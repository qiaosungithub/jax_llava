"""Minimal LLaVA-1.5-style model using CLIP ViT-L/14@336 + Gemma LM.

This is intentionally close to the existing PaliGemma training API:
``__call__`` accepts ``input_ids, images, prefix_len, attention_mask, labels``
and returns ``(loss, log_dict, debug_dict)`` during teacher-forced training.
The current implementation prepends projected image patch tokens before the
text sequence, which matches the common LLaVA prompt layout where the image
token appears at the beginning of the conversation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from gemma.gm.nn._transformer import _Inputs

from models.clip_vit import CLIP_L14_336, CLIPVisionTower
from models.gemma import load_LM
from models.paligemma import token_xent_loss, token_xent_loss_from_hidden
from utils.pjit_util import constrain_batch, constrain_batch_model

Array = jnp.ndarray
PyTree = Any
_IMAGE_TEXT_STAT_EPS = 1e-6


def _mean_square(x: Array) -> Array:
    return jnp.mean(x.astype(jnp.float32) ** 2)


def _rms_from_mean_square(x_ms: Array) -> Array:
    return jnp.sqrt(jnp.maximum(x_ms, 0.0))


def _masked_mean_square(x: Array, mask: Optional[Array]) -> Array:
    if mask is None:
        return _mean_square(x)
    mask = mask.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(mask) * x.shape[-1], 1.0)
    mask = mask[..., None]
    return jnp.sum(x.astype(jnp.float32) ** 2 * mask) / denom


def _prompt_mask(text_embeds: Array, prefix_len: Array, attention_mask: Optional[Array]) -> Array:
    B, T = text_embeds.shape[:2]
    prefix_len = jnp.asarray(prefix_len, dtype=jnp.int32)
    if prefix_len.ndim == 0:
        prefix_len = jnp.broadcast_to(prefix_len, (B,))
    mask = jnp.arange(T, dtype=jnp.int32)[None, :] < prefix_len[:, None]
    if attention_mask is not None:
        mask = mask & attention_mask.astype(bool)
    return mask


def _sample_moments(
    x: Array,
    axes: str,
    mask: Optional[Array] = None,
) -> tuple[Array, Array]:
    x = x.astype(jnp.float32)
    if axes == "scalar":
        if mask is None:
            mean = jnp.mean(x, axis=(1, 2), keepdims=True)
            var = jnp.mean((x - mean) ** 2, axis=(1, 2), keepdims=True)
        else:
            mask = mask.astype(jnp.float32)[..., None]
            denom = jnp.maximum(
                jnp.sum(mask, axis=(1, 2), keepdims=True) * x.shape[-1],
                1.0,
            )
            mean = jnp.sum(x * mask, axis=(1, 2), keepdims=True) / denom
            var = jnp.sum((x - mean) ** 2 * mask, axis=(1, 2), keepdims=True) / denom
    elif axes == "per_channel":
        if mask is None:
            mean = jnp.mean(x, axis=1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, axis=1, keepdims=True)
        else:
            mask = mask.astype(jnp.float32)[..., None]
            denom = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
            mean = jnp.sum(x * mask, axis=1, keepdims=True) / denom
            var = jnp.sum((x - mean) ** 2 * mask, axis=1, keepdims=True) / denom
    else:
        raise ValueError(f"Unsupported image_text_stat_axes: {axes}")
    return mean, jnp.sqrt(jnp.maximum(var, 0.0))


class LlavaProjector(nn.Module):
    """LLaVA multimodal projector.

    LLaVA-1.5 uses ``mlp2x_gelu``: Linear(mm_hidden -> lm_hidden), GELU,
    Linear(lm_hidden -> lm_hidden).  ``linear`` is kept for stage-1 ablations.
    """

    out_dim: int
    projector_type: str = "mlp2x_gelu"
    use_ln: bool = False

    @nn.compact
    def __call__(self, x: Array) -> Array:
        if self.projector_type == "linear":
            x = nn.Dense(self.out_dim, name="linear")(x)
        elif self.projector_type == "mlp2x_gelu":
            x = nn.Dense(self.out_dim, name="linear_1")(x)
            x = nn.gelu(x, approximate=False)
            x = nn.Dense(self.out_dim, name="linear_2")(x)
        else:
            raise ValueError(f"Unsupported projector_type: {self.projector_type}")

        if self.use_ln:
            x = nn.RMSNorm(name="rms_norm")(x)
        return x


class LlavaGemma(nn.Module):
    """LLaVA-style VLM with a CLIP vision tower and Gemma language model."""

    # Vision tower
    vision_tower_str: str = CLIP_L14_336
    vision_feature_layer: int = -2
    vision_select_feature: str = "patch"
    clip_input_format: str = "minus_one_to_one"

    # Language model
    lm_backbone_str: str = "gemma3_1B"
    attn_logits_soft_cap: float = 0.0
    final_logit_softcap: float = 0.0

    # Projector
    projector_type: str = "mlp2x_gelu"
    projector_use_ln: bool = False

    # Kept for compatibility with existing train/eval config paths.
    image_size: int = 336
    recon_loss_weight: float = 0.0
    txt_feature_layer: int = 0
    # If the first txt_feature_layer LM blocks are frozen, their text-only
    # output should usually be treated as a fixed feature. Otherwise JAX still
    # builds a backward pass through those frozen blocks for no trainable
    # parameter update, which makes late-fusion HSDP runs much slower.
    stop_gradient_text_features: bool = False
    image_post_connector_scale: float = 1.0
    image_post_connector_transform: str = "fixed_scale"
    image_text_stat_source: str = "prompt"
    image_text_stat_axes: str = "scalar"
    token_loss_mode: str = "hidden_scan"
    token_loss_chunk_size: int = 8192
    # Standard LLaVA behavior: image prefix tokens are bidirectional, while
    # prompt/text tokens remain causal. Set False to restore the old full
    # image+prompt bidirectional prefix.
    prompt_causal: bool = True
    eos_id: int = 1

    def setup(self) -> None:
        self.image_encoder = CLIPVisionTower(
            model_name=self.vision_tower_str,
            feature_layer=self.vision_feature_layer,
            select_feature=self.vision_select_feature,
            input_format=self.clip_input_format,
        )
        self.lm_backbone, self.lm_backbone_hidden_size = load_LM(
            self.lm_backbone_str,
            attn_logits_soft_cap=self.attn_logits_soft_cap,
            final_logit_softcap=self.final_logit_softcap,
        )
        self.projector = LlavaProjector(
            out_dim=self.lm_backbone_hidden_size,
            projector_type=self.projector_type,
            use_ln=self.projector_use_ln,
        )
        if self.txt_feature_layer < 0:
            raise ValueError(f"txt_feature_layer must be >= 0, got {self.txt_feature_layer}")
        if self.txt_feature_layer > len(self.lm_backbone.blocks):
            raise ValueError(
                f"txt_feature_layer={self.txt_feature_layer} exceeds "
                f"LM depth {len(self.lm_backbone.blocks)}"
            )
        if self._image_post_connector_transform() not in {"fixed_scale", "match_text_stats"}:
            raise ValueError(
                "image_post_connector_transform must be 'fixed_scale' or "
                f"'match_text_stats', got {self.image_post_connector_transform!r}"
            )
        if self._image_text_stat_source() != "prompt":
            raise ValueError(
                "Only image_text_stat_source='prompt' is currently supported, got "
                f"{self.image_text_stat_source!r}"
            )
        if self._image_text_stat_axes() not in {"scalar", "per_channel"}:
            raise ValueError(
                "image_text_stat_axes must be 'scalar' or 'per_channel', got "
                f"{self.image_text_stat_axes!r}"
            )
        if self._token_loss_mode() not in {"hidden_scan", "full_decode"}:
            raise ValueError(
                "token_loss_mode must be 'hidden_scan' or 'full_decode', got "
                f"{self.token_loss_mode!r}"
            )

    def _image_post_connector_transform(self) -> str:
        return str(self.image_post_connector_transform or "fixed_scale").lower().replace("-", "_")

    def _image_text_stat_source(self) -> str:
        return str(self.image_text_stat_source or "prompt").lower().replace("-", "_")

    def _image_text_stat_axes(self) -> str:
        return str(self.image_text_stat_axes or "scalar").lower().replace("-", "_").replace(" ", "_")

    def _token_loss_mode(self) -> str:
        return str(self.token_loss_mode or "hidden_scan").lower().replace("-", "_")

    def encode_image(self, images: Array, train: bool = False) -> Array:
        """Encode images into CLIP patch tokens."""
        return constrain_batch_model(self.image_encoder(images, train=train))

    def _scale_image_embeds(self, img_embeds: Array) -> Array:
        scale = 1.0 if self.image_post_connector_scale is None else float(
            self.image_post_connector_scale
        )
        if scale == 1.0:
            return img_embeds
        return img_embeds * jnp.asarray(scale, dtype=img_embeds.dtype)

    def _image_post_connector_scale_value(self) -> float:
        if self.image_post_connector_scale is None:
            return 1.0
        return float(self.image_post_connector_scale)

    def _transform_image_embeds(
        self,
        img_embeds: Array,
        text_embeds: Array,
        prefix_len: Array,
        attention_mask: Optional[Array],
    ) -> tuple[Array, Dict[str, Array]]:
        transform = self._image_post_connector_transform()
        if transform == "fixed_scale":
            return self._scale_image_embeds(img_embeds), {}
        if transform != "match_text_stats":
            raise ValueError(f"Unsupported image_post_connector_transform: {transform}")

        axes = self._image_text_stat_axes()
        text_mask = _prompt_mask(text_embeds, prefix_len, attention_mask)
        img_mean, img_std = _sample_moments(img_embeds, axes)
        text_mean, text_std = _sample_moments(text_embeds, axes, text_mask)
        matched = (
            (img_embeds.astype(jnp.float32) - img_mean)
            / jnp.maximum(img_std, _IMAGE_TEXT_STAT_EPS)
            * jax.lax.stop_gradient(text_std)
            + jax.lax.stop_gradient(text_mean)
        )
        log_dict = {
            "image_text_stat_prompt_tokens": jnp.sum(text_mask.astype(jnp.float32)),
            "image_text_stat_img_mean": jnp.mean(img_mean),
            "image_text_stat_img_std": jnp.mean(img_std),
            "image_text_stat_text_mean": jnp.mean(text_mean),
            "image_text_stat_text_std": jnp.mean(text_std),
        }
        return matched.astype(img_embeds.dtype), log_dict

    def make_causal_with_prefix_block(
        self,
        L: int,
        prefix_total: Array,
        cache_size: Optional[int] = None,
        image_prefix: Optional[Array] = None,
    ) -> Array:
        """Build causal attention, optionally with only image tokens bidirectional."""
        if cache_size is None:
            cache_size = L
        if image_prefix is None or not self.prompt_causal:
            bidir_prefix = prefix_total
        else:
            bidir_prefix = jnp.asarray(image_prefix, dtype=jnp.int32)
            if bidir_prefix.ndim == 0:
                bidir_prefix = jnp.broadcast_to(bidir_prefix, prefix_total.shape)
        pt = bidir_prefix[:, None, None]
        i = jnp.arange(L, dtype=jnp.int32)[None, :, None]
        j = jnp.arange(cache_size, dtype=jnp.int32)[None, None, :]
        return (j <= i) | ((i < pt) & (j < pt))

    def _cache_size(self, cache: PyTree) -> int:
        return cache[list(cache.keys())[0]]["k"].shape[1]

    def _cache_dtype(self, cache: PyTree) -> jnp.dtype:
        return cache[list(cache.keys())[0]]["v"].dtype

    def _apply_text_feature_layers(
        self,
        token_embeds: Array,
        prefix_len: Array,
        cache: Optional[PyTree] = None,
    ) -> tuple[Array, dict]:
        """Run text embeddings through the first N LM layers before fusion."""
        if self.txt_feature_layer == 0:
            return token_embeds, {}

        B, T, _ = token_embeds.shape
        positions = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32)[None, :], (B, T))
        cache_size = self._cache_size(cache) if cache is not None else None
        attn_mask = self.make_causal_with_prefix_block(
            T, prefix_len, cache_size, image_prefix=0
        )
        x = token_embeds.astype(self._cache_dtype(cache)) if cache is not None else token_embeds

        old_cache = cache or {}
        new_cache = {}
        for i in range(self.txt_feature_layer):
            layer_name = f"layer_{i}"
            layer_cache, x = self.lm_backbone.blocks[i](
                x,
                positions,
                old_cache.get(layer_name),
                attn_mask,
            )
            new_cache[layer_name] = layer_cache
        return x, new_cache

    def _apply_lm_from_layer(
        self,
        token_embeds: Array,
        positions: Array,
        attn_mask: Array,
        cache: Optional[PyTree],
        start_layer: int,
    ) -> tuple[Array, dict]:
        """Run LM blocks from ``start_layer`` to the final norm."""
        x = token_embeds.astype(self._cache_dtype(cache)) if cache is not None else token_embeds
        old_cache = cache or {}
        new_cache = {}
        for i in range(start_layer, len(self.lm_backbone.blocks)):
            layer_name = f"layer_{i}"
            layer_cache, x = self.lm_backbone.blocks[i](
                x,
                positions,
                old_cache.get(layer_name),
                attn_mask,
            )
            new_cache[layer_name] = layer_cache
        return self.lm_backbone.final_norm(x), new_cache

    def __call__(
        self,
        input_ids: Array,
        images: Optional[Array],
        prefix_len: Array,
        attention_mask: Optional[Array] = None,
        labels: Optional[Array] = None,
        mask_token_category_probs: Optional[Array] = None,
        cache: Optional[PyTree] = None,
        use_cache: bool = False,
    ) -> Any:
        del mask_token_category_probs, use_cache

        log_dict: Dict[str, Array] = {}
        text_mask = attention_mask
        token_embeds = self.lm_backbone.embedder.encode(input_ids)
        token_embeds = constrain_batch_model(token_embeds)
        token_embeds_norm = _mean_square(token_embeds)
        token_embeds_rms = _rms_from_mean_square(token_embeds_norm)
        token_embeds_valid_norm = _masked_mean_square(token_embeds, text_mask)
        token_embeds_valid_rms = _rms_from_mean_square(token_embeds_valid_norm)
        log_dict["token_embeds_valid_rms"] = token_embeds_valid_rms
        if text_mask is not None:
            valid_text_tokens = jnp.sum(text_mask.astype(jnp.float32))
            total_text_tokens = jnp.asarray(text_mask.size, dtype=jnp.float32)
            log_dict["valid_text_tokens"] = valid_text_tokens
            log_dict["text_padding_fraction"] = (
                1.0 - valid_text_tokens / jnp.maximum(total_text_tokens, 1.0)
            )

        if images is not None:
            images = constrain_batch(images)
            clip_tokens = self.encode_image(images, train=labels is not None)
            clip_tokens_norm = _mean_square(clip_tokens)
            log_dict["clip_tokens_norm"] = clip_tokens_norm
            log_dict["clip_tokens_rms"] = _rms_from_mean_square(clip_tokens_norm)
            img_embeds = self.projector(clip_tokens)
            img_embeds = constrain_batch_model(img_embeds)
            raw_img_embeds_norm = _mean_square(img_embeds)
            raw_img_embeds_rms = _rms_from_mean_square(raw_img_embeds_norm)
            # Explicit post-connector metrics. img_embeds_norm is kept for
            # backward-compatible W&B curves.
            log_dict["image_post_connector_scale"] = jnp.asarray(
                self._image_post_connector_scale_value(), dtype=jnp.float32
            )
            log_dict["img_embeds_pre_scale_rms"] = raw_img_embeds_rms

            split_txt_cache = {}
            if self.txt_feature_layer > 0:
                token_embeds, split_txt_cache = self._apply_text_feature_layers(
                    token_embeds,
                    jnp.asarray(prefix_len, dtype=jnp.int32),
                    cache,
                )
                if self.stop_gradient_text_features:
                    token_embeds = jax.lax.stop_gradient(token_embeds)
                txt_feature_embeds_norm = _mean_square(token_embeds)
                txt_feature_embeds_rms = _rms_from_mean_square(txt_feature_embeds_norm)
                txt_feature_embeds_valid_norm = _masked_mean_square(token_embeds, text_mask)
                txt_feature_embeds_valid_rms = _rms_from_mean_square(
                    txt_feature_embeds_valid_norm
                )
                log_dict["txt_feature_embeds_valid_rms"] = txt_feature_embeds_valid_rms

            img_embeds, stat_log_dict = self._transform_image_embeds(
                img_embeds,
                token_embeds,
                jnp.asarray(prefix_len, dtype=jnp.int32),
                text_mask,
            )
            log_dict.update(stat_log_dict)
            img_embeds = constrain_batch_model(img_embeds)
            img_embeds_norm = _mean_square(img_embeds)
            img_embeds_rms = _rms_from_mean_square(img_embeds_norm)
            log_dict["img_embeds_post_connector_rms"] = img_embeds_rms
            log_dict["img_token_txt_embeds_valid_ratio"] = (
                img_embeds_rms / jnp.maximum(token_embeds_valid_rms, 1e-12)
            )
            if self.txt_feature_layer > 0:
                log_dict["img_txt_feature_valid_rms_ratio"] = (
                    img_embeds_rms / jnp.maximum(txt_feature_embeds_valid_rms, 1e-12)
                )

            token_embeds = jnp.concatenate([img_embeds, token_embeds], axis=1)
            token_embeds = constrain_batch_model(token_embeds)
            K = img_embeds.shape[1]
        else:
            K = 0
            split_txt_cache = {}

        B, L, _ = token_embeds.shape
        prefix_total = jnp.asarray(prefix_len, dtype=jnp.int32) + K
        if cache is not None:
            cache_size = cache[list(cache.keys())[0]]["k"].shape[1]
            attn_mask = self.make_causal_with_prefix_block(
                L, prefix_total, cache_size, image_prefix=K
            )
        else:
            attn_mask = self.make_causal_with_prefix_block(
                L, prefix_total, image_prefix=K
            )

        positions = jnp.broadcast_to(jnp.arange(L, dtype=jnp.int32)[None, :], (B, L))
        inputs = _Inputs(
            embeddings=token_embeds,
            positions=positions,
            attention_mask=attn_mask,
            inputs_mask=jnp.ones((B, L), dtype=jnp.int32),
        )

        if self.txt_feature_layer > 0 and images is not None:
            out, rest_cache = self._apply_lm_from_layer(
                token_embeds,
                positions,
                attn_mask,
                cache,
                self.txt_feature_layer,
            )
            new_cache = {**split_txt_cache, **rest_cache}
        else:
            if cache is not None:
                inputs = _Inputs(
                    embeddings=token_embeds.astype(self._cache_dtype(cache)),
                    positions=positions,
                    attention_mask=attn_mask,
                    inputs_mask=jnp.ones((B, L), dtype=jnp.int32),
                )
            out, new_cache = self.lm_backbone._apply_attention(inputs, cache)
        out = constrain_batch_model(out)

        if labels is None:
            logits = self.lm_backbone.embedder.decode(out)
            return {"logits": logits, "cache": new_cache}

        assert cache is None
        lm_hidden = out[:, K:, :] if K > 0 else out
        if self._token_loss_mode() == "full_decode":
            logits = self.lm_backbone.embedder.decode(lm_hidden)
            if self.final_logit_softcap != 0.0:
                logits = jnp.tanh(logits / self.final_logit_softcap) * self.final_logit_softcap
            loss = token_xent_loss(logits, labels)
            pred_ids = jnp.argmax(logits, axis=-1)
        else:
            loss, pred_ids = token_xent_loss_from_hidden(
                lm_hidden,
                self.lm_backbone.embedder.input_embedding_table,
                labels,
                final_logit_softcap=self.final_logit_softcap,
                chunk_size=int(self.token_loss_chunk_size),
            )

        valid = labels != -100
        valid_count = valid.sum()
        acc = (
            jnp.sum((pred_ids == labels) * valid)
            / jnp.maximum(valid_count, 1)
        )
        log_dict["loss_vlm"] = loss
        log_dict["acc"] = acc
        log_dict["valid_tokens"] = valid_count.astype(jnp.float32)
        log_dict["valid_tokens_per_sample"] = (
            valid_count.astype(jnp.float32) / jnp.maximum(B, 1)
        )

        debug = {
            "attn_mask": attn_mask,
            "labels": labels,
            "preds": pred_ids,
            "input_ids": input_ids,
        }
        return loss, log_dict, debug

    def generate(
        self,
        prompt_ids: Array,
        prefix_len: Array,
        images: Optional[Array] = None,
        max_new_tokens: int = 64,
    ) -> Array:
        """Greedy autoregressive generation. Returns ``(B, max_new_tokens)``."""
        B = prompt_ids.shape[0]
        T_prompt = prompt_ids.shape[1]
        prefix_len = jnp.asarray(prefix_len, dtype=jnp.int32)

        token_embeds = self.lm_backbone.embedder.encode(prompt_ids)
        if images is not None:
            clip_tokens = self.encode_image(images, train=False)
            img_embeds = self.projector(clip_tokens)
            K = img_embeds.shape[1]
        else:
            img_embeds = None
            K = 0

        prefix_total = prefix_len + K
        step_pos_init = prefix_total[:, None]
        step_pos_init_txt = prefix_len[:, None]
        max_total_len = T_prompt + max_new_tokens + K

        cache = self.lm_backbone.init_cache(
            batch_size=B,
            dtype=jnp.bfloat16,
            cache_length=max_total_len,
        )
        cache_dtype = cache[list(cache.keys())[0]]["v"].dtype

        split_txt_cache = {}
        use_split = self.txt_feature_layer > 0 and images is not None
        if use_split:
            token_embeds, split_txt_cache = self._apply_text_feature_layers(
                token_embeds,
                prefix_len,
                cache,
            )
        if img_embeds is not None:
            img_embeds, _ = self._transform_image_embeds(
                img_embeds,
                token_embeds,
                prefix_len,
                attention_mask=None,
            )
            img_embeds = constrain_batch_model(img_embeds)
            token_embeds = jnp.concatenate([img_embeds, token_embeds], axis=1)

        L = token_embeds.shape[1]
        positions = jnp.broadcast_to(jnp.arange(L, dtype=jnp.int32)[None, :], (B, L))
        prefill_attn_mask = self.make_causal_with_prefix_block(
            L, prefix_total, cache_size=max_total_len, image_prefix=K
        )
        if use_split:
            prefill_out, rest_cache = self._apply_lm_from_layer(
                token_embeds,
                positions,
                prefill_attn_mask,
                cache,
                self.txt_feature_layer,
            )
            prefill_cache = {**split_txt_cache, **rest_cache}
        else:
            prefill_inputs = _Inputs(
                embeddings=token_embeds.astype(cache_dtype),
                positions=positions,
                attention_mask=prefill_attn_mask,
                inputs_mask=jnp.ones((B, L), dtype=jnp.int32),
            )
            prefill_out, prefill_cache = self.lm_backbone._apply_attention(
                prefill_inputs, cache
            )

        # Decode only the last prompt hidden state. Decoding the whole prefill
        # sequence materializes [B, T, vocab] logits and can OOM on v6e-64.
        hidden_at_last = jnp.take_along_axis(
            prefill_out,
            (prefix_total - 1)[:, None, None],
            axis=1,
        ).squeeze(1)
        logits_at_last = self.lm_backbone.embedder.decode(hidden_at_last)
        first_token = jnp.argmax(logits_at_last, axis=-1, keepdims=True)

        tokens_out = jnp.zeros((B, max_new_tokens), dtype=jnp.int32)
        tokens_out = tokens_out.at[:, 0].set(first_token.squeeze(-1))

        def cond_fn(carry):
            _, _, step, _ = carry
            return step < max_new_tokens

        def body_fn(carry):
            curr_tok, curr_cache, step, out_tokens = carry
            fk = list(curr_cache.keys())[0]
            emb = self.lm_backbone.embedder.encode(curr_tok).astype(
                curr_cache[fk]["v"].dtype
            )
            j = jnp.arange(max_total_len)[None, None, :]
            mask = (
                (j < prefix_total[:, None, None])
                | ((j >= T_prompt + K) & (j < T_prompt + K + step))
            )
            step_inputs = _Inputs(
                embeddings=emb,
                positions=step_pos_init + step,
                attention_mask=mask,
                inputs_mask=jnp.ones((B, 1), dtype=jnp.int32),
            )
            lm_out, next_cache = self.lm_backbone._apply_attention(
                step_inputs, curr_cache
            )
            next_tok = jnp.argmax(
                self.lm_backbone.embedder.decode(lm_out[:, -1, :]),
                axis=-1,
                keepdims=True,
            )
            return (
                next_tok,
                next_cache,
                step + 1,
                out_tokens.at[:, step].set(next_tok.squeeze(-1)),
            )

        def body_fn_split(m, carry):
            curr_tok, curr_cache, step, out_tokens = carry
            emb = m.lm_backbone.embedder.encode(curr_tok).astype(
                curr_cache[list(curr_cache.keys())[0]]["v"].dtype
            )
            j = jnp.arange(max_total_len)[None, None, :]
            txt_mask = (
                (j < prefix_len[:, None, None])
                | ((j >= T_prompt) & (j < T_prompt + step))
            )
            full_mask = (
                (j < prefix_total[:, None, None])
                | ((j >= T_prompt + K) & (j < T_prompt + K + step))
            )

            x = emb
            new_cache = {}
            txt_pos = step_pos_init_txt + step
            for i in range(m.txt_feature_layer):
                layer_name = f"layer_{i}"
                layer_cache, x = m.lm_backbone.blocks[i](
                    x,
                    txt_pos,
                    curr_cache.get(layer_name),
                    txt_mask,
                )
                new_cache[layer_name] = layer_cache

            full_pos = step_pos_init + step
            for i in range(m.txt_feature_layer, len(m.lm_backbone.blocks)):
                layer_name = f"layer_{i}"
                layer_cache, x = m.lm_backbone.blocks[i](
                    x,
                    full_pos,
                    curr_cache.get(layer_name),
                    full_mask,
                )
                new_cache[layer_name] = layer_cache

            lm_out = m.lm_backbone.final_norm(x)
            next_tok = jnp.argmax(
                m.lm_backbone.embedder.decode(lm_out[:, -1, :]),
                axis=-1,
                keepdims=True,
            )
            return (
                next_tok,
                new_cache,
                step + 1,
                out_tokens.at[:, step].set(next_tok.squeeze(-1)),
            )

        _, _, _, all_tokens = jax.lax.while_loop(
            cond_fn,
            (lambda carry: body_fn_split(self, carry)) if use_split else body_fn,
            (first_token, prefill_cache, 1, tokens_out),
        )
        return all_tokens

    def generate_beam_search(
        self,
        prompt_ids: Array,
        prefix_len: Array,
        images: Optional[Array] = None,
        beam_size: int = 3,
        max_new_tokens: int = 64,
    ) -> Array:
        """Beam-search autoregressive generation. Returns ``(B, max_new_tokens)``."""
        if beam_size <= 1:
            return self.generate(prompt_ids, prefix_len, images, max_new_tokens)

        B = prompt_ids.shape[0]
        T_prompt = prompt_ids.shape[1]
        prefix_len = jnp.asarray(prefix_len, dtype=jnp.int32)
        if max_new_tokens <= 0:
            return jnp.zeros((B, 0), dtype=jnp.int32)

        token_embeds = self.lm_backbone.embedder.encode(prompt_ids)
        if images is not None:
            clip_tokens = self.encode_image(images, train=False)
            img_embeds = self.projector(clip_tokens)
            K = img_embeds.shape[1]
        else:
            img_embeds = None
            K = 0

        prefix_total = prefix_len + K
        step_pos_init = prefix_total[:, None]
        step_pos_init_txt = prefix_len[:, None]
        max_total_len = T_prompt + max_new_tokens + K

        cache = self.lm_backbone.init_cache(
            batch_size=B,
            dtype=jnp.bfloat16,
            cache_length=max_total_len,
        )
        cache_dtype = cache[list(cache.keys())[0]]["v"].dtype

        split_txt_cache = {}
        use_split = self.txt_feature_layer > 0 and images is not None
        if use_split:
            token_embeds, split_txt_cache = self._apply_text_feature_layers(
                token_embeds,
                prefix_len,
                cache,
            )
        if img_embeds is not None:
            img_embeds, _ = self._transform_image_embeds(
                img_embeds,
                token_embeds,
                prefix_len,
                attention_mask=None,
            )
            img_embeds = constrain_batch_model(img_embeds)
            token_embeds = jnp.concatenate([img_embeds, token_embeds], axis=1)

        L = token_embeds.shape[1]
        positions = jnp.broadcast_to(jnp.arange(L, dtype=jnp.int32)[None, :], (B, L))
        prefill_attn_mask = self.make_causal_with_prefix_block(
            L, prefix_total, cache_size=max_total_len, image_prefix=K
        )
        if use_split:
            prefill_out, rest_cache = self._apply_lm_from_layer(
                token_embeds,
                positions,
                prefill_attn_mask,
                cache,
                self.txt_feature_layer,
            )
            prefill_cache = {**split_txt_cache, **rest_cache}
        else:
            prefill_inputs = _Inputs(
                embeddings=token_embeds.astype(cache_dtype),
                positions=positions,
                attention_mask=prefill_attn_mask,
                inputs_mask=jnp.ones((B, L), dtype=jnp.int32),
            )
            prefill_out, prefill_cache = self.lm_backbone._apply_attention(
                prefill_inputs, cache
            )

        # Decode only the last prompt hidden state. Beam search needs full-vocab
        # logits for candidate expansion, but not a [B, prompt_len, vocab] tensor.
        hidden_at_last = jnp.take_along_axis(
            prefill_out,
            (prefix_total - 1)[:, None, None],
            axis=1,
        ).squeeze(1)
        top_scores, top_tokens = jax.lax.top_k(
            jax.nn.log_softmax(self.lm_backbone.embedder.decode(hidden_at_last)),
            beam_size,
        )

        cache_tiled = jax.tree.map(
            lambda x: jnp.repeat(x, beam_size, axis=0),
            prefill_cache,
        )
        curr_tokens = top_tokens.reshape(-1, 1)
        beam_scores = top_scores.reshape(-1)
        beam_prefix = jnp.repeat(prefix_total, beam_size, axis=0)
        beam_prefix_len = jnp.repeat(prefix_len, beam_size, axis=0)
        step_pos_init_beam = beam_prefix[:, None]
        step_pos_init_txt_beam = beam_prefix_len[:, None]
        has_eos = (curr_tokens == self.eos_id).reshape(-1)

        history = jnp.zeros((B * beam_size, max_new_tokens), dtype=jnp.int32)
        history = history.at[:, 0].set(curr_tokens.squeeze(-1))

        def cond_fn(carry):
            _, _, _, _, _, _, step = carry
            return step < max_new_tokens

        def _select_next(log_probs, next_cache, b_scores, hist, hit_eos, curr_prefix, step):
            vocab_size = log_probs.shape[-1]
            eos_only = jnp.full_like(log_probs, -1e9).at[:, self.eos_id].set(0.0)
            effective_log_probs = jnp.where(hit_eos[:, None], eos_only, log_probs)
            total = (
                effective_log_probs.reshape(B, beam_size, -1)
                + b_scores.reshape(B, beam_size, 1)
            ).reshape(B, -1)

            next_scores, next_idx = jax.lax.top_k(total, beam_size)
            parent = next_idx // vocab_size
            ids = next_idx % vocab_size

            offset = jnp.arange(B)[:, None] * beam_size
            flat_parents = (parent + offset).reshape(-1)
            flat_ids = ids.reshape(-1)

            return (
                flat_ids.reshape(-1, 1),
                jax.tree.map(lambda x: x[flat_parents], next_cache),
                next_scores.reshape(-1),
                hist[flat_parents].at[:, step].set(flat_ids),
                hit_eos[flat_parents] | (flat_ids == self.eos_id),
                curr_prefix[flat_parents],
                step + 1,
            )

        def body_fn(carry):
            tokens, curr_cache, b_scores, hist, hit_eos, curr_prefix, step = carry
            fk = list(curr_cache.keys())[0]
            emb = self.lm_backbone.embedder.encode(tokens).astype(
                curr_cache[fk]["v"].dtype
            )
            j = jnp.arange(max_total_len)[None, None, :]
            mask = (
                (j < curr_prefix[:, None, None])
                | ((j >= T_prompt + K) & (j < T_prompt + K + step))
            )
            step_inputs = _Inputs(
                embeddings=emb,
                positions=step_pos_init_beam + step,
                attention_mask=mask,
                inputs_mask=jnp.ones((B * beam_size, 1), dtype=jnp.int32),
            )
            lm_out, next_cache = self.lm_backbone._apply_attention(
                step_inputs, curr_cache
            )
            log_probs = jax.nn.log_softmax(
                self.lm_backbone.embedder.decode(lm_out[:, -1, :])
            )
            return _select_next(
                log_probs,
                next_cache,
                b_scores,
                hist,
                hit_eos,
                curr_prefix,
                step,
            )

        def body_fn_split(carry):
            tokens, curr_cache, b_scores, hist, hit_eos, curr_prefix, step = carry
            fk = list(curr_cache.keys())[0]
            x = self.lm_backbone.embedder.encode(tokens).astype(
                curr_cache[fk]["v"].dtype
            )
            j = jnp.arange(max_total_len)[None, None, :]
            txt_mask = (
                (j < beam_prefix_len[:, None, None])
                | ((j >= T_prompt) & (j < T_prompt + step))
            )
            full_mask = (
                (j < curr_prefix[:, None, None])
                | ((j >= T_prompt + K) & (j < T_prompt + K + step))
            )

            new_cache = {}
            txt_pos = step_pos_init_txt_beam + step
            for i in range(self.txt_feature_layer):
                layer_name = f"layer_{i}"
                layer_cache, x = self.lm_backbone.blocks[i](
                    x,
                    txt_pos,
                    curr_cache.get(layer_name),
                    txt_mask,
                )
                new_cache[layer_name] = layer_cache

            full_pos = step_pos_init_beam + step
            for i in range(self.txt_feature_layer, len(self.lm_backbone.blocks)):
                layer_name = f"layer_{i}"
                layer_cache, x = self.lm_backbone.blocks[i](
                    x,
                    full_pos,
                    curr_cache.get(layer_name),
                    full_mask,
                )
                new_cache[layer_name] = layer_cache

            lm_out = self.lm_backbone.final_norm(x)
            log_probs = jax.nn.log_softmax(
                self.lm_backbone.embedder.decode(lm_out[:, -1, :])
            )
            return _select_next(
                log_probs,
                new_cache,
                b_scores,
                hist,
                hit_eos,
                curr_prefix,
                step,
            )

        _, _, _, final_history, _, _, _ = jax.lax.while_loop(
            cond_fn,
            body_fn_split if use_split else body_fn,
            (curr_tokens, cache_tiled, beam_scores, history, has_eos, beam_prefix, 1),
        )
        return final_history[jnp.arange(B) * beam_size]
