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
from models.paligemma import token_xent_loss

Array = jnp.ndarray
PyTree = Any


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

    def encode_image(self, images: Array, train: bool = False) -> Array:
        """Encode images into CLIP patch tokens."""
        return self.image_encoder(images, train=train)

    def make_causal_with_prefix_block(
        self,
        L: int,
        prefix_total: Array,
        cache_size: Optional[int] = None,
    ) -> Array:
        """Build causal attention with a bidirectional image+prompt prefix."""
        if cache_size is None:
            cache_size = L
        pt = prefix_total[:, None, None]
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
        attn_mask = self.make_causal_with_prefix_block(T, prefix_len, cache_size)
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
        del attention_mask, mask_token_category_probs, use_cache

        log_dict: Dict[str, Array] = {}
        token_embeds = self.lm_backbone.embedder.encode(input_ids)
        log_dict["token_embeds_norm"] = jnp.mean(token_embeds ** 2)

        if images is not None:
            clip_tokens = self.encode_image(images, train=labels is not None)
            log_dict["clip_tokens_norm"] = jnp.mean(clip_tokens ** 2)
            img_embeds = self.projector(clip_tokens)
            log_dict["img_embeds_norm"] = jnp.mean(img_embeds ** 2)

            split_txt_cache = {}
            if self.txt_feature_layer > 0:
                token_embeds, split_txt_cache = self._apply_text_feature_layers(
                    token_embeds,
                    jnp.asarray(prefix_len, dtype=jnp.int32),
                    cache,
                )
                log_dict["txt_feature_embeds_norm"] = jnp.mean(token_embeds ** 2)

            token_embeds = jnp.concatenate([img_embeds, token_embeds], axis=1)
            K = img_embeds.shape[1]
        else:
            K = 0
            split_txt_cache = {}

        B, L, _ = token_embeds.shape
        prefix_total = jnp.asarray(prefix_len, dtype=jnp.int32) + K
        if cache is not None:
            cache_size = cache[list(cache.keys())[0]]["k"].shape[1]
            attn_mask = self.make_causal_with_prefix_block(L, prefix_total, cache_size)
        else:
            attn_mask = self.make_causal_with_prefix_block(L, prefix_total)

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
        logits = self.lm_backbone.embedder.decode(out)

        if self.final_logit_softcap != 0.0 and labels is not None:
            logits = jnp.tanh(logits / self.final_logit_softcap) * self.final_logit_softcap

        if labels is None:
            return {"logits": logits, "cache": new_cache}

        assert cache is None
        labels_full = (
            jnp.concatenate([jnp.full((B, K), -100, dtype=jnp.int32), labels], axis=1)
            if K > 0 else labels
        )
        loss = token_xent_loss(logits, labels_full)

        valid = labels_full != -100
        valid_count = valid.sum()
        acc = (
            jnp.sum((jnp.argmax(logits, axis=-1) == labels_full) * valid)
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
            "labels": labels_full,
            "preds": jnp.argmax(logits, axis=-1),
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
            token_embeds = jnp.concatenate([img_embeds, token_embeds], axis=1)

        L = token_embeds.shape[1]
        positions = jnp.broadcast_to(jnp.arange(L, dtype=jnp.int32)[None, :], (B, L))
        prefill_attn_mask = self.make_causal_with_prefix_block(
            L, prefix_total, cache_size=max_total_len
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
        del beam_size
        return self.generate(prompt_ids, prefix_len, images, max_new_tokens)
