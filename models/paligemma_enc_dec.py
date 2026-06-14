"""PaliGemma with PrefixMAE image encoder-decoder.

Key differences from paligemma.py
----------------------------------
1. The image encoder is replaced by PrefixMAE (models/siglip_enc_dec.py):
     - Encoder:  image patches + K learnable tokens → N ViT blocks → K abstract tokens
     - Decoder:  first i of K encoder tokens (+ K-i mask tokens) → context ViT
                 → cross-attention with T spatial patch queries
                 → pixel prediction head  →  (B, T, P²×3)

2. Two losses are jointly optimised during training:
       loss_total = loss_vlm + recon_loss_weight × loss_recon
   where
     loss_vlm   = cross-entropy on text tokens (standard VLM objective)
     loss_recon = MSE between predicted and target image patches
                  (target = patchify(original_image))
                  i ~ Uniform[1, K] is sampled each step via the 'gen' RNG key.

3. New hyperparameters (all have sensible defaults):
     num_learnable_tokens  – K, abstract image representation size (default 256)
     recon_loss_weight     – λ  (default 0.1)
     recon_stop_grad       – if True, stop gradient through encoder output when
                             computing loss_recon; decoder only learns (default False)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
import math

from gemma.gm.nn._transformer import _Inputs

from models.siglip_enc_dec import PrefixMAE, patchify, unpatchify, recon_mse_loss
from models.gemma import load_LM
from models.paligemma import Projector, token_xent_loss_from_hidden
from utils.pjit_util import constrain_batch_model

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
    return jnp.sum(x.astype(jnp.float32) ** 2 * mask[..., None]) / denom


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


class PaliGemmaEncDec(nn.Module):
    """PaliGemma variant whose image encoder is PrefixMAE.

    Training loss
    -------------
        loss = loss_vlm  +  recon_loss_weight × loss_recon

        loss_vlm   – LM cross-entropy on text tokens (image token positions
                     are masked with label -100 as usual)
        loss_recon – MSE between decoder-predicted patches and original image
                     patches, averaged over all T patches.  The number of
                     visible encoder tokens  i  is sampled uniformly from
                     {1, …, K} using the 'gen' RNG key each training step.

    Hyperparameters
    ---------------
    Image encoder (PrefixMAE encoder)
        patch_size            patch size P          (default 16)
        image_size            input resolution      (default 224)
        num_learnable_tokens  K                     (default 256)
        enc_hidden_dim        encoder hidden dim     (default 768)
        enc_num_heads         encoder attn heads     (default 12)
        enc_head_dim          encoder head dim       (default 64)
        enc_mlp_dim           encoder MLP dim        (default 3072)
        enc_num_patch_sa_layers   Stage 1 depth      (default 6)
        enc_num_cross_attn_layers Stage 2 depth      (default 12)
        enc_num_token_sa_layers   Stage 3 depth      (default 6)

    Image decoder (PrefixMAE decoder, training only)
        dec_hidden_dim        decoder hidden dim     (default 512)
        dec_num_heads         decoder attn heads     (default 8)
        dec_head_dim          decoder head dim       (default 64)
        dec_mlp_dim           decoder MLP dim        (default 2048)
        dec_num_layers        decoder depth          (default 6)

    Language model
        lm_backbone_str       'gemma3_270M' | 'gemma2_2B'
        projector_use_ln      RMSNorm after linear projector

    Loss
        recon_loss_weight     λ                      (default 0.1)
        recon_stop_grad       stop-gradient through encoder output for
                              reconstruction loss    (default False)
    """

    # ── Image encoder ──────────────────────────────────────────────────────
    patch_size: int = 16
    image_size: int = 224
    num_learnable_tokens: int = 256
    enc_hidden_dim: int = 768
    enc_num_patch_sa_layers: int = 6
    enc_num_cross_attn_layers: int = 12
    enc_num_token_sa_layers: int = 6

    # masking config
    und_use_tokens: int = None
    und_use_mask: bool = False
    recon_use_mask: bool = False
    mask_strategy: str = "uniform"
    enc_cross_attn_split: int = 0 # position where we do nested dropout in encoder

    # ── Image decoder ──────────────────────────────────────────────────────
    dec_hidden_dim: int = 512
    dec_num_layers: int = 6
    use_decoder: bool = True
    feature_dim: int = 256

    # ── Language model ─────────────────────────────────────────────────────
    lm_backbone_str: str = 'gemma3_270M'
    projector_use_ln: bool = True
    use_2l_mlp: bool = False # whether use a 2-layer MLP for the projector, or just a single layer
    # Soft capping (Gemma2-style): 0.0 = disabled.
    attn_logits_soft_cap: float = 0.0   # applied inside every attention layer
    final_logit_softcap: float = 0.0    # applied to output logits after embedder.decode

    # ── Loss ───────────────────────────────────────────────────────────────
    # Scalar multipliers for each loss term.
    vlm_loss_weight:       float = 1.0   # image-conditioned NTP loss
    text_only_loss_weight: float = 1.0   # text-only NTP loss (unconditional branch)
    cfg_loss_weight:       float = 1.0   # CFG-style contrastive loss
    recon_loss_weight:     float = 0.1   # image reconstruction MSE loss
    # CFG-style contrastive alpha.  Set to 0.0 to skip the text-only forward
    alpha: float = 0.0

    # experiment: image embeddings align with text embeddings, by passing txt_feature_layer layers.
    txt_feature_layer: int = 0
    # Treat text features from frozen prefix LM layers as constants. This keeps
    # late-fusion runs from spending backward compute on frozen text-only blocks.
    stop_gradient_text_features: bool = False
    image_post_connector_scale: float = 1.0
    image_post_connector_transform: str = "fixed_scale"
    image_text_stat_source: str = "prompt"
    image_text_stat_axes: str = "scalar"
    # Standard LLaVA behavior: image prefix tokens are bidirectional, while
    # prompt/text tokens remain causal. Set False to restore the old full
    # image+prompt bidirectional prefix.
    prompt_causal: bool = True

    # ── Misc ───────────────────────────────────────────────────────────────
    eos_id: int = 1

    # ------------------------------------------------------------------
    def setup(self) -> None:
        self.image_encoder = PrefixMAE(
            patch_size=self.patch_size,
            image_size=self.image_size,
            enc_hidden_dim=self.enc_hidden_dim,
            feature_dim=self.feature_dim,
            enc_num_patch_sa_layers=self.enc_num_patch_sa_layers,
            enc_num_cross_attn_layers=self.enc_num_cross_attn_layers,
            enc_num_token_sa_layers=self.enc_num_token_sa_layers,
            num_learnable_tokens=self.num_learnable_tokens,
            dec_hidden_dim=self.dec_hidden_dim,
            dec_num_layers=self.dec_num_layers,
            use_decoder=self.use_decoder,
            enc_cross_attn_split=self.enc_cross_attn_split,
        )
        self.lm_backbone, self.lm_backbone_hidden_size = load_LM(
            self.lm_backbone_str,
            attn_logits_soft_cap=self.attn_logits_soft_cap,
            final_logit_softcap=self.final_logit_softcap,
        )
        self.projector = Projector(
            out_dim=self.lm_backbone_hidden_size,
            use_ln=self.projector_use_ln,
            use_2l_mlp=self.use_2l_mlp,
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def encode_image(
        self,
        images: Array,
        vis_mask: Optional[Array] = None,  # (B, K) bool; forwarded to encoder for mid-masking
    ) -> Array:
        """(B, H, W, 3) → (B, K, feature_dim)"""
        return self.image_encoder.encode(images, vis_mask)

    def _image_post_connector_scale_value(self) -> float:
        if self.image_post_connector_scale is None:
            return 1.0
        return float(self.image_post_connector_scale)

    def _image_post_connector_transform(self) -> str:
        return str(self.image_post_connector_transform or "fixed_scale").lower().replace("-", "_")

    def _image_text_stat_source(self) -> str:
        return str(self.image_text_stat_source or "prompt").lower().replace("-", "_")

    def _image_text_stat_axes(self) -> str:
        return str(self.image_text_stat_axes or "scalar").lower().replace("-", "_").replace(" ", "_")

    def _scale_image_embeds(self, img_embeds: Array) -> Array:
        scale = self._image_post_connector_scale_value()
        if scale == 1.0:
            return img_embeds
        return img_embeds * jnp.asarray(scale, dtype=img_embeds.dtype)

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
            'image_text_stat_prompt_tokens': jnp.sum(text_mask.astype(jnp.float32)),
            'image_text_stat_img_mean': jnp.mean(img_mean),
            'image_text_stat_img_std': jnp.mean(img_std),
            'image_text_stat_text_mean': jnp.mean(text_mean),
            'image_text_stat_text_std': jnp.mean(text_std),
        }
        return matched.astype(img_embeds.dtype), log_dict

    def make_causal_with_prefix_block(
        self,
        L: int,
        prefix_total: Array,            # (B,)
        cache_size: Optional[int] = None,
        image_prefix: Optional[Array] = None,
    ) -> Array:
        """(B, L, cache_size) bool attention mask.

        When prompt_causal=True, only image prefix tokens are bidirectional;
        prompt/text tokens attend causally. When False, the legacy
        image+prompt prefix is fully bidirectional.
        """
        if cache_size is None:
            cache_size = L
        if image_prefix is None or not self.prompt_causal:
            bidir_prefix = prefix_total
        else:
            bidir_prefix = jnp.asarray(image_prefix, dtype=jnp.int32)
            if bidir_prefix.ndim == 0:
                bidir_prefix = jnp.broadcast_to(bidir_prefix, prefix_total.shape)
        pt = bidir_prefix[:, None, None]                             # (B,1,1)
        i  = jnp.arange(L,          dtype=jnp.int32)[None, :, None] # (1,L,1)
        j  = jnp.arange(cache_size, dtype=jnp.int32)[None, None, :] # (1,1,C)
        return (j <= i) | ((i < pt) & (j < pt))                     # (B,L,C)

    def _sample_num_visible_from_category(
        self,
        rng: Array,
        category_probs: Array,   # (B, 7), probs over token counts [4..256]
        K: int,
    ) -> Array:
        token_values = jnp.array([4, 8, 16, 32, 64, 128, 256], dtype=jnp.int32)
        valid = token_values <= K

        probs = category_probs.astype(jnp.float32)
        probs = jnp.where(valid[None, :], probs, 0.0)
        prob_sum = probs.sum(axis=-1, keepdims=True)

        n_valid = jnp.maximum(valid.sum(), 1)
        fallback = valid.astype(jnp.float32)[None, :] / n_valid.astype(jnp.float32)
        probs = jnp.where(prob_sum > 0, probs / jnp.maximum(prob_sum, 1e-8), fallback)

        sampled_idx = jax.random.categorical(
            rng, jnp.log(jnp.clip(probs, 1e-8, 1.0)), axis=-1
        )
        num_visible = token_values[sampled_idx]
        return jnp.clip(num_visible, 1, K)

    # ------------------------------------------------------------------
    # Forward pass  (training + teacher-forced eval)
    # ------------------------------------------------------------------

    def __call__(
        self,
        input_ids: Array,                        # (B, T_text)
        images: Array,                           # (B, H, W, 3)
        prefix_len: Array,                       # (B,)
        attention_mask: Optional[Array] = None,  # unused; kept for API compat
        labels: Optional[Array] = None,          # (B, T_text), -100=ignore
        mask_token_category_probs: Optional[Array] = None,  # (B, 7) over [4,8,...,256]
        cache: Optional[PyTree] = None,
        use_cache: bool = False,
        return_hidden: bool = False,
    ) -> Any:
        """
        Training (labels is not None)
        ─────────────────────────────
        Returns (loss_total, log_dict, debug_dict)
            loss_total = loss_vlm + recon_loss_weight × loss_recon

        Inference (labels is None)
        ──────────────────────────
        Returns {'logits': (B, L, V), 'cache': new_cache}
        """
        log_dict: Dict[str, Array] = {}
        K = self.num_learnable_tokens

        # ── Text embeddings ────────────────────────────────────────────────
        token_embeds = self.lm_backbone.embedder.encode(input_ids)   # (B, T_text, D_lm)
        token_embeds = constrain_batch_model(token_embeds)
        token_embeds_norm = _mean_square(token_embeds)
        token_embeds_rms = _rms_from_mean_square(token_embeds_norm)
        token_embeds_valid_norm = _masked_mean_square(token_embeds, attention_mask)
        token_embeds_valid_rms = _rms_from_mean_square(token_embeds_valid_norm)
        log_dict['token_embeds_norm'] = token_embeds_norm
        log_dict['token_embeds_rms'] = token_embeds_rms
        log_dict['token_embeds_valid_rms'] = token_embeds_valid_rms
        if attention_mask is not None:
            valid_text_tokens = jnp.sum(attention_mask.astype(jnp.float32))
            total_text_tokens = jnp.asarray(attention_mask.size, dtype=jnp.float32)
            log_dict['valid_text_tokens'] = valid_text_tokens
            log_dict['text_padding_fraction'] = (
                1.0 - valid_text_tokens / jnp.maximum(total_text_tokens, 1.0)
            )

        # ── Image encoding + optional reconstruction loss ──────────────────
        if images is not None:
            B_img = images.shape[0]

            # Sample prefix_mask BEFORE encoding so it can be forwarded into the
            # encoder when mid-cross-attention masking is enabled.
            _und_mask_active = (
                self.und_use_mask and cache is None and labels is not None
            )
            _need_prefix_mask = (
                labels is not None and (self.recon_use_mask or self.und_use_mask)
            )
            if _need_prefix_mask:
                rng = self.make_rng('gen')
                if self.mask_strategy == 'diy':
                    if mask_token_category_probs is not None:
                        num_visible = self._sample_num_visible_from_category(
                            rng,
                            mask_token_category_probs,
                            K,
                        )
                    else:
                        # Fallback for init/eval paths that do not pass per-sample
                        # category probabilities.
                        num_visible = jax.random.randint(
                            rng, shape=(B_img,), minval=1, maxval=K + 1
                        )
                elif self.mask_strategy == 'uniform':
                    num_visible = jax.random.randint(
                        rng, shape=(B_img,), minval=1, maxval=K + 1
                    )  # (B,)
                elif self.mask_strategy == 'square':
                    x = jax.random.randint(
                        rng, shape=(B_img,), minval=1, maxval=int(K ** 0.5) + 1
                    )  # (B,)
                    num_visible = x * x
                elif self.mask_strategy == 'exp':
                    x = jax.random.randint(
                        rng, shape=(B_img,), minval=1, maxval=math.log2(K) + 1
                    )  # (B,)
                    num_visible = 2 ** x
                else:
                    raise ValueError(f"Unknown mask_strategy: {self.mask_strategy}")

                prefix_mask = jnp.arange(K)[None, :] < num_visible[:, None]  # (B, K)
            else:
                prefix_mask = None
                num_visible = None

            # When enc_cross_attn_split > 0 and und_use_mask is active, apply the
            # nested-dropout mask inside the encoder at the cross-attn midpoint.
            _use_enc_mid_mask = _und_mask_active and (self.enc_cross_attn_split > 0)
            enc_tokens = self.encode_image(
                images,
                vis_mask=prefix_mask if _use_enc_mid_mask else None,
            )                                                        # (B, K, D_enc)
            enc_tokens = constrain_batch_model(enc_tokens)
            enc_tokens_norm = _mean_square(enc_tokens)
            log_dict['enc_tokens_norm'] = enc_tokens_norm
            log_dict['enc_tokens_rms'] = _rms_from_mean_square(enc_tokens_norm)

            # reconstruction loss
            if labels is not None and self.recon_loss_weight > 0.0:

                if self.recon_use_mask:
                    # Build visibility mask: vis_mask[b, k] = True iff k < num_visible[b]
                    vis_mask_recon = prefix_mask

                    # Apply masking in encoder feature space before the decoder.
                    masked_enc = jnp.where(
                        vis_mask_recon[:, :, None], enc_tokens, jnp.zeros_like(enc_tokens)
                    )  # (B, K, D_enc)

                else:  # full reconstruction, no masking
                    masked_enc = enc_tokens
                    vis_mask_recon = None

                # Decoder forward: pass vis_mask so the context ViT only attends
                # to register tokens and visible image-token positions.
                pixel_pred = self.image_encoder.decode(masked_enc, vis_mask_recon)  # (B, T, P²×3)

                # Ground-truth patch pixels (same normalisation as model input)
                patch_targets = patchify(images, self.patch_size)    # (B, T, P²×3)

                loss_recon = recon_mse_loss(pixel_pred, patch_targets)
                log_dict['loss_recon']      = loss_recon

                # Keep only first 6 images to save memory for visualization
                recon_imgs_vis = unpatchify(pixel_pred[:6], self.patch_size, self.image_size)  # (≤6, H, W, 3)
                orig_imgs_vis  = images[:6]                                                    # (≤6, H, W, 3)
            else:
                loss_recon = jnp.zeros(())

            # if und_use_tokens is not None, use the first und_use_tokens tokens
            if self.und_use_tokens is not None:
                enc_tokens = enc_tokens[:, :self.und_use_tokens]
                K = self.und_use_tokens

            # und_use_mask: during training feed the masked encoder tokens to the LM.
            # When enc_cross_attn_split > 0: masking was already done inside the
            # encoder (masked slots are zeros + no second-half cross-attn on them).
            # When enc_cross_attn_split == 0: zero out masked positions here (old behaviour).
            if _und_mask_active and not _use_enc_mid_mask:
                enc_tokens = jnp.where(
                    prefix_mask[:, :, None], enc_tokens, jnp.zeros_like(enc_tokens)
                )

            # Project K encoder tokens to LM hidden dim
            img_embeds = self.projector(enc_tokens)                  # (B, K, D_lm)
            img_embeds = constrain_batch_model(img_embeds)
            img_embeds_pre_scale_norm = _mean_square(img_embeds)
            img_embeds_pre_scale_rms = _rms_from_mean_square(img_embeds_pre_scale_norm)
            log_dict['image_post_connector_scale'] = jnp.asarray(
                self._image_post_connector_scale_value(), dtype=jnp.float32
            )
            log_dict['img_embeds_pre_scale_norm'] = img_embeds_pre_scale_norm
            log_dict['img_embeds_pre_scale_rms'] = img_embeds_pre_scale_rms

            # Optional split: run text through first txt_feature_layer LM blocks
            # to produce "text features" before concatenating with image features.
            _split_txt_cache: dict = {}
            if self.txt_feature_layer > 0:
                _N = self.txt_feature_layer
                _Bt, _T, _ = token_embeds.shape
                _txt_pos = jnp.broadcast_to(
                    jnp.arange(_T, dtype=jnp.int32)[None, :], (_Bt, _T)
                )
                _cs = cache[next(iter(cache))]['k'].shape[1] if cache is not None else None
                _txt_amask = self.make_causal_with_prefix_block(
                    _T, prefix_len, _cs, image_prefix=0
                )
                if cache is not None:
                    _emb_dtype = cache[next(iter(cache))]['v'].dtype
                    token_embeds = token_embeds.astype(_emb_dtype)
                _old_cache = cache or {}
                _x = token_embeds
                for _i in range(_N):
                    _ln = f'layer_{_i}'
                    _lc, _x = self.lm_backbone.blocks[_i](
                        _x, _txt_pos, _old_cache.get(_ln), _txt_amask
                    )
                    _split_txt_cache[_ln] = _lc
                token_embeds = _x  # (B, T_text, D_lm) – text features after N blocks
                if self.stop_gradient_text_features:
                    token_embeds = jax.lax.stop_gradient(token_embeds)
                txt_feature_embeds_norm = _mean_square(token_embeds)
                txt_feature_embeds_rms = _rms_from_mean_square(txt_feature_embeds_norm)
                txt_feature_embeds_valid_norm = _masked_mean_square(
                    token_embeds, attention_mask
                )
                txt_feature_embeds_valid_rms = _rms_from_mean_square(
                    txt_feature_embeds_valid_norm
                )
                log_dict['txt_feature_layer'] = jnp.asarray(_N, dtype=jnp.float32)
                log_dict['txt_feature_embeds_norm'] = txt_feature_embeds_norm
                log_dict['txt_feature_embeds_rms'] = txt_feature_embeds_rms
                log_dict['txt_feature_embeds_valid_rms'] = txt_feature_embeds_valid_rms

            img_embeds, stat_log_dict = self._transform_image_embeds(
                img_embeds,
                token_embeds,
                prefix_len,
                attention_mask,
            )
            log_dict.update(stat_log_dict)
            img_embeds = constrain_batch_model(img_embeds)
            img_embeds_norm = _mean_square(img_embeds)
            img_embeds_rms = _rms_from_mean_square(img_embeds_norm)
            log_dict['img_embeds_norm'] = img_embeds_norm
            log_dict['img_embeds_post_connector_rms'] = img_embeds_rms
            log_dict['img_token_txt_embeds_valid_ratio'] = (
                img_embeds_rms / jnp.maximum(token_embeds_valid_rms, 1e-12)
            )
            log_dict['txt_embeds_valid_img_rms_ratio'] = (
                token_embeds_valid_rms / jnp.maximum(img_embeds_rms, 1e-12)
            )
            if self.txt_feature_layer > 0:
                log_dict['img_txt_feature_valid_rms_ratio'] = (
                    img_embeds_rms / jnp.maximum(txt_feature_embeds_valid_rms, 1e-12)
                )
                log_dict['txt_feature_valid_img_rms_ratio'] = (
                    txt_feature_embeds_valid_rms / jnp.maximum(img_embeds_rms, 1e-12)
                )

            # Prepend image embeddings to text embeddings (or text features)
            T_text_seq = token_embeds.shape[1]
            token_embeds = jnp.concatenate([img_embeds, token_embeds], axis=1)
            token_embeds = constrain_batch_model(token_embeds)

            B, L, _ = token_embeds.shape

            if _und_mask_active:
                # Position IDs: visible image tokens get 0..nv-1; text tokens get
                # nv..nv+T_text-1 so text appears to immediately follow the visible
                # image tokens regardless of how many mask slots sit between them.
                # The masked image positions (nv..K-1) keep sequential IDs but are
                # blocked as attention keys via the custom mask below.
                prefix_total = prefix_len + num_visible               # (B,)
                img_pos = jnp.broadcast_to(
                    jnp.arange(K, dtype=jnp.int32)[None, :], (B, K)
                )                                                     # (B, K)
                txt_pos = (
                    num_visible[:, None]
                    + jnp.arange(T_text_seq, dtype=jnp.int32)[None, :]
                )                                                     # (B, T_text)
                positions = jnp.concatenate([img_pos, txt_pos], axis=1)  # (B, L)

                # Causal + bidirectional-prefix mask; additionally block any key that
                # is a masked image token (sequence index nv <= j < K).
                pid_i  = positions[:, :, None]                        # (B, L, 1)
                pid_j  = positions[:, None, :]                        # (B, 1, L)
                bidir_prefix = jnp.where(
                    self.prompt_causal, num_visible, prefix_total
                )
                pt     = bidir_prefix[:, None, None].astype(jnp.int32)
                j_idx  = jnp.arange(L, dtype=jnp.int32)[None, None, :]
                nv_3   = num_visible[:, None, None]
                is_masked_img_key = (j_idx >= nv_3) & (j_idx < K)    # (B, 1, L)
                attn_mask = (
                    (pid_j <= pid_i) | ((pid_i < pt) & (pid_j < pt))
                ) & ~is_masked_img_key                               # (B, L, L)

                # inputs_mask: 0 for masked image slots, 1 everywhere else
                inputs_mask = jnp.concatenate([
                    prefix_mask.astype(jnp.int32),
                    jnp.ones((B, T_text_seq), dtype=jnp.int32),
                ], axis=1)                                           # (B, L)
            else:
                prefix_total = prefix_len + K
                if cache is not None:
                    cs = cache[list(cache.keys())[0]]['k'].shape[1]
                    attn_mask = self.make_causal_with_prefix_block(
                        L, prefix_total, cs, image_prefix=K
                    )
                else:
                    attn_mask = self.make_causal_with_prefix_block(
                        L, prefix_total, image_prefix=K
                    )
                positions   = jnp.broadcast_to(
                    jnp.arange(L, dtype=jnp.int32)[None, :], (B, L)
                )
                inputs_mask = jnp.ones((B, L), dtype=jnp.int32)

        else:
            # no image mode. just for debug.
            B, L, _ = token_embeds.shape
            prefix_total = prefix_len
            loss_recon = jnp.zeros(())

            if cache is not None:
                cs = cache[list(cache.keys())[0]]['k'].shape[1]
                attn_mask = self.make_causal_with_prefix_block(
                    L, prefix_total, cs, image_prefix=0
                )
            else:
                attn_mask = self.make_causal_with_prefix_block(
                    L, prefix_total, image_prefix=0
                )
            positions   = jnp.broadcast_to(
                jnp.arange(L, dtype=jnp.int32)[None, :], (B, L)
            )
            inputs_mask = jnp.ones((B, L), dtype=jnp.int32)

        # ── LM forward pass ────────────────────────────────────────────────
        if self.txt_feature_layer > 0 and images is not None:
            # Remaining blocks only (first N were already applied above).
            if cache is not None:
                _fwd_dtype = cache[list(cache.keys())[0]]['v'].dtype
                token_embeds = token_embeds.astype(_fwd_dtype)
            _old_cache = cache or {}
            _x = token_embeds
            _rest_cache: dict = {}
            for _i in range(self.txt_feature_layer, len(self.lm_backbone.blocks)):
                _ln = f'layer_{_i}'
                _lc, _x = self.lm_backbone.blocks[_i](
                    _x, positions, _old_cache.get(_ln), attn_mask
                )
                _rest_cache[_ln] = _lc
            out = self.lm_backbone.final_norm(_x)
            out = constrain_batch_model(out)
            new_cache = {**_split_txt_cache, **_rest_cache}
        else:
            if cache is not None:
                dtype = cache[list(cache.keys())[0]]['v'].dtype
                token_embeds = token_embeds.astype(dtype)
            inputs = _Inputs(
                embeddings=token_embeds,
                positions=positions,
                attention_mask=attn_mask,
                inputs_mask=inputs_mask,
            )
            out, new_cache = self.lm_backbone._apply_attention(inputs, cache)
            out = constrain_batch_model(out)

        if labels is None:
            if return_hidden:
                return {'hidden': out, 'cache': new_cache}
            logits = self.lm_backbone.embedder.decode(out)           # (B, L, V)
            return {'logits': logits, 'cache': new_cache}

        assert cache is None
        embedding_table = self.lm_backbone.embedder.input_embedding_table

        # ── Text-only forward pass for CFG-style contrastive loss ──────────
        # When alpha > 0, we run a second forward pass using text embeddings
        # only (no image prefix).  This gives the language prior p(token | text),
        # which is then subtracted from the image-conditioned distribution to
        # amplify image-specific signal.
        #
        #   normalized_logits = cond_logits − α · stop_grad(text_only_logits)
        #   loss_cfg          = CE(normalized_logits, labels)
        #   loss_text_only    = CE(text_only_logits,  labels)
        #
        # Both extra losses are only computed during training (labels is not None)
        # and only when an image is present (otherwise they would be identical to
        # loss_vlm and trivially cancel).
        if (labels is not None and images is not None and self.alpha > 0.0
            and (self.text_only_loss_weight > 0.0 or self.cfg_loss_weight > 0.0)
        ):
            # Reuse the text embeddings already embedded above.
            # After the image–text concatenation, token_embeds has shape
            # (B, K + T_text, D_lm); the text part starts at index K.
            text_embeds = token_embeds[:, K:, :]                     # (B, T_text, D_lm)
            T_text      = text_embeds.shape[1]

            # Text-only branch follows the same prompt_causal policy. With the
            # default prompt_causal=True this is purely causal.
            text_attn_mask = self.make_causal_with_prefix_block(
                T_text, prefix_len, image_prefix=0
            )                                                         # (B, T_text, T_text)
            text_positions = jnp.broadcast_to(
                jnp.arange(T_text, dtype=jnp.int32)[None, :], (B, T_text)
            )                                                         # (B, T_text)

            text_inputs = _Inputs(
                embeddings=text_embeds,
                positions=text_positions,
                attention_mask=text_attn_mask,
                inputs_mask=jnp.ones((B, T_text), dtype=jnp.int32),
            )
            if self.txt_feature_layer > 0:
                # text_embeds already has blocks 0..N-1 applied; only run N..last.
                _N = self.txt_feature_layer
                _x = text_embeds
                for _i in range(_N, len(self.lm_backbone.blocks)):
                    _ln = f'layer_{_i}'
                    _, _x = self.lm_backbone.blocks[_i](_x, text_positions, None, text_attn_mask)
                text_out = self.lm_backbone.final_norm(_x)
            else:
                text_out, _ = self.lm_backbone._apply_attention(text_inputs, None)
            text_out = constrain_batch_model(text_out)

            # Loss 1 – standard NTP on the text-only (unconditional) branch.
            if self.text_only_loss_weight > 0.0:
                loss_text_only, _ = token_xent_loss_from_hidden(
                    text_out,
                    embedding_table,
                    labels,
                    final_logit_softcap=self.final_logit_softcap,
                )
            else:
                loss_text_only = jnp.zeros(())

            # Loss 2 – CFG-style contrastive loss.
            # out[:, K:, :] are the image-conditioned text hidden states.
            # Subtracting the stop-grad text prior sharpens image-specific predictions.
            if self.cfg_loss_weight > 0.0:
                cond_text_hidden = out[:, K:, :]                     # (B, T_text, D_lm)
                loss_cfg, _ = token_xent_loss_from_hidden(
                    cond_text_hidden,
                    embedding_table,
                    labels,
                    final_logit_softcap=self.final_logit_softcap,
                    subtract_hidden=text_out,
                    subtract_alpha=self.alpha,
                )
            else:
                loss_cfg = jnp.zeros(())

            log_dict['loss_text_only'] = loss_text_only
            log_dict['loss_cfg']       = loss_cfg
        else:
            loss_text_only = jnp.zeros(())
            loss_cfg       = jnp.zeros(())

        # ── Combined loss ──────────────────────────────────────────────────
        if labels is not None:
            if images is not None:
                # Only text positions have labels; avoid materializing logits for
                # the image prefix positions.
                lm_hidden = out[:, K:, :]
                labels_for_loss = labels
            else:
                lm_hidden = out
                labels_for_loss = labels

            loss_vlm, pred_ids = token_xent_loss_from_hidden(
                lm_hidden,
                embedding_table,
                labels_for_loss,
                final_logit_softcap=self.final_logit_softcap,
            )

            valid = labels_for_loss != -100
            valid_count = valid.sum()
            acc = (
                jnp.sum((pred_ids == labels_for_loss) * valid)
                / jnp.maximum(valid_count, 1)
            )
            log_dict['loss_vlm'] = loss_vlm
            log_dict['acc']      = acc
            log_dict['valid_tokens'] = valid_count.astype(jnp.float32)
            log_dict['valid_tokens_per_sample'] = (
                valid_count.astype(jnp.float32) / jnp.maximum(B, 1)
            )

            loss_total = (
                self.vlm_loss_weight       * loss_vlm
                + self.text_only_loss_weight * loss_text_only
                + self.cfg_loss_weight       * loss_cfg
                + self.recon_loss_weight     * loss_recon
            )

            debug = {
                'attn_mask': attn_mask,
                'labels':    labels_for_loss,
                'preds':     pred_ids,
                'input_ids': input_ids,
            }
            if self.recon_loss_weight > 0.0:
                debug['recon_imgs'] = recon_imgs_vis  # (≤6, H, W, 3), normalised [-1, 1]
                debug['orig_imgs']  = orig_imgs_vis   # (≤6, H, W, 3), normalised [-1, 1]
            return loss_total, log_dict, debug

    # ------------------------------------------------------------------
    # Reconstruction  (encoder + decoder, no LM)
    # ------------------------------------------------------------------

    def reconstruct(self, images: Array, num_visible: int) -> Array:
        """Reconstruct image patches using only the first num_visible encoder tokens.

        Tokens at positions [num_visible, K) are replaced with the learnable
        mask token before the decoder forward pass, mirroring the nested-dropout
        masking used during training.

        Args:
            images:      (B, H, W, 3) – normalised input images
            num_visible: how many encoder tokens to reveal (1 ≤ num_visible ≤ K)

        Returns:
            (B, T, patch_size²×3) patch predictions in the same normalised pixel
            space as the input.
        """
        enc_tokens = self.encode_image(images)          # (B, K, D_enc)
        B = enc_tokens.shape[0]
        K = self.num_learnable_tokens

        vis_mask = jnp.arange(K)[None, :] < num_visible  # (B, K)
        masked_enc = jnp.where(vis_mask[:, :, None], enc_tokens, jnp.zeros_like(enc_tokens))

        return self.image_encoder.decode(masked_enc, vis_mask)  # (B, T, P²×3)

    # ------------------------------------------------------------------
    # Greedy generation  (decoder not used at inference)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: Array,
        prefix_len: Array,
        images: Optional[Array] = None,
        max_new_tokens: int = 64,
    ) -> Array:
        """Greedy auto-regressive generation. Returns (B, max_new_tokens) int32."""
        B        = prompt_ids.shape[0]
        T_prompt = prompt_ids.shape[1]
        prefix_len = jnp.asarray(prefix_len, dtype=jnp.int32)

        K = self.num_learnable_tokens if images is not None else 0
        if self.und_use_tokens is not None:
            K = self.und_use_tokens
        
        prefix_total    = prefix_len + K
        step_pos_init   = prefix_total[:, None]
        step_pos_init_txt = prefix_len[:, None]   # text-layer positions (no K offset)
        max_total_len   = T_prompt + max_new_tokens + K

        cache = self.lm_backbone.init_cache(
            batch_size=B, dtype=jnp.bfloat16, cache_length=max_total_len
        )

        # Prefill. Decode only the last prompt hidden state; decoding the full
        # prefill would materialize [B, prompt_len, vocab] logits.
        out_dict = self(
            input_ids=prompt_ids, images=images,
            prefix_len=prefix_len, cache=cache, use_cache=False,
            return_hidden=True,
        )
        hidden_at_last = jnp.take_along_axis(
            out_dict['hidden'],
            (prefix_total - 1)[:, None, None],
            axis=1,
        ).squeeze(1)
        logits_at_last = self.lm_backbone.embedder.decode(hidden_at_last)  # (B, V)
        first_token = jnp.argmax(logits_at_last, axis=-1, keepdims=True)

        tokens_out = jnp.zeros((B, max_new_tokens), dtype=jnp.int32)
        tokens_out = tokens_out.at[:, 0].set(first_token.squeeze(-1))

        def cond_fn(carry):
            _, _, step, _ = carry
            return step < max_new_tokens

        def body_fn(m, carry):
            curr_tok, curr_cache, step, out = carry
            fk = list(curr_cache.keys())[0]
            emb = m.lm_backbone.embedder.encode(curr_tok).astype(
                curr_cache[fk]['v'].dtype
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
            lm_out, next_cache = m.lm_backbone._apply_attention(step_inputs, curr_cache)
            next_tok = jnp.argmax(
                m.lm_backbone.embedder.decode(lm_out)[:, -1, :], axis=-1, keepdims=True
            )
            return (next_tok, next_cache, step + 1, out.at[:, step].set(next_tok.squeeze(-1)))

        def body_fn_split(m, carry):
            """body_fn for txt_feature_layer > 0: split text vs full-seq forward."""
            curr_tok, curr_cache, step, out = carry
            fk = list(curr_cache.keys())[0]
            emb = m.lm_backbone.embedder.encode(curr_tok).astype(
                curr_cache[fk]['v'].dtype
            )
            _N = m.txt_feature_layer
            j = jnp.arange(max_total_len)[None, None, :]
            # Text-layer mask: attend to text prefix + already-decoded text
            _txt_mask = (
                (j < prefix_len[:, None, None])
                | ((j >= T_prompt) & (j < T_prompt + step))
            )
            # Remaining-layer mask: attend to img+text prefix + decoded full features
            _full_mask = (
                (j < prefix_total[:, None, None])
                | ((j >= T_prompt + K) & (j < T_prompt + K + step))
            )
            # -- text layers 0..N-1 --
            _txt_pos = step_pos_init_txt + step
            _x = emb
            _new_cache: dict = {}
            for _i in range(_N):
                _ln = f'layer_{_i}'
                _lc, _x = m.lm_backbone.blocks[_i](
                    _x, _txt_pos, curr_cache.get(_ln), _txt_mask
                )
                _new_cache[_ln] = _lc
            # -- remaining layers N..last --
            _full_pos = step_pos_init + step
            for _i in range(_N, len(m.lm_backbone.blocks)):
                _ln = f'layer_{_i}'
                _lc, _x = m.lm_backbone.blocks[_i](
                    _x, _full_pos, curr_cache.get(_ln), _full_mask
                )
                _new_cache[_ln] = _lc
            lm_out = m.lm_backbone.final_norm(_x)
            next_tok = jnp.argmax(
                m.lm_backbone.embedder.decode(lm_out)[:, -1, :], axis=-1, keepdims=True
            )
            return (next_tok, _new_cache, step + 1, out.at[:, step].set(next_tok.squeeze(-1)))

        _use_split = self.txt_feature_layer > 0 and images is not None
        _, _, _, all_tokens = jax.lax.while_loop(
            cond_fn,
            (lambda c: body_fn_split(self, c)) if _use_split else (lambda c: body_fn(self, c)),
            (first_token, out_dict['cache'], 1, tokens_out),
        )
        return all_tokens

    # ------------------------------------------------------------------
    # Beam-search generation
    # ------------------------------------------------------------------

    def generate_beam_search(
        self,
        prompt_ids: Array,
        prefix_len: Array,
        images: Optional[Array] = None,
        beam_size: int = 3,
        max_new_tokens: int = 64,
    ) -> Array:
        """Beam-search generation. Returns (B, max_new_tokens) int32."""
        B        = prompt_ids.shape[0]
        T_prompt = prompt_ids.shape[1]
        prefix_len = jnp.asarray(prefix_len, dtype=jnp.int32)

        K = self.num_learnable_tokens if images is not None else 0
        if self.und_use_tokens is not None:
            K = self.und_use_tokens
        
        prefix_total  = prefix_len + K
        max_total_len = T_prompt + max_new_tokens + K

        if max_new_tokens <= 0:
            return jnp.zeros((B, 0), dtype=jnp.int32)

        cache_single = self.lm_backbone.init_cache(
            batch_size=B, dtype=jnp.bfloat16, cache_length=max_total_len
        )
        out_dict = self(
            input_ids=prompt_ids, images=images,
            prefix_len=prefix_len, cache=cache_single, use_cache=False,
            return_hidden=True,
        )

        hidden_at_last = jnp.take_along_axis(
            out_dict['hidden'],
            (prefix_total - 1)[:, None, None],
            axis=1,
        ).squeeze(1)
        logits_at_last = self.lm_backbone.embedder.decode(hidden_at_last)
        top_scores, top_tokens = jax.lax.top_k(
            jax.nn.log_softmax(logits_at_last), beam_size
        )

        cache_tiled  = jax.tree.map(lambda x: jnp.repeat(x, beam_size, axis=0), out_dict['cache'])
        curr_tokens  = top_tokens.reshape(-1, 1)
        beam_scores  = top_scores.reshape(-1)
        beam_prefix  = jnp.repeat(prefix_total, beam_size, axis=0)
        beam_prefix_len = jnp.repeat(prefix_len, beam_size, axis=0)
        step_pos_init = beam_prefix[:, None]
        step_pos_init_txt = beam_prefix_len[:, None]   # text-layer positions (no K offset)
        has_eos      = (curr_tokens == self.eos_id).reshape(-1)

        history = jnp.zeros((B * beam_size, max_new_tokens), dtype=jnp.int32)
        history = history.at[:, 0].set(curr_tokens.squeeze(-1))

        def cond_fn(carry):
            _, _, _, _, _, _, step = carry
            return step < max_new_tokens

        def body_fn(m, carry):
            tokens, curr_cache, b_scores, hist, hit_eos, curr_prefix, step = carry
            fk = list(curr_cache.keys())[0]
            emb = m.lm_backbone.embedder.encode(tokens).astype(
                curr_cache[fk]['v'].dtype
            )
            j = jnp.arange(max_total_len)[None, None, :]
            mask = (
                (j < curr_prefix[:, None, None])
                | ((j >= T_prompt + K) & (j < T_prompt + K + step))
            )
            step_inputs = _Inputs(
                embeddings=emb,
                positions=step_pos_init + step,
                attention_mask=mask,
                inputs_mask=jnp.ones((B * beam_size, 1), dtype=emb.dtype),
            )
            lm_out, next_cache = m.lm_backbone._apply_attention(step_inputs, curr_cache)
            log_probs  = jax.nn.log_softmax(
                m.lm_backbone.embedder.decode(lm_out)[:, -1, :]
            )
            vocab_size = log_probs.shape[-1]

            eos_only  = jnp.full_like(log_probs, -1e9).at[:, self.eos_id].set(0.0)
            eff_lp    = jnp.where(hit_eos[:, None], eos_only, log_probs)
            total     = (
                eff_lp.reshape(B, beam_size, -1) + b_scores.reshape(B, beam_size, 1)
            ).reshape(B, -1)

            next_scores, next_idx = jax.lax.top_k(total, beam_size)
            parent = next_idx // vocab_size
            ids    = next_idx % vocab_size

            off          = jnp.arange(B)[:, None] * beam_size
            flat_parents = (parent + off).reshape(-1)

            reshuffled_hist   = hist[flat_parents].at[:, step].set(ids.reshape(-1))
            reshuffled_cache  = jax.tree.map(lambda x: x[flat_parents], next_cache)
            reshuffled_prefix = curr_prefix[flat_parents]
            new_hit_eos       = hit_eos[flat_parents] | (ids.reshape(-1) == self.eos_id)

            return (
                ids.reshape(-1, 1), reshuffled_cache, next_scores.reshape(-1),
                reshuffled_hist, new_hit_eos, reshuffled_prefix, step + 1,
            )

        def body_fn_split(m, carry):
            """body_fn for beam search with txt_feature_layer > 0."""
            tokens, curr_cache, b_scores, hist, hit_eos, curr_prefix, step = carry
            fk = list(curr_cache.keys())[0]
            emb = m.lm_backbone.embedder.encode(tokens).astype(
                curr_cache[fk]['v'].dtype
            )
            _N = m.txt_feature_layer
            j = jnp.arange(max_total_len)[None, None, :]
            _txt_mask = (
                (j < beam_prefix_len[:, None, None])
                | ((j >= T_prompt) & (j < T_prompt + step))
            )
            _full_mask = (
                (j < curr_prefix[:, None, None])
                | ((j >= T_prompt + K) & (j < T_prompt + K + step))
            )
            _txt_pos = step_pos_init_txt + step
            _x = emb
            _new_cache: dict = {}
            for _i in range(_N):
                _ln = f'layer_{_i}'
                _lc, _x = m.lm_backbone.blocks[_i](
                    _x, _txt_pos, curr_cache.get(_ln), _txt_mask
                )
                _new_cache[_ln] = _lc
            _full_pos = step_pos_init + step
            for _i in range(_N, len(m.lm_backbone.blocks)):
                _ln = f'layer_{_i}'
                _lc, _x = m.lm_backbone.blocks[_i](
                    _x, _full_pos, curr_cache.get(_ln), _full_mask
                )
                _new_cache[_ln] = _lc
            lm_out = m.lm_backbone.final_norm(_x)
            log_probs  = jax.nn.log_softmax(
                m.lm_backbone.embedder.decode(lm_out)[:, -1, :]
            )
            vocab_size = log_probs.shape[-1]

            eos_only  = jnp.full_like(log_probs, -1e9).at[:, self.eos_id].set(0.0)
            eff_lp    = jnp.where(hit_eos[:, None], eos_only, log_probs)
            total     = (
                eff_lp.reshape(B, beam_size, -1) + b_scores.reshape(B, beam_size, 1)
            ).reshape(B, -1)

            next_scores, next_idx = jax.lax.top_k(total, beam_size)
            parent = next_idx // vocab_size
            ids    = next_idx % vocab_size

            off          = jnp.arange(B)[:, None] * beam_size
            flat_parents = (parent + off).reshape(-1)

            reshuffled_hist   = hist[flat_parents].at[:, step].set(ids.reshape(-1))
            reshuffled_cache  = jax.tree.map(lambda x: x[flat_parents], _new_cache)
            reshuffled_prefix = curr_prefix[flat_parents]
            new_hit_eos       = hit_eos[flat_parents] | (ids.reshape(-1) == self.eos_id)

            return (
                ids.reshape(-1, 1), reshuffled_cache, next_scores.reshape(-1),
                reshuffled_hist, new_hit_eos, reshuffled_prefix, step + 1,
            )

        _use_split = self.txt_feature_layer > 0 and images is not None
        _, _, _, final_hist, _, _, _ = jax.lax.while_loop(
            cond_fn,
            (lambda c: body_fn_split(self, c)) if _use_split else (lambda c: body_fn(self, c)),
            (curr_tokens, cache_tiled, beam_scores, history, has_eos, beam_prefix, 1),
        )
        return final_hist[jnp.arange(B) * beam_size]
