"""PrefixMAE: custom image encoder-decoder model.

Architecture
------------
Encoder
  - Patchify image (B, H, W, 3) → flat patch tokens (B, T, D),  T = (H/P)²
  - Stage 1 (6 layers):  self-attention on T patch tokens
  - Stage 2 (12 layers): L learnable 1-D tokens cross-attend to T patch tokens
                         (L is query, T is key/value) → (B, L, D)
  - Stage 3 (6 layers):  self-attention on L 1-D tokens → (B, L, D_enc)

Decoder  (MAE-style — reconstructs original image patch pixels)
  - Sample i ~ Uniform[1, L] at training time (number of "visible" tokens)
  - Project first i encoder tokens to decoder dim; fill positions i..L-1 with a
    learnable mask token → L tokens total.
  - Add L abstract positional embeddings.
  - Run N_dec bidirectional ViT blocks: (B, L, D_dec)  ← "context"
  - Cross-attention pixel decoder:
      · T learnable patch-position query embeddings (one per spatial patch)
      · Each query attends to ALL L context tokens via cross-attention
      · One cross-attention block (pre-norm) followed by a feed-forward layer
  - Linear head: (B, T, D_dec) → (B, T, P²×3)  (patch pixel predictions)

Loss
  - patchify() extracts raw pixel patches from the original image: (B, T, P²×3)
  - recon_mse_loss() = mean MSE over all T patches (no output masking;
    the difficulty is controlled entirely by the number of visible context tokens i)
"""
from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, use_bias=True, name='fc1')(x)
        x = nn.gelu(x)
        x = nn.Dense(self.output_dim, use_bias=True, name='fc2')(x)
        return x


class Attention(nn.Module):
    """Multi-head *self*-attention with float32 softmax for numerical stability."""
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,            # (B, T, D)
        mask: Optional[jnp.ndarray] = None,  # (B, T, T) bool, True=attend
    ) -> jnp.ndarray:
        B, T, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        inner = H * Dh

        qkv = nn.Dense(3 * inner, use_bias=False, name='qkv')(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def heads(z: jnp.ndarray) -> jnp.ndarray:
            return z.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)  # (B,H,T,Dh)

        q, k, v = heads(q), heads(k), heads(v)
        attn = jnp.einsum('bhqd,bhkd->bhqk',
                          q.astype(jnp.float32),
                          k.astype(jnp.float32)) * (Dh ** -0.5)
        if mask is not None:
            attn = jnp.where(mask[:, None, :, :], attn, -1e9)
        attn = jax.nn.softmax(attn, axis=-1).astype(x.dtype)
        out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, inner)
        return nn.Dense(D, use_bias=True, name='out_proj')(out)


class CrossAttention(nn.Module):
    """Multi-head *cross*-attention: query from one sequence, key/value from another."""
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,      # (B, Tq, D)
        context: jnp.ndarray,    # (B, Tk, D)
        qmask: Optional[jnp.ndarray] = None,  # (B, Tq) bool, True=run attention; False=zero row
    ) -> jnp.ndarray:            # (B, Tq, D)
        B, Tq, D = query.shape
        Tk = context.shape[1]
        H, Dh = self.num_heads, self.head_dim
        inner = H * Dh

        q = nn.Dense(inner, use_bias=False, name='q')(query)
        k = nn.Dense(inner, use_bias=False, name='k')(context)
        v = nn.Dense(inner, use_bias=False, name='v')(context)

        def heads(z: jnp.ndarray, T: int) -> jnp.ndarray:
            return z.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

        q, k, v = heads(q, Tq), heads(k, Tk), heads(v, Tk)
        attn = jnp.einsum('bhqd,bhkd->bhqk',
                          q.astype(jnp.float32),
                          k.astype(jnp.float32)) * (Dh ** -0.5)
        attn = jax.nn.softmax(attn, axis=-1).astype(query.dtype)
        if qmask is not None:
            # No read from any patch for dropped query rows (equiv. row-wise attention mask)
            attn = attn * qmask[:, None, :, None].astype(attn.dtype)
        out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, Tq, inner)
        out = nn.Dense(D, use_bias=True, name='out_proj')(out)
        if qmask is not None:
            out = out * qmask[:, :, None].astype(out.dtype)
        return out


class TransformerBlock(nn.Module):
    """Pre-LN self-attention ViT block."""
    num_heads: int
    head_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        D = x.shape[-1]
        h = nn.LayerNorm(name='norm1')(x)
        h = Attention(self.num_heads, self.head_dim, name='attn')(h, mask)
        x = x + h
        h = nn.LayerNorm(name='norm2')(x)
        h = MLP(self.mlp_dim, D, name='mlp')(h)
        return x + h


class CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention block (query attends to context) + MLP residual."""
    num_heads: int
    head_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,    # (B, Tq, D)
        context: jnp.ndarray,  # (B, Tk, D)
        qmask: Optional[jnp.ndarray] = None,  # (B, Tq) bool, True=participate
    ) -> jnp.ndarray:          # (B, Tq, D)
        D = query.shape[-1]
        # Cross-attention with pre-norm on both query and context
        h = CrossAttention(self.num_heads, self.head_dim, name='cross_attn')(
            nn.LayerNorm(name='q_norm')(query),
            nn.LayerNorm(name='k_norm')(context),
            qmask,
        )
        query = query + h
        # Feed-forward
        h = MLP(self.mlp_dim, D, name='mlp')(nn.LayerNorm(name='ff_norm')(query))
        if qmask is not None:
            h = h * qmask[:, :, None].astype(h.dtype)
        return query + h


# ---------------------------------------------------------------------------
# Image Encoder
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """Compress an image into L abstract 1-D token embeddings via three stages.

    Processing order:
      patch_embed(image) → add patch positional embeddings         → (B, T, D)
      Stage 1 (num_patch_sa_layers):   self-attention on T patch tokens
      Stage 2 (num_cross_attn_layers): L learnable tokens (query) cross-attend
                                       to T patch tokens (key/value) → (B, L, D)
      Stage 3 (num_token_sa_layers):   self-attention on L 1-D tokens
      LayerNorm → return (B, L, D)

    Mid-masking (enc_cross_attn_split > 0)
    ----------------------------------------
    When vis_mask (B, L) is supplied at call time AND enc_cross_attn_split > 0,
    a nested-dropout mask is applied after the first enc_cross_attn_split cross-
    attention layers:
      · masked positions are zero-filled (dropped, not a learnable token)
      · Stage 2 second half: cross-attn uses a query mask so dropped slots
        do not read from image patches; MLP residuals there are also zeroed
      · Stage 3 self-attention further blocks dropped positions as both query
        and key so visible tokens are never contaminated
    """
    patch_size: int = 16
    image_size: int = 224
    hidden_dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    mlp_dim: int = 3072
    num_patch_sa_layers: int = 6     # Stage 1: self-attn on patch tokens
    num_cross_attn_layers: int = 12  # Stage 2: cross-attn L→T
    num_token_sa_layers: int = 6     # Stage 3: self-attn on 1-D tokens
    num_learnable_tokens: int = 256  # L
    feature_dim: int = 256
    enc_cross_attn_split: int = 0    # apply mask after this many cross-attn layers
                                     # (0 = disabled; typically num_cross_attn_layers//2)

    @nn.compact
    def __call__(
        self,
        images: jnp.ndarray,
        vis_mask: Optional[jnp.ndarray] = None,  # (B, L) bool, True=visible
    ) -> jnp.ndarray:
        """images: (B, H, W, 3) → (B, L, feature_dim)

        vis_mask is only honoured when enc_cross_attn_split > 0; it triggers
        mid-masking at the split point.
        """
        B = images.shape[0]
        T = (self.image_size // self.patch_size) ** 2
        L = self.num_learnable_tokens

        # ── Patch embedding ────────────────────────────────────────────────
        x = nn.Conv(
            self.hidden_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            use_bias=True,
            name='patch_embed',
        )(images)                                       # (B, H/P, W/P, D)
        x = x.reshape(B, T, self.hidden_dim)

        patch_pos = self.param(
            'patch_pos_embed',
            nn.initializers.normal(stddev=0.02),
            (1, T, self.hidden_dim),
        )
        x = x + patch_pos                              # (B, T, D)

        # ── Stage 1: self-attention on patch tokens ────────────────────────
        for i in range(self.num_patch_sa_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                name=f'patch_sa_block_{i}',
            )(x)                                       # (B, T, D)

        # ── Learnable 1-D tokens (initialised once, used as cross-attn queries)
        learnable = self.param(
            'learnable_tokens',
            nn.initializers.normal(stddev=0.02),
            (1, L, self.hidden_dim),
        )
        learnable_pos = self.param(
            'learnable_pos_embed',
            nn.initializers.normal(stddev=0.02),
            (1, L, self.hidden_dim),
        )
        tokens = jnp.broadcast_to(
            learnable + learnable_pos,
            (B, L, self.hidden_dim),
        )                                              # (B, L, D)

        # ── Mid-masking setup ──────────────────────────────────────────────
        _do_mid_mask = (
            vis_mask is not None
            and self.enc_cross_attn_split > 0
        )
        split = (
            min(self.enc_cross_attn_split, self.num_cross_attn_layers)
            if _do_mid_mask else self.num_cross_attn_layers
        )

        # ── Stage 2 first half: cross-attention layers 0 .. split-1 ───────
        for i in range(split):
            tokens = CrossAttentionBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                name=f'cross_attn_block_{i}',
            )(tokens, x)                               # (B, L, D)

        # ── Apply mask at midpoint ─────────────────────────────────────────
        if _do_mid_mask:
            tokens = jnp.where(vis_mask[:, :, None], tokens, jnp.zeros_like(tokens))  # (B, L, D)

        # ── Stage 2 second half: cross-attention layers split .. end ───────
        for i in range(split, self.num_cross_attn_layers):
            tokens = CrossAttentionBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                name=f'cross_attn_block_{i}',
            )(
                tokens, x, vis_mask if _do_mid_mask else None
            )  # (B, L, D) — qmask: visible rows attend to patches, dropped = no CA + no MLP add

        # ── Stage 3: self-attention on 1-D tokens ─────────────────────────
        # When mid-masking: block masked positions as both query and key so
        # visible tokens cannot be contaminated by the mask-token slots.
        if _do_mid_mask:
            sa_mask = vis_mask[:, :, None] & vis_mask[:, None, :]  # (B, L, L)
        else:
            sa_mask = None

        for i in range(self.num_token_sa_layers):
            tokens = TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                name=f'token_sa_block_{i}',
            )(tokens, sa_mask)                         # (B, L, D)

        # final linear projection to feature dim
        tokens = nn.Dense(self.feature_dim, use_bias=False, name='final_proj')(tokens)

        return nn.LayerNorm(name='norm')(tokens)       # (B, L, feature_dim)


# ---------------------------------------------------------------------------
# Image Decoder
# ---------------------------------------------------------------------------

class ImageDecoder(nn.Module):
    """MAE-style decoder that reconstructs the original image patch pixels.

    Stage 1 — context ViT
      Project already-masked tokens to D_dec → prepend R learnable register tokens
      → add K abstract positional embeddings to image-token slots
      → N_dec bidirectional ViT blocks with attention mask that restricts keys
        to [register tokens] ∪ [visible image-token positions]
      → (B, R+K, D_dec)

    Stage 2 — cross-attention pixel decoder
      T learnable patch-position query embeddings (B, T, D_dec)
      × (R+K) context tokens  →  CrossAttentionBlock  →  (B, T, D_dec)

    Stage 3 — pixel head
      Linear: (B, T, D_dec) → (B, T, patch_size² × 3)

    NOTE: this module does NOT perform masking. The caller must replace masked
    encoder-token positions with the learnable mask token before calling.
    The caller should also supply vis_mask so that masked positions are excluded
    from the self-attention key set (only register + visible tokens are keys).
    """
    hidden_dim: int = 512
    num_heads: int = 8
    head_dim: int = 64
    mlp_dim: int = 2048
    num_layers: int = 6
    num_tokens: int = 256           # K
    num_patches: int = 196          # T = (image_size / patch_size)²
    patch_pixel_dim: int = 768      # patch_size² × 3  (e.g. 16²×3 = 768)
    num_registers: int = 4          # R — learnable in-context (register) tokens

    @nn.compact
    def __call__(
        self,
        tokens: jnp.ndarray,                    # (B, K, D_enc)  — masking already applied by caller
        vis_mask: Optional[jnp.ndarray] = None, # (B, K) bool, True = visible
    ) -> jnp.ndarray:                           # (B, T, patch_pixel_dim)
        B = tokens.shape[0]
        K = self.num_tokens
        T = self.num_patches
        R = self.num_registers

        # ── Stage 1: context ViT ───────────────────────────────────────────
        context = nn.Dense(
            self.hidden_dim, use_bias=True, name='input_proj'
        )(tokens)                                       # (B, K, D_dec)

        abs_pos = self.param(
            'abs_pos_embed',
            nn.initializers.normal(stddev=0.02),
            (1, K, self.hidden_dim),
        )
        context = context + abs_pos                     # (B, K, D_dec)

        # Prepend R learnable register tokens (no positional embedding needed;
        # they serve as free-floating in-context memory).
        registers = self.param(
            'register_tokens',
            nn.initializers.normal(stddev=0.02),
            (1, R, self.hidden_dim),
        )
        reg = jnp.broadcast_to(registers, (B, R, self.hidden_dim))
        context = jnp.concatenate([reg, context], axis=1)  # (B, R+K, D_dec)

        # Build self-attention mask: every position may attend to register
        # slots (indices 0..R-1) and to visible image-token slots (indices R..R+K-1).
        # Shape: (B, R+K, R+K), True = allowed.
        if vis_mask is not None:
            # key_visible[b, j] = True iff position j is always-visible (register)
            #                        or is a visible image token.
            # vis_mask may arrive as (1, K) when num_visible is a scalar (e.g. in
            # reconstruct()), so broadcast to (B, K) before concatenating.
            vis_mask_bcast = jnp.broadcast_to(vis_mask, (B, K))
            key_visible = jnp.concatenate(
                [jnp.ones((B, R), dtype=jnp.bool_), vis_mask_bcast], axis=1
            )  # (B, R+K)
            sa_mask = key_visible[:, None, :]  # (B, 1, R+K) → broadcasts to (B, R+K, R+K)
        else:
            sa_mask = None

        for i in range(self.num_layers):
            context = TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                name=f'block_{i}',
            )(context, sa_mask)

        context = nn.LayerNorm(name='context_norm')(context)   # (B, R+K, D_dec)

        # ── Stage 2: cross-attention pixel decoder ─────────────────────────
        # T learnable patch-position query embeddings; attend to all R+K context tokens.
        patch_queries = self.param(
            'patch_query_embed',
            nn.initializers.normal(stddev=0.02),
            (1, T, self.hidden_dim),
        )
        queries = jnp.broadcast_to(patch_queries, (B, T, self.hidden_dim))

        queries = CrossAttentionBlock(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            mlp_dim=self.mlp_dim,
            name='pixel_cross_attn',
        )(queries, context)                             # (B, T, D_dec)

        # ── Stage 3: pixel head ────────────────────────────────────────────
        queries = nn.LayerNorm(name='pixel_norm')(queries)
        pixel_pred = nn.Dense(
            self.patch_pixel_dim, use_bias=True, name='pixel_head'
        )(queries)                                      # (B, T, P²×3)

        return pixel_pred


# ---------------------------------------------------------------------------
# PrefixMAE: combined Encoder + Decoder wrapper
# ---------------------------------------------------------------------------

class PrefixMAE(nn.Module):
    """Encoder-decoder image model (encoder + MAE-style image-reconstruction decoder).

    Typical training usage
    ----------------------
        enc_tokens = model.encode(images)                  # (B, K, D_enc)

        # Masking is the caller's responsibility (see PaliGemmaEncDec.__call__):
        #   1. sample num_visible (B,) ~ Uniform[1, K]
        #   2. vis_mask = arange(K)[None,:] < num_visible[:,None]   # (B, K)
        #   3. masked_enc = where(vis_mask[:,:,None], enc_tokens, mask_token_tiled)
        pixel_pred = model.decode(masked_enc)              # (B, T, P²×3)

        patch_targets = patchify(images, patch_size)       # (B, T, P²×3)
        loss = recon_mse_loss(pixel_pred, patch_targets)

    Inference (encoder only)
    ------------------------
        enc_tokens = model.encode(images)                  # (B, K, D_enc)
    """
    # ── Encoder ────────────────────────────────────────────────────────────
    patch_size: int = 16
    image_size: int = 224
    enc_hidden_dim: int = 768
    enc_num_heads: int = 12
    enc_head_dim: int = 64
    enc_mlp_dim: int = 3072
    enc_num_patch_sa_layers: int = 6    # Stage 1: self-attn on patch tokens
    enc_num_cross_attn_layers: int = 12 # Stage 2: cross-attn L→T
    enc_num_token_sa_layers: int = 6    # Stage 3: self-attn on 1-D tokens
    num_learnable_tokens: int = 256     # L
    feature_dim: int = 256
    enc_cross_attn_split: int = 0       # 0 = no mid-masking; typically enc_num_cross_attn_layers//2

    # ── Decoder ────────────────────────────────────────────────────────────
    dec_hidden_dim: int = 512
    dec_num_heads: int = 8
    dec_head_dim: int = 64
    dec_mlp_dim: int = 2048
    dec_num_layers: int = 6
    use_decoder: bool = True

    def setup(self) -> None:
        num_patches = (self.image_size // self.patch_size) ** 2
        patch_pixel_dim = self.patch_size ** 2 * 3

        self.encoder = ImageEncoder(
            patch_size=self.patch_size,
            image_size=self.image_size,
            hidden_dim=self.enc_hidden_dim,
            num_heads=self.enc_num_heads,
            head_dim=self.enc_head_dim,
            mlp_dim=self.enc_mlp_dim,
            num_patch_sa_layers=self.enc_num_patch_sa_layers,
            num_cross_attn_layers=self.enc_num_cross_attn_layers,
            num_token_sa_layers=self.enc_num_token_sa_layers,
            num_learnable_tokens=self.num_learnable_tokens,
            feature_dim=self.feature_dim,
            enc_cross_attn_split=self.enc_cross_attn_split,
        )
        if self.use_decoder:
            self.decoder = ImageDecoder(
                hidden_dim=self.dec_hidden_dim,
                num_heads=self.dec_num_heads,
                head_dim=self.dec_head_dim,
                mlp_dim=self.dec_mlp_dim,
                num_layers=self.dec_num_layers,
                num_tokens=self.num_learnable_tokens,
                num_patches=num_patches,
                patch_pixel_dim=patch_pixel_dim,
            )

    def encode(
        self,
        images: jnp.ndarray,
        vis_mask: Optional[jnp.ndarray] = None,  # (B, L) bool; enables mid-masking
    ) -> jnp.ndarray:
        """(B, H, W, 3) → (B, K, feature_dim)

        Pass vis_mask to activate mid-cross-attention masking (only has effect
        when enc_cross_attn_split > 0).
        """
        return self.encoder(images, vis_mask)

    def decode(
        self,
        tokens: jnp.ndarray,                    # (B, K, D_enc)  — masking already applied by caller
        vis_mask: Optional[jnp.ndarray] = None, # (B, K) bool, True = visible
    ) -> jnp.ndarray:                           # (B, T, patch_size²×3)
        """Decoder-only forward pass. Masking must be applied by the caller.

        vis_mask restricts the decoder context ViT's self-attention keys to the
        register tokens and the visible (non-masked) image-token positions.
        """
        return self.decoder(tokens, vis_mask)


# ---------------------------------------------------------------------------
# Patch utilities & reconstruction loss
# ---------------------------------------------------------------------------

def patchify(images: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    """Extract non-overlapping patches from an image in raster order.

    Args:
        images:     (B, H, W, 3)  float, already normalised (e.g. to [-1, 1])
        patch_size: P

    Returns:
        (B, T, P²×3)  where T = (H/P) × (W/P), row-major (raster) order
    """
    B, H, W, C = images.shape
    P = patch_size
    h, w = H // P, W // P
    x = images.reshape(B, h, P, w, P, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)   # (B, h, w, P, P, C)
    return x.reshape(B, h * w, P * P * C)


def unpatchify(patches: jnp.ndarray, patch_size: int, image_size: int) -> jnp.ndarray:
    """Reconstruct images from flat patch tokens (inverse of patchify).

    Args:
        patches:    (B, T, P²×3)  normalised pixel space
        patch_size: P
        image_size: H = W (assumed square)

    Returns:
        (B, H, W, 3)
    """
    B, _T, _ = patches.shape
    P = patch_size
    h = w = image_size // P
    x = patches.reshape(B, h, w, P, P, 3)
    x = x.transpose(0, 1, 3, 2, 4, 5)   # (B, h, P, w, P, 3)
    return x.reshape(B, h * P, w * P, 3)


def recon_mse_loss(
    pixel_pred: jnp.ndarray,    # (B, T, patch_pixel_dim)
    patch_targets: jnp.ndarray, # (B, T, patch_pixel_dim)
) -> jnp.ndarray:
    """Mean squared error over all T patch positions and all pixels within each patch.

    Both tensors should be in the same normalised pixel space.

    Returns: scalar
    """
    return jnp.mean((pixel_pred - patch_targets) ** 2)
