"""Shared building blocks for PaliGemma-style models.

The PrefixMAE-based variant (PaliGemmaEncDec) is the live model; it pulls
Projector and token_xent_loss from here.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

from gemma.gm.nn._layers import RMSNorm

Array = jnp.ndarray


class Projector(nn.Module):
    out_dim: int
    use_ln: bool = True
    use_2l_mlp: bool = False  # 2-layer MLP vs single linear

    @nn.compact
    def __call__(self, x: Array) -> Array:
        # x: (B, N, Dv)
        if self.use_2l_mlp:
            x = nn.Dense(
                self.out_dim * 4,
                use_bias=False,
                name="proj1",
                kernel_init=nn.initializers.normal(stddev=0.02),
            )(x)
            x = nn.gelu(x)
            x = nn.Dense(
                self.out_dim,
                use_bias=False,
                name="proj2",
                kernel_init=nn.initializers.normal(stddev=0.02),
            )(x)
        else:
            x = nn.Dense(
                self.out_dim,
                use_bias=False,
                name="proj",
                kernel_init=nn.initializers.normal(stddev=0.02),
            )(x)
        if self.use_ln:
            x = RMSNorm(name="norm")(x)
        return x


def token_xent_loss(logits, labels, ignore_index=-100):
    """Cross-entropy loss over a token sequence.

    Avoids materialising the full (B, T, V) log-softmax tensor by computing
    only the per-position logsumexp scalar (B, T) as the normaliser.
    """
    valid = labels != ignore_index
    labels_clipped = jnp.clip(labels, 0, logits.shape[-1] - 1)

    label_logit = jnp.take_along_axis(logits, labels_clipped[..., None], axis=-1)[..., 0]
    log_normalizer = jax.nn.logsumexp(logits, axis=-1)

    nll = log_normalizer - label_logit
    denom = jnp.maximum(valid.sum(), 1)
    return (nll * valid).sum() / denom
