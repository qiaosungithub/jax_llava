"""Shared building blocks for PaliGemma-style models.

The PrefixMAE-based variant (PaliGemmaEncDec) is the live model; it pulls
Projector and token_xent_loss from here.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

from gemma.gm.nn._layers import RMSNorm
from utils.pjit_util import constrain_batch_model

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


def _maybe_softcap(logits, final_logit_softcap: float):
    if final_logit_softcap == 0.0:
        return logits
    return jnp.tanh(logits / final_logit_softcap) * final_logit_softcap


def token_xent_loss_from_hidden(
    hidden,
    embedding_table,
    labels,
    *,
    ignore_index=-100,
    final_logit_softcap: float = 0.0,
    chunk_size: int = 8192,
    subtract_hidden=None,
    subtract_alpha: float = 0.0,
):
    """Cross entropy from hidden states without materializing full-sequence logits.

    Gemma3 has a 262k vocab; a full ``(B, T, V)`` tensor is often the largest
    training allocation. This scans over vocab chunks and keeps only ``(B, T)``
    log-normalizers plus argmax state live at once. ``subtract_hidden`` supports
    the CFG loss path: ``logits = cond_logits - alpha * stop_grad(text_logits)``.
    """
    hidden = hidden.astype(jnp.float32)
    embedding_table = embedding_table.astype(jnp.float32)
    labels = labels.astype(jnp.int32)
    valid = labels != ignore_index
    labels_clipped = jnp.clip(labels, 0, embedding_table.shape[0] - 1)

    label_emb = embedding_table[labels_clipped]
    label_logit = jnp.sum(hidden * label_emb, axis=-1)
    label_logit = _maybe_softcap(label_logit, final_logit_softcap)

    if subtract_hidden is not None and subtract_alpha != 0.0:
        subtract_hidden = subtract_hidden.astype(jnp.float32)
        subtract_label_logit = jnp.sum(subtract_hidden * label_emb, axis=-1)
        subtract_label_logit = _maybe_softcap(
            subtract_label_logit,
            final_logit_softcap,
        )
        label_logit = (
            label_logit
            - subtract_alpha * jax.lax.stop_gradient(subtract_label_logit)
        )

    vocab_size, hidden_dim = embedding_table.shape
    chunk_size = max(1, min(int(chunk_size), int(vocab_size)))
    num_chunks = (int(vocab_size) + chunk_size - 1) // chunk_size
    pad_vocab = num_chunks * chunk_size - int(vocab_size)
    if pad_vocab:
        embedding_table = jnp.pad(embedding_table, ((0, pad_vocab), (0, 0)))

    init_log_norm = jnp.full(labels.shape, -jnp.inf, dtype=jnp.float32)
    init_max = jnp.full(labels.shape, -jnp.inf, dtype=jnp.float32)
    init_argmax = jnp.zeros(labels.shape, dtype=jnp.int32)

    def scan_chunk(carry, chunk_idx):
        log_norm, best_value, best_id = carry
        start = chunk_idx * chunk_size
        emb_chunk = jax.lax.dynamic_slice(
            embedding_table,
            (start, 0),
            (chunk_size, hidden_dim),
        )
        logits = jnp.einsum("...d,vd->...v", hidden, emb_chunk)
        logits = _maybe_softcap(logits, final_logit_softcap)

        if subtract_hidden is not None and subtract_alpha != 0.0:
            subtract_logits = jnp.einsum("...d,vd->...v", subtract_hidden, emb_chunk)
            subtract_logits = _maybe_softcap(subtract_logits, final_logit_softcap)
            logits = logits - subtract_alpha * jax.lax.stop_gradient(subtract_logits)
        logits = constrain_batch_model(logits)

        vocab_ids = start + jnp.arange(chunk_size, dtype=jnp.int32)
        valid_vocab = vocab_ids < vocab_size
        valid_vocab = valid_vocab.reshape((1,) * (logits.ndim - 1) + (chunk_size,))
        logits = jnp.where(valid_vocab, logits, -jnp.inf)

        chunk_log_norm = jax.nn.logsumexp(logits, axis=-1)
        chunk_max = jnp.max(logits, axis=-1)
        chunk_argmax = jnp.argmax(logits, axis=-1).astype(jnp.int32) + start

        better = chunk_max > best_value
        return (
            jnp.logaddexp(log_norm, chunk_log_norm),
            jnp.where(better, chunk_max, best_value),
            jnp.where(better, chunk_argmax, best_id),
        ), None

    (log_normalizer, _, pred_ids), _ = jax.lax.scan(
        scan_chunk,
        (init_log_norm, init_max, init_argmax),
        jnp.arange(num_chunks, dtype=jnp.int32),
    )

    nll = log_normalizer - label_logit
    denom = jnp.maximum(valid.sum(), 1)
    loss = (nll * valid).sum() / denom
    return loss, jnp.where(valid, pred_ids, 0)
