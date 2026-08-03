"""Shared building blocks for PaliGemma-style models.

The PrefixMAE-based variant (PaliGemmaEncDec) is the live model; it pulls
Projector and token_xent_loss from here.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import flax.linen as nn

from gemma.gm.nn._layers import RMSNorm
from utils.pjit_util import constrain_batch_model
from utils import pjit_util as _pjit_util
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as _P

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
    logits_scale: float = 1.0,
    subtract_hidden=None,
    subtract_alpha: float = 0.0,
):
    """Cross entropy from hidden states without materializing (B, T, V) logits.

    ``chunk_size`` counts TOKENS, not vocabulary columns: each chunk decodes
    ``chunk_size`` positions against the *whole* vocabulary and then runs one
    ordinary logsumexp / argmax / label lookup.  Peak logit memory is the same
    either way (chunk_size x V per chunk), but chunking the token axis deletes
    the entire bookkeeping layer a vocab-chunked scan needs: a zero-padded
    table, an in-vocab mask, a cross-chunk logaddexp carry, a running argmax
    carry, and a dynamic "which chunk holds this label" test.  With them go all
    the +-inf sentinels, so the loss is finite by construction whenever the
    inputs are -- there is no longer any way for a masked-out column to leak
    into it.

    ``logits_scale`` and ``subtract_hidden`` support the CFG loss path::

        logits = logits_scale * cond_logits
                 - subtract_alpha * stop_grad(text_logits)

    Each branch is soft-capped before its coefficient is applied, matching the
    full-logits definition.
    """
    hidden = hidden.astype(jnp.float32)
    embedding_table = embedding_table.astype(jnp.float32)
    # Remove the shared vocabulary-row mode before the TPU dot.  This makes a
    # separately rewritten target dot numerically harmless because every
    # decode path consumes the same centered table.  The table mean is detached
    # deliberately: the tied input lookup keeps its original parameterization,
    # while decode receives the mathematically shift-invariant forward value
    # without projecting optimizer gradients through a decode-only mean.
    # Unconditional, including under a tanh softcap. The softcap is not shift
    # invariant, which is exactly why the centering has to stay on: h @ mean(table)
    # drifts about -1.5 per 10k steps in every config measured, so around step 170k
    # it reaches -30 = the softcap value, tanh saturates the vocab bulk, and the
    # reported CE collapses onto its information-theoretic floor while accuracy
    # stops improving (measured: +0.0203 -> +0.0019 acc per 10k steps across that
    # crossing). Centering pins the vocab mean of the capped logits at 0, so the
    # loss becomes exactly independent of that drift -- verified on v4 at
    # (V=152960, D=896): loss stays 25.6314 while the common mode is swept 0 to
    # -120, versus 25.63 -> 13.31 without centering.
    table_mean = jax.lax.stop_gradient(
        embedding_table.mean(axis=0, keepdims=True)
    )
    embedding_table = embedding_table - table_mean

    labels = labels.astype(jnp.int32)
    valid = (labels != ignore_index)
    vocab_size, hidden_dim = embedding_table.shape
    batch_shape = labels.shape
    assert hidden.shape[:-1] == batch_shape, (
        f"hidden {hidden.shape} and labels {batch_shape} disagree on positions"
    )

    n_tokens = int(labels.size)
    flat_hidden = hidden.reshape(n_tokens, hidden_dim)
    # Clipping only has to bring ignore_index (-100) back in range; every label
    # that survives `valid` already indexes a real row, so the one-hot below
    # always matches exactly one column.
    flat_labels = jnp.clip(labels.reshape(n_tokens), 0, vocab_size - 1)
    flat_sub = (
        subtract_hidden.astype(jnp.float32).reshape(n_tokens, hidden_dim)
        if subtract_hidden is not None
        else None
    )

    chunk = max(1, min(int(chunk_size), n_tokens))
    num_chunks = (n_tokens + chunk - 1) // chunk
    pad_tokens = num_chunks * chunk - n_tokens
    if pad_tokens:
        # Padded positions decode a zero hidden state: every logit is 0, so
        # logZ = log V and the label logit is 0.  Finite, and dropped by `valid`
        # before it reaches the loss -- no sentinel value required.
        flat_hidden = jnp.pad(flat_hidden, ((0, pad_tokens), (0, 0)))
        flat_labels = jnp.pad(flat_labels, (0, pad_tokens))
        if flat_sub is not None:
            flat_sub = jnp.pad(flat_sub, ((0, pad_tokens), (0, 0)))

    def _decode_block(h, lab, sub, table):
        """All reductions for one token block against the FULL vocabulary.

        Every array here lives in one device frame: `table` is the complete
        (V, D) matrix and `logits` the complete (m, V) row block, so logZ,
        label_logit, argmax and the probes all read the same buffer and
        CE >= 0 holds by construction. The callers differ only in what "one
        device frame" means (whole mesh for the scan path, one shard for the
        shard_map path).
        """
        raw = jnp.einsum("nd,vd->nv", h, table)
        # Pre-softcap vocab-axis sum, reduced to centered_logit_mean below:
        # ~1e-4 while the table is centered, equal to the raw common mode (-27
        # and falling in the runs that broke) if the centering is ever bypassed.
        # Single scalar that separates the two by ~5 orders of magnitude.
        logit_sum = jnp.sum(raw, axis=-1)
        logits = _maybe_softcap(raw, final_logit_softcap)
        # Avoid an unnecessary multiply for the common coefficient of 1.0.
        if logits_scale != 1.0:
            logits = logits_scale * logits
        if sub is not None and subtract_alpha != 0.0:
            sub_logits = _maybe_softcap(
                jnp.einsum("nd,vd->nv", sub, table),
                final_logit_softcap,
            )
            logits = logits - subtract_alpha * jax.lax.stop_gradient(sub_logits)

        onehot = (
            jnp.arange(table.shape[0], dtype=jnp.int32)[None, :] == lab[:, None]
        )
        # Reduce, not take_along_axis: a sharded-vocab gather once got rewritten
        # into a separate dot with different rounding (how nll first went < 0),
        # and a 0-filled mask sum has no sentinel to leak. See README.md
        # "Sharding invariant & CE guardrails".
        label_logit = jnp.sum(jnp.where(onehot, logits, 0.0), axis=-1)
        # onehot_count must be exactly 1 per token; logged as a guardrail.
        dbg_onehot_count = jnp.sum(onehot.astype(jnp.int32), axis=-1)
        return (
            jax.nn.logsumexp(logits, axis=-1),
            label_logit,
            jnp.argmax(logits, axis=-1).astype(jnp.int32),
            logit_sum,
            dbg_onehot_count,
        )

    # On multi-host meshes run the reductions inside shard_map so the
    # partitioner has no sharded reduction to rewrite (the SPMD rewrite of this
    # exact code once returned label_logit > logZ on v5p-64). Details in
    # README.md "Sharding invariant & CE guardrails".
    _mesh = _pjit_util._mesh
    _data_axes = (
        _pjit_util._data_axis_names(_mesh, _pjit_util._mode)
        if _mesh is not None
        else ()
    )
    _n_data = 1
    if _mesh is not None:
        _sizes = dict(zip(_mesh.axis_names, _mesh.devices.shape))
        for _ax in _data_axes:
            _n_data *= int(_sizes[_ax])
    use_shard_map = (
        _mesh is not None
        and len(_data_axes) > 0
        and _n_data > 1
        and chunk % _n_data == 0
        # Escape hatch for A/B and audits; the shard_map path is the default.
        and os.environ.get("PALIGEMMA_CE_SHARDMAP", "1") != "0"
    )

    if use_shard_map:
        # Constraints, each learned from a live v5p failure (README.md has the
        # verdict matrix): the einsum must stay OUTSIDE the manual region (an
        # einsum/scan inside halts every core), the body does reductions only,
        # and chunking must never move a token between data ranks -- hence
        # PER-RANK STRIDED chunks via a metadata-only reshape along the
        # sharding. Which tokens share a chunk is irrelevant (CE is per-token;
        # chunking only bounds peak memory).
        _model_ax = _pjit_util._model_axis_names(_mesh, _pjit_util._mode)[0]
        _row_spec = _P(tuple(_data_axes), _model_ax)
        _vec_spec = _P(tuple(_data_axes))
        chunk_loc = chunk // _n_data

        def _reduce_body(logits_loc, lab_loc):
            vloc = logits_loc.shape[-1]
            col0 = (jax.lax.axis_index(_model_ax) * vloc).astype(jnp.int32)
            lmax = jax.lax.stop_gradient(jnp.max(logits_loc, axis=-1))
            # pmax has no AD rule (jax 0.6); the logsumexp shift is the
            # standard stop-gradient max anyway.
            gmax = jax.lax.stop_gradient(jax.lax.pmax(lmax, _model_ax))
            lse = jnp.sum(jnp.exp(logits_loc - gmax[:, None]), axis=-1)
            # psum(lse) >= exp(lmax-gmax) = 1 on the gmax shard, on every
            # device, so log_z >= gmax >= any single logit: CE >= 0 is
            # structural, given the hand-built onehot hits exactly one
            # column -- which col0 + arange guarantees by construction.
            log_z = jnp.log(jax.lax.psum(lse, _model_ax)) + gmax
            cols = col0 + jnp.arange(vloc, dtype=jnp.int32)
            onehot = cols[None, :] == lab_loc[:, None]
            lab_logit = jax.lax.psum(
                jnp.sum(jnp.where(onehot, logits_loc, 0.0), axis=-1),
                _model_ax,
            )
            ohc = jax.lax.psum(
                jnp.sum(onehot.astype(jnp.int32), axis=-1), _model_ax
            )
            local_arg = (
                jnp.argmax(logits_loc, axis=-1).astype(jnp.int32) + col0
            )
            cand = jnp.where(lmax >= gmax, local_arg, jnp.int32(2**31 - 1))
            pred = jax.lax.pmin(cand, _model_ax)
            return log_z, lab_logit, pred, ohc

        _hreduce = shard_map(
            _reduce_body,
            mesh=_mesh,
            in_specs=(_row_spec, _vec_spec),
            out_specs=(_vec_spec,) * 4,
        )

        def _rank_major(x, tail_shape):
            # (N, ...) -> (n_data, num_chunks, chunk_loc, ...): dim 0 folds
            # exactly along the data sharding, so this is metadata-only.
            y = x.reshape((_n_data, num_chunks, chunk_loc) + tail_shape)
            return jax.lax.with_sharding_constraint(
                y,
                jax.sharding.NamedSharding(
                    _mesh,
                    _P(tuple(_data_axes), *((None,) * (y.ndim - 1))),
                ),
            )

        Hr = _rank_major(flat_hidden, (hidden_dim,))
        Lr = _rank_major(flat_labels, ())
        Sr = _rank_major(flat_sub, (hidden_dim,)) if flat_sub is not None else None

        _outs = []
        for _ci in range(num_chunks):
            h = Hr[:, _ci].reshape(chunk, hidden_dim)
            lab = Lr[:, _ci].reshape(chunk)
            raw = jnp.einsum("nd,vd->nv", h, embedding_table)
            logit_sum = jnp.sum(raw, axis=-1)
            logits = _maybe_softcap(raw, final_logit_softcap)
            if logits_scale != 1.0:
                logits = logits_scale * logits
            if Sr is not None and subtract_alpha != 0.0:
                hs = Sr[:, _ci].reshape(chunk, hidden_dim)
                sub_logits = _maybe_softcap(
                    jnp.einsum("nd,vd->nv", hs, embedding_table),
                    final_logit_softcap,
                )
                logits = logits - subtract_alpha * jax.lax.stop_gradient(
                    sub_logits
                )
            logits = constrain_batch_model(logits)
            log_z, lab_logit, pred, ohc = _hreduce(logits, lab)
            _outs.append((log_z, lab_logit, pred, logit_sum, ohc))

        def _reorder(x):
            # (num_chunks, chunk) back to the original flat token order. The
            # transpose only swaps the two leading logical dims; the data
            # sharding stays on the rank axis, so it is metadata-only too.
            return (
                x.reshape(num_chunks, _n_data, chunk_loc)
                .transpose(1, 0, 2)
                .reshape(-1)
            )

        (
            log_normalizer,
            label_logit,
            pred_ids,
            logit_sum,
            dbg_onehot_count,
        ) = (_reorder(jnp.stack(t)) for t in zip(*_outs))
    else:
        def scan_chunk(carry, chunk_idx):
            start = chunk_idx * chunk
            h = jax.lax.dynamic_slice(
                flat_hidden, (start, 0), (chunk, hidden_dim)
            )
            lab = jax.lax.dynamic_slice(flat_labels, (start,), (chunk,))
            sub = (
                jax.lax.dynamic_slice(
                    flat_sub, (start, 0), (chunk, hidden_dim)
                )
                if flat_sub is not None
                else None
            )
            return carry, _decode_block(h, lab, sub, embedding_table)

        _, (
            log_normalizer,
            label_logit,
            pred_ids,
            logit_sum,
            dbg_onehot_count,
        ) = jax.lax.scan(
            scan_chunk, None, jnp.arange(num_chunks, dtype=jnp.int32)
        )

    def _unflatten(x):
        return x.reshape(-1)[:n_tokens].reshape(batch_shape)

    log_normalizer = _unflatten(log_normalizer)
    label_logit = _unflatten(label_logit)
    pred_ids = _unflatten(pred_ids)
    logit_sum = _unflatten(logit_sum)
    dbg_onehot_count = _unflatten(dbg_onehot_count)

    nll = log_normalizer - label_logit
    denom = jnp.maximum(valid.sum(), 1)
    loss = (nll * valid).sum() / denom
    # CE = logZ - l_label >= 0 identically; a violation means the two terms came
    # from different logit vectors (this silently corrupted long runs). Poison
    # the loss so the run dies at the first bad step. Never fires when correct.
    min_nll = jnp.min(jnp.where(valid, nll, jnp.inf))
    # A batch with no valid target would leave the +inf sentinel above in the
    # logged metric and trip the non-finite guard on a healthy step.
    min_nll = jnp.where(valid.any(), min_nll, 0.0)
    loss = loss + jnp.where(min_nll < -1e-3, jnp.nan, 0.0).astype(loss.dtype)
    log_z_mean = (log_normalizer * valid).sum() / denom
    centered_logit_mean = (
        (logit_sum / float(vocab_size)) * valid
    ).sum() / denom
    _stop = jax.lax.stop_gradient
    aux = {
        # logit-scale monitor (gradient-carrying logsumexp mean over valid).
        'log_z_mean': log_z_mean,
        # Tripwire: nll_min crossing 0 is the failure itself; watch it directly.
        'nll_min': min_nll,
        # ~1e-4 if the decode table is centered; tracks the raw common mode
        # (order 1 early, -27 late) if the centering is ever bypassed.
        'centered_logit_mean': _stop(centered_logit_mean),
        # The loss cannot manufacture a non-finite value on its own, so if one
        # shows up it came in through the activations.
        'hidden_absmax': _stop(jnp.max(jnp.abs(hidden))),
        # Guardrail: the label one-hot must hit exactly one column per token
        # ([1, 1]); 0/2 was the fake-replication signature (README.md).
        'dbg_onehot_count_min': _stop(jnp.min(dbg_onehot_count)),
        'dbg_onehot_count_max': _stop(jnp.max(dbg_onehot_count)),
    }
    return loss, jnp.where(valid, pred_ids, 0), aux
