# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Muon.

Implementation of the
[Muon optimizer](https://github.com/KellerJordan/modded-nanogpt)
by Keller Jordan
"""


import math
from typing import Any, Callable, NamedTuple, Optional, Union

import chex
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
import jax
import jax.numpy as jnp

from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
import optax.tree


_ADAM_LABEL = "adam"
_MUON_MATRIX_LABEL = "muon_matrix"
_MUON_PATCH_EMBED_LABEL = "muon_patch_embed"
_MUON_DENSE_GENERAL_IN_LABEL = "muon_dense_general_in"
_MUON_EINSUM_ATTENTION_IN_LABEL = "muon_einsum_attention_in"
_MUON_DENSE_GENERAL_OUT_LABEL = "muon_dense_general_out"


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
) -> jax.Array:
  r"""Orthogonalize via Newton-Schulz iteration.

  We opt to use a quintic iteration whose coefficients are selected to maximize
  the slope at zero. For the purpose of minimizing steps, it turns out to be
  empirically effective to keep increasing the slope at zero even beyond the
  point where the iteration no longer converges all the way to one everywhere
  on the interval. This iteration therefore does not produce UV^T but rather
  something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5),
  which turns out not to hurt model performance at all relative to UV^T, where
  USV^T = G is the SVD.

  Args:
    x: A matrix to orthogonalize.
    ns_coeffs: Coefficients for the Newton-schulz iterators.
      Must have shape (n, 3) where n is the number of iterations.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a 2D array.
    eps: Term added to denominators to improve numerical stability.

  Returns:
    The orthogonalized matrix.
  """
  if x.ndim != 2:
    raise ValueError(f'Input must have shape (m, n), got {x.shape}')
  if ns_coeffs.ndim > 2 or ns_coeffs.shape[-1] != 3:
    raise ValueError(
        'Newton-Schulz coefficients must have shape (3,) or (n, 3), '
        f'got {ns_coeffs.shape}'
    )
  def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    a = x @ x.T
    b = coeffs[1] * a + coeffs[2] * a @ a
    return coeffs[0] * x + b @ x
  transposed = False
  if x.shape[0] > x.shape[1]:
    x = x.T
    transposed = True
  x /= jnp.linalg.norm(x) + eps  # Ensure spectral norm is at most 1
  ns_coeffs = ns_coeffs.astype(x.dtype)
  if ns_coeffs.ndim == 1:
    x = jax.lax.fori_loop(
        0, ns_steps, lambda _, x: newton_schulz_iterator(x, ns_coeffs), x
    )
  else:
    x, _ = jax.lax.scan(
        lambda x, abc: (newton_schulz_iterator(x, abc), None), x, ns_coeffs
    )
  if transposed:
    x = x.T
  return x


def _to_matrix_batch(x: jax.Array, matrix_axis_policy: str) -> jax.Array:
  """Views a Dense kernel as a batch of 2D matrices for Muon."""
  if matrix_axis_policy == _MUON_MATRIX_LABEL:
    if x.ndim < 2:
      raise ValueError(f"Muon matrix params must be at least 2D, got {x.shape}")
    matrix_shape = x.shape[-2:]
    return jnp.reshape(x, (-1,) + matrix_shape)

  if matrix_axis_policy == _MUON_PATCH_EMBED_LABEL:
    # ViT patch embedding conv: (...scan, patch_h, patch_w, in_ch, out_dim)
    # -> (...scan, patch_h * patch_w * in_ch, out_dim).
    if x.ndim < 2:
      raise ValueError(
          f"Patch embedding params must be at least 2D, got {x.shape}"
      )
    matrix_shape = (math.prod(x.shape[:-1]), x.shape[-1])
    return jnp.reshape(x, (-1,) + matrix_shape)

  if matrix_axis_policy == _MUON_DENSE_GENERAL_IN_LABEL:
    # DenseGeneral q/k/v kernel: (...scan, in_dim, heads, head_dim)
    # -> (...scan, in_dim, heads * head_dim).
    if x.ndim < 3:
      raise ValueError(
          "DenseGeneral input-projection params must be at least 3D, "
          f"got {x.shape}"
      )
    matrix_shape = (x.shape[-3], math.prod(x.shape[-2:]))
    return jnp.reshape(x, (-1,) + matrix_shape)

  if matrix_axis_policy == _MUON_EINSUM_ATTENTION_IN_LABEL:
    # Gemma q/qkv/kv einsum weights: (...batch, heads, in_dim, head_dim)
    # -> (...batch, in_dim, heads * head_dim). The batch axis can be qkv/kv
    # selector and/or a scanned layer axis.
    if x.ndim < 3:
      raise ValueError(
          "Einsum attention input-projection params must be at least 3D, "
          f"got {x.shape}"
      )
    x = jnp.moveaxis(x, -3, -2)
    matrix_shape = (x.shape[-3], math.prod(x.shape[-2:]))
    return jnp.reshape(x, (-1,) + matrix_shape)

  if matrix_axis_policy == _MUON_DENSE_GENERAL_OUT_LABEL:
    # DenseGeneral output kernel: (...scan, heads, head_dim, out_dim)
    # -> (...scan, heads * head_dim, out_dim).
    if x.ndim < 3:
      raise ValueError(
          "DenseGeneral output-projection params must be at least 3D, "
          f"got {x.shape}"
      )
    matrix_shape = (math.prod(x.shape[-3:-1]), x.shape[-1])
    return jnp.reshape(x, (-1,) + matrix_shape)

  raise ValueError(f"Unknown Muon matrix-axis policy: {matrix_axis_policy}")


def _from_matrix_batch(
    matrices: jax.Array,
    original_shape: tuple[int, ...],
    matrix_axis_policy: str,
) -> jax.Array:
  if matrix_axis_policy == _MUON_EINSUM_ATTENTION_IN_LABEL:
    moved_shape = original_shape[:-3] + (
        original_shape[-2],
        original_shape[-3],
        original_shape[-1],
    )
    x = jnp.reshape(matrices, moved_shape)
    return jnp.moveaxis(x, -2, -3)
  return jnp.reshape(matrices, original_shape)


def _orthogonalize_muon_update(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int,
    eps: float,
    matrix_axis_policy: str,
    consistent_rms: bool,
) -> jax.Array:
  matrices = _to_matrix_batch(x, matrix_axis_policy)
  updates = jax.vmap(
      lambda matrix: orthogonalize_via_newton_schulz(
          matrix, ns_coeffs, ns_steps, eps
      )
  )(matrices)

  if consistent_rms:
    rms = jnp.sqrt(jnp.mean(jnp.square(updates), axis=(-2, -1), keepdims=True))
    updates = updates / (rms + eps)
  else:
    rows = matrices.shape[-2]
    cols = matrices.shape[-1]
    updates = jnp.sqrt(jnp.maximum(1, cols / rows)) * updates

  return _from_matrix_batch(updates, x.shape, matrix_axis_policy)


def _apply_muon_adaptive_scaling(
    mu_hat: jax.Array,
    updates: jax.Array,
    matrix_axis_policy: str,
) -> jax.Array:
  mu_matrices = _to_matrix_batch(mu_hat, matrix_axis_policy)
  update_matrices = _to_matrix_batch(updates, matrix_axis_policy)
  dual_norm = jnp.sum(
      mu_matrices * update_matrices, axis=(-2, -1), keepdims=True
  )
  update_matrices = dual_norm * update_matrices
  return _from_matrix_batch(update_matrices, updates.shape, matrix_axis_policy)


def _path_parts(path: tuple[Any, ...]) -> tuple[str, ...]:
  return tuple(str(x).lower() for x in path)


def _is_attention_path(parts: tuple[str, ...]) -> bool:
  joined = "/".join(parts)
  return (
      "attention" in joined
      or "attn" in joined
      or "multiheaddotproductattention" in joined
      or "cross_attn" in joined
  )


def _is_mlp_path(parts: tuple[str, ...]) -> bool:
  joined = "/".join(parts)
  return (
      "mlp" in joined
      or "ffn" in joined
      or "feedforward" in joined
      or "feed_forward" in joined
  )


def _is_patch_embedding_path(parts: tuple[str, ...]) -> bool:
  joined = "/".join(parts)
  return (
      "patch_embed" in joined
      or "patch_embedding" in joined
      or "/embedding/kernel" in joined
  )


def _muon_label_for_param(path: tuple[Any, ...], param: Any) -> str:
  """Labels only Attention/MLP Dense kernels for Muon.

  The rule is intentionally path-based rather than apparent-rank-based because
  scanned Flax modules prepend the layer axis to every parameter.
  """
  parts = _path_parts(path)
  if not parts or getattr(param, "ndim", 0) < 2:
    return _ADAM_LABEL

  leaf_name = parts[-1]
  parent = parts[-2] if len(parts) >= 2 else ""
  grandparent = parts[-3] if len(parts) >= 3 else ""
  is_attention = _is_attention_path(parts)
  is_mlp = _is_mlp_path(parts)

  if leaf_name == "kernel" and _is_patch_embedding_path(parts):
    return _MUON_PATCH_EMBED_LABEL

  if is_attention:
    if leaf_name == "w" and parent in {"q_einsum", "qkv_einsum", "kv_einsum"}:
      return _MUON_EINSUM_ATTENTION_IN_LABEL
    if leaf_name == "w" and parent == "attn_vec_einsum":
      return (
          _MUON_DENSE_GENERAL_OUT_LABEL
          if param.ndim >= 3
          else _MUON_MATRIX_LABEL
      )
    if leaf_name != "kernel":
      return _ADAM_LABEL
    if parent in {"query", "key", "value", "q", "k", "v", "qkv"}:
      return (
          _MUON_DENSE_GENERAL_IN_LABEL
          if param.ndim >= 3
          else _MUON_MATRIX_LABEL
      )
    if parent in {"out", "out_proj", "proj", "o"}:
      return (
          _MUON_DENSE_GENERAL_OUT_LABEL
          if param.ndim >= 3
          else _MUON_MATRIX_LABEL
      )
    return _MUON_MATRIX_LABEL

  if is_mlp or grandparent in {"mlp", "mlpblock"}:
    if leaf_name not in {"kernel", "gating_einsum", "linear", "w"}:
      return _ADAM_LABEL
    return _MUON_MATRIX_LABEL

  return _ADAM_LABEL


def create_muon_param_labels(params: base.Params) -> base.Params:
  """Creates the optimizer label pytree for Muon partitioning."""
  is_frozen = isinstance(params, FrozenDict)
  params_dict = params.unfreeze() if is_frozen else params
  flat_params = flatten_dict(params_dict)
  labels = {
      path: _muon_label_for_param(path, param)
      for path, param in flat_params.items()
  }
  labels = unflatten_dict(labels)
  return FrozenDict(labels) if is_frozen else labels


class MuonState(NamedTuple):
  """State for the Muon algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  ns_coeffs: chex.Array  # shape=(), dtype=jnp.int32.


def scale_by_muon(
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    matrix_axis_policy: str = _MUON_MATRIX_LABEL,
    consistent_rms: bool = False,
) -> base.GradientTransformation:
  r"""Rescale updates according to the Muon algorithm.

  Muon is a variant of Shampoo that uses the Newton-schulz method to
  orthogonalize the momentum accumulated by the optimizer. Mathematically, it
  does steepest descent under the Schatten-p norm, for some large p. With
  p=infty, it is equivalent to Shampoo without accumulation, or steepest
  descent under the Spectral norm.

  Args:
    ns_coeffs: Coefficients for the Newton-schulz method.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    beta: Decay rate for the exponentially weighted average of grads.
    eps: Term added to denominators to improve numerical stability.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.
    adaptive: Whether to scale the updates by the dual norm of the
      original updates. See <https://arxiv.org/abs/2409.20325>
    matrix_axis_policy: How to view selected Dense kernels as matrices.
    consistent_rms: If true, normalize the orthogonalized update to unit RMS
      per matrix so the Muon learning rate is comparable to Adam's normalized
      update scale.

  Returns:
    A `GradientTransformation` object.

  References:
    Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
    <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

    Bernstein et al., `Old Optimizer, New Norm: An Anthology
    <https://arxiv.org/abs/2409.20325>`_, 2024
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment
    ns_coeffs_ = jnp.asarray(ns_coeffs)
    if ns_coeffs_.ndim > 2 or ns_coeffs_.shape[-1] != 3:
      raise ValueError(
          f'ns_coeffs must have shape (3,) or (n, 3), got {ns_coeffs_.shape}'
      )
    return MuonState(
        count=jnp.zeros([], jnp.int32),
        mu=mu,
        ns_coeffs=ns_coeffs_,
    )

  def update_fn(updates, state, params=None):
    del params
    mu = optax.tree.update_moment(updates, state.mu, beta, 1)
    count_inc = numerics.safe_increment(state.count)
    if nesterov:
      mu_hat = jax.tree.map(
          lambda m, g: beta * m + (1 - beta) * g,
          optax.tree.bias_correction(
              mu, beta, numerics.safe_increment(count_inc)
          ),
          optax.tree.bias_correction(updates, beta, count_inc),
      )
    else:
      mu_hat = optax.tree.bias_correction(mu, beta, count_inc)
    # Apply Newton-schulz orthogonalization. Scanned Dense kernels are treated
    # as a batch of per-layer matrices.
    updates = jax.tree.map(
        lambda x: _orthogonalize_muon_update(
            x,
            state.ns_coeffs,
            ns_steps,
            eps,
            matrix_axis_policy,
            consistent_rms,
        ),
        mu_hat,
    )
    if adaptive:
      # Scale the orthogonalized updates by the dual norm of the original
      # updates. See https://arxiv.org/abs/2409.20325 for the derivation.
      updates = jax.tree.map(
          lambda x, y: _apply_muon_adaptive_scaling(
              x, y, matrix_axis_policy
          ),
          mu_hat,
          updates,
      )
    mu = optax.tree.cast(mu, mu_dtype)
    return updates, MuonState(
        count=count_inc,
        mu=mu,
        ns_coeffs=state.ns_coeffs,
    )
  return base.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate: base.ScalarOrSchedule,
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[
        Union[Any, Callable[[base.Params], Any]]
    ] = None,
    mu_dtype: Optional[Any] = None,
    *,
    custom_adam = None,
    nesterov: bool = True,
    adaptive: bool = False,
    consistent_rms: bool = True,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps_root: float = 0.0,
    adam_weight_decay: float = 0.0,
) -> base.GradientTransformation:
  r"""Muon: Momentum Orthogonalized by Newton-schulz.

  Muon is a variant of Shampoo that uses the Newton-schulz method to
  orthogonalize the momentum accumulated by the optimizer. Mathematically, it
  does steepest descent under the Schatten-p norm, for some large p. With
  p=infty, it is equivalent to Shampoo without accumulation, or steepest
  descent under the Spectral norm.

  Muon is applied only to Dense kernels inside Attention/MLP modules. Scanned
  Dense params and DenseGeneral attention params are reshaped into per-layer
  matrices before the Newton-Schulz iteration. All other params are passed
  through Adam.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    ns_coeffs: Coefficients for the Newton-schulz method.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    beta: Decay rate for the exponentially weighted average of grads.
    eps: Term added to the denominator to improve numerical stability.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent
      with other frameworks such as PyTorch, but different from
      (Loshchilov et al, 2019) where the weight decay is only multiplied with
      the "schedule multiplier", but not the base learning rate.
    weight_decay_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.
    adaptive: Whether to scale the updates by the dual norm of the
      original updates. See <https://arxiv.org/abs/2409.20325>
    consistent_rms: Normalize each Muon matrix update to unit RMS, matching the
      scale expected by the Adam fallback learning rate.
    adam_b1: Exponential decay rate for Adam's first moment estimates.
    adam_b2: Exponential decay rate for Adam's second moment estimates.
    adam_eps_root: Epsilon to stabilize division in Adam, square root version.
    adam_weight_decay: Weight decay factor for Adam.

  Returns:
    The corresponding `GradientTransformation`.

  References:
    Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
    <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

    Bernstein et al., `Old Optimizer, New Norm: An Anthology
    <https://arxiv.org/abs/2409.20325>`_, 2024
  """
  adam = custom_adam if custom_adam else alias.adamw(
      learning_rate=learning_rate,
      b1=adam_b1,
      b2=adam_b2,
      eps=eps,
      eps_root=adam_eps_root,
      weight_decay=adam_weight_decay,
      mu_dtype=mu_dtype,
      nesterov=nesterov,
  )
  def _muon_transform(matrix_axis_policy: str) -> base.GradientTransformation:
    return combine.chain(
        scale_by_muon(
            ns_coeffs=ns_coeffs,
            ns_steps=ns_steps,
            beta=beta,
            eps=eps,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
            adaptive=adaptive,
            matrix_axis_policy=matrix_axis_policy,
            consistent_rms=consistent_rms,
        ),
        transform.add_decayed_weights(weight_decay, weight_decay_mask),
        transform.scale_by_learning_rate(learning_rate),
    )

  return combine.partition(
      transforms={
          _MUON_MATRIX_LABEL: _muon_transform(_MUON_MATRIX_LABEL),
          _MUON_PATCH_EMBED_LABEL: _muon_transform(_MUON_PATCH_EMBED_LABEL),
          _MUON_DENSE_GENERAL_IN_LABEL: _muon_transform(
              _MUON_DENSE_GENERAL_IN_LABEL
          ),
          _MUON_EINSUM_ATTENTION_IN_LABEL: _muon_transform(
              _MUON_EINSUM_ATTENTION_IN_LABEL
          ),
          _MUON_DENSE_GENERAL_OUT_LABEL: _muon_transform(
              _MUON_DENSE_GENERAL_OUT_LABEL
          ),
          _ADAM_LABEL: adam,
      },
      param_labels=create_muon_param_labels,
  )
