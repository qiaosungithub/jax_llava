"""Utilities for splitting/merging params for partial-freeze training.

Inspired by /kmh-nfs-ssd-us-mount/code/qiao/work/TS-imgnet/utils/frozen_util.py.

The flow is:
    trainable_params, frozen_params = extract_trainable_parameters(params, prefixes)
    # … run grad_fn on trainable_params, merge with frozen_params inside loss_fn
    full_params = merge_params(trainable_params, frozen_params)

Prefixes are matched against the flattened key "a/b/c". A prefix `p` matches a
key `k` when `k == p` or `k` starts with `p + "/"`.
"""

from typing import Iterable, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax import traverse_util

from utils.llm_util import LOC_TOKEN_END, LOC_TOKEN_START
from utils.state_util import flatten_state_dict


def merge_params(params_a, params_b):
  """Merge two disjoint nested-dict pytrees back into a single tree.

  The two inputs must have non-overlapping flat keys; the resulting tree has
  the union of those keys.
  """
  flat_a = flatten_state_dict(params_a)
  flat_b = flatten_state_dict(params_b)
  overlap = set(flat_a) & set(flat_b)
  assert not overlap, f"merge_params: overlapping keys: {sorted(overlap)[:5]} …"
  merged = {**flat_a, **flat_b}
  return traverse_util.unflatten_dict(merged, sep="/")


def _key_matches_prefix(flat_key: str, prefix: str) -> bool:
  return flat_key == prefix or flat_key.startswith(prefix + "/")


def extract_trainable_parameters(
    params,
    trainable_prefixes: Sequence[str],
) -> Tuple[dict, dict]:
  """Split `params` into (trainable, frozen) according to `trainable_prefixes`.

  A leaf is trainable iff its flattened key matches at least one of
  `trainable_prefixes` under :func:`_key_matches_prefix`.

  Returns two unflattened dicts whose union round-trips back to `params`.
  """
  flat_params = flatten_state_dict(params)

  trainable_flat = {}
  frozen_flat = {}
  for k, v in flat_params.items():
    is_trainable = any(_key_matches_prefix(k, p) for p in trainable_prefixes)
    if is_trainable:
      trainable_flat[k] = v
    else:
      frozen_flat[k] = v

  trainable = traverse_util.unflatten_dict(trainable_flat, sep="/")
  frozen = traverse_util.unflatten_dict(frozen_flat, sep="/")

  # sanity: round-trip preserves structure
  merged = merge_params(trainable, frozen)
  assert jax.tree_util.tree_structure(merged) == jax.tree_util.tree_structure(
      params
  ), "extract_trainable_parameters: round-trip structure mismatch"

  return trainable, frozen


# ---------------------------------------------------------------------------
# Higher-level helper that names *which* parts of the VLM are trainable.
# ---------------------------------------------------------------------------
# Top-level groups in PaliGemmaEncDec params:
#   - image_encoder  (PrefixMAE)
#   - lm_backbone    (Gemma)
#   - projector      (image→LM projector)

_ALL_VLM_GROUPS = ("image_encoder", "lm_backbone", "projector")
_LOC_EMBEDDING_KEY = "lm_backbone/embedder/input_embedding"
_LOC_EMBEDDING_PARTS = tuple(_LOC_EMBEDDING_KEY.split("/"))


def _can_train_loc_embedding(flat_params) -> bool:
  emb = flat_params.get(_LOC_EMBEDDING_KEY, None)
  shape = getattr(emb, "shape", None)
  return shape is not None and len(shape) >= 1 and int(shape[0]) > LOC_TOKEN_START


def _get_path(tree, parts):
  cur = tree
  for part in parts:
    cur = cur[part]
  return cur


def _has_path(tree, parts) -> bool:
  try:
    _get_path(tree, parts)
    return True
  except (KeyError, TypeError):
    return False


def _set_path(tree, parts, value):
  if not parts:
    return value
  if isinstance(tree, FrozenDict):
    out = tree.unfreeze()
    out[parts[0]] = _set_path(out[parts[0]], parts[1:], value)
    return FrozenDict(out)
  out = dict(tree)
  out[parts[0]] = _set_path(out[parts[0]], parts[1:], value)
  return out


def _loc_slice_for_embedding(embedding):
  start = LOC_TOKEN_START
  end = min(LOC_TOKEN_END, int(embedding.shape[0]))
  if start >= end:
    return None
  return start, end


def merge_params_trainable_loc_embeddings(
    wrt_params,
    frozen_params,
    current_params,
    enable: bool = True,
):
  """Merge params while keeping non-loc embedding rows frozen.

  When the LM is frozen but loc embeddings are trainable, the whole embedding
  leaf must be present in `wrt_params`.  This helper uses the current params as
  a stop-gradient source for non-loc rows, then writes trainable loc rows back.
  """
  params = merge_params(wrt_params, frozen_params) if frozen_params else wrt_params
  if not enable or not _has_path(params, _LOC_EMBEDDING_PARTS):
    return params

  emb = _get_path(params, _LOC_EMBEDDING_PARTS)
  loc_slice = _loc_slice_for_embedding(emb)
  if loc_slice is None or not _has_path(current_params, _LOC_EMBEDDING_PARTS):
    return params

  start, end = loc_slice
  base_emb = _get_path(current_params, _LOC_EMBEDDING_PARTS)
  merged_emb = jax.lax.stop_gradient(base_emb).at[start:end].set(emb[start:end])
  return _set_path(params, _LOC_EMBEDDING_PARTS, merged_emb)


def zero_nonloc_embedding_rows(tree):
  """Zero every update/gradient row except the loc-token embedding rows."""
  if not _has_path(tree, _LOC_EMBEDDING_PARTS):
    return tree
  emb = _get_path(tree, _LOC_EMBEDDING_PARTS)
  loc_slice = _loc_slice_for_embedding(emb)
  if loc_slice is None:
    return tree

  start, end = loc_slice
  row_mask = jnp.zeros((emb.shape[0],), dtype=bool).at[start:end].set(True)
  while row_mask.ndim < emb.ndim:
    row_mask = row_mask[..., None]
  masked_emb = jnp.where(row_mask, emb, jnp.zeros_like(emb))
  return _set_path(tree, _LOC_EMBEDDING_PARTS, masked_emb)


def get_trainable(
    params,
    freeze_lm: bool = False,
    txt_feature_layer: int = 0,
    freeze_image_encoder: bool = False,
    train_loc_embeddings_when_lm_frozen: bool = True,
):
  """Split VLM params into (trainable, frozen) according to flags.

  Args:
    params: the model params dict (unfrozen).
    freeze_lm: if True, freeze part of lm_backbone.
    txt_feature_layer: when freeze_lm=True and txt_feature_layer > 0, only
      the lm_backbone embedder and blocks 0..txt_feature_layer-1 are frozen
      (they act as a fixed text feature extractor).  When txt_feature_layer==0
      the entire lm_backbone is frozen (original behaviour).
    freeze_image_encoder: if True, freeze the vision tower.  This matches the
      standard LLaVA-1.5 projector/LM tuning setup.
    train_loc_embeddings_when_lm_frozen: if True, keep only the
      <loc0000>..<loc1023> rows in lm_backbone/embedder/input_embedding
      trainable while the rest of the LM stays frozen. Row-level freezing is
      enforced in train_step and the optimizer update transform.

  Returns:
    (trainable_params, frozen_params). When nothing is frozen,
    `frozen_params` is an empty dict.
  """
  if not freeze_lm and not freeze_image_encoder:
    return params, {}

  flat_params = flatten_state_dict(params)
  flat_keys = list(flat_params.keys())
  train_loc_embedding = (
      bool(train_loc_embeddings_when_lm_frozen)
      and freeze_lm
      and _can_train_loc_embedding(flat_params)
  )
  top_level = {k.split("/", 1)[0] for k in flat_keys}
  unknown = top_level - set(_ALL_VLM_GROUPS)
  assert not unknown, (
      f"get_trainable: unexpected top-level params {sorted(unknown)}; "
      f"expected subset of {_ALL_VLM_GROUPS}"
  )

  frozen_prefixes = set()
  if freeze_image_encoder:
    frozen_prefixes.add("image_encoder")

  if not freeze_lm:
    trainable_prefixes = [g for g in _ALL_VLM_GROUPS if g not in frozen_prefixes]
  elif txt_feature_layer == 0:
    # Freeze all of lm_backbone except optional loc-token embedding rows.
    frozen_prefixes.add("lm_backbone")
    trainable_prefixes = [g for g in _ALL_VLM_GROUPS if g not in frozen_prefixes]
    if train_loc_embedding:
      trainable_prefixes.append(_LOC_EMBEDDING_KEY)
  else:
    # Freeze only lm_backbone/embedder + lm_backbone/layer_0..N-1.
    # Remaining LM layers (N..last) and final_norm stay trainable.
    N = txt_feature_layer
    frozen_lm_prefixes = {"lm_backbone/embedder"} | {
        f"lm_backbone/layer_{i}" for i in range(N)
    }
    # Collect all lm_backbone sub-keys to derive the trainable remainder.
    lm_flat_keys = [k for k in flat_keys if k.startswith("lm_backbone/")]
    trainable_lm_prefixes = sorted({
        "/".join(k.split("/")[:2])  # e.g. "lm_backbone/layer_12"
        for k in lm_flat_keys
        if not any(k.startswith(fp + "/") or k == fp for fp in frozen_lm_prefixes)
    })
    trainable_prefixes = (
        [g for g in _ALL_VLM_GROUPS if g not in (frozen_prefixes | {"lm_backbone"})]
        + trainable_lm_prefixes
    )
    if train_loc_embedding:
      trainable_prefixes.append(_LOC_EMBEDDING_KEY)

  trainable, frozen = extract_trainable_parameters(params, trainable_prefixes)
  assert frozen, "freeze was requested but no params were frozen"
  return trainable, frozen


def label_trainable_frozen_params(
    params,
    freeze_lm: bool = False,
    txt_feature_layer: int = 0,
    freeze_image_encoder: bool = False,
    image_prefix: str = "image_encoder",
    train_loc_embeddings_when_lm_frozen: bool = True,
):
  """Label params for optimizer partitioning.

  Labels:
    frozen: leaves excluded from training; use optax.set_to_zero().
    img: trainable image encoder leaves; use vision LR schedule.
    main: all other trainable leaves; use normal LR schedule.

  The returned pytree has the same structure as `params` and can be passed
  directly to optax.multi_transform.
  """
  _, frozen = get_trainable(
      params,
      freeze_lm=freeze_lm,
      txt_feature_layer=txt_feature_layer,
      freeze_image_encoder=freeze_image_encoder,
      train_loc_embeddings_when_lm_frozen=train_loc_embeddings_when_lm_frozen,
  )
  frozen_keys = set(flatten_state_dict(frozen).keys()) if frozen else set()
  flat_params = flatten_state_dict(params)

  labels_flat = {}
  for key in flat_params.keys():
    if key in frozen_keys:
      labels_flat[key] = "frozen"
    elif key == image_prefix or key.startswith(image_prefix + "/"):
      labels_flat[key] = "img"
    else:
      labels_flat[key] = "main"

  labels = traverse_util.unflatten_dict(labels_flat, sep="/")
  assert jax.tree_util.tree_structure(labels) == jax.tree_util.tree_structure(
      params
  ), "label_trainable_frozen_params: label tree does not match params"
  return labels
