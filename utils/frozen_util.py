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
from flax import traverse_util

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


def get_trainable(params, freeze_lm: bool = False, txt_feature_layer: int = 0):
  """Split VLM params into (trainable, frozen) according to flags.

  Args:
    params: the model params dict (unfrozen).
    freeze_lm: if True, freeze part of lm_backbone.
    txt_feature_layer: when freeze_lm=True and txt_feature_layer > 0, only
      the lm_backbone embedder and blocks 0..txt_feature_layer-1 are frozen
      (they act as a fixed text feature extractor).  When txt_feature_layer==0
      the entire lm_backbone is frozen (original behaviour).

  Returns:
    (trainable_params, frozen_params). When nothing is frozen,
    `frozen_params` is an empty dict.
  """
  if not freeze_lm:
    return params, {}

  flat_keys = list(flatten_state_dict(params).keys())
  top_level = {k.split("/", 1)[0] for k in flat_keys}
  unknown = top_level - set(_ALL_VLM_GROUPS)
  assert not unknown, (
      f"get_trainable: unexpected top-level params {sorted(unknown)}; "
      f"expected subset of {_ALL_VLM_GROUPS}"
  )

  if txt_feature_layer == 0:
    # Original behaviour: freeze all of lm_backbone.
    trainable_prefixes = [g for g in _ALL_VLM_GROUPS if g != "lm_backbone"]
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
        [g for g in _ALL_VLM_GROUPS if g != "lm_backbone"]
        + trainable_lm_prefixes
    )

  trainable, frozen = extract_trainable_parameters(params, trainable_prefixes)
  assert frozen, "freeze_lm=True but no params were frozen"
  return trainable, frozen


def label_trainable_frozen_params(
    params,
    freeze_lm: bool = False,
    txt_feature_layer: int = 0,
    image_prefix: str = "image_encoder",
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
      params, freeze_lm=freeze_lm, txt_feature_layer=txt_feature_layer
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
