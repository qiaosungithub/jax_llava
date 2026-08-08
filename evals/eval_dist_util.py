"""Shared multi-host eval utilities for evals/eval_*.py.

Collects the pieces that were duplicated (nearly verbatim) across the
per-benchmark eval files:
  - DistributedEvalSampler: deterministic strided eval sampler.
  - collate_fn: filter-None + tensor-stack + prefix_len int32 + aux-as-list.
  - write_rank_json_results / gather_rank_json_results: per-rank JSON result
    files ({prefix}.results_{process_index}.json) and rank-0 merge readback.
  - broadcast_merge_ok: multi-host guardrail so that a rank-0 merge failure
    raises cleanly on every host instead of hanging the other hosts in the
    next sync_global_devices barrier.

Per-record validation/scoring logic stays in the individual eval files.
"""

import json
import os

import fsspec
import jax
import numpy as np
import torch
from jax.experimental import multihost_utils as mu
from torch.utils.data import Sampler

from utils.logging_util import log_for_0


class DistributedEvalSampler(Sampler):
    """Deterministic eval-only sampler without padding/duplication."""

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = jax.process_count()
        if rank is None:
            rank = jax.process_index()
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.dataset_len = len(dataset)
        self.num_samples = (
            self.dataset_len - self.rank + self.num_replicas - 1
        ) // self.num_replicas

    def __iter__(self):
        return iter(range(self.rank, self.dataset_len, self.num_replicas))

    def __len__(self):
        return self.num_samples


def collate_fn(batch):
    """Shared eval collate: drop None samples, stack tensors, keep aux as list.

    ``prefix_len`` may be a 0-dim int32 tensor (most evals) or a plain python
    int (eval_mme); both end up as an int32 tensor of shape (B,). Non-tensor
    keys other than ``prefix_len``/``aux`` are dropped (none exist today).
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    collated = {}
    first = batch[0]
    for key, value in first.items():
        if isinstance(value, torch.Tensor):
            collated[key] = torch.stack([b[key] for b in batch])
        elif key == "prefix_len":
            collated[key] = torch.tensor([b[key] for b in batch], dtype=torch.int32)
        elif key == "aux":
            collated[key] = [b[key] for b in batch]
    return collated


def write_json(path, obj, indent=2):
    """Write JSON to `path`, on CNS or locally.

    Every eval writes its merged results and metrics with the builtin open(),
    which cannot create a file on Colossus -- and on Borg these now live under
    $CHECKPOINT_BUCKET. One helper so a new eval cannot reintroduce it.
    """
    payload = json.dumps(obj, ensure_ascii=False, indent=indent)
    if is_cns_path(path):
        from google3.pyglib import gfile
        with gfile.Open(str(path), "w") as f:
            f.write(payload)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload)
    return path


def read_json(path):
    """Read JSON from `path`, on CNS or locally."""
    if is_cns_path(path):
        from google3.pyglib import gfile
        with gfile.Open(str(path), "r") as f:
            return json.load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_rank_json_results(result_prefix, results):
    """Write this rank's results to ``{result_prefix}.results_{process_index}.json``."""
    res_file = f"{result_prefix}.results_{jax.process_index()}.json"
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    if is_cns_path(res_file):
        # The builtin open() cannot create a file on Colossus, and on Borg the
        # results now live under $CHECKPOINT_BUCKET.
        from google3.pyglib import gfile
        with gfile.Open(res_file, "w") as f:
            f.write(payload)
    else:
        with open(res_file, "w", encoding="utf-8") as f:
            f.write(payload)
    return res_file


def gather_rank_json_results(
    result_prefix,
    missing_file_msg=None,
    log_missing=False,
    process_count=None,
):
    """Rank-0 merge readback of all ranks' ``{prefix}.results_{rank}.json`` files.

    Args:
      result_prefix: same prefix passed to write_rank_json_results.
      missing_file_msg: format string with ``{rank}`` / ``{path}`` placeholders
        used for the FileNotFoundError when a rank file is missing. ``None``
        means silently skip missing files (eval_refcocog behavior).
      log_missing: additionally log_for_0 a missing rank file before raising.
      process_count: defaults to jax.process_count().

    Returns the concatenated list of all rank results (write order).
    """
    if process_count is None:
        process_count = jax.process_count()
    all_results = []
    for r in range(process_count):
        pf = f"{result_prefix}.results_{r}.json"
        if not _path_exists(pf):
            if missing_file_msg is None:
                continue
            if log_missing:
                log_for_0(f"Process {r} results file not found: {pf}")
            raise FileNotFoundError(missing_file_msg.format(rank=r, path=pf))
        with (eval_open(pf, "r") if is_cns_path(pf)
              else open(pf, encoding="utf-8")) as f:
            all_results.extend(json.load(f))
    return all_results


def broadcast_merge_ok(merge_exception, eval_name):
    """Multi-host guardrail after a rank-0-only merge/validation section.

    Rank 0 broadcasts whether its merge succeeded. On failure every process
    raises (rank 0 re-raises the original exception, other ranks raise a
    RuntimeError) instead of hanging in the next sync_global_devices barrier.

    MUST be called at a point reached by ALL processes, never inside a
    rank-0-only branch. ``merge_exception`` must be None on non-zero ranks.
    """
    merge_ok = mu.broadcast_one_to_all(
        np.asarray([merge_exception is None], dtype=np.int32),
        is_source=jax.process_index() == 0,
    )
    if not bool(np.asarray(jax.device_get(merge_ok)).reshape(-1)[0]):
        if merge_exception is not None:
            raise merge_exception
        raise RuntimeError(
            f"{eval_name} rank-0 merge/validation failed; see process-0 logs"
        )


def is_cns_path(path) -> bool:
    """True for a Colossus/bigstore path, which fsspec cannot see."""
    return str(path).startswith(("/cns/", "/bigstore/"))


def eval_glob(pattern: str):
    """Sorted matches for `pattern`, on whichever filesystem names it.

    fsspec's "file" backend silently reports NO MATCHES for a /cns/ path rather
    than failing, so an eval pointed at a Colossus replica sees an empty
    directory and raises "no shards found" naming the directory its shards are
    actually in. Every eval that lists shards must come through here.
    """
    if is_cns_path(pattern):
        from google3.pyglib import gfile
        return sorted(str(u) for u in gfile.Glob(pattern))
    if str(pattern).startswith("gs://"):
        fs = fsspec.filesystem("gs")
        return sorted(u if str(u).startswith("gs://") else f"gs://{u}"
                      for u in fs.glob(pattern))
    fs, fs_path = fsspec.core.url_to_fs(pattern)
    return sorted(fs.glob(fs_path))


def eval_open(path: str, mode: str = "rb"):
    """Open a file on CNS, GCS or locally."""
    if is_cns_path(path):
        from google3.pyglib import gfile
        return gfile.Open(path, mode)
    return fsspec.open(path, mode).open()


def _path_exists(path: str) -> bool:
    if is_cns_path(path):
        from google3.pyglib import gfile
        return gfile.Exists(path)
    fs, fs_path = fsspec.core.url_to_fs(path)
    return fs.exists(fs_path)


def _join_path(root: str, leaf: str) -> str:
    return f"{root.rstrip('/')}/{leaf.lstrip('/')}"
