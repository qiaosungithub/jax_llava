"""Shared helpers for eval result files."""

import os
import re
import time
import uuid

import jax
import numpy as np
from jax.experimental import multihost_utils as mu

_RUN_ID_BUF_SIZE = 128


def _safe_path_part(value) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text or "none"


def _broadcast_string_from_source(value, is_source):
    data = value.encode("utf-8") if is_source else b""
    if len(data) >= _RUN_ID_BUF_SIZE:
        raise ValueError(f"eval run id is too long: {value}")
    buf = np.zeros((_RUN_ID_BUF_SIZE,), dtype=np.uint8)
    if is_source:
        buf[:len(data)] = np.frombuffer(data, dtype=np.uint8)
    out = np.asarray(mu.broadcast_one_to_all(buf, is_source=is_source))
    zero = np.where(out == 0)[0]
    end = int(zero[0]) if len(zero) else len(out)
    return bytes(out[:end].tolist()).decode("utf-8")


def _set_eval_config_value(config, key: str, value) -> None:
    eval_cfg = config.eval
    try:
        setattr(eval_cfg, key, value)
        return
    except (AttributeError, KeyError):
        unlock = getattr(eval_cfg, "unlocked", None)
        if unlock is None:
            raise
    with unlock():
        setattr(eval_cfg, key, value)


def set_eval_result_context(config, step: int, run_id: str, suffix: str) -> None:
    _set_eval_config_value(config, "current_eval_step", int(step))
    _set_eval_config_value(config, "current_eval_run_id", run_id)
    _set_eval_config_value(config, "current_eval_suffix", suffix or "main")


def _eval_run_id(config) -> str:
    run_id = getattr(config.eval, "current_eval_run_id", "")
    if run_id and run_id != "manual":
        return _safe_path_part(run_id)

    is_source = jax.process_index() == 0
    local_run_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}" if is_source else ""
    run_id = _broadcast_string_from_source(local_run_id, is_source)
    _set_eval_config_value(config, "current_eval_run_id", run_id)
    return _safe_path_part(run_id)


def ensure_eval_result_base_dir(base_dir: str) -> None:
    """Ensure ranks can create new result files in the shared cache dir."""
    os.makedirs(base_dir, exist_ok=True)
    try:
        os.chmod(base_dir, 0o777)
    except PermissionError:
        pass
    if not os.access(base_dir, os.W_OK | os.X_OK):
        raise PermissionError(
            f"Eval result cache dir is not writable/searchable: {base_dir}. "
            "The eval writer creates new uniquely named result files here; "
            "the directory itself must allow file creation."
        )


def eval_result_prefix(
    config,
    cache_dir_attr: str,
    default_cache_dir: str,
    task_name: str,
    *name_parts,
) -> tuple[str, str]:
    """Return ``(base_dir, file_prefix)`` for fresh per-eval result files.

    Eval result files use fixed names like results_0.json. Keeping them in a
    stable persistent directory means a different user can later try to truncate
    an existing file they do not own. The run_eval_tasks wrapper sets
    current_eval_run_id once per eval pass so every process agrees on unique
    filenames while still writing flat files directly in the shared cache dir.

    The per-task cache-dir config remains useful as a namespace, but the result
    files are written to its parent directory by default. For example,
    /data/cached/zhh/vqav2_eval becomes flat files under /data/cached/zhh named
    vqav2_eval__<workdir>__step_<n>__...
    """
    namespace_source = os.path.normpath(str(getattr(config.eval, cache_dir_attr, default_cache_dir)))
    namespace = _safe_path_part(os.path.basename(namespace_source) or task_name)
    base = getattr(config.eval, "result_cache_dir", None)
    if base is None:
        base = os.path.dirname(namespace_source) or "."
    base = os.path.normpath(str(base))
    workdir_hash = _safe_path_part(getattr(config, "workdir_hash", "nohash"))
    step = int(getattr(config.eval, "current_eval_step", -1))
    run_id = _eval_run_id(config)
    suffix = _safe_path_part(getattr(config.eval, "current_eval_suffix", ""))

    parts = [
        namespace,
        workdir_hash,
        f"step_{step}",
        suffix,
        _safe_path_part(task_name),
        run_id,
    ]
    parts.extend(_safe_path_part(p) for p in name_parts)
    return base, os.path.join(base, "__".join(parts))
