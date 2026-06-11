import hashlib
import pickle

import fsspec
import jax

from utils.ckpt_util import _resolve_checkpoint_path, convert_to_gs
from utils.logging_util import log_for_0


STATE_VERSION = 1


def stateful_dataloader_enabled(config) -> bool:
    return bool(getattr(config.dataset, "stateful_dataloader", False))


def stateful_dataloader_strict(config) -> bool:
    return bool(getattr(config.dataset, "stateful_dataloader_strict", True))


def _as_plain(value):
    if isinstance(value, dict):
        return {str(k): _as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(v) for v in value]
    return value


def _storage_path(path: str) -> str:
    return path[5:] if path.startswith("gs://") else path


def dataloader_topology(config, batch_size):
    """State that must match for exact dataloader resume."""
    roots = list(getattr(config.dataset, "root", []) or [])
    types = list(getattr(config.dataset, "types", []) or [])
    weights = list(getattr(config.dataset, "mix_weights", []) or [])
    payload = {
        "jax_process_count": int(jax.process_count()),
        "process_batch_size": int(batch_size),
        "num_workers": int(getattr(config.dataset, "num_workers", 0)),
        "prefetch_factor": int(getattr(config.dataset, "prefetch_factor", 0)),
        "data_seed_offset": int(getattr(config.dataset, "data_seed_offset", 0)),
        "roots": _as_plain(roots),
        "types": _as_plain(types),
        "mix_weights": _as_plain(weights),
    }
    encoded = repr(sorted(payload.items())).encode("utf-8")
    payload["hash"] = hashlib.sha256(encoded).hexdigest()[:16]
    return payload


def _state_path_from_checkpoint(checkpoint_path: str) -> str:
    return (
        f"{checkpoint_path.rstrip('/')}/dataloader_state/"
        f"process_{jax.process_index():05d}.pkl"
    )


def _checkpoint_path_for_workdir(workdir: str, step: int) -> str:
    return f"{convert_to_gs(workdir).rstrip('/')}/checkpoint_{int(step)}"


def save_dataloader_state(train_loader, config, workdir, step, batch_size):
    if not stateful_dataloader_enabled(config):
        return
    if not hasattr(train_loader, "state_dict"):
        raise TypeError("stateful_dataloader is enabled but train_loader has no state_dict().")

    checkpoint_path = _checkpoint_path_for_workdir(workdir, step)
    state_path = _state_path_from_checkpoint(checkpoint_path)
    payload = {
        "version": STATE_VERSION,
        "step": int(step),
        "process_index": int(jax.process_index()),
        "process_count": int(jax.process_count()),
        "topology": dataloader_topology(config, batch_size),
        "loader_state": train_loader.state_dict(),
    }

    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    tmp_path = state_path + ".tmp"
    fs, _, paths = fsspec.get_fs_token_paths(tmp_path)
    tmp_storage_path = paths[0]
    final_storage_path = _storage_path(state_path)
    fs.pipe_file(tmp_storage_path, data)
    if fs.exists(final_storage_path):
        fs.rm(final_storage_path)
    fs.mv(tmp_storage_path, final_storage_path)
    log_for_0("Dataloader state saved to %s.", state_path)


def restore_dataloader_state(train_loader, config, load_from, zone, expected_step, batch_size):
    if not stateful_dataloader_enabled(config):
        return False
    if not hasattr(train_loader, "load_state_dict"):
        raise TypeError("stateful_dataloader is enabled but train_loader has no load_state_dict().")

    checkpoint_path = _resolve_checkpoint_path(load_from, zone)
    state_path = _state_path_from_checkpoint(checkpoint_path)
    fs = fsspec.open(state_path, "rb").fs
    if not fs.exists(_storage_path(state_path)):
        if stateful_dataloader_strict(config):
            raise FileNotFoundError(
                f"Missing dataloader state for exact resume: {state_path}. "
                "Use the previous complete checkpoint or disable stateful_dataloader."
            )
        log_for_0("No dataloader state at %s; continuing without exact loader resume.", state_path)
        return False

    with fsspec.open(state_path, "rb") as f:
        payload = pickle.load(f)

    if int(payload.get("version", -1)) != STATE_VERSION:
        raise ValueError(f"Unsupported dataloader state version in {state_path}: {payload.get('version')}")
    if int(payload.get("step", -1)) != int(expected_step):
        raise ValueError(
            f"Dataloader state step mismatch for {state_path}: "
            f"expected {expected_step}, found {payload.get('step')}"
        )
    if int(payload.get("process_index", -1)) != int(jax.process_index()):
        raise ValueError(f"Dataloader state process index mismatch in {state_path}.")
    if int(payload.get("process_count", -1)) != int(jax.process_count()):
        raise ValueError(f"Dataloader state process count mismatch in {state_path}.")

    expected_topology = dataloader_topology(config, batch_size)
    saved_topology = payload.get("topology", {})
    if saved_topology.get("hash") != expected_topology.get("hash"):
        raise ValueError(
            "Dataloader topology changed; exact stateful resume is not valid. "
            f"saved={saved_topology}, current={expected_topology}"
        )

    train_loader.load_state_dict(payload["loader_state"])
    log_for_0("Dataloader state restored from %s.", state_path)
    return True
