import hashlib
import pickle
import re

import fsspec
import jax

from utils.ckpt_util import _resolve_checkpoint_path, convert_to_gs
from utils.logging_util import log_for_0


STATE_VERSION = 1
_REPLICA_DATA_BUCKET_RE = re.compile(
    r"^(gs://kmh-gcp-(?:us-central1|us-east5|asia-northeast1-b)/data)(/.*)?$"
)
_LOGICAL_DATA_BUCKET_PREFIX = "gs://kmh-gcp-<replica>/data"


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


def _canonicalize_data_path(value):
    if not isinstance(value, str):
        return value
    match = _REPLICA_DATA_BUCKET_RE.match(value)
    if not match:
        return value
    suffix = match.group(2) or ""
    return _LOGICAL_DATA_BUCKET_PREFIX + suffix


def _canonicalize_data_paths(value):
    if isinstance(value, dict):
        return {str(k): _canonicalize_data_paths(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_data_paths(v) for v in value]
    return _canonicalize_data_path(value)


def _topology_compare_payload(topology):
    payload = {
        k: _as_plain(v)
        for k, v in dict(topology).items()
        if k not in ("hash", "logical_roots")
    }
    payload["roots"] = _canonicalize_data_paths(payload.get("roots", []))
    return payload


def _topology_hash(topology):
    encoded = repr(sorted(_topology_compare_payload(topology).items())).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _flatten_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_strings(item)


def _data_bucket_prefix(value):
    match = _REPLICA_DATA_BUCKET_RE.match(value) if isinstance(value, str) else None
    return match.group(1) if match else None


def _build_replica_prefix_remap(saved_roots, current_roots):
    """Map saved same-dataset replica buckets to the current zone-local buckets."""
    saved = list(_flatten_strings(saved_roots))
    current = list(_flatten_strings(current_roots))
    prefix_remap = {}
    if len(saved) != len(current):
        return prefix_remap

    for saved_root, current_root in zip(saved, current):
        if _canonicalize_data_path(saved_root) != _canonicalize_data_path(current_root):
            continue
        saved_prefix = _data_bucket_prefix(saved_root)
        current_prefix = _data_bucket_prefix(current_root)
        if not saved_prefix or not current_prefix or saved_prefix == current_prefix:
            continue
        previous = prefix_remap.get(saved_prefix)
        if previous is not None and previous != current_prefix:
            raise ValueError(
                "Ambiguous dataloader state bucket remap: "
                f"{saved_prefix} -> both {previous} and {current_prefix}"
            )
        prefix_remap[saved_prefix] = current_prefix
    return prefix_remap


def _remap_string_data_bucket(value, prefix_remap):
    for saved_prefix, current_prefix in sorted(prefix_remap.items(), key=lambda item: -len(item[0])):
        if value == saved_prefix or value.startswith(saved_prefix + "/"):
            return current_prefix + value[len(saved_prefix):], 1
    return value, 0


def _remap_state_data_buckets(value, prefix_remap):
    if not prefix_remap:
        return value, 0
    if isinstance(value, str):
        return _remap_string_data_bucket(value, prefix_remap)
    if isinstance(value, list):
        remapped = []
        count = 0
        for item in value:
            new_item, item_count = _remap_state_data_buckets(item, prefix_remap)
            remapped.append(new_item)
            count += item_count
        return remapped, count
    if isinstance(value, tuple):
        remapped = []
        count = 0
        for item in value:
            new_item, item_count = _remap_state_data_buckets(item, prefix_remap)
            remapped.append(new_item)
            count += item_count
        return tuple(remapped), count
    if isinstance(value, dict):
        remapped = {}
        count = 0
        for key, item in value.items():
            new_key, key_count = _remap_state_data_buckets(key, prefix_remap)
            new_item, item_count = _remap_state_data_buckets(item, prefix_remap)
            remapped[new_key] = new_item
            count += key_count + item_count
        return remapped, count
    return value, 0


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
    payload["logical_roots"] = _canonicalize_data_paths(payload["roots"])
    payload["hash"] = _topology_hash(payload)
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
    saved_compare = _topology_compare_payload(saved_topology)
    expected_compare = _topology_compare_payload(expected_topology)
    if saved_compare != expected_compare:
        raise ValueError(
            "Dataloader topology changed; exact stateful resume is not valid. "
            f"saved={saved_topology}, current={expected_topology}"
        )
    prefix_remap = _build_replica_prefix_remap(
        saved_topology.get("roots", []),
        expected_topology.get("roots", []),
    )
    loader_state = payload["loader_state"]
    if prefix_remap:
        loader_state, remap_count = _remap_state_data_buckets(loader_state, prefix_remap)
        log_for_0(
            "Dataloader state uses same logical dataset replicas; remapped %d "
            "state URL entries with bucket prefixes %s.",
            remap_count,
            prefix_remap,
        )

    train_loader.load_state_dict(loader_state)
    log_for_0("Dataloader state restored from %s.", state_path)
    return True
