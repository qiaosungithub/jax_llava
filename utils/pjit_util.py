"""jit/HSDP sharding helpers for JAX training."""

import enum
import functools
from typing import Tuple

import jax
import jax.tree_util as jtu
import numpy as np
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

_mesh = None
_mode = "hsdp"


def log_for_0(*args, **kwargs):
    from utils.logging_util import log_for_0 as _log_for_0
    return _log_for_0(*args, **kwargs)


class MeshMode(enum.Enum):
    DATA = enum.auto()
    MODEL = enum.auto()


TOPOLOGIES = {
    "v6": {
        4: (1, 4),
        16: (4, 4),
        32: (8, 4),
        64: (8, 8),
        128: (8, 16),
    },
    "v5": {
        4: (1, 4),
        8: (2, 4),
        16: (4, 4),
        32: (2, 4, 4),
        64: (4, 4, 4),
        128: (8, 4, 4),
    },
    "v4": {
        4: (4, 1),
        16: (4, 4),
        32: (2, 4, 4),
        64: (4, 4, 4),
    },
}


def _get_shape(x):
    if hasattr(x, "shape"):
        return "shape", x.shape
    if hasattr(x, "value"):
        return "value.shape", x.value.shape
    if isinstance(x, Tuple):
        return "self", x
    if isinstance(x, (int, float)):
        return "scalar", ()
    return "unknown", x


def _as_tuple_spec(spec):
    return tuple(spec) if spec is not None else ()


def _validate_sharding_spec(leaf, spec, mesh_shape):
    if spec is None:
        return True, "ok"
    msg, shape = _get_shape(leaf)
    if msg == "unknown":
        return False, f"cannot get shape for {leaf}"
    spec_tuple = _as_tuple_spec(spec)
    if len(shape) < len(spec_tuple):
        return False, f"shape {shape} has fewer dims than spec {spec}"
    for dim, axis_rule in enumerate(spec_tuple):
        if axis_rule is None:
            continue
        divisor = 1
        if isinstance(axis_rule, str):
            divisor = mesh_shape.get(axis_rule, 0)
        elif isinstance(axis_rule, (tuple, list)):
            for axis_name in axis_rule:
                divisor *= mesh_shape.get(axis_name, 0)
        else:
            return False, f"unsupported axis rule {axis_rule}"
        if divisor == 0:
            return False, f"axis rule {axis_rule} missing from mesh"
        if shape[dim] % divisor != 0:
            return False, f"dim {dim}={shape[dim]} not divisible by {divisor}"
    return True, "ok"


def _make_valid_spec(leaf, spec, mesh_shape):
    if spec is None:
        return P()
    msg, shape = _get_shape(leaf)
    if msg == "unknown":
        return P()
    spec_tuple = _as_tuple_spec(spec)
    if len(shape) < len(spec_tuple):
        spec_tuple = spec_tuple[-len(shape):]
    if len(shape) > len(spec_tuple):
        spec_tuple = (None,) * (len(shape) - len(spec_tuple)) + spec_tuple

    out = []
    for dim, axis_rule in enumerate(spec_tuple):
        if axis_rule is None:
            out.append(None)
            continue
        divisor = 1
        if isinstance(axis_rule, str):
            divisor = mesh_shape.get(axis_rule, 0)
        elif isinstance(axis_rule, (tuple, list)):
            for axis_name in axis_rule:
                divisor *= mesh_shape.get(axis_name, 0)
        else:
            out.append(None)
            continue
        out.append(axis_rule if divisor and shape[dim] % divisor == 0 else None)
    return P() if all(x is None for x in out) else P(*out)


def apply_spec_to_last_dims(leaf, spec):
    msg, shape = _get_shape(leaf)
    if msg == "unknown":
        return P()
    spec_tuple = _as_tuple_spec(spec)
    if len(shape) < len(spec_tuple):
        return P(*spec_tuple[-len(shape):]) if shape else P()
    return P(*((None,) * (len(shape) - len(spec_tuple)) + spec_tuple))


def _model_axis_names(mesh: Mesh, sharding_mode: str):
    if str(sharding_mode).lower() in {"hsdp", "fsdp"} and len(mesh.axis_names) > 1:
        return (mesh.axis_names[-1],)
    return ()


def _data_axis_names(mesh: Mesh, sharding_mode: str):
    mode = str(sharding_mode).lower()
    if mode in {"hsdp", "fsdp"} and len(mesh.axis_names) > 1:
        return tuple(mesh.axis_names[:-1])
    return tuple(mesh.axis_names)


def _axis_size(axis_rule, mesh_shape):
    if axis_rule is None:
        return 1
    if isinstance(axis_rule, str):
        return mesh_shape.get(axis_rule, 0)
    size = 1
    for axis_name in axis_rule:
        size *= mesh_shape.get(axis_name, 0)
    return size


def get_spec_dict(tree, mesh: Mesh, param_mode: MeshMode, sharding_mode: str):
    sharding_mode = str(sharding_mode).lower()
    if param_mode == MeshMode.DATA:
        data_axes = _data_axis_names(mesh, sharding_mode)
        data_spec = P(data_axes) if data_axes else P()
        spec_tree = jax.tree.map(lambda _: data_spec, tree)
    elif sharding_mode == "ddp":
        spec_tree = jax.tree.map(lambda _: P(), tree)
    elif sharding_mode == "hsdp":
        model_shard = P(mesh.axis_names[-1])
        spec_tree = jax.tree.map(lambda leaf: apply_spec_to_last_dims(leaf, model_shard), tree)
    elif sharding_mode == "fsdp":
        if len(mesh.axis_names) == 1:
            model_shard = P(mesh.axis_names[-1])
        else:
            model_shard = P(mesh.axis_names[-1], tuple(mesh.axis_names[:-1]))
        spec_tree = jax.tree.map(lambda leaf: apply_spec_to_last_dims(leaf, model_shard), tree)
    else:
        raise ValueError(f"Unsupported sharding mode: {sharding_mode}")

    mesh_shape = {name: size for name, size in zip(mesh.axis_names, mesh.devices.shape)}
    tree_flat, _ = jtu.tree_flatten_with_path(tree)
    spec_flat, treedef = jtu.tree_flatten_with_path(spec_tree)
    if len(tree_flat) != len(spec_flat):
        raise ValueError(f"Tree/spec length mismatch: {len(tree_flat)} vs {len(spec_flat)}")

    new_specs = []
    for (path, leaf), (spec_path, spec) in zip(tree_flat, spec_flat):
        if path != spec_path:
            raise ValueError(f"Tree/spec path mismatch: {path} vs {spec_path}")
        valid, reason = _validate_sharding_spec(leaf, spec, mesh_shape)
        if valid:
            new_specs.append(spec)
            continue
        best_spec = _make_valid_spec(leaf, spec, mesh_shape)
        log_for_0(
            "Sharding fallback at %s: %s -> %s (%s)",
            jtu.keystr(path),
            spec,
            best_spec,
            reason,
        )
        new_specs.append(best_spec)
    return jtu.tree_unflatten(treedef, new_specs)


def get_mesh() -> Mesh:
    global_device_count = jax.device_count()
    device_kind = jax.local_devices()[0].device_kind.lower()
    mesh_shape = None
    for topo_name, topo_shapes in TOPOLOGIES.items():
        if topo_name in device_kind:
            if global_device_count not in topo_shapes:
                raise ValueError(
                    f"Unsupported device count {global_device_count} for TPU kind {device_kind}"
                )
            mesh_shape = topo_shapes[global_device_count]
            break
    if mesh_shape is None:
        # Local CPU/GPU debug path.
        mesh_shape = (global_device_count,)

    devices = mesh_utils.create_device_mesh(
        mesh_shape,
        allow_split_physical_axes=(len(mesh_shape) > 2),
    )
    return Mesh(devices, tuple(f"AXIS_{i}" for i in range(len(mesh_shape))))


def prepare_pjit_funcs(mode: str = "hsdp"):
    global _mesh, _mode
    mode = str(mode).lower()
    mesh = get_mesh()
    _mesh = mesh
    _mode = mode
    mesh_size = int(np.prod(mesh.devices.shape))
    mesh_dict = {name: size for name, size in zip(mesh.axis_names, mesh.devices.shape)}
    log_for_0("Setting up jit/HSDP mesh mode=%s shape=%s size=%d axes=%s", mode, mesh.devices.shape, mesh_size, mesh_dict)

    def get_partition_spec(tree, param_mode: MeshMode):
        return get_spec_dict(tree, mesh, param_mode=param_mode, sharding_mode=mode)

    def pjit_all_gather(tree):
        data_axes = _data_axis_names(mesh, mode)
        return multihost_utils.host_local_array_to_global_array(
            tree,
            mesh,
            P(data_axes) if data_axes else P(),
        )

    def pjit_reduce_scatter(tree, param_mode=MeshMode.DATA):
        spec = get_partition_spec(tree, param_mode=param_mode)
        return multihost_utils.global_array_to_host_local_array(tree, mesh, spec)

    def _to_named_sharding(shardings):
        def convert(spec):
            if isinstance(spec, NamedSharding):
                return spec
            return NamedSharding(mesh, spec)

        return jax.tree_util.tree_map(convert, shardings)

    def pjit_compile(fn, in_shardings, out_shardings, donate_argnums=()):
        compiled = jax.jit(
            fn,
            in_shardings=_to_named_sharding(in_shardings),
            out_shardings=_to_named_sharding(out_shardings),
            donate_argnums=donate_argnums,
        )

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with Mesh(mesh.devices, mesh.axis_names):
                return compiled(*args, **kwargs)

        def lower(*args, **kwargs):
            with Mesh(mesh.devices, mesh.axis_names):
                return compiled.lower(*args, **kwargs)

        wrapped.lower = lower
        return wrapped

    return mesh, get_partition_spec, pjit_all_gather, pjit_reduce_scatter, pjit_compile


def named_sharding(mesh, spec):
    return NamedSharding(mesh, spec)


def _current_mesh_shape():
    if _mesh is None:
        return {}
    return {name: size for name, size in zip(_mesh.axis_names, _mesh.devices.shape)}


def _valid_axis_for_dim(shape, dim, axis_rule):
    if axis_rule is None:
        return False
    if not shape:
        return False
    dim = dim if dim >= 0 else len(shape) + dim
    if dim < 0 or dim >= len(shape):
        return False
    divisor = _axis_size(axis_rule, _current_mesh_shape())
    return bool(divisor and shape[dim] % divisor == 0)


def _activation_spec(x, *, model_dim=None):
    """Best-effort activation spec: shard batch on data axes and one model dim."""
    if _mesh is None:
        return P()
    shape = getattr(x, "shape", ())
    if not shape:
        return P()
    spec = [None] * len(shape)

    data_axes = _data_axis_names(_mesh, _mode)
    if data_axes and _valid_axis_for_dim(shape, 0, data_axes):
        spec[0] = data_axes

    model_axes = _model_axis_names(_mesh, _mode)
    if model_dim is not None and model_axes:
        model_dim = model_dim if model_dim >= 0 else len(shape) + model_dim
        if model_dim != 0 and _valid_axis_for_dim(shape, model_dim, model_axes[0]):
            spec[model_dim] = model_axes[0]

    return P(*spec)


def constrain_batch(x):
    """Constrain an activation to keep only data-axis batch sharding."""
    if _mesh is None:
        return x
    return jax.lax.with_sharding_constraint(x, _activation_spec(x))


def constrain_batch_model(x, model_dim=-1):
    """Constrain an activation to shard batch on data axes and model on one dim."""
    if _mesh is None:
        return x
    return jax.lax.with_sharding_constraint(
        x,
        _activation_spec(x, model_dim=model_dim),
    )


def shard_cpu_tree_to_mesh(cpu_tree, mesh, partition_specs):
    sharding_tree = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), partition_specs)

    def to_global_array(leaf, sharding):
        if isinstance(leaf, jax.Array) and leaf.is_fully_addressable is False:
            return leaf
        return jax.make_array_from_callback(
            np.shape(leaf),
            sharding,
            lambda index: np.asarray(leaf)[index],
        )

    return jax.tree_util.tree_map(to_global_array, cpu_tree, sharding_tree)
