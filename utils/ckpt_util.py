import jax
import jax.experimental.multihost_utils as mu
from flax.training import checkpoints
from utils.logging_util import log_for_0, print0, Emoji
import os
import re
import gcsfs
import contextlib
import orbax.checkpoint as ocp

FS = gcsfs.GCSFileSystem()
_CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)$")
_NORMAL_CKPT_PREFIX = "/qiao_zhicheng_hanhong_files/"
_PRETRAINED_CKPT_PREFIX = "/pretrained-ckpts/qiao_zhicheng_hanhong_files/"

def _bucket_for_zone(zone: str):
    if zone.startswith('us-central1'):
        return 'kmh-gcp-us-central1'
    if zone.startswith('us-east1'):
        return 'kmh-gcp-us-east1'
    if zone.startswith('us-east5'):
        return 'kmh-gcp-us-east5'
    if zone.startswith('us-central2'):
        return 'kmh-gcp-us-central2'
    if zone.startswith('asia-northeast1-b'):
        return 'kmh-gcp-asia-northeast1-b'
    if zone.startswith('europe-west4'):
        return 'kmh-gcp'
    return None

def _convert_known_gs_to_zone(path: str, zone: str):
    bucket = _bucket_for_zone(zone)
    if bucket is None:
        return path
    for prefix in (_PRETRAINED_CKPT_PREFIX, _NORMAL_CKPT_PREFIX):
        idx = path.find(prefix)
        if idx >= 0 and path.startswith('gs://kmh-gcp'):
            return f"gs://{bucket}{path[idx:]}"
    return path

def infer_zone_card(config, workdir):
    matched_zones = [z for z in ['us-central1', 'us-east1', 'us-east5', 'us-central2', 'asia-northeast1-b', 'europe-west4'] if z in workdir]
    if not matched_zones:
        if not config.local_debug:
            raise ValueError(f'Cannot infer zone from workdir {workdir}. Make sure your workdir contains zone info')
        else:
            return None
    assert len(matched_zones) == 1, f'Multiple matched zones {matched_zones} from workdir {workdir}, this is unexpected'
    zone = matched_zones[0]
    return zone

def convert_to_gs(path: str, zone=None):
    if path.startswith('gs://'):
        if zone is not None:
            return _convert_known_gs_to_zone(path, zone)
        return path
    assert os.path.isabs(path), f'ckpt path {path} is not absolute.'
    # assert path.startswith('/')
    
    if zone is not None: # only for restoring ckpt
        return convert_to_gs_by_zone(path, zone)

    subpaths = path.strip('/').split('/')
    assert subpaths[0] in ['kmh-nfs-ssd-us-mount', 'kmh-nfs-us-mount'], f'cannot handle checkpoint path {path}'

    matched_zones = [z for z in ['us-central1', 'us-east1', 'us-east5', 'us-central2', 'asia-northeast1-b', 'europe-west4'] if z in path]
    if not matched_zones:
        log_for_0(f'[WARNING] cannot infer GCS path from {path}, no known zone found. Using default us-central2.')
        pref = 'kmh-gcp-us-central2'
    else:
        assert len(matched_zones) == 1, f'cannot handle checkpoint path {path}, multiple zones found: {matched_zones}'
        zone = matched_zones[0]
        if zone == 'europe-west4': pref = 'kmh-gcp'
        else: pref = f'kmh-gcp-{zone}'
    out = '/' + '/'.join(subpaths[3:]) # unknown/launch*
    out = f'gs://{pref}/qiao_zhicheng_hanhong_files' + out
    return out

def exist_general(path):
    if path.startswith('gs://'):
        return FS.exists(path)
    return os.path.exists(path)

def convert_to_pretrained_gs(path: str, zone=None):
    """Maps a normal regional checkpoint path to the same bucket's durable prefix."""
    gs_path = convert_to_gs(path, zone).rstrip('/')
    if _PRETRAINED_CKPT_PREFIX in gs_path:
        return gs_path
    assert _NORMAL_CKPT_PREFIX in gs_path, (
        f'cannot convert checkpoint path to pretrained-ckpts path: {gs_path}'
    )
    return gs_path.replace(_NORMAL_CKPT_PREFIX, _PRETRAINED_CKPT_PREFIX, 1)

def _latest_checkpoint_or_none(path):
    path = path.rstrip('/')
    if os.path.basename(path).startswith('checkpoint_'):
        return path if exist_general(path) else None
    try:
        latest = checkpoints.latest_checkpoint(path)
    except Exception:
        latest = None
    if latest is not None:
        return latest.rstrip('/')
    return None

def is_checkpoint(path):
    return _latest_checkpoint_or_none(path) is not None

def checkpoint_step(load_from, zone):
    """
    Returns the step encoded in a checkpoint path without restoring checkpoint arrays.
    """
    gs_path = _resolve_checkpoint_path(load_from, zone)
    match = _CHECKPOINT_RE.match(os.path.basename(gs_path))
    assert match is not None, f'cannot infer checkpoint step from {gs_path}'
    return int(match.group(1))

def _resolve_checkpoint_path(load_from, zone, allow_pretrained_fallback=False):
    """Returns the concrete checkpoint_N path for either a ckpt or workdir path."""
    gs_path = convert_to_gs(load_from, zone).rstrip('/')
    resolved = _latest_checkpoint_or_none(gs_path)
    if resolved is None and allow_pretrained_fallback:
        pretrained_path = convert_to_pretrained_gs(load_from, zone)
        resolved = _latest_checkpoint_or_none(pretrained_path)
        if resolved is not None:
            log_for_0(
                'Checkpoint %s not found; using durable pretrained checkpoint %s.',
                gs_path,
                resolved,
            )
    assert resolved is not None, f'checkpoint {gs_path} does not exist'
    return resolved

def restore_checkpoint(state, load_from, zone, allow_pretrained_fallback=False):
    """
    Restores the model state from a checkpoint located in the specified working directory.
    """
    gs_path = _resolve_checkpoint_path(
        load_from,
        zone,
        allow_pretrained_fallback=allow_pretrained_fallback,
    )
    state = checkpoints.restore_checkpoint(gs_path, state)
    log_for_0("Restored from checkpoint at {}".format(gs_path))
    return state

def restore_checkpoint_params(params_target, load_from, zone):
    """
    Restores only the params subtree using the caller's current sharding target.

    Passing a concrete target matters for jit/HSDP checkpoints: restoring with
    target=None asks Orbax to reuse checkpoint-saved device sharding, which can
    fail when a v6e checkpoint is resumed on v5p or any different topology.
    """
    gs_path = _resolve_checkpoint_path(
        load_from,
        zone,
        allow_pretrained_fallback=True,
    )
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    restore_target = {'params': params_target}
    restore_args = ocp.checkpoint_utils.construct_restore_args(restore_target)
    restored = checkpointer.restore(
        gs_path,
        args=ocp.args.PyTreeRestore(
            item=restore_target,
            restore_args=restore_args,
            partial_restore=True,
        ),
    )
    params = restored['params'] if isinstance(restored, dict) else restored.params
    log_for_0("Restored params from checkpoint at {}".format(load_from))
    return params

def copy_latest_checkpoint_to_pretrained(checkpoint_or_workdir, zone=None):
    """Copies the latest normal checkpoint_N to the same bucket's pretrained prefix."""
    if jax.process_index() != 0:
        return False
    src = _resolve_checkpoint_path(checkpoint_or_workdir, zone, allow_pretrained_fallback=False)
    dst_root = convert_to_pretrained_gs(os.path.dirname(src.rstrip('/')), zone)
    dst = f"{dst_root.rstrip('/')}/{os.path.basename(src)}"
    if exist_general(dst):
        log_for_0("Pretrained checkpoint already exists at %s; skipping copy.", dst)
        return True
    log_for_0("Copying final checkpoint to durable pretrained path: %s -> %s", src, dst)
    FS.copy(src, dst, recursive=True)
    log_for_0("Durable pretrained checkpoint saved to %s.", dst)
    return True

def save_checkpoint(state, workdir, *, log_completion=True):
    """
    Saves the model state to a checkpoint in the specified working directory.
    """
    assert not workdir.startswith('gs://'), f'workdir {workdir} must not start with gs://'
    step = int(jax.device_get(state.step))
    gs_path = convert_to_gs(workdir)
    print0(f'{Emoji.ROCKET} Saving checkpoint at step {step} ...')
    with _orbax_set_mesh_context_compat():
        _save_sharded_checkpoint_all_processes(gs_path, state, step, keep=3)
    if log_completion:
        print0(f'{Emoji.GOOD} Checkpoint at step {step} saved to {gs_path}.')
    return step, gs_path


def _save_sharded_checkpoint_all_processes(gs_path, state, step, keep):
    """Writes a sharded Orbax checkpoint without all-gathering the TrainState."""
    gs_path = gs_path.rstrip('/')
    ckpt_path = f'{gs_path}/checkpoint_{step}'
    if jax.process_index() == 0:
        checkpoints._remove_invalid_ckpts(  # pylint: disable=protected-access
            ckpt_path,
            f'{gs_path}/checkpoint_',
            keep,
            False,
            None,
            True,
        )
    mu.sync_global_devices(f'checkpoint_prune_{step}')
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    checkpointer.save(ckpt_path, state)


@contextlib.contextmanager
def _orbax_set_mesh_context_compat():
    """
    Orbax versions used by flax checkpointing may expect
    ``jax.sharding.set_mesh`` to be a context manager. JAX 0.6.x returns the
    previous mesh instead, so wrap it during checkpoint serialization only.
    """
    original_set_mesh = jax.sharding.set_mesh

    @contextlib.contextmanager
    def set_mesh_context(mesh):
        previous_or_context = original_set_mesh(mesh)
        if hasattr(previous_or_context, "__enter__"):
            with previous_or_context as value:
                yield value
            return
        try:
            yield previous_or_context
        finally:
            original_set_mesh(previous_or_context)

    jax.sharding.set_mesh = set_mesh_context
    try:
        yield
    finally:
        jax.sharding.set_mesh = original_set_mesh

def convert_to_gs_by_zone(path: str, zone: str):
    if zone == 'us-central1':
        return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-us-central1/qiao_zhicheng_hanhong_files')
    if zone == 'us-east1':
        return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-us-east1/qiao_zhicheng_hanhong_files')
    if zone == 'us-east5':
        return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-us-east5/qiao_zhicheng_hanhong_files')
    if zone == 'us-central2':
        return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-us-central2/qiao_zhicheng_hanhong_files')
    if zone == 'asia-northeast1-b':
        return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-asia-northeast1-b/qiao_zhicheng_hanhong_files')
    if zone == 'europe-west4':
        return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp/qiao_zhicheng_hanhong_files')
    return None
