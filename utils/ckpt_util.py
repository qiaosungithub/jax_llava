import jax
from flax.training import checkpoints
from utils.logging_util import log_for_0, print0, Emoji
import os
import re
import gcsfs
import contextlib
import orbax.checkpoint as ocp

FS = gcsfs.GCSFileSystem()
_CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)$")

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

def is_checkpoint(path):
    if not exist_general(path):
        return False
    if not os.path.basename(path).startswith('checkpoint_'):
        path = checkpoints.latest_checkpoint(path)
        return path is not None and is_checkpoint(path)
    return True

def checkpoint_step(load_from, zone):
    """
    Returns the step encoded in a checkpoint path without restoring checkpoint arrays.
    """
    gs_path = _resolve_checkpoint_path(load_from, zone)
    match = _CHECKPOINT_RE.match(os.path.basename(gs_path))
    assert match is not None, f'cannot infer checkpoint step from {gs_path}'
    return int(match.group(1))

def _resolve_checkpoint_path(load_from, zone):
    """Returns the concrete checkpoint_N path for either a ckpt or workdir path."""
    gs_path = convert_to_gs(load_from, zone).rstrip('/')
    assert exist_general(gs_path), f'checkpoint {gs_path} does not exist'
    if not os.path.basename(gs_path).startswith('checkpoint_'):
        latest = checkpoints.latest_checkpoint(gs_path)
        assert latest is not None, f'no checkpoint found under {gs_path}'
        gs_path = latest.rstrip('/')
    return gs_path

def restore_checkpoint(state, load_from, zone):
    """
    Restores the model state from a checkpoint located in the specified working directory.
    """
    gs_path = convert_to_gs(load_from, zone)
    # assert gs path exists
    assert exist_general(gs_path), f'checkpoint {gs_path} does not exist'
    state = checkpoints.restore_checkpoint(gs_path, state)
    log_for_0("Restored from checkpoint at {}".format(load_from))
    return state

def restore_checkpoint_params(params_target, load_from, zone):
    """
    Restores only the params subtree using the caller's current sharding target.

    Passing a concrete target matters for jit/HSDP checkpoints: restoring with
    target=None asks Orbax to reuse checkpoint-saved device sharding, which can
    fail when a v6e checkpoint is resumed on v5p or any different topology.
    """
    gs_path = _resolve_checkpoint_path(load_from, zone)
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

def save_checkpoint(state, workdir):
    """
    Saves the model state to a checkpoint in the specified working directory.
    """
    assert not workdir.startswith('gs://'), f'workdir {workdir} must not start with gs://'
    # Save a host tree, matching the text-jit HSDP path. This avoids baking a
    # TPU topology-specific sharding into checkpoints that may later be resumed
    # on a different v5p/v6e layout.
    state = jax.tree.map(lambda x: jax.device_get(x), state)
    step = int(state.step)
    print0(f'{Emoji.ROCKET} Saving checkpoint at step {step} ...')
    gs_path = convert_to_gs(workdir)
    with _orbax_set_mesh_context_compat():
        checkpoints.save_checkpoint_multiprocess(gs_path, state, step, keep=3)
    print0(f'{Emoji.GOOD} Checkpoint at step {step} saved to {gs_path}.')


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
