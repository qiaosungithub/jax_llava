import jax
import jax.experimental.multihost_utils as mu
from flax.training import checkpoints
from utils import g3_env
from utils.logging_util import log_for_0, print0, Emoji
import os
import re
import contextlib
import json
import orbax.checkpoint as ocp


def _cns_cell(path):
    """'/cns/yucmhcg-d/home/x' -> 'yucmhcg-d'. None for a non-CNS path."""
    parts = str(path).split('/')
    return parts[2] if len(parts) > 3 and parts[1] == 'cns' else None


class _GfileFS:
    """The three `gcsfs.GCSFileSystem` methods this module uses, over gfile.

    `gcsfs` does not exist in google3, and importing it at module scope killed
    the binary before `main()` ever ran. `pyglib.gfile` covers the same
    operations and additionally reaches `/cns/`, which gcsfs never could.

    The recursive copy deliberately prefers `Snapshot` over
    `RecursivelyCopyDir`. Both were measured: within one CNS cell -- which is
    always the case here, since `checkpoints/` and `pretrained-ckpts/` are both
    under $CHECKPOINT_BUCKET -- `Snapshot` is a metadata operation that moved a
    real 1.45 GiB orbax checkpoint in 0.16 s, while `RecursivelyCopyDir`
    streams every byte through the calling machine (its own docstring says so),
    which from a Borg task is minutes of wasted bandwidth per checkpoint.

    Both also default to `overwrite=False` and raise ALREADY_EXISTS, where
    gcsfs.copy() silently overwrote. A previous aborted copy would then make
    every retry fail, so the destination is cleared first.
    """

    @staticmethod
    def _gfile():
        from google3.pyglib import gfile
        return gfile

    def exists(self, path):
        return bool(self._gfile().Exists(path))

    def listdir(self, path):
        """Entry basenames under `path`; () when it does not exist or cannot be read.

        Never raises. An auto-resume probe runs before any training has
        happened, so a failure here must degrade to "cold start", never to a
        dead job -- see engineering.md, "do not let a diagnostic kill the thing
        it watches".
        """
        try:
            gfile = self._gfile()
            if not gfile.Exists(path):
                return ()
            return tuple(gfile.ListDir(path))
        except Exception:  # noqa: BLE001 - a probe must never block startup
            return ()

    def copy(self, src, dst, recursive=False):
        gfile = self._gfile()
        src, dst = str(src).rstrip('/'), str(dst).rstrip('/')
        parent = os.path.dirname(dst)
        if parent and not gfile.Exists(parent):
            gfile.MakeDirs(parent)
        if not recursive or not gfile.IsDirectory(src):
            gfile.Copy(src, dst, overwrite=True)
            return
        if _cns_cell(src) is not None and _cns_cell(src) == _cns_cell(dst):
            if gfile.Exists(dst):
                gfile.DeleteRecursively(dst)
            gfile.Snapshot(src, dst)
        else:
            gfile.RecursivelyCopyDir(src, dst, overwrite=True)


def _open_text(path):
    """Read-mode file object for a small text file, on CNS or on a local disk.

    `open()` cannot see `/cns/`, and in google3 `gfile.Open` reaches both, so
    this is only a two-way switch rather than a scheme table.
    """
    if g3_env.in_google3():
        from google3.pyglib import gfile
        return gfile.Open(path, 'r')
    if str(path).startswith('gs://'):
        return FS.open(path, 'r')
    return open(path, 'r')


def _listdir(path):
    """Entry basenames under `path`; () if absent or unreadable. Never raises.

    `_GfileFS` answers for google3; gcsfs spells the same thing `ls()` and
    returns full paths, so the basenames are taken here. An auto-resume probe
    runs before any training has happened, so a failure must degrade to "cold
    start", never to a dead job.
    """
    try:
        if hasattr(FS, 'listdir'):
            return FS.listdir(path)
        return tuple(str(entry).rstrip('/').rsplit('/', 1)[-1]
                     for entry in FS.ls(path))
    except Exception:  # noqa: BLE001 - a probe must never block startup
        return ()


def _make_fs():
    if g3_env.in_google3():
        return _GfileFS()
    import gcsfs
    return gcsfs.GCSFileSystem()


FS = _make_fs()
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
    # On Borg the workdir is a task-local scratch path with no zone in its
    # name; the truth is the cell the scheduler placed us in. Ask the
    # environment first, and only fall back to parsing the workdir (which is
    # how the GCP TPU-VM path has always worked, where the launcher encodes
    # the zone in the path).
    env_zone = g3_env.infer_zone_from_environment()
    if env_zone:
        return env_zone
    cell = g3_env.borg_cell()
    if cell:
        # On Borg with a cell we do not recognise. Falling through to parse the
        # workdir would be worse than useless: a Borg workdir is task-local
        # scratch with no zone in its name, so the parse fails with a message
        # about workdirs that says nothing about the real problem.
        raise ValueError(
            f'Running in Borg cell {cell!r}, which utils/g3_env.py does not map '
            'to a region. Add it to _CELL_TO_METRO (and give the metro a data '
            'root) or set JAX_LLAVA_ZONE explicitly. Refusing to guess: the '
            'zone decides which storage this job reads and writes.'
        )
    matched_zones = [z for z in ['us-central1', 'us-east1', 'us-east5', 'us-central2', 'asia-northeast1-b', 'europe-west4'] if z in workdir]
    if not matched_zones:
        if not config.local_debug:
            raise ValueError(f'Cannot infer zone from workdir {workdir}. Make sure your workdir contains zone info')
        else:
            return None
    assert len(matched_zones) == 1, f'Multiple matched zones {matched_zones} from workdir {workdir}, this is unexpected'
    zone = matched_zones[0]
    return zone

# Under Borg the durable root is handed to the job as $CHECKPOINT_BUCKET (a
# /cns/ prefix derived from the XManager experiment id, so it is stable across
# restarts -- which is what makes in-process auto-resume well defined). The
# workdir itself is task-local and is wiped by the very restart a checkpoint
# exists to survive, so it is never a checkpoint location.
CNS_CKPT_SUBDIR = 'checkpoints'
CNS_PRETRAINED_SUBDIR = 'pretrained-ckpts'


def checkpoint_bucket():
    """The durable /cns/ root for this run, or '' if none was provided."""
    return (os.environ.get('CHECKPOINT_BUCKET', '') or '').strip().rstrip('/')


def _convert_to_cns(path: str):
    """Map any checkpoint-ish path to this run's durable CNS prefix."""
    if path.startswith('/cns/'):
        return path.rstrip('/')
    bucket = checkpoint_bucket()
    if not bucket:
        raise ValueError(
            f'Cannot resolve checkpoint path {path!r} in google3: no '
            '$CHECKPOINT_BUCKET in the environment. The launcher sets it; for '
            'a local run export it to a /cns/ prefix you can write.'
        )
    if not bucket.startswith('/cns/'):
        raise ValueError(
            f'$CHECKPOINT_BUCKET={bucket!r} is not a /cns/ path. A Borg task '
            'runs as <user>@prod.google.com and cannot write our gs:// '
            'buckets; CNS is the one filesystem its identity owns.'
        )
    return f'{bucket}/{CNS_CKPT_SUBDIR}'


def convert_to_gs(path: str, zone=None):
    # In google3 there is no GCS bucket to convert to; everything durable is on
    # CNS. Do this before the gs:// branch so a stale gs:// path in a config
    # cannot smuggle a cross-identity write past us.
    if g3_env.in_google3():
        return _convert_to_cns(path)
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
    # `os.path.exists` on a /cns/ path silently answers FALSE -- the stdlib has
    # no idea the filesystem exists, and it does not raise, so a checkpoint
    # that is right there reports as missing. Measured: for a file with
    # gfile.Exists() True, os.path.exists() is False.
    #
    # Prefix-matching the scheme is not enough either: it is one more list to
    # keep in sync, and getting it wrong fails the same silent way. gfile
    # handles /cns/, /bigstore/, /placer/ AND ordinary POSIX paths, so in
    # google3 it can answer every case.
    if g3_env.in_google3():
        return FS.exists(path)
    if path.startswith('gs://'):
        return FS.exists(path)
    return os.path.exists(path)

def convert_to_pretrained_gs(path: str, zone=None):
    """Maps a normal regional checkpoint path to the same bucket's durable prefix."""
    gs_path = convert_to_gs(path, zone).rstrip('/')
    if g3_env.in_google3():
        if f'/{CNS_PRETRAINED_SUBDIR}' in gs_path:
            return gs_path
        assert gs_path.endswith(f'/{CNS_CKPT_SUBDIR}') or f'/{CNS_CKPT_SUBDIR}/' in gs_path, (
            f'cannot convert {gs_path} to a pretrained path: it is not under '
            f'/{CNS_CKPT_SUBDIR}'
        )
        return gs_path.replace(f'/{CNS_CKPT_SUBDIR}', f'/{CNS_PRETRAINED_SUBDIR}', 1)
    if _PRETRAINED_CKPT_PREFIX in gs_path:
        return gs_path
    assert _NORMAL_CKPT_PREFIX in gs_path, (
        f'cannot convert checkpoint path to pretrained-ckpts path: {gs_path}'
    )
    return gs_path.replace(_NORMAL_CKPT_PREFIX, _PRETRAINED_CKPT_PREFIX, 1)

# ---------------------------------------------------------------------------
# Auto-resume: rediscovering this run's own progress at startup.
# ---------------------------------------------------------------------------
# A Borg task restart replays the SAME argv and environment on a fresh machine.
# There is no process state and `workdir` (task-local /tmp) starts empty, so
# anything the previous attempt learned survives only if it was persisted --
# and only if this process goes looking for it. That lookup is what
# `latest_complete_checkpoint()` does.
#
# $CHECKPOINT_BUCKET is derived from the XManager experiment id, so every
# restart of a given XID -- and every work unit appended by `--resume_xid` --
# resolves to the same prefix. That stability is precisely what makes
# in-process rediscovery well defined.
#
# Enumerating the prefix beats parsing logs: a rotated or lost log would
# otherwise silently restart from step 0 and throw away real progress.

# The file orbax writes LAST. `finalize()` updates _CHECKPOINT_METADATA with
# `commit_timestamp_nsecs` and only THEN renames the tmp directory into place
# (third_party/py/orbax/checkpoint/_src/path/atomicity.py::finalize, and the
# CNS2 variant in google/path/cns2_atomicity.py -- both docstrings read
# "Updates checkpoint metadata with commit_timestamp_nsecs"). A step directory
# whose metadata lacks that key was still being written when the task died.
#
# This is jax_llava's equivalent of EqR-jax's `extra.json` rule. It is NOT the
# same filename, because the two projects use different checkpoint writers;
# copying the name across would have rejected every checkpoint instead.
_CHECKPOINT_COMMIT_KEY = 'commit_timestamp_nsecs'
_CHECKPOINT_METADATA_FILE = '_CHECKPOINT_METADATA'


def _checkpoint_is_complete(step_dir):
    """(True, '') when orbax finished writing `step_dir`, else (False, reason).

    Two witnesses, both required:

    1. `_CHECKPOINT_METADATA` parses as JSON and carries a non-null
       `commit_timestamp_nsecs` -- orbax's own atomic-commit marker.
    2. If this checkpoint has a `dataloader_state/` directory at all, it is
       non-empty. `train.py::_save_training_checkpoint` publishes that
       directory by rename AFTER orbax returns, so witness (1) alone is
       satisfied during the window in between -- and a resume there would then
       hard-fail inside `restore_dataloader_state` under
       `stateful_dataloader_strict`. Absent entirely is fine: a checkpoint from
       a non-stateful run is still resumable.

    Never raises: an unreadable candidate counts as incomplete.
    """
    step_dir = str(step_dir).rstrip('/')
    meta_path = f'{step_dir}/{_CHECKPOINT_METADATA_FILE}'
    try:
        if not FS.exists(meta_path):
            return False, f'no {_CHECKPOINT_METADATA_FILE}'
        with _open_text(meta_path) as handle:
            meta = json.loads(handle.read())
    except Exception as exc:  # noqa: BLE001 - unreadable == incomplete
        return False, f'{_CHECKPOINT_METADATA_FILE} unreadable ({exc!r})'
    if not isinstance(meta, dict) or meta.get(_CHECKPOINT_COMMIT_KEY) is None:
        return False, f'{_CHECKPOINT_METADATA_FILE} has no {_CHECKPOINT_COMMIT_KEY}'

    state_dir = f'{step_dir}/dataloader_state'
    if FS.exists(state_dir) and not _listdir(state_dir):
        return False, 'dataloader_state/ present but empty'
    return True, ''


def latest_complete_checkpoint(checkpoint_root):
    """Highest-step COMPLETE `checkpoint_N` under `checkpoint_root`, or None.

    Torn writes are skipped, loudly. Returns None -- never raises -- when the
    prefix is absent or holds nothing complete: that is a cold start, a
    legitimate outcome that must not kill the job.
    """
    if not checkpoint_root:
        return None
    checkpoint_root = str(checkpoint_root).rstrip('/')
    best_step, best_dir = -1, None
    for name in _listdir(checkpoint_root):
        match = _CHECKPOINT_RE.match(str(name))
        if not match:
            continue
        step = int(match.group(1))
        if step <= best_step:
            continue
        candidate = f'{checkpoint_root}/{name}'
        complete, why = _checkpoint_is_complete(candidate)
        if not complete:
            log_for_0('Auto-resume: ignoring incomplete checkpoint %s (%s)',
                      candidate, why)
            continue
        best_step, best_dir = step, candidate
    return best_dir


def resolve_borg_autoresume(config):
    """The checkpoint this attempt should continue from, or None to start cold.

    Pure: reads the environment and the filesystem, mutates nothing. The caller
    (main.py) applies the answer to the config, so the decision and its effect
    are separately testable.

    An explicit `load_from` ALWAYS wins and disables the probe. That is the
    user asking for a specific checkpoint -- an eval target, or a warm start
    from someone else's run -- and a restart must not silently redirect it.
    The launcher relies on this: `~/work/tpu_cmd/xm_launcher.py` deliberately
    does NOT set $LOAD_FROM for `--resume_xid`, because doing so would both
    supply a path it cannot know and disable the mechanism below.
    """
    bucket = checkpoint_bucket()
    if not bucket:
        return None, 'no $CHECKPOINT_BUCKET'

    existing = str(config.get('load_from', '') or '').strip()
    if existing:
        return None, f'load_from already set to {existing!r} (explicit request wins)'
    if config.get('eval_only', False):
        return None, 'eval_only run has no progress of its own to resume'

    checkpoint_root = f'{bucket}/{CNS_CKPT_SUBDIR}'
    try:
        resume_from = latest_complete_checkpoint(checkpoint_root)
    except Exception as exc:  # noqa: BLE001 - never block a cold start
        return None, f'probe of {checkpoint_root} failed ({exc!r})'
    if not resume_from:
        return None, f'no complete checkpoint under {checkpoint_root}'
    return resume_from, ''


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
