import logging as _logging
import re
import io
import os
import sys
import threading
import time
import shutil
from typing import Any, List, Optional

import jax
from absl import logging

import numpy as np
from PIL import Image
from jax.experimental import multihost_utils
from functools import partial

# def print0(*args, **kwargs):
#     if jax.process_index() == 0:
#         print(*args, **kwargs)

def process_index_or_zero():
    """`jax.process_index()`, or 0 when JAX is not initialised yet.

    Under google3 Bazel, JAX refuses to answer before `absl.app.run()` has
    run (`jax_google.py::_lazy_initialization`), raising

        RuntimeError: Attempted call to JAX before absl.app.run() is called.

    That happens for real in a torch DataLoader worker: absl_spawn re-execs
    the binary, the child unpickles the dataset, which re-imports
    `input_pipeline` -> this module, all before the child's InitGoogle(). A
    logging helper that raises there kills the worker and the parent then
    waits forever for a batch that never comes.

    Logging is a diagnostic; it must never be able to kill the thing it
    watches. Before initialisation every process is indistinguishable from
    process 0, which is also the answer that makes `log_for_0` print.
    """
    try:
        return jax.process_index()
    except RuntimeError:
        return 0


def log_for_0(*args, stacklevel=2):
    if process_index_or_zero() == 0:
        logging.info(*args, stacklevel=stacklevel)

print0 = lambda *args, **kwargs: log_for_0(*args, stacklevel=3)

def log_for_all(msg):
    logging.info(f"[Rank {process_index_or_zero()}] {msg}")

class ExcludeInfo(_logging.Filter):
    def __init__(self, exclude_files):
        super().__init__()
        self.exclude_files = exclude_files

    def filter(self, record):
        if any(file_name in record.pathname for file_name in self.exclude_files):
            return record.levelno > _logging.INFO
        return True


# Suppress orbax/flax checkpoint INFO logs: CommitFuture blocking, "No metadata found", etc.
_EXCLUDE_FILES_BASE = [
    'orbax/checkpoint/async_checkpointer.py',
    'orbax/checkpoint/abstract_checkpointer.py',
    'orbax/checkpoint/multihost/utils.py',
    'orbax/checkpoint/future.py',
    'orbax/checkpoint/_src/handlers/base_pytree_checkpoint_handler.py',
    'orbax/checkpoint/type_handlers.py',
    'orbax/checkpoint/metadata/checkpoint.py',
    'orbax/checkpoint/metadata/sharding.py',
    'orbax/checkpoint/metadata/array_metadata_store.py',
    'array_metadata_store.py',
    'orbax/checkpoint/',  # catch any other checkpoint INFO under orbax (e.g. future.py path variants)
]
# Non-zero processes additionally silence the two files process 0 keeps, so the
# checkpoint story is told once rather than once per host.
_EXCLUDE_FILES_NONZERO = [
    'orbax/checkpoint/checkpointer.py',
    'flax/training/checkpoints.py',
]


def _exclude_files():
    # Resolved lazily: asking JAX for the process index at MODULE scope raises
    # in any process that has not run InitGoogle() yet -- see
    # process_index_or_zero() above. supress_checkpt_info() is only ever
    # called from main(), where the answer is real.
    return _EXCLUDE_FILES_BASE + _EXCLUDE_FILES_NONZERO * process_index_or_zero()


def supress_checkpt_info():
    logging.get_absl_handler().addFilter(ExcludeInfo(_exclude_files()))


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.mode = 'normal'

    def elapse_without_reset(self):
        return time.time() - self.start_time

    def elapse_with_reset(self):
        """This do both elaspse and reset"""
        a = time.time() - self.start_time
        self.reset()
        return a

    def reset(self):
        self.start_time = time.time()

    def __str__(self):
        return f'{self.elapse_with_reset():.2f} s'
    
    def skip(self):
        self.mode = 'skip'
        return self
    
    def __enter__(self):
        assert self.mode == 'skip', "Please call skip() before using 'with' statement"
        self._elapsed = self.elapse_with_reset()

    def __exit__(self, exc_type, exc_value, traceback):
        self.mode = 'normal'
        self.reset()
        self.start_time -= self._elapsed  # adjust start_time to skip the elapsed time

class MetricsTracker:
    """Accumulate per-step metrics ON DEVICE; transfer only when logging.

    THE TRANSFER IS THE COST, NOT THE ARITHMETIC. `update()` runs every step,
    and the previous implementation called `jax.device_get` on every leaf --
    which, as its own comment said, "blocks on the computation that produced
    x". JAX dispatch is asynchronous precisely so the host can queue step N+1
    while the device runs step N; a blocking read each step destroys that and
    serialises the whole loop into dispatch -> wait -> dispatch.

    Measured on a v7-32 stage-2 run at bs256: 0.312 steps/s, with the step
    itself taking 0.176 s and the input pipeline 0.041 s -- i.e. 93% of a
    3.2 s iteration was the host waiting. The reference run of the same recipe
    did 1.871 steps/s. 33 metric leaves were being fetched per step.

    Accumulating on device keeps the pipeline full: the sum is a tiny
    device-side add, and the single transfer happens in `finalize()`, once per
    `log_per_step`.
    """

    def __init__(self):
        self._sum = None   # tree of jax scalars, kept ON DEVICE
        self._n = 0        # number of steps accumulated on *this host*

    @staticmethod
    def _mean_over_local_devices(x):
        """Average a leaf over its local-device axis, without leaving the device.

        Metrics may arrive per-replica (a leading device axis) or already
        reduced (0-D). Both are handled here so `update` stays a pure
        device-side operation.
        """
        # `jax.numpy`, not a new module-level `import jax.numpy as jnp`: this
        # module is deliberately lazy about heavy imports (see the header) and
        # `jax` is already bound here.
        a = jax.numpy.asarray(x)
        if a.ndim >= 1:  # treat leading axis as local device axis
            a = a.mean(axis=0)
        return a

    def update(self, metrics_step_tree):
        """
        Incorporate one step's metrics (per-replica JAX arrays) into the running sum.
        Call this once per training step. Does NOT block: no host transfer here.
        """
        local_mean = jax.tree.map(self._mean_over_local_devices, metrics_step_tree)
        if self._sum is None:
            self._sum = local_mean
        else:
            self._sum = jax.tree.map(lambda s, x: s + x, self._sum, local_mean)
        self._n += 1

    def finalize(self):
        """
        Return global mean over steps, devices, and hosts as a tree of Python floats.
        Resets internal state. Safe to call at any logging boundary.

        This is where the single host transfer happens -- one blocking read per
        logging interval instead of one per step.
        """
        if self._n == 0:
            return {}

        # One device_get for the whole tree: fewer, larger transfers beat many
        # small ones, and it blocks exactly once.
        summed = jax.device_get(self._sum)
        out = jax.tree.map(
            lambda s: float(np.asarray(s / self._n, dtype=np.float64).mean()),
            summed,
        )

        self._sum, self._n = None, 0
        return out

def _viz_output_root(workdir):
    """Where PNGs go: `$CHECKPOINT_BUCKET/viz` when Borg gives us one.

    `workdir` on a Borg task is `/tmp/eqr_log/<name>` -- the worker's own tmpfs,
    which dies with the task. Every image this Writer produced was therefore
    discarded: wandb resolves to a mock in google3, tensorboard is refused in
    __init__, and this fallback wrote to a directory nobody could read
    afterwards. The checkpoint bucket is the one location that outlives the job.
    """
    bucket = os.environ.get("CHECKPOINT_BUCKET", "").strip()
    if bucket:
        return f"{bucket.rstrip('/')}/viz"
    return os.path.join(workdir, "writed_images")


def _save_image(img, step, key, workdir):
    """Write one PIL image under the viz root, on CNS or locally."""
    root = _viz_output_root(workdir)
    name = f"step{step:07d}_{key}.png"
    if root.startswith("/cns/") or root.startswith("/bigstore/"):
        from google3.pyglib import gfile
        # CNS is not an object store: writing into a directory that does not
        # exist fails outright, so create it first.
        if not gfile.Exists(root):
            gfile.MakeDirs(root)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        with gfile.Open(f"{root}/{name}", "wb") as handle:
            handle.write(buf.getvalue())
        return
    os.makedirs(root, exist_ok=True)
    img.save(os.path.join(root, name))


class Writer:
    def __init__(self, config, workdir, use_wandb=False, use_tb=False):
        if jax.process_index() != 0:
            return
        self.use_wandb = use_wandb
        self.use_tb = use_tb
        self.workdir = workdir
        self.wandb = None
        self._wandb_error_count = 0
        if use_wandb:
            import wandb
            self.wandb = wandb
            kwargs = {}
            wandb_resume_id = getattr(config, 'wandb_resume_id', '')
            if wandb_resume_id:
                kwargs['id'] = wandb_resume_id
                kwargs['resume'] = 'must'
            try:
                wandb.init(
                    project=config.logging.wandb_project + '_eval' * config.eval_only,
                    entity=config.logging.wandb_entity if config.logging.wandb_entity else None,
                    notes=config.logging.wandb_notes if config.logging.wandb_notes else None,
                    tags=config.logging.wandb_tags if config.logging.wandb_tags else None,
                    dir='/tmp', # avoid writing to workdir
                    settings=wandb.Settings(_service_wait=60),
                    **kwargs
                )
            except Exception as e:
                self.use_wandb = False
                self._log_wandb_failure('init', e)
            if self.use_wandb:
                self._safe_wandb_call(
                    'config.update(full_config)',
                    lambda: wandb.config.update(config.to_dict(), allow_val_change=True),
                )
                try:
                    ka = re.search(
                        r"kmh-tpuvm-v[23456e]+-(\d+)(-preemptible)?(-spot)?-.*yang-(\d+)", workdir
                    ).group()
                except AttributeError:
                    ka = ' ' * 10 + 'Failed to parse VM'
                ka = ka[10:] # remove "kmh-tpuvm-"
                self._safe_wandb_call(
                    'config.update(ka)',
                    lambda: wandb.config.update({'ka': ka}),
                )

                # Save wandb run id so resume scripts can continue the same run.
                try:
                    os.makedirs(workdir, exist_ok=True)
                    wandb_id_path = os.path.join(workdir, 'wandb_run_id.txt')
                    with open(wandb_id_path, 'w') as f:
                        f.write(self.wandb.run.id)
                    log_for_0(f'Saved wandb run id {self.wandb.run.id} to {wandb_id_path}')
                except Exception as e:
                    log_for_0(f'[WARNING] Failed to save wandb run id: {e}')

        if use_tb:
            raise ValueError("use_tb is not supported")
            from clu import metric_writers
            self.writer = metric_writers.create_default_writer(logdir=workdir, just_logging=False)

    def _log_wandb_failure(self, action, error):
        self._wandb_error_count += 1
        if self._wandb_error_count <= 20 or self._wandb_error_count % 100 == 0:
            log_for_0(
                f"[WARNING] wandb {action} failed; training will continue. "
                f"{type(error).__name__}: {error}"
            )

    def _safe_wandb_call(self, action, fn):
        try:
            fn()
            return True
        except Exception as e:
            self._log_wandb_failure(action, e)
            return False
            
    def write_scalars(self, step, scalar_dict):
        # [200] ep=0.159073, steps_per_second=6.76798, train_accuracy=0.00585938, train_loss=6.71379, train_lr=0.0127258, train_step=199
        # Fail fast on a non-finite training loss: once loss is NaN/Inf the params
        # are poisoned and every further step is wasted compute. Raise (on ALL
        # hosts, before the process-0 early return) so the job dies immediately and
        # infra marks it failed instead of burning hours logging NaN.
        _loss = scalar_dict.get("loss")
        if _loss is not None:
            try:
                _loss_f = float(np.asarray(_loss))
            except (TypeError, ValueError):
                _loss_f = None
            if _loss_f is not None and not np.isfinite(_loss_f):
                # Dump every scalar of the failing step, non-finite ones first.
                # "loss is inf" alone says nothing about which term blew up, and
                # the run is dead either way -- there is no second chance to look.
                bad, ok = [], []
                for k in sorted(scalar_dict):
                    try:
                        v = float(np.asarray(scalar_dict[k]))
                    except (TypeError, ValueError):
                        continue
                    (bad if not np.isfinite(v) else ok).append(f"{k}={v:.6g}")
                raise FloatingPointError(
                    f"[step {step}] training loss is {_loss_f} (NaN/Inf) -- aborting run. "
                    "This usually means a numerical blow-up (e.g. NaN gradient); "
                    "resume will NOT help (checkpoint params are NaN), fix the cause "
                    "and start a fresh run."
                    f"\n  HOST: process_index={jax.process_index()}"
                    f"\n  NON-FINITE: {', '.join(bad) or '(none besides loss)'}"
                    f"\n  FINITE:     {', '.join(ok)}"
                )
        # Cross-host consistency probe. Jit outputs here are global scalars and
        # must be bitwise-identical on every host, yet the PaliGemma-baseline
        # fake-replication dumps showed acc/valid_tokens/nll_min splitting into
        # two host groups (12-29% apart) -- replicated-output divergence, i.e.
        # some reduction was compiled without covering the whole mesh. Every
        # host reaches this point (the train loop syncs right after), so a tiny
        # allgather makes the divergence a first-class logged metric instead of
        # a forensic find. ~1KB per log interval.
        try:
            from jax.experimental import multihost_utils as _mu
            if jax.process_count() > 1:
                _probe_keys = [
                    k for k in ("loss", "loss_vlm", "acc", "nll_min", "valid_tokens")
                    if k in scalar_dict
                ]
                if _probe_keys:
                    _local = np.asarray(
                        [float(np.asarray(scalar_dict[k])) for k in _probe_keys],
                        dtype=np.float64,
                    )
                    _all = np.asarray(_mu.process_allgather(_local))
                    _spread = np.abs(_all - _all[0:1]).max(axis=0)
                    scalar_dict = dict(scalar_dict)
                    scalar_dict["host_metric_spread"] = float(_spread.max())
                    if _spread.max() > 0:
                        _detail = ", ".join(
                            f"{k}:{s:.3g}" for k, s in zip(_probe_keys, _spread) if s > 0
                        )
                        log_for_0(
                            "[host-spread step %s] replicated metrics differ across "
                            "hosts: %s", step, _detail,
                        )
        except Exception:  # never let the probe kill training
            logging.exception("host_metric_spread probe failed (non-fatal)")
        if jax.process_index() != 0:
            return
        log_str = f"[{step}]"
        for k, v in scalar_dict.items():
            log_str += f" {k}={v:.5g}," if isinstance(v, float) else f" {k}={v},"
        log_str = log_str.strip(",")
        logging.info(log_str)
        if self.use_wandb:
            self._safe_wandb_call(
                f'log scalars at step {step}',
                lambda: self.wandb.log(scalar_dict, step=step),
            )
        # In google3 `wandb` is a mock whose log() stores nothing, so this is
        # the only durable record of the curve. No-op outside google3, and
        # best-effort inside it (see utils/g3_metrics.py).
        try:
            from utils import g3_metrics
            g3_metrics.log_metrics(scalar_dict, step)
        except Exception:  # noqa: BLE001 - telemetry must never kill training
            logging.exception("Datatables metric write failed (non-fatal)")
        if self.use_tb:
            self.writer.write_scalars(step, scalar_dict)
            
    def write_images(self, step, image_dict):
        if jax.process_index() != 0:
            return

        def reduce_arr_func(v):
            if isinstance(v, Image.Image):
                return v
            assert isinstance(v, np.ndarray), "Invalid image type {}".format(type(v))
            assert v.dtype == np.uint8, "Invalid image dtype {}".format(v.dtype)
            assert (
                v.ndim == 3
                and 3 in [v.shape[0], v.shape[2]]
            ), "Invalid image shape {}".format(v.shape)
            if v.shape[0] == 3:
                v = v.transpose((1, 2, 0))
            return Image.fromarray(v)

        wandb_ok = False
        if self.use_wandb:
            wandb_ok = self._safe_wandb_call(
                f'log images at step {step}',
                lambda: self.wandb.log({
                    k: self.wandb.Image(reduce_arr_func(v)) for k, v in image_dict.items()
                }, step=step),
            )
        if self.use_tb:
            self.writer.write_images(step, {
                k: np.asarray(reduce_arr_func(v)) for k, v in image_dict.items()
            })
        if (not self.use_wandb or not wandb_ok) and not self.use_tb:
            log_for_0(f"[NOTE] Saving images to {_viz_output_root(self.workdir)}, "
                      f"at step {step}")
            for k, v in image_dict.items():
                try:
                    _save_image(reduce_arr_func(v), step, k, self.workdir)
                except Exception as e:  # noqa: BLE001
                    # Telemetry must never kill a run.
                    log_for_0(f"[NOTE] Could not save image {k!r}: "
                              f"{type(e).__name__}: {e}")

    def write_texts(self, step, text_key, text_list):
        if jax.process_index() != 0:
            return
        if self.use_wandb:
            def log_text_table():
                text_table = self.wandb.Table(columns=[text_key])
                for text in text_list:
                    text_table.add_data(text)
                self.wandb.log({text_key: text_table}, step=step)
            if self._safe_wandb_call(f'log text table {text_key} at step {step}', log_text_table):
                return
        log_for_0(f"[NOTE] {text_key} at step {step}:")
        for text in text_list:
            log_for_0(text)

    def flush(self):
        if jax.process_index() != 0:
            return
        if self.use_tb:
            self.writer.flush()
            
    def __del__(self):
        if jax.process_index() != 0:
            return
        if self.use_wandb:
            self._safe_wandb_call('finish', lambda: self.wandb.finish())
            shutil.rmtree('/tmp/wandb', ignore_errors=True)
        if self.use_tb:
            self.writer.flush()
            self.writer.close()
            
class Emoji:
    HAPPY = "😀"
    THUMBS = "👍"
    YEAH = "🎉"
    ROCKET = "🚀"
    SPARKLES = "✨"
    FIRE = "🔥"
    GOOD = "✅"
    WARNING = "⚠️ "
    ERROR = "❌"
    EYES = "👀"
    TRUCK = "🚛"
    ROBOT = "🤖"
    INFO = "ℹ️ "


# ===========================================================================
# GRAFTED FROM EqR-jax `utils/logging_util.py` (lines 477-1132), verbatim.
#
# WHY VERBATIM, AND WHY HERE. This project and EqR-jax launch through the same
# `~/work/tpu_cmd/xm_launcher.py` onto the same Borg cells, and on this
# workstation the CNS mirror is the ONLY readable log channel: `borg tasklog`
# SIGABRTs on PERMISSION_DENIED, `analog --remote` is refused by the RDL
# engine, Coroner's binary is not even readable, and the task is GC'd from the
# borgmaster within minutes. Debugging the two projects therefore has to work
# identically, which means the same paths (`<bucket>/logs/rank_N_attemptK.log`)
# and the same attempt semantics -- not a second implementation that is merely
# similar. Every comment below is a bug that was paid for once already; they
# are kept so the reasons cannot be lost to a re-derivation.
#
# Grafted as a hunk rather than by copying the whole file: jax_llava's own
# content above (the `ExcludeInfo` filter, `supress_checkpt_info`, `Writer`,
# `MetricsTracker`, `process_index_or_zero`) has no counterpart in EqR-jax and
# must survive. Divergence below is limited to the flusher THREAD NAME, which
# names the project it belongs to.
#
# Not grafted, because they do not exist in EqR-jax and cover a window it never
# had to care about: the pre-InitGoogle startup marker, the import-crash
# reporter and the backend marker. Those stay in `utils/g3_logmirror.py`.
# ===========================================================================

# ---------------------------------------------------------------------------
# Persistent log mirroring
#
# A Borg task's stdout/stderr only survives as long as the task does. Once the
# work unit is GC'd, `borg tasklog` has nothing, `WorkUnit.borg_job_states` goes
# empty, and XManager's own `status.message` is routinely ''. The net effect is
# that the single most useful artefact -- the traceback that killed the run --
# is exactly the thing you can no longer reach.
#
# So mirror stdout/stderr into the checkpoint bucket while the job is alive.
# The copy outlives the task, the work unit, and the experiment.
#
# Design notes:
#   * ONE writer per remote file. stdout and stderr are two streams but one
#     log, so they share a single buffer, a single lock and a single critical
#     section. The previous version gave each stream its own `_GcsTee` with its
#     own buffer pointed at the SAME file, and each flush did
#     `read_text()` + `write_text(existing + payload)`. Two independent
#     read-modify-write cycles on one file is last-writer-wins: every flush
#     silently discarded whatever the other stream had appended since its own
#     read. EqR-jax prints through `log_for_0` -> `print()` -> STDOUT while
#     absl and warnings go to STDERR, so the two halves of the log erased each
#     other continuously. A 90-minute eval (XID 275491682) survived as 496
#     bytes: one stderr RuntimeWarning, the last writer standing.
#   * APPEND, never read-modify-write. `epath.Path.open("a")` reaches
#     `pyglib.gfile.Open(path, "a")` on CNS, which is a native append (CNS
#     files are append-only by construction; `fileutil append` is the CLI form
#     of the same operation). That makes a flush O(payload) instead of
#     O(whole log), and makes a second writer on the same path -- another
#     process, or a task restart -- interleave rather than truncate. There is
#     a read-modify-write fallback for backends that reject "a", but it is a
#     degraded mode and says so.
#   * Flush on a timer, on every ERROR-ish line, and from a background thread.
#     A crash usually kills the process before a size-triggered flush fires,
#     and the lines just before the crash are the ones worth having. The
#     background flusher matters for the opposite failure: a job that HANGS
#     writes nothing more, so a purely write-triggered flush would leave the
#     last (most diagnostic) lines stuck in the buffer forever.
#   * Never let mirroring break the job: every failure path degrades to plain
#     local logging.
#   * Rank-aware filename, so multi-process runs do not clobber each other --
#     see `detect_task_rank`, which is where that actually got decided.
# ---------------------------------------------------------------------------

_MIRROR_FLUSH_SECONDS = 20.0
# Bound the in-memory buffer. A runaway logger must not turn into an OOM; past
# this many buffered characters we flush regardless of the timer.
_MIRROR_MAX_BUFFER_CHARS = 1 << 20

_URGENT_TOKENS = (
    "Traceback", "Error", "ERROR", "error:", "Exception",
    "FATAL", "Fatal", "CRITICAL",
)


def detect_task_rank(default: int = 0) -> int:
    """This process's index among the job's tasks, without touching JAX.

    Called before `jax.process_index()` is legal (see main.py: the first JAX
    call boots the backend and forecloses distributed initialisation), so every
    source here is an environment variable or an already-parsed absl flag.

    The old expression -- `int(os.environ.get("BORG_TASK_HANDLE", "0")
    .split(".")[0] or 0)` -- returned 0 for all four tasks of XID 275491682 and
    friends, so all four mirrored into `logs/rank_0.log`. Two ways it gets
    there, both real:
      * `BORG_TASK_HANDLE` unset in the container -> the "0" default;
      * a handle carrying the optional `logs.` prefix
        (`logs.<task>.<job>.<user>.<uid>`, see borg/borgletlib/go/borgletlib.go
        `taskHandleRegexp`) -> `split(".")[0]` is the literal "logs",
        `int(...)` raises, and the caller's `except` swallows it into 0.
    So it consults the unambiguous integer sources first and only then parses
    the handle, with the same regex Borg itself uses.
    """
    import re

    for key in ("BORG_TASK_INDEX", "JAX_TASK_ID", "JAX_PROCESS_ID", "TASK_ID", "RANK"):
        value = os.environ.get(key)
        if value is None or not str(value).strip():
            continue
        try:
            return int(str(value).strip())
        except ValueError:
            continue

    # XManager passes the task index as an absl flag rather than an env var
    # (learning/brain/research/jax/lib/jax_google.py). FLAGS may not be parsed
    # yet, in which case reading it raises and we move on.
    try:
        from absl import flags as _flags

        if "jax_task_id" in _flags.FLAGS:
            value = _flags.FLAGS["jax_task_id"].value
            if value is not None:
                return int(value)
    except Exception:  # noqa: BLE001 - a flag lookup must never break startup
        pass

    handle = os.environ.get("BORG_TASK_HANDLE", "")
    match = re.match(r"^(?:logs\.)?(\d+)\.", handle or "")
    if match:
        return int(match.group(1))

    return default


class _RemoteLogWriter:
    """The single owner of one remote log file.

    Every stream mirroring into a given path shares one instance, so the buffer,
    the lock and the flush are all singular and interleaved writes keep their
    relative order. Thread-safe; all state is guarded by `_lock`.
    """

    def __init__(self, remote_path: str, flush_seconds: float = _MIRROR_FLUSH_SECONDS):
        self._remote_path = remote_path
        self._buf: List[str] = []
        self._buf_chars = 0
        self._flush_seconds = flush_seconds
        self._last_flush = time.time()
        self._lock = threading.Lock()
        self._broken = False
        # None = not probed yet; True = native append; False = degraded to
        # read-modify-write because the backend rejected append mode.
        self._append_ok: Optional[bool] = None
        self._flusher: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def remote_path(self) -> str:
        return self._remote_path

    @property
    def append_mode(self) -> Optional[bool]:
        """True once a native append has succeeded, False if degraded to RMW."""
        return self._append_ok

    def write(self, text: str, *, allow_inline_flush: bool = True) -> None:
        """Buffer `text`, flushing inline if it is urgent, due, or oversized.

        `allow_inline_flush=False` buffers and returns without ever touching
        CNS. Required for any caller that can run on a thread it does not own
        -- see `_MirrorLogHandler.emit`, where an inline flush would issue a
        blocking CNS op from inside CNS's own fiber-coloured callback and
        CHECK-fail the process. The background flusher still ships the bytes.
        """
        if self._broken or not text:
            return
        with self._lock:
            self._buf.append(text)
            self._buf_chars += len(text)
            if not allow_inline_flush:
                # Bound the buffer even so: a mirror must not become an OOM.
                if self._buf_chars >= _MIRROR_MAX_BUFFER_CHARS:
                    del self._buf[: len(self._buf) // 2]
                    self._buf_chars = sum(len(chunk) for chunk in self._buf)
                return
            urgent = any(token in text for token in _URGENT_TOKENS)
            due = (time.time() - self._last_flush) >= self._flush_seconds
            if urgent or due or self._buf_chars >= _MIRROR_MAX_BUFFER_CHARS:
                self._flush_locked()

    def flush(self) -> None:
        if self._broken:
            return
        with self._lock:
            self._flush_locked()

    def start_background_flusher(self) -> None:
        """Flush on a wall clock, so a HUNG job still ships its last lines."""
        if self._flusher is not None:
            return

        def _loop():
            while not self._stop.wait(self._flush_seconds):
                if self._broken:
                    return
                try:
                    self.flush()
                except Exception:  # noqa: BLE001 - never kill the job
                    return

        self._flusher = threading.Thread(
            target=_loop, name="llava-log-mirror-flush", daemon=True
        )
        self._flusher.start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ internals

    def _flush_locked(self) -> None:
        if not self._buf:
            return
        payload = "".join(self._buf)
        try:
            from etils import epath

            path = epath.Path(self._remote_path)
            if self._append_ok is not False:
                try:
                    with path.open("a") as handle:
                        handle.write(payload)
                    self._append_ok = True
                except Exception:  # noqa: BLE001 - backend without append mode
                    self._append_ok = False
            if self._append_ok is False:
                # Degraded: last-writer-wins across processes, and O(file) per
                # flush. Correct for a single writer, which is what we have
                # once `detect_task_rank` gives each task its own file.
                existing = path.read_text() if path.exists() else ""
                path.write_text(existing + payload)
            self._buf.clear()
            self._buf_chars = 0
            self._last_flush = time.time()
        except Exception:  # noqa: BLE001 - mirroring must never kill the job
            # Drop the buffer rather than growing it without bound; the local
            # stream still has everything.
            self._buf.clear()
            self._buf_chars = 0
            self._broken = True


class _GcsTee:
    """File-like proxy: writes through to `stream` and into the shared writer.

    Deliberately owns no buffer of its own. Several tees (stdout, stderr) point
    at ONE `_RemoteLogWriter`; that is what stops the two streams from erasing
    each other's flushes.
    """

    def __init__(self, stream, writer: _RemoteLogWriter):
        self._stream = stream
        self._writer = writer

    @property
    def writer(self) -> _RemoteLogWriter:
        return self._writer

    def write(self, text):
        # Return the local stream's count: `io` callers expect the number of
        # characters written, and the old version returned None.
        written = self._stream.write(text)
        self._writer.write(text)
        return written

    def flush(self):
        self._stream.flush()
        self._writer.flush()

    def isatty(self):
        return False

    def close(self):
        """Flush, but never close the underlying stream.

        A tee stands in for `sys.stdout`/`sys.stderr`, and closing those breaks
        every subsequent write in the process. `logging.shutdown()` closes every
        handler at interpreter exit, and absl's `PythonHandler.close` guards
        only against the FOUR objects `sys.stdout/stderr/__stdout__/__stderr__`
        currently name -- a tee that has since been swapped out of `sys.stderr`
        is not one of them, so absl calls `.close()` on it. Without this method
        that fell through `__getattr__` to the wrapped stream. Explicitly
        defining it keeps mirroring alive and keeps shutdown quiet.
        """
        self.flush()

    def __getattr__(self, name):
        # Everything this class does not define is answered by the wrapped
        # stream, including by raising AttributeError when it does not have the
        # attribute either. That is the truthful answer -- but it means a tee
        # must never be installed over a stream some OTHER component expects a
        # richer interface from; see `reattach_absl_handlers`, which is careful
        # about exactly that.
        return getattr(self._stream, name)


# Set by mirror_logs_to_bucket so close_attempt_log can find this attempt's file
# without the caller threading it through.
_ATTEMPT_LOG_PATH: Optional[str] = None
_ATTEMPT_LOG_RANK: int = 0


# --------------------------------------------------------------------------- #
# DIVERGENCE FROM EqR-jax, and the reason for it.
#
# Everything above tees the STREAM OBJECTS `sys.stdout` / `sys.stderr`. Inside
# google3 that is not sufficient, and the gap is invisible until you look for
# it: `absl.app.run()` calls `use_cpp_logging()` whenever `pywrapbase` is
# importable -- which in a Blaze binary it always is -- and from that moment
# `logging.info(...)` is handled by `absl.logging.CppHandler`, whose `emit`
# calls `pywrapbase.LogMessageScript(...)`. That reaches the C++ logging
# backend and comes out of FILE DESCRIPTOR 2 directly. It never touches the
# Python object named `sys.stderr`, so a tee over that object cannot see it,
# no matter how correctly the tee is installed.
#
# Measured, not assumed: `tools/g3_mirror_probe` mirrored its `print()`, its
# raw `sys.stderr.write()` and a full traceback, and dropped exactly one line
# -- the `logging.info()` one -- while the diagnostic showed absl's own
# handler stream WAS the tee. That is the signature of a second, lower path.
#
# It matters here more than it did in EqR-jax, which logs mostly through
# `print()`. jax_llava logs through `log_for_0` -> `logging.info`, so on Borg
# the C++ path is the MAJORITY of the training log: step, loss, accuracy,
# checkpoint writes. Mirroring stdout alone would have produced a file that
# looked healthy and contained almost none of the run.
#
# So mirror at the RECORD level as well. A `logging.Handler` on the root logger
# sees every record before any backend claims it, so it is indifferent to
# whether absl is in Python mode or C++ mode today. It is strictly additive:
# the stream tee still catches `print()` and anything written to the fds by
# Python, and `_should_emit()` suppresses the one case where both would fire.
#
# What this still does NOT capture: output written to fd 2 by C++ code that
# never entered Python -- a CHECK failure inside XLA, TPU bring-up chatter,
# `LOG(FATAL)`. Capturing those needs an fd-level `dup2` pipe, which can
# deadlock the process if the pump thread ever stalls, and a mirror that can
# hang the job it is watching is a worse bargain than a mirror with a known
# blind spot. The blind spot is covered from the other side instead, by
# `g3_logmirror`'s startup/backend markers and `main.py`'s import-crash file.
# --------------------------------------------------------------------------- #


class _MirrorLogHandler(_logging.Handler):
    """Writes every `logging` record into the remote log, backend-agnostically.

    Deliberately holds the same `_RemoteLogWriter` the stream tees hold, so the
    record lands in the one file, in order, next to the `print()` output it
    belongs between.
    """

    def __init__(self, writer: "_RemoteLogWriter"):
        super().__init__()
        self._writer = writer
        self.setFormatter(_logging.Formatter(
            "%(levelname).1s%(asctime)s %(process)d %(filename)s:%(lineno)d] %(message)s",
            datefmt="%m%d %H:%M:%S",
        ))

    def _should_emit(self) -> bool:
        """False when this record already reaches the mirror via the stream tee.

        Only true-double-delivery is suppressed. When absl is in C++ mode the
        stream tee sees nothing, so this handler is the ONLY path and must fire;
        when absl is in Python mode pointed at a tee, the tee already has it and
        firing again would duplicate every line.
        """
        try:
            from absl import logging as absl_logging

            handler = absl_logging.get_absl_handler()
            using_cpp = getattr(handler, "is_using_cpp_logging", None)
            if callable(using_cpp) and using_cpp():
                return True
            python_handler = getattr(handler, "python_handler", None)
            return not isinstance(getattr(python_handler, "stream", None), _GcsTee)
        except Exception:  # noqa: BLE001 - when in doubt, keep the evidence
            return True

    def emit(self, record) -> None:
        # A logging handler that raises takes the log line AND prints a
        # "--- Logging error ---" storm to stderr. Swallow everything.
        try:
            if not self._should_emit():
                return
            # NEVER flush inline from here. A logging handler runs on whatever
            # thread called `logging.info`, and in this binary that includes
            # threads owned by the CNS client itself -- `gdm_access_logger.py`
            # logs from inside the Orbax/tfhub read path. Issuing a blocking
            # CNS write from such a thread re-enters CNS on a fiber-coloured
            # stack and trips
            #     selectables.cc:131] Must not select on fiber cancellation
            #     from functions of other colors
            # which is a CHECK, i.e. SIGABRT and an 8.9 GB core, from a
            # LOGGING call. Buffer only; the background flusher owns a plain
            # Python thread with no colour and ships the bytes within
            # _MIRROR_FLUSH_SECONDS.
            self._writer.write(self.format(record) + "\n",
                               allow_inline_flush=False)
        except Exception:  # noqa: BLE001
            pass

    def flush(self) -> None:
        # Deliberately a NO-OP. `logging.shutdown()` and `logging.Handler`
        # call this from arbitrary threads, and the whole point of `emit`
        # buffering is that this class never issues a CNS op itself. The
        # background flusher and main.py's explicit `flush_logs()` (main
        # thread, uncoloured) are what actually write.
        return


def _install_record_handler(writer: "_RemoteLogWriter") -> None:
    """Attach a `_MirrorLogHandler` to the root logger, replacing any previous.

    Idempotent: a second `mirror_logs_to_bucket` (a restart, or the probe's
    attempt-2 case) must not leave the previous attempt's writer attached, or
    records would keep flowing into the log of a life that has ended.
    """
    try:
        root = _logging.getLogger()
        for existing in list(root.handlers):
            if isinstance(existing, _MirrorLogHandler):
                root.removeHandler(existing)
        root.addHandler(_MirrorLogHandler(writer))
    except Exception:  # noqa: BLE001 - logging plumbing must never kill the job
        pass


def detect_attempt_id() -> Optional[int]:
    """A discriminator that CHANGES on every task restart, or None off Borg.

    THE WORK-UNIT ID IS THE WRONG ANSWER, which is what this used to return.
    `--resume_xid` appends a work unit, so the WID does move when a human
    resumes an experiment -- but a PREEMPTION RESTART re-runs the same work
    unit, and the WID is then identical across all of a job's lives. Every
    restart therefore computed `attempt=1` and reopened
    `rank_<n>_attempt1.log`. The bytes survived (the writer appends), but the
    file became a concatenation of unrelated lives, which is exactly the
    condition `latest_attempt_log` exists to avoid: `tpu check` reads a stale
    traceback from the FIRST life and reports a merely-preempted job as a code
    bug. Measured on XID 276576525: three lives, WID `1` in all three.

    Borg does not export a restart counter to the job, but the task's UID does
    change every life and appears in the local-ram-fs path Borg gives the task:

        /export/hda3/borglet/local_ram_fs_dirs/3.qiaos_group_276576525.1.main.qiaos.2541478647.<hash>
                                               ^task            ^XID     ^WID       ^UID

    That UID (2541478647 -> 2541564660 -> 2542345531 across the three lives) is
    monotonically increasing, so it both distinguishes and ORDERS the lives --
    which is what `latest_attempt_log`'s "highest number wins" rule needs.

    Preference order, most authoritative first:
      1. An explicit restart counter, if the environment ever grows one.
      2. The task UID, which is what actually moves today.
      3. None -- the caller then counts existing files, which is correct
         everywhere and merely needs a directory listing.

    The WID is deliberately NOT consulted any more: it is constant across the
    lives this function exists to tell apart, and preferring it is what made
    the count-the-files fallback unreachable.
    """
    import re

    # 1. A real restart counter, if one is ever exported. `BORG_RESTART_COUNT`
    #    is 0-based, so +1 keeps attempt numbers 1-based like the filenames.
    for var in ("BORG_RESTART_COUNT", "BORG_TASK_RESTART_COUNT"):
        value = os.environ.get(var, "").strip()
        if value.isdigit():
            return int(value) + 1

    # 2. The task UID. Present in the ram-fs path and in the task handle; both
    #    spell it `...<jobname>.<user>.<UID>.<hash>` after the `.main.` segment.
    for var in ("BORG_TASK_HANDLE", "TEST_TMPDIR", "TMPDIR", "BORG_LOCAL_RAM_FS_DIR"):
        value = os.environ.get(var, "")
        m = re.search(r"_group_\d+\.\d+\.[a-z0-9_]+\.[a-z0-9_]+\.(\d{6,})\.", value)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass

    # 3. Give up and let the caller count files. Returning the WID here would
    #    be worse than returning nothing: it looks authoritative and is stable
    #    across precisely the events we need to separate.
    return None


def _attempt_number_of(path) -> Optional[int]:
    """Attempt index parsed out of a `rank_<n>_attempt<k>.log` path, or None."""
    import re

    m = re.search(r"_attempt(\d+)\.log$", str(path))
    return int(m.group(1)) if m else None


def _attempt_slot_for(logs_dir: str, rank: int, discriminator: Optional[int]) -> int:
    """A small 1-based number for this life that no existing file already uses.

    The raw discriminator from `detect_attempt_id` is a task UID -- ten digits,
    and meaningful only by comparison. Filenames want 1, 2, 3. This maps one to
    the other by taking one past the highest slot present, which:

      * never reuses a name, so two lives cannot append to one file even if
        their discriminators collide or are both None;
      * keeps `latest_attempt_log`'s "highest number wins" rule true, because
        slots are handed out in start order.

    The UID itself is not thrown away -- `mirror_logs_to_bucket` writes it into
    the file's header line, so a file can still be tied back to its Borg task.

    Racing tasks are not a concern: each rank owns its own filename series.
    """
    existing = _existing_attempt_logs(logs_dir, rank)
    highest = 0
    for path in existing:
        number = _attempt_number_of(path)
        if number is not None and number > highest:
            highest = number
    return highest + 1


def _existing_attempt_logs(logs_dir: str, rank: int):
    """Previous attempts' logs for this rank, oldest first. Never raises."""
    import re

    try:
        from etils import epath

        pattern = re.compile(rf"^rank_{rank}_attempt(\d+)\.log$")
        found = []
        for entry in epath.Path(logs_dir).iterdir():
            m = pattern.match(entry.name)
            if m:
                found.append((int(m.group(1)), entry))
        return [p for _, p in sorted(found)]
    except Exception:  # noqa: BLE001 - dir may not exist yet
        return []


def mirror_logs_to_bucket(
    bucket: str, rank: Optional[int] = None, *, background_flush: bool = True
) -> Optional[str]:
    """Tee stdout+stderr into `<bucket>/logs/rank_<n>_attempt<k>.log`.

    Both streams share one `_RemoteLogWriter`, so the file is a faithfully
    interleaved merge instead of two writers overwriting each other. `rank`
    defaults to `detect_task_rank()`.

    ONE FILE PER ATTEMPT. Every attempt used to append to a single
    `rank_<n>.log`, which made the file a concatenation of unrelated runs:
    `tpu check` read a stale traceback from attempt 1 and reported XID 275793223
    as `CODE BUG: ValueError` for hours after that bug was fixed and the job was
    merely being preempted. Splitting per work-unit id means the newest file is
    unambiguously the current attempt, and `latest_attempt_log()` finds it.

    A short header links the chain backwards, and `close_attempt_log` writes the
    forward pointer plus a step summary into the PREVIOUS file, so either end of
    the chain can be navigated from the other.
    """
    if not bucket:
        return None
    if rank is None:
        rank = detect_task_rank()
    logs_dir = bucket.rstrip("/") + "/logs"
    attempt = detect_attempt_id()
    if attempt is None:
        # Not on Borg (or an unrecognised handle): one more than the highest
        # attempt already present, so a local rerun still does not clobber.
        attempt = len(_existing_attempt_logs(logs_dir, rank)) + 1
    # NEVER REOPEN A FILE ANOTHER LIFE IS USING. `attempt` is a discriminator,
    # not a slot number: it changes per restart (task UID) but says nothing
    # about ordering relative to files already on disk, and off Borg it is None
    # and the count-based fallback can collide after a partial cleanup. Both
    # cases end the same way -- two processes appending to one file, which is
    # how a run's log became three interleaved lives.
    #
    # So resolve to a name nothing else owns. `_attempt_slot_for` numbers this
    # life one past the highest already present, and records the raw
    # discriminator in the header for provenance.
    slot = _attempt_slot_for(logs_dir, rank, attempt)
    remote = f"{logs_dir}/rank_{rank}_attempt{slot}.log"
    try:
        from etils import epath

        epath.Path(remote).parent.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        return None

    global _ATTEMPT_LOG_PATH, _ATTEMPT_LOG_RANK
    _ATTEMPT_LOG_PATH = remote
    _ATTEMPT_LOG_RANK = rank

    # Backward link, written before any job output so it is the first thing in
    # the file. The forward link lands in the previous file at shutdown
    # (close_attempt_log); a task that is SIGKILLed never gets to write it,
    # which is exactly why the backward link is written eagerly here.
    #
    # Compare against the SLOT, not the raw discriminator: slots are what the
    # filenames carry. (Selecting by number rather than by list position also
    # keeps this file -- which exists by now -- from being cited as its own
    # predecessor, which is how attempt 6 once came to point at attempt 6.)
    previous = [
        p for p in _existing_attempt_logs(logs_dir, rank)
        if _attempt_number_of(p) is not None and _attempt_number_of(p) < slot
    ]
    prior = str(previous[-1]) if previous else None

    writer = _RemoteLogWriter(remote)
    # Carry the raw discriminator (the Borg task UID) alongside the slot: the
    # slot orders the lives, the UID identifies which Borg task each one was,
    # and only the pair can answer "is this file the life that died at 03:41?".
    stamp = f" [task-id {attempt}]" if attempt is not None else ""
    header = [f"=== attempt {slot} (rank {rank}){stamp} begins ==="]
    if prior:
        header.append(f"=== previous attempt log: {prior} ===")
    writer.write("\n".join(header) + "\n")
    writer.flush()
    sys.stdout = _GcsTee(sys.stdout, writer)
    sys.stderr = _GcsTee(sys.stderr, writer)

    reattach_absl_handlers()
    # Mirror at the RECORD level too. In google3 `app.run()` switches absl to
    # C++ logging, which writes to fd 2 and never touches the object above --
    # so on Borg this, not the tee, is what carries the training log. See the
    # divergence note at `_MirrorLogHandler`.
    _install_record_handler(writer)

    if background_flush:
        writer.start_background_flusher()

    # Last-ditch flush when the interpreter unwinds, including on exceptions.
    # Flush the writer itself rather than sys.stdout/sys.stderr, so it still
    # works if something reassigned those in the meantime.
    import atexit

    atexit.register(writer.flush)
    return remote


def latest_attempt_log(bucket: str, rank: int = 0) -> Optional[str]:
    """Newest attempt log for `rank` under `bucket`, or None.

    What any reader -- a human, `tpu check`, a watchdog -- actually wants: the
    CURRENT attempt, not a concatenation of every attempt ever run. Falls back
    to the pre-split `rank_<n>.log` so old runs stay readable.
    """
    logs_dir = bucket.rstrip("/") + "/logs"
    existing = _existing_attempt_logs(logs_dir, rank)
    if existing:
        return str(existing[-1])
    try:
        from etils import epath

        legacy = epath.Path(f"{logs_dir}/rank_{rank}.log")
        if legacy.exists():
            return str(legacy)
    except Exception:  # noqa: BLE001
        pass
    return None


def close_attempt_log(summary: str = "") -> None:
    """Append a footer to THIS attempt's log, and a forward link to the previous.

    Called at shutdown. `summary` should say what the attempt accomplished --
    the caller knows the step range and the last checkpoint; this module does
    not. The forward pointer is written into the PREVIOUS attempt's file too,
    so someone reading an old log is told where the run continued instead of
    concluding it died there.

    Best effort throughout: a preempted task is often SIGKILLed with no chance
    to run this, which is why the backward link in the header is the load-bearing
    half of the chain.
    """
    path = _ATTEMPT_LOG_PATH
    if not path:
        return
    try:
        from etils import epath

        footer = "=== attempt ends ==="
        if summary:
            footer = f"=== attempt ends: {summary} ==="
        with epath.Path(path).open("a") as handle:
            handle.write(footer + "\n")
    except Exception:  # noqa: BLE001
        pass

    # Forward link into the previous attempt's file.
    try:
        from etils import epath

        logs_dir = str(epath.Path(path).parent)
        mine = _attempt_number_of(path)
        # Same reasoning as the backward link: pick by attempt NUMBER. Using
        # list position would make a file point at itself.
        earlier = [
            p for p in _existing_attempt_logs(logs_dir, _ATTEMPT_LOG_RANK)
            if _attempt_number_of(p) is not None
            and mine is not None
            and _attempt_number_of(p) < mine
        ]
        if earlier:
            note = f"=== continued in: {path}"
            if summary:
                note += f" ({summary})"
            note += " ==="
            with epath.Path(earlier[-1]).open("a") as handle:
                handle.write(note + "\n")
    except Exception:  # noqa: BLE001
        pass


def _console_streams() -> list:
    """Every object that is currently, or was just now, a console stream.

    The four stdlib names, plus -- unwrapping any `_GcsTee` among them -- what
    each tee wraps. That second part is the whole point: by the time
    `reattach_absl_handlers` runs, `mirror_logs_to_bucket` has ALREADY replaced
    `sys.stderr` with a tee, so a handler built before the swap holds the
    stream the tee now wraps and matches nothing in the four names. Those
    handlers are precisely the ones that need moving. Unwrapping also makes a
    second `reattach` (mirroring re-installed after a Borg restart) work.
    """
    found = []
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        # Bounded walk: a tee wrapping a tee is legal but a cycle is not, and a
        # logging helper must not be able to hang the process.
        for _ in range(8):
            if stream is None or any(stream is seen for seen in found):
                break
            found.append(stream)
            if not isinstance(stream, _GcsTee):
                break
            stream = stream._stream  # noqa: SLF001 - same module
    return found


def _is_console_stream(stream) -> bool:
    """True if `stream` is one of the process's console streams (or a tee of one).

    The question `reattach_absl_handlers` has to answer is "was this handler
    built to write to the console?", and the only honest evidence is the
    identity of the object it holds.

    Anything not in `_console_streams()` -- a `StringIO`, a log file, a pytest
    capture buffer -- belongs to someone else and is left alone.
    """
    if stream is None:
        return False
    return any(stream is candidate for candidate in _console_streams())


def reattach_absl_handlers() -> None:
    """Point absl's and the root logger's console handlers back at `sys.stderr`.

    Idempotent, and safe to call when no mirror is installed.

    WHY THIS IS PUBLIC. `mirror_logs_to_bucket` installs a `_GcsTee` over
    stdout/stderr, but the stdlib logging handlers captured the ORIGINAL stderr
    when they were constructed, so they have to be repointed. Any library that
    later builds its own handlers, or re-runs absl's logging setup, silently
    steals the streams back and every subsequent line bypasses the mirror.

    `clu.metric_writers.create_default_writer` does exactly that. The symptom is
    brutally quiet: XID 275709629 (submitted 18:50) mirrored 173 lines, and
    every job after commit fae1898 (18:51, which added the CLU writer) mirrored
    exactly 4 -- the lines printed before the writer is constructed. The job
    itself is fine; only the durable log is lost, and on Borg that log is the
    ONLY record, since the local stream dies with the task.

    So this must be re-invoked after constructing anything that touches logging.

    SECOND BUG, same line. `get_absl_handler()` returns an `ABSLHandler`, which
    has NO `setStream` -- the stream lives on its delegate, `.python_handler`
    (an `absl.logging.PythonHandler`). The original code called
    `get_absl_handler().setStream(...)` inside a bare `except: pass`, so it
    raised AttributeError on every single invocation and was silently swallowed:
    absl records were NEVER mirrored, from the first commit onward. Reach
    through `python_handler`, and let each step fail independently so one
    broken handler cannot skip the rest.

    THIRD BUG. The root-logger sweep used to repoint EVERY `StreamHandler` it
    found, on the assumption that they all write to the console. They do not: a
    `StreamHandler` over an in-memory buffer is how capture works -- pytest's
    `LogCaptureHandler` holds a `StringIO`, and so do most log-assertion
    helpers. Repointing one of those at `sys.stderr` steals the buffer the owner
    is about to read, and the owner's next `stream.getvalue()` lands on the tee,
    which proxies it to the real stdout and raises `AttributeError:
    '...' object has no attribute 'getvalue'`. That is precisely what made five
    of `tests/test_logging_mirror.py`'s own tests fail. Only handlers that are
    currently pointed at a console stream are console handlers, so only those
    are moved -- see `_is_console_stream`.
    """
    import logging as _logging

    try:
        from absl import logging as absl_logging

        handler = absl_logging.get_absl_handler()
        # `python_handler` is the delegate that actually owns `.stream`; older
        # absl versions expose setStream on the outer handler instead.
        target = getattr(handler, "python_handler", handler)
        if hasattr(target, "setStream"):
            target.setStream(sys.stderr)
        elif hasattr(target, "stream"):
            target.stream = sys.stderr
    except Exception:  # noqa: BLE001 - logging plumbing must never kill the job
        pass

    try:
        for h in _logging.getLogger().handlers:
            if isinstance(h, _logging.StreamHandler) and _is_console_stream(
                getattr(h, "stream", None)
            ):
                h.setStream(sys.stderr)
    except Exception:  # noqa: BLE001
        pass
