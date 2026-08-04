"""Early-boot evidence for a job whose Borg logs you cannot read.

Three files, written to `$CHECKPOINT_BUCKET/logs/`, covering the windows the
normal log mirror cannot:

* `_startup_rank<N>.txt`  -- proof `main()` was entered at all, written before
  the mirror exists, because installing the mirror is itself one of the things
  that can fail.
* `_import_crash_rank<N>.txt` -- written by `main.py` (not here) for a crash in
  the module-scope imports, i.e. before `InitGoogle()` and therefore before any
  `/cns/` API is legal.
* `_backend_rank<N>.txt` -- the JAX backend and device list, written BEFORE the
  assertion that would kill the task for being on CPU, so the evidence outlives
  the assertion.

WHY THIS IS NOT THE LOG MIRROR. Tee-ing stdout/stderr into
`<bucket>/logs/rank_<n>_attempt<k>.log` lives in `utils/logging_util.py`,
grafted verbatim from EqR-jax so the two projects are debugged identically --
same paths, same attempt-slot semantics, same background flusher. This module
used to carry a second, parallel implementation of that; it was deleted rather
than maintained, and `mirror_logs`/`flush_logs` below are thin aliases so
existing call sites keep working.

What remains here is genuinely absent from EqR-jax: it never had to report from
before `InitGoogle()`, because it never died there. jax_llava did, twice (XIDs
276839294 and 276859816: FAILURE, empty status message, peak RSS ~400 MiB, dead
<40 s after RUN, against imports that take ~5 min and 11 GB to complete).

Everything here is best effort and never raises. Evidence that can kill the
thing it is evidence about is worse than no evidence.
"""

import os
import sys
import time


def _gfile():
    from google3.pyglib import gfile
    return gfile


def detect_task_rank(default: int = 0) -> int:
    """This task's index among the job's tasks, without touching JAX.

    Delegates to the grafted EqR-jax implementation so both projects derive the
    rank the same way (unambiguous integer env vars first, then the absl
    `jax_task_id` flag, then `BORG_TASK_HANDLE` parsed with Borg's own regex --
    the naive `split(".")[0]` returns the literal "logs" for a handle carrying
    the optional `logs.` prefix, which silently made every task rank 0).

    Kept as a re-export rather than a second copy: this module is imported from
    `main()` before `logging_util` is safe to touch in some paths, and callers
    already say `g3_logmirror.detect_task_rank()`.
    """
    from utils.logging_util import detect_task_rank as _detect
    return _detect(default)


def write_startup_marker(bucket, rank=None):
    """Record that `main()` was entered, as the very first thing it does.

    A Borg task that dies during startup leaves NOTHING: the work-unit message
    says only "terminated in state FAILURE", the task log is garbage-collected
    within minutes, and the log mirror does not exist yet because installing it
    is itself one of the things that can fail. Absence of evidence is then the
    only evidence, and it cannot distinguish "died before main()" from "died
    while setting up logging" from "never scheduled".

    One small file, written before anything else can fail, splits that
    ambiguity in half for the cost of a single CNS open. It also proves the
    job's identity can write the bucket at all -- which is the other thing you
    cannot otherwise tell from silence.

    Returns the path, or None. Never raises.
    """
    if not bucket:
        return None
    if rank is None:
        rank = detect_task_rank()
    path = f"{bucket.rstrip('/')}/logs/_startup_rank{rank}.txt"
    try:
        gfile = _gfile()
        parent = f"{bucket.rstrip('/')}/logs"
        if not gfile.Exists(parent):
            gfile.MakeDirs(parent)
        fields = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "rank": rank,
            "pid": os.getpid(),
            "argv": " ".join(sys.argv[:6]),
        }
        for key in ("BORG_TASK_HANDLE", "BORG_CELL", "BORG_PHYSICAL_CELL",
                    "BORG_JOB_SIZE", "XM_XID", "XM_WID", "CHECKPOINT_BUCKET",
                    "TMPDIR", "HOSTNAME"):
            value = os.environ.get(key)
            if value:
                fields[key] = value
        body = "".join(f"{k}={v}\n" for k, v in fields.items())
        with gfile.Open(path, "w") as handle:
            handle.write(body)
        return path
    except Exception:  # noqa: BLE001 - a marker must never block startup
        return None


def write_backend_marker(bucket, rank=None):
    """Record the JAX backend and device list to CNS, as a durable file.

    The expensive failure on a TPU job is not a crash, it is a SILENT CPU
    fallback: `//third_party/py/jax` alone builds a CPU-only binary, and
    google3 registers the TPU backend factory with `fail_quietly=True`, so a
    missing `//learning/brain/research/jax:tpu_support` degrades to CPU with
    nothing louder than a `logger.info`. XID 275525750 burned a v6p-16 for
    2.5 h reporting `[CpuDevice(id=0)]` before the pruner reclaimed it for a
    0.000 duty cycle.

    `_assert_accelerator_backend()` already turns that into a loud early
    death. This writes the same facts somewhere that survives the death, and
    that is readable WITHOUT any Borg log access -- which on a restricted-LOAS
    workstation is the only kind of evidence there is. Reading one small file
    from CNS then answers "did it get real chips?" definitively.

    Returns the path, or None. Never raises.
    """
    if not bucket:
        return None
    if rank is None:
        rank = detect_task_rank()
    path = f"{bucket.rstrip('/')}/logs/_backend_rank{rank}.txt"
    try:
        import jax  # local: this is only ever called after the backend is up

        fields = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "rank": rank,
            "default_backend": jax.default_backend(),
            "process_index": jax.process_index(),
            "process_count": jax.process_count(),
            "device_count": jax.device_count(),
            "local_device_count": jax.local_device_count(),
            "local_devices": repr(jax.local_devices()),
        }
        for key in ("BORG_TASK_HANDLE", "BORG_CELL", "XM_XID"):
            value = os.environ.get(key)
            if value:
                fields[key] = value
        body = "".join(f"{k}={v}\n" for k, v in fields.items())
        gfile = _gfile()
        parent = f"{bucket.rstrip('/')}/logs"
        if not gfile.Exists(parent):
            gfile.MakeDirs(parent)
        with gfile.Open(path, "w") as handle:
            handle.write(body)
        return path
    except Exception:  # noqa: BLE001 - evidence must never block startup
        return None


# --------------------------------------------------------------------------- #
# Aliases onto the grafted EqR-jax mirror.
#
# The implementation moved to `utils/logging_util.py`; these keep `main.py` and
# `utils/g3_metrics.py` reading naturally and give one place to look when
# asking "which mirror is this project using?". The answer is: EqR-jax's.
# --------------------------------------------------------------------------- #


def mirror_logs(bucket, rank=None):
    """Tee stdout+stderr into `<bucket>/logs/rank_<n>_attempt<k>.log`.

    Returns the path, or None if a mirror could not be opened (never fatal).
    MUST be called from inside `main()`: it touches CNS.
    """
    from utils.logging_util import mirror_logs_to_bucket
    return mirror_logs_to_bucket(bucket, rank=rank)


def attempt_log_path():
    """The file this process is mirroring into, or None."""
    from utils import logging_util
    return logging_util._ATTEMPT_LOG_PATH  # pylint: disable=protected-access


def flush_logs():
    """Flush the mirror, if any. Safe to call from an exception handler."""
    from utils.logging_util import _GcsTee
    for stream in (sys.stdout, sys.stderr):
        try:
            if isinstance(stream, _GcsTee):
                stream.flush()
        except Exception:  # noqa: BLE001
            pass


def close_attempt_log(summary: str = ""):
    """Footer for this attempt's log plus a forward link into the previous one."""
    try:
        from utils.logging_util import close_attempt_log as _close
        _close(summary)
    except Exception:  # noqa: BLE001 - shutdown bookkeeping must not fail a run
        pass


def _reattach_absl_handler():
    """Point absl's console handlers back at the (tee'd) `sys.stderr`.

    Anything that rebuilds logging handlers -- `clu.metric_writers`, most
    notably -- steals the streams back and every subsequent line bypasses the
    mirror. Re-invoke after constructing such a thing.
    """
    from utils.logging_util import reattach_absl_handlers
    reattach_absl_handlers()
