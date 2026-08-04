"""Tee stdout/stderr to durable storage, for a job whose logs you cannot read.

On this workstation `borg tasklog` and `analog --remote` fail with
PERMISSION_DENIED (restricted LOAS), and the Borg task itself is garbage
collected within minutes of finishing. So the application's own mirror is
usually the ONLY log that survives long enough to read. It is written from the
first line of `main()`, before JAX or the TPU backend is touched, because those
are among the most common places to die.

Design constraints, all learned the expensive way (see EqR-jax's
`utils/logging_util.py`, which this is a compact sibling of):

* **One writer per file.** stdout and stderr share a single writer so the file
  is a faithful interleave rather than two streams clobbering each other.
* **One file per attempt.** A Borg restart re-runs the same work unit, so
  anything keyed on the work-unit id appends a second life onto the first file
  and a stale traceback then reads as the current failure.
* **Never raise.** A mirror is a diagnostic. If the backend misbehaves it
  disables itself and the job carries on.
* **Flush eagerly on error lines and on a wall clock**, so a HUNG job has still
  shipped its last words.
"""

import os
import re
import sys
import threading
import time

_FLUSH_SECONDS = 20.0
_MAX_BUFFER_CHARS = 1 << 20
_URGENT_TOKENS = (
    "Traceback", "Error", "ERROR", "error:", "Exception",
    "FATAL", "Fatal", "CRITICAL", "Refusing",
)

_ATTEMPT_LOG_PATH = None


def attempt_log_path():
    """The file this process is mirroring into, or None."""
    return _ATTEMPT_LOG_PATH


def detect_task_rank(default: int = 0) -> int:
    """This task's index among the job's tasks, without touching JAX.

    Called before `jax.process_index()` is legal, so every source is an
    environment variable or an already-parsed absl flag. Unambiguous integer
    sources first; `BORG_TASK_HANDLE` is parsed last and with Borg's own regex,
    because the handle carries an optional `logs.` prefix that makes the naive
    `split(".")[0]` return the string "logs" -- which then silently becomes
    rank 0 for every task, and all of them mirror into one file.
    """
    for key in ("BORG_TASK_INDEX", "JAX_TASK_ID", "JAX_PROCESS_ID", "TASK_ID", "RANK"):
        value = os.environ.get(key)
        if value is None or not str(value).strip():
            continue
        try:
            return int(str(value).strip())
        except ValueError:
            continue
    try:
        from absl import flags as _flags
        if "jax_task_id" in _flags.FLAGS:
            value = _flags.FLAGS["jax_task_id"].value
            if value is not None:
                return int(value)
    except Exception:  # noqa: BLE001 - a flag lookup must never break startup
        pass
    match = re.match(r"^(?:logs\.)?(\d+)\.", os.environ.get("BORG_TASK_HANDLE", "") or "")
    return int(match.group(1)) if match else default


def _gfile():
    from google3.pyglib import gfile
    return gfile


class _RemoteLogWriter:
    """The single owner of one remote log file. Thread-safe; never raises."""

    def __init__(self, remote_path, flush_seconds=_FLUSH_SECONDS):
        self._path = remote_path
        self._buf = []
        self._chars = 0
        self._flush_seconds = flush_seconds
        self._last_flush = time.time()
        self._lock = threading.Lock()
        self._broken = False
        self._stop = threading.Event()
        self._flusher = None

    @property
    def remote_path(self):
        return self._path

    def write(self, text):
        if self._broken or not text:
            return
        with self._lock:
            self._buf.append(text)
            self._chars += len(text)
            urgent = any(tok in text for tok in _URGENT_TOKENS)
            due = (time.time() - self._last_flush) >= self._flush_seconds
            if urgent or due or self._chars >= _MAX_BUFFER_CHARS:
                self._flush_locked()

    def flush(self):
        if self._broken:
            return
        with self._lock:
            self._flush_locked()

    def start_background_flusher(self):
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
            target=_loop, name="llava-log-mirror-flush", daemon=True)
        self._flusher.start()

    def stop(self):
        self._stop.set()

    def _flush_locked(self):
        if not self._buf:
            return
        payload = "".join(self._buf)
        try:
            with _gfile().Open(self._path, "a") as handle:
                handle.write(payload)
            self._buf.clear()
            self._chars = 0
            self._last_flush = time.time()
        except Exception:  # noqa: BLE001 - mirroring must never kill the job
            # Drop rather than grow without bound; the local stream still has
            # everything, and a broken mirror must not become an OOM.
            self._buf.clear()
            self._chars = 0
            self._broken = True


class _Tee:
    """File-like proxy: writes through to `stream` AND into the shared writer."""

    def __init__(self, stream, writer):
        self._stream = stream
        self._writer = writer

    @property
    def writer(self):
        return self._writer

    def write(self, text):
        written = self._stream.write(text)
        self._writer.write(text)
        return written

    def flush(self):
        self._stream.flush()
        self._writer.flush()

    def isatty(self):
        return False

    def close(self):
        # Flush, but NEVER close the underlying stream: absl's
        # logging.shutdown() closes every handler at exit and only recognises
        # the four objects sys.stdout/stderr/__stdout__/__stderr__ currently
        # name. A tee that has been swapped out is not one of them, so absl
        # would close the real stdout under us.
        self.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _next_attempt_slot(logs_dir, rank):
    """One past the highest attempt already present. Never reuses a name."""
    try:
        existing = _gfile().Glob(f"{logs_dir}/rank_{rank}_attempt*.log")
    except Exception:  # noqa: BLE001
        return 1
    highest = 0
    for path in existing:
        m = re.search(r"_attempt(\d+)\.log$", str(path))
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


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


def mirror_logs(bucket, rank=None):
    """Tee stdout+stderr into `<bucket>/logs/rank_<n>_attempt<k>.log`.

    Returns the path, or None if mirroring could not be set up (which is never
    fatal). MUST be called from inside `main()`: it touches CNS.
    """
    global _ATTEMPT_LOG_PATH
    if not bucket:
        return None
    if rank is None:
        rank = detect_task_rank()
    logs_dir = bucket.rstrip("/") + "/logs"
    try:
        gfile = _gfile()
        if not gfile.Exists(logs_dir):
            gfile.MakeDirs(logs_dir)
        slot = _next_attempt_slot(logs_dir, rank)
        remote = f"{logs_dir}/rank_{rank}_attempt{slot}.log"
        writer = _RemoteLogWriter(remote)
        handle = os.environ.get("BORG_TASK_HANDLE", "")
        writer.write(
            f"=== attempt {slot} (rank {rank}) begins"
            f"{f' [task {handle}]' if handle else ''} ===\n")
        writer.flush()
        if writer._broken:  # pylint: disable=protected-access
            return None
    except Exception:  # noqa: BLE001 - a mirror must never block startup
        return None
    sys.stdout = _Tee(sys.stdout, writer)
    sys.stderr = _Tee(sys.stderr, writer)
    writer.start_background_flusher()
    _reattach_absl_handler()
    _ATTEMPT_LOG_PATH = remote
    return remote


def _reattach_absl_handler():
    """Point absl's logging handler at the NEW sys.stderr.

    absl captures the stream object when its handler is constructed, which
    normally happens at import time -- i.e. before the tee exists. Without
    this, `logging.info(...)` keeps writing to the original stderr and the
    mirror contains only bare prints, which is the majority of a training log
    missing.
    """
    try:
        from absl import logging as absl_logging
        handler = absl_logging.get_absl_handler()
        stream_handler = getattr(handler, "python_handler", None) or handler
        if hasattr(stream_handler, "stream"):
            stream_handler.stream = sys.stderr
    except Exception:  # noqa: BLE001
        pass


def flush_logs():
    """Flush the mirror, if any. Safe to call from an exception handler."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if isinstance(stream, _Tee):
                stream.flush()
        except Exception:  # noqa: BLE001
            pass
