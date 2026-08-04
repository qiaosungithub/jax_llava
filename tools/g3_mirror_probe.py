"""Prove the CNS log mirror works, in seconds instead of in a Borg smoke.

The mirror is the ONLY readable log channel for a Borg task on this
workstation (`borg tasklog` SIGABRTs on PERMISSION_DENIED, `analog --remote`
is refused, Coroner's binary is unreadable, and the task is GC'd within
minutes). A bug in the mirror therefore does not merely lose a log -- it makes
every subsequent failure undiagnosable. So it is tested directly, against real
CNS, rather than inferred from a training run that also has to survive TPU
bring-up and five minutes of imports.

What it asserts, in order:
  1. the startup marker lands and is readable;
  2. `mirror_logs_to_bucket` returns a `rank_<n>_attempt<k>.log` path;
  3. `print()` (stdout), `logging.info` (absl -> stderr) and a traceback all
     reach that ONE file -- the interleave is the property the single shared
     writer exists to provide, and the absl half is the one that was silently
     broken in the original for every commit of its life;
  4. a SECOND `mirror_logs_to_bucket` on the same bucket picks attempt k+1 and
     does not touch attempt k -- restart semantics, which is what turns a
     stale traceback into a misdiagnosis when it is wrong;
  5. `close_attempt_log` writes the footer, and the forward pointer into the
     previous attempt.

Run:
  ./blaze-bin/experimental/users/qiaos/jax_llava/g3_mirror_probe \
      --bucket=/cns/yucmhcg-d/home/qiaos/jax_llava/mirror_probe
"""

import os
import sys
import traceback

# The project is imported by path, not as Bazel targets (see BUILD): put the
# package root on sys.path so `from utils import ...` resolves the same way it
# does inside `main`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_string("bucket", "", "CNS dir to mirror into. Required.")

_FAILURES = []


def _check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}" + (f": {detail}" if detail else "")
    # Bypass the tee for the verdict lines: they must be visible locally even
    # if the thing under test is what is broken.
    print(line, file=sys.__stderr__, flush=True)
    if not ok:
        _FAILURES.append(line)


def _read(path):
    from google3.pyglib import gfile
    with gfile.Open(path, "r") as handle:
        return handle.read()


def main(argv):
    del argv
    bucket = FLAGS.bucket.strip()
    if not bucket:
        raise app.UsageError("--bucket is required")

    from utils import g3_logmirror
    from utils import logging_util

    rank = logging_util.detect_task_rank()
    _check("detect_task_rank", isinstance(rank, int), f"rank={rank}")

    marker = g3_logmirror.write_startup_marker(bucket)
    _check("startup marker written", bool(marker), str(marker))
    if marker:
        body = _read(marker)
        _check("startup marker readable", "pid=" in body, body.split("\n")[0])

    # In google3 `app.run()` has by now switched absl to C++ logging, so
    # `logging.info` goes out of fd 2 and NOT through `sys.stderr`. Recorded
    # here because it is the reason ABSL-CANARY below is the load-bearing
    # assertion rather than a formality.
    _check("absl is in google3 C++ logging mode",
           logging.get_absl_handler().is_using_cpp_logging(),
           "if False, the record-level handler is untested by this run")

    first = logging_util.mirror_logs_to_bucket(bucket, rank=rank)
    _check("attempt 1 opened", bool(first) and "_attempt" in str(first), str(first))

    print("STDOUT-CANARY: printed through sys.stdout")
    logging.info("ABSL-CANARY: emitted through absl logging")
    sys.stderr.write("STDERR-CANARY: written straight to sys.stderr\n")
    try:
        raise ValueError("TRACEBACK-CANARY")
    except ValueError:
        traceback.print_exc()
    for stream in (sys.stdout, sys.stderr):
        stream.flush()

    text = _read(first)
    for canary in ("STDOUT-CANARY", "ABSL-CANARY", "STDERR-CANARY",
                   "TRACEBACK-CANARY"):
        _check(f"{canary} in mirror", canary in text)
    _check("attempt header present", "=== attempt " in text,
           text.split("\n")[0])
    # No line may arrive twice: the stream tee and the record handler both
    # exist, and the whole point of `_should_emit` is that exactly one of them
    # fires for any given line.
    for canary in ("STDOUT-CANARY", "ABSL-CANARY", "STDERR-CANARY"):
        n = text.count(canary)
        _check(f"{canary} appears exactly once", n == 1, f"count={n}")

    # Restart semantics: a second mirror must NOT reopen the first file.
    second = logging_util.mirror_logs_to_bucket(bucket, rank=rank)
    _check("attempt 2 opened", bool(second), str(second))
    _check("attempt 2 is a different file", second != first,
           f"{first} -> {second}")
    n1 = logging_util._attempt_number_of(first)
    n2 = logging_util._attempt_number_of(second)
    _check("attempt number advanced", n1 is not None and n2 == n1 + 1,
           f"{n1} -> {n2}")
    _check("attempt 2 links back to attempt 1",
           str(first) in _read(second), "backward link")

    logging_util.close_attempt_log("mirror probe")
    _check("footer in attempt 2", "attempt ends" in _read(second))
    _check("forward pointer in attempt 1", "continued in:" in _read(first))

    latest = logging_util.latest_attempt_log(bucket, rank=rank)
    _check("latest_attempt_log finds attempt 2", str(latest) == str(second),
           str(latest))

    verdict = "ALL PASS" if not _FAILURES else f"{len(_FAILURES)} FAILURE(S)"
    print(f"\n=== mirror probe: {verdict} ===", file=sys.__stderr__, flush=True)
    for line in _FAILURES:
        print("  " + line, file=sys.__stderr__, flush=True)
    if _FAILURES:
        sys.exit(1)


if __name__ == "__main__":
    app.run(main)
