"""jax_llava entry point.

Two execution environments, one file:

* **GCP TPU VM** (the original): `python main.py --config=... --workdir=...`,
  one process per host started by a shell script, data in
  `gs://kmh-gcp-<zone>`, weights read from an NFS mount.
* **google3 / Borg**: a Blaze `py_binary` launched by XManager, data on CNS,
  weights from Placer/CNS, one task per host.

The differences that matter are all at startup, and they are collected here so
the rest of the codebase stays environment-agnostic:

1. **Entry point.** Under Bazel `sys.executable` is None, so stdlib `spawn`
   cannot re-exec the binary and the torch DataLoader's workers never start.
   `g3_multiprocessing.handle_main` teaches multiprocessing how to re-exec a
   Blaze binary and then calls `app.run` for us. It is a no-op elsewhere, but
   it is not importable elsewhere, hence the guarded import.
2. **Distributed init.** google3's JAX self-initialises from the
   `--jax_controller_address` / `--jax_num_tasks` / `--jax_task_id` flags
   XManager passes; calling `jax.distributed.initialize()` on top of that
   either duplicates the work or dies with "coordinator_address should be
   defined". So: initialise explicitly only when nobody else will.
3. **No filesystem before `main()`.** Touching `/cns/` or `/bigstore/` before
   InitGoogle() finishes CHECK-fails and core-dumps the process
   (go/no_file_or_rpc_during_init). Everything that reads a path lives inside
   `main()`.
"""

import os
import sys

# The shared NFS checkout the GCP path imports `big_vision` and `gemma` from.
# Absent under Bazel, where those come from BUILD deps instead.
SHARED_CODE_ROOT = "/kmh-nfs-ssd-us-mount/code/hanhong/shared"
if os.path.isdir(SHARED_CODE_ROOT) and SHARED_CODE_ROOT not in sys.path:
    sys.path.insert(0, SHARED_CODE_ROOT)

# Under Bazel the package directory itself is not on sys.path when the binary
# is launched from a staged snapshot; `import train` must still work.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _boot_log(message, *args):
    """Log during startup without touching JAX.

    Asking JAX anything boots the XLA backend, which in google3 also triggers
    distributed initialisation. Keep startup logging JAX-free so it cannot
    perturb that, and use stderr so it survives even if logging is not
    configured yet.
    """
    print(message % args if args else message, file=sys.stderr, flush=True)


def _add_bazel_imports_dirs():
    """Make `py_library(imports=...)` packages importable under Bazel.

    `//third_party/py/scamper:wandb_mock` ships `wandb_mock/wandb.py` and
    relies on `imports = ["wandb_mock"]` to put that subdirectory on sys.path.
    That attribute is not honoured for this target when depended on from a
    staged package, so a bare `import wandb` raises ModuleNotFoundError at
    module-import time -- before main(), before any logging exists, which on
    Borg surfaces only as an empty status message and no log at all.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    parts = here.split(os.sep)
    if "google3" in parts:
        g3root = os.sep.join(parts[: len(parts) - parts[::-1].index("google3")])
        for rel in ("third_party/py/scamper/wandb_mock",):
            cand = os.path.join(g3root, rel)
            if os.path.isdir(cand) and cand not in sys.path:
                sys.path.append(cand)


_add_bazel_imports_dirs()


def _install_webdataset_shim():
    """Register the in-tree `wds_shim` as the `webdataset` package.

    There is no `//third_party/py/webdataset` in google3 and nothing in the
    depot links the real package. `wds_shim` reimplements exactly the surface
    jax_llava uses (~330 lines) and, unlike a pip webdataset, can read shards
    from CNS. Outside google3 the real package is used and this is a no-op.
    """
    if "webdataset" in sys.modules:
        return "already imported"
    try:
        import webdataset  # noqa: F401  pylint: disable=unused-import
        return "real package"
    except ImportError:
        pass
    import types
    from wds_shim import wds_shim

    sys.modules["webdataset"] = wds_shim
    for name in ("gopen", "filters", "shardlists"):
        sub = types.ModuleType(f"webdataset.{name}")
        holder = getattr(wds_shim, name)
        for attr in dir(holder):
            if not attr.startswith("_"):
                setattr(sub, attr, getattr(holder, attr))
        sys.modules[f"webdataset.{name}"] = sub
        setattr(wds_shim, name, sub)
    sys.modules["webdataset.shardlists"].expand_urls = wds_shim.expand_urls
    sys.modules["webdataset.gopen"].gopen_schemes = wds_shim.gopen_schemes
    sys.modules["webdataset.filters"].RandomMix = wds_shim.RandomMix
    return "wds_shim"


_WDS_SOURCE = _install_webdataset_shim()

import jax  # noqa: E402

from absl import app, flags  # noqa: E402
from ml_collections import config_flags  # noqa: E402

import train  # noqa: E402
from utils import g3_env  # noqa: E402
from utils import g3_logmirror  # noqa: E402
from utils import logging_util  # noqa: E402
from utils.logging_util import log_for_0  # noqa: E402

logging_util.supress_checkpt_info()

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_bool('debug', False, 'Debugging mode.')
flags.DEFINE_string('mode', None, 'useless here')

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)


_ENV_CONFIG_OVERRIDES = {
    "load_from": ("LOAD_FROM", "load_from", "CONFIG_LOAD_FROM"),
    "wandb_resume_id": (
        "WANDB_RESUME_ID",
        "wandb_resume_id",
        "CONFIG_WANDB_RESUME_ID",
    ),
}


def _normalize_env_config_value(value):
    value = str(value).strip()
    if value.lower() in ("none", "null"):
        return ""
    return value


def _read_env_config_value(names):
    found = []
    for name in names:
        if name in os.environ:
            found.append((name, _normalize_env_config_value(os.environ[name])))
    if not found:
        return None
    first_name, first_value = found[0]
    for name, value in found[1:]:
        if value != first_value:
            raise ValueError(
                f"Conflicting environment overrides: {first_name}={first_value!r} "
                f"but {name}={value!r}"
            )
    return first_name, first_value


def _apply_env_config_overrides(config):
    updates = []
    with config.unlocked():
        for key, names in _ENV_CONFIG_OVERRIDES.items():
            env_value = _read_env_config_value(names)
            if env_value is None:
                continue
            env_name, value = env_value
            current = _normalize_env_config_value(getattr(config, key, ""))
            if current and current != value:
                raise ValueError(
                    f"Conflicting {key}: config has {current!r}, "
                    f"but environment {env_name}={value!r}"
                )
            if current != value:
                config[key] = value
                updates.append((key, env_name, value))
    for key, env_name, value in updates:
        shown_value = value if key != "wandb_resume_id" else (value or "<empty>")
        log_for_0("Applied config.%s from environment %s=%r", key, env_name, shown_value)


def get_available_bytes():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def _log_available_memory():
    """Report free memory; never block startup on it.

    The GCP path asserted >=100 GB free here, which is a property of that
    specific TPU-VM image, not of the program. Under Borg the memory limit is
    a job requirement and the right place to fail is the allocator; a hard
    assert in the binary just turns a schedulable job into a crash loop.
    """
    available = get_available_bytes()
    if available is None:
        _boot_log("MemAvailable not readable; skipping memory report")
    else:
        _boot_log("Memory available at startup: %.1f GB", available / 1e9)


def _init_distributed():
    """Bring up JAX's distributed runtime, only if nobody else will.

    google3's JAX (`learning/brain/research/jax/lib/jax_google.py`) starts the
    coordinator and connects the client itself, driven by the
    `--jax_controller_address` / `--jax_port` / `--jax_num_tasks` /
    `--jax_task_id` flags XManager passes. Calling
    `jax.distributed.initialize()` on top of that is wrong twice over: with the
    flags present it duplicates work JAX already did, and without them it
    raises `ValueError: coordinator_address should be defined`.
    """
    if os.environ.get("JAX_LLAVA_SKIP_DISTRIBUTED_INIT", "").lower() in ("1", "true", "yes"):
        _boot_log("Distributed init skipped by JAX_LLAVA_SKIP_DISTRIBUTED_INIT")
        return

    def _flag(name):
        try:
            return FLAGS[name].value if name in FLAGS else None
        except Exception:  # noqa: BLE001
            return None

    if _flag('jax_controller_address') or (_flag('jax_port') or 0):
        _boot_log(
            "JAX coordination flags present; leaving init to jax_google "
            "(controller=%r port=%r tasks=%r task_id=%r)",
            _flag('jax_controller_address'), _flag('jax_port'),
            _flag('jax_num_tasks'), _flag('jax_task_id'))
        return

    if g3_env.in_google3():
        # A google3 build with no coordination flags is a single-process run
        # (local debugging, or a one-task job). jax_google handles that itself.
        _boot_log("google3 build with no JAX coordination flags; single process")
        return

    jax.distributed.initialize()


def _assert_accelerator_backend():
    """Die immediately if this is an accelerator job that came up on CPU.

    `//third_party/py/jax` alone builds a CPU-ONLY binary; google3 registers
    the TPU backend factory with `fail_quietly=True`, so a missing
    `//learning/brain/research/jax:tpu_support` degrades to CPU with nothing
    louder than a `logger.info`. The job then holds every chip it was
    allocated at a 0.000 duty cycle until the pruner reclaims it. Failing in
    the first seconds instead costs nothing.
    """
    if os.environ.get("JAX_LLAVA_ALLOW_CPU_BACKEND", "").lower() in ("1", "true", "yes"):
        return
    if not os.environ.get("BORG_TASK_HANDLE") and not os.environ.get("XM_XID"):
        return  # not on Borg: no accelerator was promised
    if jax.default_backend() != "cpu":
        return
    raise RuntimeError(
        "FATAL: this looks like an accelerator job, but JAX came up on CPU "
        f"(devices={jax.local_devices()!r}). Add "
        '"//learning/brain/research/jax:tpu_support" to the binary\'s BUILD '
        "deps. Set JAX_LLAVA_ALLOW_CPU_BACKEND=1 to run on CPU deliberately."
    )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Mirror logs FIRST: TPU/topology bring-up is one of the most common places
  # to die, and a task that dies there leaves no retrievable log otherwise.
  bucket = os.environ.get("CHECKPOINT_BUCKET", "").strip()
  if bucket:
      mirrored = g3_logmirror.mirror_logs(bucket)
      if mirrored:
          _boot_log("[log-mirror] stdout/stderr -> %s", mirrored)
      else:
          _boot_log("[log-mirror] could not open a mirror under %s", bucket)

  _log_available_memory()
  _boot_log("webdataset provider: %s", _WDS_SOURCE)
  _boot_log("Borg cell=%r zone=%r", g3_env.borg_cell(),
            g3_env.infer_zone_from_environment())
  _init_distributed()

  _apply_env_config_overrides(FLAGS.config)
  log_for_0('JAX process: %d / %d', jax.process_index(), jax.process_count())
  log_for_0('JAX local devices: %r', jax.local_devices())
  _assert_accelerator_backend()
  log_for_0('FLAGS.config: \n{}'.format(FLAGS.config))

  try:
      if FLAGS.config.eval_only:
        train.just_evaluate(FLAGS.config, FLAGS.workdir)
      elif getattr(FLAGS.config, 'finetune', False):
        train.finetune(FLAGS.config, FLAGS.workdir)
      else:
        train.train_and_evaluate(FLAGS.config, FLAGS.workdir)
  finally:
      # The mirror flushes on a timer and on error tokens, but a clean exit or
      # an exception on the way out should not lose the last few lines.
      g3_logmirror.flush_logs()


if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir'])
    # `known_only=True`: XManager passes JAX coordination flags this binary
    # does not declare, and an undeclared flag is otherwise fatal at startup.
    _run_kwargs = dict(flags_parser=lambda a: flags.FLAGS(a, known_only=True))
    try:
        from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing
    except ImportError:
        app.run(main, **_run_kwargs)
    else:
        # Teaches multiprocessing to re-exec this Blaze binary (sys.executable
        # is None otherwise), which is what makes DataLoader workers possible;
        # then calls app.run itself. `fork` is not an option: torch's google3
        # multiprocessing asserts on it (go/python-tips/018) and forking after
        # JAX has started deadlocks.
        g3_multiprocessing.handle_main(main, **_run_kwargs)
