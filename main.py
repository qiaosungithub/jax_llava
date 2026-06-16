import os
import sys


SHARED_CODE_ROOT = "/kmh-nfs-ssd-us-mount/code/hanhong/shared"
if os.path.isdir(SHARED_CODE_ROOT) and SHARED_CODE_ROOT not in sys.path:
    sys.path.insert(0, SHARED_CODE_ROOT)


def get_available_bytes():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                parts = line.split()
                kb = int(parts[1])
                return kb * 1024
    raise RuntimeError("MemAvailable not found")


def assert_free_mem_at_least(bytes_required: int):
    available_bytes = get_available_bytes()

    assert available_bytes >= bytes_required, \
        f"Need ≥ {bytes_required/1e9:.1f} GB free, but only {available_bytes/1e9:.1f} GB available"

    print(f"OK: {available_bytes/1e9:.1f} GB available", flush=True)

# Check for at least 100 GB free memory
assert_free_mem_at_least(100 * 1024**3)

os.environ.update({
    "HF_TOKEN": open('/kmh-nfs-ssd-us-mount/code/siri/这个sqa一点用都没有').read().strip(),
})

print("Training starts. Good luck!", flush=True)

import jax

jax.distributed.initialize()

from absl import app, flags
from ml_collections import config_flags

import train
from utils import logging_util
from utils.logging_util import log_for_0

logging_util.supress_checkpt_info()

import warnings

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


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  _apply_env_config_overrides(FLAGS.config)
  log_for_0('JAX process: %d / %d', jax.process_index(), jax.process_count())
  log_for_0('JAX local devices: %r', jax.local_devices())
  log_for_0('FLAGS.config: \n{}'.format(FLAGS.config))
  
  if FLAGS.config.eval_only:
    train.just_evaluate(FLAGS.config, FLAGS.workdir)
  elif getattr(FLAGS.config, 'finetune', False):
    train.finetune(FLAGS.config, FLAGS.workdir)
  else:
    train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
