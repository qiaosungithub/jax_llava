import os


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

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

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
