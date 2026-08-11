"""Load the eval config THROUGH config_flags, the way main.py does on Borg.

`tests/test_eval_only_config.py` calls `get_config()` directly. main.py does
not: it declares `--config` via ml_collections `config_flags`, which loads
`configs/load_config.py:<mode>` BY THE PATH GIVEN ON THE COMMAND LINE and wraps
any failure in a deferred error object that only raises at the first attribute
access -- far from the cause. That is the exact failure `_find_config_yml`'s
docstring describes (XID 277033539). So the last untested link is: does the
flag path resolve our new yaml inside a real binary?

This is a probe, not a unit test: it needs the Blaze runfiles tree.
"""

import os
import sys

# The real binary is main.py, which does this on line 49 before anything else,
# and the launcher additionally sets PYTHONPATH=<pkg_path>. Without one of the
# two, config_flags cannot `import configs` and fails with a DEFERRED error
# that only surfaces at the first attribute access -- which is precisely the
# trap this probe exists to check for, so it must not trip over its own
# version of it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl import app
from absl import flags
from ml_collections import config_flags

_CONFIG = config_flags.DEFINE_config_file(
    'config', None, 'configs/load_config.py:<mode>')

EXPECTED = 17


def main(argv):
  del argv
  cfg = _CONFIG.value          # a deferred error raises HERE, as on Borg
  tasks = list(cfg.training.get('final_eval_tasks', []) or [])
  print(f'eval_only               = {cfg.eval_only}')
  print(f'curriculum              = {cfg.training.get("curriculum")!r}')
  print(f'dataset.max_txt_len     = {cfg.dataset.max_txt_len}')
  print(f'sharding                = {cfg.sharding}')
  print(f'load_from               = {cfg.load_from!r}')
  print(f'final_eval_tasks ({len(tasks):2d})    = {tasks}')
  ok = (cfg.eval_only is True
        and len(tasks) == EXPECTED
        and tasks[-1] == 'knn_full'
        and int(cfg.dataset.max_txt_len) == 512
        and not str(cfg.load_from or '').strip())
  print('\nVERDICT:', 'OK' if ok else 'FAILED')
  sys.exit(0 if ok else 1)


if __name__ == '__main__':
  app.run(main)
