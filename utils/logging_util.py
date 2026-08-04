import logging as _logging
import re
import os
import time
import shutil

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
    def __init__(self):
        self._sum = None   # tree of numpy arrays (host)
        self._n = 0        # number of steps accumulated on *this host*

    @staticmethod
    def _mean_over_local_devices(x):
        """
        Bring one leaf to host and average over local device axis if present.
        This avoids keeping per-device values around on host.
        """
        # device_get blocks on the computation that produced x.
        a = np.asarray(jax.device_get(x))
        # Under sharded multi-device execution, metrics may still carry local
        # device axes depending on the caller.
        # If it's already scalar (0-D), leave unchanged.
        if a.ndim >= 1:  # treat leading axis as local device axis
            a = a.mean(axis=0)
        return a

    def update(self, metrics_step_tree):
        """
        Incorporate one step's metrics (per-replica JAX arrays) into the running sum.
        Call this once per training step.
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
        """
        if self._n == 0:
            return {}

        out = jax.tree.map(
            lambda s: float(np.asarray(s / self._n, dtype=np.float64).mean()),
            self._sum,
        )

        self._sum, self._n = None, 0
        return out

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
            log_for_0(f"[NOTE] Saving images locally, at step {step}")
            for k, v in image_dict.items():
                v = reduce_arr_func(v)
                os.makedirs(os.path.join(self.workdir, 'writed_images'), exist_ok=True)
                v.save(os.path.join(self.workdir, 'writed_images', f"step{step:07d}_{k}.png"))

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
