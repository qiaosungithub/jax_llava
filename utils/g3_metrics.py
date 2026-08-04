"""Durable metrics for a google3/Borg run, where wandb records nothing.

`import wandb` inside google3 resolves to `//third_party/py/scamper:wandb_mock`,
which implements `init/log/finish/Table/plot/Video/run` and **stores nothing** --
`log()` is a `logging.debug`. jax_llava wraps every wandb call in try/except and
defaults `use_wandb` to False, so it degrades safely, but the consequence is
that a Borg run's only record of its own loss curve is text in a stderr stream
that is garbage-collected with the task.

This routes the same scalars into Datatables, which backs the two UIs people
actually read:

    http://datatable/xid/<XID>/data     the raw scalar table
    http://flatboard/xid/<XID>          the curves

Written through `clu.metric_writers` rather than the datatables client
directly: CLU is `//visibility:public` while the datatables client is
allowlisted and only links from //experimental/ because
`--check_visibility_for_experimental` defaults to false.

Everything here is best-effort by construction. Telemetry that can raise into a
training loop has negative value, so every entry point swallows its own
failures and the run continues without metrics.
"""

import os

_WRITER = None
_WRITER_TRIED = False
# Rows buffered since the last flush. CLU's destructor CANCELS its background
# thread rather than draining it, so an unflushed tail is lost when a job dies
# -- and jax_llava logs every `log_per_step` (100) steps, so a whole stage can
# sit inside one buffer. Flush on a small count rather than trusting shutdown.
_ROWS_SINCE_FLUSH = 0
_FLUSH_EVERY = 20


def _log(message, *args):
    from utils.logging_util import log_for_0
    log_for_0(message, *args)


def enabled():
    """False when explicitly disabled or when we are not in google3."""
    if os.environ.get("JAX_LLAVA_DISABLE_DATATABLES", "").strip() not in ("", "0", "false"):
        return False
    from utils import g3_env
    return g3_env.in_google3()


def _init_writer():
    """Build the writer once. Returns None if unavailable; never raises."""
    global _WRITER, _WRITER_TRIED
    if _WRITER_TRIED:
        return _WRITER
    _WRITER_TRIED = True
    if not enabled():
        return None
    try:
        import jax
        from clu import metric_writers
    except ImportError as exc:  # noqa: BLE001
        _log("clu.metric_writers unavailable (%s); metrics stay log-only", exc)
        return None
    try:
        _WRITER = metric_writers.create_default_writer(
            None,
            # Hosts 1..N construct no writer and issue no RPC, so the tasks of
            # one work unit cannot collide on the (wid, step) key.
            just_logging=jax.process_index() != 0,
            # MUST be explicit. The default (None) ACL-gates on
            # mdb/datatables-users and, for a non-member, writes NOTHING at all
            # with no error -- an empty chart page rather than a failure.
            write_to_datatable=True,
            # XM Measurements is deprecated and silently drops anything past
            # 1 point/sec/label.
            write_to_xm_measurements=False,
            asynchronous=True,
        )
        # create_default_writer() reinstalls absl logging handlers bound to the
        # ORIGINAL stderr, which unhooks the remote log mirror main() set up.
        # Everything after this point would otherwise never reach
        # $CHECKPOINT_BUCKET/logs/ -- the only durable log a Borg task leaves.
        from utils import g3_logmirror
        g3_logmirror._reattach_absl_handler()  # pylint: disable=protected-access
        xid = os.environ.get("XM_XID", "<XID>")
        _log("Datatables metric writer ready: curves at http://flatboard/xid/%s "
             "(raw: http://datatable/xid/%s/data)", xid, xid)
    except Exception as exc:  # noqa: BLE001 - telemetry must not kill training
        _log("Could not start the Datatables writer (%s); metrics stay log-only", exc)
        _WRITER = None
    return _WRITER


def _flatten_scalars(metrics, prefix=""):
    """Keep the float-able leaves, flattening nested dicts with '/'.

    Non-scalars are DROPPED, by design -- a scalar table cannot hold an array
    or a figure. That silence is also the trap: a key that stops being a float
    stops being a column, with no error anywhere.
    """
    out = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten_scalars(value, prefix=f"{name}/"))
            continue
        if isinstance(value, bool) or value is None:
            continue
        try:
            import numpy as np
            array = np.asarray(value)
            if array.size != 1:
                continue
            out[name] = float(array.reshape(()).item())
        except Exception:  # noqa: BLE001 - not a scalar; drop it
            continue
    return out


def log_metrics(metrics, step):
    """Write one row of scalars. Safe to call from any process, any time."""
    global _ROWS_SINCE_FLUSH
    writer = _init_writer()
    if writer is None:
        return
    scalars = _flatten_scalars(dict(metrics))
    if not scalars:
        return
    try:
        writer.write_scalars(int(step), scalars)
        _ROWS_SINCE_FLUSH += 1
        if _ROWS_SINCE_FLUSH >= _FLUSH_EVERY:
            writer.flush()
            _ROWS_SINCE_FLUSH = 0
    except Exception as exc:  # noqa: BLE001
        _log("Metric write failed at step %s (%s); continuing", step, exc)


def flush():
    """Drain the buffer. Call before exiting and after the last step."""
    global _ROWS_SINCE_FLUSH
    if _WRITER is None:
        return
    try:
        _WRITER.flush()
        _ROWS_SINCE_FLUSH = 0
    except Exception as exc:  # noqa: BLE001
        _log("Metric flush failed (%s); continuing", exc)


def close():
    """Flush and close. Never raises."""
    flush()
    if _WRITER is None:
        return
    try:
        _WRITER.close()
    except Exception:  # noqa: BLE001
        pass
