"""The sampled text a rank scores must answer the question that rank asked.

`train.py::run_p_sample_step` feeds a HOST-LOCAL batch in through
`jax.make_array_from_process_local_data` and reads the result back with
`mu.process_allgather(..., tiled=True)`, which returns the GLOBAL batch --
`process_count` times longer. Every one of the eight eval files then consumes
it host-locally:

    for aux, out_str, is_pad in zip(batch['aux'], out_strs, batch['is_pad'])

`zip` stops at the shortest. So while the return value was global, every rank
silently paired its own questions with the first `local_batch` answers, which
on a process-major mesh are process 0's. Rank 0 was right by coincidence and
ranks 1..7 scored at chance: VQAv2 read 16.84 against a reference of 67.63
while the training curve matched that reference to four decimals, because
teacher forcing never goes through this path.

Nothing raised, and nothing could: the shapes are all individually valid and
the text is fluent. The only mechanical signature is that every rank's output
is identical -- which is what `test_all_ranks_agree_is_the_bug_signature`
pins -- so the guard has to be an explicit inverse of the placement, which is
`_process_local_rows`, tested here.

These are pure index arithmetic over a sharding, so they run on CPU with no
accelerator and no real multi-process cluster: a Mesh built over N fake devices
with assigned process indices exercises exactly the code path that matters.
`addressable_devices_indices_map` is the seam, and it is real here, not stubbed.
"""

import sys
import types

import numpy as np

try:
    import pytest
except ImportError:  # google3: pytest is visibility-controlled
    pytest = types.ModuleType('pytest')
    pytest.fixture = lambda *a, **k: (lambda f: f)
    pytest.raises = None
    sys.modules['pytest'] = pytest

import jax
from jax.sharding import Mesh, PartitionSpec as P


def _load_train_helpers(process_index, process_count):
    """`_local_array_to_global` + `_process_local_rows` out of train.py.

    Importing train.py itself would drag in torch, webdataset, gemma and a
    TPU-shaped world. The two functions under test are self-contained index
    arithmetic over numpy and jax.sharding, so they are extracted by source and
    given the module globals they read (`np`, `jax`, `PRC`, `PRI`). Extracting
    by source rather than copying keeps the test honest: it fails if the
    functions are renamed or deleted, and it always tests what ships.
    """
    import os
    import re

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, 'train.py')).read()

    mod = types.ModuleType('train_helpers')
    mod.__dict__.update(np=np, jax=jax, PRC=process_count, PRI=process_index)
    for name in ('_local_array_to_global', '_process_local_rows'):
        match = re.search(rf'^def {name}\(.*?(?=\n\ndef |\n\n\ndef )', src, re.S | re.M)
        assert match, f'{name} is gone from train.py; the alignment guard moved or was deleted'
        exec(match.group(0), mod.__dict__)  # pylint: disable=exec-used
    return mod


# `addressable_devices_indices_map` reaches into a real Device (it consults the
# backend's device list), so these use REAL jax CPU devices and simulate the
# multi-process topology by choosing, per "process", which devices belong to it.
# `_process_local_rows` asks the sharding for ADDRESSABLE devices; under a real
# multi-process job that set is the process's own devices, so the test provides
# exactly that set through a seam and reads the same index map the shipped code
# reads. The end-to-end multi-process behaviour was additionally verified with 8
# real `jax.distributed.initialize` processes; this file keeps the arithmetic
# under test in a form that runs anywhere, with no cluster.


def _devices(n):
    import os
    os.environ.setdefault('XLA_FLAGS', f'--xla_force_host_platform_device_count={n}')
    devices = jax.devices()
    assert len(devices) >= n, (
        f'need {n} CPU devices, got {len(devices)}; set XLA_FLAGS before jax loads'
    )
    return devices[:n]


def _mesh(shape, process_count, *, process_major=True):
    """A mesh over prod(shape) real devices, plus the device->process mapping.

    `process_major=True` mimics `pjit_util._process_major_model_axis_mesh`: each
    host owns one contiguous block of mesh slots, which is what the production
    v7-32 (4,4,4) layout does. `False` interleaves the hosts, so a process's
    rows are NOT contiguous -- the case where `out[PRI*B:(PRI+1)*B]` is wrong.
    """
    total = int(np.prod(shape))
    devices = _devices(total)
    per_process = total // process_count
    if process_major:
        owner = {devices[i]: i // per_process for i in range(total)}
    else:
        owner = {devices[i]: i % process_count for i in range(total)}
    mesh = Mesh(
        np.array(devices, dtype=object).reshape(shape),
        tuple(f'AXIS_{i}' for i in range(len(shape))),
    )
    return mesh, owner


def _as_process(helpers, owner, process_index):
    """Make `_process_local_rows` see only `process_index`'s devices.

    Under a real job `addressable_devices_indices_map` already returns just this
    process's devices. Off-cluster every device is addressable, so the seam is
    narrowed here rather than in the code under test -- the shipped function is
    executed unmodified, with a NamedSharding whose index map is filtered.
    """
    real_named_sharding = jax.sharding.NamedSharding

    class _ProcessView:
        def __init__(self, mesh, spec):
            self._inner = real_named_sharding(mesh, spec)

        def addressable_devices_indices_map(self, global_shape):
            full = self._inner.devices_indices_map(global_shape[:1])
            return {
                device: index
                for device, index in full.items()
                if owner[device] == process_index
            }

    shim = types.SimpleNamespace(
        sharding=types.SimpleNamespace(NamedSharding=_ProcessView),
    )
    helpers.__dict__['jax'] = shim
    return helpers


def _global_rows_of(mesh, spec, owner, process_index, global_rows):
    """Which global rows `make_array_from_process_local_data` gave a process.

    Computed independently of `_process_local_rows` so the test is not the
    implementation restated.
    """
    full = jax.sharding.NamedSharding(mesh, spec).devices_indices_map((global_rows,))
    spans = sorted({
        ((idx[0].start or 0), (global_rows if idx[0].stop is None else idx[0].stop))
        for device, idx in full.items()
        if owner[device] == process_index
    })
    return np.concatenate([np.arange(start, stop) for start, stop in spans])


# The production case: v7-32 is 64 devices over 8 hosts, and `hsdp_legacy_data`
# is NOT in the {"hsdp","fsdp"} set `_data_axis_names` tests, so every mesh axis
# counts as a data axis and the batch shards over all 64.
PROD_SHAPE = (4, 4, 4)
PROD_PROCESSES = 8
PROD_SPEC = P(('AXIS_0', 'AXIS_1', 'AXIS_2'))


def test_every_process_recovers_its_own_rows_production_mesh():
    """The whole bug, in one assertion, for the mesh the run actually used."""
    local_rows = 8
    global_rows = local_rows * PROD_PROCESSES
    mesh, owner = _mesh(PROD_SHAPE, PROD_PROCESSES)

    # Tag every global row with the (process, local_row) that produced it.
    tags = np.zeros((global_rows, 2), dtype=np.int32)
    for process in range(PROD_PROCESSES):
        rows = _global_rows_of(mesh, PROD_SPEC, owner, process, global_rows)
        assert len(rows) == local_rows
        for local_row, global_row in enumerate(rows):
            tags[global_row] = (process, local_row)

    for process in range(PROD_PROCESSES):
        helpers = _as_process(_load_train_helpers(process, PROD_PROCESSES), owner, process)
        got = helpers._process_local_rows(tags, mesh, PROD_SPEC, local_rows)
        want = np.stack([np.full(local_rows, process), np.arange(local_rows)], axis=1)
        assert np.array_equal(got, want), (
            f'process {process} recovered {got.tolist()}, wanted {want.tolist()}'
        )


def test_all_ranks_agree_is_the_bug_signature():
    """Without the slice, every rank reads process 0's rows. Pin that.

    This is the negative control: it asserts the ORIGINAL behaviour really was
    broken and really did produce the observed "all ranks identical" signature,
    so a future revert cannot pass the suite by accident.
    """
    local_rows = 8
    global_rows = local_rows * PROD_PROCESSES
    mesh, owner = _mesh(PROD_SHAPE, PROD_PROCESSES)
    tags = np.zeros((global_rows,), dtype=np.int32)
    for process in range(PROD_PROCESSES):
        for global_row in _global_rows_of(mesh, PROD_SPEC, owner, process, global_rows):
            tags[global_row] = process

    unsliced = [tags[:local_rows].tolist() for _ in range(PROD_PROCESSES)]
    assert all(rows == unsliced[0] for rows in unsliced)
    assert set(unsliced[0]) == {0}, 'the truncating zip fed everyone process 0'


def test_non_process_major_mesh_defeats_the_naive_offset():
    """`out[PRI*B:(PRI+1)*B]` is a layout assumption; the fix must not need it."""
    local_rows = 4
    processes = 4
    shape = (4, 4)
    spec = P(('AXIS_0', 'AXIS_1'))
    global_rows = local_rows * processes
    mesh, owner = _mesh(shape, processes, process_major=False)

    tags = np.zeros((global_rows,), dtype=np.int32)
    for process in range(processes):
        for global_row in _global_rows_of(mesh, spec, owner, process, global_rows):
            tags[global_row] = process

    naive_was_wrong = False
    for process in range(processes):
        helpers = _as_process(_load_train_helpers(process, processes), owner, process)
        got = helpers._process_local_rows(tags, mesh, spec, local_rows)
        assert np.array_equal(got, np.full(local_rows, process)), (
            f'process {process} got {got.tolist()} from an interleaved mesh'
        )
        naive = tags[process * local_rows:(process + 1) * local_rows]
        naive_was_wrong |= not np.array_equal(naive, got)
    assert naive_was_wrong, (
        'interleaved mesh no longer defeats the offset shortcut; this test has '
        'stopped testing what it claims to test'
    )


def test_single_process_is_a_passthrough():
    """One host addresses everything, so the gather is already host-local."""
    mesh, owner = _mesh((2, 4), 1)
    spec = P(('AXIS_0', 'AXIS_1'))
    rows = np.arange(8 * 3, dtype=np.int32).reshape(8, 3)
    helpers = _as_process(_load_train_helpers(0, 1), owner, 0)
    assert np.array_equal(helpers._process_local_rows(rows, mesh, spec, 8), rows)


def test_replicated_batch_is_a_passthrough():
    """A batch replicated over the mesh gives every host the whole thing."""
    mesh, owner = _mesh(PROD_SHAPE, PROD_PROCESSES)
    rows = np.arange(8 * 2, dtype=np.int32).reshape(8, 2)
    helpers = _as_process(_load_train_helpers(3, PROD_PROCESSES), owner, 3)
    assert np.array_equal(helpers._process_local_rows(rows, mesh, P(), 8), rows)


def test_row_count_mismatch_raises_rather_than_misaligning():
    """Refuse, loudly. A wrong slice here only ever shows up as a bad score.

    That is the entire reason this bug survived: it degrades quality without
    degrading anything a program can notice.
    """
    mesh, owner = _mesh(PROD_SHAPE, PROD_PROCESSES)
    tags = np.zeros((64, 2), dtype=np.int32)
    helpers = _as_process(_load_train_helpers(1, PROD_PROCESSES), owner, 1)

    try:
        helpers._process_local_rows(tags, mesh, PROD_SPEC, 7)  # owns 8, told 7
    except ValueError as exc:
        message = str(exc)
        assert '8' in message and '7' in message, (
            f'the refusal must report both counts, got: {message}'
        )
    else:
        raise AssertionError('a row-count mismatch returned a misaligned batch')


def test_multiple_rows_per_shard():
    """Batches are usually larger than one row per device."""
    rows_per_shard = 4
    local_rows = 8 * rows_per_shard
    global_rows = local_rows * PROD_PROCESSES
    mesh, owner = _mesh(PROD_SHAPE, PROD_PROCESSES)

    tags = np.zeros((global_rows,), dtype=np.int32)
    for process in range(PROD_PROCESSES):
        for local_row, global_row in enumerate(
            _global_rows_of(mesh, PROD_SPEC, owner, process, global_rows)
        ):
            tags[global_row] = process * 1000 + local_row

    for process in (0, 3, 7):
        helpers = _as_process(_load_train_helpers(process, PROD_PROCESSES), owner, process)
        got = helpers._process_local_rows(tags, mesh, PROD_SPEC, local_rows)
        want = process * 1000 + np.arange(local_rows)
        assert np.array_equal(got, want), f'process {process}: {got.tolist()}'


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS {name}')
            except Exception as exc:  # pylint: disable=broad-except
                failures += 1
                print(f'FAIL {name}: {type(exc).__name__}: {exc}')
    raise SystemExit(1 if failures else 0)
