"""Auto-resume: the completeness rule and the decision it feeds.

These run off Borg, against a local temp directory, because the thing being
tested is a pure decision over a directory listing. `ckpt_util.FS` and the
`_open_text`/`_listdir` helpers are the only seam that touches a filesystem, so
they are the only thing stubbed.

The failing branches matter more than the passing one here: a guard whose
reject path has never run is trusted on faith. So every reason a checkpoint can
be judged incomplete gets its own case.
"""

import json
import os
import sys
import types

import pytest

# `ckpt_util` picks its filesystem at import time: google3's gfile inside a
# Blaze binary, `gcsfs` otherwise. This workstation env has neither, and the
# auto-resume code under test never calls gcsfs -- every access goes through
# `ckpt_util.FS`, which the fixture below replaces. So a stub is enough to get
# the module imported, and it is deliberately empty: if anything ever does
# reach for a real gcsfs method here, that is a bug worth an AttributeError
# rather than a silent pass.
sys.modules.setdefault('gcsfs', types.ModuleType('gcsfs'))
if not hasattr(sys.modules['gcsfs'], 'GCSFileSystem'):
    sys.modules['gcsfs'].GCSFileSystem = lambda *a, **k: None

from utils import ckpt_util


class _LocalFS:
    """The three `ckpt_util.FS` methods the auto-resume path uses, on POSIX."""

    def exists(self, path):
        return os.path.exists(path)

    def listdir(self, path):
        return tuple(sorted(os.listdir(path))) if os.path.isdir(path) else ()


@pytest.fixture(autouse=True)
def _local_fs(monkeypatch):
    monkeypatch.setattr(ckpt_util, 'FS', _LocalFS())
    # `_open_text` branches on in_google3(); force the POSIX branch.
    monkeypatch.setattr(ckpt_util.g3_env, 'in_google3', lambda: False)


def _write_checkpoint(root, step, *, committed=True, dataloader_state='full',
                      metadata='valid'):
    """One `checkpoint_<step>` directory, in a chosen state of completeness."""
    step_dir = os.path.join(root, f'checkpoint_{step}')
    os.makedirs(step_dir, exist_ok=True)
    if metadata != 'absent':
        payload = {
            'item_handlers': 'orbax...PyTreeCheckpointHandler',
            'init_timestamp_nsecs': 1785865624820660699,
        }
        if committed:
            payload['commit_timestamp_nsecs'] = 1785865785076221554
        with open(os.path.join(step_dir, '_CHECKPOINT_METADATA'), 'w') as handle:
            handle.write('{not json' if metadata == 'corrupt' else json.dumps(payload))
    if dataloader_state != 'absent':
        state_dir = os.path.join(step_dir, 'dataloader_state')
        os.makedirs(state_dir, exist_ok=True)
        if dataloader_state == 'full':
            with open(os.path.join(state_dir, 'process_00000.pkl'), 'wb') as handle:
                handle.write(b'\x80\x05')
    return step_dir


def test_picks_the_highest_complete_step(tmp_path):
    root = str(tmp_path)
    for step in (5, 10, 12):
        _write_checkpoint(root, step)
    assert ckpt_util.latest_complete_checkpoint(root) == os.path.join(
        root, 'checkpoint_12')


def test_highest_step_wins_not_newest_mtime(tmp_path):
    """Ordering is by step number, not by anything the filesystem reports."""
    root = str(tmp_path)
    _write_checkpoint(root, 12)
    _write_checkpoint(root, 5)  # written later, lower step
    assert ckpt_util.latest_complete_checkpoint(root).endswith('checkpoint_12')


def test_torn_write_is_skipped_and_the_previous_step_wins(tmp_path):
    """The whole point: a directory orbax never finalized must not be chosen."""
    root = str(tmp_path)
    _write_checkpoint(root, 10)
    _write_checkpoint(root, 15, committed=False)
    assert ckpt_util.latest_complete_checkpoint(root).endswith('checkpoint_10')


def test_missing_metadata_file_is_incomplete(tmp_path):
    root = str(tmp_path)
    _write_checkpoint(root, 10)
    _write_checkpoint(root, 15, metadata='absent')
    assert ckpt_util.latest_complete_checkpoint(root).endswith('checkpoint_10')


def test_corrupt_metadata_is_incomplete_not_an_exception(tmp_path):
    root = str(tmp_path)
    _write_checkpoint(root, 10)
    _write_checkpoint(root, 15, metadata='corrupt')
    assert ckpt_util.latest_complete_checkpoint(root).endswith('checkpoint_10')


def test_empty_dataloader_state_dir_is_incomplete(tmp_path):
    """Orbax finalized, but the pending dataloader rename had not happened."""
    root = str(tmp_path)
    _write_checkpoint(root, 10)
    _write_checkpoint(root, 15, dataloader_state='empty')
    assert ckpt_util.latest_complete_checkpoint(root).endswith('checkpoint_10')


def test_absent_dataloader_state_dir_is_still_complete(tmp_path):
    """A checkpoint from a non-stateful run must stay resumable."""
    root = str(tmp_path)
    _write_checkpoint(root, 15, dataloader_state='absent')
    assert ckpt_util.latest_complete_checkpoint(root).endswith('checkpoint_15')


def test_no_checkpoints_is_none_not_an_error(tmp_path):
    assert ckpt_util.latest_complete_checkpoint(str(tmp_path)) is None


def test_missing_root_is_none_not_an_error(tmp_path):
    assert ckpt_util.latest_complete_checkpoint(
        os.path.join(str(tmp_path), 'nope')) is None


def test_empty_root_argument_is_none(tmp_path):
    assert ckpt_util.latest_complete_checkpoint('') is None


def test_unrelated_directories_are_ignored(tmp_path):
    """`_pending_dataloader_state` and friends live beside the step dirs."""
    root = str(tmp_path)
    _write_checkpoint(root, 12)
    for junk in ('_pending_dataloader_state', '_dataloader_state_probe',
                 'checkpoint_', 'checkpoint_abc'):
        os.makedirs(os.path.join(root, junk), exist_ok=True)
    assert ckpt_util.latest_complete_checkpoint(root).endswith('checkpoint_12')


# --------------------------------------------------------------------------
# resolve_borg_autoresume: the decision, including every reason to decline.
# --------------------------------------------------------------------------

class _Config(dict):
    """The `config.get(...)` surface `resolve_borg_autoresume` uses."""


def _bucket_with_checkpoints(tmp_path, steps=(5, 10, 12)):
    bucket = str(tmp_path)
    root = os.path.join(bucket, ckpt_util.CNS_CKPT_SUBDIR)
    os.makedirs(root, exist_ok=True)
    for step in steps:
        _write_checkpoint(root, step)
    return bucket


def test_resolve_finds_the_newest_checkpoint(tmp_path, monkeypatch):
    bucket = _bucket_with_checkpoints(tmp_path)
    monkeypatch.setenv('CHECKPOINT_BUCKET', bucket)
    resume_from, why_not = ckpt_util.resolve_borg_autoresume(_Config())
    assert resume_from.endswith('checkpoints/checkpoint_12'), resume_from
    assert why_not == ''


def test_explicit_load_from_wins_and_disables_the_probe(tmp_path, monkeypatch):
    """The launcher depends on this: LOAD_FROM is an explicit user request."""
    bucket = _bucket_with_checkpoints(tmp_path)
    monkeypatch.setenv('CHECKPOINT_BUCKET', bucket)
    resume_from, why_not = ckpt_util.resolve_borg_autoresume(
        _Config(load_from='/cns/elsewhere/checkpoints/checkpoint_3'))
    assert resume_from is None
    assert 'explicit request wins' in why_not


def test_eval_only_does_not_resume(tmp_path, monkeypatch):
    bucket = _bucket_with_checkpoints(tmp_path)
    monkeypatch.setenv('CHECKPOINT_BUCKET', bucket)
    resume_from, why_not = ckpt_util.resolve_borg_autoresume(
        _Config(eval_only=True))
    assert resume_from is None
    assert 'eval_only' in why_not


def test_no_bucket_is_a_cold_start(monkeypatch):
    monkeypatch.delenv('CHECKPOINT_BUCKET', raising=False)
    resume_from, why_not = ckpt_util.resolve_borg_autoresume(_Config())
    assert resume_from is None
    assert 'CHECKPOINT_BUCKET' in why_not


def test_empty_bucket_is_a_cold_start(tmp_path, monkeypatch):
    """First attempt of a brand new experiment: prefix exists, nothing in it."""
    monkeypatch.setenv('CHECKPOINT_BUCKET', str(tmp_path))
    resume_from, why_not = ckpt_util.resolve_borg_autoresume(_Config())
    assert resume_from is None
    assert 'no complete checkpoint' in why_not


def test_only_a_torn_checkpoint_is_a_cold_start(tmp_path, monkeypatch):
    bucket = str(tmp_path)
    root = os.path.join(bucket, ckpt_util.CNS_CKPT_SUBDIR)
    os.makedirs(root)
    _write_checkpoint(root, 5, committed=False)
    monkeypatch.setenv('CHECKPOINT_BUCKET', bucket)
    resume_from, why_not = ckpt_util.resolve_borg_autoresume(_Config())
    assert resume_from is None
    assert 'no complete checkpoint' in why_not


def test_a_blank_load_from_does_not_count_as_explicit(tmp_path, monkeypatch):
    """configs/default.py seeds `load_from = ''`; that is not a user request."""
    bucket = _bucket_with_checkpoints(tmp_path)
    monkeypatch.setenv('CHECKPOINT_BUCKET', bucket)
    resume_from, _ = ckpt_util.resolve_borg_autoresume(_Config(load_from='  '))
    assert resume_from.endswith('checkpoint_12')


def test_probe_failure_degrades_to_cold_start(tmp_path, monkeypatch):
    """A diagnostic must never kill the thing it watches."""
    class _ExplodingFS(_LocalFS):
        def listdir(self, path):
            raise OSError('CNS is having a day')

    monkeypatch.setenv('CHECKPOINT_BUCKET', str(tmp_path))
    monkeypatch.setattr(ckpt_util, 'FS', _ExplodingFS())
    resume_from, why_not = ckpt_util.resolve_borg_autoresume(_Config())
    assert resume_from is None
    assert why_not
