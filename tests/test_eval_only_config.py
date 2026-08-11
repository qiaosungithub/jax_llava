"""What an eval-only run of `remote_run` would actually evaluate.

WHY THIS EXISTS: `train.just_evaluate` -> `_run_eval_from_checkpoint` reads
`config.training.final_eval_tasks` -- the TOP-LEVEL key. In the curriculum
config the 17 benchmarks live under `training.stage2`, and only
`_build_curriculum_stage_config` lifts a stage's keys to the top level. Nothing
does that on the eval-only path, so `final_eval_tasks` is `[]`, the
`if final_eval_tasks:` guard is False, and the job restores a 12 GiB checkpoint,
evaluates NOTHING, logs 'Eval Over.' and exits 0.

That is the worst shape of failure: a successful-looking run producing no
metrics, indistinguishable from a harvest bug. These tests pin the contract
that makes the final eval real, and they are written to fail loudly on the
config as it stands rather than to describe it.

Pure config assertions -- no JAX, no TPU, no checkpoint. Run:
    python3 tests/test_eval_only_config.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.load_config import get_config  # noqa: E402

# The 17 benchmarks the reproduction is scored on.
EXPECTED_TASKS = [
    'vqav2', 'mme', 'textvqa', 'pope', 'mmbench', 'knn_full', 'refcocog',
    'mmvp', 'vstar', 'ocrbench', 'countbenchqa', 'gqa', 'seed_bench',
    'cambrian_cvbench', 'vlms_are_blind', 'docvqa', 'realworldqa',
]


def _tasks(config):
    """Exactly what `_run_eval_from_checkpoint` reads."""
    return list(config.training.get('final_eval_tasks', []) or [])


def test_eval_only_config_evaluates_the_17_benchmarks():
    """The headline contract: an eval-only run must not evaluate nothing."""
    c = get_config('remote_run_eval')
    tasks = _tasks(c)
    assert tasks, (
        'training.final_eval_tasks is empty at the TOP level, which is the only '
        'place just_evaluate looks. The eval-only run would evaluate nothing '
        'and still exit 0.'
    )
    # SET equality, not list equality: the order is deliberately different from
    # the training config (see the next test) and is its own contract.
    missing = [t for t in EXPECTED_TASKS if t not in tasks]
    extra = [t for t in tasks if t not in EXPECTED_TASKS]
    assert not missing and not extra, f'missing={missing} extra={extra}'
    assert len(tasks) == len(set(tasks)) == 17, f'duplicate or short list: {tasks}'


def test_knn_full_runs_last_so_a_preemption_costs_only_knn():
    """Task order is the eval's only checkpointing mechanism.

    `evals/eval.py::run_eval_tasks` iterates the list in order and writes each
    task's scalars as soon as it finishes; the harvest reads those out of the
    rank logs. So whatever completed before a preemption is kept.

    knn_full at production settings (images_per_class None => all 1,281,167
    train images) costs ~80+ min against ~17 min for the other sixteen
    COMBINED. Anywhere but last, a preemption inside it also forfeits every
    task queued behind it.
    """
    tasks = _tasks(get_config('remote_run_eval'))
    assert tasks[-1] == 'knn_full', (
        f'knn_full must run last; the list ends with {tasks[-3:]}'
    )
    # vqav2 (7.1 min, the second most expensive) directly before it, so the
    # cheap sixteen are banked as early as possible.
    assert tasks[-2] == 'vqav2', f'expected vqav2 second-to-last, got {tasks[-2]}'
    cheap = tasks[:8]
    assert 'knn_full' not in cheap and 'vqav2' not in cheap and 'docvqa' not in cheap, (
        f'the three expensive tasks must not be in the first eight: {cheap}'
    )


def test_eval_only_flag_routes_to_just_evaluate():
    c = get_config('remote_run_eval')
    assert c.eval_only is True, 'eval_only must be True or main.py trains instead'


def test_load_from_is_left_for_the_launcher():
    """LOAD_FROM (env) must not collide with a value baked into the yaml.

    `main._apply_env_config_overrides` RAISES on a conflict, so a non-empty
    load_from here kills every task at startup the moment --load_from is passed.
    """
    c = get_config('remote_run_eval')
    assert not str(c.load_from or '').strip(), (
        f'load_from must be empty in the eval config; found {c.load_from!r}. '
        'It arrives as the LOAD_FROM env var and a mismatch raises at startup.'
    )


def test_eval_side_config_matches_the_training_run():
    """Anything an eval reads must equal what stage2 trained under.

    `_build_curriculum_stage_config` applied stage2's overrides on the training
    run; the eval config has no stage machinery, so each of these has to be
    correct at the top level in its own right. A silent mismatch here changes
    the number without failing.
    """
    train_c = get_config('remote_run')
    eval_c = get_config('remote_run_eval')
    stage2 = train_c.training.stage2
    # max_txt_len is read by 8 eval files for prompt truncation.
    assert int(eval_c.dataset.max_txt_len) == int(stage2.dataset['max_txt_len']), (
        f'dataset.max_txt_len {eval_c.dataset.max_txt_len} != stage2 '
        f"{stage2.dataset['max_txt_len']}"
    )
    for key in ('lm_backbone_str', 'vision_tower_str', 'projector_type',
                'vision_feature_layer', 'clip_input_format', 'prompt_causal'):
        assert eval_c.model[key] == train_c.model[key], (
            f'model.{key} differs: {eval_c.model[key]!r} vs {train_c.model[key]!r}'
        )
    assert int(eval_c.dataset.image_size) == int(train_c.dataset.image_size)
    assert eval_c.sharding == train_c.sharding, 'sharding must match the checkpoint'


def test_curriculum_is_off_so_num_steps_cannot_trigger_training():
    """Belt and braces: even if eval_only were dropped, this must not train."""
    c = get_config('remote_run_eval')
    assert not str(c.training.get('curriculum', '') or '').strip(), (
        'curriculum must be empty in the eval config'
    )


def test_the_training_config_is_the_reason_this_file_exists():
    """NEGATIVE CONTROL. Fails if someone 'fixes' the bug by editing the
    TRAINING config, which would change what the production run does."""
    c = get_config('remote_run')
    assert _tasks(c) == [], (
        'remote_run (training) now has top-level final_eval_tasks. That changes '
        'the training run. The eval-only fix belongs in its own config.'
    )
    assert len(list(c.training.stage2.get('final_eval_tasks') or [])) == 17


def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith('test_') and callable(f)]
    failures = []
    for name, fn in fns:
        try:
            fn()
            print(f'PASS {name}')
        except Exception:  # noqa: BLE001 -- report every failure, not the first
            failures.append(name)
            print(f'FAIL {name}\n{traceback.format_exc()}')
    print(f'\n{"FAILED: " + ", ".join(failures) if failures else "ALL TESTS PASSED"}'
          f'  ({len(fns) - len(failures)}/{len(fns)})')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(_run_all())
