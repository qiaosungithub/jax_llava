import builtins
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import train as train_module
from configs.default import get_config
from configs.load_config import get_config as load_config
from evals import eval as eval_dispatch
from evals import eval_mmstar as eval_mmstar_module
from train import _build_pjit_fns
from utils.data_util import resolve_dataset_roots


_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mmstar_defaults_and_zone_local_root():
    config = get_config()

    assert config.eval.mmstar_num_samples == 1500
    assert config.eval.mmstar_num_workers == 0
    assert config.eval.mmstar_max_txt_len == 512
    assert config.eval.eval_tokens_mmstar == 8

    resolve_dataset_roots(config, "us-east5")
    assert config.eval.mmstar_root == (
        "gs://kmh-gcp-us-east5/data/vlm_eval_benchmarks/mmstar"
    )


def test_remote_stage2_runs_mmstar_with_contract_budget():
    config = load_config("remote_run")

    assert "mmstar" in config.training.stage2.final_eval_tasks
    assert config.eval.eval_tokens_mmstar == 8


def test_eval_dependencies_are_exact_git_pins():
    requirements = [
        line.strip()
        for line in (_REPO_ROOT / "requirements-eval.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert requirements == [
        "one-dataset-suite @ git+https://github.com/hevision/one-dataset-suite.git@"
        "d09aae6de23df600e8c1cf4615aea15c1f080af7",
        "one-benchmark-suite @ git+https://github.com/hevision/one-benchmark-suite.git@"
        "7c8eb0d37d11b0b4563830f77d809941ce3fbaf7",
    ]


def test_mmstar_preflight_is_conditional_and_scans_curriculum(monkeypatch):
    calls = []
    monkeypatch.setattr(
        eval_mmstar_module, "_load_eval_impl", lambda: calls.append("loaded")
    )

    assert eval_mmstar_module.preflight_mmstar_eval(get_config()) is False
    assert calls == []

    assert eval_mmstar_module.preflight_mmstar_eval(load_config("remote_run")) is True
    assert calls == ["loaded"]


def test_training_entrypoint_preflights_before_curriculum_runner(monkeypatch):
    events = []
    monkeypatch.setattr(
        train_module,
        "preflight_mmstar_eval",
        lambda config: events.append(("preflight", config)),
    )
    monkeypatch.setattr(
        train_module,
        "_train_llava_curriculum",
        lambda config, workdir: events.append(("runner", config, workdir)),
    )
    config = load_config("remote_run")

    train_module.train_and_evaluate(config, "/tmp/mmstar-preflight-test")

    assert events == [
        ("preflight", config),
        ("runner", config, "/tmp/mmstar-preflight-test"),
    ]


def test_mmstar_adapter_forwards_native_eval_pipeline(monkeypatch):
    calls = []

    def fake_impl(*args):
        calls.append(args)
        return 12.5, ["sample"], {"final_score": 12.5}

    monkeypatch.setattr(eval_mmstar_module, "_load_eval_impl", lambda: fake_impl)
    args = tuple(object() for _ in range(6))

    result = eval_mmstar_module.eval_mmstar(*args)

    assert calls == [args]
    assert result == (12.5, ["sample"], {"final_score": 12.5})


def test_mmstar_adapter_reports_missing_exact_eval_dependencies(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "one_benchmark_suite.visual_understanding.mmstar":

            def get_eval_impl():
                raise ModuleNotFoundError(
                    "missing dataset contract", name="one_dataset_suite"
                )

            return SimpleNamespace(get_eval_impl=get_eval_impl)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match="requirements-eval.txt"):
        eval_mmstar_module._load_eval_impl()


def test_mmstar_dispatch_logs_headline_and_breakdowns(monkeypatch):
    calls = []

    def fake_eval(*args):
        calls.append(args)
        return (
            42.0,
            ["sample"],
            {
                "num_samples": 1500,
                "by_category": {
                    "science & technology": {"accuracy": 41.0, "count": 250}
                },
                "by_l2_category": {
                    "image style & quality": {"accuracy": 43.0, "count": 83}
                },
            },
        )

    class Writer:
        def __init__(self):
            self.scalars = []
            self.texts = []

        def write_scalars(self, step, values):
            self.scalars.append((step, values))

        def write_texts(self, step, name, values):
            self.texts.append((step, name, values))

    monkeypatch.setattr(eval_dispatch, "eval_mmstar", fake_eval)
    monkeypatch.setattr(
        eval_dispatch, "_broadcast_string_from_source", lambda *_args: "run"
    )
    monkeypatch.setattr(eval_dispatch, "set_eval_result_context", lambda *_args: None)
    writer = Writer()
    config = get_config()

    eval_dispatch.run_eval_tasks(
        SimpleNamespace(params="params"),
        {"default": "default", "mmstar": "mmstar_sampler"},
        ["mm-star"],
        step=7,
        run_p_sample_step="run_step",
        model="model",
        tokenizer="tokenizer",
        config=config,
        writer=writer,
    )

    assert calls[0][0] == "mmstar_sampler"
    assert writer.scalars == [
        (
            7,
            {
                "mmstar_acc": 42.0,
                "mmstar_num_samples": 1500.0,
                "step": 7,
                "mmstar_science_technology_acc": 41.0,
                "mmstar_axis_image_style_quality_acc": 43.0,
            },
        )
    ]
    assert writer.texts == [(7, "mmstar_samples", ["sample"])]


def test_mmstar_sampler_uses_own_prompt_and_generation_budgets():
    config = get_config()
    config.training.batch_size = 1
    state = SimpleNamespace(params=object())
    mesh = SimpleNamespace(devices=np.empty((1,), dtype=object))

    def get_partition_spec(tree, _mode):
        if tree is state:
            return SimpleNamespace(params="state_params")
        if isinstance(tree, dict):
            return {key: value.shape for key, value in tree.items()}
        return tree.shape

    def pjit_compile(fn, **_kwargs):
        assert isinstance(fn, partial)
        return fn

    _, _, _, samplers, _ = _build_pjit_fns(
        config,
        model=object(),
        state=state,
        mesh_bundle=(mesh, get_partition_spec, None, object(), pjit_compile),
    )

    mmstar = samplers["mmstar"]
    assert mmstar.keywords["max_new_tokens"] == 8
    assert mmstar._batch_spec["input_ids"] == (1, 512)
