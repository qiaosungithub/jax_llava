"""MMStar adapter backed by the canonical one-benchmark-suite evaluator."""

MMSTAR_TASK_NAMES = frozenset({"mmstar", "mm_star", "mm-star"})


def _load_eval_impl():
    try:
        from one_benchmark_suite.visual_understanding.mmstar import get_eval_impl

        return get_eval_impl()
    except ModuleNotFoundError as exc:
        missing_root = (exc.name or "").split(".", 1)[0]
        if missing_root in {"one_benchmark_suite", "one_dataset_suite"}:
            raise ModuleNotFoundError(
                "MMStar evaluation requires both exact Git pins from "
                "requirements-eval.txt (one-benchmark-suite and one-dataset-suite)"
            ) from exc
        raise


def eval_mmstar(
    p_sample_step,
    run_p_sample_step,
    model,
    tokenizer,
    params,
    config,
):
    """Run MMStar without duplicating its artifact, prompt, or scoring contract."""
    return _load_eval_impl()(
        p_sample_step,
        run_p_sample_step,
        model,
        tokenizer,
        params,
        config,
    )


def preflight_mmstar_eval(config):
    """Load MMStar early iff any configured train stage will evaluate it."""
    training = config.training
    sections = [training]
    for stage_name in ("stage1", "stage2"):
        stage = training.get(stage_name, None)
        if stage is not None:
            sections.append(stage)

    for section in sections:
        for key in ("online_eval_tasks", "final_eval_tasks"):
            tasks = section.get(key, []) or []
            if any(str(task).strip().lower() in MMSTAR_TASK_NAMES for task in tasks):
                _load_eval_impl()
                return True
    return False


__all__ = ["MMSTAR_TASK_NAMES", "eval_mmstar", "preflight_mmstar_eval"]
