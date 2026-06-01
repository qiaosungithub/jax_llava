import time
import uuid

import jax
import numpy as np
import jax.experimental.multihost_utils as mu

from evals.eval_vqav2 import eval_vqav2
from evals.eval_textvqa import eval_textvqa
from evals.eval_mme import eval_mme
from evals.eval_pope import eval_pope
from evals.eval_mmbench import eval_mmbench
from evals.eval_imagenet_knn import eval_imagenet_knn
from evals.eval_refcocog import eval_refcocog
from evals.eval_pixelbench import eval_pixelbench
from evals.eval_vlm_benchmarks import (
    eval_gqa,
    eval_vizwiz,
    eval_scienceqa_img,
    eval_seed_bench,
)
from utils.logging_util import log_for_0
from utils.eval_io_util import set_eval_result_context
from utils import vis_util


_RUN_ID_BUF_SIZE = 128


def _knn_scalars(metric_name, result, suffix):
    """Keep the legacy metric name as raw KNN and log whitened explicitly."""
    if isinstance(result, dict):
        scalars = {}
        if "raw" in result:
            scalars[f"{metric_name}{suffix}"] = result["raw"]
            scalars[f"{metric_name}_raw{suffix}"] = result["raw"]
        if "pca_whitened" in result:
            scalars[f"{metric_name}_pca_whitened{suffix}"] = result["pca_whitened"]
        if not scalars and result:
            first_key = next(iter(result))
            scalars[f"{metric_name}{suffix}"] = result[first_key]
        return scalars
    return {f"{metric_name}{suffix}": result}


def _broadcast_string_from_source(value, is_source):
    data = value.encode("utf-8") if is_source else b""
    if len(data) >= _RUN_ID_BUF_SIZE:
        raise ValueError(f"eval run id is too long: {value}")
    buf = np.zeros((_RUN_ID_BUF_SIZE,), dtype=np.uint8)
    if is_source:
        buf[:len(data)] = np.frombuffer(data, dtype=np.uint8)
    out = np.asarray(mu.broadcast_one_to_all(buf, is_source=is_source))
    zero = np.where(out == 0)[0]
    end = int(zero[0]) if len(zero) else len(out)
    return bytes(out[:end].tolist()).decode("utf-8")


def run_eval_tasks(
    state,
    p_sample_fn,
    eval_tasks,
    *,
    step,
    run_p_sample_step,
    model,
    tokenizer,
    config,
    writer,
    p_sample_step_mmbench=None,
    task_suffix="",
    is_online_eval=False,
    extra_args=None,
):
    """Run eval tasks sequentially and log each task immediately."""
    if not eval_tasks:
        return

    params = state.params
    suffix = task_suffix or ""
    source = jax.process_index() == 0
    local_run_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}" if source else ""
    eval_run_id = _broadcast_string_from_source(local_run_id, source)
    result_suffix = "online" if is_online_eval and not suffix else (suffix or "main")
    set_eval_result_context(config, int(step), eval_run_id, result_suffix)

    for task in eval_tasks:
        t = str(task).strip().lower()
        if not t:
            continue
        # MMBench has a separate compiled sampler with a longer prompt length.
        # Other short-answer tasks use the regular sampler so their prompt shape
        # stays tied to config.dataset.max_txt_len.
        short_sample_fn = p_sample_fn

        if t == "knn_partial":
            knn_data_dir = None
            if extra_args:
                knn_data_dir = extra_args.get("knn_imagenet_data_dir") or extra_args.get("knn_imagenet_root")
            if knn_data_dir is None:
                log_for_0("Skip knn_partial: ImageNet TFDS data_dir unavailable")
                continue
            knn_result = eval_imagenet_knn(
                params,
                model,
                config,
                knn_data_dir,
                images_per_class=config.eval.get("knn_images_per_class", 128),
                seed=config.eval.get("knn_seed", 42),
                k=config.eval.get("knn_k", 20),
                temperature=config.eval.get("knn_temperature", 0.07),
                batch_size=config.eval.get("knn_batch_size", 256),
                num_workers=config.eval.get("knn_num_workers", 4),
                val_examples=config.eval.get("knn_val_examples", None),
            )
            writer.write_scalars(step, {**_knn_scalars("knn_partial_acc", knn_result, suffix), "step": step})
            mu.sync_global_devices("knn_partial")
            continue

        if t == "knn_full":
            knn_data_dir = None
            if extra_args:
                knn_data_dir = extra_args.get("knn_imagenet_data_dir") or extra_args.get("knn_imagenet_root")
            if knn_data_dir is None:
                log_for_0("Skip knn_full: ImageNet TFDS data_dir unavailable")
                continue
            knn_result = eval_imagenet_knn(
                params,
                model,
                config,
                knn_data_dir,
                images_per_class=None,
                seed=config.eval.get("knn_seed", 42),
                k=config.eval.get("knn_k", 20),
                temperature=config.eval.get("knn_temperature", 0.07),
                batch_size=config.eval.get("knn_batch_size", 256),
                num_workers=config.eval.get("knn_num_workers", 4),
                val_examples=config.eval.get("knn_val_examples", None),
            )
            writer.write_scalars(step, {**_knn_scalars("knn_full_acc", knn_result, suffix), "step": step})
            mu.sync_global_devices("knn_full")
            continue

        if t == "recon_psnr":
            if not extra_args:
                log_for_0("Skip recon_psnr: missing extra_args")
                continue
            p_recon_steps = extra_args.get("p_recon_steps")
            vis_batch = extra_args.get("vis_batch")
            patch_size = extra_args.get("patch_size")
            image_size = extra_args.get("image_size")
            run_eval_recon_psnr = extra_args.get("run_eval_recon_psnr")
            if not p_recon_steps:
                log_for_0("Skip recon_psnr: p_recon_steps is empty")
                continue
            psnr_results, recon_vis = run_eval_recon_psnr(
                p_recon_steps,
                params,
                vis_batch["pixel_values"],
                patch_size,
                image_size,
            )
            psnr_scalars = {f"recon_psnr_k{k}{suffix}": v for k, v in psnr_results.items()}
            psnr_scalars["step"] = step
            writer.write_scalars(step, psnr_scalars)
            orig_vis = jax.device_get(
                vis_batch["pixel_values"].reshape(-1, *vis_batch["pixel_values"].shape[2:])
            )
            for k, imgs in recon_vis.items():
                n = imgs.shape[0]
                combined = np.empty((n * 2, *imgs.shape[1:]), dtype=imgs.dtype)
                combined[0::2] = orig_vis[:n]
                combined[1::2] = imgs
                grid = vis_util.make_grid_visualization(combined, grid=n, to_uint8=True, is_pt=False)
                writer.write_images(step, {f"recon_vis_k{k}{suffix}": grid})
            continue

        if t == "vqav2":
            log_for_0(f"Evaluating VQAv2 at step {step}...")
            acc, sample_outputs, _ = eval_vqav2(
                short_sample_fn, run_p_sample_step, model, tokenizer, params, config
            )
            log_for_0(f"VQAv2 accuracy: {acc:.2f}%")
            writer.write_scalars(step, {f"vqav2_acc{suffix}": acc, "step": step})
            if sample_outputs:
                writer.write_texts(step, f"vqav2_samples{suffix}", sample_outputs)
            log_for_0("VQAv2 evaluation finished.")
            continue

        if t == "mme":
            log_for_0(f"Evaluating MME at step {step}...")
            mme_p, mme_s, sample_outputs, _ = eval_mme(
                short_sample_fn, run_p_sample_step, model, tokenizer, params, config
            )
            log_for_0(f"MME-P: {mme_p:.2f}")
            log_for_0(f"MME-S: {mme_s:.2f}")
            writer.write_scalars(
                step,
                {
                    f"MME-P{suffix}": mme_p,
                    f"MME-S{suffix}": mme_s,
                    "step": step,
                },
            )
            if sample_outputs:
                writer.write_texts(step, f"mme_samples{suffix}", sample_outputs)
            log_for_0("MME evaluation finished.")
            continue

        if t == "textvqa":
            log_for_0(f"Evaluating TextVQA at step {step}...")
            acc, sample_outputs, _ = eval_textvqa(
                short_sample_fn, run_p_sample_step, model, tokenizer, params, config
            )
            log_for_0(f"TextVQA accuracy: {acc:.2f}%")
            writer.write_scalars(step, {f"textvqa_acc{suffix}": acc, "step": step})
            if sample_outputs:
                writer.write_texts(step, f"textvqa_samples{suffix}", sample_outputs)
            log_for_0("TextVQA evaluation finished.")
            continue

        if t == "gqa":
            log_for_0(f"Evaluating GQA at step {step}...")
            acc, sample_outputs, _ = eval_gqa(
                short_sample_fn, run_p_sample_step, model, tokenizer, params, config
            )
            log_for_0(f"GQA accuracy: {acc:.2f}%")
            writer.write_scalars(step, {f"gqa_acc{suffix}": acc, "step": step})
            if sample_outputs:
                writer.write_texts(step, f"gqa_samples{suffix}", sample_outputs)
            log_for_0("GQA evaluation finished.")
            continue

        if t == "vizwiz":
            log_for_0(f"Evaluating VisWiz at step {step}...")
            acc, sample_outputs, metric_dict = eval_vizwiz(
                short_sample_fn, run_p_sample_step, model, tokenizer, params, config
            )
            log_for_0(f"VisWiz accuracy: {acc:.2f}%")
            scalar_dict = {f"vizwiz_acc{suffix}": acc, "step": step}
            if metric_dict.get("num_without_gt", 0):
                scalar_dict[f"vizwiz_num_without_gt{suffix}"] = float(metric_dict["num_without_gt"])
            writer.write_scalars(step, scalar_dict)
            if sample_outputs:
                writer.write_texts(step, f"vizwiz_samples{suffix}", sample_outputs)
            log_for_0("VisWiz evaluation finished.")
            continue

        if t in {"scienceqa", "scienceqa_img", "scienceqa-img", "sciqa", "sciqa_img"}:
            log_for_0(f"Evaluating ScienceQA-IMG at step {step}...")
            acc, sample_outputs, _ = eval_scienceqa_img(
                short_sample_fn, run_p_sample_step, model, tokenizer, params, config
            )
            log_for_0(f"ScienceQA-IMG accuracy: {acc:.2f}%")
            writer.write_scalars(step, {f"scienceqa_img_acc{suffix}": acc, "step": step})
            if sample_outputs:
                writer.write_texts(step, f"scienceqa_img_samples{suffix}", sample_outputs)
            log_for_0("ScienceQA-IMG evaluation finished.")
            continue

        if t in {"seed", "seed_bench", "seed-bench", "seed_bench_image", "seed-bench-image"}:
            log_for_0(f"Evaluating SEED-Bench at step {step}...")
            acc, sample_outputs, metric_dict = eval_seed_bench(
                short_sample_fn, run_p_sample_step, model, tokenizer, params, config
            )
            log_for_0(f"SEED-Bench accuracy: {acc:.2f}%")
            scalar_dict = {f"seed_bench_acc{suffix}": acc, "step": step}
            for qtype, metrics in metric_dict.get("by_question_type", {}).items():
                safe = qtype.lower().replace(" ", "_").replace("/", "_")
                scalar_dict[f"seed_bench_{safe}_acc{suffix}"] = float(metrics["accuracy"])
            writer.write_scalars(step, scalar_dict)
            if sample_outputs:
                writer.write_texts(step, f"seed_bench_samples{suffix}", sample_outputs)
            log_for_0("SEED-Bench evaluation finished.")
            continue

        if t == "pope":
            log_for_0(f"Evaluating POPE at step {step}...")
            pope_f1, sample_outputs, metric_dict = eval_pope(
                short_sample_fn, run_p_sample_step, model, tokenizer, params, config
            )
            log_for_0(f"POPE macro F1: {pope_f1:.2f}%")
            split_metrics = metric_dict.get("splits", {})
            scalar_dict = {f"pope_f1_macro{suffix}": pope_f1, "step": step}
            for split in ["random", "popular", "adversarial"]:
                if split in split_metrics:
                    scalar_dict[f"pope_{split}_f1{suffix}"] = float(
                        split_metrics[split]["f1"] * 100.0
                    )
            writer.write_scalars(step, scalar_dict)
            if sample_outputs:
                writer.write_texts(step, f"pope_samples{suffix}", sample_outputs)
            log_for_0("POPE evaluation finished.")
            continue

        if t == "mmbench":
            if p_sample_step_mmbench is None:
                log_for_0("Skip MMBench: p_sample_step_mmbench is None")
                continue
            log_for_0(f"Evaluating MMBench at step {step}...")
            acc, sample_outputs, _ = eval_mmbench(
                p_sample_step_mmbench,
                run_p_sample_step,
                model,
                tokenizer,
                params,
                config,
            )
            log_for_0(f"MMBench accuracy: {acc:.2f}%")
            writer.write_scalars(step, {f"mmbench_acc{suffix}": acc, "step": step})
            if sample_outputs:
                writer.write_texts(step, f"mmbench_samples{suffix}", sample_outputs)
            log_for_0("MMBench evaluation finished.")
            continue

        if t == "refcocog":
            log_for_0(f"Evaluating RefCOCOg at step {step}...")
            acc, sample_outputs, metric_dict = eval_refcocog(
                short_sample_fn, run_p_sample_step, model, tokenizer, params, config
            )
            scalar_dict = {f"refcocog_acc{suffix}": acc, "step": step}
            if isinstance(metric_dict, dict) and "miou" in metric_dict:
                scalar_dict[f"refcocog_miou{suffix}"] = float(metric_dict["miou"])
            writer.write_scalars(step, scalar_dict)
            if sample_outputs:
                writer.write_texts(step, f"refcocog_samples{suffix}", sample_outputs)
            if isinstance(metric_dict, dict) and metric_dict.get("vis_image") is not None:
                writer.write_images(step, {f"refcocog_vis{suffix}": metric_dict["vis_image"]})
            log_for_0("RefCOCOg evaluation finished.")
            continue

        pixelbench_aliases = {
            "mmvp": "mmvp",
            "v*": "vstar",
            "vstar": "vstar",
            "ocrbench": "ocrbench",
            "countbench": "countbenchqa",
            "countbenchqa": "countbenchqa",
        }
        if t == "pixelbench" or t in pixelbench_aliases:
            benchmarks = None if t == "pixelbench" else [pixelbench_aliases[t]]
            log_for_0(f"Evaluating PixelBench task '{t}' at step {step}...")
            acc, sample_outputs, metric_dict = eval_pixelbench(
                short_sample_fn,
                run_p_sample_step,
                model,
                tokenizer,
                params,
                config,
                benchmarks=benchmarks,
            )
            scalar_dict = {"step": step}
            if t == "pixelbench":
                scalar_dict[f"pixelbench_macro_acc{suffix}"] = acc
                for name, metrics in metric_dict.get("benchmarks", {}).items():
                    scalar_dict[f"{name}_acc{suffix}"] = float(metrics.get("acc", 0.0))
            else:
                scalar_dict[f"{pixelbench_aliases[t]}_acc{suffix}"] = acc
            writer.write_scalars(step, scalar_dict)
            if sample_outputs:
                writer.write_texts(step, f"{t.replace('*', 'star')}_samples{suffix}", sample_outputs)
            log_for_0(f"PixelBench task '{t}' evaluation finished.")
            continue

        log_for_0(f"Unknown eval task '{task}', skipping.")
