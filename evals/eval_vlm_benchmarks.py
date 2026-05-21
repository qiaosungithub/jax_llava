"""Evaluators for image-only visual-understanding benchmarks uploaded as WDS tar shards.

Supported datasets:
  - GQA balanced testdev/val/train shards produced by VLM-Eval-Benchmarks-upload.py
  - VisWiz-VQA val/test shards
  - ScienceQA-IMG train/validation/test shards
  - SEED-Bench image-only shards

The uploader stores each record as image + json in a WebDataset tar. GQA and
SEED-Bench records are image-level and contain multiple QAs in the json; the
other datasets are one QA per image record.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from glob import glob
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import fsspec
import jax
import numpy as np
import torch
import webdataset as wds
from absl import logging
from jax.experimental import multihost_utils as mu
from torch.utils.data import DataLoader, IterableDataset

from evals.eval_vqav2 import postprocess_vqav2_text, vqa_accuracy_one
from input_pipeline import get_transforms, prepare_batch_data
from utils.eval_io_util import ensure_eval_result_base_dir, eval_result_prefix
from utils.logging_util import log_for_0, log_for_all

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _normalize_short_answer(text: str) -> str:
    text = _as_text(text).strip()
    text = re.sub(r"^\s*(answer|the answer is|final answer)\s*[:\-]?\s*", "", text, flags=re.I)
    text = text.split("\n", 1)[0].strip()
    return postprocess_vqav2_text(text)


def _exact_answer_score(pred: str, answers: Sequence[Any]) -> float:
    pred_norm = _normalize_short_answer(pred)
    for ans in answers:
        if pred_norm == _normalize_short_answer(_as_text(ans)):
            return 1.0
    return 0.0


def _extract_answers(payload: Dict[str, Any]) -> List[str]:
    raw_answers = []
    for key in ("answers", "answer", "label", "correct_answer"):
        if key in payload and payload.get(key) is not None:
            raw_answers = _as_list(payload.get(key))
            break
    answers = []
    for ans in raw_answers:
        if isinstance(ans, dict):
            ans = ans.get("answer", ans.get("text", ans.get("label", "")))
        ans = _as_text(ans).strip()
        if ans:
            answers.append(ans)
    return answers


def _record_id(payload: Dict[str, Any], fallback: str) -> str:
    for key in ("question_id", "questionId", "questionID", "id", "qid", "qa_id"):
        if key in payload and payload.get(key) not in (None, ""):
            return _as_text(payload.get(key))
    return fallback


def _sample_image(sample: Dict[str, Any]) -> Any:
    return (
        sample.get("jpg")
        or sample.get("jpeg")
        or sample.get("png")
        or sample.get("webp")
    )


def _load_json(sample: Dict[str, Any]) -> Dict[str, Any]:
    raw = sample.get("json")
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = json.loads(raw.decode("utf-8"))
    if isinstance(raw, dict):
        return raw
    return {}


def _list_tar_urls(root: str | Sequence[str]) -> List[str]:
    if isinstance(root, (list, tuple)):
        urls: List[str] = []
        for item in root:
            urls.extend(_list_tar_urls(item))
        return urls

    root = str(root)
    if root.endswith(".tar") or ("{" in root and "}" in root):
        return [root]

    pattern = root if "*" in root else root.rstrip("/") + "/shard-*.tar"
    if pattern.startswith("gs://"):
        fs, fs_path = fsspec.core.url_to_fs(pattern)
        protocol = fs.protocol[0] if isinstance(fs.protocol, (tuple, list)) else fs.protocol
        matches = sorted(fs.glob(fs_path))
        return [p if str(p).startswith("gs://") else f"{protocol}://{p}" for p in matches]
    return sorted(glob(pattern))


def _format_vqa_prompt(question: str, instruction: str) -> str:
    question = _as_text(question).strip()
    if question and not question.endswith("?"):
        question = question + "?"
    return f"{question}\n{instruction}\n"


def _extract_choices(payload: Dict[str, Any]) -> List[str]:
    for key in ("choices", "options", "choice"):
        value = payload.get(key)
        if isinstance(value, list):
            return [_as_text(x).strip() for x in value if _as_text(x).strip()]
        if isinstance(value, str):
            # Some HF conversions store choices as a JSON-ish string.
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [_as_text(x).strip() for x in parsed if _as_text(x).strip()]
            except Exception:
                parts = re.split(r"\s*\|\s*", value)
                if len(parts) > 1:
                    return [p.strip() for p in parts if p.strip()]
    choices = []
    for letter in LETTERS:
        for key in (f"choice_{letter.lower()}", f"choice_{letter}", letter, letter.lower()):
            if payload.get(key) not in (None, ""):
                choices.append(_as_text(payload.get(key)).strip())
                break
    return choices


def _answer_index(payload: Dict[str, Any], choices: Sequence[str]) -> Optional[int]:
    for key in ("answer", "answer_idx", "answer_index", "label", "correct_answer"):
        if key not in payload or payload.get(key) is None:
            continue
        value = payload.get(key)
        if isinstance(value, (int, np.integer)):
            idx = int(value)
            if 0 <= idx < len(choices):
                return idx
            if 1 <= idx <= len(choices):
                return idx - 1
        text = _as_text(value).strip()
        if not text:
            continue
        letter_match = re.match(r"^\s*([A-Z])\b", text, flags=re.I)
        if letter_match:
            idx = LETTERS.find(letter_match.group(1).upper())
            if 0 <= idx < len(choices):
                return idx
        norm = _normalize_short_answer(text)
        for i, choice in enumerate(choices):
            if norm == _normalize_short_answer(choice):
                return i
    return None


def _format_mc_prompt(question: str, choices: Sequence[str], extra_context: str = "") -> str:
    lines = []
    extra_context = _as_text(extra_context).strip()
    if extra_context:
        lines.append(f"Context: {extra_context}")
    lines.append(_as_text(question).strip())
    for idx, choice in enumerate(choices):
        if idx >= len(LETTERS):
            break
        lines.append(f"{LETTERS[idx]}. {_as_text(choice).strip()}")
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines).strip() + "\n"


def _parse_choice_prediction(pred: str, choices: Sequence[str]) -> Optional[int]:
    text = _as_text(pred).strip()
    # Prefer the first standalone option letter; this handles "A", "A.", "Answer: A".
    match = re.search(r"\b([A-Z])\b", text, flags=re.I)
    if match:
        idx = LETTERS.find(match.group(1).upper())
        if 0 <= idx < len(choices):
            return idx
    norm_pred = _normalize_short_answer(text)
    for idx, choice in enumerate(choices):
        if norm_pred == _normalize_short_answer(choice):
            return idx
    return None


# -----------------------------------------------------------------------------
# Dataset expansion
# -----------------------------------------------------------------------------


def _expand_gqa(sample: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    record = _load_json(sample)
    image = _sample_image(sample)
    if image is None:
        return []
    image_id = _as_text(record.get("image_id", record.get("id", "")))
    out = []
    for i, qa in enumerate(record.get("qas", []) or []):
        if not isinstance(qa, dict):
            continue
        question = _as_text(qa.get("question", "")).strip()
        answers = _extract_answers(qa)
        if not question or not answers:
            continue
        qid = _record_id(qa, f"{image_id}_{i}")
        out.append(
            {
                "image": image,
                "prompt": _format_vqa_prompt(
                    question,
                    "Answer the question using a single word or phrase.",
                ),
                "aux": {
                    "id": qid,
                    "image_id": image_id,
                    "question": question,
                    "answers": answers,
                    "metric": "exact",
                },
            }
        )
    return out


def _expand_vizwiz(sample: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    record = _load_json(sample)
    image = _sample_image(sample)
    question = _as_text(record.get("question", "")).strip()
    if image is None or not question:
        return []
    answers = _extract_answers(record)
    key = _as_text(sample.get("__key__", "vizwiz"))
    qid = _record_id(record, f"{key}_{question[:64]}")
    return [
        {
            "image": image,
            "prompt": _format_vqa_prompt(
                question,
                "When the provided information is insufficient, respond with 'Unanswerable'. "
                "Answer the question using a single word or phrase.",
            ),
            "aux": {
                "id": qid,
                "question": question,
                "answers": answers,
                "metric": "vqa",
            },
        }
    ]


def _expand_scienceqa(sample: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    record = _load_json(sample)
    image = _sample_image(sample)
    question = _as_text(record.get("question", "")).strip()
    choices = _extract_choices(record)
    answer_idx = _answer_index(record, choices)
    if image is None or not question or not choices:
        return []
    key = _as_text(sample.get("__key__", "scienceqa"))
    qid = _record_id(record, f"{key}_{record.get('pid', '')}")
    context = record.get("hint", "") or record.get("lecture", "")
    return [
        {
            "image": image,
            "prompt": _format_mc_prompt(question, choices, extra_context=context),
            "aux": {
                "id": qid,
                "question": question,
                "choices": choices,
                "answer_idx": answer_idx,
                "answer": LETTERS[answer_idx] if answer_idx is not None and answer_idx < len(LETTERS) else None,
                "metric": "mc",
            },
        }
    ]


def _seed_type_name(type_map: Any, type_id: Any) -> str:
    if not isinstance(type_map, dict):
        return _as_text(type_id) if type_id not in (None, "") else "unknown"
    # Upload script stores name -> id. Invert it.
    for name, value in type_map.items():
        if _as_text(value) == _as_text(type_id):
            return _as_text(name)
    return _as_text(type_id) if type_id not in (None, "") else "unknown"


def _expand_seed_bench(sample: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    record = _load_json(sample)
    image = _sample_image(sample)
    if image is None:
        return []
    data_id = _as_text(record.get("data_id", ""))
    type_map = record.get("question_type")
    out = []
    for i, qa in enumerate(record.get("qas", []) or []):
        if not isinstance(qa, dict):
            continue
        question = _as_text(qa.get("question", "")).strip()
        choices = _extract_choices(qa)
        answer_idx = _answer_index(qa, choices)
        if not question or not choices:
            continue
        qid = _record_id(qa, f"{data_id}_{i}")
        type_id = qa.get("question_type_id", qa.get("question_type", qa.get("type_id")))
        out.append(
            {
                "image": image,
                "prompt": _format_mc_prompt(question, choices),
                "aux": {
                    "id": qid,
                    "data_id": data_id,
                    "question": question,
                    "choices": choices,
                    "answer_idx": answer_idx,
                    "answer": LETTERS[answer_idx] if answer_idx is not None and answer_idx < len(LETTERS) else None,
                    "question_type": _seed_type_name(type_map, type_id),
                    "metric": "mc",
                },
            }
        )
    return out


EXPANDERS: Dict[str, Callable[[Dict[str, Any]], Iterable[Dict[str, Any]]]] = {
    "gqa": _expand_gqa,
    "vizwiz": _expand_vizwiz,
    "scienceqa_img": _expand_scienceqa,
    "seed_bench": _expand_seed_bench,
}


class WDSUnderstandingEvalDataset(IterableDataset):
    def __init__(self, root_url, benchmark: str, config, tokenizer):
        self.root_url = root_url
        self.benchmark = benchmark
        self.config = config
        self.tokenizer = tokenizer
        self.transform = get_transforms(
            config.dataset.image_size,
            is_train=False,
            resize_mode=getattr(config.dataset, "resize_mode", "letterbox"),
        )
        self.max_len = int(getattr(config.eval, f"{benchmark}_max_txt_len", config.dataset.max_txt_len))
        self.num_processes = jax.process_count()
        self.process_rank = jax.process_index()

    def __iter__(self):
        urls = _list_tar_urls(self.root_url)
        if not urls:
            raise FileNotFoundError(f"No tar shards found for {self.benchmark}: {self.root_url}")
        ds = wds.WebDataset(urls, resampled=False, shardshuffle=False).decode("pil")
        expand = EXPANDERS[self.benchmark]
        sample_idx = 0
        for wds_sample in ds:
            for item in expand(wds_sample):
                if sample_idx % self.num_processes == self.process_rank:
                    out = self._preprocess(item)
                    if out is not None:
                        yield out
                sample_idx += 1

    def _preprocess(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            pixel_values = self.transform(item["image"])
        except Exception:
            return None

        prompt = _as_text(item.get("prompt", "")).strip()
        if not prompt:
            return None
        if not prompt.endswith("\n"):
            prompt += "\n"

        ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        eff_len = min(len(ids), self.max_len)
        pad_len = self.max_len - eff_len
        pad_id = self.tokenizer.special_tokens.PAD
        input_ids = ids[:eff_len] + [pad_id] * pad_len
        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "prefix_len": torch.tensor(eff_len, dtype=torch.int32),
            "aux": item.get("aux", {}),
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "prefix_len": torch.stack([b["prefix_len"] for b in batch]).to(torch.int32),
        "aux": [b["aux"] for b in batch],
    }


def _dummy_batch(batch_size: int, image_size: int, max_len: int) -> Dict[str, Any]:
    return {
        "pixel_values": torch.zeros((batch_size, 3, image_size, image_size), dtype=torch.float32),
        "input_ids": torch.zeros((batch_size, max_len), dtype=torch.long),
        "prefix_len": torch.ones((batch_size,), dtype=torch.int32),
        "aux": [{"id": "-1", "question": "", "answers": []} for _ in range(batch_size)],
        "_all_pad": True,
    }


def _score_result(row: Dict[str, Any]) -> Optional[float]:
    metric = row.get("metric")
    pred = row.get("prediction", "")
    if metric == "exact":
        answers = row.get("answers", [])
        if not answers:
            return None
        return _exact_answer_score(pred, answers)
    if metric == "vqa":
        answers = row.get("answers", [])
        if not answers:
            return None
        if len(answers) >= 10:
            return vqa_accuracy_one(pred, answers)
        return _exact_answer_score(pred, answers)
    if metric == "mc":
        gt = row.get("answer_idx")
        choices = row.get("choices", [])
        if gt is None:
            return None
        pred_idx = _parse_choice_prediction(pred, choices)
        return float(pred_idx == int(gt)) if pred_idx is not None else 0.0
    return None


def _merge_and_score(
    config,
    benchmark: str,
    cache_key: str,
    cache_default: str,
    result_name: str,
    result_prefix: str,
) -> Tuple[float, Dict[str, Any]]:
    all_results = []
    for rank in range(jax.process_count()):
        path = f"{result_prefix}.results_{rank}.json"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {benchmark} result file from rank {rank}: {path}")
        with open(path, encoding="utf-8") as f:
            all_results.extend(json.load(f))

    dedup = {}
    for row in all_results:
        key = row.get("id", len(dedup))
        if key not in dedup:
            dedup[key] = row
    all_results = list(dedup.values())

    scores = []
    by_type = defaultdict(list)
    no_gt = 0
    for row in all_results:
        score = _score_result(row)
        if score is None:
            no_gt += 1
            continue
        scores.append(float(score))
        if row.get("question_type"):
            by_type[row["question_type"]].append(float(score))

    primary = float(np.mean(scores) * 100.0) if scores else 0.0
    metrics = {
        "benchmark": benchmark,
        "num_predictions": len(all_results),
        "num_scored": len(scores),
        "num_without_gt": no_gt,
        "accuracy": primary,
    }
    if by_type:
        metrics["by_question_type"] = {
            k: {"accuracy": float(np.mean(v) * 100.0), "count": len(v)}
            for k, v in sorted(by_type.items())
        }

    final_path = f"{result_prefix}.results_final.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    metrics_path = f"{result_prefix}.metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log_for_0(f"{result_name} results: {final_path} ({len(all_results)} predictions)")
    log_for_0(f"{result_name} metrics: {metrics_path}")
    log_for_0(f"{result_name} accuracy: {primary:.2f}% over {len(scores)} scored samples")
    return primary, metrics


def _vis_row(row: Dict[str, Any]) -> str:
    lines = [f"question: {row.get('question', '')}", f"prediction: {row.get('prediction', '')}"]
    if row.get("answers"):
        lines.append(f"gt_answers: {row.get('answers')}")
    if row.get("choices"):
        lines.append(f"choices: {row.get('choices')}")
        lines.append(f"gt: {row.get('answer')}")
    if row.get("question_type"):
        lines.append(f"type: {row.get('question_type')}")
    return "\n".join(lines)


def _eval_understanding_benchmark(
    p_sample_step,
    run_p_sample_step,
    model,
    tokenizer,
    params,
    config,
    *,
    benchmark: str,
    root_key: str,
    total_key: str,
    default_total: int,
    cache_key: str,
    cache_default: str,
    result_name: str,
):
    root_url = getattr(config.eval, root_key)
    assert "💣" not in root_url, f"bomb placeholder found in eval path {root_url}"
    log_for_0(f"{result_name} eval: loading from {root_url}")

    dataset = WDSUnderstandingEvalDataset(root_url, benchmark, config, tokenizer)
    batch_size = int(config.eval.device_batch_size) * jax.local_device_count()
    max_len = int(getattr(config.eval, f"{benchmark}_max_txt_len", config.dataset.max_txt_len))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)
    loader_iter = iter(loader)

    total_samples = int(getattr(config.eval, total_key, default_total))
    samples_per_process = (total_samples + jax.process_count() - 1) // jax.process_count()
    fixed_num_steps = (samples_per_process + batch_size - 1) // batch_size
    log_for_0(
        f"{result_name} fixed eval schedule: total_samples={total_samples}, "
        f"samples_per_process={samples_per_process}, fixed_num_steps={fixed_num_steps}, "
        f"batch_size={batch_size}"
    )

    all_outs = []
    for step_idx in range(fixed_num_steps):
        try:
            raw_batch = next(loader_iter)
            if not raw_batch:
                raw_batch = _dummy_batch(batch_size, config.dataset.image_size, max_len)
        except StopIteration:
            raw_batch = _dummy_batch(batch_size, config.dataset.image_size, max_len)

        if "aux" not in raw_batch:
            raw_batch["aux"] = []
        if len(raw_batch["aux"]) < batch_size:
            raw_batch["aux"].extend([{"id": "-1", "question": "", "answers": []}] * (batch_size - len(raw_batch["aux"])))

        batch = prepare_batch_data(raw_batch, batch_size=batch_size)
        if raw_batch.get("_all_pad", False):
            batch["is_pad"] = np.ones((batch_size,), dtype=bool)

        out_strs = run_p_sample_step(
            p_sample_step,
            model,
            tokenizer,
            params,
            batch["pixel_values"],
            batch["input_ids"],
            prefix_len=batch["prefix_len"],
        )

        for aux, pred, is_pad in zip(batch["aux"], out_strs, batch["is_pad"].tolist()):
            if is_pad:
                continue
            row = dict(aux)
            row["prediction"] = _as_text(pred).strip()
            all_outs.append(row)

        if step_idx % 50 == 0:
            logging.info(
                f"rank {jax.process_index()}, {result_name} batch {step_idx}, "
                f"collected {len(all_outs)} results..."
            )

    mu.sync_global_devices(f"{benchmark} inference done")

    base_dir, prefix = eval_result_prefix(config, cache_key, cache_default, benchmark)
    ensure_eval_result_base_dir(base_dir)
    out_file = f"{prefix}.results_{jax.process_index()}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_outs, f, ensure_ascii=False, indent=2)

    mu.sync_global_devices(f"{benchmark} write done")

    if jax.process_index() == 0:
        primary, metrics = _merge_and_score(config, benchmark, cache_key, cache_default, result_name, prefix)
    else:
        log_for_all(f"Process {jax.process_index()} waiting for {result_name} evaluation to finish...")
        primary, metrics = 0.0, {}

    mu.sync_global_devices(f"{benchmark} eval done")
    return primary, [_vis_row(o) for o in all_outs[:16]], metrics


# -----------------------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------------------


def eval_gqa(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    return _eval_understanding_benchmark(
        p_sample_step,
        run_p_sample_step,
        model,
        tokenizer,
        params,
        config,
        benchmark="gqa",
        root_key="gqa_root",
        total_key="gqa_num_samples",
        default_total=12578,
        cache_key="gqa_cache_dir",
        cache_default="/kmh-nfs-ssd-us-mount/data/cached/zhh/gqa_eval",
        result_name="GQA",
    )


def eval_vizwiz(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    return _eval_understanding_benchmark(
        p_sample_step,
        run_p_sample_step,
        model,
        tokenizer,
        params,
        config,
        benchmark="vizwiz",
        root_key="vizwiz_root",
        total_key="vizwiz_num_samples",
        default_total=4319,
        cache_key="vizwiz_cache_dir",
        cache_default="/kmh-nfs-ssd-us-mount/data/cached/zhh/vizwiz_eval",
        result_name="VisWiz",
    )


def eval_scienceqa_img(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    return _eval_understanding_benchmark(
        p_sample_step,
        run_p_sample_step,
        model,
        tokenizer,
        params,
        config,
        benchmark="scienceqa_img",
        root_key="scienceqa_img_root",
        total_key="scienceqa_img_num_samples",
        default_total=2017,
        cache_key="scienceqa_img_cache_dir",
        cache_default="/kmh-nfs-ssd-us-mount/data/cached/zhh/scienceqa_img_eval",
        result_name="ScienceQA-IMG",
    )


def eval_seed_bench(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    return _eval_understanding_benchmark(
        p_sample_step,
        run_p_sample_step,
        model,
        tokenizer,
        params,
        config,
        benchmark="seed_bench",
        root_key="seed_bench_root",
        total_key="seed_bench_num_samples",
        default_total=14233,
        cache_key="seed_bench_cache_dir",
        cache_default="/kmh-nfs-ssd-us-mount/data/cached/zhh/seed_bench_eval",
        result_name="SEED-Bench",
    )
