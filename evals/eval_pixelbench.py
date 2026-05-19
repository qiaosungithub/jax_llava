"""Pixel-centric VQA evaluations in one file.

Supported benchmarks stored under a common tar-only root:
  <root>/<benchmark>/images.tar
  <root>/<benchmark>/metadata.tar

The metadata tar must contain manifest.jsonl. Each manifest row has an image
member name and task-specific answer fields. This matches the tar artifacts
created by ../beifen/PixelBench-upload.py.
"""

import io
import json
import os
import re
import tarfile
from collections import defaultdict

import fsspec
import jax
import numpy as np
import torch
from PIL import Image
from jax.experimental import multihost_utils as mu
from torch.utils.data import DataLoader, Dataset, Sampler

from input_pipeline import get_transforms, prepare_batch_data
from utils.logging_util import log_for_0, log_for_all
from utils.eval_io_util import ensure_eval_result_base_dir, eval_result_prefix


PIXELBENCH_BENCHMARKS = ("mmvp", "vstar", "ocrbench", "countbenchqa")
DEFAULT_SAMPLES_PER_BENCHMARK = 8
BENCH_ALIASES = {
    "mmvp": "mmvp",
    "v*": "vstar",
    "vstar": "vstar",
    "vstar_bench": "vstar",
    "ocrbench": "ocrbench",
    "countbench": "countbenchqa",
    "countbenchqa": "countbenchqa",
}

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
}


class DistributedEvalSampler(Sampler):
    """Deterministic eval sampler without duplicate samples."""

    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = int(num_replicas if num_replicas is not None else jax.process_count())
        self.rank = int(rank if rank is not None else jax.process_index())
        self.dataset_len = len(dataset)
        self.num_samples = (self.dataset_len - self.rank + self.num_replicas - 1) // self.num_replicas

    def __iter__(self):
        return iter(range(self.rank, self.dataset_len, self.num_replicas))

    def __len__(self):
        return self.num_samples


def _canon_bench(name):
    key = str(name).strip().lower()
    if key not in BENCH_ALIASES:
        raise ValueError(f"Unknown pixelbench benchmark: {name}. Expected one of {sorted(BENCH_ALIASES)}")
    return BENCH_ALIASES[key]


def _join_url(root, *parts):
    return "/".join([root.rstrip("/")] + [p.strip("/") for p in parts])


def _bench_tar_path(root, bench, kind):
    root = str(root).rstrip("/")
    filename = f"{kind}.tar"
    if "{benchmark}" in root or "{bench}" in root:
        return root.format(benchmark=bench, bench=bench, kind=kind)
    if root.endswith(filename):
        return root
    if root.rsplit("/", 1)[-1] == bench:
        return _join_url(root, filename)
    return _join_url(root, bench, filename)


def _open_binary(path):
    return fsspec.open(path, "rb").open()


def _read_manifest(metadata_tar):
    with _open_binary(metadata_tar) as f:
        with tarfile.open(fileobj=f, mode="r|*") as tf:
            for member in tf:
                if member.isfile() and member.name == "manifest.jsonl":
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        break
                    text = extracted.read().decode("utf-8")
                    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
                    if not rows:
                        raise ValueError(f"Empty manifest.jsonl in {metadata_tar}")
                    return rows
    raise FileNotFoundError(f"manifest.jsonl not found in {metadata_tar}")


def _read_image_tar(images_tar):
    images = {}
    with _open_binary(images_tar) as f:
        with tarfile.open(fileobj=f, mode="r|*") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                images[member.name] = extracted.read()
    if not images:
        raise ValueError(f"No image files found in {images_tar}")
    return images


def _safe_str(value):
    if value is None:
        return ""
    return str(value).strip()


def _build_prompt(row):
    bench = row.get("benchmark", "")
    question = _safe_str(row.get("question"))
    if bench == "mmvp":
        options = _safe_str(row.get("options"))
        prompt = question
        if options:
            prompt += f"\nOptions: {options}"
        prompt += "\nAnswer with the option letter only.\n"
        return prompt
    if bench == "vstar":
        if "answer with" in question.lower():
            return question.rstrip() + "\n"
        return question.rstrip() + "\nAnswer with the option letter only.\n"
    return question.rstrip() + "\n"


def _encode_prompt(prompt, tokenizer, max_len):
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    eff_len = min(len(ids), max_len)
    pad_id = tokenizer.special_tokens.PAD
    input_ids = ids[:eff_len] + [pad_id] * (max_len - eff_len)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(eff_len, dtype=torch.int32)


class PixelBenchDataset(Dataset):
    def __init__(self, root, bench, config, tokenizer):
        self.root = root
        self.bench = _canon_bench(bench)
        self.config = config
        self.tokenizer = tokenizer
        self.transform = get_transforms(
            config.dataset.image_size,
            is_train=False,
            resize_mode=getattr(config.dataset, "resize_mode", "letterbox"),
        )
        self.max_len = int(getattr(config.eval, "pixelbench_max_txt_len", config.dataset.max_txt_len))
        self.metadata_tar = _bench_tar_path(root, self.bench, "metadata")
        self.images_tar = _bench_tar_path(root, self.bench, "images")
        log_for_0(f"PixelBench/{self.bench}: loading metadata from {self.metadata_tar}")
        self.rows = _read_manifest(self.metadata_tar)
        log_for_0(f"PixelBench/{self.bench}: loading images from {self.images_tar}")
        self.images = _read_image_tar(self.images_tar)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = dict(self.rows[idx])
        image_name = row.get("image")
        image_bytes = self.images.get(image_name)
        if image_bytes is None:
            return None
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pixel_values = self.transform(image)
        except Exception:
            return None

        prompt = _build_prompt(row)
        input_ids, prefix_len = _encode_prompt(prompt, self.tokenizer, self.max_len)
        aux = dict(row)
        aux["prompt"] = prompt
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "prefix_len": prefix_len,
            "aux": aux,
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


def _dummy_aux(bench):
    return {
        "benchmark": bench,
        "id": "-1",
        "image": "",
        "question": "",
        "answer": "",
        "prompt": "",
    }


def _make_dummy_batch(batch_size, image_size, max_len, bench):
    return {
        "pixel_values": torch.zeros((batch_size, 3, image_size, image_size), dtype=torch.float32),
        "input_ids": torch.zeros((batch_size, max_len), dtype=torch.long),
        "prefix_len": torch.ones((batch_size,), dtype=torch.int32),
        "aux": [_dummy_aux(bench) for _ in range(batch_size)],
        "_all_pad": True,
    }


def _result_prefix(config, bench):
    return eval_result_prefix(
        config,
        "pixelbench_cache_dir",
        "/kmh-nfs-ssd-us-mount/data/cached/zhh/pixelbench_eval",
        "pixelbench",
        bench,
    )


def _normalize_option(text):
    s = _safe_str(text).upper()
    m = re.search(r"(?:^|[^A-Z0-9])([A-D])(?:[^A-Z0-9]|$)", s)
    if m:
        return m.group(1)
    if len(s) == 1 and s in "ABCD":
        return s
    return ""


def _normalize_text(text):
    text = _safe_str(text).lower()
    text = re.sub(r"\b(the answer is|answer is|it is|it's)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _normalize_ocr(text):
    return re.sub(r"[^a-z0-9]+", "", _safe_str(text).lower())


def _extract_number(text):
    text = _safe_str(text).lower()
    m = re.search(r"-?\d+", text)
    if m:
        try:
            return int(m.group(0))
        except ValueError:
            pass
    words = re.findall(r"[a-z]+", text)
    for i, word in enumerate(words):
        if word in _NUMBER_WORDS:
            value = _NUMBER_WORDS[word]
            if i + 1 < len(words) and words[i + 1] in _NUMBER_WORDS and value >= 20:
                value += _NUMBER_WORDS[words[i + 1]]
            return value
    return None


def _score_one(record):
    bench = record.get("benchmark")
    pred = _safe_str(record.get("prediction"))
    gt = _safe_str(record.get("answer"))

    if bench in {"mmvp", "vstar"}:
        pred_option = _normalize_option(pred)
        gt_option = _normalize_option(gt)
        return float(pred_option == gt_option), {"pred_norm": pred_option, "gt_norm": gt_option}

    if bench == "ocrbench":
        pred_norm = _normalize_ocr(pred)
        answers = record.get("answers") or [gt]
        gt_norms = [_normalize_ocr(a) for a in answers if _safe_str(a)]
        correct = any(g and (pred_norm == g or g in pred_norm) for g in gt_norms)
        return float(correct), {"pred_norm": pred_norm, "gt_norm": gt_norms[0] if gt_norms else ""}

    if bench == "countbenchqa":
        pred_num = _extract_number(pred)
        gt_num = int(record.get("number", gt))
        return float(pred_num == gt_num), {"pred_norm": pred_num, "gt_norm": gt_num}

    raise ValueError(f"Cannot score unknown benchmark: {bench}")


def _score_records(records):
    if not records:
        return {"acc": 0.0, "num_samples": 0, "by_category": {}, "scored": []}

    scored = []
    by_category = defaultdict(list)
    for item in records:
        score, norm = _score_one(item)
        out = dict(item)
        out["score"] = score
        out.update(norm)
        scored.append(out)
        category = item.get("category") or item.get("dataset") or item.get("question_type") or "all"
        by_category[str(category)].append(score)

    metrics = {
        "acc": float(np.mean([x["score"] for x in scored]) * 100.0),
        "num_samples": len(scored),
        "by_category": {k: float(np.mean(v) * 100.0) for k, v in sorted(by_category.items())},
        "scored": scored,
    }
    return metrics


def _vis_record(item):
    return (
        f"benchmark: {item.get('benchmark', '')}\n"
        f"question: {item.get('question', '')}\n"
        f"prediction: {item.get('prediction', '')}\n"
        f"answer: {item.get('answer', '')}\n"
        f"score: {item.get('score', '')}"
    )


def _run_one_benchmark(p_sample_step, run_p_sample_step, model, tokenizer, params, config, bench):
    bench = _canon_bench(bench)
    samples_per_benchmark = int(
        getattr(config.eval, "pixelbench_samples_per_benchmark", DEFAULT_SAMPLES_PER_BENCHMARK)
    )
    root = getattr(config.eval, "pixelbench_root", None)
    bench_root_attr = f"{bench}_root"
    root = getattr(config.eval, bench_root_attr, root)
    if not root:
        raise ValueError("config.eval.pixelbench_root is required for PixelBench evaluation.")
    assert "\U0001f4a3" not in root, f"Unresolved zone marker found in PixelBench root: {root}"

    log_for_0(f"PixelBench/{bench}: root={root}")
    dataset = PixelBenchDataset(root, bench, config, tokenizer)
    device_batch_size = int(getattr(config.eval, "pixelbench_device_batch_size", config.eval.device_batch_size))
    batch_size = device_batch_size * jax.local_device_count()
    sampler = DistributedEvalSampler(dataset, num_replicas=jax.process_count(), rank=jax.process_index())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(getattr(config.eval, "pixelbench_num_workers", 0)),
        collate_fn=collate_fn,
    )
    loader_iter = iter(loader)

    samples_per_process = (len(dataset) + jax.process_count() - 1) // jax.process_count()
    fixed_num_steps = max(1, (samples_per_process + batch_size - 1) // batch_size)
    max_len = int(getattr(config.eval, "pixelbench_max_txt_len", config.dataset.max_txt_len))
    log_for_0(
        f"PixelBench/{bench}: rows={len(dataset)}, samples_per_process={samples_per_process}, "
        f"fixed_num_steps={fixed_num_steps}, batch_size={batch_size}"
    )

    all_outs = []
    sample_outputs = []
    for i in range(fixed_num_steps):
        try:
            raw_batch = next(loader_iter)
            if not raw_batch:
                raw_batch = _make_dummy_batch(batch_size, config.dataset.image_size, max_len, bench)
        except StopIteration:
            raw_batch = _make_dummy_batch(batch_size, config.dataset.image_size, max_len, bench)

        if "aux" not in raw_batch:
            raw_batch["aux"] = []
        if len(raw_batch["aux"]) < batch_size:
            raw_batch["aux"].extend([_dummy_aux(bench) for _ in range(batch_size - len(raw_batch["aux"]))])

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

        for aux, out_str, is_pad in zip(batch["aux"], out_strs, batch["is_pad"].tolist()):
            if is_pad:
                continue
            record = dict(aux)
            record["prediction"] = _safe_str(out_str)
            all_outs.append(record)
            if len(sample_outputs) < samples_per_benchmark:
                sample_outputs.append(_vis_record(record))

        if i % 20 == 0:
            log_for_0(f"PixelBench/{bench}: batch {i}, local collected {len(all_outs)} results")

    mu.sync_global_devices(f"pixelbench {bench} inference done")

    base_dir, result_prefix = _result_prefix(config, bench)
    ensure_eval_result_base_dir(base_dir)

    rank_file = f"{result_prefix}.results_{jax.process_index()}.json"
    with open(rank_file, "w", encoding="utf-8") as f:
        json.dump(all_outs, f, ensure_ascii=False, indent=2)

    mu.sync_global_devices(f"pixelbench {bench} write done")

    if jax.process_index() == 0:
        merged = []
        for rank in range(jax.process_count()):
            path = f"{result_prefix}.results_{rank}.json"
            if not os.path.exists(path):
                raise FileNotFoundError(f"During PixelBench/{bench} eval, process {rank} results file missing: {path}")
            with open(path, "r", encoding="utf-8") as f:
                merged.extend(json.load(f))

        dedup = {}
        for item in merged:
            key = (item.get("benchmark", bench), str(item.get("id", "")))
            dedup[key] = item
        merged = [dedup[k] for k in sorted(dedup)]
        metrics = _score_records(merged)
        scored = metrics.pop("scored")

        with open(f"{result_prefix}.results_final.json", "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        with open(f"{result_prefix}.scored_results.json", "w", encoding="utf-8") as f:
            json.dump(scored, f, ensure_ascii=False, indent=2)
        with open(f"{result_prefix}.metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        log_for_0(f"PixelBench/{bench}: accuracy={metrics['acc']:.2f}% ({metrics['num_samples']} samples)")
        log_for_0(f"PixelBench/{bench}: merged results saved with prefix: {result_prefix}")
        sample_outputs = [_vis_record(item) for item in scored[:samples_per_benchmark]]
    else:
        log_for_all(f"Process {jax.process_index()} waiting for PixelBench/{bench} scoring...")
        metrics = {"acc": 0.0, "num_samples": 0, "by_category": {}}

    mu.sync_global_devices(f"pixelbench {bench} eval done")
    return metrics["acc"], sample_outputs, metrics


def eval_pixelbench(p_sample_step, run_p_sample_step, model, tokenizer, params, config, benchmarks=None):
    if benchmarks is None:
        benchmarks = getattr(config.eval, "pixelbench_benchmarks", PIXELBENCH_BENCHMARKS)
    benchmarks = [_canon_bench(b) for b in benchmarks]

    all_metrics = {}
    all_samples = []
    for bench in benchmarks:
        acc, samples, metrics = _run_one_benchmark(
            p_sample_step,
            run_p_sample_step,
            model,
            tokenizer,
            params,
            config,
            bench,
        )
        all_metrics[bench] = dict(metrics)
        all_metrics[bench]["acc"] = float(acc)
        if samples:
            all_samples.extend(samples)

    if jax.process_index() == 0:
        macro_acc = float(np.mean([m["acc"] for m in all_metrics.values()])) if all_metrics else 0.0
    else:
        macro_acc = 0.0
    metric_dict = {"macro_acc": macro_acc, "benchmarks": all_metrics}
    return macro_acc, all_samples, metric_dict


def eval_mmvp(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    return eval_pixelbench(p_sample_step, run_p_sample_step, model, tokenizer, params, config, benchmarks=["mmvp"])


def eval_vstar(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    return eval_pixelbench(p_sample_step, run_p_sample_step, model, tokenizer, params, config, benchmarks=["vstar"])


def eval_ocrbench(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    return eval_pixelbench(p_sample_step, run_p_sample_step, model, tokenizer, params, config, benchmarks=["ocrbench"])


def eval_countbenchqa(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    return eval_pixelbench(p_sample_step, run_p_sample_step, model, tokenizer, params, config, benchmarks=["countbenchqa"])
