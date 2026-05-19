import io
import json
import os
import random
from collections import defaultdict
from functools import partial
from glob import glob

import fsspec
import jax
import pandas as pd
import torch
from PIL import Image
from jax.experimental import multihost_utils as mu
from torch.utils.data import DataLoader, Dataset, Sampler

from input_pipeline import get_transforms, prepare_batch_data
from utils.logging_util import log_for_0, log_for_all
from utils.eval_io_util import ensure_eval_result_base_dir, eval_result_prefix


PERCEPTION_TASKS = [
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
]

COGNITION_TASKS = [
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
]


class DistributedEvalSampler(Sampler):
    """Deterministic eval-only sampler without padding/duplication."""

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = jax.process_count()
        if rank is None:
            rank = jax.process_index()
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.dataset_len = len(dataset)
        self.num_samples = (
            self.dataset_len - self.rank + self.num_replicas - 1
        ) // self.num_replicas

    def __iter__(self):
        return iter(range(self.rank, self.dataset_len, self.num_replicas))

    def __len__(self):
        return self.num_samples


_mme_load_count = 0


def _list_parquet_files(root: str):
    if root.startswith("gs://"):
        fs, fs_path = fsspec.core.url_to_fs(root)
        proto = fs.protocol[0] if isinstance(fs.protocol, (tuple, list)) else fs.protocol
        files = sorted(fs.glob(fs_path.rstrip("/") + "/*.parquet"))
        return [f"{proto}://{p}" for p in files]
    return sorted(glob(os.path.join(root, "*.parquet")))


def _load_mme_rows(root: str):
    global _mme_load_count
    files = _list_parquet_files(root)
    if not files:
        raise FileNotFoundError(f"No parquet files found under MME root: {root}")
    if len(files) != 2:
        raise ValueError(f"Expected exactly 2 parquet files for MME evaluation, but found {len(files)} under {root}.")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    needed_cols = ["question_id", "image", "question", "answer", "category"]
    missing = set(needed_cols) - set(df.columns)
    if missing:
        raise ValueError(f"MME parquet missing columns: {missing}")
    rows = df[needed_cols].to_dict(orient="records")
    random.Random(42 + _mme_load_count).shuffle(rows)
    _mme_load_count += 1
    return rows


def preprocess_mme_sample(sample, transform, tokenizer, max_len):
    """Preprocess one MME sample with natural question prompt."""
    try:
        image = sample.get("jpg") or sample.get("png")
        if image is None:
            return None
        pixel_values = transform(image)
    except Exception:
        return None

    question = (sample.get("question") or "").strip()
    if not question:
        return None

    if not question.endswith("?"):
        question = question + "?"
    prefix = f"{question}\n"
    full_ids = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    prefix_len = min(len(full_ids), max_len)

    cur_len = len(full_ids)
    pad_len = max_len - cur_len
    pad_id = tokenizer.special_tokens.PAD
    if pad_len > 0:
        input_ids_list = full_ids + [pad_id] * pad_len
    else:
        input_ids_list = full_ids[:max_len]

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    labels = torch.zeros(max_len, dtype=torch.long)  # not used for inference
    attention_mask = torch.ones(max_len, dtype=torch.bool)
    if pad_len > 0:
        attention_mask[cur_len:] = False

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prefix_len": prefix_len,
        "aux": sample.get("aux"),
    }


class MMEDataset(Dataset):
    def __init__(self, root, config, tokenizer):
        self.rows = _load_mme_rows(root)
        self.preprocess_fn = partial(
            preprocess_mme_sample,
            transform=get_transforms(
                config.dataset.image_size,
                is_train=False,
                resize_mode=getattr(config.dataset, "resize_mode", "letterbox"),
            ),
            tokenizer=tokenizer,
            max_len=config.dataset.max_txt_len,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        image_info = item["image"]
        if isinstance(image_info, dict):
            img_bytes = image_info.get("bytes", None)
        else:
            img_bytes = image_info
        if img_bytes is None:
            return None
        if isinstance(img_bytes, memoryview):
            img_bytes = img_bytes.tobytes()
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return None

        sample = {
            "jpg": img,
            "question": item["question"],
            "aux": {
                "question_id": str(item["question_id"]),
                "question": str(item["question"]),
                "answer": str(item["answer"]),
                "category": str(item["category"]),
            },
        }
        return self.preprocess_fn(sample)


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    collated = {}
    first = batch[0]
    for key, value in first.items():
        if isinstance(value, torch.Tensor):
            collated[key] = torch.stack([b[key] for b in batch])
        elif key == "prefix_len":
            collated[key] = torch.tensor([b[key] for b in batch], dtype=torch.int32)
        elif key == "aux":
            collated[key] = [b[key] for b in batch]
    return collated


def parse_pred_ans(pred_ans: str):
    pred_ans = (pred_ans or "").strip().lower()
    if pred_ans in ["yes", "no"]:
        return pred_ans
    prefix_pred_ans = pred_ans[:4]
    if "yes" in prefix_pred_ans:
        return "yes"
    if "no" in prefix_pred_ans:
        return "no"
    return "other"


def _compute_task_score(rows):
    """rows: list of dicts with keys question_id, gt, pred."""
    if not rows:
        return {
            "acc": 0.0,
            "acc_plus": 0.0,
            "task_score": 0.0,
            "num_samples": 0,
            "num_pairs": 0,
        }

    correct = []
    by_qid = defaultdict(list)
    for r in rows:
        is_correct = int(r["gt"] == r["pred"])
        correct.append(is_correct)
        by_qid[r["question_id"]].append(is_correct)

    acc = sum(correct) / len(correct)
    both_correct = 0
    for vals in by_qid.values():
        both_correct += int(len(vals) == 2 and vals[0] == 1 and vals[1] == 1)
    acc_plus = both_correct / max(len(by_qid), 1)
    task_score = (acc + acc_plus) * 100.0
    return {
        "acc": acc,
        "acc_plus": acc_plus,
        "task_score": task_score,
        "num_samples": len(rows),
        "num_pairs": len(by_qid),
    }


def score_mme(all_results):
    per_task = defaultdict(list)

    for r in all_results:
        category = r["category"]
        gt = str(r["gt_answer"]).strip().lower()
        pred = parse_pred_ans(r["pred_answer"])
        if gt not in ["yes", "no"]:
            continue
        per_task[category].append(
            {
                "question_id": str(r["question_id"]),
                "gt": gt,
                "pred": pred,
            }
        )

    task_metrics = {}
    perception_score = 0.0
    cognition_score = 0.0

    for task in PERCEPTION_TASKS + COGNITION_TASKS:
        metric = _compute_task_score(per_task.get(task, []))
        task_metrics[task] = metric
        if task in PERCEPTION_TASKS:
            perception_score += metric["task_score"]
        else:
            cognition_score += metric["task_score"]

    mme_s = perception_score + cognition_score
    return {
        "MME-P": perception_score,
        "MME-C": cognition_score,
        "MME-S": mme_s,
        "task_metrics": task_metrics,
    }


def vis_mme_qa(o):
    return (
        f'question: {o.get("question", "")}\n'
        f'answer: {o.get("pred_answer", "")}\n'
        f'gt_answer: {o.get("gt_answer", "")}'
    )


def eval_mme(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    mme_root = getattr(config.eval, "mme_root", None)
    if not mme_root:
        raise ValueError("config.eval.mme_root is required for MME evaluation.")

    log_for_0(f"MME eval: loading parquet files from {mme_root}")
    dataset = MMEDataset(mme_root, config, tokenizer)
    batch_size = config.eval.device_batch_size * jax.local_device_count()
    sampler = DistributedEvalSampler(dataset, num_replicas=jax.process_count(), rank=jax.process_index())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=getattr(config.eval, "mme_num_workers", 0),
        collate_fn=collate_fn,
    )

    all_outs = []
    sample_outputs = []

    for i, batch in enumerate(loader):
        if not batch:
            continue
        batch = prepare_batch_data(batch, batch_size=batch_size)
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
            out_str = out_str.strip()
            all_outs.append(
                {
                    "question_id": aux["question_id"],
                    "category": aux["category"],
                    "question": aux["question"],
                    "gt_answer": aux["answer"],
                    "pred_answer": out_str,
                }
            )
            if len(sample_outputs) < 16:
                sample_outputs.append(
                    vis_mme_qa(
                        {
                            "question": aux["question"],
                            "pred_answer": out_str,
                            "gt_answer": aux["answer"],
                        }
                    )
                )

        if i % 20 == 0:
            log_for_0(f"MME batch {i}, collected {len(all_outs)} results...")

    mu.sync_global_devices("mme inference done")

    base_dir, result_prefix = eval_result_prefix(
        config,
        "mme_cache_dir",
        "/kmh-nfs-ssd-us-mount/data/cached/zhh/mme_eval",
        "mme",
    )
    ensure_eval_result_base_dir(base_dir)

    rank_file = f"{result_prefix}.results_{jax.process_index()}.json"
    with open(rank_file, "w", encoding="utf-8") as f:
        json.dump(all_outs, f, ensure_ascii=False, indent=2)

    mu.sync_global_devices("mme write done")

    if jax.process_index() == 0:
        merged = []
        for r in range(jax.process_count()):
            pf = f"{result_prefix}.results_{r}.json"
            if not os.path.exists(pf):
                raise FileNotFoundError(f"During MME eval, process {r} results file missing: {pf}")
            with open(pf, "r", encoding="utf-8") as f:
                merged.extend(json.load(f))

        with open(f"{result_prefix}.results_final.json", "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        metric_dict = score_mme(merged)
        with open(f"{result_prefix}.metrics.json", "w", encoding="utf-8") as f:
            json.dump(metric_dict, f, ensure_ascii=False, indent=2)

        mme_p = float(metric_dict["MME-P"])
        mme_s = float(metric_dict["MME-S"])
        log_for_0(f"MME-P: {mme_p:.2f}")
        log_for_0(f"MME-S: {mme_s:.2f}")
        log_for_0(f"MME merged results saved with prefix: {result_prefix}")
    else:
        log_for_all(f"Process {jax.process_index()} waiting for MME scoring...")
        metric_dict = {"task_metrics": {}}
        mme_p = 0.0
        mme_s = 0.0

    mu.sync_global_devices("mme eval done")
    return mme_p, mme_s, sample_outputs, metric_dict

def collate_fn(batch):
    """
    加强版 Collate Function:
    1. 过滤 None (处理坏图)
    2. 智能堆叠: 只会对 Tensor 类型的字段进行 Stack
    3. 自动忽略: 字符串(str)、数字(int/float)等非 Tensor 字段，防止报错
    """
    # 1. 过滤 None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    collated = {}
    
    # 2. 获取第一个样本的 Keys 作为参考
    first_sample = batch[0]
    
    for key, value in first_sample.items():
        # 3. type check: only stack tensors
        if isinstance(value, torch.Tensor):
            # ensure all samples have this key, prevent KeyError
            try:
                collated[key] = torch.stack([b[key] for b in batch])
            except RuntimeError as e:
                log_for_0(f"⚠️ Stack error for key '{key}': {e}")
                # 可能是 tensor 形状不一致 (比如没 resize 好)，跳过该字段
                raise e
        elif key == 'prefix_len':
            collated[key] = torch.tensor([b[key] for b in batch], dtype=torch.int32)
        else:
            # 如果是字符串 (如 'txt', '__key__', 'url')，直接忽略，不传给模型
            # pass
            if key == 'aux':
                collated[key] = [b[key] for b in batch] # 保留 aux 供后续使用
            
    return collated
