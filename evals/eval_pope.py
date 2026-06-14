import glob
import hashlib
import io
import json
import os
import tarfile
import time
from functools import partial

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


def _path_exists(path: str) -> bool:
    fs, fs_path = fsspec.core.url_to_fs(path)
    return fs.exists(fs_path)


def _join_path(root: str, leaf: str) -> str:
    return f"{root.rstrip('/')}/{leaf.lstrip('/')}"


def _is_json_annotation_path(path: str) -> bool:
    return path.rstrip("/").endswith((".json", ".jsonl"))


def _maybe_add_gs_scheme(path: str) -> str:
    if path.startswith("gs://"):
        return path
    return f"gs://{path}"


def _glob_tar_shards(root: str):
    root = root.rstrip("/")
    if root.endswith(".tar"):
        return [root] if _path_exists(root) else []
    if root.startswith("gs://"):
        fs, fs_path = fsspec.core.url_to_fs(root)
        matched = sorted(fs.glob(f"{fs_path}/shard-*.tar"))
        if not matched:
            matched = sorted(fs.glob(f"{fs_path}/*.tar"))
        return [_maybe_add_gs_scheme(p) for p in matched]
    matched = sorted(glob.glob(os.path.join(root, "shard-*.tar")))
    if not matched:
        matched = sorted(glob.glob(os.path.join(root, "*.tar")))
    return matched


def resolve_pope_split_file(pope_root: str, split: str, dataset: str) -> str:
    root = pope_root.rstrip("/")

    if root.endswith(".json") or root.endswith(".jsonl"):
        return root

    candidates = []
    if dataset:
        candidates.extend(
            [
                _join_path(root, f"{dataset}_pope_{split}.json"),
                _join_path(root, f"{dataset}_pope_{split}.jsonl"),
                _join_path(root, f"{dataset}_pope_seem_{split}.json"),
                _join_path(root, f"{dataset}_pope_seem_{split}.jsonl"),
            ]
        )

    candidates.extend(
        [
            _join_path(root, f"{split}.json"),
            _join_path(root, f"{split}.jsonl"),
        ]
    )

    for path in candidates:
        if _path_exists(path):
            return path

    if root.startswith("gs://"):
        fs = fsspec.filesystem("gs")
        matched = sorted(fs.glob(f"{root}/*{split}*.json"))
        if matched:
            first = matched[0]
            return first if first.startswith("gs://") else f"gs://{first}"
    else:
        matched = sorted(glob.glob(os.path.join(root, f"*{split}*.json")))
        if matched:
            return matched[0]

    raise FileNotFoundError(
        f"Cannot resolve POPE file for split='{split}' under pope_root='{pope_root}'."
    )


def load_pope_questions(path: str, split: str):
    with fsspec.open(path, "rb").open() as f:
        content = f.read().decode("utf-8")

    stripped = content.strip()
    if not stripped:
        raise ValueError(f"POPE file is empty: {path}")

    if stripped[0] == "[":
        raw = json.loads(stripped)
    else:
        raw = [json.loads(line) for line in stripped.splitlines() if line.strip()]

    rows = []
    for i, item in enumerate(raw):
        question = (
            item.get("text") or item.get("query") or item.get("question") or ""
        ).strip()
        image = (item.get("image") or item.get("image_name") or "").strip()
        label = str(item.get("label", item.get("answer", ""))).strip().lower()
        if label not in {"yes", "no"}:
            if label.startswith("y"):
                label = "yes"
            elif label.startswith("n"):
                label = "no"
            else:
                continue

        if not question or not image:
            continue

        question_id = item.get("question_id", i + 1)
        rows.append(
            {
                "split": split,
                "sample_uid": f"{split}:{i}",
                "question_id": question_id,
                "question": question,
                "image": image,
                "label": label,
            }
        )

    if not rows:
        raise ValueError(f"No valid POPE rows found in: {path}")
    return rows


def load_pope_image_record_rows(root: str, splits, max_samples_per_split: int = 0):
    wanted_splits = set(splits)
    tar_paths = _glob_tar_shards(root)
    if not tar_paths:
        raise FileNotFoundError(f"No POPE tar shards found under: {root}")

    start = time.time()
    rows = []
    rows_per_split = {split: 0 for split in wanted_splits}
    image_records = 0
    for tar_path in tar_paths:
        with fsspec.open(tar_path, "rb").open() as f:
            with tarfile.open(fileobj=f, mode="r:*") as tar:
                members = {m.name: m for m in tar.getmembers() if m.isfile()}
                json_names = sorted(n for n in members if n.endswith(".json"))
                for json_name in json_names:
                    meta_f = tar.extractfile(members[json_name])
                    if meta_f is None:
                        continue
                    record = json.loads(meta_f.read().decode("utf-8"))
                    key = os.path.splitext(json_name)[0]
                    image_member = None
                    for ext in (".jpg", ".jpeg", ".png"):
                        candidate = f"{key}{ext}"
                        if candidate in members:
                            image_member = members[candidate]
                            break
                    if image_member is None:
                        continue

                    image_ref = {
                        "tar_path": tar_path,
                        "member": image_member.name,
                        "offset": int(image_member.offset_data),
                        "size": int(image_member.size),
                    }
                    image_records += 1
                    for i, q in enumerate(record.get("questions", [])):
                        split = str(q.get("split", "")).strip()
                        if split not in wanted_splits:
                            continue
                        if (
                            max_samples_per_split > 0
                            and rows_per_split[split] >= max_samples_per_split
                        ):
                            continue
                        question = str(q.get("question", "")).strip()
                        image = str(record.get("image", "")).strip()
                        label = str(q.get("label", "")).strip().lower()
                        if not question or not image or label not in {"yes", "no"}:
                            continue
                        rows.append(
                            {
                                "split": split,
                                "sample_uid": q.get("sample_uid", f"{split}:{len(rows)}"),
                                "question_id": q.get("question_id", i + 1),
                                "question": question,
                                "image": image,
                                "label": label,
                                "_image_record": image_ref,
                            }
                        )
                        rows_per_split[split] += 1
                    if (
                        max_samples_per_split > 0
                        and all(rows_per_split[s] >= max_samples_per_split for s in wanted_splits)
                    ):
                        break
                if (
                    max_samples_per_split > 0
                    and all(rows_per_split[s] >= max_samples_per_split for s in wanted_splits)
                ):
                    break
        if (
            max_samples_per_split > 0
            and all(rows_per_split[s] >= max_samples_per_split for s in wanted_splits)
        ):
            break

    if not rows:
        raise ValueError(f"No valid POPE rows found in image-record root: {root}")
    log_for_0(
        f"POPE image-record rows loaded: {len(rows)} questions, "
        f"{image_records} images from {len(tar_paths)} shards in {time.time() - start:.1f}s"
    )
    return rows


def _pope_image_record_rows_cache_path(config, pope_root, splits, max_samples_per_split):
    base_dir = getattr(
        config.eval,
        "pope_cache_dir",
        "/kmh-nfs-ssd-us-mount/data/cached/zhh/pope_eval",
    )
    ensure_eval_result_base_dir(base_dir)
    payload = json.dumps(
        {
            "pope_root": pope_root,
            "splits": list(splits),
            "max_samples_per_split": int(max_samples_per_split),
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return os.path.join(base_dir, f"pope_image_record_rows_{digest}.json")


def load_pope_image_record_rows_once(config, pope_root, splits, max_samples_per_split):
    """Loads tar metadata on host 0 only, then shares small row refs via NFS."""
    cache_path = _pope_image_record_rows_cache_path(
        config, pope_root, splits, max_samples_per_split
    )
    rows = None
    if jax.process_index() == 0:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            log_for_0(f"POPE image-record rows loaded from cache: {cache_path}")
        else:
            rows = load_pope_image_record_rows(
                pope_root, splits, max_samples_per_split=max_samples_per_split
            )
            tmp_path = f"{cache_path}.tmp.{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False)
            os.replace(tmp_path, cache_path)
            log_for_0(f"POPE image-record rows cache written: {cache_path}")
    mu.sync_global_devices("pope image-record rows cache ready")
    if rows is None:
        with open(cache_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    return rows


def resolve_image_path(image_name: str, image_root: str) -> str:
    if "://" in image_name:
        return image_name
    if os.path.isabs(image_name) and _path_exists(image_name):
        return image_name
    if image_root:
        return _join_path(image_root, image_name)
    return image_name


def _format_prompt(prompt_template: str, question: str) -> str:
    if "{question}" in prompt_template:
        prompt = prompt_template.format(question=question)
    elif "{}" in prompt_template:
        prompt = prompt_template.format(question)
    else:
        prompt = f"{prompt_template}{question}"
    if not prompt.endswith("\n"):
        prompt = prompt + "\n"
    return prompt


def preprocess_pope_sample(sample, transform, tokenizer, max_len, prompt_template):
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

    prompt = _format_prompt(prompt_template, question)
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    aux = dict(sample.get("aux") or {})
    aux["prompt"] = prompt

    eff_len = min(len(ids), max_len)
    pad_len = max_len - eff_len
    pad_id = tokenizer.special_tokens.PAD

    if pad_len > 0:
        input_ids_list = ids[:eff_len] + [pad_id] * pad_len
    else:
        input_ids_list = ids[:max_len]

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    prefix_len = torch.tensor(eff_len, dtype=torch.int32)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "prefix_len": prefix_len,
        "aux": aux,
    }


class POPEDataset(Dataset):
    def __init__(self, rows, config, tokenizer, image_root, prompt_template):
        self.rows = rows
        self.image_root = image_root
        self._image_record_streams = {}
        self.preprocess_fn = partial(
            preprocess_pope_sample,
            transform=get_transforms(
                config.dataset.image_size,
                is_train=False,
                resize_mode=getattr(config.dataset, "resize_mode", "letterbox"),
            ),
            tokenizer=tokenizer,
            max_len=config.dataset.max_txt_len,
            prompt_template=prompt_template,
        )

    def __len__(self):
        return len(self.rows)

    def _load_image_record(self, image_ref):
        tar_path = image_ref["tar_path"]
        stream = self._image_record_streams.get(tar_path)
        if stream is None:
            stream = fsspec.open(tar_path, "rb").open()
            self._image_record_streams[tar_path] = stream
        stream.seek(int(image_ref["offset"]))
        payload = stream.read(int(image_ref["size"]))
        return Image.open(io.BytesIO(payload)).convert("RGB")

    def __getitem__(self, idx):
        row = self.rows[idx]
        try:
            if "_image_record" in row:
                image = self._load_image_record(row["_image_record"])
            else:
                image_path = resolve_image_path(row["image"], self.image_root)
                with fsspec.open(image_path, "rb").open() as f:
                    image = Image.open(f).convert("RGB")
        except Exception:
            return None

        sample = {
            "jpg": image,
            "question": row["question"],
            "aux": {
                "split": row["split"],
                "sample_uid": row["sample_uid"],
                "question_id": row["question_id"],
                "question": row["question"],
                "image": row["image"],
                "label": row["label"],
            },
        }
        return self.preprocess_fn(sample)


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "prefix_len": torch.stack([b["prefix_len"] for b in batch]).to(torch.int32),
        "aux": [b["aux"] for b in batch],
    }


def _make_dummy_pope_batch(batch_size, image_size, max_len):
    return {
        "pixel_values": torch.zeros((batch_size, 3, image_size, image_size), dtype=torch.float32),
        "input_ids": torch.zeros((batch_size, max_len), dtype=torch.long),
        "prefix_len": torch.ones((batch_size,), dtype=torch.int32),
        "aux": [
            {
                "split": "",
                "sample_uid": "",
                "question_id": -1,
                "question": "",
                "image": "",
                "label": "yes",
                "prompt": "",
            }
            for _ in range(batch_size)
        ],
        "_all_pad": True,
    }


def normalize_pope_answer(text: str) -> str:
    text = "" if text is None else str(text)

    if text.find(".") != -1:
        text = text.split(".")[0]

    text = text.replace(",", "")
    words = text.split(" ")
    if "No" in words or "not" in words or "no" in words:
        return "no"
    return "yes"


def _label_to_yes_no(label: str) -> str:
    label = str(label).strip().lower()
    if label.startswith("n"):
        return "no"
    return "yes"


def compute_pope_metrics(records):
    if not records:
        return {
            "num_samples": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "yes_ratio": 0.0,
        }

    tp, tn, fp, fn = 0, 0, 0, 0
    num_yes_pred = 0

    for r in records:
        pred = 1 if r["pred_answer_norm"] == "yes" else 0
        label = 1 if r["gt_answer_norm"] == "yes" else 0
        num_yes_pred += pred

        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1

    total = len(records)
    acc = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    yes_ratio = num_yes_pred / total

    return {
        "num_samples": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_ratio,
    }


def vis_pope_qa(o):
    return (
        f"split: {o.get('split', '')}\n"
        f"question: {o.get('question', '')}\n"
        f"prompt: {o.get('prompt', '')}\n"
        f"answer: {o.get('pred_answer_raw', '')}\n"
        f"gt_answer: {o.get('gt_answer_norm', '')}"
    )


def eval_pope(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    pope_root = getattr(config.eval, "pope_root", None)
    if not pope_root:
        raise ValueError("config.eval.pope_root is required for POPE evaluation.")
    assert "💣" not in pope_root, f"💣 found in POPE path: {pope_root}"

    image_root = getattr(config.eval, "pope_image_root", "")
    if image_root:
        assert "💣" not in image_root, f"💣 found in POPE image root: {image_root}"

    splits = list(
        getattr(config.eval, "pope_splits", ["random", "popular", "adversarial"])
    )
    dataset_name = getattr(config.eval, "pope_dataset", "coco")
    prompt_template = getattr(
        config.eval, "pope_prompt_template", "{question}\nPlease answer yes or no.\n"
    )
    num_workers = int(getattr(config.eval, "pope_num_workers", 0))

    batch_size = config.eval.device_batch_size * jax.local_device_count()
    log_for_0(
        f"POPE eval: root={pope_root}, image_root={image_root}, splits={splits}, "
        f"batch_size={batch_size}"
    )

    all_outs = []
    sample_outputs = []
    image_record_rows = None
    image_record_shards = [] if _is_json_annotation_path(pope_root) else _glob_tar_shards(pope_root)
    max_samples = int(
        getattr(config.eval, "pope_max_samples_per_split", 0)
        or getattr(config.eval, "debug_max_samples", 0)
        or 0
    )
    if image_record_shards:
        image_record_rows = load_pope_image_record_rows_once(
            config, pope_root, splits, max_samples_per_split=max_samples
        )
        log_for_0(
            f"POPE eval using image-record tar package with "
            f"{len(image_record_shards)} shards"
        )

    for split in splits:
        if image_record_rows is not None:
            split_file = pope_root
            rows = [r for r in image_record_rows if r["split"] == split]
        else:
            split_file = resolve_pope_split_file(pope_root, split, dataset_name)
            rows = load_pope_questions(split_file, split)
        if max_samples > 0 and len(rows) > max_samples:
            rows = rows[:max_samples]
            log_for_0(f"POPE/{split}: capped to {len(rows)} samples")
        log_for_0(f"POPE/{split}: loaded {len(rows)} rows from {split_file}")

        dataset = POPEDataset(
            rows=rows,
            config=config,
            tokenizer=tokenizer,
            image_root=image_root,
            prompt_template=prompt_template,
        )
        sampler = DistributedEvalSampler(
            dataset,
            num_replicas=jax.process_count(),
            rank=jax.process_index(),
        )
        local_num_samples = len(sampler)
        local_num_steps = (local_num_samples + batch_size - 1) // batch_size
        log_for_0(
            f"POPE/{split}: global_samples={len(rows)}, "
            f"local_samples={local_num_samples}, local_steps={local_num_steps}"
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        loader_iter = iter(loader)
        samples_per_process = (len(dataset) + jax.process_count() - 1) // jax.process_count()
        fixed_num_steps = (samples_per_process + batch_size - 1) // batch_size
        log_for_0(
            f"POPE/{split}: fixed_num_steps={fixed_num_steps}, "
            f"samples_per_process={samples_per_process}, batch_size={batch_size}"
        )

        for i in range(fixed_num_steps):
            try:
                raw_batch = next(loader_iter)
                if not raw_batch:
                    raw_batch = _make_dummy_pope_batch(
                        batch_size, config.dataset.image_size, config.dataset.max_txt_len
                    )
            except StopIteration:
                raw_batch = _make_dummy_pope_batch(
                    batch_size, config.dataset.image_size, config.dataset.max_txt_len
                )

            if "aux" not in raw_batch:
                raw_batch["aux"] = []
            if len(raw_batch["aux"]) < batch_size:
                raw_batch["aux"].extend(
                    [
                        {
                            "split": split,
                            "sample_uid": "",
                            "question_id": -1,
                            "question": "",
                            "image": "",
                            "label": "yes",
                            "prompt": "",
                        }
                        for _ in range(batch_size - len(raw_batch["aux"]))
                    ]
                )

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

            for aux, out_str, is_pad in zip(
                batch["aux"], out_strs, batch["is_pad"].tolist()
            ):
                if is_pad:
                    continue

                pred_norm = normalize_pope_answer(out_str)
                gt_norm = _label_to_yes_no(aux.get("label", "yes"))

                record = {
                    "split": aux.get("split", split),
                    "sample_uid": aux.get("sample_uid", ""),
                    "question_id": aux.get("question_id", -1),
                    "image": aux.get("image", ""),
                    "question": aux.get("question", ""),
                    "prompt": aux.get("prompt", ""),
                    "pred_answer_raw": out_str,
                    "pred_answer_norm": pred_norm,
                    "gt_answer_norm": gt_norm,
                }
                all_outs.append(record)

                if len(sample_outputs) < 16:
                    sample_outputs.append(vis_pope_qa(record))

            if i % 20 == 0:
                log_for_0(
                    f"POPE/{split}: batch {i}, local collected {len(all_outs)} results"
                )

    mu.sync_global_devices("pope inference done")

    base_dir, result_prefix = eval_result_prefix(
        config,
        "pope_cache_dir",
        "/kmh-nfs-ssd-us-mount/data/cached/zhh/pope_eval",
        "pope",
    )
    ensure_eval_result_base_dir(base_dir)

    rank_file = f"{result_prefix}.results_{jax.process_index()}.json"
    with open(rank_file, "w", encoding="utf-8") as f:
        json.dump(all_outs, f, ensure_ascii=False, indent=2)

    mu.sync_global_devices("pope write done")

    if jax.process_index() == 0:
        merged = []
        for r in range(jax.process_count()):
            pf = f"{result_prefix}.results_{r}.json"
            if not os.path.exists(pf):
                raise FileNotFoundError(
                    f"During POPE eval, process {r} results file missing: {pf}"
                )
            with open(pf, "r", encoding="utf-8") as f:
                merged.extend(json.load(f))

        dedup = {}
        for rec in merged:
            uid = rec.get("sample_uid")
            if uid not in dedup:
                dedup[uid] = rec
        merged = list(dedup.values())

        split_metrics = {}
        for split in splits:
            split_records = [r for r in merged if r.get("split") == split]
            split_metrics[split] = compute_pope_metrics(split_records)

        f1s = [split_metrics[s]["f1"] for s in splits if s in split_metrics]
        accs = [split_metrics[s]["acc"] for s in splits if s in split_metrics]
        macro_f1 = float(np.mean(f1s) * 100.0) if f1s else 0.0
        macro_acc = float(np.mean(accs) * 100.0) if accs else 0.0

        metrics_dict = {
            "macro": {
                "f1": float(np.mean(f1s)) if f1s else 0.0,
                "acc": float(np.mean(accs)) if accs else 0.0,
                "f1_percent": macro_f1,
                "acc_percent": macro_acc,
            },
            "splits": split_metrics,
            "num_samples": len(merged),
        }

        with open(
            f"{result_prefix}.results_final.json", "w", encoding="utf-8"
        ) as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        with open(f"{result_prefix}.metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

        for split in splits:
            m = split_metrics[split]
            log_for_0(
                f"POPE/{split}: acc={m['acc'] * 100:.2f}%, "
                f"f1={m['f1'] * 100:.2f}%, yes_ratio={m['yes_ratio'] * 100:.2f}%"
            )
        log_for_0(f"POPE macro Acc: {macro_acc:.2f}%")
        log_for_0(f"POPE macro F1: {macro_f1:.2f}%")
        log_for_0(f"POPE merged results saved with prefix: {result_prefix}")
    else:
        log_for_all(f"Process {jax.process_index()} waiting for POPE scoring...")
        macro_f1 = 0.0
        metrics_dict = {"macro": {}, "splits": {}, "num_samples": 0}

    mu.sync_global_devices("pope eval done")
    return macro_f1, sample_outputs, metrics_dict
