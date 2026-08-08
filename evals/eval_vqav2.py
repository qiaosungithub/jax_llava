"""
VQAv2 evaluation for PaliGemma.
Data: WebDataset tar shards at gs://.../vqav2_image_records_wds/val2014/shard-*.tar
Format: each sample = {image_id}.jpg + {image_id}.json with {"image_id", "qas": [{question_id, question, answers, answer_type}, ...]}
"""
import json
from absl import logging

import fsspec
import jax
import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset, DataLoader
from jax.experimental import multihost_utils as mu

from utils.logging_util import log_for_0, log_for_all
from utils.eval_io_util import ensure_eval_result_base_dir, eval_result_prefix
from input_pipeline import get_transforms, prepare_batch_data
from evals.vqa_scoring import postprocess_vqav2_text, stripspace_vqav2, vqa_accuracy_one
from evals.eval_dist_util import (
    eval_glob,
    broadcast_merge_ok,
    collate_fn,
    gather_rank_json_results,
    write_rank_json_results,
)

# GCS support: input_pipeline.register_gcsfs() is called on import


def _format_vqa_prompt(question: str) -> str:
    question = (question or "").strip()
    if question and not question.endswith("?"):
        question = question + "?"
    return f"{question}\nAnswer the question using a single word or phrase.\n"


def preprocess_vqa_sample(sample, transform, tokenizer, max_len):
    """Preprocess one (image, question) for VQA inference.
    Prompt format matches LLaVA-1.5 short-answer VQA evaluation.
    prefix_len is the number of valid (non-pad) tokens in input_ids (clipped to max_len).
    """
    try:
        image = sample.get("jpg") or sample.get("png")
        if image is None:
            return None
        pixel_values = transform(image)
    except Exception:
        return None

    question = (sample.get("question", "") or "").strip()
    if not question:
        return None

    prompt = _format_vqa_prompt(question)
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    aux = dict(sample.get("aux") or {})
    aux["prompt"] = prompt

    # Effective (clipped) length
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


def expand_vqa_sample(sample):
    """Expand one (jpg, json) into list of (image, qa) for each question."""
    j = sample.get("json")
    if j is None:
        return []
    if isinstance(j, bytes):
        j = json.loads(j.decode("utf-8"))
    qas = j.get("qas", [])
    img = sample.get("jpg")
    if img is None or not qas:
        return []
    out = []
    for qa in qas:
        answers = [a.get("answer", a) if isinstance(a, dict) else a for a in qa.get("answers", [])]
        out.append({
            "jpg": img,
            "question": qa.get("question", ""),
            "aux": {
                "question_id": int(qa.get("question_id", 0)),
                "question": qa.get("question", ""),
                "answers": answers,
                "answer_type": qa.get("answer_type", "other"),
            },
        })
    return out


class VQAv2IterableDataset(IterableDataset):
    """IterableDataset over VQAv2 WebDataset shards. Expands each image's QAs."""

    def __init__(self, root_url, config, tokenizer, num_shards=None, shard_rank=None):
        self.root_url = root_url.rstrip("/")
        self.config = config
        self.tokenizer = tokenizer
        self.transform = get_transforms(
            config.dataset.image_size,
            is_train=False,
            resize_mode=getattr(config.dataset, "resize_mode", "letterbox"),
        )
        self.max_len = config.dataset.max_txt_len
        self.num_shards = num_shards or jax.process_count()
        self.shard_rank = shard_rank if shard_rank is not None else jax.process_index()

    def _list_urls(self):
        """List all VQAv2 shard URLs under root_url: /cns/, gs:// or local.

        The /cns/ case has to go through gfile. fsspec's "file" backend cannot
        see Colossus and reports no matches rather than failing, so a replica
        holding 128 shards looked empty and the eval died with "No VQAv2 shards
        found" naming the very directory they were in.
        """
        return eval_glob(f"{self.root_url}/shard-*.tar")

    def __iter__(self):
        all_urls = self._list_urls()
        if not all_urls:
            raise FileNotFoundError(f"No VQAv2 shards found under {self.root_url}")

        my_urls = all_urls[self.shard_rank::self.num_shards]
        if not my_urls:
            return

        ds = wds.WebDataset(my_urls, resampled=False, shardshuffle=False).decode("pil")

        for sample in ds:
            for item in expand_vqa_sample(sample):
                out = preprocess_vqa_sample(item, self.transform, self.tokenizer, self.max_len)
                if out is not None:
                    yield out


def _make_dummy_vqav2_batch(batch_size, image_size, max_len):
    """Create a full-size dummy batch so all ranks can keep pmap calls in sync."""
    return {
        "pixel_values": torch.zeros((batch_size, 3, image_size, image_size), dtype=torch.float32),
        "input_ids": torch.zeros((batch_size, max_len), dtype=torch.long),
        "prefix_len": torch.ones((batch_size,), dtype=torch.int32),
        "aux": [{
            "question_id": -1,
            "question": "",
            "answers": [],
            "answer_type": "other",
        } for _ in range(batch_size)],
        "_all_pad": True,
    }


def _resolve_vqav2_num_samples(config):
    full_total = int(getattr(config.eval, "vqav2_num_samples", 214354))
    eval_suffix = str(getattr(config.eval, "current_eval_suffix", "")).lower()
    if eval_suffix != "online":
        return full_total

    explicit = getattr(config.eval, "online_vqav2_num_samples", None)
    if explicit not in (None, ""):
        online_total = int(explicit)
    else:
        fraction = float(
            getattr(
                config.eval,
                "online_vqav2_sample_fraction",
                getattr(config.eval, "online_eval_sample_fraction", 1.0),
            )
        )
        online_total = int(np.ceil(full_total * fraction))
    online_total = max(1, min(full_total, online_total))
    log_for_0(
        f"VQAv2 online sample cap: {online_total}/{full_total} "
        f"({online_total / max(full_total, 1):.2%})"
    )
    return online_total


def _allocate_vqav2_batch(local_counts, remaining):
    """Deterministically accept up to ``remaining`` interleaved rank samples."""
    local_counts = [int(value) for value in local_counts]
    if remaining < 0 or any(value < 0 for value in local_counts):
        raise ValueError(f"Invalid VQAv2 allocation: counts={local_counts}, remaining={remaining}")
    accepted = [0] * len(local_counts)
    left = int(remaining)
    for local_index in range(max(local_counts, default=0)):
        for rank, count in enumerate(local_counts):
            if left == 0:
                return accepted
            if local_index < count:
                accepted[rank] += 1
                left -= 1
    return accepted


def _gather_vqav2_batch_counts(local_count):
    gathered = mu.process_allgather(np.asarray(local_count, dtype=np.int32))
    return [int(value) for value in np.asarray(jax.device_get(gathered)).reshape(-1)]


def eval_vqav2(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    """
    Run VQAv2 evaluation.
    Uses same interfaces as eval_cider: prepare_batch_data, run_p_sample_step, collate_fn.
    """
    # root_url = getattr(config.eval, "vqav2_root", "gs://kmh-gcp-us-east5/data/vqav2/vqav2_image_records_wds/val2014")
    root_url = config.eval.vqav2_root
    assert '💣' not in root_url, f'💣 found in dataset path {root_url}'
    log_for_0(f"VQAv2 eval: loading from {root_url}")

    dataset = VQAv2IterableDataset(root_url, config, tokenizer)
    batch_size = config.eval.device_batch_size * jax.local_device_count()
    log_for_0(f"Batch size: {batch_size}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # IterableDataset + multiprocess can be tricky; 0 is safest
        collate_fn=collate_fn,
    )
    loader_iter = iter(loader)

    total_vqav2_samples = _resolve_vqav2_num_samples(config)
    if total_vqav2_samples <= 0:
        raise ValueError(f"VQAv2 target sample count must be positive, got {total_vqav2_samples}")
    log_for_0(
        "VQAv2 synchronized exact-target schedule: "
        f"total_samples={total_vqav2_samples}, process_count={jax.process_count()}, "
        f"batch_size={batch_size}"
    )

    ALL_OUTS = []
    accuracies_by_type = {"yes/no": [], "number": [], "other": []}

    global_collected = 0
    i = 0
    while global_collected < total_vqav2_samples:
        try:
            raw_batch = next(loader_iter)
            if not raw_batch:
                raw_batch = _make_dummy_vqav2_batch(
                    batch_size=batch_size,
                    image_size=config.dataset.image_size,
                    max_len=config.dataset.max_txt_len,
                )
        except StopIteration:
            raw_batch = _make_dummy_vqav2_batch(
                batch_size=batch_size,
                image_size=config.dataset.image_size,
                max_len=config.dataset.max_txt_len,
            )

        local_real_count = 0 if raw_batch.get("_all_pad", False) else len(raw_batch.get("aux", []))
        local_counts = _gather_vqav2_batch_counts(local_real_count)
        if sum(local_counts) == 0:
            raise RuntimeError(
                "VQAv2 dataset exhausted before reaching configured target: "
                f"collected={global_collected}, target={total_vqav2_samples}"
            )
        accepted_counts = _allocate_vqav2_batch(
            local_counts,
            total_vqav2_samples - global_collected,
        )
        local_accepted = accepted_counts[jax.process_index()]

        # Keep aux aligned with is_pad length so zip can cover the whole batch.
        if "aux" not in raw_batch:
            raw_batch["aux"] = []
        if len(raw_batch["aux"]) < batch_size:
            raw_batch["aux"].extend([{
                "question_id": -1,
                "question": "",
                "answers": [],
                "answer_type": "other",
            }] * (batch_size - len(raw_batch["aux"])))

        batch = prepare_batch_data(raw_batch, batch_size=batch_size)
        if raw_batch.get("_all_pad", False):
            batch["is_pad"] = np.ones((batch_size,), dtype=bool)
        elif local_accepted < local_real_count:
            batch["is_pad"][local_accepted:local_real_count] = True

        input_ids = batch["input_ids"]
        prefix_len = batch["prefix_len"]  # (LDC, B) or (B,) - per-sample prefix length

        out_strs = run_p_sample_step(p_sample_step, model, tokenizer, params, batch["pixel_values"], input_ids, prefix_len=prefix_len)

        local_collected = 0
        for aux, out_str, is_pad in zip(batch["aux"], out_strs, batch["is_pad"].tolist()):
            if is_pad:
                continue
            local_collected += 1
            qid = aux["question_id"]
            answers = aux.get("answers", [])
            answer_type = aux.get("answer_type", "other")
            if answer_type not in accuracies_by_type:
                answer_type = "other"

            acc = vqa_accuracy_one(out_str, answers)
            accuracies_by_type[answer_type].append(acc)
            ALL_OUTS.append(
                {
                    "question_id": qid,
                    "question": aux.get("question", ""),
                    "prompt": aux.get("prompt", ""),
                    "answer": out_str,
                    "answers": answers,
                    "answer_type": answer_type,
                }
            )
        
        collected_counts = _gather_vqav2_batch_counts(local_collected)
        if collected_counts != accepted_counts:
            raise RuntimeError(
                "VQAv2 accepted/output count mismatch across ranks: "
                f"accepted={accepted_counts}, collected={collected_counts}"
            )
        global_collected += sum(accepted_counts)
        if i % 50 == 0:
            logging.info(
                "rank %d, VQAv2 batch %d, local=%d, global=%d/%d",
                jax.process_index(),
                i,
                len(ALL_OUTS),
                global_collected,
                total_vqav2_samples,
            )
        i += 1
    

    # All-reduce for multi-host
    mu.sync_global_devices("vqav2 inference done")

    # Save results (same pattern as eval_cider: zhh shared)
    base_dir, result_prefix = eval_result_prefix(
        config,
        "vqav2_cache_dir",
        "/kmh-nfs-ssd-us-mount/data/cached/zhh/vqav2_eval",
        "vqav2",
    )
    ensure_eval_result_base_dir(base_dir)

    write_rank_json_results(result_prefix, ALL_OUTS)

    mu.sync_global_devices("vqav2 write done")

    # Merge and recompute global accuracy on rank 0 (answers in each output for correct gather)
    merge_exception = None
    if jax.process_index() == 0:
        try:
            all_results = gather_rank_json_results(
                result_prefix,
                missing_file_msg="During VQAv2 evaluation, process {rank} results file not found: {path}",
            )

            raw_result_count = len(all_results)
            dedup_by_qid = {}
            for o in all_results:
                qid = o.get("question_id")
                if qid in (None, -1):
                    raise ValueError(f"Invalid VQAv2 question_id in result: {qid!r}")
                if qid in dedup_by_qid:
                    raise ValueError(f"Duplicate VQAv2 question_id across rank outputs: {qid}")
                dedup_by_qid[qid] = o
            all_results = list(dedup_by_qid.values())
            if raw_result_count != total_vqav2_samples or len(all_results) != total_vqav2_samples:
                raise ValueError(
                    "VQAv2 result count mismatch: "
                    f"raw={raw_result_count}, unique={len(all_results)}, "
                    f"expected={total_vqav2_samples}"
                )

            out_path = f"{result_prefix}.results_final.json"
            # Save merged (question_id, answer) for submission; full for debug
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    [{"question_id": o["question_id"], "answer": o["answer"]} for o in all_results],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            # Recompute and round exactly like the official VQA-v2 evaluator.
            all_accs = []
            for o in all_results:
                answers = o.get("answers", [])
                if answers:
                    all_accs.append(vqa_accuracy_one(o["answer"], answers))
            overall_acc = round(float(np.mean(all_accs) * 100), 2) if all_accs else 0.0
            log_for_0(f"VQAv2 results: {out_path} ({len(all_results)} samples)")
            log_for_0(f"VQAv2 accuracy: {overall_acc:.2f}%")
        except Exception as exc:  # Keep every rank out of the final barrier on failure.
            logging.exception("VQAv2 rank-0 merge/validation failed")
            merge_exception = exc
    else:
        log_for_all(f"Process {jax.process_index()} waiting for evaluation to finish...")
        overall_acc = 0.0

    broadcast_merge_ok(merge_exception, "VQAv2")

    mu.sync_global_devices("vqav2 eval done")
    return overall_acc, [vis_qa(o) for o in ALL_OUTS[:16]], []

def vis_qa(o):
    return (
        f'question: {o.get("question", "")}\n'
        f'prompt: {o.get("prompt", "")}\n'
        f'answer: {o.get("answer", "")}\n'
        f'gt_answers: {o.get("answers", [])}'
    )

if __name__ == "__main__":
    # Quick test
    from utils.llm_util import create_tokenizer
    from types import SimpleNamespace
    config = SimpleNamespace(
        dataset=SimpleNamespace(image_size=224, max_txt_len=64),
        eval=SimpleNamespace(device_batch_size=4, vqav2_root="gs://kmh-gcp-us-east5/data/vqav2/vqav2_image_records_wds/val2014"),
        workdir_hash="test",
    )
    tokenizer = create_tokenizer("gemma3_270M")
    ds = VQAv2IterableDataset(config.eval.vqav2_root, config, tokenizer, num_shards=1, shard_rank=0)
    it = iter(ds)
    for _ in range(3):
        s = next(it)
        print(s["aux"]["question_id"], s["aux"]["question"][:50], "...")
