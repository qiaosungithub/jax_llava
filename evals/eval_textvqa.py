"""
TextVQA evaluation for PaliGemma.
Data: WebDataset tar shards at gs://.../textvqa_image_records_wds/val/shard-*.tar
Format: each sample = {image_id}.jpg + {image_id}.json with
  {"image_id", "split", "qas": [{"question_id", "question", "answers": [...10 strings...], ...}, ...]}

Key differences from VQAv2:
  - Prefix: "{question}\\n" (TextVQA requires reading scene text)
  - answers: list of 10 plain strings (no dict wrapping)
  - No answer_type field
  - image_id is an OpenImages hex string, not an integer
  - Dataset is small (val: 5000 Qs over 3166 images, 1 shard) so we cannot shard by tar file.
    Instead every process reads the full shard, but only runs inference on its own slice
    (sample_index % num_processes == process_index). Dummy batches keep pmap calls in sync
    across all processes for steps where a process has no real work. Each process writes its
    own results file; rank-0 merges and deduplicates by question_id before computing accuracy.
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
from evals.vqa_scoring import vqa_accuracy_one as _shared_vqa_accuracy_one
from evals.eval_dist_util import (
    write_json,
    broadcast_merge_ok,
    collate_fn,
    gather_rank_json_results,
    write_rank_json_results,
)


def vqa_accuracy_one(answer: str, gt_answers: list) -> float:
    """TextVQA applies VQA text normalization even for unanimous answers."""
    return _shared_vqa_accuracy_one(answer, gt_answers, normalize_unanimous=True)


def _format_vqa_prompt(question: str) -> str:
    question = (question or "").strip()
    if question and not question.endswith("?"):
        question = question + "?"
    return f"{question}\nAnswer the question using a single word or phrase.\n"


def preprocess_textvqa_sample(sample, transform, tokenizer, max_len):
    """Preprocess one (image, question) for TextVQA inference.
    Prompt format matches LLaVA-1.5 short-answer VQA evaluation.
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


def expand_textvqa_sample(sample):
    """Expand one (jpg, json) into one item per question.
    TextVQA has ~1.58 questions per image on average, so expansion is necessary.
    """
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
        answers = qa.get("answers", [])  # plain list of strings
        out.append({
            "jpg": img,
            "question": qa.get("question", ""),
            "aux": {
                "question_id": int(qa.get("question_id", 0)),
                "question": qa.get("question", ""),
                "answers": answers,
            },
        })
    return out


class TextVQAIterableDataset(IterableDataset):
    """
    IterableDataset over TextVQA WebDataset shards.

    Since TextVQA val has only 1 shard, we cannot split shards across processes.
    Instead, every process reads the complete shard but only yields samples whose
    global index satisfies: index % num_processes == process_rank.
    This ensures no two processes handle the same question while keeping the
    union of all processes' outputs equal to the full dataset.
    """

    def __init__(self, root_url, config, tokenizer, num_processes=None, process_rank=None):
        self.root_url = root_url.rstrip("/")
        self.config = config
        self.tokenizer = tokenizer
        self.transform = get_transforms(
            config.dataset.image_size,
            is_train=False,
            resize_mode=getattr(config.dataset, "resize_mode", "letterbox"),
        )
        self.max_len = config.dataset.max_txt_len
        self.num_processes = num_processes if num_processes is not None else jax.process_count()
        self.process_rank = process_rank if process_rank is not None else jax.process_index()

    # def _list_urls(self):
    #     # Accept either a directory prefix (gs://.../val) or a direct tar URL.
    #     root = self.root_url
    #     if root.endswith(".tar"):
    #         return [root]
    #     pattern = f"{root}/shard-*.tar"
    #     if root.startswith("gs://"):
    #         fs = fsspec.filesystem("gs")
    #         matched = sorted(fs.glob(pattern))
    #         return [u if u.startswith("gs://") else f"gs://{u}" for u in matched]
    #     fs = fsspec.filesystem("file")
    #     return sorted(fs.glob(pattern))

    def __iter__(self):
        all_urls = [self.root_url]

        # All processes read the only 1 shard; sample-level interleaving is done below.
        ds = wds.WebDataset(all_urls, resampled=False, shardshuffle=False).decode("pil")

        sample_idx = 0
        for wds_sample in ds:
            for item in expand_textvqa_sample(wds_sample):
                # Only yield samples assigned to this process.
                if sample_idx % self.num_processes == self.process_rank:
                    out = preprocess_textvqa_sample(item, self.transform, self.tokenizer, self.max_len)
                    if out is not None:
                        yield out
                sample_idx += 1


def _make_dummy_textvqa_batch(batch_size, image_size, max_len):
    return {
        "pixel_values": torch.zeros((batch_size, 3, image_size, image_size), dtype=torch.float32),
        "input_ids": torch.zeros((batch_size, max_len), dtype=torch.long),
        "prefix_len": torch.ones((batch_size,), dtype=torch.int32),
        "aux": [{
            "question_id": -1,
            "question": "",
            "answers": [],
        } for _ in range(batch_size)],
        "_all_pad": True,
    }


def eval_textvqa(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    """
    Run TextVQA evaluation.

    Each process reads the full val shard but only runs inference on its assigned
    sample slice (sample_index % num_processes == process_rank). Dummy batches pad
    out the fixed step budget so all processes execute the same number of pmap calls.
    Rank-0 merges per-process result files, deduplicates by question_id, and computes
    the final VQA accuracy.
    """
    root_url = config.eval.textvqa_root
    assert '💣' not in root_url, f'💣 found in dataset path {root_url}'
    log_for_0(f"TextVQA eval: loading from {root_url}")

    dataset = TextVQAIterableDataset(root_url, config, tokenizer)
    batch_size = config.eval.device_batch_size * jax.local_device_count()
    log_for_0(f"Batch size: {batch_size}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
    )
    loader_iter = iter(loader)

    # Each process handles ceil(total / num_processes) samples at most.
    total_textvqa_samples = int(getattr(config.eval, "textvqa_num_samples", 5000))
    samples_per_process = (total_textvqa_samples + jax.process_count() - 1) // jax.process_count()
    fixed_num_steps = (samples_per_process + batch_size - 1) // batch_size
    log_for_0(
        "TextVQA fixed eval schedule: "
        f"total_samples={total_textvqa_samples}, "
        f"samples_per_process={samples_per_process}, "
        f"fixed_num_steps={fixed_num_steps}, "
        f"batch_size={batch_size}"
    )

    ALL_OUTS = []

    for i in range(fixed_num_steps):
        try:
            raw_batch = next(loader_iter)
            if not raw_batch:
                raw_batch = _make_dummy_textvqa_batch(
                    batch_size=batch_size,
                    image_size=config.dataset.image_size,
                    max_len=config.dataset.max_txt_len,
                )
        except StopIteration:
            raw_batch = _make_dummy_textvqa_batch(
                batch_size=batch_size,
                image_size=config.dataset.image_size,
                max_len=config.dataset.max_txt_len,
            )

        if "aux" not in raw_batch:
            raw_batch["aux"] = []
        if len(raw_batch["aux"]) < batch_size:
            raw_batch["aux"].extend([{
                "question_id": -1,
                "question": "",
                "answers": [],
            }] * (batch_size - len(raw_batch["aux"])))

        batch = prepare_batch_data(raw_batch, batch_size=batch_size)
        if raw_batch.get("_all_pad", False):
            batch["is_pad"] = np.ones((batch_size,), dtype=bool)

        input_ids = batch["input_ids"]
        prefix_len = batch["prefix_len"]

        out_strs = run_p_sample_step(p_sample_step, model, tokenizer, params, batch["pixel_values"], input_ids, prefix_len=prefix_len)

        for aux, out_str, is_pad in zip(batch["aux"], out_strs, batch["is_pad"].tolist()):
            if is_pad:
                continue
            ALL_OUTS.append({
                "question_id": aux["question_id"],
                "question": aux.get("question", ""),
                "prompt": aux.get("prompt", ""),
                "answer": out_str,
                "answers": aux.get("answers", []),
            })

        if i % 50 == 0:
            logging.info(f"rank {jax.process_index()}, TextVQA batch {i}, collected {len(ALL_OUTS)} results...")

    mu.sync_global_devices("textvqa inference done")

    base_dir, result_prefix = eval_result_prefix(
        config,
        "textvqa_cache_dir",
        "/kmh-nfs-ssd-us-mount/data/cached/zhh/textvqa_eval",
        "textvqa",
    )
    ensure_eval_result_base_dir(base_dir)

    write_rank_json_results(result_prefix, ALL_OUTS)

    mu.sync_global_devices("textvqa write done")

    merge_exception = None
    if jax.process_index() == 0:
        try:
            all_results = gather_rank_json_results(
                result_prefix,
                missing_file_msg="During TextVQA evaluation, process {rank} results file not found: {path}",
                log_missing=True,
            )

            # Each process handled a disjoint subset of questions, so there should be no
            # duplicates in normal operation. Dedup anyway as a safety net.
            dedup_by_qid = {}
            for o in all_results:
                qid = o.get("question_id")
                if qid not in dedup_by_qid:
                    dedup_by_qid[qid] = o
            all_results = list(dedup_by_qid.values())

            out_path = f"{result_prefix}.results_final.json"
            write_json(out_path, [{"question_id": o["question_id"], "answer": o["answer"]} for o in all_results], indent=2)

            all_accs = []
            for o in all_results:
                answers = o.get("answers", [])
                if answers:
                    all_accs.append(vqa_accuracy_one(o["answer"], answers))
            overall_acc = np.mean(all_accs) * 100 if all_accs else 0.0
            log_for_0(f"TextVQA results: {out_path} ({len(all_results)} samples)")
            log_for_0(f"TextVQA accuracy: {overall_acc:.2f}%")
        except Exception as exc:  # Keep every rank out of the final barrier on failure.
            logging.exception("TextVQA rank-0 merge/validation failed")
            merge_exception = exc
    else:
        log_for_all(f"Process {jax.process_index()} waiting for evaluation to finish...")
        overall_acc = 0.0

    broadcast_merge_ok(merge_exception, "TextVQA")

    mu.sync_global_devices("textvqa eval done")
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
        eval=SimpleNamespace(device_batch_size=4, textvqa_root="gs://kmh-gcp-us-east5/data/textvqa/textvqa_image_records_wds/val"),
        workdir_hash="test",
    )
    tokenizer = create_tokenizer("gemma3_270M")
    ds = TextVQAIterableDataset(config.eval.textvqa_root, config, tokenizer, num_processes=1, process_rank=0)
    it = iter(ds)
    for _ in range(3):
        s = next(it)
        print(s["aux"]["question_id"], s["aux"]["question"][:50], "...")
