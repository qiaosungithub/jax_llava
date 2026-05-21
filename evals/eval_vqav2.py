"""
VQAv2 evaluation for PaliGemma.
Data: WebDataset tar shards at gs://.../vqav2_image_records_wds/val2014/shard-*.tar
Format: each sample = {image_id}.jpg + {image_id}.json with {"image_id", "qas": [{question_id, question, answers, answer_type}, ...]}
"""
import json
import re
import os
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

# GCS support: input_pipeline.register_gcsfs() is called on import


# --- VQA accuracy (from big_vision, https://visualqa.org/evaluation.html) ---
REPLACEMENTS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "hows": "how's", "Im": "I'm", "Ive": "I've", "isnt": "isn't",
    "itd": "it'd", "itll": "it'll", "maam": "ma'am", "mustve": "must've", "neednt": "needn't",
    "oclock": "o'clock", "shant": "shan't", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "somebodys": "somebody's", "someones": "someone's",
    "somethings": "something's", "thats": "that's", "thered": "there'd", "theres": "there's",
    "theyd": "they'd", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed": "we'd", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whod": "who'd", "wholl": "who'll",
    "whos": "who's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "yall": "y'all", "youd": "you'd", "youll": "you'll", "youre": "you're", "youve": "you've",
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
PUNCT = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
ARTICLES = {"a", "an", "the"}


def stripspace_vqav2(txt):
    return txt.replace("\n", " ").replace("\t", " ").strip()


def postprocess_vqav2_text(txt):
    has_digit_comma = re.search(r"(\d)(\,)(\d)", txt) is not None
    out = txt
    for p in PUNCT:
        if has_digit_comma or f"{p} " in txt or f" {p}" in txt:
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = re.sub(r"(?<!\d)\.(?!\d)", "", out)
    words = []
    for word in out.lower().split():
        if word not in ARTICLES:
            words.append(REPLACEMENTS.get(word, word))
    return " ".join(words)


def vqa_accuracy_one(answer: str, gt_answers: list) -> float:
    """Per-question accuracy: avg over 10 leave-one-out GT sets. min(1, count/3)."""
    if not gt_answers or len(gt_answers) < 10:
        return 0.0
    gt_answers = [stripspace_vqav2(a) for a in gt_answers[:10]]
    answer = stripspace_vqav2(answer)
    if len(set(gt_answers)) > 1:
        answer = postprocess_vqav2_text(answer)
        gt_answers = [postprocess_vqav2_text(a) for a in gt_answers]
    gt_arr = np.array(gt_answers)
    matches = (answer == gt_arr)
    accs = []
    for i_leave_out in range(10):
        m = np.delete(matches, i_leave_out)
        accs.append(min(1.0, np.sum(m) / 3))
    return float(np.mean(accs))


def preprocess_vqa_sample(sample, transform, tokenizer, max_len):
    """Preprocess one (image, question) for VQA inference.
    Prompt format: '{question}\\n' with BOS, no EOS, padded to max_len.
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

    # Match your sanity + common big_vision formatting
    if not question.endswith("?"):
        question = question + "?"
    prompt = f"{question}\n"
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)

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
        "aux": sample.get("aux"),
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
        """List all VQAv2 shard URLs under root_url, supporting gs:// and local paths."""
        pattern = f"{self.root_url}/shard-*.tar"
        if self.root_url.startswith("gs://"):
            fs = fsspec.filesystem("gs")
            matched = sorted(fs.glob(pattern))
            # gcsfs.glob() typically returns paths without gs:// prefix.
            return [u if u.startswith("gs://") else f"gs://{u}" for u in matched]

        fs = fsspec.filesystem("file")
        return sorted(fs.glob(pattern))

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


def _make_dummy_vqav2_batch(batch_size, image_size, max_len):
    """Create a full-size dummy batch so all ranks keep compiled calls in sync."""
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

    # Force all ranks to execute the same number of sampling steps.
    total_vqav2_samples = int(getattr(config.eval, "vqav2_num_samples", 214354))
    samples_per_process = (total_vqav2_samples + jax.process_count() - 1) // jax.process_count()
    fixed_num_steps = (samples_per_process + batch_size - 1) // batch_size
    log_for_0(
        "VQAv2 fixed eval schedule: "
        f"total_samples={total_vqav2_samples}, "
        f"samples_per_process={samples_per_process}, "
        f"fixed_num_steps={fixed_num_steps}, "
        f"batch_size={batch_size}"
    )

    ALL_OUTS = []
    accuracies_by_type = {"yes/no": [], "number": [], "other": []}

    for i in range(fixed_num_steps):
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

        input_ids = batch["input_ids"]
        prefix_len = batch["prefix_len"]  # (LDC, B) or (B,) - per-sample prefix length

        out_strs = run_p_sample_step(p_sample_step, model, tokenizer, params, batch["pixel_values"], input_ids, prefix_len=prefix_len)

        for aux, out_str, is_pad in zip(batch["aux"], out_strs, batch["is_pad"].tolist()):
            if is_pad:
                continue
            qid = aux["question_id"]
            answers = aux.get("answers", [])
            answer_type = aux.get("answer_type", "other")
            if answer_type not in accuracies_by_type:
                answer_type = "other"

            # TODO: postprocess out_str, by normalizing output. e.g. remove 'the answer is ' and so on

            acc = vqa_accuracy_one(out_str, answers)
            accuracies_by_type[answer_type].append(acc)
            ALL_OUTS.append(
                {
                    "question_id": qid,
                    "question": aux.get("question", ""),
                    "answer": out_str,
                    "answers": answers,
                    "answer_type": answer_type,
                }
            )
        
        if i % 50 == 0:
            n = len(ALL_OUTS)
            logging.info(f"rank {jax.process_index()}, VQAv2 batch {i}, collected {n} results...")
    

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

    res_file = f"{result_prefix}.results_{jax.process_index()}.json"
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(ALL_OUTS, f, ensure_ascii=False, indent=2)

    mu.sync_global_devices("vqav2 write done")

    # Merge and recompute global accuracy on rank 0 (answers in each output for correct gather)
    if jax.process_index() == 0:
        all_results = []
        for r in range(jax.process_count()):
            pf = f"{result_prefix}.results_{r}.json"
            if os.path.exists(pf):
                all_results.extend(json.load(open(pf, encoding="utf-8")))
            else:
                log_for_0(f"Process {r} results file not found: {pf}")
                raise FileNotFoundError(f"During VQAv2 evaluation, process {r} results file not found: {pf}")

        # All ranks run the same eval stream for synchronized compiled calls, so
        # merged outputs contain duplicates. Keep one prediction per question_id.
        dedup_by_qid = {}
        for o in all_results:
            qid = o.get("question_id")
            if qid not in dedup_by_qid:
                dedup_by_qid[qid] = o
        all_results = list(dedup_by_qid.values())

        out_path = f"{result_prefix}.results_final.json"
        # Save merged (question_id, answer) for submission; full for debug
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([{"question_id": o["question_id"], "answer": o["answer"]} for o in all_results], f, ensure_ascii=False, indent=2)

        # Recompute global accuracy from merged (each sample has answers)
        all_accs = []
        for o in all_results:
            answers = o.get("answers", [])
            if answers:
                all_accs.append(vqa_accuracy_one(o["answer"], answers))
        overall_acc = np.mean(all_accs) * 100 if all_accs else 0.0
        log_for_0(f"VQAv2 results: {out_path} ({len(all_results)} samples)")
        log_for_0(f"VQAv2 accuracy: {overall_acc:.2f}%")
    else:
        log_for_all(f"Process {jax.process_index()} waiting for evaluation to finish...")
        overall_acc = 0.0

    mu.sync_global_devices("vqav2 eval done")
    return overall_acc, [vis_qa(o) for o in ALL_OUTS[:16]], []

def vis_qa(o):
    return (
        f'question: {o.get("question", "")}\n'
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
