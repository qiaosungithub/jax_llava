"""MMBench evaluation and TEST submission export for PaliGemma.

The official MMBench repository delegates evaluation to VLMEvalKit. This file
implements the small subset we need in this JAX training loop: TSV loading,
prompt building, exact option extraction, circular scoring, and TEST xlsx export.
"""

import base64
import io
import json
import os
import re
import ssl
import string
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from xml.sax.saxutils import escape

import jax
import numpy as np
import pandas as pd
import torch
from PIL import Image
from jax.experimental import multihost_utils as mu
from torch.utils.data import DataLoader, Dataset, Sampler

from input_pipeline import get_transforms, prepare_batch_data
from utils.logging_util import log_for_0, log_for_all
from utils.eval_io_util import ensure_eval_result_base_dir, eval_result_prefix


DEFAULT_DEV_URL = "https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv"
DEFAULT_TEST_URL = "https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_EN.tsv"

def _is_url(path):
    return isinstance(path, str) and path.startswith(("http://", "https://"))


def _safe_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    value = str(value).strip()
    if value.lower() in ("nan", "none"):
        return ""
    return value


def _download_file(url, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp_path = local_path + ".tmp"
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=ctx, timeout=120) as src, open(tmp_path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    os.replace(tmp_path, local_path)


def _resolve_tsv_path(root, cache_dir, filename):
    if not _is_url(root):
        return root

    local_path = os.path.join(cache_dir, filename)
    if jax.process_index() == 0 and not os.path.exists(local_path):
        log_for_0(f"Downloading MMBench TSV from {root} to {local_path}")
        _download_file(root, local_path)
    mu.sync_global_devices(f"mmbench download {filename}")

    while not os.path.exists(local_path):
        time.sleep(0.5)
    return local_path


def _read_tsv(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def _valid_option_columns(row):
    return [c for c in string.ascii_uppercase if c in row and _safe_text(row[c])]


_DEFAULT_MMBENCH_SUFFIX = "Answer with the option letter only.\n"


def build_mmbench_prompt(row, prefix="", suffix=None):
    if suffix is None:
        suffix = _DEFAULT_MMBENCH_SUFFIX

    hint = _safe_text(row.get("hint"))
    question = _safe_text(row.get("question"))
    options = {c: _safe_text(row.get(c)) for c in _valid_option_columns(row)}

    body = ""
    if hint:
        body += f"Hint: {hint}\n"
    body += f"Question: {question}\n"
    if options:
        body += "Options:\n"
        for key, value in options.items():
            body += f"{key}. {value}\n"
        body += suffix
    return prefix + body


def _token_len(tokenizer, text):
    return len(tokenizer.encode(text, add_bos=True, add_eos=False))


def _clip_text(text, max_chars):
    text = _safe_text(text)
    if max_chars is None or len(text) <= max_chars:
        return text
    max_chars = max(0, int(max_chars))
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _row_with_limits(row, *, keep_hint, question_chars=None, option_chars=None):
    out = dict(row)
    if not keep_hint:
        out["hint"] = ""
    if question_chars is not None:
        out["question"] = _clip_text(out.get("question"), question_chars)
    if option_chars is not None:
        for key in _valid_option_columns(out):
            out[key] = _clip_text(out.get(key), option_chars)
    return out


def _fit_mmbench_prompt(row, tokenizer, max_len, prefix, suffix):
    """Build a prompt that fits max_len while preserving answer options.

    Long MMBench rows often overflow because of hints or verbose choices. The
    important part for multiple choice scoring is the question, option labels,
    option text, and final "letter only" instruction. So we drop hints first,
    then progressively shorten question/options instead of blindly chopping off
    the tail where the options and instruction live.
    """
    attempts = []
    full_prompt = build_mmbench_prompt(row, prefix=prefix, suffix=suffix)
    attempts.append((full_prompt, False, _token_len(tokenizer, full_prompt)))

    no_hint = _row_with_limits(row, keep_hint=False)
    prompt = build_mmbench_prompt(no_hint, prefix=prefix, suffix=suffix)
    attempts.append((prompt, True, _token_len(tokenizer, prompt)))

    # Progressively tighten text fields. Question gets twice the option budget.
    for option_chars in (512, 384, 256, 192, 128, 96, 64, 48, 32, 24, 16):
        fitted = _row_with_limits(
            row,
            keep_hint=False,
            question_chars=option_chars * 2,
            option_chars=option_chars,
        )
        prompt = build_mmbench_prompt(fitted, prefix=prefix, suffix=suffix)
        attempts.append((prompt, True, _token_len(tokenizer, prompt)))

    for prompt, truncated, n_tokens in attempts:
        if n_tokens <= max_len:
            return prompt, tokenizer.encode(prompt, add_bos=True, add_eos=False), truncated, n_tokens

    # Last resort for pathological rows or very small max_len: keep the shortest
    # structured prompt and let token truncation apply only after options were
    # aggressively shortened.
    prompt, _, n_tokens = attempts[-1]
    return prompt, tokenizer.encode(prompt, add_bos=True, add_eos=False), True, n_tokens


def _encode_prompt(row, tokenizer, max_len, prefix, suffix):
    prompt, ids, truncated, raw_len = _fit_mmbench_prompt(
        row,
        tokenizer,
        max_len,
        prefix,
        suffix,
    )

    eff_len = min(len(ids), max_len)
    pad_id = tokenizer.special_tokens.PAD
    input_ids = ids[:eff_len] + [pad_id] * (max_len - eff_len)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(eff_len, dtype=torch.int32),
        prompt,
        truncated or raw_len > max_len,
        int(raw_len),
    )


def can_infer_option(answer, choices):
    answer = str(answer)
    if "Failed to obtain answer via API" in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    ]
    for err in reject_to_answer:
        if err in answer:
            return "Z"

    answer_mod = answer
    for char in ".()[],:;!*#{}":
        answer_mod = answer_mod.replace(char, " ")
    splits = [x.strip() for x in answer_mod.split()]

    count = sum(1 for c in choices if c in splits)
    if count == 1:
        for ch in choices:
            if ch in splits and splits.index(ch) > (len(splits) - 5):
                return ch
    if count == 0 and sum(1 for c in ("Z", "") if c in splits) == 1:
        return "Z"
    return False


def can_infer_text(answer, choices):
    answer = str(answer).lower()
    if len(answer) > 2 * sum(len(str(v)) for v in choices.values()):
        return False
    lowered = {k: str(v).lower() for k, v in choices.items()}
    cands = [k for k, v in lowered.items() if v and v in answer]
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    copt = can_infer_option(str(answer), choices)
    return copt if copt else can_infer_text(str(answer), choices)


def extract_prediction_option(prediction, choices):
    ret = can_infer(prediction, choices)
    return ret if ret else "Z"


def _group_index(row, index):
    value = _safe_text(row.get("g_index"))
    if value:
        return int(float(value))
    return int(index % 1_000_000)


class DistributedEvalSampler(Sampler):
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


class MMBenchDataset(Dataset):
    def __init__(self, tsv_path, config, tokenizer):
        self.tsv_path = tsv_path
        self.data = _read_tsv(tsv_path)
        self.config = config
        self.tokenizer = tokenizer
        self.transform = get_transforms(
            config.dataset.image_size,
            is_train=False,
            resize_mode=getattr(config.dataset, "resize_mode", "letterbox"),
        )
        self.max_len = int(getattr(config.eval, "mmbench_max_txt_len", config.dataset.max_txt_len))
        self.prompt_prefix = getattr(config.eval, "mmbench_prompt_prefix", "")
        self.prompt_suffix = getattr(
            config.eval,
            "mmbench_prompt_suffix",
            _DEFAULT_MMBENCH_SUFFIX,
        )
        self.image_map = {}
        if "image" in self.data:
            for _, row in self.data.iterrows():
                self.image_map[str(row["index"])] = _safe_text(row.get("image"))
        self.image_cache = {}

    def __len__(self):
        return len(self.data)

    def _resolve_image_value(self, row):
        value = _safe_text(row.get("image"))
        if not value:
            return None, None
        if len(value) <= 64 and value in self.image_map:
            return value, self.image_map[value]
        return str(row["index"]), value

    def _load_image(self, row):
        key, value = self._resolve_image_value(row)
        if value is None:
            return None
        if key in self.image_cache:
            return self.image_cache[key]

        try:
            if os.path.exists(value):
                image = Image.open(value).convert("RGB")
            else:
                image = Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
        except Exception:
            return None

        self.image_cache[key] = image
        return image

    def __getitem__(self, idx):
        row = self.data.iloc[idx].to_dict()
        image = self._load_image(row)
        if image is None:
            return None
        try:
            pixel_values = self.transform(image)
        except Exception:
            return None

        input_ids, prefix_len, prompt, prompt_truncated, prompt_token_len = _encode_prompt(
            row,
            self.tokenizer,
            self.max_len,
            self.prompt_prefix,
            self.prompt_suffix,
        )
        choices = {c: _safe_text(row.get(c)) for c in _valid_option_columns(row)}
        index = int(row["index"])
        aux = {
            "index": index,
            "g_index": _group_index(row, index),
            "question": _safe_text(row.get("question")),
            "hint": _safe_text(row.get("hint")),
            "choices": choices,
            "answer": _safe_text(row.get("answer")),
            "category": _safe_text(row.get("category")),
            "l2-category": _safe_text(row.get("l2-category")),
            "split": _safe_text(row.get("split")) or "dev",
            "prompt": prompt,
            "prompt_truncated": bool(prompt_truncated),
            "prompt_token_len": int(prompt_token_len),
            "prompt_max_len": int(self.max_len),
        }
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


def _make_dummy_mmbench_batch(batch_size, image_size, max_len):
    return {
        "pixel_values": torch.zeros((batch_size, 3, image_size, image_size), dtype=torch.float32),
        "input_ids": torch.zeros((batch_size, max_len), dtype=torch.long),
        "prefix_len": torch.ones((batch_size,), dtype=torch.int32),
        "aux": [_dummy_aux() for _ in range(batch_size)],
        "_all_pad": True,
    }


def _dummy_aux():
    return {
        "index": -1,
        "g_index": -1,
        "question": "",
        "hint": "",
        "choices": {},
        "answer": "",
        "category": "",
        "l2-category": "",
        "split": "",
        "prompt": "",
        "prompt_truncated": False,
        "prompt_token_len": 0,
        "prompt_max_len": 0,
    }


def _result_prefix(config, split_name):
    return eval_result_prefix(
        config,
        "mmbench_cache_dir",
        "/kmh-nfs-ssd-us-mount/data/cached/zhh/mmbench_eval",
        "mmbench",
        split_name.lower(),
    )


def _run_mmbench_predictions(p_sample_step, run_p_sample_step, model, tokenizer, params, config, split_name, root):
    cache_dir = getattr(config.eval, "mmbench_data_cache_dir", "/kmh-nfs-ssd-us-mount/data/cached/zhh/mmbench_data")
    tsv_path = _resolve_tsv_path(root, cache_dir, f"{split_name}.tsv")
    dataset = MMBenchDataset(tsv_path, config, tokenizer)
    device_batch_size = int(getattr(config.eval, "mmbench_device_batch_size", config.eval.device_batch_size))
    batch_size = device_batch_size * jax.local_device_count()
    sampler = DistributedEvalSampler(dataset, num_replicas=jax.process_count(), rank=jax.process_index())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(getattr(config.eval, "mmbench_num_workers", 0)),
        collate_fn=collate_fn,
    )
    loader_iter = iter(loader)

    samples_per_process = (len(dataset) + jax.process_count() - 1) // jax.process_count()
    fixed_num_steps = (samples_per_process + batch_size - 1) // batch_size
    log_for_0(
        f"MMBench {split_name}: rows={len(dataset)}, samples_per_process={samples_per_process}, "
        f"fixed_num_steps={fixed_num_steps}, batch_size={batch_size}"
    )

    all_outs = []
    truncated_prompts = 0
    max_len = int(getattr(config.eval, "mmbench_max_txt_len", config.dataset.max_txt_len))
    for i in range(fixed_num_steps):
        try:
            raw_batch = next(loader_iter)
            if not raw_batch:
                raw_batch = _make_dummy_mmbench_batch(batch_size, config.dataset.image_size, max_len)
        except StopIteration:
            raw_batch = _make_dummy_mmbench_batch(batch_size, config.dataset.image_size, max_len)

        if "aux" not in raw_batch:
            raw_batch["aux"] = []
        if len(raw_batch["aux"]) < batch_size:
            raw_batch["aux"].extend([_dummy_aux() for _ in range(batch_size - len(raw_batch["aux"]))])

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
            pred = str(out_str).strip()
            pred_option = extract_prediction_option(pred, aux.get("choices", {}))
            item = dict(aux)
            item["prediction"] = pred
            item["pred_option"] = pred_option
            all_outs.append(item)
            truncated_prompts += int(bool(aux.get("prompt_truncated", False)))

        if i % 20 == 0:
            log_for_0(f"MMBench {split_name} batch {i}, collected {len(all_outs)} results...")

    log_for_0(
        f"MMBench {split_name}: prompt fitting truncated/dropped hint for "
        f"{truncated_prompts}/{len(all_outs)} local samples (max_len={max_len})."
    )
    mu.sync_global_devices(f"mmbench {split_name} inference done")

    base_dir, result_prefix = _result_prefix(config, split_name)
    ensure_eval_result_base_dir(base_dir)

    if not os.access(base_dir, os.W_OK | os.X_OK):
        raise PermissionError(
            f"MMBench result cache dir is not writable on process {jax.process_index()}: {base_dir}."
        )

    rank_file = f"{result_prefix}.results_{jax.process_index()}.json"
    with open(rank_file, "w", encoding="utf-8") as f:
        json.dump(all_outs, f, ensure_ascii=False, indent=2)

    mu.sync_global_devices(f"mmbench {split_name} write done")

    merged = None
    if jax.process_index() == 0:
        merged = []
        for rank in range(jax.process_count()):
            path = f"{result_prefix}.results_{rank}.json"
            if not os.path.exists(path):
                raise FileNotFoundError(f"During MMBench eval, process {rank} results file missing: {path}")
            with open(path, "r", encoding="utf-8") as f:
                merged.extend(json.load(f))

        dedup = {}
        for item in merged:
            dedup[int(item["index"])] = item
        merged = [dedup[idx] for idx in sorted(dedup)]
        with open(f"{result_prefix}.results_final.json", "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        log_for_0(f"MMBench {split_name} merged results saved with prefix: {result_prefix}")
    else:
        log_for_all(f"Process {jax.process_index()} waiting for MMBench {split_name} merge...")

    mu.sync_global_devices(f"mmbench {split_name} merge done")
    return merged if jax.process_index() == 0 else all_outs


def _score_mmbench(results):
    by_group = defaultdict(list)
    main_by_group = {}
    for item in results:
        if not item.get("answer"):
            continue
        item = dict(item)
        item["row_hit"] = int(item.get("pred_option") == item.get("answer"))
        g_index = int(item["g_index"])
        by_group[g_index].append(item)
        if int(item["index"]) == g_index:
            main_by_group[g_index] = item

    scored_rows = []
    for g_index, rows in by_group.items():
        hit = int(all(r["row_hit"] for r in rows))
        main = dict(main_by_group.get(g_index, rows[0]))
        main["hit"] = hit
        main["num_circular_rows"] = len(rows)
        scored_rows.append(main)

    if not scored_rows:
        return 0.0, {"Overall": 0.0, "num_questions": 0}, []

    overall = float(np.mean([r["hit"] for r in scored_rows]) * 100.0)
    metrics = {"Overall": overall, "num_questions": len(scored_rows)}

    for group_key in ("l2-category", "category", "split"):
        buckets = defaultdict(list)
        for row in scored_rows:
            key = row.get(group_key) or "none"
            buckets[key].append(row["hit"])
        for key, vals in buckets.items():
            metrics[f"{group_key}/{key}"] = float(np.mean(vals) * 100.0)

    metrics["option_match_fail_rows"] = int(sum(1 for r in results if r.get("pred_option") == "Z"))
    metrics["circular_group_size_counts"] = dict(Counter(len(v) for v in by_group.values()))
    return overall, metrics, scored_rows


def _vis_mmbench(item):
    choices = item.get("choices", {})
    options = "\n".join([f"{k}. {v}" for k, v in choices.items()])
    return (
        f"question: {item.get('question', '')}\n"
        f"prompt: {item.get('prompt', '')}\n"
        f"options:\n{options}\n"
        f"prediction: {item.get('prediction', '')}\n"
        f"pred_option: {item.get('pred_option', '')}\n"
        f"answer: {item.get('answer', '')}"
    )


def eval_mmbench(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    root = getattr(config.eval, "mmbench_root", DEFAULT_DEV_URL)
    results = _run_mmbench_predictions(
        p_sample_step,
        run_p_sample_step,
        model,
        tokenizer,
        params,
        config,
        split_name="MMBench_DEV_EN",
        root=root,
    )

    if jax.process_index() == 0:
        overall, metric_dict, scored_rows = _score_mmbench(results)
        base_dir, result_prefix = _result_prefix(config, "MMBench_DEV_EN")
        ensure_eval_result_base_dir(base_dir)
        with open(f"{result_prefix}.metrics.json", "w", encoding="utf-8") as f:
            json.dump(metric_dict, f, ensure_ascii=False, indent=2)
        with open(f"{result_prefix}.scored_rows.json", "w", encoding="utf-8") as f:
            json.dump(scored_rows, f, ensure_ascii=False, indent=2)
        log_for_0(f"MMBench DEV EN accuracy: {overall:.2f}%")
        sample_outputs = [_vis_mmbench(item) for item in results[:16]]
    else:
        overall = 0.0
        metric_dict = {}
        sample_outputs = []

    mu.sync_global_devices("mmbench dev eval done")
    return overall, sample_outputs, metric_dict


def _test_xlsx_path(config):
    base_dir, result_prefix = _result_prefix(config, "MMBench_TEST_EN")
    ensure_eval_result_base_dir(base_dir)
    return f"{result_prefix}.predictions.xlsx"


def _xlsx_col_name(idx):
    name = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        name = chr(ord("A") + rem) + name
    return name


def _xml_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return escape(text, {"\"": "&quot;"})


def _write_xlsx_minimal(df, path):
    """Write a simple single-sheet xlsx without optional Excel packages."""
    rows = [list(df.columns)] + df.astype(object).where(pd.notna(df), "").values.tolist()
    sheet_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row):
            ref = f"{_xlsx_col_name(c_idx)}{r_idx}"
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{_xml_text(value)}</t></is></c>')
        sheet_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        '</worksheet>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '</Relationships>'
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def _write_prediction_xlsx(df, path):
    try:
        df.to_excel(path, index=False, engine="xlsxwriter")
    except (ImportError, ModuleNotFoundError, ValueError):
        try:
            df.to_excel(path, index=False, engine="openpyxl")
        except (ImportError, ModuleNotFoundError, ValueError):
            _write_xlsx_minimal(df, path)


def export_mmbench_test_xlsx(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    root = getattr(config.eval, "mmbench_test_root", DEFAULT_TEST_URL)
    cache_dir = getattr(config.eval, "mmbench_data_cache_dir", "/kmh-nfs-ssd-us-mount/data/cached/zhh/mmbench_data")
    tsv_path = _resolve_tsv_path(root, cache_dir, "MMBench_TEST_EN.tsv")
    results = _run_mmbench_predictions(
        p_sample_step,
        run_p_sample_step,
        model,
        tokenizer,
        params,
        config,
        split_name="MMBench_TEST_EN",
        root=root,
    )

    out_path = None
    if jax.process_index() == 0:
        data = _read_tsv(tsv_path)
        pred_map = {int(item["index"]): item["prediction"] for item in results}
        missing = [int(idx) for idx in data["index"] if int(idx) not in pred_map]
        if missing:
            raise ValueError(f"MMBench TEST export missing {len(missing)} predictions, first missing={missing[:5]}")
        data["prediction"] = [pred_map[int(idx)] for idx in data["index"]]
        if "image" in data:
            data = data.drop(columns=["image"])
        out_path = _test_xlsx_path(config)
        _write_prediction_xlsx(data, out_path)
        log_for_0(f"MMBench TEST EN xlsx exported to {out_path}")

    mu.sync_global_devices("mmbench test export done")
    return out_path


if __name__ == "__main__":
    pass
