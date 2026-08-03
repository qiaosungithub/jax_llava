"""Evaluators for image-only visual-understanding benchmarks uploaded as WDS tar shards.

Supported datasets:
  - GQA balanced testdev/val/train shards produced by VLM-Eval-Benchmarks-upload.py
  - VisWiz-VQA val/test shards
  - ScienceQA-IMG train/validation/test shards
  - SEED-Bench image-only shards
  - Cambrian CV-Bench official test set
  - VLMs Are Blind (BlindTest) official validation set
  - DocVQA 2020 single-document validation set
  - xAI RealWorldQA official test set

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

from evals.vqa_scoring import postprocess_vqav2_text, vqa_accuracy_one
from evals.eval_dist_util import (
    broadcast_merge_ok,
    collate_fn,
    gather_rank_json_results,
    write_rank_json_results,
)
from input_pipeline import get_transforms, prepare_batch_data
from utils.eval_io_util import ensure_eval_result_base_dir, eval_result_prefix
from utils.logging_util import log_for_0, log_for_all

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

CVBENCH_ANSWER_INSTRUCTION = (
    "Answer with the option's letter from the given choices directly."
)

DOCVQA_ANSWER_INSTRUCTION = "Answer the question using a single word or phrase."

BLINDTEST_TASK_MAP = {
    "counting grid - blank grids": "counting_grid",
    "counting grid - word grids": "counting_grid",
    "line plot intersections": "line_plot_intersections",
    "touching circles": "touching_circles",
    "circled letter": "circled_letter",
    "olympic counting - circles": "olympic_counting_circles",
    "olympic counting - pentagons": "olympic_counting_pentagons",
    "nested squares": "nested_squares",
    "subway connections": "subway_connections",
}
BLINDTEST_REPORT_TASKS = (
    "line_plot_intersections",
    "touching_circles",
    "circled_letter",
    "olympic_counting_circles",
    "olympic_counting_pentagons",
    "nested_squares",
    "counting_grid",
    "subway_connections",
)


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


def _normalize_docvqa_text(text: Any) -> str:
    """Match the case-insensitive, otherwise text-preserving DocVQA protocol."""
    return _as_text(text).strip().lower()


def _levenshtein_distance(left: str, right: str) -> int:
    """Compute exact character edit distance without an optional dependency."""
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _docvqa_anls_score(
    pred: str,
    answers: Sequence[Any],
    threshold: float = 0.5,
) -> float:
    """Return the best official-style ANLS score over accepted answer variants."""
    prediction = _normalize_docvqa_text(pred)
    best = 0.0
    for answer in answers:
        target = _normalize_docvqa_text(answer)
        if not target:
            similarity = float(prediction == "")
        else:
            normalized_distance = _levenshtein_distance(target, prediction) / max(
                len(target), len(prediction)
            )
            similarity = (
                1.0 - normalized_distance
                if normalized_distance < threshold
                else 0.0
            )
        best = max(best, similarity)
    return best


def _docvqa_exact_score(pred: str, answers: Sequence[Any]) -> float:
    prediction = _normalize_docvqa_text(pred)
    return float(
        any(prediction == _normalize_docvqa_text(answer) for answer in answers)
    )


_REALWORLDQA_ANSWER_PHRASES = (
    "the answer is",
    "answer is",
    "the correct answer is",
    "correct answer is",
    "the best answer is",
    "best answer is",
    "the correct option is",
    "correct option is",
    "the best option is",
    "best option is",
    "the choice is",
    "choice is",
    "the correct choice is",
    "correct choice is",
    "i choose",
    "i select",
    "i pick",
    "my answer is",
    "my choice is",
    "答案是",
    "答案为",
    "选",
)


def _extract_realworldqa_mcq_answer(response: str) -> str:
    """Port lmms-eval's ranked MCQ extractor for the A-D RealWorldQA rows."""
    choices = ("A", "B", "C", "D")
    text = _as_text(response).strip()
    if not text:
        return ""
    for char in ",.!?;:'\"":
        text = text.strip(char)
    padded = f" {text} "
    candidates = []
    priorities = {
        "start": 10,
        "end": 9,
        "phrase": 7,
        "parentheses": 6,
        "period": 5,
        "colon": 4,
        "right_paren": 3,
        "space": 2,
        "fallback": 0,
    }
    for choice in choices:
        for marker, format_name in (
            (f"({choice})", "parentheses"),
            (f"{choice}.", "period"),
            (f"{choice}:", "colon"),
            (f"{choice})", "right_paren"),
            (f"{choice} ", "space"),
        ):
            if marker in padded:
                candidates.append((choice, padded.rfind(marker), format_name))

    lowered = padded.lower()
    for phrase in _REALWORLDQA_ANSWER_PHRASES:
        phrase_index = lowered.find(phrase)
        if phrase_index == -1:
            continue
        after = phrase_index + len(phrase)
        for choice in choices:
            choice_index = padded.find(choice, after)
            if choice_index != -1:
                candidates.append((choice, choice_index, "phrase"))

    stripped = padded.strip()
    for choice in choices:
        if stripped.startswith(choice) and (
            len(stripped) == 1 or not stripped[1].isalpha()
        ):
            candidates.append((choice, 0, "start"))
        if stripped.endswith(choice) and (
            len(stripped) == 1 or not stripped[-2].isalpha()
        ):
            candidates.append((choice, len(padded) - 1, "end"))

    if not candidates:
        for choice in choices:
            if choice in padded:
                candidates.append((choice, padded.rfind(choice), "fallback"))
    if not candidates:
        return ""
    candidates.sort(
        key=lambda item: (priorities[item[2]], item[1]),
        reverse=True,
    )
    return candidates[0][0]


def _realworldqa_exact_score(pred: str, answer: Any) -> float:
    """Match the public lmms-eval RealWorldQA exact-match protocol."""
    target = _as_text(answer).strip()
    prediction = _as_text(pred).strip()
    if not target:
        return 0.0
    if target.upper() in {"A", "B", "C", "D"}:
        return float(_extract_realworldqa_mcq_answer(prediction) == target.upper())
    return float(prediction.lower().rstrip(".") == target.lower())


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


def _require_success_marker(root: str, benchmark: str) -> None:
    """Refuse a partially committed strict benchmark replica."""
    root = str(root).rstrip("/")
    if root.endswith(".tar") or "*" in root or "{" in root:
        raise ValueError(
            f"{benchmark} requires a committed dataset root, got shard pattern {root!r}"
        )
    marker = f"{root}/_SUCCESS"
    if marker.startswith("gs://"):
        fs, fs_path = fsspec.core.url_to_fs(marker)
        exists = fs.exists(fs_path)
    else:
        exists = os.path.isfile(marker)
    if not exists:
        raise FileNotFoundError(
            f"{benchmark} replica is not committed: missing {marker}"
        )


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


def _parse_cvbench_choice_prediction(pred: str, choices: Sequence[str]) -> Optional[int]:
    """Parse only direct option-label answers accepted by CV-Bench.

    A model may emit a bare label, a parenthesized label, or a short
    ``answer`` prefix.  Do not search arbitrary prose for a convenient option
    letter.  In particular, ordinary words such as ``Blue`` and ``Cat`` must
    not be interpreted as options B and C.
    """
    text = _as_text(pred).strip()
    match = re.match(
        r"^(?:(?:the\s+)?answer(?:\s+is)?\s*[:\-]?\s*)?"
        r"\(?([A-Z])\)?(?:[.)](?:\s+|$)|\s+|$)",
        text,
        flags=re.I,
    )
    if match:
        idx = LETTERS.find(match.group(1).upper())
        if 0 <= idx < len(choices):
            return idx
    return None


def _parse_blindtest_prediction(pred: str, task: str) -> Optional[str]:
    """Parse only BlindTest's official constrained answer formats."""
    text = _as_text(pred).strip().lower()
    task_key = BLINDTEST_TASK_MAP.get(_as_text(task).strip().lower())
    if task_key is None:
        return None

    if task_key == "touching_circles":
        match = re.fullmatch(r"(yes|no)\s*[.!]?", text)
        return match.group(1) if match else None

    if task_key in {
        "line_plot_intersections",
        "nested_squares",
        "olympic_counting_circles",
        "olympic_counting_pentagons",
        "subway_connections",
    }:
        match = re.fullmatch(r"(?:\{\s*)?(\d+)(?:\s*\})?\s*[.!]?", text)
        return str(int(match.group(1))) if match else None

    if task_key == "circled_letter":
        match = re.fullmatch(r"(?:\{\s*)?([a-z])(?:\s*\})?\s*[.!]?", text)
        return match.group(1) if match else None

    if task_key == "counting_grid":
        patterns = (
            r"rows\s*=\s*\{\s*(\d+)\s*\}\s*columns\s*=\s*\{\s*(\d+)\s*\}",
            r"\(\s*(\d+)\s*,\s*(\d+)\s*\)",
            r"\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}",
            r"(\d+)\s*,\s*(\d+)",
        )
        for pattern in patterns:
            match = re.fullmatch(pattern + r"\s*[.!]?", text)
            if match:
                return f"{int(match.group(1))},{int(match.group(2))}"
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
    # `lecture` and `solution` are target-side explanations in ScienceQA. They
    # must never be exposed to the model; only the input-side hint is allowed.
    context = record.get("hint", "")
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


def _expand_cambrian_cvbench(sample: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Expand one official CV-Bench row without rebuilding its content prompt."""
    record = _load_json(sample)
    image = _sample_image(sample)
    official_prompt = _as_text(record.get("prompt", "")).strip()
    choices = _extract_choices(record)
    if image is None or not official_prompt or not choices:
        return []

    raw_answer = _as_text(record.get("answer", "")).strip()
    answer_match = re.fullmatch(r"\(?\s*([A-Z])\s*\)?", raw_answer, flags=re.I)
    answer_idx = None
    if answer_match:
        candidate = LETTERS.find(answer_match.group(1).upper())
        if 0 <= candidate < len(choices):
            answer_idx = candidate

    idx = _as_text(record.get("idx", sample.get("__key__", "cvbench")))
    prompt = f"{official_prompt}\n{CVBENCH_ANSWER_INSTRUCTION}"
    return [
        {
            "image": image,
            "prompt": prompt,
            "aux": {
                "id": idx,
                "question": _as_text(record.get("question", "")).strip(),
                "official_prompt": official_prompt,
                "choices": choices,
                "answer_idx": answer_idx,
                "answer": raw_answer,
                "metric": "cvbench_mc",
                "type": _as_text(record.get("type", "")),
                "task": _as_text(record.get("task", "")),
                "source": _as_text(record.get("source", "")),
                "source_dataset": _as_text(record.get("source_dataset", "")),
                "filename": _as_text(record.get("filename", "")),
            },
        }
    ]


def _expand_vlms_are_blind(sample: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Expand one BlindTest row, preserving the benchmark-provided prompt."""
    record = _load_json(sample)
    image = _sample_image(sample)
    prompt = _as_text(record.get("prompt", "")).strip()
    task = _as_text(record.get("task", "")).strip()
    groundtruth = _as_text(record.get("groundtruth", "")).strip()
    if image is None or not prompt or not task or not groundtruth:
        return []

    row_id = _as_text(
        record.get("id", record.get("idx", sample.get("__key__", "blindtest")))
    )
    return [
        {
            "image": image,
            "prompt": prompt,
            "aux": {
                "id": row_id,
                "question": prompt,
                "official_prompt": prompt,
                "task": task,
                "groundtruth": groundtruth,
                "metadata": record.get("metadata", ""),
                "metric": "blindtest",
            },
        }
    ]


def _expand_docvqa(sample: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Expand one DocVQA validation row using the community-standard prompt."""
    record = _load_json(sample)
    image = _sample_image(sample)
    question = _as_text(record.get("question", "")).strip()
    answers = _extract_answers(record)
    if image is None or not question or not answers:
        return []

    key = _as_text(sample.get("__key__", "docvqa"))
    question_id = _record_id(record, key)
    prompt = f"{question}\n{DOCVQA_ANSWER_INSTRUCTION}"
    return [
        {
            "image": image,
            "prompt": prompt,
            "aux": {
                "id": question_id,
                "question_id": question_id,
                "question": question,
                "answers": answers,
                "question_types": _as_list(record.get("question_types")),
                "doc_id": record.get("docId"),
                "metric": "docvqa_anls",
            },
        }
    ]


def _expand_realworldqa(sample: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Expand one xAI RealWorldQA row without altering its bundled prompt."""
    record = _load_json(sample)
    image = _sample_image(sample)
    question = _as_text(record.get("question", "")).strip()
    answer = _as_text(record.get("answer", "")).strip()
    if image is None or not question or not answer:
        return []

    key = _as_text(sample.get("__key__", "realworldqa"))
    row_id = _record_id(record, key)
    return [
        {
            "image": image,
            "prompt": question,
            "aux": {
                "id": row_id,
                "question": question,
                "official_prompt": question,
                "answer": answer,
                "metric": "realworldqa_exact",
            },
        }
    ]


EXPANDERS: Dict[str, Callable[[Dict[str, Any]], Iterable[Dict[str, Any]]]] = {
    "gqa": _expand_gqa,
    "vizwiz": _expand_vizwiz,
    "scienceqa_img": _expand_scienceqa,
    "seed_bench": _expand_seed_bench,
    "cambrian_cvbench": _expand_cambrian_cvbench,
    "vlms_are_blind": _expand_vlms_are_blind,
    "docvqa": _expand_docvqa,
    "realworldqa": _expand_realworldqa,
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
        if self.benchmark in {"docvqa", "realworldqa"}:
            _require_success_marker(self.root_url, self.benchmark)
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
        aux = dict(item.get("aux") or {})
        aux["prompt"] = prompt
        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "prefix_len": torch.tensor(eff_len, dtype=torch.int32),
            "aux": aux,
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
    if metric == "cvbench_mc":
        gt = row.get("answer_idx")
        choices = row.get("choices", [])
        if gt is None:
            return None
        pred_idx = _parse_cvbench_choice_prediction(pred, choices)
        return float(pred_idx == int(gt)) if pred_idx is not None else 0.0
    if metric == "blindtest":
        target = _as_text(row.get("groundtruth", "")).strip().lower()
        if not target:
            return None
        parsed = _parse_blindtest_prediction(pred, row.get("task", ""))
        return float(parsed == target) if parsed is not None else 0.0
    if metric == "docvqa_anls":
        answers = row.get("answers", [])
        if not answers:
            return None
        return _docvqa_anls_score(pred, answers)
    if metric == "realworldqa_exact":
        answer = row.get("answer", "")
        if not _as_text(answer).strip():
            return None
        return _realworldqa_exact_score(pred, answer)
    return None


def _percent(scores: Sequence[float]) -> float:
    return float(np.mean(scores) * 100.0) if scores else 0.0


def _aggregate_cambrian_cvbench(
    scored_rows: Sequence[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    by_source = defaultdict(list)
    by_task = defaultdict(list)
    for row in scored_rows:
        score = float(row["score"])
        by_source[_as_text(row.get("source", "")).strip()].append(score)
        by_task[_as_text(row.get("task", "")).strip()].append(score)

    required_sources = ("ADE20K", "COCO", "Omni3D")
    missing = [source for source in required_sources if not by_source.get(source)]
    if missing:
        raise ValueError(f"CV-Bench is missing required source slices: {missing}")

    source_acc = {source: _percent(by_source[source]) for source in required_sources}
    acc_2d = (source_acc["ADE20K"] + source_acc["COCO"]) / 2.0
    acc_3d = source_acc["Omni3D"]
    official = (acc_2d + acc_3d) / 2.0
    micro = _percent([float(row["score"]) for row in scored_rows])
    metrics = {
        "accuracy": official,
        "official_accuracy": official,
        "micro_accuracy": micro,
        "by_type": {
            "2D": {
                "accuracy": acc_2d,
                "count": sum(len(by_source[s]) for s in ("ADE20K", "COCO")),
            },
            "3D": {"accuracy": acc_3d, "count": len(by_source["Omni3D"])},
        },
        "by_source": {
            source: {"accuracy": source_acc[source], "count": len(by_source[source])}
            for source in required_sources
        },
        "by_task": {
            task: {"accuracy": _percent(scores), "count": len(scores)}
            for task, scores in sorted(by_task.items())
        },
    }
    return official, metrics


def _aggregate_vlms_are_blind(
    scored_rows: Sequence[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    by_task = {task: [] for task in BLINDTEST_REPORT_TASKS}
    unknown = defaultdict(int)
    for row in scored_rows:
        raw_task = _as_text(row.get("task", "")).strip().lower()
        task = BLINDTEST_TASK_MAP.get(raw_task)
        if task is None:
            unknown[raw_task or "<empty>"] += 1
            continue
        by_task[task].append(float(row["score"]))
    if unknown:
        raise ValueError(f"Unknown VLMs Are Blind task labels: {dict(unknown)}")

    missing = [task for task, scores in by_task.items() if not scores]
    if missing:
        raise ValueError(f"VLMs Are Blind is missing required report tasks: {missing}")

    task_metrics = {
        task: {"accuracy": _percent(scores), "count": len(scores)}
        for task, scores in by_task.items()
    }
    task_mean = float(
        np.mean([task_metrics[task]["accuracy"] for task in BLINDTEST_REPORT_TASKS])
    )
    micro = _percent([float(row["score"]) for row in scored_rows])
    metrics = {
        "accuracy": task_mean,
        "task_mean": task_mean,
        "micro_accuracy": micro,
        "by_task": task_metrics,
    }
    return task_mean, metrics


def _aggregate_docvqa(
    scored_rows: Sequence[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    anls = _percent([float(row["score"]) for row in scored_rows])
    exact = _percent(
        [
            _docvqa_exact_score(
                row.get("prediction", ""),
                row.get("answers", []),
            )
            for row in scored_rows
        ]
    )
    return anls, {
        "anls": anls,
        "exact_accuracy": exact,
    }


def _merge_and_score(
    config,
    benchmark: str,
    cache_key: str,
    cache_default: str,
    result_name: str,
    result_prefix: str,
) -> Tuple[float, Dict[str, Any]]:
    all_results = gather_rank_json_results(
        result_prefix,
        missing_file_msg=f"Missing {benchmark} result file from rank {{rank}}: {{path}}",
    )

    dedup = {}
    for row in all_results:
        key = row.get("id", len(dedup))
        if key not in dedup:
            dedup[key] = row
    all_results = list(dedup.values())

    strict_benchmarks = {
        "cambrian_cvbench",
        "vlms_are_blind",
        "docvqa",
        "realworldqa",
    }
    expected = None
    if benchmark in strict_benchmarks:
        expected = int(getattr(config.eval, f"{benchmark}_num_samples"))
        if len(all_results) != expected:
            raise ValueError(
                f"{result_name} expected {expected} unique predictions, "
                f"got {len(all_results)}"
            )

    scores = []
    scored_rows = []
    by_type = defaultdict(list)
    no_gt = 0
    for row in all_results:
        score = _score_result(row)
        if score is None:
            no_gt += 1
            continue
        scores.append(float(score))
        scored_rows.append({**row, "score": float(score)})
        if row.get("question_type"):
            by_type[row["question_type"]].append(float(score))

    if expected is not None and len(scores) != expected:
        raise ValueError(
            f"{result_name} expected {expected} scored predictions, got {len(scores)}; "
            f"{no_gt} rows were missing usable ground truth"
        )

    if benchmark == "cambrian_cvbench":
        primary, benchmark_metrics = _aggregate_cambrian_cvbench(scored_rows)
    elif benchmark == "vlms_are_blind":
        primary, benchmark_metrics = _aggregate_vlms_are_blind(scored_rows)
    elif benchmark == "docvqa":
        primary, benchmark_metrics = _aggregate_docvqa(scored_rows)
    else:
        primary = _percent(scores)
        benchmark_metrics = {"accuracy": primary}
    metrics = {
        "benchmark": benchmark,
        "num_predictions": len(all_results),
        "num_scored": len(scores),
        "num_without_gt": no_gt,
        **benchmark_metrics,
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
    log_for_0(
        f"{result_name} primary score: {primary:.2f}% "
        f"over {len(scores)} scored samples"
    )
    return primary, metrics


def _vis_row(row: Dict[str, Any]) -> str:
    lines = [f"question: {row.get('question', '')}"]
    if row.get("prompt"):
        lines.append(f"prompt: {row.get('prompt', '')}")
    lines.append(f"prediction: {row.get('prediction', '')}")
    if row.get("answers"):
        lines.append(f"gt_answers: {row.get('answers')}")
    elif row.get("answer") not in (None, ""):
        lines.append(f"gt_answer: {row.get('answer')}")
    if row.get("choices"):
        lines.append(f"choices: {row.get('choices')}")
        lines.append(f"gt: {row.get('answer')}")
    if row.get("question_type"):
        lines.append(f"type: {row.get('question_type')}")
    return "\n".join(lines)


def _understanding_loader_settings(config, benchmark: str) -> Tuple[int, int]:
    """Resolve safe per-benchmark loader settings.

    The iterable stream is partitioned at sample level across JAX processes so
    every host executes a synchronized, exact-count decode schedule. PyTorch
    workers independently split WebDataset shards, which breaks that global
    sample ordering and can leave the merged result short or duplicated. Keep
    these exact-count benchmarks single-worker until the dataset owns a joint
    process/worker partitioner.
    """
    device_batch_size = int(
        getattr(
            config.eval,
            f"{benchmark}_device_batch_size",
            config.eval.device_batch_size,
        )
    )
    num_workers = int(getattr(config.eval, f"{benchmark}_num_workers", 0))
    if num_workers != 0:
        raise ValueError(
            f"{benchmark}_num_workers must be 0 for synchronized exact-count "
            f"evaluation, got {num_workers}"
        )
    return device_batch_size, num_workers


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
    device_batch_size, num_workers = _understanding_loader_settings(config, benchmark)
    batch_size = device_batch_size * jax.local_device_count()
    max_len = int(getattr(config.eval, f"{benchmark}_max_txt_len", config.dataset.max_txt_len))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
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
    write_rank_json_results(prefix, all_outs)

    mu.sync_global_devices(f"{benchmark} write done")

    merge_exception = None
    if jax.process_index() == 0:
        try:
            primary, metrics = _merge_and_score(config, benchmark, cache_key, cache_default, result_name, prefix)
        except Exception as exc:  # Keep every rank out of the final barrier on failure.
            logging.exception(f"{result_name} rank-0 merge/scoring failed")
            merge_exception = exc
    else:
        log_for_all(f"Process {jax.process_index()} waiting for {result_name} evaluation to finish...")
        primary, metrics = 0.0, {}

    broadcast_merge_ok(merge_exception, result_name)

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


def eval_cambrian_cvbench(
    p_sample_step, run_p_sample_step, model, tokenizer, params, config
):
    return _eval_understanding_benchmark(
        p_sample_step,
        run_p_sample_step,
        model,
        tokenizer,
        params,
        config,
        benchmark="cambrian_cvbench",
        root_key="cambrian_cvbench_root",
        total_key="cambrian_cvbench_num_samples",
        default_total=2638,
        cache_key="cambrian_cvbench_cache_dir",
        cache_default="/kmh-nfs-ssd-us-mount/data/cached/zhh/cambrian_cvbench_eval",
        result_name="Cambrian CV-Bench",
    )


def eval_vlms_are_blind(
    p_sample_step, run_p_sample_step, model, tokenizer, params, config
):
    return _eval_understanding_benchmark(
        p_sample_step,
        run_p_sample_step,
        model,
        tokenizer,
        params,
        config,
        benchmark="vlms_are_blind",
        root_key="vlms_are_blind_root",
        total_key="vlms_are_blind_num_samples",
        default_total=8016,
        cache_key="vlms_are_blind_cache_dir",
        cache_default="/kmh-nfs-ssd-us-mount/data/cached/zhh/vlms_are_blind_eval",
        result_name="VLMs Are Blind",
    )


def eval_docvqa(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    return _eval_understanding_benchmark(
        p_sample_step,
        run_p_sample_step,
        model,
        tokenizer,
        params,
        config,
        benchmark="docvqa",
        root_key="docvqa_root",
        total_key="docvqa_num_samples",
        default_total=5349,
        cache_key="docvqa_cache_dir",
        cache_default="/kmh-nfs-ssd-us-mount/data/cached/zhh/docvqa_eval",
        result_name="DocVQA validation",
    )


def eval_realworldqa(
    p_sample_step,
    run_p_sample_step,
    model,
    tokenizer,
    params,
    config,
):
    return _eval_understanding_benchmark(
        p_sample_step,
        run_p_sample_step,
        model,
        tokenizer,
        params,
        config,
        benchmark="realworldqa",
        root_key="realworldqa_root",
        total_key="realworldqa_num_samples",
        default_total=765,
        cache_key="realworldqa_cache_dir",
        cache_default="/kmh-nfs-ssd-us-mount/data/cached/zhh/realworldqa_eval",
        result_name="RealWorldQA",
    )
