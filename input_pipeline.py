import importlib
import io
import json
import math
import os
import re
import subprocess
import warnings

import fsspec
import numpy as np
import torch, jax, random
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms
from torchvision.transforms import functional as TF
from functools import partial
from PIL import Image, ImageFile

from utils.logging_util import log_for_0
from utils.llm_util import create_tokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from webdataset.filters import RandomMix
except ImportError:
    RandomMix = getattr(wds, "RandomMix", None)

# ---------------------------------------------------------------------------
# Visual Genome Grounded Caption: region annotation cache
# ---------------------------------------------------------------------------
_REGION_DESC_LOCAL = "/dev/shm/vg_region_descriptions.json"
_DATA_SEED_STRIDE = 1_000_003
_GCS_GLOB_CACHE = {}
_NUMERIC_BRACE_RE = re.compile(r"\{(\d+)\.\.(\d+)(?:\.\.(\d+))?\}")


def _region_desc_gcs_from_root(root_url: str) -> str:
    """Derive the region_descriptions.json GCS path from the shard root URL.

    Real GCS layout:
      gs://kmh-gcp-<zone>/data/visual_genome/wds/shard-000000.tar
      gs://kmh-gcp-<zone>/data/visual_genome/annotations/region_descriptions.json
    """
    base = root_url.split("/wds/")[0]   # "gs://bucket/data/visual_genome"
    return f"{base}/annotations/region_descriptions.json"


def _load_region_lookup(gcs_path: str, local_path: str = _REGION_DESC_LOCAL) -> dict:
    """Download region_descriptions.json from GCS (once) and return {image_id: regions}."""
    if not os.path.exists(local_path):
        log_for_0(f"[genome_gcap] Downloading {gcs_path} -> {local_path} ...")
        r = subprocess.run(
            f"gcloud storage cp {gcs_path} {local_path}",
            shell=True, capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"[genome_gcap] Download failed:\n{r.stderr}")
        log_for_0("[genome_gcap] Download complete.")
    log_for_0("[genome_gcap] Loading region_descriptions.json ...")
    with open(local_path, encoding="utf-8") as f:
        data = json.load(f)
    # top-level list: [{"id": image_id, "regions": [...]}, ...]
    lookup = {entry["id"]: entry["regions"] for entry in data}
    log_for_0(f"[genome_gcap] Loaded {len(lookup)} images with region annotations.")
    return lookup


def register_gcsfs():
    """Patches webdataset to use fsspec for gs:// urls."""
    try:
        gopen_module = importlib.import_module("webdataset.gopen")

        def gopen_gcsfs(url, mode="rb", bufsize=8192, **kwargs):
            return fsspec.open(url, mode=mode).open()

        gopen_module.gopen_schemes["gs"] = gopen_gcsfs
    except ImportError as e:
        print("[Warning] Could not import webdataset.gopen, GCS hack skipped.")
        raise e


register_gcsfs()


_CAPTION_PROMPTS_COMMON = [
    "Describe this image.",
    "Write a caption for this image.",
    "What is happening in this image?",
    "Provide an image caption.",
    "Summarize this image in one caption.",
]

_CAPTION_PROMPTS_DETAILED = [
    "Describe this image in detail.",
    "Write a detailed caption for this image.",
    "Provide a detailed description of this image.",
]

_TEXTCAPS_PROMPTS = [
    "Describe this image and include important visible text.",
    "Write a caption for this image, mentioning key text you can read.",
    "Give a natural caption that captures both scene and visible text.",
    "Caption this image with attention to readable text.",
]

_GCAP_REGION_PROMPTS = [
    "Describe the region {loc}.",
    "What is in the region {loc}?",
    "Give a short caption for region {loc}.",
]

_OCR_TEXT_PROMPTS = [
    "Read the text in this image.",
    "Transcribe the visible text from this image.",
    "What text is shown in this image?",
]

def _ensure_question_line(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return ""
    if not q.endswith("?"):
        q = q + "?"
    return q


def _sample_caption_prompt(dataset_type: str) -> str:
    if dataset_type == "cc12m":
        # Favor detailed prompts for CC12M (long recaptioned style).
        pool = _CAPTION_PROMPTS_COMMON + _CAPTION_PROMPTS_DETAILED + _CAPTION_PROMPTS_DETAILED
    elif dataset_type == "textcaps":
        # Keep concise, text-aware prompts; no forced long-form instruction.
        pool = _TEXTCAPS_PROMPTS
    elif dataset_type == "rendered_text":
        pool = _OCR_TEXT_PROMPTS
    else:
        pool = _CAPTION_PROMPTS_COMMON
    return random.choice(pool)


def _sample_qa_prompt(question: str) -> str:
    qline = _ensure_question_line(question)
    if not qline:
        return ""
    templates = [
        "{question}",
        "{question}",
        "{question}",
        "Question: {question}",
        "Please answer: {question}",
    ]
    return random.choice(templates).format(question=qline)


_MASK_TOKEN_VALUES = np.array([4, 8, 16, 32, 64, 128, 256], dtype=np.int32)
_MASK_EPS = 1e-6


def _dataset_type_to_mask_category(dataset_type: str) -> str:
    if dataset_type in {
        "vqav2",
        "genome",
        "gqa",
        "llava15",
        "llava150k",
        "llava_ov15",
        "textvqa",
        "tallyqa",
        "dvqa",
    }:
        return "vqa"
    if dataset_type in {"rendered_text", "textcaps"}:
        return "ocr"
    if dataset_type in {"genome_gcap", "genome_det"}:
        return "grounded_caption"
    return "caption"


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    erf_x = np.vectorize(math.erf)(x / np.sqrt(2.0))
    return 0.5 * (1.0 + erf_x)


def _logit_normal_discrete_probs(mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    n_bins = len(_MASK_TOKEN_VALUES)  # 7 bins for 4..256
    probs = np.zeros((n_bins,), dtype=np.float64)

    for idx in range(n_bins):
        # Match ideas/plot_discrete_logit_normal.py logic exactly:
        #   k = clip(round(u * n_bins), 0, n_bins - 1)
        # so bin edges are (idx ± 0.5) / n_bins.
        u_low = (idx - 0.5) / n_bins
        u_high = (idx + 0.5) / n_bins

        u_low = max(0.0, u_low)
        u_high = min(1.0, u_high)

        if u_low <= 0.0:
            z_low = -np.inf
        else:
            p_low = np.clip(u_low, _MASK_EPS, 1.0 - _MASK_EPS)
            z_low = np.log(p_low) - np.log1p(-p_low)

        if u_high >= 1.0:
            z_high = np.inf
        else:
            p_high = np.clip(u_high, _MASK_EPS, 1.0 - _MASK_EPS)
            z_high = np.log(p_high) - np.log1p(-p_high)

        low_cdf = 0.0 if np.isneginf(z_low) else _normal_cdf((z_low - mu) / sigma)
        high_cdf = 1.0 if np.isposinf(z_high) else _normal_cdf((z_high - mu) / sigma)
        probs[idx] = max(0.0, float(high_cdf - low_cdf))

    probs_sum = probs.sum()
    if probs_sum <= 0:
        probs = np.full_like(probs, 1.0 / n_bins)
    else:
        probs = probs / probs_sum
    return probs.astype(np.float32)


def _build_mask_category_distribution(dataset_config, dataset_type: str) -> torch.Tensor:
    category = _dataset_type_to_mask_category(dataset_type)
    dist_cfg = getattr(dataset_config, "nested_mask_logit_normal", None)

    if dist_cfg is None or category not in dist_cfg:
        probs = np.full((len(_MASK_TOKEN_VALUES),), 1.0 / len(_MASK_TOKEN_VALUES), dtype=np.float32)
    else:
        mu = float(dist_cfg[category].get("mu", 0.0))
        sigma = float(dist_cfg[category].get("sigma", 1.0))
        probs = _logit_normal_discrete_probs(mu, sigma)

    return torch.tensor(probs, dtype=torch.float32)


def _item_shuffle_size(dataset_config, dataset_type: str, default: int) -> int:
    value = getattr(dataset_config, "item_shuffle_size", None)
    if value is None:
        value = getattr(dataset_config, "shuffle_buffer_size", None)
    if value is None:
        return int(default)
    if hasattr(value, "get") and not isinstance(value, (str, bytes)):
        value = value.get(dataset_type, value.get("default", default))
    return max(1, int(value))


def _decode_image_if_needed(image):
    if isinstance(image, (bytes, bytearray)):
        with Image.open(io.BytesIO(image)) as img:
            return img.convert("RGB")
    return image


_FATAL_WDS_ERROR_MARKERS = (
    "no such file",
    "not found",
    "404",
    "403",
    "forbidden",
    "permission denied",
    "access denied",
    "unauthorized",
    "bucket not found",
    "does not exist",
)


def _is_fatal_wds_error(exn):
    if isinstance(exn, (FileNotFoundError, PermissionError)):
        return True
    text = " ".join(
        str(arg) for arg in getattr(exn, "args", ()) if arg is not None
    ).lower()
    return any(marker in text for marker in _FATAL_WDS_ERROR_MARKERS)


def make_stop_after_n_errors(max_errors=50, fatal_on_missing=True):
    """Skip sporadic bad samples; stop after too many errors."""
    count = [0]

    def handler(exn):
        if fatal_on_missing and _is_fatal_wds_error(exn):
            raise exn
        count[0] += 1
        if count[0] >= max_errors or max_errors <= 0:
            raise exn
        warnings.warn(
            f"Ignoring error ({count[0]}/{max_errors}): {exn}",
            UserWarning,
            stacklevel=2,
        )
        return True

    return handler


def _max_wds_errors(config):
    return int(getattr(config, "max_wds_errors", 50))


class LetterboxPadTransform:
    """Resize while preserving aspect ratio, then pad to a square canvas."""

    def __init__(
        self,
        image_size,
        interpolation=transforms.InterpolationMode.BICUBIC,
        fill=127,
    ):
        self.image_size = int(image_size)
        self.target_width = self.image_size
        self.target_height = self.image_size
        self.resize_mode = "letterbox"
        self.interpolation = interpolation
        self.fill = fill
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

    def get_params(self, width, height):
        width = max(int(width), 1)
        height = max(int(height), 1)
        scale = min(self.image_size / width, self.image_size / height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        pad_left = (self.image_size - new_w) // 2
        pad_top = (self.image_size - new_h) // 2
        pad_right = self.image_size - new_w - pad_left
        pad_bottom = self.image_size - new_h - pad_top
        return scale, new_w, new_h, pad_left, pad_top, pad_right, pad_bottom

    def __call__(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        _, new_w, new_h, pad_left, pad_top, pad_right, pad_bottom = self.get_params(width, height)
        image = TF.resize(image, [new_h, new_w], interpolation=self.interpolation)
        image = TF.pad(
            image,
            [pad_left, pad_top, pad_right, pad_bottom],
            fill=self.fill,
        )
        return self.normalize(self.to_tensor(image))

    def transform_box(self, x1, y1, x2, y2, src_w, src_h):
        scale, _, _, pad_left, pad_top, _, _ = self.get_params(src_w, src_h)
        x1 = x1 * scale + pad_left
        x2 = x2 * scale + pad_left
        y1 = y1 * scale + pad_top
        y2 = y2 * scale + pad_top
        x1 = max(0.0, min(float(x1), float(self.image_size)))
        x2 = max(0.0, min(float(x2), float(self.image_size)))
        y1 = max(0.0, min(float(y1), float(self.image_size)))
        y2 = max(0.0, min(float(y2), float(self.image_size)))
        return x1, y1, x2, y2

    def inverse_box(self, x1, y1, x2, y2, src_w, src_h):
        scale, _, _, pad_left, pad_top, _, _ = self.get_params(src_w, src_h)
        x1 = (x1 - pad_left) / scale
        x2 = (x2 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        y2 = (y2 - pad_top) / scale
        src_w = float(src_w)
        src_h = float(src_h)
        x1 = max(0.0, min(float(x1), src_w))
        x2 = max(0.0, min(float(x2), src_w))
        y1 = max(0.0, min(float(y1), src_h))
        y2 = max(0.0, min(float(y2), src_h))
        return x1, y1, x2, y2


class DirectResizeTransform:
    """Resize directly to the target canvas, without preserving aspect ratio."""

    def __init__(
        self,
        image_size,
        interpolation=transforms.InterpolationMode.BICUBIC,
    ):
        if isinstance(image_size, (tuple, list)):
            if len(image_size) != 2:
                raise ValueError(f"image_size tuple/list must be (height, width), got {image_size}")
            self.target_height = int(image_size[0])
            self.target_width = int(image_size[1])
        else:
            self.target_height = int(image_size)
            self.target_width = int(image_size)
        if self.target_height <= 0 or self.target_width <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.image_size = self.target_height if self.target_height == self.target_width else None
        self.resize_mode = "stretch"
        self.interpolation = interpolation
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

    def __call__(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = TF.resize(
            image,
            [self.target_height, self.target_width],
            interpolation=self.interpolation,
        )
        return self.normalize(self.to_tensor(image))

    def transform_box(self, x1, y1, x2, y2, src_w, src_h):
        src_w = max(float(src_w), 1.0)
        src_h = max(float(src_h), 1.0)
        scale_x = float(self.target_width) / src_w
        scale_y = float(self.target_height) / src_h
        x1 = x1 * scale_x
        x2 = x2 * scale_x
        y1 = y1 * scale_y
        y2 = y2 * scale_y
        x1 = max(0.0, min(float(x1), float(self.target_width)))
        x2 = max(0.0, min(float(x2), float(self.target_width)))
        y1 = max(0.0, min(float(y1), float(self.target_height)))
        y2 = max(0.0, min(float(y2), float(self.target_height)))
        return x1, y1, x2, y2

    def inverse_box(self, x1, y1, x2, y2, src_w, src_h):
        src_w = max(float(src_w), 1.0)
        src_h = max(float(src_h), 1.0)
        scale_x = src_w / float(self.target_width)
        scale_y = src_h / float(self.target_height)
        x1 = x1 * scale_x
        x2 = x2 * scale_x
        y1 = y1 * scale_y
        y2 = y2 * scale_y
        x1 = max(0.0, min(float(x1), src_w))
        x2 = max(0.0, min(float(x2), src_w))
        y1 = max(0.0, min(float(y1), src_h))
        y2 = max(0.0, min(float(y2), src_h))
        return x1, y1, x2, y2


def _resize_mode_from_config(config):
    return str(getattr(config, "resize_mode", "letterbox")).lower()


def _transform_target_size(transform):
    target_w = getattr(transform, "target_width", None)
    target_h = getattr(transform, "target_height", None)
    if target_w is None or target_h is None:
        image_size = getattr(transform, "image_size", None)
        if image_size is None:
            return None, None
        target_w = target_h = image_size
    return float(target_w), float(target_h)


def get_transforms(image_size, is_train=True, resize_mode="letterbox"):
    resize_mode = str(resize_mode or "letterbox").lower()
    if resize_mode in {"letterbox", "letterbox_pad", "pad"}:
        return LetterboxPadTransform(image_size)
    if resize_mode in {"stretch", "direct_resize", "resize"}:
        return DirectResizeTransform(image_size)
    raise ValueError(f"Unknown resize_mode: {resize_mode}")


def _box_to_loc_tokens(transform, x, y, w, h, img_w, img_h):
    img_w = max(float(img_w), 1.0)
    img_h = max(float(img_h), 1.0)
    x1 = max(0.0, min(float(x), img_w))
    y1 = max(0.0, min(float(y), img_h))
    x2 = max(0.0, min(float(x) + float(w), img_w))
    y2 = max(0.0, min(float(y) + float(h), img_h))
    if x2 <= x1 or y2 <= y1:
        return None

    if hasattr(transform, "transform_box"):
        x1, y1, x2, y2 = transform.transform_box(x1, y1, x2, y2, img_w, img_h)
        norm_w, norm_h = _transform_target_size(transform)
        if norm_w is None or norm_h is None:
            norm_w, norm_h = img_w, img_h
    else:
        norm_w, norm_h = img_w, img_h

    if x2 <= x1 or y2 <= y1:
        return None

    ymin = int((y1 / norm_h) * 1023)
    xmin = int((x1 / norm_w) * 1023)
    ymax = int((y2 / norm_h) * 1023)
    xmax = int((x2 / norm_w) * 1023)
    ymin = max(0, min(ymin, 1023))
    xmin = max(0, min(xmin, 1023))
    ymax = max(0, min(ymax, 1023))
    xmax = max(0, min(xmax, 1023))
    return f"<loc{ymin:04d}><loc{xmin:04d}><loc{ymax:04d}><loc{xmax:04d}>"


def _get_text_from_sample(sample, dataset_type):
    if dataset_type in {"llava150k", "llava15", "llava_ov15"}:
        raw = sample.get("json")
        if raw is None:
            return ("", "")
        if isinstance(raw, bytes):
            raw = json.loads(raw.decode("utf-8"))

        convs = raw
        if isinstance(raw, dict):
            convs = raw.get("conversations", [])
        else:
            convs = raw
        if not isinstance(convs, list):
            return ("", "")

        turns = []
        for c in convs:
            if not isinstance(c, dict):
                continue
            speaker = (c.get("from") or c.get("role") or "").strip().lower()
            value = (c.get("value") or c.get("content") or "").replace("<image>", "").strip()
            if not value:
                continue
            if speaker in {"human", "user"}:
                turns.append(("human", value))
            elif speaker in {"gpt", "assistant"}:
                turns.append(("assistant", value))

        if not turns:
            return ("", "")

        last_assistant_idx = -1
        for i in range(len(turns) - 1, -1, -1):
            if turns[i][0] == "assistant":
                last_assistant_idx = i
                break
        if last_assistant_idx < 0:
            return ("", "")

        question_parts = [v for role, v in turns[:last_assistant_idx] if role == "human"]
        answer_part = turns[last_assistant_idx][1]
        if not answer_part:
            return ("", "")

        question_part = "\n".join(question_parts).strip()
        return (question_part, answer_part)

    if dataset_type == "rendered_text":
        raw = sample.get("json")
        if raw is None:
            return ""
        if isinstance(raw, bytes):
            raw = json.loads(raw.decode("utf-8"))
        lines = raw.get("ocr_annotation", {}).get("text", [])
        return " ".join(lines).strip()

    if dataset_type == "textcaps":
        raw = sample.get("json")
        if raw is None:
            return []
        if isinstance(raw, bytes):
            raw = json.loads(raw.decode("utf-8"))
        caps = raw.get("captions", raw.get("caption_str", []))
        if isinstance(caps, str):
            caps = [caps]
        if not isinstance(caps, list):
            caps = []
        return [str(x).strip() for x in caps if str(x).strip()]

    caption = sample.get("txt") or sample.get("caption") or ""
    if not isinstance(caption, str):
        caption = str(caption)
    return caption


def preprocess_fn(
    sample,
    transform,
    tokenizer,
    max_len,
    dataset_type="default",
    mask_token_category_probs=None,
):
    try:
        image = sample.get("jpg") or sample.get("jpeg") or sample.get("png") or sample.get("webp")
        if image is None:
            return None
        image = _decode_image_if_needed(image)
        pixel_values = transform(image)
    except Exception:
        return None

    text_out = _get_text_from_sample(sample, dataset_type)
    if dataset_type in {"llava150k", "llava15"}:
        question_part = (sample.get("question", "") or "").strip()
        answer_part = (sample.get("aux", {}) or {}).get("answer", "")
        answer_part = "" if answer_part is None else str(answer_part).strip()
        if not answer_part:
            question_part, answer_part = text_out
        if not answer_part:
            return None
        prompt_for_mask = _sample_qa_prompt(question_part) if question_part else "Describe the image."
        prompt_for_mask = prompt_for_mask + "\n"
        full_text = f"{prompt_for_mask}{answer_part}"
        prefix_tokens = tokenizer.encode(prompt_for_mask, add_bos=True, add_eos=False)
    elif dataset_type == "llava_ov15":
        question_part = (sample.get("question", "") or "").strip()
        answer_part = (sample.get("aux", {}) or {}).get("answer", "")
        answer_part = "" if answer_part is None else str(answer_part).strip()
        if not answer_part:
            question_part, answer_part = text_out
        if not answer_part:
            return None
        prompt_for_mask = f"{question_part}\n" if question_part else ""
        full_text = f"{prompt_for_mask}{answer_part}"
        prefix_tokens = tokenizer.encode(prompt_for_mask, add_bos=True, add_eos=False)
    elif dataset_type == "rendered_text":
        caption = text_out # the words rendered
        if not caption:
            raise ValueError(f'caption is empty: {sample}')
        prefix = _sample_caption_prompt("rendered_text") + "\n"
        full_text = f"{prefix}{caption}"
        prefix_tokens = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    elif dataset_type in {"vqav2", "gqa"}:
        question = (sample.get("question", "") or "").strip()
        if not question:
            log_for_0(f'question is empty')
            return None
        prompt = _sample_qa_prompt(question)
        if not prompt:
            return None
        prefix = f"{prompt}\n"
        aux = sample.get("aux", None) or {}
        answers = aux.get("answers", [])
        if not answers:
            log_for_0(f'answers is empty')
            return None
        answer = random.choice(answers)
        full_text = f"{prefix}{answer}"
        prefix_tokens = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    elif dataset_type in {"textvqa", "tallyqa", "dvqa"}:
        question = (sample.get("question", "") or "").strip()
        if not question:
            return None
        prompt = _sample_qa_prompt(question)
        if not prompt:
            return None
        prefix = f"{prompt}\n"
        aux = sample.get("aux", None) or {}
        answers = aux.get("answers", [])
        answers = [
            str(a).strip()
            for a in answers
            if str(a).strip() and str(a).strip().lower() != "unanswerable"
        ]
        if not answers:
            return None
        answer = random.choice(answers)
        full_text = f"{prefix}{answer}"
        prefix_tokens = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    elif dataset_type == "genome":
        question = (sample.get("question", "") or "").strip()
        if not question:
            return None
        aux = sample.get("aux", None) or {}
        answer = (aux.get("answer", "") or "").strip()
        if not answer:
            return None
        prompt = _sample_qa_prompt(question)
        if not prompt:
            return None
        prefix = f"{prompt}\n"
        full_text = f"{prefix}{answer}"
        prefix_tokens = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    elif dataset_type == "genome_gcap":
        # Grounded captioning: prompt = "caption <loc_ymin><loc_xmin><loc_ymax><loc_xmax>\n"
        # label  = region phrase (natural language description of the box)
        region = sample.get("region")
        if region is None:
            return None
        phrase = (region.get("phrase") or "").strip()
        if len(phrase.split()) < 2:
            return None
        img_w = sample.get("img_w") or 1
        img_h = sample.get("img_h") or 1
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("width", 0)
        h = region.get("height", 0)
        if w <= 0 or h <= 0:
            return None
        loc = _box_to_loc_tokens(transform, x, y, w, h, img_w, img_h)
        if loc is None:
            return None
        prefix = random.choice(_GCAP_REGION_PROMPTS).format(loc=loc) + "\n"
        full_text = f"{prefix}{phrase}"
        prefix_tokens = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    elif dataset_type == "genome_det":
        # Grounded detection: prompt is a phrase, target is bbox tokens.
        # Format aligned with RefCOCO-style evaluation prompting.
        region = sample.get("region")
        if region is None:
            return None
        phrase = (region.get("phrase") or "").strip()
        if len(phrase.split()) < 2:
            return None
        img_w = sample.get("img_w") or 1
        img_h = sample.get("img_h") or 1
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("width", 0)
        h = region.get("height", 0)
        if w <= 0 or h <= 0:
            return None
        loc = _box_to_loc_tokens(transform, x, y, w, h, img_w, img_h)
        if loc is None:
            return None
        prefix = f"Locate the region described by this phrase: {phrase}\n"
        full_text = f"{prefix}{loc}"
        prefix_tokens = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    elif dataset_type == "textcaps":
        captions = text_out if isinstance(text_out, list) else []
        if not captions:
            return None
        caption = random.choice(captions)
        prefix = _sample_caption_prompt("textcaps") + "\n"
        full_text = f"{prefix}{caption}"
        prefix_tokens = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    else:
        caption = text_out
        prefix = _sample_caption_prompt(dataset_type) + "\n"
        full_text = f"{prefix}{caption}"
        prefix_tokens = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    prefix_len = min(len(prefix_tokens), max_len)

    # gemma tokenizer returns python list
    token_ids = tokenizer.encode(full_text, add_bos=True, add_eos=True)

    # ensure input_ids/labels length == max_len
    if len(token_ids) > max_len + 1:
        token_ids = token_ids[:max_len + 1]

    input_ids_list = token_ids[:-1]
    labels_list = token_ids[1:]

    cur_len = len(input_ids_list)
    pad_len = max_len - cur_len
    assert pad_len >= 0, f"pad_len is negative: {pad_len}"

    pad_id = tokenizer.special_tokens.PAD
    if pad_len > 0:
        input_ids_list = input_ids_list + [pad_id] * pad_len
        labels_list = labels_list + [-100] * pad_len
        attention_mask_list = [1] * cur_len + [0] * pad_len
    else:
        attention_mask_list = [1] * max_len

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask_list, dtype=torch.bool)
    labels = torch.tensor(labels_list, dtype=torch.long)

    labels[attention_mask == 0] = -100
    if prefix_len > 1:
        mask_len = min(prefix_len - 1, max_len)
        labels[:mask_len] = -100

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prefix_len": prefix_len,
        "mask_token_category_probs": (
            mask_token_category_probs
            if mask_token_category_probs is not None
            else torch.full((len(_MASK_TOKEN_VALUES),), 1.0 / len(_MASK_TOKEN_VALUES), dtype=torch.float32)
        ),
        "aux": sample.get("aux", None),
    }


def expand_vqa_sample(sample):
    """Expand one (image, json) into list of (image, qa) for each question."""
    j = sample.get("json")
    if j is None:
        return []
    if isinstance(j, bytes):
        j = json.loads(j.decode("utf-8"))
    qas = j.get("qas", [])
    img = sample.get("jpg") or sample.get("jpeg") or sample.get("png") or sample.get("webp")
    if img is None or not qas:
        return []
    out = []
    for qa in qas:
        raw_answers = qa.get("answers", [])
        if not raw_answers and "answer" in qa:
            raw_answers = [qa.get("answer")]
        answers = [a.get("answer", a) if isinstance(a, dict) else a for a in raw_answers]
        answers = [str(a).strip() for a in answers if str(a).strip()]
        if not answers:
            continue
        out.append({
            "jpg": img,
            "question": qa.get("question", ""),
            "aux": {
                "question_id": qa.get("question_id", 0),
                "question": qa.get("question", ""),
                "answers": answers,
                # "answer_type": qa.get("answer_type", "other"),
            },
        })
    return out


def expand_genome_sample(sample):
    """Expand one (jpg, json) into list of (image, qa) for each question.
    Visual Genome QAs each have a single answer string (no list).
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
        out.append({
            "jpg": img,
            "question": qa.get("question", ""),
            "aux": {
                "qa_id":    int(qa.get("qa_id", 0)),
                "question": qa.get("question", ""),
                "answer":   qa.get("answer", ""),
            },
        })
    return out


def expand_llava_sample(sample):
    """Expand one LLaVA conversation sample into per-turn QA samples."""
    raw = sample.get("json")
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = json.loads(raw.decode("utf-8"))

    convs = raw.get("conversations", []) if isinstance(raw, dict) else raw
    if not isinstance(convs, list):
        return []

    img = sample.get("jpg") or sample.get("jpeg") or sample.get("png") or sample.get("webp")
    if img is None:
        return []

    turns = []
    for c in convs:
        if not isinstance(c, dict):
            continue
        speaker = (c.get("from") or c.get("role") or "").strip().lower()
        value = (c.get("value") or c.get("content") or "").replace("<image>", "").strip()
        if not value:
            continue
        if speaker in {"human", "user"}:
            turns.append(("human", value))
        elif speaker in {"gpt", "assistant"}:
            turns.append(("assistant", value))

    out = []
    sample_id = raw.get("id") if isinstance(raw, dict) else None
    for i, (role, answer) in enumerate(turns):
        if role != "assistant":
            continue
        question = ""
        for j in range(i - 1, -1, -1):
            if turns[j][0] == "human":
                question = turns[j][1]
                break
        if not question or not answer:
            continue
        out.append({
            "jpg": img,
            "question": question,
            "aux": {
                "answer": answer,
                "conversation_id": sample_id,
                "turn_idx": i,
            },
        })
    return out


def expand_genome_gcap_sample(sample, region_lookup: dict) -> list:
    """Expand one shard sample into (image, region) dicts — one per region.

    Each output dict has:
      "jpg"    : PIL Image
      "region" : single region dict  {x, y, width, height, phrase, ...}
      "img_w"  : int
      "img_h"  : int
    """
    j = sample.get("json")
    if j is None:
        return []
    if isinstance(j, bytes):
        j = json.loads(j.decode("utf-8"))
    image_id = j.get("image_id")
    if image_id is None:
        return []
    img_w = j.get("width") or 1
    img_h = j.get("height") or 1
    img = sample.get("jpg")
    if img is None:
        return []
    regions = region_lookup.get(image_id)
    if not regions:
        return []
    out = []
    for region in regions:
        phrase = (region.get("phrase") or "").strip()
        if len(phrase.split()) < 2:
            continue
        if (region.get("width", 0) <= 0) or (region.get("height", 0) <= 0):
            continue
        out.append({
            "jpg":    img,
            "region": region,
            "img_w":  img_w,
            "img_h":  img_h,
        })
    return out


_EXPAND_FN = {
    "vqav2":   expand_vqa_sample,
    "gqa":     expand_vqa_sample,
    "textvqa": expand_vqa_sample,
    "tallyqa": expand_vqa_sample,
    "dvqa":    expand_vqa_sample,
    "genome":  expand_genome_sample,
    "llava15": expand_llava_sample,
    "llava150k": expand_llava_sample,
    "llava_ov15": expand_llava_sample,
    # genome_gcap needs region_lookup; handled separately in GenomeGCapIterableDataset
}


def _expand_gcs_glob_if_needed(root):
    if isinstance(root, (list, tuple)):
        urls = []
        for item in root:
            expanded = _expand_gcs_glob_if_needed(item)
            if isinstance(expanded, list):
                urls.extend(expanded)
            else:
                urls.append(expanded)
        return urls
    if not isinstance(root, str):
        return root

    if root in _GCS_GLOB_CACHE:
        return list(_GCS_GLOB_CACHE[root])

    urls = []
    for part in root.split("::"):
        for expanded_part in _expand_numeric_braces(part):
            if expanded_part.startswith("gs://") and "*" in expanded_part:
                fs = fsspec.filesystem("gs")
                matches = sorted(fs.glob(expanded_part))
                assert len(matches) > 0, f"No GCS files matched dataset glob: {expanded_part}"
                urls.extend(
                    m if str(m).startswith("gs://") else f"gs://{m}"
                    for m in matches
                )
            else:
                urls.append(expanded_part)

    if len(urls) == 1 and urls[0] == root:
        return root

    _GCS_GLOB_CACHE[root] = tuple(urls)
    log_for_0(f"Expanded dataset URL pattern to {len(urls)} shards: {root}")
    return urls


def _expand_numeric_braces(url):
    """Expand numeric shard ranges like shard-{000000..000040}.tar."""
    match = _NUMERIC_BRACE_RE.search(url)
    if match is None:
        return [url]

    start_s, end_s, step_s = match.groups()
    start = int(start_s)
    end = int(end_s)
    step = int(step_s or 1)
    if step <= 0:
        raise ValueError(f"Invalid non-positive brace step in URL pattern: {url}")

    width = max(len(start_s), len(end_s))
    stop = end + 1 if start <= end else end - 1
    signed_step = step if start <= end else -step

    expanded = []
    for value in range(start, stop, signed_step):
        replacement = f"{value:0{width}d}"
        next_url = url[:match.start()] + replacement + url[match.end():]
        expanded.extend(_expand_numeric_braces(next_url))
    return expanded


def expand_genome_det_sample(sample, region_lookup: dict) -> list:
    """Expand one shard sample into (image, region) dicts for detection pretraining.

    Each output dict has the same fields as genome_gcap expansion, but uses
    phrase->bbox supervision in preprocess_fn(dataset_type='genome_det').
    """
    return expand_genome_gcap_sample(sample, region_lookup)


def _fold_data_seed(base_seed: int, data_seed_offset: int = 0) -> int:
    return int(base_seed) + int(data_seed_offset) * _DATA_SEED_STRIDE


def _worker_seed(base_seed: int, rank: int, data_seed_offset: int = 0) -> int:
    worker = get_worker_info()
    worker_id = 0 if worker is None else int(worker.id)
    return _fold_data_seed(base_seed, data_seed_offset) + int(rank) * 10007 + worker_id * 1009


def _shuffled_worker_urls(root_url, data_seed_offset: int, epoch: int):
    urls = _expand_gcs_glob_if_needed(root_url)
    urls = [urls] if isinstance(urls, str) else list(urls)
    if not urls:
        return []

    worker = get_worker_info()
    worker_id = 0 if worker is None else int(worker.id)
    num_workers = 1 if worker is None else int(worker.num_workers)
    rank = jax.process_index()
    world = jax.process_count()
    stream_id = rank * num_workers + worker_id
    num_streams = max(1, world * num_workers)

    rng = random.Random(_fold_data_seed(7919 + int(epoch), data_seed_offset))
    rng.shuffle(urls)
    selected = urls[stream_id::num_streams]
    if not selected:
        selected = [urls[stream_id % len(urls)]]
    return selected


class VQAv2IterableDataset(IterableDataset):
    """IterableDataset over VQA-style WebDataset shards.
    Expands (image, qas) -> (image, qa) per question, then performs strong item-level shuffling.
    """

    def __init__(self, root_url, config, tokenizer, num_shards=None, dataset_type="vqav2", data_seed_offset=0):
        # Expand expensive GCS globs once in the parent process. Otherwise each
        # DataLoader worker repeats the same bucket listing on first iteration.
        expanded_root = _expand_gcs_glob_if_needed(root_url)
        self.root_url = expanded_root.rstrip("/") if isinstance(expanded_root, str) else list(expanded_root)
        self.config = config
        self.tokenizer = tokenizer
        self.transform = get_transforms(
            config.image_size,
            is_train=False,
            resize_mode=_resize_mode_from_config(config),
        )
        self.max_len = config.max_txt_len
        self.num_shards = num_shards or jax.process_count()
        self.shard_rank = jax.process_index()
        self.dataset_type = dataset_type
        self.data_seed_offset = int(data_seed_offset)
        self.mask_token_category_probs = _build_mask_category_distribution(config, dataset_type)

    def __iter__(self):
        rng = random.Random(_worker_seed(2027, self.shard_rank, self.data_seed_offset))
        expand_fn = _EXPAND_FN.get(self.dataset_type, expand_vqa_sample)

        # Strong item-level shuffle AFTER expand.
        # Keep PIL-image buffers bounded for image-heavy QA datasets.
        shuffle_buf = []
        shuffle_sizes = {
            "textvqa": 2000,
            "dvqa": 20000,
            "tallyqa": 50000,
        }
        SHUFFLE_SIZE = _item_shuffle_size(
            self.config,
            self.dataset_type,
            shuffle_sizes.get(self.dataset_type, 10000),
        )

        epoch = 0
        error_handler = make_stop_after_n_errors(_max_wds_errors(self.config))
        while True:
            urls = _shuffled_worker_urls(self.root_url, self.data_seed_offset, epoch)
            epoch += 1
            if not urls:
                continue

            # Use the low-level pipeline so manually sharded workers are not
            # split a second time by WebDataset's default worker splitter.
            ds = wds.DataPipeline(
                wds.SimpleShardList(urls),
                wds.tarfile_to_samples(handler=error_handler),
            )

            for sample in ds:
                if sample is None:
                    continue

                items = expand_fn(sample)
                if not items:
                    continue

                # Optional: break local correlation within one image
                rng.shuffle(items)

                for item in items:
                    shuffle_buf.append(item)

                    if len(shuffle_buf) >= SHUFFLE_SIZE:
                        # Randomly pop one item, keep buffer full
                        idx = rng.randrange(len(shuffle_buf))
                        chosen = shuffle_buf[idx]
                        shuffle_buf[idx] = shuffle_buf[-1]
                        shuffle_buf.pop()

                        out = preprocess_fn(
                            chosen,
                            self.transform,
                            self.tokenizer,
                            self.max_len,
                            dataset_type=self.dataset_type,
                            mask_token_category_probs=self.mask_token_category_probs,
                        )
                        if out is not None:
                            yield out


class GenomeGCapIterableDataset(IterableDataset):
    """Grounded captioning dataset built from Visual Genome region_descriptions.

    Each sample is one region: prompt = "caption <loc_ymin><loc_xmin><loc_ymax><loc_xmax>\\n"
                                label  = region phrase

    region_lookup is loaded once at __init__ and shared to DataLoader workers
    via Linux fork (copy-on-write), so it is not re-downloaded per worker.
    """

    def __init__(self, root_url: str, config, tokenizer, data_seed_offset=0):
        self.root_url    = root_url.rstrip("/")
        self.config      = config
        self.tokenizer   = tokenizer
        self.transform   = get_transforms(
            config.image_size,
            is_train=True,
            resize_mode=_resize_mode_from_config(config),
        )
        self.max_len     = config.max_txt_len
        self.data_seed_offset = int(data_seed_offset)
        self.mask_token_category_probs = _build_mask_category_distribution(config, "genome_gcap")
        # Derive annotation path from shard root (same bucket, /annotations/ subdir).
        region_json_gcs = _region_desc_gcs_from_root(root_url)
        # Load once in the main process; workers inherit via fork (copy-on-write).
        self.region_lookup = _load_region_lookup(region_json_gcs)

    def __iter__(self):
        rng = random.Random(_worker_seed(2027, jax.process_index(), self.data_seed_offset))
        region_lookup = self.region_lookup  # local ref inside worker

        ds = (
            wds.WebDataset(
                self.root_url,
                resampled=True,
                shardshuffle=1000,
                handler=make_stop_after_n_errors(_max_wds_errors(self.config)),
            )
            .select(lambda x: x is not None)
        )

        # VG has ~3.8M regions total; keep a large shuffle buffer to break
        # the strong locality (all regions from the same image arrive together).
        shuffle_buf = []
        SHUFFLE_SIZE = _item_shuffle_size(self.config, "genome_gcap", 10000)

        for sample in ds:
            items = expand_genome_gcap_sample(sample, region_lookup)
            if not items:
                continue

            # Shuffle within one image to break intra-image order
            rng.shuffle(items)

            for item in items:
                shuffle_buf.append(item)

                if len(shuffle_buf) >= SHUFFLE_SIZE:
                    idx = rng.randrange(len(shuffle_buf))
                    chosen = shuffle_buf[idx]
                    shuffle_buf[idx] = shuffle_buf[-1]
                    shuffle_buf.pop()

                    out = preprocess_fn(
                        chosen,
                        self.transform,
                        self.tokenizer,
                        self.max_len,
                        dataset_type="genome_gcap",
                        mask_token_category_probs=self.mask_token_category_probs,
                    )
                    if out is not None:
                        yield out


class GenomeDetIterableDataset(IterableDataset):
    """Grounded detection dataset from Visual Genome region_descriptions.

    Each sample is one region: prompt = "detect en\\n{phrase}\\n"
                                label  = "<loc_ymin><loc_xmin><loc_ymax><loc_xmax>"
    """

    def __init__(self, root_url: str, config, tokenizer, data_seed_offset=0):
        self.root_url = root_url.rstrip("/")
        self.config = config
        self.tokenizer = tokenizer
        self.transform = get_transforms(
            config.image_size,
            is_train=True,
            resize_mode=_resize_mode_from_config(config),
        )
        self.max_len = config.max_txt_len
        self.data_seed_offset = int(data_seed_offset)
        self.mask_token_category_probs = _build_mask_category_distribution(config, "genome_det")
        region_json_gcs = _region_desc_gcs_from_root(root_url)
        self.region_lookup = _load_region_lookup(region_json_gcs)

    def __iter__(self):
        rng = random.Random(_worker_seed(2029, jax.process_index(), self.data_seed_offset))
        region_lookup = self.region_lookup

        ds = (
            wds.WebDataset(
                self.root_url,
                resampled=True,
                shardshuffle=1000,
                handler=make_stop_after_n_errors(_max_wds_errors(self.config)),
            )
            .select(lambda x: x is not None)
        )

        shuffle_buf = []
        SHUFFLE_SIZE = _item_shuffle_size(self.config, "genome_det", 10000)

        for sample in ds:
            items = expand_genome_det_sample(sample, region_lookup)
            if not items:
                continue
            rng.shuffle(items)

            for item in items:
                shuffle_buf.append(item)
                if len(shuffle_buf) >= SHUFFLE_SIZE:
                    idx = rng.randrange(len(shuffle_buf))
                    chosen = shuffle_buf[idx]
                    shuffle_buf[idx] = shuffle_buf[-1]
                    shuffle_buf.pop()

                    out = preprocess_fn(
                        chosen,
                        self.transform,
                        self.tokenizer,
                        self.max_len,
                        dataset_type="genome_det",
                        mask_token_category_probs=self.mask_token_category_probs,
                    )
                    if out is not None:
                        yield out


def make_dataset(root, dataset_config, tokenizer, is_train=True, dataset_type="default", data_seed_offset=0):
    log_for_0(f"Making dataset for {dataset_type} with root {root}")
    assert dataset_type in [
        "default", "laion_aes", "cc12m", "blip3o", "textcaps", "llava150k", "llava15", "llava_ov15", "vqav2", "gqa", "textvqa", "tallyqa", "dvqa", "genome", "genome_gcap", "genome_det", "rendered_text"
    ], f"Invalid dataset type: {dataset_type}"

    if dataset_type in ["vqav2", "gqa", "textvqa", "tallyqa", "dvqa", "genome", "llava15", "llava150k", "llava_ov15"]:
        ds = VQAv2IterableDataset(
            root,
            dataset_config,
            tokenizer,
            dataset_type=dataset_type,
            data_seed_offset=data_seed_offset,
        )
        log_for_0(f'VQAv2IterableDataset created.')
        return ds

    if dataset_type == "genome_gcap":
        ds = GenomeGCapIterableDataset(root, dataset_config, tokenizer, data_seed_offset=data_seed_offset)
        log_for_0("GenomeGCapIterableDataset created.")
        return ds

    if dataset_type == "genome_det":
        ds = GenomeDetIterableDataset(root, dataset_config, tokenizer, data_seed_offset=data_seed_offset)
        log_for_0("GenomeDetIterableDataset created.")
        return ds

    img_transform = get_transforms(
        dataset_config.image_size,
        is_train=is_train,
        resize_mode=_resize_mode_from_config(dataset_config),
    )
    mask_token_category_probs = _build_mask_category_distribution(dataset_config, dataset_type)

    rank = jax.process_index()

    ds = (
        wds.WebDataset(
            _expand_gcs_glob_if_needed(root),
            resampled=True,
            handler=make_stop_after_n_errors(_max_wds_errors(dataset_config)),
            shardshuffle=True,
        )
        .shuffle(
            int(getattr(dataset_config, "webdataset_shuffle_size", 10000)),
            rng=random.Random(_fold_data_seed(115 + rank * 514, data_seed_offset)),
        )
        .decode("pil")
        .map(partial(
            preprocess_fn,
            transform=img_transform,
            tokenizer=tokenizer,
            max_len=dataset_config.max_txt_len,
            dataset_type=dataset_type,
            mask_token_category_probs=mask_token_category_probs,
        ))
        .select(lambda x: x is not None)
    )
    log_for_0("WebDataset created.")
    return ds


def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    collated = {}
    first_sample = batch[0]
    for key, value in first_sample.items():
        if isinstance(value, torch.Tensor):
            collated[key] = torch.stack([b[key] for b in batch])
        elif key == "prefix_len":
            collated[key] = torch.tensor([b[key] for b in batch], dtype=torch.int32)
        else:
            pass
    return collated


def worker_init_fn(worker_id, rank, data_seed_offset=0):
    seed = _fold_data_seed(worker_id + rank * 1000, data_seed_offset)
    torch.manual_seed(seed % (2**63 - 1))
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def create_split(config, batch_size, data_seed_offset=0):
    rank = jax.process_index()
    data_seed_offset = int(data_seed_offset)
    tokenizer = create_tokenizer(config.model.lm_backbone_str)
    log_for_0("Tokenizer loaded.")

    log_for_0(f"Creating dataset with data_seed_offset={data_seed_offset}...")
    datasets = []
    roots = config.dataset.root
    assert isinstance(roots, list), f"Root must be a list, got {type(roots)}"
    # types is always populated by resolve_dataset_roots (from items or legacy root).
    types = list(getattr(config.dataset, "types", []) or [])
    assert len(types) == len(roots), (
        f"dataset.types length ({len(types)}) != dataset.root length ({len(roots)}). "
        "Ensure resolve_dataset_roots() was called before create_split()."
    )
    for root, dataset_type in zip(roots, types):
        assert "💣" not in root, f"💣 found in dataset path {root}"
        if not dataset_type:
            dataset_type = "default"
        datasets.append(
            make_dataset(
                root,
                config.dataset,
                tokenizer,
                is_train=True,
                dataset_type=dataset_type,
                data_seed_offset=data_seed_offset,
            )
        )
    log_for_0("Datasets created.")

    weights = getattr(config.dataset, "mix_weights", [])
    assert len(weights) == len(roots) or len(roots) == 1
    if len(roots) == 1:
        dataset = datasets[0]
    else:
        if RandomMix is None:
            raise ImportError("webdataset RandomMix is unavailable in current environment")
        dataset = RandomMix(datasets, weights)
        log_for_0(f"RandomMix dataset created with roots {roots} and weights {weights}.")

    dl_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        worker_init_fn=partial(worker_init_fn, rank=rank, data_seed_offset=data_seed_offset),
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        persistent_workers=True if config.dataset.num_workers > 0 else False,
        collate_fn=custom_collate_fn,
        drop_last=True,
    )
    if config.dataset.num_workers > 0:
        dl_kwargs["prefetch_factor"] = config.dataset.prefetch_factor
        dl_kwargs["timeout"] = int(getattr(config.dataset, "dataloader_timeout", 0))

    loader = DataLoader(**dl_kwargs)
    return loader, tokenizer


def prepare_batch_data(batch, batch_size=None):
    """Reformat a PyTorch dataloader batch to numpy NHWC tensors."""
    keys = [
        "pixel_values",
        "input_ids",
        "attention_mask",
        "labels",
        "prefix_len",
        "mask_token_category_probs",
    ]

    if batch_size is not None:
        current_batch_size = batch["pixel_values"].shape[0]
        if current_batch_size < batch_size:
            log_for_0(f"Current batch size {current_batch_size} < required {batch_size}. Padding.")
            pad_size = batch_size - current_batch_size
            for k in keys:
                if k not in batch:
                    continue
                pad_shape = [(0, pad_size)] + [(0, 0) for _ in range(batch[k].ndim - 1)]
                batch[k] = np.pad(batch[k], pad_shape, mode="constant", constant_values=0)
            batch["is_pad"] = np.zeros((batch_size,), dtype=bool)
            batch["is_pad"][current_batch_size:] = True
        else:
            batch["is_pad"] = np.zeros((current_batch_size,), dtype=bool)

    LDC = jax.local_device_count()

    for k in keys:
        if k not in batch:
            continue
        if hasattr(batch[k], "numpy"):
            batch[k] = batch[k].numpy()

    pixel_values = batch["pixel_values"]
    if pixel_values.ndim == 4:
        already_sharded = False
        if pixel_values.shape[0] % LDC != 0:
            raise ValueError(
                f"Batch size {pixel_values.shape[0]} must be divisible by "
                f"local_device_count={LDC}."
            )
    elif pixel_values.ndim == 5:
        already_sharded = True
        if pixel_values.shape[0] != LDC:
            raise ValueError(
                f"Sharded batch leading axis {pixel_values.shape[0]} must equal "
                f"local_device_count={LDC}."
            )
    else:
        raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")

    if not already_sharded:
        for k in keys:
            if k not in batch:
                continue
            batch[k] = batch[k].reshape((LDC, -1) + batch[k].shape[1:])
    else:
        for k in keys:
            if k not in batch:
                continue
            if batch[k].shape[0] != LDC:
                raise ValueError(
                    f"Sharded key {k} has leading axis {batch[k].shape[0]}, "
                    f"expected {LDC}."
                )

    # pixel_values: LDC, B, C, H, W -> LDC, B, H, W, C
    ldc, b, c, h, w = batch['pixel_values'].shape
    assert h == w, f'wrong shape: {batch["pixel_values"].shape}'
    batch['pixel_values'] = batch['pixel_values'].transpose(0, 1, 3, 4, 2)

    return batch


if __name__ == "__main__":
    pass
    # shard_list = get_gcs_shards("gs://kmh-gcp-us-central1/data/laion-aesthetic/**/*.tar")
    # print(shard_list)
    # # --- Smoke test for dataloader ---
    # # Expect: you have a config object with:
    # #   config.dataset.root (can be LAION-aes or DataComp)
    # #   config.dataset.image_size, max_txt_len, num_workers, prefetch_factor, pin_memory
    # #   config.model.lm_backbone_str
    # #
    # # If you don't have config wiring here, import your config builder and create it.

    # from types import SimpleNamespace

    # # -------------------------
    # # Debug-only config (minimal)
    # # -------------------------
    # # NOTE:
    # # - root 这里既可以指 DataComp，也可以指 LAION-aesthetic
    # # - 你现在已经改成只抓 *.tar，所以 LAION 用 **/*.tar
    # # - batch_size 最好是 jax.local_device_count() 的整数倍，避免 reshape 失败
    # # -------------------------

    # LDC = jax.local_device_count()
    # debug_global_batch = 8
    # if debug_global_batch % LDC != 0:
    #     # 向上凑成 LDC 的倍数，避免 prepare_batch_data reshape 崩
    #     debug_global_batch = ((debug_global_batch + LDC - 1) // LDC) * LDC

    # config = SimpleNamespace(
    #     model=SimpleNamespace(
    #         lm_backbone_str="gemma3_270M",
    #     ),
    #     dataset=SimpleNamespace(
    #         # ✅ 用 LAION-aesthetic（webdataset tar）
    #         root="gs://kmh-gcp-us-central1/data/laion-aesthetic/**/*.tar",

    #         # 如果你要测 DataComp，换成：
    #         # root="gs://kmh-gcp-us-central1/data/datacomp/small/**/*.tar",

    #         batch_size=debug_global_batch,
    #         num_workers=0,          # debug 时建议 0，避免多进程把问题复杂化
    #         prefetch_factor=2,      # num_workers==0 时 DataLoader 不会用到（我们代码里也不会传）
    #         pin_memory=False,

    #         image_size=224,
    #         max_txt_len=64,
    #     ),
    # )

    # print(f"\n🚀 Dataloader smoke test")
    # print(f"  dataset base: {config.dataset.root}")
    # print(f"  batch_size:   {config.dataset.batch_size} (LDC={LDC}, per_device={config.dataset.batch_size // LDC})")
    # print(f"  num_workers:  {config.dataset.num_workers}")
    # print(f"  process_index={jax.process_index()} process_count={jax.process_count()}")

    # # build loader
    # loader, tokenizer = create_split(config, batch_size=config.dataset.batch_size)
    # it = iter(loader)

    # print("\n⏳ Fetching one batch from PyTorch DataLoader...")
    # batch = next(it)

    # if not batch:
    #     raise RuntimeError("Batch is empty (collate_fn returned {}). Check dataset keys / preprocessing.")

    # print("\n✅ Raw batch (PyTorch) keys and shapes:")
    # for k, v in batch.items():
    #     if hasattr(v, "shape"):
    #         print(f"  - {k:15s} shape={tuple(v.shape)} dtype={getattr(v, 'dtype', type(v))}")
    #     else:
    #         print(f"  - {k:15s} type={type(v)} value={v}")

    # # Convert to JAX-friendly batch
    # print("\n🔁 Converting batch via prepare_batch_data(...) ...")
    # jbatch = prepare_batch_data(batch, batch_size=config.dataset.batch_size)

    # print("\n✅ JAX batch keys and shapes:")
    # for k, v in jbatch.items():
    #     if isinstance(v, np.ndarray):
    #         print(f"  - {k:15s} shape={tuple(v.shape)} dtype={v.dtype}")
    #     else:
    #         print(f"  - {k:15s} type={type(v)} value={v}")

    # # pixel statistics (NHWC after prepare_batch_data)
    # x = jbatch["pixel_values"]
    # # x: (LDC, B, H, W, C)
    # print("\n🖼️ pixel_values stats (after transform, NHWC):")
    # print(f"  shape: {x.shape}")
    # print(f"  min/max: {x.min():.4f} / {x.max():.4f}")
    # print(f"  mean/std: {x.mean():.4f} / {x.std():.4f}")

    # # Print a decoded example (first device, first sample)
    # ids = jbatch["input_ids"][0, 0]  # (T,)
    # # trim at first PAD if you want cleaner print
    # pad_id = tokenizer.special_tokens.PAD
    # ids_list = ids.tolist()
    # if pad_id in ids_list:
    #     ids_list = ids_list[:ids_list.index(pad_id)]
    # try:
    #     decoded = tokenizer.decode(ids_list)
    # except Exception:
    #     decoded = str(ids_list[:64])

    # print("\n📝 decoded sample[0,0] (trimmed at first PAD):")
    # print(decoded)

    # print("\n✅ Smoke test done.\n")
