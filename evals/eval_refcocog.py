import json
import io
import os
import re
import tarfile
import time
from functools import partial

import fsspec
import jax
import numpy as np
import torch
from PIL import Image, ImageDraw
from jax.experimental import multihost_utils as mu
from torch.utils.data import DataLoader, Dataset, Sampler

from input_pipeline import LetterboxPadTransform, format_detection_prompt, get_transforms, prepare_batch_data
from utils.logging_util import log_for_0, log_for_all
from utils.eval_io_util import ensure_eval_result_base_dir, eval_result_prefix


DEFAULT_REFCOCOG_NUM_VIS = 8


class DistributedEvalSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = jax.process_count()
        if rank is None:
            rank = jax.process_index()
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.dataset_len = len(dataset)
        self.num_samples = (self.dataset_len - self.rank + self.num_replicas - 1) // self.num_replicas

    def __iter__(self):
        return iter(range(self.rank, self.dataset_len, self.num_replicas))

    def __len__(self):
        return self.num_samples


def _join_path(root: str, leaf: str) -> str:
    return f"{root.rstrip('/')}/{leaf.lstrip('/')}"


def _path_exists(path: str) -> bool:
    fs, fs_path = fsspec.core.url_to_fs(path)
    return fs.exists(fs_path)


def _glob_tar_shards(root: str):
    fs, fs_path = fsspec.core.url_to_fs(root)
    matches = sorted(fs.glob(f"{fs_path.rstrip('/')}/shard-*.tar"))
    out = []
    for m in matches:
        if m.startswith("gs://") or m.startswith("/"):
            out.append(m)
        elif root.startswith("gs://"):
            out.append(f"gs://{m}")
        else:
            out.append(m)
    return out


class TarShardImageResolver:
    def __init__(self, tar_paths):
        self.tar_paths = list(tar_paths)
        self.name_to_tar_member = {}
        self._tar_streams = {}
        self._build_index()

    def _build_index(self):
        t0 = time.time()
        for tar_path in self.tar_paths:
            try:
                with fsspec.open(tar_path, "rb").open() as f:
                    with tarfile.open(fileobj=f, mode="r:*") as tf:
                        for m in tf.getmembers():
                            if not m.isfile():
                                continue
                            base = os.path.basename(m.name)
                            if not base:
                                continue
                            if base not in self.name_to_tar_member:
                                self.name_to_tar_member[base] = (tar_path, int(m.offset_data), int(m.size))
            except Exception as e:
                log_for_0(f"RefCOCOg tar index warning: skip {tar_path}, err={e}")
        log_for_0(
            f"RefCOCOg tar index built: {len(self.name_to_tar_member)} files from {len(self.tar_paths)} shards "
            f"in {time.time()-t0:.1f}s"
        )

    def load_image(self, image_name_candidates):
        for name in image_name_candidates:
            hit = self.name_to_tar_member.get(name)
            if hit is None:
                continue
            tar_path, offset, size = hit
            try:
                stream = self._tar_streams.get(tar_path)
                if stream is None:
                    stream = fsspec.open(tar_path, "rb").open()
                    self._tar_streams[tar_path] = stream
                stream.seek(offset)
                payload = stream.read(size)
                if not payload:
                    continue
                return Image.open(io.BytesIO(payload)).convert("RGB")
            except Exception:
                continue
        return None


def _image_name_candidates(name: str):
    name = (name or "").strip()
    if not name:
        return []
    cands = [name]

    # Some annotation dumps append question/ann ids to COCO filenames, e.g.
    # COCO_train2014_000000546154_298801.jpg -> COCO_train2014_000000546154.jpg
    m = re.match(r"^(COCO_(?:train|val)2014_\d{12})_\d+\.(jpg|jpeg|png)$", name, flags=re.IGNORECASE)
    if m is not None:
        cands.append(f"{m.group(1)}.{m.group(2)}")

    # General fallback: remove one trailing _<digits> before extension.
    m2 = re.match(r"^(.+)_\d+\.(jpg|jpeg|png)$", name, flags=re.IGNORECASE)
    if m2 is not None:
        cands.append(f"{m2.group(1)}.{m2.group(2)}")

    out = []
    seen = set()
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _as_list(path_or_dir: str):
    if path_or_dir.endswith(".json") or path_or_dir.endswith(".jsonl"):
        return [path_or_dir]
    fs, _ = fsspec.core.url_to_fs(path_or_dir)
    cand = sorted(fs.glob(f"{path_or_dir.rstrip('/')}/*.json")) + sorted(fs.glob(f"{path_or_dir.rstrip('/')}/*.jsonl"))
    out = []
    for p in cand:
        if p.startswith("gs://") or p.startswith("/"):
            out.append(p)
        elif path_or_dir.startswith("gs://"):
            out.append(f"gs://{p}")
        else:
            out.append(p)
    return out


def _normalize_bbox_xyxy(box):
    if box is None or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _xywh_to_xyxy(box):
    if box is None or len(box) != 4:
        return None
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + max(0.0, w), y + max(0.0, h)]


def _parse_row(row, idx):
    phrase = row.get("phrase") or row.get("ref") or row.get("expression") or row.get("caption") or row.get("sent") or row.get("sentence")
    if phrase is None:
        ans = row.get("answer")
        if isinstance(ans, list):
            for a in ans:
                a = "" if a is None else str(a).strip()
                if a:
                    phrase = a
                    break
        elif ans is not None:
            a = str(ans).strip()
            if a:
                phrase = a
    if phrase is None:
        q = row.get("question")
        q = "" if q is None else str(q).strip()
        if q and "come up with a caption for the area" not in q.lower():
            phrase = q

    phrase = "" if phrase is None else str(phrase).strip()
    if not phrase:
        return None

    image = row.get("image") or row.get("image_path") or row.get("file_name") or row.get("img")
    image = "" if image is None else str(image).strip()
    if not image:
        return None

    gt = None
    if "bbox" in row:
        b = row["bbox"]
        if isinstance(b, dict):
            if all(k in b for k in ["x", "y", "w", "h"]):
                gt = _xywh_to_xyxy([b["x"], b["y"], b["w"], b["h"]])
            elif all(k in b for k in ["x1", "y1", "x2", "y2"]):
                gt = _normalize_bbox_xyxy([b["x1"], b["y1"], b["x2"], b["y2"]])
        elif isinstance(b, (list, tuple)) and len(b) == 4:
            # Default as xywh for common RefCOCO dumps.
            gt = _xywh_to_xyxy(b)

    if gt is None and "box" in row and isinstance(row["box"], (list, tuple)) and len(row["box"]) == 4:
        gt = _xywh_to_xyxy(row["box"])
    if gt is None and all(k in row for k in ["x", "y", "w", "h"]):
        gt = _xywh_to_xyxy([row["x"], row["y"], row["w"], row["h"]])
    if gt is None and all(k in row for k in ["x1", "y1", "x2", "y2"]):
        gt = _normalize_bbox_xyxy([row["x1"], row["y1"], row["x2"], row["y2"]])
    if gt is None:
        return None

    sample_id = row.get("id", row.get("ann_id", row.get("ref_id", idx)))
    return {
        "id": str(sample_id),
        "image": image,
        "phrase": phrase,
        "gt_bbox_xyxy": gt,
    }


def load_refcocog_rows(root_or_file):
    files = _as_list(root_or_file)
    if not files:
        raise FileNotFoundError(f"No RefCOCOg json/jsonl files under {root_or_file}")

    rows = []
    for path in files:
        with fsspec.open(path, "rb").open() as f:
            txt = f.read().decode("utf-8").strip()
        if not txt:
            continue
        if txt[0] == "[":
            data = json.loads(txt)
        else:
            data = [json.loads(line) for line in txt.splitlines() if line.strip()]
        for i, row in enumerate(data):
            item = _parse_row(row, i)
            if item is not None:
                rows.append(item)

    if not rows:
        raise ValueError(f"No valid RefCOCOg rows from {root_or_file}")
    return rows


def _extract_loc_tokens(text: str):
    pat = re.compile(r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>")
    m = pat.search(text)
    if m is None:
        return None
    vals = [int(m.group(i)) for i in range(1, 5)]
    # model outputs y1 x1 y2 x2
    y1, x1, y2, x2 = vals
    return [x1, y1, x2, y2]


def _loc1023_to_xyxy(loc_xyxy, img_w, img_h, transform_aux=None):
    x1, y1, x2, y2 = loc_xyxy
    if transform_aux is not None:
        if isinstance(transform_aux, dict):
            resize_mode = str(transform_aux.get("resize_mode", "letterbox")).lower()
            target_w = transform_aux.get("target_w", transform_aux.get("target_width"))
            target_h = transform_aux.get("target_h", transform_aux.get("target_height"))
            legacy_size = transform_aux.get("letterbox_image_size")
            if target_w is None:
                target_w = legacy_size
            if target_h is None:
                target_h = legacy_size
        else:
            resize_mode = "letterbox"
            target_w = target_h = transform_aux

        if target_w is None or target_h is None:
            raise ValueError(f"Missing transform target size in aux: {transform_aux}")

        target_w = float(target_w)
        target_h = float(target_h)
        x1 = float(np.clip(x1, 0, 1023)) / 1023.0 * target_w
        x2 = float(np.clip(x2, 0, 1023)) / 1023.0 * target_w
        y1 = float(np.clip(y1, 0, 1023)) / 1023.0 * target_h
        y2 = float(np.clip(y2, 0, 1023)) / 1023.0 * target_h

        if resize_mode in {"stretch", "direct_resize", "resize"}:
            x1 = x1 / target_w * float(img_w)
            x2 = x2 / target_w * float(img_w)
            y1 = y1 / target_h * float(img_h)
            y2 = y2 / target_h * float(img_h)
        elif resize_mode in {"letterbox", "letterbox_pad", "pad"}:
            if int(round(target_w)) != int(round(target_h)):
                raise ValueError(f"Letterbox target must be square, got {target_w}x{target_h}")
            transform = LetterboxPadTransform(int(round(target_w)))
            x1, y1, x2, y2 = transform.inverse_box(x1, y1, x2, y2, img_w, img_h)
        else:
            raise ValueError(f"Unknown resize_mode: {resize_mode}")
        return _normalize_bbox_xyxy([x1, y1, x2, y2])

    x1 = float(np.clip(x1, 0, 1023)) / 1023.0 * img_w
    x2 = float(np.clip(x2, 0, 1023)) / 1023.0 * img_w
    y1 = float(np.clip(y1, 0, 1023)) / 1023.0 * img_h
    y2 = float(np.clip(y2, 0, 1023)) / 1023.0 * img_h
    return _normalize_bbox_xyxy([x1, y1, x2, y2])


def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def _clip_text(text, max_chars=70):
    text = "" if text is None else str(text).replace("\n", " ").strip()
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _scale_box_for_canvas(box, scale, offset_x, offset_y):
    if box is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        x1 * scale + offset_x,
        y1 * scale + offset_y,
        x2 * scale + offset_x,
        y2 * scale + offset_y,
    ]


def _draw_labeled_box(draw, box, color, label, width):
    if box is None:
        return
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    text_bbox = draw.textbbox((0, 0), label)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    label_y = max(0, y1 - th - 4)
    draw.rectangle([x1, label_y, x1 + tw + 6, label_y + th + 4], fill=color)
    draw.text((x1 + 3, label_y + 2), label, fill=(255, 255, 255))


def _make_refcocog_vis_tile(item, tile_size=384, caption_h=96):
    image = item.get("vis_image")
    if image is None:
        return None
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        return None

    pil_img = Image.fromarray(image, mode="RGB")
    src_w, src_h = pil_img.size
    scale = min(tile_size / max(src_w, 1), tile_size / max(src_h, 1))
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    pil_img = pil_img.resize((dst_w, dst_h), Image.BILINEAR)

    tile = Image.new("RGB", (tile_size, tile_size + caption_h), (255, 255, 255))
    offset_x = (tile_size - dst_w) // 2
    offset_y = (tile_size - dst_h) // 2
    tile.paste(pil_img, (offset_x, offset_y))
    draw = ImageDraw.Draw(tile)

    line_w = max(2, tile_size // 160)
    gt_box = _scale_box_for_canvas(item.get("gt_bbox_xyxy"), scale, offset_x, offset_y)
    pred_box = _scale_box_for_canvas(item.get("pred_bbox_xyxy"), scale, offset_x, offset_y)
    _draw_labeled_box(draw, gt_box, (36, 160, 80), "GT", line_w)
    _draw_labeled_box(draw, pred_box, (220, 60, 50), "Pred", line_w)

    cap_y = tile_size + 6
    lines = [
        f"phrase: {_clip_text(item.get('phrase'), 58)}",
        f"pred: {_clip_text(item.get('pred_text'), 62)}",
        f"iou: {float(item.get('iou', 0.0)):.4f}",
    ]
    for line in lines:
        draw.text((8, cap_y), line, fill=(0, 0, 0))
        cap_y += 28
    return tile


def _make_refcocog_vis_grid(items, tile_size=384, caption_h=96, cols=4):
    tiles = [
        tile for tile in (_make_refcocog_vis_tile(x, tile_size, caption_h) for x in items)
        if tile is not None
    ]
    if not tiles:
        return None
    cols = max(1, int(cols))
    rows = (len(tiles) + cols - 1) // cols
    tile_w, tile_h = tiles[0].size
    grid = Image.new("RGB", (cols * tile_w, rows * tile_h), (255, 255, 255))
    for i, tile in enumerate(tiles):
        x = (i % cols) * tile_w
        y = (i // cols) * tile_h
        grid.paste(tile, (x, y))
    return np.asarray(grid, dtype=np.uint8)


def preprocess_refcocog_sample(sample, transform, tokenizer, max_len):
    try:
        image = sample.get("jpg") or sample.get("png")
        if image is None:
            return None
        pixel_values = transform(image)
        img_w, img_h = image.size
    except Exception:
        return None

    phrase = (sample.get("phrase") or "").strip()
    if not phrase:
        return None

    prompt = format_detection_prompt(phrase)
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    eff_len = min(len(ids), max_len)
    pad_len = max_len - eff_len
    pad_id = tokenizer.special_tokens.PAD
    input_ids_list = ids[:eff_len] + [pad_id] * max(0, pad_len)
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    prefix_len = torch.tensor(eff_len, dtype=torch.int32)

    aux = dict(sample.get("aux", {}))
    aux["prompt"] = prompt
    aux["img_w"] = int(img_w)
    aux["img_h"] = int(img_h)
    aux["vis_image"] = np.asarray(image, dtype=np.uint8)
    target_w = getattr(transform, "target_width", getattr(transform, "image_size", None))
    target_h = getattr(transform, "target_height", getattr(transform, "image_size", None))
    if target_w is not None and target_h is not None:
        aux["resize_mode"] = str(getattr(transform, "resize_mode", "letterbox"))
        aux["target_w"] = int(target_w)
        aux["target_h"] = int(target_h)
    if getattr(transform, "resize_mode", "letterbox") == "letterbox" and hasattr(transform, "image_size"):
        aux["letterbox_image_size"] = int(transform.image_size)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "prefix_len": prefix_len,
        "aux": aux,
    }


class RefCOCOgDataset(Dataset):
    def __init__(self, rows, config, tokenizer, image_root, tar_resolver=None):
        self.rows = rows
        self.image_root = image_root
        self.tar_resolver = tar_resolver
        self.preprocess_fn = partial(
            preprocess_refcocog_sample,
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
        row = self.rows[idx]
        image_path = row["image"]
        image_name_candidates = _image_name_candidates(image_path)
        image = None
        if self.tar_resolver is not None and image_name_candidates:
            image = self.tar_resolver.load_image(image_name_candidates)
        if image is None:
            candidate_paths = []
            if "://" in image_path or os.path.isabs(image_path):
                candidate_paths = [image_path]
            else:
                candidate_paths = [_join_path(self.image_root, name) for name in image_name_candidates]

            resolved_path = None
            for p in candidate_paths:
                if _path_exists(p):
                    resolved_path = p
                    break
            if resolved_path is None:
                return None

            try:
                with fsspec.open(resolved_path, "rb").open() as f:
                    image = Image.open(f).convert("RGB")
            except Exception:
                return None

        sample = {
            "jpg": image,
            "phrase": row["phrase"],
            "aux": {
                "id": row["id"],
                "phrase": row["phrase"],
                "image": row["image"],
                "gt_bbox_xyxy": row["gt_bbox_xyxy"],
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


def eval_refcocog(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    ann_root = config.eval.refcocog_root
    image_root = config.eval.refcocog_image_root
    iou_thr = float(getattr(config.eval, "refcocog_iou_threshold", 0.5))
    num_workers = int(getattr(config.eval, "refcocog_num_workers", 0))
    num_vis = int(getattr(config.eval, "refcocog_num_vis", DEFAULT_REFCOCOG_NUM_VIS))
    vis_tile_size = int(getattr(config.eval, "refcocog_vis_tile_size", 384))
    vis_cols = int(getattr(config.eval, "refcocog_vis_cols", 4))
    log_for_0(f"RefCOCOg eval: ann={ann_root}, image_root={image_root}, IoU@{iou_thr}")

    rows = load_refcocog_rows(ann_root)
    tar_shards = _glob_tar_shards(image_root)
    tar_resolver = TarShardImageResolver(tar_shards) if tar_shards else None
    # Quick path sanity check to surface filename/root mismatches early.
    n_probe = min(64, len(rows))
    n_exist = 0
    for r in rows[:n_probe]:
        image_name = r["image"]
        name_cands = _image_name_candidates(image_name)
        ok = False
        if tar_resolver is not None and any(x in tar_resolver.name_to_tar_member for x in name_cands):
            ok = True
        cands = [image_name] if ("://" in image_name or os.path.isabs(image_name)) else [
            _join_path(image_root, x) for x in name_cands
        ]
        if not ok:
            for p in cands:
                if _path_exists(p):
                    ok = True
                    break
        if ok:
            n_exist += 1
    log_for_0(f"RefCOCOg probe: {n_exist}/{n_probe} image paths resolved under image_root")

    dataset = RefCOCOgDataset(rows, config, tokenizer, image_root=image_root, tar_resolver=tar_resolver)
    sampler = DistributedEvalSampler(dataset)
    batch_size = config.eval.device_batch_size * jax.local_device_count()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    all_outs = []
    vis_outs = []
    for i, raw_batch in enumerate(loader):
        if not raw_batch:
            continue
        batch = prepare_batch_data(raw_batch, batch_size=batch_size)
        out_strs = run_p_sample_step(
            p_sample_step,
            model,
            tokenizer,
            params,
            batch["pixel_values"],
            batch["input_ids"],
            prefix_len=batch["prefix_len"],
        )
        for aux, out_str in zip(batch["aux"], out_strs):
            pred_loc = _extract_loc_tokens(out_str)
            pred_xyxy = None
            iou = 0.0
            if pred_loc is not None:
                pred_xyxy = _loc1023_to_xyxy(
                    pred_loc,
                    aux["img_w"],
                    aux["img_h"],
                    aux,
                )
                iou = _iou_xyxy(pred_xyxy, aux["gt_bbox_xyxy"])
            record = {
                "id": aux["id"],
                "phrase": aux["phrase"],
                "image": aux["image"],
                "prompt": aux.get("prompt", ""),
                "pred_text": out_str,
                "pred_bbox_xyxy": pred_xyxy,
                "gt_bbox_xyxy": aux["gt_bbox_xyxy"],
                "iou": float(iou),
                "hit": float(iou >= iou_thr),
            }
            all_outs.append(record)
            if jax.process_index() == 0 and len(vis_outs) < num_vis:
                vis_record = dict(record)
                vis_record["vis_image"] = aux.get("vis_image")
                vis_outs.append(vis_record)
        if i % 50 == 0:
            log_for_all(f"rank {jax.process_index()}, refcocog batch {i}, collected {len(all_outs)}")

    mu.sync_global_devices("refcocog inference done")

    base_dir, result_prefix = eval_result_prefix(
        config,
        "refcocog_cache_dir",
        "/kmh-nfs-ssd-us-mount/data/cached/zhh/refcocog_eval",
        "refcocog",
    )
    ensure_eval_result_base_dir(base_dir)

    res_file = f"{result_prefix}.results_{jax.process_index()}.json"
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(all_outs, f, ensure_ascii=False, indent=2)

    mu.sync_global_devices("refcocog write done")

    if jax.process_index() == 0:
        merged = []
        for r in range(jax.process_count()):
            pf = f"{result_prefix}.results_{r}.json"
            if os.path.exists(pf):
                merged.extend(json.load(open(pf, encoding="utf-8")))
        dedup = {}
        for o in merged:
            if o["id"] not in dedup:
                dedup[o["id"]] = o
        merged = list(dedup.values())
        out_path = f"{result_prefix}.results_final.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        ious = [o["iou"] for o in merged]
        hits = [o["hit"] for o in merged]
        miou = float(np.mean(ious)) if ious else 0.0
        acc = float(np.mean(hits) * 100.0) if hits else 0.0
        log_for_0(f"RefCOCOg results: {out_path} ({len(merged)} samples)")
        log_for_0(f"RefCOCOg Acc@IoU{iou_thr:.2f}: {acc:.2f}% | mIoU: {miou:.4f}")
    else:
        log_for_all(f"Process {jax.process_index()} waiting for RefCOCOg merge...")
        acc = 0.0
        miou = 0.0

    mu.sync_global_devices("refcocog eval done")
    sample_texts = [
        (
            f"phrase: {o['phrase']}\n"
            f"prompt: {o.get('prompt', '')}\n"
            f"pred: {o['pred_text']}\n"
            f"pred_box: {o['pred_bbox_xyxy']}\n"
            f"gt_box: {o['gt_bbox_xyxy']}\n"
            f"iou: {o['iou']:.4f}"
        )
        for o in all_outs[:16]
    ]
    metric_dict = {"miou": miou, "iou_threshold": iou_thr}
    vis_grid = _make_refcocog_vis_grid(
        vis_outs,
        tile_size=vis_tile_size,
        cols=vis_cols,
    )
    if vis_grid is not None:
        metric_dict["vis_image"] = vis_grid
    return acc, sample_texts, metric_dict
