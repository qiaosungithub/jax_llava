#!/usr/bin/env python3
"""Same-region evaluation/train overlap checker.

This script is intended to run on a VM/TPU in the same region as the GCS
bucket. It streams tar objects and writes compact JSON/JSONL reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import tarfile
import time
from io import BytesIO
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
TEXT_EXTS = (".json", ".jsonl", ".txt", ".csv")
PIXELBENCH_BENCHMARKS = ("mmvp", "vstar", "ocrbench", "countbenchqa")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}")
    return p


def gcs_ls(pattern: str) -> List[str]:
    p = run(f"gcloud storage ls '{pattern}'", check=False)
    if p.returncode != 0:
        return []
    return [x.strip() for x in p.stdout.splitlines() if x.strip()]


def stream_tar_members(uri: str) -> Iterator[Tuple[tarfile.TarInfo, object]]:
    """Yield regular tar members from a GCS or local tar path."""
    if uri.startswith("gs://"):
        proc = subprocess.Popen(
            ["gcloud", "storage", "cat", uri],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout is not None
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|*") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    yield member, f
        finally:
            if proc.stdout:
                proc.stdout.close()
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            code = proc.wait()
            if code != 0:
                raise RuntimeError(f"gcloud storage cat failed for {uri}: {stderr}")
    else:
        with tarfile.open(uri, mode="r|*") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                yield member, f


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_image_sha256(data: bytes, size: int = 256) -> str:
    with Image.open(BytesIO(data)) as img:
        img = img.convert("RGB").resize((size, size), Image.BICUBIC)
        return hashlib.sha256(img.tobytes()).hexdigest()


def norm_text(value) -> str:
    s = "" if value is None else str(value)
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def first_answer(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        if not value:
            return ""
        x = value[0]
        if isinstance(x, dict):
            return str(x.get("answer", ""))
        return str(x)
    if isinstance(value, dict):
        return str(value.get("answer", ""))
    return str(value)


def flatten_answers(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for x in value:
            if isinstance(x, dict):
                out.append(str(x.get("answer", "")))
            else:
                out.append(str(x))
        return out
    return [str(value)]


def qa_keys(question: str, answers) -> List[str]:
    q = norm_text(question)
    keys = set()
    for a in flatten_answers(answers):
        na = norm_text(a)
        if q and na:
            keys.add(f"{q}\t{na}")
    return sorted(keys)


def conv_question_answer(conversations) -> Tuple[str, str]:
    if not isinstance(conversations, list):
        return "", ""
    q = ""
    a = ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", turn.get("from", ""))).lower()
        text = str(turn.get("content", turn.get("value", "")))
        if not q and role in {"user", "human"}:
            q = text.replace("<image>", " ")
        elif q and not a and role in {"assistant", "gpt"}:
            a = text
            break
    return q, a


@dataclass
class EvalSample:
    eval_dataset: str
    eval_index: int
    sample_id: str
    image_name: str = ""
    image_id: str = ""
    question_id: str = ""
    question: str = ""
    answers: List[str] = None
    image_sha256: str = ""
    image_canon_sha256: str = ""
    source: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["answers"] = d["answers"] or []
        return d


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, rows: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_textvqa_eval(uri: str) -> List[EvalSample]:
    log(f"loading eval TextVQA: {uri}")
    image_bytes: Dict[str, bytes] = {}
    records: Dict[str, dict] = {}
    for member, f in stream_tar_members(uri):
        name = os.path.basename(member.name)
        stem, ext = os.path.splitext(name)
        ext = ext.lower()
        data = f.read()
        if ext in IMAGE_EXTS:
            image_bytes[stem] = data
        elif ext == ".json":
            records[stem] = json.loads(data.decode("utf-8"))
    out: List[EvalSample] = []
    for image_id in sorted(records):
        rec = records[image_id]
        img_sha = sha256_bytes(image_bytes[image_id]) if image_id in image_bytes else ""
        img_canon = canonical_image_sha256(image_bytes[image_id]) if image_id in image_bytes else ""
        for qa in rec.get("qas", []):
            idx = len(out)
            qid = str(qa.get("question_id", ""))
            out.append(
                EvalSample(
                    eval_dataset="textvqa_val",
                    eval_index=idx,
                    sample_id=qid or f"{image_id}:{idx}",
                    image_name=f"{image_id}.jpg",
                    image_id=str(rec.get("image_id", image_id)),
                    question_id=qid,
                    question=qa.get("question", ""),
                    answers=flatten_answers(qa.get("answers", [])),
                    image_sha256=img_sha,
                    image_canon_sha256=img_canon,
                    source="textvqa",
                )
            )
    log(f"TextVQA eval samples={len(out)} images={len(records)}")
    return out


def read_pixelbench_manifest(root: str, bench: str) -> List[dict]:
    uri = f"{root.rstrip('/')}/{bench}/metadata.tar"
    for member, f in stream_tar_members(uri):
        if member.name == "manifest.jsonl":
            return [json.loads(line) for line in f.read().decode("utf-8").splitlines() if line.strip()]
    raise FileNotFoundError(f"manifest.jsonl not found in {uri}")


def read_pixelbench_hashes(root: str, bench: str) -> Dict[str, Tuple[str, str]]:
    uri = f"{root.rstrip('/')}/{bench}/images.tar"
    out = {}
    for member, f in stream_tar_members(uri):
        data = f.read()
        out[member.name] = (sha256_bytes(data), canonical_image_sha256(data))
    return out


def load_pixelbench_eval(root: str, benches: Iterable[str]) -> List[EvalSample]:
    out: List[EvalSample] = []
    for bench in benches:
        log(f"loading eval PixelBench/{bench}: {root}")
        rows = read_pixelbench_manifest(root, bench)
        hashes = read_pixelbench_hashes(root, bench)
        for row_i, row in enumerate(rows):
            answers = row.get("answers", None)
            if answers is None:
                answers = [row.get("answer", "")]
            image_name = row.get("image", "")
            out.append(
                EvalSample(
                    eval_dataset=bench,
                    eval_index=row_i,
                    sample_id=str(row.get("id", row_i)),
                    image_name=image_name,
                    image_id=str(row.get("source_image", image_name)),
                    question_id=str(row.get("id", row_i)),
                    question=row.get("question", row.get("text", "")),
                    answers=flatten_answers(answers),
                    image_sha256=hashes.get(image_name, ("", ""))[0],
                    image_canon_sha256=hashes.get(image_name, ("", ""))[1],
                    source=str(row.get("benchmark", bench)),
                )
            )
        log(f"PixelBench/{bench} samples={len(rows)} images={len(hashes)}")
    return out


def load_vqav2_eval(root: str) -> List[EvalSample]:
    urls = gcs_ls(f"{root.rstrip('/')}/shard-*.tar")
    log(f"loading eval VQAv2: shards={len(urls)} root={root}")
    out: List[EvalSample] = []
    for shard_i, uri in enumerate(urls):
        images: Dict[str, bytes] = {}
        records: Dict[str, dict] = {}
        for member, f in stream_tar_members(uri):
            name = os.path.basename(member.name)
            stem, ext = os.path.splitext(name)
            ext = ext.lower()
            data = f.read()
            if ext in IMAGE_EXTS:
                images[stem] = data
            elif ext == ".json":
                records[stem] = json.loads(data.decode("utf-8"))
        for image_key in sorted(records):
            rec = records[image_key]
            img_sha = sha256_bytes(images[image_key]) if image_key in images else ""
            img_canon = canonical_image_sha256(images[image_key]) if image_key in images else ""
            image_id = str(rec.get("image_id", image_key))
            for qa in rec.get("qas", []):
                idx = len(out)
                qid = str(qa.get("question_id", ""))
                out.append(
                    EvalSample(
                        eval_dataset="vqav2_val",
                        eval_index=idx,
                        sample_id=qid or f"{image_id}:{idx}",
                        image_name=f"{image_key}.jpg",
                        image_id=image_id,
                        question_id=qid,
                        question=qa.get("question", ""),
                        answers=flatten_answers(qa.get("answers", [])),
                        image_sha256=img_sha,
                        image_canon_sha256=img_canon,
                        source="vqav2",
                    )
                )
        if (shard_i + 1) % 16 == 0:
            log(f"VQAv2 eval loaded shards {shard_i + 1}/{len(urls)} samples={len(out)}")
    log(f"VQAv2 eval samples={len(out)}")
    return out


def refcocog_image_name_candidates(name: str) -> List[str]:
    name = (name or "").strip()
    if not name:
        return []
    cands = [name]
    m = re.match(r"^(COCO_(?:train|val)2014_\d{12})_\d+\.(jpg|jpeg|png)$", name, flags=re.IGNORECASE)
    if m is not None:
        cands.append(f"{m.group(1)}.{m.group(2)}")
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


def load_refcocog_eval(ann_path: str, image_root: str) -> List[EvalSample]:
    log(f"loading eval RefCOCOg ann={ann_path} image_root={image_root}")
    with open(ann_path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    rows = json.loads(txt) if txt.startswith("[") else [json.loads(x) for x in txt.splitlines() if x.strip()]
    wanted = {}
    parsed = []
    for idx, row in enumerate(rows):
        image = str(row.get("image") or row.get("image_path") or row.get("file_name") or row.get("img") or "").strip()
        if not image:
            continue
        phrase = row.get("phrase") or row.get("ref") or row.get("expression") or row.get("caption") or row.get("sent") or row.get("sentence")
        if phrase is None:
            ans = row.get("answer")
            if isinstance(ans, list):
                phrase = next((str(x).strip() for x in ans if str(x).strip()), "")
            elif ans is not None:
                phrase = str(ans)
        phrase = "" if phrase is None else str(phrase).strip()
        sample_id = str(row.get("id", row.get("ann_id", row.get("ref_id", row.get("question_id", idx)))))
        cands = refcocog_image_name_candidates(image)
        parsed.append((idx, sample_id, image, cands, phrase))
        for c in cands:
            wanted[os.path.basename(c)] = None

    urls = gcs_ls(f"{image_root.rstrip('/')}/shard-*.tar")
    log(f"RefCOCOg wants {len(wanted)} candidate image names; scanning COCO shards={len(urls)}")
    found_hashes = {}
    remaining = set(wanted)
    for shard_i, uri in enumerate(urls):
        if not remaining:
            break
        for member, f in stream_tar_members(uri):
            base = os.path.basename(member.name)
            if base not in remaining:
                # Consume member stream.
                while f.read(1024 * 1024):
                    pass
                continue
            data = f.read()
            found_hashes[base] = (sha256_bytes(data), canonical_image_sha256(data))
            remaining.remove(base)
        log(f"RefCOCOg image scan shard {shard_i + 1}/{len(urls)} found={len(found_hashes)} remaining={len(remaining)}")

    out = []
    for idx, sample_id, image, cands, phrase in parsed:
        exact = canon = matched = ""
        for c in cands:
            base = os.path.basename(c)
            if base in found_hashes:
                matched = base
                exact, canon = found_hashes[base]
                break
        out.append(
            EvalSample(
                eval_dataset="refcocog_val",
                eval_index=len(out),
                sample_id=sample_id,
                image_name=matched or image,
                image_id=matched or image,
                question_id=sample_id,
                question=phrase,
                answers=[phrase] if phrase else [],
                image_sha256=exact,
                image_canon_sha256=canon,
                source="refcocog",
            )
        )
    log(f"RefCOCOg eval samples={len(out)} images_found={len(found_hashes)}")
    return out


def build_eval_index(samples: List[EvalSample]) -> dict:
    by_hash = defaultdict(list)
    by_canon_hash = defaultdict(list)
    by_question = defaultdict(list)
    by_qa = defaultdict(list)
    by_image_id = defaultdict(list)
    for s in samples:
        d = s.to_json()
        if s.image_sha256:
            by_hash[s.image_sha256].append(d)
        if s.image_canon_sha256:
            by_canon_hash[s.image_canon_sha256].append(d)
        if s.image_id:
            by_image_id[f"{s.source}:{s.image_id}"].append(d)
            by_image_id[s.image_id].append(d)
        nq = norm_text(s.question)
        if nq:
            by_question[nq].append(d)
        for key in qa_keys(s.question, s.answers):
            by_qa[key].append(d)
    return {
        "by_hash": by_hash,
        "by_canon_hash": by_canon_hash,
        "by_question": by_question,
        "by_qa": by_qa,
        "by_image_id": by_image_id,
    }


def list_llava_configs(llava_root: str) -> List[str]:
    paths = gcs_ls(f"{llava_root.rstrip('/')}/")
    names = []
    for p in paths:
        p = p.rstrip("/")
        name = p.rsplit("/", 1)[-1]
        if name and not name.endswith(".tar") and name not in {"_SUCCESS", "summary.json", ":"}:
            names.append(name)
    return sorted(set(names))


def list_llava_shards(llava_root: str, configs: Optional[List[str]] = None) -> List[str]:
    cfgs = configs if configs is not None else list_llava_configs(llava_root)
    urls = []
    for cfg in cfgs:
        urls.extend(gcs_ls(f"{llava_root.rstrip('/')}/{cfg}/shard-*.tar"))
    return sorted(urls)


def parse_llava_shard(
    uri: str,
    eval_index: dict,
    out_dir: str,
    scan_images: bool,
    scan_text: bool,
    scan_canonical: bool,
) -> Counter:
    counts = Counter()
    config = uri.rstrip("/").split("/")[-2]
    shard = uri.rsplit("/", 1)[-1]
    pending_image_hits = {}
    for member, f in stream_tar_members(uri):
        name = member.name
        stem, ext = os.path.splitext(os.path.basename(name))
        ext = ext.lower()
        if ext in IMAGE_EXTS:
            counts["train_images"] += 1
            if not scan_images:
                # Consume the stream without keeping bytes.
                while f.read(1024 * 1024):
                    pass
                continue
            h = hashlib.sha256()
            chunks = [] if scan_canonical else None
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            digest = h.hexdigest()
            hits = eval_index["by_hash"].get(digest)
            if hits:
                rows = []
                for ev in hits:
                    rows.append(
                        {
                            "type": "exact_image_sha256",
                            "train_config": config,
                            "train_shard": shard,
                            "train_member": name,
                            "train_key": stem,
                            "image_sha256": digest,
                            "eval": ev,
                        }
                    )
                append_jsonl(os.path.join(out_dir, "duplicates_image_exact.jsonl"), rows)
                counts["image_hash_hits"] += len(rows)
                pending_image_hits[stem] = rows
            if scan_canonical and chunks is not None:
                try:
                    canon = canonical_image_sha256(b"".join(chunks))
                except Exception:
                    counts["bad_image_decode"] += 1
                    canon = ""
                if canon:
                    chits = eval_index["by_canon_hash"].get(canon)
                    if chits:
                        rows = []
                        for ev in chits:
                            rows.append(
                                {
                                    "type": "canonical_image_sha256_rgb256",
                                    "train_config": config,
                                    "train_shard": shard,
                                    "train_member": name,
                                    "train_key": stem,
                                    "image_canon_sha256": canon,
                                    "eval": ev,
                                }
                            )
                        append_jsonl(os.path.join(out_dir, "duplicates_image_canonical.jsonl"), rows)
                        counts["image_canonical_hits"] += len(rows)
        elif ext == ".json":
            counts["train_json"] += 1
            if not scan_text:
                f.read()
                continue
            try:
                meta = json.loads(f.read().decode("utf-8"))
            except Exception as e:
                counts["bad_json"] += 1
                continue
            q, a = conv_question_answer(meta.get("conversations", []))
            rows = []
            nq = norm_text(q)
            if nq and nq in eval_index["by_question"]:
                for ev in eval_index["by_question"][nq]:
                    rows.append(
                        {
                            "type": "question_text",
                            "train_config": config,
                            "train_shard": shard,
                            "train_member": name,
                            "train_key": stem,
                            "train_id": meta.get("id", ""),
                            "train_orig_row_index": meta.get("orig_row_index", None),
                            "train_question": q,
                            "train_answer": a,
                            "eval": ev,
                        }
                    )
            for key in qa_keys(q, [a]):
                if key in eval_index["by_qa"]:
                    for ev in eval_index["by_qa"][key]:
                        rows.append(
                            {
                                "type": "qa_pair_text",
                                "train_config": config,
                                "train_shard": shard,
                                "train_member": name,
                                "train_key": stem,
                                "train_id": meta.get("id", ""),
                                "train_orig_row_index": meta.get("orig_row_index", None),
                                "train_question": q,
                                "train_answer": a,
                                "eval": ev,
                            }
                        )
            if rows:
                append_jsonl(os.path.join(out_dir, "duplicates_text.jsonl"), rows)
                counts["text_hits"] += len(rows)
    return counts


def quick_textvqa_check(args, eval_samples: List[EvalSample], eval_index: dict, out_dir: str) -> dict:
    log("quick check: scanning LLaVA-OV config textvqa")
    counts = Counter()
    for uri in list_llava_shards(args.llava_root, ["textvqa"]):
        log(f"scan {uri}")
        c = parse_llava_shard(uri, eval_index, out_dir, scan_images=True, scan_text=True, scan_canonical=True)
        counts.update(c)
        log(f"textvqa shard done {uri}: {dict(c)}")
    return dict(counts)


def all_exact_check(args, eval_index: dict, out_dir: str) -> dict:
    configs = list_llava_configs(args.llava_root)
    if args.include_configs:
        keep = set(args.include_configs.split(","))
        configs = [c for c in configs if c in keep]
    if args.exclude_configs:
        drop = set(args.exclude_configs.split(","))
        configs = [c for c in configs if c not in drop]
    urls = list_llava_shards(args.llava_root, configs)
    if args.max_train_tars > 0:
        urls = urls[: args.max_train_tars]
    log(f"all-exact: configs={len(configs)} shards={len(urls)} scan_images={args.scan_images} scan_text={args.scan_text}")
    counts = Counter()
    start = time.time()
    for i, uri in enumerate(urls):
        log(f"train shard {i + 1}/{len(urls)} {uri}")
        c = parse_llava_shard(
            uri,
            eval_index,
            out_dir,
            scan_images=args.scan_images,
            scan_text=args.scan_text,
            scan_canonical=args.scan_canonical,
        )
        counts.update(c)
        if (i + 1) % args.progress_every == 0:
            write_json(
                os.path.join(out_dir, "progress.json"),
                {
                    "done_shards": i + 1,
                    "total_shards": len(urls),
                    "counts": dict(counts),
                    "elapsed_sec": time.time() - start,
                },
            )
            log(f"progress {i + 1}/{len(urls)} counts={dict(counts)}")
    return dict(counts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["quick", "all-exact"], default="quick")
    p.add_argument("--zone-short", default="us-east5")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--llava-root", default="gs://kmh-gcp-us-east5/data/llava-ov-1.5-instruct/configs")
    p.add_argument("--textvqa-root", default="gs://kmh-gcp-us-east5/data/textvqa/val/shard-000000.tar")
    p.add_argument("--pixelbench-root", default="gs://kmh-gcp-us-east5/data/eval/pixelbench")
    p.add_argument("--vqav2-root", default="gs://kmh-gcp-us-east5/data/vqav2/vqav2_image_records_wds/val2014")
    p.add_argument("--refcocog-ann", default="/kmh-nfs-ssd-us-mount/code/hanhong/shared/refcocog/val.json")
    p.add_argument("--refcocog-image-root", default="gs://kmh-gcp-us-east5/data/coco/train2014")
    p.add_argument("--evals", default="textvqa,pixelbench", help="comma list: textvqa,pixelbench,vqav2,refcocog")
    p.add_argument("--pixelbench-benches", default="mmvp,vstar,ocrbench,countbenchqa")
    p.add_argument("--scan-images", action="store_true", default=False)
    p.add_argument("--scan-canonical", action="store_true", default=False)
    p.add_argument("--scan-text", action="store_true", default=False)
    p.add_argument("--include-configs", default="")
    p.add_argument("--exclude-configs", default="")
    p.add_argument("--max-train-tars", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=25)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log(f"out_dir={args.out_dir}")
    write_json(os.path.join(args.out_dir, "args.json"), vars(args))

    configs = list_llava_configs(args.llava_root)
    write_json(os.path.join(args.out_dir, "llava_configs.json"), configs)
    suspicious = [c for c in configs if re.search(r"textvqa|vstar|mmvp|mme|pope|mmbench|vqav2|refcoco", c, re.I)]
    log(f"LLaVA-OV configs={len(configs)} suspicious={suspicious}")

    eval_names = {x.strip() for x in args.evals.split(",") if x.strip()}
    eval_samples: List[EvalSample] = []
    if "textvqa" in eval_names:
        eval_samples.extend(load_textvqa_eval(args.textvqa_root))
    if "pixelbench" in eval_names:
        benches = [x.strip() for x in args.pixelbench_benches.split(",") if x.strip()]
        eval_samples.extend(load_pixelbench_eval(args.pixelbench_root, benches))
    if "vqav2" in eval_names:
        eval_samples.extend(load_vqav2_eval(args.vqav2_root))
    if "refcocog" in eval_names:
        eval_samples.extend(load_refcocog_eval(args.refcocog_ann, args.refcocog_image_root))

    append_jsonl(os.path.join(args.out_dir, "eval_manifest.jsonl"), [s.to_json() for s in eval_samples])
    by_dataset = Counter(s.eval_dataset for s in eval_samples)
    eval_images_by_dataset = defaultdict(set)
    eval_canon_images_by_dataset = defaultdict(set)
    for s in eval_samples:
        if s.image_sha256:
            eval_images_by_dataset[s.eval_dataset].add(s.image_sha256)
        if s.image_canon_sha256:
            eval_canon_images_by_dataset[s.eval_dataset].add(s.image_canon_sha256)
    eval_index = build_eval_index(eval_samples)
    write_json(
        os.path.join(args.out_dir, "eval_summary.json"),
        {
            "samples_by_dataset": dict(by_dataset),
            "unique_image_hashes_by_dataset": {k: len(v) for k, v in eval_images_by_dataset.items()},
            "unique_canonical_image_hashes_by_dataset": {k: len(v) for k, v in eval_canon_images_by_dataset.items()},
            "unique_image_hashes_total": len(eval_index["by_hash"]),
            "unique_canonical_image_hashes_total": len(eval_index["by_canon_hash"]),
            "unique_questions_total": len(eval_index["by_question"]),
            "unique_qa_pairs_total": len(eval_index["by_qa"]),
            "llava_suspicious_configs": suspicious,
        },
    )

    start = time.time()
    if args.mode == "quick":
        train_counts = quick_textvqa_check(args, eval_samples, eval_index, args.out_dir)
    else:
        train_counts = all_exact_check(args, eval_index, args.out_dir)

    summary = {
        "mode": args.mode,
        "elapsed_sec": time.time() - start,
        "eval": {
            "samples_by_dataset": dict(by_dataset),
            "unique_image_hashes_total": len(eval_index["by_hash"]),
            "unique_canonical_image_hashes_total": len(eval_index["by_canon_hash"]),
            "unique_questions_total": len(eval_index["by_question"]),
            "unique_qa_pairs_total": len(eval_index["by_qa"]),
        },
        "llava": {
            "num_configs": len(configs),
            "suspicious_configs": suspicious,
            "train_counts": train_counts,
        },
        "outputs": {
            "eval_manifest": os.path.join(args.out_dir, "eval_manifest.jsonl"),
            "duplicates_image_exact": os.path.join(args.out_dir, "duplicates_image_exact.jsonl"),
            "duplicates_image_canonical": os.path.join(args.out_dir, "duplicates_image_canonical.jsonl"),
            "duplicates_text": os.path.join(args.out_dir, "duplicates_text.jsonl"),
        },
    }
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    log(f"done summary={os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
