#!/usr/bin/env python3
"""Count per-config LLaVA-OV1.5 image and question (QA-pair) totals.

Run on a VM in the same region as the dataset bucket. Image counts are read
exactly from each config's summary.json. Question counts are estimated by
sampling a few shards per config, expanding each image with the SAME
`expand_llava_sample` used in training (so a "question" == one emitted training
example), computing the average QA-pairs-per-image, and scaling by the exact
image total. Writes a JSON mapping for embedding into llava_ov15_groups.py.
"""

import argparse
import io
import json
import os
import signal
import subprocess
import sys
import time


class _ShardTimeout(Exception):
    pass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import webdataset as wds  # noqa: E402

from utils.llava_ov15_groups import (  # noqa: E402
    LLAVA_OV15_GROUPS,
    LLAVA_OV15_CONFIG_SHARDS,
    LLAVA_OV15_CONFIG_ROOT,
)
import input_pipeline  # noqa: E402

expand_llava_sample = input_pipeline.expand_llava_sample


class _PopenStdout:
    def __init__(self, proc):
        self._proc = proc
        self._stdout = proc.stdout

    def read(self, *a, **k):
        return self._stdout.read(*a, **k)

    def close(self):
        try:
            self._stdout.close()
        finally:
            rc = self._proc.wait()
            if rc not in (0, -13, 141):
                raise RuntimeError(f"gsutil cat exited {rc}")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def _register_gsutil():
    import importlib
    gopen = importlib.import_module("webdataset.gopen")

    def opener(url, mode="rb", bufsize=8192, **kw):
        proc = subprocess.Popen(["gsutil", "cat", url], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=bufsize)
        return _PopenStdout(proc)
    gopen.gopen_schemes["gs"] = opener


def _gsutil_cat_text(url, timeout=60):
    try:
        out = subprocess.run(["gsutil", "cat", url], capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] summary read {url}", flush=True)
        return None
    if out.returncode != 0:
        return None
    return out.stdout.decode("utf-8", "replace")


def _sample_shard_indices(n_shards, n_sample):
    if n_shards <= n_sample:
        return list(range(n_shards))
    # evenly spread across the range (captures any per-shard answer structure)
    return sorted({int(round(i * (n_shards - 1) / (n_sample - 1))) for i in range(n_sample)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone-short", default="us-east5",
                    choices=["us-central1", "us-east5", "asia-northeast1-b"])
    ap.add_argument("--sample-shards", type=int, default=3,
                    help="Shards sampled per config to estimate QA-per-image.")
    ap.add_argument("--max-images-per-shard", type=int, default=1200,
                    help="Stop reading a sampled shard after this many images (bounds time).")
    ap.add_argument("--shard-timeout", type=int, default=90,
                    help="Skip a sampled shard if it has not finished in this many seconds.")
    ap.add_argument("--out", default="tmp_audit/ov15_counts.json")
    args = ap.parse_args()

    _register_gsutil()
    root = LLAVA_OV15_CONFIG_ROOT.replace("💣", args.zone_short)
    configs = [c for g in LLAVA_OV15_GROUPS for c in g["configs"]]
    print(f"configs={len(configs)} zone={args.zone_short} root={root}", flush=True)

    # resume: keep already-measured configs
    results = {}
    if os.path.exists(args.out):
        try:
            results = json.load(open(args.out))
            print(f"resume: {len(results)} configs already done", flush=True)
        except Exception:
            results = {}

    def _alarm(signum, frame):
        raise _ShardTimeout()
    signal.signal(signal.SIGALRM, _alarm)

    t0 = time.time()
    for idx, cfg in enumerate(configs):
        if cfg in results:
            continue
        n_shards = int(LLAVA_OV15_CONFIG_SHARDS[cfg])
        # exact image count from summary.json
        summ = _gsutil_cat_text(f"{root}/{cfg}/summary.json")
        images = None
        if summ:
            try:
                images = int(json.loads(summ).get("samples"))
            except Exception:
                images = None
        # sample shards -> avg QA-pairs per image
        n_img_s = 0
        n_qa_s = 0
        for si in _sample_shard_indices(n_shards, args.sample_shards):
            url = f"{root}/{cfg}/shard-{si:06d}.tar"
            shard_imgs = 0
            signal.alarm(int(args.shard_timeout))
            try:
                ds = wds.DataPipeline(wds.SimpleShardList([url]),
                                      wds.tarfile_to_samples())
                for sample in ds:
                    if sample is None:
                        continue
                    n_img_s += 1
                    shard_imgs += 1
                    try:
                        n_qa_s += len(expand_llava_sample(sample))
                    except Exception:
                        pass
                    if shard_imgs >= int(args.max_images_per_shard):
                        break
            except _ShardTimeout:
                print(f"  [TIMEOUT] {cfg} shard {si} after {args.shard_timeout}s "
                      f"({shard_imgs} imgs read)", flush=True)
            except Exception as e:
                print(f"  [WARN] {cfg} shard {si}: {e}", flush=True)
            finally:
                signal.alarm(0)
        avg_k = (n_qa_s / n_img_s) if n_img_s > 0 else 0.0
        if images is None:
            # fall back to shard-proxy image estimate if summary.json missing
            images = int(round(n_shards * (n_img_s / max(1, len(_sample_shard_indices(n_shards, args.sample_shards))))))
        questions = int(round(images * avg_k))
        results[cfg] = {"images": images, "shards": n_shards,
                        "avg_qa_per_image": round(avg_k, 4), "questions": questions,
                        "sampled_images": n_img_s}
        print(f"[{idx+1:3d}/{len(configs)}] {cfg:32s} images={images:>9d} "
              f"avg_k={avg_k:5.2f} questions={questions:>10d} "
              f"(sampled {n_img_s} imgs) elapsed={time.time()-t0:.0f}s", flush=True)
        # incremental write so a preemption still leaves partial results
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    tot_img = sum(r["images"] for r in results.values())
    tot_q = sum(r["questions"] for r in results.values())
    print(f"DONE configs={len(results)} total_images={tot_img} total_questions={tot_q} "
          f"global_avg_k={tot_q/max(1,tot_img):.3f} out={args.out}", flush=True)


if __name__ == "__main__":
    main()
