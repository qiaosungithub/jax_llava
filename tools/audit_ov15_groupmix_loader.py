#!/usr/bin/env python3
"""Audit LLaVA-OV1.5 grouped-mixture dataloader shuffle stability.

Run this on a VM in the same region as the dataset bucket. It never downloads
whole datasets manually; it streams WebDataset samples through the normal input
pipeline and prints valid-token statistics over time.
"""

import argparse
import copy
import csv
import os
import pickle
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Keep this script a CPU-side loader audit. Some TPU VMs have accelerators, but
# importing JAX for process metadata is enough; avoid accidental multi-process
# TPU initialization in subprocesses.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from configs.load_config import get_config
from utils.data_util import resolve_dataset_roots
import input_pipeline


class _PopenStdout:
    def __init__(self, proc):
        self._proc = proc
        self._stdout = proc.stdout

    def read(self, *args, **kwargs):
        return self._stdout.read(*args, **kwargs)

    def close(self):
        try:
            self._stdout.close()
        finally:
            rc = self._proc.wait()
            if rc not in (0, -13, 141):
                raise RuntimeError(f"gsutil cat exited with code {rc}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _register_gsutil_gopen():
    """Use gsutil cat for gs:// streams to avoid requiring gcsfs on debug VMs."""
    import importlib

    gopen_module = importlib.import_module("webdataset.gopen")

    def gopen_gsutil(url, mode="rb", bufsize=8192, **kwargs):
        if "r" not in mode:
            raise ValueError(f"gsutil opener is read-only, got mode={mode}")
        proc = subprocess.Popen(
            ["gsutil", "cat", url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=bufsize,
        )
        return _PopenStdout(proc)

    gopen_module.gopen_schemes["gs"] = gopen_gsutil


def _build_curriculum_stage_config(config, stage_key, *, stage_start_step, stage_end_step, total_steps):
    stage = config.training.get(stage_key, None)
    if stage is None:
        raise ValueError(f"Missing training.{stage_key} in curriculum config")
    phase_config = copy.deepcopy(config)
    stage_steps = int(stage_end_step) - int(stage_start_step)
    if stage_steps <= 0:
        raise ValueError(f"Invalid {stage_key} step range: {stage_start_step} -> {stage_end_step}")

    dataset_items = stage.get("dataset_items", stage.get("items", None))
    if dataset_items is not None:
        phase_config.dataset["items"] = copy.deepcopy(dataset_items)
    if stage.get("mix_weights", None) is not None:
        phase_config.dataset["mix_weights"] = copy.deepcopy(stage.mix_weights)
    if stage.get("dataset", None) is not None:
        for key, value in stage.dataset.items():
            target_key = "max_txt_len" if key == "max_txt_length" else key
            if target_key in {"items", "dataset_items"}:
                phase_config.dataset["items"] = copy.deepcopy(value)
            elif target_key == "mix_weights":
                phase_config.dataset["mix_weights"] = copy.deepcopy(value)
            else:
                if target_key not in phase_config.dataset:
                    raise ValueError(f"Unsupported stage dataset override: {key}")
                phase_config.dataset[target_key] = copy.deepcopy(value)
    if stage.get("model", None) is not None:
        phase_config.model.update(copy.deepcopy(stage.model))

    training_keys = [
        "batch_size", "freeze_lm", "freeze_lm_embed", "freeze_lm_late",
        "freeze_image_encoder", "vision_tower_from_scratch",
        "clip_from_pt", "hf_cache_dir", "optimizer", "grad_clip_norm", "log_per_step",
        "checkpoint_per_step", "log_vis_per_step", "sample_per_step", "online_eval_per_step",
        "online_eval_tasks", "final_eval_tasks", "warmup_steps",
        "lr_schedule", "seed", "vision_encoder_learning_rate",
        "connector_learning_rate", "projector_learning_rate",
        "exclude_bias_norm_from_weight_decay",
    ]
    for key in training_keys:
        if stage.get(key, None) is not None:
            phase_config.training[key] = copy.deepcopy(stage[key])
    for nested_key in ["adam", "muon"]:
        if stage.get(nested_key, None) is not None:
            phase_config.training[nested_key].update(copy.deepcopy(stage[nested_key]))
    if stage.get("sampling", None) is not None:
        phase_config.sampling.update(copy.deepcopy(stage.sampling))
    if stage.get("eval", None) is not None:
        phase_config.eval.update(copy.deepcopy(stage.eval))
    if stage.get("logging", None) is not None:
        phase_config.logging.update(copy.deepcopy(stage.logging))

    phase_config.training.num_steps = stage_steps
    phase_config.training.curriculum_stage_name = stage.get("name", stage_key)
    phase_config.training.curriculum_stage_key = stage_key
    phase_config.training.curriculum_stage_index = 1 if stage_key == "stage1" else 2
    phase_config.training.curriculum_stage_start_step = int(stage_start_step)
    phase_config.training.curriculum_stage_end_step = int(stage_end_step)
    phase_config.training.curriculum_global_num_steps = int(total_steps)
    phase_config.finetune = False
    return phase_config


def _token_valid_count_llava_ov15(sample, tokenizer, max_len):
    question_part = (sample.get("question", "") or "").strip()
    answer_part = (sample.get("aux", {}) or {}).get("answer", "")
    answer_part = "" if answer_part is None else str(answer_part).strip()
    if not answer_part:
        return None

    prompt_for_mask = f"{question_part}\n" if question_part else ""
    full_text = f"{prompt_for_mask}{answer_part}"
    prefix_tokens = tokenizer.encode(prompt_for_mask, add_bos=True, add_eos=False)
    prefix_len = min(len(prefix_tokens), int(max_len))

    token_ids = tokenizer.encode(full_text, add_bos=True, add_eos=True)
    if len(token_ids) > int(max_len) + 1:
        token_ids = token_ids[: int(max_len) + 1]

    input_len = max(0, min(int(max_len), len(token_ids) - 1))
    mask_len = min(prefix_len - 1, int(max_len)) if prefix_len > 1 else 0
    return max(0, input_len - min(mask_len, input_len))


def _config_name_from_url(url):
    """Parse the OV1.5 config name from a shard URL.

    URLs look like .../llava-ov-1.5-instruct/configs/<config>/shard-XXXXXX.tar
    """
    if not url:
        return "<unknown>"
    marker = "/configs/"
    idx = url.find(marker)
    if idx < 0:
        return "<unknown>"
    rest = url[idx + len(marker):]
    return rest.split("/", 1)[0] or "<unknown>"


class FastOVTextDataset:
    def __init__(self, root_url, config, tokenizer, shuffle_size, data_seed_offset=0,
                 collect_meta=False, source_name=None, expand_at_fill=False):
        self.root_url = root_url
        self.config = config
        self.tokenizer = tokenizer
        self.max_len = int(config.max_txt_len)
        self.shuffle_size = int(shuffle_size)
        self.data_seed_offset = int(data_seed_offset)
        self.shard_rank = 0
        self.dataset_type = "llava_ov15"
        self.collect_meta = bool(collect_meta)
        self.source_name = source_name
        self.expand_at_fill = bool(expand_at_fill)

    def __iter__(self):
        return FastOVTextIterator(self)


class FastOVTextIterator:
    """Text-only audit iterator with the same raw-image shuffle state machine.

    This skips PIL decode/resize and tensor collation, but keeps:
      * the exact raw shard iterator,
      * the same raw-image shuffle buffer and pending-question behavior,
      * the same LLaVA conversation expansion,
      * the same weighted StatefulRandomMix selection upstream.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.raw_iter = input_pipeline._StatefulRawShardIterator(
            dataset.root_url,
            dataset.config,
            dataset.data_seed_offset,
        )
        self.rng = random.Random(
            input_pipeline._worker_seed(2027, dataset.shard_rank, dataset.data_seed_offset)
        )
        self.choice_rng = random.Random(
            input_pipeline._worker_seed(2039, dataset.shard_rank, dataset.data_seed_offset)
        )
        self.expand_fn = input_pipeline._EXPAND_FN["llava_ov15"]
        self.shuffle_buf = []
        start_skip_max = input_pipeline._stream_start_skip(dataset.config, dataset.dataset_type)
        self.start_skip_remaining = (
            self.rng.randrange(start_skip_max + 1) if start_skip_max > 0 else 0
        )

    def __iter__(self):
        return self

    def _pop_random_entry(self):
        idx = self.rng.randrange(len(self.shuffle_buf))
        entry = self.shuffle_buf[idx]
        self.shuffle_buf[idx] = self.shuffle_buf[-1]
        self.shuffle_buf.pop()
        return entry

    def _emit_one_from_buffer(self):
        kind, payload = self._pop_random_entry()
        if kind == "pending":
            config_name, items = payload
            from_pending = True
        else:
            from_pending = False
            config_name = _config_name_from_url(payload.get("__url__"))
            items = input_pipeline._with_module_random(
                self.choice_rng, self.expand_fn, payload
            )
            if not items:
                return None
            self.rng.shuffle(items)

        chosen = items.pop()
        if items:
            self.shuffle_buf.append(("pending", (config_name, items)))

        valid = _token_valid_count_llava_ov15(
            chosen,
            self.dataset.tokenizer,
            self.dataset.max_len,
        )
        if valid is None:
            return None
        if not self.dataset.collect_meta:
            return valid
        turn_idx = int((chosen.get("aux", {}) or {}).get("turn_idx", -1))
        return (
            int(valid),
            self.dataset.source_name or config_name,
            config_name,
            turn_idx,
            1 if from_pending else 0,
        )

    def _emit_one_item(self):
        # expand_at_fill mode: buffer holds individual expanded QA items, so the
        # emitted turn distribution is stationary from step 1 (no slow-draining
        # pending slot). Each entry is ("item", (config_name, item)).
        _kind, (config_name, chosen) = self._pop_random_entry()
        valid = _token_valid_count_llava_ov15(
            chosen,
            self.dataset.tokenizer,
            self.dataset.max_len,
        )
        if valid is None:
            return None
        if not self.dataset.collect_meta:
            return valid
        turn_idx = int((chosen.get("aux", {}) or {}).get("turn_idx", -1))
        return (
            int(valid),
            self.dataset.source_name or config_name,
            config_name,
            turn_idx,
            0,
        )

    def _fill_items(self):
        # Read raw samples and expand each into independent item entries until
        # the buffer reaches shuffle_size.
        while len(self.shuffle_buf) < self.dataset.shuffle_size:
            sample = next(self.raw_iter)
            if sample is None:
                continue
            if self.start_skip_remaining > 0:
                self.start_skip_remaining -= 1
                continue
            config_name = _config_name_from_url(sample.get("__url__"))
            items = input_pipeline._with_module_random(
                self.choice_rng, self.expand_fn, sample
            )
            for item in items:
                self.shuffle_buf.append(("item", (config_name, item)))

    def __next__(self):
        if self.dataset.expand_at_fill:
            while True:
                self._fill_items()
                out = self._emit_one_item()
                if out is not None:
                    return out
        while True:
            while len(self.shuffle_buf) < self.dataset.shuffle_size:
                sample = next(self.raw_iter)
                if sample is None:
                    continue
                if self.start_skip_remaining > 0:
                    self.start_skip_remaining -= 1
                    continue
                self.shuffle_buf.append(("raw", sample))

            out = self._emit_one_from_buffer()
            if out is not None:
                return out


def _fast_text_shuffle_sizes(cfg):
    weights = list(cfg.dataset.mix_weights)
    types = list(cfg.dataset.types)
    weighted_cfg = getattr(cfg.dataset, "weighted_item_shuffle_size", None)
    include_types = []
    if weighted_cfg is not None and bool(weighted_cfg.get("enabled", True)):
        include_types = input_pipeline._as_config_list(
            weighted_cfg.get("include_types", ["llava_ov15"])
        )
    eligible_weight_sum = sum(
        float(weight)
        for weight, dtype in zip(weights, types)
        if (not include_types or dtype in include_types) and float(weight) > 0
    )
    out = []
    for dtype, weight in zip(types, weights):
        override = input_pipeline._weighted_item_shuffle_size_override(
            cfg.dataset,
            dtype,
            float(weight),
            eligible_weight_sum,
        )
        if override is None:
            override = input_pipeline._item_shuffle_size(cfg.dataset, dtype, 10000)
        out.append(int(override))
    return out


def _create_fast_text_tokenizer(model_name, tokenizer_path=None):
    if tokenizer_path:
        if not str(model_name).startswith("gemma3"):
            raise ValueError("--tokenizer-path is currently implemented for gemma3 tokenizers")
        from utils.llm_util import PaliGemma3Tokenizer

        return PaliGemma3Tokenizer(path=tokenizer_path)
    return input_pipeline.create_tokenizer(model_name)


def _create_fast_text_iterator(cfg, data_seed_offset=0, tokenizer_path=None):
    dataset = _create_fast_text_dataset(
        cfg,
        data_seed_offset=data_seed_offset,
        tokenizer_path=tokenizer_path,
    )
    return iter(dataset)


def _create_fast_text_dataset(cfg, data_seed_offset=0, tokenizer_path=None, collect_meta=False,
                              expand_at_fill=False):
    tokenizer = _create_fast_text_tokenizer(
        cfg.model.lm_backbone_str,
        tokenizer_path=tokenizer_path,
    )
    datasets = []
    shuffle_sizes = _fast_text_shuffle_sizes(cfg)
    for idx, (root, dtype, shuffle_size) in enumerate(
        zip(cfg.dataset.root, cfg.dataset.types, shuffle_sizes)
    ):
        if dtype != "llava_ov15":
            raise ValueError("fast text audit currently supports only llava_ov15 roots")
        source_name = cfg.dataset.resolved_names[idx]
        print(
            f"fast_source[{idx:02d}] name={source_name} "
            f"shuffle_size={shuffle_size}",
            flush=True,
        )
        datasets.append(
            FastOVTextDataset(
                root,
                cfg.dataset,
                tokenizer,
                shuffle_size=shuffle_size,
                data_seed_offset=data_seed_offset,
                collect_meta=collect_meta,
                source_name=source_name,
                expand_at_fill=expand_at_fill,
            )
        )
    return input_pipeline.StatefulRandomMix(
        datasets,
        list(cfg.dataset.mix_weights),
        data_seed_offset=data_seed_offset,
    )


def _collate_valid_counts(batch):
    return np.asarray(batch, dtype=np.int32)


def _collate_meta(batch):
    # batch is a list of (valid, source_name, config_name, turn_idx, from_pending)
    valid = np.asarray([b[0] for b in batch], dtype=np.int32)
    sources = [b[1] for b in batch]
    configs = [b[2] for b in batch]
    turn_idx = np.asarray([b[3] for b in batch], dtype=np.int32)
    from_pending = np.asarray([b[4] for b in batch], dtype=np.int32)
    return {
        "valid": valid,
        "sources": sources,
        "configs": configs,
        "turn_idx": turn_idx,
        "from_pending": from_pending,
    }


def _build_config(args):
    cfg = get_config(args.config)
    if args.stage == "stage2":
        stage1_steps = int(cfg.training.get("stage1_steps", 0))
        stage2_steps = int(cfg.training.get("stage2_steps", 1))
        cfg = _build_curriculum_stage_config(
            cfg,
            "stage2",
            stage_start_step=stage1_steps,
            stage_end_step=stage1_steps + stage2_steps,
            total_steps=stage1_steps + stage2_steps,
        )
    elif args.stage != "raw":
        raise ValueError(f"Unsupported stage: {args.stage}")

    cfg.zone = args.zone_short
    cfg.local_debug = False
    cfg.model.lm_backbone_str = args.lm_backbone
    cfg.dataset.image_size = int(args.image_size)
    cfg.dataset.max_txt_len = int(args.max_txt_len)
    cfg.dataset.num_workers = int(args.num_workers)
    cfg.dataset.prefetch_factor = int(args.prefetch_factor)
    cfg.dataset.pin_memory = False
    cfg.dataset.dataloader_timeout = int(args.dataloader_timeout)
    cfg.dataset.stateful_dataloader = bool(args.stateful)
    cfg.dataset.stateful_dataloader_strict = True
    cfg.dataset.stateful_snapshot_every_n_steps = int(args.snapshot_every)
    cfg.dataset.shuffle_total_streams_override = (
        None
        if int(args.shuffle_total_streams_override) <= 0
        else int(args.shuffle_total_streams_override)
    )
    cfg.training.checkpoint_per_step = int(args.snapshot_every)

    if args.only_ov:
        cfg.dataset.items = [{"name": "llava-ov-1.5-instruct-grouped", "weight": 1.0}]
        cfg.dataset.mix_weights = []

    if int(args.ov_min_shards) > 0:
        try:
            cfg.dataset.llava_ov15_min_shards_standalone = int(args.ov_min_shards)
        except Exception:
            with cfg.unlocked():
                cfg.dataset.llava_ov15_min_shards_standalone = int(args.ov_min_shards)

    resolve_dataset_roots(cfg, args.zone_short)
    return cfg


def _valid_token_stats(batch):
    labels = batch["labels"]
    if hasattr(labels, "numpy"):
        labels = labels.numpy()
    valid_per_sample = (labels != -100).sum(axis=1).astype(np.float64)
    return {
        "mean": float(valid_per_sample.mean()),
        "std": float(valid_per_sample.std()),
        "p05": float(np.percentile(valid_per_sample, 5)),
        "p50": float(np.percentile(valid_per_sample, 50)),
        "p95": float(np.percentile(valid_per_sample, 95)),
        "min": float(valid_per_sample.min()),
        "max": float(valid_per_sample.max()),
    }


def _valid_count_stats(valid_counts):
    valid_per_sample = np.asarray(valid_counts, dtype=np.float64)
    return {
        "mean": float(valid_per_sample.mean()),
        "std": float(valid_per_sample.std()),
        "p05": float(np.percentile(valid_per_sample, 5)),
        "p50": float(np.percentile(valid_per_sample, 50)),
        "p95": float(np.percentile(valid_per_sample, 95)),
        "min": float(valid_per_sample.min()),
        "max": float(valid_per_sample.max()),
    }


def _window_slope(values):
    if len(values) < 2:
        return 0.0
    y = np.asarray(values, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return 0.0
    return float((x * (y - y.mean())).sum() / denom)


def _same_batch(a, b):
    keys = ["input_ids", "attention_mask", "labels", "prefix_len"]
    for key in keys:
        av = a[key].numpy() if hasattr(a[key], "numpy") else np.asarray(a[key])
        bv = b[key].numpy() if hasattr(b[key], "numpy") else np.asarray(b[key])
        if not np.array_equal(av, bv):
            return False, key
    return True, ""


def _state_roundtrip(cfg, batch_size, loader, iterator, step):
    if not hasattr(loader, "state_dict"):
        raise RuntimeError("Stateful loader requested, but loader has no state_dict().")
    state = loader.state_dict()
    original_next = next(iterator)
    restored_loader, _ = input_pipeline.create_split(cfg, batch_size=batch_size)
    restored_loader.load_state_dict(state)
    restored_next = next(iter(restored_loader))
    same, key = _same_batch(original_next, restored_next)
    if not same:
        raise RuntimeError(f"State roundtrip mismatch at step {step}, key={key}")
    return _valid_token_stats(original_next)


def _run_meta_audit(args, cfg):
    """Per-config drift diagnostic for the fast text-only OV1.5 path.

    Aggregates valid-token mean, source/config mixture share, pending-emit
    fraction, and mean turn index per fixed step window, then decomposes the
    early->late win_mean change into a mixture-shift effect and a
    within-config-mean-shift effect (the key diagnosis for whether the drift is
    driven by coarse mixture non-stationarity or by within-config effects).
    """
    from collections import defaultdict
    from torch.utils.data import DataLoader

    fast_dataset = _create_fast_text_dataset(
        cfg, tokenizer_path=args.tokenizer_path or None, collect_meta=True,
        expand_at_fill=bool(args.expand_at_fill),
    )
    loader = DataLoader(
        fast_dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        timeout=int(args.dataloader_timeout),
        collate_fn=_collate_meta,
        pin_memory=False,
    )
    iterator = iter(loader)

    window = int(args.window)
    long_csv = Path(args.csv).with_suffix(".perconfig.csv")
    win_csv = Path(args.csv).with_suffix(".window.csv")
    long_csv.parent.mkdir(parents=True, exist_ok=True)

    # Per-window accumulators.
    win_idx = 0
    win_n = 0
    win_valid_sum = 0.0
    win_pending = 0
    win_turn_sum = 0.0
    win_cfg_n = defaultdict(int)
    win_cfg_valid = defaultdict(float)
    win_src_n = defaultdict(int)
    win_src_valid = defaultdict(float)
    window_records = []  # (win_idx, mean, pending_frac, turn_mean, {cfg:(n,meanvalid)}, {src:...})
    t0 = time.time()

    lf = long_csv.open("w", newline="")
    long_writer = csv.DictWriter(
        lf, fieldnames=["window", "level", "name", "count", "share", "mean_valid"]
    )
    long_writer.writeheader()

    def flush_window():
        nonlocal win_idx, win_n, win_valid_sum, win_pending, win_turn_sum
        if win_n == 0:
            return
        mean = win_valid_sum / win_n
        pending_frac = win_pending / win_n
        turn_mean = win_turn_sum / win_n
        cfg_stats = {
            c: (win_cfg_n[c], win_cfg_valid[c] / max(1, win_cfg_n[c]))
            for c in win_cfg_n
        }
        src_stats = {
            s: (win_src_n[s], win_src_valid[s] / max(1, win_src_n[s]))
            for s in win_src_n
        }
        window_records.append((win_idx, mean, pending_frac, turn_mean, cfg_stats, src_stats))
        for c, (n, mv) in sorted(cfg_stats.items(), key=lambda kv: -kv[1][0]):
            long_writer.writerow({
                "window": win_idx, "level": "config", "name": c,
                "count": n, "share": n / win_n, "mean_valid": round(mv, 4),
            })
        for s, (n, mv) in sorted(src_stats.items(), key=lambda kv: -kv[1][0]):
            long_writer.writerow({
                "window": win_idx, "level": "source", "name": s,
                "count": n, "share": n / win_n, "mean_valid": round(mv, 4),
            })
        lf.flush()
        print(
            f"[win {win_idx:04d}] step~{(win_idx + 1) * window} mean={mean:.3f} "
            f"pending_frac={pending_frac:.3f} turn_mean={turn_mean:.2f} "
            f"n_cfg={len(cfg_stats)} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        # reset
        win_idx += 1
        win_n = 0
        win_valid_sum = 0.0
        win_pending = 0
        win_turn_sum = 0.0
        win_cfg_n.clear()
        win_cfg_valid.clear()
        win_src_n.clear()
        win_src_valid.clear()

    steps_in_win = 0
    for step in range(1, int(args.max_steps) + 1):
        batch = next(iterator)
        valid = batch["valid"]
        win_n += int(valid.shape[0])
        win_valid_sum += float(valid.sum())
        win_pending += int(batch["from_pending"].sum())
        win_turn_sum += float(batch["turn_idx"].clip(min=0).sum())
        for c, v in zip(batch["configs"], valid.tolist()):
            win_cfg_n[c] += 1
            win_cfg_valid[c] += v
        for s, v in zip(batch["sources"], valid.tolist()):
            win_src_n[s] += 1
            win_src_valid[s] += v
        steps_in_win += 1
        if steps_in_win >= window:
            flush_window()
            steps_in_win = 0
        if args.min_seconds > 0 and (time.time() - t0) >= args.min_seconds:
            print(f"min_seconds reached at step={step}", flush=True)
            break
    flush_window()
    lf.close()

    # Decompose early->late win_mean change at config level.
    if len(window_records) >= 2:
        early = window_records[0]
        late = window_records[-1]
        e_cfg, l_cfg = early[4], late[4]
        e_n_total = sum(n for n, _ in e_cfg.values())
        l_n_total = sum(n for n, _ in l_cfg.values())
        names = set(e_cfg) | set(l_cfg)
        mix_effect = 0.0
        within_effect = 0.0
        contrib = []
        for c in names:
            en, em = e_cfg.get(c, (0, 0.0))
            ln, lm = l_cfg.get(c, (0, 0.0))
            es = en / e_n_total if e_n_total else 0.0
            ls = ln / l_n_total if l_n_total else 0.0
            mbar = (em + lm) / 2.0
            sbar = (es + ls) / 2.0
            me = (ls - es) * mbar          # mixture-shift contribution
            we = sbar * (lm - em)          # within-config-mean-shift contribution
            mix_effect += me
            within_effect += we
            contrib.append((c, me, we, es, ls, em, lm))
        with win_csv.open("w", newline="") as wf:
            ww = csv.writer(wf)
            ww.writerow(["window", "mean", "pending_frac", "turn_mean"])
            for rec in window_records:
                ww.writerow([rec[0], round(rec[1], 4), round(rec[2], 4), round(rec[3], 3)])
        print("\n=== EARLY->LATE win_mean DECOMPOSITION ===", flush=True)
        print(f"early_win={early[0]} mean={early[1]:.3f} pending={early[2]:.3f} turn={early[3]:.2f}", flush=True)
        print(f"late_win ={late[0]} mean={late[1]:.3f} pending={late[2]:.3f} turn={late[3]:.2f}", flush=True)
        print(f"total_delta={late[1]-early[1]:+.3f}  mix_effect={mix_effect:+.3f}  within_effect={within_effect:+.3f}", flush=True)
        contrib.sort(key=lambda x: x[1] + x[2])
        print("top mixture+within contributors (most negative first):", flush=True)
        for c, me, we, es, ls, em, lm in contrib[:15]:
            print(
                f"  {c:32s} mix={me:+.3f} within={we:+.3f} "
                f"share {es*100:5.2f}%->{ls*100:5.2f}% mean {em:6.1f}->{lm:6.1f}",
                flush=True,
            )
    print(f"DONE meta perconfig_csv={long_csv} window_csv={win_csv}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="remote_run")
    ap.add_argument("--stage", default="stage2", choices=["stage2", "raw"])
    ap.add_argument("--zone-short", default="us-central1", choices=["us-central1", "us-east5", "asia-northeast1-b"])
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=90000)
    ap.add_argument("--min-seconds", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--csv", default="/tmp/ov15_groupmix_loader_audit.csv")
    ap.add_argument("--only-ov", action="store_true")
    ap.add_argument("--stateful", action="store_true", default=True)
    ap.add_argument("--no-stateful", dest="stateful", action="store_false")
    ap.add_argument("--state-test-step", type=int, default=5)
    ap.add_argument("--snapshot-every", type=int, default=1000000)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--dataloader-timeout", type=int, default=900)
    ap.add_argument("--image-size", type=int, default=336)
    ap.add_argument("--max-txt-len", type=int, default=512)
    ap.add_argument("--lm-backbone", default="gemma3_270M")
    ap.add_argument("--shuffle-total-streams-override", type=int, default=0)
    ap.add_argument(
        "--fast-text-only",
        action="store_true",
        help="Skip image decode/resize and audit the OV1.5 text/token shuffle stream.",
    )
    ap.add_argument(
        "--collect-meta",
        action="store_true",
        help="Per-config drift diagnostic: aggregate per-config share/mean and "
             "decompose early->late win_mean drift. Implies --fast-text-only.",
    )
    ap.add_argument(
        "--ov-min-shards",
        type=int,
        default=0,
        help="If >0, split OV1.5 grouped sources into finer per-config sources: "
             "each config with >= this many shards becomes its own fixed-weight "
             "StatefulRandomMix source; smaller configs merge into a per-group tail. "
             "Makes the config mixture stationary. 0 keeps the coarse 13 groups.",
    )
    ap.add_argument(
        "--expand-at-fill",
        action="store_true",
        help="Fix candidate: expand multi-QA conversations at buffer-fill time so "
             "every turn is an independent shuffle-buffer element (stationary turn "
             "distribution; no slow-draining pending slot).",
    )
    ap.add_argument(
        "--gcs-open",
        default="gsutil",
        choices=["gsutil", "gcsfs"],
        help="How WebDataset opens gs:// shards. gsutil avoids requiring gcsfs.",
    )
    ap.add_argument(
        "--tokenizer-path",
        default="",
        help="Optional local tokenizer model path for fast text audits.",
    )
    args = ap.parse_args()

    cfg = _build_config(args)
    if args.gcs_open == "gsutil":
        _register_gsutil_gopen()
    print("resolved_roots", len(cfg.dataset.root), flush=True)
    print("resolved_weight_sum", sum(float(x) for x in cfg.dataset.mix_weights), flush=True)
    for idx, (name, dtype, weight, root) in enumerate(zip(
        cfg.dataset.resolved_names,
        cfg.dataset.types,
        cfg.dataset.mix_weights,
        cfg.dataset.root,
    )):
        first = root[0] if isinstance(root, list) else root
        n_patterns = len(root) if isinstance(root, list) else 1
        print(f"source[{idx:02d}] name={name} type={dtype} weight={float(weight):.9g} patterns={n_patterns} first={first}", flush=True)

    if args.collect_meta:
        args.fast_text_only = True
        if not args.only_ov:
            raise ValueError("--collect-meta currently requires --only-ov")
        _run_meta_audit(args, cfg)
        return

    if args.fast_text_only:
        if not args.only_ov:
            raise ValueError("--fast-text-only currently requires --only-ov")
        fast_dataset = _create_fast_text_dataset(
            cfg,
            tokenizer_path=args.tokenizer_path or None,
        )
        if int(args.num_workers) > 0:
            from torch.utils.data import DataLoader

            loader = DataLoader(
                fast_dataset,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                prefetch_factor=int(args.prefetch_factor),
                timeout=int(args.dataloader_timeout),
                collate_fn=_collate_valid_counts,
                pin_memory=False,
            )
            iterator = iter(loader)
        else:
            loader = None
            iterator = iter(fast_dataset)
    else:
        loader, _ = input_pipeline.create_split(cfg, batch_size=int(args.batch_size))
        iterator = iter(loader)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    recent = []
    all_means = []
    t0 = time.time()
    rows_written = 0
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "elapsed_sec", "mean", "std", "p05", "p50", "p95", "min", "max", "window_mean", "window_slope"],
        )
        writer.writeheader()
        for step in range(1, int(args.max_steps) + 1):
            if (
                not args.fast_text_only
                and args.stateful
                and args.state_test_step > 0
                and step == int(args.state_test_step)
            ):
                stats = _state_roundtrip(cfg, int(args.batch_size), loader, iterator, step)
                print(f"state_roundtrip_ok step={step} next_mean={stats['mean']:.3f}", flush=True)
            if args.fast_text_only:
                if loader is None:
                    valid_counts = [next(iterator) for _ in range(int(args.batch_size))]
                else:
                    valid_counts = next(iterator)
                stats = _valid_count_stats(valid_counts)
            else:
                batch = next(iterator)
                stats = _valid_token_stats(batch)
            recent.append(stats["mean"])
            all_means.append(stats["mean"])
            if len(recent) > int(args.window):
                recent.pop(0)
            row = {
                "step": step,
                "elapsed_sec": time.time() - t0,
                **stats,
                "window_mean": float(statistics.mean(recent)),
                "window_slope": _window_slope(recent),
            }
            writer.writerow(row)
            rows_written += 1
            if step % int(args.log_every) == 0 or step == 1:
                f.flush()
                print(
                    "step={step} elapsed={elapsed:.1f}s mean={mean:.3f} p50={p50:.1f} "
                    "p95={p95:.1f} win_mean={wmean:.3f} win_slope={slope:.6f}".format(
                        step=step,
                        elapsed=row["elapsed_sec"],
                        mean=row["mean"],
                        p50=row["p50"],
                        p95=row["p95"],
                        wmean=row["window_mean"],
                        slope=row["window_slope"],
                    ),
                    flush=True,
                )
            if args.min_seconds > 0 and (time.time() - t0) >= args.min_seconds:
                print(f"min_seconds reached at step={step}", flush=True)
                break

    if all_means:
        total_slope = _window_slope(all_means)
        print(
            f"DONE rows={rows_written} csv={csv_path} global_mean={statistics.mean(all_means):.3f} "
            f"global_slope={total_slope:.8f} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
    else:
        print(f"DONE rows=0 csv={csv_path} elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
