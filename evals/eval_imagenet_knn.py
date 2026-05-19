"""ImageNet KNN accuracy evaluation using only the image encoder.

Two modes
---------
  partial  – 128 images/class from train (seed-controlled, ~128 k images),
             ~5 min wall-time; used for online eval during training.
  full     – all ~1.28 M train images; used for the final eval at the end.

The eval always queries against the full val set (50 k images).

Distributed feature extraction
-------------------------------
All processes participate in feature extraction in parallel.

``ImageNetSubset`` first builds the **complete** sample list identically on
every process (identical class-sort → same ``random.sample`` with the same
seed).  Each process then takes a strided shard:

    local_samples = all_samples[process_index :: total_processes]

This guarantees that the 128 selected images/class are **bit-identical**
regardless of the number of processes.

After extracting local features with a *local* ``pmap`` (no cross-process
collectives inside), a ``process_allgather`` gathers all shards.
Only ``process_index == 0`` then computes KNN and returns the accuracy.

Padding notes
-------------
Two padding levels exist:

1. **LDC padding** (within one batch): each DataLoader batch of real size *B*
   is zero-padded at the *end* to the next multiple of ``LDC`` so that it can
   be reshaped to ``(LDC, B_per_device, …)`` for ``pmap``.  The padded outputs
   are discarded by slicing ``feats[:B]`` after ``pmap``.

2. **Process padding** (for ``process_allgather``): shard sizes differ by at
   most 1 (strided split).  Before allgather each process zero-pads its local
   feature array to ``ceil(n_total / PRC)``.  After allgather the padding is
   stripped using the exact shard-size formula
   ``n_i = n_total // PRC + (1 if i < n_total % PRC else 0)``.

Data access
-----------
On remote TPUs call ``ensure_imagenet_available(zone)`` *once* before the
first eval call.  On a local debug machine pass ``local_debug=True`` to use
the NFS path directly.

Public API
----------
    imagenet_root = ensure_imagenet_available(zone, local_debug=False)
    acc = eval_imagenet_knn(state_params, model, config, imagenet_root,
                            images_per_class=128, seed=42, k=20)
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils as mu
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from utils.logging_util import log_for_0, log_for_all

# ── JAX process / device info ────────────────────────────────────────────────
LDC = jax.local_device_count()
PRI = jax.process_index()
PRC = jax.process_count()

# ── GCS paths per zone (mirrors jax_dev_v6_base WARMUP_ARGS) ────────────────
_KNN_WARMUP_ARGS: dict[str, tuple] = {
    "us-central1": (
        "gs://kmh-gcp-us-central1/data/imagenet/imagenet",
        "/mnt/zhhm/zhh/imagenet", "JPEG", 1_281_167,
        "/mnt/zhhm/zhh/imagenet/imagenet/train",
    ),
    "us-east5": (
        "gs://kmh-gcp-us-east5/data/imagenet/imagenet",
        "/mnt/zhhm/zhh/imagenet", "JPEG", 1_281_167,
        "/mnt/zhhm/zhh/imagenet/imagenet/train",
    ),
    "asia-northeast1-b": (
        "gs://kmh-gcp-asia-northeast1-b/data/imagenet/imagenet",
        "/mnt/zhhm/zhh/imagenet", "JPEG", 1_281_167,
        "/mnt/zhhm/zhh/imagenet/imagenet/train",
    ),
    "europe-west4": (
        "gs://kmh-gcp/data/imagenet/imagenet",
        "/mnt/zhhm/zhh/imagenet", "JPEG", 1_281_167,
        "/mnt/zhhm/zhh/imagenet/imagenet/train",
    ),
}

_NFS_IMAGENET_ROOT = "/kmh-nfs-ssd-us-mount/data/imagenet"
_imagenet_root_cache: Optional[str] = None


def ensure_imagenet_available(zone: str, local_debug: bool = False) -> str:
    """Download ImageNet from GCS to tmpfs on first call; return the root path.

    The returned path contains ``train/`` and ``val/`` sub-directories with
    the standard WordNet-id folder structure.  Subsequent calls return the
    cached path immediately (no re-download).
    """
    global _imagenet_root_cache

    if _imagenet_root_cache is not None:
        train_dir = os.path.join(_imagenet_root_cache, "train")
        if os.path.isdir(train_dir) and len(os.listdir(train_dir)) == 1000:
            log_for_0(f"[KNN] Using cached imagenet at {_imagenet_root_cache}")
            return _imagenet_root_cache

    if local_debug:
        log_for_0(f"[KNN] local_debug=True – using NFS imagenet at {_NFS_IMAGENET_ROOT}")
        _imagenet_root_cache = _NFS_IMAGENET_ROOT
        return _NFS_IMAGENET_ROOT

    if zone not in _KNN_WARMUP_ARGS:
        raise ValueError(
            f"[KNN] Unknown zone '{zone}'. Supported: {list(_KNN_WARMUP_ARGS.keys())}"
        )

    gs_root, shm_dest, suffix, num, new_root = _KNN_WARMUP_ARGS[zone]

    if os.path.isdir(new_root) and len(os.listdir(new_root)) == 1000:
        log_for_0(f"[KNN] ImageNet already present at {new_root}, skipping download.")
        imagenet_root = str(Path(new_root).parent)
        _imagenet_root_cache = imagenet_root
        return imagenet_root

    log_for_0(f"[KNN] Downloading ImageNet from {gs_root} → {shm_dest} …")
    # Lazy import: warmup_util does a process_allgather at import time.
    # Importing it here ensures all processes are active when it fires.
    from utils.warmup_util import run_warmup_main  # noqa: PLC0415
    run_warmup_main(gs_root, shm_dest, suffix, num, new_root)

    imagenet_root = str(Path(new_root).parent)   # has train/ and val/
    _imagenet_root_cache = imagenet_root
    log_for_0(f"[KNN] ImageNet ready at {imagenet_root}")
    return imagenet_root


# ── ImageNet subset dataset ───────────────────────────────────────────────────

class ImageNetSubset(Dataset):
    """ImageNet split with optional per-class image-count cap.

    The full sample list is built **identically on every process** (same seed,
    same alphabetical class order, same ``random.sample`` call).  Each process
    then takes a strided shard so that the selected images are consistent
    regardless of process count:

        local_samples = all_samples[process_index :: total_processes]

    Attributes:
        n_total: total sample count across ALL processes (used for allgather).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        images_per_class: Optional[int] = 128,
        seed: int = 42,
        transform=None,
        process_index: int = 0,
        total_processes: int = 1,
    ):
        self.transform = transform
        split_dir = Path(root) / split

        # Sort classes for bit-identical ordering across machines
        classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        # Build the complete list with a shared seed – SAME on all processes
        rng = random.Random(seed)
        all_samples: list[tuple[str, int]] = []

        for cls_name in classes:
            cls_dir = split_dir / cls_name
            imgs = sorted(
                f for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpeg", ".jpg", ".png"))
            )
            if images_per_class is not None and len(imgs) > images_per_class:
                imgs = rng.sample(imgs, images_per_class)
            label = self.class_to_idx[cls_name]
            for img_name in imgs:
                all_samples.append((str(cls_dir / img_name), label))

        # Strided shard: process i gets indices i, i+PRC, i+2*PRC, …
        self.n_total: int = len(all_samples)
        self.samples: list[tuple[str, int]] = (
            all_samples[process_index::total_processes]
        )

        log_for_0(
            f"[KNN Dataset] {split}: images_per_class={images_per_class}, "
            f"seed={seed} → total={self.n_total:,}  "
            f"(process {process_index}/{total_processes}: {len(self.samples):,} local)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _knn_transform(image_size: int):
    """Val-time transform: resize → center-crop → tensor → normalise to [-1, 1]."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ── Feature extraction ────────────────────────────────────────────────────────

def _make_p_encode(model):
    """Build a local pmap'd encoder that mean-pools over learnable tokens.

    Purely local – no cross-process collectives → safe to call independently
    on each process.  Uses ``devices=jax.local_devices()`` to be explicit.
    """
    def _encode(params, images):
        # params: unreplicated on this device (shape without leading LDC dim)
        # images: (B_local, H, W, 3) float32 in [-1, 1]
        tokens = model.apply(
            {"params": params},
            images,
            method=model.encode_image,
        )  # (B_local, K, D)
        return tokens.mean(axis=1)  # (B_local, D)

    return jax.pmap(_encode, devices=jax.local_devices())


def _extract_features_local(
    p_encode,
    state_params,          # replicated params (LDC, ...)
    dataset: ImageNetSubset,
    image_size: int,
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features for this process's local shard.

    LDC padding strategy
    --------------------
    For a DataLoader batch of *B* real images:

        pad  = (LDC - B % LDC) % LDC
        x_np shape after pad: (B + pad, H, W, 3)   ← zeros at the END

    Reshape to ``(LDC, (B+pad)//LDC, H, W, 3)`` and feed to pmap.
    After pmap, flatten to ``(B + pad, D)`` and keep only ``[:B]``:

        feats_np = np.array(feats).reshape(-1, D)[:B]   # (B, D)

    The zero-padded inputs produce garbage features that are simply discarded.

    Returns
    -------
    local_feats:  ``(N_local, D)`` float32
    local_labels: ``(N_local,)``  int32
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    all_feats:  list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    t0 = time.time()

    for batch_idx, (imgs, labels) in enumerate(loader):
        B = imgs.shape[0]

        # (B, 3, H, W) → (B, H, W, 3) float32
        x_np = imgs.permute(0, 2, 3, 1).numpy().astype(np.float32)

        # ── LDC padding: zero-fill at the END ────────────────────────────────
        pad = (LDC - B % LDC) % LDC
        if pad > 0:
            x_np = np.concatenate(
                [x_np, np.zeros((pad,) + x_np.shape[1:], dtype=np.float32)],
                axis=0,
            )   # shape: (B + pad, H, W, 3)

        # Reshape to (LDC, B_per_device, H, W, 3) for pmap
        x_jax = jnp.array(x_np.reshape(LDC, -1, image_size, image_size, 3))
        feats = p_encode(state_params, x_jax)   # (LDC, B_per_device, D)
        D = feats.shape[-1]

        # Flatten: (LDC * B_per_device, D) = (B + pad, D)
        # Trim:    [:B]  → (B, D)  – discards the zero-padded tail exactly
        feats_np = np.array(feats).reshape(-1, D)[:B]

        all_feats.append(feats_np)
        all_labels.append(labels.numpy().astype(np.int32))

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            done = sum(f.shape[0] for f in all_feats)
            log_for_all(
                f"  [KNN:{PRI}] [{batch_idx+1}/{len(loader)}] "
                f"{done:,}/{len(dataset):,}  elapsed {time.time()-t0:.1f}s"
            )

    local_feats  = np.concatenate(all_feats,  axis=0).astype(np.float32)
    local_labels = np.concatenate(all_labels, axis=0).astype(np.int32)
    return local_feats, local_labels


# ── Distributed all-gather of features ───────────────────────────────────────

def _shard_size(n_total: int, proc_idx: int, n_procs: int) -> int:
    """Exact local-shard size for a strided split.

    For a strided split ``all[proc_idx::n_procs]`` on a list of length
    ``n_total``, the number of elements owned by ``proc_idx`` is:

        n_total // n_procs  +  (1 if proc_idx < n_total % n_procs else 0)
    """
    return n_total // n_procs + (1 if proc_idx < n_total % n_procs else 0)


def _allgather_feats(
    local_feats:  np.ndarray,   # (N_local, D)
    local_labels: np.ndarray,   # (N_local,)
    n_total:      int,          # total samples across ALL processes
) -> tuple[np.ndarray, np.ndarray]:
    """All-gather (feature, label) pairs from all processes.

    Steps
    -----
    1. Compute the maximum shard size: ``n_max = ceil(n_total / PRC)``.
    2. Each process pads its arrays to ``n_max`` with zeros / -1 labels.
    3. ``process_allgather`` collects all shards:
           gathered_feats  shape: (PRC, n_max, D)
           gathered_labels shape: (PRC, n_max)
    4. For each process ``i``, trim to the exact shard size
       ``n_i = n_total // PRC + (1 if i < n_total % PRC else 0)`` and
       discard the padding.
    5. Concatenate across processes (order: proc-0 shard, proc-1 shard, …).

    Returns
    -------
    all_feats:  ``(n_total, D)`` float32
    all_labels: ``(n_total,)``  int32
    """
    D = local_feats.shape[1]
    n_max   = (n_total + PRC - 1) // PRC    # upper bound on any shard size
    n_local = local_feats.shape[0]

    # ── Pad to n_max (zeros / sentinel label -1 at the end) ──────────────────
    pad = n_max - n_local
    assert pad >= 0, f"Local shard ({n_local}) larger than n_max ({n_max})?"
    if pad > 0:
        local_feats  = np.concatenate(
            [local_feats, np.zeros((pad, D), dtype=np.float32)], axis=0
        )
        local_labels = np.concatenate(
            [local_labels, -np.ones(pad, dtype=np.int32)], axis=0
        )
    # shapes now: (n_max, D) and (n_max,)

    # ── All-gather across processes ───────────────────────────────────────────
    gathered_feats  = jax.device_get(
        mu.process_allgather(jnp.array(local_feats))
    )   # (PRC, n_max, D)
    gathered_labels = jax.device_get(
        mu.process_allgather(jnp.array(local_labels))
    )   # (PRC, n_max)

    # ── Trim padding per process and concatenate ──────────────────────────────
    result_feats:  list[np.ndarray] = []
    result_labels: list[np.ndarray] = []
    for proc_idx in range(PRC):
        n_i = _shard_size(n_total, proc_idx, PRC)
        result_feats.append(gathered_feats[proc_idx, :n_i])
        result_labels.append(gathered_labels[proc_idx, :n_i])

    all_feats  = np.concatenate(result_feats,  axis=0).astype(np.float32)
    all_labels = np.concatenate(result_labels, axis=0).astype(np.int32)

    # Sanity check
    assert all_feats.shape[0] == n_total, (
        f"Gathered {all_feats.shape[0]} features but expected {n_total}"
    )
    return all_feats, all_labels


# ── KNN evaluation (process 0 only) ──────────────────────────────────────────

def _knn_accuracy_jax(
    train_feats:  np.ndarray,   # (N_train, D)
    train_labels: np.ndarray,   # (N_train,)
    val_feats:    np.ndarray,   # (N_val, D)
    val_labels:   np.ndarray,   # (N_val,)
    k:            int   = 20,
    temperature:  float = 0.07,
    num_classes:  int   = 1000,
    query_chunk:  int   = 2000,
) -> float:
    """Weighted cosine-similarity KNN (DINO/MAE convention).

    Each of the *k* nearest neighbours votes with weight
    ``exp(cosine_sim(q, n_i) / temperature)``.  Returns accuracy in [0, 100].
    """
    def _l2norm(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    db_feats  = jnp.array(_l2norm(train_feats))    # (N_train, D)
    db_labels = jnp.array(train_labels)             # (N_train,)
    val_n     = _l2norm(val_feats)                  # (N_val, D)

    @jax.jit
    def _knn_chunk(q_feats):
        sims         = jnp.einsum("qd,nd->qn", q_feats, db_feats)   # (Q, N_train)
        topk_sims, topk_idx = jax.lax.top_k(sims, k)                # (Q, k)
        topk_labels  = db_labels[topk_idx]                           # (Q, k)
        weights      = jnp.exp(topk_sims / temperature)              # (Q, k)
        one_hot      = (
            topk_labels[:, :, None] == jnp.arange(num_classes)[None, None, :]
        )                                                             # (Q, k, C)
        scores       = (weights[:, :, None] * one_hot).sum(axis=1)   # (Q, C)
        return jnp.argmax(scores, axis=1)                            # (Q,)

    n_val   = val_n.shape[0]
    correct = 0
    t0      = time.time()
    for start in range(0, n_val, query_chunk):
        end    = min(start + query_chunk, n_val)
        preds  = np.array(_knn_chunk(jnp.array(val_n[start:end])))
        correct += int((preds == val_labels[start:end]).sum())
        if (start // query_chunk + 1) % 10 == 0 or end == n_val:
            log_for_0(
                f"  [KNN query] [{end}/{n_val}]  "
                f"running acc {correct / end * 100:.2f}%  "
                f"elapsed {time.time()-t0:.1f}s"
            )

    return correct / n_val * 100.0


# ── Public eval function ──────────────────────────────────────────────────────

def eval_imagenet_knn(
    state_params,                       # replicated params (from jax_utils.replicate)
    model,                              # PaliGemmaEncDec instance
    config,                             # training config (needs dataset.image_size)
    imagenet_root: str,                 # path with train/ and val/ sub-dirs
    images_per_class: Optional[int] = 128,
    seed: int = 42,
    k: int = 20,
    temperature: float = 0.07,
    batch_size: int = 256,
    num_workers: int = 4,
) -> float:
    """Evaluate KNN accuracy on (partial or full) ImageNet.

    All processes participate in feature extraction (each handles a strided
    shard, so the work is evenly distributed).  Features are gathered via
    ``process_allgather`` and KNN is computed **only on process 0**.

    Args:
        state_params:     replicated params from ``jax_utils.replicate(state).params``.
        model:            ``PaliGemmaEncDec`` instance (Flax module, not applied).
        config:           training config; uses ``config.dataset.image_size``.
        imagenet_root:    path whose children are ``train/`` and ``val/``.
        images_per_class: train images per class (128 for partial eval;
                          ``None`` for full eval).
        seed:             RNG seed for reproducible partial sampling.
        k:                number of nearest neighbours.
        temperature:      softmax temperature for weighted voting.
        batch_size:       images per encode batch (per process).
        num_workers:      DataLoader workers per process.

    Returns:
        KNN top-1 accuracy in percent (0–100) on process 0; ``0.0`` elsewhere.
    """
    image_size = config.dataset.image_size
    transform  = _knn_transform(image_size)

    log_for_0(
        f"[KNN] Starting eval: images_per_class={images_per_class}, "
        f"seed={seed}, k={k}, T={temperature}, image_size={image_size}, "
        f"processes={PRC}, LDC={LDC}"
    )

    # Local pmap encoder – no cross-process ops inside, purely local
    p_encode = _make_p_encode(model)

    # ── Train features ────────────────────────────────────────────────────────
    train_ds = ImageNetSubset(
        imagenet_root, split="train",
        images_per_class=images_per_class, seed=seed, transform=transform,
        process_index=PRI, total_processes=PRC,
    )
    log_for_all(f"[KNN:{PRI}] Extracting train features ({len(train_ds):,} local samples) …")
    t0 = time.time()
    local_train_feats, local_train_labels = _extract_features_local(
        p_encode, state_params, train_ds, image_size, batch_size, num_workers
    )
    log_for_all(
        f"[KNN:{PRI}] Train local feats {local_train_feats.shape} "
        f"in {time.time()-t0:.1f}s"
    )

    log_for_0("[KNN] All-gathering train features …")
    train_feats, train_labels = _allgather_feats(
        local_train_feats, local_train_labels, train_ds.n_total
    )
    log_for_0(f"[KNN] Train feats gathered: {train_feats.shape}")

    # ── Val features ──────────────────────────────────────────────────────────
    val_ds = ImageNetSubset(
        imagenet_root, split="val",
        images_per_class=None,   # always use all 50 k val images
        seed=seed, transform=transform,
        process_index=PRI, total_processes=PRC,
    )
    log_for_all(f"[KNN:{PRI}] Extracting val features ({len(val_ds):,} local samples) …")
    t0 = time.time()
    local_val_feats, local_val_labels = _extract_features_local(
        p_encode, state_params, val_ds, image_size, batch_size, num_workers
    )
    log_for_all(
        f"[KNN:{PRI}] Val local feats {local_val_feats.shape} "
        f"in {time.time()-t0:.1f}s"
    )

    log_for_0("[KNN] All-gathering val features …")
    val_feats, val_labels = _allgather_feats(
        local_val_feats, local_val_labels, val_ds.n_total
    )
    log_for_0(f"[KNN] Val feats gathered: {val_feats.shape}")

    # ── KNN (process 0 only) ──────────────────────────────────────────────────
    # Other processes have already done their share of the work (feature
    # extraction + allgather) and can return immediately.
    if PRI != 0:
        return 0.0

    log_for_0(f"[KNN] Running KNN-{k} on process 0 …")
    t0 = time.time()
    acc = _knn_accuracy_jax(
        train_feats, train_labels,
        val_feats,   val_labels,
        k=k, temperature=temperature,
    )
    log_for_0(f"[KNN] Done in {time.time()-t0:.1f}s  →  acc = {acc:.2f}%")
    return acc
