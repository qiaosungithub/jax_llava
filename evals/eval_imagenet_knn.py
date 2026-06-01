"""ImageNet KNN accuracy evaluation using only the image encoder.

Two modes
---------
  partial  – 128 images/class from train (seed-controlled, ~128 k images),
             ~5 min wall-time; used for online eval during training.
  full     – all ~1.28 M train images; used for the final eval at the end.

The eval queries against the full val set (50 k images) unless the caller sets
``val_examples`` for a small remote data-path smoke test.

Distributed feature extraction
-------------------------------
All processes participate in feature extraction in parallel.

The ImageNet data is read directly from TFDS.  Each JAX process uses TFDS
input-context sharding, so hosts read disjoint TFRecord shards from the
zone-local bucket instead of first copying ImageNet into local tmpfs.  For
partial KNN eval, each process keeps its own quota per class; the combined
quota is exactly ``images_per_class`` when every process shard has enough
examples for every class.

After extracting local features with the same jit/HSDP sharding style as
training, a ``process_allgather`` gathers all shards.
Only ``process_index == 0`` then computes KNN and returns the accuracy.

Padding notes
-------------
Two padding levels exist:

1. **local-device padding** (within one batch): each TFDS batch of real
   size *B* is zero-padded at the *end* to the next multiple of ``LDC`` so the
   implied global batch is divisible by the full jit/HSDP mesh. The padded outputs
   are discarded by slicing ``feats[:B]`` after inference.

2. **Process padding** (for ``process_allgather``): TFDS shard sizes can differ
   across processes.  Before allgather each process shares its local count and
   zero-pads to the maximum local count.  After allgather the padding is
   stripped using the gathered per-process counts.

Data access
-----------
On remote TPUs call ``ensure_imagenet_available(zone)`` *once* before the
first eval call.  It returns the zone-local TFDS data directory and does not
download or copy ImageNet.  On a local debug machine pass ``local_debug=True``
to use the local TFDS path or ``TFDS_DATA_DIR``.

Public API
----------
    imagenet_data_dir = ensure_imagenet_available(zone, local_debug=False)
    acc = eval_imagenet_knn(state_params, model, config, imagenet_data_dir,
                            images_per_class=128, seed=42, k=20,
                            val_examples=None)
"""

from __future__ import annotations

import os
import time
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils as mu
import numpy as np

from utils.logging_util import log_for_0, log_for_all
from utils.pjit_util import MeshMode, prepare_pjit_funcs

# ── JAX process / device info ────────────────────────────────────────────────
LDC = jax.local_device_count()
PRI = jax.process_index()
PRC = jax.process_count()

# ── Zone-local TFDS paths ────────────────────────────────────────────────────
_KNN_TFDS_DATA_DIRS: dict[str, str] = {
    "us-central1": "gs://kmh-gcp-us-central1/tensorflow_datasets",
    "us-east5": "gs://kmh-gcp-us-east5/tensorflow_datasets",
    "asia-northeast1-b": "gs://kmh-gcp-asia-northeast1-b/tensorflow_datasets",
    "europe-west4": "gs://kmh-gcp/tensorflow_datasets",
}

_LOCAL_TFDS_DATA_DIR = "/kmh-nfs-ssd-us-mount/data/tensorflow_datasets"
_imagenet_data_dir_cache: Optional[str] = None


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _process_split_count(total: int, process_index: int, process_count: int) -> int:
    """Deterministically split a global example budget across JAX processes."""
    total = max(int(total), 0)
    process_index = int(process_index)
    process_count = max(int(process_count), 1)
    base = total // process_count
    extra = total % process_count
    return base + (1 if process_index < extra else 0)


def _tfds_modules():
    """Import TensorFlow/TFDS lazily so non-KNN runs do not need the dependency."""
    import tensorflow as tf  # noqa: PLC0415
    import tensorflow_datasets as tfds  # noqa: PLC0415

    return tf, tfds


def _normalize_zone(zone: str) -> str:
    """Map a full zone or region-ish string to the TFDS bucket key."""
    if zone in _KNN_TFDS_DATA_DIRS:
        return zone
    if zone.startswith("us-central1"):
        return "us-central1"
    if zone.startswith("us-east5"):
        return "us-east5"
    if zone.startswith("asia-northeast1"):
        return "asia-northeast1-b"
    if zone.startswith("europe-west4"):
        return "europe-west4"
    return zone


def ensure_imagenet_available(zone: str, local_debug: bool = False) -> str:
    """Return the TFDS ImageNet data_dir for this run; never copy ImageNet.

    The historical implementation copied a torchvision-style ImageNet tree
    into ``/mnt/zhhm`` before KNN eval.  ImageNet is now prepared as TFDS in the
    zone-local GCS buckets, so KNN eval should read TFDS directly.
    """
    global _imagenet_data_dir_cache

    if _imagenet_data_dir_cache is not None:
        log_for_0(f"[KNN] Using cached TFDS ImageNet data_dir={_imagenet_data_dir_cache}")
        return _imagenet_data_dir_cache

    if local_debug:
        data_dir = (
            os.environ.get("KNN_TFDS_DATA_DIR")
            or os.environ.get("TFDS_DATA_DIR")
            or _LOCAL_TFDS_DATA_DIR
        )
    else:
        data_dir = os.environ.get("KNN_TFDS_DATA_DIR")
        if not data_dir:
            zone_key = _normalize_zone(zone)
            if zone_key not in _KNN_TFDS_DATA_DIRS:
                raise ValueError(
                    f"[KNN] Unknown zone '{zone}'. Supported: {list(_KNN_TFDS_DATA_DIRS.keys())}"
                )
            data_dir = _KNN_TFDS_DATA_DIRS[zone_key]

    # Metadata-only builder creation is a cheap early sanity check and gives a
    # clearer failure than discovering a missing TFDS dependency inside eval.
    _, tfds = _tfds_modules()
    builder = tfds.builder("imagenet2012", data_dir=data_dir)
    if "train" not in builder.info.splits or "validation" not in builder.info.splits:
        raise ValueError(
            f"[KNN] imagenet2012 TFDS at {data_dir} is missing train/validation splits"
        )

    _imagenet_data_dir_cache = data_dir
    log_for_0(f"[KNN] Using TFDS ImageNet data_dir={data_dir}")
    return data_dir


# ── TFDS ImageNet input ───────────────────────────────────────────────────────

def _preprocess_imagenet_example(example, image_size: int, resize_mode: str, tf):
    """Match train/eval image geometry: direct stretch or letterbox padding."""
    image = tf.image.convert_image_dtype(example["image"], tf.float32)
    resize_mode = str(resize_mode or "letterbox").lower()

    if resize_mode in {"stretch", "direct_resize", "resize"}:
        image = tf.image.resize(
            image, [image_size, image_size], method="bicubic", antialias=True
        )
    elif resize_mode in {"letterbox", "letterbox_pad", "pad"}:
        shape = tf.shape(image)
        height = tf.cast(shape[0], tf.float32)
        width = tf.cast(shape[1], tf.float32)
        scale = tf.cast(image_size, tf.float32) / tf.maximum(height, width)
        new_height = tf.cast(tf.round(height * scale), tf.int32)
        new_width = tf.cast(tf.round(width * scale), tf.int32)
        image = tf.image.resize(
            image, [new_height, new_width], method="bicubic", antialias=True
        )
        pad_top = (image_size - new_height) // 2
        pad_bottom = image_size - new_height - pad_top
        pad_left = (image_size - new_width) // 2
        pad_right = image_size - new_width - pad_left
        image = tf.pad(
            image,
            [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            constant_values=127.0 / 255.0,
        )
    else:
        raise ValueError(f"Unknown resize_mode: {resize_mode}")

    image = tf.clip_by_value(image, 0.0, 1.0)
    image = image * 2.0 - 1.0
    label = tf.cast(example["label"], tf.int32)
    return image, label


class TFDSImageNetSplit:
    """Process-local TFDS ImageNet split used by KNN feature extraction."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        image_size: int,
        resize_mode: str,
        images_per_class: Optional[int],
        seed: int,
        process_index: int,
        total_processes: int,
        num_parallel_calls: int,
        max_examples: Optional[int] = None,
    ):
        self.tf, self.tfds = _tfds_modules()
        self.data_dir = data_dir
        self.split = split
        self.image_size = int(image_size)
        self.resize_mode = str(resize_mode or "letterbox")
        self.images_per_class = images_per_class
        self.seed = int(seed)
        self.process_index = int(process_index)
        self.total_processes = int(total_processes)
        self.num_parallel_calls = max(int(num_parallel_calls), 1)

        builder = self.tfds.builder("imagenet2012", data_dir=data_dir)
        split_info = builder.info.splits["validation" if split == "validation" else split]
        self.global_num_examples = int(split_info.num_examples)

        if split == "train" and images_per_class is not None:
            base = int(images_per_class) // self.total_processes
            extra = int(images_per_class) % self.total_processes
            self.local_quota_per_class = base + (1 if self.process_index < extra else 0)
            self.local_max_examples = None
            self.local_target_examples = 1000 * self.local_quota_per_class
            self.max_target_examples = 1000 * _ceil_div(int(images_per_class), self.total_processes)
        else:
            self.local_quota_per_class = None
            target_global_examples = self.global_num_examples
            if max_examples is not None:
                target_global_examples = min(max(int(max_examples), 0), self.global_num_examples)
            self.local_max_examples = _process_split_count(
                target_global_examples,
                self.process_index,
                self.total_processes,
            )
            self.local_target_examples = self.local_max_examples
            self.max_target_examples = _ceil_div(target_global_examples, self.total_processes)

        log_for_all(
            f"[KNN:{self.process_index}] TFDS {split}: data_dir={data_dir}, "
            f"resize_mode={self.resize_mode}, "
            f"global_examples={self.global_num_examples:,}, "
            f"local_quota_per_class={self.local_quota_per_class}, "
            f"local_target_examples={self.local_target_examples}, "
            f"max_target_examples={self.max_target_examples}"
        )

    def synchronized_num_steps(self, batch_size: int) -> int:
        """Return the same fixed step count on every host for HSDP eval."""
        return _ceil_div(self.max_target_examples, int(batch_size))

    def _make_dataset(self):
        input_context = self.tf.distribute.InputContext(
            num_input_pipelines=self.total_processes,
            input_pipeline_id=self.process_index,
            num_replicas_in_sync=self.total_processes,
        )
        read_config = self.tfds.ReadConfig(
            input_context=input_context,
            shuffle_seed=self.seed,
        )
        shuffle_files = self.split == "train" and self.local_quota_per_class is not None
        ds = self.tfds.load(
            "imagenet2012",
            split=self.split,
            data_dir=self.data_dir,
            download=False,
            shuffle_files=shuffle_files,
            read_config=read_config,
        )
        options = self.tf.data.Options()
        options.experimental_deterministic = True
        ds = ds.with_options(options)
        if shuffle_files:
            ds = ds.shuffle(
                8192,
                seed=self.seed + self.process_index,
                reshuffle_each_iteration=False,
            )
        ds = ds.map(
            lambda x: _preprocess_imagenet_example(
                x, self.image_size, self.resize_mode, self.tf
            ),
            num_parallel_calls=self.num_parallel_calls,
        )
        if self.local_max_examples is not None:
            # Apply the debug cap after TFDS process sharding so every host reads
            # a small disjoint slice from the zone-local TFDS files.
            ds = ds.take(self.local_max_examples)
        return ds

    def iter_batches(self, batch_size: int):
        ds = self._make_dataset()
        if self.local_quota_per_class is None:
            ds = ds.batch(batch_size, drop_remainder=False).prefetch(self.tf.data.AUTOTUNE)
            for images, labels in ds.as_numpy_iterator():
                yield images.astype(np.float32), labels.astype(np.int32)
            return

        counts = np.zeros((1000,), dtype=np.int32)
        images_batch: list[np.ndarray] = []
        labels_batch: list[int] = []
        finished_classes = 0

        for image, label in ds.as_numpy_iterator():
            label = int(label)
            if counts[label] >= self.local_quota_per_class:
                continue
            if counts[label] == self.local_quota_per_class - 1:
                finished_classes += 1
            counts[label] += 1
            images_batch.append(image.astype(np.float32))
            labels_batch.append(label)

            if len(images_batch) == batch_size:
                yield np.stack(images_batch, axis=0), np.asarray(labels_batch, dtype=np.int32)
                images_batch.clear()
                labels_batch.clear()

            if finished_classes == 1000:
                break

        if images_batch:
            yield np.stack(images_batch, axis=0), np.asarray(labels_batch, dtype=np.int32)

        if finished_classes < 1000:
            missing = int(1000 - finished_classes)
            log_for_all(
                f"[KNN:{self.process_index}] Warning: TFDS shard only satisfied "
                f"{finished_classes}/1000 class quotas; missing={missing}"
            )


# ── Feature extraction ────────────────────────────────────────────────────────

def _make_p_encode(model, state_params, config, global_batch_size):
    """Build a jit/HSDP encoder that mean-pools image tokens."""
    mesh, get_partition_spec, _, reduce_scatter, pjit_compile = prepare_pjit_funcs(
        getattr(config, "sharding", "hsdp")
    )
    params_spec = get_partition_spec(state_params, MeshMode.MODEL)
    image_spec = get_partition_spec(
        jax.ShapeDtypeStruct(
            (global_batch_size, config.dataset.image_size, config.dataset.image_size, 3),
            jnp.float32,
        ),
        MeshMode.DATA,
    )
    output_spec = get_partition_spec(
        jax.ShapeDtypeStruct((global_batch_size, 1), jnp.float32),
        MeshMode.DATA,
    )

    def _encode(params, images):
        tokens = model.apply(
            {"params": params},
            images,
            method=model.encode_image,
        )
        return tokens.mean(axis=1)

    p_encode = pjit_compile(
        _encode,
        in_shardings=(params_spec, image_spec),
        out_shardings=output_spec,
    )
    p_encode._mesh = mesh
    p_encode._image_spec = image_spec
    p_encode._reduce_scatter = reduce_scatter
    return p_encode


def _extract_features_local(
    p_encode,
    state_params,
    dataset: TFDSImageNetSplit,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features for this process's local shard.

    jit/HSDP padding strategy
    --------------------
    For a TFDS batch of *B* real images:

        pad  = (LDC - B % LDC) % LDC
        x_np shape after pad: (B + pad, H, W, 3)

    The local padded batch is converted into a globally sharded JAX array.
    After jit/HSDP inference, the host-local output is trimmed back to the real
    batch size.

    The zero-padded inputs produce garbage features that are simply discarded.

    Returns
    -------
    local_feats:  ``(N_local, D)`` float32
    local_labels: ``(N_local,)``  int32
    """
    all_feats:  list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    iterator = iter(dataset.iter_batches(batch_size))
    num_steps = dataset.synchronized_num_steps(batch_size)
    seen_examples = 0
    feature_dim = None
    exhausted_early = False
    t0 = time.time()

    log_for_all(
        f"[KNN:{PRI}] HSDP fixed-shape extraction: steps={num_steps}, "
        f"local_batch={batch_size}, target_local={dataset.local_target_examples}"
    )

    for batch_idx in range(num_steps):
        B = 0
        labels = np.zeros((0,), dtype=np.int32)
        x_np = np.zeros(
            (batch_size, dataset.image_size, dataset.image_size, 3),
            dtype=np.float32,
        )

        if seen_examples < dataset.local_target_examples:
            try:
                batch_images, batch_labels = next(iterator)
            except StopIteration:
                if not exhausted_early:
                    log_for_all(
                        f"[KNN:{PRI}] TFDS shard exhausted after "
                        f"{seen_examples}/{dataset.local_target_examples} examples; "
                        "padding remaining synchronized HSDP steps."
                    )
                    exhausted_early = True
            else:
                remaining = dataset.local_target_examples - seen_examples
                B = min(int(batch_images.shape[0]), int(remaining), int(batch_size))
                if B > 0:
                    x_np[:B] = batch_images[:B].astype(np.float32)
                    labels = batch_labels[:B].astype(np.int32)
                    seen_examples += B

        global_shape = (batch_size * PRC,) + x_np.shape[1:]
        global_images = jax.make_array_from_process_local_data(
            jax.sharding.NamedSharding(p_encode._mesh, p_encode._image_spec),
            x_np,
            global_shape,
        )
        feats = p_encode(state_params, global_images)
        feats = p_encode._reduce_scatter(feats, MeshMode.DATA)
        D = feats.shape[-1]
        feature_dim = D
        feats_np = np.array(jax.device_get(feats)).reshape(-1, D)[:B]

        if B > 0:
            all_feats.append(feats_np)
            all_labels.append(labels.astype(np.int32))

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_steps:
            done = sum(f.shape[0] for f in all_feats)
            log_for_all(
                f"  [KNN:{PRI}] [{batch_idx+1} batches] "
                f"{done:,} local examples  elapsed {time.time()-t0:.1f}s"
            )

    if not all_feats:
        local_feats = np.zeros((0, int(feature_dim or 1)), dtype=np.float32)
        local_labels = np.zeros((0,), dtype=np.int32)
        return local_feats, local_labels

    local_feats = np.concatenate(all_feats, axis=0).astype(np.float32)
    local_labels = np.concatenate(all_labels, axis=0).astype(np.int32)
    return local_feats, local_labels


# ── Distributed all-gather of features ───────────────────────────────────────

def _allgather_feats(
    local_feats:  np.ndarray,   # (N_local, D)
    local_labels: np.ndarray,   # (N_local,)
) -> tuple[np.ndarray, np.ndarray]:
    """All-gather variable-size (feature, label) pairs from all processes.

    Steps
     -----
    1. Gather local example counts and compute ``n_max``.
    2. Each process pads its arrays to ``n_max`` with zeros / -1 labels.
    3. ``process_allgather`` collects all shards:
           gathered_feats  shape: (PRC, n_max, D)
           gathered_labels shape: (PRC, n_max)
    4. For each process ``i``, trim to the gathered local count.
    5. Concatenate across processes (order: proc-0 shard, proc-1 shard, …).

    Returns
    -------
    all_feats:  ``(sum(N_local_i), D)`` float32
    all_labels: ``(sum(N_local_i),)``  int32
    """
    D = local_feats.shape[1]
    n_local = local_feats.shape[0]
    counts = np.asarray(
        jax.device_get(mu.process_allgather(jnp.asarray([n_local], dtype=jnp.int32)))
    ).reshape(-1)
    n_max = int(counts.max()) if counts.size else 0

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
    for proc_idx, n_i in enumerate(counts):
        n_i = int(n_i)
        result_feats.append(gathered_feats[proc_idx, :n_i])
        result_labels.append(gathered_labels[proc_idx, :n_i])

    all_feats  = np.concatenate(result_feats,  axis=0).astype(np.float32)
    all_labels = np.concatenate(result_labels, axis=0).astype(np.int32)
    return all_feats, all_labels


# ── KNN evaluation (process 0 only) ──────────────────────────────────────────

def _config_bool(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _pca_whiten_features(
    train_feats: np.ndarray,
    val_feats: np.ndarray,
    *,
    eps: float = 1e-5,
    keep_dim: Optional[int] = None,
    batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA whitening on train features and apply it to train/val features.

    The transform is fitted only on the KNN reference/train set to avoid using
    validation distribution information.  Both sets then get:

        (x - train_mean) @ eigvecs / sqrt(eigvals + eps)

    The downstream KNN code still L2-normalizes before cosine similarity.
    """
    train_feats = np.asarray(train_feats, dtype=np.float32)
    val_feats = np.asarray(val_feats, dtype=np.float32)
    n_train, dim = train_feats.shape
    if n_train < 2 or dim == 0:
        log_for_0("[KNN] Skip PCA whitening: need at least two train features.")
        return train_feats, val_feats

    keep_dim = dim if keep_dim is None or int(keep_dim) <= 0 else min(int(keep_dim), dim)
    batch_size = max(int(batch_size), 1)
    t0 = time.time()
    log_for_0(
        f"[KNN] Fitting PCA whitening on train feats: n={n_train:,}, dim={dim}, "
        f"keep_dim={keep_dim}, eps={eps:g}, cov_batch={batch_size}"
    )

    mean = train_feats.mean(axis=0, dtype=np.float64)
    cov = np.zeros((dim, dim), dtype=np.float64)
    for start in range(0, n_train, batch_size):
        chunk = train_feats[start : start + batch_size].astype(np.float64, copy=False)
        chunk = chunk - mean
        cov += chunk.T @ chunk
    cov /= max(n_train - 1, 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order][:keep_dim]
    eigvecs = eigvecs[:, order][:, :keep_dim]
    projection = eigvecs * (1.0 / np.sqrt(np.maximum(eigvals, 0.0) + float(eps)))[None, :]

    log_for_0(
        f"[KNN] PCA whitening eigvals: max={float(eigvals[0]):.6g}, "
        f"median={float(np.median(eigvals)):.6g}, min_kept={float(eigvals[-1]):.6g}"
    )

    def _transform(feats: np.ndarray, name: str) -> np.ndarray:
        can_write_in_place = keep_dim == feats.shape[1] and feats.dtype == np.float32
        if can_write_in_place:
            out = feats
        else:
            out = np.empty((feats.shape[0], keep_dim), dtype=np.float32)
        for start in range(0, feats.shape[0], batch_size):
            chunk = feats[start : start + batch_size].astype(np.float64, copy=False)
            chunk = chunk - mean
            out[start : start + batch_size] = (chunk @ projection).astype(np.float32)
        log_for_0(f"[KNN] PCA-whitened {name} feats: {out.shape}")
        return out

    train_w = _transform(train_feats, "train")
    val_w = _transform(val_feats, "val")
    log_for_0(f"[KNN] PCA whitening finished in {time.time()-t0:.1f}s")
    return train_w, val_w


def _maybe_apply_knn_feature_transform(
    train_feats: np.ndarray,
    val_feats: np.ndarray,
    config,
) -> tuple[np.ndarray, np.ndarray]:
    if not _config_bool(config.eval.get("knn_pca_whitening", True), default=True):
        log_for_0("[KNN] PCA whitening disabled; using raw features + L2 norm.")
        return train_feats, val_feats
    return _pca_whiten_features(
        train_feats,
        val_feats,
        eps=float(config.eval.get("knn_pca_whitening_eps", 1e-5)),
        keep_dim=config.eval.get("knn_pca_whitening_dim", 0),
        batch_size=int(config.eval.get("knn_pca_whitening_batch_size", 65536)),
    )


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
    state_params,
    model,                              # PaliGemmaEncDec instance
    config,                             # training config (needs dataset.image_size)
    imagenet_data_dir: str,             # TFDS data_dir containing imagenet2012
    images_per_class: Optional[int] = 128,
    seed: int = 42,
    k: int = 20,
    temperature: float = 0.07,
    batch_size: int = 256,
    num_workers: int = 4,
    val_examples: Optional[int] = None,
) -> float:
    """Evaluate KNN accuracy on (partial or full) ImageNet.

    All processes participate in feature extraction (each handles a TFDS
    input-context shard, so the work is distributed).  Features are gathered via
    ``process_allgather`` and KNN is computed **only on process 0**.

    Args:
        state_params:     sharded params from the jit/HSDP TrainState.
        model:            ``PaliGemmaEncDec`` instance (Flax module, not applied).
        config:           training config; uses dataset image size/resize mode.
        imagenet_data_dir: TFDS data_dir containing prepared ``imagenet2012``.
        images_per_class: train images per class (128 for partial eval;
                          ``None`` for full eval).
        seed:             RNG seed for reproducible partial sampling.
        k:                number of nearest neighbours.
        temperature:      softmax temperature for weighted voting.
        batch_size:       images per encode batch (per process).
        num_workers:      TFDS map parallelism per process.
        val_examples:     optional global cap for validation examples; useful
                          for remote path/debug checks that should not scan all
                          50 k validation images.

    Returns:
        KNN top-1 accuracy in percent (0–100) on process 0; ``0.0`` elsewhere.
    """
    image_size = int(config.dataset.image_size)
    resize_mode = getattr(config.dataset, "resize_mode", "letterbox")
    if images_per_class is not None and int(images_per_class) < PRC:
        log_for_0(
            f"[KNN] Raising images_per_class from {images_per_class} to "
            f"process_count={PRC} so every process has work."
        )
        images_per_class = PRC
    if val_examples is not None:
        val_examples = int(val_examples)
        if val_examples <= 0:
            val_examples = None
        elif val_examples < PRC:
            log_for_0(
                f"[KNN] Raising val_examples from {val_examples} to "
                f"process_count={PRC} so every process has work."
            )
            val_examples = PRC

    log_for_0(
        f"[KNN] Starting eval: images_per_class={images_per_class}, "
        f"seed={seed}, k={k}, T={temperature}, image_size={image_size}, "
        f"resize_mode={resize_mode}, processes={PRC}, LDC={LDC}, val_examples={val_examples}, "
        f"tfds_data_dir={imagenet_data_dir}"
    )

    local_batch = int(batch_size)
    if local_batch % LDC != 0:
        local_batch = ((local_batch + LDC - 1) // LDC) * LDC
    p_encode = _make_p_encode(model, state_params, config, local_batch * PRC)

    # ── Train features ────────────────────────────────────────────────────────
    train_ds = TFDSImageNetSplit(
        imagenet_data_dir,
        split="train",
        image_size=image_size,
        resize_mode=resize_mode,
        images_per_class=images_per_class,
        seed=seed,
        process_index=PRI,
        total_processes=PRC,
        num_parallel_calls=num_workers,
    )
    log_for_all(f"[KNN:{PRI}] Extracting train features from TFDS …")
    t0 = time.time()
    local_train_feats, local_train_labels = _extract_features_local(
        p_encode, state_params, train_ds, local_batch
    )
    log_for_all(
        f"[KNN:{PRI}] Train local feats {local_train_feats.shape} "
        f"in {time.time()-t0:.1f}s"
    )

    log_for_0("[KNN] All-gathering train features …")
    train_feats, train_labels = _allgather_feats(
        local_train_feats, local_train_labels
    )
    log_for_0(f"[KNN] Train feats gathered: {train_feats.shape}")
    mu.sync_global_devices("knn_train_features_gathered")

    # ── Val features ──────────────────────────────────────────────────────────
    val_ds = TFDSImageNetSplit(
        imagenet_data_dir,
        split="validation",
        image_size=image_size,
        resize_mode=resize_mode,
        images_per_class=None,
        seed=seed,
        process_index=PRI,
        total_processes=PRC,
        num_parallel_calls=num_workers,
        max_examples=val_examples,
    )
    log_for_all(f"[KNN:{PRI}] Extracting val features from TFDS …")
    t0 = time.time()
    local_val_feats, local_val_labels = _extract_features_local(
        p_encode, state_params, val_ds, local_batch
    )
    log_for_all(
        f"[KNN:{PRI}] Val local feats {local_val_feats.shape} "
        f"in {time.time()-t0:.1f}s"
    )

    log_for_0("[KNN] All-gathering val features …")
    val_feats, val_labels = _allgather_feats(
        local_val_feats, local_val_labels
    )
    log_for_0(f"[KNN] Val feats gathered: {val_feats.shape}")
    mu.sync_global_devices("knn_val_features_gathered")

    train_feats, val_feats = _maybe_apply_knn_feature_transform(
        train_feats,
        val_feats,
        config,
    )
    mu.sync_global_devices("knn_feature_transform_done")

    # Every host runs the same JAX KNN program.  Running JAX only on process 0
    # can desynchronize multi-controller TPU programs after HSDP feature
    # extraction, even if the input arrays are already fully gathered.
    log_for_0(f"[KNN] Running replicated KNN-{k} on every process …")
    t0 = time.time()
    acc = _knn_accuracy_jax(
        train_feats, train_labels,
        val_feats,   val_labels,
        k=k, temperature=temperature,
    )
    accs = np.asarray(
        jax.device_get(mu.process_allgather(jnp.asarray([acc], dtype=jnp.float32)))
    ).reshape(-1)
    if PRI == 0 and accs.size and float(accs.max() - accs.min()) > 1e-3:
        log_for_0(f"[KNN] Warning: replicated KNN accuracies differ across hosts: {accs}")
    acc = float(accs[0]) if accs.size else float(acc)
    log_for_0(f"[KNN] Done in {time.time()-t0:.1f}s  →  acc = {acc:.2f}%")
    return acc
