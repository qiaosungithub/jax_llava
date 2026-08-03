"""Pull real training batches from the real CNS shards, locally.

This is step 2 of the google3 port: prove the whole data path -- CNS ->
wds_shim -> jax_llava's real preprocess/tokenizer/collate -> torch DataLoader
with absl_spawn workers -> prepare_batch_data -- produces genuine batches,
BEFORE paying for a Borg round trip. It deliberately reuses jax_llava's own
`create_split` rather than reimplementing a loader, so what it proves is what
training will do.

Run:
  blaze build //experimental/users/qiaos/jax_llava:g3_dataloader_probe
  TMPDIR=/tmp HF_HOME=/tmp/hf \\
    ./blaze-bin/experimental/users/qiaos/jax_llava/g3_dataloader_probe \\
      --num_workers=2 --batches=3
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as jax_llava_main  # noqa: F401,E402 -- installs the webdataset shim

import numpy as np  # noqa: E402
from absl import app, flags  # noqa: E402

_NUM_WORKERS = flags.DEFINE_integer('num_workers', 2, 'DataLoader workers.')
_BATCHES = flags.DEFINE_integer('batches', 3, 'Batches to pull.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 4, 'Per-process batch size.')
_MAX_TXT_LEN = flags.DEFINE_integer('max_txt_len', 160, 'Stage-1 token budget.')
_SHUFFLE = flags.DEFINE_integer(
    'shuffle_size', 64,
    'WebDataset shuffle buffer. Production is 10000; a probe that filled it '
    'would read ~10k images before the first batch.')
_ZONE = flags.DEFINE_string(
    'zone', 'us-east5', 'jax_llava zone. The locality guard checks the data '
    'roots against it, so this must match the cell holding the shards.')


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def describe(name, array):
    a = np.asarray(array)
    return (f"{name:24s} shape={str(a.shape):22s} dtype={str(a.dtype):9s} "
            f"min={a.min():>10.4f} max={a.max():>10.4f} mean={a.mean():>9.4f} "
            f"std={a.std():>8.4f}")


def build_config():
    from configs import default as default_config
    from utils import data_util

    config = default_config.get_config()
    with config.unlocked():
        config.local_debug = False
        config.zone = _ZONE.value
        config.dataset.items = [{'name': 'cc12m'}]
        config.dataset.root = []
        config.dataset.types = []
        config.dataset.mix_weights = [1.0]
        config.dataset.num_workers = _NUM_WORKERS.value
        config.dataset.prefetch_factor = 2
        config.dataset.max_txt_len = _MAX_TXT_LEN.value
        config.dataset.image_size = 336
        config.dataset.resize_mode = 'stretch'
        config.dataset.webdataset_shuffle_size = _SHUFFLE.value
        config.dataset.stateful_dataloader = True
        config.dataset.stateful_dataloader_strict = True
        config.dataset.stateful_snapshot_every_n_steps = 1
        config.dataset.dataloader_timeout = 1800
        config.model.lm_backbone_str = 'gemma3_1B'
        config.training.checkpoint_per_step = 800
    data_util.resolve_dataset_roots(config, _ZONE.value)
    return config


def main(argv):
    del argv
    import jax
    import input_pipeline

    section('0. environment')
    from utils import g3_env
    print(f"in_google3        : {g3_env.in_google3()}")
    print(f"borg cell         : {g3_env.borg_cell()}")
    print(f"jax backend       : {jax.default_backend()}")
    print(f"jax devices       : {jax.local_devices()}")
    print(f"process           : {jax.process_index()} / {jax.process_count()}")
    print(f"webdataset module : {sys.modules['webdataset'].__name__}")

    section('1. resolve dataset roots (the CNS replica, not the GCS mirror)')
    config = build_config()
    roots = list(config.dataset.root)
    print(f"roots  : {roots}")
    print(f"types  : {list(config.dataset.types)}")
    from google3.pyglib import gfile
    for root in roots:
        urls = input_pipeline._expand_gcs_glob_if_needed(root)
        urls = [urls] if isinstance(urls, str) else list(urls)
        print(f"expanded to {len(urls)} shards; first={urls[0]} last={urls[-1]}")
        missing = [u for u in urls if not gfile.Exists(u)]
        print(f"MISSING SHARDS: {len(missing)}"
              + (f" -> {missing[:5]}" if missing else "  (all present)"))
        assert not missing, "missing shards are a configuration error"

    section('2. the fail-closed locality guard, both branches')
    from input_pipeline import _assert_same_zone_roots
    _assert_same_zone_roots(roots, _ZONE.value, local_debug=False)
    print("PASS  real roots accepted for zone", _ZONE.value)
    for bad_roots, bad_zone, why in (
            (roots, 'us-central1', 'right data, wrong zone'),
            (['/cns/lu-d/home/qiaos/data/cc12m/{00000..00001}.tar'],
             _ZONE.value, 'right zone, wrong CNS cell'),
            (['gs://kmh-gcp-us-central1/data/cc12m/x.tar'], _ZONE.value,
             'a GCS bucket in another region'),
            (['/tmp/local/shard.tar'], _ZONE.value, 'an unrecognised scheme'),
    ):
        try:
            _assert_same_zone_roots(bad_roots, bad_zone, local_debug=False)
        except ValueError as exc:
            print(f"PASS  rejected ({why}): {str(exc)[:110]}")
        else:
            raise AssertionError(f"GUARD DID NOT FIRE for {why}: {bad_roots}")

    section(f'3. create_split (num_workers={_NUM_WORKERS.value})')
    t0 = time.time()
    loader, tokenizer = input_pipeline.create_split(
        config, _BATCH_SIZE.value, data_seed_offset=0)
    print(f"loader built in {time.time() - t0:.1f}s: {type(loader).__name__}")
    print(f"topology captured: {input_pipeline._PROCESS_TOPOLOGY}")

    section(f'4. pull {_BATCHES.value} real batches')
    it = iter(loader)
    for i in range(_BATCHES.value):
        t0 = time.time()
        raw = next(it)
        dt = time.time() - t0
        batch = input_pipeline.prepare_batch_data(raw, _BATCH_SIZE.value)
        print(f"\n--- batch {i} ({dt:.1f}s) ---")
        for key in ('pixel_values', 'input_ids', 'attention_mask', 'labels'):
            if key in batch:
                print(' ', describe(key, batch[key]))

        pv = np.asarray(batch['pixel_values'])
        # (local_devices, per_device, H, W, C) -> (batch, H, W, C)
        pv = pv.reshape(-1, *pv.shape[2:])
        # A degenerate batch (all-black, all-identical, or a constant fill) is
        # the failure this whole probe exists to catch, so check it explicitly
        # rather than eyeballing the numbers above.
        per_image_std = pv.reshape(pv.shape[0], -1).std(axis=1)
        print(f"  per-image pixel std : "
              f"{np.array2string(per_image_std, precision=4)}")
        assert per_image_std.min() > 0.05, (
            f"degenerate image in batch {i}: std={per_image_std}")
        flat = pv.reshape(pv.shape[0], -1)
        for a in range(flat.shape[0]):
            for b in range(a + 1, flat.shape[0]):
                assert not np.allclose(flat[a], flat[b]), (
                    f"images {a} and {b} are identical in batch {i}")

        # prepare_batch_data returns arrays shaped (local_devices, per_device,
        # ...), so flatten the leading device axis before treating rows as
        # sequences.
        ids = np.asarray(batch['input_ids']).reshape(-1, np.asarray(batch['input_ids']).shape[-1])
        mask = np.asarray(batch['attention_mask']).reshape(ids.shape)
        texts = [tokenizer.decode([int(t) for t in row.tolist() if int(t) != 0])
                 for row in ids]
        lengths = [int((mask[r] != 0).sum()) for r in range(ids.shape[0])]
        print(f"  valid token counts  : {lengths}")
        for r, text in enumerate(texts):
            print(f"  caption[{r}] ({len(text)} chars): {text[:220]!r}")
        assert min(lengths) > 4, f"suspiciously short sequences: {lengths}"

    section('5. loader state (exact resume)')
    state = loader.state_dict()
    print(f"state_dict keys: {sorted(state)[:8]} ... ({len(state)} total)")
    import pickle
    blob = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"pickled loader state: {len(blob)} bytes")

    print("\nPROBE PASSED", flush=True)


if __name__ == '__main__':
    try:
        from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing
    except ImportError:
        app.run(main)
    else:
        g3_multiprocessing.handle_main(main)
