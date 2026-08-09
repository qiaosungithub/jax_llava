"""Load a config through the real loader and print what training would see.

Cheap dry run of everything that happens before the first step: config merge,
curriculum stage expansion, dataset-root resolution, and the locality guard.
Catches a bad config in seconds instead of after packaging + scheduling.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as jax_llava_main  # noqa: F401,E402 -- installs the webdataset shim

from absl import app, flags  # noqa: E402

_CONFIG = flags.DEFINE_string('config_mode', 'g3_smoke', 'configs/<mode>_config.yml')
_ZONE = flags.DEFINE_string('zone', 'us-east5', 'jax_llava zone.')


def _exists(path) -> bool:
    """True if `path` is really there, asked of whatever filesystem owns it."""
    from utils import g3_env
    if str(path).startswith(('/cns/', '/bigstore/')):
        return g3_env.cns_dir_exists(str(path))
    return os.path.exists(str(path))


def _runtime_target(url):
    """The path this shard will REALLY be opened from, or None if unreadable.

    A gs:// root is not itself a failure on Borg: input_pipeline's webdataset
    opener rewrites to the co-located replica at the last hop, deliberately,
    because shard roots are assembled in several modules and not all of them
    pass through data_util's resolution. So the honest check is the one the
    opener performs -- rewrite, then confirm the target exists -- and NOT
    "does this string start with gs://", which would condemn a path that works.
    """
    url = str(url)
    if url.startswith('gs://'):
        from utils.data_util import _rewrite_bucket_to_cns
        url = _rewrite_bucket_to_cns(url) or ''
        if not url:
            return None
    return url if _exists(url) else None


def _probe_eval_roots(config, tasks):
    """Every `*_root` the eval tasks will open, checked before a launch.

    The final eval runs once, after 57 hours. A root that still says gs:// or
    points at nothing is invisible until then, which is the most expensive
    moment to discover it.

    Only roots belonging to an ENABLED task are fatal. `default.py` declares
    roots for benchmarks no config runs (scienceqa_img, vizwiz), whose data was
    never copied because nothing asks for it -- failing on those would be the
    probe inventing work rather than reporting risk.
    """
    print("\n--- eval roots ---")
    # A root belongs to a task if the task name appears in the key, modulo the
    # spelling drift between config keys and task names.
    aliases = {'seed_bench': 'seed_bench', 'knn_full': 'imagenet',
               'mmvp': 'pixelbench', 'vstar': 'pixelbench',
               'ocrbench': 'pixelbench', 'countbenchqa': 'pixelbench'}
    wanted = {aliases.get(t, t) for t in tasks}
    n_bad = 0
    for key in sorted(k for k in config.eval.keys() if k.endswith('_root')):
        value = config.eval[key]
        if not isinstance(value, str) or not value:
            continue
        stem = key[:-len('_root')].replace('_image', '').replace('_test', '')
        used = any(stem.startswith(w) or w.startswith(stem) for w in wanted)
        if not used:
            print(f"  {'(unused)':<16} {key:<28} {value}")
            continue
        if value.startswith(('http://', 'https://')):
            status, n_bad = 'HTTP(no egress)', n_bad + 1
        elif value.startswith('gs://'):
            status, n_bad = 'GS://', n_bad + 1
        elif value.startswith('/kmh-nfs'):
            status, n_bad = 'NFS(absent)', n_bad + 1
        elif value.startswith('/cns/'):
            probe = value.split('*', 1)[0].split('{', 1)[0].rstrip('/')
            status = 'OK' if _exists(probe) else 'MISSING'
            n_bad += status == 'MISSING'
        else:
            status = 'unused' if value == 'unused_for_image_records_wds' else '?'
        print(f"  {status:<16} {key:<28} {value}")
    return n_bad


def _probe_knn(config):
    """The TFDS ImageNet data_dir, resolved the way the eval will resolve it."""
    print("\n--- knn ---")
    from evals.eval_imagenet_knn import ensure_imagenet_available
    try:
        data_dir = ensure_imagenet_available('unused-on-borg',
                                             local_debug=config.local_debug)
        print(f"  OK               imagenet TFDS data_dir: {data_dir}")
        return 0
    except Exception as e:  # noqa: BLE001 -- reporting, not handling
        print(f"  FAIL             {type(e).__name__}: {e}")
        return 1


def main(argv):
    del argv
    from configs import load_config
    from utils import data_util, g3_env
    import input_pipeline
    import train

    print(f"in_google3={g3_env.in_google3()} cell={g3_env.borg_cell()} "
          f"zone_from_env={g3_env.infer_zone_from_environment()}")

    config = load_config.get_config(_CONFIG.value)
    with config.unlocked():
        config.zone = _ZONE.value
    print(f"\ncurriculum        : {config.training.get('curriculum')}")
    print(f"stage1/stage2/tot : {config.training.stage1_steps} / "
          f"{config.training.stage2_steps} / {config.training.num_steps}")
    print(f"checkpoint_per_step: {config.training.checkpoint_per_step}")
    print(f"sharding          : {config.sharding}")
    print(f"lm_backbone       : {config.model.lm_backbone_str} "
          f"({config.model.lm_checkpoint_variant})")

    resolved_stage = None
    for stage_key in ('stage1', 'stage2'):
        stage = train._build_curriculum_stage_config(
            config, stage_key,
            stage_start_step=0 if stage_key == 'stage1' else config.training.stage1_steps,
            stage_end_step=(config.training.stage1_steps if stage_key == 'stage1'
                            else config.training.num_steps),
            total_steps=config.training.num_steps)
        data_util.resolve_dataset_roots(stage, _ZONE.value)
        print(f"\n--- {stage_key} ---")
        print(f"  batch_size   : {stage.training.batch_size}")
        print(f"  num_workers  : {stage.dataset.num_workers}")
        print(f"  max_txt_len  : {stage.dataset.max_txt_len}")
        print(f"  freeze_lm    : {stage.training.get('freeze_lm')}")
        print(f"  types        : {list(stage.dataset.types)}")
        print(f"  mix_weights  : {list(stage.dataset.get('mix_weights', []))}")
        input_pipeline._assert_same_zone_roots(
            stage.dataset.root, _ZONE.value, local_debug=False)
        print("  locality guard: PASS")

        # EVERY root, expanded and existence-checked -- not just root[0].
        # A 12-source mix resolves twelve different ways, and the previous
        # version proved only the first one; the eleven behind it then failed
        # one remote launch at a time. Each remote attempt costs ~10 minutes
        # and this loop costs seconds.
        n_bad = 0
        for name, root in zip(stage.dataset.get('resolved_names', []),
                              list(stage.dataset.root)):
            urls = input_pipeline._expand_gcs_glob_if_needed(root)
            urls = [urls] if isinstance(urls, str) else list(urls)
            # Sample the ends, not the middle: a truncated copy loses its tail,
            # and a wrong root loses its head.
            sample = urls[:2] + urls[-2:]
            bad = [u for u in sample if _runtime_target(u) is None]
            n_gs = sum(1 for u in urls if str(u).startswith('gs://'))
            status = 'OK' if not n_gs else f'OK(gs->cns {n_gs})'
            if not urls:
                status, n_bad = 'EMPTY', n_bad + 1
            elif bad:
                status, n_bad = f'UNREADABLE {bad[0]}', n_bad + 1
            print(f"    {status:<18} {name:<34} {len(urls):>5} shards  "
                  f"{_runtime_target(urls[0]) or urls[0] if urls else root}")
        if n_bad:
            raise SystemExit(f"{n_bad} unusable dataset root(s) in {stage_key}")
        resolved_stage = stage

    # The startup gate itself. `_init_run` asserted a hardcoded zone allowlist
    # that predated the CNS replicas and rejected tul outright -- 4 minutes of
    # packaging and scheduling to discover a one-line check the probe could
    # have run locally. Anything that can refuse to start belongs here.
    print("\n--- startup gate (train._init_run preconditions) ---")
    from utils import ckpt_util
    inferred = ckpt_util.infer_zone_card(config, '/tmp/wd')
    print(f"  infer_zone_card    : {inferred}")
    if g3_env.in_google3():
        print(f"  cns_data_roots     : {list(g3_env.cns_data_roots())}")
    else:
        assert inferred in ['us-central1', 'us-east5', 'asia-northeast1-b']
    # The dataloader replica regex only bites at the FIRST CHECKPOINT, ~40 min
    # into a run, so it is worth asserting here rather than discovering later.
    from utils.dataloader_state_util import _REPLICA_DATA_BUCKET_RE as _RX
    # Check the path the loader will REALLY hand the state tracker: a gs://
    # root is rewritten to CNS at the opener, so testing the raw string tests
    # nothing. Every distinct CNS prefix in the mix gets checked.
    _prefixes = set()
    for _root in list(resolved_stage.dataset.root):
        _t = _runtime_target(str(_root)) or str(_root)
        if _t.startswith('/cns/'):
            _prefixes.add('/'.join(_t.split('/')[:6]))
    for _r in sorted(_prefixes):
        if True:
            ok = bool(_RX.match(_r))
            print(f"  replica regex      : {'OK' if ok else 'REJECT'}  {_r}")
            if not ok:
                raise SystemExit(
                    f"dataloader replica regex rejects {_r}: a checkpoint "
                    "written here could not be resumed under strict mode.")

    print("\n--- weights ---")
    from models import clip_vit
    print(f"  CLIP source  : {clip_vit.resolve_clip_source()}")
    from gemma import gm
    print(f"  Gemma ckpt   : {train._gemma_checkpoint_path(config.model.lm_backbone_str, config.model.lm_checkpoint_variant)}")

    print("\n--- checkpoint destination ---")
    from utils import ckpt_util
    bucket = ckpt_util.checkpoint_bucket()
    print(f"  CHECKPOINT_BUCKET : {bucket or '<unset>'}")
    if bucket:
        print(f"  convert_to_gs('/tmp/wd') -> {ckpt_util.convert_to_gs('/tmp/wd')}")
        print(f"  pretrained path          -> {ckpt_util.convert_to_pretrained_gs('/tmp/wd')}")

    tasks = set()
    for scope in (config.training, resolved_stage.training):
        for key in ('online_eval_tasks', 'final_eval_tasks'):
            tasks.update(scope.get(key, []) or [])
    print(f"\neval tasks enabled: {sorted(tasks)}")
    n_bad = _probe_eval_roots(resolved_stage, tasks)
    if 'knn_full' in tasks or 'knn_partial' in tasks:
        n_bad += _probe_knn(config)
    if n_bad:
        raise SystemExit(f"{n_bad} unusable eval root(s) / knn failure")

    print("\nCONFIG PROBE PASSED")


if __name__ == '__main__':
    app.run(main, flags_parser=lambda a: flags.FLAGS(a, known_only=True))
