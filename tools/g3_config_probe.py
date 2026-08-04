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
        print(f"  roots        : {list(stage.dataset.root)}")
        print(f"  types        : {list(stage.dataset.types)}")
        print(f"  mix_weights  : {list(stage.dataset.get('mix_weights', []))}")
        input_pipeline._assert_same_zone_roots(
            stage.dataset.root, _ZONE.value, local_debug=False)
        print("  locality guard: PASS")
        urls = input_pipeline._expand_gcs_glob_if_needed(list(stage.dataset.root)[0])
        urls = [urls] if isinstance(urls, str) else list(urls)
        print(f"  shards       : {len(urls)}  [{urls[0]} .. {urls[-1]}]")

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

    print("\nCONFIG PROBE PASSED")


if __name__ == '__main__':
    app.run(main, flags_parser=lambda a: flags.FLAGS(a, known_only=True))
