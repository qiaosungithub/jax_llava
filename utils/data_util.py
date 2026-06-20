
from utils.llava_ov15_groups import (
    LLAVA_OV15_GROUP_ALIAS,
    LLAVA_OV15_GROUP_ALIASES,
    LLAVA_OV15_GROUP_ALIAS_SHORT,
    LLAVA_OV15_GROUP_TOTAL_WEIGHT,
    llava_ov15_group_roots,
    llava_ov15_finegrained_roots,
)

_LLAVA_OV15_CONFIG_ROOT = 'gs://kmh-gcp-💣/data/llava-ov-1.5-instruct/configs'
_UREADER_CONFIGS = (
    'ureader_tr',
    'ureader_ocr',
    'ureader_qa',
    'ureader_chart',
    'ureader_cap',
    'ureader_ie',
    'ureader_kg',
)

_ZONE_LOCKED_DATASETS = {
    'laion-400m': 'asia-northeast1-b',
}


def _assert_zone_allowed(name, zone: str):
    locked_zone = _ZONE_LOCKED_DATASETS.get(name)
    if locked_zone is None and isinstance(name, str) and '/data/laion-400m/' in name:
        locked_zone = _ZONE_LOCKED_DATASETS['laion-400m']
    assert locked_zone is None or zone == locked_zone, (
        f'{name} is only available in {locked_zone}; got zone={zone}'
    )


dataset_name_to_path_dict = {
    'laion-aes':         'gs://kmh-gcp-💣/data/laion-aesthetic/part-{00000..00127}-cad4a140-cebd-46fa-b874-e8968f93e32e-c000.snappy/{00000..00040}.tar',
    'laion-400m':        'gs://kmh-gcp-asia-northeast1-b/data/laion-400m/part-{00000..00127}/{00000..00282}.tar',
    'cc12m':             'gs://kmh-gcp-💣/data/cc12m/{00000..01096}.tar',
    # NOTE: shards 00842.tar and 01812.tar are missing from the upstream
    # HuggingFace dataset (and therefore from our mirrors). Use webdataset's
    # '::' separator (split before braceexpand in wds.expand_urls) to chain
    # three brace ranges that skip the holes. Full valid range is
    # 00000..01832 minus {842, 1812} -> 1831 shards.
    # NB: must stay a single str (not a list) -- expand_urls only does
    # braceexpand when it gets a str input.
    'blip3o-short':      (
        'gs://kmh-gcp-💣/data/BLIP3o-Pretrain-Short-Caption/{00000..00841}.tar'
        '::gs://kmh-gcp-💣/data/BLIP3o-Pretrain-Short-Caption/{00843..01811}.tar'
        '::gs://kmh-gcp-💣/data/BLIP3o-Pretrain-Short-Caption/{01813..01832}.tar'
    ),
    'textcaps-train':    'gs://kmh-gcp-💣/data/textcaps/train/shard-{000000..000002}.tar',
    'textcaps-val':      'gs://kmh-gcp-💣/data/textcaps/val/shard-000000.tar',
    'rendered-text-512': 'gs://kmh-gcp-💣/data/rendered-text-512/{000000..009979}.tar',
    'visual-genome':     'gs://kmh-gcp-💣/data/visual_genome/wds/shard-{000000..000040}.tar',
    'visual-genome-gcap':'gs://kmh-gcp-💣/data/visual_genome/wds/shard-{000000..000040}.tar',
    'visual-genome-det': 'gs://kmh-gcp-💣/data/visual_genome/wds/shard-{000000..000040}.tar',
    'vqav2':             'gs://kmh-gcp-💣/data/vqav2/vqav2_image_records_wds/train2014/shard-{000000..000008}.tar',
    'okvqa':             'gs://kmh-gcp-💣/data/okvqa/train/shard-{000000..000004}.tar',
    'okvqa-train':       'gs://kmh-gcp-💣/data/okvqa/train/shard-{000000..000004}.tar',
    'aokvqa':            'gs://kmh-gcp-💣/data/aokvqa/train/shard-{000000..000008}.tar',
    'aokvqa-train':      'gs://kmh-gcp-💣/data/aokvqa/train/shard-{000000..000008}.tar',
    'ocrvqa':            'gs://kmh-gcp-💣/data/ocrvqa/train/shard-{000000..000083}.tar',
    'ocrvqa-train':      'gs://kmh-gcp-💣/data/ocrvqa/train/shard-{000000..000083}.tar',
    'refcoco':           'gs://kmh-gcp-💣/data/refcoco/train/shard-{000000..000008}.tar',
    'refcoco-train':     'gs://kmh-gcp-💣/data/refcoco/train/shard-{000000..000008}.tar',
    'refcocog':          'gs://kmh-gcp-💣/data/refcocog/image_records_wds/train/shard-*.tar',
    'refcocog-train':    'gs://kmh-gcp-💣/data/refcocog/image_records_wds/train/shard-*.tar',
    'gqa':               'gs://kmh-gcp-💣/data/vlm_eval_benchmarks/gqa-balanced/train/shard-{000000..000036}.tar',
    'gqa-train':         'gs://kmh-gcp-💣/data/vlm_eval_benchmarks/gqa-balanced/train/shard-{000000..000036}.tar',
    'gqa-val':           'gs://kmh-gcp-💣/data/vlm_eval_benchmarks/gqa-balanced/val/shard-{000000..000005}.tar',
    'gqa-testdev':       'gs://kmh-gcp-💣/data/vlm_eval_benchmarks/gqa-balanced/testdev/shard-000000.tar',
    'textvqa':           'gs://kmh-gcp-💣/data/textvqa/train/shard-000000.tar',
    'vizwiz-vqa-val':    'gs://kmh-gcp-💣/data/vlm_eval_benchmarks/vizwiz-vqa/val/shard-{000000..000002}.tar',
    'vizwiz-vqa-test':   'gs://kmh-gcp-💣/data/vlm_eval_benchmarks/vizwiz-vqa/test/shard-{000000..000003}.tar',
    'scienceqa-img-train':'gs://kmh-gcp-💣/data/vlm_eval_benchmarks/scienceqa-img/train/shard-{000000..000003}.tar',
    'scienceqa-img-val': 'gs://kmh-gcp-💣/data/vlm_eval_benchmarks/scienceqa-img/validation/shard-{000000..000001}.tar',
    'scienceqa-img-test':'gs://kmh-gcp-💣/data/vlm_eval_benchmarks/scienceqa-img/test/shard-{000000..000001}.tar',
    'seed-bench-image':  'gs://kmh-gcp-💣/data/vlm_eval_benchmarks/seed-bench-image/shard-{000000..000002}.tar',
    'tallyqa':           'gs://kmh-gcp-💣/data/tallyqa/wds/shard-{000000..000067}.tar',
    'dvqa': [
        'gs://kmh-gcp-💣/data/dvqa/wds/train/shard-{000000..000099}.tar',
        'gs://kmh-gcp-💣/data/dvqa/wds/val_easy/shard-{000000..000024}.tar',
        'gs://kmh-gcp-💣/data/dvqa/wds/val_hard/shard-{000000..000024}.tar',
    ],
    'dvqa-train':        'gs://kmh-gcp-💣/data/dvqa/wds/train/shard-{000000..000099}.tar',
    'dvqa-val-easy':     'gs://kmh-gcp-💣/data/dvqa/wds/val_easy/shard-{000000..000024}.tar',
    'dvqa-val-hard':     'gs://kmh-gcp-💣/data/dvqa/wds/val_hard/shard-{000000..000024}.tar',
    'pixmo-count':       'gs://kmh-gcp-💣/data/pixmo-count/image_records_wds/train/shard-*.tar',
    'pixmo-count-train': 'gs://kmh-gcp-💣/data/pixmo-count/image_records_wds/train/shard-*.tar',
    'pixmo-count-val':   'gs://kmh-gcp-💣/data/pixmo-count/image_records_wds/validation/shard-*.tar',
    'pixmo-count-validation': 'gs://kmh-gcp-💣/data/pixmo-count/image_records_wds/validation/shard-*.tar',
    'pixmo-count-test':  'gs://kmh-gcp-💣/data/pixmo-count/image_records_wds/test/shard-*.tar',
    'pixmo-points':      'gs://kmh-gcp-💣/data/pixmo-points/image_records_wds/train/shard-*.tar',
    'pixmo-points-train':'gs://kmh-gcp-💣/data/pixmo-points/image_records_wds/train/shard-*.tar',
    'pixmo-cap-qa':      'gs://kmh-gcp-💣/data/pixmo-cap-qa/image_records_wds/train/shard-*.tar',
    'pixmo-capqa':       'gs://kmh-gcp-💣/data/pixmo-cap-qa/image_records_wds/train/shard-*.tar',
    'pixmo-cap-qa-train':'gs://kmh-gcp-💣/data/pixmo-cap-qa/image_records_wds/train/shard-*.tar',
    'pixmo-capqa-train': 'gs://kmh-gcp-💣/data/pixmo-cap-qa/image_records_wds/train/shard-*.tar',
    'llava-1.5':         'gs://kmh-gcp-💣/data/llava-v1-5-mix665k/shards/llava_v1_5_mix665k-{000000..000003}.tar',
    'llava-ov-1.5-instruct': f'{_LLAVA_OV15_CONFIG_ROOT}/*/shard-*.tar',
    'llava-ov1.5':       f'{_LLAVA_OV15_CONFIG_ROOT}/*/shard-*.tar',
    'llava-ov-1.5-instruct-image-shuffled-v1': 'gs://kmh-gcp-💣/data/llava-ov-1.5-instruct-image-shuffled-v1/part-*/shard-*.tar',
    'llava-ov-1.5-instruct-image-shuffled-v1-pilot': 'gs://kmh-gcp-💣/data/llava-ov-1.5-instruct-image-shuffled-v1-pilot/shard-*.tar',
    'ai2d':              f'{_LLAVA_OV15_CONFIG_ROOT}/ai2d/shard-*.tar',
    'ai2d-train':        f'{_LLAVA_OV15_CONFIG_ROOT}/ai2d/shard-*.tar',
    'ureader':           [f'{_LLAVA_OV15_CONFIG_ROOT}/{name}/shard-*.tar' for name in _UREADER_CONFIGS],
    'ureader-tr':        f'{_LLAVA_OV15_CONFIG_ROOT}/ureader_tr/shard-*.tar',
    'ureader-train':     f'{_LLAVA_OV15_CONFIG_ROOT}/ureader_tr/shard-*.tar',
    'ureader-ocr':       f'{_LLAVA_OV15_CONFIG_ROOT}/ureader_ocr/shard-*.tar',
    'ureader-qa':        f'{_LLAVA_OV15_CONFIG_ROOT}/ureader_qa/shard-*.tar',
    'ureader-chart':     f'{_LLAVA_OV15_CONFIG_ROOT}/ureader_chart/shard-*.tar',
    'ureader-cap':       f'{_LLAVA_OV15_CONFIG_ROOT}/ureader_cap/shard-*.tar',
    'ureader-ie':        f'{_LLAVA_OV15_CONFIG_ROOT}/ureader_ie/shard-*.tar',
    'ureader-kg':        f'{_LLAVA_OV15_CONFIG_ROOT}/ureader_kg/shard-*.tar',
    'ureader_tr':        f'{_LLAVA_OV15_CONFIG_ROOT}/ureader_tr/shard-*.tar',
    'ureader_ocr':       f'{_LLAVA_OV15_CONFIG_ROOT}/ureader_ocr/shard-*.tar',
    'ureader_qa':        f'{_LLAVA_OV15_CONFIG_ROOT}/ureader_qa/shard-*.tar',
    'ureader_chart':     f'{_LLAVA_OV15_CONFIG_ROOT}/ureader_chart/shard-*.tar',
    'ureader_cap':       f'{_LLAVA_OV15_CONFIG_ROOT}/ureader_cap/shard-*.tar',
    'ureader_ie':        f'{_LLAVA_OV15_CONFIG_ROOT}/ureader_ie/shard-*.tar',
    'ureader_kg':        f'{_LLAVA_OV15_CONFIG_ROOT}/ureader_kg/shard-*.tar',
}

# Default dataset_type for each named dataset.
# Used by resolve_dataset_roots when no explicit 'type' is given in an item.
dataset_name_to_type_dict = {
    'laion-aes':         'laion_aes',
    'laion-400m':        'laion_aes',
    'cc12m':             'cc12m',
    'blip3o-short':      'blip3o',
    'textcaps-train':    'textcaps',
    'textcaps-val':      'textcaps',
    'rendered-text-512': 'rendered_text',
    'visual-genome':     'genome',
    'visual-genome-gcap':'genome_gcap',
    'visual-genome-det': 'genome_det',
    'vqav2':             'vqav2',
    'okvqa':             'okvqa',
    'okvqa-train':       'okvqa',
    'aokvqa':            'aokvqa',
    'aokvqa-train':      'aokvqa',
    'ocrvqa':            'ocrvqa',
    'ocrvqa-train':      'ocrvqa',
    'refcoco':           'refcoco',
    'refcoco-train':     'refcoco',
    'refcocog':          'refcoco',
    'refcocog-train':    'refcoco',
    'gqa':               'gqa',
    'gqa-train':         'gqa',
    'gqa-val':           'gqa',
    'gqa-testdev':       'gqa',
    'textvqa':           'textvqa',
    'vizwiz-vqa-val':    'vizwiz',
    'vizwiz-vqa-test':   'vizwiz',
    'scienceqa-img-train':'scienceqa_img',
    'scienceqa-img-val': 'scienceqa_img',
    'scienceqa-img-test':'scienceqa_img',
    'seed-bench-image':  'seed_bench',
    'tallyqa':           'tallyqa',
    'dvqa':              'dvqa',
    'dvqa-train':        'dvqa',
    'dvqa-val-easy':     'dvqa',
    'dvqa-val-hard':     'dvqa',
    'pixmo-count':       'pixmo_count',
    'pixmo-count-train': 'pixmo_count',
    'pixmo-count-val':   'pixmo_count',
    'pixmo-count-validation': 'pixmo_count',
    'pixmo-count-test':  'pixmo_count',
    'pixmo-points':      'pixmo_points',
    'pixmo-points-train':'pixmo_points',
    'pixmo-cap-qa':      'pixmo_cap_qa',
    'pixmo-capqa':       'pixmo_cap_qa',
    'pixmo-cap-qa-train':'pixmo_cap_qa',
    'pixmo-capqa-train': 'pixmo_cap_qa',
    'llava-1.5':         'llava15',
    'llava-ov-1.5-instruct': 'llava_ov15',
    'llava-ov1.5':       'llava_ov15',
    'llava-ov-1.5-instruct-image-shuffled-v1': 'llava_ov15',
    'llava-ov-1.5-instruct-image-shuffled-v1-pilot': 'llava_ov15',
    LLAVA_OV15_GROUP_ALIAS: 'llava_ov15',
    LLAVA_OV15_GROUP_ALIAS_SHORT: 'llava_ov15',
    'ai2d':              'ai2d',
    'ai2d-train':        'ai2d',
    'ureader':           'ureader',
    'ureader-tr':        'ureader',
    'ureader-train':     'ureader',
    'ureader-ocr':       'ureader',
    'ureader-qa':        'ureader',
    'ureader-chart':     'ureader',
    'ureader-cap':       'ureader',
    'ureader-ie':        'ureader',
    'ureader-kg':        'ureader',
    'ureader_tr':        'ureader',
    'ureader_ocr':       'ureader',
    'ureader_qa':        'ureader',
    'ureader_chart':     'ureader',
    'ureader_cap':       'ureader',
    'ureader_ie':        'ureader',
    'ureader_kg':        'ureader',
}


def _resolve_one(name, zone: str):
    """Resolve a single dataset name or raw GCS path to a full path."""
    if isinstance(name, (list, tuple)):
        return [_resolve_one(item, zone) for item in name]
    _assert_zone_allowed(name, zone)
    if name in dataset_name_to_path_dict:
        return _resolve_one(dataset_name_to_path_dict[name], zone)
    if '💣' in name:
        return name.replace('💣', zone)
    return name


def _item_name_and_type(item):
    if hasattr(item, 'get') and not isinstance(item, str):
        name = item.get('name', '')
        dtype = item.get('type', dataset_name_to_type_dict.get(name, ''))
    else:
        name = str(item)
        dtype = dataset_name_to_type_dict.get(name, '')
    return name, dtype


def _item_weight(item, raw_weights, item_index):
    if (
        hasattr(item, 'get')
        and not isinstance(item, str)
        and item.get('weight', None) is not None
    ):
        return float(item['weight'])
    if raw_weights and len(raw_weights) > item_index:
        return float(raw_weights[item_index])
    return None


def _append_llava_ov15_grouped(
    *,
    zone,
    alias_name,
    base_weight,
    resolved_roots,
    resolved_types,
    resolved_weights,
    resolved_names,
    min_shards_standalone=None,
    weight_basis="samples",
    group_weights=None,
    config_weights=None,
):
    """Expand the OV1.5 grouped alias into per-(group/config) sources.

    Absolute-M weighting: each source's weight is its base count (images for
    'samples', emitted QA pairs for 'questions') in MILLIONS, times the per-group
    and per-config multipliers, times `base_weight` (an overall OV1.5 multiplier;
    the alias item's `weight`, default 1.0). Each OV1.5 sub-source therefore sits
    in the same "millions of examples" unit as the other datasets in the mix, and
    OV1.5's total share emerges from the data (~54 for 'questions' / ~14.7 for
    'samples') instead of being a number you must set.
    """
    overall_mult = float(base_weight)
    if min_shards_standalone is not None and int(min_shards_standalone) > 0:
        names, roots_list, counts = llava_ov15_finegrained_roots(
            int(min_shards_standalone),
            basis=weight_basis,
            group_weights=group_weights,
            config_weights=config_weights,
        )
    else:
        names, roots_list, counts = llava_ov15_group_roots()
    for group_name, roots, count in zip(names, roots_list, counts):
        resolved_roots.append(_resolve_one(roots, zone))
        resolved_types.append('llava_ov15')
        resolved_weights.append(float(count) / 1e6 * overall_mult)
        resolved_names.append(f"{alias_name}:{group_name}")


def resolve_dataset_roots(config, zone):
    """Resolve dataset names/paths to full GCS paths for the given zone.

    Supports two config formats (checked in order):

    New — config.dataset.items (list of dicts):
        items:
          - {name: laion-aes}
          - {name: visual-genome-gcap, type: genome_gcap}
      Populates config.dataset.root (resolved paths) and
      config.dataset.types (dataset_type per entry) for create_split.

    Legacy — config.dataset.root (plain list of name strings):
        root:
          - laion-aes
          - visual-genome
      Populates config.dataset.root and config.dataset.types from
      dataset_name_to_type_dict.

    Eval roots (vqav2_root, mme_root, …) are always resolved.
    """
    ov15_min_shards = config.dataset.get('llava_ov15_min_shards_standalone', None)
    ov15_basis = config.dataset.get('llava_ov15_weight_basis', 'samples') or 'samples'
    ov15_group_weights = dict(config.dataset.get('llava_ov15_group_weights', {}) or {})
    ov15_config_weights = dict(config.dataset.get('llava_ov15_config_weights', {}) or {})
    items = list(config.dataset.get('items', []) or [])
    if items:
        resolved_roots = []
        resolved_types = []
        resolved_weights = []
        resolved_names = []
        raw_weights = list(config.dataset.get('mix_weights', []) or [])
        raw_weights_match_items = len(raw_weights) == len(items)
        for item_index, item in enumerate(items):
            name, dtype = _item_name_and_type(item)
            item_weight = _item_weight(
                item,
                raw_weights if raw_weights_match_items else [],
                item_index,
            )
            if name in LLAVA_OV15_GROUP_ALIASES:
                _append_llava_ov15_grouped(
                    zone=zone,
                    alias_name=name,
                    base_weight=(
                        item_weight
                        if item_weight is not None
                        else 1.0
                    ),
                    resolved_roots=resolved_roots,
                    resolved_types=resolved_types,
                    resolved_weights=resolved_weights,
                    resolved_names=resolved_names,
                    min_shards_standalone=ov15_min_shards,
                    weight_basis=ov15_basis,
                    group_weights=ov15_group_weights,
                    config_weights=ov15_config_weights,
                )
                continue

            resolved_roots.append(_resolve_one(name, zone))
            resolved_types.append(dtype)
            resolved_names.append(name)
            if item_weight is not None:
                resolved_weights.append(float(item_weight))
        config.dataset.root  = resolved_roots
        config.dataset.types = resolved_types
        config.dataset.resolved_names = resolved_names
        if resolved_weights:
            if len(resolved_weights) != len(resolved_roots):
                raise ValueError(
                    "Partial dataset weights after resolving dataset.items: "
                    f"{len(resolved_weights)} weights for {len(resolved_roots)} roots. "
                    "Use either item.weight on every item or a mix_weights list matching items."
                )
            config.dataset.mix_weights = resolved_weights
    else:
        # Legacy: plain list of name strings / raw paths
        roots = list(config.dataset.root or [])
        if roots:
            resolved_roots = []
            resolved_types = []
            resolved_weights = []
            resolved_names = []
            raw_weights = list(config.dataset.get('mix_weights', []) or [])
            raw_weights_match_roots = len(raw_weights) == len(roots)
            for root_index, root in enumerate(roots):
                if root in LLAVA_OV15_GROUP_ALIASES:
                    _append_llava_ov15_grouped(
                        zone=zone,
                        alias_name=root,
                        base_weight=(
                            raw_weights[root_index]
                            if raw_weights_match_roots
                            else 1.0
                        ),
                        resolved_roots=resolved_roots,
                        resolved_types=resolved_types,
                        resolved_weights=resolved_weights,
                        resolved_names=resolved_names,
                        min_shards_standalone=ov15_min_shards,
                        weight_basis=ov15_basis,
                        group_weights=ov15_group_weights,
                        config_weights=ov15_config_weights,
                    )
                    continue
                resolved_roots.append(_resolve_one(root, zone))
                resolved_types.append(dataset_name_to_type_dict.get(root, ''))
                resolved_names.append(str(root))
                if raw_weights_match_roots:
                    resolved_weights.append(float(raw_weights[root_index]))
            config.dataset.root  = resolved_roots
            config.dataset.types = resolved_types
            config.dataset.resolved_names = resolved_names
            if resolved_weights:
                if len(resolved_weights) != len(resolved_roots):
                    raise ValueError(
                        "Partial dataset weights after resolving dataset.root: "
                        f"{len(resolved_weights)} weights for {len(resolved_roots)} roots. "
                        "Use either no mix_weights or a mix_weights list matching root."
                    )
                config.dataset.mix_weights = resolved_weights

    for _eval_root_key in [
        'vqav2_root',
        'mme_root',
        'textvqa_root',
        'gqa_root',
        'vizwiz_root',
        'scienceqa_img_root',
        'seed_bench_root',
    ]:
        if config.eval.get(_eval_root_key, False) and '💣' in config.eval[_eval_root_key]:
            config.eval[_eval_root_key] = config.eval[_eval_root_key].replace('💣', zone)
    if config.eval.get("pope_root", False) and "💣" in config.eval.pope_root:
        config.eval.pope_root = config.eval.pope_root.replace("💣", zone)
    if (
        config.eval.get("pope_image_root", False)
        and "💣" in config.eval.pope_image_root
    ):
        config.eval.pope_image_root = config.eval.pope_image_root.replace("💣", zone)
    if config.eval.get('mmbench_root', False) and '💣' in config.eval.mmbench_root:
        config.eval.mmbench_root = config.eval.mmbench_root.replace('💣', zone)
    if config.eval.get('mmbench_test_root', False) and '💣' in config.eval.mmbench_test_root:
        config.eval.mmbench_test_root = config.eval.mmbench_test_root.replace('💣', zone)
    if config.eval.get('refcocog_root', False) and '💣' in config.eval.refcocog_root:
        config.eval.refcocog_root = config.eval.refcocog_root.replace('💣', zone)
    if config.eval.get('refcocog_image_root', False) and '💣' in config.eval.refcocog_image_root:
        config.eval.refcocog_image_root = config.eval.refcocog_image_root.replace('💣', zone)
    if config.eval.get('pixelbench_root', False) and '💣' in config.eval.pixelbench_root:
        config.eval.pixelbench_root = config.eval.pixelbench_root.replace('💣', zone)
    for _eval_root_key in ['mmvp_root', 'vstar_root', 'ocrbench_root', 'countbenchqa_root']:
        if config.eval.get(_eval_root_key, False) and '💣' in config.eval[_eval_root_key]:
            config.eval[_eval_root_key] = config.eval[_eval_root_key].replace('💣', zone)
