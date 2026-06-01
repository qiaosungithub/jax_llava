
dataset_name_to_path_dict = {
    'laion-aes':         'gs://kmh-gcp-💣/data/laion-aesthetic/part-{00000..00127}-cad4a140-cebd-46fa-b874-e8968f93e32e-c000.snappy/{00000..00040}.tar',
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
    'llava-1.5':         'gs://kmh-gcp-💣/data/llava-v1-5-mix665k/shards/llava_v1_5_mix665k-{000000..000003}.tar',
    'llava-ov-1.5-instruct': 'gs://kmh-gcp-💣/data/llava-ov-1.5-instruct/configs/*/shard-*.tar',
    'llava-ov1.5':       'gs://kmh-gcp-💣/data/llava-ov-1.5-instruct/configs/*/shard-*.tar',
    'llava-ov-1.5-instruct-image-shuffled-v1': 'gs://kmh-gcp-💣/data/llava-ov-1.5-instruct-image-shuffled-v1/part-*/shard-*.tar',
    'llava-ov-1.5-instruct-image-shuffled-v1-pilot': 'gs://kmh-gcp-💣/data/llava-ov-1.5-instruct-image-shuffled-v1-pilot/shard-*.tar',
}

# Default dataset_type for each named dataset.
# Used by resolve_dataset_roots when no explicit 'type' is given in an item.
dataset_name_to_type_dict = {
    'laion-aes':         'laion_aes',
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
    'llava-1.5':         'llava15',
    'llava-ov-1.5-instruct': 'llava_ov15',
    'llava-ov1.5':       'llava_ov15',
    'llava-ov-1.5-instruct-image-shuffled-v1': 'llava_ov15',
    'llava-ov-1.5-instruct-image-shuffled-v1-pilot': 'llava_ov15',
}


def _resolve_one(name, zone: str):
    """Resolve a single dataset name or raw GCS path to a full path."""
    if isinstance(name, (list, tuple)):
        return [_resolve_one(item, zone) for item in name]
    if name in dataset_name_to_path_dict:
        return _resolve_one(dataset_name_to_path_dict[name], zone)
    if '💣' in name:
        return name.replace('💣', zone)
    return name


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
    items = list(config.dataset.get('items', []) or [])
    if items:
        resolved_roots = []
        resolved_types = []
        for item in items:
            if isinstance(item, dict):
                name = item.get('name', '')
                dtype = item.get('type', dataset_name_to_type_dict.get(name, ''))
            else:
                name = str(item)
                dtype = dataset_name_to_type_dict.get(name, '')
            resolved_roots.append(_resolve_one(name, zone))
            resolved_types.append(dtype)
        config.dataset.root  = resolved_roots
        config.dataset.types = resolved_types
    else:
        # Legacy: plain list of name strings / raw paths
        roots = list(config.dataset.root or [])
        if roots:
            config.dataset.root  = [_resolve_one(n, zone) for n in roots]
            config.dataset.types = [
                dataset_name_to_type_dict.get(n, '') for n in roots
            ]

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
