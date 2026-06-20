"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # ------------------------------------------------------------
    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()

    # Preferred format: list of {name, type(optional)} dicts.
    # resolve_dataset_roots() will expand this into dataset.root and dataset.types.
    # e.g. [{'name': 'visual-genome-gcap', 'type': 'genome_gcap'}, {'name': 'laion-aes'}, {'name': 'llava-1.5'}]
    dataset.items = []

    # Legacy: plain list of dataset name strings (still supported).
    dataset.root  = ['/kmh-nfs-ssd-us-mount/data/imagenet']
    # Populated automatically by resolve_dataset_roots(); do not set manually.
    dataset.types = []
    # Auto-populated by resolve_dataset_roots() (the OV1.5 grouped expansion
    # needs per-source names). Predeclared so it can be set on a LOCKED config.
    dataset.resolved_names = []
    dataset.mix_weights = []

    dataset.num_workers = 8
    dataset.prefetch_factor = 4
    dataset.pin_memory = False
    # Base offset added to the checkpoint step when seeding dataloader shuffles.
    dataset.data_seed_offset = 0
    dataset.stateful_dataloader = False
    dataset.stateful_dataloader_strict = True
    # None means snapshot at checkpoint cadence. Avoid torchdata's default of
    # snapshotting every batch, which serializes large shuffle buffers.
    dataset.stateful_snapshot_every_n_steps = None
    # Regular WebDataset loaders fill this buffer before yielding. Keep the
    # production default large, but allow remote smoke tests to lower it.
    dataset.webdataset_shuffle_size = 10000
    # Keep custom QA/region loaders from reading huge buffers before first yield.
    dataset.item_shuffle_size = ml_collections.ConfigDict()
    dataset.item_shuffle_size.default = 2048
    dataset.item_shuffle_size.llava_ov15 = 50000
    dataset.item_shuffle_size.vqav2 = 2048
    dataset.item_shuffle_size.okvqa = 2048
    dataset.item_shuffle_size.aokvqa = 2048
    dataset.item_shuffle_size.ocrvqa = 2048
    dataset.item_shuffle_size.gqa = 2048
    dataset.item_shuffle_size.textvqa = 2048
    dataset.item_shuffle_size.textcaps = 2048
    dataset.item_shuffle_size.genome = 2048
    dataset.item_shuffle_size.genome_gcap = 2048
    dataset.item_shuffle_size.genome_det = 2048
    dataset.item_shuffle_size.refcoco = 2048
    dataset.item_shuffle_size.tallyqa = 2048
    dataset.item_shuffle_size.dvqa = 2048
    dataset.item_shuffle_size.pixmo_count = 2048
    dataset.item_shuffle_size.pixmo_points = 2048
    dataset.item_shuffle_size.pixmo_cap_qa = 2048
    # Optional per-child shuffle allocation for expanded mixtures. Disabled by
    # default; configs can enable it for LLaVA-OV1.5 grouped SFT.
    dataset.weighted_item_shuffle_size = ml_collections.ConfigDict()
    dataset.weighted_item_shuffle_size.enabled = False
    dataset.weighted_item_shuffle_size.total = 65536
    dataset.weighted_item_shuffle_size.min = 512
    dataset.weighted_item_shuffle_size.max = None
    dataset.weighted_item_shuffle_size.include_types = ["llava_ov15"]
    # LLaVA-OV1.5 grouped mixture granularity. None/0 keeps the coarse 13
    # task-family groups. A positive value splits each group into per-config
    # StatefulRandomMix sources (each config with >= this many shards becomes its
    # own fixed-weight source; smaller configs merge into a per-group tail),
    # which makes the config mixture stationary and removes the valid-token
    # window-mean drift. Weights split each group's weight proportional to
    # per-config shard count, preserving the group-level mixture.
    dataset.llava_ov15_min_shards_standalone = None
    # When True, expand multi-turn conversations at shuffle-buffer fill time so
    # each QA turn is an independent buffer element (stationary turn distribution,
    # no slow-draining pending slot). Pairs with the finer-grained mixture above.
    dataset.expand_conversations_at_fill = False
    # Base for the per-config OV1.5 mixture weight:
    #   'samples'   -> per-config image count (reproduces the image-proportional
    #                  13-group mixture; default).
    #   'questions' -> per-config emitted QA-pair count (images x avg turns); this
    #                  matches the natural distribution of a uniform image read +
    #                  conversation expansion.
    #   'shards'    -> tar-shard count (image proxy, no count tables needed).
    dataset.llava_ov15_weight_basis = "samples"
    # Optional per-group / per-config weight MULTIPLIERS (default 1.0 each). Keys
    # are group names (e.g. 'ocr_text_reading') / config names (e.g. 'coco').
    # Final per-source weight = base-count-in-millions x group_mult x config_mult
    # x the OV1.5 item weight (an overall multiplier, default 1.0). NOT normalised:
    # OV1.5's total share emerges from the data. Dynamic keys; filled from yaml.
    dataset.llava_ov15_group_weights = ml_collections.ConfigDict()
    dataset.llava_ov15_config_weights = ml_collections.ConfigDict()
    # Debug-only override for loader audits that run in a single Python process
    # but want to simulate multi-host stream-count shuffle scaling.
    dataset.shuffle_total_streams_override = None
    # Match the LLaVA-1.5/PaliGemma SFT recipe: sample 10 Visual Genome region
    # annotations per image for genome_det.
    dataset.genome_det_regions_per_image = 10
    # Optional per-dataset startup skip for iterable WebDataset streams.
    dataset.stream_start_skip = ml_collections.ConfigDict()
    dataset.stream_start_skip.default = 0
    dataset.stream_start_skip.pixmo_count = 0
    dataset.stream_start_skip.pixmo_points = 0
    dataset.stream_start_skip.pixmo_cap_qa = 0
    # Remote production default. Debug configs can lower this to fail faster.
    dataset.dataloader_timeout = 900

    dataset.image_size = 224
    # Image geometry before normalization:
    #   "letterbox": preserve aspect ratio and pad to image_size x image_size.
    #   "stretch": directly resize to image_size x image_size.
    dataset.resize_mode = "letterbox"

    dataset.max_txt_len = 64

    # Nested-mask sampling config (logit-normal over token budgets).
    # Distribution is discretized to token counts in {4, 8, 16, 32, 64, 128, 256}.
    # input_pipeline maps each dataset_type to one of these categories.
    dataset.nested_mask_logit_normal = ml_collections.ConfigDict()
    dataset.nested_mask_logit_normal.caption = ml_collections.ConfigDict()
    dataset.nested_mask_logit_normal.caption.mu = 0.0
    dataset.nested_mask_logit_normal.caption.sigma = 1.2
    dataset.nested_mask_logit_normal.vqa = ml_collections.ConfigDict()
    dataset.nested_mask_logit_normal.vqa.mu = 0.0
    dataset.nested_mask_logit_normal.vqa.sigma = 0.6
    dataset.nested_mask_logit_normal.ocr = ml_collections.ConfigDict()
    dataset.nested_mask_logit_normal.ocr.mu = 1.0
    dataset.nested_mask_logit_normal.ocr.sigma = 0.6
    dataset.nested_mask_logit_normal.grounded_caption = ml_collections.ConfigDict()
    dataset.nested_mask_logit_normal.grounded_caption.mu = 1.0
    dataset.nested_mask_logit_normal.grounded_caption.sigma = 0.6

    # ------------------------------------------------------------
    # Training
    config.training = training = ml_collections.ConfigDict()

    training.adam = adam = ml_collections.ConfigDict()
    adam.learning_rate = 1e-4
    adam.adam_b2 = 0.95
    adam.weight_decay = 0.0

    training.muon = muon = ml_collections.ConfigDict()
    muon.learning_rate = 5e-4
    muon.weight_decay = 0.0
    muon.consistent_rms = True
    # Optional dedicated LR for the vision encoder param group.
    # If unset, the vision encoder uses the main optimizer LR.
    training.vision_encoder_learning_rate = None
    # Optional dedicated LR for the multimodal connector/projector param group.
    # If unset, the connector uses the main optimizer LR.
    training.connector_learning_rate = None
    # Backward-compatible aliases for the same connector/projector group.
    training.projector_learning_rate = None
    training.exclude_bias_norm_from_weight_decay = True
    training.grad_clip_norm = 0.0 # disabled

    training.batch_size = 256

    training.num_steps = 1000

    training.log_per_step = 100
    training.sample_per_step = -1
    training.checkpoint_per_step = -1
    training.log_vis_per_step = -1
    training.online_eval_per_step = -1
    training.online_eval_tasks = []
    training.final_eval_tasks = []
    training.warmup_steps = 1
    # Empty means legacy single-stage train. "llava15_two_stage" keeps one
    # workdir/checkpoint stream and switches dataloaders/optimizer at stage1_steps.
    training.curriculum = ""
    training.stage1_steps = 0
    training.stage2_steps = 0
    training.curriculum_stage_name = ""
    training.curriculum_stage_key = ""
    training.curriculum_stage_index = 0
    training.curriculum_stage_start_step = 0
    training.curriculum_stage_end_step = 0
    training.curriculum_global_num_steps = 0
    training.copy_stage1_checkpoint_to_pretrained = True
    training.copy_final_checkpoint_to_pretrained = True

    training.seed = 42

    # training.ema_val = [0.9999]

    training.optimizer = "adam"
    training.lr_schedule = "cos"

    # Whether to freeze the LM backbone params during training (only the
    # image encoder + projector receive gradients).
    training.freeze_lm = False
    # Optional split LM freeze flags for late-fusion runs. If both are None,
    # freeze_lm keeps its legacy behavior. Otherwise:
    # freeze_lm_embed freezes the embedder + layers before model.txt_feature_layer;
    # freeze_lm_late freezes layers from model.txt_feature_layer onward + final_norm.
    training.freeze_lm_embed = None
    training.freeze_lm_late = None
    # If freeze_lm=True, keep the Gemma <loc0000>..<loc1023> embedding rows
    # trainable while all original token embedding rows remain frozen.
    training.train_loc_embeddings_when_lm_frozen = True
    training.freeze_image_encoder = False
    training.vision_tower_from_scratch = False
    training.clip_from_pt = True
    training.hf_cache_dir = "/dev/shm/huggingface"
    # If enabled, only load pretrained Gemma embedder + first N transformer
    # layers into lm_backbone; later multimodal layers stay randomly initialized.
    # When lm_init_pretrained_num_layers is None, model.txt_feature_layer is used.
    training.lm_init_pretrained_text_layers_only = False
    training.lm_init_pretrained_num_layers = None

    # ------------------------------------------------------------
    # MeanFlow
    config.model = model = ml_collections.ConfigDict()
    model.lm_backbone_str = "gemma_dummy"
    # PT by default for backward compatibility. LLaVA/Gemma sanity configs use
    # IT because Vicuna-v1.5 is instruction-tuned.
    model.lm_checkpoint_variant = "pt"
    model.mask_strategy = "diy"
    model.attn_logits_soft_cap = 0.0   # 0.0 = disabled; e.g. 50.0 for Gemma2-style
    model.final_logit_softcap = 0.0    # 0.0 = disabled; e.g. 30.0 for Gemma2-style
    model.txt_feature_layer = 0 # 0: disabled
    # CFG training. Defaults keep the standard image-conditioned NTP objective.
    model.vlm_loss_weight = 1.0
    model.text_only_loss_weight = 0.0
    model.cfg_loss_weight = 0.0
    model.alpha = 0.0
    # None means auto: when txt_feature_layer > 0 and the prefix text LM blocks
    # are frozen, treat their outputs as fixed features to avoid useless HSDP
    # backward compute. Set False if you intentionally train through that path.
    model.stop_gradient_text_features = None
    model.image_post_connector_scale = 1.0
    # fixed_scale keeps legacy behavior. match_text_stats normalizes projected
    # image tokens per sample and matches prompt text-token mean/std before concat.
    model.image_post_connector_transform = "fixed_scale"
    model.image_text_stat_source = "prompt"
    model.image_text_stat_axes = "scalar"  # scalar or per_channel
    model.prompt_causal = True
    model.token_loss_mode = "hidden_scan"  # hidden_scan or full_decode
    model.token_loss_chunk_size = 8192

    # ------------------------------------------------------------
    # Sampling
    config.sampling = sampling = ml_collections.ConfigDict()

    # ------------------------------------------------------------
    # Eval
    config.eval = eval = ml_collections.ConfigDict()
    eval.device_batch_size = 32
    eval.result_cache_dir = "/kmh-nfs-ssd-us-mount/data/cached/zhh"
    eval.cache_ref = '/kmh-nfs-ssd-us-mount/data/cached/zhh/imgnet_256_train_jax_stats_20250205.npz'
    eval.vqav2_root = 'gs://kmh-gcp-💣/data/vqav2/vqav2_image_records_wds/val2014'
    eval.vqav2_cache_dir = "/kmh-nfs-ssd-us-mount/data/cached/zhh/vqav2_eval"
    eval.mme_root = 'gs://kmh-gcp-💣/data/mme'
    eval.mme_cache_dir = '/kmh-nfs-ssd-us-mount/data/cached/zhh/mme_eval'
    eval.textvqa_root = 'gs://kmh-gcp-💣/data/textvqa/val/shard-000000.tar'
    eval.textvqa_cache_dir = "/kmh-nfs-ssd-us-mount/data/cached/zhh/textvqa_eval"
    eval.gqa_root = 'gs://kmh-gcp-💣/data/vlm_eval_benchmarks/gqa-balanced/testdev'
    eval.gqa_num_samples = 12578
    eval.gqa_cache_dir = "/kmh-nfs-ssd-us-mount/data/cached/zhh/gqa_eval"
    eval.vizwiz_root = 'gs://kmh-gcp-💣/data/vlm_eval_benchmarks/vizwiz-vqa/val'
    eval.vizwiz_num_samples = 4319
    eval.vizwiz_cache_dir = "/kmh-nfs-ssd-us-mount/data/cached/zhh/vizwiz_eval"
    eval.scienceqa_img_root = 'gs://kmh-gcp-💣/data/vlm_eval_benchmarks/scienceqa-img/test'
    eval.scienceqa_img_num_samples = 2017
    eval.scienceqa_img_cache_dir = "/kmh-nfs-ssd-us-mount/data/cached/zhh/scienceqa_img_eval"
    eval.seed_bench_root = 'gs://kmh-gcp-💣/data/vlm_eval_benchmarks/seed-bench-image'
    eval.seed_bench_num_samples = 14233
    eval.seed_bench_cache_dir = "/kmh-nfs-ssd-us-mount/data/cached/zhh/seed_bench_eval"
    eval.cider_cache_dir = "/kmh-nfs-ssd-us-mount/data/cached/zhh/coco_caption_eval"
    eval.pope_root = "gs://kmh-gcp-💣/data/pope/coco_image_records_wds/val2014"
    eval.pope_dataset = "coco"
    eval.pope_image_root = "unused_for_image_records_wds"
    eval.pope_splits = ["random", "popular", "adversarial"]
    eval.pope_prompt_template = "{question}\nPlease answer yes or no.\n"
    eval.mmbench_prompt_prefix = ""
    eval.pope_cache_dir = "/kmh-nfs-ssd-us-mount/data/cached/zhh/pope_eval"
    eval.refcocog_root = "/kmh-nfs-ssd-us-mount/code/hanhong/shared/refcocog/val.json"
    eval.refcocog_image_root = "gs://kmh-gcp-💣/data/coco/train2014"
    eval.refcocog_cache_dir = "/kmh-nfs-ssd-us-mount/data/cached/zhh/refcocog_eval"
    eval.refcocog_iou_threshold = 0.5
    eval.refcocog_num_workers = 0
    eval.pixelbench_root = "gs://kmh-gcp-💣/data/eval/pixelbench"
    eval.pixelbench_benchmarks = ["mmvp", "vstar", "ocrbench", "countbenchqa"]
    eval.pixelbench_cache_dir = "/kmh-nfs-ssd-us-mount/data/cached/zhh/pixelbench_eval"
    # PixelBench also runs VStar with beam_size=5; using the global eval batch
    # can beam-expand into a huge generation batch and OOM during decode.
    eval.pixelbench_device_batch_size = 4
    eval.pixelbench_num_workers = 0
    eval.mmbench_root = "https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv"
    eval.mmbench_test_root = "https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_EN.tsv"
    eval.mmbench_cache_dir = "/kmh-nfs-ssd-us-mount/data/cached/zhh/mmbench_eval"
    eval.mmbench_data_cache_dir = "/kmh-nfs-ssd-us-mount/data/cached/zhh/mmbench_data"
    eval.mmbench_export_test = False
    # MMBench prompts can be much longer than short VQA prompts. This is eval-only
    # and uses a separate sampler, so it does not change train sequence length.
    eval.mmbench_max_txt_len = 512
    # Generation budgets for eval. Keep these task-facing knobs separate from
    # `sampling.max_new_tokens`, which is used for generic visualization/sample
    # generation and may be longer than short-answer benchmark outputs need.
    eval.eval_tokens_default = 64
    eval.eval_tokens_shortqa = 8
    eval.eval_tokens_mid = 16
    eval.eval_tokens_ocr = 32
    eval.eval_tokens_refcoco = 16
    eval.eval_tokens_pixelbench = 32
    eval.eval_tokens_mmbench = 8
    # Backward-compatible aliases.
    eval.short_answer_max_new_tokens = eval.eval_tokens_shortqa
    eval.mmbench_max_new_tokens = eval.eval_tokens_mmbench
    # Online VQAv2 is too expensive at full val scale; final eval still uses
    # eval.vqav2_num_samples unless explicitly overridden.
    eval.online_vqav2_sample_fraction = 0.1
    eval.online_vqav2_num_samples = None
    eval.pope_num_workers = 0
    eval.mme_num_workers = 0
    # Optional smoke-test caps. 0/None keeps the full benchmark.
    eval.debug_max_samples = 0
    eval.mme_max_samples = 0
    eval.pope_max_samples_per_split = 0
    eval.refcocog_max_samples = 0
    eval.pixelbench_max_samples = 0
    eval.mmbench_max_samples = 0
    eval.current_eval_step = -1
    eval.current_eval_run_id = "manual"
    eval.current_eval_suffix = "main"
    
    eval.eval_recon_tokens = []

    # ImageNet KNN eval
    # eval_knn_partial: run online KNN eval (128 images/class) during training
    # eval_knn_full:    run full-dataset KNN eval at end of training
    eval.eval_knn_partial = False
    eval.eval_knn_full = False
    eval.knn_k = 20
    eval.knn_temperature = 0.07
    eval.knn_seed = 42
    eval.knn_images_per_class = 128   # for partial eval
    # Optional remote debug cap.  None/0 keeps the full 50 k ImageNet val set.
    eval.knn_val_examples = None
    eval.knn_batch_size = 256
    eval.knn_num_workers = 4
    # Extract KNN features once, then evaluate both raw cosine KNN and PCA-
    # whitened cosine KNN. The legacy knn_*_acc metric name records raw KNN;
    # knn_*_acc_pca_whitened records the whitened result. 0 keeps all dims.
    eval.knn_eval_raw_and_pca_whitened = True
    eval.knn_pca_whitening = True
    eval.knn_pca_whitening_eps = 1e-5
    eval.knn_pca_whitening_dim = 0
    eval.knn_pca_whitening_batch_size = 65536

    config.finetune = False  # set to True in remote_run_config.yml to load finetune_config.yml

    # ------------------------------------------------------------
    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.wandb_project = ""
    logging.use_wandb = False
    logging.wandb_entity = ""
    logging.wandb_notes = ""
    logging.wandb_tags = []

    # others
    config.load_from = ''
    config.load_from_pretrained = ''
    config.eval_all = False
    config.eval_only = False
    # jit sharding mode: "ddp", "hsdp", "hsdp_legacy_data", or "fsdp".
    # HSDP is the default production path; it shards model/optimizer states over
    # the last mesh axis. "hsdp_legacy_data" keeps HSDP parameter sharding but
    # shards DATA inputs on every mesh axis, matching the older fast stage1
    # layout.
    config.sharding = "hsdp"
    config.wandb_resume_id = ""
    config.local_debug = False

    # Auto-populated at runtime; do not set in yaml.
    config.workdir_hash = None
    config.zone = None
    return config
