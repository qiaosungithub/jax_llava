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
    dataset.mix_weights = []

    dataset.num_workers = 8
    dataset.prefetch_factor = 4
    dataset.pin_memory = False
    # Base offset added to the checkpoint step when seeding dataloader shuffles.
    dataset.data_seed_offset = 0
    # Regular WebDataset loaders fill this buffer before yielding. Keep the
    # production default large, but allow remote smoke tests to lower it.
    dataset.webdataset_shuffle_size = 10000
    # Keep custom QA/region loaders from reading huge buffers before first yield.
    dataset.item_shuffle_size = ml_collections.ConfigDict()
    dataset.item_shuffle_size.default = 512
    dataset.item_shuffle_size.llava_ov15 = 50000
    # 0 means use all Visual Genome region annotations for genome_det.
    # LLaVA-1.5-style SFT configs override this to 10.
    dataset.genome_det_regions_per_image = 0
    # Optional per-dataset startup skip for iterable WebDataset streams.
    dataset.stream_start_skip = ml_collections.ConfigDict()
    dataset.stream_start_skip.default = 0
    # 0 keeps PyTorch's default; remote configs can set this to fail fast.
    dataset.dataloader_timeout = 0

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
    training.hf_cache_dir = None
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
    eval.pixelbench_num_workers = 0
    eval.mmbench_root = "https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv"
    eval.mmbench_test_root = "https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_EN.tsv"
    eval.mmbench_cache_dir = "/kmh-nfs-ssd-us-mount/data/cached/zhh/mmbench_eval"
    eval.mmbench_data_cache_dir = "/kmh-nfs-ssd-us-mount/data/cached/zhh/mmbench_data"
    eval.mmbench_export_test = False
    # MMBench prompts can be much longer than short VQA prompts. This is eval-only
    # and uses the short-answer sampler, so it does not change train sequence length.
    eval.mmbench_max_txt_len = 512
    eval.short_answer_max_new_tokens = 8
    eval.mmbench_max_new_tokens = 8
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
    # jit sharding mode: "ddp", "hsdp", or "fsdp". HSDP is the default
    # production path; it shards model/optimizer states over the last mesh axis.
    config.sharding = "hsdp"
    config.wandb_resume_id = ""
    config.local_debug = False

    # Auto-populated at runtime; do not set in yaml.
    config.workdir_hash = None
    config.zone = None
    return config
