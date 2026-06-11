from absl import logging as absl_logging
from functools import partial
import copy
import warnings
from hashlib import md5

import jax
import jax.numpy as jnp
import jax.experimental.multihost_utils as mu
import ml_collections
import numpy as np
import optax
from gemma import gm
from jax import random
from PIL import Image

import input_pipeline
from evals.eval import run_eval_tasks
from evals.eval_imagenet_knn import ensure_imagenet_available
from evals.eval_mmbench import export_mmbench_test_xlsx
from evals.eval_mme import collate_fn
from models.clip_vit import load_clip_vision_params
from models.llava import LlavaGemma
from models.paligemma_enc_dec import PaliGemmaEncDec
from utils import vis_util
from utils.ckpt_util import (
    checkpoint_step,
    copy_latest_checkpoint_to_pretrained,
    infer_zone_card,
    restore_checkpoint,
    restore_checkpoint_params,
    save_checkpoint,
)
from utils.data_util import resolve_dataset_roots
from utils.dataloader_state_util import (
    restore_dataloader_state,
    save_dataloader_state,
    stateful_dataloader_enabled,
)
from utils.frozen_util import (
    get_trainable,
    merge_params,
    merge_params_trainable_loc_embeddings,
    resolve_lm_freeze_flags,
    zero_nonloc_embedding_rows,
)
from utils.llm_util import create_tokenizer, init_loc_token_embeddings
from utils.logging_util import MetricsTracker, Timer, Writer, log_for_0
from utils.pjit_util import MeshMode, prepare_pjit_funcs, shard_cpu_tree_to_mesh
from utils.trainstate_util import create_train_state

warnings.filterwarnings("ignore", message=".*EOF occurred in violation of protocol.*")
absl_logging.set_verbosity(absl_logging.INFO)

LDC = jax.local_device_count()
PRC = jax.process_count()
PRI = jax.process_index()
GDC = jax.device_count()
assert GDC == LDC * PRC, f"{GDC} != {LDC} * {PRC}"


FIXED_PAIRS = {
    'COCO_val2014_000000391895.jpg': "A man with a red helmet on a small moped on a dirt road. ",
    'COCO_val2014_000000060623.jpg': "A young girl inhales with the intent of blowing out a candle. ",
    'COCO_val2014_000000483108.jpg': "A man on a bicycle riding next to a train",
    'COCO_val2014_000000384213.jpg': "A kitchen is shown with a variety of items on the counters.",
    'COCO_val2014_000000386164.jpg': "A wooden ball on top of a wooden stick.",
    'COCO_val2014_000000223648.jpg': "Multiple wooden spoons are shown on a table top.",
    'COCO_val2014_000000403385.jpg': "A bathroom that has a broken wall in the shower.",
    'COCO_val2014_000000294832.jpg': "A bathroom with an enclosed shower next to a sink and a toilet.",
    'COCO_val2014_000000462565.jpg': "people on bicycles ride down a busy street ",
    'COCO_val2014_000000436141.jpg': "A clean bathroom is seen in this image.",
}


def _state_step(state) -> int:
    return int(jax.device_get(state.step))


def _replace_state_step(state, step):
    dtype = getattr(state.step, "dtype", jnp.int32)
    return state.replace(step=jnp.asarray(int(step), dtype=dtype))


def compute_metrics(dict_losses):
    return {k: jnp.mean(v) for k, v in dict_losses.items()}


def train_step(state, batch, rng_init, config):
    rng_step = random.fold_in(rng_init, state.step)
    assert batch['pixel_values'].shape[1:] == (
        config.dataset.image_size,
        config.dataset.image_size,
        3,
    ), f"Unexpected image shape {batch['pixel_values'].shape}"

    freeze_lm = bool(config.training.get('freeze_lm', False))
    freeze_image_encoder = bool(config.training.get('freeze_image_encoder', False))
    train_loc_embeddings_when_lm_frozen = bool(config.training.get('train_loc_embeddings_when_lm_frozen', True))
    txt_feature_layer = int(config.model.get('txt_feature_layer', 0))
    freeze_lm_embed, _ = resolve_lm_freeze_flags(
        freeze_lm=freeze_lm,
        txt_feature_layer=txt_feature_layer,
        freeze_lm_embed=config.training.get('freeze_lm_embed', None),
        freeze_lm_late=config.training.get('freeze_lm_late', None),
    )
    trainable_params, frozen_params = get_trainable(
        state.params,
        freeze_lm=freeze_lm,
        txt_feature_layer=txt_feature_layer,
        freeze_lm_embed=config.training.get('freeze_lm_embed', None),
        freeze_lm_late=config.training.get('freeze_lm_late', None),
        freeze_image_encoder=freeze_image_encoder,
        train_loc_embeddings_when_lm_frozen=train_loc_embeddings_when_lm_frozen,
    )

    def loss_fn(wrt_params):
        params = merge_params_trainable_loc_embeddings(
            wrt_params,
            frozen_params,
            state.params,
            enable=freeze_lm_embed and train_loc_embeddings_when_lm_frozen,
        )
        outputs = state.apply_fn(
            {"params": params},
            input_ids=batch['input_ids'],
            images=batch['pixel_values'],
            prefix_len=batch['prefix_len'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            mask_token_category_probs=batch.get('mask_token_category_probs', None),
            rngs=dict(gen=rng_step),
        )
        loss, log_dict, model_debug = outputs
        return loss, (log_dict, model_debug)

    (loss, (log_dict, model_debug)), grads = jax.value_and_grad(loss_fn, has_aux=True)(trainable_params)

    if frozen_params:
        frozen_zero_grads = jax.tree.map(jnp.zeros_like, frozen_params)
        grads = merge_params(grads, frozen_zero_grads)
    if freeze_lm_embed and train_loc_embeddings_when_lm_frozen:
        grads = zero_nonloc_embedding_rows(grads)

    grad_norm = optax.global_norm(grads)
    metrics = compute_metrics(log_dict)
    metrics['grad_norm'] = grad_norm
    metrics['loss'] = jnp.mean(loss)

    debug = {}
    if config.model.get('recon_loss_weight', 0.0) > 0.0:
        for key in ['recon_imgs', 'orig_imgs']:
            if key in model_debug:
                debug[key] = model_debug[key]

    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics, debug


def sample_step(params, images, prompt_ids, prefix_len, model, max_new_tokens=64, beam_size=1):
    if beam_size > 1:
        return model.apply(
            {'params': params},
            prompt_ids,
            prefix_len,
            images,
            method=model.generate_beam_search,
            beam_size=beam_size,
            max_new_tokens=max_new_tokens,
        )
    return model.apply(
        {'params': params},
        prompt_ids,
        prefix_len,
        images,
        method=model.generate,
        max_new_tokens=max_new_tokens,
    )


def _eval_token_budget(config, key, default, legacy_key=None):
    value = config.eval.get(key, None)
    if value is None and legacy_key:
        value = config.eval.get(legacy_key, None)
    if value is None:
        value = default
    return int(value)


def recon_step(params, images, model: PaliGemmaEncDec, num_visible: int):
    return model.apply({'params': params}, images, num_visible, method=model.reconstruct)


def _partial_init_lm_from_pretrained(random_lm_params, pretrained_lm_params, num_text_layers: int):
    if num_text_layers < 0:
        raise ValueError(f"num_text_layers must be >= 0, got {num_text_layers}")
    required = ["embedder"] + [f"layer_{i}" for i in range(num_text_layers)]
    missing_random = [k for k in required if k not in random_lm_params]
    missing_pretrained = [k for k in required if k not in pretrained_lm_params]
    if missing_random or missing_pretrained:
        raise KeyError(
            "Cannot partially initialize LM. "
            f"missing_random={missing_random}, missing_pretrained={missing_pretrained}"
        )
    merged = dict(random_lm_params)
    for key in required:
        merged[key] = pretrained_lm_params[key]
    return merged


def _get_lm_pretrained_text_layers(config):
    if not bool(config.training.get("lm_init_pretrained_text_layers_only", False)):
        return None
    value = config.training.get("lm_init_pretrained_num_layers", None)
    if value is None:
        value = config.model.get("txt_feature_layer", 0)
    return int(value)


def _infer_stop_gradient_text_features(config, model_kwargs):
    explicit = model_kwargs.get('stop_gradient_text_features', None)
    if explicit is not None:
        return bool(explicit)

    txt_feature_layer = int(model_kwargs.get('txt_feature_layer', 0))
    if txt_feature_layer <= 0:
        return False

    freeze_lm_embed, _ = resolve_lm_freeze_flags(
        freeze_lm=bool(config.training.get('freeze_lm', False)),
        txt_feature_layer=txt_feature_layer,
        freeze_lm_embed=config.training.get('freeze_lm_embed', None),
        freeze_lm_late=config.training.get('freeze_lm_late', None),
    )
    return bool(freeze_lm_embed)


def _create_model(config, *, finetune_mode: bool = False):
    arch = str(config.model.get('arch', 'paligemma_enc_dec')).lower()
    model_kwargs = config.model.to_dict()
    model_kwargs.pop('arch', None)
    model_kwargs.pop('image_size', None)
    model_kwargs.pop('lm_checkpoint_variant', None)
    model_kwargs.pop('rope_scale_when_interpolate', None)

    if arch in {'paligemma_enc_dec', 'prefixmae_paligemma'}:
        if finetune_mode and 'use_decoder' not in model_kwargs:
            model_kwargs['use_decoder'] = False
        model_kwargs['stop_gradient_text_features'] = _infer_stop_gradient_text_features(
            config, model_kwargs
        )
        allowed = set(PaliGemmaEncDec.__dataclass_fields__.keys())
        model_kwargs = {k: v for k, v in model_kwargs.items() if k in allowed}
        return PaliGemmaEncDec(**model_kwargs, image_size=config.dataset.image_size)
    if arch in {'llava', 'llava_gemma'}:
        model_kwargs['stop_gradient_text_features'] = _infer_stop_gradient_text_features(
            config, model_kwargs
        )
        allowed = set(LlavaGemma.__dataclass_fields__.keys())
        model_kwargs = {k: v for k, v in model_kwargs.items() if k in allowed}
        return LlavaGemma(**model_kwargs, image_size=config.dataset.image_size)
    raise ValueError(f'Unsupported model.arch: {arch}')


def _gemma_checkpoint_path(model_name: str, variant: str = "pt"):
    variant = str(variant or "pt").lower()
    if variant not in {"pt", "it"}:
        raise ValueError(f'Unsupported LM checkpoint variant: {variant}')
    paths = {
        ("gemma2_2B", "pt"): gm.ckpts.CheckpointPath.GEMMA2_2B_PT,
        ("gemma2_2B", "it"): gm.ckpts.CheckpointPath.GEMMA2_2B_IT,
        ("gemma3_270M", "pt"): gm.ckpts.CheckpointPath.GEMMA3_270M_PT,
        ("gemma3_270M", "it"): gm.ckpts.CheckpointPath.GEMMA3_270M_IT,
        ("gemma3_1B", "pt"): gm.ckpts.CheckpointPath.GEMMA3_1B_PT,
        ("gemma3_1B", "it"): gm.ckpts.CheckpointPath.GEMMA3_1B_IT,
    }
    if (model_name, variant) in paths:
        return paths[(model_name, variant)]
    raise ValueError(f'Unsupported LM backbone: {model_name}')


def _model_patch_size(config):
    if 'patch_size' in config.model:
        return config.model.patch_size
    if str(config.model.get('arch', '')).lower() in {'llava', 'llava_gemma'}:
        return 14
    return None


def _validate_and_get_local_batch_size(config):
    batch_size = int(config.training.batch_size)
    if batch_size % GDC > 0:
        raise ValueError(f'Global batch size {batch_size} must be divisible by device_count={GDC}')
    if batch_size % PRC > 0:
        raise ValueError('Batch size must be divisible by process_count')
    local_batch_size = batch_size // PRC
    if local_batch_size % LDC > 0:
        raise ValueError('Local batch size must be divisible by local_device_count')
    return local_batch_size


def _create_train_iterator(
    config,
    local_batch_size,
    step_offset,
    *,
    load_from=None,
    zone=None,
    checkpoint_step_for_state=None,
):
    if stateful_dataloader_enabled(config):
        # Exact stateful resume restores iterator cursors/RNG. Do not change the
        # seed by checkpoint step or the restored stream will not match.
        data_seed_offset = int(getattr(config.dataset, "data_seed_offset", 0))
    else:
        data_seed_offset = int(getattr(config.dataset, "data_seed_offset", 0)) + int(step_offset)
    log_for_0(
        'Creating train loader with batch size: %d, data_seed_offset=%d',
        local_batch_size,
        data_seed_offset,
    )
    train_loader, tokenizer = input_pipeline.create_split(
        config,
        local_batch_size,
        data_seed_offset=data_seed_offset,
    )
    if stateful_dataloader_enabled(config) and load_from and int(step_offset) > 0:
        expected_state_step = (
            int(checkpoint_step_for_state)
            if checkpoint_step_for_state is not None
            else int(step_offset)
        )
        restore_dataloader_state(
            train_loader,
            config,
            load_from,
            zone,
            expected_state_step,
            local_batch_size,
        )
    return train_loader, iter(train_loader), tokenizer


def _save_training_checkpoint(state, train_loader, config, workdir, step, local_batch_size):
    save_checkpoint(state, workdir)
    save_dataloader_state(train_loader, config, workdir, step, local_batch_size)


def _flatten_host_local_batch(batch):
    out = {}
    for key, value in batch.items():
        if key == 'is_pad':
            continue
        if hasattr(value, 'numpy'):
            value = value.numpy()
        value = np.asarray(value)
        if key == 'pixel_values':
            if value.ndim == 5:
                value = value.reshape((-1,) + value.shape[2:])
            elif value.ndim != 4:
                raise ValueError(f'Unexpected pixel_values shape: {value.shape}')
        else:
            if value.ndim >= 2 and value.shape[0] == LDC:
                value = value.reshape((-1,) + value.shape[2:])
        if key in {'input_ids', 'labels', 'prefix_len'}:
            value = value.astype(np.int32)
        elif key == 'attention_mask':
            value = value.astype(np.bool_)
        elif key in {'pixel_values', 'mask_token_category_probs'}:
            value = value.astype(np.float32)
        out[key] = value
    return out


def _prepare_host_batch(batch, batch_size=None):
    return _flatten_host_local_batch(input_pipeline.prepare_batch_data(batch, batch_size=batch_size))


def _local_array_to_global(local_arr, mesh, spec):
    local_np = local_arr.numpy() if hasattr(local_arr, 'numpy') else np.asarray(local_arr)
    global_shape = (local_np.shape[0] * PRC,) + local_np.shape[1:]
    sharding = jax.sharding.NamedSharding(mesh, spec)
    return jax.make_array_from_process_local_data(sharding, local_np, global_shape)


def _make_global_batch(batch, partition_specs, mesh):
    return jax.tree_util.tree_map(
        lambda local_arr, spec: _local_array_to_global(local_arr, mesh, spec),
        batch,
        partition_specs,
    )


def _fake_batch(config, global_batch_size, *, max_txt_len=None):
    max_txt_len = int(config.dataset.max_txt_len if max_txt_len is None else max_txt_len)
    return {
        'pixel_values': jax.ShapeDtypeStruct(
            (global_batch_size, config.dataset.image_size, config.dataset.image_size, 3),
            jnp.float32,
        ),
        'input_ids': jax.ShapeDtypeStruct((global_batch_size, max_txt_len), jnp.int32),
        'attention_mask': jax.ShapeDtypeStruct((global_batch_size, max_txt_len), jnp.bool_),
        'labels': jax.ShapeDtypeStruct((global_batch_size, max_txt_len), jnp.int32),
        'prefix_len': jax.ShapeDtypeStruct((global_batch_size,), jnp.int32),
        'mask_token_category_probs': jax.ShapeDtypeStruct((global_batch_size, 7), jnp.float32),
    }


def _attach_sample_metadata(fn, mesh, batch_spec, reduce_scatter):
    fn._mesh = mesh
    fn._batch_spec = batch_spec
    fn._reduce_scatter = reduce_scatter
    return fn


def _build_pjit_fns(config, model, state, mesh_bundle):
    mesh, get_partition_spec, _, reduce_scatter, pjit_compile = mesh_bundle
    state_spec = get_partition_spec(state, MeshMode.MODEL)
    fake_bs = int(config.training.batch_size)
    mesh_total = int(np.prod(mesh.devices.shape))
    if fake_bs % mesh_total != 0:
        fake_bs = mesh_total
    batch_spec = get_partition_spec(_fake_batch(config, fake_bs), MeshMode.DATA)
    mmbench_max_txt_len = int(getattr(config.eval, 'mmbench_max_txt_len', config.dataset.max_txt_len))
    mmbench_batch_spec = get_partition_spec(
        _fake_batch(config, fake_bs, max_txt_len=mmbench_max_txt_len),
        MeshMode.DATA,
    )

    p_train_step = pjit_compile(
        partial(train_step, rng_init=random.PRNGKey(config.training.seed), config=config),
        in_shardings=(state_spec, batch_spec),
        out_shardings=(state_spec, None, None),
        donate_argnums=(0, 1),
    )

    default_tokens = int(config.eval.get(
        'eval_tokens_default',
        config.sampling.get('max_new_tokens', 64),
    ))
    sampling_tokens = int(config.sampling.get('max_new_tokens', default_tokens))
    default_beam = int(config.sampling.get('beam_size', 1))
    short_tokens = _eval_token_budget(
        config,
        'eval_tokens_shortqa',
        8,
        legacy_key='short_answer_max_new_tokens',
    )
    mid_tokens = _eval_token_budget(config, 'eval_tokens_mid', 16)
    ocr_tokens = _eval_token_budget(config, 'eval_tokens_ocr', 32)
    refcoco_tokens = _eval_token_budget(config, 'eval_tokens_refcoco', mid_tokens)
    pixelbench_tokens = _eval_token_budget(config, 'eval_tokens_pixelbench', ocr_tokens)
    mmbench_tokens = _eval_token_budget(
        config,
        'eval_tokens_mmbench',
        short_tokens,
        legacy_key='mmbench_max_new_tokens',
    )

    sample_cache = {}

    def compile_sample_step(max_new_tokens, beam_size=default_beam, sampler_batch_spec=batch_spec, spec_name='default'):
        max_new_tokens = int(max_new_tokens)
        beam_size = int(beam_size)
        cache_key = (max_new_tokens, beam_size, spec_name)
        if cache_key in sample_cache:
            return sample_cache[cache_key]
        sample_out_spec = get_partition_spec(
            jax.ShapeDtypeStruct((fake_bs, max_new_tokens), jnp.int32),
            MeshMode.DATA,
        )
        fn = pjit_compile(
            partial(
                sample_step,
                model=model,
                max_new_tokens=max_new_tokens,
                beam_size=beam_size,
            ),
            in_shardings=(
                state_spec.params,
                sampler_batch_spec['pixel_values'],
                sampler_batch_spec['input_ids'],
                sampler_batch_spec['prefix_len'],
            ),
            out_shardings=sample_out_spec,
        )
        _attach_sample_metadata(fn, mesh, sampler_batch_spec, reduce_scatter)
        sample_cache[cache_key] = fn
        return fn

    p_sample_steps = {
        'default': compile_sample_step(sampling_tokens, default_beam),
        'shortqa': compile_sample_step(short_tokens, default_beam),
        'mid': compile_sample_step(mid_tokens, default_beam),
        'mid_beam5': compile_sample_step(mid_tokens, 5, spec_name='mid_beam5'),
        'ocr': compile_sample_step(ocr_tokens, default_beam),
        'refcoco': compile_sample_step(refcoco_tokens, default_beam),
        'pixelbench': compile_sample_step(pixelbench_tokens, default_beam),
        # MMBench can need a longer prompt length than training/VQA prompts, so
        # it gets its own input spec instead of reusing batch_spec['input_ids'].
        'mmbench': compile_sample_step(
            mmbench_tokens,
            default_beam,
            sampler_batch_spec=mmbench_batch_spec,
            spec_name='mmbench',
        ),
    }

    return state_spec, batch_spec, p_train_step, p_sample_steps, p_sample_steps['mmbench']


def run_p_sample_step(p_sample_step, model, tokenizer, params, images, prompt_ids, prefix_len=None):
    del model
    if prefix_len is None:
        prefix_len = np.zeros((np.asarray(prompt_ids).reshape(-1, prompt_ids.shape[-1]).shape[0],), dtype=np.int32)

    mesh = p_sample_step._mesh
    batch_spec = p_sample_step._batch_spec
    reduce_scatter = p_sample_step._reduce_scatter

    images = _flatten_host_local_batch({'pixel_values': images})['pixel_values']
    prompt_ids = _flatten_host_local_batch({'input_ids': prompt_ids})['input_ids'].astype(np.int32)
    prefix_len = _flatten_host_local_batch({'prefix_len': prefix_len})['prefix_len'].astype(np.int32)

    global_images = _local_array_to_global(images, mesh, batch_spec['pixel_values'])
    global_prompt_ids = _local_array_to_global(prompt_ids, mesh, batch_spec['input_ids'])
    global_prefix_len = _local_array_to_global(prefix_len, mesh, batch_spec['prefix_len'])

    output = p_sample_step(params, global_images, global_prompt_ids, global_prefix_len)
    local_output = reduce_scatter(output, MeshMode.DATA)
    output = np.asarray(jax.device_get(local_output)).reshape(-1, local_output.shape[-1])

    def post_process(token_ids):
        indices = np.where(token_ids == tokenizer.special_tokens.EOS)[0]
        if len(indices) > 0:
            token_ids = token_ids[:indices[0]]
        return token_ids.tolist()

    return [tokenizer.decode(post_process(o)) for o in output]


def run_eval_recon_psnr(p_recon_steps, state_params, eval_images, patch_size, image_size, n_vis=6):
    del p_recon_steps, state_params, eval_images, patch_size, image_size, n_vis
    raise NotImplementedError('recon_psnr eval is not wired for jit/HSDP yet')


def _prepare_knn_if_needed(config, zone, tasks):
    if any(t in {"knn_partial", "knn_full"} for t in tasks):
        log_for_0('[KNN] Resolving TFDS ImageNet data_dir ...')
        data_dir = ensure_imagenet_available(zone, local_debug=config.local_debug)
        log_for_0(f'[KNN] ImageNet TFDS data_dir: {data_dir}')
        return data_dir
    return None


def _prepare_fixed_vis_batch(config, tokenizer):
    log_for_0('Preparing sample pairs for sampling...')
    vis_pairs = [
        input_pipeline.preprocess_fn(
            {
                'jpg': Image.open(f'/kmh-nfs-ssd-us-mount/code/hanhong/shared/COCO/val2014/{k}'),
                'aux': {'gt': v},
            },
            transform=input_pipeline.get_transforms(
                config.dataset.image_size,
                is_train=False,
                resize_mode=getattr(config.dataset, 'resize_mode', 'letterbox'),
            ),
            tokenizer=tokenizer,
            max_len=config.dataset.max_txt_len,
        )
        for k, v in FIXED_PAIRS.items()
    ]
    padded_size = ((len(vis_pairs) + LDC - 1) // LDC) * LDC
    vis_batch = input_pipeline.prepare_batch_data(collate_fn(vis_pairs), batch_size=padded_size)
    return vis_pairs, vis_batch


def _maybe_shard_params(params, mesh, params_spec):
    leaves = jax.tree_util.tree_leaves(params)
    if leaves and isinstance(leaves[0], jax.Array) and not leaves[0].is_fully_addressable:
        return params
    return shard_cpu_tree_to_mesh(params, mesh, params_spec)


def _interpolate_patch_pos_embed(params, target_image_size, patch_size):
    """Interpolate learned patch_pos_embed before restoring at a new resolution."""
    enc_params = params.get('image_encoder', {})
    encoder = enc_params.get('encoder', {}) if hasattr(enc_params, 'get') else {}
    if not hasattr(encoder, 'get') or 'patch_pos_embed' not in encoder:
        return params

    old_posemb = encoder['patch_pos_embed']
    t_old = int(old_posemb.shape[1])
    dim = int(old_posemb.shape[2])
    grid_old = int(np.sqrt(t_old))
    if grid_old * grid_old != t_old:
        raise ValueError(f'patch_pos_embed token count is not square: {t_old}')
    grid_new = int(target_image_size) // int(patch_size)
    t_new = grid_new * grid_new
    if t_old == t_new:
        return params

    log_for_0(
        'Interpolating patch_pos_embed: (%d x %d = %d) -> (%d x %d = %d), '
        'resolution %d -> %d (patch_size=%d)',
        grid_old,
        grid_old,
        t_old,
        grid_new,
        grid_new,
        t_new,
        grid_old * int(patch_size),
        int(target_image_size),
        int(patch_size),
    )

    import scipy.ndimage

    old_grid = np.asarray(jax.device_get(old_posemb[0])).reshape(grid_old, grid_old, dim)
    zoom_factors = (grid_new / grid_old, grid_new / grid_old, 1)
    new_grid = scipy.ndimage.zoom(old_grid, zoom_factors, order=1)
    new_posemb = new_grid.reshape(1, t_new, dim).astype(old_grid.dtype, copy=False)

    params = dict(params)
    params['image_encoder'] = dict(params['image_encoder'])
    params['image_encoder']['encoder'] = dict(params['image_encoder']['encoder'])
    params['image_encoder']['encoder']['patch_pos_embed'] = new_posemb
    return params


def _load_initial_pretrained_params(state, config, mesh, state_spec, step_offset=0):
    to_fp32 = lambda x: x.astype(jnp.float32) if hasattr(x, 'astype') else x
    params = dict(state.params)
    arch = str(config.model.get('arch', 'paligemma_enc_dec')).lower()

    if arch in {'llava', 'llava_gemma'}:
        if bool(config.training.get('vision_tower_from_scratch', False)):
            log_for_0('Using CLIP vision tower from random initialization.')
        else:
            log_for_0(f'Loading CLIP vision tower: {config.model.vision_tower_str}')
            clip_params = load_clip_vision_params(
                config.model.vision_tower_str,
                cache_dir=config.training.get('hf_cache_dir', None),
                from_pt=bool(config.training.get('clip_from_pt', True)),
            )
            clip_params = jax.tree.map(to_fp32, clip_params)
            params['image_encoder'] = shard_cpu_tree_to_mesh(clip_params, mesh, state_spec.params['image_encoder'])
            log_for_0('Loaded pretrained CLIP vision tower.')
    else:
        assert config.training.get('siglip_from_scratch', False), (
            'Loading pretrained SigLip is not currently wired up; '
            'set training.siglip_from_scratch=True.'
        )
        log_for_0('Using image encoder from scratch.')

    pretrained_text_layers = _get_lm_pretrained_text_layers(config)
    if pretrained_text_layers is not None:
        raise NotImplementedError('Partial LM initialization is not implemented for jit/HSDP.')

    lm_variant = config.model.get('lm_checkpoint_variant', 'pt')
    log_for_0(f'Loading LM backbone: {config.model.lm_backbone_str} ({lm_variant})')
    gemma_path = _gemma_checkpoint_path(config.model.lm_backbone_str, lm_variant)
    gemma_params = gm.ckpts.load_params(gemma_path)
    if gemma_params is None:
        raise ValueError(f'{config.model.lm_backbone_str} checkpoint is empty!')
    gemma_params = jax.tree.map(to_fp32, gemma_params)
    if str(config.model.lm_backbone_str).startswith('gemma3_'):
        gemma_params = init_loc_token_embeddings(gemma_params)
        log_for_0('Initialized <loc0000>~<loc1023> embeddings with sinusoidal encoding.')
    params['lm_backbone'] = shard_cpu_tree_to_mesh(gemma_params, mesh, state_spec.params['lm_backbone'])
    del gemma_params

    state = state.replace(params=params)
    assert _state_step(state) == int(step_offset), f'Expected initial step {step_offset}, got {_state_step(state)}'
    return state


def _restore_params_only(state, params_source, zone, mesh, state_spec, current_step, config=None):
    if isinstance(params_source, str):
        params = restore_checkpoint_params(state.params, params_source, zone=zone)
    else:
        params = params_source
    if 'image_encoder' in params and isinstance(params['image_encoder'], dict) and 'decoder' in params['image_encoder']:
        params = copy.deepcopy(params)
        params['image_encoder'].pop('decoder')
    if config is not None and 'patch_size' in config.model:
        params = _interpolate_patch_pos_embed(
            params,
            target_image_size=int(config.dataset.image_size),
            patch_size=int(config.model.patch_size),
        )
    params = _maybe_shard_params(params, mesh, state_spec.params)
    state = state.replace(params=params)
    return _replace_state_step(state, current_step)


def _stage_overrides(config, stage_key):
    stage = config.training.get(stage_key, None)
    if stage is None:
        raise ValueError(f'Missing training.{stage_key} in curriculum config')
    return stage


def _build_curriculum_stage_config(config, stage_key, *, stage_start_step, stage_end_step, total_steps):
    stage = _stage_overrides(config, stage_key)
    phase_config = copy.deepcopy(config)
    stage_steps = int(stage_end_step) - int(stage_start_step)
    if stage_steps <= 0:
        raise ValueError(f'Invalid {stage_key} step range: {stage_start_step} -> {stage_end_step}')

    dataset_items = stage.get('dataset_items', stage.get('items', None))
    if dataset_items is not None:
        phase_config.dataset['items'] = copy.deepcopy(dataset_items)
    if stage.get('mix_weights', None) is not None:
        phase_config.dataset['mix_weights'] = copy.deepcopy(stage.mix_weights)
    if stage.get('dataset', None) is not None:
        for key, value in stage.dataset.items():
            target_key = 'max_txt_len' if key == 'max_txt_length' else key
            if target_key in {'items', 'dataset_items'}:
                phase_config.dataset['items'] = copy.deepcopy(value)
            elif target_key == 'mix_weights':
                phase_config.dataset['mix_weights'] = copy.deepcopy(value)
            else:
                if target_key not in phase_config.dataset:
                    raise ValueError(f'Unsupported stage dataset override: {key}')
                phase_config.dataset[target_key] = copy.deepcopy(value)
    if stage.get('model', None) is not None:
        phase_config.model.update(copy.deepcopy(stage.model))

    training_keys = [
        'batch_size', 'freeze_lm', 'freeze_lm_embed', 'freeze_lm_late',
        'freeze_image_encoder', 'vision_tower_from_scratch',
        'clip_from_pt', 'hf_cache_dir', 'optimizer', 'grad_clip_norm', 'log_per_step',
        'checkpoint_per_step', 'log_vis_per_step', 'sample_per_step', 'online_eval_per_step',
        'online_eval_tasks', 'final_eval_tasks', 'warmup_steps',
        'lr_schedule', 'seed', 'vision_encoder_learning_rate',
        'connector_learning_rate', 'projector_learning_rate',
        'exclude_bias_norm_from_weight_decay',
    ]
    for key in training_keys:
        if stage.get(key, None) is not None:
            phase_config.training[key] = copy.deepcopy(stage[key])
    for nested_key in ['adam', 'muon']:
        if stage.get(nested_key, None) is not None:
            phase_config.training[nested_key].update(copy.deepcopy(stage[nested_key]))
    if stage.get('sampling', None) is not None:
        phase_config.sampling.update(copy.deepcopy(stage.sampling))
    if stage.get('eval', None) is not None:
        phase_config.eval.update(copy.deepcopy(stage.eval))
    if stage.get('logging', None) is not None:
        phase_config.logging.update(copy.deepcopy(stage.logging))

    phase_config.training.num_steps = stage_steps
    phase_config.training.curriculum_stage_name = stage.get('name', stage_key)
    phase_config.training.curriculum_stage_key = stage_key
    phase_config.training.curriculum_stage_index = 1 if stage_key == 'stage1' else 2
    phase_config.training.curriculum_stage_start_step = int(stage_start_step)
    phase_config.training.curriculum_stage_end_step = int(stage_end_step)
    phase_config.training.curriculum_global_num_steps = int(total_steps)
    phase_config.finetune = False
    return phase_config


def _run_train_phase(
    *,
    config,
    workdir,
    writer,
    rng,
    zone,
    mesh_bundle,
    stage_key='train',
    stage_start_step=0,
    current_step=0,
    stage_end_step=None,
    restore_mode='fresh_pretrained',
    params_source=None,
    finetune_mode=False,
):
    stage_end_step = int(config.training.num_steps if stage_end_step is None else stage_end_step)
    stage_name = config.training.get('curriculum_stage_name', stage_key)
    local_step_offset = int(current_step) - int(stage_start_step)
    if local_step_offset < 0:
        raise ValueError(f'{stage_key}: current_step={current_step} < stage_start_step={stage_start_step}')

    resolve_dataset_roots(config, zone)
    local_batch_size = _validate_and_get_local_batch_size(config)
    train_loader, train_iter, tokenizer = _create_train_iterator(
        config,
        local_batch_size,
        local_step_offset,
        load_from=config.load_from if restore_mode == 'full' else None,
        zone=zone,
        checkpoint_step_for_state=current_step,
    )

    online_eval_tasks = list(config.training.get('online_eval_tasks', []) or [])
    final_eval_tasks = list(config.training.get('final_eval_tasks', []) or [])
    knn_data_dir = _prepare_knn_if_needed(config, zone, online_eval_tasks + final_eval_tasks)

    mesh, get_partition_spec, _, _, _ = mesh_bundle
    model = _create_model(config, finetune_mode=finetune_mode)
    state, normal_lr_fn, vision_lr_fn, connector_lr_fn = create_train_state(
        rng,
        config,
        model,
        mesh_bundle=mesh_bundle,
    )
    state_spec = get_partition_spec(state, MeshMode.MODEL)

    if restore_mode == 'fresh_pretrained':
        state = _load_initial_pretrained_params(state, config, mesh, state_spec, step_offset=0)
    elif restore_mode == 'full':
        state = restore_checkpoint(state, config.load_from, zone=zone)
        assert _state_step(state) == int(current_step), (
            f'Checkpoint step mismatch: expected {current_step}, restored {_state_step(state)}'
        )
        log_for_0(f'[{stage_name}] Full checkpoint loaded from {config.load_from}.')
    elif restore_mode == 'params_only':
        if params_source is None:
            params_source = config.load_from_pretrained or config.load_from
        state = _restore_params_only(
            state,
            params_source,
            zone,
            mesh,
            state_spec,
            int(current_step),
            config=config,
        )
        log_for_0(f'[{stage_name}] Params-only restore at global step {current_step}; optimizer reinitialized.')
    else:
        raise ValueError(f'Unknown restore_mode: {restore_mode}')

    assert _state_step(state) == int(current_step), f'Expected step {current_step}, got {_state_step(state)}'
    state_spec, batch_spec, p_train_step, p_sample_steps, p_sample_step_mmbench = _build_pjit_fns(
        config, model, state, mesh_bundle
    )
    del state_spec

    vis_pairs, vis_batch = _prepare_fixed_vis_batch(config, tokenizer)
    metrics_tracker = MetricsTracker()
    timer = Timer()
    timer.reset()

    log_for_0(
        '[%s] Starting jit/HSDP phase from global step %d to %d (local offset %d).',
        stage_name,
        current_step,
        stage_end_step,
        local_step_offset,
    )
    log_for_0(f'[{stage_name}] The initial training step may take a while....')

    for step in range(int(current_step), int(stage_end_step)):
        raw_batch = next(train_iter)
        batch = _prepare_host_batch(raw_batch)
        if step == int(current_step):
            log_for_0(f'[{stage_name}] first batch ready')
        global_batch = _make_global_batch(batch, batch_spec, mesh)
        state, metrics, all_debug = p_train_step(state, global_batch)
        if step == int(current_step):
            log_for_0(f'[{stage_name}] Train step compiled in {timer}.')

        metrics_tracker.update(metrics)
        local_step = step - int(stage_start_step)
        if (step + 1) % int(config.training.log_per_step) == 0:
            summary = metrics_tracker.finalize()
            summary['steps_per_second'] = config.training.log_per_step / timer.elapse_with_reset()
            summary['normal_lr'] = normal_lr_fn(local_step + 1)
            summary['vision_encoder_lr'] = vision_lr_fn(local_step + 1)
            summary['connector_lr'] = connector_lr_fn(local_step + 1)
            summary['siglip_lr'] = summary['vision_encoder_lr']
            summary['step'] = step + 1
            if 'curriculum_stage_index' in config.training:
                summary['curriculum_stage'] = int(config.training.curriculum_stage_index)
                summary['stage_step'] = local_step + 1
            writer.write_scalars(step + 1, summary)
            mu.sync_global_devices('log')

        log_vis_per_step = int(config.training.get('log_vis_per_step', -1))
        if log_vis_per_step > 0 and (
            step == int(current_step) or (step + 1) % log_vis_per_step == 0
        ):
            with timer.skip():
                pixels_src = raw_batch['pixel_values'][:16]
                pixels = pixels_src.numpy() if hasattr(pixels_src, 'numpy') else np.asarray(pixels_src)
                if pixels.ndim == 5:
                    pixels = pixels.reshape((-1,) + pixels.shape[2:])
                is_pt = pixels.ndim == 4 and pixels.shape[1] in (1, 3)
                img_grid = vis_util.make_grid_visualization(pixels, to_uint8=True, is_pt=is_pt)
                writer.write_images(step + 1, {f'{stage_key}_train_images': img_grid})
                ids_src = raw_batch['input_ids'][:16]
                ids_list = ids_src.numpy() if hasattr(ids_src, 'numpy') else np.asarray(ids_src)
                writer.write_texts(
                    step + 1,
                    f'{stage_key}_train_captions',
                    [tokenizer.decode(ids) for ids in ids_list],
                )

        sample_per_step = int(config.training.get('sample_per_step', -1))
        if sample_per_step > 0 and (
            step == int(current_step) or (step + 1) % sample_per_step == 0
        ):
            with timer.skip():
                out_strs = run_p_sample_step(
                    p_sample_steps['default'],
                    model,
                    tokenizer,
                    state.params,
                    vis_batch['pixel_values'],
                    vis_batch['input_ids'],
                    vis_batch['prefix_len'],
                )[:len(vis_pairs)]
                log_for_0(f'[{stage_name}] sample outputs: {out_strs}')
                writer.write_texts(step + 1, f'{stage_key}_vis_samples', out_strs)

        if config.model.recon_loss_weight > 0.0 and 'recon_imgs' in all_debug:
            with timer.skip():
                recon_imgs = np.asarray(jax.device_get(all_debug['recon_imgs']))
                orig_imgs = np.asarray(jax.device_get(all_debug['orig_imgs']))
                n_vis = orig_imgs.shape[0]
                combined = np.empty((n_vis * 2, *orig_imgs.shape[1:]), dtype=orig_imgs.dtype)
                combined[0::2] = orig_imgs
                combined[1::2] = recon_imgs
                recon_grid = vis_util.make_grid_visualization(combined, grid=n_vis, to_uint8=True, is_pt=False)
                writer.write_images(step + 1, {f'{stage_key}_recon_comparison': recon_grid})

        checkpoint_per_step = int(config.training.get('checkpoint_per_step', -1))
        if (
            (checkpoint_per_step > 0 and (step + 1) % checkpoint_per_step == 0)
            or (step + 1) == int(stage_end_step)
        ):
            with timer.skip():
                log_for_0(f'[{stage_name}] Saving checkpoint at global step {step + 1}...')
                host_state = mu.process_allgather(state, tiled=True)
                _save_training_checkpoint(
                    host_state,
                    train_loader,
                    config,
                    workdir,
                    step + 1,
                    local_batch_size,
                )
                del host_state
                mu.sync_global_devices('ckpt')

        online_eval_per_step = int(config.training.get('online_eval_per_step', -1))
        if online_eval_per_step > 0 and online_eval_tasks and (step + 1) % online_eval_per_step == 0:
            with timer.skip():
                run_eval_tasks(
                    state,
                    p_sample_steps,
                    online_eval_tasks,
                    step=step + 1,
                    run_p_sample_step=run_p_sample_step,
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    writer=writer,
                    p_sample_step_mmbench=p_sample_step_mmbench,
                    is_online_eval=True,
                    extra_args={
                        'knn_imagenet_data_dir': knn_data_dir,
                        'knn_imagenet_root': knn_data_dir,
                        'p_recon_steps': {},
                        'vis_batch': vis_batch,
                        'patch_size': _model_patch_size(config),
                        'image_size': config.dataset.image_size,
                        'run_eval_recon_psnr': run_eval_recon_psnr,
                    },
                )

    if final_eval_tasks:
        log_for_0(
            f'[{stage_name}] Running final evaluation with '
            f'beam_size={int(config.sampling.get("beam_size", 1))}...'
        )
        run_eval_tasks(
            state,
            p_sample_steps,
            final_eval_tasks,
            step=int(stage_end_step),
            run_p_sample_step=run_p_sample_step,
            model=model,
            tokenizer=tokenizer,
            config=config,
            writer=writer,
            p_sample_step_mmbench=p_sample_step_mmbench,
            task_suffix=f'_{stage_key}_final' if stage_key.startswith('stage') else '_final',
            extra_args={
                'knn_imagenet_data_dir': knn_data_dir,
                'knn_imagenet_root': knn_data_dir,
                'p_recon_steps': {},
                'vis_batch': vis_batch,
                'patch_size': _model_patch_size(config),
                'image_size': config.dataset.image_size,
                'run_eval_recon_psnr': run_eval_recon_psnr,
            },
        )

    if config.eval.get('mmbench_export_test', False):
        export_mmbench_test_xlsx(p_sample_step_mmbench, run_p_sample_step, model, tokenizer, state.params, config)

    log_for_0(f'[{stage_name}] Phase done.')
    return state


def _init_run(config, workdir):
    config.workdir_hash = md5(workdir.encode()).hexdigest()[:8]
    zone = infer_zone_card(config, workdir)
    assert zone in ['us-central1', 'us-east5', 'asia-northeast1-b'], (
        'We only support us-central1, us-east5 and asia-northeast1-b for now.'
    )
    config.zone = zone
    log_for_0(config)
    mesh_bundle = prepare_pjit_funcs(getattr(config, 'sharding', 'hsdp'))
    writer = Writer(config, workdir, use_wandb=config.logging.use_wandb, use_tb=False)
    return zone, mesh_bundle, writer


def _single_stage_train(config, workdir, *, finetune_mode=False):
    rng = random.PRNGKey(config.training.seed)
    zone, mesh_bundle, writer = _init_run(config, workdir)
    current_step = checkpoint_step(config.load_from, zone=zone) if config.load_from else 0
    final_step = int(config.training.num_steps)
    if config.load_from:
        restore_mode = 'full'
        params_source = None
    elif config.load_from_pretrained:
        restore_mode = 'params_only'
        params_source = config.load_from_pretrained
    else:
        restore_mode = 'fresh_pretrained'
        params_source = None
    state = _run_train_phase(
        config=config,
        workdir=workdir,
        writer=writer,
        rng=rng,
        zone=zone,
        mesh_bundle=mesh_bundle,
        stage_key='train',
        stage_start_step=0,
        current_step=current_step,
        stage_end_step=final_step,
        restore_mode=restore_mode,
        params_source=params_source,
        finetune_mode=finetune_mode,
    )
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    if bool(config.training.get('copy_final_checkpoint_to_pretrained', True)):
        final_checkpoint_source = workdir if current_step < final_step else config.load_from
        if final_checkpoint_source:
            copy_latest_checkpoint_to_pretrained(final_checkpoint_source, zone=zone)
            mu.sync_global_devices('pretrained_ckpt')
        else:
            log_for_0('Skipping durable checkpoint copy; no checkpoint was produced or loaded.')
    return state


def _train_llava_curriculum(config: ml_collections.ConfigDict, workdir: str):
    assert not config.finetune, 'Curriculum training expects finetune=False'
    stage1_steps = int(config.training.get('stage1_steps', 0))
    stage2_steps = int(config.training.get('stage2_steps', 0))
    if stage1_steps <= 0 or stage2_steps <= 0:
        raise ValueError('Curriculum config requires positive training.stage1_steps and stage2_steps')
    total_steps = stage1_steps + stage2_steps
    if int(config.training.num_steps) != total_steps:
        log_for_0(
            f'Curriculum overriding training.num_steps={config.training.num_steps} '
            f'to stage1_steps + stage2_steps = {total_steps}'
        )
        config.training.num_steps = total_steps

    rng = random.PRNGKey(config.training.seed)
    zone, mesh_bundle, writer = _init_run(config, workdir)
    current_step = checkpoint_step(config.load_from, zone=zone) if config.load_from else 0
    initial_step = current_step
    if current_step > total_steps:
        raise ValueError(f'Checkpoint step {current_step} is beyond curriculum total steps {total_steps}')

    params_for_next = None
    state = None
    if current_step < stage1_steps:
        stage1_config = _build_curriculum_stage_config(
            config,
            'stage1',
            stage_start_step=0,
            stage_end_step=stage1_steps,
            total_steps=total_steps,
        )
        stage1_config.load_from = config.load_from
        state = _run_train_phase(
            config=stage1_config,
            workdir=workdir,
            writer=writer,
            rng=rng,
            zone=zone,
            mesh_bundle=mesh_bundle,
            stage_key='stage1',
            stage_start_step=0,
            current_step=current_step,
            stage_end_step=stage1_steps,
            restore_mode='full' if config.load_from else 'fresh_pretrained',
        )
        current_step = stage1_steps
        if current_step < total_steps:
            params_for_next = state.params
            del state
            mu.sync_global_devices('curriculum_stage1_to_stage2')

    if current_step < total_steps:
        stage2_config = _build_curriculum_stage_config(
            config,
            'stage2',
            stage_start_step=stage1_steps,
            stage_end_step=total_steps,
            total_steps=total_steps,
        )
        stage2_config.load_from = config.load_from
        state = _run_train_phase(
            config=stage2_config,
            workdir=workdir,
            writer=writer,
            rng=random.fold_in(rng, 2),
            zone=zone,
            mesh_bundle=mesh_bundle,
            stage_key='stage2',
            stage_start_step=stage1_steps,
            current_step=current_step,
            stage_end_step=total_steps,
            restore_mode='params_only' if current_step == stage1_steps else 'full',
            params_source=params_for_next,
        )
    else:
        log_for_0(f'Curriculum checkpoint already reaches total_steps={total_steps}.')
        final_stage2_config = _build_curriculum_stage_config(
            config,
            'stage2',
            stage_start_step=stage1_steps,
            stage_end_step=total_steps,
            total_steps=total_steps,
        )
        final_stage2_config.load_from = config.load_from
        if config.load_from and list(final_stage2_config.training.get('final_eval_tasks', []) or []):
            state = _run_train_phase(
                config=final_stage2_config,
                workdir=workdir,
                writer=writer,
                rng=random.fold_in(rng, 2),
                zone=zone,
                mesh_bundle=mesh_bundle,
                stage_key='stage2',
                stage_start_step=stage1_steps,
                current_step=current_step,
                stage_end_step=total_steps,
                restore_mode='full',
            )

    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    if bool(config.training.get('copy_final_checkpoint_to_pretrained', True)):
        final_checkpoint_source = workdir if initial_step < total_steps else config.load_from
        if final_checkpoint_source:
            copy_latest_checkpoint_to_pretrained(final_checkpoint_source, zone=zone)
            mu.sync_global_devices('pretrained_ckpt')
        else:
            log_for_0('Skipping durable checkpoint copy; no checkpoint was produced or loaded.')
    return state


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    if str(config.training.get('curriculum', '')).lower() == 'llava15_two_stage':
        return _train_llava_curriculum(config, workdir)
    assert not config.finetune, 'train_and_evaluate expects finetune=False'
    return _single_stage_train(config, workdir, finetune_mode=False)


def just_evaluate(config: ml_collections.ConfigDict, workdir: str):
    config.workdir_hash = md5(workdir.encode()).hexdigest()[:8]
    rng = random.PRNGKey(config.training.seed)
    zone, mesh_bundle, writer = _init_run(config, workdir)
    resolve_dataset_roots(config, zone)
    tokenizer = create_tokenizer(config.model.lm_backbone_str)
    model = _create_model(config)
    state, _, _, _ = create_train_state(rng, config, model, mesh_bundle=mesh_bundle)
    if config.load_from:
        state = restore_checkpoint(state, config.load_from, zone=zone)
    else:
        mesh, get_partition_spec, _, _, _ = mesh_bundle
        state_spec = get_partition_spec(state, MeshMode.MODEL)
        log_for_0('Eval-only run has no load_from; initializing pretrained params.')
        state = _load_initial_pretrained_params(state, config, mesh, state_spec, step_offset=0)
    step = _state_step(state)
    state_spec, batch_spec, _, p_sample_steps, p_sample_step_mmbench = _build_pjit_fns(config, model, state, mesh_bundle)
    del state_spec, batch_spec

    final_eval_tasks = list(config.training.get('final_eval_tasks', []) or [])
    knn_data_dir = _prepare_knn_if_needed(config, zone, final_eval_tasks)
    if final_eval_tasks:
        run_eval_tasks(
            state,
            p_sample_steps,
            final_eval_tasks,
            step=step,
            run_p_sample_step=run_p_sample_step,
            model=model,
            tokenizer=tokenizer,
            config=config,
            writer=writer,
            p_sample_step_mmbench=p_sample_step_mmbench,
            task_suffix='_final',
            extra_args={'knn_imagenet_data_dir': knn_data_dir, 'knn_imagenet_root': knn_data_dir},
        )
    if config.eval.get('mmbench_export_test', False):
        export_mmbench_test_xlsx(p_sample_step_mmbench, run_p_sample_step, model, tokenizer, state.params, config)
    log_for_0('Eval Over.')
    return 'Eval Over.'


def finetune(config: ml_collections.ConfigDict, workdir: str):
    assert getattr(config, 'finetune', False), 'Expected finetune=True in config'
    assert config.model.recon_loss_weight == 0.0, 'Expected recon_loss_weight == 0.0 in config'
    assert config.load_from_pretrained or config.load_from
    return _single_stage_train(config, workdir, finetune_mode=True)
