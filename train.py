from absl import logging as absl_logging
from functools import partial
import jax
import jax.numpy as jnp
import jax.experimental.multihost_utils as mu
from flax import jax_utils
import numpy as np

import warnings
warnings.filterwarnings("ignore", message=".*EOF occurred in violation of protocol.*")

import ml_collections
import optax
from jax import lax, random
from gemma import gm

import input_pipeline
from utils.data_util import resolve_dataset_roots
from models.paligemma_enc_dec import PaliGemmaEncDec
from models.siglip_enc_dec import patchify, unpatchify
from models.gemma import load_LM
from utils.logging_util import MetricsTracker, Timer, log_for_0, Writer
from utils.trainstate_util import create_train_state
from utils.ckpt_util import checkpoint_step, infer_zone_card, save_checkpoint, restore_checkpoint
from utils import vis_util
from utils.llm_util import create_tokenizer, init_loc_token_embeddings
from utils.frozen_util import get_trainable, merge_params
from hashlib import md5
from evals.eval_mme import collate_fn
from evals.eval_mmbench import export_mmbench_test_xlsx
from evals.eval_imagenet_knn import ensure_imagenet_available
from evals.eval import run_eval_tasks
from PIL import Image

# JAX consts
LDC = jax.local_device_count()
PRC = jax.process_count()
PRI = jax.process_index()
GDC = jax.device_count() # global device count = LDC * PRC

assert GDC == LDC * PRC, f"{GDC} != {LDC} * {PRC}"

# absl verbosity
absl_logging.set_verbosity(absl_logging.INFO)

def compute_metrics(dict_losses):
    metrics = {k: jnp.mean(v) for k, v in dict_losses.items()}
    metrics = lax.pmean(metrics, axis_name="batch")
    return metrics

def train_step(state, batch, rng_init, config):
    """
    Perform a single training step.
    """
    rng_step = random.fold_in(rng_init, state.step)

    assert batch['pixel_values'].shape[1:] == (config.dataset.image_size, config.dataset.image_size, 3), f"Unexpected image shape {batch['pixel_values'].shape}"

  
    freeze_lm = bool(config.training.get('freeze_lm', False))
    txt_feature_layer = int(config.model.get('txt_feature_layer', 0))
    trainable_params, frozen_params = get_trainable(
        state.params, freeze_lm=freeze_lm, txt_feature_layer=txt_feature_layer
    )

    def loss_fn(wrt_params):
        """loss function used for training."""
        params = merge_params(wrt_params, frozen_params) if frozen_params else wrt_params
        variables = {
            "params": params,
        }
        outputs = state.apply_fn(
            variables,
            input_ids=batch['input_ids'],
            images=batch['pixel_values'],
            prefix_len=batch['prefix_len'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            mask_token_category_probs=batch.get('mask_token_category_probs', None),
            rngs=dict(gen=rng_step),
        )
        loss, log_dict, all_debug = outputs
        return loss, (log_dict, all_debug)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    aux, grads = grad_fn(trainable_params)
    loss, (log_dict, all_debug) = aux

    grads = lax.pmean(grads, axis_name="batch")

    if frozen_params:
        frozen_zero_grads = jax.tree.map(jnp.zeros_like, frozen_params)
        grads = merge_params(grads, frozen_zero_grads)
    
    grad_norm = optax.global_norm(grads)

    metrics = compute_metrics(log_dict)
    metrics["grad_norm"] = grad_norm
    metrics["loss"] = lax.pmean(loss, axis_name="batch")

    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics, all_debug

def sample_step(params, images, prompt_ids, model: PaliGemmaEncDec, max_new_tokens=64, beam_size=1, prefix_len=None):
    if beam_size > 1:
        output = model.apply({'params': params}, prompt_ids, prefix_len, images, method=model.generate_beam_search, beam_size=beam_size, max_new_tokens=max_new_tokens)
    else:
        output = model.apply({'params': params}, prompt_ids, prefix_len, images, method=model.generate, max_new_tokens=max_new_tokens)
    return output


def make_mmbench_sample_step(model, config):
    return jax.pmap(
        partial(
            sample_step,
            model=model,
            max_new_tokens=int(getattr(config.eval, "mmbench_max_new_tokens", 8)),
            beam_size=1,
        ),
        axis_name="batch",
    )

def run_p_sample_step(p_sample_step, model, tokenizer, params, images, prompt_ids, prefix_len=None):

    output = p_sample_step(params, images, prompt_ids, prefix_len=prefix_len)  # shape (LDC, B, T)
    output = output.reshape(-1, output.shape[2]) # shape (LDC * B, T)
    output = jax.device_get(output)

    def post_process(token_ids):
        # token_ids: (T,)
        # 找到第一个 eos 的索引
        indices = np.where(token_ids == tokenizer.special_tokens.EOS)[0]
        if len(indices) > 0:
            # 截取到第一个 eos 
            token_ids = token_ids[:indices[0]]
        return token_ids.tolist()
    
    token_ids = [post_process(o) for o in output]
    # batch decode
    output_strs = [tokenizer.decode(token_id) for token_id in token_ids]

    return output_strs

def recon_step(params, images, model: PaliGemmaEncDec, num_visible: int):
    """Reconstruct images using the first num_visible encoder tokens.

    Designed to be pmapped – each device processes its local shard.

    Args:
        params:      model parameters (unsharded on each device after pmap)
        images:      (B, H, W, 3) images for this device
        model:       PaliGemmaEncDec instance (static via partial)
        num_visible: number of encoder tokens to reveal (Python int, static)

    Returns:
        (B, T, P²×3) patch predictions in the same normalised pixel space
    """
    return model.apply({'params': params}, images, num_visible, method=model.reconstruct)


def run_eval_recon_psnr(p_recon_steps, state_params, eval_images, patch_size, image_size, n_vis=6):
    """Evaluate reconstruction PSNR for each K in p_recon_steps.

    PSNR is computed as  10 * log10(1 / MSE)  with the pixel space being
    whatever normalisation the model uses (typically [-1, 1]).  Per-image PSNR
    values are averaged across all eval images.

    Args:
        p_recon_steps: dict  {num_visible (int) -> pmapped recon_step fn}
        state_params:  replicated model params  (LDC, ...)
        eval_images:   (LDC, B_per_device, H, W, 3) – already sharded for pmap
        patch_size:    patch size P used by the model
        image_size:    spatial resolution H (= W) used by the model
        n_vis:         how many images to keep for visualization per K

    Returns:
        psnr_results:  dict  {num_visible (int) -> mean_psnr (float, dB)}
        recon_vis:     dict  {num_visible (int) -> (n_vis, H, W, 3) numpy array}
    """
    # Flat images on CPU for computing patch targets once
    flat_images = jax.device_get(
        eval_images.reshape(-1, *eval_images.shape[2:])
    )  # (N, H, W, 3)
    patch_targets = patchify(flat_images, patch_size)  # (N, T, P²×3)

    psnr_results = {}
    recon_vis = {}
    for k, p_recon_k in p_recon_steps.items():
        pixel_pred = p_recon_k(state_params, eval_images)   # (LDC, B_local, T, P²×3)
        pixel_pred = jax.device_get(
            pixel_pred.reshape(-1, pixel_pred.shape[-2], pixel_pred.shape[-1])
        )  # (N, T, P²×3)

        # Per-image MSE → per-image PSNR → mean (standard convention)
        mse_per_image = np.mean(
            (pixel_pred - patch_targets) ** 2, axis=(1, 2)
        )  # (N,)
        psnr_per_image = 10.0 * np.log10(1.0 / (mse_per_image + 1e-10))
        psnr_results[k] = float(np.mean(psnr_per_image))

        # Keep first n_vis reconstructions for visualization
        recon_vis[k] = unpatchify(pixel_pred[:n_vis], patch_size, image_size)  # (n_vis, H, W, 3)

    return psnr_results, recon_vis


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
    'COCO_val2014_000000436141.jpg': "A clean bathroom is seen in this image."
}


def _partial_init_lm_from_pretrained(random_lm_params, pretrained_lm_params, num_text_layers: int):
    """Load only Gemma text-feature params; keep remaining LM params random."""
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


def _create_train_iterator(config, local_batch_size, step_offset):
    data_seed_offset = int(getattr(config.dataset, "data_seed_offset", 0)) + int(step_offset)
    log_for_0(
        f'Creating train loader with batch size: {local_batch_size}, '
        f'data_seed_offset={data_seed_offset}'
    )
    train_loader, tokenizer = input_pipeline.create_split(
        config,
        local_batch_size,
        data_seed_offset=data_seed_offset,
    )
    train_iter = iter(train_loader)
    log_for_0(f'Train loader iterator created. Batch size: {local_batch_size}')
    return train_loader, train_iter, tokenizer


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
):
    assert not config.finetune, 'Finetune mode is not supported for now!!!'
    config.workdir_hash = md5(workdir.encode()).hexdigest()[:8]
    log_for_0(config)
    rng = random.PRNGKey(config.training.seed)
    # tpu_type = jax.local_devices()[0].device_kind
    zone = infer_zone_card(config, workdir)
    assert zone in ['us-central1', 'us-east5', 'asia-northeast1-b'], f'We only support us-central1 and us-east5 and asia-northeast1-b for now!!!'
    config.zone = zone

    resolve_dataset_roots(config, zone)

    # Download ImageNet to tmpfs if any KNN eval is requested.
    # This is a one-time distributed operation; subsequent calls are instant.
    _knn_imagenet_root = None
    online_eval_tasks = list(config.training.get("online_eval_tasks", []) or [])
    final_eval_tasks = list(config.training.get("final_eval_tasks", []) or [])
    _need_knn = any(t in {"knn_partial", "knn_full"} for t in (online_eval_tasks + final_eval_tasks))
    if _need_knn:
        log_for_0('[KNN] Preparing ImageNet dataset …')
        _knn_imagenet_root = ensure_imagenet_available(
            zone, local_debug=config.local_debug
        )
        log_for_0(f'[KNN] ImageNet root: {_knn_imagenet_root}')

    batch_size = config.training.batch_size
    if batch_size % PRC > 0:
        raise ValueError('Batch size must be divisible by the number of processes')
    local_batch_size = batch_size // PRC
    if local_batch_size % LDC > 0:
        raise ValueError('Local batch size must be divisible by the number of local devices')

    if config.load_from:
        assert zone is not None, "Cannot infer zone from workdir."
        step_offset = checkpoint_step(config.load_from, zone=zone)
    else:
        step_offset = 0
    train_loader, train_iter, tokenizer = _create_train_iterator(config, local_batch_size, step_offset)

    ################## Create Model ##################
    model = PaliGemmaEncDec(
        **config.model,
        image_size=config.dataset.image_size,
    )

    state, normal_lr_fn, siglip_lr_fn = create_train_state(rng, config, model)
    if config.load_from:
        assert zone is not None, "Cannot infer zone from workdir."
        state = restore_checkpoint(
            state, 
            config.load_from,
            zone=zone
        )
        assert int(state.step) == step_offset, (
            f'Checkpoint step mismatch: inferred {step_offset}, restored {int(state.step)}'
        )
        log_for_0(f'Checkpoint loaded from {config.load_from}. Current step: {int(state.step)}')
    else:
        # Keep pretrained weights in fp32 to avoid tiny updates being lost in bf16.
        to_fp32 = lambda x: x.astype(jnp.float32) if hasattr(x, "astype") else x

        assert config.training.get('siglip_from_scratch', False), (
            "Loading pretrained SigLip is not currently wired up; "
            "set training.siglip_from_scratch=True (image_encoder trains from scratch)."
        )
        log_for_0('Using image encoder from scratch.')

        if config.model.lm_backbone_str == 'gemma2_2B':
            log_for_0('Loading Gemma2...')
            gemma_path = gm.ckpts.CheckpointPath.GEMMA2_2B_PT
        elif config.model.lm_backbone_str == 'gemma3_270M':
            log_for_0('Loading Gemma3...')
            gemma_path = gm.ckpts.CheckpointPath.GEMMA3_270M_PT
        else:
            raise ValueError(f'Unsupported LM backbone: {config.model.lm_backbone_str}')

        gemma_params = gm.ckpts.load_params(gemma_path)
        if gemma_params is None:
            raise ValueError(f'{config.model.lm_backbone_str} checkpoint is empty!')
        gemma_params = jax.tree.map(to_fp32, gemma_params)

        if config.model.lm_backbone_str == 'gemma3_270M':
            gemma_params = init_loc_token_embeddings(gemma_params)
            log_for_0('Initialized <loc0000>~<loc1023> embeddings with sinusoidal encoding.')

        pretrained_text_layers = _get_lm_pretrained_text_layers(config)
        if pretrained_text_layers is None:
            state.params['lm_backbone'] = gemma_params
            log_for_0('Loaded full pretrained LM backbone.')
        else:
            state.params['lm_backbone'] = _partial_init_lm_from_pretrained(
                state.params['lm_backbone'],
                gemma_params,
                pretrained_text_layers,
            )
            total_layers = sum(1 for k in state.params['lm_backbone'] if k.startswith('layer_'))
            loaded_layers = (
                "no transformer layers"
                if pretrained_text_layers == 0
                else f"layers 0..{pretrained_text_layers - 1}"
            )
            scratch_layers = (
                "no transformer layers"
                if pretrained_text_layers >= total_layers
                else f"layers {pretrained_text_layers}..{total_layers - 1}"
            )
            log_for_0(
                f'Loaded pretrained LM embedder + {loaded_layers}; '
                f'kept {scratch_layers} and final_norm from scratch.'
            )
        del gemma_params
        assert int(state.step) == step_offset, (
            f'Expected initial step {step_offset}, got {int(state.step)}'
        )

    state = jax_utils.replicate(state)

    p_train_step = jax.pmap(
        partial(
            train_step,
            rng_init=rng,
            config=config,
        ),
        axis_name="batch",
        donate_argnums=(0, 1),
    )
    p_sample_step = jax.pmap(
        partial(
            sample_step,
            model=model,
            **config.sampling.to_dict()
        ),
        axis_name="batch",
    )
    p_sample_step_mmbench = make_mmbench_sample_step(model, config)

    # Build one pmapped recon step per requested token count.
    # Only meaningful when the model has a decoder (recon_loss_weight > 0).
    eval_recon_tokens = list(config.eval.get('eval_recon_tokens', []))
    p_recon_steps = {}
    if eval_recon_tokens and config.model.recon_loss_weight > 0.0:
        for _k in eval_recon_tokens:
            p_recon_steps[_k] = jax.pmap(
                partial(recon_step, model=model, num_visible=_k),
                axis_name="batch",
            )
        log_for_0(f'Built p_recon_steps for K={eval_recon_tokens}')

    writer = Writer(config, workdir, use_wandb=config.logging.use_wandb, use_tb=False)
    metrics_tracker = MetricsTracker()
    timer = Timer()
    timer.reset()

    log_for_0(f'Preparing sample pairs for sampling...')
    vis_pairs = [input_pipeline.preprocess_fn({
        'jpg': Image.open(f'/kmh-nfs-ssd-us-mount/code/hanhong/shared/COCO/val2014/{k}'),
        'aux': {'gt': v},
    }, transform=input_pipeline.get_transforms(
        config.dataset.image_size,
        is_train=False,
        resize_mode=getattr(config.dataset, "resize_mode", "letterbox"),
    ), tokenizer=tokenizer, max_len=config.dataset.max_txt_len) for k, v in FIXED_PAIRS.items()]
    # padded_size is the first multiple of LDC that is greater than or equal to len(vis_pairs)
    padded_size = ((len(vis_pairs) + LDC - 1) // LDC) * LDC
    vis_batch = input_pipeline.prepare_batch_data(collate_fn(vis_pairs), batch_size=padded_size)

    log_for_0(f'Starting training from step {step_offset} to {config.training.num_steps}...')
    log_for_0(f'The initial training step may take a while....')

    for step in range(step_offset, config.training.num_steps):
        batch = next(train_iter)

        ########### Train ###########
        raw_batch = {k: v for k, v in batch.items()} # keep a copy of raw batch for visualization
        batch = input_pipeline.prepare_batch_data(batch)
        if step == step_offset:
            log_for_0(f'first batch ready')
        state, metrics, all_debug = p_train_step(state, batch)
        if step == step_offset:
            log_for_0(f'Train step compiled in {timer}.')

        ########### Metrics ###########
        metrics_tracker.update(metrics)  # stream one step in
        if (step+1) % config.training.log_per_step == 0:
            summary = metrics_tracker.finalize()
            summary['steps_per_second'] = config.training.log_per_step / timer.elapse_with_reset()
            summary['normal_lr'] = normal_lr_fn(step + 1)
            summary['siglip_lr'] = siglip_lr_fn(step + 1)
            summary['step'] = step + 1
            writer.write_scalars(step + 1, summary)
            mu.sync_global_devices('log')

        # visualize training images
        if step == 0 or (config.training.log_vis_per_step > 0 and (step + 1) % config.training.log_vis_per_step == 0):
            with timer.skip():
                log_for_0("Logging visualization at step {}...".format(step))
                # log some training images
                pixels = raw_batch['pixel_values'][:16].numpy()
                # print pixels range (max / min /mean /std)
                img_grid = vis_util.make_grid_visualization(pixels, to_uint8=True, is_pt=True) # (H, W, C)
                writer.write_images(step + 1, {"train_images": img_grid})
                writer.write_texts(step + 1, "train_captions", [tokenizer.decode(ids) for ids in raw_batch['input_ids'][:16]])
                log_for_0("Visualization logged.")

        if step == 0 or (config.training.sample_per_step > 0 and (step + 1) % config.training.sample_per_step == 0):
        # if True:
            with timer.skip():
                log_for_0("Sampling at step {}...".format(step))
                input_ids = vis_batch["input_ids"]
                prefix_len = vis_batch["prefix_len"]
                out_strs = run_p_sample_step(
                    p_sample_step,
                    model,
                    tokenizer,
                    state.params,
                    vis_batch['pixel_values'],
                    input_ids,
                    prefix_len
                )
                out_strs = out_strs[:len(vis_pairs)]
                log_for_0(f'sample outputs: {out_strs}')
                writer.write_texts(step + 1, "vis_samples", out_strs)
                log_for_0("Sample finished.")

        visualize_recon_per_step = config.training.get('visualize_recon_per_step', 0)
        if config.model.recon_loss_weight > 0.0 and 'recon_imgs' in all_debug:
            if step == 0 or (visualize_recon_per_step > 0 and (step + 1) % visualize_recon_per_step == 0):
                with timer.skip():
                    log_for_0("Logging recon visualization at step {}...".format(step))
                    # all_debug['recon_imgs']: (LDC, ≤6, H, W, 3), normalised [-1, 1]
                    recon_imgs = jax.device_get(all_debug['recon_imgs'][0])  # (n, H, W, 3)
                    orig_imgs  = jax.device_get(all_debug['orig_imgs'][0])   # (n, H, W, 3)
                    # Interleave orig and recon: [orig0, recon0, orig1, recon1, ...]
                    # → make_grid_visualization with grid=n gives a 2-row layout:
                    #   top row = orig,  bottom row = recon  (each column is one pair)
                    n_vis = orig_imgs.shape[0]
                    combined = np.empty((n_vis * 2, *orig_imgs.shape[1:]), dtype=orig_imgs.dtype)
                    combined[0::2] = orig_imgs
                    combined[1::2] = recon_imgs
                    recon_grid = vis_util.make_grid_visualization(combined, grid=n_vis, to_uint8=True, is_pt=False)
                    writer.write_images(step + 1, {"recon_comparison": recon_grid})
                    log_for_0("Recon visualization logged.")

        ########### Save Checkpoint ###########
        if (step + 1) % config.training.checkpoint_per_step == 0 \
            or (step + 1) == config.training.num_steps:
            with timer.skip():
                log_for_0("Saving checkpoint at step {}...".format(step))
                save_checkpoint(state, workdir)
                mu.sync_global_devices('ckpt')

        # if step == 0 or (config.training.cider_per_step > 0 and (step + 1) % config.training.cider_per_step == 0):
        # # if config.training.cider_per_step > 0 and (step + 1) % config.training.cider_per_step == 0:
        # # if True:
        #     with timer.skip():
        #         log_for_0("Evaluating 醋 at step {}...".format(step))
        #         acc, sample_outputs, sample_images = eval_cider(p_sample_step, run_p_sample_step, model, tokenizer, state.params, config)
        #         log_for_0(f'Current CIDEr score: {acc}')
        #         log_for_0(f'sample outputs: {sample_outputs}')
        #         writer.write_scalars(step + 1, {"cider": acc, "step": step + 1})
        #         writer.write_texts(step + 1, "samples", sample_outputs)
        #         writer.write_images(step + 1, {"sample_images": vis_util.make_grid_visualization(np.stack(sample_images[:16]), to_uint8=True, is_pt=True)})
        #             # write sample_outputs to wandb
        #         log_for_0("Evaluation finished.")

        online_eval_per_step = int(config.training.get("online_eval_per_step", -1))
        online_eval_tasks = list(config.training.get("online_eval_tasks", []) or [])
        if online_eval_per_step > 0 and online_eval_tasks and (step + 1) % online_eval_per_step == 0:
            with timer.skip():
                run_eval_tasks(
                    state,
                    p_sample_step,
                    online_eval_tasks,
                    step=step + 1,
                    run_p_sample_step=run_p_sample_step,
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    writer=writer,
                    p_sample_step_mmbench=p_sample_step_mmbench,
                    extra_args={
                        "knn_imagenet_root": _knn_imagenet_root,
                        "p_recon_steps": p_recon_steps,
                        "vis_batch": vis_batch,
                        "patch_size": config.model.patch_size,
                        "image_size": config.dataset.image_size,
                        "run_eval_recon_psnr": run_eval_recon_psnr,
                    },
                )

    ########### Final ImageNet KNN eval ###########
    final_step = config.training.num_steps

    ########### Final eval with beam_size=1 ###########
    log_for_0("Running final evaluation with beam_size=1...")

    final_eval_tasks = list(config.training.get("final_eval_tasks", []) or [])
    if final_eval_tasks:
        run_eval_tasks(
            state,
            p_sample_step,
            final_eval_tasks,
            step=final_step,
            run_p_sample_step=run_p_sample_step,
            model=model,
            tokenizer=tokenizer,
            config=config,
            writer=writer,
            p_sample_step_mmbench=p_sample_step_mmbench,
            task_suffix="_final",
            extra_args={
                "knn_imagenet_root": _knn_imagenet_root,
                "p_recon_steps": p_recon_steps,
                "vis_batch": vis_batch,
                "patch_size": config.model.patch_size,
                "image_size": config.dataset.image_size,
                "run_eval_recon_psnr": run_eval_recon_psnr,
            },
        )

    if config.eval.get("mmbench_export_test", False):
        log_for_0("Exporting MMBench TEST EN xlsx...")
        export_mmbench_test_xlsx(p_sample_step_mmbench, run_p_sample_step, model, tokenizer, state.params, config)

    log_for_0("Final beam=1 evaluation done.")

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state

def just_evaluate(config: ml_collections.ConfigDict, workdir: str):
    config.workdir_hash = md5(workdir.encode()).hexdigest()[:8]
    log_for_0(config)
    rng = random.PRNGKey(config.training.seed)
    # tpu_type = jax.local_devices()[0].device_kind
    zone = infer_zone_card(config, workdir)
    assert zone in ['us-central1', 'us-east5', 'asia-northeast1-b'], f'We only support us-central1 and us-east5 and asia-northeast1-b for now!!!'
    config.zone = zone

    resolve_dataset_roots(config, zone)

    # Download ImageNet to tmpfs if any KNN eval is requested.
    _knn_imagenet_root = None
    final_eval_tasks = list(config.training.get("final_eval_tasks", []) or [])
    _need_knn = any(t in {"knn_partial", "knn_full"} for t in final_eval_tasks)
    if _need_knn:
        log_for_0('[KNN] Preparing ImageNet dataset …')
        _knn_imagenet_root = ensure_imagenet_available(
            zone, local_debug=config.local_debug
        )
        log_for_0(f'[KNN] ImageNet root: {_knn_imagenet_root}')

    tokenizer = create_tokenizer(config.model.lm_backbone_str)

    ################## Create Model ##################
    model = PaliGemmaEncDec(
        image_size=config.dataset.image_size,
        **config.model
    )
    state, normal_lr_fn, siglip_lr_fn = create_train_state(rng, config, model)
    assert config.load_from
    assert zone is not None, "Cannot infer zone from workdir."
    state = restore_checkpoint(
        state, 
        config.load_from,
        zone=zone
    )
    log_for_0(f'Checkpoint loaded from {config.load_from}. Current step: {int(state.step)}')

    step = int(state.step)
    state = jax_utils.replicate(state)

    p_sample_step = jax.pmap(
        partial(
            sample_step,
            model=model,
            **config.sampling.to_dict()
        ),
        axis_name="batch",
    )
    p_sample_step_mmbench = make_mmbench_sample_step(model, config)

    writer = Writer(config, workdir, use_wandb=config.logging.use_wandb, use_tb=False)
    metrics_tracker = MetricsTracker()
    timer = Timer()
    timer.reset()

    final_eval_tasks = list(config.training.get("final_eval_tasks", []) or [])
    if final_eval_tasks:
        run_eval_tasks(
            state,
            p_sample_step,
            final_eval_tasks,
            step=step,
            run_p_sample_step=run_p_sample_step,
            model=model,
            tokenizer=tokenizer,
            config=config,
            writer=writer,
            p_sample_step_mmbench=p_sample_step_mmbench,
            task_suffix="_final",
            extra_args={
                "knn_imagenet_root": _knn_imagenet_root,
            },
        )

    if config.eval.get("mmbench_export_test", False):
        log_for_0("Exporting MMBench TEST EN xlsx...")
        export_mmbench_test_xlsx(p_sample_step_mmbench, run_p_sample_step, model, tokenizer, state.params, config)

    log_for_0("Eval Over.")
    return "Eval Over."

def finetune(config: ml_collections.ConfigDict, workdir: str):
    assert getattr(config, 'finetune', False), 'Expected finetune=True in config'
    config.workdir_hash = md5(workdir.encode()).hexdigest()[:8]
    log_for_0(config)
    rng = random.PRNGKey(config.training.seed)
    zone = infer_zone_card(config, workdir)
    assert zone in ['us-central1', 'us-east5', 'asia-northeast1-b'], f'We only support us-central1 and us-east5 and asia-northeast1-b for now!!!'
    config.zone = zone

    assert config.model.recon_loss_weight == 0.0, 'Expected recon_loss_weight == 0.0 in config'

    resolve_dataset_roots(config, zone)

    # Download ImageNet to tmpfs if any KNN eval is requested.
    _knn_imagenet_root = None
    online_eval_tasks = list(config.training.get("online_eval_tasks", []) or [])
    final_eval_tasks = list(config.training.get("final_eval_tasks", []) or [])
    _need_knn = any(t in {"knn_partial", "knn_full"} for t in (online_eval_tasks + final_eval_tasks))
    if _need_knn:
        log_for_0('[KNN] Preparing ImageNet dataset …')
        _knn_imagenet_root = ensure_imagenet_available(
            zone, local_debug=config.local_debug
        )
        log_for_0(f'[KNN] ImageNet root: {_knn_imagenet_root}')

    batch_size = config.training.batch_size
    if batch_size % PRC > 0:
        raise ValueError('Batch size must be divisible by the number of processes')
    local_batch_size = batch_size // PRC
    if local_batch_size % LDC > 0:
        raise ValueError('Local batch size must be divisible by the number of local devices')

    assert config.load_from_pretrained or config.load_from
    assert zone is not None, "Cannot infer zone from workdir."
    if config.load_from:
        step_offset = checkpoint_step(config.load_from, zone=zone)
    else:
        step_offset = 0
    train_loader, train_iter, tokenizer = _create_train_iterator(config, local_batch_size, step_offset)

    ################## Create Model ##################
    model = PaliGemmaEncDec(
        **config.model,
        image_size=config.dataset.image_size,
        use_decoder=False,
    )
    state, normal_lr_fn, siglip_lr_fn = create_train_state(rng, config, model)
    
    if config.load_from:
        # load from a finetuned to half checkpoint
        state = restore_checkpoint(
            state, 
            config.load_from,
            zone=zone
        )
        assert int(state.step) == step_offset, (
            f'Checkpoint step mismatch: inferred {step_offset}, restored {int(state.step)}'
        )
        log_for_0(f'Checkpoint loaded from {config.load_from}.')
    else:
        # only restore params, not optimizer state
        restored_state = restore_checkpoint(
            None, 
            config.load_from_pretrained,
            zone=zone
        )
        params = restored_state['params']
        # delete the decoder params (['image_encoder']['decoder'])
        if 'decoder' in params['image_encoder']:
            params['image_encoder'].pop('decoder')
        # only keep params, not optimizer state
        state = state.replace(params=params)
        del params
        del restored_state
        assert int(state.step) == step_offset, (
            f'Expected initial step {step_offset}, got {int(state.step)}'
        )
        log_for_0(f'Checkpoint loaded from {config.load_from_pretrained}.')

    state = jax_utils.replicate(state)

    p_train_step = jax.pmap(
        partial(
            train_step,
            rng_init=rng,
            config=config,
        ),
        axis_name="batch",
        donate_argnums=(0, 1),
    )
    p_sample_step = jax.pmap(
        partial(
            sample_step,
            model=model,
            **config.sampling.to_dict()
        ),
        axis_name="batch",
    )
    p_sample_step_mmbench = make_mmbench_sample_step(model, config)

    writer = Writer(config, workdir, use_wandb=config.logging.use_wandb, use_tb=False)
    metrics_tracker = MetricsTracker()
    timer = Timer()
    timer.reset()

    log_for_0(f'Starting training from step {step_offset} to {config.training.num_steps}...')
    log_for_0(f'The initial training step may take a while....')


    for step in range(step_offset, config.training.num_steps):
        batch = next(train_iter)
        ########### Train ###########
        raw_batch = {k: v for k, v in batch.items()} # keep a copy of raw batch for visualization
        batch = input_pipeline.prepare_batch_data(batch)
        if config.local_debug and step == step_offset:
            log_for_0(f'first batch ready')
        state, metrics, all_debug = p_train_step(state, batch)
        if step == step_offset:
            log_for_0(f'Train step compiled in {timer}.')
        
        ########### Metrics ###########
        metrics_tracker.update(metrics)  # stream one step in
        if (step+1) % config.training.log_per_step == 0:
            summary = metrics_tracker.finalize()
            summary['steps_per_second'] = config.training.log_per_step / timer.elapse_with_reset()
            summary['normal_lr'] = normal_lr_fn(step + 1)
            summary['siglip_lr'] = siglip_lr_fn(step + 1)
            summary['step'] = step + 1
            writer.write_scalars(step + 1, summary)
            mu.sync_global_devices('log')
        
        # visualize training images
        if step == 0 or (config.training.log_vis_per_step > 0 and (step + 1) % config.training.log_vis_per_step == 0):
            with timer.skip():
                log_for_0("Logging visualization at step {}...".format(step))
                # log some training images
                pixels = raw_batch['pixel_values'][:16].numpy()
                # print pixels range (max / min /mean /std)
                img_grid = vis_util.make_grid_visualization(pixels, to_uint8=True, is_pt=True) # (H, W, C)
                writer.write_images(step + 1, {"train_images": img_grid})
                writer.write_texts(step + 1, "train_captions", [tokenizer.decode(ids) for ids in raw_batch['input_ids'][:16]])
                log_for_0("Visualization logged.")
        
        ########### Save Checkpoint ###########
        if (step + 1) % config.training.checkpoint_per_step == 0 \
            or (step + 1) == config.training.num_steps:
            with timer.skip():
                log_for_0("Saving checkpoint at step {}...".format(step))
                save_checkpoint(state, workdir)
                mu.sync_global_devices('ckpt')

        online_eval_per_step = int(config.training.get("online_eval_per_step", -1))
        online_eval_tasks = list(config.training.get("online_eval_tasks", []) or [])
        if online_eval_per_step > 0 and online_eval_tasks and (step + 1) % online_eval_per_step == 0:
            with timer.skip():
                run_eval_tasks(
                    state,
                    p_sample_step,
                    online_eval_tasks,
                    step=step + 1,
                    run_p_sample_step=run_p_sample_step,
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    writer=writer,
                    p_sample_step_mmbench=p_sample_step_mmbench,
                )

    ########### Final ImageNet KNN eval ###########
    final_step = config.training.num_steps

    # Defer ImageNet download to here so training is never blocked waiting for it.
    _knn_imagenet_root = None
    _need_knn = any(t in {"knn_partial", "knn_full"} for t in final_eval_tasks)
    if _need_knn:
        log_for_0('[KNN] Preparing ImageNet dataset …')
        _knn_imagenet_root = ensure_imagenet_available(
            zone, local_debug=config.local_debug
        )
        log_for_0(f'[KNN] ImageNet root: {_knn_imagenet_root}')

    ########### Final eval with beam_size=1 ###########
    log_for_0("Running final evaluation with beam_size=1...")
    final_step = config.training.num_steps

    final_eval_tasks = list(config.training.get("final_eval_tasks", []) or [])
    if final_eval_tasks:
        run_eval_tasks(
            state,
            p_sample_step,
            final_eval_tasks,
            step=final_step,
            run_p_sample_step=run_p_sample_step,
            model=model,
            tokenizer=tokenizer,
            config=config,
            writer=writer,
            p_sample_step_mmbench=p_sample_step_mmbench,
            task_suffix="_final",
            extra_args={
                "knn_imagenet_root": _knn_imagenet_root,
            },
        )

    if config.eval.get("mmbench_export_test", False):
        log_for_0("Exporting MMBench TEST EN xlsx...")
        export_mmbench_test_xlsx(p_sample_step_mmbench, run_p_sample_step, model, tokenizer, state.params, config)

    log_for_0("Final beam=1 evaluation done.")

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state
    
