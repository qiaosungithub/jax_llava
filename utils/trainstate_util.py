import optax, ml_collections
import numpy as np
from jax import random
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict, unflatten_dict

from utils.info_util import print_params
from utils.optim_util import muon
from utils.logging_util import log_for_0
from utils.frozen_util import label_trainable_frozen_params
from utils.pjit_util import MeshMode


def create_lr_schedule(training_config: ml_collections.ConfigDict, base_lr):
  warmup_steps = int(training_config.warmup_steps)
  warmup_fn = optax.linear_schedule(
      init_value=1e-6,
      end_value=base_lr,
      transition_steps=max(warmup_steps, 1),
  )
  if training_config.lr_schedule in ['cosine', 'cos']:
    total_steps = training_config.num_steps
    cosine_steps = max(total_steps - warmup_steps, 1)
    schedule_fn = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=cosine_steps,
        alpha=1e-6,
    )
  elif training_config.lr_schedule == 'const':
    schedule_fn = optax.constant_schedule(value=base_lr)
  elif training_config.lr_schedule == 'zero_const':
    warmup_fn = optax.constant_schedule(value=0.0)
    schedule_fn = optax.constant_schedule(value=base_lr)
    if warmup_steps <= 0:
      return schedule_fn
    return optax.join_schedules(
        schedules=[warmup_fn, schedule_fn],
        boundaries=[warmup_steps],
    )
  else:
    raise NotImplementedError
  if warmup_steps <= 0:
    return schedule_fn
  return optax.join_schedules(
      schedules=[warmup_fn, schedule_fn],
      boundaries=[warmup_steps],
  )


def _init_params_raw(key, image_size, model, text_len):
  input_ids = jnp.ones((1, text_len), dtype=jnp.int32)
  images = jnp.ones((1, image_size, image_size, 3), dtype=jnp.float32)
  prefix_len = jnp.ones((1,), dtype=jnp.int32)
  attention_mask = jnp.ones((1, text_len), dtype=jnp.bool_)
  labels = jnp.ones((1, text_len), dtype=jnp.int32)
  variables = model.init({"params": key}, input_ids, images, prefix_len, attention_mask, labels)
  return variables["params"]


def initialized(key, image_size, model, text_len):
  @jax.jit
  def init(k):
    return _init_params_raw(k, image_size, model, text_len)

  log_for_0("Initializing params...")
  params = init(key)
  log_for_0("Initializing params done.")
  param_count = sum(x.size for x in jax.tree.leaves(params))
  log_for_0("Total trainable parameters: " + str(param_count))
  return params


def _get_weight_decay(config: ml_collections.ConfigDict) -> float:
  if config.training.optimizer == "muon":
    return float(config.training.muon.weight_decay)
  return float(config.training.adam.weight_decay)


def _get_grad_clip_norm(config: ml_collections.ConfigDict):
  grad_clip_norm = float(config.training.grad_clip_norm)
  return grad_clip_norm if grad_clip_norm > 0 else None


def _create_weight_decay_mask(params):
  is_frozen = isinstance(params, FrozenDict)
  d = params.unfreeze() if is_frozen else params
  flat = flatten_dict(d)
  mask_flat = {}

  for key_tuple in flat.keys():
    parts = [str(x).lower() for x in key_tuple]
    leaf_name = parts[-1]
    joined = "/".join(parts)
    is_bias = leaf_name == "bias"
    is_norm = (
      "norm" in joined
      or "layernorm" in joined
      or "rmsnorm" in joined
      or "/ln" in joined
      or joined.endswith("/ln")
    )
    is_embedding = (
      "embedding_table" in leaf_name
      or leaf_name.endswith("_embed")
      or leaf_name.endswith("_tokens")
      or leaf_name in {"pos_embedding", "position_embedding", "positional_embedding"}
      or joined.endswith("/embedding")
    )
    mask_flat[key_tuple] = not (is_bias or is_norm or is_embedding)

  mask = unflatten_dict(mask_flat)
  return FrozenDict(mask) if is_frozen else mask


def create_base_optimizer(config, lr_fn, weight_decay: float = 0.0, weight_decay_mask=None):
  adam = optax.adamw(
      learning_rate=lr_fn,
      weight_decay=weight_decay,
      b2=config.training.adam.adam_b2,
      mask=weight_decay_mask,
  )
  if config.training.optimizer == 'muon':
    muon_cfg = config.training.get('muon', {})
    base_tx = muon(
      learning_rate=lr_fn,
      weight_decay=weight_decay,
      weight_decay_mask=weight_decay_mask,
      custom_adam=adam,
      consistent_rms=bool(muon_cfg.get('consistent_rms', True)),
    )
  else:
    base_tx = adam

  grad_clip_norm = _get_grad_clip_norm(config)
  if grad_clip_norm is not None:
    return optax.chain(optax.clip_by_global_norm(grad_clip_norm), base_tx)
  return base_tx


def _build_optimizer(config, params):
  if config.eval_only:
    tx = optax.sgd(learning_rate=0.0)
    zero = optax.constant_schedule(value=0.0)
    return tx, zero, zero

  base_lr = config.training.adam.learning_rate if config.training.optimizer == 'adam' else config.training.muon.learning_rate
  vision_base_lr = config.training.get('vision_encoder_learning_rate', None)
  if vision_base_lr is None:
    vision_lr_scale = float(config.training.get('vision_encoder_lr_scale', 1.0))
    vision_base_lr = float(base_lr) * vision_lr_scale

  normal_lr_fn = create_lr_schedule(config.training, base_lr)
  siglip_lr_fn = create_lr_schedule(config.training, vision_base_lr)

  weight_decay = _get_weight_decay(config)
  exclude_bias_norm = bool(config.training.get("exclude_bias_norm_from_weight_decay", True))
  weight_decay_mask = _create_weight_decay_mask(params) if exclude_bias_norm else None

  tx_main = create_base_optimizer(config, normal_lr_fn, weight_decay=weight_decay, weight_decay_mask=weight_decay_mask)
  tx_siglip = create_base_optimizer(config, siglip_lr_fn, weight_decay=weight_decay, weight_decay_mask=weight_decay_mask)
  param_groups = label_trainable_frozen_params(
      params,
      freeze_lm=bool(config.training.get('freeze_lm', False)),
      txt_feature_layer=int(config.model.get('txt_feature_layer', 0)),
      freeze_image_encoder=bool(config.training.get('freeze_image_encoder', False)),
      image_prefix='image_encoder',
  )
  tx = optax.multi_transform(
      {
          'img': tx_siglip,
          'main': tx_main,
          'frozen': optax.set_to_zero(),
      },
      param_groups,
  )
  return tx, normal_lr_fn, siglip_lr_fn


def create_train_state(
    rng,
    config: ml_collections.ConfigDict,
    model,
    print_info: bool = True,
    mesh_bundle=None,
):
  rng, rng_init = random.split(rng)

  if mesh_bundle is None:
    params = initialized(rng_init, config.dataset.image_size, model, config.dataset.max_txt_len)
    if print_info:
      print_params(params)
    tx, normal_lr_fn, siglip_lr_fn = _build_optimizer(config, params)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, normal_lr_fn, siglip_lr_fn

  mesh, get_partition_spec, _, _, pjit_compile = mesh_bundle
  del mesh
  log_for_0("Inferring parameter shapes for sharded init...")
  params_shape = jax.eval_shape(
      lambda key: _init_params_raw(key, config.dataset.image_size, model, config.dataset.max_txt_len),
      rng_init,
  )
  param_count = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(params_shape))
  log_for_0("Total trainable parameters: " + str(param_count))

  tx, normal_lr_fn, siglip_lr_fn = _build_optimizer(config, params_shape)
  log_for_0("Inferring optimizer state shapes...")
  opt_shape = jax.eval_shape(tx.init, params_shape)
  state_shape = train_state.TrainState(
      step=0,
      apply_fn=model.apply,
      params=params_shape,
      tx=tx,
      opt_state=opt_shape,
  )
  state_spec = get_partition_spec(state_shape, MeshMode.MODEL)

  log_for_0("Initializing params with jit/HSDP sharding...")
  params = pjit_compile(
      lambda key: _init_params_raw(key, config.dataset.image_size, model, config.dataset.max_txt_len),
      in_shardings=(None,),
      out_shardings=state_spec.params,
  )(rng_init)

  log_for_0("Initializing optimizer state with jit/HSDP sharding...")
  opt_state = pjit_compile(
      tx.init,
      in_shardings=(state_spec.params,),
      out_shardings=state_spec.opt_state,
  )(params)
  state = train_state.TrainState(step=0, apply_fn=model.apply, params=params, tx=tx, opt_state=opt_state)
  if print_info:
    print_params(params)
  return state, normal_lr_fn, siglip_lr_fn
