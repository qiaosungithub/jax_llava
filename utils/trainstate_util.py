import optax, ml_collections
from jax import random
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict, unflatten_dict

from utils.info_util import print_params, print_params_compact
from utils.optim_util import muon
from functools import partial
from utils.logging_util import log_for_0
from utils.frozen_util import label_trainable_frozen_params

def create_lr_schedule(
    training_config: ml_collections.ConfigDict,
    base_lr,
):
    '''
    Create learning rate schedule.
    '''
    warmup_fn = optax.linear_schedule(
        init_value=1e-6,
        end_value=base_lr,
        transition_steps=training_config.warmup_steps,
    )
    if training_config.lr_schedule in ['cosine', 'cos']:
        total_steps = training_config.num_steps
        cosine_steps = max(total_steps - training_config.warmup_steps, 1)
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
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, schedule_fn],
            boundaries=[training_config.warmup_steps],
        )
        return schedule_fn
    else: raise NotImplementedError
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, schedule_fn],
        boundaries=[training_config.warmup_steps],
    )
    return schedule_fn

def create_paligemma_style_slow_warmup_schedule(
    training_config: ml_collections.ConfigDict,
    base_lr,
    lr_fn,
):
    """
    Create learning rate schedule.
    """
    def new_lr_fn(step):
      normal_lr = lr_fn(step)
      warmup_lr = base_lr * (step / training_config.siglip_warmup_steps)
      return jnp.minimum(normal_lr, warmup_lr)
    return new_lr_fn

def initialized(key, image_size, model, text_len):

    input_ids = jnp.ones((1, text_len), dtype=int)
    images = jnp.ones((1, image_size, image_size, 3), dtype=jnp.float32)
    prefix_len = jnp.ones((1,), dtype=int)
    attention_mask = jnp.ones((1, text_len), dtype=int)
    labels = jnp.ones((1, text_len), dtype=int)

    @jax.jit
    def init(*args):
        return model.init(*args)

    log_for_0("Initializing params...")
    variables = init({"params": key}, input_ids, images, prefix_len, attention_mask, labels)
    log_for_0("Initializing params done.")

    param_count = sum(x.size for x in jax.tree.leaves(variables["params"]))
    log_for_0("Total trainable parameters: " + str(param_count))
    return variables["params"]

def _get_weight_decay(config: ml_collections.ConfigDict) -> float:
  if config.training.optimizer == "muon":
    return float(config.training.muon.weight_decay)
  return float(config.training.adam.weight_decay)

def _get_grad_clip_norm(config: ml_collections.ConfigDict):
  # Classic VLM setup usually uses global grad norm clipping around 1.0.
  grad_clip_norm = float(config.training.grad_clip_norm)
  return grad_clip_norm if grad_clip_norm > 0 else None

def _create_weight_decay_mask(params):
  """
  Return a pytree mask for decayed weights.
  True -> apply weight decay; False -> skip weight decay.
  """
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
    # Match embedding-like params. Three families are checked separately:
    #   - Gemma's lookup table: `input_embedding_table`, plus `per_layer_input_embedding_table` in gemma3n.
    #   - Standard pos-embedding suffixes (`pos_embedding`, `position_embedding`, …) and `_embed`-style
    #     names used by PrefixMAE: `patch_pos_embed`, `learnable_pos_embed`, `abs_pos_embed`,
    #     `patch_query_embed`.
    #   - Learnable token banks used as CLS/query/register tokens: `learnable_tokens`, `register_tokens`.
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
    base_tx = muon(
      learning_rate=lr_fn,
      weight_decay=weight_decay,
      weight_decay_mask=weight_decay_mask,
      custom_adam=adam,
    )
  else:
    base_tx = adam

  grad_clip_norm = _get_grad_clip_norm(config)
  if grad_clip_norm is not None:
    tx = optax.chain(optax.clip_by_global_norm(grad_clip_norm), base_tx)
  else:
    tx = base_tx
  return tx

def create_train_state(rng, config: ml_collections.ConfigDict, model, print_info: bool = True) -> train_state.TrainState:
  """
  Create initial training state.
  ---
  apply_fn: output a dict, with key 'loss', 'mse'
  """

  rng, rng_init = random.split(rng)

  params = initialized(rng_init, config.dataset.image_size, model, config.dataset.max_txt_len)

  # params = abstract_model.params
  # ema_params = deepcopy(params) # no ema
  # ema_params = update_ema(params, params, 0.0)
  if print_info:
    print_params(params)

  if config.eval_only:
    # dull optimizer
    tx = optax.sgd(learning_rate=0.0)
    normal_lr_fn = optax.constant_schedule(value=0.0)
    siglip_lr_fn = optax.constant_schedule(value=0.0)
  else:
    base_lr = config.training.adam.learning_rate if config.training.optimizer == 'adam' else config.training.muon.learning_rate
    vision_base_lr = config.training.get('vision_encoder_learning_rate', None)
    if vision_base_lr is None:
      vision_lr_scale = float(config.training.get('vision_encoder_lr_scale', 1.0))
      vision_base_lr = float(base_lr) * vision_lr_scale

    normal_lr_fn = create_lr_schedule(config.training, base_lr)
    vision_normal_lr_fn = create_lr_schedule(config.training, vision_base_lr)
    siglip_lr_fn = create_paligemma_style_slow_warmup_schedule(
      config.training, vision_base_lr, vision_normal_lr_fn
    )

    weight_decay = _get_weight_decay(config)
    exclude_bias_norm = bool(config.training.get("exclude_bias_norm_from_weight_decay", True))
    weight_decay_mask = _create_weight_decay_mask(params) if exclude_bias_norm else None

    tx_main = create_base_optimizer(
      config, normal_lr_fn, weight_decay=weight_decay, weight_decay_mask=weight_decay_mask
    )
    tx_siglip = create_base_optimizer(
      config, siglip_lr_fn, weight_decay=weight_decay, weight_decay_mask=weight_decay_mask
    )

    param_groups = label_trainable_frozen_params(
      params,
      freeze_lm=bool(config.training.get('freeze_lm', False)),
      txt_feature_layer=int(config.model.get('txt_feature_layer', 0)),
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
  
  state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
  )
  return state, normal_lr_fn, siglip_lr_fn
