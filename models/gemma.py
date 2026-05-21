import dataclasses

import jax
import jax.numpy as jnp
from gemma import gm


def _apply_soft_caps(base_config, attn_logits_soft_cap: float, final_logit_softcap: float):
    """Return a config with the given soft caps applied (0.0 means disabled → None)."""
    attn_cap = attn_logits_soft_cap if attn_logits_soft_cap != 0.0 else None
    final_cap = final_logit_softcap if final_logit_softcap != 0.0 else None
    return dataclasses.replace(
        base_config,
        attn_logits_soft_cap=attn_cap,
        final_logit_softcap=final_cap,
    )


def load_gemma2_2B(attn_logits_soft_cap: float = 0.0, final_logit_softcap: float = 0.0):
    config = _apply_soft_caps(gm.nn.Gemma2_2B.config, attn_logits_soft_cap, final_logit_softcap)
    model = gm.nn.Gemma2_2B(config=config)
    embed_dim = 2304
    return model, embed_dim


def load_gemma3_270M(attn_logits_soft_cap: float = 0.0, final_logit_softcap: float = 0.0):
    config = _apply_soft_caps(gm.nn.Gemma3_270M.config, attn_logits_soft_cap, final_logit_softcap)
    model = gm.nn.Gemma3_270M(tokens="tokens", config=config)
    embed_dim = 640
    return model, embed_dim


def load_gemma3_1B(attn_logits_soft_cap: float = 0.0, final_logit_softcap: float = 0.0):
    config = _apply_soft_caps(gm.nn.Gemma3_1B.config, attn_logits_soft_cap, final_logit_softcap)
    model = gm.nn.Gemma3_1B(tokens="tokens", config=config)
    embed_dim = 1152
    return model, embed_dim


def load_LM(model_str, attn_logits_soft_cap: float = 0.0, final_logit_softcap: float = 0.0):
    if model_str == 'gemma2_2B':
        return load_gemma2_2B(attn_logits_soft_cap, final_logit_softcap)
    elif model_str == 'gemma3_270M':
        return load_gemma3_270M(attn_logits_soft_cap, final_logit_softcap)
    elif model_str == 'gemma3_1B':
        return load_gemma3_1B(attn_logits_soft_cap, final_logit_softcap)
    else:
        raise ValueError(f"Unsupported model string: {model_str}")
