"""CLIP vision tower utilities for LLaVA-style models.

The canonical LLaVA-1.5 vision tower is OpenAI CLIP ViT-L/14 at 336px.  This
module keeps the model-side wrapper small and uses HuggingFace's Flax CLIP
implementation for both parameter loading and forward execution.
"""
from __future__ import annotations

import contextlib

from typing import Any, Optional

import jax.numpy as jnp
import flax.linen as nn
from flax.core import unfreeze
from transformers import CLIPImageProcessor, CLIPVisionConfig, FlaxCLIPVisionModel
from transformers import logging as hf_logging
from transformers.models.clip.modeling_flax_clip import FlaxCLIPVisionTransformer
from utils import g3_env
from utils.logging_util import log_for_0
from utils.pjit_util import constrain_batch, constrain_batch_model


@contextlib.contextmanager
def _silence_hf_from_pretrained_warnings():
    """Temporarily silence HF's noisy 'Some weights ... were not used' /
    'This IS expected ...' messages emitted by from_pretrained when loading
    a vision-only subset out of a full CLIP checkpoint."""
    prev = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        yield
    finally:
        hf_logging.set_verbosity(prev)


# The model's IDENTITY: the HuggingFace repo id. Every comparison and default
# in this module keys off it, so it stays a plain string constant. WHERE the
# weights are read from is a separate question, answered by
# `resolve_clip_source` below -- in google3 there is no network to resolve a
# repo id against (`huggingface_hub`'s ENDPOINT is unpatched and there is no
# internal mirror), so the same identity maps to a CNS directory.
CLIP_L14_336 = "openai/clip-vit-large-patch14-336"


def resolve_clip_source(model_name: str = None) -> str:
    """Map a CLIP model identity to a location this process can actually read.

    In google3 the weights come from a CNS copy of the same HF snapshot.
    `transformers` reaches it directly: copybara rewrites `import os` to
    `import g3lib.os as os` throughout transformers v4, and safetensors'
    google3 fork has a real Colossus backend -- so `from_pretrained` on a
    `/cns/` directory just works, with no staging step.

    Verified equal to the HF snapshot: 303,507,456 params over 391 tensors,
    0 all-zero, position embeddings (577, 1024), and `preprocessor_config`'s
    image_mean/image_std bitwise equal to the _CLIP_MEAN/_CLIP_STD below.

    Do NOT wrap the path in `/readahead/512M/`: `g3lib.os.path.isdir` answers
    True for the prefixed form, so transformers sails through its checks and
    then dies inside the Rust safetensors loader, which matches a literal
    "/cns/" prefix.
    """
    import os as _os

    model_name = model_name or CLIP_L14_336
    # An explicit path (someone already resolved it, or a test points at a
    # staged copy) is passed through untouched.
    if model_name.startswith(("/", "gs://")):
        return model_name
    if model_name != CLIP_L14_336:
        return model_name
    override = (_os.environ.get("JAX_LLAVA_CLIP_PATH") or "").strip()
    if override:
        return override
    if not g3_env.in_google3():
        return CLIP_L14_336
    for root in g3_env.cns_model_roots():
        candidate = f"{root}/clip-vit-large-patch14-336"
        if g3_env.cns_dir_exists(candidate):
            return candidate
    raise FileNotFoundError(
        "No CNS copy of clip-vit-large-patch14-336 found under "
        f"{list(g3_env.cns_model_roots())}. Set JAX_LLAVA_CLIP_PATH to point "
        "at one, or copy the snapshot (config.json + model.safetensors + "
        "preprocessor_config.json is enough)."
    )


_CLIP_L14_336_CONFIG = dict(
    image_size=336,
    patch_size=14,
    num_channels=3,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=24,
    num_attention_heads=16,
    projection_dim=768,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
    attention_dropout=0.0,
    dropout=0.0,
    initializer_factor=1.0,
    initializer_range=0.02,
)

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def clip_vision_config(model_name: str = CLIP_L14_336) -> CLIPVisionConfig:
    """Return a CLIP vision config without doing network I/O."""
    if model_name != CLIP_L14_336:
        raise ValueError(
            f"Unsupported CLIP vision tower {model_name!r}. "
            f"Only {CLIP_L14_336!r} is wired for offline config creation."
        )
    return CLIPVisionConfig(**_CLIP_L14_336_CONFIG)


def load_clip_vision_params(
    model_name: str = CLIP_L14_336,
    *,
    cache_dir: Optional[str] = None,
    dtype: Any = jnp.float32,
    from_pt: bool = True,
) -> dict:
    """Load CLIP vision params in the layout used by :class:`CLIPVisionTower`.

    The HuggingFace repo for OpenAI CLIP-L/14@336 publishes PyTorch weights, so
    ``from_pt=True`` is the default.  The returned dict is shaped as
    ``{"vision_model": ...}``, matching the submodule name used below.
    """
    kwargs = dict(dtype=dtype, from_pt=from_pt)
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    # `model_name` names the model; `source` says where its bytes live. In
    # google3 that is a CNS directory, which transformers reads natively.
    source = resolve_clip_source(model_name)
    log_for_0("Loading CLIP vision tower %s from %s", model_name, source)
    with _silence_hf_from_pretrained_warnings():
        model = FlaxCLIPVisionModel.from_pretrained(source, **kwargs)
    params = unfreeze(model.params)
    if "vision_model" not in params:
        raise KeyError(
            "Unexpected CLIP params layout: expected top-level key "
            f"'vision_model', got {list(params.keys())}"
        )
    return params


def load_clip_image_processor(
    model_name: str = CLIP_L14_336,
    *,
    cache_dir: Optional[str] = None,
) -> CLIPImageProcessor:
    """Load the matching HuggingFace image processor for offline checks."""
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    with _silence_hf_from_pretrained_warnings():
        return CLIPImageProcessor.from_pretrained(
            resolve_clip_source(model_name), **kwargs)


class CLIPVisionTower(nn.Module):
    """OpenAI CLIP ViT-L/14@336 wrapper returning LLaVA patch tokens.

    Inputs are expected to be NHWC tensors from the existing input pipeline.
    That pipeline normalizes images to ``[-1, 1]`` by default, so the wrapper
    converts back to ``[0, 1]`` and applies CLIP mean/std normalization inside
    the model.  This avoids forking the dataloader path for the first sanity
    implementation.
    """

    model_name: str = CLIP_L14_336
    feature_layer: int = -2
    select_feature: str = "patch"
    input_format: str = "minus_one_to_one"
    dtype: Any = jnp.float32

    def setup(self) -> None:
        self.config_obj = clip_vision_config(self.model_name)
        # Use FlaxCLIPVisionTransformer directly (not FlaxCLIPVisionModule) to
        # avoid an extra 'vision_model/' nesting level. HuggingFace's
        # FlaxCLIPVisionModel.params already collapses the outer module name,
        # so its layout is {'vision_model': {'embeddings', 'pre_layrnorm',
        # 'encoder', 'post_layernorm'}}. Wrapping FlaxCLIPVisionModule here
        # would produce {'vision_model': {'vision_model': {...}}} and break
        # tree_map against the loaded checkpoint.
        self.vision_model = FlaxCLIPVisionTransformer(
            config=self.config_obj,
            dtype=self.dtype,
        )

    def _normalize(self, images: jnp.ndarray) -> jnp.ndarray:
        if self.input_format == "minus_one_to_one":
            images = (images + 1.0) * 0.5
        elif self.input_format == "zero_one":
            pass
        elif self.input_format == "clip_normalized":
            return images
        else:
            raise ValueError(f"Unsupported CLIP input_format: {self.input_format}")

        mean = jnp.asarray(_CLIP_MEAN, dtype=images.dtype).reshape(1, 1, 1, 3)
        std = jnp.asarray(_CLIP_STD, dtype=images.dtype).reshape(1, 1, 1, 3)
        return (images - mean) / std

    def __call__(self, images: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        images = constrain_batch(images)
        pixel_values = self._normalize(images)
        outputs = self.vision_model(
            pixel_values,
            deterministic=not train,
            output_hidden_states=True,
            return_dict=True,
        )

        if self.feature_layer == -1:
            features = outputs.last_hidden_state
        else:
            features = outputs.hidden_states[self.feature_layer]
        features = constrain_batch_model(features)

        if self.select_feature == "patch":
            return features[:, 1:, :]
        if self.select_feature == "cls_patch":
            return features
        raise ValueError(f"Unsupported CLIP select_feature: {self.select_feature}")
