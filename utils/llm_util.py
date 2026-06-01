# import jax
# import numpy as np
# import gc
# from transformers import AutoTokenizer
# from models.t5_ae_jax import T5Config
# from models.models_t5 import create_t5_encode_fn
# from utils.logging_util import log_for_0

import dataclasses
import functools

import numpy as np
from gemma import gm
from gemma.gm.utils import _file_cache
from etils import epath
from sentencepiece import sentencepiece_model_pb2
import sentencepiece as spm

# ---------------------------------------------------------------------------
# PaliGemma loc-token tokenizer
# ---------------------------------------------------------------------------
# <loc0000> ~ <loc{N_LOC-1}> are mapped onto the existing <unusedX> slots
# (token ids CUSTOM + 0 ~ CUSTOM + N_LOC - 1 = 6 ~ 6+N_LOC-1).
# No model embedding extension is needed since those ids already exist.
N_LOC = 1024   # coordinate bins, shared for x and y

# In the Gemma3 SP vocab (262144 pieces):
#   pieces[6..104]        = <unused0>..<unused98>    (99 tokens, ids 6..104)
#   pieces[105..255999]   = real BPE vocab
#   pieces[256001..262143] = <unused99>..<unused6241> (6143 tokens)
# We remap pieces[_LOC_BASE .. _LOC_BASE+N_LOC-1]
# (<unused99>..<unused1122>) → (<loc0000>..<loc1023>).
_LOC_BASE = 256001         # proto piece index of <unused99> → <loc0000>
_LOC_UNUSED_OFFSET = 99    # <unusedX> where X = _LOC_UNUSED_OFFSET + loc_index

LOC_TOKEN_START = _LOC_BASE
LOC_TOKEN_COUNT = N_LOC
LOC_TOKEN_END = LOC_TOKEN_START + LOC_TOKEN_COUNT


@dataclasses.dataclass(frozen=True)
class PaliGemma3Tokenizer(gm.text.Gemma3Tokenizer):
    """Gemma3 tokenizer extended with <loc0000>…<loc{N_LOC-1}> tokens.

    Overwrites <unused99>…<unused{N_LOC+98}> in the SentencePiece proto,
    bypassing the 99-slot limit of the base `custom_tokens` API.

    Token id for <loc{n:04d}> = _LOC_BASE + n  (= 256001 + n).
    """

    @functools.cached_property
    def _sp(self) -> spm.SentencePieceProcessor:
        model_file = _file_cache.maybe_get_from_cache(
            remote_file_path=self.path,
            cache_subdir='tokenizer',
        )
        raw = epath.Path(model_file).read_bytes()
        raw = _patch_loc_tokens(raw, N_LOC)
        sp = spm.SentencePieceProcessor()
        sp.LoadFromSerializedProto(raw)
        return sp

    @functools.cached_property
    def loc_token_ids(self) -> list[int]:
        """Token ids for <loc0000> ~ <loc{N_LOC-1}> (= [256001, 257024])."""
        return list(range(_LOC_BASE, _LOC_BASE + N_LOC))

    def loc_id(self, n: int) -> int:
        """Token id for <loc{n:04d}>."""
        assert 0 <= n < N_LOC, f"loc index {n} out of range [0, {N_LOC})"
        return _LOC_BASE + n


def _patch_loc_tokens(model_bytes: bytes, n: int) -> bytes:
    """Rename <unused{99+k}> → <loc{k:04d}> for k in [0, n) in the SP proto."""
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(model_bytes)
    for k in range(n):
        piece = proto.pieces[_LOC_BASE + k]
        expected = f"<unused{_LOC_UNUSED_OFFSET + k}>"
        assert piece.piece == expected, (
            f"Expected proto.pieces[{_LOC_BASE + k}].piece == {expected!r}, "
            f"got {piece.piece!r}. Tokenizer vocab layout may have changed."
        )
        piece.piece = f"<loc{k:04d}>"
    return proto.SerializeToString()


def sinusoidal_loc_embeddings(n: int, d_model: int) -> np.ndarray:
    """Sin/cos embeddings for loc tokens, shaped (n, d_model).

    Coordinate value is normalised to [0, 1]: position = idx / (n - 1).
    Uses the same log-spaced frequency scheme as transformer positional
    encodings (and diffusion timestep embeddings).
    """
    positions = np.arange(n, dtype=np.float32) / (n - 1)          # [0, 1]
    half = d_model // 2
    freqs = np.arange(half, dtype=np.float32)
    omega = 1.0 / (10000.0 ** (freqs / half))                      # (half,)
    angles = positions[:, None] * omega[None, :]                   # (n, half)
    emb = np.concatenate([np.sin(angles), np.cos(angles)], axis=1) # (n, d_model)
    if d_model % 2 == 1:   # odd d_model: pad last dim with zeros
        emb = np.concatenate([emb, np.zeros((n, 1), dtype=np.float32)], axis=1)
    return emb.astype(np.float32)


def init_loc_token_embeddings(lm_params: dict) -> dict:
    """Replace <loc> embedding rows with sinusoidal init.

    Called once after loading the pretrained Gemma checkpoint, before training.

    The embedding table lives at lm_params['embedder']['input_embedding'],
    shape (vocab_size, d_model).  We overwrite rows
    [_LOC_BASE : _LOC_BASE + N_LOC]  (= 256001 : 257025) with
    sinusoidal_loc_embeddings(N_LOC, d_model).

    Args:
        lm_params: The raw Gemma parameter dict as loaded by gm.ckpts.load_params.
    Returns:
        lm_params with embedding rows for loc tokens replaced.
    """
    import jax.numpy as jnp

    emb = lm_params['embedder']['input_embedding']   # (V, D), may be jax array
    d_model = emb.shape[1]
    loc_emb = sinusoidal_loc_embeddings(N_LOC, d_model)  # (N_LOC, D) float32

    # Cast to match existing embedding dtype (often bf16 after load)
    loc_emb = loc_emb.astype(emb.dtype)

    # Functional update: replace rows _LOC_BASE : _LOC_BASE + N_LOC
    new_emb = emb.at[_LOC_BASE : _LOC_BASE + N_LOC].set(loc_emb)
    lm_params = dict(lm_params)
    lm_params['embedder'] = dict(lm_params['embedder'])
    lm_params['embedder']['input_embedding'] = new_emb
    return lm_params


# class LLM:
    
#     def __init__(self, config):
#         self.config = config
#         self.model_name = config.dataset.llm
#         assert self.model_name in [
#             'google/flan-t5-small',
#             'google/flan-t5-base', 
#             'google/flan-t5-large',
#             'google/flan-t5-xxl',
#             'debug-llm'
#         ], f'Unsupported model: {self.model_name}'

#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name if self.model_name != 'debug-llm' else 'google/flan-t5-base')

#         if self.model_name == 'debug-llm':
#             self.model_config = T5Config(d_model=16, d_kv=16, d_ff=16, num_layers=1, num_heads=1)
#         else:
#             self.model_config = T5Config.from_pretrained(self.model_name)

#         self.hidden_dim = self.model_config.d_model

#         # encoder are initialized later
#         self.encode_fn = None
#         self.model = None
#         self.params = None
    
#     def init_encoder(self, mesh_bundle):
#         """
#         Initialize encoder after DataLoader is constructed to avoid forking a large model.
#         """
#         if self.encode_fn is not None:
#             return  # already initialized

#         log_for_0(f'Building LLM encoder {self.model_name} after dataloader setup...')
#         log_for_0(f"Before loading LLM encoder, memory allocated: {jax.device_get(jax.local_devices()[0].memory_stats()['bytes_in_use']) / (1024**3):.2f} GB")

#         self.encode_fn, self.model, self.params = create_t5_encode_fn(
#             model_name=self.model_name,
#             max_encoder_length=self.config.dataset.prompt_length,
#             mesh_bundle=mesh_bundle,
#             model_config=self.model_config,
#         )

#         # remove decoder to save memory
#         if isinstance(self.params, dict) and "params" in self.params and "decoder" in self.params["params"]:
#             del self.params["params"]["decoder"]
#         if isinstance(self.params, dict) and "decoder" in self.params:
#             del self.params["decoder"]
#         if hasattr(self.model, "decoder"):
#             delattr(self.model, "decoder")
#         gc.collect()
        
#         log_for_0(f'After loading LLM encoder, memory allocated: {jax.device_get(jax.local_devices()[0].memory_stats()["bytes_in_use"]) / (1024**3):.2f} GB')
        
#     def tokenize_single(self, text):
#         o = self.tokenizer(
#             text,
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=self.config.dataset.prompt_length,
#         )
#         input_ids, attention_mask = o.input_ids, o.attention_mask
#         # input_ids may have shape (1, seq_length), we need to squeeze it
#         if input_ids.ndim != 1:
#             assert input_ids.ndim == 2 and input_ids.shape[0] == 1, f'Unexpected input_ids shape: {input_ids.shape}'
#             input_ids = input_ids.squeeze(0)
#         if attention_mask.ndim != 1:
#             assert attention_mask.ndim == 2 and attention_mask.shape[0] == 1, f'Unexpected attention_mask shape: {attention_mask.shape}'
#             attention_mask = attention_mask.squeeze(0)
#         return input_ids, attention_mask
    
#     def tokenize_batch(self, texts, to_np=True):
#         input_ids, attention_masks = zip(*[self.tokenize_single(text) for text in texts])
#         if to_np:
#             input_ids = np.stack([x.numpy() for x in input_ids], axis=0)
#             attention_masks = np.stack([x.numpy() for x in attention_masks], axis=0)
#         else:
#             raise NotImplementedError('We donnot think you should come to here')
#         return input_ids, attention_masks
    
#     def encode_batch(self, params, input_ids, attention_mask):
#         if self.encode_fn is None or self.params is None:
#             raise RuntimeError("LLM encoder has not been initialized. Call init_encoder() first.")
#         return self.encode_fn(params, input_ids, attention_mask)
    
def create_tokenizer(model_name):
    if model_name.startswith("gemma2"):
        return gm.text.Gemma2Tokenizer()
    elif model_name.startswith("gemma3"):
        return PaliGemma3Tokenizer()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
