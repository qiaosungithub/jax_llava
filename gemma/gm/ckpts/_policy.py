# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint loader for pair-wise models."""

import dataclasses
from typing import Any, Protocol, Optional

from gemma.gm.ckpts import _checkpoint
import jax
import jax.numpy as jnp
# from kauldron import kd

class PartialLoader(Protocol):
    def transform(self, state: Any) -> Any: ...


def _state_replace(state: Any, **kwargs) -> Any:
    # flax.struct.dataclass / TrainState 常见的 replace
    if hasattr(state, "replace") and callable(state.replace):
        return state.replace(**kwargs)
    return dataclasses.replace(state, **kwargs)

@dataclasses.dataclass(frozen=True, kw_only=True)
class AnchoredPolicyLoader:
  """Loader for `gm.nn.AnchoredPolicy` models.

  Loaded load policy and anchor separately by providing
  sub-transforms.

  This assume the sub-loaders only overwrite the `state.params` without
  modifying the rest of the state.
  """

  policy: PartialLoader
  anchor: Optional[PartialLoader] = None

  def transform(self, state: Any) -> Any:
    if not isinstance(state.params, dict) or set(state.params.keys()) != {"policy", "anchor"}:
        raise ValueError(
            "AnchoredPolicyLoader expects state.params keys to be {'policy', 'anchor'} "
            "(intended for AnchoredPolicy-style models)."
        )

    # --- Load policy params ---
    policy_state = _state_replace(state, params=state.params["policy"])
    policy_state = self.policy.transform(policy_state)

    # --- Load anchor params ---
    if self.anchor is None:
        # If no anchor loader, initialize anchor as a copy of policy params.
        # (release_memory is optional; safe to omit if you don't have it)
        anchor_params = jax.tree_util.tree_map(lambda x: jnp.copy(x), policy_state.params)
        anchor_state = _state_replace(policy_state, params=anchor_params)
    else:
        anchor_state = _state_replace(state, params=state.params["anchor"])
        anchor_state = self.anchor.transform(anchor_state)

    # --- Merge back ---
    return _state_replace(
        state,
        params={
            "policy": policy_state.params,
            "anchor": anchor_state.params,
        },
    )