# The code is adapted and modified from https://github.com/facebookresearch/flip
# LICENSE: https://github.com/facebookresearch/flip/blob/main/LICENSE

# --------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# --------------------------------------------------------
# References:
# The code is adapted and modified from https://github.com/google-research/t5x/tree/main/t5x
# LICENSE: https://github.com/google-research/t5x/blob/2a62e14fd2806a28c8b24c7674fdd5423aa95e3d/LICENSE
# --------------------------------------------------------


"""Utilities for processing optimizer states."""

from flax import traverse_util
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict, unflatten_dict

def label_params(params, prefix_str):
    """Return a pytree of same structure, each leaf is 'img' or 'main'."""
    is_frozen = isinstance(params, FrozenDict)
    d = params.unfreeze() if is_frozen else params

    flat = flatten_dict(d, sep="/")  # key: "a/b/c"
    labels_flat = {}

    for k in flat.keys():
        # k 是类似 "net/image_encoder/...."
        labels_flat[k] = "img" if (k == prefix_str or k.startswith(prefix_str + "/")) else "main"
    
    # sanity check: assert there is both 'img' and 'main'
    assert 'img' in labels_flat.values() and 'main' in labels_flat.values()

    labels = unflatten_dict({tuple(k.split("/")): v for k, v in labels_flat.items()})
    return FrozenDict(labels) if is_frozen else labels

def tensorstore_leaf(_, value):
    """Detect if the node is a serialized tensorstore spec.

    Args:
      _: The unused name of the current item.
      value: The value of the possible leaf.

    Returns:
      True if the value represents a tensorstore spec, False otherwise.
    """
    # It is a tensorstore leaf if it at least has `driver`, `kvstore` and
    # `metadata` in its keys, sometime they have additional ones like `dtype` or
    # `transform`.
    return set(value.keys()) >= {"driver", "kvstore", "metadata"}


def flatten_state_dict(state_dict, keep_empty_nodes: bool = False):
    """Flatten a dictionary until an array or tensorstore is reached.

    Args:
      state_dict: Optimizer state as nested dictionary.
      keep_empty_nodes: Whether to keep empty node, for example, empty param
        states from simple optimizers or non-touched parameter states in a
        multioptimizer.

    Returns:
      Flattened dictionary, though keeping tensor store state unflattened.
    """
    return traverse_util.flatten_dict(
        state_dict, is_leaf=tensorstore_leaf, keep_empty_nodes=keep_empty_nodes, sep="/"
    )