#!/bin/bash
# Mirror this checkout into the google3 CitC package, for building/running the
# Bazel artifact locally. The XManager launcher does its own staging copy; this
# is the fast inner loop.
#
# `rsync -L` dereferences symlinks, and this tree has several DANGLING ones
# (big_vision, gemma, debug_remote.sh, ... -> a kmh-nfs mount that does not
# exist here). rsync exits 23 on those, so they are excluded explicitly rather
# than by ignoring the exit code -- an ignored exit code would also hide a real
# transfer failure.
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${1:-/google/src/cloud/qiaos/jax_llava_g3/google3/experimental/users/qiaos/jax_llava}"
mkdir -p "$DST"
rsync -a --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='bazel-*' \
  --exclude='big_vision' --exclude='gemma' \
  --exclude='debug_remote.sh' --exclude='staging.sh' --exclude='run_remote.sh' \
  --exclude='新.sh' \
  --exclude='*.npy' --exclude='*.npz' --exclude='*.ckpt' --exclude='*.pth' \
  --exclude='*.pt' --exclude='*.safetensors' --exclude='logs' --exclude='wandb' \
  "$SRC/" "$DST/"
echo "synced $SRC -> $DST"
