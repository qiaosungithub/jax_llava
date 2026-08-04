#!/bin/bash
# Mirror this checkout into the google3 CitC package, for building/running the
# Bazel artifact locally. The XManager launcher does its own staging copy; this
# is the fast inner loop.
#
# `-L` DEREFERENCES symlinks, and it is required, not cosmetic: Blaze refuses
# to glob a package containing an absolute symlink --
#   "Absolute symlinks are forbidden: .../xm_launcher.py"
# -- so `xm_launcher.py -> ~/work/tpu_cmd/xm_launcher.py` (the symlink that
# makes this project provably run the SAME launcher as EqR-jax) must arrive in
# the CitC package as a regular file. EqR-jax's package is a regular file for
# the same reason; the XManager wrapper's own staging pass already uses `-L`.
#
# The tree also has several DANGLING symlinks (big_vision, gemma,
# debug_remote.sh, ... -> a kmh-nfs mount that does not exist here), which make
# rsync exit 23. They are excluded explicitly rather than by ignoring the exit
# code -- an ignored exit code would also hide a real transfer failure.
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${1:-/google/src/cloud/qiaos/jax_llava_g3/google3/experimental/users/qiaos/jax_llava}"
mkdir -p "$DST"
rsync -aL --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='bazel-*' \
  --exclude='big_vision' --exclude='gemma' \
  --exclude='debug_remote.sh' --exclude='staging.sh' --exclude='run_remote.sh' \
  --exclude='新.sh' \
  --exclude='*.npy' --exclude='*.npz' --exclude='*.ckpt' --exclude='*.pth' \
  --exclude='*.pt' --exclude='*.safetensors' --exclude='logs' --exclude='wandb' \
  "$SRC/" "$DST/"
echo "synced $SRC -> $DST"
