# GCP TPU-VM path (ka.sh and the debug_remote_* scripts read these).
export TASKNAME=paligemma-baseline
export WANDB_API_KEY=73f8ff40bb7f8589e9bd1f476196a896f662cdfa
export DATA_ROOT=kmh-nfs-ssd-us-mount

# google3/Borg path. Read by ~/work/tpu_cmd/tpu_wrapper.sh and xm_launcher.py.
# The wrapper rewrites TARGET_LABEL to point at the per-launch staged snapshot,
# so the value here is the fallback for a direct build from the CitC package.
export PROJECT_NAME="jax_llava"
export PACKAGE_MODE="bazel"
export TARGET_LABEL="//experimental/users/qiaos/jax_llava:main"
