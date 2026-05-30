#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./remote_debug_knn_eval.sh [LOAD_FROM_CHECKPOINT|none] [ZONE]

Environment overrides:
  WORKDIR                 Output dir. Default is a timestamped logs/sqa path.
  KNN_IMAGES_PER_CLASS    Train examples per class for partial KNN. Default: 32.
  KNN_VAL_EXAMPLES        Global validation example cap. Default: 2048.
  KNN_BATCH_SIZE          Per-process encode batch. Default: 32.
  KNN_K                   KNN neighbors. Default: 1.
  KNN_NUM_WORKERS         TFDS map parallelism. Default: 2.
  ALLOW_KNN_TFDS_OVERRIDE Respect existing KNN_TFDS_DATA_DIR when set to 1.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

LOAD_FROM="${1:-${LOAD_FROM:-}}"
if [[ "${LOAD_FROM}" == "none" || "${LOAD_FROM}" == "fresh" ]]; then
    LOAD_FROM=""
fi

ZONE="${2:-${ZONE:-us-central1-b}}"
case "${ZONE}" in
    us-central1*) TFDS_BUCKET="gs://kmh-gcp-us-central1/tensorflow_datasets" ;;
    us-east5*) TFDS_BUCKET="gs://kmh-gcp-us-east5/tensorflow_datasets" ;;
    asia-northeast1*) TFDS_BUCKET="gs://kmh-gcp-asia-northeast1-b/tensorflow_datasets" ;;
    europe-west4*) TFDS_BUCKET="gs://kmh-gcp/tensorflow_datasets" ;;
    *)
        echo "ERROR: unsupported zone '${ZONE}' for KNN TFDS debug." >&2
        exit 2
        ;;
esac

if [[ "${ALLOW_KNN_TFDS_OVERRIDE:-0}" == "1" ]]; then
    export KNN_TFDS_DATA_DIR="${KNN_TFDS_DATA_DIR:-${TFDS_BUCKET}}"
else
    # Cost guard: do not inherit ka.sh's NFS TFDS_DATA_DIR or a stale cross-zone
    # KNN_TFDS_DATA_DIR.  The debug run should read zone-local prepared TFDS.
    export KNN_TFDS_DATA_DIR="${TFDS_BUCKET}"
fi

export WANDB_MODE="${WANDB_MODE:-disabled}"
PYTHON_BIN="${CONDA_PY_PATH:-python}"
USER_NAME="${USER:-sqa}"
STAMP="$(date '+%Y%m%d_%H%M%S')"
WORKDIR="${WORKDIR:-/kmh-nfs-ssd-us-mount/logs/${USER_NAME}/jax_llava_knn_debug/${STAMP}_${ZONE}__eval_only}"

KNN_IMAGES_PER_CLASS="${KNN_IMAGES_PER_CLASS:-32}"
KNN_VAL_EXAMPLES="${KNN_VAL_EXAMPLES:-2048}"
KNN_BATCH_SIZE="${KNN_BATCH_SIZE:-32}"
KNN_K="${KNN_K:-1}"
KNN_NUM_WORKERS="${KNN_NUM_WORKERS:-2}"

mkdir -p "${WORKDIR}"

echo "[remote_debug_knn] zone=${ZONE}"
echo "[remote_debug_knn] workdir=${WORKDIR}"
echo "[remote_debug_knn] load_from=${LOAD_FROM}"
echo "[remote_debug_knn] KNN_TFDS_DATA_DIR=${KNN_TFDS_DATA_DIR}"
echo "[remote_debug_knn] images_per_class=${KNN_IMAGES_PER_CLASS} val_examples=${KNN_VAL_EXAMPLES}"

LOAD_FROM_ARGS=()
if [[ -n "${LOAD_FROM}" ]]; then
    LOAD_FROM_ARGS=(--config.load_from="${LOAD_FROM}")
else
    echo "[remote_debug_knn] no load_from; eval will initialize pretrained params"
fi

exec "${PYTHON_BIN}" main.py \
    --workdir="${WORKDIR}" \
    --mode=remote_debug_knn \
    --config=configs/load_config.py:remote_debug_knn \
    "${LOAD_FROM_ARGS[@]}" \
    --config.eval.knn_images_per_class="${KNN_IMAGES_PER_CLASS}" \
    --config.eval.knn_val_examples="${KNN_VAL_EXAMPLES}" \
    --config.eval.knn_batch_size="${KNN_BATCH_SIZE}" \
    --config.eval.knn_k="${KNN_K}" \
    --config.eval.knn_num_workers="${KNN_NUM_WORKERS}"
