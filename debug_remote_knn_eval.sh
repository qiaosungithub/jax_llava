#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./debug_remote_knn_eval.sh <TPU_NAME> <ZONE> [LOAD_FROM_CHECKPOINT|none]

This is a remote-debug launcher, not a tpu-manager run job.  It stages the
current checkout to a debug staging directory, SSHes to all TPU workers, and
runs remote_debug_knn_eval.sh there.

Cost guards:
  - ImageNet is read from the TFDS bucket local to <ZONE>.
  - Cross-zone checkpoint restores are refused unless ALLOW_CROSS_ZONE_CKPT=1.
  - KNN reads only a capped partial train/val subset by default.

Optional overrides:
  KNN_IMAGES_PER_CLASS=32
  KNN_VAL_EXAMPLES=2048
  KNN_BATCH_SIZE=32
  KNN_K=1
  KNN_NUM_WORKERS=2
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 2 ]]; then
    usage
    exit 2
fi

VM_NAME="$1"
ZONE="$2"
LOAD_FROM="${3:-none}"
if [[ "${LOAD_FROM}" == "fresh" ]]; then
    LOAD_FROM="none"
fi

source config.sh
source ka.sh "${VM_NAME}" "${ZONE}"

case "${ZONE}" in
    us-central1*) EXPECTED_CKPT_BUCKET="gs://kmh-gcp-us-central1/" ;;
    us-east5*) EXPECTED_CKPT_BUCKET="gs://kmh-gcp-us-east5/" ;;
    asia-northeast1*) EXPECTED_CKPT_BUCKET="gs://kmh-gcp-asia-northeast1-b/" ;;
    europe-west4*) EXPECTED_CKPT_BUCKET="gs://kmh-gcp/" ;;
    *)
        echo "ERROR: unsupported zone '${ZONE}'." >&2
        exit 2
        ;;
esac

if [[ "${LOAD_FROM}" == gs://* && "${LOAD_FROM}" != "${EXPECTED_CKPT_BUCKET}"* ]]; then
    if [[ "${ALLOW_CROSS_ZONE_CKPT:-0}" != "1" ]]; then
        echo "ERROR: checkpoint bucket does not match TPU zone." >&2
        echo "  zone: ${ZONE}" >&2
        echo "  expected prefix: ${EXPECTED_CKPT_BUCKET}" >&2
        echo "  load_from: ${LOAD_FROM}" >&2
        echo "Set ALLOW_CROSS_ZONE_CKPT=1 only if this cross-zone restore is intended." >&2
        exit 2
    fi
fi

STAGEDIR="/${DATA_ROOT}/staging/${USER}/debug-knn-${VM_NAME}-${ZONE}"
LOGDIR="${STAGEDIR}/log"
DEBUG_STAMP="$(date '+%Y%m%d_%H%M%S')"
REMOTE_LOG_ROOT="/kmh-nfs-ssd-us-mount/logs/${USER}/jax_llava_knn_debug"
REMOTE_WORKDIR="${REMOTE_LOG_ROOT}/${DEBUG_STAMP}_${ZONE}__eval_only"

echo "[debug_remote_knn] TPU=${VM_NAME}"
echo "[debug_remote_knn] zone=${ZONE}"
echo "[debug_remote_knn] load_from=${LOAD_FROM}"
echo "[debug_remote_knn] staging_dir=${STAGEDIR}"
echo "[debug_remote_knn] remote_workdir=${REMOTE_WORKDIR}"

if [[ -d "${STAGEDIR}" ]]; then
    echo "[debug_remote_knn] previous staging size:"
    du -sh "${STAGEDIR}" || true
fi

sudo mkdir -p "${STAGEDIR}"
sudo find "${STAGEDIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
sudo chmod 777 -R "${STAGEDIR}"

echo "[debug_remote_knn] staging files..."
sudo rsync -a . "${STAGEDIR}" \
    --exclude=tmp \
    --exclude=.git \
    --exclude=__pycache__ \
    --exclude='*.png' \
    --exclude=wandb \
    --exclude='*err.log' \
    --exclude=datasets \
    --exclude=datasets_sqa \
    --exclude=dataset_wzh \
    --exclude=dataset_new
sudo chmod 777 -R "${STAGEDIR}"

sudo mkdir -p "${LOGDIR}"
sudo chmod 777 -R "${LOGDIR}"
sudo mkdir -p "${REMOTE_WORKDIR}"
sudo chmod 777 "${REMOTE_LOG_ROOT}" "${REMOTE_WORKDIR}"

ZONE_SHORT="${ZONE::-2}"
REMOTE_ENV="export GOOGLE_APPLICATION_CREDENTIALS=/kmh-nfs-ssd-us-mount/code/qiao/${ZONE_SHORT}.json
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-disabled}
export KNN_IMAGES_PER_CLASS=${KNN_IMAGES_PER_CLASS:-32}
export KNN_VAL_EXAMPLES=${KNN_VAL_EXAMPLES:-2048}
export KNN_BATCH_SIZE=${KNN_BATCH_SIZE:-32}
export KNN_K=${KNN_K:-1}
export KNN_NUM_WORKERS=${KNN_NUM_WORKERS:-2}
export CONDA_PY_PATH=${CONDA_PY_PATH}
export WORKDIR=${REMOTE_WORKDIR}
"

echo "[debug_remote_knn] clearing stale main.py processes on target workers..."
gcloud compute tpus tpu-vm ssh "${VM_NAME}" --zone "${ZONE}" \
    --worker=all --command "
pgrep -af python | grep 'main.py' | grep -v 'grep' | awk '{print \"sudo kill -9 \" \$1}' | sh
sudo rm -rf /tmp/tpu_logs
" --project=he-vision-group

echo "[debug_remote_knn] launching eval-only KNN debug..."
gcloud compute tpus tpu-vm ssh "${VM_NAME}" --zone "${ZONE}" \
    --worker=all --command "
cd ${STAGEDIR}
echo 'Current dir:'
pwd
${REMOTE_ENV}
./remote_debug_knn_eval.sh '${LOAD_FROM}' '${ZONE}'
" --project=he-vision-group 2>&1 | tee -a "${LOGDIR}/output.log"
