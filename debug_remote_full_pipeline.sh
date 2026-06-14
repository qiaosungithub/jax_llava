#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./debug_remote_full_pipeline.sh <TPU_NAME> <ZONE>

Stages the current checkout to NFS, clears the debug checkpoint prefix in the
same-region bucket, kills stale main.py on the target workers, and runs:
  main.py --config=configs/load_config.py:remote_debug_full_pipeline
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

source config.sh
source ka.sh "${VM_NAME}" "${ZONE}"

case "${ZONE}" in
    us-central1*) BUCKET="kmh-gcp-us-central1"; CRED_ZONE="us-central1" ;;
    us-east5*) BUCKET="kmh-gcp-us-east5"; CRED_ZONE="us-east5" ;;
    asia-northeast1-b) BUCKET="kmh-gcp-asia-northeast1-b"; CRED_ZONE="asia-northeast1" ;;
    *)
        echo "ERROR: unsupported debug zone '${ZONE}'." >&2
        exit 2
        ;;
esac

STAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_ID="${STAMP}_${VM_NAME}_${ZONE}__full_pipeline"
STAGEDIR="/${DATA_ROOT}/staging/${USER}/debug-full-${VM_NAME}-${ZONE}"
LOCAL_LOGDIR="${STAGEDIR}/log"
WORKDIR="/kmh-nfs-ssd-us-mount/logs/${USER}/jax_llava_remote_debug/${RUN_ID}"
GCS_LOGDIR="gs://${BUCKET}/qiao_zhicheng_hanhong_files/jax_llava_remote_debug/${RUN_ID}"
FIRST_CHECKPOINT_STEP=3
RESUME_CHECKPOINT_STEP=4

echo "[debug_full] TPU=${VM_NAME}"
echo "[debug_full] zone=${ZONE}"
echo "[debug_full] staging_dir=${STAGEDIR}"
echo "[debug_full] workdir=${WORKDIR}"
echo "[debug_full] gcs_logdir=${GCS_LOGDIR}"

sudo mkdir -p "${STAGEDIR}"
sudo find "${STAGEDIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
sudo chmod 777 -R "${STAGEDIR}"

echo "[debug_full] staging files..."
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

sudo mkdir -p "${LOCAL_LOGDIR}" "${WORKDIR}"
sudo chmod 777 -R "${LOCAL_LOGDIR}" "${WORKDIR}"

echo "[debug_full] clearing same-region debug GCS checkpoint prefix..."
gsutil -m rm -r "${GCS_LOGDIR}" 2>/dev/null || true

kill_stale_main() {
    echo "[debug_full] clearing stale main.py processes on target workers..."
    gcloud compute tpus tpu-vm ssh "${VM_NAME}" --zone "${ZONE}" \
        --worker=all --ssh-flag="-n" --project=he-vision-group --command "
# Use [m]ain.py so the matcher does not match this cleanup command itself.
pgrep -f '[m]ain.py' | xargs -r sudo kill -9 || true
sudo rm -rf /tmp/tpu_logs
" || true
    sleep 5
}

run_remote_debug() {
    local config_name="$1"
    local extra_args="${2:-}"
    echo "[debug_full] launching ${config_name} ${extra_args}"
    gcloud compute tpus tpu-vm ssh "${VM_NAME}" --zone "${ZONE}" \
        --worker=all --ssh-flag="-n" --project=he-vision-group --command "
cd ${STAGEDIR}
echo 'Current dir:'
pwd
export GOOGLE_APPLICATION_CREDENTIALS=/kmh-nfs-ssd-us-mount/code/qiao/${CRED_ZONE}.json
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export HF_HOME=/dev/shm/huggingface
export TRANSFORMERS_CACHE=/dev/shm/huggingface
mkdir -p /dev/shm/huggingface
${CONDA_PY_PATH} main.py \
    --workdir=${WORKDIR} \
    --mode=remote_debug_full_pipeline \
    --config=configs/load_config.py:${config_name} ${extra_args}
" 2>&1 | tee -a "${LOCAL_LOGDIR}/output.log"
}

check_dataloader_state() {
    local step="$1"
    local state_glob="${GCS_LOGDIR}/checkpoint_${step}/dataloader_state/process_*.pkl"
    echo "[debug_full] checking dataloader sidecars: ${state_glob}"
    local count
    count="$(gsutil ls "${state_glob}" 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${count}" == "0" ]]; then
        echo "ERROR: no dataloader sidecar found for checkpoint_${step}" >&2
        exit 1
    fi
    echo "[debug_full] checkpoint_${step} dataloader sidecars: ${count}"
}

kill_stale_main
run_remote_debug remote_debug_full_pipeline
check_dataloader_state "${FIRST_CHECKPOINT_STEP}"

kill_stale_main
run_remote_debug remote_debug_full_pipeline_resume "--config.load_from=${WORKDIR}"
check_dataloader_state "${RESUME_CHECKPOINT_STEP}"

echo "[debug_full] full pipeline fresh+resume smoke passed"
