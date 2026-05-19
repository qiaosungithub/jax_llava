# 卡.sh

# This is the newest script on 2025.4.11 14:10
source config.sh

if [ -z "$TASKNAME" ]; then
    echo "Please set your own config.sh. See README for reference"
    sleep 60
    exit 1
fi

if [ -z "$1" ]; then

############## TPU VMs ##############

# export VM_NAME=kmh-tpuvm-v4-8-1
# export VM_NAME=kmh-tpuvm-v4-8-2
# export VM_NAME=kmh-tpuvm-v4-8-6
# export VM_NAME=kmh-tpuvm-v5p-32-nopreeeee-1
# export VM_NAME=kmh-tpuvm-v4-64-spot-yiyang
# export VM_NAME=kmh-tpuvm-v5p-64-spot-203
# export VM_NAME=kmh-tpuvm-v5p-64-spot-llqxccvqb
export VM_NAME=kmh-tpuvm-v6e-64-spot-sqaeanuqy
# export VM_NAME=kmh-tpuvm-v6e-32-spot-103
# export VM_NAME=kmh-tpuvm-v6e-64-spot-keyartveow
# export VM_NAME=kmh-tpuvm-v6e-64-spot-52
# export VM_NAME=kmh-tpuvm-v6e-64-spot-105
# export VM_NAME=kmh-tpuvm-v6e-64-spot-202
# export VM_NAME=kmh-tpuvm-v6e-64-spot-303

#####################################
else
    echo ka: use command line arguments
    export VM_NAME=$1
fi

# get zone
if [ -z "$2" ]; then
# auto infer zone
    if [[ $VM_NAME == *"v4-"* ]]; then
        export ZONE=us-central2-b
    elif [[ $VM_NAME == *"v5"* ]]; then
        export ZONE=us-central1-a
        # export ZONE=us-east5-a
    elif [[ $VM_NAME == *"v6"* ]]; then
        # export ZONE=us-east1-d
        export ZONE=us-east5-b
        # export ZONE=us-central1-b
        # export ZONE=asia-northeast1-b
        # export ZONE=europe-west4-a
    else
        export ZONE=us-central1-a
    fi

    echo inferred zone: $ZONE
else
    echo zone: use command line arguments
    export ZONE=$2
fi
# Zone: your TPU VM zone

# DATA_ROOT: the disk mounted
# FAKE_DATA_ROOT: the fake data (imagenet_fake) link
# USE_CONDA: 1 for europe, 2 for us (common conda env)

export DATA_ROOT="kmh-nfs-ssd-us-mount"
if [[ $VM_NAME == *"v4-"* ]]; then
    export USE_CONDA=1
else
    export USE_CONDA=2
fi
# export TFDS_DATA_DIR='gs://kmh-gcp-us-central2/tensorflow_datasets'  # use this for imagenet
export TFDS_DATA_DIR='/kmh-nfs-ssd-us-mount/data/tensorflow_datasets'

if [[ $USE_CONDA == 1 ]]; then
    export CONDA_PY_PATH=/kmh-nfs-ssd-us-mount/code/hanhong/miniforge3/bin/python
    export CONDA_PIP_PATH=/kmh-nfs-ssd-us-mount/code/hanhong/miniforge3/bin/pip
    echo $CONDA_PY_PATH
    echo $CONDA_PIP_PATH
else
    export CONDA_PY_PATH=python
    export CONDA_PIP_PATH=pip
    echo $CONDA_PY_PATH
    echo $CONDA_PIP_PATH
fi
