set -ex

# export JAX_PLATFORMS=cpu

WORKDIR=$(pwd)/tmp
sudo rm -rf $WORKDIR
sudo rm -rf ./wandb
mkdir -p $WORKDIR

export GOOGLE_APPLICATION_CREDENTIALS=/kmh-nfs-ssd-us-mount/code/siri/bu/bucket-us-central2.json

sleep 2
python main.py --config configs/load_config.py:local_debug \
    --workdir  $WORKDIR \
    --mode local_debug