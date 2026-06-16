now=`date '+%y%m%d%H%M%S'`
salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
git config --global --add safe.directory $(pwd)
HERE=$(pwd)
commitid=`git show -s --format=%h`  # latest commit id; may not be exactly the same as the commit
export STAGEDIR=/kmh-nfs-ssd-us-mount/staging/$USER/${now}-${salt}-${commitid}-code

echo 'Staging files...'
rsync -a . $STAGEDIR --exclude=tmp --exclude=.git --exclude=__pycache__ --exclude="*.png" --exclude="history" --exclude=wandb --exclude="zhh_code" --exclude="zhh" --exclude=big_vision --exclude=gemma
# cp -r /kmh-nfs-ssd-eu-mount/code/hanhong/MyFile/research_utils/Jax/zhh $STAGEDIR
echo 'Done staging.'

echo $STAGEDIR
cd $STAGEDIR

alias back='cd $HERE'

current_window=$(tmux display-message -p -t "$pane_id" '#S:#I')
tpu upd-staging-info $1 $current_window $STAGEDIR # $1 is the id that passed in
