echo "[worker] $(hostname): fighting with apt..."
export DEBIAN_FRONTEND=noninteractive

systemctl stop apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service || true
systemctl disable apt-daily.timer apt-daily-upgrade.timer || true
systemctl mask unattended-upgrades.service apt-daily.service apt-daily-upgrade.service || true
pkill -9 unattended-upgrade apt.systemd.daily || true

rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock
dpkg --configure -a || true

until apt-get install -y zstd pv python3-crcmod >/dev/null; do
  echo "[worker] apt install failed, retrying..."
  ( 
    systemctl stop unattended-upgrades || true 
    killall unattended-upgrade || true 
    for f in /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock; do
      sudo kill -9 $(sudo lsof -t "$f" 2>/dev/null) 2>/dev/null || true
    done
    rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock || true
    dpkg --configure -a || true
  ) || true
  sleep 5
done

# check if crcmod is compiled
if ! gsutil version -l | grep 'compiled crcmod: True' -q; then
  echo '[worker] failed to install compiled crcmod, exiting...'
  exit 7
fi

echo '[worker] dependencies installed.'

echo "[worker] Using zhh new script to mount a fast tmpfs"
rm -rf /dev/shm/* # clean up old data in /dev/shm
(
  ps -ef | grep gsutil | grep kmh-gcp | awk '{ print "sudo kill -9 " $2 }' | sh
  pids=$(grep -l "/dev/shm" /proc/*/maps 2>/dev/null | awk -F'/' '{print $3}' | sort -u)
  echo "[worker] found 杀不死的小强s that are using /dev/shm: $pids"
  # display each process's cmdline
  for pid in $pids; do
    (ps -ef | grep "$pid " | grep -v grep) || true
  done
) || true

# if /mnt/zhhm does not exist, create it
if [ ! -d /mnt/zhhm ]; then
  mkdir -p /mnt/zhhm
  mount -t tmpfs -o size=270G,mode=0755,uid=0,gid=0 tmpfs /mnt/zhhm
fi
echo "[worker] /mnt/zhhm exists, with contents:"
ls -lh /mnt/zhhm

echo "[worker] tmpfs mounted at /mnt/zhhm"
echo '[worker] lree -h:'; free -h || true
echo '[worker] df -h /dev/shm:'; df -h /dev/shm || true
echo '[worker] df -h /mnt/zhhm:'; df -h /mnt/zhhm

sudo mkdir -p /dev/shm/tmp_data
sudo chmod a+w /dev/shm/tmp_data
