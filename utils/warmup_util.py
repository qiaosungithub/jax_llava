# New warmup.sh!
# credit: the algorithms are proposed by Jiangqi Dai, Kangyang Zhou and Xianbang Wang

import traceback
import jax
import jax.numpy as jnp
import subprocess
import os
import signal
import time
import glob
import socket
import pickle
from functools import partial
from jax.experimental import multihost_utils as mu
from threading import Thread
import abc

def goodpartial(f, *args, **kwargs):
    o = partial(f, *args, **kwargs)
    o.__name__ = f.__name__
    return o

try:
    jax.distributed.initialize()
except:
    # already initialized
    pass

PRI = jax.process_index()
PRC = jax.process_count()
VERBOSITY = 0  # 0: silent, 1: normal, 2: debug

def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

WARMUP_MOUNT_TIMEOUT_SECONDS = _env_int("WARMUP_MOUNT_TIMEOUT_SECONDS", 10 * 60)
WARMUP_DOWNLOAD_TIMEOUT_SECONDS = _env_int("WARMUP_DOWNLOAD_TIMEOUT_SECONDS", 20 * 60)
WARMUP_EXTRACT_TIMEOUT_SECONDS = _env_int("WARMUP_EXTRACT_TIMEOUT_SECONDS", 45 * 60)
WARMUP_CREATE_DATA_TIMEOUT_SECONDS = _env_int("WARMUP_CREATE_DATA_TIMEOUT_SECONDS", 25 * 60)
WARMUP_POST_PROCESS_TIMEOUT_SECONDS = _env_int("WARMUP_POST_PROCESS_TIMEOUT_SECONDS", 50 * 60)

################ LOGGING UTILITIES ####################
def log_for_all(msg):
    print(f"[Rank {PRI}/{PRC}] {msg}", flush=True)
    
def log_for_0(msg):
    if PRI == 0:
        print(f"[Master] {msg}", flush=True)
        
def set_verb(l):
    def inner(f):
        if VERBOSITY >= l:
            return f
        else:
            def noop(*args, **kwargs):
                pass
            return noop
    return inner

v0 = set_verb(0)
v1 = set_verb(1)
v2 = set_verb(2)

v0(log_for_all)("JAX process initialized.")

################ GATHER ALL IP ADDRESSES ####################
IP_ADDR = subprocess.check_output(["hostname", "-I"]).decode("utf-8").strip().split()[0]
WHOAMI = os.getenv("USER")

def ip_to_list(addr: str):
    MAX_NUMS = 4 # max uint 32 required to encode an ip
    int_val = int.from_bytes(addr.encode("utf-8"), byteorder="big")
    # int_val is too long for uint32, split into multiple numbers
    uint_max = (1 << 32) - 1
    out = []
    for _ in range(MAX_NUMS):
        out.append(int_val & uint_max)
        int_val >>= 32
    assert int_val == 0, ZeroDivisionError()
    return out

def list_to_ip(addr_ints: list):
    addr_int = sum(x << (32 * i) for i, x in enumerate(addr_ints))
    n_bytes = (addr_int.bit_length() + 7) // 8
    addr_bytes = addr_int.to_bytes(n_bytes, byteorder="big")
    out = addr_bytes.decode("utf-8")
    assert all(32 <= ord(c) <= 126 for c in out), f"Invalid IP address decoded: {addr_bytes} -> {out}"
    return out

# gather all address
NOW = int(time.time())
d = jax.device_get(mu.process_allgather({
    'arrs' : jnp.array([PRI] + ip_to_list(IP_ADDR), dtype=jnp.uint32),
    'now' : jnp.array(NOW, dtype=jnp.int32)  # padding
}))
all_addr_ints = d['arrs']
try:
    NOW = d['now'][0]
    PORT_WILL_USE = NOW % 20000 + 40000  # port in [40000, 60000], enough random
    RSYNC_PORT = PORT_WILL_USE ^ 1
    v2(log_for_all)(f"Using port {PORT_WILL_USE} for signal transfer")
    v2(log_for_0)(f'Gathered: {all_addr_ints}')
    # decode
    ALL_IP_ADDRS = {
        int(l[0]): list_to_ip(l[1:].tolist())
        for l in all_addr_ints
    }
    v2(log_for_all)(f"All IP addresses: {ALL_IP_ADDRS}")
    v0(log_for_0)(f'IP addresses synced')
except:
    print(f'warning!!! Failed to gather IP addresses: {traceback.format_exc()}. Expect to run only in debug mode.', flush=True)


################# DATA TRANSFER ####################
# helpers for synchronization, not use for now
def server_wait():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", PORT_WILL_USE))
    while True:
        data, addr = s.recvfrom(16)
        if data == b"c2s":
            s.sendto(b"s2c", addr)
            v2(log_for_all)(f"Got signal from {addr}")
            return True

def client_connect(src_ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for _ in range(10):
        s.sendto(b"c2s", (src_ip, PORT_WILL_USE))
        s.settimeout(0.2)
        try:
            data, _ = s.recvfrom(16)
            if data == b"s2c":
                v2(log_for_all)(f"ACK received from server at {src_ip}:{PORT_WILL_USE}")
                return True
        except socket.timeout:
            time.sleep(0.1)
    v0(log_for_all)(f"[client] Failed to connect to server at {src_ip}:{PORT_WILL_USE}")
    return False

# Mutable port state – updated globally whenever we restart the rsync daemon.
# All processes always agree on _rsync_active_port because they derive it from
# the same RSYNC_PORT base and the same _rsync_restart_count.
_rsync_active_port: int = RSYNC_PORT
_rsync_restart_count: int = 0


def _restart_rsync_daemon(data_root: str) -> None:
    """Kill all rsync processes (daemon + any active clients on this node) and
    start a fresh daemon on a new port.

    Updates *_rsync_active_port* and *_rsync_restart_count* in-place.
    Called on **every** process simultaneously (inside wait_run's main loop,
    not inside the background thread), so the port change is coherent.

    Importantly, killing ALL rsync (not just the daemon) ensures that the
    background thread's in-progress rsync client also dies quickly, which lets
    wait_run.thread.join() return fast before the next attempt starts.

    After launching the new daemon, we verify it is actually listening (up to
    3 bind attempts) rather than blindly sleeping.
    """
    global _rsync_active_port, _rsync_restart_count
    _rsync_restart_count += 1
    new_port = RSYNC_PORT + 2 * _rsync_restart_count
    log_for_all(
        f"[rsync restart {_rsync_restart_count}] "
        f"Killing ALL rsync on this node (daemon + any active clients), "
        f"port {_rsync_active_port} → {new_port} …"
    )

    def _write_conf(port: int) -> None:
        rsyncd_conf = (
            "use chroot = no\n"
            "address = 0.0.0.0\n"
            f"port = {port}\n"
            "pid file = /tmp/rsyncd.pid\n"
            "lock file = /tmp/rsyncd.lock\n"
            "\n"
            "[files]\n"
            f"\tpath = {os.path.dirname(data_root)}\n"
            "\tread only = yes\n"
        )
        with open(os.path.expanduser('~/rsyncd.conf'), 'w') as f:
            f.write(rsyncd_conf)
        os.system('sudo mv ~/rsyncd.conf /etc/rsyncd.conf')

    # Kill everything rsync on this node (daemon AND any active client processes).
    # This is intentional: the background thread's rsync client will receive
    # SIGTERM, causing it to exit quickly, so thread.join() in wait_run returns fast.
    os.system('sudo pkill rsync || true')
    os.system('sudo rm -f /tmp/rsyncd.pid /tmp/rsyncd.lock || true')
    time.sleep(3)  # let processes actually die

    # Try to start the daemon and verify it is actually listening.
    # Retry the bind up to 3 times in case the port is momentarily still busy.
    _write_conf(new_port)
    bound = False
    for bind_try in range(3):
        subprocess.Popen(
            'sudo rsync --daemon --no-detach --config=/etc/rsyncd.conf',
            shell=True,
        )
        time.sleep(5)  # give the daemon time to bind
        res = subprocess.run(
            f'sudo lsof -i :{new_port}',
            shell=True, capture_output=True,
        )
        if res.returncode == 0:
            bound = True
            break
        log_for_all(
            f"[rsync restart {_rsync_restart_count}] "
            f"Daemon not yet listening on {new_port} (bind try {bind_try + 1}/3), "
            f"retrying …"
        )
        os.system('sudo pkill rsync || true')
        os.system('sudo rm -f /tmp/rsyncd.pid /tmp/rsyncd.lock || true')
        time.sleep(2)

    if not bound:
        log_for_all(
            f"\033[31m[rsync restart {_rsync_restart_count}] "
            f"WARNING: daemon may not be listening on {new_port} after 3 tries\033[0m"
        )

    _rsync_active_port = new_port
    log_for_all(f"[rsync restart {_rsync_restart_count}] Daemon ready on port {new_port}")


def worker_main_fn(read_from, file_this_round):
    """
    Main propagate function. The worker reads data from `read_from` rank via rsync.

    Uses *_rsync_active_port*, which may be updated between retries by
    wait_run's coordinated restart mechanism.  This function itself does NOT
    retry – retries (with daemon restart on a fresh port) are handled at the
    wait_run level so that all processes participate simultaneously.

    Cases:
      1. file_this_round is a path  – rsync that file/folder.
      2. file_this_round is a list  – rsync all listed files (same directory).
    """
    if read_from is None:
        return
    if not file_this_round:
        return

    src_ip = ALL_IP_ADDRS[read_from]
    port   = _rsync_active_port          # read current port (may have been updated)

    v1(log_for_all)(f"Going to read {file_this_round} from Rank {read_from} via port {port}...")
    if isinstance(file_this_round, list):
        dirs = [os.path.dirname(f) for f in file_this_round]
        assert len(set(dirs)) == 1, "If provided a file list, all files must be in the same directory for rsync"
        common_ancestor = os.path.basename(dirs[0])
        files = ' '.join([
            f'rsync://{src_ip}:{port}/files/{common_ancestor}/{os.path.basename(f)}'
            for f in file_this_round
        ])
        dest = dirs[0]
        os.makedirs(dest, exist_ok=True)
    else:
        files = f'rsync://{src_ip}:{port}/files/{os.path.basename(file_this_round)}'
        dest  = os.path.dirname(file_this_round)

    status = f'exist, with content {os.listdir(dest)}' if os.path.exists(dest) else 'not exist'
    v2(log_for_all)(f"Before receiving, destination {dest} {status}")
    rsync_cmd = f'sudo rsync -avP {files} {dest}'
    v2(log_for_all)(f"Running command: {rsync_cmd}")

    res = subprocess.run(rsync_cmd, shell=True, capture_output=True)
    if res.returncode != 0:
        v0(log_for_all)(f"Error receiving from Rank {read_from}: {res.stderr.decode('utf-8')}")
        raise RuntimeError(f"[Rank {PRI}] Failed to receive {file_this_round} from Rank {read_from}")

def get_matmul_fn():
    from flax.jax_utils import replicate as R
    N = 1 << 15
    key = R(jax.random.key(0))
    x = jax.pmap(lambda key: jax.random.normal(key, (N, N)))(key)
    mm = jax.pmap(lambda x: x@x.T / jnp.linalg.norm(x.T@x))
    return x, mm

def wait_run(fn, *args, max_retries: int = 5, restart_fn=None, timeout_s=None, **kwargs):
    """Run fn(*args, **kwargs) in a background thread; wait for ALL processes to succeed.

    Retry strategy
    --------------
    If any process fails AND max_retries > 0 AND restart_fn is provided:
      1. All processes detect the failure via process_allgather (they are all
         in this loop simultaneously).
      2. restart_fn() is called on every process (port change + daemon restart).
      3. A sync barrier ensures every process has the new daemon up before
         the next attempt.
      4. The function is retried from scratch on every process.

    This is correct because worker_main_fn reads _rsync_active_port at call
    time, so after _restart_rsync_daemon() updates the global, the next
    attempt automatically uses the new port on every node.
    """
    start = time.time()
    x, mm = get_matmul_fn()

    def _run_once(d):
        try:
            fn(*args, **kwargs)
            d['success'] = True
        except Exception as e:
            v0(log_for_all)(
                f"\033[31mException in wait_run({fn.__name__}): {e}. "
                f"traceback: {traceback.format_exc()}\033[0m"
            )
            d['success'] = False
            d['err'] = f'{e.__class__.__name__}: {e}'

    for attempt in range(max_retries + 1):
        if attempt > 0:
            v0(log_for_0)(
                f"\033[33mRetrying \033[0m\033[33m{fn.__name__}\033[0m "
                f"(attempt {attempt + 1}/{max_retries + 1}) …"
            )

        d = {}
        # Daemon threads let the process exit if a worker-side subprocess wedges
        # and the coordinated timeout below turns the warmup into a real failure.
        thread = Thread(target=_run_once, args=(d,), daemon=True)
        thread.start()
        attempt_start = time.time()
        timeout_reported = False
        v0(log_for_0)(
            f"Waiting for function \033[33m{fn.__name__}\033[0m "
            f"(attempt {attempt + 1}/{max_retries + 1}) to complete…"
        )
        time.sleep(10)  # give the thread a moment to start

        success = False
        failed  = False
        needs_retry = False

        while True:
            if not thread.is_alive():
                success = d.get('success', False)
                if not success:
                    failed = True
            elif timeout_s is not None and time.time() - attempt_start > timeout_s:
                failed = True
                d['success'] = False
                d['err'] = f"Timeout after {timeout_s}s"
                if not timeout_reported:
                    timeout_reported = True
                    v0(log_for_all)(
                        f"\033[31mwait_run({fn.__name__}) timed out after "
                        f"{timeout_s}s on this rank. Marking warmup failed "
                        "instead of waiting forever.\033[0m"
                    )

            joined_arr = jax.device_get(
                mu.process_allgather(jnp.array([int(success), int(failed)], dtype=jnp.uint8))
            )

            if all(joined_arr[:, 0].tolist()):
                v0(log_for_0)(
                    f"\033[32m{fn.__name__}\033[0m: "
                    f"All processes succeeded in {time.time() - start:.1f}s."
                )
                x.block_until_ready()
                jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
                return  # ← success path

            elif any(joined_arr[:, 1].tolist()):
                failed_procs = [i for i in range(len(joined_arr)) if joined_arr[i, 1]]
                if attempt < max_retries and restart_fn is not None:
                    # ── coordinated daemon restart on ALL processes ────────────
                    v0(log_for_0)(
                        f"\033[33mProcesses {failed_procs} failed on attempt "
                        f"{attempt + 1}. Restarting rsync daemon on all nodes "
                        f"with a new port and retrying…\033[0m"
                    )
                    # restart_fn kills ALL rsync (daemon + active clients), so
                    # the background thread's rsync client also receives SIGTERM
                    # and dies quickly.  join() ensures the old thread is fully
                    # done before we start a fresh one – no competing processes.
                    restart_fn()                                         # updates _rsync_active_port
                    thread.join(timeout=30)                              # wait for old thread to exit
                    mu.sync_global_devices(f'rsync_restart_{attempt}')  # barrier: all daemons up
                    needs_retry = True
                    break  # exit inner while → outer for loop increments attempt
                else:
                    v0(log_for_0)(
                        f'\033[31mFATAL: Some processes failed: {failed_procs}\033[0m'
                    )
                    raise RuntimeError('两极反转！warmup failed. Contact ZHH')

            else:
                v0(log_for_0)(
                    f'Elapsed: {time.time() - start:.1f}s. '
                    f'Waiting for workers {[i for i in range(len(joined_arr)) if not joined_arr[i, 0]]}…'
                )

            for _ in range(100):
                x = mm(x)

        if not needs_retry:
            break  # shouldn't happen, but safety exit

    # Exhausted all retries
    raise RuntimeError('两极反转！warmup failed after all retries. Contact ZHH')

################ PREPARATION AND CLEANUP ####################
def run_sync_prep(data_root, enable_rsync=True):
    if enable_rsync:
        ######## kill all rsync process #######
        os.system('sudo systemctl stop rsync || true')
        os.system('sudo service rsync stop || true')
        os.system('sudo pkill rsync || true')
        os.system('sudo rm -f /tmp/rsyncd.pid /tmp/rsyncd.lock /tmp/rsyncd.log || true')

        # kill anything holding PORT_WILL_USE or RSYNC_PORT
        for _port in (PORT_WILL_USE, RSYNC_PORT):
            os.system(f'sudo lsof -i :{_port}' + ' | awk \'{ print "sudo kill -9 " $2 }\' | sh')
        time.sleep(5)  # wait for kill to take effect

        ######## setup rsync daemon ########
        v2(log_for_all)("Setting up rsync daemon to serve files...")
        # write rsyncd.conf
        # rsyncd_conf = f'use chroot = no\naddress = 0.0.0.0\n\n[files]\n\tpath = {os.path.dirname(data_root)}\n\tread only = yes\n'
        rsyncd_conf = (
            "use chroot = no\n"
            "address = 0.0.0.0\n"
            f"port = {RSYNC_PORT}\n"
            "pid file = /tmp/rsyncd.pid\n"
            "lock file = /tmp/rsyncd.lock\n"
            "\n"
            "[files]\n"
            f"\tpath = {os.path.dirname(data_root)}\n"
            "\tread only = yes\n"
        )
        with open(os.path.expanduser('~/rsyncd.conf'), 'w') as f:
            f.write(rsyncd_conf)
        os.system('sudo mv ~/rsyncd.conf /etc/rsyncd.conf')
        v2(log_for_all)("Wrote /etc/rsyncd.conf:")
        v2(os.system)('cat /etc/rsyncd.conf')
        # start rsync daemon
        rsync_daemon_cmd = 'sudo rsync --daemon --no-detach --config=/etc/rsyncd.conf'
        v2(log_for_all)(f"Starting rsync daemon with command: {rsync_daemon_cmd}")
        subprocess.Popen(rsync_daemon_cmd, shell=True)
        time.sleep(5)  # wait for daemon to start
        # check if rsync is listening
        v2(log_for_all)("Checking if rsync daemon is listening...")
        res = subprocess.run(f"sudo lsof -i :{RSYNC_PORT}", shell=True, capture_output=True)
        if res.returncode != 0:
            v0(log_for_all)("\033[31m[WARNING] rsync daemon is not listening on port 873\033[0m")
        else:
            v2(log_for_all)("rsync daemon is listening on port 873")
    else:
        v1(log_for_all)("Skipping rsync daemon setup (direct-download mode).")
    
    
    ######## clean up old data ########
    os.system(f'sudo rm -rf {data_root}')
    time.sleep(5) # wait for cleanup

    v0(log_for_0)("\033[32mSynchronize Preparation done.\033[0m")

def run_sync_cleanup(data_root, enable_rsync=True):
    if enable_rsync:
        # kill all rsync process
        os.system('sudo pkill rsync || true') # if don't do this, python program may fail to exit
    v0(log_for_0)("Cleanup done.")

################## TASKS #################
class Task(abc.ABC):
    def __init__(self, data_root, **aux_info):
        self.data_root = data_root
        self.aux_info = aux_info

    @abc.abstractmethod
    def create_data(self):
        """Create data to be transferred."""
        pass
    
    @abc.abstractmethod
    def post_process(self):
        """Post process after data is received."""
        pass
    
class SimpleDemo(Task):
    def create_data(self):
        if PRI == 0:
            foo = {'bar': 123, 'baz': [1, 2, 3], 'now': time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
            bar = '💗 may the tpu be with you 💗'
            os.makedirs(self.data_root, exist_ok=True)
            with open(os.path.join(self.data_root, '1.info'), 'wb') as f:
                pickle.dump(foo, f)
            with open(os.path.join(self.data_root, '2.info'), 'wb') as f:
                pickle.dump(bar, f)

            v1(log_for_0)(f"🖊️ Wrote {self.data_root}")
    
    def post_process(self):
        with open(os.path.join(self.data_root, '1.info'), 'rb') as f:
            obj1 = pickle.load(f)
        with open(os.path.join(self.data_root, '2.info'), 'rb') as f:
            obj2 = pickle.load(f)
        v0(log_for_all)(f"\033[32m 📦 Received object: {obj1}, {obj2}\033[0m")
        
class LargeFileDemo(Task):
    """Demo for large file"""
    # this is written by copilot, not tested yet
    def create_data(self):
        assert 'size_g' in self.aux_info, "LargeFileDemo requires 'size_g' in aux_info"
        if PRI == 0:
            size_g = self.aux_info['size_g']
            DATA_FILE = os.path.join(self.data_root, 'large_file.dat')
            os.makedirs(self.data_root, exist_ok=True)
            os.system(f'head -c {size_g}G /dev/urandom > {DATA_FILE}')
            os.system(f'ls -lh {DATA_FILE}')
            os.system(f'md5sum {DATA_FILE}')
            v1(log_for_0)(f"🖊️ Wrote large file {DATA_FILE} of size {size_g} GB")
    
    def post_process(self):
        DATA_FILE = os.path.join(self.data_root, 'large_file.dat')
        size = os.path.getsize(DATA_FILE)
        md5_val = subprocess.check_output(f'md5sum {DATA_FILE}', shell=True).decode('utf-8').strip()
        v0(log_for_all)(f"\033[32m 📦 Received file size: {size} bytes, md5sum: {md5_val}\033[0m")

# helper
def safe_cmd(cmd, msg, timeout=None):
    start = time.time()
    v1(log_for_all)(f"Starting to {msg} with command: {cmd}")
    capture_output = PRI != 0
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        if proc.returncode != 0:
            e = subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
            raise e
    except subprocess.TimeoutExpired as e:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.communicate(timeout=15)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        raise RuntimeError(f"Timed out while trying to {msg} after {timeout}s") from e
    except subprocess.CalledProcessError as e:
        v0(log_for_all)(f"\033[31mFailed to {msg} (error code {e.returncode}): ---- STDOUT ----\n{e.stdout}\n---- STDERR ----\n{e.stderr}\033[0m")
        raise RuntimeError(f"Failed to {msg}") from e
    v1(log_for_all)(f"Successfully completed {msg} in {time.time() - start} seconds.")

def _get_instance_region():
    zone_path = subprocess.check_output(
        "curl -fs -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone",
        shell=True,
        text=True,
    ).strip()
    zone = zone_path.split('/')[-1]
    parts = zone.split('-')
    if len(parts) < 3:
        raise RuntimeError(f"Unexpected instance zone format: {zone}")
    return '-'.join(parts[:-1])

def _get_bucket_region(gs_path):
    if not gs_path.startswith('gs://'):
        raise ValueError(f"Expected gs:// path, got: {gs_path}")
    bucket = gs_path[len('gs://'):].split('/')[0]
    location = subprocess.check_output(
        f"gcloud storage buckets describe gs://{bucket} --format='value(location)'",
        shell=True,
        text=True,
    ).strip()
    return location.lower()

def _assert_same_region_transfer(gs_path):
    instance_region = _get_instance_region().lower()
    bucket_region = _get_bucket_region(gs_path)
    if bucket_region != instance_region:
        raise RuntimeError(
            f"Bucket region ({bucket_region}) does not match instance region ({instance_region}); "
            "aborting to guarantee same-region transfer."
        )
    v0(log_for_all)(
        f"[same-region-check] instance_region={instance_region}, bucket_region={bucket_region}"
    )

class Warmup(Task):
    """The warmup.sh task"""
    def create_data(self):
        assert self.data_root == '/dev/shm/tmp_data', "Warmup task must use /dev/shm/tmp_data as data_root"
        assert all(k in self.aux_info for k in ['gs_root', 'shm_dest', 'suffix', 'num', 'root']), f"Warmup task requires 'gs_root', 'shm_dest', 'suffix', 'num', 'root' in aux_info, got {self.aux_info.keys()}"
        self.gs_root = self.aux_info['gs_root']
        self.shm_dest = self.aux_info['shm_dest']
        self.suffix = self.aux_info['suffix']
        self.num = self.aux_info['num']
        self.root = self.aux_info['root']
        
        
        # step 1: mount disk
        fight_sh_path = os.path.join(os.path.dirname(__file__), 'mount_disk.sh')
        safe_cmd(f'sudo bash {fight_sh_path}', msg='mount disk', timeout=WARMUP_MOUNT_TIMEOUT_SECONDS)
        
        # step 2: each process downloads directly from GCS
        gs_sources = self.gs_root if isinstance(self.gs_root, (list, tuple)) else [self.gs_root]
        _assert_same_region_transfer(gs_sources[0])
        safe_cmd(
            f"gcloud storage cp -r {' '.join(gs_sources)} /dev/shm/tmp_data",
            msg='download data from GCS',
            timeout=WARMUP_DOWNLOAD_TIMEOUT_SECONDS,
        )
        # safe_cmd(f'sudo gsutil -m cp -r {self.gs_root} /dev/shm/tmp_data', msg='download data from GCS')
        v0(log_for_all)(f'\033[32m 📥 Downloaded data from GCS to /dev/shm/tmp_data \033[0m')
        
        return True

    def post_process(self):
        # step 3: extract data
        dest = self.shm_dest
        CMD = f'''
            set -eu
            mkfifo /dev/shm/zhh_stream
            (
            for f in /dev/shm/tmp_data/*; do
                sudo cat "$f"
                sudo rm -f "$f"
            done > /dev/shm/zhh_stream
            ) &
            sudo rm -rf {dest}
            sudo mkdir -p {dest}
            sudo chmod a+r {dest}
            sudo tar -C "{dest}" -xf /dev/shm/zhh_stream
            ls /dev/shm/tmp_data || echo "tmp_data 已清空 ✅"
            sudo rm -f /dev/shm/zhh_stream
            sudo rm -rf /dev/shm/tmp_data # clean up
        '''
        safe_cmd(CMD, msg='extract data', timeout=WARMUP_EXTRACT_TIMEOUT_SECONDS)

        all_pt = glob.glob(os.path.join(self.root, '**', f'*.{self.suffix}'), recursive=True)
        log_for_all(f'Found {len(all_pt)} {self.suffix} files in {self.root}')
        if len(all_pt) < self.num:
            ds = subprocess.check_output(f'find {self.shm_dest} -type d', shell=True).decode('utf-8').strip().split('\n')
            ds = [d for d in ds if not os.path.basename(d)[-1].isdigit()]
            raise RuntimeError(f"Extracted file count is less than expected! This may be due to bugs in `WARMUP_ARGS` or `class LatentDataset`. For a supplementary info, we have folder names listed as {ds}.")
        log_for_all(f'\033[32m 📂 Data extraction and verification completed. \033[0m')
        return True

################ ALGORITHMS ###############
class Algorithm(abc.ABC):
    def __init__(self, task: Task, **aux_info):
        self.task = task
        self.data_root = task.data_root
        self.create_data_fn = task.create_data
        self.post_process_fn = task.post_process
        self.aux_info = aux_info | task.aux_info
        self.use_rsync = True
        self.setup()
        run_sync_prep(self.data_root, enable_rsync=self.use_rsync)
        
        # before all distributed part, all worker have to be synchronized
        v0(log_for_0)(f'Using {self.__class__.__name__} algorithm for distributed data synchronization.')
        mu.sync_global_devices('algorithm initialized')
        
        # create data (this corresponds to download from gs)
        wait_run(self.create_data_fn, timeout_s=WARMUP_CREATE_DATA_TIMEOUT_SECONDS)

    @abc.abstractmethod
    def setup(self):
        """Describe the communication pattern."""
        pass
    
    @abc.abstractmethod
    def run(self):
        """Run the algorithm."""
        pass
    
    def finish(self):
        wait_run(self.post_process_fn, timeout_s=WARMUP_POST_PROCESS_TIMEOUT_SECONDS)
        run_sync_cleanup(self.data_root, enable_rsync=self.use_rsync)
        
    def __del__(self):
        v2(log_for_all)("Algorithm object is being deleted.")
        if self.use_rsync:
            os.system('sudo pkill rsync || true') # ensure rsync is killed

class Zak(Algorithm):
    """
    zak's algorithm:
    (round 1) 0 -> 1
    (round 2) 0 -> 2, 1 -> 3
    ...
    """
    def setup(self):
        max_idx = 0
        self.l = []
        while max_idx < PRC - 1:
            self.l.append([(i, max_idx + i + 1) for i in range(max_idx + 1)])
            max_idx = self.l[-1][-1][1]
        v1(log_for_0)(f"Zak communication rounds: {self.l}")
        
    def run(self):
        for round, pairs in enumerate(self.l):
            read_from = [src for (src, dst) in pairs if dst == PRI]
            read_from = read_from[0] if read_from else None
            send_to = [dst for (src, dst) in pairs if src == PRI]
            v1(log_for_all)(f"Zak Round {round}: Rank {PRI} read_from={read_from}, send_to={send_to}")
            wait_run(
                worker_main_fn, read_from, self.data_root,
                restart_fn=partial(_restart_rsync_daemon, self.data_root),
            )

class DirectDownload(Algorithm):
    """Every process downloads from GCS directly; no peer-to-peer rsync."""

    def setup(self):
        self.use_rsync = False

    def run(self):
        v0(log_for_0)("DirectDownload mode: skipped inter-device rsync propagation.")
            
# export
def run_warmup_main(gs_root, shm_dest, suffix, num, root):
    files = subprocess.check_output(f'gsutil ls {gs_root}.tar.*', shell=True).decode('utf-8').strip().split('\n')
    files = sorted([f for f in files if f])
    gs_sources = files if files else [gs_root]
    v1(log_for_0)(f"Files in GCS: {gs_sources}")
    task = Warmup(data_root='/dev/shm/tmp_data', gs_root=gs_sources, shm_dest=shm_dest, suffix=suffix, num=num, root=root)
    algo = DirectDownload(task)
    algo.run()
    algo.finish() # post process & clean up

if __name__ == "__main__":
    # demo
    # task = SimpleDemo(data_root='/dev/shm/data')
    # algo = DirectDownload(task=task)
    # algo.run()
    # algo.finish()
    
    # run_warmup_main(
    #     "gs://kmh-gcp-us-east5/hanhong/imagenet_latents_zhh/imagenet_latent_zhh",
    #     "/mnt/zhhm/zhh/latents",
    #     'pt',
    #     1281167,
    #     "/mnt/zhhm/zhh/latents/vae_cached_muvar_imagenet_zhh/train" # new data root
    # )
    run_warmup_main(
        "gs://kmh-gcp-us-east5/data/imagenet/imagenet",
        "/mnt/zhhm/zhh/imagenet",
        'JPEG',
        1281167,
        "/mnt/zhhm/zhh/imagenet/imagenet/train" # new data root
    )
    log_for_all("✅ All done, exiting...")
