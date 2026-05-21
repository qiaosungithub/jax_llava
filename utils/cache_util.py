import jax
import os
import pathlib
import glob
from typing import Any, Sequence
import logging
logger = logging.getLogger('jax')
from jax.experimental import multihost_utils as mu
from utils.logging_util import print0, Emoji

ZHH_MIN_CACHE_COMPILE_TIME_SECS = 5.0

# from jax._src.gfile_cache import GFileCache

SHARED_PATH = '/kmh-nfs-ssd-us-mount/code/hanhong/shared/jax_cache'

# class HHFileCache(GFileCache):
class HHFileCache:

  def __init__(self, path):
    if path != SHARED_PATH:
        print0(f'{Emoji.WARNING} Warning: Using shared path {SHARED_PATH} instead of args path={path}.', flush=True)
    self._path = pathlib.Path(SHARED_PATH)
    self._path.mkdir(parents=True, exist_ok=True)
    
    # use project to tag:
    here = os.path.dirname(os.path.abspath(__file__))
    paths = here.strip('/').split('/')
    if paths[0] == 'home':
      assert paths[1] == 'sqa', f'failed to init cache. code dir {here} not supported.'
      paths = paths[2:] # skip /home/sqa
    assert paths[0] == 'kmh-nfs-ssd-us-mount' and paths[1] in ['code', 'staging'], f'failed to init cache. code dir {here} not supported.'
    self.rel_to_root = '/'.join(paths[2:-1]) # last dir is utils
    
    # recursive ls
    all_files = glob.glob(os.path.join(SHARED_PATH, '**'), recursive=True)
    all_files = [f for f in all_files if len(os.path.basename(f)) > 68] # jit-fnnanmeHASH / pjit-fnnanmeHASH, HASH is 64 chars
    print0(f'zhh: Found {len(all_files)} cached files in HHFileCache', flush=True)
    self.key_to_file = {os.path.basename(f): f for f in all_files}

  def get(self, key: str):
    if key not in self.key_to_file:
      # print0(f'zhh: Key {key} not found in cache.', flush=True)
      return None
    print0(f'{Emoji.ROCKET} cache hit: {key}', flush=True)
    path_to_file = self.key_to_file[key]
    # convert to relative path
    assert not path_to_file.startswith('gs://'), 'gs:// paths not supported in HHFileCache get'
    o = pathlib.Path(path_to_file).read_bytes()
    print0(f"{Emoji.GOOD} Read {len(o)} bytes from cache for key {key}", flush=True)
    # logger.debug(f'zhh: read key {key}, got {len(o)} bytes')
    return o

  def put(self, key: str, value: bytes):
    # convert to relative path
    print0(f'{Emoji.TRUCK} caching key {key}', flush=True)
    rel_path = os.path.join(self.rel_to_root, key)
    # super().put(pathlib.Path(rel_path), value)
    file_path = self._path / rel_path
    if not file_path.parent.exists():
      os.system('sudo mkdir -p ' + str(file_path.parent))
    #     file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.parent / f"_temp_{key[:8]}" # ensure temp file is no longer than 68 chars
    os.system('sudo chmod 777 -R ' + str(file_path.parent))
    with open(str(tmp_path), "wb") as f:
      f.write(value)
      f.flush()
      os.fsync(f.fileno())
    os.replace(tmp_path, file_path)
    os.chmod(file_path, 0o777)
    self.key_to_file[key] = str(self._path / rel_path)
    
    
def get_file_cache(path: str):
  """Returns the file cache and the path to the cache."""
  return HHFileCache(path), path

def zhh_cache_read(
    module_name: str, cache_key: str, compile_options: Any,
    backend: Any
) -> tuple[Any | None, int | None]:
  """Looks up the `computation` and it's compilation time in the persistent
  compilation cache repository.
  """
  import jax._src.compilation_cache as compilation_cache
  try:
    return compilation_cache.get_executable_and_time(
        cache_key, compile_options, backend)
  except Exception as ex:
    raise RuntimeError(f'zhh: failed to read cache') from ex
    return None, None

def zhh_cache_write(cache_key: str,
                 compile_time_secs: float,
                 module_name: str,
                 backend: Any, executable: Any,
                 host_callbacks: Sequence[Any]) -> None:
  # ZHH: maintain minimal dependency on jax internals, since jax version may change
  """Writes the `serialized_computation` and its compilation time to the
  persistent compilation cache repository.
  """
  # log_for_0(f'zhh: zhh_cache_write called for key {cache_key}, module {module_name}, compile_time_secs {compile_time_secs}')
  if module_name == 'jit__psum':
    # skip in-module jax compilations, since mu depend on them
    return
  import jax._src.compilation_cache as compilation_cache
  # Only write cache entries from the first process. Otherwise we create
  # problems with contention for writes on some filesystems, e.g., GCS.
  log_priority = logging.DEBUG
  
  if host_callbacks:
    raise NotImplementedError('host_callbacks check not implemented in zhh custom cache')
    logger.log(
        log_priority,
        "Not writing persistent cache entry for '%s' because it uses host "
        "callbacks (e.g. from jax.debug.print or breakpoint)", module_name)
    return
  
  min_compile_time = ZHH_MIN_CACHE_COMPILE_TIME_SECS
  if compile_time_secs < min_compile_time:
    logger.log(
        log_priority,
        "Not writing persistent cache entry for '%s' because it took < %.2f "
        "seconds to compile (%.2fs)", module_name, min_compile_time,
        compile_time_secs)
    # return
  else:
    logger.debug(
        "'%s' took at least %.2f seconds to compile (%.2fs)",
        module_name, min_compile_time, compile_time_secs)
  
    if jax.process_index() != 0:
      logger.log(log_priority,
                "Not writing persistent cache entry since process_id != 0")
    else:
      # worker 0
      # Write the cache entry.
      if cache_key.startswith('pjit') or cache_key.startswith('jit'):
        print0(f'{Emoji.ROBOT} Detected jit/pjit cache key {cache_key}. Caching...', flush=True)
        try:
          compilation_cache.put_executable_and_time(
              cache_key, module_name, executable, backend, int(compile_time_secs))
        except Exception as ex:
          raise RuntimeError(f'zhh: failed to write cache') from ex
        print0(f'{Emoji.GOOD} caching key {cache_key} finished', flush=True)
      else:
        print0(f'{Emoji.WARNING} WARNING: Not caching key {cache_key} since it do not begin with jit or pjit.', flush=True)
        print0(f'{Emoji.THUMBS} Please use jit/HSDP compilation on multiple devices.', flush=True)

  # sync devices before proceeding
  mu.sync_global_devices(f'cache write')
