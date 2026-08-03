"""A minimal `webdataset` replacement covering exactly what jax_llava uses.

There is no `//third_party/py/webdataset` in google3 and nothing in the depot
links the real package (the two files that `import webdataset` either wrap it
in `try/except ImportError` -- //third_party/py/timm/data/readers/reader_wds.py
-- or have no BUILD target at all -- //third_party/deepmind/ufo4d/...).

This module exists to answer ONE question: is the webdataset surface
jax_llava depends on small enough to reimplement, and does
`input_pipeline.py` import and run against a reimplementation unmodified?

Surface reimplemented (the complete set jax_llava touches, from a repo-wide
grep):
    wds.WebDataset(urls, resampled=, shardshuffle=, handler=)
        .shuffle(n, rng=) .decode("pil") .map(fn) .select(fn)
    wds.DataPipeline(*stages)
    wds.SimpleShardList(urls)
    wds.tarfile_to_samples(handler=)
    wds.shardlists.expand_urls(brace_pattern)
    wds.filters.RandomMix(datasets, weights)
    webdataset.gopen.gopen_schemes   (the dict jax_llava monkey-patches)

Shard bytes are read through `open_shard()`, which dispatches on scheme:
gopen_schemes first (so jax_llava's register_gcsfs patch still wins), then
pyglib.gfile for /cns/, /bigstore/, /placer/ and plain POSIX paths. That is
the piece a pip-installed webdataset could never do.
"""

import io
import random as _random
import re
import tarfile

# jax_llava's register_gcsfs() writes into this dict:
#     gopen_module.gopen_schemes["gs"] = gopen_gcsfs
# Keeping the same name and semantics means that monkey-patch keeps working
# unmodified, and lets a CNS/bigstore opener be registered the same way.
gopen_schemes = {}


def _gfile():
  return __import__("google3.pyglib.gfile", fromlist=["gfile"])


def open_shard(url, mode="rb"):
  """Open one shard by URL. gopen_schemes wins, then gfile, then builtins."""
  scheme = url.split("://", 1)[0] if "://" in url else ""
  if scheme and scheme in gopen_schemes:
    return gopen_schemes[scheme](url, mode=mode)
  if url.startswith(("/cns/", "/bigstore/", "/placer/", "/namespace/")):
    return _gfile().Open(url, mode)
  return open(url, mode)


# --------------------------------------------------------------------------
# shardlists
# --------------------------------------------------------------------------
_BRACE = re.compile(r"\{(\d+)\.\.(\d+)\}")


def expand_urls(urls):
  """Expand `shard-{000000..000123}.tar` brace notation."""
  if isinstance(urls, (list, tuple)):
    out = []
    for u in urls:
      out.extend(expand_urls(u))
    return out
  m = _BRACE.search(urls)
  if not m:
    return [urls]
  lo, hi = m.group(1), m.group(2)
  width = len(lo)
  return [
      urls[: m.start()] + str(i).zfill(width) + urls[m.end():]
      for i in range(int(lo), int(hi) + 1)
  ]


class shardlists:  # noqa: N801 -- mirrors `wds.shardlists.expand_urls`
  expand_urls = staticmethod(expand_urls)


def SimpleShardList(urls, seed=None):  # noqa: N802 -- upstream name
  """Yield {"url": ...} dicts, the shape tarfile_to_samples expects."""
  if isinstance(urls, str):
    urls = expand_urls(urls)
  urls = list(urls)
  if seed is not None:
    _random.Random(seed).shuffle(urls)

  def _stage(_src=None):
    for u in urls:
      yield {"url": u}

  _stage.urls = urls
  return _stage


# --------------------------------------------------------------------------
# tar iteration
# --------------------------------------------------------------------------
def base_plus_ext(name):
  """'a/b/000123.left.jpg' -> ('a/b/000123', 'left.jpg')."""
  i = name.find(".", name.rfind("/") + 1)
  if i < 0:
    return None, None
  return name[:i], name[i + 1:]


def _iter_tar(url, handler=None):
  """Yield {'__key__','__url__', <ext>: bytes} dicts from one tar shard."""
  try:
    fh = open_shard(url)
  except Exception as exc:  # noqa: BLE001
    if handler is None or not handler(exc):
      raise
    return
  try:
    with tarfile.open(fileobj=fh, mode="r|*") as tf:
      current, key = None, None
      for member in tf:
        if not member.isfile():
          continue
        k, ext = base_plus_ext(member.name)
        if k is None:
          continue
        if k != key:
          if current:
            yield current
          current, key = {"__key__": k, "__url__": url}, k
        try:
          current[ext] = tf.extractfile(member).read()
        except Exception as exc:  # noqa: BLE001
          if handler is None or not handler(exc):
            raise
      if current:
        yield current
  except Exception as exc:  # noqa: BLE001
    if handler is None or not handler(exc):
      raise
  finally:
    try:
      fh.close()
    except Exception:  # noqa: BLE001
      pass


def tarfile_to_samples(handler=None):
  """Pipeline stage: {'url':...} dicts -> sample dicts."""

  def _stage(src):
    for shard in src:
      url = shard["url"] if isinstance(shard, dict) else shard
      yield from _iter_tar(url, handler=handler)

  return _stage


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------
_IMG_EXT = ("jpg", "jpeg", "png", "webp", "bmp", "ppm")


def _decode_pil(sample):
  from PIL import Image
  out = dict(sample)
  for k, v in sample.items():
    if not isinstance(v, bytes):
      continue
    ext = k.rsplit(".", 1)[-1].lower()
    if ext in _IMG_EXT:
      out[k] = Image.open(io.BytesIO(v)).convert("RGB")
    elif ext == "json":
      import json
      out[k] = json.loads(v)
    elif ext in ("txt", "text", "cls", "cap"):
      out[k] = v.decode("utf-8", "replace")
  return out


# --------------------------------------------------------------------------
# pipelines
# --------------------------------------------------------------------------
class DataPipeline:
  """`wds.DataPipeline(stage, stage, ...)`; each stage is a generator fn."""

  def __init__(self, *stages):
    self.stages = [s for s in stages if s is not None]

  def __iter__(self):
    src = None
    for i, stage in enumerate(self.stages):
      src = stage() if i == 0 else stage(src)
    return iter(src if src is not None else [])


class WebDataset:
  """`wds.WebDataset(urls, resampled=, shardshuffle=, handler=)` + chaining."""

  def __init__(self, urls, resampled=False, shardshuffle=False, handler=None,
               nodesplitter=None, empty_check=True, seed=None, **_kwargs):
    if isinstance(urls, str):
      urls = expand_urls(urls)
    self.urls = list(urls)
    self.resampled = resampled
    self.shardshuffle = shardshuffle
    self.handler = handler
    self.seed = seed
    self._ops = []

  # -- chained ops (each returns self, as upstream does) -------------------
  def shuffle(self, size, rng=None, **_kw):
    self._ops.append(("shuffle", (int(size), rng)))
    return self

  def decode(self, *args, **_kw):
    self._ops.append(("decode", args))
    return self

  def map(self, fn, handler=None):
    self._ops.append(("map", (fn, handler)))
    return self

  def select(self, fn):
    self._ops.append(("select", fn))
    return self

  def with_epoch(self, n):
    self._ops.append(("with_epoch", int(n)))
    return self

  def repeat(self, n=None, **_kw):
    self._ops.append(("repeat", n))
    return self

  # -- iteration -----------------------------------------------------------
  def _shard_urls(self):
    rng = _random.Random(self.seed)
    if self.resampled:
      while True:
        yield rng.choice(self.urls)
    else:
      urls = list(self.urls)
      if self.shardshuffle:
        rng.shuffle(urls)
      yield from urls

  def _raw(self):
    for url in self._shard_urls():
      yield from _iter_tar(url, handler=self.handler)

  def __iter__(self):
    it = self._raw()
    for op, arg in self._ops:
      if op == "shuffle":
        it = _shuffle_iter(it, arg[0], arg[1])
      elif op == "decode":
        it = (_decode_pil(s) for s in it)
      elif op == "map":
        it = _map_iter(it, arg[0], arg[1])
      elif op == "select":
        it = (s for s in it if arg(s))
      elif op == "with_epoch":
        it = _take(it, arg)
      elif op == "repeat":
        pass
    return it


def _map_iter(src, fn, handler):
  for s in src:
    try:
      out = fn(s)
    except Exception as exc:  # noqa: BLE001
      if handler is None or not handler(exc):
        raise
      continue
    if out is not None:
      yield out


def _shuffle_iter(src, size, rng):
  rng = rng or _random.Random()
  buf = []
  for s in src:
    buf.append(s)
    if len(buf) >= size:
      i = rng.randrange(len(buf))
      buf[i], buf[-1] = buf[-1], buf[i]
      yield buf.pop()
  rng.shuffle(buf)
  yield from buf


def _take(src, n):
  for i, s in enumerate(src):
    if i >= n:
      return
    yield s


class RandomMix:
  """`webdataset.filters.RandomMix(datasets, weights)`."""

  def __init__(self, datasets, probs=None, longest=False):
    self.datasets = list(datasets)
    self.probs = list(probs) if probs else [1.0] * len(self.datasets)
    self.longest = longest

  def __iter__(self):
    rng = _random.Random()
    its = [iter(d) for d in self.datasets]
    probs = list(self.probs)
    while its:
      i = rng.choices(range(len(its)), weights=probs, k=1)[0]
      try:
        yield next(its[i])
      except StopIteration:
        its.pop(i)
        probs.pop(i)


class filters:  # noqa: N801 -- mirrors `webdataset.filters.RandomMix`
  RandomMix = RandomMix


class gopen:  # noqa: N801 -- mirrors `webdataset.gopen.gopen_schemes`
  gopen_schemes = gopen_schemes
