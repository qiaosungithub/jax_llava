"""Where this process is running, and which storage is local to it.

jax_llava was written for TPU VMs in GCP: compute sits in a zone like
`us-east5`, data sits in the bucket `gs://kmh-gcp-us-east5`, and the two are
kept together by the `💣` placeholder in `utils/data_util.py`.

Under google3 the same job runs on Borg. Compute sits in a *cell* (`go`),
storage is Colossus (`/cns/<cell>-d/...`), and the placeholder machinery has
nothing to substitute. This module is the translation layer, and it is
deliberately the ONLY place that knows the mapping:

    Borg cell  ->  metro  ->  GCP region  ->  jax_llava "zone"
        go     ->  cmh    ->  us-east5    ->  us-east5

The metro-to-region relation is verified against
`//production/borg/cloud_iam/slicer_regions/slicer_metros.pi` ("CMH":
"us-east5"), not from memory. Keeping it in one table means the fail-closed
locality guard in `input_pipeline.py` and the data-root resolution in
`utils/data_util.py` cannot drift apart.

Nothing here touches the filesystem, so it is safe to import at module scope
(`/cns/` and `/bigstore/` CHECK-fail before InitGoogle(); reading environment
variables does not).
"""

import os


def in_google3() -> bool:
    """True when this binary was built by Blaze rather than run from a checkout.

    `google3.pyglib` only exists inside a Bazel py_binary's runfiles, so its
    importability is the fact itself rather than a proxy for it. Note this is
    an import check, not a filesystem check.
    """
    global _IN_GOOGLE3
    if _IN_GOOGLE3 is None:
        try:
            import google3.pyglib.gfile  # noqa: F401  pylint: disable=unused-import
            _IN_GOOGLE3 = True
        except ImportError:
            _IN_GOOGLE3 = False
    return _IN_GOOGLE3


_IN_GOOGLE3 = None


# ---------------------------------------------------------------------------
# cell -> metro -> region
# ---------------------------------------------------------------------------
# Only cells we can actually name are listed. An unlisted cell is an ERROR,
# not a default: silently guessing a region is exactly how a job ends up
# streaming 200 GiB across a continent.
# Verified with `mach_locality -k metro <cell>`, not from memory.
_CELL_TO_METRO = {
    # --- cmh: where the data lives, and where compute now goes -------------
    # `go` is the compute cell this project pins (`tpu queue --cell=go`), and
    # `go-d` is the Colossus cell it reads and checkpoints to. Same metro, and
    # `mach_locality -k campus` says nby vs clb for go-d and yucmhcg-d -- two
    # campuses of one metro, which storage.md's boundary treats as neighbours.
    "go": "cmh",
    "yucmhcg": "cmh",
    "yucmhfq": "cmh",
    "yucmhqa": "cmh",
    "yucmhps": "cmh",
    "yucmhty": "cmh",
    # --- other metros where the GROUP has Colossus headroom ----------------
    # Registered so a job that lands in one gets a legible "no dataset replica
    # in this metro" failure naming the metro, instead of the far vaguer
    # "Cannot determine where this task is running".
    "oe": "tul",
    "nz": "cbf",
    "rs": "dfw",
    "ej": "grq",
    "yuphxrp": "phx",
    # NOT a data metro, and not a quota metro either: `yuskedq-d` has no group
    # registration, so it is capped at the personal 500 GiB per-cell ceiling.
    # It has TPU capacity our allocation can obtain, which is why the smoke
    # runs landed here and read cc12m across the Atlantic. `ske` is therefore
    # absent from `_METRO_TO_CNS_CELLS` on purpose, and a long run must not
    # come back here.
    "yuskedq": "ske",
}

# //production/borg/cloud_iam/slicer_regions/slicer_metros.pi, read from the
# depot rather than from memory:
#   "CMH": "us-east5"   "CBF": "us-central1"   "TUL": "us-central2"
#   "DFW": "us-south1"  "GRQ": "europe-west4"  "PHX": "us-west8"
#
# `ske` is absent ON PURPOSE even though `_CELL_TO_METRO` knows the cell. This
# map is what `infer_zone_from_environment()` and the dataset locality guard
# compare against, and there is no jax_llava data or verified GCP region
# mapping for ske. A guessed entry would let a cross-metro read pass the guard
# silently, which is the one thing the guard exists to prevent.
#
# SECOND GATE, and it is not in this file: `train.py::_init_run` asserts the
# resolved zone is one of us-central1 / us-east5 / asia-northeast1-b. So a job
# in tul, dfw, grq or phx resolves a region here and then stops there, with a
# message naming what is supported. That is fail-closed and legible, but it
# means group Colossus headroom in a metro is necessary and NOT sufficient --
# a new training metro needs that assert widened and a dataset replica placed,
# in that order.
_METRO_TO_REGION = {
    "cmh": "us-east5",
    "cbf": "us-central1",
    "tul": "us-central2",
    "dfw": "us-south1",
    "grq": "europe-west4",
    "phx": "us-west8",
}

# The metros this project may run in, and the CNS cell it reads/writes there.
#
# THE ENTRY CRITERION IS GROUP QUOTA, NOT PROXIMITY. Every cell listed here is
# registered to `deepmind-resources-colossus` in the flex registry, which is
# the only authoritative source -- `fileutil quota <group> <cell>` reports a
# plausible-looking 500.00G for an UNREGISTERED group because that is the
# default bucket it falls through to, and writing against that reply dies with
# "Group <g> has no quota (partition=hdd)" plus a poisoned file handle, which
# is strictly worse than not setting the accounting at all.
#
# Verified group headroom per metro, with
#   flex.par list_ceiling -s colossus -g deepmind-resources-colossus -l <cell>
#   fileutil quota deepmind-resources-colossus <cell>
#
#   cmh  go-d 5.40 PiB, yucmhty-d 5.16 PiB    tul  oe-d       3.87 PiB
#   cbf  nz-d 6.06 PiB                        dfw  rs-d       9.51 PiB
#   grq  ej-d 4.50 PiB                        phx  yuphxrp-d  94 TiB
#
# DELIBERATELY EXCLUDED, and these are the load-bearing omissions:
#
#   ske / yuskedq-d -- no group registration, so it is capped at the PERSONAL
#     500 GiB per-cell ceiling. cc12m alone is 289 GiB at rs=9.4 and the group
#     is at 0 B there, so a replica would fit exactly once and leave no room
#     for checkpoints. This is why the smoke runs read cross-metro and why
#     compute moved to cmh rather than the data moving to ske.
#   yucmhcg-d, yucbfpv-d, yuchspe-d -- likewise no group registration. Note
#     yucmhcg-d is where the data lives TODAY: it is readable and stays
#     readable, it is simply not somewhere to keep growing (measured 398 GiB
#     of a 500 GiB personal ceiling, shared with every run's checkpoints).
#
# An unlisted metro raises. That is the whole point: silently guessing a
# region is how a job ends up streaming 200 GiB across a continent.
_METRO_TO_CNS_CELLS = {
    "cmh": ("go-d", "yucmhcg-d"),
    "tul": ("oe-d",),
    "cbf": ("nz-d",),
    "dfw": ("rs-d",),
    "grq": ("ej-d",),
    "phx": ("yuphxrp-d",),
}

# Where the datasets live, keyed by CNS cell root, so a job in one cell of a
# metro picks ITS OWN replica rather than reaching across the metro.
#
# A root appears here only when its payload is verified present. A fallback
# that resolves to an empty directory is worse than no fallback, because it
# looks like a valid root and then yields a partial stream -- so
# `cns_dataset_path()` additionally requires a `_SUCCESS` marker before using
# one, and the order below is a preference, not a promise.
#
# `go-d` is FIRST for cmh: same metro as `yucmhcg-d` (both cmh, campuses nby
# and clb, so cross-cell reads there are effectively free), but charged to the
# group instead of to a 500 GiB personal ceiling, and holding
# throughput_spindles=50 -- an actual performance floor. A default quota circle
# carries ZERO spindle commitment, which is the documented condition behind a
# 12-hour throughput collapse to 0.04 MiB/s.
#
# Colossus quota is charged on POST-REPLICATION disk bytes, so the encoding
# decides whether 199 GiB of cc12m costs 289 GiB (`rs=9.4`, 1.4505x) or
# 601 GiB (`r=3.2`, 3.0166x). Against a group ceiling that is an efficiency
# question; against the personal one it is feasibility, and getting it wrong
# once already poisoned a cell.
_CNS_DATA_ROOTS = {
    "go-d": "/cns/go-d/home/qiaos/data",
    "yucmhcg-d": "/cns/yucmhcg-d/home/qiaos/data",
}

# Pretrained third-party weights (CLIP). Unlike the dataset these are small and
# read once at startup, so the list is tried in order and a same-metro read is
# acceptable. Gemma is absent on purpose: it comes from /tfhub/, which is
# globally addressable and needs no replica choice.
# `go-d` first for the same reason as the dataset -- group quota and a real
# spindle commitment -- with `yucmhcg-d` kept as a second entry because a model
# read is small, once, at startup, and having a working fallback beats
# refusing to start. That is deliberately the opposite of the dataset rule,
# where a wrong choice means streaming 200 GiB for hours.
_CNS_MODEL_ROOTS = {
    "cmh": ("/cns/go-d/home/qiaos/models",
            "/cns/yucmhcg-d/home/qiaos/models"),
}


def borg_cell():
    """The Borg cell this task runs in, or None off Borg.

    `BORG_CELL` / `BORG_PHYSICAL_CELL` are the documented sources
    (`borg/borgletlib/python/pyborgletinfo.py` reads exactly these), but they
    are not guaranteed to be exported into every container, and a job that
    depends on them dies at startup with no log when they are missing. So this
    also parses the task handle, whose shape is
    `<task>.<job>.<user>.<cell>.<uid>` -- the cell is in there even when the
    convenience variable is not.
    """
    for name in ("BORG_PHYSICAL_CELL", "BORG_CELL"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    # e.g. "0.qiaos_group_276839294.1.main.qiaos.yucmhps.42"; take the last
    # field we recognise rather than a fixed index, since the job-name segment
    # itself contains dots.
    handle = (os.environ.get("BORG_TASK_HANDLE") or "").strip()
    for part in reversed(handle.split(".")):
        if part in _CELL_TO_METRO:
            return part
    return None


def metro_of_cell(cell):
    return _CELL_TO_METRO.get((cell or "").strip())


def region_of_cell(cell):
    """GCP region for a Borg cell, or None if we do not know the cell.

    None means "cannot tell", which every caller must treat as fatal rather
    than as permission to proceed.
    """
    metro = metro_of_cell(cell)
    return _METRO_TO_REGION.get(metro) if metro else None


def cns_cells_for_zone(zone):
    """The CNS cell roots (e.g. 'yucmhcg-d') local to a jax_llava zone."""
    for metro, region in _METRO_TO_REGION.items():
        if region == zone:
            return _METRO_TO_CNS_CELLS.get(metro, ())
    return ()


def cns_cell_of_path(path):
    """'/cns/yucmhcg-d/home/...' -> 'yucmhcg-d'. None for anything else."""
    if not isinstance(path, str) or not path.startswith("/cns/"):
        return None
    rest = path[len("/cns/"):]
    return rest.split("/", 1)[0] or None


def cns_data_root(cell=None):
    """The dataset root local to `cell` (default: this task's cell).

    Raises rather than guessing: an unknown cell means we cannot prove
    locality, and a wrong guess is a cross-region read.
    """
    cell = cell or borg_cell()
    if not cell:
        raise ValueError(
            "No Borg cell in the environment ($BORG_CELL / "
            "$BORG_PHYSICAL_CELL); cannot choose a co-located data root. "
            "Set JAX_LLAVA_CNS_CELL to pick one explicitly for local runs."
        )
    metro = metro_of_cell(cell)
    if metro is None:
        raise ValueError(
            f"Unknown Borg cell {cell!r}: refusing to guess which CNS cell is "
            f"local to it. Known cells: {sorted(_CELL_TO_METRO)}."
        )
    candidates = _METRO_TO_CNS_CELLS.get(metro, ())
    roots = [_CNS_DATA_ROOTS[c] for c in candidates if c in _CNS_DATA_ROOTS]
    if not roots:
        raise ValueError(f"No CNS data root registered for metro {metro!r}.")
    return roots[0]


def cns_data_roots(cell=None):
    """Every dataset root local to `cell`, in preference order.

    A list rather than a single answer because completeness is a property of
    the replica, not of the cell: the caller picks the first one that carries
    its completion marker.
    """
    explicit = (os.environ.get("JAX_LLAVA_DATA_ROOT") or "").strip()
    if explicit:
        return (explicit.rstrip("/"),)
    cell_override = (os.environ.get("JAX_LLAVA_CNS_CELL") or "").strip()
    if cell_override:
        root = _CNS_DATA_ROOTS.get(cell_override)
        if root is None:
            raise ValueError(
                f"JAX_LLAVA_CNS_CELL={cell_override!r} has no registered data "
                f"root. Known: {sorted(_CNS_DATA_ROOTS)}."
            )
        return (root,)
    cell = cell or borg_cell()
    metro = metro_of_cell(cell) if cell else None
    if metro is None:
        # No legible cell. Fall back to the zone, which can also be derived
        # from $CHECKPOINT_BUCKET -- see infer_zone_from_environment(). This is
        # still fail-closed: an unknown zone raises, and whichever root we pick
        # is re-checked against the zone by the dataset locality guard before a
        # single byte is read.
        zone = infer_zone_from_environment()
        metro = next((m for m, r in _METRO_TO_REGION.items() if r == zone), None)
        if metro is None:
            raise ValueError(
                f"Cannot determine where this task is running (cell={cell!r}, "
                f"zone={zone!r}, handle="
                f"{os.environ.get('BORG_TASK_HANDLE', '')!r}), so it is not "
                "possible to choose a co-located data root. Known cells: "
                f"{sorted(_CELL_TO_METRO)}. Set JAX_LLAVA_CNS_CELL or "
                "JAX_LLAVA_DATA_ROOT to be explicit."
            )
    roots = tuple(_CNS_DATA_ROOTS[c] for c in _METRO_TO_CNS_CELLS.get(metro, ())
                  if c in _CNS_DATA_ROOTS)
    if not roots:
        # LAST RESORT, and only because the alternative is worse. A job in a
        # compute-only metro (ske, today) is a legitimate situation -- it is
        # where the chips our allocation can actually obtain are -- and the
        # launcher offers no way to pass JAX_LLAVA_CNS_CELL into the job
        # (`~/work/tpu_cmd/xm_launcher.py` builds `job_env_vars` from a fixed
        # list, and it is shared with other projects, so it is not ours to
        # edit).
        #
        # $CHECKPOINT_BUCKET, however, IS passed, and it is chosen explicitly
        # at submit time (`--bucket=/cns/yucmhcg-d/...`). A human naming a
        # durable root in cmh while scheduling compute in ske has already made
        # the cross-metro decision; honouring it is reporting that decision,
        # not guessing. Anything else -- an unset bucket, a bucket in a metro
        # with no registered data -- still raises.
        #
        # It is deliberately LOUD: co-location is the thing this module exists
        # to protect, and a cross-metro read must never be silent. See
        # storage.md for the incident where compute and checkpoints on
        # different continents cost 4-5x throughput and got the job pruned.
        bucket_cell = cns_cell_of_path(
            (os.environ.get("CHECKPOINT_BUCKET") or "").strip())
        if bucket_cell and bucket_cell in _CNS_DATA_ROOTS:
            bucket_metro = next(
                (m for m, cells in _METRO_TO_CNS_CELLS.items()
                 if bucket_cell in cells), None)
            message = (
                f"CROSS-METRO DATA READ: compute is in metro {metro!r} "
                f"(cell={cell!r}), which has no dataset replica, so falling "
                f"back to {_CNS_DATA_ROOTS[bucket_cell]} in metro "
                f"{bucket_metro!r} -- taken from $CHECKPOINT_BUCKET, i.e. from "
                "an explicit --bucket at submit time. Acceptable for a short "
                "smoke; for a long run replicate the data into the compute "
                "metro or move the compute, or throughput drops several-fold "
                "and the utilisation pruner eventually kills the job."
            )
            # NOT `warnings.warn`: warnings are deduplicated, are silenced by
            # whatever filter the process last installed, and did not appear
            # at all in a Blaze binary when this was tested. A message that
            # can be swallowed is not a warning. absl logging reaches both the
            # console and -- via the record-level handler in
            # `utils/logging_util.py` -- the durable CNS mirror, which on Borg
            # is the only log anyone can read.
            try:
                from absl import logging as _absl_logging
                _absl_logging.warning("%s", message)
            except Exception:  # noqa: BLE001 - never block on reporting
                pass
            print(f"[locality] {message}", flush=True)
            return (_CNS_DATA_ROOTS[bucket_cell],)
        raise ValueError(
            f"No CNS data root registered for metro {metro!r} (cell={cell!r}). "
            f"Metros with data: {sorted(_METRO_TO_CNS_CELLS)}. Either replicate "
            "the dataset into this metro, or set JAX_LLAVA_CNS_CELL="
            "<cell>-d / JAX_LLAVA_DATA_ROOT=<path> to read across metros "
            "deliberately -- which costs throughput and, on a long run, gets "
            "the job killed by the utilisation pruner."
        )
    return roots


def resolve_data_root():
    """The dataset root for this process, honouring an explicit override.

    Order: `$JAX_LLAVA_DATA_ROOT` (a full path, for local debugging and for
    pointing a run at a replica), then `$JAX_LLAVA_CNS_CELL` (a cell root such
    as `yucmhcg-d`), then the Borg cell.
    """
    explicit = (os.environ.get("JAX_LLAVA_DATA_ROOT") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    cell_override = (os.environ.get("JAX_LLAVA_CNS_CELL") or "").strip()
    if cell_override:
        root = _CNS_DATA_ROOTS.get(cell_override)
        if root is None:
            raise ValueError(
                f"JAX_LLAVA_CNS_CELL={cell_override!r} has no registered data "
                f"root. Known: {sorted(_CNS_DATA_ROOTS)}."
            )
        return root
    return cns_data_root()


def cns_model_roots(cell=None):
    """CNS directories to search for pretrained weights, nearest metro first.

    Falls back to every registered root when the cell is unknown -- a model
    read is a few hundred MB, once, at startup, so "slower but correct" beats
    "refuses to start". That is deliberately the opposite of the dataset rule,
    where a wrong choice means streaming 200 GiB across a region for hours.
    """
    override = (os.environ.get("JAX_LLAVA_MODEL_ROOT") or "").strip()
    if override:
        return (override.rstrip("/"),)
    metro = metro_of_cell(cell or borg_cell())
    if metro and metro in _CNS_MODEL_ROOTS:
        return _CNS_MODEL_ROOTS[metro]
    return tuple(r for roots in _CNS_MODEL_ROOTS.values() for r in roots)


def cns_dir_exists(path):
    """True if `path` exists on CNS. MUST be called after InitGoogle()."""
    try:
        from google3.pyglib import gfile
        return bool(gfile.Exists(path))
    except Exception:  # noqa: BLE001 - an unreadable path is not an existing one
        return False


def cns_file_size(path):
    """Size of `path` in bytes, or None if it is absent or unreadable.

    Existence is a weaker claim than it looks: a copy that is mid-flight or
    that hit a quota wall leaves files that exist, are readable, and are the
    wrong length. Callers that care whether a payload is COMPLETE must ask for
    the size.
    """
    try:
        from google3.pyglib import gfile
        return int(gfile.Stat(path).length)
    except Exception:  # noqa: BLE001
        return None


def describe_placement():
    """A short, never-raising summary of where this process thinks it is.

    Logged at startup. Placement errors otherwise surface much later as a
    message about workdirs or buckets that names neither the cell nor the
    reason, which is the hardest kind of failure to read from a Borg job whose
    logs you cannot fetch.
    """
    try:
        roots = cns_data_roots()
    except Exception as exc:  # noqa: BLE001
        roots = f"<unresolved: {exc}>"
    return {
        "cell": borg_cell(),
        "metro": metro_of_cell(borg_cell()),
        "zone": infer_zone_from_environment(),
        "data_roots": roots,
        "handle": os.environ.get("BORG_TASK_HANDLE", ""),
    }


def infer_zone_from_environment():
    """The jax_llava `zone` string implied by where this task is running.

    Returns None off Borg / in an unknown cell, so the caller can fall back to
    parsing the workdir the way the GCP path always has.
    """
    override = (os.environ.get("JAX_LLAVA_ZONE") or "").strip()
    if override:
        return override
    zone = region_of_cell(borg_cell())
    if zone:
        return zone
    # Last resort: the durable root the launcher handed us. $CHECKPOINT_BUCKET
    # is a /cns/<cell>-d/... path chosen at submit time and is present in every
    # launched job, so it pins the region even when the cell is not legible
    # from the environment. It is a weaker claim than the cell -- it says where
    # our STORAGE is, and co-location is then an assumption rather than a
    # measurement -- but the alternative is dying at startup with a message
    # about workdirs, which helps nobody. The mismatch is caught anyway: the
    # dataset locality guard compares the resolved roots against this zone.
    bucket_cell = cns_cell_of_path(
        (os.environ.get("CHECKPOINT_BUCKET") or "").strip())
    if bucket_cell:
        for metro, cells in _METRO_TO_CNS_CELLS.items():
            if bucket_cell in cells:
                return _METRO_TO_REGION.get(metro)
    return None


# ---------------------------------------------------------------------------
# Deferred topology constants
# ---------------------------------------------------------------------------
#
# `train.py` and `evals/eval_imagenet_knn.py` each define four module-level
# constants -- LDC / PRC / PRI / GDC -- by calling `jax.local_device_count()`
# and friends at import time. Under google3 that is fatal: JAX refuses to
# answer before `absl.app.run()` has run InitGoogle(), so the binary dies
# during `import`, before main() and before any logging exists. On Borg that
# surfaces as an empty status message and no log at all.
#
# The obvious repair -- a PEP 562 module `__getattr__` -- DOES NOT WORK here,
# and the way it fails is worth recording. `__getattr__` is consulted only for
# *attribute* access on the module object (`train.GDC`). A bare name inside a
# function in that same module compiles to LOAD_GLOBAL, which looks in the
# module dict and then in builtins and then raises NameError -- it never
# consults `__getattr__`. So `train.GDC` from outside resolved fine while the
# 50 in-module uses all raised
#
#   NameError: name 'GDC' is not defined
#
# ...at the first real training step, i.e. deep enough to look like a
# different bug. The lesson: PEP 562 defers a module's EXPORTS, not its
# INTERNALS.
#
# What does work is to leave the names genuinely absent during import and
# BIND them, once, from inside main(), when JAX is legal. Modules opt in with
# a `_DEFERRED_TOPOLOGY_NAMES` marker so this stays a declaration by the
# module rather than a list maintained at a distance.

DEFERRED_TOPOLOGY_MARKER = "_DEFERRED_TOPOLOGY_NAMES"


def topology_values():
    """{LDC, PRC, PRI, GDC} from JAX. Only legal after InitGoogle()."""
    import jax  # local: importing jax is fine, CALLING it early is not.

    ldc = int(jax.local_device_count())
    prc = int(jax.process_count())
    gdc = int(jax.device_count())
    if gdc != ldc * prc:
        raise ValueError(
            f"Inconsistent JAX topology: device_count={gdc} != "
            f"local_device_count={ldc} * process_count={prc}")
    return {"LDC": ldc, "PRC": prc, "GDC": gdc, "PRI": int(jax.process_index())}


def _declared_topology_names(module):
    """The marker tuple a module declares, or () -- via `__dict__`, on purpose.

    `getattr`/`hasattr` are the wrong probes for a scan over `sys.modules`.
    Some module-like objects answer *every* attribute name from a custom
    `__getattr__`: `torch.classes` is a `_ClassNamespace` that hands back
    another namespace for anything you ask, so `hasattr(m, MARKER)` is True
    for it and the scan then tried to iterate a namespace --

      TypeError: '_ClassNamespace' object is not iterable

    Reading `__dict__` asks the only question that is actually meaningful
    here -- "did this module really assign that name at module level?" -- and
    is immune to descriptor and PEP 562 magic. The type check then keeps a
    coincidental same-named attribute from turning into a silent no-op.
    """
    namespace = getattr(module, "__dict__", None)
    if not isinstance(namespace, dict):
        return ()
    names = namespace.get(DEFERRED_TOPOLOGY_MARKER)
    if not isinstance(names, (tuple, list)):
        return ()
    if not all(isinstance(name, str) for name in names):
        raise ValueError(
            f"Module {namespace.get('__name__', module)!r} declares a "
            f"{DEFERRED_TOPOLOGY_MARKER} that is not a tuple of str: {names!r}")
    return tuple(names)


def bind_topology_constants(modules=None):
    """Bind the deferred topology constants into every opted-in module.

    Call once from `main()`, after `handle_main` has run InitGoogle() and
    after any distributed initialisation, but BEFORE the first use. Returns
    the values it bound, for logging.

    Idempotent, and deliberately NOT silent about a module that asks for a
    name this does not know -- a typo in the marker would otherwise reproduce
    exactly the NameError this exists to prevent.
    """
    import sys as _sys

    values = topology_values()
    if modules is None:
        modules = [
            module for module in list(_sys.modules.values())
            if _declared_topology_names(module)
        ]
    bound = []
    for module in modules:
        names = _declared_topology_names(module)
        unknown = [name for name in names if name not in values]
        if unknown:
            raise ValueError(
                f"Module {getattr(module, '__name__', module)!r} declares "
                f"unknown deferred topology names {unknown}; "
                f"known names are {sorted(values)}")
        for name in names:
            setattr(module, name, values[name])
        if names:
            bound.append(getattr(module, "__name__", str(module)))
    values["_bound_in"] = bound
    return values
