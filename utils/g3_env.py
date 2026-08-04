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
# Only metros we actually have data in are listed. An unlisted cell is an
# ERROR, not a default: silently guessing a region is exactly how a job ends up
# streaming 200 GiB across a continent.
# Verified with `mach_locality -k metro <cell>`, not from memory.
_CELL_TO_METRO = {
    "go": "cmh",
    "yucmhcg": "cmh",
    "yucmhfq": "cmh",
    "yucmhqa": "cmh",
    # TPU cells in the same metro. `go` itself has no TPU capacity for our
    # allocation, so an accelerator job lands in one of these -- still cmh,
    # still us-east5, still co-located with the data on yucmhcg-d.
    "yucmhps": "cmh",
    "yucmhty": "cmh",
}

# //production/borg/cloud_iam/slicer_regions/slicer_metros.pi
_METRO_TO_REGION = {
    "cmh": "us-east5",
}

# The CNS cell to write/read from, per metro. `<cell>-d` is the durable root.
_METRO_TO_CNS_CELLS = {
    "cmh": ("yucmhcg-d",),
}

# Where the datasets live. Keyed by CNS cell root so a job in another cell of
# the same metro picks its own replica rather than reaching across the metro.
#
# There is deliberately ONE cmh replica. A second copy existed on `go-d` and
# was deleted, and the reason is worth keeping: Colossus quota is charged on
# POST-REPLICATION disk bytes. That copy was written with the default `r=3.2`
# (3-way replication, 3.0166x), so 199 GiB of cc12m occupied 598.7 GiB against
# a 500 GiB per-user-per-cell disk limit -- the cell went over quota and every
# write to it failed with "Poisoned file handle ... over Colossus bytes HDD
# quota". The surviving replica is written `rs=9.4` (Reed-Solomon, 1.4505x =
# 289 GiB) and fits comfortably. Same bytes, same shard count, 2.08x less disk.
#
# Do NOT add a root back as a "read-only fallback" unless its payload is
# verified present: a fallback that resolves to an empty directory is worse
# than no fallback, because it looks like a valid root and then yields a
# partial stream. `cns_dataset_path()` additionally requires a _SUCCESS marker
# for exactly this reason.
#
# Check with `fileutil quota qiaos <cell>` and `fileutil ls -le <path>`.
_CNS_DATA_ROOTS = {
    "yucmhcg-d": "/cns/yucmhcg-d/home/qiaos/data",
}

# Pretrained third-party weights (CLIP). Unlike the dataset these are small and
# read once at startup, so the list is tried in order and a same-metro read is
# acceptable. Gemma is absent on purpose: it comes from /tfhub/, which is
# globally addressable and needs no replica choice.
_CNS_MODEL_ROOTS = {
    "cmh": ("/cns/yucmhcg-d/home/qiaos/models",),
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
        raise ValueError(f"No CNS data root registered for metro {metro!r}.")
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
            if module is not None and hasattr(module, DEFERRED_TOPOLOGY_MARKER)
        ]
    bound = []
    for module in modules:
        names = getattr(module, DEFERRED_TOPOLOGY_MARKER, ())
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
