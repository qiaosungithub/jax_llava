"""Where this process is running, and which storage is local to it.

jax_llava was written for TPU VMs in GCP: compute sits in a zone like
`us-east5`, data sits in the bucket `gs://kmh-gcp-us-east5`, and the two are
kept together by the `💣` placeholder in `utils/data_util.py`.

Under google3 the same job runs on Borg. Compute sits in a *cell* (`go`),
storage is Colossus (`/cns/go-d/...`), and the placeholder machinery has
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
_CELL_TO_METRO = {
    "go": "cmh",
    "yucmhcg": "cmh",
    "yucmhfq": "cmh",
    "yucmhqa": "cmh",
}

# //production/borg/cloud_iam/slicer_regions/slicer_metros.pi
_METRO_TO_REGION = {
    "cmh": "us-east5",
}

# The CNS cell to write/read from, per metro. `<cell>-d` is the durable root.
_METRO_TO_CNS_CELLS = {
    "cmh": ("go-d", "yucmhcg-d"),
}

# Where the datasets were copied. Keyed by CNS cell root so a job in another
# cmh cell picks its own replica rather than reaching across the metro.
_CNS_DATA_ROOTS = {
    "go-d": "/cns/go-d/home/qiaos/data",
    "yucmhcg-d": "/cns/yucmhcg-d/home/qiaos/data",
}


def borg_cell():
    """The Borg cell this task runs in, or None off Borg."""
    for name in ("BORG_PHYSICAL_CELL", "BORG_CELL"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
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
    """The CNS cell roots (e.g. 'go-d') that are local to a jax_llava zone."""
    for metro, region in _METRO_TO_REGION.items():
        if region == zone:
            return _METRO_TO_CNS_CELLS.get(metro, ())
    return ()


def cns_cell_of_path(path):
    """'/cns/go-d/home/...' -> 'go-d'. None for anything else."""
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
    for cns_cell in candidates:
        root = _CNS_DATA_ROOTS.get(cns_cell)
        if root:
            return root
    raise ValueError(f"No CNS data root registered for metro {metro!r}.")


def resolve_data_root():
    """The dataset root for this process, honouring an explicit override.

    Order: `$JAX_LLAVA_DATA_ROOT` (a full path, for local debugging and for
    pointing a run at a replica), then `$JAX_LLAVA_CNS_CELL` (a cell root such
    as `go-d`), then the Borg cell.
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


def infer_zone_from_environment():
    """The jax_llava `zone` string implied by where this task is running.

    Returns None off Borg / in an unknown cell, so the caller can fall back to
    parsing the workdir the way the GCP path always has.
    """
    override = (os.environ.get("JAX_LLAVA_ZONE") or "").strip()
    if override:
        return override
    return region_of_cell(borg_cell())
