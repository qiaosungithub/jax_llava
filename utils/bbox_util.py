"""Canonical bounding-box conversions shared by training and evaluation.

The only internal representation produced by this module is absolute ``xyxy``
coordinates on an explicit ``(width, height)`` canvas. Dataset adapters must
name the source format; this module deliberately never guesses from values.
"""

import math


CANONICAL_BBOX_FORMAT = "xyxy_abs"

_FORMAT_ALIASES = {
    "xyxy": "xyxy_abs",
    "xyxy_abs": "xyxy_abs",
    "absolute_xyxy": "xyxy_abs",
    "xywh": "xywh_abs",
    "xywh_abs": "xywh_abs",
    "absolute_xywh": "xywh_abs",
    "xyxy_norm": "xyxy_norm",
    "normalized_xyxy": "xyxy_norm",
    "norm_xyxy": "xyxy_norm",
    "xywh_norm": "xywh_norm",
    "normalized_xywh": "xywh_norm",
    "norm_xywh": "xywh_norm",
}


def normalize_bbox_format(bbox_format):
    """Return one supported explicit format name or raise on ambiguity."""
    key = "" if bbox_format is None else str(bbox_format).strip().lower()
    if not key:
        raise ValueError("bbox_format is required; coordinate formats must not be guessed")
    try:
        return _FORMAT_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported bbox_format: {bbox_format!r}") from exc


def _box4(box):
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        values = tuple(float(box[i]) for i in range(4))
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def _size2(size, *, required=False):
    if size is None:
        if required:
            raise ValueError("coord_size is required for normalized bbox coordinates")
        return None
    if not isinstance(size, (list, tuple)) or len(size) < 2:
        raise ValueError(f"coord_size must be (width, height), got {size!r}")
    try:
        width, height = float(size[0]), float(size[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid coord_size: {size!r}") from exc
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError(f"Invalid coord_size: {size!r}")
    return width, height


def canonicalize_bbox_xyxy(
    box,
    bbox_format,
    *,
    coord_size=None,
    target_size=None,
    clip=True,
):
    """Convert one explicitly formatted box to canonical absolute ``xyxy``.

    ``coord_size`` is the canvas on which an absolute source box is defined and
    is mandatory for normalized formats. If ``target_size`` is supplied, the
    canonical box is rescaled onto that canvas. Invalid or empty boxes return
    ``None``; missing/unknown formats raise because silently guessing them is a
    data-corruption bug.
    """
    values = _box4(box)
    if values is None:
        return None
    fmt = normalize_bbox_format(bbox_format)
    source_size = _size2(coord_size, required=fmt.endswith("_norm"))
    destination_size = _size2(target_size)

    a, b, c, d = values
    if fmt == "xyxy_abs":
        x1, y1, x2, y2 = a, b, c, d
    elif fmt == "xywh_abs":
        if c <= 0.0 or d <= 0.0:
            return None
        x1, y1, x2, y2 = a, b, a + c, b + d
    elif fmt == "xyxy_norm":
        width, height = source_size
        x1, y1, x2, y2 = a * width, b * height, c * width, d * height
    elif fmt == "xywh_norm":
        if c <= 0.0 or d <= 0.0:
            return None
        width, height = source_size
        x1, y1 = a * width, b * height
        x2, y2 = (a + c) * width, (b + d) * height
    else:  # pragma: no cover - normalize_bbox_format makes this unreachable.
        raise AssertionError(fmt)

    if source_size is not None and destination_size is not None:
        source_w, source_h = source_size
        target_w, target_h = destination_size
        scale_x, scale_y = target_w / source_w, target_h / source_h
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y

    bounds = destination_size or source_size
    if clip and bounds is not None:
        width, height = bounds
        x1 = max(0.0, min(x1, width))
        x2 = max(0.0, min(x2, width))
        y1 = max(0.0, min(y1, height))
        y2 = max(0.0, min(y2, height))

    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


LEGACY_REFCOCO_STORAGE_FORMATS = {
    # The old uploader comment claimed xywh, but jxu124/refcoco actually stored
    # absolute xyxy in the untagged bbox field.
    "refcoco": "xyxy_abs",
    # Older rearranged RefCOCOg metadata used COCO-style xywh.
    "refcocog": "xywh_abs",
}

_BBOX_LIST_FIELDS = (
    ("bbox_xyxy", "xyxy_abs"),
    ("bbox_xywh", "xywh_abs"),
)


def legacy_refcoco_untagged_format(dataset_name):
    """Storage format of an untagged ``bbox`` for one RefCOCO-family dataset."""
    return LEGACY_REFCOCO_STORAGE_FORMATS.get(str(dataset_name or "").strip().lower())


def record_coord_size(record):
    """Read the canvas an absolute box in ``record`` is defined on, if stated."""
    size = record.get("bbox_coord_size") or record.get("image_size")
    if isinstance(size, dict):
        size = (size.get("width"), size.get("height"))
    if size is None:
        width = record.get("image_width", record.get("width"))
        height = record.get("image_height", record.get("height"))
        if width is not None and height is not None:
            size = (width, height)
    return size


def explicit_bbox_format(record, *keys):
    """First non-empty format tag among ``keys``; blank tags count as absent."""
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def collect_bbox_candidates(record, *, untagged_format=None):
    """List every box representation present in ``record`` with its format.

    Returns ``(box, bbox_format, field_name)`` triples. ``untagged_format`` is
    the storage convention for a bare ``bbox``/``box`` list carrying no format
    tag; leaving it ``None`` makes such a field raise downstream rather than be
    guessed from its values.
    """
    candidates = []
    for field, field_format in _BBOX_LIST_FIELDS:
        if record.get(field) is not None:
            candidates.append((record[field], field_format, field))

    box = record.get("bbox")
    if isinstance(box, dict):
        # A dict box states its layout through its key names; only the units of
        # those numbers can still be tagged.
        if all(key in box for key in ("x", "y", "w", "h")):
            candidates.append((
                [box["x"], box["y"], box["w"], box["h"]],
                explicit_bbox_format(box, "bbox_format") or explicit_bbox_format(record, "bbox_format") or "xywh_abs",
                "bbox{x,y,w,h}",
            ))
        elif all(key in box for key in ("x1", "y1", "x2", "y2")):
            candidates.append((
                [box["x1"], box["y1"], box["x2"], box["y2"]],
                explicit_bbox_format(box, "bbox_format") or explicit_bbox_format(record, "bbox_format") or "xyxy_abs",
                "bbox{x1,y1,x2,y2}",
            ))
    elif box is not None:
        candidates.append((box, explicit_bbox_format(record, "bbox_format") or untagged_format, "bbox"))

    if record.get("box") is not None and not isinstance(record["box"], dict):
        candidates.append((
            record["box"],
            explicit_bbox_format(record, "box_format", "bbox_format") or untagged_format,
            "box",
        ))
    if all(key in record for key in ("x", "y", "w", "h")):
        candidates.append((
            [record["x"], record["y"], record["w"], record["h"]], "xywh_abs", "x/y/w/h"))
    if all(key in record for key in ("x1", "y1", "x2", "y2")):
        candidates.append((
            [record["x1"], record["y1"], record["x2"], record["y2"]], "xyxy_abs", "x1/y1/x2/y2"))
    return candidates


def resolve_canonical_bbox(record, *, untagged_format=None, coord_size=None, label="bbox"):
    """Canonical absolute ``xyxy`` agreed on by every box field of one record.

    Returns ``(box, source_formats)``; ``box`` is ``None`` when the record holds
    no usable box. Fields that disagree by more than a pixel raise, because that
    means the source schema was misread rather than that one field is stale.
    """
    canonical = []
    for box, bbox_format, field_name in collect_bbox_candidates(
        record, untagged_format=untagged_format
    ):
        converted = canonicalize_bbox_xyxy(
            box,
            bbox_format,
            coord_size=coord_size,
            clip=False,
        )
        if converted is not None:
            canonical.append((converted, field_name, str(bbox_format)))
    if not canonical:
        return None, []

    expected, expected_field, _ = canonical[0]
    for converted, field_name, _ in canonical[1:]:
        if any(abs(a - b) > 1e-3 for a, b in zip(expected, converted)):
            raise ValueError(
                f"Conflicting {label} bbox fields: "
                f"{expected_field}={expected}, {field_name}={converted}"
            )
    return list(expected), [f"{field}:{fmt}" for _, field, fmt in canonical]


def canonical_bbox_record(box, bbox_format, *, coord_size, target_size=None):
    """Build a serializable canonical bbox record for expanded samples."""
    output_size = target_size or coord_size
    xyxy = canonicalize_bbox_xyxy(
        box,
        bbox_format,
        coord_size=coord_size,
        target_size=target_size,
    )
    if xyxy is None:
        return None
    width, height = _size2(output_size, required=True)
    return {
        "bbox_xyxy": [float(value) for value in xyxy],
        "bbox_format": CANONICAL_BBOX_FORMAT,
        "bbox_coord_size": [float(width), float(height)],
    }
