import json

import pytest

from evals.eval_refcocog import _canonical_gt_bbox, _parse_row
from input_pipeline import _refcoco_bbox_xyxy


def test_real_refcocog_val_row_uses_legacy_coco_xywh_adapter():
    path = "/kmh-nfs-ssd-us-mount/code/hanhong/shared/refcocog/val.json"
    with open(path, "r", encoding="utf-8") as handle:
        row = json.load(handle)[0]

    parsed = _parse_row(row, 0)

    assert parsed["gt_bbox_format"] == "xyxy_abs"
    assert parsed["gt_bbox_source_formats"] == ["bbox:xywh_abs"]
    assert parsed["gt_bbox_xyxy"] == pytest.approx(
        [286.760009765625, 233.1699981689453, 351.8400115966797, 469.9199981689453]
    )
    xs = row["segmentation"][0::2]
    ys = row["segmentation"][1::2]
    assert parsed["gt_bbox_xyxy"] == pytest.approx(
        [min(xs), min(ys), max(xs), max(ys)], abs=2e-3
    )


@pytest.mark.parametrize(
    "row",
    [
        {"bbox_xyxy": [10, 20, 50, 80]},
        {"bbox_xywh": [10, 20, 40, 60]},
        {"bbox": [10, 20, 50, 80], "bbox_format": "xyxy_abs"},
        {"bbox": [0.1, 0.2, 0.5, 0.8], "bbox_format": "xyxy_norm", "width": 100, "height": 100},
        {"bbox": {"x": 10, "y": 20, "w": 40, "h": 60}},
        {"bbox": {"x1": 10, "y1": 20, "x2": 50, "y2": 80}},
    ],
)
def test_supported_eval_schemas_converge_to_canonical_xyxy(row):
    box, source_formats = _canonical_gt_bbox(row)
    assert box == pytest.approx([10, 20, 50, 80])
    assert source_formats


def test_conflicting_eval_bbox_fields_fail_loudly():
    with pytest.raises(ValueError, match="Conflicting RefCOCOg bbox fields"):
        _canonical_gt_bbox({
            "bbox_xyxy": [10, 20, 50, 80],
            "bbox_xywh": [10, 20, 10, 10],
        })


def test_conflicting_training_bbox_fields_fail_loudly():
    with pytest.raises(ValueError, match="Conflicting RefCOCO bbox fields"):
        _refcoco_bbox_xyxy(
            {"bbox_xyxy": [10, 20, 50, 80], "bbox_xywh": [10, 20, 10, 10]},
            "refcoco",
        )


def test_untagged_refcoco_bbox_of_an_unknown_dataset_is_never_guessed():
    with pytest.raises(ValueError, match="Ambiguous untagged RefCOCO bbox"):
        _refcoco_bbox_xyxy({"bbox": [10, 20, 50, 80]}, "refcoco+")


@pytest.mark.parametrize(
    "row",
    [
        {"bbox": [10, 20, 40, 60]},
        {"bbox_xyxy": [10, 20, 50, 80]},
        {"bbox_xywh": [10, 20, 40, 60]},
        {"bbox": [10, 20, 50, 80], "bbox_format": "xyxy_abs"},
        {"bbox": [0.1, 0.2, 0.5, 0.8], "bbox_format": "xyxy_norm", "width": 100, "height": 100},
        {"bbox": {"x": 10, "y": 20, "w": 40, "h": 60}},
        {"bbox": {"x1": 10, "y1": 20, "x2": 50, "y2": 80}},
    ],
)
def test_training_and_eval_adapters_read_one_record_identically(row):
    """Both call sites share one adapter, so neither may drift from the other."""
    assert _refcoco_bbox_xyxy(row, "refcocog") == pytest.approx(_canonical_gt_bbox(row)[0])


def test_real_refcocog_val_row_reads_the_same_in_training_and_eval():
    path = "/kmh-nfs-ssd-us-mount/code/hanhong/shared/refcocog/val.json"
    with open(path, "r", encoding="utf-8") as handle:
        row = json.load(handle)[0]

    assert _refcoco_bbox_xyxy(row, "refcocog") == pytest.approx(_canonical_gt_bbox(row)[0])
