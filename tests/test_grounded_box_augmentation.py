"""Grounded-box supervision tests, ported from beifen-Paligemma.

jax_llava emits loc_tokens only, so beifen's coord_format/mae_transform cases and
its OV1.5 conversation-text canonicalization are omitted; everything covering the
structured-grounding adapter and the RefCOCO task routing is kept verbatim.
"""

import random
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import input_pipeline
from configs.load_config import get_config
from input_pipeline import (
    LetterboxPadTransform,
    _box_to_loc_tokens,
    _box_xyxy_to_model_tokens,
    _draw_region_box,
    _grounded_caption_fields,
    _sample_grounded_caption_box_color,
    _sample_grounded_caption_box_mode,
    _sample_refcoco_task_type,
    _with_module_random,
    expand_refcoco_sample,
    preprocess_fn,
)
from utils.data_util import dataset_name_to_path_dict, dataset_name_to_type_dict


class _RecordingTransform:
    def __init__(self, width=20, height=10):
        self.target_width = width
        self.target_height = height
        self.images = []

    def __call__(self, image):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        self.images.append(array)
        return torch.from_numpy(array).permute(2, 0, 1).float()

    def transform_box(self, x1, y1, x2, y2, src_w, src_h):
        return (
            x1 / src_w * self.target_width,
            y1 / src_h * self.target_height,
            x2 / src_w * self.target_width,
            y2 / src_h * self.target_height,
        )


class _Tokenizer:
    def __init__(self):
        self.special_tokens = SimpleNamespace(PAD=0)
        self.texts = []

    def encode(self, text, add_bos=True, add_eos=False):
        self.texts.append(text)
        tokens = [1] if add_bos else []
        tokens.extend(3 + (ord(char) % 251) for char in text)
        if add_eos:
            tokens.append(2)
        return tokens


def _vg_sample(image):
    return {
        "jpg": image,
        "region": {
            "phrase": "a blue ceramic mug",
            "x": 50,
            "y": 20,
            "width": 100,
            "height": 60,
        },
        # Deliberately differs from decoded resolution to exercise scaling.
        "img_w": 200,
        "img_h": 100,
    }


def _run(sample, dataset_type, transform, tokenizer):
    return preprocess_fn(
        sample,
        transform=transform,
        tokenizer=tokenizer,
        max_len=256,
        dataset_type=dataset_type,
    )


def test_drawn_box_branch_marks_the_image_without_mutating_source(monkeypatch):
    source = Image.new("RGB", (20, 10), (10, 20, 30))
    source_before = np.asarray(source).copy()
    transform = _RecordingTransform()
    tokenizer = _Tokenizer()
    monkeypatch.setattr(input_pipeline.random, "random", lambda: 0.499)
    monkeypatch.setattr(input_pipeline.random, "choice", lambda values: values[0])

    output = _run(_vg_sample(source), "genome_gcap", transform, tokenizer)

    assert output is not None
    assert tokenizer.texts[0] == "Describe the region highlighted by the red box.\n"
    assert tokenizer.texts[-1].endswith("a blue ceramic mug")
    assert "<loc" not in tokenizer.texts[0]
    np.testing.assert_array_equal(np.asarray(source), source_before)
    assert np.any(np.all(transform.images[0] == (255, 0, 0), axis=-1))


def test_uniform_boundary_uses_loc_tokens_at_one_half(monkeypatch):
    source = Image.new("RGB", (20, 10), (10, 20, 30))
    transform = _RecordingTransform()
    tokenizer = _Tokenizer()
    monkeypatch.setattr(input_pipeline.random, "random", lambda: 0.5)
    monkeypatch.setattr(input_pipeline.random, "choice", lambda values: values[0])
    monkeypatch.setattr(
        input_pipeline,
        "_sample_grounded_caption_box_color",
        lambda: (_ for _ in ()).throw(AssertionError("loc-token branch sampled a color")),
    )

    output = _run(_vg_sample(source), "genome_gcap", transform, tokenizer)

    assert output is not None
    assert tokenizer.texts[0].startswith("Describe the region <loc0204><loc0255>")
    assert tokenizer.texts[0].endswith("<loc0818><loc0767>.\n")
    assert "red box" not in tokenizer.texts[0]
    assert not np.any(np.all(transform.images[0] == (255, 0, 0), axis=-1))


def test_drawn_box_color_and_prompt_are_synchronized_for_all_colors(monkeypatch):
    source = Image.new("RGB", (20, 10), (10, 20, 30))
    monkeypatch.setattr(input_pipeline.random, "random", lambda: 0.0)
    monkeypatch.setattr(input_pipeline.random, "choice", lambda values: values[0])

    for color_name, color_rgb in input_pipeline._GCAP_DRAWN_BOX_COLORS:
        monkeypatch.setattr(
            input_pipeline,
            "_sample_grounded_caption_box_color",
            lambda sampled=(color_name, color_rgb): sampled,
        )
        transform = _RecordingTransform()
        tokenizer = _Tokenizer()
        output = _run(_vg_sample(source), "genome_gcap", transform, tokenizer)

        assert output is not None
        assert f"highlighted by the {color_name} box" in tokenizer.texts[0]
        assert np.any(np.all(transform.images[0] == color_rgb, axis=-1))


def test_box_color_sampling_is_uniform_over_rgb_primaries(monkeypatch):
    seen = []

    def record_choice(values):
        assert values == input_pipeline._GCAP_DRAWN_BOX_COLORS
        seen.extend(values)
        return values[-1]

    monkeypatch.setattr(input_pipeline.random, "choice", record_choice)
    assert _sample_grounded_caption_box_color() == ("blue", (0, 0, 255))
    assert seen == [
        ("red", (255, 0, 0)),
        ("green", (0, 255, 0)),
        ("blue", (0, 0, 255)),
    ]


def test_single_refcoco_stream_can_emit_gcap_or_detection(monkeypatch):
    source = Image.new("RGB", (20, 10), (0, 0, 0))
    sample = {
        "jpg": source,
        "phrase": "the person on the left",
        "bbox_xyxy": [2, 1, 10, 7],
        "bbox_format": "xyxy_abs",
        "bbox_coord_size": [20, 10],
    }
    monkeypatch.setattr(input_pipeline.random, "random", lambda: 0.0)
    monkeypatch.setattr(input_pipeline.random, "choice", lambda values: values[0])
    monkeypatch.setattr(input_pipeline, "_sample_refcoco_task_type", lambda: "refcoco_gcap")
    gcap_tokenizer = _Tokenizer()

    gcap_output = _run(sample, "refcoco", _RecordingTransform(), gcap_tokenizer)

    assert gcap_output is not None
    assert "the person on the left" not in gcap_tokenizer.texts[0]
    assert gcap_tokenizer.texts[-1].endswith("the person on the left")
    assert "<loc" not in gcap_tokenizer.texts[-1]

    detection_tokenizer = _Tokenizer()
    detection_transform = _RecordingTransform()
    monkeypatch.setattr(input_pipeline, "_sample_refcoco_task_type", lambda: "refcoco")
    detection_output = _run(sample, "refcoco", detection_transform, detection_tokenizer)

    assert detection_output is not None
    assert "the person on the left" in detection_tokenizer.texts[0]
    assert detection_tokenizer.texts[-1].endswith(
        "<loc0102><loc0102><loc0716><loc0511>"
    )
    assert not np.any(np.all(detection_transform.images[0] == (255, 0, 0), axis=-1))


def test_refcoco_expansion_normalizes_legacy_xyxy_and_explicit_xywh(monkeypatch):
    monkeypatch.setattr(input_pipeline.random, "choice", lambda values: values[0])
    image = Image.new("RGB", (20, 10), (0, 0, 0))
    legacy = {
        "jpg": image,
        "json": {
            "dataset": "refcoco",
            "refs": [{"bbox": [2, 1, 10, 7], "sentences": ["left person"]}],
        },
    }
    explicit = {
        "jpg": image,
        "json": {
            "dataset": "refcocog",
            "refs": [{
                "bbox": [2, 1, 8, 6],
                "bbox_format": "xywh",
                "sentences": ["left person"],
            }],
        },
    }

    legacy_item = expand_refcoco_sample(legacy)[0]
    explicit_item = expand_refcoco_sample(explicit)[0]
    assert legacy_item["bbox_xyxy"] == [2.0, 1.0, 10.0, 7.0]
    assert explicit_item["bbox_xyxy"] == [2.0, 1.0, 10.0, 7.0]
    assert legacy_item["bbox_format"] == explicit_item["bbox_format"] == "xyxy_abs"
    assert legacy_item["bbox_coord_size"] == explicit_item["bbox_coord_size"] == [20.0, 10.0]


def test_box_drawing_scales_annotation_canvas_and_rejects_invalid_boxes():
    source = Image.new("RGB", (20, 10), (0, 0, 0))
    marked = _draw_region_box(
        source,
        bbox_xyxy=(50, 20, 150, 80),
        coord_size=(200, 100),
    )

    assert marked is not None
    pixels = np.asarray(marked)
    assert tuple(pixels[2, 5]) == (255, 0, 0)
    assert tuple(pixels[8, 15]) == (255, 0, 0)
    assert tuple(np.asarray(source)[2, 5]) == (0, 0, 0)

    bad_sample = _vg_sample(source)
    bad_sample["region"] = dict(bad_sample["region"], width=float("nan"))
    assert _grounded_caption_fields(bad_sample, "genome_gcap", source.size) is None


def test_letterbox_loc_tokens_and_drawn_box_refer_to_the_same_region():
    source = Image.new("RGB", (20, 10), (0, 0, 0))
    bbox_xywh = (5, 2, 10, 6)
    bbox_xyxy = (5, 2, 15, 8)
    transform = LetterboxPadTransform(20)
    loc = _box_to_loc_tokens(transform, *bbox_xywh, 20, 10)
    assert loc == _box_xyxy_to_model_tokens(transform, bbox_xyxy, (20, 10))
    marked = _draw_region_box(source, bbox_xyxy=bbox_xyxy, coord_size=(20, 10))
    transformed = transform(marked)

    assert loc == "<loc0358><loc0255><loc0664><loc0767>"
    # y=2 receives 5px top letterbox padding; x is unchanged.
    assert float(transformed[0, 7, 5]) > 0.9
    assert float(transformed[1, 7, 5]) < -0.9
    assert float(transformed[2, 7, 5]) < -0.9


def test_refcoco_uses_one_config_alias_per_dataset():
    assert dataset_name_to_type_dict["refcoco-train"] == "refcoco"
    assert dataset_name_to_type_dict["refcocog-train"] == "refcoco"
    assert dataset_name_to_type_dict["refcoco"] == "refcoco"
    assert dataset_name_to_type_dict["refcocog"] == "refcoco"
    assert "refcoco-gcap-train" not in dataset_name_to_type_dict
    assert "refcocog-gcap-train" not in dataset_name_to_type_dict
    assert "refcoco-gcap-train" not in dataset_name_to_path_dict
    assert "refcocog-gcap-train" not in dataset_name_to_path_dict


def test_active_config_uses_one_refcoco_stream_with_doubled_weight():
    config = get_config("remote_run")
    items = list(config.training.stage2.dataset_items)
    weights = list(config.training.stage2.mix_weights)

    assert len(items) == len(weights)
    assert weights[items.index("refcoco-train")] == 0.084
    assert weights[items.index("refcocog-train")] == 0.084
    assert all("-gcap-" not in name for name in items)


def test_refcoco_task_direction_is_uniform_at_one_half(monkeypatch):
    monkeypatch.setattr(input_pipeline.random, "random", lambda: 0.499999)
    assert _sample_refcoco_task_type() == "refcoco_gcap"
    monkeypatch.setattr(input_pipeline.random, "random", lambda: 0.5)
    assert _sample_refcoco_task_type() == "refcoco"


def test_box_mode_sequence_replays_from_stateful_choice_rng_state():
    choice_rng = random.Random(1234)
    for _ in range(7):
        _with_module_random(choice_rng, _sample_grounded_caption_box_mode)
    saved_state = choice_rng.getstate()
    expected = [
        _with_module_random(choice_rng, _sample_grounded_caption_box_mode)
        for _ in range(20)
    ]

    restored_rng = random.Random()
    restored_rng.setstate(saved_state)
    replayed = [
        _with_module_random(restored_rng, _sample_grounded_caption_box_mode)
        for _ in range(20)
    ]

    assert replayed == expected


def test_refcoco_direction_box_mode_and_color_replay_together():
    def sample_supervision():
        task = _sample_refcoco_task_type()
        if task == "refcoco":
            return task, None, None
        mode = _sample_grounded_caption_box_mode()
        color = _sample_grounded_caption_box_color() if mode == "drawn_box" else None
        return task, mode, color

    choice_rng = random.Random(5678)
    for _ in range(9):
        _with_module_random(choice_rng, sample_supervision)
    saved_state = choice_rng.getstate()
    expected = [
        _with_module_random(choice_rng, sample_supervision)
        for _ in range(40)
    ]

    restored_rng = random.Random()
    restored_rng.setstate(saved_state)
    replayed = [
        _with_module_random(restored_rng, sample_supervision)
        for _ in range(40)
    ]

    assert replayed == expected
    assert {task for task, _, _ in expected} == {"refcoco", "refcoco_gcap"}
    assert {color[0] for _, mode, color in expected if mode == "drawn_box"} == {
        "red",
        "green",
        "blue",
    }
