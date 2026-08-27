'''
File:   test_detection_renderer.py
Brief:  Unit tests for the YOLO-style detection overlay engine.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.3.0
'''

from __future__ import annotations

import numpy as np
import pytest
from comfyui_lukutar_nodes.core.detection_renderer import (
    COLOR_SINGLE,
    LABEL_CONFIDENCE,
    LABEL_TEXT,
    PALETTE,
    Detection,
    DetectionRenderer,
    DetectionRenderError,
    format_caption,
    scale_detections,
)


def _black_frame(height: int = 128, width: int = 128) -> np.ndarray:
    '''All-zero HWC RGB uint8 frame.'''
    return np.zeros((height, width, 3), dtype=np.uint8)


def _box_detection(
    label: str = "face",
    bbox: tuple[float, float, float, float] = (16.0, 40.0, 112.0, 112.0),
    confidence: float | None = 0.9124,
) -> Detection:
    '''Detection without a mask, safe to place anywhere on the frame.'''
    return Detection(bbox=bbox, label=label, confidence=confidence)


def test_rejects_unknown_label_format() -> None:
    with pytest.raises(DetectionRenderError):
        DetectionRenderer(label_format="fancy")


def test_rejects_unknown_color_mode() -> None:
    with pytest.raises(DetectionRenderError):
        DetectionRenderer(color_mode="rainbow")


def test_rejects_out_of_range_options() -> None:
    with pytest.raises(DetectionRenderError):
        DetectionRenderer(mask_alpha=1.5)
    with pytest.raises(DetectionRenderError):
        DetectionRenderer(thickness=-1)
    with pytest.raises(DetectionRenderError):
        DetectionRenderer(font_scale=0.0)
    with pytest.raises(DetectionRenderError):
        DetectionRenderer(single_color=(300, 0, 0))
    with pytest.raises(DetectionRenderError):
        DetectionRenderer(single_color="green")


def test_output_shape_and_dtype() -> None:
    frame = _black_frame()
    result = DetectionRenderer(draw_masks=False).render(
        frame, [_box_detection()]
    )
    assert result.shape == frame.shape
    assert result.dtype == np.uint8


def test_box_is_drawn_and_box_center_untouched() -> None:
    frame = _black_frame()
    renderer = DetectionRenderer(draw_masks=False, thickness=2)
    result = renderer.render(frame, [_box_detection()])
    # Top border of the box, away from the caption plate.
    assert not np.array_equal(result[40, 64], np.zeros(3, np.uint8))
    # Box interior stays clean with masks off.
    assert np.array_equal(result[80, 64], np.zeros(3, np.uint8))


def test_caption_plate_rendered_above_box() -> None:
    frame = _black_frame()
    renderer = DetectionRenderer(draw_masks=False)
    result = renderer.render(frame, [_box_detection(label="face")])
    # Left edge of the plate (x=17 is inside the plate padding zone),
    # above the box top edge (y=40) -- must carry the label color.
    assert tuple(result[30, 17]) == PALETTE[0]


def test_mask_tints_region_and_respects_toggle() -> None:
    frame = _black_frame()
    mask = np.ones((48, 48), dtype=np.float32)
    detection = Detection(
        bbox=(40.0, 40.0, 88.0, 88.0),
        label="face",
        mask=mask,
        mask_origin=(40, 40),
    )
    with_mask = DetectionRenderer(mask_alpha=0.5).render(frame, [detection])
    expected = np.rint(np.asarray(PALETTE[0], np.float32) * 0.5)
    assert np.array_equal(with_mask[64, 64], expected.astype(np.uint8))
    without_mask = DetectionRenderer(draw_masks=False).render(
        frame, [detection]
    )
    assert np.array_equal(without_mask[64, 64], np.zeros(3, np.uint8))


def test_mask_partially_outside_frame_is_clipped() -> None:
    frame = _black_frame()
    detection = Detection(
        bbox=(100.0, 100.0, 127.0, 127.0),
        label="face",
        mask=np.ones((40, 40), dtype=np.float32),
        mask_origin=(100, 100),
    )
    result = DetectionRenderer(mask_alpha=1.0).render(frame, [detection])
    expected = np.asarray(PALETTE[0], np.uint8)
    assert np.array_equal(result[120, 120], expected)
    # A point left of the mask origin, away from box and caption plate,
    # stays untouched.
    assert np.array_equal(result[60, 80], np.zeros(3, np.uint8))


def test_same_label_shares_color_and_labels_differ() -> None:
    frame = _black_frame()
    same = [
        _box_detection(label="a", bbox=(8.0, 40.0, 40.0, 72.0)),
        _box_detection(label="a", bbox=(88.0, 40.0, 120.0, 72.0)),
    ]
    result = DetectionRenderer(draw_masks=False).render(frame, same)
    assert tuple(result[40, 24]) == tuple(result[40, 104])

    distinct = [
        _box_detection(label="a", bbox=(8.0, 40.0, 40.0, 72.0)),
        _box_detection(label="b", bbox=(88.0, 40.0, 120.0, 72.0)),
    ]
    result = DetectionRenderer(draw_masks=False).render(frame, distinct)
    assert tuple(result[40, 24]) != tuple(result[40, 104])


def test_single_color_mode_uses_user_color() -> None:
    frame = _black_frame()
    renderer = DetectionRenderer(
        color_mode=COLOR_SINGLE,
        single_color=(0, 255, 0),
        draw_masks=False,
    )
    result = renderer.render(frame, [_box_detection()])
    assert tuple(result[40, 64]) == (0, 255, 0)


def test_bbox_clipped_at_frame_edge() -> None:
    frame = _black_frame()
    detection = _box_detection(bbox=(-10.0, -10.0, 50.0, 50.0))
    result = DetectionRenderer(draw_masks=False).render(frame, [detection])
    assert result.shape == frame.shape
    assert not np.array_equal(result[0, 25], np.zeros(3, np.uint8))


def test_off_frame_and_degenerate_bboxes_skipped() -> None:
    frame = _black_frame()
    detections = [
        _box_detection(bbox=(130.0, 130.0, 140.0, 140.0)),
        _box_detection(bbox=(40.0, 40.0, 10.0, 10.0)),
    ]
    result = DetectionRenderer(draw_masks=False).render(frame, detections)
    assert np.array_equal(result, frame)


def test_empty_detections_return_untouched_copy() -> None:
    frame = _black_frame()
    result = DetectionRenderer().render(frame, [])
    assert np.array_equal(result, frame)
    assert result is not frame


def test_input_frame_is_not_mutated() -> None:
    frame = _black_frame()
    detection = Detection(
        bbox=(40.0, 40.0, 88.0, 88.0),
        label="face",
        mask=np.ones((48, 48), dtype=np.float32),
        mask_origin=(40, 40),
    )
    DetectionRenderer(mask_alpha=0.5).render(frame, [detection])
    assert np.array_equal(frame, np.zeros_like(frame))


def test_float_frame_is_converted_to_uint8() -> None:
    frame = np.full((64, 64, 3), 100.6, dtype=np.float32)
    result = DetectionRenderer(draw_masks=False).render(frame, [])
    assert result.dtype == np.uint8
    assert np.all(result == 101)


def test_format_caption_variants() -> None:
    assert format_caption("face", None, LABEL_TEXT) == "face"
    assert format_caption("face", None, LABEL_CONFIDENCE) == "face"
    assert format_caption("face", 0.9124, LABEL_TEXT) == "face"
    assert format_caption("face", 0.9124, LABEL_CONFIDENCE) == "face 91%"
    with pytest.raises(DetectionRenderError):
        format_caption("face", 0.9, "fancy")


def test_scale_detections_scales_coordinates_and_masks() -> None:
    detection = Detection(
        bbox=(100.0, 200.0, 300.0, 400.0),
        label="face",
        mask=np.ones((4, 6), dtype=np.float32),
        mask_origin=(100, 200),
    )
    scaled = scale_detections([detection], (400, 600), (200, 300))
    assert len(scaled) == 1
    assert scaled[0].bbox == (50.0, 100.0, 150.0, 200.0)
    assert scaled[0].mask_origin == (50, 100)
    assert scaled[0].mask is not None
    assert scaled[0].mask.shape == (2, 3)
    # The source detection stays untouched.
    assert detection.bbox == (100.0, 200.0, 300.0, 400.0)


def test_scale_detections_rejects_non_positive_size() -> None:
    detection = _box_detection()
    with pytest.raises(DetectionRenderError):
        scale_detections([detection], (0, 0), (200, 300))
