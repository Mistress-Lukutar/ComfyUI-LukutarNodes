'''
File:   test_segs_label_filter.py
Brief:  Unit tests for the SEGS label assignment + numeric-index filter engine.
Author: Mistress-Lukutar
Date:   2026-11-18
Version: v0.1.0

Tests parse_index_filter, parse_labels, split_segs, assign_seg_labels and
build_segment_previews from core/segs_label_filter.py.  Torch-free; uses
numpy (preview tests go through draw_corner_caption).
'''

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np
import pytest
from comfyui_lukutar_nodes.core.detection_renderer import PALETTE
from comfyui_lukutar_nodes.core.segs_label_filter import (
    DIM_FACTOR,
    DIM_GRAY,
    MIN_ALPHA,
    assign_seg_labels,
    build_segment_previews,
    parse_index_filter,
    parse_labels,
    split_segs,
)

# ======================================================================
# Fixture helpers                                                        #
# ======================================================================


def _seg_factory(
    label: str = "face",
    bbox: list[float] | None = None,
    crop_region: list[int] | None = None,
    cropped_image: np.ndarray | None = None,
) -> collections.namedtuple:
    '''Build a single Impact-Pack-style SEG namedtuple.

    Args:
        label: The segment label string.
        bbox: Optional [x1, y1, x2, y2]; defaults to [0.0, 0.0, 16.0, 16.0].
        crop_region: Optional [x, y, w, h]; defaults to [0, 0, 32, 32].
        cropped_image: Optional crop source handed to the preview builder.

    Returns:
        A SEG namedtuple instance.
    '''
    if bbox is None:
        bbox = [0.0, 0.0, 16.0, 16.0]
    if crop_region is None:
        crop_region = [0, 0, 32, 32]
    seg_type = collections.namedtuple(
        "SEG",
        [
            "cropped_image",
            "cropped_mask",
            "confidence",
            "crop_region",
            "bbox",
            "label",
            "control_net_wrapper",
        ],
    )
    return seg_type(
        cropped_image=cropped_image,
        cropped_mask=None,
        confidence=np.array([0.9], dtype=np.float32),
        crop_region=crop_region,
        bbox=bbox,
        label=label,
        control_net_wrapper=None,
    )


def _fake_segs(
    labels: list[str] | None = None,
    bboxes: list[list[float]] | None = None,
) -> tuple:
    '''Build a SEGS payload ((h, w), [SEG, ...]) from label/bbox lists.

    Args:
        labels: Per-segment label strings; defaults to ["face"].
        bboxes: Per-segment bbox lists; defaults to one per-label slot.

    Returns:
        A ((int, int), list[SEG]) tuple shaped like Impact Pack SEGS.
    '''
    if labels is None:
        labels = ["face"]
    if bboxes is None:
        # Generate a default bbox for each label so zip doesn't truncate.
        bboxes = [[0.0, 0.0, 16.0, 16.0] for _ in labels]
    segs_list = [
        _seg_factory(label=lbl, bbox=bbox)
        for lbl, bbox in zip(labels, bboxes, strict=True)
    ]
    return ((72, 96), segs_list)


# ======================================================================
# parse_index_filter tests                                               #
# ======================================================================


def test_parse_empty_string_returns_none() -> None:
    """Empty string and whitespace-only strings yield None."""
    assert parse_index_filter("", 10) is None
    assert parse_index_filter("   ", 10) is None


def test_parse_comma_separated_indices() -> None:
    """'1,2,3,4' with count 6 → {0, 1, 2, 3} (singles are 0-based)."""
    result = parse_index_filter("1,2,3,4", 6)
    assert result == {0, 1, 2, 3}


def test_parse_range_and_single() -> None:
    """'1-5, 8' with count 10 → {0, 1, 2, 3, 4, 7} (range '1-5' → 0-based
    {0..4}, single '8' converts to 0-based index 7)."""
    result = parse_index_filter("1-5, 8", 10)
    assert result == {0, 1, 2, 3, 4, 7}


def test_parse_reversed_range() -> None:
    """'5-2' with count 10 → {1, 2, 3, 4} (min/max swapped, 0-based)."""
    result = parse_index_filter("5-2", 10)
    assert result == {1, 2, 3, 4}


def test_parse_partial_clamp_low() -> None:
    """'2-4' with count 3 → {1, 2} (clamped to segments 2..3 → 0-based)."""
    result = parse_index_filter("2-4", 3)
    assert result == {1, 2}


def test_parse_full_clamp() -> None:
    """'1-5' with count 3 → {0, 1, 2} (hi clamped to segment 3)."""
    result = parse_index_filter("1-5", 3)
    assert result == {0, 1, 2}


def test_parse_zero_is_invalid() -> None:
    """Index 0 is invalid and skipped → set()."""
    result = parse_index_filter("0", 6)
    assert result == set()


def test_parse_out_of_range_skipped() -> None:
    """'99' with count 6 → set() (out of range)."""
    result = parse_index_filter("99", 6)
    assert result == set()


def test_parse_non_numeric_token() -> None:
    """'abc' → set() (invalid token, skipped)."""
    result = parse_index_filter("abc", 6)
    assert result == set()


def test_parse_mixed_valid_invalid() -> None:
    """'1,abc,3' with count 6 → {0, 2}."""
    result = parse_index_filter("1,abc,3", 6)
    assert result == {0, 2}


def test_parse_whitespace_separators() -> None:
    """' 1 , 2 ' with count 6 → {0, 1} (whitespace stripped)."""
    result = parse_index_filter(" 1 , 2 ", 6)
    assert result == {0, 1}


def test_parse_newline_separator() -> None:
    """'1\\n2' with count 6 → {0, 1} (newline is a separator)."""
    result = parse_index_filter("1\n2", 6)
    assert result == {0, 1}


def test_parse_duplicates_collapse() -> None:
    """'2,2,1-2' with count 6 → {0, 1} (duplicates collapse; single '2'
    and range '1-2' both map to 0-based {0, 1})."""
    result = parse_index_filter("2,2,1-2", 6)
    assert result == {0, 1}


def test_parse_range_fully_out_of_range() -> None:
    """'1-3' with count 0 → set() (everything out of range)."""
    result = parse_index_filter("1-3", 0)
    assert result == set()


# ======================================================================
# parse_labels tests                                                     #
# ======================================================================


def test_parse_labels_empty_count_3() -> None:
    """Empty string with count 3 → ['', '', '']."""
    result = parse_labels("", 3)
    assert result == ["", "", ""]


def test_parse_labels_exact_count() -> None:
    """'bg,body,body,bg,body,hand' with count 6 → 6 entries."""
    result = parse_labels("bg,body,body,bg,body,hand", 6)
    assert len(result) == 6
    assert result == ["bg", "body", "body", "bg", "body", "hand"]


def test_parse_labels_padded() -> None:
    """'bg,body' with count 4 → ['bg', 'body', '', '']."""
    result = parse_labels("bg,body", 4)
    assert result == ["bg", "body", "", ""]


def test_parse_labels_extras_dropped() -> None:
    """'a,b,c,d,e' with count 3 → ['a', 'b', 'c']."""
    result = parse_labels("a,b,c,d,e", 3)
    assert result == ["a", "b", "c"]


def test_parse_labels_empty_piece_positional() -> None:
    """'bg,,body' with count 3 → ['bg', '', 'body']."""
    result = parse_labels("bg,,body", 3)
    assert result == ["bg", "", "body"]


def test_parse_labels_stripped() -> None:
    """' bg , body ' with count 2 → ['bg', 'body']."""
    result = parse_labels(" bg , body ", 2)
    assert result == ["bg", "body"]


def test_parse_labels_newline_separator() -> None:
    """'bg\\nbody' with count 2 → ['bg', 'body']."""
    result = parse_labels("bg\nbody", 2)
    assert result == ["bg", "body"]


def test_parse_labels_inner_spaces_preserved() -> None:
    """' red car ,x' with count 2 → ['red car', 'x']."""
    result = parse_labels(" red car ,x", 2)
    assert result == ["red car", "x"]


# ======================================================================
# split_segs tests                                                       #
# ======================================================================


def test_split_segs_selected_none_returns_identity() -> None:
    """selected=None → first element IS the passed-in segs object."""
    segs = _fake_segs(labels=["a", "b"])
    filtered, remaining = split_segs(segs, None)
    assert filtered is segs
    # Remaining is a fresh tuple with empty list.
    assert isinstance(remaining[1], list) and len(remaining[1]) == 0


def test_split_segs_selected_specific_indices() -> None:
    """selected {1, 2} over 3 segs → filtered [s1, s2], remaining [s0].

    Engine returns 1-based indices for ranges; the set {1, 2} matches
    segments at enumeration indices 1 and 2.
    """
    segs = _fake_segs(labels=["a", "b", "c"])
    filtered, remaining = split_segs(segs, {1, 2})
    assert len(filtered[1]) == 2
    assert filtered[1][0].label == "b"
    assert filtered[1][1].label == "c"
    assert len(remaining[1]) == 1
    assert remaining[1][0].label == "a"


def test_split_segs_size_preserved() -> None:
    """Both outputs keep outer ((h, w),) with int h/w."""
    segs = _fake_segs(labels=["a"])
    filtered, remaining = split_segs(segs, {0})
    assert isinstance(filtered[0][0], int) and isinstance(filtered[0][1], int)
    assert filtered[0] == (72, 96)


def test_split_segs_out_of_range_selected() -> None:
    """selected {5} over 2 segs → filtered empty, remaining [s0, s1]."""
    segs = _fake_segs(labels=["a", "b"])
    filtered, remaining = split_segs(segs, {5})
    assert len(filtered[1]) == 0
    assert len(remaining[1]) == 2


def test_split_segs_malformed_input() -> None:
    """Malformed input raises ValueError with 'Expected a SEGS'."""
    with pytest.raises(ValueError, match="Expected a SEGS"):
        split_segs("not-a-segs", {0})
    with pytest.raises(ValueError, match="Expected a SEGS"):
        split_segs((1, 2), {0})


def test_split_segs_empty_list_with_selected() -> None:
    """Empty seg list with selected {0} → both outputs empty lists."""
    segs = ((72, 96), [])
    filtered, remaining = split_segs(segs, {0})
    assert filtered[1] == []
    assert remaining[1] == []
    assert filtered[0] == (72, 96)


# ======================================================================
# assign_seg_labels tests                                                #
# ======================================================================


def test_assign_namedtuple_segs() -> None:
    """namedtuple segs: labels applied by index, originals unchanged."""
    segs = _fake_segs(labels=["a", "b"])
    result = assign_seg_labels(segs[1], ["x", "y"])
    assert len(result) == 2
    assert result[0].label == "x"
    assert result[1].label == "y"
    # Original namedtuple objects are immutable.
    assert segs[1][0].label == "a"
    assert segs[1][1].label == "b"


def test_assign_namedtuple_returns_new_objects() -> None:
    """Returned objects are new, not the originals."""
    segs = _fake_segs(labels=["a"])
    result = assign_seg_labels(segs[1], ["x"])
    assert result[0] is not segs[1][0]


def test_assign_fewer_labels_than_segments() -> None:
    """Fewer labels than segments → remaining get ''."""
    segs = _fake_segs(labels=["a", "b", "c"])
    result = assign_seg_labels(segs[1], ["x"])
    assert len(result) == 3
    assert result[0].label == "x"
    assert result[1].label == ""
    assert result[2].label == ""


def test_assign_more_labels_than_segments() -> None:
    """More labels than segments → extras ignored."""
    segs = _fake_segs(labels=["a"])
    result = assign_seg_labels(segs[1], ["x", "y", "z"])
    assert len(result) == 1
    assert result[0].label == "x"


def test_assign_plain_object() -> None:
    """Plain object without _replace: copy has label set, original unchanged."""

    @dataclass
    class PlainSeg:
        bbox: list[float]
        label: str = ""

    segs_list = [PlainSeg(bbox=[0.0, 0.0, 16.0, 16.0], label="orig")]
    original_label = segs_list[0].label
    result = assign_seg_labels(segs_list, ["new"])
    assert len(result) == 1
    assert result[0].label == "new"
    assert segs_list[0].label == original_label


# ======================================================================
# build_segment_previews tests                                           #
# ======================================================================


def _frame(
    value: float = 100.0, height: int = 72, width: int = 96
) -> np.ndarray:
    """Constant HWC float32 [0, 255] frame for preview tests."""
    return np.full((height, width, 3), value, dtype=np.float32)


def test_preview_uses_cropped_image_source() -> None:
    """Provided crop source is used as-is (plus corner caption)."""
    seg = _seg_factory(label="a", bbox=[0.0, 0.0, 16.0, 16.0])
    crop = np.full((40, 48, 3), 50.0, dtype=np.float32)
    previews = build_segment_previews(
        [seg], [crop], None, (72, 96), (72, 96), None
    )
    assert len(previews) == 1
    assert previews[0].shape == (40, 48, 4)
    assert previews[0].dtype == np.uint8
    # Pixel between the two corner plates keeps the crop brightness.
    assert abs(int(previews[0][20, 20, 0]) - 50) <= 1


def test_preview_crops_frame_when_no_source() -> None:
    """No cropped_image → frame is cropped by the scaled crop region
    (an [x1, y1, x2, y2] rect, like Impact Pack)."""
    seg = _seg_factory(
        label="a",
        bbox=[16.0, 16.0, 48.0, 48.0],
        crop_region=[8, 8, 56, 56],
    )
    previews = build_segment_previews(
        [seg], [None], _frame(100.0), (72, 96), (72, 96), None
    )
    assert len(previews) == 1
    # Rect (8,8)-(56,56) → 48x48 crop.
    assert previews[0].shape == (48, 48, 4)
    # Pixel between the two corner plates keeps the brightness.
    assert abs(int(previews[0][10, 25, 0]) - 100) <= 1


def test_preview_darkens_non_passing_whole_image() -> None:
    """Non-passing segment preview is darkened by the fixed factor."""
    segs = [
        _seg_factory(label="a", bbox=[0.0, 0.0, 16.0, 16.0]),
        _seg_factory(label="b", bbox=[0.0, 0.0, 16.0, 16.0]),
    ]
    previews = build_segment_previews(
        segs, [None, None], _frame(100.0), (72, 96), (72, 96), {0}
    )
    assert len(previews) == 2
    # Segment 1 passes: keeps brightness.
    assert abs(int(previews[0][8, 20, 0]) - 100) <= 1
    # Segment 2 does not pass: 100 * (1 - DIM_FACTOR) = 40.
    assert abs(int(previews[1][8, 20, 0]) - 100 * (1 - DIM_FACTOR)) <= 2


def test_preview_caption_colors_palette_vs_gray() -> None:
    """Passing preview carries its palette color, non-passing DIM_GRAY."""
    segs = [
        _seg_factory(label="a", bbox=[0.0, 0.0, 16.0, 16.0]),
        _seg_factory(label="b", bbox=[0.0, 0.0, 16.0, 16.0]),
    ]
    previews = build_segment_previews(
        segs, [None, None], _frame(), (72, 96), (72, 96), {0}
    )
    assert np.any(
        np.all(previews[0][..., :3] == np.asarray(PALETTE[0]), axis=-1)
    )
    assert np.any(
        np.all(previews[1][..., :3] == np.asarray(DIM_GRAY), axis=-1)
    )


def test_preview_rescales_when_sizes_differ() -> None:
    """SEGS smaller than the frame → coordinates rescaled by 2x."""
    seg = _seg_factory(
        label="a",
        bbox=[8.0, 8.0, 24.0, 24.0],
        crop_region=[4, 4, 28, 28],
    )
    previews = build_segment_previews(
        [seg], [None], _frame(100.0), (36, 48), (72, 96), None
    )
    # crop_region [4,4,28,28] * 2 → rect (8,8)-(56,56) → 48x48 crop.
    assert previews[0].shape == (48, 48, 4)


def test_preview_skips_unusable_segments() -> None:
    """A degenerate crop region with no crop source is skipped."""
    good = _seg_factory(label="a", bbox=[0.0, 0.0, 16.0, 16.0])
    bad = _seg_factory(
        label="b", bbox=[0.0, 0.0, 16.0, 16.0], crop_region=[0, 0, 0, 0]
    )
    previews = build_segment_previews(
        [good, bad], [None, None], _frame(), (72, 96), (72, 96), None
    )
    assert len(previews) == 1


# ======================================================================
# build_segment_previews alpha-mode tests                                #
# ======================================================================


def _left_half_transparent_mask(size: int = 64) -> np.ndarray:
    """64x64 mask: left half 0.0 (transparent), right half 1.0."""
    mask = np.zeros((size, size), dtype=np.float32)
    mask[:, size // 2 :] = 1.0
    return mask


def _alpha_fixture() -> tuple:
    """One segment centered on a 64x64 crop (bbox (16,16)-(48,48))."""
    seg = _seg_factory(
        label="a",
        bbox=[16.0, 16.0, 48.0, 48.0],
        crop_region=[0, 0, 64, 64],
    )
    return seg, _frame(100.0)


def test_preview_alpha_makes_non_mask_transparent() -> None:
    """RGBA preview; the non-masked part is transparent (floored at
    the fixed MIN_ALPHA) while the caption plate stays opaque."""
    seg, frame = _alpha_fixture()
    previews = build_segment_previews(
        [seg],
        [None],
        frame,
        (72, 96),
        (72, 96),
        None,
        masks=[_left_half_transparent_mask()],
    )
    assert len(previews) == 1
    assert previews[0].shape == (64, 64, 4)
    # Masked (right half) → opaque; unmasked area below the corner
    # plate → alpha floor.
    assert int(previews[0][40, 50, 3]) == 255
    assert int(previews[0][40, 10, 3]) == round(MIN_ALPHA * 255)
    # The caption plate sits over the transparent half but stays
    # opaque.
    assert int(previews[0][10, 10, 3]) == 255


def test_preview_alpha_floor_value() -> None:
    """The fixed alpha floor is exactly MIN_ALPHA (0.25)."""
    assert MIN_ALPHA == 0.25


def test_preview_alpha_resizes_mask_to_crop() -> None:
    """A mask smaller than the crop is resized to the crop size."""
    seg, frame = _alpha_fixture()
    previews = build_segment_previews(
        [seg],
        [None],
        frame,
        (72, 96),
        (72, 96),
        None,
        masks=[np.ones((32, 32), dtype=np.float32)],
    )
    assert previews[0].shape == (64, 64, 4)
    assert int(previews[0][10, 10, 3]) == 255


def test_preview_alpha_without_mask_is_opaque() -> None:
    """No usable mask → fully opaque alpha."""
    seg, frame = _alpha_fixture()
    previews = build_segment_previews(
        [seg],
        [None],
        frame,
        (72, 96),
        (72, 96),
        None,
        masks=[None],
    )
    assert previews[0].shape == (64, 64, 4)
    assert int(previews[0][5, 5, 3]) == 255


def test_preview_darken_with_alpha_keeps_mask_alpha() -> None:
    """Darkening a non-passing preview only affects RGB, not alpha."""
    seg, frame = _alpha_fixture()
    previews = build_segment_previews(
        [seg],
        [None],
        frame,
        (72, 96),
        (72, 96),
        set(),  # nothing passes → darkened
        masks=[_left_half_transparent_mask()],
    )
    # Masked area: RGB darkened to 100 * (1 - DIM_FACTOR) = 40,
    # alpha stays 255.
    assert abs(int(previews[0][40, 50, 0]) - 40) <= 2
    assert int(previews[0][40, 50, 3]) == 255
    # Unmasked area below the plate: alpha stays at the floor despite
    # darkening.
    assert int(previews[0][40, 10, 3]) == round(MIN_ALPHA * 255)


def test_preview_caption_at_top_left_no_box() -> None:
    """Label plate at the top-left, number plate at the bottom-right;
    no bbox rectangle."""
    seg, frame = _alpha_fixture()
    previews = build_segment_previews(
        [seg], [None], frame, (72, 96), (72, 96), None, masks=[None]
    )
    # Label plate fill (PALETTE[0] for passing label 'a') at the corner.
    assert np.all(previews[0][1, 1, :3] == np.asarray(PALETTE[0]))
    # Number plate fill at the opposite corner.
    assert np.all(previews[0][62, 62, :3] == np.asarray(PALETTE[0]))
    # The bbox border (bottom edge row 48, col 32) is NOT drawn.
    assert int(previews[0][48, 32, 0]) == 100
