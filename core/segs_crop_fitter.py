'''
File:   segs_crop_fitter.py
Brief:  Fit Impact Pack SEGS crop regions to an absolute target size.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v1.0.0
'''

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

#: Sizing mode names, exposed as the node's combo choices.
FIT_MODES: tuple[str, ...] = ("exact", "aspect")

#: ``exact`` mode: the crop region is exactly width x height.
MODE_EXACT = "exact"
#: ``aspect`` mode: the bbox is scaled uniformly so its longer side
#: equals ``max(width, height)``.
MODE_ASPECT = "aspect"

#: Divisor the fitted sizes are rounded up to (SD-friendly default).
DEFAULT_ROUND_TO: int = 8


class FittedRegion(NamedTuple):
    '''Outcome of fitting one segment's crop region.

    Attributes:
        region: Fitted crop region ``(x1, y1, x2, y2)`` in image pixels.
        clamped: True if the requested size was reduced because it
            exceeded the canvas.
        grown: True if the size ended up larger than requested because
            the bbox had to stay inside.
    '''

    region: tuple[int, int, int, int]
    clamped: bool
    grown: bool


def _round_up(value: float, step: int) -> int:
    '''Round ``value`` up to the nearest multiple of ``step``.'''
    return int(math.ceil(value / step)) * step


def _normalize_rect(rect: Sequence[float]) -> tuple[int, int, int, int]:
    '''Coerce a bbox rect to ints with at least a 1-pixel extent.'''
    x1, y1, x2, y2 = (int(round(float(v))) for v in rect[:4])
    return x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)


def fit_crop_region(
    bbox: Sequence[float],
    canvas_size: tuple[int, int],
    target_size: tuple[int, int],
    mode: str = MODE_EXACT,
    round_to: int = DEFAULT_ROUND_TO,
) -> FittedRegion:
    '''Compute a crop region of the requested absolute size around a bbox.

    The region is centered on the bbox center (the same anchoring as
    Impact Pack's ``make_crop_region``) and shifted to stay inside the
    canvas. It always contains the bbox; when the target is too small
    for the bbox, the size grows just enough (rounded up to a multiple
    of ``round_to``) so a detection is never cut off.

    Args:
        bbox: Segment bbox ``(x1, y1, x2, y2)`` in image pixels.
        canvas_size: Full-image size ``(width, height)``.
        target_size: Requested crop size ``(width, height)``.
        mode: One of FIT_MODES. ``exact`` uses the target verbatim;
            ``aspect`` scales the bbox uniformly so its longer side
            equals ``max(target)`` (never below 1x).
        round_to: Round sizes up to multiples of this value; applied to
            aspect-mode sizes and to growth forced by a bbox larger
            than the target (the exact target is used verbatim).

    Returns:
        The fitted crop rect plus clamp/grow flags.

    Raises:
        ValueError: If ``mode`` is not one of FIT_MODES.
    '''
    if mode not in FIT_MODES:
        raise ValueError(
            f"Unknown fit mode {mode!r}, expected one of {FIT_MODES}"
        )

    x1, y1, x2, y2 = _normalize_rect(bbox)
    bbox_w, bbox_h = x2 - x1, y2 - y1
    canvas_w, canvas_h = (max(1, int(v)) for v in canvas_size)
    target_w, target_h = (max(1, int(v)) for v in target_size)
    step = max(1, int(round_to))

    if mode == MODE_ASPECT:
        scale = max(1.0, max(target_w, target_h) / max(bbox_w, bbox_h))
        fit_w = _round_up(bbox_w * scale, step)
        fit_h = _round_up(bbox_h * scale, step)
    else:
        fit_w, fit_h = target_w, target_h

    if fit_w < bbox_w:
        fit_w = _round_up(bbox_w, step)
    if fit_h < bbox_h:
        fit_h = _round_up(bbox_h, step)
    grown = fit_w > target_w or fit_h > target_h

    clamped = False
    if fit_w > canvas_w:
        fit_w = canvas_w
        clamped = True
    if fit_h > canvas_h:
        fit_h = canvas_h
        clamped = True

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    new_x1 = min(
        max(int(round(center_x - fit_w / 2.0)), 0), canvas_w - fit_w
    )
    new_y1 = min(
        max(int(round(center_y - fit_h / 2.0)), 0), canvas_h - fit_h
    )

    return FittedRegion(
        region=(new_x1, new_y1, new_x1 + fit_w, new_y1 + fit_h),
        clamped=clamped,
        grown=grown,
    )


def realign_mask(
    mask: np.ndarray,
    old_region: Sequence[int],
    new_region: Sequence[int],
) -> np.ndarray:
    '''Re-cut a cropped mask array for a resized crop region.

    Pixels keep their image coordinates: the old crop's content is
    copied into the output at the offset between the two regions, and
    the area outside the old region is zero-filled. Batched masks with
    leading dimensions are preserved.

    Args:
        mask: Cropped mask array ``(..., H, W)`` aligned to old_region.
        old_region: Previous crop region ``(x1, y1, x2, y2)``.
        new_region: New crop region ``(x1, y1, x2, y2)``.

    Returns:
        A float32 array shaped like ``mask`` with the new (H, W).
    '''
    array = np.asarray(mask, dtype=np.float32)
    old_x1, old_y1, old_x2, old_y2 = (int(v) for v in old_region[:4])
    new_x1, new_y1, new_x2, new_y2 = (int(v) for v in new_region[:4])
    result = np.zeros(
        (*array.shape[:-2], new_y2 - new_y1, new_x2 - new_x1),
        dtype=np.float32,
    )

    x1 = max(old_x1, new_x1)
    y1 = max(old_y1, new_y1)
    x2 = min(old_x2, new_x2)
    y2 = min(old_y2, new_y2)
    if x2 > x1 and y2 > y1:
        result[
            ..., y1 - new_y1 : y2 - new_y1, x1 - new_x1 : x2 - new_x1
        ] = array[
            ..., y1 - old_y1 : y2 - old_y1, x1 - old_x1 : x2 - old_x1
        ]
    return result
