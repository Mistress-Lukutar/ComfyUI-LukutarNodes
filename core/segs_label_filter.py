'''
File:   segs_label_filter.py
Brief:  SEGS label assignment + numeric-index filter + per-segment
        preview engine.
Author: Mistress-Lukutar
Date:   2026-11-18
Version: v0.5.1
'''

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from .detection_renderer import PALETTE, draw_corner_caption

logger = logging.getLogger(__name__)

#: Fixed darken factor for previews of segments not passing the filter.
DIM_FACTOR = 0.6
#: Fixed alpha floor for the transparent part of every preview.
MIN_ALPHA = 0.25
#: Caption color for segments not passing the filter (RGB).
DIM_GRAY = (80, 80, 80)


def _split_filter_tokens(text: str) -> list[str]:
    '''Split a filter string on commas and whitespace, keeping quoted
    spans (single or double quotes) intact — separators inside quotes
    are preserved.

    Args:
        text: Raw filter text.

    Returns:
        The non-empty tokens, in order; quotes remain part of each
        quoted token.  An unclosed quote extends to the end of the
        text.
    '''
    tokens: list[str] = []
    current: list[str] = []
    quote = ''
    for ch in text:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = ''
        elif ch in '\'"':
            quote = ch
            current.append(ch)
        elif ch.isspace() or ch == ',':
            if current:
                tokens.append(''.join(current))
                current = []
        else:
            current.append(ch)
    if current:
        tokens.append(''.join(current))
    return tokens


def parse_index_filter(
    text: str,
    count: int,
    labels: list[str] | None = None,
) -> set[int]:
    '''Parse a filter string into a set of passing segment indices.

    Tokens are separated by commas and whitespace.  Numbers are
    1-based; ``0`` is invalid and skipped.  Ranges ``A-B`` expand to
    all integers from ``min(A,B)`` through ``max(A,B)`` (inclusive),
    clamped to ``[1, count]``.  A quoted token (``'face'`` or
    ``"bg"``) selects every segment whose assigned label equals the
    quoted text exactly (case-sensitive); quotes may contain spaces
    and commas.  An EMPTY text selects nothing.

    Args:
        text: Filter string such as "1,2,3,4", "1-5, 8" or
            "1,3,'face', 10".
        count: Total number of segments (upper bound for indices).
        labels: Assigned label per segment (len == count); required to
            resolve quoted tokens.  When None, quoted tokens are
            skipped with a warning.

    Returns:
        A set of **0-based** indices that pass the filter.  An empty
        *text* yields an empty set — nothing is selected.  Never
        raises.
    '''
    stripped = text.strip()
    if not stripped:
        return set()

    result: set[int] = set()
    for token in _split_filter_tokens(stripped):
        if not token:
            continue
        # Quoted token: match segments by assigned label.
        if len(token) >= 2 and token[0] == token[-1] and token[0] in '\'"':
            quoted = token[1:-1]
            if labels is None:
                logger.warning(
                    "Label token %r ignored: no labels to match against",
                    token,
                )
                continue
            matched = [
                i
                for i, label in enumerate(labels[:count])
                if label == quoted
            ]
            if not matched:
                logger.info("Label %r matches no segment, skipping", quoted)
                continue
            result.update(matched)
            continue
        # Try range form A-B first.
        if '-' in token:
            a_str, b_str = token.split('-', 1)
            if not a_str.isdigit() or not b_str.isdigit():
                logger.warning("Non-numeric range token %r, skipping", token)
                continue
            a, b = int(a_str), int(b_str)
            lo, hi = min(a, b), max(a, b)
            clamped_lo = max(lo, 1)
            clamped_hi = min(hi, count)
            if clamped_lo > clamped_hi:
                logger.warning(
                    "Range %d-%d fully out of [1, %d], skipping",
                    lo, hi, count,
                )
                continue
            if clamped_lo != lo or clamped_hi != hi:
                logger.info(
                    "Clamping range %d-%d to [%d, %d]",
                    lo, hi, clamped_lo, clamped_hi,
                )
            result.update(range(clamped_lo - 1, clamped_hi))
        else:
            # Single number.
            if not token.isdigit():
                logger.warning("Non-numeric token %r, skipping", token)
                continue
            n = int(token)
            if n == 0:
                logger.warning("Index 0 is invalid, skipping")
                continue
            if n > count:
                logger.warning("Index %d > count %d, skipping", n, count)
                continue
            result.add(n - 1)  # Convert to 0-based.

    return result


def parse_labels(text: str, count: int) -> list[str]:
    '''Split a label string into exactly *count* entries.

    Split on ``\",\"`` and newlines; empty pieces are valid positional
    entries (e.g. ``"bg,,body\"`` → ``["bg", "", "body"]``).  Inner
    spaces are preserved.

    Args:
        text: Comma/newline-separated label string.
        count: Number of segments to produce labels for.

    Returns:
        A list of exactly *count* strings (padded with ``""`` if needed,
        extras silently dropped).
    '''
    if not text:
        return [''] * count

    pieces = re.split(r'[,\n]', text)
    stripped_pieces = [p.strip() for p in pieces]

    if len(stripped_pieces) > count:
        logger.debug(
            "parse_labels: %d tokens for %d segments; dropping extras",
            len(stripped_pieces), count,
        )

    result = stripped_pieces[:count]
    while len(result) < count:
        result.append('')
    return result


def split_segs(
    segs: Any,
    selected: set[int] | None,
) -> tuple[Any, Any]:
    '''Split a SEGS payload into passing and remaining segments.

    Args:
        segs: A ``((h, w), [SEG, ...])`` payload.
        selected: Set of 0-based indices that pass the filter, or
            ``None`` to keep everything in the first tuple.

    Returns:
        A ``(filtered, remaining)`` pair, each a fresh
        ``((int(h), int(w)), [SEG, ...])`` tuple.  When *selected* is
        ``None``, *filtered* is the original object (identity).

    Raises:
        ValueError: If *segs* is not shaped like a SEGS payload.
    '''
    try:
        (h, w), seg_list = segs
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"Expected a SEGS ((h, w), [SEG, ...]) payload, got {segs!r}"
        ) from err

    if selected is None:
        return (segs, ((int(h), int(w)), []))

    filtered_segs: list[Any] = []
    remaining_segs: list[Any] = []
    for idx, seg in enumerate(seg_list):
        if idx in selected:
            filtered_segs.append(seg)
        else:
            remaining_segs.append(seg)

    return (
        ((int(h), int(w)), filtered_segs),
        ((int(h), int(w)), remaining_segs),
    )


def assign_seg_labels(
    segments: list[Any],
    labels: list[str],
) -> list[Any]:
    '''Assign labels to a list of segments.

    Mirrors the ``_replace_seg`` helper pattern from
    ``nodes/segs_set_crop_size.py``.

    Args:
        segments: List of SEG objects (namedtuple or plain object).
        labels: Label strings; index *i* is assigned to segment *i*.

    Returns:
        A new list with the same number of elements, each having its
        ``label`` attribute set (via ``_replace`` or ``setattr``).
    '''
    result: list[Any] = []
    for i, seg in enumerate(segments):
        lbl = labels[i] if i < len(labels) else ''
        try:
            updated = seg._replace(label=lbl)
        except AttributeError:
            updated = copy.copy(seg)
            updated.label = lbl
        result.append(updated)
    return result


def _label_color_map(
    segments: list[Any],
    selected: set[int] | None,
) -> dict[str, tuple[int, int, int]]:
    '''Map each distinct base label of the passing segments to its
    palette color (same label = same color).'''
    if selected is None:
        passing_indices: Sequence[int] = range(len(segments))
    else:
        passing_indices = [i for i in range(len(segments)) if i in selected]
    passing_labels = sorted(
        {str(getattr(segments[i], "label", "")) for i in passing_indices}
    )
    return {
        lbl: PALETTE[idx % len(PALETTE)]
        for idx, lbl in enumerate(passing_labels)
    }


def _crop_frame(
    frame: np.ndarray | None,
    crop_region: Any,
    factor_x: float,
    factor_y: float,
) -> np.ndarray | None:
    '''Crop `frame` by a ``[x1, y1, x2, y2]`` crop region (coordinates
    are rescaled by the given factors); None when the frame is missing
    or the region is unusable/off-frame.'''
    if frame is None:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in crop_region[:4])
    except (TypeError, ValueError):
        return None
    frame_h, frame_w = frame.shape[:2]
    ix1 = max(int(round(x1 * factor_x)), 0)
    iy1 = max(int(round(y1 * factor_y)), 0)
    ix2 = min(int(round(x2 * factor_x)), frame_w)
    iy2 = min(int(round(y2 * factor_y)), frame_h)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return frame[iy1:iy2, ix1:ix2]


def _segment_alpha(
    mask: np.ndarray | None,
    crop_shape: tuple[int, int],
    min_alpha: float,
) -> np.ndarray:
    '''Alpha channel for a segment preview, SEGSPreview-style.

    The cropped mask is resized to the crop size and floored at
    ``min_alpha`` (batched ``(B, H, W)`` masks are combined by maximum
    — the union).  None/unusable mask → fully opaque.

    Args:
        mask: Cropped mask (H, W) or (B, H, W) float in [0, 1], or
            None.
        crop_shape: (height, width) of the crop the alpha must match.
        min_alpha: Alpha floor in [0, 1].

    Returns:
        (H, W) float32 alpha in [0, 255].
    '''
    crop_h, crop_w = crop_shape[:2]
    if mask is None:
        return np.full((crop_h, crop_w), 255.0, dtype=np.float32)
    alpha = np.asarray(mask, dtype=np.float32)
    if alpha.ndim == 3:
        alpha = alpha.max(axis=0)
    if alpha.ndim != 2 or alpha.size == 0:
        return np.full((crop_h, crop_w), 255.0, dtype=np.float32)
    if alpha.shape[:2] != (crop_h, crop_w):
        alpha = cv2.resize(
            alpha, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
    alpha = np.clip(alpha, 0.0, 1.0) * 255.0
    floor = min(max(min_alpha, 0.0), 1.0) * 255.0
    if floor > 0.0:
        alpha = np.maximum(alpha, floor)
    return alpha


def build_segment_previews(
    segments: list[Any],
    crops: list[np.ndarray | None],
    frame: np.ndarray | None,
    segs_size: tuple[int, int],
    frame_size: tuple[int, int],
    selected: set[int] | None,
    masks: list[np.ndarray | None] | None = None,
    font_size: float = 0.5,
) -> list[np.ndarray]:
    '''Build one preview image per segment, SEGSPreview-style.

    Each preview shows the segment as a crop: ``crops[i]`` (the SEG's
    own ``cropped_image``) when provided, otherwise `frame` cropped by
    the segment's ``crop_region`` (an ``[x1, y1, x2, y2]`` rect).  The
    image size is the crop's own size, matching SEGSPreview.  The
    non-masked part of every preview is transparent (alpha from the
    cropped mask, floored at ``MIN_ALPHA``); segments not in
    `selected` are darkened over their whole preview by
    ``(1 - DIM_FACTOR)``.  Every preview carries the segment label as
    a caption plate at the top-left corner (when labeled) and its
    1-based index (over the full input list) as a plate at the
    bottom-right corner; no bbox rectangle is drawn.  Segment
    coordinates are rescaled proportionally when SEGS and frame
    resolutions differ.  Segments without a usable crop are skipped.

    Args:
        segments: Labeled SEG objects (``crop_region``, ``label``).
        crops: Per-segment crop source, HWC RGB float32 [0, 255], or
            None to fall back to cropping `frame`.
        frame: Full HWC RGB float32 [0, 255] frame used when a SEG has
            no crop source; None disables the frame-crop fallback.
        segs_size: SEGS (height, width) the segment coordinates refer
            to.
        frame_size: Actual frame (height, width).
        selected: 0-based indices that pass the filter, or None for
            all passing.
        masks: Per-segment cropped mask (H, W) or (B, H, W) float
            [0, 1], or None; drives the alpha channel.
        font_size: OpenCV font size used as-is for the caption plates.

    Returns:
        HWC RGBA uint8 preview images in segment order; segments
        without a usable crop produce no preview.
    '''
    segs_h, segs_w = segs_size
    frame_h, frame_w = frame_size
    factor_x = frame_w / segs_w if segs_w else 1.0
    factor_y = frame_h / segs_h if segs_h else 1.0
    color_map = _label_color_map(segments, selected)

    previews: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        crop = crops[i] if i < len(crops) else None
        if crop is None:
            crop = _crop_frame(frame, seg.crop_region, factor_x, factor_y)
        if crop is None:
            logger.debug(
                "SegsLabelFilter: no usable crop for segment %d, skipped",
                i + 1,
            )
            continue

        passing = selected is None or i in selected
        if not passing:
            crop = crop * (1.0 - DIM_FACTOR)

        mask = masks[i] if masks is not None and i < len(masks) else None
        alpha = _segment_alpha(mask, crop.shape, MIN_ALPHA)
        canvas = np.concatenate([crop, alpha[..., None]], axis=-1)
        canvas = np.clip(np.rint(canvas), 0.0, 255.0).astype(np.uint8)

        base_label = str(getattr(seg, "label", ""))
        color = (
            color_map.get(base_label, PALETTE[0]) if passing else DIM_GRAY
        )
        if base_label:
            draw_corner_caption(
                canvas, base_label, color, font_size=font_size
            )
        draw_corner_caption(
            canvas,
            str(i + 1),
            color,
            font_size=font_size,
            corner="bottom-right",
        )
        previews.append(canvas)
    return previews
