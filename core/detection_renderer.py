'''
File:   detection_renderer.py
Brief:  YOLO-style bbox/caption/mask overlay engine (numpy/OpenCV, RGB).
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.3.1
'''

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Caption carries the class name only.
LABEL_TEXT = "label"
#: Caption carries the class name followed by the confidence percentage.
LABEL_CONFIDENCE = "label+confidence"
#: Supported caption formats.
LABEL_FORMATS: tuple[str, ...] = (LABEL_TEXT, LABEL_CONFIDENCE)

#: Assign each distinct label a stable color from the built-in palette.
COLOR_AUTO = "auto"
#: Draw every detection with the single user-provided color.
COLOR_SINGLE = "single"
#: Supported color modes.
COLOR_MODES: tuple[str, ...] = (COLOR_AUTO, COLOR_SINGLE)

#: Vivid box/caption colors cycled per distinct label (RGB).
PALETTE: tuple[tuple[int, int, int], ...] = (
    (220, 40, 40),
    (240, 150, 30),
    (250, 220, 40),
    (60, 200, 70),
    (40, 220, 220),
    (50, 120, 240),
    (150, 80, 240),
    (240, 60, 190),
    (150, 230, 40),
    (30, 170, 150),
    (170, 100, 40),
    (250, 130, 170),
)

#: Hershey font used for captions (OpenCV fonts are ASCII-only).
_FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Luminance threshold above which caption text flips to black.
_TEXT_FLIP_LUMA = 150.0


class DetectionRenderError(Exception):
    '''Base exception for detection overlay failures.'''


@dataclass(frozen=True, slots=True)
class Detection:
    '''Single detection to overlay onto a frame.

    Attributes:
        bbox: (x1, y1, x2, y2) absolute pixel coordinates in the frame.
        label: Class name shown in the caption.
        confidence: Detector score in [0, 1]; None hides the percentage.
        mask: Optional (H, W) mask in [0, 1] to tint, sized to the crop
            region it came from.
        mask_origin: (x, y) top-left corner of `mask` in frame pixels.
    '''

    bbox: tuple[float, float, float, float]
    label: str
    confidence: float | None = None
    mask: np.ndarray | None = None
    mask_origin: tuple[int, int] = (0, 0)


def format_caption(
    label: str,
    confidence: float | None,
    label_format: str = LABEL_CONFIDENCE,
) -> str:
    '''Compose the caption text shown next to a detection box.

    Args:
        label: Class name of the detection.
        confidence: Detector score in [0, 1]; ignored when None.
        label_format: One of LABEL_FORMATS.

    Returns:
        The caption string, e.g. ``"face"`` or ``"face 91%"``.

    Raises:
        DetectionRenderError: If the format is unknown.
    '''
    if label_format == LABEL_TEXT:
        return label
    if label_format == LABEL_CONFIDENCE:
        if confidence is None:
            return label
        return f"{label} {confidence:.0%}"
    raise DetectionRenderError(f"Unknown label format: {label_format!r}")


class DetectionRenderer:
    '''Renders detections onto frames in a YOLO-demo style.

    Draws an outlined bounding box per detection, a filled caption plate
    above the box (inside its top edge when there is no room above) and
    an optional semi-transparent tint of the detection mask. Colors come
    either from a fixed per-label palette or from a single user color.
    '''

    def __init__(
        self,
        *,
        label_format: str = LABEL_CONFIDENCE,
        color_mode: str = COLOR_AUTO,
        single_color: tuple[int, int, int] = (0, 255, 0),
        draw_masks: bool = True,
        mask_alpha: float = 0.45,
        thickness: int = 0,
        font_scale: float = 1.0,
    ) -> None:
        '''Create a renderer with fixed drawing options.

        Args:
            label_format: One of LABEL_FORMATS.
            color_mode: One of COLOR_MODES.
            single_color: RGB color used by the `single` color mode.
            draw_masks: Whether to tint detection masks.
            mask_alpha: Mask tint strength in [0, 1].
            thickness: Box border width in pixels; 0 = auto from the
                frame height.
            font_scale: Multiplier on the auto caption text size.

        Raises:
            DetectionRenderError: If any option is out of range.
        '''
        if label_format not in LABEL_FORMATS:
            raise DetectionRenderError(
                f"Unknown label format: {label_format!r}"
            )
        if color_mode not in COLOR_MODES:
            raise DetectionRenderError(f"Unknown color mode: {color_mode!r}")
        try:
            red, green, blue = single_color
        except (TypeError, ValueError) as err:
            raise DetectionRenderError(
                f"single_color must be (R, G, B), got {single_color!r}"
            ) from err
        if any(not 0 <= channel <= 255 for channel in (red, green, blue)):
            raise DetectionRenderError(
                f"single_color channels must be in [0, 255], got "
                f"{single_color!r}"
            )
        if not 0.0 <= mask_alpha <= 1.0:
            raise DetectionRenderError(
                f"mask_alpha must be in [0, 1], got {mask_alpha}"
            )
        if thickness < 0:
            raise DetectionRenderError(
                f"thickness must be >= 0, got {thickness}"
            )
        if font_scale <= 0.0:
            raise DetectionRenderError(
                f"font_scale must be positive, got {font_scale}"
            )
        self._label_format = label_format
        self._color_mode = color_mode
        self._single_color = (int(red), int(green), int(blue))
        self._draw_masks = draw_masks
        self._mask_alpha = mask_alpha
        self._thickness = thickness
        self._font_scale = font_scale

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def render(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
    ) -> np.ndarray:
        '''Return a copy of `frame` with the detections drawn on top.

        Args:
            frame: HWC RGB image, uint8 or float32 [0, 255].
            detections: Detections with absolute frame coordinates.

        Returns:
            Annotated HWC RGB uint8 image of the same resolution.
        '''
        height, width = frame.shape[:2]
        # cv2 draws only on interleaved C-layout arrays; frames derived
        # from permuted (VAE-decoded) tensors arrive channel-planar.
        canvas = np.array(frame, dtype=np.float32, order="C", copy=True)
        colors = self._label_colors(detections)

        if self._draw_masks and self._mask_alpha > 0.0:
            for detection in detections:
                self._blend_mask(
                    canvas,
                    detection,
                    colors[detection.label],
                    self._mask_alpha,
                    width,
                    height,
                )

        canvas = np.clip(np.rint(canvas), 0.0, 255.0).astype(np.uint8)
        box_thickness = (
            max(1, round(height / 512.0))
            if self._thickness == 0
            else self._thickness
        )
        font_size = max(0.4, height / 1024.0) * self._font_scale
        text_thickness = max(1, round(font_size))

        for detection in detections:
            color = colors[detection.label]
            clipped = self._clip_bbox(detection.bbox, width, height)
            if clipped is None:
                continue
            cv2.rectangle(canvas, clipped[:2], clipped[2:], color, box_thickness)
            caption = format_caption(
                detection.label, detection.confidence, self._label_format
            )
            if caption:
                self._draw_caption(
                    canvas,
                    caption,
                    color,
                    clipped,
                    width,
                    font_size,
                    text_thickness,
                )
        return canvas

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _label_colors(
        self,
        detections: Sequence[Detection],
    ) -> dict[str, tuple[int, int, int]]:
        '''Map each label to its display color.'''
        if self._color_mode == COLOR_SINGLE:
            return {d.label: self._single_color for d in detections}
        labels = sorted({d.label for d in detections})
        return {
            label: PALETTE[index % len(PALETTE)]
            for index, label in enumerate(labels)
        }

    @staticmethod
    def _blend_mask(
        canvas: np.ndarray,
        detection: Detection,
        color: tuple[int, int, int],
        alpha: float,
        width: int,
        height: int,
    ) -> None:
        '''Tint the mask area of `detection` on the float canvas, in place.'''
        mask = detection.mask
        if mask is None:
            return
        mask_h, mask_w = mask.shape[:2]
        origin_x, origin_y = detection.mask_origin
        x1 = max(origin_x, 0)
        y1 = max(origin_y, 0)
        x2 = min(origin_x + mask_w, width)
        y2 = min(origin_y + mask_h, height)
        if x2 <= x1 or y2 <= y1:
            logger.debug("Mask of %r fully outside the frame", detection.label)
            return
        sub = mask[
            y1 - origin_y : y2 - origin_y, x1 - origin_x : x2 - origin_x, None
        ].astype(np.float32)
        strength = alpha * sub
        region = canvas[y1:y2, x1:x2]
        canvas[y1:y2, x1:x2] = region * (1.0 - strength) + np.asarray(
            color, dtype=np.float32
        ) * strength

    @staticmethod
    def _clip_bbox(
        bbox: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        '''Clip a bbox to the frame; None when nothing is visible.'''
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            logger.debug("Skipping degenerate bbox %s", bbox)
            return None
        if x2 < 0 or y2 < 0 or x1 >= width or y1 >= height:
            logger.debug("Skipping off-frame bbox %s", bbox)
            return None
        ix1 = int(np.clip(round(x1), 0, width - 1))
        iy1 = int(np.clip(round(y1), 0, height - 1))
        ix2 = int(np.clip(round(x2), 0, width - 1))
        iy2 = int(np.clip(round(y2), 0, height - 1))
        return ix1, iy1, ix2, iy2

    @staticmethod
    def _draw_caption(
        canvas: np.ndarray,
        caption: str,
        color: tuple[int, int, int],
        clipped_bbox: tuple[int, int, int, int],
        width: int,
        font_size: float,
        text_thickness: int,
    ) -> None:
        '''Draw a filled caption plate with text above (or inside) the box.'''
        (text_w, text_h), baseline = cv2.getTextSize(
            caption, _FONT, font_size, text_thickness
        )
        pad = max(2, round(text_h * 0.3))
        plate_h = text_h + baseline + 2 * pad
        plate_w = text_w + 2 * pad
        plate_x = min(max(clipped_bbox[0], 0), max(width - plate_w, 0))
        plate_y = clipped_bbox[1] - plate_h
        if plate_y < 0:
            plate_y = clipped_bbox[1]
        text_color: tuple[int, int, int]
        if _luminance(color) > _TEXT_FLIP_LUMA:
            text_color = (0, 0, 0)
        else:
            text_color = (255, 255, 255)
        cv2.rectangle(
            canvas,
            (plate_x, plate_y),
            (plate_x + plate_w, plate_y + plate_h),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            caption,
            (plate_x + pad, plate_y + pad + text_h),
            _FONT,
            font_size,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )


def _luminance(color: tuple[int, int, int]) -> float:
    '''Perceived brightness of an RGB color on the 0..255 scale.'''
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]


def scale_detections(
    detections: Sequence[Detection],
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> list[Detection]:
    '''Rescale detections from one frame resolution to another.

    Args:
        detections: Detections in `src_size` coordinates.
        src_size: Source frame size as (height, width).
        dst_size: Target frame size as (height, width).

    Returns:
        New detections with scaled bbox and mask coordinates; masks are
        resized with bilinear interpolation.

    Raises:
        DetectionRenderError: If any size dimension is not positive.
    '''
    src_h, src_w = src_size
    dst_h, dst_w = dst_size
    if min(src_h, src_w, dst_h, dst_w) <= 0:
        raise DetectionRenderError(
            f"Sizes must be positive, got src={src_size} dst={dst_size}"
        )
    factor_x = dst_w / src_w
    factor_y = dst_h / src_h

    scaled: list[Detection] = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        mask = detection.mask
        if mask is not None:
            new_w = max(1, round(mask.shape[1] * factor_x))
            new_h = max(1, round(mask.shape[0] * factor_y))
            mask = cv2.resize(
                mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
        origin_x, origin_y = detection.mask_origin
        scaled.append(
            replace(
                detection,
                bbox=(x1 * factor_x, y1 * factor_y, x2 * factor_x, y2 * factor_y),
                mask=mask,
                mask_origin=(round(origin_x * factor_x), round(origin_y * factor_y)),
            )
        )
    return scaled
