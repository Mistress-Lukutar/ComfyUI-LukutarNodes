'''
File:   segs_overlay.py
Brief:  ComfyUI node drawing Impact Pack SEGS detections on an image.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.3.0
'''

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from ..core.detection_renderer import (
    COLOR_AUTO,
    COLOR_MODES,
    LABEL_CONFIDENCE,
    LABEL_FORMATS,
    Detection,
    DetectionRenderer,
    scale_detections,
)
from ..utils.images import frames_to_tensor, tensor_to_frames

logger = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray:
    '''Convert a torch tensor or array-like SEGS field to a numpy array.'''
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_confidence(value: Any) -> float | None:
    '''Normalize a SEG confidence (float/array/tensor) to a plain float.'''
    if value is None:
        return None
    return float(_to_numpy(value).ravel()[0])


def _as_frame_mask(value: Any) -> np.ndarray | None:
    '''Normalize a SEG cropped_mask to a (H, W) float32 array in [0, 1].'''
    if value is None:
        return None
    array = _to_numpy(value).astype(np.float32)
    if array.ndim == 3:
        # (B, H, W) batched masks (AnimateDiff) -> first frame.
        array = array[0]
    if array.ndim != 2:
        raise ValueError(
            f"cropped_mask must be 2D or 3D, got shape {array.shape}"
        )
    return np.clip(array, 0.0, 1.0)


def _parse_segs(segs: Any) -> tuple[tuple[int, int], list[Detection]]:
    '''Convert an Impact Pack SEGS payload into renderer detections.

    SEGS is consumed duck-typed (no impact imports): a
    ``((height, width), [SEG, ...])`` tuple whose SEG elements expose
    ``bbox``, ``crop_region``, ``label``, ``confidence`` and
    ``cropped_mask``.

    Args:
        segs: SEGS payload from e.g. SEGM Detector (SEGS).

    Returns:
        The SEGS (height, width) size and the parsed detections.

    Raises:
        ValueError: If the payload is not shaped like SEGS.
    '''
    try:
        (height, width), seg_list = segs
        segs_size = (int(height), int(width))
        segments = list(seg_list)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"Expected a SEGS ((h, w), [SEG, ...]) payload, got {segs!r}"
        ) from err

    detections: list[Detection] = []
    for seg in segments:
        try:
            x1, y1, x2, y2 = (float(v) for v in seg.bbox)
            crop_x1, crop_y1 = (int(v) for v in seg.crop_region[:2])
            label = str(seg.label)
        except (AttributeError, TypeError, ValueError) as err:
            raise ValueError(
                f"Segment {seg!r} does not look like an Impact Pack SEG"
            ) from err
        detections.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                label=label,
                confidence=_as_confidence(seg.confidence),
                mask=_as_frame_mask(seg.cropped_mask),
                mask_origin=(crop_x1, crop_y1),
            )
        )
    return segs_size, detections


class SegsOverlayNode:
    '''Draw Impact Pack SEGS detections on an image, YOLO-demo style.

    Renders one outlined bounding box per segment with a filled caption
    plate (class name, optionally with the confidence percentage) and
    optionally tints the segment masks. The SEGS input is passed through
    unchanged, so the node can sit between a detector (e.g. SEGM
    Detector (SEGS)) and a Detailer pipeline while previewing what was
    detected.
    '''

    CATEGORY = "Lukutar/Image"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Image to draw on, e.g. the one the detector"
                            " ran on"
                        )
                    },
                ),
                "segs": (
                    "SEGS",
                    {
                        "tooltip": (
                            "Segments from an Impact Pack detector, e.g."
                            " SEGM Detector (SEGS)"
                        )
                    },
                ),
                "label_format": (
                    list(LABEL_FORMATS),
                    {
                        "default": LABEL_CONFIDENCE,
                        "tooltip": (
                            "label: class name only; label+confidence:"
                            " append the score, e.g. 'face 91%'"
                        ),
                    },
                ),
                "draw_masks": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "masks",
                        "label_off": "boxes only",
                        "tooltip": "Tint the segment masks",
                    },
                ),
                "mask_alpha": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Mask tint strength",
                    },
                ),
                "thickness": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": (
                            "Box border width in pixels; 0 = auto from the"
                            " image height"
                        ),
                    },
                ),
                "font_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Multiplier on the auto caption size",
                    },
                ),
            },
            "optional": {
                "color_mode": (
                    list(COLOR_MODES),
                    {
                        "default": COLOR_AUTO,
                        "tooltip": (
                            "auto: stable color per class from a built-in"
                            " palette; single: one user color for all"
                        ),
                    },
                ),
                "color_r": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "tooltip": "single color mode: red channel",
                    },
                ),
                "color_g": (
                    "INT",
                    {
                        "default": 255,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "tooltip": "single color mode: green channel",
                    },
                ),
                "color_b": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "tooltip": "single color mode: blue channel",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "SEGS")
    RETURN_NAMES = ("image", "segs")
    FUNCTION = "overlay"

    def overlay(
        self,
        image: torch.Tensor,
        segs: Any,
        label_format: str = LABEL_CONFIDENCE,
        draw_masks: bool = True,
        mask_alpha: float = 0.45,
        thickness: int = 0,
        font_scale: float = 1.0,
        color_mode: str = COLOR_AUTO,
        color_r: int = 0,
        color_g: int = 255,
        color_b: int = 0,
    ) -> tuple[torch.Tensor, Any]:
        '''Draw the SEGS detections frame by frame.

        When the image resolution differs from the one recorded in SEGS,
        all coordinates and masks are rescaled proportionally.

        Args:
            image: ComfyUI IMAGE batch to annotate, (B, H, W, 3) [0, 1].
            segs: SEGS payload from an Impact Pack detector.
            label_format: Caption format, one of LABEL_FORMATS.
            draw_masks: Whether to tint the segment masks.
            mask_alpha: Mask tint strength in [0, 1].
            thickness: Box border width; 0 = auto from the image height.
            font_scale: Multiplier on the auto caption text size.
            color_mode: One of COLOR_MODES.
            color_r: single color mode: red channel.
            color_g: single color mode: green channel.
            color_b: single color mode: blue channel.

        Returns:
            Tuple of (annotated IMAGE batch, `segs` passed through).

        Raises:
            ValueError: If the SEGS payload is malformed.
        '''
        frames = tensor_to_frames(image)
        segs_size, detections = _parse_segs(segs)
        frame_size = tuple(int(v) for v in frames[0].shape[:2])

        if detections and segs_size != frame_size:
            detections = scale_detections(detections, segs_size, frame_size)
            logger.info(
                "SegsOverlay: rescaled SEGS %s -> %s", segs_size, frame_size
            )

        if not detections:
            logger.info("SegsOverlay: empty SEGS, image passed through")
            return (frames_to_tensor(frames), segs)

        renderer = DetectionRenderer(
            label_format=label_format,
            color_mode=color_mode,
            single_color=(color_r, color_g, color_b),
            draw_masks=draw_masks,
            mask_alpha=mask_alpha,
            thickness=thickness,
            font_scale=font_scale,
        )
        results = [renderer.render(frame, detections) for frame in frames]
        logger.info(
            "SegsOverlay: %d detection(s) drawn on %d frame(s)",
            len(detections),
            len(results),
        )
        return (frames_to_tensor(results), segs)
