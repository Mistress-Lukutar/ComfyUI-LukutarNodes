'''
File:   segs_set_crop_size.py
Brief:  ComfyUI node refitting Impact Pack SEGS crop regions to a target size.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v1.0.0
'''

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch

from ..core.segs_crop_fitter import (
    DEFAULT_ROUND_TO,
    FIT_MODES,
    MODE_EXACT,
    fit_crop_region,
    realign_mask,
)

logger = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray:
    '''Convert a torch tensor or array-like SEGS field to a numpy array.'''
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _replace_seg(seg: Any, **fields: Any) -> Any:
    '''Update SEG fields on a namedtuple (or plain object) segment.'''
    try:
        return seg._replace(**fields)
    except AttributeError:
        updated = copy.copy(seg)
        for name, value in fields.items():
            setattr(updated, name, value)
        return updated


class SegsSetCropSizeNode:
    '''Refit every Impact Pack SEGS crop region to an absolute target size.

    A detector's ``crop_factor`` is relative, so the same factor yields
    wildly different crop sizes per segment (128x128 on a small
    detection, 1024x1024 on a large one) and often sampler-unfriendly
    sizes like 552x239. This node pins each segment's crop region to a
    size you choose, e.g. 512x512, while leaving the bbox, label,
    confidence and the mask content itself untouched — only the crop
    rectangle (and the mask's alignment to it) changes, so the node can
    sit between a detector and a Detailer pipeline.
    '''

    CATEGORY = "Lukutar/Image"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "segs": (
                    "SEGS",
                    {
                        "tooltip": (
                            "Segments from an Impact Pack detector,"
                            " e.g. SEGM Detector (SEGS)"
                        )
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 8,
                        "max": 16384,
                        "step": 8,
                        "tooltip": "Target crop width in pixels",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 8,
                        "max": 16384,
                        "step": 8,
                        "tooltip": "Target crop height in pixels",
                    },
                ),
                "mode": (
                    list(FIT_MODES),
                    {
                        "default": MODE_EXACT,
                        "tooltip": (
                            "exact: crop exactly width x height (grown"
                            " only if the bbox is larger); aspect: scale"
                            " the bbox uniformly so its longer side"
                            " equals max(width, height)"
                        ),
                    },
                ),
                "round_to": (
                    "INT",
                    {
                        "default": DEFAULT_ROUND_TO,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "tooltip": (
                            "Round fitted sizes up to multiples of this"
                            " (SD-friendly 8); the exact-mode target is"
                            " used verbatim"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "fit"

    def fit(
        self,
        segs: Any,
        width: int = 512,
        height: int = 512,
        mode: str = MODE_EXACT,
        round_to: int = DEFAULT_ROUND_TO,
    ) -> tuple[Any]:
        '''Refit every segment's crop region to the target size.

        Each new region is centered on its bbox center and kept inside
        the SEGS canvas; the cropped mask is re-cut into the new region
        (zero-filling the added context area) and any cached
        ``cropped_image`` is dropped so consumers re-crop from the
        source image.

        Args:
            segs: SEGS payload from an Impact Pack detector, shaped
                ``((h, w), [SEG, ...])``.
            width: Target crop width in pixels.
            height: Target crop height in pixels.
            mode: One of FIT_MODES.
            round_to: Round fitted sizes up to multiples of this value.

        Returns:
            Tuple of the SEGS payload with refitted crop regions.

        Raises:
            ValueError: If the payload is not shaped like SEGS.
        '''
        try:
            (shape_h, shape_w), seg_list = segs
            segs_h, segs_w = int(shape_h), int(shape_w)
            canvas = (segs_w, segs_h)
            segments = list(seg_list)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Expected a SEGS ((h, w), [SEG, ...]) payload, got {segs!r}"
            ) from err

        if not segments:
            logger.info("SegsSetCropSize: empty SEGS, passed through")
            return (segs,)

        target = (int(width), int(height))
        fitted_items = []
        clamped_count = 0
        grown_count = 0
        for seg in segments:
            try:
                bbox = seg.bbox
                crop_region = seg.crop_region
                cropped_mask = seg.cropped_mask
            except AttributeError as err:
                raise ValueError(
                    f"Segment {seg!r} does not look like an Impact Pack SEG"
                ) from err

            fitted = fit_crop_region(bbox, canvas, target, mode, round_to)
            clamped_count += fitted.clamped
            grown_count += fitted.grown
            new_mask = (
                None
                if cropped_mask is None
                else realign_mask(
                    _to_numpy(cropped_mask), crop_region, fitted.region
                )
            )
            fitted_items.append(
                _replace_seg(
                    seg,
                    crop_region=list(fitted.region),
                    cropped_mask=new_mask,
                    cropped_image=None,
                )
            )

        logger.info(
            "SegsSetCropSize: %d segment(s) refit to %dx%d (mode=%s)",
            len(fitted_items),
            target[0],
            target[1],
            mode,
        )
        if grown_count:
            logger.warning(
                "SegsSetCropSize: %d segment(s) exceeded the target size:"
                " bbox larger than requested",
                grown_count,
            )
        if clamped_count:
            logger.warning(
                "SegsSetCropSize: %d segment(s) clamped to the %dx%d canvas",
                clamped_count,
                canvas[0],
                canvas[1],
            )
        return (((segs_h, segs_w), fitted_items),)
