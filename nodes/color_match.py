'''
File:   color_match.py
Brief:  ComfyUI node wrapping the frequency-separation color matcher.
Author: Mistress-Lukutar
Date:   2026-08-24
Version: v0.1.0
'''

from __future__ import annotations

import logging

import torch

from ..core.color_matcher import (
    DEFAULT_EVAL_SIGMA,
    EVAL_METRICS,
    MATCH_METHODS,
    METHOD_REINHARD,
    ColorMatcher,
    tune_sigma,
)
from ..utils.images import frames_to_tensor, tensor_to_frames

logger = logging.getLogger(__name__)


class ColorMatchNode:
    '''Restore the color envelope of a reference onto a processed image.

    Splits the input into low/high frequency layers with a Gaussian blur,
    transfers the color statistics of the reference onto the low layer and
    recombines it with the original high-frequency detail. Typical use:
    fixing VAE encode-decode color drift on upscales while keeping the
    upscaled detail intact.
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
                            "Processed image (e.g. upscaled) to recolor"
                        )
                    },
                ),
                "reference": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Image with the target color distribution,"
                            " e.g. the original before upscaling"
                        )
                    },
                ),
                "auto_tune": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "auto",
                        "label_off": "manual",
                        "tooltip": (
                            "Grid-search sigma on the first frame and reuse"
                            " the best value for the whole batch"
                        ),
                    },
                ),
                "sigma": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 0.1,
                        "max": 500.0,
                        "step": 0.5,
                        "tooltip": (
                            "Gaussian blur radius = frequency cutoff"
                            " (ignored when auto_tune is on)"
                        ),
                    },
                ),
                "method": (
                    list(MATCH_METHODS),
                    {
                        "default": METHOD_REINHARD,
                        "tooltip": (
                            "reinhard: transfer LAB mean/std; replace: take"
                            " the reference low layer as-is"
                        ),
                    },
                ),
            },
            "optional": {
                "sigma_min": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.1,
                        "max": 500.0,
                        "step": 0.5,
                        "tooltip": "Auto-tune grid start",
                    },
                ),
                "sigma_max": (
                    "FLOAT",
                    {
                        "default": 60.0,
                        "min": 0.1,
                        "max": 500.0,
                        "step": 0.5,
                        "tooltip": "Auto-tune grid end",
                    },
                ),
                "sigma_step": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.1,
                        "max": 100.0,
                        "step": 0.5,
                        "tooltip": "Auto-tune grid step",
                    },
                ),
                "eval_sigma": (
                    "FLOAT",
                    {
                        "default": DEFAULT_EVAL_SIGMA,
                        "min": 0.1,
                        "max": 500.0,
                        "step": 0.5,
                        "tooltip": (
                            "Fixed blur radius used for auto-tune scoring"
                        ),
                    },
                ),
                "metric": (
                    list(EVAL_METRICS),
                    {
                        "default": EVAL_METRICS[0],
                        "tooltip": (
                            "envelope: color only; full: color + structure"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("image", "sigma")
    FUNCTION = "match_colors"

    def match_colors(
        self,
        image: torch.Tensor,
        reference: torch.Tensor,
        auto_tune: bool,
        sigma: float,
        method: str = METHOD_REINHARD,
        sigma_min: float = 5.0,
        sigma_max: float = 60.0,
        sigma_step: float = 5.0,
        eval_sigma: float = DEFAULT_EVAL_SIGMA,
        metric: str = EVAL_METRICS[0],
    ) -> tuple[torch.Tensor, float]:
        '''Match the reference color envelope frame by frame.

        Args:
            image: ComfyUI IMAGE batch to recolor, (B, H, W, 3) in [0, 1].
            reference: ComfyUI IMAGE batch with target colors. A single
                frame is broadcast to every frame of `image`; otherwise
                batch sizes must match.
            auto_tune: Grid-search sigma on the first frame pair.
            sigma: Manual sigma, returned unchanged when auto_tune is off.
            method: Color transfer method, one of MATCH_METHODS.
            sigma_min: Auto-tune grid start.
            sigma_max: Auto-tune grid end.
            sigma_step: Auto-tune grid step.
            eval_sigma: Fixed blur radius for the envelope metric.
            metric: Auto-tune scoring metric, one of EVAL_METRICS.

        Returns:
            Tuple of (recolored IMAGE batch, sigma actually used).

        Raises:
            ValueError: If batch sizes cannot be aligned.
        '''
        frames = tensor_to_frames(image)
        references = tensor_to_frames(reference)

        if len(references) == 1 and len(frames) > 1:
            references = references * len(frames)
        if len(references) != len(frames):
            raise ValueError(
                f"Batch mismatch: image has {len(frames)} frame(s) but "
                f"reference has {len(tensor_to_frames(reference))}"
            )

        if auto_tune:
            tuned = tune_sigma(
                references[0],
                frames[0],
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sigma_step=sigma_step,
                eval_sigma=eval_sigma,
                metric=metric,
                method=method,
            )
            sigma = tuned.sigma
            logger.info(
                "ColorMatch auto-tune: sigma=%.2f score=%.3f",
                tuned.sigma,
                tuned.score,
            )

        matcher = ColorMatcher(sigma=sigma, method=method)
        results = [
            matcher.process(ref, frame)
            for ref, frame in zip(references, frames, strict=True)
        ]
        logger.debug(
            "ColorMatch processed %d frame(s), sigma=%.2f", len(results), sigma
        )
        return (frames_to_tensor(results), float(sigma))
