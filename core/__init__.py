'''
File:   __init__.py
Brief:  Core processing engines for the LukutarNodes pack.
Author: Mistress-Lukutar
Date:   2026-08-24
Version: v0.3.0
'''

from __future__ import annotations

from .color_matcher import (
    DEFAULT_EVAL_SIGMA,
    EVAL_METRICS,
    MATCH_METHODS,
    METHOD_REINHARD,
    METHOD_REPLACE,
    METRIC_ENVELOPE,
    METRIC_FULL,
    ColorMatcher,
    ColorMatchError,
    FrequencySplit,
    TuneResult,
    tune_sigma,
)
from .detection_renderer import (
    COLOR_AUTO,
    COLOR_MODES,
    COLOR_SINGLE,
    LABEL_CONFIDENCE,
    LABEL_FORMATS,
    LABEL_TEXT,
    PALETTE,
    Detection,
    DetectionRenderer,
    DetectionRenderError,
    format_caption,
    scale_detections,
)

__all__ = [
    "COLOR_AUTO",
    "COLOR_MODES",
    "COLOR_SINGLE",
    "DEFAULT_EVAL_SIGMA",
    "EVAL_METRICS",
    "LABEL_CONFIDENCE",
    "LABEL_FORMATS",
    "LABEL_TEXT",
    "MATCH_METHODS",
    "METRIC_ENVELOPE",
    "METRIC_FULL",
    "METHOD_REINHARD",
    "METHOD_REPLACE",
    "PALETTE",
    "ColorMatchError",
    "ColorMatcher",
    "Detection",
    "DetectionRenderError",
    "DetectionRenderer",
    "FrequencySplit",
    "TuneResult",
    "format_caption",
    "scale_detections",
    "tune_sigma",
]
