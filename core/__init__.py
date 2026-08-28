'''
File:   __init__.py
Brief:  Core processing engines for the LukutarNodes pack.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.6.0
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
from .prompt_annotator import (
    DEFAULT_LABEL,
    EDIT_MODES,
    IMPACT_ALL_KEY,
    IMPACT_LAB_HEADER,
    AnnotatedPrompt,
    PromptAnnotateError,
    PromptSpan,
    edit_segment,
    labels_text,
    parse_annotated_prompt,
    segment_text,
    to_impact_wildcard,
    to_markup,
)
from .segs_crop_fitter import (
    DEFAULT_ROUND_TO,
    FIT_MODES,
    MODE_ASPECT,
    MODE_EXACT,
    FittedRegion,
    fit_crop_region,
    realign_mask,
)

__all__ = [
    "COLOR_AUTO",
    "COLOR_MODES",
    "COLOR_SINGLE",
    "DEFAULT_EVAL_SIGMA",
    "DEFAULT_LABEL",
    "DEFAULT_ROUND_TO",
    "EDIT_MODES",
    "EVAL_METRICS",
    "FIT_MODES",
    "IMPACT_ALL_KEY",
    "IMPACT_LAB_HEADER",
    "LABEL_CONFIDENCE",
    "LABEL_FORMATS",
    "LABEL_TEXT",
    "MATCH_METHODS",
    "METRIC_ENVELOPE",
    "METRIC_FULL",
    "METHOD_REINHARD",
    "METHOD_REPLACE",
    "MODE_ASPECT",
    "MODE_EXACT",
    "PALETTE",
    "AnnotatedPrompt",
    "ColorMatchError",
    "ColorMatcher",
    "Detection",
    "DetectionRenderError",
    "DetectionRenderer",
    "FittedRegion",
    "FrequencySplit",
    "PromptAnnotateError",
    "PromptSpan",
    "TuneResult",
    "edit_segment",
    "fit_crop_region",
    "format_caption",
    "labels_text",
    "parse_annotated_prompt",
    "realign_mask",
    "scale_detections",
    "segment_text",
    "to_impact_wildcard",
    "to_markup",
    "tune_sigma",
]
