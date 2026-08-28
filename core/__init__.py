'''
File:   __init__.py
Brief:  Core processing engines for the LukutarNodes pack.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.5.0
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

__all__ = [
    "COLOR_AUTO",
    "COLOR_MODES",
    "COLOR_SINGLE",
    "DEFAULT_EVAL_SIGMA",
    "DEFAULT_LABEL",
    "EDIT_MODES",
    "EVAL_METRICS",
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
    "PALETTE",
    "AnnotatedPrompt",
    "ColorMatchError",
    "ColorMatcher",
    "Detection",
    "DetectionRenderError",
    "DetectionRenderer",
    "FrequencySplit",
    "PromptAnnotateError",
    "PromptSpan",
    "TuneResult",
    "edit_segment",
    "format_caption",
    "labels_text",
    "parse_annotated_prompt",
    "scale_detections",
    "segment_text",
    "to_impact_wildcard",
    "to_markup",
    "tune_sigma",
]
