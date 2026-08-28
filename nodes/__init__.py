'''
File:   __init__.py
Brief:  Aggregation of node classes for the LukutarNodes pack.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.6.0
'''

from __future__ import annotations

from .color_match import ColorMatchNode
from .prompt_annotate import (
    AnnotationLabelsNode,
    AnnotationSegmentEditNode,
    AnnotationSegmentNode,
    AnnotationsWildcardNode,
    PromptAnnotateNode,
)
from .segs_overlay import SegsOverlayNode

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "ColorMatch": ColorMatchNode,
    "SegsOverlay": SegsOverlayNode,
    "PromptAnnotate": PromptAnnotateNode,
    "AnnotationsWildcard": AnnotationsWildcardNode,
    "AnnotationSegment": AnnotationSegmentNode,
    "AnnotationLabels": AnnotationLabelsNode,
    "AnnotationSegmentEdit": AnnotationSegmentEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "ColorMatch": "Color Match (Frequency Split)",
    "SegsOverlay": "SEGS BBox Overlay",
    "PromptAnnotate": "Prompt Annotate",
    "AnnotationsWildcard": "Annotations to Wildcard (LAB)",
    "AnnotationSegment": "Annotation Segment",
    "AnnotationLabels": "Annotation Labels",
    "AnnotationSegmentEdit": "Annotation Segment Edit",
}

__all__ = [
    "AnnotationLabelsNode",
    "AnnotationSegmentEditNode",
    "AnnotationSegmentNode",
    "AnnotationsWildcardNode",
    "ColorMatchNode",
    "PromptAnnotateNode",
    "SegsOverlayNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
