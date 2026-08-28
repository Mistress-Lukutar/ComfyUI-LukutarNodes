'''
File:   __init__.py
Brief:  Aggregation of node classes for the LukutarNodes pack.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.7.0
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
from .variables import GetVariableNode, SetVariableNode

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "ColorMatch": ColorMatchNode,
    "SegsOverlay": SegsOverlayNode,
    "PromptAnnotate": PromptAnnotateNode,
    "AnnotationsWildcard": AnnotationsWildcardNode,
    "AnnotationSegment": AnnotationSegmentNode,
    "AnnotationLabels": AnnotationLabelsNode,
    "AnnotationSegmentEdit": AnnotationSegmentEditNode,
    "SetVariable": SetVariableNode,
    "GetVariable": GetVariableNode,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "ColorMatch": "Color Match (Frequency Split)",
    "SegsOverlay": "SEGS BBox Overlay",
    "PromptAnnotate": "Prompt Annotate",
    "AnnotationsWildcard": "Annotations to Wildcard (LAB)",
    "AnnotationSegment": "Annotation Segment",
    "AnnotationLabels": "Annotation Labels",
    "AnnotationSegmentEdit": "Annotation Segment Edit",
    "SetVariable": "Set Variable",
    "GetVariable": "Get Variable",
}

__all__ = [
    "AnnotationLabelsNode",
    "AnnotationSegmentEditNode",
    "AnnotationSegmentNode",
    "AnnotationsWildcardNode",
    "ColorMatchNode",
    "GetVariableNode",
    "PromptAnnotateNode",
    "SegsOverlayNode",
    "SetVariableNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
