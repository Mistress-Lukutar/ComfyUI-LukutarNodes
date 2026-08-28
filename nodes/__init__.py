'''
File:   __init__.py
Brief:  Aggregation of node classes for the LukutarNodes pack.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.8.0
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
from .segs_set_crop_size import SegsSetCropSizeNode
from .variables import GetVariableNode, SetVariableNode

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "ColorMatch": ColorMatchNode,
    "SegsOverlay": SegsOverlayNode,
    "SegsSetCropSize": SegsSetCropSizeNode,
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
    "SegsSetCropSize": "SEGS Set Crop Size",
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
    "SegsSetCropSizeNode",
    "SetVariableNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
