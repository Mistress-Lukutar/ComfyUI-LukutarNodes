'''
File:   __init__.py
Brief:  Aggregation of node classes for the LukutarNodes pack.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.8.0
'''

from __future__ import annotations

from .bundle_variables import GetBundleNode, SetBundleNode
from .color_match import ColorMatchNode
from .group_switch import GroupSwitchNode
from .prompt_annotate import (
    AnnotationLabelsNode,
    AnnotationSegmentEditNode,
    AnnotationSegmentNode,
    AnnotationsWildcardNode,
    PromptAnnotateNode,
)
from .seed_history import SeedHistoryNode
from .segs_overlay import SegsOverlayNode
from .segs_set_crop_size import SegsSetCropSizeNode
from .variables import GetVariableNode, SetVariableNode

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "SetBundle": SetBundleNode,
    "GetBundle": GetBundleNode,
    "ColorMatch": ColorMatchNode,
    "GroupSwitch": GroupSwitchNode,
    "SegsOverlay": SegsOverlayNode,
    "SegsSetCropSize": SegsSetCropSizeNode,
    "PromptAnnotate": PromptAnnotateNode,
    "AnnotationsWildcard": AnnotationsWildcardNode,
    "AnnotationSegment": AnnotationSegmentNode,
    "AnnotationLabels": AnnotationLabelsNode,
    "AnnotationSegmentEdit": AnnotationSegmentEditNode,
    "SeedHistory": SeedHistoryNode,
    "SetVariable": SetVariableNode,
    "GetVariable": GetVariableNode,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "SetBundle": "Set Bundle",
    "GetBundle": "Get Bundle",
    "ColorMatch": "Color Match (Frequency Split)",
    "GroupSwitch": "Group Switch",
    "SegsOverlay": "SEGS BBox Overlay",
    "SegsSetCropSize": "SEGS Set Crop Size",
    "PromptAnnotate": "Prompt Annotate",
    "AnnotationsWildcard": "Annotations to Wildcard (LAB)",
    "AnnotationSegment": "Annotation Segment",
    "AnnotationLabels": "Annotation Labels",
    "AnnotationSegmentEdit": "Annotation Segment Edit",
    "SeedHistory": "Seed History",
    "SetVariable": "Set Variable",
    "GetVariable": "Get Variable",
}

__all__ = [
    "AnnotationLabelsNode",
    "GetBundleNode",
    "SetBundleNode",
    "AnnotationSegmentEditNode",
    "AnnotationSegmentNode",
    "AnnotationsWildcardNode",
    "ColorMatchNode",
    "GetVariableNode",
    "GroupSwitchNode",
    "PromptAnnotateNode",
    "SeedHistoryNode",
    "SegsOverlayNode",
    "SegsSetCropSizeNode",
    "SetVariableNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
