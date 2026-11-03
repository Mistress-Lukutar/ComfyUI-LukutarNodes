'''
File:   __init__.py
Brief:  Aggregation of node classes for the LukutarNodes pack.
Author: Mistress-Lukutar
Date:   2026-11-02
Version: v0.10.0
'''

from __future__ import annotations

from .a1111 import (
    A1111HiresFixNode,
    A1111LoraHashesNode,
    A1111ModelInfoNode,
    A1111ParametersNode,
    A1111SamplingNode,
)
from .bundle_variables import GetBundleNode, SetBundleNode
from .color_match import ColorMatchNode
from .group_switch import GroupSwitchNode
from .metadata import (
    MergeMetadataNode,
    MetadataBlocksNode,
    StandardMetadataNode,
)
from .prompt_annotate import (
    AnnotationLabelsNode,
    AnnotationSegmentEditNode,
    AnnotationSegmentNode,
    AnnotationsToStringNode,
    AnnotationsWildcardNode,
    PromptAnnotateNode,
)
from .save_image import SaveImageNode
from .save_image_jxl import SaveImageJxlNode
from .seed_history import SeedHistoryNode
from .segs_overlay import SegsOverlayNode
from .segs_set_crop_size import SegsSetCropSizeNode
from .t2i import (
    T2IBundlePackNode,
    T2IBundleUnpackNode,
    T2IInfoPackNode,
    T2IInfoToA1111Node,
    T2IInfoUnpackNode,
    T2ILoaderNode,
    T2ILoraListNode,
    T2ISamplerNode,
)
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
    "AnnotationsToString": AnnotationsToStringNode,
    "AnnotationSegmentEdit": AnnotationSegmentEditNode,
    "SeedHistory": SeedHistoryNode,
    "SetVariable": SetVariableNode,
    "GetVariable": GetVariableNode,
    "T2ILoader": T2ILoaderNode,
    "T2ILoraList": T2ILoraListNode,
    "T2ISampler": T2ISamplerNode,
    "T2IInfoToA1111": T2IInfoToA1111Node,
    "T2IInfoUnpack": T2IInfoUnpackNode,
    "T2IInfoPack": T2IInfoPackNode,
    "T2IBundleUnpack": T2IBundleUnpackNode,
    "T2IBundlePack": T2IBundlePackNode,
    "LukutarSaveImage": SaveImageNode,
    "LukutarSaveImageJxl": SaveImageJxlNode,
    "MetadataBlocks": MetadataBlocksNode,
    "StandardMetadata": StandardMetadataNode,
    "MergeMetadata": MergeMetadataNode,
    "A1111HiresFix": A1111HiresFixNode,
    "A1111LoraHashes": A1111LoraHashesNode,
    "A1111ModelInfo": A1111ModelInfoNode,
    "A1111Parameters": A1111ParametersNode,
    "A1111Sampling": A1111SamplingNode,
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
    "AnnotationsToString": "Annotations to String",
    "AnnotationSegmentEdit": "Annotation Segment Edit",
    "SeedHistory": "Seed History",
    "SetVariable": "Set Variable",
    "GetVariable": "Get Variable",
    "T2ILoader": "T2I Loader",
    "T2ILoraList": "T2I LoRA List",
    "T2ISampler": "T2I Sampler",
    "T2IInfoToA1111": "Info to A1111 Parameters",
    "T2IInfoUnpack": "T2I Info Unpack",
    "T2IInfoPack": "T2I Info Pack",
    "T2IBundleUnpack": "T2I Bundle Unpack",
    "T2IBundlePack": "T2I Bundle Pack",
    "LukutarSaveImage": "Save Image (Metadata)",
    "LukutarSaveImageJxl": "Save Image JXL",
    "A1111HiresFix": "A1111 Hires Fix",
    "A1111LoraHashes": "A1111 Lora Hashes",
    "A1111ModelInfo": "A1111 Model Info",
    "A1111Parameters": "A1111 Parameters",
    "A1111Sampling": "A1111 Sampling",
    "MetadataBlocks": "Metadata Blocks",
    "StandardMetadata": "Standard Metadata",
    "MergeMetadata": "Merge Metadata",
}

__all__ = [
    "AnnotationLabelsNode",
    "GetBundleNode",
    "SetBundleNode",
    "AnnotationSegmentEditNode",
    "AnnotationSegmentNode",
    "AnnotationsToStringNode",
    "AnnotationsWildcardNode",
    "ColorMatchNode",
    "GetVariableNode",
    "A1111HiresFixNode",
    "A1111LoraHashesNode",
    "A1111ModelInfoNode",
    "A1111ParametersNode",
    "A1111SamplingNode",
    "GroupSwitchNode",
    "MergeMetadataNode",
    "MetadataBlocksNode",
    "PromptAnnotateNode",
    "SaveImageNode",
    "SaveImageJxlNode",
    "SeedHistoryNode",
    "SegsOverlayNode",
    "SegsSetCropSizeNode",
    "SetVariableNode",
    "StandardMetadataNode",
    "T2IBundlePackNode",
    "T2IBundleUnpackNode",
    "T2ILoaderNode",
    "T2ILoraListNode",
    "T2IInfoPackNode",
    "T2IInfoToA1111Node",
    "T2IInfoUnpackNode",
    "T2ISamplerNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
