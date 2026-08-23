'''
File:   __init__.py
Brief:  Aggregation of node classes for the LukutarNodes pack.
Author: Mistress-Lukutar
Date:   2026-08-24
Version: v0.1.0
'''

from __future__ import annotations

from .color_match import ColorMatchNode

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "ColorMatch": ColorMatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "ColorMatch": "Color Match (Frequency Split)",
}

__all__ = [
    "ColorMatchNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
