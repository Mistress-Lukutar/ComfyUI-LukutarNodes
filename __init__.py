'''
File:   __init__.py
Brief:  ComfyUI-LukutarNodes custom node pack entry point.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.5.1
'''

from __future__ import annotations

__version__ = "0.5.1"

#: Served to the ComfyUI frontend (prompt annotator popup editor).
WEB_DIRECTORY = "./web"

if __package__:
    # Normal case: imported as a package (ComfyUI's custom node loader,
    # smoke test) — relative imports resolve as usual.
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    # Imported as a bare module (e.g. pytest 9 imports the rootdir package
    # __init__ without package context). The real node classes need torch
    # from the ComfyUI runtime, so expose empty mappings instead of failing;
    # ComfyUI itself never takes this branch.
    NODE_CLASS_MAPPINGS: dict[str, type] = {}
    NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
]
