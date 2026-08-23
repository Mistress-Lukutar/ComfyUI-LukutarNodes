'''
File:   conftest.py
Brief:  Pytest configuration for the pack repository.
Author: Mistress-Lukutar
Date:   2026-08-24
Version: v0.1.0

The root __init__.py is the ComfyUI entry point, not a test module: pytest 9
imports the root package __init__ when the rootdir is a package, which fails
(the folder name has a hyphen and the node classes need torch from the
ComfyUI runtime). Explicitly ignore it during collection.
'''

from __future__ import annotations

collect_ignore = ["__init__.py"]
