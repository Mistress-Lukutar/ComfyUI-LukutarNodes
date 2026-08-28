'''
File:   conftest.py
Brief:  Pytest bootstrap: load the torch-free subpackages for unit tests.
Author: Mistress-Lukutar
Date:   2026-08-24
Version: v0.2.0

Loads only ``core`` and the torch-free node modules (``nodes.variables``)
so the engine tests run under any python with numpy and opencv (no torch
required). Node-level behaviour is validated by
``tests/smoke_test_comfyui_load.py`` under ComfyUI's own python.
'''

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parent.parent
#: Import alias: the real folder name has a hyphen and cannot be imported.
PACKAGE_ALIAS = "comfyui_lukutar_nodes"


def _load_core() -> types.ModuleType:
    '''Expose the pack directory as a parent package and import ``core``.

    The root ``__init__.py`` is intentionally NOT executed: it imports the
    node classes, which require torch from the ComfyUI runtime.
    '''
    if f"{PACKAGE_ALIAS}.core" in sys.modules:
        return sys.modules[f"{PACKAGE_ALIAS}.core"]

    parent = types.ModuleType(PACKAGE_ALIAS)
    parent.__path__ = [str(PACK_ROOT)]
    sys.modules[PACKAGE_ALIAS] = parent
    return importlib.import_module(f"{PACKAGE_ALIAS}.core")


def _load_nodes_stub() -> None:
    '''Expose ``nodes`` as a stub package without executing its __init__.

    ``nodes/__init__.py`` imports every node module, several of which need
    torch; ``nodes/variables.py`` alone is torch-free, so tests import it
    through this stub parent instead.
    '''
    name = f"{PACKAGE_ALIAS}.nodes"
    if name in sys.modules:
        return
    stub = types.ModuleType(name)
    stub.__path__ = [str(PACK_ROOT / "nodes")]
    sys.modules[name] = stub


_load_core()
_load_nodes_stub()
