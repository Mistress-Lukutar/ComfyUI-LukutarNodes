'''
File:   test_variables.py
Brief:  Unit tests for the Set/Get Variable nodes (torch-free).
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.9.0
'''

from __future__ import annotations

import pytest
from comfyui_lukutar_nodes.nodes.variables import (
    GetVariableNode,
    SetVariableNode,
)


def test_set_passes_value_through_unchanged() -> None:
    value = object()
    (result,) = SetVariableNode().set_value("img_t2i", value)
    assert result is value


def test_get_returns_wired_value() -> None:
    value = object()
    (result,) = GetVariableNode().get_value("img_t2i", value)
    assert result is value


def test_get_without_value_raises_with_name() -> None:
    with pytest.raises(RuntimeError, match="img_t2i"):
        GetVariableNode().get_value("img_t2i")


@pytest.mark.parametrize("name", ["", "   ", "\t"])
def test_blank_name_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        SetVariableNode().set_value(name, object())
    with pytest.raises(ValueError, match="name"):
        GetVariableNode().get_value(name, object())


def test_name_is_trimmed_not_mutated() -> None:
    value = object()
    (result,) = SetVariableNode().set_value("  img_t2i  ", value)
    assert result is value
