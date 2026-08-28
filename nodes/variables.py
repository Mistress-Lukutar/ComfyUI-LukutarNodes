'''
File:   variables.py
Brief:  ComfyUI nodes for named workflow variables (wireless Set/Get pair).
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.9.0
'''

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _checked_name(var_name: str) -> str:
    '''Trim a variable name and refuse blank ones.

    Args:
        var_name: Raw widget value.

    Returns:
        The trimmed variable name.

    Raises:
        ValueError: If the name is empty after trimming.
    '''
    name = var_name.strip()
    if not name:
        raise ValueError("Variable name must not be empty (Lukutar)")
    return name


class SetVariableNode:
    '''Publish any value under a name — the wireless anchor of a variable.

    A value fed here becomes available to every Get Variable node with
    the same ``var_name`` anywhere in the workflow, without dragging a
    wire across the canvas. The value keeps its exact type (IMAGE, MODEL,
    CONDITIONING — anything) and is passed through unchanged, so the
    output may also be wired normally. Several Set nodes may share one
    name on alternative branches as long as only one branch is active
    (muted/bypassed nodes never reach the prompt); two simultaneously
    active Sets with one name are ambiguous and fail the queue.
    '''

    CATEGORY = "Lukutar/Variables"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "var_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Variable name, e.g. img_t2i; Get Variable "
                            "nodes with this name receive the value"
                        ),
                    },
                ),
                "value": (
                    "*",
                    {
                        "tooltip": (
                            "Value to publish; any type, passed through "
                            "unchanged"
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "set_value"

    def set_value(self, var_name: str, value: Any) -> tuple[Any]:
        '''Validate the name and pass the value through.

        Args:
            var_name: Variable name from the widget.
            value: The value being published.

        Returns:
            One-element tuple with the same value.

        Raises:
            ValueError: If the variable name is blank.
        '''
        name = _checked_name(var_name)
        logger.debug("SetVariable [%s]: %s", name, type(value).__name__)
        return (value,)


class GetVariableNode:
    '''Read a named variable wherever it is needed — no wires required.

    The web extension normally connects this node to the matching Set
    Variable with an invisible real link, so execution order and output
    caching are handled by ComfyUI itself. Without the web assets (or to
    override the name) the ``value`` input can be wired manually; a
    manual wire always wins over the variable name.
    '''

    CATEGORY = "Lukutar/Variables"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "var_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Variable name to read, e.g. img_t2i; a "
                            "manual wire into value overrides it"
                        ),
                    },
                ),
            },
            "optional": {
                "value": (
                    "*",
                    {
                        "tooltip": (
                            "Connected automatically by the web "
                            "extension; may be wired manually instead"
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"

    def get_value(self, var_name: str, value: Any = None) -> tuple[Any]:
        '''Emit the wired value, with a helpful error when it is missing.

        Args:
            var_name: Variable name from the widget.
            value: Value arriving on the (invisible or manual) link.

        Returns:
            One-element tuple with the value.

        Raises:
            ValueError: If the variable name is blank.
            RuntimeError: If no value arrived on the input.
        '''
        name = _checked_name(var_name)
        if value is None:
            raise RuntimeError(
                f"GetVariable [{name}]: no value arrived. Check that a "
                "Set Variable with this name exists and its branch is "
                "active (not muted/bypassed), that it is the only active "
                "Set with this name, that its value input is connected — "
                "or wire this node's value input manually."
            )
        logger.debug("GetVariable [%s]: %s", name, type(value).__name__)
        return (value,)
