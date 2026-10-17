'''
File:   group_switch.py
Brief:  Pure helpers for the Group Switch node (state payload + ordering).
Author: Mistress-Lukutar
Date:   2026-10-17
Version: v1.1.0
'''

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Group disable methods, exposed as the node's combo choices.
DISABLE_METHODS: tuple[str, ...] = ("bypass", "disable")

#: ``bypass`` method: group nodes are switched to ComfyUI bypass mode.
METHOD_BYPASS = "bypass"
#: ``disable`` method: group nodes are switched to ComfyUI never (mute) mode.
METHOD_DISABLE = "disable"

#: Empty state payload, used until the first group row is created.
EMPTY_STATES = "{}"


class GroupSwitchError(ValueError):
    '''Raised when a Group Switch payload cannot be interpreted.'''


def parse_group_states(raw: str) -> dict[str, bool]:
    '''Parse the ``group_states`` widget value into an ordered mapping.

    Args:
        raw: JSON object literal produced by the web extension, e.g.
            ``'{"render": true, "upscale": false}'``. A blank value is
            treated as an empty mapping.

    Returns:
        Ordered ``{group title: enabled}`` mapping; the JSON object order
        is the row order shown on the node.

    Raises:
        GroupSwitchError: If the payload is not a JSON object with string
            keys, or a value cannot be read as a boolean.
    '''
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GroupSwitchError(
            f"group_states is not valid JSON: {exc.msg} (Lukutar)"
        ) from exc
    if not isinstance(data, dict):
        raise GroupSwitchError(
            "group_states must be a JSON object, got "
            f"{type(data).__name__} (Lukutar)"
        )
    states: dict[str, bool] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise GroupSwitchError(
                f"group_states keys must be strings, got {key!r} (Lukutar)"
            )
        states[key] = _as_bool(value)
    return states


def _as_bool(value: Any) -> bool:
    '''Coerce a JSON value to a boolean without surprising strings.'''
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def parse_sort_order(raw: str) -> list[str]:
    '''Split the comma-separated sort alphabet into lowercase tokens.

    Args:
        raw: e.g. ``"b, c, dg, de"`` -> ``["b", "c", "dg", "de"]``.

    Returns:
        Non-blank, trimmed, lowercased tokens in the given order.
    '''
    return [tok.strip().lower() for tok in (raw or "").split(",") if tok.strip()]


def sort_group_titles(
    titles: list[str], sort_order: str | list[str]
) -> list[str]:
    '''Order group titles by the sort alphabet of prefixes (stable).

    A title takes the position of the FIRST token it starts with
    (case-insensitive), so "in,im,de" orders "Inpaint Fix" before
    "Image Upscale" before "Detection"/"Detailer" without typing the
    full titles. Titles starting with no token keep their relative
    order and go last; several titles may share one token (stable).
    Tokens matching no title are ignored. With no tokens the input
    order is returned unchanged.

    Args:
        titles: Titles in their current (row) order.
        sort_order: Comma-separated alphabet or pre-parsed tokens.

    Returns:
        A new list holding the same titles, reordered.
    '''
    tokens = (
        parse_sort_order(sort_order)
        if isinstance(sort_order, str)
        else [tok.strip().lower() for tok in sort_order]
    )
    if not tokens:
        return list(titles)

    def rank(item: tuple[int, str]) -> int:
        index, title = item
        lowered = title.strip().lower()
        for position, token in enumerate(tokens):
            if lowered.startswith(token):
                return position
        return len(tokens) + index

    ordered = sorted(enumerate(titles), key=lambda item: (rank(item), item[0]))
    return [title for _, title in ordered]


def order_states(
    states: dict[str, bool], sort_order: str | list[str]
) -> dict[str, bool]:
    '''Re-sort an ordered state mapping by the sort alphabet.

    Args:
        states: Ordered ``{title: enabled}`` mapping.
        sort_order: Comma-separated alphabet or pre-parsed tokens.

    Returns:
        New mapping with the same entries in sorted order.
    '''
    titles = sort_group_titles(list(states), sort_order)
    return {title: states[title] for title in titles}


def check_disable_method(method: str) -> str:
    '''Validate the disable-method combo value.

    Args:
        method: Widget value; one of ``DISABLE_METHODS``.

    Returns:
        The validated method name.

    Raises:
        GroupSwitchError: If the value is not a known method.
    '''
    if method not in DISABLE_METHODS:
        raise GroupSwitchError(
            f"disable_method must be one of {list(DISABLE_METHODS)}, "
            f"got {method!r} (Lukutar)"
        )
    return method
