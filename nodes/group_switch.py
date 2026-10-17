'''
File:   group_switch.py
Brief:  ComfyUI node for switching graph groups on/off from one panel.
Author: Mistress-Lukutar
Date:   2026-10-17
Version: v1.1.0
'''

from __future__ import annotations

import logging

from ..core.group_switch import (
    DISABLE_METHODS,
    EMPTY_STATES,
    METHOD_BYPASS,
    check_disable_method,
    order_states,
    parse_group_states,
)

logger = logging.getLogger(__name__)


class GroupSwitchNode:
    '''Toggle graph groups from a single panel node.

    The ``web/js/group_switch.js`` extension renders one boolean row per
    group on the canvas (rows appear, reorder and disappear as groups
    change) and applies the toggles: ``bypass`` sets the group's nodes to
    ComfyUI bypass mode, ``disable`` sets them to never (mute) — the same
    mechanism the native group context menu uses. This class is the
    backend contract: it validates the parameters and republishes the
    current group states as one ``GROUP_STATES`` dictionary that can be
    wired anywhere (e.g. into Set Variable) or left unconnected. Without
    the web assets the node still loads; wire the ``group_states`` JSON
    widget manually to drive it from API workflows.
    '''

    CATEGORY = "Lukutar/Groups"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "disable_method": (
                    DISABLE_METHODS,
                    {
                        "default": METHOD_BYPASS,
                        "tooltip": (
                            "How an unchecked group is switched off: "
                            "bypass (nodes pass inputs through) or disable "
                            "(nodes are muted)"
                        ),
                    },
                ),
                "change_group_color": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Recolor a switched-off group's frame: purple "
                            "for bypass, grey for disable; restored when "
                            "the group is switched back on"
                        ),
                    },
                ),
                "sort_order": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Comma-separated title prefixes for the row "
                            "order, e.g. inp,im,det: a group whose title "
                            "starts with a token takes the token's "
                            "position (case-insensitive); unlisted "
                            "groups keep their order at the end"
                        ),
                    },
                ),
                "group_states": (
                    "STRING",
                    {
                        "default": EMPTY_STATES,
                        "tooltip": (
                            "Current states as JSON, managed automatically "
                            "by the web extension: "
                            '{"group title": true/false}'
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("GROUP_STATES",)
    RETURN_NAMES = ("states",)
    FUNCTION = "switch"

    def switch(
        self,
        disable_method: str,
        change_group_color: bool,
        sort_order: str,
        group_states: str,
    ) -> tuple[dict[str, bool]]:
        '''Validate the panel state and republish it as a dictionary.

        Args:
            disable_method: ``bypass`` or ``disable``.
            change_group_color: Whether the extension recolors disabled
                groups (frontend-only; accepted for contract symmetry).
            sort_order: Comma-separated title prefixes applied to the
                output ordering.
            group_states: JSON object with the current row states.

        Returns:
            One-element tuple with the ordered ``{title: enabled}``
            dictionary.

        Raises:
            GroupSwitchError: If the method or the state payload is
                invalid.
        '''
        method = check_disable_method(disable_method)
        states = parse_group_states(group_states)
        ordered = order_states(states, sort_order)
        logger.debug(
            "GroupSwitch: %d group(s), %d on, method=%s, color=%s",
            len(ordered),
            sum(1 for on in ordered.values() if on),
            method,
            bool(change_group_color),
        )
        return (ordered,)
