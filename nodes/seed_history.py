'''
File:   seed_history.py
Brief:  ComfyUI node — seed generator with persistent history.
Author: Mistress-Lukutar
Date:   2026-10-17
Version: v0.3.0

No inputs, one INT output (``seed``).  Parameters mirror the built-in
seed widgets: ``seed`` plus ``history_size``.  The
``control_after_generate`` combo is intentionally NOT declared here —
the ComfyUI frontend adds it automatically to every INT widget named
``seed``/``noise_seed``, applies it client-side (fixed / increment /
decrement / randomize) and never sends it to the backend, so declaring
it breaks prompt validation with "Required input is missing".

The rolling seed log is kept by ``web/js/seed_history.js`` in the node
property ``seed_history``, which ComfyUI serializes with the workflow,
so the history survives reloads.  Without the web assets the node
still works as a plain seed source — the history is simply not
displayed.
'''

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SeedHistoryNode:
    '''Generate a seed and let the frontend keep a history of used seeds.

    This node has no inputs and produces one ``INT`` output so it wires
    into any ``seed``/``noise_seed`` input.  The seed value and its
    after-generate behavior are configured through the panel widgets
    exactly like ComfyUI's built-in seed sources: the frontend applies
    ``control_after_generate`` before this node ever runs.  The
    additional ``history_size`` parameter tells the web extension how
    many past seeds to keep rendered as copyable text below the
    parameters.
    '''

    CATEGORY = "Lukutar/Seeds"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**64 - 1,
                        "tooltip": "Starting seed value",
                    },
                ),
                "history_size": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 1000,
                        "tooltip": (
                            "Maximum number of past seeds kept in the "
                            "history shown below the parameters"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "generate"

    def generate(self, seed: int, history_size: int) -> tuple[int]:
        '''Pass the seed the frontend already prepared for this run.

        Args:
            seed: Widget value after the frontend applied
                ``control_after_generate`` (the combo widget is
                frontend-only and is never part of the prompt).
            history_size: Maximum history length; consumed by the web
                extension, validated here so API workflows stay honest.

        Returns:
            One-element tuple with the seed for this generation.
        '''
        logger.debug(
            "SeedHistory: seed=%d, history_size=%d", seed, history_size
        )
        return (int(seed),)
