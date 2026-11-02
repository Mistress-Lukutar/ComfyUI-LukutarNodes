'''
File:   metadata.py
Brief:  ComfyUI nodes for LUKUTAR_METADATA socket type.
Author: Mistress-Lukutar
Date:   2026-11-02
Version: v0.1.0
'''

from __future__ import annotations

import logging
from typing import Any

from ..core.metadata import (
    DEFAULT_DATETIME_FORMAT,
    MAX_METADATA_BLOCKS,
    METADATA_TYPE,
    build_metadata,
    merge_metadata,
    sort_meta_inputs,
    standard_metadata,
)

logger = logging.getLogger(__name__)


class MetadataBlocksNode:
    '''Create a metadata dict from *N* editable name/value pairs.

    The node statically declares 16 ``name_i`` / ``value_i`` input pairs
    (``i`` in ``1..MAX_METADATA_BLOCKS``) so it works fully without the
    web extension; :data:`num_blocks` controls how many are actually
    passed to the engine. The web extension (``web/js/metadata_blocks.js``)
    hides the pairs beyond ``num_blocks`` at runtime.

    Every call returns a fresh ``dict[str, str]`` — ComfyUI caches
    outputs so callers must never receive a shared mutable object.
    '''

    CATEGORY = "Lukutar/Metadata"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        required: dict[str, tuple] = {
            "num_blocks": (
                "INT",
                {
                    "default": 1,
                    "min": 1,
                    "max": MAX_METADATA_BLOCKS,
                    "tooltip": (
                        "How many name/value metadata blocks to expose."
                    ),
                },
            ),
        }
        for i in range(1, MAX_METADATA_BLOCKS + 1):
            required[f"name_{i}"] = ("STRING", {"default": ""})
            required[f"value_{i}"] = (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                },
            )
        return {"required": required}

    RETURN_TYPES = (METADATA_TYPE,)
    RETURN_NAMES = ("metadata",)
    FUNCTION = "create_blocks"

    def create_blocks(
        self, num_blocks: int, **kwargs: Any
    ) -> tuple[dict[str, str]]:
        '''Collect inputs and delegate to :func:`build_metadata`.

        Args:
            num_blocks: How many name/value pairs to process.
            **kwargs: The ``name_i`` / ``value_i`` widgets from ComfyUI.

        Returns:
            One-element tuple with the resulting metadata dict.
        '''
        names = [
            str(kwargs.get(f"name_{i}", ""))
            for i in range(1, MAX_METADATA_BLOCKS + 1)
        ]
        values = [
            str(kwargs.get(f"value_{i}", ""))
            for i in range(1, MAX_METADATA_BLOCKS + 1)
        ]
        result = build_metadata(num_blocks, names, values)
        return (result,)


class StandardMetadataNode:
    '''Emit standard ComfyUI metadata entries (workflow / prompt / datetime).

    Reads the hidden ``PROMPT`` and ``EXTRA_PNGINFO`` inputs that
    ComfyUI injects into every node's kwargs, serialises the selected
    fields to JSON, and returns them as a single metadata dict.
    '''

    CATEGORY = "Lukutar/Metadata"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "include_workflow": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Include the UI workflow JSON "
                            "(ComfyUI's hidden EXTRA_PNGINFO)."
                        ),
                    },
                ),
                "include_prompt": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Include the execution prompt JSON "
                            "(ComfyUI's hidden PROMPT input)."
                        ),
                    },
                ),
                "include_datetime": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Add the current local date/time.",
                    },
                ),
                "datetime_format": (
                    "STRING",
                    {
                        "default": DEFAULT_DATETIME_FORMAT,
                        "tooltip": (
                            "strftime format used when include_datetime is on."
                        ),
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (METADATA_TYPE,)
    RETURN_NAMES = ("metadata",)
    FUNCTION = "build"

    def build(
        self,
        include_workflow: bool,
        include_prompt: bool,
        include_datetime: bool,
        datetime_format: str,
        prompt: dict[str, Any] | None = None,
        extra_pnginfo: dict[str, Any] | None = None,
    ) -> tuple[dict[str, str]]:
        '''Delegate to :func:`standard_metadata` and return the result.

        Args:
            include_workflow: Emit the workflow entry.
            include_prompt: Emit the prompt entry.
            include_datetime: Emit the datetime entry.
            datetime_format: strftime format for the datetime value.
            prompt: ComfyUI's hidden PROMPT input (may be ``None``).
            extra_pnginfo: ComfyUI's hidden EXTRA_PNGINFO dict
                (may be ``None``).

        Returns:
            One-element tuple with the resulting metadata dict.
        '''
        result = standard_metadata(
            include_workflow=include_workflow,
            include_prompt=include_prompt,
            include_datetime=include_datetime,
            datetime_format=datetime_format,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
        return (result,)


class MergeMetadataNode:
    '''Merge several metadata blocks into one.

    The ``INPUT_TYPES`` returns an empty ``{"required": {}}`` because
    slot inputs are grown client-side by
    ``web/js/metadata_merge.js`` (same mechanism as Set/Get Bundle).
    Hand-written API prompts may pass ``meta_1``, ``meta_2``, ... links
    ordered by natural ``meta_N`` order. The backend resolves them via
    :func:`sort_meta_inputs` and delegates to :func:`merge_metadata`.
    '''

    CATEGORY = "Lukutar/Metadata"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {"required": {}}

    RETURN_TYPES = (METADATA_TYPE,)
    RETURN_NAMES = ("metadata",)
    FUNCTION = "merge"

    def merge(self, **kwargs: Any) -> tuple[dict[str, str]]:
        '''Sort the incoming metadata kwargs and merge them.

        Args:
            **kwargs: Metadata block inputs arriving as ``meta_N`` keys.

        Returns:
            One-element tuple with the merged metadata dict.
        '''
        ordered = sort_meta_inputs(kwargs)
        blocks = [kwargs[key] for key in ordered if kwargs[key] is not None]
        result = merge_metadata(*blocks)
        return (result,)
