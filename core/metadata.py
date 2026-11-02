'''
File:   metadata.py
Brief:  Metadata-block primitives for LUKUTAR_METADATA socket type.
Author: Mistress-Lukutar
Date:   2026-11-02
Version: v0.1.0
'''

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

#: The ComfyUI socket type name shared by all metadata nodes.
METADATA_TYPE: str = "LUKUTAR_METADATA"

#: Maximum number of editable name/value blocks a MetadataBlocksNode
#: may expose. Mirrors the static input declaration in the node class.
MAX_METADATA_BLOCKS: int = 16

#: Fallback format when the user leaves datetime_format blank.
DEFAULT_DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"

#: Pattern matching ``meta_N`` kwargs that arrive from the Merge node's
#: dynamically-grown sockets (managed by web/js/metadata_merge.js).
_META_INPUT_RE = re.compile(r"^meta_(\d+)$")


def build_metadata(
    num_blocks: int,
    names: Sequence[str],
    values: Sequence[str],
) -> dict[str, str]:
    '''Build a metadata dict from name/value pairs.

    Takes the first ``num_blocks`` pairs (clamped to ``[1,
    MAX_METADATA_BLOCKS]`` and to the shortest of ``names``/``values``),
    skips entries whose stripped name is empty, strips both name and
    value, coerces non-str values via ``str()``, and lets later-wins
    handle duplicates.

    Args:
        num_blocks: Desired number of blocks (clamped internally).
        names: Raw name values; each is stripped and coerced to str.
        values: Raw value values; each is stripped and coerced to str.

    Returns:
        A fresh insertion-ordered ``dict[str, str]``.
    '''
    lo = 1
    hi = MAX_METADATA_BLOCKS
    count = max(lo, min(num_blocks, hi))
    length = min(count, len(names), len(values))

    result: dict[str, str] = {}
    skipped = 0
    for i in range(length):
        raw_name = str(names[i]).strip()
        raw_value = str(values[i]).strip()
        if not raw_name:
            skipped += 1
            continue
        result[raw_name] = raw_value

    if skipped > 0:
        logger.info(
            "build_metadata: skipped %d pair(s) with empty name", skipped
        )

    return result


def merge_metadata(
    *blocks: Mapping[str, Any] | None,
) -> dict[str, str]:
    '''Merge several metadata mappings into one.

    Iterates blocks in order, skipping ``None``; each block's items
    update the result so later blocks override duplicate keys. All
    values are coerced via ``str()``.

    Args:
        *blocks: One or more ``Mapping[str, Any]`` (or ``None``).

    Returns:
        A fresh insertion-ordered ``dict[str, str]``.
    '''
    result: dict[str, str] = {}
    for block in blocks:
        if block is None:
            continue
        for key, value in block.items():
            result[key] = str(value)
    return result


def standard_metadata(
    include_workflow: bool,
    include_prompt: bool,
    include_datetime: bool,
    datetime_format: str,
    prompt: Mapping[str, Any] | None = None,
    extra_pnginfo: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    '''Build a metadata dict with workflow/prompt/datetime entries.

    Keys are emitted in this fixed order when their flag is ``True``:

    * ``"prompt"`` — JSON-serialised ComfyUI execution prompt.
    * ``"workflow"`` — JSON-serialised ``extra_pnginfo["workflow"]``.
    * ``"datetime"`` — local time formatted with ``datetime_format``
      (falls back to :data:`DEFAULT_DATETIME_FORMAT` when blank).

    Args:
        include_workflow: Emit the workflow entry.
        include_prompt: Emit the prompt entry.
        include_datetime: Emit the datetime entry.
        datetime_format: strftime format string for the datetime value.
        prompt: ComfyUI's hidden PROMPT input dict (may be ``None``).
        extra_pnginfo: ComfyUI's hidden EXTRA_PNGINFO dict (may be
            ``None``).

    Returns:
        A fresh ``dict[str, str]`` with the selected entries.
    '''
    result: dict[str, str] = {}

    if include_prompt and prompt is not None:
        result["prompt"] = json.dumps(prompt, ensure_ascii=False, default=str)

    if (
        include_workflow
        and extra_pnginfo is not None
        and extra_pnginfo.get("workflow") is not None
    ):
        result["workflow"] = json.dumps(
            extra_pnginfo["workflow"], ensure_ascii=False, default=str
        )

    if include_datetime:
        fmt = datetime_format.strip() or DEFAULT_DATETIME_FORMAT
        result["datetime"] = datetime.now().strftime(fmt)

    return result


# ------------------------------------------------------------------ #
# Natural-sort helpers for merge-node kwargs ordering                #
# ------------------------------------------------------------------ #

#: Compiled regex patterns cached per prefix (keyed by prefix string).
_NUMBERED_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _numbered_input_key(name: str, prefix: str) -> tuple[int, int, str]:
    '''Sort key for ``{prefix}N`` names.

    Names matching ``^{prefix}(\\d+)$`` sort first ascending by their
    integer suffix; all other names sort after alphabetically.

    Args:
        name: The name to produce a sort key for.
        prefix: The literal prefix before the numeric suffix (e.g.
            ``"meta_"``).

    Returns:
        A tuple suitable for use in ``sorted()`` / ``list.sort()``.
    '''
    pattern = _NUMBERED_RE_CACHE.get(prefix)
    if pattern is None:
        escaped_prefix = re.escape(prefix)
        pattern = re.compile(rf"^{escaped_prefix}(\d+)$")
        _NUMBERED_RE_CACHE[prefix] = pattern

    match = pattern.fullmatch(name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, 0, name)


def sort_numbered_inputs(
    names: Iterable[str],
    prefix: str = "meta_",
) -> list[str]:
    '''Natural-sort a collection of names by numeric suffix.

    Names matching ``{prefix}N`` are sorted first in ascending order
    by *N*; all other names follow, sorted alphabetically.

    Args:
        names: The names to sort.
        prefix: The literal prefix before the numeric suffix (default
            ``"meta_"``).

    Returns:
        A new list of names in natural sorted order.
    '''
    return sorted(names, key=lambda n: _numbered_input_key(n, prefix))


def sort_meta_inputs(names: Iterable[str]) -> list[str]:
    '''Return names sorted in natural ``meta_N`` order.

    Used to order the kwargs received by :class:`MergeMetadataNode`.

    Args:
        names: The kwarg keys arriving from ComfyUI's executor.

    Returns:
        A new list of names in the desired iteration order.
    '''
    return sort_numbered_inputs(names, "meta_")
