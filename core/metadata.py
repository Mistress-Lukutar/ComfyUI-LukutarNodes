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

from .a1111_parameters import quote_a1111

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


# ------------------------------------------------------------------ #
# Smart merge: fold entries into existing block text                 #
# ------------------------------------------------------------------ #


def _join_values(existing: str, addition: str) -> str:
    '''Join two metadata values belonging to the same block name.

    Single-line values are comma-joined (matching A1111 list entries
    such as ``Lora hashes``); when either side is multiline the values
    are newline-joined instead.

    Args:
        existing: The accumulated value.
        addition: The incoming value.

    Returns:
        The combined value string.
    '''
    if "\n" in existing or "\n" in addition:
        return f"{existing}\n{addition}"
    return f"{existing}, {addition}"


def _split_top_level(line: str) -> list[str]:
    '''Split an A1111 settings line on commas outside double quotes.

    JSON-escaped characters (``\\"``) inside quoted segments are kept
    verbatim and never treated as quote toggles.

    Args:
        line: The settings line to split.

    Returns:
        The raw comma-separated segments, whitespace untrimmed.
    '''
    parts: list[str] = []
    buf: list[str] = []
    in_quote = False
    i = 0
    while i < len(line):
        ch = line[i]
        if in_quote:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(line):
                buf.append(line[i + 1])
                i += 1
            elif ch == '"':
                in_quote = False
        elif ch == '"':
            in_quote = True
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _parse_settings_line(line: str) -> list[tuple[str, str]] | None:
    '''Parse an A1111 settings line into ``(key, raw_value)`` entries.

    Args:
        line: Candidate settings line (values may be quoted).

    Returns:
        The entries with keys stripped and raw (possibly quoted)
        values, or ``None`` when the line is not a settings line
        (any segment lacking a ``": "`` separator or a non-empty key).
    '''
    entries: list[tuple[str, str]] = []
    for part in _split_top_level(line):
        key, sep, value = part.partition(": ")
        key = key.strip()
        if not sep or not key:
            return None
        entries.append((key, value.strip()))
    return entries or None


def _unquote_a1111(text: str) -> str:
    '''Undo A1111 value quoting produced by :func:`quote_a1111`.

    Args:
        text: Raw entry value, optionally JSON-quoted.

    Returns:
        The unquoted value; a malformed quoted value falls back to
        plain quote-stripping.
    '''
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text[1:-1]
    return text


def _fold_into_entry(value: str, key: str, addition: str) -> str | None:
    '''Fold *addition* into a named entry of *value*'s settings line.

    Only the **last** line of *value* is considered (the A1111 infotext
    settings line). When it parses as ``"Key: value, ..."`` entries and
    one of them is named *key*, the addition is appended to that
    entry's value (comma-joined) and the entry is re-quoted with
    :func:`quote_a1111`.

    Args:
        value: The existing block text.
        key: The entry name to fold into.
        addition: The incoming data (already stripped).

    Returns:
        The rewritten block text, or ``None`` when *value* has no
        settings line or no entry named *key*.
    '''
    lines = value.split("\n")
    entries = _parse_settings_line(lines[-1])
    if not entries:
        return None
    for idx, (entry_key, entry_value) in enumerate(entries):
        if entry_key != key:
            continue
        merged = f"{_unquote_a1111(entry_value)}, {addition}"
        entries[idx] = (entry_key, quote_a1111(merged))
        lines[-1] = ", ".join(f"{k}: {v}" for k, v in entries)
        return "\n".join(lines)
    return None


def merge_metadata_smart(
    *blocks: Mapping[str, Any] | None,
) -> dict[str, str]:
    '''Merge metadata blocks without duplicating existing data.

    Blocks are processed in argument order, skipping ``None``. For
    every entry (empty names/values are skipped) the resolution is:

    #. **Same-name block** — the entry's name already exists in the
       result: the values are joined (comma for single-line values,
       newline when either side is multiline).
    #. **Settings-line fold** — the name appears as a ``"Key: value"``
       entry inside another block's *last* line (the A1111 infotext
       settings line, e.g. ``"Lora hashes"`` inside ``"parameters"``):
       the data is folded into that entry, honouring A1111 quoting, so
       no separate block is created.
    #. **New block** — otherwise a fresh block is appended.

    Args:
        *blocks: One or more ``Mapping[str, Any]`` (or ``None``).

    Returns:
        A fresh insertion-ordered ``dict[str, str]``; input mappings
        are never mutated.
    '''
    result: dict[str, str] = {}
    for block in blocks:
        if block is None:
            continue
        for key, value in block.items():
            name = str(key).strip()
            text = str(value).strip()
            if not name or not text:
                continue
            if name in result:
                result[name] = _join_values(result[name], text)
                continue
            folded = False
            for existing_key, existing_value in result.items():
                merged_value = _fold_into_entry(existing_value, name, text)
                if merged_value is not None:
                    result[existing_key] = merged_value
                    folded = True
                    break
            if not folded:
                result[name] = text
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
