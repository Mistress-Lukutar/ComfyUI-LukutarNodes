'''
File:   bundle_variables.py
Brief:  ComfyUI nodes for multi-value named bundles (Set/Get Bundle pair).
Author: Mistress-Lukutar
Date:   2026-10-17
Version: v0.12.0
'''

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: Hard cap on the number of slots per bundle. Must stay in sync with
#: MAX_SLOTS in ``web/js/bundle_variables.js``: the web extension never
#: grows a bundle past this, and both backends pad their return tuples
#: to exactly this length so ComfyUI's output indexing never overruns.
MAX_SLOTS = 16

#: Widget names that are real node inputs — slot names may not shadow them.
_RESERVED_NAMES = frozenset({"bundle_name", "slots"})

#: ``slot_<n>`` — the fallback input names the nodes expose without the
#: web extension (wired manually, value lands on the n-th output).
_FALLBACK_RE = re.compile(r"slot_(\d+)$")


def _checked_bundle_name(bundle_name: str) -> str:
    '''Trim a bundle name and refuse blank ones.

    Args:
        bundle_name: Raw widget value.

    Returns:
        The trimmed bundle name.

    Raises:
        ValueError: If the name is empty after trimming.
    '''
    name = bundle_name.strip()
    if not name:
        raise ValueError("Bundle name must not be empty (Lukutar)")
    return name


def _natural_slot_key(name: str) -> tuple[int, int, str]:
    '''Sort key putting ``slot_1..slot_9`` (numeric) before other names.

    Args:
        name: Slot name.

    Returns:
        Tuple ordering fallback slots by index first, everything else
        alphabetically after them.
    '''
    match = _FALLBACK_RE.fullmatch(name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, 0, name)


def _parse_slots(spec: str) -> list[tuple[str, str]]:
    '''Parse the slot spec maintained by the web extension.

    Args:
        spec: Comma-separated ``name:TYPE`` entries, e.g.
            ``"model:MODEL, vae:VAE"``. Blank entries are skipped.

    Returns:
        The ``(name, type)`` pairs in spec order.

    Raises:
        ValueError: If an entry is malformed, a name is reserved or
            duplicated, or the spec exceeds MAX_SLOTS entries.
    '''
    slots: list[tuple[str, str]] = []
    for raw in spec.split(","):
        entry = raw.strip()
        if not entry:
            continue
        name, sep, slot_type = entry.partition(":")
        name = name.strip()
        slot_type = slot_type.strip()
        if not sep or not name or not slot_type:
            raise ValueError(
                f"Malformed bundle slot entry {entry!r} "
                "(expected name:TYPE)"
            )
        if name in _RESERVED_NAMES:
            raise ValueError(
                f"Bundle slot name {name!r} is reserved (Lukutar)"
            )
        slots.append((name, slot_type))
    names = [name for name, _ in slots]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate bundle slot names: {', '.join(duplicates)}"
        )
    if len(slots) > MAX_SLOTS:
        raise ValueError(
            f"A bundle holds at most {MAX_SLOTS} slots "
            f"(got {len(slots)})"
        )
    return slots


def _ordered_values(
    kind: str,
    name: str,
    spec: list[tuple[str, str]],
    kwargs: dict[str, Any],
) -> list[Any]:
    '''Collect the incoming slot values in output order.

    With a spec (the normal, web-extension-serialized case) the values
    are ordered by the spec and both missing and unexpected slot links
    fail loudly — that combination can only come from a desynced spec
    widget or a hand-edited prompt. Without a spec (hand-wired
    slot_1..slot_N fallback inputs) the values are ordered by natural
    slot name so wiring slot_1..slot_N lands on outputs 1..N.

    Args:
        kind: "Set Bundle" or "Get Bundle", for error messages.
        name: The bundle name, for error messages.
        spec: Parsed slot spec (possibly empty).
        kwargs: Values arriving on the slot input links.

    Returns:
        The values in output order, without padding.

    Raises:
        RuntimeError: If declared slots are missing, unexpected links
            arrived, or there are more values than MAX_SLOTS.
    '''
    if spec:
        expected = dict(spec)
        missing = [slot for slot, _ in spec if slot not in kwargs]
        if missing:
            raise RuntimeError(
                f"{kind} [{name}]: slot(s) not connected: "
                f"{', '.join(missing)}. Wire them on the Set Bundle or "
                "remove the slots there."
            )
        unknown = sorted(
            key for key in kwargs if key not in expected
        )
        if unknown:
            raise RuntimeError(
                f"{kind} [{name}]: value(s) without a slot in the spec: "
                f"{', '.join(unknown)}. Re-select the bundle name on the "
                "Get Bundle node (the web extension keeps the two in "
                "sync)."
            )
        return [kwargs[slot] for slot, _ in spec]
    if len(kwargs) > MAX_SLOTS:
        raise RuntimeError(
            f"{kind} [{name}]: received {len(kwargs)} values but a "
            f"bundle holds at most {MAX_SLOTS}"
        )
    return [kwargs[key] for key in sorted(kwargs, key=_natural_slot_key)]


def _padded(values: list[Any]) -> tuple[Any, ...]:
    '''Pad the ordered values to exactly MAX_SLOTS output positions.'''
    return tuple(values) + (None,) * (MAX_SLOTS - len(values))


def _input_types() -> dict[str, dict[str, tuple]]:
    '''The INPUT_TYPES shared by both bundle nodes.

    Returns:
        The ``bundle_name`` and ``slots`` widget definitions. The slot
        links themselves are NOT declared: ComfyUI's validator only
        inspects declared inputs while the executor resolves every
        linked prompt input, so the web extension is free to grow typed
        sockets client-side and the functions receive them via kwargs.
    '''
    return {
        "required": {
            "bundle_name": (
                "STRING",
                {
                    "default": "",
                    "tooltip": (
                        "Bundle name, e.g. sdxl_base; Get Bundle nodes "
                        "with this name receive all slots"
                    ),
                },
            ),
            "slots": (
                "STRING",
                {
                    "default": "",
                    "tooltip": (
                        "Slot spec 'name:TYPE,...' managed by the web "
                        "extension (hidden in the UI); hand-written "
                        "API prompts may set it to order the slots"
                    ),
                },
            ),
        },
    }


class SetBundleNode:
    '''Publish several typed values under one name — a wireless bus.

    Each input slot (MODEL, VAE, CLIP, IMAGE — any mix) is mirrored to
    an output of the same name, so a bundle feeds every Get Bundle node
    with the same ``bundle_name`` anywhere in the workflow without
    dragging wires across the canvas. The web extension adds the slots
    (SEGS-Concat style: connecting the free ``+ add`` socket spawns the
    next one), names them, keeps the ``slots`` spec widget in sync and
    wires invisible real links to the Get side — real, so execution
    order and output caching stay ComfyUI's own.

    Several Set Bundles may share one name on alternative branches as
    long as only one branch is active (muted/bypassed nodes never reach
    the prompt); two simultaneously active Sets with one name are
    ambiguous and fail the queue. Without the web assets the node
    degrades to a fixed bus: wire the ``slot_1..slot_16`` fallback
    inputs, values pass through to the outputs positionally.
    '''

    CATEGORY = "Lukutar/Variables"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return _input_types()

    RETURN_TYPES = ("*",) * MAX_SLOTS
    RETURN_NAMES = tuple(f"slot_{i}" for i in range(1, MAX_SLOTS + 1))
    FUNCTION = "set_bundle"

    def set_bundle(
        self, bundle_name: str, slots: str = "", **kwargs: Any
    ) -> tuple[Any, ...]:
        '''Validate name and spec, and mirror every slot to its output.

        Args:
            bundle_name: Bundle name from the widget.
            slots: Slot spec from the widget (may be empty).
            kwargs: Values arriving on the slot input links.

        Returns:
            MAX_SLOTS-tuple with the slot values in spec order, padded
            with None.

        Raises:
            ValueError: If the bundle name is blank or the spec is
                malformed.
            RuntimeError: If declared slots are missing or unexpected
                links arrived.
        '''
        name = _checked_bundle_name(bundle_name)
        spec = _parse_slots(slots)
        values = _ordered_values("Set Bundle", name, spec, kwargs)
        logger.debug(
            "SetBundle [%s]: %d slot(s)", name, len(values)
        )
        return _padded(values)


class GetBundleNode:
    '''Read a named bundle wherever it is needed — no wires required.

    The web extension mirrors the matching Set Bundle's slots onto this
    node: one named input (the anchor for an invisible real link from
    the Set side) and one typed output per slot. A manual wire into an
    input always wins over the bundle name. Without the web assets the
    node exposes its ``slot_1..slot_16`` fallback inputs/outputs instead
    and passes values through in name order.
    '''

    CATEGORY = "Lukutar/Variables"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return _input_types()

    RETURN_TYPES = ("*",) * MAX_SLOTS
    RETURN_NAMES = tuple(f"slot_{i}" for i in range(1, MAX_SLOTS + 1))
    FUNCTION = "get_bundle"

    def get_bundle(
        self, bundle_name: str, slots: str = "", **kwargs: Any
    ) -> tuple[Any, ...]:
        '''Emit the wired slot values, with helpful errors when empty.

        Args:
            bundle_name: Bundle name from the widget.
            slots: Slot spec from the widget (may be empty).
            kwargs: Values arriving on the slot input links.

        Returns:
            MAX_SLOTS-tuple with the slot values in spec order, padded
            with None.

        Raises:
            ValueError: If the bundle name is blank or the spec is
                malformed.
            RuntimeError: If no values arrived or declared slots are
                missing.
        '''
        name = _checked_bundle_name(bundle_name)
        spec = _parse_slots(slots)
        if not kwargs:
            raise RuntimeError(
                f"Get Bundle [{name}]: no values arrived. Check that a "
                "Set Bundle with this name exists and its branch is "
                "active (not muted/bypassed), that it is the only "
                "active Set with this name, that its slots are "
                "connected — or wire this node's inputs manually."
            )
        values = _ordered_values("Get Bundle", name, spec, kwargs)
        logger.debug("GetBundle [%s]: %d slot(s)", name, len(values))
        return _padded(values)
