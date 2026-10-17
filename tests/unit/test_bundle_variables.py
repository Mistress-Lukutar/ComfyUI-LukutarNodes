'''
File:   test_bundle_variables.py
Brief:  Unit tests for the Set/Get Bundle nodes (torch-free).
Author: Mistress-Lukutar
Date:   2026-10-17
Version: v0.12.0
'''

from __future__ import annotations

import pytest
from comfyui_lukutar_nodes.nodes.bundle_variables import (
    MAX_SLOTS,
    GetBundleNode,
    SetBundleNode,
)


def test_input_types_declare_widgets_only() -> None:
    required = SetBundleNode.INPUT_TYPES()["required"]
    assert set(required) == {"bundle_name", "slots"}
    assert set(GetBundleNode.INPUT_TYPES()["required"]) == {
        "bundle_name",
        "slots",
    }


def test_return_types_are_padded_any_slots() -> None:
    for cls in (SetBundleNode, GetBundleNode):
        assert cls.RETURN_TYPES == ("*",) * MAX_SLOTS
        assert len(cls.RETURN_NAMES) == MAX_SLOTS


def test_set_mirrors_slots_in_spec_order() -> None:
    model, vae, clip = object(), object(), object()
    result = SetBundleNode().set_bundle(
        "sdxl",
        "model:MODEL, vae:VAE, clip:CLIP",
        clip=clip,
        model=model,
        vae=vae,
    )
    assert len(result) == MAX_SLOTS
    assert result[:3] == (model, vae, clip)
    assert result[3:] == (None,) * (MAX_SLOTS - 3)


def test_get_mirrors_slots_in_spec_order() -> None:
    model, vae = object(), object()
    result = GetBundleNode().get_bundle(
        "sdxl", "model:MODEL,vae:VAE", vae=vae, model=model
    )
    assert result[:2] == (model, vae)
    assert all(value is None for value in result[2:])


def test_set_without_spec_orders_fallback_slots_naturally() -> None:
    first, second, other = object(), object(), object()
    result = SetBundleNode().set_bundle(
        "sdxl", "", other=other, slot_2=second, slot_1=first
    )
    assert result[:3] == (first, second, other)


def test_missing_declared_slot_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="vae"):
        SetBundleNode().set_bundle("sdxl", "model:MODEL,vae:VAE", model=1)
    with pytest.raises(RuntimeError, match="vae"):
        GetBundleNode().get_bundle("sdxl", "model:MODEL,vae:VAE", model=1)


def test_value_without_spec_entry_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="stray"):
        SetBundleNode().set_bundle(
            "sdxl", "model:MODEL", model=1, stray=2
        )


def test_bundle_without_any_value_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="sdxl"):
        GetBundleNode().get_bundle("sdxl", "", )


def test_blank_bundle_name_is_rejected() -> None:
    for blank in ("", "   ", "\t"):
        with pytest.raises(ValueError, match="name"):
            SetBundleNode().set_bundle(blank, "", model=1)
        with pytest.raises(ValueError, match="name"):
            GetBundleNode().get_bundle(blank, "", model=1)


def test_name_is_trimmed_not_mutated() -> None:
    value = object()
    (first, *_rest) = SetBundleNode().set_bundle(
        "  sdxl  ", "model:MODEL", model=value
    )
    assert first is value


@pytest.mark.parametrize(
    "spec",
    [
        "model",  # missing :TYPE
        "model:",  # missing TYPE
        ":MODEL",  # missing name
        "model:MODEL,,vae:VAE,x",  # trailing malformed entry
        "model:MODEL,model:VAE",  # duplicate name
        "bundle_name:MODEL",  # reserved name
        "slots:MODEL",  # reserved name
    ],
)
def test_malformed_spec_is_rejected(spec: str) -> None:
    with pytest.raises(ValueError):
        SetBundleNode().set_bundle("sdxl", spec, model=1)


def test_spec_longer_than_max_slots_is_rejected() -> None:
    spec = ",".join(f"s{i}:ANY" for i in range(MAX_SLOTS + 1))
    kwargs = {f"s{i}": 1 for i in range(MAX_SLOTS + 1)}
    with pytest.raises(ValueError, match="at most"):
        SetBundleNode().set_bundle("sdxl", spec, **kwargs)


def test_more_values_than_slots_without_spec_is_rejected() -> None:
    kwargs = {f"slot_{i}": object() for i in range(MAX_SLOTS + 1)}
    with pytest.raises(RuntimeError, match="at most"):
        GetBundleNode().get_bundle("sdxl", "", **kwargs)
