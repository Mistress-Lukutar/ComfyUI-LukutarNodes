'''
File:   test_t2i_nodes.py
Brief:  Unit tests for the torch-free T2I node parts.
Author: Mistress-Lukutar
Date:   2026-11-03
Version: v0.1.0

nodes/t2i.py imports torch only through a guarded try, so the LoRA
list / bundle pack logic and the bundle validation run on any python.
'''

from __future__ import annotations

import pytest
from comfyui_lukutar_nodes.nodes.t2i import (
    BUNDLE_KEYS,
    MAX_LORA_SLOTS,
    T2IBundlePackNode,
    T2ILoaderNode,
    T2ILoraListNode,
    validate_bundle,
)

# ======================================================================
# T2ILoraListNode
# ======================================================================


def test_lora_list_node_claims_slots() -> None:
    """Claimed slots appear as (name, weight) pairs in order."""
    node = T2ILoraListNode()
    kwargs: dict[str, object] = {}
    for i in range(1, MAX_LORA_SLOTS + 1):
        kwargs[f"lora_{i}"] = "None"
        kwargs[f"weight_{i}"] = 1.0
    kwargs["lora_1"] = "first.safetensors"
    kwargs["weight_1"] = 0.8
    kwargs["lora_3"] = "third.safetensors"
    kwargs["weight_3"] = 1.25
    (result,) = node.build(**kwargs)
    assert result == (
        ("first.safetensors", 0.8),
        ("third.safetensors", 1.25),
    )


def test_lora_list_node_skips_zero_weights() -> None:
    """Zero-weight entries are dropped entirely."""
    node = T2ILoraListNode()
    kwargs: dict[str, object] = {
        "lora_1": "a.safetensors",
        "weight_1": 0.0,
    }
    (result,) = node.build(**kwargs)
    assert result == ()


def test_lora_list_node_empty() -> None:
    """All-None slots produce an empty tuple."""
    node = T2ILoraListNode()
    (result,) = node.build()
    assert result == ()


def test_lora_list_node_input_types_pairing() -> None:
    """Every lora_N combo is paired with a weight_N float."""
    inputs = T2ILoraListNode.INPUT_TYPES()["required"]
    for i in range(1, MAX_LORA_SLOTS + 1):
        combo = inputs[f"lora_{i}"]
        weight = inputs[f"weight_{i}"]
        assert combo[0][0] == "None"  # sentinel first, works w/o runtime
        assert weight[0] == "FLOAT"
        assert weight[1]["min"] == -3.0
        assert weight[1]["max"] == 3.0


# ======================================================================
# T2IBundlePackNode / validate_bundle
# ======================================================================


def test_bundle_pack_roundtrip() -> None:
    """pack() produces a dict carrying every bundle key."""
    node = T2IBundlePackNode()
    model, clip, vae = object(), object(), object()
    pos, neg, latent = object(), object(), object()
    (bundle,) = node.pack(
        model=model,
        clip=clip,
        vae=vae,
        pos_cond=pos,
        neg_cond=neg,
        latent=latent,
    )
    assert bundle["model"] is model
    assert bundle["clip"] is clip
    assert bundle["vae"] is vae
    assert bundle["pos_cond"] is pos
    assert bundle["neg_cond"] is neg
    assert bundle["latent"] is latent
    assert bundle["img"] is None


def test_bundle_pack_with_image() -> None:
    """An optional img lands in the bundle as-is."""
    node = T2IBundlePackNode()
    img = object()
    (bundle,) = node.pack(
        model=1,
        clip=2,
        vae=3,
        pos_cond=4,
        neg_cond=5,
        latent=6,
        img=img,
    )
    assert bundle["img"] is img


def test_validate_bundle_rejects_non_dict() -> None:
    """Non-dict bundles raise with a helpful message."""
    with pytest.raises(RuntimeError, match="LUKUTAR_BUNDLE"):
        validate_bundle([1, 2, 3])


def test_validate_bundle_rejects_missing_keys() -> None:
    """Bundles missing keys name the missing keys."""
    with pytest.raises(RuntimeError, match="vae"):
        validate_bundle({"model": 1, "clip": 2})


def test_validate_bundle_accepts_complete() -> None:
    """A complete bundle passes validation (img=None is fine)."""
    bundle = {key: None for key in BUNDLE_KEYS}
    validate_bundle(bundle)


# ======================================================================
# T2ILoaderNode (outside-runtime behavior)
# ======================================================================


def test_loader_node_raises_outside_runtime() -> None:
    """load() refuses to run without the ComfyUI runtime."""
    node = T2ILoaderNode()
    with pytest.raises(RuntimeError, match="ComfyUI runtime"):
        node.load(
            ckpt_name="m.safetensors",
            clip_skip=-1,
            vae="baked",
            positive="",
            negative="",
            width=512,
            height=512,
            token_normalization="none",
            weight_interpretation="comfy",
            batch_size=1,
            lora_list=None,
        )


def test_loader_node_has_lora_cache_slot() -> None:
    """The per-instance LoRA cache starts empty."""
    node = T2ILoaderNode()
    assert node.loaded_lora is None
