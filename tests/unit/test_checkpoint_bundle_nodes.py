'''
File:   test_checkpoint_bundle_nodes.py
Brief:  Unit tests for the torch-free checkpoint bundle node parts.
Author: Mistress-Lukutar
Date:   2026-11-03
Version: v0.1.0

nodes/checkpoint_bundle.py imports torch only through a guarded try, so the LoRA
list / bundle pack logic and the bundle validation run on any python.
'''

from __future__ import annotations

import pytest
from comfyui_lukutar_nodes.nodes.checkpoint_bundle import (
    BUNDLE_KEYS,
    BUNDLE_TYPE,
    INFO_TYPE,
    MAX_LORA_SLOTS,
    BundleKSamplerNode,
    CheckpointBundleLoaderNode,
    CheckpointBundlePackNode,
    LoraListNode,
    _interpret_key,
    validate_bundle,
)

# ======================================================================
# LoraListNode
# ======================================================================


def test_lora_list_node_claims_slots() -> None:
    """Claimed slots appear as (name, weight) pairs in order."""
    node = LoraListNode()
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
    node = LoraListNode()
    kwargs: dict[str, object] = {
        "lora_1": "a.safetensors",
        "weight_1": 0.0,
    }
    (result,) = node.build(**kwargs)
    assert result == ()


def test_lora_list_node_empty() -> None:
    """All-None slots produce an empty tuple."""
    node = LoraListNode()
    (result,) = node.build()
    assert result == ()


def test_lora_list_node_input_types_pairing() -> None:
    """Every lora_N combo is paired with a weight_N float."""
    inputs = LoraListNode.INPUT_TYPES()["required"]
    for i in range(1, MAX_LORA_SLOTS + 1):
        combo = inputs[f"lora_{i}"]
        weight = inputs[f"weight_{i}"]
        assert combo[0][0] == "None"  # sentinel first, works w/o runtime
        assert weight[0] == "FLOAT"
        assert weight[1]["min"] == -3.0
        assert weight[1]["max"] == 3.0


# ======================================================================
# CheckpointBundlePackNode / validate_bundle
# ======================================================================


def test_bundle_pack_roundtrip() -> None:
    """pack() produces a dict carrying every bundle key."""
    node = CheckpointBundlePackNode()
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
    node = CheckpointBundlePackNode()
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
# CheckpointBundleLoaderNode (outside-runtime behavior)
# ======================================================================


def test_loader_node_raises_outside_runtime() -> None:
    """load() refuses to run without the ComfyUI runtime."""
    node = CheckpointBundleLoaderNode()
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
    node = CheckpointBundleLoaderNode()
    assert node.loaded_lora is None


# ======================================================================
# BundleKSamplerNode (outside-runtime behavior)
# ======================================================================


def test_sampler_node_is_output_node() -> None:
    """The sampler must be an OUTPUT_NODE so previews render."""
    assert BundleKSamplerNode.OUTPUT_NODE is True


def test_sampler_node_io_types() -> None:
    """The sampler consumes and re-emits bundle + info."""
    assert BundleKSamplerNode.RETURN_TYPES == (BUNDLE_TYPE, INFO_TYPE)
    assert BundleKSamplerNode.RETURN_NAMES == ("bundle", "info")


def test_sampler_node_input_widgets() -> None:
    """Required widget set covers seed..vae_decode."""
    inputs = BundleKSamplerNode.INPUT_TYPES()["required"]
    for name in (
        "bundle",
        "info",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "denoise",
        "vae_decode",
    ):
        assert name in inputs
    assert inputs["vae_decode"][0] == "BOOLEAN"
    assert inputs["vae_decode"][1]["default"] is True


def test_sampler_node_raises_outside_runtime() -> None:
    """sample() refuses to run without the ComfyUI runtime."""
    node = BundleKSamplerNode()
    with pytest.raises(RuntimeError, match="ComfyUI runtime"):
        node.sample(
            bundle={key: None for key in BUNDLE_KEYS},
            info={},
            seed=0,
            steps=1,
            cfg=7.0,
            sampler_name="euler",
            scheduler="normal",
            denoise=1.0,
            vae_decode=True,
        )


# ======================================================================
# _interpret_key token structure (regression: double-wrapped pairs)
# ======================================================================


def _fake_rows() -> list[list[tuple[int, float, int]]]:
    """Mimic CLIP.tokenize(return_word_ids=True) output for one line."""
    # BOS, weighted word ("cat" spread over 2 tokens), EOS, pads.
    return [
        [
            (49406, 1.0, 0),
            (2368, 1.3, 1),
            (320, 1.3, 1),
            (49407, 1.0, 0),
        ]
        + [(0, 1.0, 0)] * 4
    ]


def test_interpret_key_comfy_passes_flat_pairs() -> None:
    """The comfy branch must pass flat (token, weight) pairs onward.

    Regression: zipping the (token, weight) pair rows with the weight
    matrix again produced ((token, weight), weight) entries, which
    ComfyUI's encode_token_weights forwards unstripped and
    process_tokens crashes on ('tuple' has no attribute 'get').
    """
    captured: list = []

    def fake_encode(lines):
        captured.append(lines)
        return object(), object()

    _interpret_key(
        fake_encode, _fake_rows(), "none", "comfy", 1.0
    )
    (line,) = captured[-1]
    assert line[0] == (49406, 1.0)
    assert line[1] == (2368, 1.3)
    assert all(isinstance(t, int) and isinstance(w, float) for t, w in line)


def test_interpret_key_comfy_applies_normalization() -> None:
    """'mean' normalization shifts the weights before encoding."""
    captured: list = []

    def fake_encode(lines):
        captured.append(lines)
        return object(), object()

    _interpret_key(fake_encode, _fake_rows(), "mean", "comfy", 1.0)
    (line,) = captured[-1]
    pairs = zip(line, _fake_rows()[0], strict=True)
    real = [w for (t, w), (rt, _, rid) in pairs if rid != 0]
    assert sum(real) / len(real) == pytest.approx(1.0)


def test_interpret_key_a1111_encodes_unweighted() -> None:
    """The A1111 branch encodes the unweighted base (all weights 1.0)."""
    captured: list = []

    def fake_encode(lines):
        captured.append(lines)
        # Base embedding path needs torch; only assert the call shape.
        raise RuntimeError("stop before tensor math")

    with pytest.raises(RuntimeError, match="stop before tensor math"):
        _interpret_key(fake_encode, _fake_rows(), "none", "A1111", 1.0)
    (line,) = captured[-1]
    assert line[1] == (2368, 1.0)  # weight neutralized in the base encode


def test_push_final_preview_noop_outside_runtime() -> None:
    """The final-preview push is a silent no-op without the runtime."""
    from comfyui_lukutar_nodes.nodes.checkpoint_bundle import _push_final_preview

    class FakePbar:
        total = 10
        calls: list = []

        def update_absolute(self, value, total=None, preview=None):
            self.calls.append((value, total, preview))

    fake = FakePbar()
    _push_final_preview(fake, None)  # Image/np are None outside runtime
    assert fake.calls == []
    _push_final_preview(None, None)
