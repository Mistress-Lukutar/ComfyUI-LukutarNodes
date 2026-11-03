'''
File:   test_t2i_info_nodes.py
Brief:  Unit tests for the T2I info converter and utility nodes.
Author: Mistress-Lukutar
Date:   2026-11-03
Version: v0.1.0

Tests T2IInfoToA1111Node, T2IInfoUnpackNode and T2IInfoPackNode
against the shared A1111 build_parameters engine.
'''

from __future__ import annotations

from comfyui_lukutar_nodes.core.a1111_parameters import build_parameters
from comfyui_lukutar_nodes.core.t2i_info import (
    NEGATIVE_KEY,
    PROMPT_KEY,
    SEED_KEY,
    STEPS_KEY,
    build_loader_info,
    with_sampling,
)
from comfyui_lukutar_nodes.nodes.t2i import (
    T2IInfoPackNode,
    T2IInfoToA1111Node,
    T2IInfoUnpackNode,
)

# ======================================================================
# T2IInfoToA1111Node
# ======================================================================


def make_info() -> dict[str, str]:
    """Build a loader+sampling info dict for tests."""
    info = build_loader_info(
        prompt="a cat",
        negative="blurry",
        model_name="model_a",
        model_hash="0123456789",
        vae_name="baked",
        clip_skip=-2,
        lora_entries=[("l1", 0.8)],
        lora_hashes=[("l1", "deadbeef00")],
        width=832,
        height=1216,
        batch_size=1,
        token_normalization="none",
        weight_interpretation="A1111",
    )
    return with_sampling(
        info,
        seed=42,
        steps=25,
        cfg=7.0,
        sampler="euler",
        scheduler="normal",
        denoise=1.0,
        vae_decode=True,
    )


def test_converter_matches_build_parameters() -> None:
    """The composed text equals build_parameters on the same inputs."""
    info = make_info()
    (metadata,) = T2IInfoToA1111Node().convert(info)

    expected = build_parameters(
        "a cat",
        "blurry",
        {k: v for k, v in info.items() if k not in (PROMPT_KEY, NEGATIVE_KEY)},
    )
    assert metadata == {"parameters": expected}


def test_converter_prompt_first_settings_after() -> None:
    """The text starts with the prompt, then settings in order."""
    (metadata,) = T2IInfoToA1111Node().convert(make_info())
    text = metadata["parameters"]
    lines = text.splitlines()
    assert lines[0] == "a cat"
    assert lines[1] == "Negative prompt: blurry"
    assert "Model: model_a" in text
    assert "Seed: 42" in text


def test_converter_empty_info_yields_empty() -> None:
    """An empty info produces an empty metadata dict."""
    (metadata,) = T2IInfoToA1111Node().convert({})
    assert metadata == {}


def test_converter_none_info_yields_empty() -> None:
    """A None info produces an empty metadata dict."""
    (metadata,) = T2IInfoToA1111Node().convert(None)
    assert metadata == {}


def test_converter_does_not_mutate_input() -> None:
    """The input info keeps its prompt keys."""
    info = make_info()
    T2IInfoToA1111Node().convert(info)
    assert info[PROMPT_KEY] == "a cat"
    assert info[NEGATIVE_KEY] == "blurry"


# ======================================================================
# T2IInfoUnpackNode
# ======================================================================


def test_unpack_reads_standard_entries() -> None:
    """Named outputs carry the matching info values."""
    outputs = T2IInfoUnpackNode().unpack(make_info())
    names = T2IInfoUnpackNode.RETURN_NAMES
    values = dict(zip(names, outputs, strict=True))
    assert values["model_name"] == "model_a"
    assert values["model_hash"] == "0123456789"
    assert values["lora"] == "l1:0.8"
    assert values["prompt"] == "a cat"
    assert values["negative"] == "blurry"
    assert values["size"] == "832 x 1216"
    assert values["seed"] == "42"
    assert values["vae_decode"] == "enabled"


def test_unpack_metadata_is_full_copy() -> None:
    """The metadata output equals the whole info mapping."""
    info = make_info()
    outputs = T2IInfoUnpackNode().unpack(info)
    assert outputs[-1] == info
    assert outputs[-1] is not info


def test_unpack_missing_entries_are_empty() -> None:
    """Missing entries yield empty strings; metadata stays a dict."""
    names = T2IInfoUnpackNode.RETURN_NAMES
    outputs = T2IInfoUnpackNode().unpack({SEED_KEY: "1"})
    values = dict(zip(names, outputs, strict=True))
    assert values["seed"] == "1"
    assert all(
        v == "" for name, v in values.items() if name not in ("seed", "metadata")
    )
    assert outputs[-1] == {SEED_KEY: "1"}


def test_unpack_none_info() -> None:
    """A None info yields empty strings and an empty dict."""
    outputs = T2IInfoUnpackNode().unpack(None)
    assert all(s == "" for s in outputs[:-1])
    assert outputs[-1] == {}


# ======================================================================
# T2IInfoPackNode
# ======================================================================


def test_pack_builds_from_pairs() -> None:
    """Claimed pairs land in the output in order."""
    (info,) = T2IInfoPackNode().pack(
        key_1="Model",
        value_1="m1",
        key_2="Steps",
        value_2="20",
    )
    assert info == {"Model": "m1", "Steps": "20"}


def test_pack_extends_base_and_overrides() -> None:
    """Later pairs win over the base info."""
    (info,) = T2IInfoPackNode().pack(
        info_in={SEED_KEY: "1", STEPS_KEY: "20"},
        key_1=SEED_KEY,
        value_1="99",
    )
    assert info == {SEED_KEY: "99", STEPS_KEY: "20"}


def test_pack_skips_empty_keys() -> None:
    """Pairs with empty keys are dropped."""
    (info,) = T2IInfoPackNode().pack(key_1="", value_1="x", key_2="A", value_2="b")
    assert info == {"A": "b"}


def test_pack_does_not_mutate_base() -> None:
    """The input info mapping stays untouched."""
    base = {SEED_KEY: "1"}
    T2IInfoPackNode().pack(info_in=base, key_1="A", value_1="b")
    assert base == {SEED_KEY: "1"}


def test_pack_empty() -> None:
    """No inputs yield an empty dict."""
    (info,) = T2IInfoPackNode().pack()
    assert info == {}
