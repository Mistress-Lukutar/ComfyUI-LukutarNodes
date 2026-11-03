'''
File:   test_generation_info.py
Brief:  Unit tests for the generation-info primitives (torch-free).
Author: Mistress-Lukutar
Date:   2026-11-03
Version: v0.1.0

Tests core.generation_info: formatters, build_loader_info, with_sampling,
get, settings_pairs.
'''

from __future__ import annotations

from comfyui_lukutar_nodes.core.generation_info import (
    BATCH_KEY,
    CFG_KEY,
    DENOISE_KEY,
    LORA_HASHES_KEY,
    LORA_KEY,
    MODEL_HASH_KEY,
    MODEL_KEY,
    NEGATIVE_KEY,
    PROMPT_KEY,
    SAMPLER_KEY,
    SCHEDULER_KEY,
    SEED_KEY,
    SIZE_KEY,
    STEPS_KEY,
    VAE_DECODE_KEY,
    VAE_KEY,
    a1111_weight_str,
    build_loader_info,
    format_lora_entries,
    format_lora_hashes,
    get,
    settings_pairs,
    size_str,
    with_sampling,
)

# ======================================================================
# a1111_weight_str
# ======================================================================


def test_weight_str_plain() -> None:
    """1.0 renders as '1.0'."""
    assert a1111_weight_str(1.0) == "1.0"


def test_weight_str_trims_trailing_zeros() -> None:
    """0.50 renders as '0.5'."""
    assert a1111_weight_str(0.5) == "0.5"


def test_weight_str_negative() -> None:
    """Negative weights keep the sign."""
    assert a1111_weight_str(-3.0) == "-3.0"


def test_weight_str_rounds_to_two_decimals() -> None:
    """More than two decimals round to two."""
    assert a1111_weight_str(1.234) == "1.23"


# ======================================================================
# format_lora_entries / format_lora_hashes
# ======================================================================


def test_format_lora_entries_basic() -> None:
    """Pairs render as 'name:weight' comma-joined."""
    assert (
        format_lora_entries([("a", 1.0), ("b", 0.75)]) == "a:1.0, b:0.75"
    )


def test_format_lora_entries_skips_zero_and_empty() -> None:
    """Zero weights and blank names are dropped."""
    result = format_lora_entries([("a", 0.0), ("", 1.0), ("b", 0.5)])
    assert result == "b:0.5"


def test_format_lora_entries_empty() -> None:
    """No entries -> empty string."""
    assert format_lora_entries([]) == ""


def test_format_lora_hashes_basic() -> None:
    """Pairs render in the A1111 'name: hash' style."""
    assert (
        format_lora_hashes([("a", "abc123"), ("b", "def456")])
        == "a: abc123, b: def456"
    )


def test_format_lora_hashes_skips_incomplete() -> None:
    """Entries missing a name or hash are dropped."""
    assert format_lora_hashes([("a", ""), ("", "x"), ("b", "y")]) == "b: y"


# ======================================================================
# size_str
# ======================================================================


def test_size_str() -> None:
    """Size renders as '{width} x {height}'."""
    assert size_str(832, 1216) == "832 x 1216"


# ======================================================================
# build_loader_info
# ======================================================================


def test_build_loader_info_full_order() -> None:
    """The info dict carries the A1111-compatible keys in order."""
    info = build_loader_info(
        prompt="pos",
        negative="neg",
        model_name="model_a",
        model_hash="0123456789",
        vae_name="baked",
        clip_skip=-2,
        lora_entries=[("l1", 0.8)],
        lora_hashes=[("l1", "deadbeef00")],
        width=832,
        height=1216,
        batch_size=2,
        token_normalization="none",
        weight_interpretation="A1111",
    )
    assert list(info.items()) == [
        (PROMPT_KEY, "pos"),
        (NEGATIVE_KEY, "neg"),
        (MODEL_KEY, "model_a"),
        (MODEL_HASH_KEY, "0123456789"),
        (VAE_KEY, "baked"),
        ("Clip skip", "-2"),
        (LORA_KEY, "l1:0.8"),
        (LORA_HASHES_KEY, "l1: deadbeef00"),
        (SIZE_KEY, "832 x 1216"),
        (BATCH_KEY, "2"),
        ("Token normalization", "none"),
        ("Weight interpretation", "A1111"),
    ]


def test_build_loader_info_no_loras() -> None:
    """Without LoRAs both LoRA entries are empty strings."""
    info = build_loader_info(
        prompt="",
        negative="",
        model_name="m",
        model_hash="",
        vae_name="baked",
        clip_skip=-1,
        lora_entries=[],
        lora_hashes=[],
        width=512,
        height=512,
        batch_size=1,
        token_normalization="none",
        weight_interpretation="comfy",
    )
    assert info[LORA_KEY] == ""
    assert info[LORA_HASHES_KEY] == ""


# ======================================================================
# with_sampling
# ======================================================================


def test_with_sampling_appends_keys_and_does_not_mutate() -> None:
    """Sampling keys are appended; the input dict stays untouched."""
    base = {PROMPT_KEY: "p", MODEL_KEY: "m"}
    result = with_sampling(
        base,
        seed=42,
        steps=25,
        cfg=7.0,
        sampler="euler",
        scheduler="normal",
        denoise=1.0,
        vae_decode=True,
    )
    assert list(result.keys()) == [
        PROMPT_KEY,
        MODEL_KEY,
        SEED_KEY,
        STEPS_KEY,
        CFG_KEY,
        SAMPLER_KEY,
        SCHEDULER_KEY,
        DENOISE_KEY,
        VAE_DECODE_KEY,
    ]
    assert base == {PROMPT_KEY: "p", MODEL_KEY: "m"}


def test_with_sampling_disabled_decode() -> None:
    """vae_decode=False records 'disabled'."""
    result = with_sampling(
        {}, seed=1, steps=1, cfg=1.0, sampler="euler",
        scheduler="normal", denoise=0.5, vae_decode=False,
    )
    assert result[VAE_DECODE_KEY] == "disabled"


# ======================================================================
# get / settings_pairs
# ======================================================================


def test_get_missing_key_returns_default() -> None:
    """Missing keys yield the default."""
    assert get({"a": "x"}, "b") == ""
    assert get({"a": "x"}, "b", "fallback") == "fallback"


def test_get_strips_and_stringifies() -> None:
    """Values are stripped and coerced to str."""
    assert get({"a": "  7  "}, "a") == "7"
    assert get({"a": 7}, "a") == "7"


def test_get_none_mapping() -> None:
    """A None mapping yields the default."""
    assert get(None, "a") == ""


def test_settings_pairs_excludes_prompts() -> None:
    """Prompt keys never leak into the A1111 settings block."""
    info = {
        PROMPT_KEY: "p",
        NEGATIVE_KEY: "n",
        MODEL_KEY: "m",
        STEPS_KEY: "20",
    }
    result = settings_pairs(info)
    assert result == {MODEL_KEY: "m", STEPS_KEY: "20"}
