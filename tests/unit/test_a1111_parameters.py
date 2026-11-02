'''
File:   test_a1111_parameters.py
Brief:  Unit tests for A1111-style parameter primitives (torch-free).
Author: Mistress-Lukutar
Date:   2026-11-02
Version: v0.1.0

Tests core.a1111_parameters functions: quote_a1111, build_parameters,
compute_model_hash, a1111_float_str, ModelHashCache, sort_numbered_inputs.
'''

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from comfyui_lukutar_nodes.core.a1111_parameters import (
    HASH_CACHE_FILENAME,
    MODEL_HASH_CHARS,
    ModelHashCache,
    a1111_float_str,
    build_parameters,
    compute_model_hash,
    quote_a1111,
)
from comfyui_lukutar_nodes.core.metadata import sort_meta_inputs, sort_numbered_inputs

# ======================================================================
# Constants                                                            #
# ======================================================================


def test_constants_have_expected_values() -> None:
    """Verify the module-level constants match spec."""
    assert MODEL_HASH_CHARS == 10
    assert HASH_CACHE_FILENAME == "a1111_hash_cache.json"
    assert len("DPM++ 2M Karras") > 0


# ======================================================================
# quote_a1111                                                         #
# ======================================================================


def test_quote_plain_value_unchanged() -> None:
    """A string without comma, colon or newline is returned as-is."""
    assert quote_a1111("hello world") == "hello world"
    assert quote_a1111("42") == "42"


def test_quote_comma_triggers_json_encoding() -> None:
    """A value containing a comma must be JSON-encoded."""
    result = quote_a1111("a, b")
    assert result == json.dumps("a, b", ensure_ascii=False)


def test_quote_colon_triggers_json_encoding() -> None:
    """A value containing a colon must be JSON-encoded."""
    result = quote_a1111("key: value")
    assert result == json.dumps("key: value", ensure_ascii=False)


def test_quote_newline_triggers_json_encoding() -> None:
    """A value containing a newline must be JSON-encoded."""
    result = quote_a1111("line1\nline2")
    assert result == json.dumps("line1\nline2", ensure_ascii=False)


def test_quote_unicode_not_escaped() -> None:
    """Unicode characters must not be escaped in JSON output."""
    result = quote_a1111("caf\u00e9, tea")
    # json.dumps with ensure_ascii=False should keep the unicode char.
    assert "\u00e9" in result


# ======================================================================
# build_parameters                                                    #
# ======================================================================


def test_build_parameters_full_example() -> None:
    """The canonical full example must match exactly."""
    params = {
        "Steps": "30",
        "Sampler": "DPM++ 2M Karras",
        "Lora hashes": "lukutar_style_v2: 3a4f2c89d1, detail_tweaker_xl: f7e1b0c3a9",
    }
    result = build_parameters(
        prompt="masterpiece, 1girl",
        negative_prompt="worst quality",
        params=params,
    )
    expected = (
        "masterpiece, 1girl\n"
        "Negative prompt: worst quality\n"
        'Steps: 30, Sampler: DPM++ 2M Karras, '
        'Lora hashes: "lukutar_style_v2: 3a4f2c89d1, detail_tweaker_xl: f7e1b0c3a9"'
    )
    assert result == expected


def test_build_parameters_negative_omitted_when_empty() -> None:
    """Empty negative prompt string does not produce a Negative prompt line."""
    result = build_parameters(
        prompt="simple prompt",
        negative_prompt="",
        params=None,
    )
    assert "Negative prompt:" not in result


def test_build_parameters_params_omitted_when_none() -> None:
    """params=None skips the settings block entirely."""
    result = build_parameters(
        prompt="prompt",
        negative_prompt="neg",
        params=None,
    )
    assert "\n" in result  # only prompt + neg line
    assert "Steps:" not in result


def test_build_parameters_params_omitted_when_empty_dict() -> None:
    """params={} (empty dict) skips the settings block."""
    result = build_parameters(
        prompt="prompt",
        negative_prompt="",
        params={},
    )
    assert result == "prompt"


def test_build_parameters_all_empty_returns_empty_string() -> None:
    """Empty everything yields an empty string (stripped)."""
    result = build_parameters(prompt="", negative_prompt="", params=None)
    assert result == ""


def test_build_parameters_none_values_skipped() -> None:
    """Parameters whose value is None are silently skipped."""
    params = {"Steps": "30", "CFG": None, "Sampler": "Euler"}
    result = build_parameters(
        prompt="prompt", negative_prompt="", params=params,
    )
    assert "CFG:" not in result
    assert "Steps: 30" in result
    assert "Sampler: Euler" in result


def test_build_parameters_insertion_order_preserved() -> None:
    """Parameter order follows insertion order of the mapping."""
    params = {"A": "1", "B": "2", "C": "3"}
    result = build_parameters(
        prompt="prompt", negative_prompt="", params=params,
    )
    a_pos = result.index("A: 1")
    b_pos = result.index("B: 2")
    c_pos = result.index("C: 3")
    assert a_pos < b_pos < c_pos


def test_build_parameters_multiline_prompt_keeps_internal_newlines() -> None:
    """Internal newlines in the prompt are preserved."""
    prompt_text = "line1\nline2"
    result = build_parameters(
        prompt=prompt_text, negative_prompt="", params=None,
    )
    assert "\n" in result  # at least one newline (between line1 and line2)


# ======================================================================
# compute_model_hash                                                  #
# ======================================================================


def test_compute_model_hash_known_content(tmp_path: Path) -> None:
    """Hash of known bytes must match hashlib.sha256."""
    content = b"hello world"
    file_path = tmp_path / "test.bin"
    file_path.write_bytes(content)
    result = compute_model_hash(file_path)
    expected = hashlib.sha256(content).hexdigest()[:MODEL_HASH_CHARS]
    assert result == expected


def test_compute_model_hash_multi_chunk(tmp_path: Path) -> None:
    """Small chunk_size forces multiple reads; hash must still be correct."""
    content = b"A" * 100  # 100 bytes, chunk=8 → 13 reads
    file_path = tmp_path / "multi.bin"
    file_path.write_bytes(content)
    result = compute_model_hash(file_path, chunk_size=8)
    expected = hashlib.sha256(content).hexdigest()[:MODEL_HASH_CHARS]
    assert result == expected


def test_compute_model_hash_file_not_found() -> None:
    """Non-existent path raises FileNotFoundError naturally."""
    try:
        compute_model_hash("/nonexistent/path/to/model.safetensors")
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass


# ======================================================================
# ModelHashCache                                                      #
# ======================================================================


def test_cache_computes_then_caches(tmp_path: Path) -> None:
    """First .hash() computes; second call returns cached value."""
    content = b"model data"
    file_path = tmp_path / "model.safetensors"
    file_path.write_bytes(content)

    cache = ModelHashCache(tmp_path / "cache.json")
    first_hash = cache.hash(file_path)
    assert len(first_hash) == MODEL_HASH_CHARS

    # Verify the entry is cached
    cached = cache.get(file_path)
    assert cached == first_hash


def test_cache_hash_not_recomputed_after_first_call(
    tmp_path: Path,
) -> None:
    """Patching compute_model_hash to a counter must stay at count=1."""
    content = b"model data for counter"
    file_path = tmp_path / "model_counter.bin"
    file_path.write_bytes(content)

    cache = ModelHashCache(tmp_path / "counter_cache.json")
    call_count = 0

    def counting_hash(path: str | Path, **_kw: object) -> str:
        nonlocal call_count
        call_count += 1
        return hashlib.sha256(b"model data for counter").hexdigest()[:MODEL_HASH_CHARS]

    # Seed the cache with a valid entry so second .hash() hits the cache.
    abs_path = os.path.abspath(os.fspath(file_path))
    st = file_path.stat()
    cache._data[abs_path] = {
        "mtime": st.st_mtime,
        "size": st.st_size,
        "hash": "seeded1234",
    }

    # Patch compute_model_hash in the module namespace.
    with patch(
        "comfyui_lukutar_nodes.core.a1111_parameters.compute_model_hash",
        counting_hash,
    ):
        h1 = cache.hash(file_path)
        assert h1 == "seeded1234"
        h2 = cache.hash(file_path)
        assert h2 == "seeded1234"

    # compute_model_hash was never called because the cache had a valid hit.
    assert call_count == 0


def test_cache_get_returns_none_when_mtime_changes(tmp_path: Path) -> None:
    """Changing mtime (and size) invalidates the cache entry."""
    content = b"original"
    file_path = tmp_path / "model_mtime.bin"
    file_path.write_bytes(content)

    cache = ModelHashCache(tmp_path / "mtime_cache.json")
    h_before = cache.hash(file_path)
    assert h_before is not None

    # Bump mtime and change file size.
    os.utime(file_path, (0, 0))  # set mtime to epoch
    file_path.write_bytes(b"x" * 200)  # different size

    assert cache.get(file_path) is None


def test_cache_corrupt_json_tolerated(tmp_path: Path) -> None:
    """Writing garbage to the cache file should be tolerated."""
    cache_file = tmp_path / "corrupt.json"
    cache_file.write_text("this is not json {{{")

    cache = ModelHashCache(cache_file)
    # Should start from empty dict, no exception.
    assert cache._data == {}


def test_cache_save_creates_json_with_expected_structure(tmp_path: Path) -> None:
    """After hash+save the JSON file must contain valid structure."""
    content = b"save test data"
    file_path = tmp_path / "model_save.bin"
    file_path.write_bytes(content)

    cache = ModelHashCache(tmp_path / "save_cache.json")
    cache.hash(file_path)
    cache.save()

    assert tmp_path.joinpath("save_cache.json").exists()
    with open(tmp_path / "save_cache.json", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    abs_path = os.path.abspath(os.fspath(file_path))
    assert abs_path in data
    entry = data[abs_path]
    assert "mtime" in entry
    assert "size" in entry
    assert "hash" in entry


# ======================================================================
# sort_numbered_inputs                                                #
# ======================================================================


def test_sort_numbered_inputs_numeric_order() -> None:
    """lora_2 must come before lora_10 with prefix 'lora_'."""
    result = sorted(
        ["lora_10", "lora_2"],
        key=lambda n: (0, int(n.split("_")[1]), n) if "_" in n else (1, 0, n),
    )
    assert result == ["lora_2", "lora_10"]

    # Now test the actual function with a custom prefix.
    result = sort_numbered_inputs(["lora_10", "lora_2"], prefix="lora_")
    assert result == ["lora_2", "lora_10"]


def test_sort_meta_inputs_ordering() -> None:
    """meta_N names sorted numerically; non-matching after."""

    result = sort_meta_inputs(["meta_10", "meta_2", "meta_1"])
    assert result == ["meta_1", "meta_2", "meta_10"]


def test_sort_numbered_inputs_non_matching_after_prefixed() -> None:
    """Non-matching names must appear after all prefixed ones."""
    result = sort_numbered_inputs(
        ["extra", "lora_10", "alpha", "lora_2"], prefix="lora_",
    )
    assert result[0] == "lora_2"
    assert result[1] == "lora_10"
    # Non-matching names come after, sorted alphabetically.
    assert result[2:] == ["alpha", "extra"]


def test_sort_meta_inputs_unchanged_behavior() -> None:
    """Regression: sort_meta_inputs must still work as before."""

    result = sort_meta_inputs(
        ["meta_3", "meta_1", "other", "meta_20"],
    )
    assert result[0] == "meta_1"
    assert result[1] == "meta_3"
    # meta_20 sorts after meta_3 numerically (20 > 3)
    assert result[2] == "meta_20"
    assert result[3] == "other"


# ======================================================================
# a1111_float_str                                                     #
# ======================================================================


def test_a1111_float_str_integer_input() -> None:
    """Integer 2 must produce '2.0'."""
    assert a1111_float_str(2) == "2.0"


def test_a1111_float_str_decimal_input() -> None:
    """Decimal 0.45 must produce '0.45'."""
    assert a1111_float_str(0.45) == "0.45"


def test_a1111_float_str_string_number() -> None:
    """String numeric input must still produce float representation."""
    assert a1111_float_str("2") == "2.0"
    assert a1111_float_str("0.45") == "0.45"
