'''
File:   test_metadata.py
Brief:  Unit tests for the metadata-block primitives (torch-free).
Author: Mistress-Lukutar
Date:   2026-11-02
Version: v0.1.0

Tests core.metadata functions: build_metadata, merge_metadata,
standard_metadata, sort_meta_inputs.
'''

from __future__ import annotations

import json
from unittest.mock import patch

from comfyui_lukutar_nodes.core.metadata import (
    build_metadata,
    merge_metadata,
    sort_meta_inputs,
    standard_metadata,
)

# ======================================================================
# build_metadata
# ======================================================================


def test_build_metadata_clamps_zero_to_one() -> None:
    """num_blocks=0 should clamp to 1 and process one pair."""
    result = build_metadata(0, ["a"], ["v"])
    assert "a" in result
    assert result["a"] == "v"


def test_build_metadata_clamps_above_max_to_16() -> None:
    """num_blocks=99 should clamp to MAX_METADATA_BLOCKS (16)."""
    names = [f"name_{i}" for i in range(20)]
    values = [f"val_{i}" for i in range(20)]
    result = build_metadata(99, names, values)
    assert len(result) == 16


def test_build_metadata_selects_first_n_pairs() -> None:
    """Only the first num_blocks pairs should be selected."""
    names = ["a", "b", "c"]
    values = ["v1", "v2", "v3"]
    result = build_metadata(2, names, values)
    assert list(result.keys()) == ["a", "b"]


def test_build_metadata_shortest_sequence_wins() -> None:
    """Iteration stops at the shortest of names/values/num_blocks."""
    names = ["a", "b", "c"]
    values = ["v1"]  # only one value
    result = build_metadata(3, names, values)
    assert len(result) == 1


def test_build_metadata_skips_empty_name() -> None:
    """Entries with empty (after strip) name are skipped."""
    names = ["a", "  ", "c"]
    values = ["v1", "v2", "v3"]
    result = build_metadata(3, names, values)
    assert list(result.keys()) == ["a", "c"]


def test_build_metadata_strips_whitespace_on_name_and_value() -> None:
    """Leading/trailing whitespace is stripped from both name and value."""
    names = ["  a  "]
    values = ["  v\n"]
    result = build_metadata(1, names, values)
    assert list(result.keys()) == ["a"]
    assert result["a"] == "v"


def test_build_metadata_coerces_non_str_via_str() -> None:
    """Non-string values are coerced via str()."""
    names = [123]  # non-str name
    values = [45.6]  # non-str value
    result = build_metadata(1, names, values)
    assert "123" in result
    assert result["123"] == "45.6"


def test_build_metadata_later_value_wins_on_duplicate_names() -> None:
    """Duplicate names get the later value."""
    names = ["key", "key"]
    values = ["first", "second"]
    result = build_metadata(2, names, values)
    assert result["key"] == "second"


def test_build_metadata_returns_fresh_dict() -> None:
    """Each call returns a new dict (not shared)."""
    r1 = build_metadata(1, ["a"], ["v"])
    r2 = build_metadata(1, ["b"], ["w"])
    r1["a"] = "modified"
    assert r2 == {"b": "w"}


# ======================================================================
# merge_metadata
# ======================================================================


def test_merge_skips_none_blocks() -> None:
    """None blocks are silently skipped."""
    result = merge_metadata(None, {"a": "1"}, None, {"b": "2"})
    assert result == {"a": "1", "b": "2"}


def test_merge_later_overrides_earlier_keys() -> None:
    """Later blocks override earlier ones for duplicate keys."""
    result = merge_metadata({"key": "old"}, {"key": "new"})
    assert result["key"] == "new"


def test_merge_does_not_mutate_inputs() -> None:
    """Input dicts are not modified by merge."""
    before = {"a": "1"}
    original_a = before["a"]
    merge_metadata(before, {"b": "2"})
    assert before == {"a": original_a}


def test_merge_coerces_non_str_values() -> None:
    """All values are coerced via str()."""
    result = merge_metadata({"num": 42, "flt": 3.14})
    assert result["num"] == "42"
    assert result["flt"] == "3.14"


# ======================================================================
# standard_metadata
# ======================================================================


def test_standard_prompt_only_when_include_and_given() -> None:
    """'prompt' key emitted only when include_prompt=True and prompt is not None."""
    result = standard_metadata(
        include_workflow=False,
        include_prompt=True,
        include_datetime=False,
        datetime_format="",
        prompt={"test": True},
        extra_pnginfo=None,
    )
    assert "prompt" in result

    result_no_prompt = standard_metadata(
        include_workflow=False,
        include_prompt=True,
        include_datetime=False,
        datetime_format="",
        prompt=None,
        extra_pnginfo=None,
    )
    assert "prompt" not in result_no_prompt


def test_standard_prompt_json_roundtrips() -> None:
    """The emitted 'prompt' value round-trips through json.loads."""
    original = {"node_a": {"inputs": {"val": 42}}, "node_b": {}}
    result = standard_metadata(
        include_workflow=False,
        include_prompt=True,
        include_datetime=False,
        datetime_format="",
        prompt=original,
        extra_pnginfo=None,
    )
    loaded = json.loads(result["prompt"])
    assert loaded == original


def test_standard_workflow_only_when_include_and_valid() -> None:
    """'workflow' emitted only when include_workflow=True and workflow is not None."""
    result_ok = standard_metadata(
        include_workflow=True,
        include_prompt=False,
        include_datetime=False,
        datetime_format="",
        prompt=None,
        extra_pnginfo={"workflow": {"nodes": []}},
    )
    assert "workflow" in result_ok

    result_none_wf = standard_metadata(
        include_workflow=True,
        include_prompt=False,
        include_datetime=False,
        datetime_format="",
        prompt=None,
        extra_pnginfo={"workflow": None},
    )
    assert "workflow" not in result_none_wf

    result_no_include = standard_metadata(
        include_workflow=False,
        include_prompt=False,
        include_datetime=False,
        datetime_format="",
        prompt=None,
        extra_pnginfo={"workflow": {"nodes": []}},
    )
    assert "workflow" not in result_no_include


def test_standard_key_order_prompt_workflow_datetime() -> None:
    """Keys are emitted in order: prompt, workflow, datetime."""
    result = standard_metadata(
        include_workflow=True,
        include_prompt=True,
        include_datetime=True,
        datetime_format="%Y-%m-%d",
        prompt={"x": 1},
        extra_pnginfo={"workflow": {"nodes": []}},
    )
    assert list(result.keys()) == ["prompt", "workflow", "datetime"]


def test_standard_datetime_fallback_on_blank_format() -> None:
    """Blank/whitespace datetime_format falls back to DEFAULT_DATETIME_FORMAT."""
    result = standard_metadata(
        include_workflow=False,
        include_prompt=False,
        include_datetime=True,
        datetime_format="   ",
        prompt=None,
        extra_pnginfo=None,
    )
    assert "datetime" in result
    # The fallback format %Y-%m-%d %H:%M:%S should produce a value like
    # "2026-11-02 12:34:56" — at least match the pattern.
    import re

    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result["datetime"])


def test_standard_custom_datetime_format_respected() -> None:
    """Custom strftime format is used verbatim."""
    with patch("comfyui_lukutar_nodes.core.metadata.datetime") as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "CUSTOM"
        result = standard_metadata(
            include_workflow=False,
            include_prompt=False,
            include_datetime=True,
            datetime_format="%Y%m%d",
            prompt=None,
            extra_pnginfo=None,
        )
    assert result["datetime"] == "CUSTOM"


# ======================================================================
# sort_meta_inputs
# ======================================================================


def test_sort_meta_inputs_natural_order() -> None:
    """meta_N names are sorted by their integer N."""
    result = sort_meta_inputs(["meta_10", "meta_2", "meta_1"])
    assert result == ["meta_1", "meta_2", "meta_10"]


def test_sort_meta_inputs_non_matching_after_all_meta() -> None:
    """Non-matching names sort after all meta_N alphabetically."""
    result = sort_meta_inputs(["extra", "meta_10", "alpha", "meta_2"])
    # meta_1 < meta_2 < alpha < extra (alphabetical among non-meta)
    assert result[0] == "meta_2"
    assert result[1] == "meta_10"
    assert result[2] == "alpha"
    assert result[3] == "extra"


def test_sort_meta_inputs_empty_list() -> None:
    """An empty list returns an empty list."""
    assert sort_meta_inputs([]) == []
