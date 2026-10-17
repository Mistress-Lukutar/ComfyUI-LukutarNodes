'''
File:   test_group_switch.py
Brief:  Unit tests for the Group Switch engine + node (torch-free).
Author: Mistress-Lukutar
Date:   2026-10-15
Version: v1.0.0
'''

from __future__ import annotations

import pytest
from comfyui_lukutar_nodes.core.group_switch import (
    GroupSwitchError,
    check_disable_method,
    order_states,
    parse_group_states,
    parse_sort_order,
    sort_group_titles,
)
from comfyui_lukutar_nodes.nodes.group_switch import GroupSwitchNode


class TestParseGroupStates:
    def test_blank_means_empty(self) -> None:
        assert parse_group_states("") == {}
        assert parse_group_states("   ") == {}

    def test_order_is_preserved(self) -> None:
        states = parse_group_states('{"b": true, "a": false, "c": true}')
        assert list(states) == ["b", "a", "c"]
        assert states == {"b": True, "a": False, "c": True}

    @pytest.mark.parametrize(
        "raw",
        ["{not json", "[1, 2]", '"text"', "42", "null", "true"],
    )
    def test_non_object_payloads_are_rejected(self, raw: str) -> None:
        with pytest.raises(GroupSwitchError, match="JSON"):
            parse_group_states(raw)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("true", True),
            ("false", False),
            (" TRUE ", True),
            ("yes", True),
            ("on", True),
            ("1", True),
            ("0", False),
            ("off", False),
            ("1.5", False),
            (1, True),
            (0, False),
            (2, True),
            (0.0, False),
            (None, False),
        ],
    )
    def test_values_are_coerced_to_bool(
        self, value: object, expected: bool
    ) -> None:
        import json

        assert parse_group_states(json.dumps({"g": value})) == {"g": expected}


class TestParseSortOrder:
    def test_tokens_are_trimmed_and_lowered(self) -> None:
        assert parse_sort_order(" B, c ,DG ,de ") == ["b", "c", "dg", "de"]

    def test_blanks_are_dropped(self) -> None:
        assert parse_sort_order("b,, ,c,") == ["b", "c"]

    def test_empty(self) -> None:
        assert parse_sort_order("") == []
        assert parse_sort_order(" , ") == []


class TestSortGroupTitles:
    def test_no_tokens_keeps_input_order(self) -> None:
        titles = ["z", "a", "m"]
        assert sort_group_titles(titles, "") == titles
        assert sort_group_titles(titles, []) == titles

    def test_tokens_define_order(self) -> None:
        titles = ["dg", "b", "de", "c"]
        assert sort_group_titles(titles, "b,c,dg,de") == [
            "b",
            "c",
            "dg",
            "de",
        ]

    def test_case_insensitive_matching(self) -> None:
        titles = ["Beta", "alpha"]
        assert sort_group_titles(titles, "ALPHA, beta") == ["alpha", "Beta"]

    def test_unlisted_titles_keep_relative_order_at_end(self) -> None:
        titles = ["x", "b", "y", "a", "z"]
        assert sort_group_titles(titles, "a,b") == ["a", "b", "x", "y", "z"]

    def test_unknown_tokens_are_ignored(self) -> None:
        titles = ["b", "a"]
        assert sort_group_titles(titles, "zz,a,yy,b") == ["a", "b"]

    def test_prefix_tokens_match_titles(self) -> None:
        titles = ["Image Upscale Iterative", "Inpaint Fix", "Detection"]
        assert sort_group_titles(titles, "in,de") == [
            "Inpaint Fix",
            "Detection",
            "Image Upscale Iterative",
        ]

    def test_first_matching_token_wins(self) -> None:
        # "det" and "de" both prefix both titles; scanning tokens in
        # order ranks both at "det", keeping their relative order.
        titles = ["Detailer", "Detection"]
        assert sort_group_titles(titles, "det,de") == [
            "Detailer",
            "Detection",
        ]

    def test_whole_title_still_matches(self) -> None:
        # A full title is itself a prefix, so exact tokens keep working.
        titles = ["b", "beta"]
        assert sort_group_titles(titles, "be,b") == ["beta", "b"]

    def test_input_list_is_not_mutated(self) -> None:
        titles = ["b", "a"]
        sort_group_titles(titles, "a,b")
        assert titles == ["b", "a"]


class TestOrderStates:
    def test_states_follow_sorted_titles(self) -> None:
        states = {"dg": True, "b": False, "x": True, "a": True}
        ordered = order_states(states, "a,b")
        assert list(ordered) == ["a", "b", "dg", "x"]
        assert ordered == states

    def test_states_follow_prefix_tokens(self) -> None:
        states = {"Inpaint Fix": True, "T2I": False}
        ordered = order_states(states, "t2,in")
        assert list(ordered) == ["T2I", "Inpaint Fix"]

    def test_no_tokens_preserves_json_order(self) -> None:
        states = parse_group_states('{"b": true, "a": false}')
        assert list(order_states(states, "")) == ["b", "a"]


class TestCheckDisableMethod:
    def test_valid_methods_pass_through(self) -> None:
        assert check_disable_method("bypass") == "bypass"
        assert check_disable_method("disable") == "disable"

    @pytest.mark.parametrize("method", ["", "mute", "BYPASS", "off"])
    def test_unknown_methods_raise(self, method: str) -> None:
        with pytest.raises(GroupSwitchError, match="disable_method"):
            check_disable_method(method)


class TestGroupSwitchNode:
    def test_switch_publishes_ordered_states(self) -> None:
        (states,) = GroupSwitchNode().switch(
            "bypass",
            True,
            "b,c",
            '{"b": true, "a": false, "c": true}',
        )
        assert list(states) == ["b", "c", "a"]
        assert states == {"b": True, "c": True, "a": False}

    def test_switch_with_empty_graph(self) -> None:
        (states,) = GroupSwitchNode().switch("disable", False, "", "{}")
        assert states == {}

    def test_bad_method_raises(self) -> None:
        with pytest.raises(GroupSwitchError, match="disable_method"):
            GroupSwitchNode().switch("mute", True, "", "{}")

    def test_bad_payload_raises(self) -> None:
        with pytest.raises(GroupSwitchError, match="JSON"):
            GroupSwitchNode().switch("bypass", True, "", "{broken")

    def test_input_types_contract(self) -> None:
        required = GroupSwitchNode.INPUT_TYPES()["required"]
        assert set(required) == {
            "disable_method",
            "change_group_color",
            "sort_order",
            "group_states",
        }
        methods = required["disable_method"][0]
        assert "bypass" in methods and "disable" in methods
        assert GroupSwitchNode.RETURN_TYPES == ("GROUP_STATES",)
