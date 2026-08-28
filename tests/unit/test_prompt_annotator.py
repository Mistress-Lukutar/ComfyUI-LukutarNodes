'''
File:   test_prompt_annotator.py
Brief:  Unit tests for the inline prompt annotation markup engine.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.4.0
'''

from __future__ import annotations

import pytest
from comfyui_lukutar_nodes.core.prompt_annotator import (
    DEFAULT_LABEL,
    AnnotatedPrompt,
    PromptAnnotateError,
    parse_annotated_prompt,
    segment_text,
    to_impact_wildcard,
)

_EXAMPLE = (
    "masterpiece, |body:1girl, thin|, |face:blue eyes, smirk|, "
    "|body,hair:red hair|, |body:stands|, |background:outdoors, park|"
)
_EXAMPLE_CLEAN = (
    "masterpiece, 1girl, thin, blue eyes, smirk, red hair, stands, "
    "outdoors, park"
)


def test_example_clean_text_and_span_offsets() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    assert annotated.clean == _EXAMPLE_CLEAN
    assert len(annotated.spans) == 5
    for span in annotated.spans:
        assert annotated.clean[span.start : span.end] == span.text


def test_labels_and_segments() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    assert annotated.labels == ("all", "body", "face", "hair", "background")
    segments = annotated.segments_by_label()
    assert segments["body"] == "1girl, thin, red hair, stands"
    assert segments["face"] == "blue eyes, smirk"
    assert segments["hair"] == "red hair"
    assert segments["background"] == "outdoors, park"
    assert segments[DEFAULT_LABEL] == "masterpiece"


def test_plain_prompt_is_all_default() -> None:
    annotated = parse_annotated_prompt("masterpiece, 1girl, outdoors")
    assert annotated.clean == "masterpiece, 1girl, outdoors"
    assert annotated.spans == ()
    assert annotated.default_text == "masterpiece, 1girl, outdoors"
    assert annotated.labels == (DEFAULT_LABEL,)


def test_empty_text() -> None:
    annotated = parse_annotated_prompt("")
    assert annotated.clean == ""
    assert annotated.spans == ()
    assert annotated.labels == ()
    assert to_impact_wildcard(annotated) == "[LAB]"


def test_label_normalization_and_dedup() -> None:
    annotated = parse_annotated_prompt("| face , body ,face:smirk|")
    assert annotated.spans[0].labels == ("face", "body")


def test_empty_tag_dropped_with_separator_cleanup() -> None:
    annotated = parse_annotated_prompt("a, |face:|, b")
    assert annotated.clean == "a, b"
    assert annotated.spans == ()
    trailing = parse_annotated_prompt("a, |face:   |")
    assert trailing.clean == "a"
    leading = parse_annotated_prompt("|face:|, b")
    assert leading.clean == "b"


def test_whitespace_only_inner_is_empty_tag() -> None:
    annotated = parse_annotated_prompt("a |face:  | b")
    assert annotated.clean == "a b"


def test_unclosed_tag_raises_with_position() -> None:
    with pytest.raises(PromptAnnotateError, match="position 3"):
        parse_annotated_prompt("a, |face:blue eyes")


def test_stray_pipe_raises() -> None:
    with pytest.raises(PromptAnnotateError):
        parse_annotated_prompt("blue | eyes")


def test_missing_label_separator_raises() -> None:
    with pytest.raises(PromptAnnotateError, match="no ':'"):
        parse_annotated_prompt("|face blue eyes|")


def test_empty_label_raises() -> None:
    with pytest.raises(PromptAnnotateError, match="Empty label"):
        parse_annotated_prompt("|:smirk|")
    with pytest.raises(PromptAnnotateError, match="Empty label"):
        parse_annotated_prompt("|face, :smirk|")


def test_invalid_label_characters_raise() -> None:
    with pytest.raises(PromptAnnotateError, match="Invalid label"):
        parse_annotated_prompt("|my label:smirk|")
    with pytest.raises(PromptAnnotateError, match="Invalid label"):
        parse_annotated_prompt("|fa-ce:smirk|")


def test_impact_wildcard_format() -> None:
    wildcard = to_impact_wildcard(parse_annotated_prompt(_EXAMPLE))
    assert wildcard == "\n".join(
        [
            "[LAB]",
            "[ALL] masterpiece,",
            "[body] 1girl, thin, red hair, stands",
            "[face] blue eyes, smirk",
            "[hair] red hair",
            "[background] outdoors, park",
        ]
    )


def test_impact_wildcard_without_default_text() -> None:
    wildcard = to_impact_wildcard(parse_annotated_prompt("|face:smirk|"))
    assert wildcard == "[LAB]\n[face] smirk"


def test_impact_wildcard_starts_with_lab_header() -> None:
    # Impact Pack matches the header with a plain startswith, so the
    # output must not carry leading whitespace.
    wildcard = to_impact_wildcard(parse_annotated_prompt("x, |face:y|"))
    assert wildcard.startswith("[LAB]")


def test_explicit_all_tag_merges_with_unmarked_text() -> None:
    annotated = parse_annotated_prompt(
        "masterpiece, |all:detailed|, |face:smirk|"
    )
    segments = annotated.segments_by_label()
    assert segments["all"] == "masterpiece, detailed"


def test_segment_text_variants() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    assert segment_text(annotated, "face") == "blue eyes, smirk"
    assert segment_text(annotated, "face", include_common=True) == (
        "masterpiece, blue eyes, smirk"
    )
    assert segment_text(annotated, DEFAULT_LABEL) == "masterpiece"


def test_segment_text_unknown_label_raises() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    with pytest.raises(ValueError, match="available labels"):
        segment_text(annotated, "paws")


def test_raw_is_preserved() -> None:
    annotated: AnnotatedPrompt = parse_annotated_prompt(_EXAMPLE)
    assert annotated.raw == _EXAMPLE
