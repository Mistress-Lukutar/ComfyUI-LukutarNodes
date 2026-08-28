'''
File:   test_prompt_annotator.py
Brief:  Unit tests for the inline prompt annotation markup engine.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.7.0
'''

from __future__ import annotations

import pytest
from comfyui_lukutar_nodes.core.prompt_annotator import (
    DEFAULT_LABEL,
    AnnotatedPrompt,
    PromptAnnotateError,
    edit_segment,
    labels_text,
    parse_annotated_prompt,
    segment_text,
    to_impact_wildcard,
    to_markup,
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


def test_labels_text_variants() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    assert labels_text(annotated) == "body, face, hair, background"
    assert labels_text(annotated, include_common=True) == (
        "all, body, face, hair, background"
    )


def test_labels_text_without_regions() -> None:
    plain = parse_annotated_prompt("masterpiece, 1girl, outdoors")
    assert labels_text(plain) == ""
    assert labels_text(plain, include_common=True) == DEFAULT_LABEL
    assert labels_text(parse_annotated_prompt("")) == ""


def test_to_markup_round_trip() -> None:
    for markup in (
        _EXAMPLE,
        "masterpiece, 1girl, outdoors",
        "",
        "|face:x|, |body:y|",
        "a, |face:x|, shared, |body:y|, b",
        "|body,hair:red hair|",
    ):
        annotated = parse_annotated_prompt(markup)
        reparse = parse_annotated_prompt(to_markup(annotated))
        assert reparse.clean == annotated.clean, markup
        assert reparse.spans == annotated.spans, markup
        assert reparse.segments_by_label() == annotated.segments_by_label()


def test_edit_segment_prepend_and_append() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    prepended = edit_segment(annotated, "face", "prepend", "detailed eyes")
    assert segment_text(prepended, "face") == "detailed eyes, blue eyes, smirk"
    assert len(prepended.spans) == len(annotated.spans)
    # Multi-span label: prepend goes to the first span, append to the last.
    solo = edit_segment(annotated, "body", "prepend", "solo")
    assert segment_text(solo, "body") == "solo, 1girl, thin, red hair, stands"
    pose = edit_segment(annotated, "body", "append", "standing pose")
    assert segment_text(pose, "body") == (
        "1girl, thin, red hair, stands, standing pose"
    )


def test_edit_segment_remove() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    edited = edit_segment(annotated, "face", "remove", "smirk")
    assert segment_text(edited, "face") == "blue eyes"
    assert edited.clean == (
        "masterpiece, 1girl, thin, blue eyes, red hair, stands, "
        "outdoors, park"
    )


def test_edit_segment_remove_emptying_shared_span() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    # |body,hair:red hair| is shared: emptying it drops the span, so
    # body loses the text too and the hair label disappears.
    edited = edit_segment(annotated, "hair", "remove", "red hair")
    assert edited.labels == ("all", "body", "face", "background")
    assert segment_text(edited, "body") == "1girl, thin, stands"


def test_edit_segment_remove_no_match_is_noop() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    assert edit_segment(annotated, "face", "remove", "freckles") is annotated


def test_edit_segment_all_label_edits_common_part_in_place() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    prepended = edit_segment(annotated, DEFAULT_LABEL, "prepend", "best")
    assert segment_text(prepended, DEFAULT_LABEL) == "best, masterpiece"
    assert prepended.clean.startswith("best, masterpiece")
    assert len(prepended.spans) == len(annotated.spans)
    removed = edit_segment(prepended, DEFAULT_LABEL, "remove", "masterpiece")
    assert segment_text(removed, DEFAULT_LABEL) == "best"


def test_edit_segment_all_label_interior_plain() -> None:
    annotated = parse_annotated_prompt("a, |face:x|, shared, |body:y|, b")
    appended = edit_segment(annotated, DEFAULT_LABEL, "append", "common")
    assert segment_text(appended, DEFAULT_LABEL) == "a, shared, b, common"
    prepended = edit_segment(annotated, DEFAULT_LABEL, "prepend", "common")
    assert segment_text(prepended, DEFAULT_LABEL) == "common, a, shared, b"


def test_edit_segment_errors() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    with pytest.raises(ValueError, match="available labels"):
        edit_segment(annotated, "paws", "prepend", "fur")
    with pytest.raises(ValueError, match="Unknown edit mode"):
        edit_segment(annotated, "face", "replace", "x")
    with pytest.raises(ValueError, match="must not be empty"):
        edit_segment(annotated, "face", "append", "   ")


def test_edit_segment_new_mode() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    edited = edit_segment(
        annotated, "hands, weapon", "new", "delicate fingers"
    )
    assert edited.spans[-1].labels == ("hands", "weapon")
    assert edited.spans[-1].text == "delicate fingers"
    assert segment_text(edited, "hands") == "delicate fingers"
    assert segment_text(edited, "weapon") == "delicate fingers"
    assert edited.clean == _EXAMPLE_CLEAN + ", delicate fingers"


def test_edit_segment_new_mode_empty_annotation() -> None:
    edited = edit_segment(parse_annotated_prompt(""), "face", "new", "smirk")
    assert to_markup(edited) == "|face: smirk|"
    assert edited.clean == "smirk"


def test_edit_segment_new_mode_existing_label_raises() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    with pytest.raises(ValueError, match="already exist"):
        edit_segment(annotated, "face", "new", "freckles")
    with pytest.raises(ValueError, match="already exist"):
        edit_segment(annotated, "hands, face", "new", "freckles")


def test_edit_segment_delete_mode() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    edited = edit_segment(annotated, "face, background", "delete", "")
    assert edited.labels == ("all", "body", "hair")
    assert edited.clean == "masterpiece, 1girl, thin, red hair, stands"


def test_edit_segment_delete_keeps_shared_span_text() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    # |body,hair: red hair| loses only the hair label; body keeps the text.
    edited = edit_segment(annotated, "hair", "delete", "")
    assert edited.labels == ("all", "body", "face", "background")
    assert segment_text(edited, "body") == "1girl, thin, red hair, stands"


def test_edit_segment_delete_all_label() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    edited = edit_segment(annotated, DEFAULT_LABEL, "delete", "")
    assert DEFAULT_LABEL not in edited.segments_by_label()
    assert edited.clean == (
        "1girl, thin, blue eyes, smirk, red hair, stands, outdoors, park"
    )
    interior = parse_annotated_prompt("a, |face:x|, shared, |body:y|, b")
    assert edit_segment(interior, DEFAULT_LABEL, "delete", "").clean == "x, y"


def test_edit_segment_multiple_labels() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    appended = edit_segment(
        annotated, "face, background", "append", "detail"
    )
    assert segment_text(appended, "face") == "blue eyes, smirk, detail"
    assert segment_text(appended, "background") == "outdoors, park, detail"
    removed = edit_segment(
        annotated, "face, background", "remove", "smirk, park"
    )
    assert segment_text(removed, "face") == "blue eyes"
    assert segment_text(removed, "background") == "outdoors"


def test_edit_segment_label_list_validation() -> None:
    annotated = parse_annotated_prompt(_EXAMPLE)
    with pytest.raises(ValueError, match="Invalid label"):
        edit_segment(annotated, "my label", "new", "x")
    with pytest.raises(ValueError, match="must not be empty"):
        edit_segment(annotated, " , ", "append", "x")
    with pytest.raises(ValueError, match="available labels"):
        edit_segment(annotated, "paws, face", "prepend", "fur")


def test_raw_is_preserved() -> None:
    annotated: AnnotatedPrompt = parse_annotated_prompt(_EXAMPLE)
    assert annotated.raw == _EXAMPLE
