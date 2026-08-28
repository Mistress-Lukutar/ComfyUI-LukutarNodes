'''
File:   prompt_annotator.py
Brief:  Inline label-markup parser for region-wise prompt annotation.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.4.0
'''

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Character that opens and closes a labelled span in the prompt text.
SPAN_DELIMITER = "|"
#: Character separating a span's label list from the span text.
LABEL_SEPARATOR = ":"
#: Character separating multiple labels inside one span tag.
LABELS_SEPARATOR = ","

#: Label implicitly assigned to unmarked prompt text.
DEFAULT_LABEL = "all"

#: Impact Pack wildcard key applied to every segment regardless of label.
IMPACT_ALL_KEY = "ALL"
#: Impact Pack wildcard header that switches the text into label mode.
IMPACT_LAB_HEADER = "[LAB]"

#: Allowed label characters; a subset of Impact Pack's label charset
#: (``[A-Za-z0-9_. ]``), minus the characters that would be ambiguous
#: inside the inline tag syntax.
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")

#: Characters treated as prompt separators when tidying up around
#: dropped empty tags and when extracting the unmarked (default) text.
_SEPARATORS = " \t\r\n,;"

#: Collapse comma/whitespace runs inside plain runs merged across a
#: dropped empty tag: comma runs to ", ", leftover whitespace to " ".
_COMMA_RUN = re.compile(r"(?:\s*,\s*)+")
_WHITESPACE_RUN = re.compile(r"\s{2,}")

#: Length of the snippet quoted in parse error messages.
_ERROR_SNIPPET = 24


class PromptAnnotateError(Exception):
    '''Raised when annotated prompt text cannot be parsed.'''


@dataclass(frozen=True)
class PromptSpan:
    '''One tagged region of the prompt.

    Attributes:
        labels: Labels carried by the span, deduplicated, in tag order.
        text: Span text without the tag markup.
        start: Offset of `text` within the cleaned prompt.
        end: Offset one past the end of `text` within the cleaned prompt.
    '''

    labels: tuple[str, ...]
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class AnnotatedPrompt:
    '''Parsed representation of one annotated prompt.

    Attributes:
        raw: Original text including the tag markup.
        clean: Prompt with all tag markup removed, ready for CLIP.
        spans: Tagged spans ordered by their position in `clean`.
    '''

    raw: str
    clean: str
    spans: tuple[PromptSpan, ...]

    @property
    def default_text(self) -> str:
        '''Unmarked prompt content (the implicit `all` label text).'''
        pieces: list[str] = []
        cursor = 0
        for span in self.spans:
            if span.start > cursor:
                pieces.append(self.clean[cursor : span.start])
            cursor = max(cursor, span.end)
        if cursor < len(self.clean):
            pieces.append(self.clean[cursor:])
        stripped = (piece.strip(_SEPARATORS) for piece in pieces)
        return ", ".join(piece for piece in stripped if piece)

    def segments_by_label(self) -> dict[str, str]:
        '''Map every label to its prompt text, spans joined in order.

        The implicit `all` label is present only when the prompt has
        non-empty unmarked content. Multi-label spans contribute to each
        of their labels.
        '''
        collected: dict[str, list[str]] = {}
        default = self.default_text
        if default:
            collected[DEFAULT_LABEL] = [default]
        for span in self.spans:
            for label in span.labels:
                collected.setdefault(label, []).append(span.text)
        return {label: ", ".join(parts) for label, parts in collected.items()}

    @property
    def labels(self) -> tuple[str, ...]:
        '''All labels with content, in first-appearance order.

        The implicit `all` label comes first when unmarked content
        exists.
        '''
        return tuple(self.segments_by_label())


@dataclass(frozen=True)
class _Plain:
    '''Untagged text run between (or around) span tags.'''

    text: str


@dataclass(frozen=True)
class _Tag:
    '''One parsed ``|labels: text|`` tag.'''

    labels: tuple[str, ...]
    text: str


def _snippet(text: str, position: int) -> str:
    '''Quote a short fragment of `text` at `position` for messages.'''
    return text[position : position + _ERROR_SNIPPET]


def _parse_labels(labels_part: str, position: int) -> tuple[str, ...]:
    '''Split and validate the label list of one tag.

    Args:
        labels_part: Raw label list from inside the tag.
        position: Offset of the opening delimiter, for error messages.

    Returns:
        Deduplicated labels in tag order.

    Raises:
        PromptAnnotateError: On empty or invalid label names.
    '''
    labels: list[str] = []
    for label in labels_part.split(LABELS_SEPARATOR):
        label = label.strip()
        if not label:
            raise PromptAnnotateError(
                f"Empty label in tag starting at position {position}"
            )
        if not _LABEL_PATTERN.match(label):
            raise PromptAnnotateError(
                f"Invalid label {label!r} in tag starting at position "
                f"{position}: labels may only contain letters, digits and "
                "underscores"
            )
        if label not in labels:
            labels.append(label)
    return tuple(labels)


def _tokenize(text: str) -> list[_Plain | _Tag]:
    '''Split the raw text into plain runs and parsed tags.

    Args:
        text: Raw annotated prompt.

    Returns:
        Tokens in text order; plain runs may be empty strings.

    Raises:
        PromptAnnotateError: On unclosed tags, missing ``:`` separators
            or invalid label lists (all reported with positions).
    '''
    tokens: list[_Plain | _Tag] = []
    pos = 0
    while True:
        pipe = text.find(SPAN_DELIMITER, pos)
        if pipe == -1:
            tokens.append(_Plain(text[pos:]))
            return tokens
        tokens.append(_Plain(text[pos:pipe]))
        close = text.find(SPAN_DELIMITER, pipe + 1)
        if close == -1:
            raise PromptAnnotateError(
                f"Unclosed tag starting at position {pipe}: "
                f"{_snippet(text, pipe)!r}"
            )
        body = text[pipe + 1 : close]
        separator = body.find(LABEL_SEPARATOR)
        if separator == -1:
            raise PromptAnnotateError(
                f"Tag starting at position {pipe} has no "
                f"{LABEL_SEPARATOR!r} between labels and text: "
                f"{_snippet(text, pipe)!r}"
            )
        labels = _parse_labels(body[:separator], pipe)
        tokens.append(_Tag(labels=labels, text=body[separator + 1 :].strip()))
        pos = close + 1


def _assemble(raw: str, tokens: list[_Plain | _Tag]) -> AnnotatedPrompt:
    '''Build the clean prompt and the spans from parsed tokens.

    Empty tags (whitespace-only span text) are dropped. Plain runs that
    become adjacent across a dropped tag are merged with normalized
    comma spacing, and separator runs at the prompt edges are stripped —
    but only inside plain runs, never inside span text, so every span's
    offsets always slice its exact text out of the clean prompt.

    Args:
        raw: Original annotated text, kept on the result.
        tokens: Tokens from `_tokenize`.

    Returns:
        The assembled annotated prompt.
    '''
    # (is_span, text, labels) fragments in final order.
    fragments: list[tuple[bool, str, tuple[str, ...]]] = []
    merge_pending = False
    for token in tokens:
        if isinstance(token, _Tag):
            if token.text:
                fragments.append((True, token.text, token.labels))
            else:
                merge_pending = True
            continue
        if merge_pending and fragments and not fragments[-1][0]:
            merged = fragments[-1][1] + token.text
            merged = _COMMA_RUN.sub(", ", merged)
            merged = _WHITESPACE_RUN.sub(" ", merged)
            fragments[-1] = (False, merged, ())
        else:
            fragments.append((False, token.text, ()))
        merge_pending = False

    if fragments and not fragments[0][0]:
        fragments[0] = (
            False,
            fragments[0][1].lstrip(_SEPARATORS),
            (),
        )
    if fragments and not fragments[-1][0]:
        fragments[-1] = (
            False,
            fragments[-1][1].rstrip(_SEPARATORS),
            (),
        )

    clean_parts: list[str] = []
    spans: list[PromptSpan] = []
    length = 0
    for is_span, text, labels in fragments:
        if not text:
            continue
        if is_span:
            spans.append(
                PromptSpan(
                    labels=labels,
                    text=text,
                    start=length,
                    end=length + len(text),
                )
            )
        clean_parts.append(text)
        length += len(text)

    return AnnotatedPrompt(raw=raw, clean="".join(clean_parts), spans=tuple(spans))


def parse_annotated_prompt(text: str) -> AnnotatedPrompt:
    '''Parse a prompt with inline label markup.

    The markup is flat, non-nested: ``|label1,label2: text|`` marks
    `text` as belonging to `label1` and `label2`. Text outside tags is
    the unmarked remainder, implicitly labelled `all`. Tags are removed
    from the clean prompt; repeated labels are the way to express
    interleaved regions, so nesting is not supported.

    Args:
        text: Raw prompt with tag markup (may be empty).

    Returns:
        The parsed prompt (clean text plus tagged spans).

    Raises:
        PromptAnnotateError: On malformed markup; the message carries
            the offending position and a text snippet.
    '''
    return _assemble(text, _tokenize(text))


def to_impact_wildcard(annotated: AnnotatedPrompt) -> str:
    '''Render annotations as an Impact Pack label-mode wildcard.

    The output starts with the ``[LAB]`` header followed by one
    ``[label] text`` line per label. Unmarked prompt content becomes
    the ``[ALL]`` line. Impact Pack concatenates the ``[ALL]`` value
    and the matching label value with no separator, so the ``[ALL]``
    line ends with a comma whenever label lines follow.

    Args:
        annotated: Parsed prompt from `parse_annotated_prompt`.

    Returns:
        Wildcard text for Detailer (SEGS)-style `wildcard` inputs.
    '''
    segments = annotated.segments_by_label()
    lines = [IMPACT_LAB_HEADER]
    if DEFAULT_LABEL in segments:
        line = f"[{IMPACT_ALL_KEY}] {segments[DEFAULT_LABEL]}"
        if len(segments) > 1:
            line += ","
        lines.append(line)
    for label in annotated.labels:
        if label != DEFAULT_LABEL:
            lines.append(f"[{label}] {segments[label]}")
    return "\n".join(lines)


def segment_text(
    annotated: AnnotatedPrompt, label: str, include_common: bool = False
) -> str:
    '''Return the prompt text of one label, optionally with the common part.

    Args:
        annotated: Parsed prompt from `parse_annotated_prompt`.
        label: Label to extract.
        include_common: Prepend the unmarked (`all`) content, e.g. to
            build an inpaint prompt from the common quality tags plus
            the region-specific tags.

    Returns:
        The composed prompt text.

    Raises:
        ValueError: If `label` has no content in the annotation.
    '''
    segments = annotated.segments_by_label()
    if label not in segments:
        available = ", ".join(segments) or "(none)"
        raise ValueError(
            f"Label {label!r} has no content in the annotation; "
            f"available labels: {available}"
        )
    parts: list[str] = []
    if (
        include_common
        and label != DEFAULT_LABEL
        and DEFAULT_LABEL in segments
    ):
        parts.append(segments[DEFAULT_LABEL])
    parts.append(segments[label])
    return ", ".join(parts)
