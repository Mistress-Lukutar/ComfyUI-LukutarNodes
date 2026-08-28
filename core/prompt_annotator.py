'''
File:   prompt_annotator.py
Brief:  Inline label-markup parser for region-wise prompt annotation.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.7.0
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
#: Separator used when rendering the label set as one string.
LABEL_LIST_SEPARATOR = ", "

#: Label implicitly assigned to unmarked prompt text.
DEFAULT_LABEL = "all"

#: Impact Pack wildcard key applied to every segment regardless of label.
IMPACT_ALL_KEY = "ALL"
#: Impact Pack wildcard header that switches the text into label mode.
IMPACT_LAB_HEADER = "[LAB]"

#: Editing modes of `edit_segment`: put the text before the labels'
#: first fragment, after their last one, remove the listed tags from
#: them, create a new span for labels that do not exist yet, or delete
#: the labels with their content entirely.
EDIT_MODES = ("prepend", "append", "remove", "new", "delete")

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


def _missing_labels_message(missing: list[str], segments: dict[str, str]) -> str:
    '''Build the shared "labels not in annotation" error message.'''
    quoted = ", ".join(repr(label) for label in missing)
    available = LABEL_LIST_SEPARATOR.join(segments) or "(none)"
    return (
        f"Label(s) {quoted} have no content in the annotation; "
        f"available labels: {available}"
    )


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
        raise ValueError(_missing_labels_message([label], segments))
    parts: list[str] = []
    if (
        include_common
        and label != DEFAULT_LABEL
        and DEFAULT_LABEL in segments
    ):
        parts.append(segments[DEFAULT_LABEL])
    parts.append(segments[label])
    return ", ".join(parts)


def labels_text(
    annotated: AnnotatedPrompt, include_common: bool = False
) -> str:
    '''Render the annotation's label set as one comma-separated string.

    Labels keep their first-appearance order and are deduplicated,
    e.g. ``"body, face, hair"``. The implicit `all` label of the
    unmarked common part is included only when `include_common` is set.

    Args:
        annotated: Parsed prompt from `parse_annotated_prompt`.
        include_common: Include the implicit `all` label.

    Returns:
        The label list joined with `", "`; an empty string when the
        annotation has no (selected) labels.
    '''
    labels = annotated.labels
    if not include_common:
        labels = tuple(label for label in labels if label != DEFAULT_LABEL)
    return LABEL_LIST_SEPARATOR.join(labels)


def _structure_tokens(annotated: AnnotatedPrompt) -> list[_Plain | _Tag]:
    '''Recover the plain/tag token sequence of a parsed annotation.

    Plain runs are sliced out of `clean` between the spans (verbatim,
    separators included); spans become `_Tag` tokens with their exact
    label lists and texts.
    '''
    tokens: list[_Plain | _Tag] = []
    cursor = 0
    for span in annotated.spans:
        if span.start > cursor:
            tokens.append(_Plain(annotated.clean[cursor : span.start]))
        tokens.append(_Tag(labels=span.labels, text=span.text))
        cursor = max(cursor, span.end)
    if cursor < len(annotated.clean):
        tokens.append(_Plain(annotated.clean[cursor:]))
    return tokens


def _tokens_to_markup(tokens: list[_Plain | _Tag]) -> str:
    '''Join a token sequence back into inline ``|label: text|`` markup.'''
    parts: list[str] = []
    for token in tokens:
        if isinstance(token, _Tag):
            parts.append(
                f"{SPAN_DELIMITER}{LABELS_SEPARATOR.join(token.labels)}"
                f"{LABEL_SEPARATOR} {token.text}{SPAN_DELIMITER}"
            )
        else:
            parts.append(token.text)
    return "".join(parts)


def _carries(token: _Plain | _Tag, label: str) -> bool:
    '''Whether a structure token carries `label`.

    Tags carry each of their labels; plain runs carry only the implicit
    `all` label.
    '''
    if isinstance(token, _Tag):
        return label in token.labels
    return label == DEFAULT_LABEL


def _with_text(token: _Plain | _Tag, text: str) -> _Plain | _Tag:
    '''Copy a token with its text replaced.'''
    if isinstance(token, _Tag):
        return _Tag(labels=token.labels, text=text)
    return _Plain(text=text)


def _insert_text(fragment: str, text: str, mode: str) -> str:
    '''Insert `text` at the start or end of a fragment.

    Separator runs at the touched edge (plain runs keep their `, `
    glue between spans) are preserved so the surrounding markup still
    re-parses to a sensibly comma-spaced clean prompt.
    '''
    if mode == "prepend":
        head = fragment[: len(fragment) - len(fragment.lstrip(_SEPARATORS))]
        return f"{head}{text}, {fragment.lstrip(_SEPARATORS)}"
    tail = fragment[len(fragment.rstrip(_SEPARATORS)) :]
    return f"{fragment.rstrip(_SEPARATORS)}, {text}{tail}"


def _remove_tags(fragment: str, removals: set[str]) -> str:
    '''Drop the comma-separated tags in `removals` from a fragment.

    Tags match exactly after trimming; surviving tags are re-joined
    with normalized `", "` spacing.
    '''
    kept = [
        part.strip()
        for part in fragment.split(LABELS_SEPARATOR)
        if part.strip() not in removals
    ]
    return LABEL_LIST_SEPARATOR.join(kept)


def to_markup(annotated: AnnotatedPrompt) -> str:
    '''Render the annotation back as inline ``|label: text|`` markup.

    Round-trips through `parse_annotated_prompt`: re-parsing the output
    yields the same clean prompt and the same spans. Multi-label spans
    serialize as ``|body,hair: text|``; plain runs are emitted verbatim
    between the tags.

    Args:
        annotated: Parsed prompt from `parse_annotated_prompt`.

    Returns:
        Markup text that parses back to an equivalent annotation.
    '''
    return _tokens_to_markup(_structure_tokens(annotated))


def _parse_label_list(value: str) -> tuple[str, ...]:
    '''Split a comma-separated label list.

    Args:
        value: Raw label field content, e.g. ``"face, body"``.

    Returns:
        Deduplicated labels in typed order.

    Raises:
        ValueError: On an empty list or an invalid label name.
    '''
    labels: list[str] = []
    for label in value.split(LABELS_SEPARATOR):
        label = label.strip()
        if not label:
            continue
        if not _LABEL_PATTERN.match(label):
            raise ValueError(
                f"Invalid label {label!r}: labels may only contain "
                "letters, digits and underscores"
            )
        if label not in labels:
            labels.append(label)
    if not labels:
        raise ValueError("Label list must not be empty")
    return tuple(labels)


def _drop_labels(
    tokens: list[_Plain | _Tag], labels: tuple[str, ...]
) -> list[_Plain | _Tag]:
    '''Remove labels together with their content from a token sequence.

    Tags lose the dropped labels; a tag left without any label vanishes
    with its text. When the implicit `all` label is dropped, plain runs
    with content are replaced by plain `", "` glue so surviving spans
    stay comma-separated in the clean prompt (pure separator runs are
    kept as they were).
    '''
    dropped = set(labels)
    kept: list[_Plain | _Tag] = []
    for token in tokens:
        if isinstance(token, _Tag):
            remaining = tuple(
                label for label in token.labels if label not in dropped
            )
            if remaining:
                kept.append(_Tag(labels=remaining, text=token.text))
        elif DEFAULT_LABEL in dropped and token.text.strip(_SEPARATORS):
            kept.append(_Plain(text=LABEL_LIST_SEPARATOR))
        else:
            kept.append(token)
    # A dropped tag leaves its neighbouring separator glues adjacent;
    # coalesce plain runs with the same comma normalization _assemble
    # applies across dropped empty tags.
    result: list[_Plain | _Tag] = []
    for token in kept:
        if isinstance(token, _Plain) and result and isinstance(
            result[-1], _Plain
        ):
            merged = result[-1].text + token.text
            merged = _COMMA_RUN.sub(", ", merged)
            merged = _WHITESPACE_RUN.sub(" ", merged)
            result[-1] = _Plain(text=merged)
        else:
            result.append(token)
    return result


def edit_segment(
    annotated: AnnotatedPrompt, labels: str, mode: str, text: str
) -> AnnotatedPrompt:
    '''Edit labels inside the annotation.

    `labels` is a comma-separated list; every listed label gets the
    edit applied. `mode` picks the edit:

    - ``prepend`` puts `text` before each label's first fragment,
      ``append`` after its last one (comma-joined);
    - ``remove`` deletes the comma-separated tags listed in `text`
      from every fragment of the labels (exact match after trimming;
      removing tags that are not present changes nothing and returns
      the input object as-is);
    - ``new`` appends one new span ``|labels: text|`` to the end of
      the prompt; every listed label must not exist yet;
    - ``delete`` removes the labels with their content entirely —
      labels are stripped from multi-label tags, spans left without a
      label vanish with their text; `text` is ignored.

    Editing the implicit `all` label edits the unmarked common part in
    place (it stays unmarked) and deleting it removes the common text,
    keeping the separators between the surviving spans. A span shared
    between several labels (``|body,hair: text|``) holds one text, so
    edits to any of its labels change that shared text for all of them;
    a span emptied by ``remove`` disappears from the annotation.

    Args:
        annotated: Parsed prompt from `parse_annotated_prompt`.
        labels: Comma-separated label list to apply the edit to.
        mode: One of `EDIT_MODES`.
        text: Tag(s) to add, the comma-separated tags to remove, or
            the new span's text; ignored in ``delete`` mode.

    Returns:
        The re-parsed edited annotation (offsets recalculated).

    Raises:
        ValueError: On an unknown `mode`, an empty label list, an
            invalid label name, blank `text` (except in ``delete``),
            labels missing from the annotation (except in ``new``), or
            labels already present in ``new`` mode.
    '''
    if mode not in EDIT_MODES:
        raise ValueError(
            f"Unknown edit mode {mode!r}; expected one of: "
            f"{LABEL_LIST_SEPARATOR.join(EDIT_MODES)}"
        )
    wanted = _parse_label_list(labels)
    segments = annotated.segments_by_label()
    if mode == "new":
        existing = [label for label in wanted if label in segments]
        if existing:
            quoted = ", ".join(repr(label) for label in existing)
            raise ValueError(
                f"Label(s) {quoted} already exist in the annotation; "
                "use append or prepend to extend them"
            )
    else:
        missing = [label for label in wanted if label not in segments]
        if missing:
            raise ValueError(_missing_labels_message(missing, segments))
    text = text.strip()
    if mode != "delete" and not text:
        raise ValueError("Edit text must not be empty")

    if mode == "new":
        tag = (
            f"{SPAN_DELIMITER}{LABELS_SEPARATOR.join(wanted)}"
            f"{LABEL_SEPARATOR} {text}{SPAN_DELIMITER}"
        )
        markup = to_markup(annotated)
        markup = f"{markup}{LABEL_LIST_SEPARATOR}{tag}" if markup else tag
        return parse_annotated_prompt(markup)

    tokens = _structure_tokens(annotated)
    if mode == "delete":
        tokens = _drop_labels(tokens, wanted)
    elif mode == "remove":
        removals = {
            part.strip()
            for part in text.split(LABELS_SEPARATOR)
            if part.strip()
        }
        edited = [
            _with_text(token, _remove_tags(token.text, removals))
            if any(_carries(token, label) for label in wanted)
            else token
            for token in tokens
        ]
        if _tokens_to_markup(edited) == _tokens_to_markup(tokens):
            return annotated
        tokens = edited
    else:
        for label in wanted:
            targets = [
                index
                for index, token in enumerate(tokens)
                if _carries(token, label) and token.text.strip(_SEPARATORS)
            ]
            index = targets[0] if mode == "prepend" else targets[-1]
            tokens[index] = _with_text(
                tokens[index], _insert_text(tokens[index].text, text, mode)
            )
    return parse_annotated_prompt(_tokens_to_markup(tokens))
