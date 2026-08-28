'''
File:   prompt_annotate.py
Brief:  ComfyUI nodes for region-wise prompt annotation markup.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.7.0
'''

from __future__ import annotations

import logging
from typing import Any

from ..core.prompt_annotator import (
    EDIT_MODES,
    AnnotatedPrompt,
    edit_segment,
    labels_text,
    parse_annotated_prompt,
    segment_text,
    to_impact_wildcard,
    to_markup,
)

logger = logging.getLogger(__name__)


def _as_annotations(value: Any) -> AnnotatedPrompt:
    '''Narrow an input to an AnnotatedPrompt with a helpful error.

    Args:
        value: Value received on the ANNOTATIONS input.

    Returns:
        The annotated prompt.

    Raises:
        ValueError: If the value did not come from Prompt Annotate.
    '''
    if not isinstance(value, AnnotatedPrompt):
        raise ValueError(
            "Expected an ANNOTATIONS output from Prompt Annotate "
            f"(Lukutar), got {type(value).__name__}"
        )
    return value


class PromptAnnotateNode:
    '''Annotate one prompt with inline region labels.

    The prompt stays a single text: ``|label1,label2: text|`` tags mark
    which parts belong to which region (face, body, background — the
    label set is free-form and should match the classifier's; labels
    are simply typed into the markup, there is no fixed list). Text
    outside tags is the unmarked common part. The node outputs the
    annotations object plus the tag-free prompt, so the same text drives
    both the base generation and the per-region passes.
    '''

    CATEGORY = "Lukutar/Prompt"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Prompt with |label: text| markup, e.g. "
                            "'masterpiece, |face:blue eyes, smirk|, "
                            "|body:thin|'; the Annotate button (web UI) "
                            "edits it visually"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("ANNOTATIONS", "STRING")
    RETURN_NAMES = ("annotations", "clean_prompt")
    FUNCTION = "annotate"

    def annotate(self, text: str) -> tuple[AnnotatedPrompt, str]:
        '''Parse the markup and emit annotations plus the clean prompt.

        Args:
            text: Raw prompt with tag markup.

        Returns:
            Tuple of (annotations object, tag-free prompt).
        '''
        annotated = parse_annotated_prompt(text)
        logger.info(
            "PromptAnnotate: %d span(s), labels [%s]",
            len(annotated.spans),
            ", ".join(annotated.labels) or "none",
        )
        return (annotated, annotated.clean)


class AnnotationsWildcardNode:
    '''Convert prompt annotations to an Impact Pack label-mode wildcard.

    Produces the ``[LAB]`` text consumed by Detailer (SEGS)-style
    `wildcard` inputs: one ``[label] text`` line per label, with the
    unmarked common part as the ``[ALL]`` line.
    '''

    CATEGORY = "Lukutar/Prompt"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "annotations": (
                    "ANNOTATIONS",
                    {
                        "tooltip": (
                            "Annotations from Prompt Annotate (Lukutar)"
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("wildcard",)
    FUNCTION = "convert"

    def convert(self, annotations: Any) -> tuple[str]:
        '''Render the annotations as a ``[LAB]`` wildcard string.

        Args:
            annotations: ANNOTATIONS output of Prompt Annotate.

        Returns:
            One-element tuple with the wildcard text.
        '''
        wildcard = to_impact_wildcard(_as_annotations(annotations))
        logger.debug("AnnotationsWildcard:\n%s", wildcard)
        return (wildcard,)


class AnnotationSegmentNode:
    '''Extract one label's prompt text from the annotations.

    With `include_common` on, the unmarked part (implicit `all`) is
    prepended — e.g. an inpaint prompt made of the shared quality tags
    plus the region-specific ones.
    '''

    CATEGORY = "Lukutar/Prompt"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "annotations": (
                    "ANNOTATIONS",
                    {
                        "tooltip": (
                            "Annotations from Prompt Annotate (Lukutar)"
                        )
                    },
                ),
                "label": (
                    "STRING",
                    {
                        "default": "face",
                        "tooltip": "Label to extract, e.g. face / body",
                    },
                ),
                "include_common": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "common + label",
                        "label_off": "label only",
                        "tooltip": (
                            "Prepend the unmarked common part of the prompt"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "segment"

    def segment(
        self, annotations: Any, label: str, include_common: bool = True
    ) -> tuple[str]:
        '''Compose the prompt text for one label.

        Args:
            annotations: ANNOTATIONS output of Prompt Annotate.
            label: Label whose text to extract.
            include_common: Prepend the unmarked common part.

        Returns:
            One-element tuple with the composed text.

        Raises:
            ValueError: If the label has no content in the annotation.
        '''
        text = segment_text(
            _as_annotations(annotations), label, include_common=include_common
        )
        logger.debug("AnnotationSegment [%s]: %s", label, text)
        return (text,)


class AnnotationLabelsNode:
    '''List the annotation's labels as one comma-separated string.

    Handy to feed a downstream label picker or to log which regions a
    workflow covers: ``body, face, hair``. The implicit `all` label of
    the unmarked common part is included only with `include_common` on.
    '''

    CATEGORY = "Lukutar/Prompt"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "annotations": (
                    "ANNOTATIONS",
                    {
                        "tooltip": (
                            "Annotations from Prompt Annotate (Lukutar)"
                        )
                    },
                ),
                "include_common": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "regions + common",
                        "label_off": "regions only",
                        "tooltip": (
                            "Include the implicit 'all' label of the "
                            "unmarked common part"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("labels",)
    FUNCTION = "collect"

    def collect(
        self, annotations: Any, include_common: bool = False
    ) -> tuple[str]:
        '''Join the annotation's labels into one string.

        Args:
            annotations: ANNOTATIONS output of Prompt Annotate.
            include_common: Include the implicit `all` label.

        Returns:
            One-element tuple with the comma-separated label list.
        '''
        text = labels_text(
            _as_annotations(annotations), include_common=include_common
        )
        logger.debug("AnnotationLabels: %s", text or "(none)")
        return (text,)


class AnnotationSegmentEditNode:
    '''Edit one label's text inside the annotations (pass-through).

    Takes ANNOTATIONS in and emits the edited ANNOTATIONS out, so one
    shared Prompt Annotate can feed several branches with different
    tweaks: `prepend`/`append` put the typed tags before/after the
    label's text, `remove` deletes the typed comma-separated tags from
    it. The implicit `all` label edits the unmarked common part. A span
    shared by several labels is edited for all of them.
    '''

    CATEGORY = "Lukutar/Prompt"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "annotations": (
                    "ANNOTATIONS",
                    {
                        "tooltip": (
                            "Annotations from Prompt Annotate (Lukutar)"
                        )
                    },
                ),
                "label": (
                    "STRING",
                    {
                        "default": "face",
                        "tooltip": (
                            "Label whose text to edit; unknown labels "
                            "fail the node listing the available ones"
                        ),
                    },
                ),
                "mode": (
                    list(EDIT_MODES),
                    {
                        "default": "prepend",
                        "tooltip": (
                            "prepend: put the text before the label's "
                            "text; append: after it; remove: delete the "
                            "listed comma-separated tags from it"
                        ),
                    },
                ),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Tags to add, or the comma-separated tags to "
                            "remove (matched exactly after trimming)"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("ANNOTATIONS",)
    RETURN_NAMES = ("annotations",)
    FUNCTION = "edit"

    def edit(
        self, annotations: Any, label: str, mode: str, text: str
    ) -> tuple[AnnotatedPrompt]:
        '''Apply one text edit to the chosen label of the annotation.

        Args:
            annotations: ANNOTATIONS output of Prompt Annotate.
            label: Label whose text to edit.
            mode: One of ``prepend`` / ``append`` / ``remove``.
            text: Tag(s) to add, or the comma-separated tags to remove.

        Returns:
            One-element tuple with the edited annotations.

        Raises:
            ValueError: On an unknown label, unknown mode or empty text.
        '''
        edited = edit_segment(
            _as_annotations(annotations), label, mode, text
        )
        logger.debug(
            "AnnotationSegmentEdit [%s/%s]: %s",
            label,
            mode,
            to_markup(edited),
        )
        return (edited,)
