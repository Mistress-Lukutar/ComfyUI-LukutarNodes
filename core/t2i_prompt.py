'''
File:   t2i_prompt.py
Brief:  Token-weight normalization math for the T2I prompt engine.
Author: Mistress-Lukutar
Date:   2026-11-03
Version: v0.1.0

This module is torch-free on purpose (see AGENTS.md architecture
boundaries).  It holds the pure-python parts of the weighted prompt
encoding engine used by the T2I nodes: the combo option tables and the
list-based weight normalization strategies.  The tensor-side weight
interpretations live in ``nodes/t2i.py`` where the CLIP models are
available.

Token lists arrive from ``CLIP.tokenize(text, return_word_ids=True)``:
a sequence (one entry per batch line) of token rows, where each row is
a list of ``(token, weight, word_id)`` tuples.  The helpers below work
on the parallel ``weights``/``word_ids`` matrices (lists of lists of
floats / ints) extracted from that structure.

Word id 0 is the sentinel reserved for padding / framing tokens (BOS,
EOS, pads); normalization strategies never touch those weights.
'''

from __future__ import annotations

import math
from typing import TypeAlias

#: Token normalization options (mirrors the BNK-style combos).
#:
#: * ``none``        — use token weights exactly as tokenized.
#: * ``mean``        — shift weights so the mean over real words is 1.
#: * ``length``      — soften each word's weight by its token count
#:   (magnitude preserved: ``(w-1)/sqrt(n)`` added to 1).
#: * ``length+mean`` — both of the above, length first.
TOKEN_NORMALIZATION: tuple[str, ...] = (
    "none",
    "mean",
    "length",
    "length+mean",
)

#: Default token normalization combo entry.
DEFAULT_TOKEN_NORMALIZATION: str = "mean"

#: Weight interpretation options (mirrors the BNK-style combos).
#:
#: * ``comfy``       — native ComfyUI weighted tokenization.
#: * ``A1111``       — unweighted base embedding scaled from zero and
#:   renormalized to the base mean.
#: * ``compel``      — weights >= 1 encoded natively, < 1 mixed down.
#: * ``comfy++``     — down-weighted mix plus masked residual for > 1.
#: * ``down_weight`` — iterative masked mixing after scaling to norm.
WEIGHT_INTERPRETATION: tuple[str, ...] = (
    "comfy",
    "A1111",
    "compel",
    "comfy++",
    "down_weight",
)

#: Default weight interpretation combo entry.
DEFAULT_WEIGHT_INTERPRETATION: str = "comfy"

#: Upper bound used by the ``down_weight`` interpretation when scaling
#: weights to a common norm.
DEFAULT_W_MAX: float = 1.0

#: Parallel weight matrix: one row per batch line, one float per token.
WeightMatrix: TypeAlias = list[list[float]]

#: Parallel word-id matrix: one row per batch line, one int per token.
WordIdMatrix: TypeAlias = list[list[int]]


def validate_options(
    token_normalization: str,
    weight_interpretation: str,
) -> tuple[str, str]:
    '''Validate the two prompt-engine combo choices.

    Args:
        token_normalization: One of :data:`TOKEN_NORMALIZATION`.
        weight_interpretation: One of :data:`WEIGHT_INTERPRETATION`.

    Returns:
        The validated ``(token_normalization, weight_interpretation)``
        pair (stripped).

    Raises:
        ValueError: If either choice is not a known option.
    '''
    token_normalization = str(token_normalization).strip()
    weight_interpretation = str(weight_interpretation).strip()
    if token_normalization not in TOKEN_NORMALIZATION:
        raise ValueError(
            f"Unknown token normalization {token_normalization!r}; "
            f"expected one of {TOKEN_NORMALIZATION}"
        )
    if weight_interpretation not in WEIGHT_INTERPRETATION:
        raise ValueError(
            f"Unknown weight interpretation {weight_interpretation!r}; "
            f"expected one of {WEIGHT_INTERPRETATION}"
        )
    return token_normalization, weight_interpretation


def _norm_mag(weight: float, word_len: int) -> float:
    '''Softening factor for a word weight spread over ``word_len`` tokens.

    Keeps the weight magnitude constant while diluting its per-token
    impact: the deviation from 1 is divided by ``sqrt(word_len)``.

    Args:
        weight: The word's token weight.
        word_len: Number of tokens the word spans (>= 1).

    Returns:
        The adjusted per-token weight.
    '''
    if word_len < 1:
        word_len = 1
    delta = weight - 1.0
    return 1.0 + math.copysign(math.sqrt(delta * delta / word_len), delta)


def _word_lengths(word_ids: WordIdMatrix) -> dict[int, int]:
    '''Count tokens per word id across the whole batch.

    Args:
        word_ids: Parallel word-id matrix.

    Returns:
        Mapping word id -> number of tokens carrying it. Word id 0 is
        excluded (it marks padding/framing tokens).
    '''
    counts: dict[int, int] = {}
    for row in word_ids:
        for wid in row:
            if wid != 0:
                counts[wid] = counts.get(wid, 0) + 1
    return counts


def divide_length(
    word_ids: WordIdMatrix,
    weights: WeightMatrix,
) -> WeightMatrix:
    '''``length`` normalization: soften weights by word token counts.

    Each real word's weight is passed through :func:`_norm_mag` with the
    word's token count; padding tokens (word id 0) collapse to 1.0.

    Args:
        word_ids: Parallel word-id matrix.
        weights: Parallel weight matrix (same shape).

    Returns:
        A new weight matrix; the input is not modified.
    '''
    counts = _word_lengths(word_ids)
    return [
        [
            1.0 if wid == 0 else _norm_mag(w, counts.get(wid, 1))
            for w, wid in zip(wrow, irow, strict=True)
        ]
        for wrow, irow in zip(weights, word_ids, strict=True)
    ]


def shift_mean_weight(
    word_ids: WordIdMatrix,
    weights: WeightMatrix,
) -> WeightMatrix:
    '''``mean`` normalization: shift weights so real words average 1.

    Computes the mean weight over all tokens with a non-zero word id
    and adds ``1 - mean`` to each of them. Padding tokens keep their
    weight.

    Args:
        word_ids: Parallel word-id matrix.
        weights: Parallel weight matrix (same shape).

    Returns:
        A new weight matrix; the input is not modified.
    '''
    total = 0.0
    count = 0
    for wrow, irow in zip(weights, word_ids, strict=True):
        for w, wid in zip(wrow, irow, strict=True):
            if wid != 0:
                total += w
                count += 1
    delta = 1.0 - (total / count) if count > 0 else 0.0
    return [
        [w if wid == 0 else w + delta for w, wid in zip(wrow, irow, strict=True)]
        for wrow, irow in zip(weights, word_ids, strict=True)
    ]


def normalize_weights(
    token_normalization: str,
    word_ids: WordIdMatrix,
    weights: WeightMatrix,
) -> WeightMatrix:
    '''Apply the selected token normalization strategy.

    Args:
        token_normalization: One of :data:`TOKEN_NORMALIZATION`.
        word_ids: Parallel word-id matrix.
        weights: Parallel weight matrix (same shape).

    Returns:
        The normalized weight matrix (``weights`` unchanged for
        ``none``).
    '''
    if token_normalization == "length":
        return divide_length(word_ids, weights)
    if token_normalization == "mean":
        return shift_mean_weight(word_ids, weights)
    if token_normalization == "length+mean":
        return shift_mean_weight(
            word_ids, divide_length(word_ids, weights)
        )
    return [list(row) for row in weights]


def scale_to_norm(
    word_ids: WordIdMatrix,
    weights: WeightMatrix,
    w_max: float,
) -> WeightMatrix:
    '''Scale all weights so the batch maximum equals ``w_max``.

    Used by the ``down_weight`` interpretation: every token weight is
    remapped to ``(w / max) * w_max`` and padding tokens (word id 0)
    get ``w_max`` directly.

    Args:
        word_ids: Parallel word-id matrix.
        weights: Parallel weight matrix (same shape).
        w_max: Target maximum weight.

    Returns:
        A new weight matrix; the input is not modified.
    '''
    top = max((w for row in weights for w in row), default=0.0)
    if top <= 0.0:
        return [list(row) for row in weights]
    cap = min(top, w_max)
    return [
        [
            cap if wid == 0 else (w / top) * cap
            for w, wid in zip(wrow, irow, strict=True)
        ]
        for wrow, irow in zip(weights, word_ids, strict=True)
    ]
