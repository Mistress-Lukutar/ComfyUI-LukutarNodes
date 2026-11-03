'''
File:   test_t2i_prompt.py
Brief:  Unit tests for the T2I prompt-weight normalization (torch-free).
Author: Mistress-Lukutar
Date:   2026-11-03
Version: v0.1.0

Tests core.t2i_prompt: option validation, divide_length,
shift_mean_weight, normalize_weights, scale_to_norm.
'''

from __future__ import annotations

import math

import pytest
from comfyui_lukutar_nodes.core.t2i_prompt import (
    TOKEN_NORMALIZATION,
    WEIGHT_INTERPRETATION,
    _norm_mag,
    divide_length,
    normalize_weights,
    scale_to_norm,
    shift_mean_weight,
    validate_options,
)

# ======================================================================
# validate_options
# ======================================================================


def test_validate_options_accepts_all_combos() -> None:
    """Every listed combo pair must validate to itself."""
    for norm in TOKEN_NORMALIZATION:
        for interp in WEIGHT_INTERPRETATION:
            assert validate_options(norm, interp) == (norm, interp)


def test_validate_options_rejects_unknown() -> None:
    """Unknown entries must raise ValueError."""
    with pytest.raises(ValueError):
        validate_options("bogus", "A1111")
    with pytest.raises(ValueError):
        validate_options("mean", "bogus")


# ======================================================================
# _norm_mag
# ======================================================================


def test_norm_mag_single_token_is_identity() -> None:
    """A one-token word keeps its weight."""
    assert _norm_mag(2.0, 1) == pytest.approx(2.0)
    assert _norm_mag(1.0, 1) == pytest.approx(1.0)


def test_norm_mag_dilutes_by_sqrt_length() -> None:
    """A two-token word with weight 2 dilutes to 1 + sqrt(1/2)."""
    assert _norm_mag(2.0, 2) == pytest.approx(1.0 + math.sqrt(0.5))


def test_norm_mag_below_one() -> None:
    """Weights below 1 dilute symmetrically (sign preserved)."""
    assert _norm_mag(0.0, 2) == pytest.approx(1.0 - math.sqrt(0.5))


def test_norm_mag_clamps_nonpositive_length() -> None:
    """word_len < 1 is treated as 1."""
    assert _norm_mag(2.0, 0) == pytest.approx(2.0)


# ======================================================================
# divide_length
# ======================================================================


def test_divide_length_softens_multi_token_words() -> None:
    """A two-token word with weight 2.5 gets _norm_mag per token."""
    word_ids = [[0, 1, 1, 2]]
    weights = [[1.0, 2.5, 2.5, 1.0]]
    result = divide_length(word_ids, weights)
    expected = 1.0 + math.sqrt(1.5 ** 2 / 2)
    assert result[0][0] == pytest.approx(1.0)  # padding token
    assert result[0][1] == pytest.approx(expected)
    assert result[0][2] == pytest.approx(expected)
    assert result[0][3] == pytest.approx(1.0)


def test_divide_length_handles_multiple_words() -> None:
    """Two words of different lengths dilute independently."""
    word_ids = [[1, 1, 2, 2, 2]]
    weights = [[2.0, 2.0, 4.0, 4.0, 4.0]]
    result = divide_length(word_ids, weights)
    assert result[0][0] == pytest.approx(1.0 + math.sqrt(0.5))
    assert result[0][2] == pytest.approx(1.0 + math.sqrt(9.0 / 3))


# ======================================================================
# shift_mean_weight
# ======================================================================


def test_shift_mean_weight_shifts_to_one() -> None:
    """After shifting, the mean over real-word weights equals 1."""
    word_ids = [[1, 2, 3]]
    weights = [[0.5, 0.5, 0.5]]
    result = shift_mean_weight(word_ids, weights)
    mean = sum(result[0]) / 3
    assert mean == pytest.approx(1.0)


def test_shift_mean_weight_keeps_padding() -> None:
    """Padding tokens (word id 0) keep their weight."""
    word_ids = [[0, 1]]
    weights = [[1.7, 1.0]]
    result = shift_mean_weight(word_ids, weights)
    assert result[0][0] == pytest.approx(1.7)


def test_shift_mean_weight_all_padding_is_noop() -> None:
    """No real words -> unchanged weights."""
    word_ids = [[0, 0]]
    weights = [[0.3, 0.7]]
    result = shift_mean_weight(word_ids, weights)
    assert result[0] == pytest.approx([0.3, 0.7])


# ======================================================================
# normalize_weights
# ======================================================================


def test_normalize_weights_none_returns_copy() -> None:
    """'none' returns an equal copy, not the same object."""
    word_ids = [[1, 2]]
    weights = [[1.0, 2.0]]
    result = normalize_weights("none", word_ids, weights)
    assert result == weights
    assert result is not weights
    assert result[0] is not weights[0]


def test_normalize_weights_length_uses_divide() -> None:
    """'length' matches divide_length exactly."""
    word_ids = [[1, 1]]
    weights = [[2.0, 2.0]]
    assert normalize_weights("length", word_ids, weights) == (
        divide_length(word_ids, weights)
    )


def test_normalize_weights_mean_matches_shift() -> None:
    """'mean' matches shift_mean_weight exactly."""
    word_ids = [[1, 2]]
    weights = [[0.5, 0.5]]
    assert normalize_weights("mean", word_ids, weights) == (
        shift_mean_weight(word_ids, weights)
    )


def test_normalize_weights_length_mean_applies_both() -> None:
    """'length+mean' divides first, then shifts the mean to 1."""
    word_ids = [[1, 1, 2]]
    weights = [[1.0, 1.0, 1.0]]
    result = normalize_weights("length+mean", word_ids, weights)
    real = [w for w, wid in zip(result[0], word_ids[0], strict=True) if wid != 0]
    assert sum(real) / len(real) == pytest.approx(1.0)


# ======================================================================
# scale_to_norm
# ======================================================================


def test_scale_to_norm_scales_to_max() -> None:
    """The largest weight maps exactly onto w_max."""
    word_ids = [[1, 2]]
    weights = [[0.5, 2.0]]
    result = scale_to_norm(word_ids, weights, 1.0)
    assert result[0][1] == pytest.approx(1.0)
    assert result[0][0] == pytest.approx(0.25)


def test_scale_to_norm_caps_at_w_max() -> None:
    """When w_max is below the current max, it caps the scale."""
    word_ids = [[1]]
    weights = [[4.0]]
    result = scale_to_norm(word_ids, weights, 1.0)
    assert result[0][0] == pytest.approx(1.0)


def test_scale_to_norm_padding_gets_w_max() -> None:
    """Padding tokens (word id 0) receive w_max directly."""
    word_ids = [[0, 1]]
    weights = [[9.0, 3.0]]
    result = scale_to_norm(word_ids, weights, 1.0)
    assert result[0][0] == pytest.approx(1.0)
    assert result[0][1] == pytest.approx(1.0 / 3)


def test_scale_to_norm_empty_is_noop() -> None:
    """An empty matrix returns an empty copy."""
    assert scale_to_norm([], [], 1.0) == []
