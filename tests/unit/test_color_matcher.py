'''
File:   test_color_matcher.py
Brief:  Unit tests for the frequency-separation color matching engine.
Author: Mistress-Lukutar
Date:   2026-08-24
Version: v0.1.0
'''

from __future__ import annotations

import cv2
import numpy as np
import pytest

from comfyui_lukutar_nodes.core.color_matcher import (
    METRIC_ENVELOPE,
    METHOD_REINHARD,
    METHOD_REPLACE,
    ColorMatchError,
    ColorMatcher,
    tune_sigma,
)


def _make_reference(width: int = 96, height: int = 64) -> np.ndarray:
    '''Synthetic warm-toned reference: smooth gradients plus texture.'''
    xs = np.linspace(0, 255, width, dtype=np.float32)
    ys = np.linspace(0, 255, height, dtype=np.float32)
    grid = xs[np.newaxis, :] + ys[:, np.newaxis]
    rng = np.random.default_rng(42)
    texture = rng.uniform(-12.0, 12.0, size=(height, width)).astype(np.float32)
    frame = np.stack(
        [
            np.clip(grid + texture, 0, 255),
            np.clip(grid * 0.6 + 40.0 - texture, 0, 255),
            np.clip(255.0 - grid * 0.4, 0, 255),
        ],
        axis=-1,
    )
    return frame.astype(np.float32)


def _color_shifted(frame: np.ndarray) -> np.ndarray:
    '''Simulate VAE drift: cool color shift on the same structure.'''
    shifted = frame.copy()
    shifted[..., 0] *= 0.65
    shifted[..., 2] = np.clip(shifted[..., 2] * 1.2 + 25.0, 0, 255)
    return np.clip(shifted, 0, 255).astype(np.float32)


def _lab_mean_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_lab = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(
        np.float32
    )
    b_lab = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(
        np.float32
    )
    return float(np.linalg.norm(a_lab.mean(axis=(0, 1)) - b_lab.mean(axis=(0, 1))))


def test_rejects_non_positive_sigma() -> None:
    with pytest.raises(ColorMatchError):
        ColorMatcher(sigma=0.0)


def test_rejects_unknown_method() -> None:
    with pytest.raises(ColorMatchError):
        ColorMatcher(sigma=10.0, method="magic")


def test_output_shape_and_dtype() -> None:
    reference = _make_reference()
    target = _color_shifted(reference)
    result = ColorMatcher(sigma=12.0).process(reference, target)
    assert result.shape == target.shape
    assert result.dtype == np.uint8


def test_reference_is_resized_to_target() -> None:
    reference = _make_reference(width=48, height=32)
    target = _color_shifted(_make_reference())
    result = ColorMatcher(sigma=12.0).process(reference, target)
    assert result.shape[:2] == target.shape[:2]


def test_reinhard_moves_colors_toward_reference() -> None:
    reference = _make_reference()
    target = _color_shifted(reference)
    result = ColorMatcher(sigma=12.0, method=METHOD_REINHARD).process(
        reference, target
    )
    before = _lab_mean_distance(target, reference)
    after = _lab_mean_distance(result, reference)
    assert after < before


def test_replace_borrows_reference_low_layer() -> None:
    reference = _make_reference()
    target = _color_shifted(reference)
    matcher = ColorMatcher(sigma=15.0, method=METHOD_REPLACE)
    result = matcher.process(reference, target)
    # "replace" recomposes the reference low layer with the target high
    # layer exactly: result = uint8(clip(ref_low + target_high)).
    target_f = target.astype(np.float32)
    target_low = cv2.GaussianBlur(target_f, (0, 0), sigmaX=15.0)
    ref_low = cv2.GaussianBlur(
        reference.astype(np.float32), (0, 0), sigmaX=15.0
    )
    expected = np.clip(
        ref_low + (target_f - target_low), 0, 255
    ).astype(np.uint8)
    assert np.array_equal(result, expected)


def test_tune_sigma_returns_grid_candidate() -> None:
    reference = _make_reference()
    target = _color_shifted(reference)
    tuned = tune_sigma(
        reference,
        target,
        sigma_min=4.0,
        sigma_max=24.0,
        sigma_step=4.0,
        eval_sigma=10.0,
        metric=METRIC_ENVELOPE,
        method=METHOD_REINHARD,
    )
    grid = {4.0, 8.0, 12.0, 16.0, 20.0, 24.0}
    assert tuned.sigma in grid
    assert tuned.score >= 0.0
    assert tuned.image.shape == target.shape
    assert tuned.image.dtype == np.uint8


def test_tune_sigma_rejects_inverted_grid() -> None:
    reference = _make_reference()
    with pytest.raises(ColorMatchError):
        tune_sigma(reference, reference, sigma_min=10.0, sigma_max=2.0)


def test_tune_sigma_rejects_zero_step() -> None:
    reference = _make_reference()
    with pytest.raises(ColorMatchError):
        tune_sigma(reference, reference, sigma_step=0.0)
