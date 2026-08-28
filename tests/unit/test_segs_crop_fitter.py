'''
File:   test_segs_crop_fitter.py
Brief:  Unit tests for the SEGS crop region fitter engine.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v1.0.0
'''

from __future__ import annotations

import numpy as np
import pytest
from comfyui_lukutar_nodes.core.segs_crop_fitter import (
    FIT_MODES,
    MODE_ASPECT,
    MODE_EXACT,
    fit_crop_region,
    realign_mask,
)


def region_size(region: tuple[int, int, int, int]) -> tuple[int, int]:
    '''(width, height) of a crop region rect.'''
    return region[2] - region[0], region[3] - region[1]


class TestFitCropRegionExact:
    '''``exact`` mode: the crop is the target size around the bbox.'''

    def test_target_hit_exactly_when_bbox_fits(self):
        fitted = fit_crop_region(
            (400, 300, 600, 500), (1000, 800), (512, 512), MODE_EXACT
        )
        # Centered on the bbox center (500, 400).
        assert fitted.region == (244, 144, 756, 656)
        assert region_size(fitted.region) == (512, 512)
        assert not fitted.clamped
        assert not fitted.grown

    def test_region_shifted_inside_canvas(self):
        fitted = fit_crop_region(
            (10, 10, 60, 60), (1000, 800), (512, 512), MODE_EXACT
        )
        assert fitted.region == (0, 0, 512, 512)
        assert not fitted.clamped
        assert not fitted.grown

    def test_region_shifted_inside_canvas_bottom_right(self):
        fitted = fit_crop_region(
            (940, 740, 990, 790), (1000, 800), (512, 512), MODE_EXACT
        )
        assert fitted.region == (488, 288, 1000, 800)

    def test_grows_to_contain_larger_bbox(self):
        # bbox 401x300 vs target 256x256: grows, rounded up to /8.
        fitted = fit_crop_region(
            (100, 100, 501, 400), (1000, 800), (256, 256), MODE_EXACT
        )
        assert region_size(fitted.region) == (408, 304)
        assert fitted.grown
        assert not fitted.clamped
        # The bbox stays fully inside.
        x1, y1, x2, y2 = fitted.region
        assert x1 <= 100 and y1 <= 100 and x2 >= 501 and y2 >= 400

    def test_target_used_verbatim_without_rounding(self):
        fitted = fit_crop_region(
            (400, 300, 500, 400), (1000, 800), (333, 217), MODE_EXACT, 8
        )
        assert region_size(fitted.region) == (333, 217)

    def test_clamped_to_canvas(self):
        fitted = fit_crop_region(
            (100, 100, 200, 200), (300, 200), (512, 512), MODE_EXACT
        )
        assert fitted.region == (0, 0, 300, 200)
        assert fitted.clamped

    def test_float_bbox_accepted(self):
        fitted = fit_crop_region(
            (16.0, 16.0, 48.0, 48.0), (96, 72), (64, 64), MODE_EXACT
        )
        assert fitted.region == (0, 0, 64, 64)

    def test_degenerate_bbox_normalized(self):
        fitted = fit_crop_region(
            (100, 100, 100, 100), (1000, 800), (64, 64), MODE_EXACT
        )
        assert region_size(fitted.region) == (64, 64)
        x1, y1, x2, y2 = fitted.region
        assert x1 <= 100 <= x2 and y1 <= 100 <= y2

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown fit mode"):
            fit_crop_region((0, 0, 10, 10), (100, 100), (64, 64), "nope")


class TestFitCropRegionAspect:
    '''``aspect`` mode: uniform scale to the target's longer side.'''

    def test_scales_to_longest_side(self):
        # bbox 400x173 @ target 512 -> 512x224 (221.44 rounded up to /8).
        fitted = fit_crop_region(
            (0, 0, 400, 173), (1000, 800), (512, 512), MODE_ASPECT
        )
        assert region_size(fitted.region) == (512, 224)
        assert not fitted.grown
        assert not fitted.clamped

    def test_never_shrinks_below_bbox(self):
        # bbox 552x239 already longer than 512: stays at 1x; 552 is a
        # multiple of 8, 239 rounds up to 240.
        fitted = fit_crop_region(
            (0, 0, 552, 239), (1000, 800), (512, 512), MODE_ASPECT
        )
        assert region_size(fitted.region) == (552, 240)
        assert fitted.grown

    def test_square_bbox_hits_target(self):
        fitted = fit_crop_region(
            (100, 100, 200, 200), (1000, 800), (512, 512), MODE_ASPECT
        )
        assert region_size(fitted.region) == (512, 512)
        assert not fitted.grown

    def test_round_to_one_keeps_raw_scale(self):
        fitted = fit_crop_region(
            (0, 0, 400, 173), (1000, 800), (512, 512), MODE_ASPECT, 1
        )
        assert region_size(fitted.region) == (512, 222)

    def test_centered_on_bbox_center(self):
        fitted = fit_crop_region(
            (300, 200, 500, 372), (1000, 800), (512, 512), MODE_ASPECT
        )
        x1, y1, x2, y2 = fitted.region
        # Bbox center (400, 286) stays the region center.
        assert x1 + x2 == 800
        assert y1 + y2 == 572
        assert x1 <= 300 and x2 >= 500 and y1 <= 200 and y2 >= 372


class TestRealignMask:
    '''Re-cutting a cropped mask for a new crop region.'''

    def test_expand_zero_fills_added_context(self):
        # Old region (8, 8, 56, 56), ones at mask rows/cols 8:40 ->
        # image coords 16:48. New region (0, 0, 64, 64).
        mask = np.zeros((48, 48), dtype=np.float32)
        mask[8:40, 8:40] = 1.0
        result = realign_mask(mask, (8, 8, 56, 56), (0, 0, 64, 64))
        assert result.shape == (64, 64)
        assert result.dtype == np.float32
        assert result[16, 16] == 1.0
        assert result[47, 47] == 1.0
        assert result[10, 10] == 0.0
        assert result[63, 63] == 0.0
        assert result[:16, :].max() == 0.0

    def test_coordinates_stick_to_image_pixels(self):
        old = np.arange(48 * 48, dtype=np.float32).reshape(48, 48)
        result = realign_mask(old, (8, 8, 56, 56), (8, 8, 56, 56))
        np.testing.assert_array_equal(result, old)
        # Shifted by 8: the new region's (0, 0) is the old (8, 8) pixel.
        shifted = realign_mask(old, (8, 8, 56, 56), (16, 16, 56, 56))
        assert shifted.shape == (40, 40)
        assert shifted[0, 0] == old[8, 8]
        assert shifted[39, 39] == old[47, 47]

    def test_shrink_keeps_center_crop(self):
        mask = np.ones((64, 64), dtype=np.float32)
        result = realign_mask(mask, (0, 0, 64, 64), (16, 16, 48, 48))
        assert result.shape == (32, 32)
        assert result.min() == 1.0

    def test_preserves_leading_batch_dims(self):
        mask = np.zeros((2, 48, 48), dtype=np.float32)
        mask[0, 8:40, 8:40] = 1.0
        result = realign_mask(mask, (8, 8, 56, 56), (0, 0, 64, 64))
        assert result.shape == (2, 64, 64)
        assert result[0, 20, 20] == 1.0
        assert result[1].max() == 0.0

    def test_no_overlap_returns_zeros(self):
        mask = np.ones((10, 10), dtype=np.float32)
        result = realign_mask(mask, (0, 0, 10, 10), (20, 20, 30, 30))
        assert result.shape == (10, 10)
        assert result.max() == 0.0


def test_fit_modes_tuple():
    assert FIT_MODES == ("exact", "aspect")
