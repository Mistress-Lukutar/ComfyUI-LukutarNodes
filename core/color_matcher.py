'''
File:   color_matcher.py
Brief:  Frequency-separation color matching engine (numpy/OpenCV, RGB).
Author: Mistress-Lukutar
Date:   2026-08-24
Version: v0.1.0
'''

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Transfer LAB mean/std from the reference onto the target low layer.
METHOD_REINHARD = "reinhard"
#: Use the reference low-frequency layer verbatim.
METHOD_REPLACE = "replace"
#: Supported color transfer methods.
MATCH_METHODS: tuple[str, ...] = (METHOD_REINHARD, METHOD_REPLACE)

#: MSE between blurred layers: color/tonal accuracy, detail ignored.
METRIC_ENVELOPE = "envelope"
#: MSE between raw images: color and structure combined.
METRIC_FULL = "full"
#: Supported evaluation metrics.
EVAL_METRICS: tuple[str, ...] = (METRIC_ENVELOPE, METRIC_FULL)

#: Fixed blur radius for scoring; prevents degenerate minima at high sigma.
DEFAULT_EVAL_SIGMA = 20.0

#: Numerical guard for division by a near-zero standard deviation.
_STD_EPSILON = 1e-6


class ColorMatchError(Exception):
    '''Base exception for color matching failures.'''


@dataclass(frozen=True, slots=True)
class FrequencySplit:
    '''Container for frequency-separated image layers.

    Attributes:
        low: Blurred (low-frequency) layer, float32 [0, 255].
        high: Residual detail (high-frequency) layer, float32.
    '''

    low: np.ndarray
    high: np.ndarray


@dataclass(frozen=True, slots=True)
class TuneResult:
    '''Best sigma search outcome.

    Attributes:
        sigma: Sigma value that produced the best score.
        score: Evaluation score of the best candidate (lower is better).
        image: Result image processed with the best sigma, RGB uint8.
    '''

    sigma: float
    score: float
    image: np.ndarray


class ColorMatcher:
    '''Frequency-separation color restoration engine.

    Splits an image into low/high frequency layers with a Gaussian blur,
    transfers the color envelope of a reference image onto the low layer
    and recombines it with the original high-frequency detail.
    '''

    def __init__(self, sigma: float, method: str = METHOD_REINHARD) -> None:
        '''Create a matcher for a fixed frequency cutoff.

        Args:
            sigma: Gaussian blur radius (frequency cutoff), positive.
            method: Color transfer method, one of MATCH_METHODS.

        Raises:
            ColorMatchError: If sigma is not positive or the method is
                unknown.
        '''
        if sigma <= 0.0:
            raise ColorMatchError(f"Sigma must be positive, got {sigma}")
        if method not in MATCH_METHODS:
            raise ColorMatchError(f"Unknown method: {method!r}")
        self._sigma = sigma
        self._method = method

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def process(
        self,
        original: np.ndarray,
        upscaled: np.ndarray,
    ) -> np.ndarray:
        '''Return the upscaled image with the original color envelope.

        Args:
            original: Reference image, HWC RGB, uint8 or float32 [0, 255].
            upscaled: Target image, HWC RGB, uint8 or float32 [0, 255].
                May differ in resolution; the reference is resized to
                match it.

        Returns:
            Color-matched image, HWC RGB uint8, resolution of `upscaled`.
        '''
        if original.shape[:2] != upscaled.shape[:2]:
            original = self._resize_to_match(original, upscaled)

        target_split = self._frequency_split(upscaled)
        reference_low = cv2.GaussianBlur(
            original.astype(np.float32),
            (0, 0),
            sigmaX=self._sigma,
        )

        if self._method == METHOD_REINHARD:
            matched_low = self._reinhard_transfer(
                target_split.low, reference_low
            )
        else:
            matched_low = reference_low

        result = np.clip(matched_low + target_split.high, 0.0, 255.0)
        return result.astype(np.uint8)

    def evaluate(
        self,
        original: np.ndarray,
        upscaled: np.ndarray,
        eval_sigma: float = DEFAULT_EVAL_SIGMA,
        metric: str = METRIC_ENVELOPE,
    ) -> float:
        '''Process the pair and score the result. Lower is better.

        Args:
            original: Reference image, HWC RGB, uint8 or float32.
            upscaled: Target image, HWC RGB, uint8 or float32.
            eval_sigma: Fixed blur radius for the envelope metric.
            metric: Scoring metric, one of EVAL_METRICS.

        Returns:
            Scalar LAB-space MSE score.

        Raises:
            ColorMatchError: If the metric is unknown.
        '''
        result = self.process(original, upscaled)
        return self.score_result(
            result, original, eval_sigma=eval_sigma, metric=metric
        )

    @staticmethod
    def score_result(
        result: np.ndarray,
        original: np.ndarray,
        eval_sigma: float = DEFAULT_EVAL_SIGMA,
        metric: str = METRIC_ENVELOPE,
    ) -> float:
        '''Score a processed image against its reference. Lower is better.

        Args:
            result: Processed image, HWC RGB, uint8.
            original: Reference image, HWC RGB; resized to `result` if
                needed.
            eval_sigma: Fixed blur radius for the envelope metric.
            metric: Scoring metric, one of EVAL_METRICS.

        Returns:
            Scalar LAB-space MSE score.

        Raises:
            ColorMatchError: If the metric is unknown.
        '''
        if metric not in EVAL_METRICS:
            raise ColorMatchError(f"Unknown metric: {metric!r}")
        if original.shape[:2] != result.shape[:2]:
            original = ColorMatcher._resize_to_match(original, result)

        if metric == METRIC_ENVELOPE:
            result_blur = cv2.GaussianBlur(
                result.astype(np.float32), (0, 0), sigmaX=eval_sigma
            )
            orig_blur = cv2.GaussianBlur(
                original.astype(np.float32), (0, 0), sigmaX=eval_sigma
            )
            return float(ColorMatcher._lab_mse(result_blur, orig_blur))
        return float(ColorMatcher._lab_mse(result, original))

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _resize_to_match(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        height, width = target.shape[:2]
        resized = cv2.resize(
            source, (width, height), interpolation=cv2.INTER_LINEAR
        )
        logger.debug(
            "Resized reference %s -> %s", source.shape[:2], (height, width)
        )
        return resized

    def _frequency_split(self, image: np.ndarray) -> FrequencySplit:
        image_f = image.astype(np.float32)
        low = cv2.GaussianBlur(image_f, (0, 0), sigmaX=self._sigma)
        return FrequencySplit(low=low, high=image_f - low)

    @staticmethod
    def _reinhard_transfer(
        target: np.ndarray,
        source: np.ndarray,
    ) -> np.ndarray:
        '''Transfer per-channel LAB mean/std from source to target.'''
        target_lab = cv2.cvtColor(
            target.astype(np.uint8), cv2.COLOR_RGB2LAB
        ).astype(np.float32)
        source_lab = cv2.cvtColor(
            source.astype(np.uint8), cv2.COLOR_RGB2LAB
        ).astype(np.float32)

        result = target_lab.copy()
        for channel in range(3):
            t_mean = target_lab[:, :, channel].mean()
            t_std = target_lab[:, :, channel].std()
            s_mean = source_lab[:, :, channel].mean()
            s_std = source_lab[:, :, channel].std()
            scale = s_std / (t_std + _STD_EPSILON)
            result[:, :, channel] = (
                (target_lab[:, :, channel] - t_mean) * scale + s_mean
            )

        clipped = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(clipped, cv2.COLOR_LAB2RGB).astype(np.float32)

    @staticmethod
    def _lab_mse(a: np.ndarray, b: np.ndarray) -> float:
        a_lab = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(
            np.float32
        )
        b_lab = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(
            np.float32
        )
        return float(np.mean((a_lab - b_lab) ** 2))


def tune_sigma(
    original: np.ndarray,
    upscaled: np.ndarray,
    sigma_min: float = 5.0,
    sigma_max: float = 60.0,
    sigma_step: float = 5.0,
    eval_sigma: float = DEFAULT_EVAL_SIGMA,
    metric: str = METRIC_ENVELOPE,
    method: str = METHOD_REINHARD,
) -> TuneResult:
    '''Grid-search sigma to minimize the color envelope error.

    Args:
        original: Reference image, HWC RGB, uint8 or float32 [0, 255].
        upscaled: Target image, HWC RGB; resized to match if needed.
        sigma_min: Grid start (inclusive).
        sigma_max: Grid end (inclusive).
        sigma_step: Grid step, positive.
        eval_sigma: Fixed blur radius for the envelope metric.
        metric: Scoring metric, one of EVAL_METRICS.
        method: Color transfer method, one of MATCH_METHODS.

    Returns:
        Best candidate: sigma, score and the processed image.

    Raises:
        ColorMatchError: If the grid is empty or arguments are invalid.
    '''
    if sigma_step <= 0.0:
        raise ColorMatchError(f"sigma_step must be positive, got {sigma_step}")
    if sigma_max < sigma_min:
        raise ColorMatchError(
            f"sigma_max ({sigma_max}) must be >= sigma_min ({sigma_min})"
        )
    if metric not in EVAL_METRICS:
        raise ColorMatchError(f"Unknown metric: {metric!r}")

    best: TuneResult | None = None
    for candidate in np.arange(sigma_min, sigma_max + sigma_step, sigma_step):
        matcher = ColorMatcher(sigma=float(candidate), method=method)
        result = matcher.process(original, upscaled)
        score = ColorMatcher.score_result(
            result, original, eval_sigma=eval_sigma, metric=metric
        )
        logger.debug("tune_sigma: candidate=%.2f score=%.3f", candidate, score)
        if best is None or score < best.score:
            best = TuneResult(sigma=float(candidate), score=score, image=result)

    if best is None:
        raise ColorMatchError("Sigma grid produced no candidates")
    logger.info("tune_sigma: best sigma=%.2f score=%.3f", best.sigma, best.score)
    return best
