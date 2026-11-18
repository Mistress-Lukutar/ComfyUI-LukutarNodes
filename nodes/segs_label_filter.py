'''
File:   segs_label_filter.py
Brief:  ComfyUI node that assigns labels to SEGS segments and filters
        them by numeric index, previewing one image per segment
        (SEGSPreview-style, non-passing darkened).
Author: Mistress-Lukutar
Date:   2026-11-18
Version: v0.4.2
'''

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch

from ..core.segs_label_filter import (
    assign_seg_labels,
    build_segment_previews,
    parse_index_filter,
    parse_labels,
    split_segs,
)
from ..utils.images import tensor_to_frames

logger = logging.getLogger(__name__)

try:
    import folder_paths  # ComfyUI runtime only
except ImportError:  # pragma: no cover - exercised only outside runtime
    folder_paths = None  # type: ignore[assignment]

try:
    from PIL import Image  # Pillow from the ComfyUI runtime
except ImportError:  # pragma: no cover - exercised only outside runtime
    Image = None  # type: ignore[assignment]


#: Temp-file prefix for in-node previews.
PREVIEW_PREFIX = "tmp_segs_labels"


def _to_numpy(value: Any) -> np.ndarray:
    '''Convert a torch tensor or array-like SEGS field to a numpy array.'''
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_crop_image(value: Any) -> np.ndarray | None:
    '''Normalize a SEG ``cropped_image`` (tensor or array) to an HWC RGB
    float32 [0, 255] crop; None when unusable.'''
    if value is None:
        return None
    arr = _to_numpy(value).astype(np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    if arr.shape[-1] not in (1, 3, 4) and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))  # planar CHW
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.size and float(arr.max()) <= 1.5:
        arr = arr * 255.0  # IMAGE-style [0, 1] content
    return np.clip(arr, 0.0, 255.0)


def _as_mask(value: Any) -> np.ndarray | None:
    '''Normalize a SEG ``cropped_mask`` (tensor or array) to a float32
    (H, W) or (B, H, W) array in [0, 1]; None when unusable.'''
    if value is None:
        return None
    arr = _to_numpy(value).astype(np.float32)
    if arr.ndim not in (2, 3):
        return None
    return np.clip(arr, 0.0, 1.0)


def _save_temp_previews(frames: list[np.ndarray]) -> list[dict[str, str]]:
    '''Save rendered frames as temp PNGs for the frontend preview.

    Files are named ``tmp_segs_labels_*.png`` in the ComfyUI temp
    directory.

    Args:
        frames: List of HWC uint8 or float32 RGB frames to save.

    Returns:
        List of ``{"filename", "subfolder", "type"}`` ui references.

    Raises:
        RuntimeError: When called outside the ComfyUI runtime.
    '''
    if folder_paths is None or Image is None:
        raise RuntimeError("Previews require the ComfyUI runtime.")
    full_output_folder, filename, counter, subfolder, _ = (
        folder_paths.get_save_image_path(
            PREVIEW_PREFIX,
            folder_paths.get_temp_directory(),
            int(frames[0].shape[0]),
            int(frames[0].shape[1]),
        )
    )
    results: list[dict[str, str]] = []
    for frame in frames:
        arr = np.clip(frame, 0.0, 255.0).astype(np.uint8)
        pil_image = Image.fromarray(arr)
        file = f"{filename}_{counter:05}.png"
        pil_image.save(
            os.path.join(full_output_folder, file),
            compress_level=1,
        )
        results.append(
            {"filename": file, "subfolder": subfolder, "type": "temp"}
        )
        counter += 1
    return results


class SegsLabelFilterNode:
    '''Assign labels to Impact Pack SEGS segments and filter by index.

    Writes label strings onto the SEG namedtuples so they travel
    downstream.  Segments passing the numeric-index filter are placed
    in ``filtered_segs``; the rest go into ``remaining_segs``.  The
    preview shows one image per segment (SEGSPreview style): the SEG's
    ``cropped_image`` when the detector provided one, otherwise the
    input image cropped by the segment's crop region.  The non-masked
    part of every preview is transparent (alpha from the segment's
    cropped mask, floored at 0.25 — always on, like Impact Pack's SEGS
    Preview); previews of non-passing segments are darkened to 40%
    brightness (fixed factor 0.6), passing ones keep palette caption
    colors.  No bbox rectangle is drawn — each preview carries the
    segment label as a caption plate at its top-left corner and its
    1-based number as a plate at the bottom-right corner, with the
    caption font size set by ``font_size``.

    Numbering is 1-based from the full input list, matching what the
    user sees on the preview.

    OUTPUT_NODE = True: this node pushes its preview frames through the
    frontend context-menu channel without requiring downstream consumers.

    Args:
        image: The SEGS-detected image, used as preview background.
        segs: Impact Pack SEGS payload from a detector (e.g. SEGM).
        labels: Comma/newline-separated label per segment in order;
            empty entries are valid positional slots.
        filter: 1-based index filter like ``"1,2,3\" or \"1-5, 8\";
            empty string means all segments pass.
        dim: Darkness of non-passing bbox regions on the preview.

    Returns:
        A dict with ``ui.images`` (preview refs) and ``result``
        containing ``(filtered_segs, remaining_segs)``.

    Raises:
        ValueError: If the SEGS payload is malformed.
    '''

    CATEGORY = "Lukutar/Image"
    OUTPUT_NODE = True
    RETURN_TYPES = ("SEGS", "SEGS")
    RETURN_NAMES = ("filtered_segs", "remaining_segs")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Image the SEGS were detected on, used as the"
                            " preview background"
                        )
                    },
                ),
                "segs": (
                    "SEGS",
                    {
                        "tooltip": (
                            "Segments from an Impact Pack detector,"
                            " e.g. SEGM Detector (SEGS)"
                        )
                    },
                ),
                "labels": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Comma/newline-separated label per segment in"
                            " order, e.g. 'bg,body,body,bg,body,hand';"
                            " missing labels result in empty strings"
                        ),
                    },
                ),
                "filter": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "1-based index filter like '1,2,3,4' or '1-5, 8';"
                            " empty = all segments pass; numbers are the"
                            " indices shown on the preview"
                        ),
                    },
                ),
                "font_size": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.05,
                        "tooltip": "Caption font size (OpenCV font scale)",
                    },
                ),
            },
        }

    @staticmethod
    def _save_previews(frames: list[np.ndarray]) -> list[dict[str, str]]:
        '''Save preview frames as temp files; a preview failure must
        never kill execution.'''
        try:
            return _save_temp_previews(frames)
        except Exception:  # noqa: BLE001 - preview must not kill execution
            logger.exception("SegsLabelFilter: preview failed")
            return []

    def run(
        self,
        image: torch.Tensor,
        segs: Any,
        labels: str = "",
        filter: str = "",  # noqa: A002 - intentional parameter name
        font_size: float = 0.5,
    ) -> dict[str, Any]:
        '''Run the SEGS label filter node.

        Args:
            image: ComfyUI IMAGE batch (B, H, W, 3) [0, 1] RGB.
            segs: Impact Pack SEGS payload shaped ((h, w), [SEG, ...]).
            labels: Comma/newline-separated label string per segment.
            filter: 1-based numeric index filter string.
            font_size: Caption font size (OpenCV font scale).

        Returns:
            Dict with ``ui.images`` (one preview ref per segment) and
            ``result`` containing ``(filtered_segs, remaining_segs)``.

        Raises:
            ValueError: If the SEGS payload is malformed.
        '''
        # 1. Convert image to frames; crops without a SEG
        # cropped_image are taken from the first frame.
        frames = tensor_to_frames(image)
        frame = frames[0]
        frame_size = (int(frame.shape[0]), int(frame.shape[1]))

        # 2. Unpack SEGS payload.
        try:
            (segs_h, segs_w), seg_list = segs
            segs_h_int = int(segs_h)
            segs_w_int = int(segs_w)
            segments = list(seg_list)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Expected a SEGS ((h, w), [SEG, ...]) payload, got {segs!r}"
            ) from err

        # 3. Parse labels and filter.
        labels_list = parse_labels(labels, len(segments))
        selected = parse_index_filter(filter, len(segments))

        # 4. Assign labels to segments.
        labeled = assign_seg_labels(segments, labels_list)

        # 5. Split into filtered/remaining.
        filtered_segs, remaining_segs = split_segs(
            ((segs_h_int, segs_w_int), labeled), selected
        )

        # 6. Handle empty SEGS: plain-frame preview, pass-through.
        if not segments:
            logger.info("SegsLabelFilter: empty SEGS")
            return {
                "ui": {"images": self._save_previews(frames)},
                "result": (segs, segs),
            }

        # 7. Build one preview image per segment (SEGSPreview style).
        crops = [_as_crop_image(seg.cropped_image) for seg in labeled]
        masks = [_as_mask(seg.cropped_mask) for seg in labeled]
        previews = build_segment_previews(
            labeled,
            crops,
            frame,
            (segs_h_int, segs_w_int),
            frame_size,
            selected,
            masks=masks,
            font_size=font_size,
        )

        # 8. Save previews (never kill execution on failure).
        ui_images = self._save_previews(previews)

        # 9. Log summary.
        logger.info(
            "SegsLabelFilter: filter=%r, %d passing, %d remaining,"
            " %d preview(s)",
            filter,
            len(filtered_segs[1]),
            len(remaining_segs[1]),
            len(ui_images),
        )

        return {"ui": {"images": ui_images}, "result": (filtered_segs, remaining_segs)}
