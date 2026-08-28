'''
File:   smoke_test_comfyui_load.py
Brief:  Simulates ComfyUI custom node loading without launching the server.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.5.3

Run with ComfyUI's own python (no server launch required):

    C:/Ai/ComfyUI_windows_portable/python_embeded/python.exe \
        tests/smoke_test_comfyui_load.py
'''

from __future__ import annotations

import collections
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

PACK_ROOT = Path(__file__).resolve().parent.parent


def load_pack() -> object:
    '''Import the pack exactly the way ComfyUI's load_custom_node does.'''
    module_name = "custom_nodes.ComfyUI-LukutarNodes"
    spec = importlib.util.spec_from_file_location(
        module_name,
        PACK_ROOT / "__init__.py",
        submodule_search_locations=[str(PACK_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create import spec for the pack")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_pair() -> tuple[torch.Tensor, torch.Tensor]:
    '''Build a drifted image and its color reference as IMAGE tensors.'''
    height, width = 72, 96
    xs = np.linspace(0, 255, width, dtype=np.float32)
    ys = np.linspace(0, 255, height, dtype=np.float32)
    grid = xs[np.newaxis, :] + ys[:, np.newaxis]

    reference = np.stack(
        [
            np.clip(grid, 0, 255),
            np.clip(grid * 0.5 + 60.0, 0, 255),
            np.clip(255.0 - grid * 0.3, 0, 255),
        ],
        axis=-1,
    )
    drifted = np.stack(
        [
            np.clip(grid * 0.6, 0, 255),
            np.clip(grid * 0.5 + 60.0, 0, 255),
            np.clip(255.0 - grid * 0.15 + 40.0, 0, 255),
        ],
        axis=-1,
    )

    def to_tensor(frame: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((frame / 255.0).astype(np.float32))[None]

    return to_tensor(drifted), to_tensor(reference)


def _check_image_tensor(tensor: torch.Tensor, expected: torch.Tensor) -> None:
    assert tensor.dtype == torch.float32, f"bad dtype: {tensor.dtype}"
    assert tensor.shape == expected.shape, f"bad shape: {tensor.shape}"
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0


def _fake_segs(height: int = 72, width: int = 96) -> tuple:
    '''Build a SEGS payload shaped like SEGM Detector (SEGS) output.'''
    seg_type = collections.namedtuple(
        "SEG",
        [
            "cropped_image",
            "cropped_mask",
            "confidence",
            "crop_region",
            "bbox",
            "label",
            "control_net_wrapper",
        ],
    )
    mask = np.zeros((48, 48), dtype=np.float32)
    mask[8:40, 8:40] = 1.0
    seg = seg_type(
        cropped_image=None,
        cropped_mask=mask,
        confidence=np.array([0.9124], dtype=np.float32),
        crop_region=[8, 8, 56, 56],
        bbox=[16.0, 16.0, 48.0, 48.0],
        label="face",
        control_net_wrapper=None,
    )
    return (height, width), [seg]


def main() -> None:
    pack = load_pack()
    mappings = getattr(pack, "NODE_CLASS_MAPPINGS", None)
    assert mappings and "ColorMatch" in mappings, "ColorMatch not registered"
    assert "ColorMatch" in pack.NODE_DISPLAY_NAME_MAPPINGS  # type: ignore[attr-defined]

    node = mappings["ColorMatch"]()
    image, reference = _synthetic_pair()

    # Manual mode.
    result, used_sigma = node.match_colors(
        image, reference, auto_tune=False, sigma=15.0, method="reinhard"
    )
    _check_image_tensor(result, image)
    assert used_sigma == 15.0

    # Auto-tune mode.
    result_tuned, tuned_sigma = node.match_colors(
        image,
        reference,
        auto_tune=True,
        sigma=15.0,
        method="reinhard",
        sigma_min=4.0,
        sigma_max=24.0,
        sigma_step=4.0,
    )
    _check_image_tensor(result_tuned, image)
    assert 4.0 <= tuned_sigma <= 24.0, f"tuned sigma out of grid: {tuned_sigma}"

    # Batch broadcast: several drifted frames, single reference.
    batch = image.repeat(3, 1, 1, 1)
    result_batch, _ = node.match_colors(
        batch, reference, auto_tune=False, sigma=15.0, method="reinhard"
    )
    assert result_batch.shape[0] == 3, "single reference must broadcast"

    # Batch mismatch must fail loudly.
    try:
        node.match_colors(
            image.repeat(2, 1, 1, 1),
            reference.repeat(3, 1, 1, 1),
            auto_tune=False,
            sigma=15.0,
            method="reinhard",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("batch mismatch must raise ValueError")

    # SEGS overlay node.
    assert "SegsOverlay" in mappings, "SegsOverlay not registered"
    assert "SegsOverlay" in pack.NODE_DISPLAY_NAME_MAPPINGS  # type: ignore[attr-defined]

    overlay_node = mappings["SegsOverlay"]()
    segs = _fake_segs()  # size matches the 72x96 synthetic frame
    overlay_result, segs_out = overlay_node.overlay(image, segs)
    _check_image_tensor(overlay_result, image)
    assert segs_out is segs, "SEGS must pass through unchanged"
    assert not torch.allclose(overlay_result, image), "overlay must draw"

    # Size mismatch: SEGS recorded at double resolution is rescaled.
    big_segs = _fake_segs(height=144, width=192)
    scaled_result, _ = overlay_node.overlay(image, big_segs)
    _check_image_tensor(scaled_result, image)
    assert not torch.allclose(scaled_result, image), "scaled overlay must draw"

    # Empty SEGS leaves the image untouched.
    empty_result, _ = overlay_node.overlay(image, ((0, 0), []))
    _check_image_tensor(empty_result, image)
    assert torch.allclose(empty_result, image), "empty SEGS must not draw"

    # Prompt annotation nodes.
    assert "PromptAnnotate" in mappings, "PromptAnnotate not registered"
    assert "AnnotationsWildcard" in mappings
    assert "AnnotationSegment" in mappings
    assert "AnnotationLabels" in mappings
    assert "AnnotationSegmentEdit" in mappings
    assert "PromptAnnotate" in pack.NODE_DISPLAY_NAME_MAPPINGS  # type: ignore[attr-defined]

    annotate_node = mappings["PromptAnnotate"]()
    annotations, clean_prompt = annotate_node.annotate(
        "masterpiece, |body:1girl, thin|, |face:blue eyes, smirk|, "
        "|body,hair:red hair|, |body:stands|, |background:outdoors, park|",
    )
    assert clean_prompt == (
        "masterpiece, 1girl, thin, blue eyes, smirk, red hair, stands, "
        "outdoors, park"
    ), f"unexpected clean prompt: {clean_prompt!r}"

    (wildcard,) = mappings["AnnotationsWildcard"]().convert(annotations)
    assert wildcard.startswith("[LAB]"), "wildcard must start with [LAB]"
    assert "[ALL] masterpiece," in wildcard
    assert "[body] 1girl, thin, red hair, stands" in wildcard
    assert "[hair] red hair" in wildcard

    (face_text,) = mappings["AnnotationSegment"]().segment(
        annotations, "face", include_common=True
    )
    assert face_text == "masterpiece, blue eyes, smirk"
    try:
        mappings["AnnotationSegment"]().segment(annotations, "paws")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown label must raise ValueError")

    (labels_str,) = mappings["AnnotationLabels"]().collect(annotations)
    assert labels_str == "body, face, hair, background", (
        f"unexpected label list: {labels_str!r}"
    )
    (labels_all,) = mappings["AnnotationLabels"]().collect(
        annotations, include_common=True
    )
    assert labels_all == "all, body, face, hair, background"

    # Pass-through edit: append to a label, then strip it back out.
    edit_node = mappings["AnnotationSegmentEdit"]()
    (edited,) = edit_node.edit(annotations, "face", "append", "smile")
    (edited_face,) = mappings["AnnotationSegment"]().segment(
        edited, "face", include_common=False
    )
    assert edited_face == "blue eyes, smirk, smile", (
        f"unexpected edited face text: {edited_face!r}"
    )
    (removed,) = edit_node.edit(
        annotations, "face", "remove", "blue eyes, smirk"
    )
    (labels_after,) = mappings["AnnotationLabels"]().collect(removed)
    assert labels_after == "body, hair, background", (
        f"face should be gone after removing its text: {labels_after!r}"
    )
    try:
        edit_node.edit(annotations, "paws", "prepend", "fur")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown label must raise ValueError")

    # new / delete modes with a comma-separated label list.
    (grown,) = edit_node.edit(
        annotations, "hands, weapon", "new", "delicate fingers"
    )
    (labels_grown,) = mappings["AnnotationLabels"]().collect(grown)
    assert labels_grown == "body, face, hair, background, hands, weapon", (
        f"new mode must append the labels: {labels_grown!r}"
    )
    (trimmed,) = edit_node.edit(grown, "face, hands, weapon", "delete", "")
    (labels_trimmed,) = mappings["AnnotationLabels"]().collect(trimmed)
    assert labels_trimmed == "body, hair, background", (
        f"delete mode must remove the labels: {labels_trimmed!r}"
    )
    try:
        edit_node.edit(annotations, "face", "new", "freckles")
    except ValueError:
        pass
    else:
        raise AssertionError("new mode on an existing label must raise")

    print("SMOKE TEST PASSED: nodes load and behave as expected")


if __name__ == "__main__":
    main()
