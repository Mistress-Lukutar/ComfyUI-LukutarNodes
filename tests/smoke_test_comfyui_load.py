'''
File:   smoke_test_comfyui_load.py
Brief:  Simulates ComfyUI custom node loading without launching the server.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.7.3

Run with ComfyUI's own python (no server launch required):

    C:/Ai/ComfyUI_windows_portable/python_embeded/python.exe \
        tests/smoke_test_comfyui_load.py
'''

from __future__ import annotations

import collections
import contextlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PACK_ROOT = Path(__file__).resolve().parent.parent
#: Portable-install ComfyUI root, used by the prompt validation e2e part.
COMFYUI_ROOT = Path("C:/Ai/ComfyUI_windows_portable/ComfyUI")


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


def _validate_variables_prompt(mappings: dict[str, type]) -> None:
    '''Validate a prompt wired the way the web extension serializes it.

    The extension keeps real links EmptyImage -> SetVariable ->
    GetVariable -> PreviewImage; running that prompt through ComfyUI's
    own validator proves the ``*`` links and the node contract hold end
    to end (execution order and caching are the server's own).
    '''
    if not COMFYUI_ROOT.exists():
        print("NOTE: ComfyUI root not found, skipping prompt validation e2e")
        return
    sys.path.insert(0, str(COMFYUI_ROOT))
    try:
        import asyncio

        import execution

        import nodes as comfy_nodes

        for key in ("SetVariable", "GetVariable"):
            comfy_nodes.NODE_CLASS_MAPPINGS[key] = mappings[key]

        prompt = {
            "1": {
                "class_type": "EmptyImage",
                "inputs": {
                    "width": 8,
                    "height": 8,
                    "batch_size": 1,
                    "color": 0,
                },
            },
            "2": {
                "class_type": "SetVariable",
                "inputs": {"var_name": "img_t2i", "value": ["1", 0]},
            },
            "3": {
                "class_type": "GetVariable",
                "inputs": {"var_name": "img_t2i", "value": ["2", 0]},
            },
            "4": {
                "class_type": "PreviewImage",
                "inputs": {"images": ["3", 0]},
            },
        }
        ok, error, outputs, _node_errors = asyncio.run(
            execution.validate_prompt("lukutar-smoke", prompt, None)
        )
        assert ok, f"variable prompt failed validation: {error}"
        assert outputs == ["4"], f"unexpected outputs: {outputs}"
    finally:
        sys.path.remove(str(COMFYUI_ROOT))


def _validate_bundle_prompt(mappings: dict[str, type]) -> None:
    '''Validate the bundle bus prompt the web extension serializes.

    The slot links (image, latent) are NOT declared in the bundle
    nodes' INPUT_TYPES — ComfyUI's validator only inspects declared
    inputs while the executor resolves every linked prompt input, which
    is the contract that lets the web extension grow sockets
    client-side. Running the shaped prompt through the real validator
    proves that contract holds end to end, heterogeneous slots
    included.
    '''
    if not COMFYUI_ROOT.exists():
        print("NOTE: ComfyUI root not found, skipping prompt validation e2e")
        return
    sys.path.insert(0, str(COMFYUI_ROOT))
    try:
        import asyncio

        import execution

        import nodes as comfy_nodes

        for key in ("SetBundle", "GetBundle"):
            comfy_nodes.NODE_CLASS_MAPPINGS[key] = mappings[key]

        spec = "image:IMAGE,latent:LATENT"
        prompt = {
            "1": {
                "class_type": "EmptyImage",
                "inputs": {
                    "width": 8,
                    "height": 8,
                    "batch_size": 1,
                    "color": 0,
                },
            },
            "2": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 8, "height": 8, "batch_size": 1},
            },
            "3": {
                "class_type": "SetBundle",
                "inputs": {
                    "bundle_name": "sdxl",
                    "slots": spec,
                    "image": ["1", 0],
                    "latent": ["2", 0],
                },
            },
            "4": {
                "class_type": "GetBundle",
                "inputs": {
                    "bundle_name": "sdxl",
                    "slots": spec,
                    "image": ["3", 0],
                    "latent": ["3", 1],
                },
            },
            "5": {
                "class_type": "PreviewImage",
                "inputs": {"images": ["4", 0]},
            },
        }
        ok, error, outputs, _node_errors = asyncio.run(
            execution.validate_prompt("lukutar-smoke", prompt, None)
        )
        assert ok, f"bundle prompt failed validation: {error}"
        assert outputs == ["5"], f"unexpected outputs: {outputs}"
    finally:
        sys.path.remove(str(COMFYUI_ROOT))


def _validate_metadata_prompt(mappings: dict[str, type]) -> None:
    '''Validate a prompt wired the way the metadata web extension serializes it.

    Builds: MetadataBlocks → MergeMetadata (meta_1/meta_2 links) →
    LukutarSaveImage (metadata link to Merge).  The meta_1/meta_2 inputs
    are NOT declared in MergeMetadataNode.INPUT_TYPES — they arrive as
    undeclared kwargs, same pattern as Set/Get Bundle slots.
    '''
    if not COMFYUI_ROOT.exists():
        print("NOTE: ComfyUI root not found, skipping metadata prompt validation e2e")
        return
    sys.path.insert(0, str(COMFYUI_ROOT))
    try:
        import asyncio

        import execution

        import nodes as comfy_nodes

        for key in ("MetadataBlocks", "MergeMetadata", "LukutarSaveImage"):
            comfy_nodes.NODE_CLASS_MAPPINGS[key] = mappings[key]

        prompt = {
            "0": {
                "class_type": "EmptyImage",
                "inputs": {
                    "width": 8,
                    "height": 8,
                    "batch_size": 1,
                    "color": 0,
                },
            },
            "1": {
                "class_type": "MetadataBlocks",
                "inputs": {
                    "num_blocks": 2,
                    "name_1": "env",
                    "value_1": "production",
                    "name_2": "model",
                    "value_2": "sd-xl",
                },
            },
            "2": {
                "class_type": "MetadataBlocks",
                "inputs": {
                    "num_blocks": 2,
                    "name_1": "model",
                    "value_1": "sdxl-v2",
                    "name_2": "rev",
                    "value_2": "beta",
                },
            },
            "3": {
                "class_type": "MergeMetadata",
                "inputs": {
                    "meta_1": ["1", 0],
                    "meta_2": ["2", 0],
                },
            },
            "4": {
                "class_type": "LukutarSaveImage",
                "inputs": {
                    "images": ["0", 0],
                    "path": "output",
                    "filename": "metadata_smoke",
                    "format": "PNG",
                    "quality": 95,
                    "metadata": ["3", 0],
                },
            },
        }
        ok, error, outputs, _node_errors = asyncio.run(
            execution.validate_prompt("lukutar-metadata-smoke", prompt, None)
        )
        assert ok, f"metadata prompt failed validation: {error}"
        assert outputs == ["4"], f"unexpected outputs: {outputs}"
    finally:
        sys.path.remove(str(COMFYUI_ROOT))


def _validate_a1111_prompt(mappings: dict[str, type]) -> None:
    '''Validate the A1111 metadata prompt chain end to end.

    Builds: A1111Sampling (meta_1) + A1111HiresFix (meta_2) into
    A1111Parameters, whose metadata output feeds MergeMetadata alongside
    MetadataBlocks, then LukutarSaveImage.  Proves the LUKUTAR_METADATA
    socket chains through undeclared kwargs and the ``*`` return bus.
    '''
    if not COMFYUI_ROOT.exists():
        print("NOTE: ComfyUI root not found, skipping A1111 prompt validation e2e")
        return
    sys.path.insert(0, str(COMFYUI_ROOT))
    try:
        import asyncio

        import execution

        import nodes as comfy_nodes_a1111

        for key in (
            "A1111Sampling",
            "A1111HiresFix",
            "A1111Parameters",
            "MetadataBlocks",
            "MergeMetadata",
            "LukutarSaveImage",
        ):
            if key in mappings:
                comfy_nodes_a1111.NODE_CLASS_MAPPINGS[key] = mappings[key]

        prompt = {
            "1": {
                "class_type": "EmptyImage",
                "inputs": {
                    "width": 8,
                    "height": 8,
                    "batch_size": 1,
                    "color": 0,
                },
            },
            "2": {
                "class_type": "A1111Sampling",
                "inputs": {
                    "steps": 30,
                    "sampler": "DPM++ 2M Karras",
                    "cfg": 7.0,
                    "seed": 3928472037,
                    "width": 832,
                    "height": 1216,
                    "clip_skip": 2,
                },
            },
            "3": {
                "class_type": "A1111HiresFix",
                "inputs": {
                    "upscaler": "4x-UltraSharp",
                    "upscale": 2.0,
                    "hires_steps": 15,
                    "denoising": 0.45,
                },
            },
            "4": {
                "class_type": "A1111Parameters",
                "inputs": {
                    "positive": "masterpiece, best quality, 1girl",
                    "negative": "worst quality",
                    "version": "forge-0.9.1",
                    "meta_1": ["2", 0],
                    "meta_2": ["3", 0],
                },
            },
            "5": {
                "class_type": "MetadataBlocks",
                "inputs": {
                    "num_blocks": 1,
                    "name_1": "env",
                    "value_1": "smoke-test",
                },
            },
            "6": {
                "class_type": "MergeMetadata",
                "inputs": {
                    "meta_1": ["4", 0],
                    "meta_2": ["5", 0],
                },
            },
            "7": {
                "class_type": "LukutarSaveImage",
                "inputs": {
                    "images": ["1", 0],
                    "path": "output",
                    "filename": "a1111_smoke",
                    "format": "PNG",
                    "quality": 95,
                    "metadata": ["6", 0],
                },
            },
        }
        ok, error, outputs, _node_errors = asyncio.run(
            execution.validate_prompt("lukutar-a1111-smoke", prompt, None)
        )
        assert ok, f"A1111 metadata prompt failed validation: {error}"
        assert outputs == ["7"], f"unexpected A1111 outputs: {outputs}"
    finally:
        sys.path.remove(str(COMFYUI_ROOT))


def _validate_jxl_prompt(mappings: dict[str, type]) -> None:
    '''Validate a prompt wired to LukutarSaveImageJxl.

    Builds: EmptyImage → LukutarSaveImageJxl with all required widget
    values.  Runs regardless of whether pillow-jxl-plugin is installed
    — only the node's INPUT_TYPES contract is being checked.
    '''
    if not COMFYUI_ROOT.exists():
        print("NOTE: ComfyUI root not found, skipping JXL prompt validation e2e")
        return
    sys.path.insert(0, str(COMFYUI_ROOT))
    try:
        import asyncio

        import execution

        import nodes as comfy_nodes3

        for key in ("LukutarSaveImageJxl", "SetVariable", "GetVariable"):
            if key in mappings:
                comfy_nodes3.NODE_CLASS_MAPPINGS[key] = mappings[key]

        prompt = {
            "1": {
                "class_type": "EmptyImage",
                "inputs": {
                    "width": 8,
                    "height": 8,
                    "batch_size": 1,
                    "color": 0,
                },
            },
            "2": {
                "class_type": "LukutarSaveImageJxl",
                "inputs": {
                    "images": ["1", 0],
                    "path": "output",
                    "filename": "jxl_smoke",
                    "lossless": False,
                    "quality": 90,
                    "effort": 7,
                    "decoding_speed": 0,
                    "use_container": False,
                    "use_original_profile": False,
                    "num_threads": -1,
                    "compress_metadata": False,
                },
            },
        }
        ok, error, outputs, _node_errors = asyncio.run(
            execution.validate_prompt("lukutar-jxl-smoke", prompt, None)
        )
        assert ok, f"JXL prompt failed validation: {error}"
        assert outputs == ["2"], f"unexpected outputs: {outputs}"
    finally:
        sys.path.remove(str(COMFYUI_ROOT))


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

    # VAE-decode style input: a BHWC permuted view is non-contiguous and
    # used to break cv2 drawing on the overlay canvas.
    bchw = image[0].permute(2, 0, 1).contiguous()[None]
    permuted = bchw.permute(0, 2, 3, 1)
    assert not permuted.is_contiguous(), "test tensor must be a permuted view"
    perm_overlay, _ = overlay_node.overlay(permuted, _fake_segs())
    _check_image_tensor(perm_overlay, image)
    assert not torch.allclose(perm_overlay, permuted), (
        "overlay must draw on permuted views"
    )
    perm_matched, _ = node.match_colors(
        permuted, reference, auto_tune=False, sigma=15.0, method="reinhard"
    )
    _check_image_tensor(perm_matched, image)

    # SEGS set crop size node.
    assert "SegsSetCropSize" in mappings, "SegsSetCropSize not registered"
    assert "SegsSetCropSize" in pack.NODE_DISPLAY_NAME_MAPPINGS  # type: ignore[attr-defined]

    fit_node = mappings["SegsSetCropSize"]()
    segs = _fake_segs()  # canvas 96x72, bbox (16,16,48,48), crop 48x48
    (resized,) = fit_node.fit(segs, 64, 64)
    assert resized[0] == (72, 96), "canvas size must be preserved"
    seg = resized[1][0]
    assert seg.crop_region == [0, 0, 64, 64], "64x64 centered on the bbox"
    assert seg.bbox == [16.0, 16.0, 48.0, 48.0], "bbox must be untouched"
    assert seg.cropped_mask.shape == (64, 64)
    assert seg.cropped_mask.dtype == np.float32
    assert seg.cropped_mask[20, 20] == 1.0, "mask must keep image coords"
    assert seg.cropped_mask[4, 4] == 0.0, "added context must be zero"

    # Target below the bbox size: grows to contain it (rounded to /8).
    (grown,) = fit_node.fit(_fake_segs(), 24, 24)
    assert grown[1][0].crop_region == [16, 16, 48, 48]

    # Target above the canvas: clamped to the canvas size.
    (clamped,) = fit_node.fit(_fake_segs(), 512, 512)
    assert clamped[1][0].crop_region == [0, 0, 96, 72]

    # aspect mode: 32x32 bbox scaled so the longer side is 64.
    (aspect,) = fit_node.fit(_fake_segs(), 64, 64, mode="aspect")
    region = aspect[1][0].crop_region
    assert region[2] - region[0] == 64 and region[3] - region[1] == 64

    # Empty SEGS passes through untouched.
    empty_segs = ((0, 0), [])
    (same_segs,) = fit_node.fit(empty_segs, 64, 64)
    assert same_segs is empty_segs

    # Prompt annotation nodes.
    assert "PromptAnnotate" in mappings, "PromptAnnotate not registered"
    assert "AnnotationsWildcard" in mappings
    assert "AnnotationSegment" in mappings
    assert "AnnotationLabels" in mappings
    assert "AnnotationSegmentEdit" in mappings
    assert "AnnotationsToString" in mappings
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

    # AnnotationsToString — clean and raw modes.
    (clean_str,) = mappings["AnnotationsToString"]().convert(annotations)
    assert clean_str == (
        "masterpiece, 1girl, thin, blue eyes, smirk, red hair, stands, "
        "outdoors, park"
    ), f"unexpected clean string: {clean_str!r}"
    (raw_str,) = mappings["AnnotationsToString"]().convert(
        annotations, "raw"
    )
    assert raw_str.startswith("masterpiece, |body:"), (
        f"unexpected raw string: {raw_str!r}"
    )

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

    # Variable nodes: pass-through Set / Get pair with name validation.
    assert "SetVariable" in mappings, "SetVariable not registered"
    assert "GetVariable" in mappings, "GetVariable not registered"
    assert "SetVariable" in pack.NODE_DISPLAY_NAME_MAPPINGS  # type: ignore[attr-defined]
    assert "GetVariable" in pack.NODE_DISPLAY_NAME_MAPPINGS  # type: ignore[attr-defined]

    marker = object()
    (passed_through,) = mappings["SetVariable"]().set_value("img_t2i", marker)
    assert passed_through is marker, "Set Variable must pass the value through"
    (fetched,) = mappings["GetVariable"]().get_value("img_t2i", marker)
    assert fetched is marker, "Get Variable must emit the wired value"
    try:
        mappings["GetVariable"]().get_value("img_t2i")
    except RuntimeError:
        pass
    else:
        raise AssertionError("unwired GetVariable must raise RuntimeError")
    try:
        mappings["SetVariable"]().set_value("   ", marker)
    except ValueError:
        pass
    else:
        raise AssertionError("blank variable name must raise ValueError")

    _validate_variables_prompt(mappings)

    # Bundle nodes: multi-slot named bus. Direct contract checks, then
    # the web-extension-shaped prompt through ComfyUI's validator.
    assert "SetBundle" in mappings, "SetBundle not registered"
    assert "GetBundle" in mappings, "GetBundle not registered"
    assert "SetBundle" in pack.NODE_DISPLAY_NAME_MAPPINGS  # type: ignore[attr-defined]
    assert "GetBundle" in pack.NODE_DISPLAY_NAME_MAPPINGS  # type: ignore[attr-defined]

    image_marker, seed_marker = object(), object()
    bundle_set = mappings["SetBundle"]()
    slots_out = bundle_set.set_bundle(
        "sdxl", "image:IMAGE, seed:INT", seed=seed_marker, image=image_marker
    )
    assert slots_out[:2] == (image_marker, seed_marker), (
        "slots must pass through in spec order"
    )
    assert all(value is None for value in slots_out[2:]), (
        "padding beyond the spec must be None"
    )
    try:
        bundle_set.set_bundle("sdxl", "image:IMAGE,seed:INT", image=image_marker)
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing declared slot must raise RuntimeError")
    try:
        mappings["GetBundle"]().get_bundle("sdxl")
    except RuntimeError:
        pass
    else:
        raise AssertionError("valueless GetBundle must raise RuntimeError")
    try:
        bundle_set.set_bundle("sdxl", "image:IMAGE,image:VAE", image=image_marker)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate slot names must raise ValueError")

    _validate_bundle_prompt(mappings)

    # Metadata nodes: undeclared-kwargs slot inputs (meta_1/meta_2) and the
    # LUKUTAR_METADATA links pass ComfyUI's validator.
    _validate_metadata_prompt(mappings)

    # A1111 metadata chain: Sampling + HiresFix → Parameters → Merge → Save.
    _validate_a1111_prompt(mappings)

    # Group Switch node: validate + order the states payload the way the
    # web extension serializes it, then prove the prompt contract end to
    # end (GROUP_STATES output feeds the wildcard Set/Get value input).
    assert "GroupSwitch" in mappings, "GroupSwitch not registered"
    assert "GroupSwitch" in pack.NODE_DISPLAY_NAME_MAPPINGS  # type: ignore[attr-defined]

    switch_node = mappings["GroupSwitch"]()
    (states,) = switch_node.switch(
        "bypass", True, "b,c", '{"b": true, "a": false, "c": true}'
    )
    assert list(states) == ["b", "c", "a"], f"unexpected order: {list(states)}"
    assert states == {"b": True, "c": True, "a": False}
    (empty_states,) = switch_node.switch("disable", False, "", "{}")
    assert empty_states == {}
    try:
        switch_node.switch("mute", True, "", "{}")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown disable method must raise ValueError")
    try:
        switch_node.switch("bypass", True, "", "{broken")
    except ValueError:
        pass
    else:
        raise AssertionError("broken group_states JSON must raise ValueError")

    if COMFYUI_ROOT.exists():
        sys.path.insert(0, str(COMFYUI_ROOT))
        try:
            import asyncio

            import execution as comfy_execution

            import nodes as comfy_nodes2

            for key in ("GroupSwitch", "SetVariable", "GetVariable"):
                comfy_nodes2.NODE_CLASS_MAPPINGS[key] = mappings[key]

            prompt = {
                "1": {
                    "class_type": "GroupSwitch",
                    "inputs": {
                        "disable_method": "bypass",
                        "change_group_color": True,
                        "sort_order": "b,c",
                        "group_states": '{"b": true, "a": false}',
                    },
                },
                "2": {
                    "class_type": "SetVariable",
                    "inputs": {"var_name": "grp", "value": ["1", 0]},
                },
                "3": {
                    "class_type": "GetVariable",
                    "inputs": {"var_name": "grp", "value": ["2", 0]},
                },
            }
            result = asyncio.run(
                comfy_execution.validate_inputs("lukutar-smoke", prompt, "3", {})
            )
            assert result[0] is True, (
                f"group switch prompt failed validation: {result[1]}"
            )
        finally:
            sys.path.remove(str(COMFYUI_ROOT))

    # Seed History node. Contract regression: the ComfyUI frontend adds
    # control_after_generate to every seed widget client-side with
    # serialize:false and never sends it — declaring it server-side made
    # prompts fail validation with "Required input is missing". The
    # output must be plain INT so it wires into sampler seed inputs.
    assert "SeedHistory" in mappings, "SeedHistory not registered"
    assert "SeedHistory" in pack.NODE_DISPLAY_NAME_MAPPINGS  # type: ignore[attr-defined]

    seed_history_node = mappings["SeedHistory"]()
    seed_required = seed_history_node.INPUT_TYPES()["required"]
    assert set(seed_required) == {"seed", "history_size"}, (
        f"unexpected SeedHistory inputs: {sorted(seed_required)}"
    )
    assert seed_history_node.RETURN_TYPES == ("INT",)
    assert seed_history_node.generate(seed=42, history_size=5) == (42,)

    if COMFYUI_ROOT.exists():
        sys.path.insert(0, str(COMFYUI_ROOT))
        try:
            import asyncio

            import execution

            import nodes as comfy_nodes3

            for key in ("SeedHistory", "SetVariable", "GetVariable"):
                comfy_nodes3.NODE_CLASS_MAPPINGS[key] = mappings[key]

            # Shaped exactly like the prompt the frontend builds for the
            # node: declared widget inputs only, no control_after_generate.
            prompt = {
                "1": {
                    "class_type": "SeedHistory",
                    "inputs": {"seed": 42, "history_size": 5},
                },
                "2": {
                    "class_type": "SetVariable",
                    "inputs": {"var_name": "sd_seed", "value": ["1", 0]},
                },
                "3": {
                    "class_type": "GetVariable",
                    "inputs": {"var_name": "sd_seed", "value": ["2", 0]},
                },
                "4": {
                    "class_type": "PreviewImage",
                    "inputs": {"images": ["3", 0]},
                },
            }
            ok, error, outputs, _node_errors = asyncio.run(
                execution.validate_prompt("lukutar-smoke", prompt, None)
            )
            assert ok, f"seed history prompt failed validation: {error}"
            assert outputs == ["4"], f"unexpected outputs: {outputs}"
        finally:
            sys.path.remove(str(COMFYUI_ROOT))

    # ------------------------------------------------------------------ #
    # Direct-contract: A1111Sampling, A1111HiresFix, A1111Parameters      #
    # ------------------------------------------------------------------ #

    # A1111Sampling — exact metadata dict contract.
    sampling_node = mappings["A1111Sampling"]()
    (sampling_meta,) = sampling_node.build(
        steps=30,
        sampler="DPM++ 2M Karras",
        cfg=7.0,
        seed=3928472037,
        width=832,
        height=1216,
        clip_skip=2,
    )
    assert sampling_meta == {
        "Steps": "30",
        "Sampler": "DPM++ 2M Karras",
        "CFG scale": "7",
        "Seed": "3928472037",
        "Size": "832x1216",
        "Clip skip": "2",
    }, f"A1111Sampling metadata mismatch: {sampling_meta!r}"

    # A1111HiresFix — exact metadata dict contract.
    hires_node = mappings["A1111HiresFix"]()
    (hires_meta,) = hires_node.build(
        upscaler="4x-UltraSharp",
        upscale=2.0,
        hires_steps=15,
        denoising=0.45,
    )
    assert hires_meta == {
        "Hires upscale": "2.0",
        "Hires upscaler": "4x-UltraSharp",
        "Hires steps": "15",
        "Denoising strength": "0.45",
    }, f"A1111HiresFix metadata mismatch: {hires_meta!r}"

    # A1111Parameters — combined build with sampling + hires metadata blocks.
    params_node = mappings["A1111Parameters"]()
    (params_meta, params_text) = params_node.build(
        positive="masterpiece, best quality, 1girl",
        negative="worst quality",
        version="forge-0.9.1",
        meta_1=sampling_meta,
        meta_2=hires_meta,
    )
    expected_text = (
        "masterpiece, best quality, 1girl\n"
        "Negative prompt: worst quality\n"
        "Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7, "
        "Seed: 3928472037, Size: 832x1216, Clip skip: 2, "
        "Hires upscale: 2.0, Hires upscaler: 4x-UltraSharp, "
        "Hires steps: 15, Denoising strength: 0.45, "
        "Version: forge-0.9.1"
    )
    assert params_text == expected_text, (
        f"A1111Parameters text mismatch:\n"
        f"  expected: {expected_text!r}\n"
        f"  got:      {params_text!r}"
    )
    assert params_meta == {"parameters": expected_text}, (
        f"A1111Parameters metadata mismatch: {params_meta!r}"
    )

    # A1111Parameters — empty inputs return empty contract.
    (empty_meta, empty_text) = params_node.build(
        positive="", negative="", version=""
    )
    assert empty_meta == {}, f"empty parameters metadata: {empty_meta!r}"
    assert empty_text == "", f"empty parameters text: {empty_text!r}"

    # ------------------------------------------------------------------ #
    # Direct-contract: A1111ModelInfo + A1111LoraHashes (guarded)         #
    # ------------------------------------------------------------------ #

    # Import pack modules BEFORE inserting COMFYUI_ROOT on sys.path,
    # otherwise ``import nodes.a1111`` resolves to ComfyUI's nodes pkg.
    _pack_path = str(PACK_ROOT)
    if _pack_path not in sys.path:
        sys.path.insert(0, _pack_path)

    import importlib  # noqa: E402

    _a1111_mod = importlib.import_module(  # noqa: E402
        "custom_nodes.ComfyUI-LukutarNodes.nodes.a1111"
    )
    _core_mod = importlib.import_module(  # noqa: E402
        "custom_nodes.ComfyUI-LukutarNodes.core.a1111_parameters"
    )

    ModelHashCache = _core_mod.ModelHashCache  # noqa: F811
    compute_model_hash = _core_mod.compute_model_hash  # noqa: F811

    if COMFYUI_ROOT.exists():
        sys.path.insert(0, str(COMFYUI_ROOT))
        try:
            import folder_paths as comfy_fp_a1111

            # Patch get_model_hash_cache to use a temp-file cache.
            _tmp_cache_path = tempfile.mktemp(suffix=".json")
            _fresh_cache = ModelHashCache(_tmp_cache_path)
            _orig_get_cache = _a1111_mod.get_model_hash_cache

            def _patched_get_cache() -> ModelHashCache:
                return _fresh_cache

            _a1111_mod.get_model_hash_cache = _patched_get_cache  # type: ignore[assignment]

            try:
                # A1111ModelInfo — find smallest checkpoint < 100 MB.
                ckpt_files = comfy_fp_a1111.get_filename_list("checkpoints")
                if ckpt_files:
                    _small_ckpt = min(
                        (f for f in ckpt_files),
                        key=lambda f: os.path.getsize(
                            comfy_fp_a1111.get_full_path("checkpoints", f)
                        ),
                    )
                    _ckpt_size = os.path.getsize(
                        comfy_fp_a1111.get_full_path("checkpoints", _small_ckpt)
                    )
                    if _ckpt_size <= 100 * 1024 * 1024:
                        model_node = mappings["A1111ModelInfo"]()
                        (ret_name, ret_hash, ret_meta) = model_node.build(
                            model_type="Checkpoint",
                            model_name=_small_ckpt,
                        )
                        expected_hash = compute_model_hash(
                            comfy_fp_a1111.get_full_path("checkpoints", _small_ckpt)
                        )
                        assert ret_hash == expected_hash, (
                            f"Model hash mismatch: {ret_hash!r} != "
                            f"{expected_hash!r}"
                        )
                        import os as _os

                        expected_name = _os.path.splitext(_small_ckpt)[0]
                        assert ret_name == expected_name, (
                            f"Model name mismatch: {ret_name!r} != "
                            f"{expected_name!r}"
                        )
                        assert list(ret_meta.keys()) == [
                            "Model hash",
                            "Model",
                        ], f"unexpected metadata keys: {list(ret_meta.keys())}"
                    else:
                        print(
                            f"A1111 Model Info node skipped (smallest checkpoint "
                            f"{_ckpt_size / (1024 * 1024):.1f} MB > 100 MB)"
                        )
                else:
                    print("A1111 Model Info node skipped (no checkpoints found)")

                # A1111LoraHashes — all-None contract.
                lora_node = mappings["A1111LoraHashes"]()
                (lora_meta, lora_text) = lora_node.build(
                    lora_1="None",
                    lora_2="None",
                    lora_3="None",
                    lora_4="None",
                    lora_5="None",
                    lora_6="None",
                    lora_7="None",
                    lora_8="None",
                )
                assert lora_meta == {}, (
                    f"all-None LoRA metadata: {lora_meta!r}"
                )
                assert lora_text == "", f"all-None LoRA text: {lora_text!r}"

                # A1111LoraHashes — real slot if loras folder has files.
                lora_files = comfy_fp_a1111.get_filename_list("loras")
                if lora_files:
                    _first_lora = lora_files[0]
                    (lora_meta2, lora_text2) = lora_node.build(
                        **{"lora_1": _first_lora},
                        lora_2="None",
                        lora_3="None",
                        lora_4="None",
                        lora_5="None",
                        lora_6="None",
                        lora_7="None",
                        lora_8="None",
                    )
                    assert "Lora hashes" in lora_meta2, (
                        f"missing Lora hashes key: {lora_meta2!r}"
                    )
                    assert lora_text2 != "", (
                        f"real LoRA should produce non-empty text: {lora_text2!r}"
                    )
                else:
                    print("A1111 Lora Hashes node skipped (no LoRA files found)")

            finally:
                _a1111_mod.get_model_hash_cache = _orig_get_cache  # type: ignore[assignment]
                with contextlib.suppress(OSError):
                    os.unlink(_tmp_cache_path)
        finally:
            sys.path.remove(str(COMFYUI_ROOT))

    # ------------------------------------------------------------------ #
    # E2E: LukutarSaveImage — save + metadata round-trip                  #
    # ------------------------------------------------------------------ #
    if COMFYUI_ROOT.exists():
        sys.path.insert(0, str(COMFYUI_ROOT))
        try:
            from PIL import Image as PILImage  # noqa: F401

            assert "LukutarSaveImage" in mappings, (
                "LukutarSaveImage not registered"
            )
            assert "LukutarSaveImage" in pack.NODE_DISPLAY_NAME_MAPPINGS  # type: ignore[attr-defined]

            save_node = mappings["LukutarSaveImage"]()
            fn_cfg = save_node.INPUT_TYPES()["required"]["filename"][1]
            assert fn_cfg["default"] == "img", (
                "filename default must be 'img'"
            )
            small_tensor = torch.rand(1, 8, 8, 3)  # B,H,W,3 float [0,1]

            with tempfile.TemporaryDirectory() as tmpdir:
                # First save — PNG with metadata
                ret1 = save_node.save(
                    images=small_tensor,
                    path=tmpdir,
                    filename="LukutarSmoke",
                    format="PNG",
                    quality=95,
                    metadata={"desc": "smoke", "version": "1.1"},
                )
                assert ret1 and "ui" in ret1 and "images" in ret1["ui"], (
                    f"save returned unexpected: {ret1}"
                )

                # Verify the pass-through return shape: (images, metadata).
                result1 = ret1.get("result")
                assert isinstance(result1, tuple) and len(result1) == 2, (
                    f"expected 2-tuple result, got {result1!r}"
                )
                assert result1[0] is small_tensor, (
                    "save() must return the input tensor by identity"
                )
                assert result1[1] == {"desc": "smoke", "version": "1.1"}, (
                    f"metadata mismatch in result: {result1[1]!r}"
                )

                # Locate the saved PNG — files may be under output/ or tmpdir
                png_candidates = list(Path(tmpdir).rglob("LukutarSmoke.png"))
                if not png_candidates:
                    # ComfyUI SaveImageNode writes to its own output dir;
                    # check there too.
                    import folder_paths as comfy_fp

                    base = getattr(comfy_fp, "base_path", Path.cwd())
                    output_dir = Path(base) / "output"
                    png_candidates = list(output_dir.rglob("LukutarSmoke.png"))
                assert png_candidates, "PNG file not found after save"
                png_path = png_candidates[0]

                # Verify PIL reads back the tEXt chunks
                with PILImage.open(png_path) as img:
                    info = getattr(img, "info", {})
                    keys = list(info.keys())
                    assert "desc" in info, f"desc missing from PNG: {keys}"
                    assert "version" in info, f"version missing from PNG: {keys}"

                # Second save with identical args → collision suffix _00001
                ret2 = save_node.save(
                    images=small_tensor,
                    path=tmpdir,
                    filename="LukutarSmoke",
                    format="PNG",
                    quality=95,
                    metadata={"desc": "smoke", "version": "1.1"},
                )
                assert ret2 and "ui" in ret2

                png_candidates2 = list(Path(tmpdir).rglob("LukutarSmoke_00001.png"))
                if not png_candidates2:
                    png_candidates2 = list(output_dir.rglob("LukutarSmoke_00001.png"))
                assert png_candidates2, "PNG collision file not found"

                # Third save — PNG *without* metadata → result[1] must be None.
                ret_no_meta = save_node.save(
                    images=small_tensor,
                    path=tmpdir,
                    filename="LukutarSmokeNoMeta",
                    format="PNG",
                    quality=95,
                )
                assert ret_no_meta and "ui" in ret_no_meta
                result_no_meta = ret_no_meta.get("result")
                assert isinstance(result_no_meta, tuple) and len(result_no_meta) == 2, (
                    f"expected 2-tuple result, got {result_no_meta!r}"
                )
                assert result_no_meta[0] is small_tensor, (
                    "save() without metadata must still return input tensor by identity"
                )
                assert result_no_meta[1] is None, (
                    f"result[1] must be None when no metadata "
                    f"passed, got {result_no_meta[1]!r}"
                )

                # JPG save + EXIF round-trip
                jpg_ret = save_node.save(
                    images=small_tensor,
                    path=tmpdir,
                    filename="LukutarSmokeJPG",
                    format="JPG",
                    quality=95,
                    metadata={"desc": "smoke", "version": "1.1"},
                )
                assert jpg_ret and "ui" in jpg_ret

                # Verify the pass-through for the JPG call too.
                result_jpg = jpg_ret.get("result")
                assert isinstance(result_jpg, tuple) and len(result_jpg) == 2, (
                    f"expected 2-tuple result, got {result_jpg!r}"
                )
                assert result_jpg[0] is small_tensor, (
                    "JPG save() must return the input tensor by identity"
                )
                assert result_jpg[1] == {"desc": "smoke", "version": "1.1"}, (
                    f"metadata mismatch in JPG result: {result_jpg[1]!r}"
                )

                jpg_candidates = list(Path(tmpdir).rglob("LukutarSmokeJPG.jpg"))
                if not jpg_candidates:
                    jpg_candidates = list(output_dir.rglob("LukutarSmokeJPG.jpg"))
                assert jpg_candidates, "JPG file not found"

                with PILImage.open(jpg_candidates[0]) as img:
                    exif = img.getexif()
                    raw = exif.get(0x9286)
                    assert raw is not None, "EXIF UserComment missing in JPG"
                    data = raw[8:].decode("utf-16").rstrip("\x00")
                    import json as _json

                    loaded = _json.loads(data)
                    assert loaded == {"desc": "smoke", "version": "1.1"}, (
                        f"JPG EXIF round-trip mismatch: {loaded}"
                    )

                # Fourth save — JPG *without* metadata → must not crash, no EXIF.
                jpg_no_meta_ret = save_node.save(
                    images=small_tensor,
                    path=tmpdir,
                    filename="LukutarSmokeJPGNoMeta",
                    format="JPG",
                    quality=95,
                )
                assert jpg_no_meta_ret and "ui" in jpg_no_meta_ret
                jpg_no_meta_result = jpg_no_meta_ret.get("result")
                assert isinstance(jpg_no_meta_result, tuple) and len(
                    jpg_no_meta_result
                ) == 2, f"expected 2-tuple result, got {jpg_no_meta_result!r}"
                assert jpg_no_meta_result[0] is small_tensor, (
                    "JPG save() without metadata must return input tensor by identity"
                )
                assert jpg_no_meta_result[1] is None, (
                    f"result[1] must be None when no metadata passed, "
                    f"got {jpg_no_meta_result[1]!r}"
                )

                jpg_no_meta_candidates = list(
                    Path(tmpdir).rglob("LukutarSmokeJPGNoMeta.jpg")
                )
                if not jpg_no_meta_candidates:
                    import folder_paths as comfy_fp2

                    base2 = getattr(comfy_fp2, "base_path", Path.cwd())
                    jpg_no_meta_candidates = list(
                        (Path(base2) / "output").rglob(
                            "LukutarSmokeJPGNoMeta.jpg"
                        )
                    )
                assert jpg_no_meta_candidates, (
                    "JPG file without metadata not found"
                )

                with PILImage.open(jpg_no_meta_candidates[0]) as img:
                    assert img.format == "JPEG", f"unexpected format: {img.format}"
                    exif = img.getexif()
                    assert exif.get(0x9286) is None, (
                        "EXIF UserComment must be absent when no metadata passed"
                    )
        finally:
            sys.path.remove(str(COMFYUI_ROOT))

    # ------------------------------------------------------------------ #
    # E2E: Chain MetadataBlocks → MergeMetadata                          #
    # ------------------------------------------------------------------ #
    blocks_node = mappings["MetadataBlocks"]()
    (block_a,) = blocks_node.create_blocks(
        num_blocks=3,
        name_1="env", value_1="production",
        name_2="model", value_2="sd-xl",
        name_3="", value_3="ignored",
    )
    assert block_a == {"env": "production", "model": "sd-xl"}

    (block_b,) = blocks_node.create_blocks(
        num_blocks=2,
        name_1="model", value_1="sdxl-v2",
        name_2="revision", value_2="beta",
    )
    assert block_b == {"model": "sdxl-v2", "revision": "beta"}

    merge_node = mappings["MergeMetadata"]()
    (merged,) = merge_node.merge(meta_1=block_a, meta_2=block_b)
    # Smart merge: the same-name "model" entries are comma-joined
    assert merged == {
        "env": "production",
        "model": "sd-xl, sdxl-v2",
        "revision": "beta",
    }

    # ------------------------------------------------------------------ #
    # E2E: LukutarSaveImageJxl — save + metadata round-trip             #
    # ------------------------------------------------------------------ #
    if "LukutarSaveImageJxl" in mappings:
        sys.path.insert(0, str(COMFYUI_ROOT))
        try:
            from PIL import Image as PILImage  # noqa: F401

            jxl_node = mappings["LukutarSaveImageJxl"]()
            jxl_fn_cfg = jxl_node.INPUT_TYPES()["required"]["filename"][1]
            assert jxl_fn_cfg["default"] == "img", (
                "filename default must be 'img'"
            )

            small_tensor = torch.rand(1, 8, 8, 3)  # B,H,W,3 float [0,1]

            # Probe pillow-jxl-plugin availability.
            try:
                import pillow_jxl  # noqa: F401
            except ImportError:
                print("JXL node skipped: pillow-jxl-plugin not installed")
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Lossy save + EXIF round-trip.
                    ret_lossy = jxl_node.save(
                        images=small_tensor,
                        path=tmpdir,
                        filename="LukutarJxlSmoke",
                        lossless=False,
                        quality=90,
                        effort=7,
                        decoding_speed=0,
                        use_container=False,
                        use_original_profile=False,
                        num_threads=-1,
                        compress_metadata=False,
                        metadata={"desc": "smoke"},
                    )
                    assert ret_lossy and "ui" in ret_lossy

                    # Verify the pass-through return shape: (images, metadata).
                    result_lossy = ret_lossy.get("result")
                    assert isinstance(result_lossy, tuple) and len(result_lossy) == 2, (
                        f"expected 2-tuple result, got {result_lossy!r}"
                    )
                    assert result_lossy[0] is small_tensor, (
                        "JXL save() must return the input tensor by identity"
                    )
                    assert result_lossy[1] == {"desc": "smoke"}, (
                        f"metadata mismatch in JXL lossy result: {result_lossy[1]!r}"
                    )

                    jxl_candidates = list(
                        Path(tmpdir).rglob("LukutarJxlSmoke.jxl")
                    )
                    if not jxl_candidates:
                        import folder_paths as comfy_fp

                        base = getattr(comfy_fp, "base_path", Path.cwd())
                        output_dir = Path(base) / "output"
                        jxl_candidates = list(
                            output_dir.rglob("LukutarJxlSmoke.jxl")
                        )
                    assert jxl_candidates, "JXL file not found after save"
                    jxl_path = jxl_candidates[0]

                    # Verify EXIF round-trip via PIL.
                    with PILImage.open(jxl_path) as img:
                        exif = img.getexif()
                        raw = exif.get(0x9286)
                        assert raw is not None, (
                            "EXIF UserComment missing in JXL"
                        )
                        data = raw[8:].decode("utf-16").rstrip("\x00")
                        loaded = json.loads(data)
                        assert loaded == {"desc": "smoke"}, (
                            f"JXL EXIF round-trip mismatch: {loaded}"
                        )

                    # Lossless save with pixel-exact read-back.
                    lossless_frame = np.array(
                        [
                            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
                            [[70.0, 80.0, 90.0], [100.0, 110.0, 120.0]],
                        ],
                        dtype=np.float32,
                    )
                    lossless_tensor = torch.from_numpy(
                        (lossless_frame / 255.0).astype(np.float32)
                    )[None]

                    ret_lossless = jxl_node.save(
                        images=lossless_tensor,
                        path=tmpdir,
                        filename="LukutarJxlLossless",
                        lossless=True,
                        quality=50,  # ignored in lossless mode
                        effort=7,
                        decoding_speed=0,
                        use_container=False,
                        use_original_profile=False,
                        num_threads=-1,
                        compress_metadata=False,
                    )
                    assert ret_lossless and "ui" in ret_lossless

                    # Verify the pass-through for lossless call (no metadata).
                    result_lossless = ret_lossless.get("result")
                    assert isinstance(result_lossless, tuple) and len(
                        result_lossless
                    ) == 2, f"expected 2-tuple result, got {result_lossless!r}"
                    assert result_lossless[0] is lossless_tensor, (
                        "JXL lossless save() must return the input tensor by identity"
                    )
                    assert result_lossless[1] is None, (
                        f"JXL lossless result[1] must be None "
                        f"when no metadata passed, got "
                        f"{result_lossless[1]!r}"
                    )

                    ll_candidates = list(
                        Path(tmpdir).rglob("LukutarJxlLossless.jxl")
                    )
                    if not ll_candidates:
                        ll_candidates = list(
                            output_dir.rglob("LukutarJxlLossless.jxl")
                        )
                    assert ll_candidates, (
                        "Lossless JXL file not found"
                    )

                    with PILImage.open(ll_candidates[0]) as img:
                        back = np.array(img)
                        expected = np.clip(
                            np.rint(lossless_frame), 0, 255
                        ).astype(np.uint8)
                        assert np.array_equal(
                            back, expected
                        ), "Lossless JXL round-trip pixel mismatch"

        finally:
            sys.path.remove(str(COMFYUI_ROOT))

    # ------------------------------------------------------------------ #
    # Validate LukutarSaveImageJxl prompt (always runs if node registered)
    # ------------------------------------------------------------------ #
    if "LukutarSaveImageJxl" in mappings:
        _validate_jxl_prompt(mappings)

    print("SMOKE TEST PASSED: nodes load, save I/O works, and "
          "save outputs (images, metadata) pass-through verified")


if __name__ == "__main__":
    main()
