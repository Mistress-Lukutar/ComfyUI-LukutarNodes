'''
File:   test_a1111_nodes.py
Brief:  Unit tests for the A1111 node classes (torch-free).
Author: Mistress-Lukutar
Date:   2026-11-02
Version: v0.1.0

Tests the five node classes in nodes/a1111.py by instantiating them
directly and calling their ``build`` methods.
'''

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from comfyui_lukutar_nodes.core.a1111_parameters import (
    ModelHashCache,
    compute_model_hash,
)
from comfyui_lukutar_nodes.nodes.a1111 import (
    LORA_NONE,
    A1111HiresFixNode,
    A1111LoraHashesNode,
    A1111ModelInfoNode,
    A1111ParametersNode,
    A1111SamplingNode,
)

# ======================================================================
# A1111SamplingNode                                                      #
# ======================================================================


def test_sampling_node_full_case_exact_dict() -> None:
    """Full parameter set must produce the exact metadata dict in order."""
    node = A1111SamplingNode()
    result, = node.build(
        steps=30,
        sampler="DPM++ 2M Karras",
        cfg=7.0,
        seed=42,
        width=832,
        height=1216,
        clip_skip=2,
    )
    expected: dict[str, str] = {
        "Steps": "30",
        "Sampler": "DPM++ 2M Karras",
        "CFG scale": "7",
        "Seed": "42",
        "Size": "832x1216",
        "Clip skip": "2",
    }
    assert list(result.items()) == list(expected.items())


def test_sampling_node_width_zero_omits_size() -> None:
    """width=0 must omit the Size entry even when height > 0."""
    node = A1111SamplingNode()
    result, = node.build(
        steps=20, sampler="Euler", cfg=8.0, seed=0,
        width=0, height=768, clip_skip=0,
    )
    assert "Size" not in result


def test_sampling_node_height_zero_omits_size() -> None:
    """height=0 must omit the Size entry."""
    node = A1111SamplingNode()
    result, = node.build(
        steps=20, sampler="Euler", cfg=8.0, seed=0,
        width=512, height=0, clip_skip=0,
    )
    assert "Size" not in result


def test_sampling_node_clip_skip_zero_omits_entry() -> None:
    """clip_skip=0 must omit the Clip skip entry."""
    node = A1111SamplingNode()
    result, = node.build(
        steps=20, sampler="Euler", cfg=8.0, seed=0,
        width=512, height=512, clip_skip=0,
    )
    assert "Clip skip" not in result


def test_sampling_node_clip_skip_one_emits() -> None:
    """clip_skip=1 must emit the Clip skip entry as '1'."""
    node = A1111SamplingNode()
    result, = node.build(
        steps=20, sampler="Euler", cfg=8.0, seed=0,
        width=512, height=512, clip_skip=1,
    )
    assert result["Clip skip"] == "1"


def test_sampling_node_cfg_g_formatting() -> None:
    """CFG values like 7.0 must render as '7', not '7.0'."""
    node = A1111SamplingNode()
    result, = node.build(
        steps=20, sampler="Euler", cfg=7.0, seed=0,
        width=512, height=512, clip_skip=0,
    )
    assert result["CFG scale"] == "7"

    result2, = node.build(
        steps=20, sampler="Euler", cfg=0.45, seed=0,
        width=512, height=512, clip_skip=0,
    )
    assert result2["CFG scale"] == "0.45"

# ======================================================================
# A1111HiresFixNode                                                      #
# ======================================================================


def test_hires_fix_node_exact_dict() -> None:
    """Default params must produce exact metadata dict in order."""
    node = A1111HiresFixNode()
    result, = node.build(
        upscaler="4x-UltraSharp",
        upscale=2.0,
        hires_steps=10,
        denoising=0.45,
    )
    expected: dict[str, str] = {
        "Hires upscale": "2.0",
        "Hires upscaler": "4x-UltraSharp",
        "Hires steps": "10",
        "Denoising strength": "0.45",
    }
    assert list(result.items()) == list(expected.items())


def test_hires_fix_steps_zero_omits() -> None:
    """hires_steps=0 must omit the 'Hires steps' entry."""
    node = A1111HiresFixNode()
    result, = node.build(
        upscaler="4x-UltraSharp",
        upscale=2.0,
        hires_steps=0,
        denoising=0.45,
    )
    assert "Hires steps" not in result


def test_hires_fix_steps_one_emits() -> None:
    """hires_steps=1 must emit 'Hires steps': '1'."""
    node = A1111HiresFixNode()
    result, = node.build(
        upscaler="4x-UltraSharp",
        upscale=2.0,
        hires_steps=1,
        denoising=0.45,
    )
    assert result["Hires steps"] == "1"

# ======================================================================
# A1111ParametersNode                                                    #
# ======================================================================


def test_parameters_node_no_meta_slots() -> None:
    """No meta slots + version must produce settings line with Version."""
    node = A1111ParametersNode()
    meta, text = node.build(positive="p", negative="n", version="forge-0.9.1")
    assert meta == {"parameters": text}
    assert text == "p\nNegative prompt: n\nVersion: forge-0.9.1"


def test_parameters_node_empty_everything() -> None:
    """All empty inputs must return ({}, '')."""
    node = A1111ParametersNode()
    meta, text = node.build(positive="", negative="", version="")
    assert meta == {}
    assert text == ""


def test_parameters_node_with_meta_slots() -> None:
    """meta_1 and meta_2 must appear in the settings line."""
    node = A1111ParametersNode()
    meta, text = node.build(
        positive="",
        negative="",
        version="",
        meta_1={"Steps": "30"},
        meta_2={"Sampler": "Euler"},
    )
    assert meta is not None
    assert "parameters" in meta
    settings = meta["parameters"]
    assert "Steps: 30, Sampler: Euler" in settings


def test_parameters_node_none_meta_block_skipped() -> None:
    """A meta block equal to None must be skipped."""
    node = A1111ParametersNode()
    meta, text = node.build(
        positive="",
        negative="",
        version="",
        meta_1={"Steps": "30"},
        meta_2=None,
        meta_3={"Sampler": "Euler"},
    )
    assert meta is not None
    settings = meta["parameters"]
    # Steps and Sampler should both be present; None block skipped.
    assert "Steps: 30" in settings
    assert "Sampler: Euler" in settings

# ======================================================================
# A1111LoraHashesNode                                                    #
# ======================================================================


def test_lora_hashes_all_none(tmp_path: Path) -> None:
    """All slots set to LORA_NONE must return ({}, '')."""
    node = A1111LoraHashesNode()
    kwargs = {f"lora_{i}": LORA_NONE for i in range(1, 9)}
    meta, hashes = node.build(**kwargs)
    assert meta == {}
    assert hashes == ""


def test_lora_hashes_with_real_slots(tmp_path: Path) -> None:
    """Real LoRA slots must produce correct hashes and metadata."""
    # Create tiny temp files that act as LoRA models.
    lora_files = {}
    for idx in (1, 3):
        content = f"lora_{idx}_data".encode()
        p = tmp_path / f"lora_{idx}.safetensors"
        p.write_bytes(content)
        lora_files[idx] = p

    expected_hash_1 = compute_model_hash(lora_files[1])
    expected_hash_3 = compute_model_hash(lora_files[3])

    # Monkeypatch the module-level folder_paths and cache.
    fake_fp = MagicMock()
    fake_fp.get_full_path_or_raise.return_value = str(
        lora_files[lora_files.keys().__iter__().__next__()]
    )

    def mock_get_full_path_or_raise(folder, name):
        # Extract index from filename like "lora_1.safetensors"
        stem = os.path.splitext(name)[0]  # e.g. "lora_1"
        idx = int(stem.split("_")[1])
        return str(lora_files[idx])

    fake_fp.get_full_path_or_raise = mock_get_full_path_or_raise

    fake_cache = ModelHashCache(tmp_path / "lora_cache.json")

    import comfyui_lukutar_nodes.nodes.a1111 as nodes_a1111

    with (
        patch.object(
            A1111LoraHashesNode, "__module__", "comfyui_lukutar_nodes.nodes.a1111"
        ),
        patch.object(nodes_a1111, "folder_paths", fake_fp),
        patch.object(
            nodes_a1111, "get_model_hash_cache", lambda: fake_cache
        ),
    ):
                node = A1111LoraHashesNode()
                kwargs: dict[str, str] = {}
                kwargs["lora_1"] = "lora_1.safetensors"
                kwargs["lora_2"] = LORA_NONE  # skipped
                kwargs["lora_3"] = "lora_3.safetensors"
                for i in (4, 5, 6, 7, 8):
                    kwargs[f"lora_{i}"] = LORA_NONE

                meta, hashes_str = node.build(**kwargs)

    # Verify hash correctness.
    assert "Lora hashes" in meta
    expected_joined = (
        f"lora_1: {expected_hash_1}, lora_3: {expected_hash_3}"
    )
    # The value is stored raw; quoting happens at serialization time.
    assert meta["Lora hashes"] == expected_joined
    assert hashes_str == expected_joined


def test_lora_hashes_numeric_order(tmp_path: Path) -> None:
    """lora_1 and lora_3 must appear in numeric order (not lora_10 > lora_2)."""
    content = b"order_test_data"
    p1 = tmp_path / "lora_1.safetensors"
    p1.write_bytes(content)
    p10 = tmp_path / "lora_10.safetensors"
    p10.write_bytes(content + b"x")

    hash_1 = compute_model_hash(p1)
    hash_10 = compute_model_hash(p10)

    import comfyui_lukutar_nodes.nodes.a1111 as nodes_a1111

    fake_fp = MagicMock()

    def mock_get_full_path_or_raise(folder, name):
        return str(tmp_path / name)

    fake_fp.get_full_path_or_raise = mock_get_full_path_or_raise
    fake_cache = ModelHashCache(tmp_path / "order_cache.json")

    with (
        patch.object(nodes_a1111, "folder_paths", fake_fp),
        patch.object(
            nodes_a1111, "get_model_hash_cache", lambda: fake_cache
        ),
    ):
            node = A1111LoraHashesNode()
            kwargs = {
                "lora_1": "lora_1.safetensors",
                "lora_2": LORA_NONE,
                "lora_3": "lora_10.safetensors",
            }
            for i in (4, 5, 6, 7, 8):
                kwargs[f"lora_{i}"] = LORA_NONE

            meta, _ = node.build(**kwargs)

    # lora_1 should come before lora_3 (= lora_10 file).
    expected_joined = (
        f"lora_1: {hash_1}, lora_10: {hash_10}"
    )
    assert meta["Lora hashes"] == expected_joined

# ======================================================================
# A1111ModelInfoNode                                                     #
# ======================================================================


@pytest.mark.parametrize(
    "model_type,model_name,expected_name,mk",
    [
        (
            "Checkpoint", "my_model.safetensors", "my_model",
            {"Model hash": None, "Model": "my_model"},
        ),
        (
            "VAE", "my_vae.safetensors", "my_vae.safetensors",
            {"VAE hash": None, "VAE": "my_vae.safetensors"},
        ),
        (
            "LoRA", "style_lora.safetensors", "style_lora",
            {"Lora hashes": None},
        ),
        (
            "Embedding", "deepbooru.pt", "deepbooru",
            {"TI hashes": None},
        ),
    ],
)
def test_model_info_type_specific_name_formatting(
    tmp_path: Path,
    model_type: str,
    model_name: str,
    expected_name: str,
    mk: dict[str, str],
) -> None:
    """Each model type must produce the correct display name and metadata keys."""
    content = f"{model_type}_content".encode()
    p = tmp_path / model_name
    p.write_bytes(content)

    expected_hash = compute_model_hash(p)
    mk.get("Model hash")  # noqa: SIM118, used for side-effect check

    import comfyui_lukutar_nodes.nodes.a1111 as nodes_a1111

    fake_fp = MagicMock()

    def mock_get_full_path_or_raise(folder, name):
        return str(tmp_path / name)

    fake_fp.get_full_path_or_raise = mock_get_full_path_or_raise
    fake_cache = ModelHashCache(tmp_path / "modelinfo_cache.json")

    with (
        patch.object(nodes_a1111, "folder_paths", fake_fp),
        patch.object(
            nodes_a1111, "get_model_hash_cache", lambda: fake_cache
        ),
    ):
            node = A1111ModelInfoNode()
            name, h, meta = node.build(model_type=model_type, model_name=model_name)

    # Name check.
    assert name == expected_name
    # Hash check.
    assert h == expected_hash
    # Metadata keys and values.
    if model_type == "Checkpoint":
        assert list(meta.keys()) == ["Model hash", "Model"]
        assert meta["Model hash"] == expected_hash
        assert meta["Model"] == expected_name
    elif model_type == "VAE":
        assert list(meta.keys()) == ["VAE hash", "VAE"]
        assert meta["VAE hash"] == expected_hash
        assert meta["VAE"] == expected_name
    elif model_type == "LoRA":
        assert "Lora hashes" in meta
        # Hash must be present in the quoted string.
        assert expected_hash in meta["Lora hashes"]
    elif model_type == "Embedding":
        assert "TI hashes" in meta
        assert expected_hash in meta["TI hashes"]


def test_model_info_checkpoint_order_model_hash_before_model(
    tmp_path: Path,
) -> None:
    """For Checkpoint, 'Model hash' must come before 'Model'."""
    content = b"checkpoint_test"
    p = tmp_path / "ckpt.safetensors"
    p.write_bytes(content)

    expected_hash = compute_model_hash(p)

    import comfyui_lukutar_nodes.nodes.a1111 as nodes_a1111

    fake_fp = MagicMock()
    fake_fp.get_full_path_or_raise = lambda f, n: str(tmp_path / n)
    fake_cache = ModelHashCache(tmp_path / "order_cache2.json")

    with (
        patch.object(nodes_a1111, "folder_paths", fake_fp),
        patch.object(
            nodes_a1111, "get_model_hash_cache", lambda: fake_cache
        ),
    ):
            node = A1111ModelInfoNode()
            _, _, meta = node.build(
                model_type="Checkpoint",
                model_name="ckpt.safetensors",
            )

    keys = list(meta.keys())
    assert keys == ["Model hash", "Model"]
    assert meta["Model hash"] == expected_hash


def test_model_info_raises_when_no_folder_paths() -> None:
    """A1111ModelInfoNode must raise RuntimeError when folder_paths is None."""
    import comfyui_lukutar_nodes.nodes.a1111 as nodes_a1111

    with patch.object(nodes_a1111, "folder_paths", None):
        node = A1111ModelInfoNode()
        with pytest.raises(RuntimeError, match="folder_paths"):
            node.build(model_type="Checkpoint", model_name="test.safetensors")


def test_lora_hashes_raises_when_no_folder_paths_with_real_slot() -> None:
    """A1111LoraHashesNode raises when folder_paths is None and there
    is at least one non-skip slot."""
    import comfyui_lukutar_nodes.nodes.a1111 as nodes_a1111

    with patch.object(nodes_a1111, "folder_paths", None):
        node = A1111LoraHashesNode()
        kwargs = {"lora_1": "some_lora.safetensors"}
        for i in (2, 3, 4, 5, 6, 7, 8):
            kwargs[f"lora_{i}"] = LORA_NONE
        with pytest.raises(RuntimeError, match="folder_paths"):
            node.build(**kwargs)


def test_lora_hashes_no_raise_when_all_none_folder_paths_none() -> None:
    """When all slots are 'None' and folder_paths is None, must return ({}, '')."""
    import comfyui_lukutar_nodes.nodes.a1111 as nodes_a1111

    with patch.object(nodes_a1111, "folder_paths", None):
        node = A1111LoraHashesNode()
        kwargs = {f"lora_{i}": LORA_NONE for i in range(1, 9)}
        meta, hashes = node.build(**kwargs)
        assert meta == {}
        assert hashes == ""


def test_parameters_node_quotes_lora_hashes_once() -> None:
    """A raw metadata dict with Lora hashes must be quoted exactly once."""
    node = A1111ParametersNode()
    meta, text = node.build(
        positive="p",
        negative="",
        version="",
        meta_1={"Lora hashes": "vhf: 47a7a957d7"},
    )
    # quote_a1111 detects the colon and JSON-encodes to '"vhf: 47a7a957d7"'.
    assert 'Lora hashes: "vhf: 47a7a957d7"' in text
    assert '\\"' not in text
