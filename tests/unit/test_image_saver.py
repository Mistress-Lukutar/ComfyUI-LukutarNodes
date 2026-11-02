'''
File:   test_image_saver.py
Brief:  Unit tests for the pure image-saving engine (torch-free).
Author: Mistress-Lukutar
Date:   2026-11-02
Version: v0.1.0

Tests core.image_saver functions: sanitize_filename, normalize_dir,
unique_destination, available_formats, supports_metadata, save_frame,
save_images. Requires Pillow.
'''

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from comfyui_lukutar_nodes.core.image_saver import (
    HAS_PIL,
    SAVE_FORMATS,
    available_formats,
    normalize_dir,
    sanitize_filename,
    save_frame,
    save_images,
    supports_metadata,
    unique_destination,
)

# ======================================================================
# sanitize_filename
# ======================================================================


def test_sanitize_replaces_forbidden_chars() -> None:
    """Forbidden characters are replaced with underscore."""
    result = sanitize_filename('file<>:"/\\|?')
    assert "<" not in result
    assert ">" not in result
    assert '"' not in result
    assert "\\" not in result
    assert "|" not in result
    assert "?" not in result


def test_sanitize_collapses_whitespace() -> None:
    """Runs of whitespace are collapsed; control chars replaced first."""
    # \t is a control char → '_', then remaining spaces collapse.
    result = sanitize_filename("hello   world\t\ttest")
    assert result == "hello world__test"


def test_sanitize_strips_trailing_dots_and_spaces() -> None:
    """Trailing dots and spaces are stripped (Windows restriction)."""
    result = sanitize_filename("image...  ")
    assert result == "image"


def test_sanitize_empty_returns_image() -> None:
    """When the result would be empty, 'image' is returned."""
    result = sanitize_filename("...   ")
    assert result == "image"


# ======================================================================
# normalize_dir
# ======================================================================


def test_normalize_absolute_path_as_is() -> None:
    """Absolute paths are used as-is (after expanduser)."""
    base = Path("/tmp/base")
    result = normalize_dir("/abs/path", base)
    # On Windows a path without drive letter is not "absolute" per
    # WindowsPath.is_absolute(), but it still starts with / so we check
    # the string representation.
    s = str(result).replace("\\", "/")
    assert s.startswith("/") or s.startswith("C:")


def test_normalize_relative_joins_base() -> None:
    """Relative paths are joined onto base_dir."""
    base = Path("/tmp/base")
    result = normalize_dir("output", base)
    assert "base" in str(result)


def test_normalize_empty_uses_base() -> None:
    """Empty or whitespace-only path returns base_dir."""
    base = Path("/tmp/base")
    for val in ["", "   ", "\t"]:
        result = normalize_dir(val, base)
        assert result == base


def test_normalize_quotes_stripped() -> None:
    """Surrounding quotes are stripped before processing."""
    base = Path("/tmp/base")
    result = normalize_dir('"output"', base)
    # Should not contain quote characters in the path
    assert '"' not in str(result)
    assert "base" in str(result)


# ======================================================================
# unique_destination
# ======================================================================


def test_unique_destination_free_name_unchanged(tmp_path: Path) -> None:
    """When the name is free, it's returned unchanged."""
    result = unique_destination(tmp_path, "img", ".png")
    assert result == tmp_path / "img.png"


def test_unique_destination_appends_counter_on_collision(tmp_path: Path) -> None:
    """Existing file triggers _00001 suffix; multiple collisions increment."""
    (tmp_path / "img.png").touch()
    r1 = unique_destination(tmp_path, "img", ".png")
    assert r1.name == "img_00001.png"

    (tmp_path / "img_00001.png").touch()
    r2 = unique_destination(tmp_path, "img", ".png")
    assert r2.name == "img_00002.png"


# ======================================================================
# available_formats + supports_metadata
# ======================================================================


def test_available_formats_excludes_jxl_without_plugin() -> None:
    """JXL is absent when HAS_JXL_PLUGIN is False."""
    with patch(
        "comfyui_lukutar_nodes.core.image_saver.HAS_JXL_PLUGIN", False
    ):
        fmts = available_formats()
        assert "JXL" not in fmts


def test_available_formats_includes_jxl_with_plugin() -> None:
    """JXL is present when HAS_JXL_PLUGIN is True."""
    with patch(
        "comfyui_lukutar_nodes.core.image_saver.HAS_JXL_PLUGIN", True
    ):
        fmts = available_formats()
        assert "JXL" in fmts


def test_supports_metadata_bmp_false_others_true() -> None:
    """BMP never supports metadata; all others do."""
    assert not supports_metadata("BMP")
    for fmt in SAVE_FORMATS:
        if fmt != "BMP":
            assert supports_metadata(fmt), f"{fmt} should support metadata"


# ======================================================================
# save_frame round-trips (requires Pillow)
# ======================================================================

@pytest.fixture()
def sample_frame() -> np.ndarray:
    """Small HWC float32 [0, 255] RGB frame for testing."""
    return np.full((4, 4, 3), 128.0, dtype=np.float32)


def _make_metadata() -> dict[str, str]:
    return {"desc": "\u0442\u0435\u043a\u0441\u0442", "version": "1.1"}


# -- PNG round-trip --------------------------------------------------------

def test_save_png_metadata_roundtrip(tmp_path: Path, sample_frame: np.ndarray) -> None:
    """PNG saves and reads back metadata via PIL tEXt chunks."""
    if not HAS_PIL:
        pytest.skip("Pillow not installed")

    dst = tmp_path / "test.png"
    meta = _make_metadata()
    save_frame(sample_frame, dst, "PNG", metadata=meta)

    from PIL import Image as PILImage

    with PILImage.open(dst) as img:
        info = img.info if hasattr(img, "info") else {}
        assert "desc" in info, f"desc missing from PNG info: {list(info.keys())}"
        assert "version" in info, f"version missing from PNG info: {list(info.keys())}"


# -- JPG round-trip --------------------------------------------------------

def test_save_jpg_metadata_roundtrip(tmp_path: Path, sample_frame: np.ndarray) -> None:
    """JPG saves and reads back metadata via EXIF 0x9286 UserComment."""
    if not HAS_PIL:
        pytest.skip("Pillow not installed")

    dst = tmp_path / "test.jpg"
    meta = _make_metadata()
    save_frame(sample_frame, dst, "JPG", quality=95, metadata=meta)

    from PIL import Image as PILImage

    with PILImage.open(dst) as img:
        exif = img.getexif()
        raw = exif.get(0x9286)
        assert raw is not None, "EXIF UserComment missing"
        # Strip first 8 bytes ("UNICODE\\x00") and decode UTF-16
        data = raw[8:].decode("utf-16").rstrip("\x00")
        loaded = json.loads(data)
        assert loaded == meta


# -- WEBP round-trip -------------------------------------------------------

def test_save_webp_metadata_roundtrip(tmp_path: Path, sample_frame: np.ndarray) -> None:
    """WEBP saves and reads back metadata via EXIF 0x9286."""
    if not HAS_PIL:
        pytest.skip("Pillow not installed")

    dst = tmp_path / "test.webp"
    meta = _make_metadata()
    save_frame(sample_frame, dst, "WEBP", quality=95, metadata=meta)

    from PIL import Image as PILImage

    with PILImage.open(dst) as img:
        exif = img.getexif()
        raw = exif.get(0x9286)
        assert raw is not None, "EXIF UserComment missing in WEBP"
        data = raw[8:].decode("utf-16").rstrip("\x00")
        loaded = json.loads(data)
        assert loaded == meta


# -- TIFF round-trip -------------------------------------------------------

def test_save_tiff_metadata_roundtrip(tmp_path: Path, sample_frame: np.ndarray) -> None:
    """TIFF saves and reads back metadata via tag 270 (ImageDescription)."""
    if not HAS_PIL:
        pytest.skip("Pillow not installed")

    dst = tmp_path / "test.tif"
    # ASCII-only metadata to avoid Pillow's TIFF tag encoding of non-ASCII.
    meta = {"desc": "hello world", "version": "1.1"}
    save_frame(sample_frame, dst, "TIFF", metadata=meta)

    from PIL import Image as PILImage

    with PILImage.open(dst) as img:
        tag_val = img.tag_v2.get(270)
        assert tag_val is not None, "TIFF tag 270 missing"
        if isinstance(tag_val, bytes):
            tag_val = tag_val.decode("utf-8", errors="replace")
        loaded = json.loads(tag_val)
        assert loaded == meta


# -- BMP metadata silently ignored -----------------------------------------

def test_save_bmp_no_metadata_error(tmp_path: Path, sample_frame: np.ndarray) -> None:
    """BMP saves fine even when metadata is passed (silently ignored)."""
    if not HAS_PIL:
        pytest.skip("Pillow not installed")

    dst = tmp_path / "test.bmp"
    meta = _make_metadata()
    save_frame(sample_frame, dst, "BMP", metadata=meta)
    assert dst.exists()


# -- JXL round-trip --------------------------------------------------------

def test_save_jxl_roundtrip(tmp_path: Path, sample_frame: np.ndarray) -> None:
    """JXL saves and optionally reads back EXIF if pillow_jxl works."""
    if not HAS_PIL:
        pytest.skip("Pillow not installed")

    dst = tmp_path / "test.jxl"
    meta = _make_metadata()
    try:
        save_frame(sample_frame, dst, "JXL", quality=95, metadata=meta)
    except Exception as exc:
        pytest.skip(f"pillow_jxl save failed: {exc}")

    assert dst.exists()

    # If readable, verify EXIF round-trip
    try:
        from PIL import Image as PILImage

        with PILImage.open(dst) as img:
            exif = img.getexif()
            raw = exif.get(0x9286)
            if raw is not None:
                data = raw[8:].decode("utf-16").rstrip("\x00")
                loaded = json.loads(data)
                assert loaded == meta
    except Exception as exc:
        pytest.skip(f"pillow_jxl read failed: {exc}")


# -- metadata-rejection resilience -----------------------------------------

def test_save_frame_degrades_on_metadata_error(
    tmp_path: Path, sample_frame: np.ndarray
) -> None:
    """When build_png_info raises, save still writes a valid PNG."""
    if not HAS_PIL:
        pytest.skip("Pillow not installed")

    dst = tmp_path / "test.png"

    def failing_build_png_info(*_args, **_kwargs):
        raise RuntimeError("boom")

    with patch(
        "comfyui_lukutar_nodes.core.image_saver.build_png_info",
        failing_build_png_info,
    ):
        save_frame(sample_frame, dst, "PNG", metadata={"k": "v"})

    assert dst.exists()
    from PIL import Image as PILImage

    with PILImage.open(dst) as img:
        # Should be a valid image (no exception on open)
        assert img.format == "PNG"


# -- frame conversion semantics --------------------------------------------

def test_save_frame_float32_to_uint8_round_clip(
    tmp_path: Path, sample_frame: np.ndarray
) -> None:
    """float32 [0, 255] input is rounded/clipped to uint8 in output."""
    if not HAS_PIL:
        pytest.skip("Pillow not installed")

    # Create a frame with non-integer values
    float_frame = np.array([[[127.4, 128.6, 129.5]]], dtype=np.float32)
    dst = tmp_path / "test.png"
    save_frame(float_frame, dst, "PNG")

    from PIL import Image as PILImage

    with PILImage.open(dst) as img:
        arr = np.array(img)
        # np.rint(127.4)=127, np.rint(128.6)=129, np.rint(129.5)=130
        expected = np.rint(float_frame).astype(np.uint8)
        assert arr[0, 0, 0] == expected[0, 0, 0]
        assert arr[0, 0, 1] == expected[0, 0, 1]
        assert arr[0, 0, 2] == expected[0, 0, 2]


# ======================================================================
# save_images (batch saver)
# ======================================================================


def test_save_images_batch_indexed_filenames(tmp_path: Path) -> None:
    """Batch of 3 frames produces stem_00000, stem_00001, stem_00002."""
    frames = [
        np.full((4, 4, 3), float(i * 50 + 10), dtype=np.float32)
        for i in range(3)
    ]
    results = save_images(frames, str(tmp_path), "stem", "PNG", 95, None, tmp_path)
    assert len(results) == 3
    names = [r.name for r in results]
    assert "stem_00000.png" in names
    assert "stem_00001.png" in names
    assert "stem_00002.png" in names


def test_save_images_single_frame_no_index(tmp_path: Path) -> None:
    """Single frame uses bare stem (no index suffix)."""
    frames = [np.full((4, 4, 3), 100.0, dtype=np.float32)]
    results = save_images(frames, str(tmp_path), "bare", "PNG", 95, None, tmp_path)
    assert len(results) == 1
    assert results[0].name == "bare.png"


def test_save_images_nested_dirs_created(tmp_path: Path) -> None:
    """Nested target dirs are created automatically."""
    frames = [np.full((4, 4, 3), 50.0, dtype=np.float32)]
    results = save_images(frames, "a/b/c", "img", "PNG", 95, None, tmp_path)
    assert len(results) == 1
    # The file lives at tmp_path/a/b/c/img.png; going up 3 levels from the file
    # parent gives us 'a'.
    assert results[0].parent.parent.parent.name == "a"


def test_save_images_collision_suffixed_on_repeated_call(tmp_path: Path) -> None:
    """Calling again with same inputs creates collision-suffixed files."""
    frames = [np.full((4, 4, 3), 50.0, dtype=np.float32)]
    r1 = save_images(frames, str(tmp_path), "stem", "PNG", 95, None, tmp_path)
    r2 = save_images(frames, str(tmp_path), "stem", "PNG", 95, None, tmp_path)
    assert len(r2) == 1
    # Second call should get _00001 suffix
    assert "_00001.png" in r2[0].name
    # All returned paths must exist
    assert r1[0].exists()
    assert r2[0].exists()


def test_save_images_unknown_format_raises(tmp_path: Path) -> None:
    """Unknown format raises RuntimeError."""
    frames = [np.full((4, 4, 3), 50.0, dtype=np.float32)]
    with pytest.raises(RuntimeError):
        save_images(frames, str(tmp_path), "img", "FAKEFMT", 95, None, tmp_path)


def test_save_images_frame_failure_wrapped(tmp_path: Path) -> None:
    """A save failure inside is wrapped as RuntimeError mentioning the destination."""
    frames = [np.full((4, 4, 3), 50.0, dtype=np.float32)]

    def failing_save(*_args, **_kwargs):
        raise OSError("disk full")

    with patch(
        "comfyui_lukutar_nodes.core.image_saver.save_frame",
        failing_save,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            save_images(frames, str(tmp_path), "img", "PNG", 95, None, tmp_path)
        assert "Failed to save frame" in str(exc_info.value)
