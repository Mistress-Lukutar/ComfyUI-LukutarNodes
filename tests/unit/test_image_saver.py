'''
File:   test_image_saver.py
Brief:  Unit tests for the pure image-saving engine (torch-free).
Author: Mistress-Lukutar
Date:   2026-11-02
Version: v0.2.0

Tests core.image_saver functions: sanitize_filename, normalize_dir,
unique_destination, available_formats, supports_metadata, save_frame,
save_images, JxlSettings, save_frame_jxl, save_images_jxl. Requires
Pillow.
'''

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from comfyui_lukutar_nodes.core import image_saver as _image_saver_mod
from comfyui_lukutar_nodes.core.image_saver import (
    HAS_JXL_PLUGIN,
    HAS_PIL,
    SAVE_FORMATS,
    JxlSettings,
    available_formats,
    normalize_dir,
    sanitize_filename,
    save_frame,
    save_frame_jxl,
    save_images,
    save_images_jxl,
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


def test_available_formats_excludes_jxl() -> None:
    """JXL is not in SAVE_FORMATS — it has its own dedicated saver."""
    fmts = available_formats()
    assert fmts == ("PNG", "JPG", "WEBP", "TIFF", "BMP")
    assert "JXL" not in fmts


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


# -- JXL format removed from generic saver ---------------------------------

def test_save_frame_jxl_format_raises_valueerror(tmp_path: Path) -> None:
    """save_frame with fmt='JXL' raises ValueError (unknown format)."""
    frame = np.full((4, 4, 3), 128.0, dtype=np.float32)
    dst = tmp_path / "test.jxl"
    with pytest.raises(ValueError, match="Unknown format"):
        save_frame(frame, dst, "JXL")


# ======================================================================
# JxlSettings dataclass
# ======================================================================


def test_jxl_settings_defaults() -> None:
    """Default values match the spec."""
    s = JxlSettings()
    assert s.lossless is False
    assert s.quality == 90
    assert s.effort == 7
    assert s.decoding_speed == 0
    assert s.use_container is False
    assert s.use_original_profile is False
    assert s.num_threads == -1
    assert s.compress_metadata is False


def test_jxl_settings_frozen() -> None:
    """Mutation raises FrozenInstanceError."""
    s = JxlSettings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.lossless = True  # type: ignore[assignment]


def test_jxl_settings_custom_values() -> None:
    """Custom values are stored."""
    s = JxlSettings(lossless=True, quality=50, effort=9)
    assert s.lossless is True
    assert s.quality == 50
    assert s.effort == 9


# ======================================================================
# save_frame_jxl (requires pillow-jxl-plugin)
# ======================================================================

@pytest.fixture()
def jxl_plugin_available() -> bool:
    """Skip fixture for JXL tests when plugin is absent."""
    if not HAS_JXL_PLUGIN:
        pytest.skip("pillow-jxl-plugin not installed")
    return True  # noqa: RET504


# -- Lossy round-trip ------------------------------------------------------

def test_save_frame_jxl_lossy_roundtrip(
    tmp_path: Path, sample_frame: np.ndarray, jxl_plugin_available: bool
) -> None:
    """Lossy JXL saves and reads back EXIF metadata."""
    from PIL import Image as PILImage

    dst = tmp_path / "test_lossy.jxl"
    meta = _make_metadata()
    save_frame_jxl(sample_frame, dst, JxlSettings(quality=90), metadata=meta)

    assert dst.exists()

    with PILImage.open(dst) as img:
        exif = img.getexif()
        raw = exif.get(0x9286)
        assert raw is not None, "EXIF UserComment missing in lossy JXL"
        data = raw[8:].decode("utf-16").rstrip("\x00")
        loaded = json.loads(data)
        assert loaded == meta


# -- Lossless round-trip ---------------------------------------------------

def test_save_frame_jxl_lossless_exact(
    tmp_path: Path, sample_frame: np.ndarray, jxl_plugin_available: bool
) -> None:
    """Lossless JXL preserves every pixel exactly."""
    from PIL import Image as PILImage

    # Use non-uniform values to detect any rounding.
    frame = np.array(
        [
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
            [[70.0, 80.0, 90.0], [100.0, 110.0, 120.0]],
        ],
        dtype=np.float32,
    )
    dst = tmp_path / "test_lossless.jxl"
    save_frame_jxl(frame, dst, JxlSettings(lossless=True, quality=50))

    assert dst.exists()

    with PILImage.open(dst) as img:
        back = np.array(img)
        expected = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
        assert np.array_equal(back, expected), "Lossless round-trip pixel mismatch"


# -- Degrade path (metadata error → retry without metadata) ----------------

def test_save_frame_jxl_degrade_on_metadata_error(
    tmp_path: Path, sample_frame: np.ndarray, jxl_plugin_available: bool
) -> None:
    """When build_exif_payload raises inside the try block, save retries."""
    from PIL import Image as PILImage

    dst = tmp_path / "test_degrade.jxl"
    meta = {"key": "val"}

    # Patch to succeed on first call (builds exif), fail on second save.
    original_build_exif = _image_saver_mod.build_exif_payload
    call_count = 0

    def counting_build_exif(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return original_build_exif(*args, **kwargs)
        raise RuntimeError("exif boom on retry")

    with patch.object(
        _image_saver_mod, "build_exif_payload", counting_build_exif
    ):
        save_frame_jxl(sample_frame, dst, JxlSettings(), metadata=meta)

    assert dst.exists()

    with PILImage.open(dst) as img:
        # Should be a valid image (no exception on open).
        assert img.format == "JXL"


def test_save_frame_jxl_degrade_on_first_build_error(
    tmp_path: Path, sample_frame: np.ndarray, jxl_plugin_available: bool
) -> None:
    """When the EXIF payload itself fails to build, a clean save is written."""
    from PIL import Image as PILImage

    dst = tmp_path / "test_degrade_build.jxl"

    def failing_build_exif(*args, **kwargs):
        raise RuntimeError("exif boom on build")

    with patch.object(
        _image_saver_mod, "build_exif_payload", failing_build_exif
    ):
        save_frame_jxl(
            sample_frame, dst, JxlSettings(), metadata={"key": "val"}
        )

    assert dst.exists()

    with PILImage.open(dst) as img:
        assert img.format == "JXL"


# -- Plugin-missing contract -----------------------------------------------

def test_save_frame_jxl_plugin_missing(
    tmp_path: Path, sample_frame: np.ndarray
) -> None:
    """When HAS_JXL_PLUGIN is False, save_frame_jxl raises RuntimeError."""
    dst = tmp_path / "test.jxl"

    with patch.object(_image_saver_mod, "HAS_JXL_PLUGIN", False):
        with pytest.raises(RuntimeError) as exc_info:
            save_frame_jxl(sample_frame, dst, JxlSettings())
        assert "pillow-jxl-plugin" in str(exc_info.value).lower()


# ======================================================================
# save_images_jxl (requires pillow-jxl-plugin)
# ======================================================================

# -- Single frame → bare stem ----------------------------------------------

def test_save_images_jxl_single_frame(
    tmp_path: Path, jxl_plugin_available: bool
) -> None:
    """Single-frame batch writes bare-stem .jxl."""
    frames = [np.full((4, 4, 3), 50.0, dtype=np.float32)]
    results = save_images_jxl(
        frames, str(tmp_path), "bare", JxlSettings(), None, tmp_path
    )
    assert len(results) == 1
    assert results[0].name == "bare.jxl"


# -- Multi-frame batch → indexed names -------------------------------------

def test_save_images_jxl_batch_indexed(
    tmp_path: Path, jxl_plugin_available: bool
) -> None:
    """3-frame batch produces stem_00000..stem_00002."""
    frames = [
        np.full((4, 4, 3), float(i * 50 + 10), dtype=np.float32)
        for i in range(3)
    ]
    results = save_images_jxl(
        frames, str(tmp_path), "stem", JxlSettings(), None, tmp_path
    )
    assert len(results) == 3
    names = sorted(r.name for r in results)
    assert names == ["stem_00000.jxl", "stem_00001.jxl", "stem_00002.jxl"]


# -- Collision suffix on repeated call -------------------------------------

def test_save_images_jxl_collision_suffix(
    tmp_path: Path, jxl_plugin_available: bool
) -> None:
    """Second identical call produces collision-suffixed files."""
    frames = [np.full((4, 4, 3), 50.0, dtype=np.float32)]
    r1 = save_images_jxl(
        frames, str(tmp_path), "stem", JxlSettings(), None, tmp_path
    )
    r2 = save_images_jxl(
        frames, str(tmp_path), "stem", JxlSettings(), None, tmp_path
    )
    assert len(r2) == 1
    assert "_00001.jxl" in r2[0].name
    assert r1[0].exists()
    assert r2[0].exists()


# -- Nested dirs created ---------------------------------------------------

def test_save_images_jxl_nested_dirs(
    tmp_path: Path, jxl_plugin_available: bool
) -> None:
    """Nested target directories are created automatically."""
    frames = [np.full((4, 4, 3), 50.0, dtype=np.float32)]
    results = save_images_jxl(
        frames, "a/b/c", "img", JxlSettings(), None, tmp_path
    )
    assert len(results) == 1
    # The file lives at tmp_path/a/b/c/img.jxl; going up 3 levels from the file
    # parent gives us 'a'.
    assert results[0].parent.parent.parent.name == "a"


# -- All returned paths exist ----------------------------------------------

def test_save_images_jxl_all_paths_exist(
    tmp_path: Path, jxl_plugin_available: bool
) -> None:
    """Every path returned by save_images_jxl exists on disk."""
    frames = [
        np.full((4, 4, 3), float(i * 50 + 10), dtype=np.float32)
        for i in range(3)
    ]
    results = save_images_jxl(
        frames, str(tmp_path), "batch", JxlSettings(), None, tmp_path
    )
    assert all(r.exists() for r in results)


# ======================================================================
# metadata-rejection resilience (generic)
# ======================================================================

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
# Bug-regression: save_frame with None / empty metadata + degrade retry
# ======================================================================


def test_save_jpg_no_metadata_succeeds(
    tmp_path: Path, sample_frame: np.ndarray
) -> None:
    """JPG with metadata=None must not raise; file opens as JPEG.

    Bug: ``save_frame`` passes ``exif=None`` to Pillow which raises
    ``TypeError: object of type 'NoneType' has no len()``.
    """
    if not HAS_PIL:
        pytest.skip("Pillow not installed")

    dst = tmp_path / "test_no_meta.jpg"
    save_frame(sample_frame, dst, "JPG", quality=95, metadata=None)

    from PIL import Image as PILImage

    with PILImage.open(dst) as img:
        assert img.format == "JPEG"


def test_save_jpg_empty_metadata_succeeds(
    tmp_path: Path, sample_frame: np.ndarray
) -> None:
    """JPG with metadata={} (empty dict is falsy → no exif) must succeed.

    Bug: same as above — empty dict evaluates falsy but ``exif=None`` is
    still passed to Pillow's JPEG encoder.
    """
    if not HAS_PIL:
        pytest.skip("Pillow not installed")

    dst = tmp_path / "test_empty_meta.jpg"
    save_frame(sample_frame, dst, "JPG", quality=95, metadata={})

    from PIL import Image as PILImage

    with PILImage.open(dst) as img:
        assert img.format == "JPEG"


def test_save_webp_no_metadata_succeeds(
    tmp_path: Path, sample_frame: np.ndarray
) -> None:
    """WEBP with metadata=None must not raise; file opens as WEBP.

    Bug: ``exif=None`` causes
    ``AttributeError: 'NoneType' object has no attribute 'startswith'``.
    """
    if not HAS_PIL:
        pytest.skip("Pillow not installed")

    dst = tmp_path / "test_webp_no_meta.webp"
    save_frame(sample_frame, dst, "WEBP", quality=95, metadata=None)

    from PIL import Image as PILImage

    with PILImage.open(dst) as img:
        assert img.format == "WEBP"


def test_save_webp_empty_metadata_succeeds(
    tmp_path: Path, sample_frame: np.ndarray
) -> None:
    """WEBP with metadata={} must succeed (empty dict → no exif).

    Bug: same as WEBP-no-metadata — ``exif=None`` passed to Pillow.
    """
    if not HAS_PIL:
        pytest.skip("Pillow not installed")

    dst = tmp_path / "test_webp_empty_meta.webp"
    save_frame(sample_frame, dst, "WEBP", quality=95, metadata={})

    from PIL import Image as PILImage

    with PILImage.open(dst) as img:
        assert img.format == "WEBP"


def test_save_jpg_oversized_metadata_degrades(
    tmp_path: Path,
    sample_frame: np.ndarray,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Oversized EXIF payload on JPG triggers degrade → metadata-less save.

    Bug: the retry path calls ``img.save(dst, format=fmt)`` with fmt="JPG"
    but Pillow only registers "JPEG" → ``KeyError: 'JPG'``.

    The large metadata dict produces a ~80 KB UTF-16 payload that exceeds
    Pillow's 65 533-byte EXIF limit, so the first save raises
    ``ValueError: EXIF data is too long``, the degrade path fires, and
    then dies on the bad format string.
    """
    if not HAS_PIL:
        pytest.skip("Pillow not installed")

    dst = tmp_path / "test_big_meta.jpg"
    meta = {"workflow": "x" * 40_000}

    caplog.set_level(logging.WARNING)

    save_frame(sample_frame, dst, "JPG", quality=95, metadata=meta)

    assert dst.exists()

    from PIL import Image as PILImage

    with PILImage.open(dst) as img:
        assert img.format == "JPEG"
        # Must have been saved without metadata after the retry.
        assert img.getexif().get(0x9286) is None

    assert caplog.record_tuples, (
        "Expected a warning about retrying without metadata"
    )
    retry_msgs = [
        r.message for r in caplog.records if "retrying without metadata" in r.message
    ]
    assert retry_msgs


def test_save_webp_large_metadata_roundtrip(
    tmp_path: Path, sample_frame: np.ndarray
) -> None:
    """WEBP stores the same large payload and round-trips successfully.

    WEBP has no 64 KB EXIF cap in Pillow so this test should PASS both
    before and after the fix — it pins the WEBP/JPG asymmetry.
    """
    if not HAS_PIL:
        pytest.skip("Pillow not installed")

    dst = tmp_path / "test_webp_big.webp"
    meta = {"workflow": "x" * 40_000}

    save_frame(sample_frame, dst, "WEBP", quality=95, metadata=meta)

    assert dst.exists()

    from PIL import Image as PILImage

    with PILImage.open(dst) as img:
        exif = img.getexif()
        raw = exif.get(0x9286)
        assert raw is not None, "EXIF UserComment missing in WEBP"
        data = raw[8:].decode("utf-16").rstrip("\x00")
        loaded = json.loads(data)
        assert loaded == meta


def test_save_frame_jxl_no_metadata_succeeds(
    tmp_path: Path, sample_frame: np.ndarray, jxl_plugin_available: bool
) -> None:
    """metadata-less JXL save keeps working after the fix.

    Pins the contract that ``save_frame_jxl(..., metadata=None)`` produces
    a valid file without raising.
    """
    from PIL import Image as PILImage

    dst = tmp_path / "test_jxl_no_meta.jxl"
    save_frame_jxl(sample_frame, dst, JxlSettings(), metadata=None)

    assert dst.exists()

    with PILImage.open(dst) as img:
        assert img.format == "JXL"


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
