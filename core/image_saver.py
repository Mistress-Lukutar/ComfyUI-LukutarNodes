'''
File:   image_saver.py
Brief:  Pure engine for saving numpy frames to disk in various image formats.
Author: Mistress-Lukutar
Date:   2026-11-02
Version: v0.1.1
'''

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

try:
    import pillow_jxl  # type: ignore[import-not-found]  # noqa: F401
    HAS_JXL_PLUGIN: bool = True
except ImportError:
    HAS_JXL_PLUGIN = False
try:
    from PIL import (  # type: ignore[import-not-found]
        Image,
        PngImagePlugin,
        TiffImagePlugin,
    )
    HAS_PIL: bool = True
except ImportError:
    HAS_PIL = False

logger = logging.getLogger(__name__)

#: Display order for save formats.
SAVE_FORMATS: tuple[str, ...] = ("PNG", "JPG", "JXL", "WEBP", "TIFF", "BMP")

#: Mapping from display name to file-extension (with dot).
EXTENSIONS: dict[str, str] = {
    "PNG": ".png",
    "JPG": ".jpg",
    "JXL": ".jxl",
    "WEBP": ".webp",
    "TIFF": ".tif",
    "BMP": ".bmp",
}

#: Formats that cannot embed any metadata dict.
METADATALESS_FORMATS: tuple[str, ...] = ("BMP",)

#: Default JPEG/WEBP/JXL quality when not specified by the user.
DEFAULT_QUALITY: int = 95

#: Socket type consumed from the metadata inputs (mirrors core.metadata.METADATA_TYPE).
METADATA_TYPE: str = "LUKUTAR_METADATA"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def available_formats() -> tuple[str, ...]:
    '''Return the list of save formats minus JXL when the plugin is absent.

    Returns:
        Tuple of format names usable on this runtime.
    '''
    if HAS_JXL_PLUGIN:
        return SAVE_FORMATS
    return tuple(f for f in SAVE_FORMATS if f != "JXL")


def supports_metadata(fmt: str) -> bool:
    '''Check whether *fmt* can embed a metadata dict.

    Args:
        fmt: Format name (e.g. ``"PNG"``).

    Returns:
        ``True`` unless the format is in :data:`METADATALESS_FORMATS`.
    '''
    return fmt not in METADATALESS_FORMATS


def sanitize_filename(name: str) -> str:
    '''Sanitise a string for use as a file stem on Windows.

    Args:
        name: Raw filename stem.

    Returns:
        A cleaned version safe for filesystem creation. Returns ``"image"``
        when the result is otherwise empty.
    '''
    name = name.strip()
    # Replace forbidden / control characters with underscore.
    name = re.sub(r'[<>:"/\\|?\x00-\x1f]', '_', name)
    # Collapse whitespace runs to a single space.
    name = re.sub(r'\s+', ' ', name)
    # Strip trailing dots and spaces (Windows restriction).
    name = name.rstrip('. ')
    return name if name else "image"


def normalize_dir(path: str, base_dir: Path) -> Path:
    '''Normalise a user-supplied path against *base_dir*.

    Args:
        path: Raw path string from the widget.
        base_dir: ComfyUI root (``folder_paths.base_path``).

    Returns:
        An absolute :class:`Path`.
    '''
    stripped = path.strip().strip('"').strip("'")
    if not stripped or not stripped.strip():
        return base_dir
    p = Path(stripped)
    if p.is_absolute():
        return p.expanduser()
    return (base_dir / p).expanduser()


def unique_destination(directory: Path, stem: str, ext: str) -> Path:
    '''Return a free path under *directory* named ``stem + ext``.

    When the exact name is taken, appends a zero-padded five-digit
    counter (``_00001``, ``_00002``, …).

    Args:
        directory: Target directory (must exist or be creatable).
        stem: Filename stem without extension.
        ext: Extension including the leading dot (e.g. ``".png"``).

    Returns:
        A :class:`Path` that does not yet exist on disk.
    '''
    candidate = directory / f"{stem}{ext}"
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        candidate = directory / f"{stem}_{counter:05d}{ext}"
        if not candidate.exists():
            return candidate
        counter += 1


# ---------------------------------------------------------------------------
# Metadata payload builders
# ---------------------------------------------------------------------------

def build_png_info(metadata: Mapping[str, str]) -> PngImagePlugin.PngInfo:
    '''Build a ``PngInfo`` with one tEXt chunk per metadata item.

    Args:
        metadata: Key-value pairs to embed.

    Returns:
        A populated :class:`PngImagePlugin.PngInfo`.

    Raises:
        RuntimeError: When Pillow is not available.
    '''
    if not HAS_PIL:
        raise RuntimeError("Pillow is required to build PNG metadata")
    info = PngImagePlugin.PngInfo()
    for key, value in metadata.items():
        info.add_text(key, str(value))
    return info


def build_exif_payload(metadata: Mapping[str, str]) -> bytes:
    '''Build an EXIF TIFF blob carrying *metadata* as JSON in UserComment.

    The payload starts with ``b"UNICODE\\x00"`` followed by the UTF-16
    (with BOM) encoded JSON string — matching A1111-style readers.

    Args:
        metadata: Key-value pairs to embed.

    Returns:
        Raw EXIF bytes ready for ``Image.save(exif=...)``.

    Raises:
        RuntimeError: When Pillow is not available.
    '''
    if not HAS_PIL:
        raise RuntimeError("Pillow is required to build EXIF metadata")
    exif = Image.Exif()
    charset = b"UNICODE\x00"
    data = json.dumps(metadata, ensure_ascii=False).encode("utf-16")
    exif[0x9286] = charset + data  # UserComment tag
    return exif.tobytes()


# ---------------------------------------------------------------------------
# Frame saver
# ---------------------------------------------------------------------------

def save_frame(
    frame: np.ndarray,
    dst: Path,
    fmt: str,
    quality: int = DEFAULT_QUALITY,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    '''Save a single HWC RGB numpy frame to *dst*.

    Args:
        frame: HWC RGB array (float or uint8).
        dst: Destination path.
        fmt: Format name (one of :data:`SAVE_FORMATS`).
        quality: JPEG/WEBP/JXL quality (1-100). Ignored by PNG, TIFF, BMP.
        metadata: Optional key-value pairs to embed in the file.

    Returns:
        The absolute *dst* path.

    Raises:
        ValueError: On unknown *fmt*.
        RuntimeError: When Pillow is missing or save fails twice.
    '''
    if fmt not in SAVE_FORMATS:
        raise ValueError(f"Unknown format: {fmt}")
    if not HAS_PIL:
        raise RuntimeError("Pillow is required to save frames")

    cleaned = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
    img = Image.fromarray(cleaned, mode="RGB")

    def _save_with_meta() -> None:
        if fmt == "PNG":
            pnginfo = build_png_info(metadata) if metadata else None
            img.save(dst, format="PNG", pnginfo=pnginfo)
        elif fmt == "JPG":
            exif = build_exif_payload(metadata) if metadata else None
            img.save(dst, format="JPEG", quality=quality, exif=exif)
        elif fmt == "WEBP":
            exif = build_exif_payload(metadata) if metadata else None
            img.save(dst, format="WEBP", quality=quality, exif=exif)
        elif fmt == "TIFF":
            tiffinfo = None
            if metadata:
                td = TiffImagePlugin.ImageFileDirectory_v2()
                td[270] = json.dumps(metadata, ensure_ascii=False)
                tiffinfo = td
            img.save(dst, format="TIFF", tiffinfo=tiffinfo)
        elif fmt == "JXL":
            exif = build_exif_payload(metadata) if metadata else None
            img.save(dst, format="JXL", quality=quality, exif=exif)
        elif fmt == "BMP":
            img.save(dst, format="BMP")

    # Attempt save with metadata; degrade once on failure.
    try:
        _save_with_meta()
    except Exception as orig_exc:
        if metadata is not None:
            logger.warning(
                "Save %s with metadata failed (%s); retrying without metadata",
                dst, fmt,
            )
            try:
                img.save(dst, format=fmt)
            except Exception:
                raise RuntimeError(
                    f"Failed to save {dst} ({fmt})"
                ) from orig_exc
        else:
            raise

    return dst


# ---------------------------------------------------------------------------
# Batch saver
# ---------------------------------------------------------------------------

def save_images(
    frames: Sequence[np.ndarray],
    path: str,
    filename: str,
    fmt: str,
    quality: int,
    metadata: Mapping[str, str] | None,
    base_dir: Path,
) -> list[Path]:
    '''Save a batch of frames to disk.

    Args:
        frames: Sequence of HWC RGB numpy arrays.
        path: User-supplied directory string.
        filename: Base filename stem (extension added automatically).
        fmt: Target format name.
        quality: JPEG/WEBP/JXL quality value.
        metadata: Optional metadata dict to embed in each file.
        base_dir: ComfyUI root for resolving relative paths.

    Returns:
        List of destination :class:`Path` objects (one per frame).

    Raises:
        RuntimeError: When any individual frame fails to save.
    '''
    target_dir = normalize_dir(path, base_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    base_stem = sanitize_filename(filename)
    ext = EXTENSIONS.get(fmt, ".png")
    results: list[Path] = []

    single = len(frames) == 1

    for i, frame in enumerate(frames):
        stem = base_stem if single else f"{base_stem}_{i:05d}"

        dst = unique_destination(target_dir, stem, ext)

        try:
            save_frame(frame, dst, fmt, quality, metadata)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to save frame {i} to {dst}"
            ) from exc

        results.append(dst)

    return results
