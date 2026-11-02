'''
File:   save_image.py
Brief:  ComfyUI node for saving IMAGE batches to disk in various formats.
Author: Mistress-Lukutar
Date:   2026-11-02
Version: v0.2.0
'''

from __future__ import annotations

import logging
from pathlib import Path

try:
    import folder_paths  # ComfyUI runtime only
    HAS_COMFY: bool = True
except ImportError:
    folder_paths = None  # type: ignore[assignment]
    HAS_COMFY = False

from ..core.image_saver import (
    DEFAULT_QUALITY,
    METADATA_TYPE,
    available_formats,
    save_images,
    supports_metadata,
)
from ..utils.images import tensor_to_frames

logger = logging.getLogger(__name__)


def _comfy_base_dir() -> Path:
    '''Return the ComfyUI root directory.

    Returns:
        Absolute :class:`Path` to ``folder_paths.base_path`` or cwd.
    '''
    if HAS_COMFY:
        return Path(getattr(folder_paths, "base_path", Path.cwd()))
    return Path.cwd()


def build_ui_entries(saved: list[Path]) -> list[dict[str, str]]:
    '''Build the ``ui.images`` list for ComfyUI's preview panel.

    Each entry maps a saved file to its subfolder and output type so
    that ComfyUI can display it in the history / output viewer.

    Args:
        saved: List of destination :class:`Path` objects returned by
            a saver function.

    Returns:
        List of dicts with keys ``filename``, ``subfolder``, ``type``.
    '''
    entries: list[dict[str, str]] = []
    if HAS_COMFY:
        output_dir = folder_paths.get_output_directory()
        for p in saved:
            try:
                resolved = p.resolve()
                out_resolved = Path(output_dir).resolve()
                if resolved.is_relative_to(out_resolved):
                    rel_parent = resolved.parent.relative_to(
                        out_resolved
                    )
                    subfolder = rel_parent.as_posix()
                    if subfolder == ".":
                        subfolder = ""
                    entries.append(
                        {
                            "filename": p.name,
                            "subfolder": subfolder,
                            "type": "output",
                        }
                    )
            except (ValueError, OSError):
                pass
    return entries


class SaveImageNode:
    '''Save IMAGE batches to the filesystem in a chosen format.

    Writes one file per frame (appending a five-digit index when the
    batch has more than one frame).  Optionally embeds metadata key-value
    pairs into the file headers where supported.
    '''

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        fmt_list = available_formats()
        meta_fmts = [f for f in fmt_list if supports_metadata(f)]
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch of frames to save."}),
                "path": (
                    "STRING",
                    {
                        "default": "output",
                        "tooltip": (
                            "Target directory. Relative paths are resolved "
                            "against the ComfyUI root "
                            "(folder_paths.base_path); absolute paths are "
                            "used as-is."
                        ),
                    },
                ),
                "filename": (
                    "STRING",
                    {
                        "default": "Lukutar",
                        "tooltip": "Base file name without extension.",
                    },
                ),
                "format": (
                    fmt_list,
                    {
                        "default": "PNG",
                        "tooltip": (
                            "Output image format.  Metadata is embedded for "
                            f"{meta_fmts}."
                        ),
                        "metadata_formats": meta_fmts,
                    },
                ),
                "quality": (
                    "INT",
                    {
                        "default": DEFAULT_QUALITY,
                        "min": 1,
                        "max": 100,
                        "tooltip": (
                            "Used by JPG and WEBP; ignored by PNG, TIFF and BMP."
                        ),
                    },
                ),
            },
            "optional": {
                "metadata": (
                    METADATA_TYPE,
                    {
                        "tooltip": (
                            "Metadata blocks written into the file. Hidden "
                            "for formats that cannot store metadata."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Lukutar/Image"

    @classmethod
    def IS_CHANGED(cls, **kwargs: object) -> float:
        '''Always re-execute (mirrors ComfyUI's own SaveImage behaviour).

        Returns:
            ``nan`` to force a fresh execution every time.
        '''
        return float("nan")

    def save(
        self,
        images: object,
        path: str,
        filename: str,
        format: str,  # noqa: A002 — shadows builtin by necessity
        quality: int,
        metadata: dict[str, str] | None = None,
    ) -> dict:
        '''Save the image batch to disk.

        Args:
            images: ComfyUI IMAGE tensor (B, H, W, 3), float in [0, 1].
            path: Target directory string.
            filename: Base filename stem.
            format: Output format name.
            quality: JPEG/WEBP quality value.
            metadata: Optional metadata dict to embed.

        Returns:
            Dict with ``ui.images`` for the ComfyUI preview panel.
        '''
        if metadata and not supports_metadata(format):
            logger.warning(
                "Format %s does not support metadata — embedding skipped",
                format,
            )

        frames = tensor_to_frames(images)  # type: ignore[arg-type]
        saved = save_images(
            frames, path, filename, format, quality, metadata, _comfy_base_dir()
        )

        return {"ui": {"images": build_ui_entries(saved)}}
