'''
File:   save_image_jxl.py
Brief:  ComfyUI node for saving IMAGE batches as JPEG-XL with full encoder control.
Author: Mistress-Lukutar
Date:   2026-11-02
Version: v0.2.1
'''

from __future__ import annotations

import logging

try:
    import folder_paths  # ComfyUI runtime only
    HAS_COMFY: bool = True
except ImportError:
    folder_paths = None  # type: ignore[assignment]
    HAS_COMFY = False

from ..core.image_saver import (
    METADATA_TYPE,
    JxlSettings,
    save_images_jxl,
)
from ..utils.images import tensor_to_frames
from .save_image import _comfy_base_dir, build_ui_entries

logger = logging.getLogger(__name__)


class SaveImageJxlNode:
    '''Dedicated JPEG-XL saver backed by pillow-jxl-plugin.

    Supports both lossy and lossless modes with full encoder control.
    Metadata is embedded as an EXIF box.  Raises a clear
    :exc:`RuntimeError` at execution when the plugin is missing.

    Pass-through outputs: ``images`` (the original IMAGE tensor, untouched)
    and ``metadata`` (the LUKUTAR_METADATA dict or ``None``).
    '''

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
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
                        "default": "img",
                        "tooltip": "Base file name without extension.",
                    },
                ),
                "lossless": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Lossless (mathematically exact) encoding. "
                            "The quality slider is ignored when this is on."
                        ),
                    },
                ),
                "quality": (
                    "INT",
                    {
                        "default": 90,
                        "min": 0,
                        "max": 100,
                        "tooltip": (
                            "JPEG-style lossy quality; 100 is visually "
                            "lossless. Ignored in lossless mode."
                        ),
                    },
                ),
                "effort": (
                    "INT",
                    {
                        "default": 7,
                        "min": 1,
                        "max": 9,
                        "tooltip": (
                            "Encoder effort: higher values compress better "
                            "but run slower."
                        ),
                    },
                ),
                "decoding_speed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4,
                        "tooltip": (
                            "Decode-speed tier: 0 = smallest files, slowest "
                            "decode ... 4 = fastest decode, larger files."
                        ),
                    },
                ),
                "use_container": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Wrap the codestream in the JXL container format.",
                    },
                ),
                "use_original_profile": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Keep the original color profile "
                            "(skip YCbCr conversion)."
                        ),
                    },
                ),
                "num_threads": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 64,
                        "tooltip": "Encoder threads; -1 = automatic.",
                    },
                ),
                "compress_metadata": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Compress the embedded metadata boxes.",
                    },
                ),
            },
            "optional": {
                "metadata": (
                    METADATA_TYPE,
                    {
                        "tooltip": (
                            "Metadata blocks written into the file "
                            "as an EXIF box."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", METADATA_TYPE)
    RETURN_NAMES = ("images", "metadata")
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
        lossless: bool,
        quality: int,
        effort: int,
        decoding_speed: int,
        use_container: bool,
        use_original_profile: bool,
        num_threads: int,
        compress_metadata: bool,
        metadata: dict[str, str] | None = None,
    ) -> dict:
        '''Save the image batch as JPEG-XL to disk.

        Uses the ``pillow-jxl-plugin`` encoder.  In lossless mode the
        quality parameter is ignored (forced to 100 internally).

        Args:
            images: ComfyUI IMAGE tensor (B, H, W, 3), float in [0, 1].
            path: Target directory string.
            filename: Base filename stem.
            lossless: Lossless encoding flag.
            quality: JPEG-style lossy quality; ignored in lossless mode.
            effort: Encoder effort (1-9).
            decoding_speed: Decode-speed tier (0-4).
            use_container: Wrap codestream in JXL container.
            use_original_profile: Keep original color profile.
            num_threads: Encoder threads (-1 = auto).
            compress_metadata: Compress embedded metadata boxes.
            metadata: Optional metadata dict to embed as EXIF.

        Returns:
            Dict with ``ui.images`` for the ComfyUI preview panel and
            ``result`` = ``(images, metadata)`` pass-through.
        '''
        settings = JxlSettings(
            lossless=lossless,
            quality=quality,
            effort=effort,
            decoding_speed=decoding_speed,
            use_container=use_container,
            use_original_profile=use_original_profile,
            num_threads=num_threads,
            compress_metadata=compress_metadata,
        )

        frames = tensor_to_frames(images)  # type: ignore[arg-type]
        saved = save_images_jxl(
            frames, path, filename, settings, metadata, _comfy_base_dir()
        )

        return {"ui": {"images": build_ui_entries(saved)}, "result": (images, metadata)}
