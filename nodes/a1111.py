'''
File:   a1111.py
Brief:  ComfyUI nodes for A1111-style generation-parameter metadata.
Author: Mistress-Lukutar
Date:   2026-11-02
Version: v0.1.0
'''

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import folder_paths  # ComfyUI runtime only
except ImportError:
    folder_paths = None  # type: ignore[assignment]

from ..core.a1111_parameters import (
    A1111_SAMPLERS,
    DEFAULT_SAMPLER,
    a1111_float_str,
    build_parameters,
    get_model_hash_cache,
)
from ..core.metadata import (
    METADATA_TYPE,
    merge_metadata,
    sort_meta_inputs,
    sort_numbered_inputs,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Module constants                                                   #
# ------------------------------------------------------------------ #

#: Fallback combo entry marking an unused LoRA slot.
LORA_NONE: str = "None"

#: Model types offered by the A1111 Model Info node.
MODEL_TYPES: tuple[str, ...] = ("Checkpoint", "VAE", "LoRA", "Embedding")

#: Mapping from model type label to the corresponding ``folder_paths`` folder name.
_MODEL_TYPE_FOLDERS: dict[str, str] = {
    "Checkpoint": "checkpoints",
    "VAE": "vae",
    "LoRA": "loras",
    "Embedding": "embeddings",
}


# ======================================================================
# Node classes                                                         #
# ======================================================================


class A1111ParametersNode:
    '''Build an A1111-style infotext string and metadata block.

    Combines positive/negative prompts, optional version, and any number
    of ``meta_N`` metadata blocks into a single A1111 settings line.
    '''

    CATEGORY = "Lukutar/Metadata"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "positive": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Positive prompt; may span multiple lines.",
                    },
                ),
                "negative": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Negative prompt; the 'Negative prompt:' line is "
                            "omitted when empty."
                        ),
                    },
                ),
                "version": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Value for the trailing 'Version:' entry "
                            "(e.g. 'forge-0.9.1'). Empty = omit."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = (METADATA_TYPE, "STRING")
    RETURN_NAMES = ("metadata", "parameters_text")
    FUNCTION = "build"

    def build(
        self,
        positive: str,
        negative: str,
        version: str,
        **kwargs: Any,
    ) -> tuple[dict[str, str], str]:
        '''Build the A1111 parameters string and metadata block.

        Args:
            positive: The positive prompt text.
            negative: The negative prompt text.
            version: Optional version string for the trailing entry.
            **kwargs: Optional ``meta_N`` metadata blocks.

        Returns:
            Tuple of (metadata dict, parameters text string).
        '''
        ordered = sort_meta_inputs(kwargs)
        params = merge_metadata(
            *(kwargs[k] for k in ordered if kwargs[k] is not None)
        )

        version = str(version).strip()
        if version:
            params["Version"] = version

        text = build_parameters(positive, negative, params)

        result: dict[str, str] = {"parameters": text} if text else {}

        return (result, text)


class A1111SamplingNode:
    '''Emit sampling parameters as a metadata block.

    Produces the standard A1111 settings entries for sampler configuration.
    '''

    CATEGORY = "Lukutar/Metadata"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 150,
                        "tooltip": "Number of inference steps.",
                    },
                ),
                "sampler": (
                    A1111_SAMPLERS,
                    {
                        "default": DEFAULT_SAMPLER,
                        "tooltip": (
                            "Sampler algorithm. Custom names can be added via "
                            "'convert widget to input'."
                        ),
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.1,
                        "tooltip": "Classifier-free guidance scale.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        # The frontend adds control_after_generate on its
                        # own; declaring it server-side would break
                        # validate_prompt (see Seed History).
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Generation seed.",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": 16384,
                        "step": 8,
                        "tooltip": "0 = omit the Size entry.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": 16384,
                        "step": 8,
                        "tooltip": "Image height in pixels.",
                    },
                ),
                "clip_skip": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 12,
                        "tooltip": "0 = omit the Clip skip entry.",
                    },
                ),
            }
        }

    RETURN_TYPES = (METADATA_TYPE,)
    RETURN_NAMES = ("metadata",)
    FUNCTION = "build"

    def build(
        self,
        steps: int,
        sampler: str,
        cfg: float,
        seed: int,
        width: int,
        height: int,
        clip_skip: int,
    ) -> tuple[dict[str, str]]:
        '''Build the sampling metadata block.

        Args:
            steps: Number of inference steps.
            sampler: Sampler algorithm name.
            cfg: Classifier-free guidance scale.
            seed: Generation seed value.
            width: Image width in pixels (0 = omit Size).
            height: Image height in pixels (0 = omit Size).
            clip_skip: CLIP skip value (0 = omit Clip skip).

        Returns:
            One-element tuple with the metadata dict.
        '''
        result: dict[str, str] = {}
        result["Steps"] = str(int(steps))
        result["Sampler"] = str(sampler)
        result["CFG scale"] = f"{float(cfg):g}"
        result["Seed"] = str(int(seed))

        if width > 0 and height > 0:
            result["Size"] = f"{int(width)}x{int(height)}"

        if clip_skip >= 1:
            result["Clip skip"] = str(int(clip_skip))

        return (result,)


class A1111HiresFixNode:
    '''Emit hires-fix parameters as a metadata block.

    Produces the standard A1111 settings entries for high-resolution fix.
    '''

    CATEGORY = "Lukutar/Metadata"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "upscaler": (
                    "STRING",
                    {
                        "default": "4x-UltraSharp",
                        "tooltip": "A1111 upscaler display name, e.g. '4x-UltraSharp'.",
                    },
                ),
                "upscale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 8.0,
                        "step": 0.05,
                        "tooltip": "Upscaling factor.",
                    },
                ),
                "hires_steps": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 150,
                        "tooltip": "0 = omit the 'Hires steps' entry.",
                    },
                ),
                "denoising": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Denoising strength for the hires pass.",
                    },
                ),
            }
        }

    RETURN_TYPES = (METADATA_TYPE,)
    RETURN_NAMES = ("metadata",)
    FUNCTION = "build"

    def build(
        self,
        upscaler: str,
        upscale: float,
        hires_steps: int,
        denoising: float,
    ) -> tuple[dict[str, str]]:
        '''Build the hires-fix metadata block.

        Args:
            upscaler: Upscaler display name.
            upscale: Upscaling factor.
            hires_steps: Hires pass steps (0 = omit).
            denoising: Denoising strength.

        Returns:
            One-element tuple with the metadata dict.
        '''
        result: dict[str, str] = {}
        result["Hires upscale"] = a1111_float_str(upscale)
        result["Hires upscaler"] = str(upscaler)

        if hires_steps >= 1:
            result["Hires steps"] = str(int(hires_steps))

        result["Denoising strength"] = a1111_float_str(denoising)

        return (result,)


class A1111ModelInfoNode:
    '''Emit model hash and name information as a metadata block.

    Looks up the file in ``folder_paths``, computes its SHA-256 prefix,
    and emits type-specific metadata entries matching A1111 conventions.
    '''

    CATEGORY = "Lukutar/Metadata"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        if folder_paths is not None:
            choices = folder_paths.get_filename_list("checkpoints")
            if not choices:
                choices = [""]
            model_name_choices = tuple(choices)
        else:
            model_name_choices = (
                "(model list requires the ComfyUI runtime)",
            )

        return {
            "required": {
                "model_type": (
                    MODEL_TYPES,
                    {
                        "default": "Checkpoint",
                        "tooltip": (
                            "Which model folder to list. The web extension "
                            "refills the file list when this changes; without "
                            "it the combo shows checkpoints."
                        ),
                    },
                ),
                "model_name": (
                    model_name_choices,
                    {
                        "default": model_name_choices[0] if model_name_choices else "",
                        "tooltip": "Model file to look up.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", METADATA_TYPE)
    RETURN_NAMES = ("model_name", "model_hash", "metadata")
    FUNCTION = "build"

    def build(
        self,
        model_type: str,
        model_name: str,
    ) -> tuple[str, str, dict[str, str]]:
        '''Build the model info metadata block.

        Args:
            model_type: One of ``MODEL_TYPES``.
            model_name: Filename within the selected folder.

        Returns:
            Tuple of (display name, hash string, metadata dict).

        Raises:
            RuntimeError: When called outside the ComfyUI runtime.
        '''
        if folder_paths is None:
            raise RuntimeError(
                "A1111 Model Info requires the ComfyUI runtime (folder_paths)."
            )

        folder = _MODEL_TYPE_FOLDERS.get(model_type, "checkpoints")
        path = folder_paths.get_full_path_or_raise(folder, model_name)
        h = get_model_hash_cache().hash(path)

        if model_type == "Checkpoint":
            name = os.path.splitext(model_name)[0]
            meta: dict[str, str] = {
                "Model hash": h,
                "Model": name,
            }
        elif model_type == "VAE":
            name = model_name  # A1111 keeps extension for VAE
            meta = {
                "VAE hash": h,
                "VAE": name,
            }
        elif model_type == "LoRA":
            name = os.path.splitext(model_name)[0]
            meta = {"Lora hashes": f"{name}: {h}"}
        elif model_type == "Embedding":
            name = os.path.splitext(model_name)[0]
            meta = {"TI hashes": f"{name}: {h}"}
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return (name, h, meta)


class A1111LoraHashesNode:
    '''Compute and emit LoRA hashes as a metadata block.

    Up to 8 LoRA slots are accepted; unused slots default to ``"None"``.
    The value is stored raw; A1111 quoting (``quote_a1111``) is applied once
    at serialization time by the A1111 Parameters node.
    '''

    CATEGORY = "Lukutar/Metadata"

    #: Maximum number of LoRA slots the node accepts.
    MAX_LORA_SLOTS: int = 8

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        if folder_paths is not None:
            lora_choices = (LORA_NONE,) + tuple(
                folder_paths.get_filename_list("loras")
            )
        else:
            lora_choices = (
                LORA_NONE,
                "(model list requires the ComfyUI runtime)",
            )

        required: dict[str, tuple] = {}
        for i in range(1, cls.MAX_LORA_SLOTS + 1):
            tooltip = (
                "LoRA file; 'None' skips the slot. "
                "The web extension hides unused slots."
                if i == 1
                else ""
            )
            required[f"lora_{i}"] = (
                lora_choices,
                {
                    "default": LORA_NONE,
                    "tooltip": tooltip,
                },
            )

        return {"required": required}

    RETURN_TYPES = (METADATA_TYPE, "STRING")
    RETURN_NAMES = ("metadata", "lora_hashes")
    FUNCTION = "build"

    def build(self, **kwargs: Any) -> tuple[dict[str, str], str]:
        '''Build the LoRA hashes metadata block.

        Args:
            **kwargs: The ``lora_N`` inputs from ComfyUI.

        Returns:
            Tuple of (metadata dict, joined hash string).

        Raises:
            RuntimeError: When a non-empty slot is used outside the runtime.
        '''
        ordered = sort_numbered_inputs(kwargs, "lora_")

        parts: list[str] = []
        has_real_slot = False

        for name in ordered:
            value = kwargs[name]
            if value is None:
                continue
            str_value = str(value).strip()
            if not str_value or str_value == LORA_NONE:
                continue
            has_real_slot = True
            parts.append(str_value)

        # All slots skipped — return empty without touching folder_paths.
        if not has_real_slot:
            return ({}, "")

        if folder_paths is None:
            raise RuntimeError(
                "A1111 LoRA Hashes requires the ComfyUI runtime (folder_paths)."
            )

        hashes: list[str] = []
        for value in parts:
            path = folder_paths.get_full_path_or_raise("loras", value)
            h = get_model_hash_cache().hash(path)
            stem = os.path.splitext(value)[0]
            hashes.append(f"{stem}: {h}")

        joined = ", ".join(hashes)
        return ({"Lora hashes": joined}, joined)
