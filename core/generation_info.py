'''
File:   generation_info.py
Brief:  Generation-info primitives for the LUKUTAR_INFO socket type.
Author: Mistress-Lukutar
Date:   2026-11-03
Version: v0.1.0

The bundle node set threads a portable "info" entity through the
graph: an insertion-ordered ``dict[str, str]`` describing everything
that was fed into and produced by the pipeline.  Keys deliberately
reuse the A1111 infotext names (``Model``, ``Model hash``, ``Steps``,
...) so the ``InfoToA1111`` node can pass the mapping straight into
the A1111 parameters engine (``core/a1111_parameters.py``).

Prompts are stored under the lowercase keys :data:`PROMPT_KEY` /
:data:`NEGATIVE_KEY` and are excluded from the settings block, exactly
like the A1111 infotext format itself.
'''

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

#: The ComfyUI socket type name shared by all info nodes.
INFO_TYPE: str = "LUKUTAR_INFO"

#: Positive prompt key (excluded from the A1111 settings block).
PROMPT_KEY: str = "prompt"

#: Negative prompt key (excluded from the A1111 settings block).
NEGATIVE_KEY: str = "negative_prompt"

#: Checkpoint file name key.
MODEL_KEY: str = "Model"

#: Checkpoint short hash key.
MODEL_HASH_KEY: str = "Model hash"

#: VAE selection key ("baked" or a VAE file name).
VAE_KEY: str = "VAE"

#: CLIP skip layer key (negative INT as string, e.g. "-2").
CLIP_SKIP_KEY: str = "Clip skip"

#: Applied LoRA list with weights, ``"name:weight"`` comma-joined.
LORA_KEY: str = "Lora"

#: LoRA short hashes, A1111 format ``"name: hash"`` comma-joined.
LORA_HASHES_KEY: str = "Lora hashes"

#: Latent geometry key, ``"{width} x {height}"``.
SIZE_KEY: str = "Size"

#: Batch size key.
BATCH_KEY: str = "Batch size"

#: Token normalization option key (see core/weighted_prompt.py).
TOKEN_NORM_KEY: str = "Token normalization"

#: Weight interpretation option key (see core/weighted_prompt.py).
WEIGHT_INTERP_KEY: str = "Weight interpretation"

#: Sampling seed key.
SEED_KEY: str = "Seed"

#: Sampling steps key.
STEPS_KEY: str = "Steps"

#: CFG scale key.
CFG_KEY: str = "CFG scale"

#: ComfyUI sampler name key.
SAMPLER_KEY: str = "Sampler"

#: ComfyUI scheduler name key.
SCHEDULER_KEY: str = "Schedule type"

#: Denoise strength key.
DENOISE_KEY: str = "Denoising strength"

#: VAE decode flag key ("enabled"/"disabled").
VAE_DECODE_KEY: str = "VAE decode"

#: Keys holding the raw prompts (never emitted as settings pairs).
_PROMPT_KEYS: tuple[str, ...] = (PROMPT_KEY, NEGATIVE_KEY)


def format_lora_entries(
    entries: Iterable[tuple[str, Any]],
) -> str:
    '''Format ``(name, weight)`` LoRA pairs as an info value.

    Args:
        entries: Iterable of ``(name, weight)`` pairs; zero-weight
            entries are skipped, names are stripped.

    Returns:
        ``"name1:weight1, name2:weight2"`` (weights via ``str``),
        ``""`` when nothing remains.
    '''
    parts: list[str] = []
    for name, weight in entries:
        name = str(name).strip()
        if not name:
            continue
        weight_f = float(weight)
        if weight_f == 0.0:
            continue
        parts.append(f"{name}:{a1111_weight_str(weight_f)}")
    return ", ".join(parts)


def a1111_weight_str(weight: float) -> str:
    '''Render a LoRA weight without float noise.

    Args:
        weight: The weight value.

    Returns:
        ``"1.0"`` style string with trailing zeros trimmed to at most
        two decimals (``1`` -> ``"1"`` is avoided: ``"1.0"`` kept).
    '''
    text = f"{weight:.2f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text


def format_lora_hashes(
    entries: Iterable[tuple[str, str]],
) -> str:
    '''Format ``(name, hash)`` pairs in the A1111 ``Lora hashes`` style.

    Args:
        entries: Iterable of ``(name, hash)`` pairs; empty names or
            empty hashes are skipped.

    Returns:
        ``"name1: hash1, name2: hash2"``, ``""`` when nothing remains.
    '''
    parts: list[str] = []
    for name, hash_str in entries:
        name = str(name).strip()
        hash_str = str(hash_str).strip()
        if not name or not hash_str:
            continue
        parts.append(f"{name}: {hash_str}")
    return ", ".join(parts)


def size_str(width: int, height: int) -> str:
    '''Render the ``Size`` info value.

    Args:
        width: Latent geometry width in pixels.
        height: Latent geometry height in pixels.

    Returns:
        ``"{width} x {height}"``.
    '''
    return f"{int(width)} x {int(height)}"


def build_loader_info(
    prompt: str,
    negative: str,
    model_name: str,
    model_hash: str,
    vae_name: str,
    clip_skip: int,
    lora_entries: Iterable[tuple[str, Any]],
    lora_hashes: Iterable[tuple[str, str]],
    width: int,
    height: int,
    batch_size: int,
    token_normalization: str,
    weight_interpretation: str,
) -> dict[str, str]:
    '''Build the info dict emitted by the checkpoint bundle loader.

    Args:
        prompt: Positive prompt text.
        negative: Negative prompt text.
        model_name: Checkpoint combo name.
        model_hash: Checkpoint short hash (may be empty when hashing is
            unavailable).
        vae_name: ``"baked"`` or a VAE file name.
        clip_skip: Negative CLIP layer index.
        lora_entries: ``(name, weight)`` pairs applied by the loader.
        lora_hashes: ``(name, hash)`` pairs for the LoRA files.
        width: Latent width in pixels.
        height: Latent height in pixels.
        batch_size: Requested batch size.
        token_normalization: Prompt engine option.
        weight_interpretation: Prompt engine option.

    Returns:
        A fresh insertion-ordered ``dict[str, str]``.
    '''
    return {
        PROMPT_KEY: str(prompt),
        NEGATIVE_KEY: str(negative),
        MODEL_KEY: str(model_name),
        MODEL_HASH_KEY: str(model_hash),
        VAE_KEY: str(vae_name),
        CLIP_SKIP_KEY: str(int(clip_skip)),
        LORA_KEY: format_lora_entries(lora_entries),
        LORA_HASHES_KEY: format_lora_hashes(lora_hashes),
        SIZE_KEY: size_str(width, height),
        BATCH_KEY: str(int(batch_size)),
        TOKEN_NORM_KEY: str(token_normalization),
        WEIGHT_INTERP_KEY: str(weight_interpretation),
    }


def with_sampling(
    info: Mapping[str, str],
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    denoise: float,
    vae_decode: bool,
) -> dict[str, str]:
    '''Return a copy of *info* extended with the sampling parameters.

    The input mapping is never mutated; a new ordered dict is returned
    with the sampling keys appended after the loader keys.

    Args:
        info: Info dict from the checkpoint bundle loader (or another source).
        seed: Sampling seed.
        steps: Sampling step count.
        cfg: CFG scale.
        sampler: ComfyUI sampler name.
        scheduler: ComfyUI scheduler name.
        denoise: Denoise strength.
        vae_decode: Whether the sampler decodes the result to IMAGE.

    Returns:
        A new ``dict[str, str]``.
    '''
    merged = dict(info)
    merged[SEED_KEY] = str(int(seed))
    merged[STEPS_KEY] = str(int(steps))
    merged[CFG_KEY] = a1111_weight_str(float(cfg))
    merged[SAMPLER_KEY] = str(sampler)
    merged[SCHEDULER_KEY] = str(scheduler)
    merged[DENOISE_KEY] = a1111_weight_str(float(denoise))
    merged[VAE_DECODE_KEY] = "enabled" if vae_decode else "disabled"
    return merged


def get(info: Mapping[str, Any], key: str, default: str = "") -> str:
    '''Read an info entry as a stripped string.

    Args:
        info: Info mapping (missing mapping yields ``default``).
        key: Entry key.
        default: Value returned when the key is absent or ``None``.

    Returns:
        The stringified entry, or ``default``.
    '''
    if info is None:
        return default
    value = info.get(key)
    if value is None:
        return default
    return str(value).strip()


def settings_pairs(info: Mapping[str, Any]) -> dict[str, str]:
    '''Extract the A1111 settings block from an info dict.

    Returns every entry except the prompt keys, order preserved; this
    is the ``params`` mapping for
    ``core.a1111_parameters.build_parameters``.

    Args:
        info: Info mapping.

    Returns:
        A fresh ``dict[str, str]`` without :data:`PROMPT_KEY` and
        :data:`NEGATIVE_KEY`; ``None`` values are skipped.
    '''
    return {
        str(k): str(v)
        for k, v in info.items()
        if k not in _PROMPT_KEYS and v is not None
    }
