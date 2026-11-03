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
...) so the ``InfoToA1111`` node can convert the mapping into the
A1111 parameters engine's settings block
(``core/a1111_parameters.py``) via :func:`a1111_settings`, which
filters pack-only entries and normalises A1111-convention values.

Prompts are stored under the lowercase keys :data:`PROMPT_KEY` /
:data:`NEGATIVE_KEY` and are excluded from the settings block, exactly
like the A1111 infotext format itself.
'''

from __future__ import annotations

import logging
import re
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


# ------------------------------------------------------------------ #
# A1111-compatible settings conversion                               #
# ------------------------------------------------------------------ #

#: Pack-engine keys with no A1111 infotext counterpart: emitted by the
#: bundle pipeline but never valid in A1111 ``parameters`` text. The
#: ``Lora`` entry is pack-only too — A1111 records LoRA weights via
#: ``<lora:name:weight>`` prompt tags and hashes via ``Lora hashes``.
NON_A1111_KEYS: frozenset[str] = frozenset(
    {
        BATCH_KEY,
        LORA_KEY,
        TOKEN_NORM_KEY,
        WEIGHT_INTERP_KEY,
        VAE_DECODE_KEY,
    }
)

#: ComfyUI sampler name -> A1111 infotext sampler name. Names outside
#: this map (CFG++ variants, custom samplers) pass through unchanged.
A1111_SAMPLER_NAMES: dict[str, str] = {
    "euler": "Euler",
    "euler_ancestral": "Euler a",
    "heun": "Heun",
    "lms": "LMS",
    "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a",
    "dpmpp_2s_ancestral": "DPM++ 2S a",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_2m_sde_heun": "DPM++ 2M SDE Heun",
    "dpmpp_3m_sde": "DPM++ 3M SDE",
    "dpm_fast": "DPM fast",
    "dpm_adaptive": "DPM adaptive",
    "ddim": "DDIM",
    "plms": "PLMS",
    "uni_pc": "UniPC",
    "restart": "Restart",
}

#: ComfyUI scheduler name -> A1111 infotext ``Schedule type`` value.
A1111_SCHEDULE_TYPES: dict[str, str] = {
    "normal": "Normal",
    "karras": "Karras",
    "exponential": "Exponential",
    "sgm_uniform": "SGM Uniform",
    "simple": "Simple",
    "linear": "Linear",
    "ddim_uniform": "DDIM Uniform",
    "beta": "Beta",
    "kl_optimal": "KL Optimal",
    "align_your_steps": "Align Your Steps",
}

#: ``Size`` values in the ``"{width} x {height}"`` loader format.
_SIZE_SPACES_RE = re.compile(r"^(\d+)\s*x\s*(\d+)$")

#: Model-file extensions stripped from ``Lora hashes`` entry names —
#: A1111 writes the LoRA *name* there (``"Mawshot (1): 9ec5889be9"``),
#: never the file name with its extension.
_MODEL_FILE_EXTS: tuple[str, ...] = (
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".sft",
    ".gguf",
)

#: Canonical A1111 infotext key order (a subset of A1111's
#: ``create_infotext`` insertion order). Known keys are emitted in this
#: sequence — external parsers often match unit patterns such as
#: ``Model hash: X, Model: Y`` — and every other key follows in its
#: original info order.
_A1111_KEY_ORDER: tuple[str, ...] = (
    STEPS_KEY,
    SAMPLER_KEY,
    SCHEDULER_KEY,
    CFG_KEY,
    SEED_KEY,
    SIZE_KEY,
    MODEL_HASH_KEY,
    MODEL_KEY,
    "VAE hash",
    VAE_KEY,
    DENOISE_KEY,
    "Conditional mask weight",
    CLIP_SKIP_KEY,
    "Hires checkpoint",
    "Hires upscale",
    "Hires upscaler",
    "Hires steps",
    "Hires resize-1",
    "Hires resize-2",
    LORA_HASHES_KEY,
    "TI hashes",
    "ENSD",
    "Eta",
    "Eta DDIM",
    "RNG",
    "Tiling",
    "Seed resize from",
    "Version",
    "Hash",
    "User",
    "Batch mask weight",
    "N first pass sizes",
    "Original Size",
)


def _strip_model_ext(name: str) -> str:
    '''Strip a known model-file extension from a LoRA name.

    Args:
        name: LoRA name, possibly a file name like
            ``"Mawshot (1).safetensors"``.

    Returns:
        The name without a known model-file extension; dotted names
        whose suffix is not a known model extension stay untouched.
    '''
    lowered = name.lower()
    for ext in _MODEL_FILE_EXTS:
        if lowered.endswith(ext):
            return name[: -len(ext)]
    return name


def _strip_lora_hash_exts(text: str) -> str:
    '''Strip model-file extensions from ``Lora hashes`` entry names.

    Args:
        text: Raw ``"name: hash"`` entries comma-joined (unquoted).

    Returns:
        The entries with A1111-style extension-free names; malformed
        entries pass through untouched.
    '''
    parts: list[str] = []
    for entry in text.split(", "):
        name, sep, hash_str = entry.partition(": ")
        parts.append(
            f"{_strip_model_ext(name)}: {hash_str}" if sep else entry
        )
    return ", ".join(parts)


def a1111_settings(info: Mapping[str, Any]) -> dict[str, str]:
    '''Build an A1111-compatible settings dict from an info mapping.

    Everything that has no A1111 infotext counterpart is removed, and
    values deviating from A1111 conventions are normalised:

    * :data:`NON_A1111_KEYS` (``Batch size``, ``Lora``, ``Token
      normalization``, ``Weight interpretation``, ``VAE decode``) are
      dropped — A1111's infotext never contains them and some parsers
      choke on them.
    * ``VAE: baked`` is dropped (ComfyUI-ese; A1111 expects a VAE file
      name).
    * ``Clip skip`` is emitted as the positive layers-to-skip count
      (``-2`` -> ``2``); a count of ``1`` (no skip) drops the key,
      matching A1111 which only writes it when skipping.
    * ``Size`` ``"{width} x {height}"`` is normalised to A1111's
      ``"{width}x{height}"`` (no spaces).
    * ``Sampler`` / ``Schedule type`` values are translated through
      :data:`A1111_SAMPLER_NAMES` / :data:`A1111_SCHEDULE_TYPES`.
    * ``Lora hashes`` entry names lose their model-file extension
      (``"Mawshot (1).safetensors: 9ec5889be9"`` ->
      ``"Mawshot (1): 9ec5889be9"``), the A1111 convention that
      lets parsers match the LoRA.
    * The entries are reordered into the canonical A1111 infotext
      order (:data:`_A1111_KEY_ORDER`), so pairs like ``Model hash,
      Model`` appear where parsers expect them.

    Unknown keys pass through verbatim after the known ones, so
    user-added entries survive.

    Args:
        info: Info mapping (prompt keys and ``None``/empty values are
            skipped).

    Returns:
        A fresh ``dict[str, str]`` for ``build_parameters``.
    '''
    result: dict[str, str] = {}
    for key, value in info.items():
        if key in _PROMPT_KEYS or key in NON_A1111_KEYS or value is None:
            continue
        text = str(value).strip()
        if not text:
            continue

        if key == CLIP_SKIP_KEY:
            try:
                skip = abs(int(text))
            except ValueError:
                pass
            else:
                if skip > 1:
                    result[key] = str(skip)
                continue

        elif key == SIZE_KEY:
            match = _SIZE_SPACES_RE.fullmatch(text)
            if match:
                text = f"{match.group(1)}x{match.group(2)}"

        elif key == VAE_KEY and text.lower() == "baked":
            continue

        elif key == SAMPLER_KEY:
            text = A1111_SAMPLER_NAMES.get(text, text)

        elif key == SCHEDULER_KEY:
            text = A1111_SCHEDULE_TYPES.get(text, text)

        elif key == LORA_HASHES_KEY:
            text = _strip_lora_hash_exts(text)

        result[key] = text

    order = {k: i for i, k in enumerate(_A1111_KEY_ORDER)}
    return dict(
        sorted(result.items(), key=lambda kv: order.get(kv[0], len(order)))
    )
