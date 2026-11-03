'''
File:   t2i.py
Brief:  ComfyUI nodes for the T2I bundle pipeline (loader, LoRA list, utilities).
Author: Mistress-Lukutar
Date:   2026-11-03
Version: v0.1.0

The T2I set threads two custom entities through the graph:

* ``LUKUTAR_BUNDLE`` — a dict carrying ``model``/``clip``/``vae``/
  ``pos_cond``/``neg_cond``/``latent``/``img``; produced by the loader,
  consumed and re-emitted by the sampler.
* ``LUKUTAR_INFO`` — an insertion-ordered ``dict[str, str]`` describing
  every generation parameter (A1111-compatible keys), produced by the
  loader, extended by the sampler, converted to A1111 metadata later.

The weighted prompt engine implements BNK-style token normalization
(pure math in ``core/t2i_prompt.py``) and weight interpretations
(tensor math here) on top of ComfyUI's own CLIP tokenizer.
'''

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

try:
    import torch  # ComfyUI runtime only
except ImportError:  # pragma: no cover - exercised only outside runtime
    torch = None  # type: ignore[assignment]

try:
    import folder_paths  # ComfyUI runtime only
except ImportError:  # pragma: no cover - exercised only outside runtime
    folder_paths = None  # type: ignore[assignment]

try:
    import comfy.model_management as comfy_model_management  # runtime only
    import comfy.sd as comfy_sd
    import comfy.utils as comfy_utils
except ImportError:  # pragma: no cover - exercised only outside runtime
    comfy_model_management = None  # type: ignore[assignment]
    comfy_sd = None  # type: ignore[assignment]
    comfy_utils = None  # type: ignore[assignment]

from ..core.a1111_parameters import get_model_hash_cache
from ..core.t2i_info import (
    INFO_TYPE,
    build_loader_info,
    format_lora_entries,
)
from ..core.t2i_prompt import (
    DEFAULT_W_MAX,
    TOKEN_NORMALIZATION,
    WEIGHT_INTERPRETATION,
    normalize_weights,
    scale_to_norm,
    validate_options,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Module constants                                                   #
# ------------------------------------------------------------------ #

#: The ComfyUI socket type name for the T2I generation bundle.
BUNDLE_TYPE: str = "LUKUTAR_BUNDLE"

#: The ComfyUI socket type name for a weighted LoRA list.
LORA_LIST_TYPE: str = "LUKUTAR_LORA_LIST"

#: Ordered bundle member keys; every producer must emit all of them.
BUNDLE_KEYS: tuple[str, ...] = (
    "model",
    "clip",
    "vae",
    "pos_cond",
    "neg_cond",
    "latent",
    "img",
)

#: Maximum number of LoRA slots on the T2I LoRA List node.
MAX_LORA_SLOTS: int = 8

#: Combo sentinel marking an unused LoRA slot.
LORA_NONE: str = "None"

#: Maximum latent resolution edge (matches ComfyUI core).
MAX_RESOLUTION: int = 16384

#: Mask token id used by the embedding-mixing interpretations.  The
#: comma token proves a robust neutral embedding for SD-1.x CLIP
#: (empirical choice inherited from the BNK-style implementations).
_MASK_TOKEN: int = 266

#: Number of token lines encoded per CLIP call when building variants.
_ENCODE_CHUNK: int = 32

#: Weight row type alias: one float per token, one row per batch line.
#:
#: Imported lazily for typing only; kept as plain aliases to avoid a
#: runtime dependency on the core module layout.
_WeightMatrix = list[list[float]]
_WordIdMatrix = list[list[int]]
_TokenLine = list[tuple[Any, float]]


# ======================================================================
# Weighted prompt engine (torch side)                                  #
# ======================================================================


def _prepare_clip(clip: Any) -> Any:
    '''Apply layer/execution options and make sure CLIP is on GPU.

    Mirrors the setup ``CLIP.encode_from_tokens`` performs before
    touching the sub-models.

    Args:
        clip: The ComfyUI ``CLIP`` object.

    Returns:
        The ``cond_stage_model`` for convenience.
    '''
    csm = clip.cond_stage_model
    csm.reset_clip_options()
    layer_idx = getattr(clip, "layer_idx", None)
    if layer_idx is not None:
        csm.set_clip_options({"layer": layer_idx})
    csm.set_clip_options({"execution_device": clip.patcher.load_device})
    comfy_model_management.load_model_gpu(clip.patcher)
    return csm


def _make_encode_func(
    csm: Any,
    key: str,
) -> Callable[[list[_TokenLine]], tuple[Any, Any]]:
    '''Build an encode callable for one tokenizer key.

    The returned callable takes token lines (lists of
    ``(token, weight)``) and returns ``(emb, pooled)`` where ``emb`` is
    the ``(1, n_lines * token_len, dim)`` conditioning tensor.

    Args:
        csm: The ``cond_stage_model``.
        key: Tokenizer key such as ``"l"`` or ``"g"``.

    Returns:
        The encode callable.
    '''
    sub_attr = "clip_" + key
    if hasattr(csm, sub_attr):
        submodel = getattr(csm, sub_attr)

        def encode(lines: list[_TokenLine]) -> tuple[Any, Any]:
            out = submodel.encode_token_weights(lines)
            pooled = out[1] if len(out) > 1 else None
            return out[0], pooled

        return encode

    def encode_wrapped(lines: list[_TokenLine]) -> tuple[Any, Any]:
        out = csm.encode_token_weights({key: lines})
        pooled = out[1] if len(out) > 1 else None
        return out[0], pooled

    return encode_wrapped


def _encode_lines(
    encode_func: Callable[[list[_TokenLine]], tuple[Any, Any]],
    lines: list[_TokenLine],
) -> Any:
    '''Encode variant token lines in chunks.

    Args:
        encode_func: Callable from :func:`_make_encode_func`.
        lines: Variant token lines (all the same token length).

    Returns:
        Tensor of shape ``(len(lines), token_len, dim)``.
    '''
    chunks = []
    for i in range(0, len(lines), _ENCODE_CHUNK):
        part = lines[i : i + _ENCODE_CHUNK]
        emb, _ = encode_func(part)
        emb = emb.reshape(len(part), -1, emb.shape[-1])
        chunks.append(emb)
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, dim=0)


def _from_zero(flat_weights: list[float], base_emb: Any) -> Any:
    '''Scale the base embedding token-wise from zero.

    Args:
        flat_weights: One weight per token position, row-major over
            ``(batch, token)``.
        base_emb: ``(1, batch * token_len, dim)`` base embedding.

    Returns:
        The weighted embedding (same shape).
    '''
    weight_tensor = torch.tensor(
        flat_weights, dtype=base_emb.dtype, device=base_emb.device
    ).reshape(1, -1, 1)
    return base_emb * weight_tensor


def _renorm_a1111(base_emb: Any, weighted_emb: Any) -> Any:
    '''Renormalize a weighted embedding to the base embedding's mean.

    Args:
        base_emb: Unweighted base embedding.
        weighted_emb: Weighted embedding.

    Returns:
        The renormalized embedding.
    '''
    return (base_emb.mean() / weighted_emb.mean()) * weighted_emb


def _mix_coefficients(weights_asc: list[float]) -> list[float]:
    '''Incremental mixing weights for the embedding-mixing variants.

    Args:
        weights_asc: Distinct weights below 1, ascending.

    Returns:
        ``diff([0] + weights_asc)``: one coefficient per variant
        (including the base at index 0).
    '''
    padded = [0.0] + list(weights_asc)
    return [b - a for a, b in zip(padded, padded[1:], strict=False)]


def _down_weight(
    encode_func: Callable[[list[_TokenLine]], tuple[Any, Any]],
    tokens: list[_TokenLine],
    weights: _WeightMatrix,
    base_emb: Any,
    token_len: int,
) -> Any:
    '''Mix down-weighted token ranges via masked variants.

    For each batch line, tokens are progressively masked out in
    ascending weight thresholds below 1; the resulting variants are
    mixed with incremental coefficients so lighter weights contribute
    proportionally less of the base embedding.

    Args:
        encode_func: Callable from :func:`_make_encode_func`.
        tokens: Token lines (``(token, weight)`` tuples; weights in the
            tuples are irrelevant to the masking).
        weights: Normalized per-token weights.
        base_emb: ``(1, batch * token_len, dim)`` embedding to mix
            from.
        token_len: Tokens per line.

    Returns:
        The mixed embedding, shape ``(1, batch * token_len, dim)``.
    '''
    mask_token = (_MASK_TOKEN, 1.0)
    batch = len(tokens)

    variant_lines: list[_TokenLine] = []
    variant_slices: list[list[int]] = []  # variant line idx per batch line
    mix_weights: list[list[float]] = []
    for b in range(batch):
        distinct = sorted({w for w in weights[b] if w < 1.0})
        if not distinct:
            variant_slices.append([])
            mix_weights.append([])
            continue
        current = list(tokens[b])
        variants = []
        for threshold in distinct:
            current = [
                mask_token if weights[b][p] == threshold else t
                for p, t in enumerate(current)
            ]
            variants.append(list(current))
        start = len(variant_lines)
        variant_lines.extend(variants)
        variant_slices.append(list(range(start, start + len(variants))))
        mix_weights.append(_mix_coefficients(distinct))

    if not variant_lines:
        return base_emb

    embs = _encode_lines(encode_func, variant_lines)

    rows = []
    for b in range(batch):
        base_row = base_emb[0, b * token_len : (b + 1) * token_len]
        idxs = variant_slices[b]
        if not idxs:
            rows.append(base_row)
            continue
        parts = [base_row.unsqueeze(0)] + [embs[i] for i in idxs]
        stacked = torch.cat(parts, dim=0).to(dtype=base_emb.dtype)
        coefs = torch.tensor(
            mix_weights[b], dtype=base_emb.dtype, device=base_emb.device
        ).reshape(-1, 1, 1)
        rows.append((coefs * stacked).sum(dim=0))

    return torch.cat(rows, dim=0).unsqueeze(0)


def _masked_word_residual(
    encode_func: Callable[[list[_TokenLine]], tuple[Any, Any]],
    tokens: list[_TokenLine],
    weights: _WeightMatrix,
    word_ids: _WordIdMatrix,
    base_emb: Any,
    token_len: int,
) -> tuple[Any, Any]:
    '''Residual embedding for above-1 weights (the comfy++ upper half).

    Every real word with a weight different from 1 is masked out in its
    own variant line; the difference between the base and the variant
    isolates the word's embedding contribution, which is then scaled by
    ``(weight - 1)`` and summed back.

    Args:
        encode_func: Callable from :func:`_make_encode_func`.
        tokens: Token lines (unweighted tuples).
        weights: Per-token weights (only values != 1 matter).
        word_ids: Per-token word ids (0 = padding/framing).
        base_emb: ``(1, batch * token_len, dim)`` base embedding.
        token_len: Tokens per line.

    Returns:
        Tuple of (residual embedding with the base shape, pooled
        delta tensor or ``None``).
    '''
    mask_token = (_MASK_TOKEN, 1.0)
    batch = len(tokens)

    variant_lines: list[_TokenLine] = []
    metas: list[tuple[int, float, set[int]]] = []
    for b in range(batch):
        first_weight: dict[int, float] = {}
        for p, wid in enumerate(word_ids[b]):
            if wid != 0 and wid not in first_weight:
                first_weight[wid] = weights[b][p]
        for wid, w in first_weight.items():
            if w == 1.0:
                continue
            positions = {
                p for p, wid2 in enumerate(word_ids[b]) if wid2 == wid
            }
            variant_lines.append(
                [
                    mask_token if p in positions else t
                    for p, t in enumerate(tokens[b])
                ]
            )
            metas.append((b, w, positions))

    if not variant_lines:
        return torch.zeros_like(base_emb), None

    embs = _encode_lines(encode_func, variant_lines)

    residual = torch.zeros_like(base_emb)
    pooled_deltas: list[Any] = []
    for idx, (b, w, positions) in enumerate(metas):
        variant_row = embs[idx].to(dtype=base_emb.dtype)
        base_row = base_emb[0, b * token_len : (b + 1) * token_len]
        mask_vec = torch.tensor(
            [1.0 if p in positions else 0.0 for p in range(token_len)],
            dtype=base_emb.dtype,
            device=base_emb.device,
        ).reshape(1, -1, 1)
        contribution = (base_row.unsqueeze(0) - variant_row.unsqueeze(0)) * (
            mask_vec * (w - 1.0)
        )
        residual[0, b * token_len : (b + 1) * token_len] += contribution[0]
        pooled_deltas.append(
            (
                variant_row[token_len - 1 : token_len]
                - base_row[token_len - 1 : token_len]
            )
            * (w - 1.0)
        )

    pooled_delta = torch.stack(pooled_deltas, dim=0).mean(dim=0)
    return residual, pooled_delta


def _interpret_key(
    encode_func: Callable[[list[_TokenLine]], tuple[Any, Any]],
    token_rows: list[list[tuple[Any, float, int]]],
    token_normalization: str,
    weight_interpretation: str,
    w_max: float,
) -> tuple[Any, Any]:
    '''Run the prompt engine for one tokenizer key.

    Args:
        encode_func: Callable from :func:`_make_encode_func`.
        token_rows: Tokenized prompt rows of ``(token, weight,
            word_id)``.
        token_normalization: One of ``core.t2i_prompt.TOKEN_NORMALIZATION``.
        weight_interpretation: One of
            ``core.t2i_prompt.WEIGHT_INTERPRETATION``.
        w_max: Norm cap for the ``down_weight`` interpretation.

    Returns:
        Tuple of (conditioning embedding ``(1, batch * token_len,
        dim)``, pooled output tensor or ``None``).
    '''
    tokens: list[_TokenLine] = [[(t, w) for t, w, _ in row] for row in token_rows]
    weights: _WeightMatrix = [[w for _, w, _ in row] for row in token_rows]
    word_ids: _WordIdMatrix = [[wid for _, _, wid in row] for row in token_rows]
    weights = normalize_weights(token_normalization, word_ids, weights)
    token_len = len(token_rows[0])

    if weight_interpretation == "comfy":
        weighted = [
            [(t, w) for t, w in zip(trow, wrow, strict=True)]
            for trow, wrow in zip(tokens, weights, strict=True)
        ]
        return encode_func(weighted)

    unweighted: list[_TokenLine] = [
        [(t, 1.0) for t in trow] for trow in tokens
    ]
    base_emb, pooled_base = encode_func(unweighted)

    if weight_interpretation == "A1111":
        weighted_emb = _from_zero(
            [w for row in weights for w in row], base_emb
        )
        weighted_emb = _renorm_a1111(base_emb, weighted_emb)
        return weighted_emb, pooled_base

    if weight_interpretation == "compel":
        pos_tokens: list[_TokenLine] = [
            [
                (t, w) if w >= 1.0 else (t, 1.0)
                for t, w in zip(trow, wrow, strict=True)
            ]
            for trow, wrow in zip(tokens, weights, strict=True)
        ]
        pos_emb, _ = encode_func(pos_tokens)
        mixed = _down_weight(
            encode_func, pos_tokens, weights, pos_emb, token_len
        )
        pooled = mixed[0:1, token_len - 1 : token_len, :]
        return mixed, pooled

    if weight_interpretation == "comfy++":
        mixed = _down_weight(
            encode_func, unweighted, weights, base_emb, token_len
        )
        clipped = [[w if w > 1.0 else 1.0 for w in row] for row in weights]
        residual, pooled_delta = _masked_word_residual(
            encode_func, unweighted, clipped, word_ids, base_emb, token_len
        )
        pooled = pooled_base
        if pooled_base is not None and pooled_delta is not None:
            pooled = pooled_base + pooled_delta.to(
                device=pooled_base.device, dtype=pooled_base.dtype
            )
        return mixed + residual, pooled

    # down_weight
    scaled = scale_to_norm(word_ids, weights, w_max)
    mixed = _down_weight(
        encode_func, unweighted, scaled, base_emb, token_len
    )
    pooled = mixed[0:1, token_len - 1 : token_len, :]
    return mixed, pooled


def advanced_encode(
    clip: Any,
    text: str,
    token_normalization: str,
    weight_interpretation: str,
    w_max: float = DEFAULT_W_MAX,
) -> tuple[Any, Any]:
    '''Encode a prompt with selectable weighting semantics.

    SD1.x / SD2.x (single ``l`` key) and SDXL (``l`` + ``g`` keys)
    take the full engine; every other model composition falls back to
    ComfyUI's native weighted encoding with a log note.

    Args:
        clip: The ComfyUI ``CLIP`` object.
        text: The prompt text (may use ComfyUI ``(text:weight)``
            syntax).
        token_normalization: One of
            ``core.t2i_prompt.TOKEN_NORMALIZATION``.
        weight_interpretation: One of
            ``core.t2i_prompt.WEIGHT_INTERPRETATION``.
        w_max: Norm cap for the ``down_weight`` interpretation.

    Returns:
        Tuple of (conditioning embedding, pooled output or ``None``).

    Raises:
        RuntimeError: When called outside the ComfyUI runtime.
    '''
    if torch is None:
        raise RuntimeError("advanced_encode requires the ComfyUI runtime.")
    token_normalization, weight_interpretation = validate_options(
        token_normalization, weight_interpretation
    )
    tokenized = clip.tokenize(text, return_word_ids=True)
    csm = _prepare_clip(clip)

    has_l = bool(tokenized.get("l"))
    has_g = bool(tokenized.get("g"))

    if has_l and has_g:
        # SDXL: encode both keys with the engine, concatenate in
        # comfy's own order (l first), pooled from 'g'.
        emb_l, _ = _interpret_key(
            _make_encode_func(csm, "l"),
            tokenized["l"],
            token_normalization,
            weight_interpretation,
            w_max,
        )
        emb_g, pooled = _interpret_key(
            _make_encode_func(csm, "g"),
            tokenized["g"],
            token_normalization,
            weight_interpretation,
            w_max,
        )
        return torch.cat([emb_l, emb_g], dim=-1), pooled

    if has_g and hasattr(csm, "clip_g"):
        # SDXL refiner: 'g' only.
        return _interpret_key(
            _make_encode_func(csm, "g"),
            tokenized["g"],
            token_normalization,
            weight_interpretation,
            w_max,
        )

    if has_l:
        return _interpret_key(
            _make_encode_func(csm, "l"),
            tokenized["l"],
            token_normalization,
            weight_interpretation,
            w_max,
        )

    # Non-CLIP composition (e.g. Flux l + t5xxl): comfy native weights.
    logger.debug(
        "T2I prompt engine: no l/g CLIP keys (%s); using comfy native "
        "weighting",
        sorted(tokenized.keys()),
    )
    out = clip.encode_from_tokens(tokenized, return_pooled=True)
    return out[0], out[1]


def encode_conditioning(
    clip: Any,
    text: str,
    token_normalization: str,
    weight_interpretation: str,
) -> list[list[Any]]:
    '''Encode one prompt into a ComfyUI conditioning list.

    Args:
        clip: The ComfyUI ``CLIP`` object.
        text: The prompt text.
        token_normalization: Prompt engine option.
        weight_interpretation: Prompt engine option.

    Returns:
        A single-entry conditioning list.
    '''
    emb, pooled = advanced_encode(
        clip, text, token_normalization, weight_interpretation
    )
    entry: dict[str, Any] = {}
    if pooled is not None:
        entry["pooled_output"] = pooled
    return [[emb, entry]]


# ======================================================================
# Bundle helpers                                                      #
# ======================================================================


def validate_bundle(bundle: Any) -> None:
    '''Validate a T2I bundle entity.

    Args:
        bundle: The entity to check.

    Raises:
        RuntimeError: If the value is not a bundle-shaped dict or is
            missing required keys.
    '''
    if not isinstance(bundle, dict):
        raise RuntimeError(
            "T2I bundle input is not a LUKUTAR_BUNDLE entity; connect a "
            "T2I Loader or T2I Bundle Pack output."
        )
    missing = [key for key in BUNDLE_KEYS if key not in bundle]
    if missing:
        raise RuntimeError(
            "T2I bundle is missing key(s): " + ", ".join(missing)
        )


# ======================================================================
# Node classes                                                        #
# ======================================================================


class T2ILoraListNode:
    '''Build a weighted LoRA list from dynamic combo slots.

    Mirrors the A1111 Lora Hashes interaction: slots beyond the first
    free one stay hidden (managed by ``web/js/t2i_lora_list.js``).
    Zero-weight and unused entries are dropped by the backend.
    '''

    CATEGORY = "Lukutar/T2I"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        lora_choices = (
            ("None",) + tuple(folder_paths.get_filename_list("loras"))
            if folder_paths
            else ("None",)
        )
        required: dict[str, Any] = {}
        for i in range(1, MAX_LORA_SLOTS + 1):
            required[f"lora_{i}"] = (
                lora_choices,
                {"tooltip": f"LoRA slot {i}; 'None' leaves the slot unused."},
            )
            required[f"weight_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -3.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": f"Strength of LoRA slot {i} "
                    "(applied to both model and CLIP).",
                },
            )
        return {"required": required}

    RETURN_TYPES = (LORA_LIST_TYPE,)
    RETURN_NAMES = ("lora_list",)
    FUNCTION = "build"
    DESCRIPTION = (
        "Collect several LoRAs with weights into one list socket for "
        "the T2I Loader."
    )

    def build(self, **kwargs: Any) -> tuple[tuple[tuple[str, float], ...]]:
        '''Collect the claimed slots into a weighted tuple list.

        Args:
            **kwargs: ``lora_N`` combo values and ``weight_N`` floats.

        Returns:
            Tuple of ``(name, weight)`` pairs; unused (``None``) and
            zero-weight entries are skipped.
        '''
        entries: list[tuple[str, float]] = []
        for i in range(1, MAX_LORA_SLOTS + 1):
            name = str(kwargs.get(f"lora_{i}", LORA_NONE)).strip()
            weight = float(kwargs.get(f"weight_{i}", 1.0))
            if not name or name == LORA_NONE:
                continue
            if weight == 0.0:
                logger.info(
                    "T2I LoRA List: skipping zero-weight LoRA %s", name
                )
                continue
            entries.append((name, weight))
        return (tuple(entries),)


class T2ILoaderNode:
    '''Load a checkpoint (+ optional LoRAs) and build the T2I bundle.

    Encodes positive/negative prompts with the weighted prompt engine,
    creates the empty latent and assembles the portable info entity.
    '''

    CATEGORY = "Lukutar/T2I"

    def __init__(self) -> None:
        #: Per-instance LoRA file cache ``(path, state_dict, metadata)``.
        self.loaded_lora: tuple[str, Any, Any] | None = None

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "ckpt_name": (
                    tuple(folder_paths.get_filename_list("checkpoints"))
                    if folder_paths
                    else (),
                    {"tooltip": "The checkpoint to load."},
                ),
                "clip_skip": (
                    "INT",
                    {
                        "default": -1,
                        "min": -24,
                        "max": -1,
                        "step": 1,
                        "tooltip": "CLIP layer to stop at "
                        "(-1 = standard output layer).",
                    },
                ),
                "vae": (
                    (
                        ("baked",)
                        + tuple(folder_paths.get_filename_list("vae"))
                    )
                    if folder_paths
                    else ("baked",),
                    {"tooltip": "'baked' uses the checkpoint's own VAE."},
                ),
                "positive": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Positive prompt.",
                    },
                ),
                "negative": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Negative prompt.",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": MAX_RESOLUTION,
                        "step": 8,
                        "tooltip": "Latent width in pixels.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": MAX_RESOLUTION,
                        "step": 8,
                        "tooltip": "Latent height in pixels.",
                    },
                ),
                "token_normalization": (
                    TOKEN_NORMALIZATION,
                    {"tooltip": "How token weights are normalized."},
                ),
                "weight_interpretation": (
                    WEIGHT_INTERPRETATION,
                    {"tooltip": "How (text:weight) syntax is interpreted."},
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "Number of latent images in the batch.",
                    },
                ),
            },
            "optional": {
                "lora_list": (
                    LORA_LIST_TYPE,
                    {
                        "tooltip": "Weighted LoRA list from the T2I LoRA "
                        "List node."
                    },
                ),
            },
        }

    RETURN_TYPES = (BUNDLE_TYPE, INFO_TYPE)
    RETURN_NAMES = ("bundle", "info")
    FUNCTION = "load"
    DESCRIPTION = (
        "Loads a checkpoint with optional LoRAs, encodes both prompts "
        "and emits the T2I bundle plus its textual info entity."
    )

    def _apply_loras(
        self,
        model: Any,
        clip: Any,
        lora_list: Any,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, str]]]:
        '''Apply every LoRA from the list onto model and CLIP.

        Args:
            model: The diffusion model patcher.
            clip: The CLIP object.
            lora_list: ``(name, weight)`` pairs or ``None``.

        Returns:
            Tuple of (applied ``(name, weight)`` entries, ``(name,
            hash)`` pairs for the info entity).
        '''
        applied: list[tuple[str, float]] = []
        hashes: list[tuple[str, str]] = []
        if not lora_list:
            return applied, hashes

        for name, weight in lora_list:
            path = folder_paths.get_full_path_or_raise("loras", name)
            lora_sd = None
            metadata = None
            if self.loaded_lora is not None:
                if self.loaded_lora[0] == path:
                    lora_sd = self.loaded_lora[1]
                    metadata = self.loaded_lora[2]
                else:
                    self.loaded_lora = None
            if lora_sd is None:
                lora_sd, metadata = comfy_utils_load_torch_file(path)
                self.loaded_lora = (path, lora_sd, metadata)

            weight_f = float(weight)
            if weight_f == 0.0:
                continue
            model, clip = comfy_sd.load_lora_for_models(
                model, clip, lora_sd, weight_f, weight_f
            )
            applied.append((name, weight_f))
            hashes.append((name, get_model_hash_cache().hash(path)))
        return applied, hashes

    def load(
        self,
        ckpt_name: str,
        clip_skip: int,
        vae: str,
        positive: str,
        negative: str,
        width: int,
        height: int,
        token_normalization: str,
        weight_interpretation: str,
        batch_size: int,
        lora_list: Any = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        '''Load the checkpoint and assemble the bundle and info.

        Args:
            ckpt_name: Checkpoint combo name.
            clip_skip: Negative CLIP layer index.
            vae: ``"baked"`` or a VAE file name.
            positive: Positive prompt.
            negative: Negative prompt.
            width: Latent width in pixels.
            height: Latent height in pixels.
            token_normalization: Prompt engine option.
            weight_interpretation: Prompt engine option.
            batch_size: Latent batch size.
            lora_list: Optional weighted LoRA list.

        Returns:
            Tuple of (bundle dict, info dict).

        Raises:
            RuntimeError: When called outside the ComfyUI runtime.
        '''
        if folder_paths is None or torch is None:
            raise RuntimeError("T2I Loader requires the ComfyUI runtime.")

        token_normalization, weight_interpretation = validate_options(
            token_normalization, weight_interpretation
        )

        ckpt_path = folder_paths.get_full_path_or_raise(
            "checkpoints", ckpt_name
        )
        model, clip, vae_model = comfy_sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )[:3]

        applied, lora_hashes = self._apply_loras(model, clip, lora_list)

        clip = clip.clone()
        clip.clip_layer(clip_skip)

        if vae != "baked":
            vae_path = folder_paths.get_full_path_or_raise("vae", vae)
            vae_model = comfy_sd.VAE(
                sd=comfy_utils.load_torch_file(vae_path)
            )

        pos_cond = encode_conditioning(
            clip, positive, token_normalization, weight_interpretation
        )
        neg_cond = encode_conditioning(
            clip, negative, token_normalization, weight_interpretation
        )

        latent_channels = model.model.latent_format.latent_channels
        latent = torch.zeros(
            [batch_size, latent_channels, height // 8, width // 8],
            device=comfy_model_management.intermediate_device(),
            dtype=comfy_model_management.intermediate_dtype(),
        )
        latent_dict = {"samples": latent, "downscale_ratio_spacial": 8}

        model_hash = get_model_hash_cache().hash(ckpt_path)
        info = build_loader_info(
            prompt=positive,
            negative=negative,
            model_name=os.path.splitext(ckpt_name)[0],
            model_hash=model_hash,
            vae_name=vae,
            clip_skip=clip_skip,
            lora_entries=applied,
            lora_hashes=lora_hashes,
            width=width,
            height=height,
            batch_size=batch_size,
            token_normalization=token_normalization,
            weight_interpretation=weight_interpretation,
        )
        if applied:
            logger.info(
                "T2I Loader: applied LoRAs %s",
                format_lora_entries(applied) or "-",
            )

        bundle = {
            "model": model,
            "clip": clip,
            "vae": vae_model,
            "pos_cond": pos_cond,
            "neg_cond": neg_cond,
            "latent": latent_dict,
            "img": None,
        }
        return (bundle, info)


def comfy_utils_load_torch_file(path: str) -> tuple[Any, Any]:
    '''Load a torch file with metadata (runtime only).

    Args:
        path: Absolute model file path.

    Returns:
        Tuple of (state dict, metadata or ``None``).

    Raises:
        RuntimeError: When called outside the ComfyUI runtime.
    '''
    return comfy_utils.load_torch_file(
        path, safe_load=True, return_metadata=True
    )


class T2IBundleUnpackNode:
    '''Split a T2I bundle back into standard ComfyUI sockets.

    The ``img`` output yields an empty ``(0, h, w, 3)`` tensor until a
    sampler with VAE decode fills the bundle image slot.
    '''

    CATEGORY = "Lukutar/T2I"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "bundle": (
                    BUNDLE_TYPE,
                    {"tooltip": "Bundle from the T2I Loader or Sampler."},
                ),
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "CLIP",
        "VAE",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "IMAGE",
    )
    RETURN_NAMES = (
        "model",
        "clip",
        "vae",
        "pos_cond",
        "neg_cond",
        "latent",
        "img",
    )
    FUNCTION = "unpack"
    DESCRIPTION = "Unpacks the T2I bundle into its components."

    def unpack(
        self, bundle: Any
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        '''Split the bundle into component sockets.

        Args:
            bundle: The bundle entity.

        Returns:
            Tuple of (model, clip, vae, pos_cond, neg_cond, latent,
            img). The image is an empty tensor while no sampler has
            decoded one yet.

        Raises:
            RuntimeError: On a malformed bundle.
        '''
        validate_bundle(bundle)
        img = bundle["img"]
        if img is None:
            samples = (bundle.get("latent") or {}).get("samples")
            if samples is not None:
                height = samples.shape[-2] * 8
                width = samples.shape[-1] * 8
                device = samples.device
                dtype = samples.dtype
            else:
                height = width = 64
                device = comfy_model_management.intermediate_device()
                dtype = comfy_model_management.intermediate_dtype()
            img = torch.zeros((0, height, width, 3), device=device, dtype=dtype)
        return (
            bundle["model"],
            bundle["clip"],
            bundle["vae"],
            bundle["pos_cond"],
            bundle["neg_cond"],
            bundle["latent"],
            img,
        )


class T2IBundlePackNode:
    '''Assemble a T2I bundle from standard ComfyUI sockets.'''

    CATEGORY = "Lukutar/T2I"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Diffusion model."}),
                "clip": ("CLIP", {"tooltip": "CLIP text encoder."}),
                "vae": ("VAE", {"tooltip": "VAE decoder."}),
                "pos_cond": (
                    "CONDITIONING",
                    {"tooltip": "Positive conditioning."},
                ),
                "neg_cond": (
                    "CONDITIONING",
                    {"tooltip": "Negative conditioning."},
                ),
                "latent": ("LATENT", {"tooltip": "Latent image batch."}),
            },
            "optional": {
                "img": (
                    "IMAGE",
                    {"tooltip": "Optional decoded image tensor."},
                ),
            },
        }

    RETURN_TYPES = (BUNDLE_TYPE,)
    RETURN_NAMES = ("bundle",)
    FUNCTION = "pack"
    DESCRIPTION = "Packs standard sockets into a T2I bundle."

    def pack(
        self,
        model: Any,
        clip: Any,
        vae: Any,
        pos_cond: Any,
        neg_cond: Any,
        latent: Any,
        img: Any = None,
    ) -> tuple[dict[str, Any]]:
        '''Collect the components into a bundle.

        Args:
            model: Diffusion model.
            clip: CLIP text encoder.
            vae: VAE.
            pos_cond: Positive conditioning.
            neg_cond: Negative conditioning.
            latent: Latent dict.
            img: Optional image tensor.

        Returns:
            Tuple of (bundle dict,).
        '''
        bundle = {
            "model": model,
            "clip": clip,
            "vae": vae,
            "pos_cond": pos_cond,
            "neg_cond": neg_cond,
            "latent": latent,
            "img": img,
        }
        return (bundle,)


__all__ = [
    "BUNDLE_KEYS",
    "BUNDLE_TYPE",
    "LORA_LIST_TYPE",
    "MAX_LORA_SLOTS",
    "T2IBundlePackNode",
    "T2IBundleUnpackNode",
    "T2ILoaderNode",
    "T2ILoraListNode",
    "advanced_encode",
    "encode_conditioning",
    "validate_bundle",
]
