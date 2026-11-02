'''
File:   a1111_parameters.py
Brief:  A1111-style generation-parameter (infotext) primitives.
Author: Mistress-Lukutar
Date:   2026-11-02
Version: v0.1.0
'''

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Constants                                                          #
# ------------------------------------------------------------------ #

#: Widely-used A1111 sampler names for the Sampling node's combo widget.
A1111_SAMPLERS: tuple[str, ...] = (
    "Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a",
    "DPM++ 2S a", "DPM++ 2M", "DPM++ SDE", "DPM++ 2M SDE",
    "DPM++ 2M SDE Heun", "DPM++ 3M SDE", "DPM fast", "DPM adaptive",
    "LMS Karras", "DPM2 Karras", "DPM2 a Karras", "DPM++ 2S a Karras",
    "DPM++ 2M Karras", "DPM++ SDE Karras", "DPM++ 3M SDE Karras",
    "DDIM", "PLMS", "UniPC", "Restart",
)

#: The sampler preselected on new A1111 Sampling nodes.
DEFAULT_SAMPLER: str = "DPM++ 2M Karras"

#: Length of an A1111 model hash (first N hex chars of the file sha256).
MODEL_HASH_CHARS: int = 10

#: Default cache file name stored next to the pack root.
HASH_CACHE_FILENAME: str = "a1111_hash_cache.json"

# ------------------------------------------------------------------ #
# Infotext helpers                                                   #
# ------------------------------------------------------------------ #


def quote_a1111(text: Any) -> str:
    '''Mirror A1111's ``quote()`` behaviour.

    Coerces *text* to string; if the result contains none of ```,```,
    ``\\n``, or ``:`` it is returned as-is, otherwise it is JSON-encoded
    (with unicode left unescaped).

    Args:
        text: Value to quote.

    Returns:
        The quoted string suitable for A1111-style infotext.
    '''
    s = str(text)
    if "," not in s and "\n" not in s and ":" not in s:
        return s
    return json.dumps(s, ensure_ascii=False)


def build_parameters(
    prompt: Any,
    negative_prompt: Any,
    params: Mapping[str, Any] | None = None,
) -> str:
    '''Compose an A1111-style infotext string.

    The output follows the format:

    * ``prompt`` line (raw, may be multiline).
    * Optional ``Negative prompt: ...`` line when *negative_prompt* is
      truthy (bare truthiness check, no strip).
    * Optional settings block: each non-None parameter rendered as
      ``f"{key}: {quoted_value}"``, joined by ``", "``.

    Args:
        prompt: The generation prompt (coerced to str).
        negative_prompt: The negative prompt (truthy check, coerced to str).
        params: Optional mapping of parameter name -> value. ``None`` or
            empty dict skips the settings block entirely; entries with
            ``None`` values are silently skipped.

    Returns:
        The complete infotext string, stripped of leading/trailing
        whitespace (empty input yields ``""``).
    '''
    prompt_str = str(prompt)
    parts: list[str] = [prompt_str]

    neg_str = str(negative_prompt)
    if neg_str:
        parts.append(f"Negative prompt: {neg_str}")

    settings_lines: list[str] = []
    if params is not None:
        for key, value in params.items():
            if value is None:
                continue
            settings_lines.append(
                f"{key}: {quote_a1111(str(value))}"
            )

    if settings_lines:
        parts.append(", ".join(settings_lines))

    return "\n".join(parts).strip()


def compute_model_hash(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    '''Compute the first ``MODEL_HASH_CHARS`` hex chars of a file's SHA-256.

    Reads the file in streaming chunks so large models never fit fully
    into memory.

    Args:
        path: Path to the model file.
        chunk_size: Number of bytes per read iteration (default 1 MiB).

    Returns:
        The truncated hex digest string.

    Raises:
        FileNotFoundError: When *path* does not exist.
    '''
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:MODEL_HASH_CHARS]


def a1111_float_str(value: Any) -> str:
    '''Return *value* as A1111-style float string.

    Coerces to ``float`` first so that integer-like inputs (e.g. ``2``)
    produce ``"2.0"`` rather than ``"2"``.

    Args:
        value: Any numeric-compatible value.

    Returns:
        The A1111 repr-style string, e.g. ``"2.0"``, ``"0.45"``.
    '''
    return str(float(value))


# ------------------------------------------------------------------ #
# Model hash cache                                                   #
# ------------------------------------------------------------------ #


class ModelHashCache:
    '''On-disk cache for model SHA-256 prefixes keyed by file mtime/size.

    Cache entries are stored as ``{"mtime": float, "size": int, "hash": str}``
    and keyed by the absolute path of the model file.
    '''

    def __init__(self, cache_path: str | Path) -> None:
        '''Initialise the cache from disk.

        Args:
            cache_path: Path to the JSON cache file.
        '''
        self._cache_path = os.path.abspath(os.fspath(cache_path))
        try:
            with open(self._cache_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self._data: dict[str, dict[str, Any]] = data  # type: ignore[assignment]
                else:
                    logger.warning(
                        "ModelHashCache: %s is not a JSON object; starting empty",
                        self._cache_path,
                    )
                    self._data = {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._data = {}

    def get(self, path: str | Path) -> str | None:
        '''Return a cached hash if the file's mtime/size still match.

        Args:
            path: The model file to look up.

        Returns:
            The 10-character hex hash, or ``None`` when there is no valid
            cache entry.
        '''
        abs_path = os.path.abspath(os.fspath(path))
        entry = self._data.get(abs_path)
        if entry is None:
            return None
        try:
            st = os.stat(abs_path)
        except OSError:
            return None
        if st.st_mtime != entry.get("mtime") or st.st_size != entry.get(
            "size"
        ):
            return None
        return entry["hash"]

    def hash(self, path: str | Path) -> str:
        '''Return the cached (or newly computed) model hash.

        Args:
            path: The model file to hash.

        Returns:
            The 10-character hex SHA-256 prefix.
        '''
        abs_path = os.path.abspath(os.fspath(path))
        cached = self.get(abs_path)
        if cached is not None:
            return cached
        h = compute_model_hash(path)
        try:
            st = os.stat(abs_path)
            self._data[abs_path] = {
                "mtime": st.st_mtime,
                "size": st.st_size,
                "hash": h,
            }
        except OSError:
            # File disappeared between stat and now -- store without mtime/size.
            self._data[abs_path] = {"hash": h}
        self.save()
        return h

    def save(self) -> None:
        '''Persist the cache to disk atomically.

        Writes to ``<cache_path>.tmp`` then renames via ``os.replace``.
        Warnings are logged on failure; no exception is raised.
        '''
        tmp_path = self._cache_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
            os.replace(tmp_path, self._cache_path)
        except Exception:  # noqa: BLE001
            logger.warning(
                "ModelHashCache: failed to save %s", self._cache_path,
            )


# ------------------------------------------------------------------ #
# Module-level singleton                                               #
# ------------------------------------------------------------------ #

_cache: ModelHashCache | None = None


def get_model_hash_cache() -> ModelHashCache:
    '''Return the shared cache stored next to the pack root.

    The cache file lives at ``core/../a1111_hash_cache.json`` (i.e.
    one level above ``core/``).  Lazy-initialised into a module-global
    variable; thread-safety is not required.

    Returns:
        The singleton :class:`ModelHashCache` instance.
    '''
    global _cache
    if _cache is None:
        pack_root = Path(__file__).resolve().parent.parent
        cache_file = pack_root / HASH_CACHE_FILENAME
        _cache = ModelHashCache(cache_file)
    return _cache
