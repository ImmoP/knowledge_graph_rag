"""Disk-backed content-addressed cache for provider calls (#19).

Implements ``docs/specs/provider-abstractions.md`` §5 verbatim: cache
identity is canonical JSON + SHA256; layout is
``${LLM_CACHE_DIR}/{llm,embeddings}/<key>.json``; atomic writes protect
against partial-file corruption; nothing transient (budget state,
retry history, reachability checks) is cached.

The cache key builder is a verbatim transcription of spec §5.1's
construction. Per-operation identity inputs are:

* ``generate``: cache_format_version, operation, provider.name,
  provider.model, system, user, max_tokens, temperature.
* ``generate_json``: as ``generate`` plus the canonical-JSON schema
  hash (SHA256 of ``json.dumps(schema, sort_keys=True,
  separators=(",", ":"))``).
* ``embed``: cache_format_version, operation, provider.name,
  provider.model, ordered full texts list.

Caller-side augmentation (e.g. ``#7`` extraction's chunk identity and
prompt-template SHA) may add extra identity dimensions on top — see
spec §5.5 — but the canonical identity inputs above are the floor and
must not be weakened.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Single shared format-version constant. Bumped on adapter changes that
# affect parsing or output shape so old entries become orphaned rather
# than silently returned for the new format. Spec §5.3.
CACHE_FORMAT_VERSION = "1"

# Standard subdirectories under LLM_CACHE_DIR. Spec §5.2.
LLM_SUBDIR = "llm"
EMBEDDING_SUBDIR = "embeddings"

# Canonical-JSON separator pair used both inside individual values
# (max_tokens, temperature, schema) and at the top-level hash blob.
_SEPARATORS = (",", ":")


def _canonical(value: Any) -> str:
    """Return the canonical-JSON string used in cache-key hashing.

    Numbers, lists, dicts, and strings all go through ``json.dumps`` with
    ``sort_keys`` and the tight separators; ``ensure_ascii=False`` keeps
    non-ASCII text from being expanded (which would change the hash) for
    text-bearing parts.
    """
    return json.dumps(value, sort_keys=True, separators=_SEPARATORS, ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def schema_hash(schema: dict) -> str:
    """SHA256 (hex) of the canonical-JSON serialisation of ``schema``.

    Used as the ``schema_canonical_json`` member of the cache key for
    ``generate_json``. Spec §5.1.
    """
    return _sha256_hex(_canonical(schema))


def make_key(
    *,
    cache_format_version: str,
    operation: str,
    provider_name: str,
    provider_model: str,
    system: str | None = None,
    user: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    schema_canonical_json: str | None = None,
    texts: list[str] | None = None,
) -> str:
    """Build the canonical cache key for one provider call.

    Mirrors the spec §5.1 construction exactly. Each named argument
    contributes to identity iff the corresponding operation actually
    uses it; absent inputs (``None``) are omitted from the hashed blob
    so the key shape is consistent across operations.
    """
    parts: list[str] = [cache_format_version, operation, provider_name, provider_model]
    if system is not None:
        parts.append(system)
    if user is not None:
        parts.append(user)
    if max_tokens is not None:
        parts.append(_canonical(max_tokens))
    if temperature is not None:
        parts.append(_canonical(temperature))
    if schema_canonical_json is not None:
        parts.append(schema_canonical_json)
    if texts is not None:
        parts.append(_canonical(texts))
    return _sha256_hex(_canonical(parts))


def _key_for_generate(
    *,
    cache_format_version: str,
    provider_name: str,
    provider_model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> str:
    return make_key(
        cache_format_version=cache_format_version,
        operation="generate",
        provider_name=provider_name,
        provider_model=provider_model,
        system=system,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _key_for_generate_json(
    *,
    cache_format_version: str,
    provider_name: str,
    provider_model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    schema: dict,
) -> str:
    return make_key(
        cache_format_version=cache_format_version,
        operation="generate_json",
        provider_name=provider_name,
        provider_model=provider_model,
        system=system,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
        schema_canonical_json=schema_hash(schema),
    )


def _key_for_embed(
    *,
    cache_format_version: str,
    provider_name: str,
    provider_model: str,
    texts: list[str],
) -> str:
    return make_key(
        cache_format_version=cache_format_version,
        operation="embed",
        provider_name=provider_name,
        provider_model=provider_model,
        texts=texts,
    )


@dataclass(frozen=True)
class CachePaths:
    """Resolved on-disk cache layout for a single cache root."""

    root: Path
    llm: Path
    embeddings: Path

    @classmethod
    def from_root(cls, root: Path) -> "CachePaths":
        root = Path(root)
        return cls(root=root, llm=root / LLM_SUBDIR, embeddings=root / EMBEDDING_SUBDIR)

    def ensure(self) -> None:
        self.llm.mkdir(parents=True, exist_ok=True)
        self.embeddings.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` as JSON to ``path`` atomically.

    A crash or concurrent reader sees either the old file or the new
    one, never a half-written file. Uses ``NamedTemporaryFile`` in the
    same directory so the final ``os.replace`` is atomic on POSIX and
    Windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class ProviderCache:
    """Disk-backed cache rooted at ``LLM_CACHE_DIR``.

    The cache is a property of the abstraction layer (spec §5 opening
    line); callers receive a thin facade and never see the key
    construction. Methods correspond one-to-one to the three adapter
    operations and return ``None`` on miss.
    """

    def __init__(
        self,
        paths: CachePaths,
        *,
        cache_format_version: str = CACHE_FORMAT_VERSION,
    ) -> None:
        self.paths = paths
        self.cache_format_version = cache_format_version
        self.paths.ensure()

    # --- lookup ------------------------------------------------------

    def get_generate(
        self,
        *,
        provider_name: str,
        provider_model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> str | None:
        key = _key_for_generate(
            cache_format_version=self.cache_format_version,
            provider_name=provider_name,
            provider_model=provider_model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return self._read_text(self.paths.llm / f"{key}.json")

    def get_generate_json(
        self,
        *,
        provider_name: str,
        provider_model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        schema: dict,
    ) -> dict | None:
        key = _key_for_generate_json(
            cache_format_version=self.cache_format_version,
            provider_name=provider_name,
            provider_model=provider_model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            schema=schema,
        )
        rec = self._read_record(self.paths.llm / f"{key}.json")
        if rec is None:
            return None
        payload = rec.get("payload")
        return payload if isinstance(payload, dict) else None

    def get_embed(
        self,
        *,
        provider_name: str,
        provider_model: str,
        texts: list[str],
    ) -> list[list[float]] | None:
        key = _key_for_embed(
            cache_format_version=self.cache_format_version,
            provider_name=provider_name,
            provider_model=provider_model,
            texts=texts,
        )
        rec = self._read_record(self.paths.embeddings / f"{key}.json")
        if rec is None:
            return None
        payload = rec.get("payload")
        return payload if isinstance(payload, list) else None

    # --- write -------------------------------------------------------

    def put_generate(
        self,
        *,
        provider_name: str,
        provider_model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        text: str,
    ) -> str:
        key = _key_for_generate(
            cache_format_version=self.cache_format_version,
            provider_name=provider_name,
            provider_model=provider_model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        _atomic_write_json(
            self.paths.llm / f"{key}.json",
            {
                "operation": "generate",
                "provider_name": provider_name,
                "provider_model": provider_model,
                "payload": text,
            },
        )
        return key

    def put_generate_json(
        self,
        *,
        provider_name: str,
        provider_model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        schema: dict,
        parsed: dict,
    ) -> str:
        key = _key_for_generate_json(
            cache_format_version=self.cache_format_version,
            provider_name=provider_name,
            provider_model=provider_model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            schema=schema,
        )
        _atomic_write_json(
            self.paths.llm / f"{key}.json",
            {
                "operation": "generate_json",
                "provider_name": provider_name,
                "provider_model": provider_model,
                "payload": parsed,
            },
        )
        return key

    def put_embed(
        self,
        *,
        provider_name: str,
        provider_model: str,
        texts: list[str],
        vectors: list[list[float]],
    ) -> str:
        key = _key_for_embed(
            cache_format_version=self.cache_format_version,
            provider_name=provider_name,
            provider_model=provider_model,
            texts=texts,
        )
        _atomic_write_json(
            self.paths.embeddings / f"{key}.json",
            {
                "operation": "embed",
                "provider_name": provider_name,
                "provider_model": provider_model,
                "payload": vectors,
            },
        )
        return key

    # --- key exposure (for tests + spec §5 inspection) --------------

    def key_generate(
        self,
        *,
        provider_name: str,
        provider_model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        return _key_for_generate(
            cache_format_version=self.cache_format_version,
            provider_name=provider_name,
            provider_model=provider_model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def key_generate_json(
        self,
        *,
        provider_name: str,
        provider_model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        schema: dict,
    ) -> str:
        return _key_for_generate_json(
            cache_format_version=self.cache_format_version,
            provider_name=provider_name,
            provider_model=provider_model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            schema=schema,
        )

    def key_embed(
        self,
        *,
        provider_name: str,
        provider_model: str,
        texts: list[str],
    ) -> str:
        return _key_for_embed(
            cache_format_version=self.cache_format_version,
            provider_name=provider_name,
            provider_model=provider_model,
            texts=texts,
        )

    # --- low-level IO ------------------------------------------------

    @staticmethod
    def _read_text(path: Path) -> str | None:
        rec = ProviderCache._read_record(path)
        if rec is None:
            return None
        payload = rec.get("payload")
        return payload if isinstance(payload, str) else None

    @staticmethod
    def _read_record(path: Path) -> dict | None:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            # Treat any IO error as a miss; the spec calls out malformed
            # entries as "never silently returned for the new format",
            # which we satisfy by never reading them at all.
            return None
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None