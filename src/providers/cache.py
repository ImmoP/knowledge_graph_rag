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

import contextlib
import contextvars
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


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

# --- extraction cache scope (repair #23, spec §5.5a) ----------------
#
# ``extraction_cache_scope(**dims)`` activates strict-superset identity
# dimensions on top of the canonical inputs (spec §5.5a). The six
# allowed extra-dimension keys are: ``chunk_id``, ``source_text_sha256``,
# ``ontology_version``, ``ontology_sha256``, ``prompt_template_filename``,
# ``prompt_template_content_sha256``. ``LLMProvider`` Protocol signatures
# are unchanged — the scope is read inside ``make_key`` (only when
# ``operation == "generate_json"``) via a ``contextvars`` token.
#
# Scope semantics:
# - Concurrent call isolation via ``contextvars`` is supported.
# - Nested extraction scopes are FORBIDDEN (spec §5.5a fail-closed).
#   Entering a scope while one is already active raises
#   ``RuntimeError`` and does NOT mutate or replace the outer scope.
_EXTRACTION_SCOPE_KEYS: tuple[str, ...] = (
    "chunk_id",
    "source_text_sha256",
    "ontology_version",
    "ontology_sha256",
    "prompt_template_filename",
    "prompt_template_content_sha256",
)
_active_extraction_scope: contextvars.ContextVar[dict[str, str] | None] = (
    contextvars.ContextVar("_active_extraction_scope", default=None)
)


@contextlib.contextmanager
def extraction_cache_scope(**dims: str) -> Iterator[None]:
    """Activate extraction cache-scope identity dimensions (spec §5.5a).

    The ``LLMProvider.generate_json`` cache key includes these six
    dimensions as a strict superset of the canonical identity inputs
    (spec §5.5 / §5.5a). When no scope is active, the produced key is
    byte-identical to the canonical key (spec §5.5 floor is preserved).

    Concurrent call isolation via ``contextvars`` is supported, so two
    independent calls running concurrently each see their own active
    scope. Nested extraction scopes are NOT supported: entering this
    context manager while another ``extraction_cache_scope`` is already
    active raises ``RuntimeError`` immediately and does not mutate or
    replace the outer scope. The previous value is restored on exit
    even if the body raises.

    Only the six allowed dimension keys are accepted. Unknown keys
    raise ``ValueError``; missing keys raise ``ValueError``; non-string
    values raise ``TypeError``. The active caller (extraction) is
    expected to provide all six for every ``generate_json`` call —
    partial scope is a programming error.
    """
    if set(dims) != set(_EXTRACTION_SCOPE_KEYS):
        unknown = set(dims) - set(_EXTRACTION_SCOPE_KEYS)
        missing = set(_EXTRACTION_SCOPE_KEYS) - set(dims)
        problems: list[str] = []
        if unknown:
            problems.append(
                f"unknown keys: {sorted(unknown)!r}"
            )
        if missing:
            problems.append(
                f"missing keys: {sorted(missing)!r}"
            )
        raise ValueError(
            "extraction_cache_scope requires exactly "
            f"{list(_EXTRACTION_SCOPE_KEYS)}; got {problems}"
        )
    for k, v in dims.items():
        if not isinstance(v, str):
            raise TypeError(
                f"extraction_cache_scope[{k!r}] must be str; got {type(v).__name__}"
            )
    # Fail closed on nesting (spec §5.5a). If a scope is already
    # active, refuse entry without mutating the outer scope.
    if _active_extraction_scope.get() is not None:
        raise RuntimeError(
            "nested extraction_cache_scope is forbidden (spec §5.5a); "
            "an extraction_cache_scope is already active"
        )
    # Defensive copy so a caller mutating the dict after entry does not
    # affect the active scope. Sorted on entry so two callers that pass
    # the same six strings in different orders get the same hash.
    scope = {k: dims[k] for k in sorted(dims)}
    token = _active_extraction_scope.set(scope)
    try:
        yield
    finally:
        _active_extraction_scope.reset(token)


def current_extraction_cache_scope() -> dict[str, str] | None:
    """Return a copy of the active extraction cache scope, if any.

    Observability wrappers use this to tag provider attempts with the
    chunk id and prompt phase already established by the extraction
    pipeline. Returning a copy prevents callers from mutating the
    context-local cache identity. This does not change cache keys or
    any extraction validator.
    """
    scope = _active_extraction_scope.get()
    return dict(scope) if scope is not None else None


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

    For ``operation == "generate_json"`` the key is a strict superset
    of the canonical key when an ``extraction_cache_scope`` is active
    (spec §5.5a); for ``"generate"`` and ``"embed"`` the extraction
    scope is ignored. When no scope is active the generate_json key is
    byte-identical to the §5.1 canonical key — this is the single
    source of truth for base-key assembly.
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
    # spec §5.5a (repair #23): if an extraction_cache_scope is active
    # at the moment of key construction, mix its six identity
    # dimensions into the hash as one trailing list element so the
    # key is a strict superset of the canonical key. Only generate_json
    # participates; generate() and embed() ignore the scope so the
    # §5.5 floor is preserved for those operations.
    if operation == "generate_json":
        scope = _active_extraction_scope.get()
        if scope is not None:
            parts.append(_canonical(scope))
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
    # Single source of truth for base-key assembly is ``make_key``;
    # the extraction-scope augmentation lives there too (only for
    # ``operation == "generate_json"``).
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