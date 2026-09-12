"""Extraction-layer cache-identity helpers (#23).

Implements ``docs/specs/extraction-contract.md`` §10.1 helper surface
as a deterministic identity/logging/verification helper. The LIVE
cache lookup is owned by the provider seam via
:func:`src.providers.cache.extraction_cache_scope` (provider-abstractions
§5.5a); this module never opens, writes, or reads cache files.

The real cache identity lives inside the provider layer. Around each
``LLMProvider.generate_json`` call the pipeline enters the six-dimension
scope:

    with extraction_cache_scope(
        chunk_id=...,
        source_text_sha256=...,
        ontology_version=...,
        ontology_sha256=...,
        prompt_template_filename=...,
        prompt_template_content_sha256=...,
    ):
        provider.generate_json(...)

This module exposes two small, pure helpers used elsewhere:

* :func:`schema_hash` — SHA256 hex of canonical-JSON serialised schema.
  The provider seam computes its own identical :func:`schema_hash`;
  we re-export the same construction so extraction code can reason
  about the schema hash without crossing the provider boundary for a
  single value.
* :func:`prompt_content_sha256` — SHA256 hex of raw prompt file
  bytes; matches spec §10.1 ``prompt_template_content_sha256``.

It also exposes :func:`required_identity_dimensions`, the ordered
tuple of §10.1 caller-specific dimension keys, used by tests to
confirm the pipeline passes all six to ``extraction_cache_scope``.

What this module DOES NOT do (per spec):

* It does not own cache I/O. No read/write of any disk path.
* It does not call into the provider seam for live cache lookups.
* It does not smuggle otherwise-unused metadata into the prompt text.
  The §10.1 dimensions participate in the cache key only because the
  spec names them; they are not a workaround for the provider cache.

Spec §10.5 ("What is NOT cached by extraction") — budget state,
chunk_status records, and reject records — is not mixed into any
key by this module either.
"""

from __future__ import annotations

import hashlib


def schema_hash(schema: dict) -> str:
    """SHA256 (hex) of the canonical-JSON serialisation of ``schema``.

    Mirrors :func:`src.providers.cache.schema_hash`. Both functions
    use the same canonical-JSON rules (``sort_keys=True``,
    ``separators=(",", ":")``, ``ensure_ascii=False``) so the same
    schema dict hashes byte-identically on both sides. The
    extraction-layer copy exists so the cache key construction can
    reason about the hash without crossing the provider boundary for
    a single value.
    """
    import json

    canonical = json.dumps(
        schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prompt_content_sha256(text) -> str:
    """SHA256 (hex) of the prompt template file's exact raw bytes.

    Spec §10.1: ``prompt_template_file_content`` is the verbatim
    template file content. ``text`` may be a ``str`` (the file
    contents), ``bytes``, or a :class:`Path`. The hash is byte-exact
    so an unintended whitespace or BOM edit invalidates the cache.
    """
    if hasattr(text, "read_bytes"):
        data = text.read_bytes()
    elif isinstance(text, bytes):
        data = text
    else:
        data = text.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def required_identity_dimensions() -> tuple[str, ...]:
    """The ordered tuple of six §10.1 caller-specific dimension keys.

    The provider-layer :func:`extraction_cache_scope` enforces that
    exactly these six kwargs are passed at every call site; this
    helper is the test-facing surface for that contract.
    """
    return (
        "chunk_id",
        "source_text_sha256",
        "ontology_version",
        "ontology_sha256",
        "prompt_template_filename",
        "prompt_template_content_sha256",
    )
