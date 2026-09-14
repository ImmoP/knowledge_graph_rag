"""Pure-data helpers for Neo4j ingestion (#25).

These helpers do not touch Neo4j, the network, or the filesystem. They
encode the deterministic projection rules from
``docs/specs/neo4j-ingestion.md`` §3.4 (PascalCase label mapping), §6
(cross-store ID contract), §8.1 (canonical logical projection), and §8.2
(citation-source digest).

Every helper is pure: given the same bytes in, the same bytes out.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


# Spec §6.1 — every identifier that flows through the graph must re-emerge
# byte-for-byte from a single canonical JSON serialisation. These separators
# match the rest of the codebase (src/graph/entity_resolution.py, etc.).
_JSON_SEPARATORS = (",", ":")
_JSON_KW: dict[str, Any] = {
    "sort_keys": True,
    "separators": _JSON_SEPARATORS,
    "ensure_ascii": False,
}


def canonical_json(payload: Any) -> str:
    """§8.1 canonical JSON serialisation: sort keys, compact separators,
    UTF-8 (no ``\\u`` escapes). Identical payload bytes always serialise
    to identical output bytes.
    """
    return json.dumps(payload, **_JSON_KW)


def canonical_json_bytes(payload: Any) -> bytes:
    """Same as :func:`canonical_json` but returns ``bytes`` for hashing."""
    return canonical_json(payload).encode("utf-8")


def sha256_hex(payload: Any) -> str:
    """§8.1 canonical logical projection hash — SHA256 over the canonical
    JSON bytes (no trailing newline).
    """
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


# ---------------------------------------------------------------------------
# ID regexes (§6 / chunk-record spec / §1).
# ---------------------------------------------------------------------------

CHUNK_ID_RE = re.compile(r"^c_[0-9a-f]{16}$")
OPENALEX_ID_RE = re.compile(r"^W[0-9]+$")
ENTITY_ID_RE = re.compile(r"^e_[0-9a-f]{16}$")


def is_chunk_id(value: Any) -> bool:
    return isinstance(value, str) and bool(CHUNK_ID_RE.match(value))


def is_openalex_id(value: Any) -> bool:
    return isinstance(value, str) and bool(OPENALEX_ID_RE.match(value))


def is_entity_id(value: Any) -> bool:
    return isinstance(value, str) and bool(ENTITY_ID_RE.match(value))


# ---------------------------------------------------------------------------
# OpenAlex URL / id normalisation (§4.4).
# ---------------------------------------------------------------------------


def normalize_openalex_id(raw: Any) -> str | None:
    """Map an OpenAlex URL or short ID to its canonical ``W...`` form.

    Returns ``None`` when the input does not match any known form. The
    ingestion spec (§4.4) requires every ID used in ``CITES`` to be the
    canonical short form; this helper is the single entry point.
    """
    if not isinstance(raw, str):
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    # Strip URL forms: https://openalex.org/W... or https://api.openalex.org/W...
    for prefix in (
        "https://openalex.org/",
        "http://openalex.org/",
        "https://api.openalex.org/",
        "http://api.openalex.org/",
    ):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):]
            break
    return candidate if is_openalex_id(candidate) else None


# ---------------------------------------------------------------------------
# Label mapping (§3.3).
# ---------------------------------------------------------------------------


def pascal_case(entity_type: str) -> str:
    """§3.3 PascalCase mapping from snake-case ontology entity type.

    Deterministic and locale-independent; not a general transliteration.
    Every letter that follows an underscore is uppercased; underscores
    are dropped. ``method`` -> ``Method``; ``open_domain_qa`` ->
    ``OpenDomainQa``. The ontology validator caps the v1 set to five
    structural types; this helper accepts any well-formed snake-case
    string and is the single source of truth for label string form.
    """
    if not isinstance(entity_type, str) or not entity_type:
        raise ValueError(f"entity_type must be a non-empty string: {entity_type!r}")
    parts = entity_type.split("_")
    return "".join(p[:1].upper() + p[1:] for p in parts)


def entity_subtype_label(entity_type: str) -> str | None:
    """§3.3 — return the Entity subtype label for an extracted entity
    type, or ``None`` if the type maps to a structural node (Document or
    Chunk) that is not also an Entity.

    Per the ontology, ``document`` and ``chunk`` are structural; the
    four extracted entity types are ``method``, ``dataset``, ``metric``,
    ``task``. Everything else is a preflight violation.
    """
    if entity_type in {"document", "chunk"}:
        return None
    return pascal_case(entity_type)


# ---------------------------------------------------------------------------
# Logical projection entries (§8.1).
# ---------------------------------------------------------------------------


def node_entry(
    *, labels: list[str], logical_id: str, properties: dict[str, Any]
) -> dict[str, Any]:
    """Build a single §8.1 node entry. Labels are sorted lexicographically
    here so callers may pass any order.
    """
    return {
        "labels": sorted(labels),
        "logical_id": logical_id,
        "properties": dict(properties),
    }


def relationship_entry(
    *,
    type: str,
    logical_id: Any,
    from_kind: str,
    from_id: str,
    to_kind: str,
    to_id: str,
    properties: dict[str, Any],
) -> dict[str, Any]:
    """Build a single §8.1 relationship entry with the canonical four-key
    shape. ``logical_id`` is serialised through :func:`canonical_json` so
    a list identity (e.g. ``["W...", "c_..."]``) sorts canonically.
    """
    return {
        "type": type,
        "logical_id": logical_id,
        "from": {"kind": from_kind, "id": from_id},
        "to": {"kind": to_kind, "id": to_id},
        "properties": dict(properties),
    }