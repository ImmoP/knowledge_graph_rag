"""Content-addressed ID derivation for extraction (#23).

Implements ``docs/specs/extraction-contract.md`` §6 verbatim. The IDs
are content-addressed hashes over a canonical-JSON payload of the
record's own identity-bearing fields. They are deterministic across
clean full runs and resumed runs because the payload contains no
processing-order, run-id, sequence, or timestamp component.

Canonical-JSON rules (spec §6.1):

* ``sort_keys=True``
* ``separators=(",", ":")``
* ``ensure_ascii=False`` with UTF-8 encoding
* integer fields serialised as JSON numbers (not strings)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


_MENTION_PREFIX = "m_"
_ASSERTION_PREFIX = "a_"

#: Number of hex chars of the sha256 digest that form the visible ID.
#: Spec §6.1: "The 16-hex prefix is the full ID; there is no chunk
#: suffix, run suffix, or sequence number."
HEX_PREFIX_LEN = 16

#: Canonical-JSON separator pair used everywhere IDs are serialised.
_SEPARATORS = (",", ":")


def _canonical_json(payload: Any) -> str:
    """Serialise ``payload`` as canonical JSON per spec §6.1."""
    return json.dumps(payload, sort_keys=True, separators=_SEPARATORS, ensure_ascii=False)


def mention_id(
    *,
    chunk_id: str,
    source_text_sha256: str,
    entity_type: str,
    char_start: int,
    char_end: int,
) -> str:
    """Derive the content-addressed mention ID per spec §6.1.

    The payload contains exactly the fields listed in spec §6.1; no
    ``surface_form``, ``canonical_form``, ``confidence``, ``mention_id``,
    run id, sequence number, or timestamp participates.
    """
    payload = {
        "chunk_id": chunk_id,
        "source_text_sha256": source_text_sha256,
        "entity_type": entity_type,
        "char_start": char_start,
        "char_end": char_end,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return _MENTION_PREFIX + digest[:HEX_PREFIX_LEN]


def assertion_id(
    *,
    chunk_id: str,
    source_text_sha256: str,
    relation_type: str,
    subject_ref: dict,
    object_ref: dict,
    char_start: int,
    char_end: int,
) -> str:
    """Derive the content-addressed assertion ID per spec §6.1.

    Endpoints are serialised as ``{"kind": ..., "id": ...}`` under the
    same canonical-JSON rules as the outer payload. No ``confidence``,
    run id, sequence, or timestamp participates.
    """
    payload = {
        "chunk_id": chunk_id,
        "source_text_sha256": source_text_sha256,
        "relation_type": relation_type,
        "subject_ref": {
            "kind": subject_ref["kind"],
            "id": subject_ref["id"],
        },
        "object_ref": {
            "kind": object_ref["kind"],
            "id": object_ref["id"],
        },
        "char_start": char_start,
        "char_end": char_end,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return _ASSERTION_PREFIX + digest[:HEX_PREFIX_LEN]


def rederive_mention_id(record: dict) -> str:
    """Re-derive a mention ID from a record's own fields.

    Implements the §15 acceptance harness: the tripwire against any
    future regression that re-introduces a position-dependent ID rule.
    The record is a JSONL mention record; only the spec-mandated
    identity fields participate.
    """
    evidence = record["evidence"]
    return mention_id(
        chunk_id=record["chunk_id"],
        source_text_sha256=evidence["source_text_sha256"],
        entity_type=record["entity_type"],
        char_start=evidence["char_start"],
        char_end=evidence["char_end"],
    )


def rederive_assertion_id(record: dict) -> str:
    """Re-derive an assertion ID from a record's own fields."""
    evidence = record["evidence"]
    return assertion_id(
        chunk_id=record["chunk_id"],
        source_text_sha256=evidence["source_text_sha256"],
        relation_type=record["relation_type"],
        subject_ref=record["subject_ref"],
        object_ref=record["object_ref"],
        char_start=evidence["char_start"],
        char_end=evidence["char_end"],
    )


# --- v2 ID derivation (#51, #47 superseding resolution) --------------
#
# v2 mentions and v2 assertions are derived from the same identity
# fields as v1 plus an explicit ``schema_version: SCHEMA_VERSION_V2``
# discriminator baked into the canonical payload. The discriminator
# makes otherwise-identical v1 and v2 inputs derive distinct IDs under
# the documented canonical hash construction.
#
# The canonical-JSON rules (``sort_keys=True``,
# ``separators=(",", ":")``, UTF-8 ``ensure_ascii=False``), SHA-256,
# the ``m_`` / ``a_`` prefixes, and ``HEX_PREFIX_LEN`` are reused
# unchanged from the v1 helpers above.


def mention_id_v2(
    *,
    chunk_id: str,
    source_text_sha256: str,
    entity_type: str,
    char_start: int,
    char_end: int,
) -> str:
    """Derive the v2 content-addressed mention ID (#51, #47).

    The payload is identical to ``mention_id``'s payload plus an
    explicit ``schema_version`` discriminator pulled from
    ``src.extract.schema.SCHEMA_VERSION_V2``. The discriminator is
    imported locally to avoid the historical cycle between
    ``schema`` and ``ids``.
    """
    from src.extract.schema import SCHEMA_VERSION_V2

    payload = {
        "chunk_id": chunk_id,
        "entity_type": entity_type,
        "char_start": char_start,
        "char_end": char_end,
        "source_text_sha256": source_text_sha256,
        "schema_version": SCHEMA_VERSION_V2,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return _MENTION_PREFIX + digest[:HEX_PREFIX_LEN]


def assertion_id_v2(
    *,
    chunk_id: str,
    source_text_sha256: str,
    relation_type: str,
    subject_ref: dict,
    object_ref: dict,
    char_start: int,
    char_end: int,
) -> str:
    """Derive the v2 content-addressed assertion ID (#51, #47)."""
    from src.extract.schema import SCHEMA_VERSION_V2

    payload = {
        "chunk_id": chunk_id,
        "relation_type": relation_type,
        "subject_ref": {
            "kind": subject_ref["kind"],
            "id": subject_ref["id"],
        },
        "object_ref": {
            "kind": object_ref["kind"],
            "id": object_ref["id"],
        },
        "char_start": char_start,
        "char_end": char_end,
        "source_text_sha256": source_text_sha256,
        "schema_version": SCHEMA_VERSION_V2,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return _ASSERTION_PREFIX + digest[:HEX_PREFIX_LEN]


def rederive_v2_mention_id(record: dict) -> str:
    """Re-derive a v2 mention ID from a v2 record's own fields (#51).

    Fail-closed v2 contract:

    1. the record must carry a top-level ``schema_version``;
    2. ``type(record["schema_version"])`` must be ``int`` (rejects
       ``bool``, which is a subclass of ``int`` in Python);
    3. ``record["schema_version"]`` must equal
       ``SCHEMA_VERSION_V2``;
    4. only then is the ID derived via ``mention_id_v2``.

    Raises ``ValueError`` on any of the first three conditions. The
    helper never reinterprets a v1 record as v2.
    """
    from src.extract.schema import SCHEMA_VERSION_V2

    sv = record.get("schema_version")
    if type(sv) is not int:
        raise ValueError(
            f"rederive_v2_mention_id: record.schema_version must be a "
            f"plain int equal to SCHEMA_VERSION_V2 ({SCHEMA_VERSION_V2}); "
            f"got type={type(sv).__name__} value={sv!r}"
        )
    if sv != SCHEMA_VERSION_V2:
        raise ValueError(
            f"rederive_v2_mention_id: record.schema_version must equal "
            f"SCHEMA_VERSION_V2 ({SCHEMA_VERSION_V2}); got {sv!r}"
        )
    evidence = record["evidence"]
    return mention_id_v2(
        chunk_id=record["chunk_id"],
        source_text_sha256=evidence["source_text_sha256"],
        entity_type=record["entity_type"],
        char_start=evidence["char_start"],
        char_end=evidence["char_end"],
    )


def rederive_v2_assertion_id(record: dict) -> str:
    """Re-derive a v2 assertion ID from a v2 record's own fields (#51).

    Same fail-closed contract as ``rederive_v2_mention_id``.
    """
    from src.extract.schema import SCHEMA_VERSION_V2

    sv = record.get("schema_version")
    if type(sv) is not int:
        raise ValueError(
            f"rederive_v2_assertion_id: record.schema_version must be a "
            f"plain int equal to SCHEMA_VERSION_V2 ({SCHEMA_VERSION_V2}); "
            f"got type={type(sv).__name__} value={sv!r}"
        )
    if sv != SCHEMA_VERSION_V2:
        raise ValueError(
            f"rederive_v2_assertion_id: record.schema_version must equal "
            f"SCHEMA_VERSION_V2 ({SCHEMA_VERSION_V2}); got {sv!r}"
        )
    evidence = record["evidence"]
    return assertion_id_v2(
        chunk_id=record["chunk_id"],
        source_text_sha256=evidence["source_text_sha256"],
        relation_type=record["relation_type"],
        subject_ref=record["subject_ref"],
        object_ref=record["object_ref"],
        char_start=evidence["char_start"],
        char_end=evidence["char_end"],
    )