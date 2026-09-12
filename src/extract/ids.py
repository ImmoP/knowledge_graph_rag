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