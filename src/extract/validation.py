"""Validation gates for extraction candidates (#23).

Implements ``docs/specs/extraction-contract.md`` §7 and §8 verbatim.

* Gate 1 (schema): owned by ``generate_json`` (spec §8.1). After its
  one retry is exhausted, ``LLMValidationError`` propagates; the
  pipeline catches it and routes the failure to the reject log. The
  extraction layer never stacks another schema retry.
* Gate 2 (ontology conformity): checked here per spec §8.2.
* Gate 3 (evidence integrity): checked here per spec §7 / §8.3.

A failed gate emits a reject-log record. The chunk is not
extraction-complete until both phases pass all three gates.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from src.extract import ids as ids_module


# ---------- shared helpers -------------------------------------------


def source_text_sha256(chunk_text: str) -> str:
    """Spec §7: ``sha256(chunk.text.encode("utf-8")).hexdigest()``.

    The chunk text is read exactly as it appears in
    ``chunks.jsonl``; no normalisation.
    """
    return hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()


_WS_RE = re.compile(r"\s+")


def _whitespace_normalise(s: str) -> str:
    """Spec §7.1: collapse internal whitespace runs, trim edges.

    Used only for the "contains" check in §7.1; chunk text itself is
    never rewritten.
    """
    return _WS_RE.sub(" ", s).strip()


# ---------- evidence gate --------------------------------------------


class EvidenceError(ValueError):
    """Raised when a mention / assertion fails gate 3 (evidence integrity)."""


def _check_offsets(
    *,
    char_start: int,
    char_end: int,
    chunk_text: str,
    where: str,
) -> None:
    if not (isinstance(char_start, int) and isinstance(char_end, int)):
        raise EvidenceError(f"{where}: char offsets must be integers")
    if char_start < 0:
        raise EvidenceError(f"{where}: char_start must be >= 0")
    if char_end <= char_start:
        raise EvidenceError(f"{where}: char_end must be > char_start")
    if char_end > len(chunk_text):
        raise EvidenceError(
            f"{where}: char_end {char_end} exceeds chunk text length {len(chunk_text)}"
        )


def check_mention_evidence(
    *,
    mention: dict,
    chunk_text: str,
    expected_source_text_sha256: str,
) -> None:
    """Gate 3 for mentions per spec §7.1.

    Rules:

    1. offsets in range;
    2. ``chunk.text[char_start:char_end]`` (whitespace-normalised on
       both sides) contains ``surface_form``;
    3. ``source_text_sha256`` matches ``sha256(chunk.text)``.
    """
    ev = mention["evidence"]
    char_start = ev["char_start"]
    char_end = ev["char_end"]
    _check_offsets(
        char_start=char_start,
        char_end=char_end,
        chunk_text=chunk_text,
        where="mention evidence",
    )
    span_text = chunk_text[char_start:char_end]
    surface = mention.get("surface_form", "")
    span_norm = _whitespace_normalise(span_text)
    surface_norm = _whitespace_normalise(surface)
    if not surface_norm or surface_norm not in span_norm:
        raise EvidenceError(
            f"mention surface_form {surface!r} not contained in "
            f"chunk span [{char_start},{char_end}) {span_text!r}"
        )
    if ev["source_text_sha256"] != expected_source_text_sha256:
        raise EvidenceError(
            f"mention source_text_sha256 drift: stored "
            f"{ev['source_text_sha256']!r} != chunk "
            f"{expected_source_text_sha256!r}"
        )


def check_assertion_evidence(
    *,
    assertion: dict,
    chunk_text: str,
    expected_source_text_sha256: str,
) -> None:
    """Gate 3 for assertions per spec §7.1.

    Mechanical checks only (per spec §7.1 and the relation prompt's
    span-selection obligation, §5.3):

    1. ``0 <= char_start < char_end <= len(chunk.text)`` for the
       relevant chunk;
    2. the whitespace-normalised selected span
       ``chunk.text[char_start:char_end]`` is non-empty;
    3. ``evidence.source_text_sha256 == sha256(chunk.text)``.

    Whether the cited span directly hosts or expresses the asserted
    relation is the model's responsibility (relation prompt §5.3,
    the explicit "directly express the asserted relation"
    span-selection obligation). This validator applies no semantic
    check: no paraphrase comparison, no embeddings, no similarity
    threshold, no LLM judge, no regex heuristic, no extra LLM call.
    The §7.1 amended rule reads: "the mechanical evidence gate does
    not compare the assertion span against a paraphrased form of the
    relation" — the assertion model shape carries no relation-text
    field to compare against.
    """
    ev = assertion["evidence"]
    char_start = ev["char_start"]
    char_end = ev["char_end"]
    _check_offsets(
        char_start=char_start,
        char_end=char_end,
        chunk_text=chunk_text,
        where="assertion evidence",
    )
    span_text = chunk_text[char_start:char_end]
    span_norm = _whitespace_normalise(span_text)
    if not span_norm:
        raise EvidenceError(
            f"assertion evidence span [{char_start},{char_end}) "
            f"is whitespace-only after normalisation: {span_text!r}"
        )
    if ev["source_text_sha256"] != expected_source_text_sha256:
        raise EvidenceError(
            f"assertion source_text_sha256 drift: stored "
            f"{ev['source_text_sha256']!r} != chunk "
            f"{expected_source_text_sha256!r}"
        )


# ---------- ontology gate --------------------------------------------


class OntologyError(ValueError):
    """Raised when a candidate fails gate 2 (ontology conformity)."""


def _is_string(v: Any) -> bool:
    return isinstance(v, str) and bool(v)


def _validate_endpoint(
    *,
    ref: dict,
    endpoint_label: str,
    expected_population_mode: str,
    doc_id: str | None,
    same_chunk_mentions: dict[str, dict] | None,
    entity_type_for_mention: dict[str, str] | None,
) -> tuple[str, str]:
    """Validate one ``{kind, id}`` endpoint per spec §5.3 / §8.2.

    Returns the resolved ``(kind, id)`` so the caller can build the
    canonical assertion record. Raises :class:`OntologyError` for
    any structural mismatch.
    """
    if not isinstance(ref, dict):
        raise OntologyError(f"{endpoint_label}: ref must be an object")
    if set(ref.keys()) - {"kind", "id"} or {"kind", "id"} - set(ref.keys()):
        raise OntologyError(
            f"{endpoint_label}: ref must have exactly {{kind, id}}; got {sorted(ref.keys())}"
        )
    kind = ref.get("kind")
    if kind not in ("document", "mention"):
        raise OntologyError(
            f"{endpoint_label}: kind must be 'document' or 'mention'; got {kind!r}"
        )
    ref_id = ref.get("id")
    if not _is_string(ref_id):
        raise OntologyError(f"{endpoint_label}: id must be a non-empty string")

    # Map endpoint population_mode to expected endpoint kind.
    if expected_population_mode == "structural":
        if kind != "document":
            raise OntologyError(
                f"{endpoint_label}: structural endpoint expects kind=document; got {kind!r}"
            )
        if doc_id is None:
            raise OntologyError(
                f"{endpoint_label}: structural document endpoint requires chunk.doc_id"
            )
        if ref_id != doc_id:
            raise OntologyError(
                f"{endpoint_label}: document endpoint id {ref_id!r} != "
                f"current chunk doc_id {doc_id!r}"
            )
        return kind, ref_id
    if expected_population_mode == "extracted":
        if kind != "mention":
            raise OntologyError(
                f"{endpoint_label}: extracted endpoint expects kind=mention; got {kind!r}"
            )
        if same_chunk_mentions is None or ref_id not in same_chunk_mentions:
            raise OntologyError(
                f"{endpoint_label}: dangling mention ref {ref_id!r} "
                f"(not in same-chunk mention surface)"
            )
        # Cross-document inference is forbidden (spec §5.3). Mentions
        # in the same-chunk surface always share the current chunk's
        # doc_id by construction, so no further check is needed here.
        return kind, ref_id
    raise OntologyError(
        f"{endpoint_label}: unsupported population_mode {expected_population_mode!r}"
    )


def check_assertion_ontology(
    *,
    assertion: dict,
    ontology: dict,
    doc_id: str,
    same_chunk_mentions: dict[str, dict],
    entity_type_for_mention: dict[str, str],
) -> None:
    """Gate 2 for assertions per spec §8.2.

    Validates:

    * ``relation_type`` ∈ extracted ontology relation names;
    * ``subject_ref.kind/id`` matches the from-endpoint type;
    * ``object_ref.kind/id`` matches the to-endpoint type;
    * document endpoints equal current chunk's ``doc_id``;
    * mention endpoints resolve to a validated same-chunk mention
      whose ``entity_type`` matches the relation endpoint type.

    There is no blanket self-loop prohibition (spec §8.2 final
    sentence); ontology v1 does not pin one. A self-loop would only
    be rejected if both endpoints resolve to the same id AND the
    ontology or a relation-specific contract forbids it; we pass
    that through unchanged and rely on the underlying relations
    having structurally distinct endpoint types (e.g. document →
    method never self-loops because document.kind != method.kind).
    """
    from src.extract import schema as schema_module  # local import to avoid cycle

    relation_type = assertion.get("relation_type")
    if not _is_string(relation_type):
        raise OntologyError("assertion: relation_type must be a non-empty string")
    rt_map = schema_module.relation_endpoints(ontology)
    if relation_type not in rt_map:
        raise OntologyError(
            f"assertion: relation_type {relation_type!r} not in extracted ontology"
        )
    et_modes = schema_module.entity_type_endpoint_table(ontology)
    from_type, to_type = rt_map[relation_type]
    from_mode = et_modes.get(from_type)
    to_mode = et_modes.get(to_type)
    if from_mode is None or to_mode is None:
        raise OntologyError(
            f"assertion: relation {relation_type!r} reference unknown entity types"
        )

    # subject_ref
    sub_ref = assertion.get("subject_ref")
    sub_kind, sub_id = _validate_endpoint(
        ref=sub_ref,
        endpoint_label="assertion subject_ref",
        expected_population_mode=from_mode,
        doc_id=doc_id,
        same_chunk_mentions=same_chunk_mentions,
        entity_type_for_mention=entity_type_for_mention,
    )
    # If subject is a mention, its entity_type must match from_type.
    if sub_kind == "mention":
        sub_mention = same_chunk_mentions[sub_id]
        if sub_mention["entity_type"] != from_type:
            raise OntologyError(
                f"assertion subject_ref: mention {sub_id!r} has entity_type "
                f"{sub_mention['entity_type']!r} but relation {relation_type!r} "
                f"expects {from_type!r}"
            )

    # object_ref
    obj_ref = assertion.get("object_ref")
    obj_kind, obj_id = _validate_endpoint(
        ref=obj_ref,
        endpoint_label="assertion object_ref",
        expected_population_mode=to_mode,
        doc_id=doc_id,
        same_chunk_mentions=same_chunk_mentions,
        entity_type_for_mention=entity_type_for_mention,
    )
    if obj_kind == "mention":
        obj_mention = same_chunk_mentions[obj_id]
        if obj_mention["entity_type"] != to_type:
            raise OntologyError(
                f"assertion object_ref: mention {obj_id!r} has entity_type "
                f"{obj_mention['entity_type']!r} but relation {relation_type!r} "
                f"expects {to_type!r}"
            )


def check_mention_ontology(
    *,
    mention: dict,
    ontology: dict,
) -> None:
    """Gate 2 for mentions per spec §8.2.

    Only ``entity_type`` membership is checked here; the other
    mention fields (surface_form, canonical_form, char_start,
    char_end, confidence) are validated structurally by the JSON
    Schema passed to ``generate_json``.
    """
    from src.extract import schema as schema_module  # local import to avoid cycle

    entity_type = mention.get("entity_type")
    if not _is_string(entity_type):
        raise OntologyError("mention: entity_type must be a non-empty string")
    allowed = [et["name"] for et in schema_module.extracted_entity_types(ontology)]
    if entity_type not in allowed:
        raise OntologyError(
            f"mention: entity_type {entity_type!r} not in extracted ontology"
        )


# ---------- per-chunk validators (used after validation, with ID) ---


@dataclass(frozen=True)
class ValidatedMention:
    """A mention record that has passed all three gates.

    The pipeline uses this to build the canonical JSONL record and to
    populate the same-chunk mention surface for the relation pass.
    """

    mention_id: str
    chunk_id: str
    doc_id: str
    entity_type: str
    surface_form: str
    canonical_form: str
    evidence: dict
    confidence: str

    def to_record(self) -> dict:
        return {
            "kind": "mention",
            "mention_id": self.mention_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "entity_type": self.entity_type,
            "surface_form": self.surface_form,
            "canonical_form": self.canonical_form,
            "evidence": dict(self.evidence),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ValidatedAssertion:
    """An assertion record that has passed all three gates."""

    assertion_id: str
    chunk_id: str
    doc_id: str
    relation_type: str
    subject_ref: dict
    object_ref: dict
    evidence: dict
    confidence: str

    def to_record(self) -> dict:
        return {
            "kind": "assertion",
            "assertion_id": self.assertion_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "relation_type": self.relation_type,
            "subject_ref": dict(self.subject_ref),
            "object_ref": dict(self.object_ref),
            "evidence": dict(self.evidence),
            "confidence": self.confidence,
        }


def build_validated_mention(
    *,
    raw_mention: dict,
    chunk_id: str,
    doc_id: str,
    source_text_sha256_hex: str,
) -> ValidatedMention:
    """Assign a content-addressed mention_id after the gates pass.

    The ``raw_mention`` is the model-output candidate; this function
    assumes the caller has already passed it through gates 1–3.
    """
    evidence = dict(raw_mention["evidence"])
    # Normalise the stored evidence to the canonical shape: only
    # the three spec-mandated keys, with the SHA256 byte-exact.
    canonical_evidence = {
        "char_start": int(evidence["char_start"]),
        "char_end": int(evidence["char_end"]),
        "source_text_sha256": source_text_sha256_hex,
    }
    mid = ids_module.mention_id(
        chunk_id=chunk_id,
        source_text_sha256=source_text_sha256_hex,
        entity_type=raw_mention["entity_type"],
        char_start=canonical_evidence["char_start"],
        char_end=canonical_evidence["char_end"],
    )
    return ValidatedMention(
        mention_id=mid,
        chunk_id=chunk_id,
        doc_id=doc_id,
        entity_type=raw_mention["entity_type"],
        surface_form=raw_mention["surface_form"],
        canonical_form=raw_mention["canonical_form"],
        evidence=canonical_evidence,
        confidence=raw_mention["confidence"],
    )


def build_validated_assertion(
    *,
    raw_assertion: dict,
    chunk_id: str,
    doc_id: str,
    source_text_sha256_hex: str,
) -> ValidatedAssertion:
    """Assign a content-addressed assertion_id after the gates pass."""
    evidence = dict(raw_assertion["evidence"])
    canonical_evidence = {
        "char_start": int(evidence["char_start"]),
        "char_end": int(evidence["char_end"]),
        "source_text_sha256": source_text_sha256_hex,
    }
    subject_ref = {"kind": raw_assertion["subject_ref"]["kind"],
                   "id": raw_assertion["subject_ref"]["id"]}
    object_ref = {"kind": raw_assertion["object_ref"]["kind"],
                  "id": raw_assertion["object_ref"]["id"]}
    aid = ids_module.assertion_id(
        chunk_id=chunk_id,
        source_text_sha256=source_text_sha256_hex,
        relation_type=raw_assertion["relation_type"],
        subject_ref=subject_ref,
        object_ref=object_ref,
        char_start=canonical_evidence["char_start"],
        char_end=canonical_evidence["char_end"],
    )
    return ValidatedAssertion(
        assertion_id=aid,
        chunk_id=chunk_id,
        doc_id=doc_id,
        relation_type=raw_assertion["relation_type"],
        subject_ref=subject_ref,
        object_ref=object_ref,
        evidence=canonical_evidence,
        confidence=raw_assertion["confidence"],
    )


# ---------- v2 persisted-record dataclasses (#51) --------------------
#
# Following the v1 convention used above, "Validated" means the
# caller has already passed the relevant v2 validation / alignment
# gates. The dataclasses themselves do not perform the gates and
# do not call the deterministic exact-span v2 aligner (#49). They
# carry the canonical v2 record fields and emit the canonical JSONL
# record via ``to_record()``.


@dataclass(frozen=True)
class ValidatedV2Mention:
    """A v2 mention record that has passed the relevant v2 gates (#51).

    Precondition: the caller has already run the v2 evidence and
    ontology gates and obtained the v2 ``mention_id`` from
    ``ids.mention_id_v2``. The literal text (``surface_form``)
    lives at the top level; ``evidence`` contains only the three
    spec-mandated offsets + sha fields, with no literal
    duplication.
    """

    mention_id: str
    chunk_id: str
    doc_id: str
    entity_type: str
    surface_form: str
    canonical_form: str
    evidence: dict
    confidence: str

    def to_record(self) -> dict:
        from src.extract.schema import SCHEMA_VERSION_V2

        return {
            "kind": "mention",
            "schema_version": SCHEMA_VERSION_V2,
            "mention_id": self.mention_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "entity_type": self.entity_type,
            "surface_form": self.surface_form,
            "canonical_form": self.canonical_form,
            "evidence": dict(self.evidence),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ValidatedV2Assertion:
    """A v2 assertion record that has passed the relevant v2 gates (#51).

    Precondition: the caller has already run the v2 evidence and
    ontology gates and obtained the v2 ``assertion_id`` from
    ``ids.assertion_id_v2``. The literal text (``evidence_quote``)
    lives at the top level; ``evidence`` contains only the three
    spec-mandated offsets + sha fields, with no literal
    duplication.
    """

    assertion_id: str
    chunk_id: str
    doc_id: str
    relation_type: str
    subject_ref: dict
    object_ref: dict
    evidence_quote: str
    evidence: dict
    confidence: str

    def to_record(self) -> dict:
        from src.extract.schema import SCHEMA_VERSION_V2

        return {
            "kind": "assertion",
            "schema_version": SCHEMA_VERSION_V2,
            "assertion_id": self.assertion_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "relation_type": self.relation_type,
            "subject_ref": dict(self.subject_ref),
            "object_ref": dict(self.object_ref),
            "evidence_quote": self.evidence_quote,
            "evidence": dict(self.evidence),
            "confidence": self.confidence,
        }