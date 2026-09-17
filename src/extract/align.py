"""Deterministic exact-span aligner for extraction-contract v2 (#49).

Turns the model's candidate literals (mention ``surface_form`` and
relation ``evidence_quote``) into deterministic Python-string character
offsets into canonical ``chunk.text``. The accepted literal itself
matches canonical text exactly; no semantic repair, no fuzzy match,
no whitespace / Unicode / case normalization, no offset-hint fallback.
UTF-8 byte integrity is carried separately by ``source_text_sha256``.

Pure functions only: no I/O, no provider imports, no manifest writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RejectReason(Enum):
    """Structured rejection reasons emitted by the v2 aligner (#49)."""

    ZERO_OCCURRENCES = "ZERO_OCCURRENCES"
    INVALID_OCCURRENCE_INDEX = "INVALID_OCCURRENCE_INDEX"
    ENDPOINT_NOT_CONTAINED = "ENDPOINT_NOT_CONTAINED"
    EMPTY_SURFACE = "EMPTY_SURFACE"
    NON_STRING_SURFACE = "NON_STRING_SURFACE"


@dataclass(frozen=True)
class Reject:
    """A single structured rejection (#49 acceptance 18).

    ``candidate_id`` is the model's emitted ``id`` field when present,
    else ``None``. ``offending_value`` is the literal/value that
    triggered the rejection.
    """

    chunk_id: str
    candidate_id: str | None
    reason: RejectReason
    offending_value: Any

    def to_record(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "candidate_id": self.candidate_id,
            "reason": self.reason.value,
            "offending_value": self.offending_value,
        }


@dataclass(frozen=True)
class AlignedMention:
    """A mention whose literal ``surface_form`` was aligned into
    canonical ``chunk_text`` via exact Python-string equality."""

    chunk_id: str
    candidate_id: str | None
    entity_type: str
    surface_form: str
    canonical_form: str
    confidence: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class AlignedAssertion:
    """A relation whose literal ``evidence_quote`` was aligned and
    whose mention-kind endpoint spans are contained within it."""

    chunk_id: str
    candidate_id: str | None
    relation_type: str
    subject_ref: dict = field(hash=False)
    object_ref: dict = field(hash=False)
    evidence_quote: str
    confidence: str
    char_start: int
    char_end: int


#: Version stamp of the alignment algorithm itself (#55). Bumped on any
#: change to the literal-alignment rules (occurrence enumeration,
#: endpoint containment, occurrence-index resolution, etc.). The
#: ``ALIGNMENT_ALGORITHM_SHA256`` recorded in the canonical manifest
#: is the sha256 of this module's source bytes and changes whenever
#: the source changes; this constant changes only when the algorithm
#: semantically changes.
ALIGNMENT_ALGORITHM_VERSION = "v2-exact-span-1"


# ---------- enumeration helpers ----------


def _enumerate_occurrences(chunk_text: str, literal: str) -> list[int]:
    """Return the left-to-right, non-overlapping start offsets of
    ``literal`` in ``chunk_text`` using exact Python-string equality.

    After a match at ``start``, the next search begins at
    ``start + len(literal)`` (#49 module-level invariant 16). This
    guarantees a single deterministic ordering and prevents
    ``start + 1`` from generating overlapping / adjacent matches
    across non-contiguous occurrences.
    """
    starts: list[int] = []
    needle_len = len(literal)
    if needle_len == 0:
        return starts
    start = chunk_text.find(literal, 0)
    while start != -1:
        starts.append(start)
        start = chunk_text.find(literal, start + needle_len)
    return starts


def _resolve_occurrence_index(
    count: int,
    index: Any,
    *,
    allow_missing_when_unique: bool,
) -> int | RejectReason:
    """Validate ``occurrence_index`` against the occurrence count.

    ``bool`` is rejected as a non-integer via ``type(index) is int``
    (#49 acceptance 8). ``allow_missing_when_unique`` is ``True`` for
    unique literals so the absence path can be accepted.
    """
    if index is None:
        if count == 1 and allow_missing_when_unique:
            return 0
        return RejectReason.INVALID_OCCURRENCE_INDEX
    # bool is a subclass of int in Python; explicit type() check
    # rejects ``True``/``False`` per #49 acceptance 8.
    if type(index) is not int:
        return RejectReason.INVALID_OCCURRENCE_INDEX
    if index < 0:
        return RejectReason.INVALID_OCCURRENCE_INDEX
    if index >= count:
        return RejectReason.INVALID_OCCURRENCE_INDEX
    return index


def _validate_literal_shape(literal: Any) -> RejectReason | None:
    """Apply the literal-shape rules of #49 (acceptance 9-11)."""
    if not isinstance(literal, str):
        return RejectReason.NON_STRING_SURFACE
    if literal == "":
        return RejectReason.EMPTY_SURFACE
    return None


def _candidate_id(candidate: dict) -> str | None:
    """Return the candidate's model-emitted ``id`` field, if any."""
    cid = candidate.get("id")
    if isinstance(cid, str) and cid:
        return cid
    return None


# ---------- mentions ----------


def align_mentions(
    chunk_text: str,
    candidates: list[dict],
    *,
    chunk_id: str,
) -> tuple[list[AlignedMention], list[Reject]]:
    """Align each candidate's ``surface_form`` into ``chunk_text``.

    Returns a tuple ``(accepted, rejects)``. Inputs are not mutated
    (#49 acceptance 19). Each accepted mention carries derived
    ``char_start`` / ``char_end`` such that
    ``chunk_text[char_start:char_end] == surface_form``
    (#20).
    """
    accepted: list[AlignedMention] = []
    rejects: list[Reject] = []
    for candidate in candidates:
        cid = _candidate_id(candidate)
        literal = candidate.get("surface_form")
        shape_err = _validate_literal_shape(literal)
        if shape_err is not None:
            rejects.append(
                Reject(
                    chunk_id=chunk_id,
                    candidate_id=cid,
                    reason=shape_err,
                    offending_value=literal,
                )
            )
            continue
        starts = _enumerate_occurrences(chunk_text, literal)  # type: ignore[arg-type]
        if not starts:
            rejects.append(
                Reject(
                    chunk_id=chunk_id,
                    candidate_id=cid,
                    reason=RejectReason.ZERO_OCCURRENCES,
                    offending_value=literal,
                )
            )
            continue
        index = candidate.get("occurrence_index", None)
        # Unique literal: missing or index 0 are accepted; any other
        # integer index is invalid (#1, #2, #3).
        resolved = _resolve_occurrence_index(
            len(starts),
            index,
            allow_missing_when_unique=(len(starts) == 1),
        )
        if isinstance(resolved, RejectReason):
            rejects.append(
                Reject(
                    chunk_id=chunk_id,
                    candidate_id=cid,
                    reason=resolved,
                    offending_value=index,
                )
            )
            continue
        char_start = starts[resolved]
        char_end = char_start + len(literal)  # type: ignore[arg-type]
        accepted.append(
            AlignedMention(
                chunk_id=chunk_id,
                candidate_id=cid,
                entity_type=candidate.get("entity_type", ""),
                surface_form=literal,  # type: ignore[arg-type]
                canonical_form=candidate.get("canonical_form", ""),
                confidence=candidate.get("confidence", ""),
                char_start=char_start,
                char_end=char_end,
            )
        )
    return accepted, rejects


# ---------- relations ----------


def _endpoint_span_contained(
    *,
    relation_start: int,
    relation_end: int,
    endpoint: Any,
    aligned_mentions_by_id: dict,
) -> bool:
    """Return True iff a mention-kind endpoint is fully contained in
    the relation span. Document endpoints impose no requirement (#14).
    """
    if not isinstance(endpoint, dict):
        return False
    kind = endpoint.get("kind")
    if kind == "document":
        return True
    if kind != "mention":
        return False
    eid = endpoint.get("id")
    if not isinstance(eid, str) or not eid:
        return False
    span = aligned_mentions_by_id.get(eid)
    if span is None:
        return False
    m_start, m_end = span
    return relation_start <= m_start and m_end <= relation_end


def align_relations(
    chunk_text: str,
    candidates: list[dict],
    aligned_mentions_by_id: dict,
    *,
    chunk_id: str,
) -> tuple[list[AlignedAssertion], list[Reject]]:
    """Align each candidate's ``evidence_quote`` and verify endpoint
    containment (#12-15). Inputs are not mutated (#19).
    """
    accepted: list[AlignedAssertion] = []
    rejects: list[Reject] = []
    for candidate in candidates:
        cid = _candidate_id(candidate)
        literal = candidate.get("evidence_quote")
        shape_err = _validate_literal_shape(literal)
        if shape_err is not None:
            rejects.append(
                Reject(
                    chunk_id=chunk_id,
                    candidate_id=cid,
                    reason=shape_err,
                    offending_value=literal,
                )
            )
            continue
        starts = _enumerate_occurrences(chunk_text, literal)  # type: ignore[arg-type]
        if not starts:
            rejects.append(
                Reject(
                    chunk_id=chunk_id,
                    candidate_id=cid,
                    reason=RejectReason.ZERO_OCCURRENCES,
                    offending_value=literal,
                )
            )
            continue
        index = candidate.get("occurrence_index", None)
        resolved = _resolve_occurrence_index(
            len(starts),
            index,
            allow_missing_when_unique=(len(starts) == 1),
        )
        if isinstance(resolved, RejectReason):
            rejects.append(
                Reject(
                    chunk_id=chunk_id,
                    candidate_id=cid,
                    reason=resolved,
                    offending_value=index,
                )
            )
            continue
        char_start = starts[resolved]
        char_end = char_start + len(literal)  # type: ignore[arg-type]
        subject_ref = candidate.get("subject_ref")
        object_ref = candidate.get("object_ref")
        if not _endpoint_span_contained(
            relation_start=char_start,
            relation_end=char_end,
            endpoint=subject_ref,
            aligned_mentions_by_id=aligned_mentions_by_id,
        ) or not _endpoint_span_contained(
            relation_start=char_start,
            relation_end=char_end,
            endpoint=object_ref,
            aligned_mentions_by_id=aligned_mentions_by_id,
        ):
            rejects.append(
                Reject(
                    chunk_id=chunk_id,
                    candidate_id=cid,
                    reason=RejectReason.ENDPOINT_NOT_CONTAINED,
                    offending_value={
                        "evidence_quote": literal,  # type: ignore[arg-type]
                        "subject_ref": subject_ref,
                        "object_ref": object_ref,
                    },
                )
            )
            continue
        accepted.append(
            AlignedAssertion(
                chunk_id=chunk_id,
                candidate_id=cid,
                relation_type=candidate.get("relation_type", ""),
                subject_ref=dict(subject_ref) if isinstance(subject_ref, dict) else {},
                object_ref=dict(object_ref) if isinstance(object_ref, dict) else {},
                evidence_quote=literal,  # type: ignore[arg-type]
                confidence=candidate.get("confidence", ""),
                char_start=char_start,
                char_end=char_end,
            )
        )
    return accepted, rejects
