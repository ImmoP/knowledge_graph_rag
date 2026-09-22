"""Fail-closed gold-set ann-B review workflow scaffolding (#95, #48 §4).

This module implements the v1 pilot adjudication protocol defined in
``gold_set/GUIDELINES.md`` §9. It freezes the deterministic 24/120 ann-B
sample, gates self-review behind a 7-day wash, validates ann-A-self drafts
without leaking canonical labels into templates, computes cell-level
Cohen's κ on the frozen sample, and emits a deterministic ann-C handoff
report. Canonical ann-A records are read-only inputs.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gold_set import author  # noqa: E402

# --- Frozen protocol constants ------------------------------------------------

ANN_B_ALGORITHM_ID = "ann-b-frozen-sample.v1"
ANN_B_ALGORITHM_VERSION = 1
ANN_B_SEED_HEX = "9e3779b1"
ANN_B_SEED_INT = 0x9E3779B1
ANN_B_SAMPLE_SIZE = 24
ANN_B_DIGEST_NS = b"ann-b-review-v1"
WASH_DAYS = 7

ANN_A_ANNOTATOR = "ann-A"
ANN_A_SELF_ANNOTATOR = "ann-A-self"
ANN_B_ANNOTATOR = "ann-B"
ANN_C_ANNOTATOR = "ann-C"

SINGLE_PASS_STATUS = "single_pass"
DOUBLE_PASS_STATUS = "double_pass"
ADJUDICATED_STATUS = "adjudicated"

REVIEW_REL = "data/gold_set/sample.ann_b.audit.json"
REVIEW_ROOT = "gold_set/reviews"
DRAFTS_ROOT = "gold_set/reviews/drafts"

SIDECAR_SCHEMA_VERSION = "gold-review.v1"

KAPPA_THRESHOLD = 0.70

ENTITY_TYPES = ("method", "dataset", "metric", "task")
CONFIDENCE_LABELS = ("stated", "inferred", "hedged")


# --- Errors ------------------------------------------------------------------


class ReviewError(RuntimeError):
    """Raised when an input or operation violates the review contract."""


# --- Data structures ---------------------------------------------------------


@dataclass(frozen=True)
class FrozenSample:
    """Deterministic 24-chunk ann-B frozen sample."""

    selected: tuple[str, ...]
    fingerprint: str
    source_sample_sha256: str
    source_sample_size: int

    @property
    def selected_set(self) -> frozenset[str]:
        return frozenset(self.selected)


@dataclass(frozen=True)
class WashGuard:
    boundary: datetime
    max_annotated_at: datetime
    remaining: timedelta
    passed: bool


@dataclass(frozen=True)
class Disagreement:
    kind: str
    chunk_id: str
    detail: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "chunk_id": self.chunk_id, "detail": dict(self.detail)}


# --- Path helpers -------------------------------------------------------------


def _paths() -> author.Paths:
    return author._default_paths()


def audit_path(root: Path | None = None) -> Path:
    base = root or ROOT
    return base / REVIEW_REL


def reviews_dir(root: Path | None = None) -> Path:
    base = root or ROOT
    return base / REVIEW_ROOT


def drafts_dir(root: Path | None = None) -> Path:
    base = root or ROOT
    return base / DRAFTS_ROOT


def self_review_path(chunk_id: str, root: Path | None = None) -> Path:
    if not author.CHUNK_ID_RE.match(chunk_id):
        raise ReviewError(f"chunk_id malformed: {chunk_id}")
    return reviews_dir(root) / f"{chunk_id}.json"


# --- Sample enumeration ------------------------------------------------------


def _sample_chunk_ids(paths: author.Paths) -> list[str]:
    rows = author._jsonl(paths.sample)
    if len(rows) != author.SAMPLE_SIZE:
        raise ReviewError(f"sample has {len(rows)} rows; expected {author.SAMPLE_SIZE}")
    ids = [author._string(row.get("chunk_id"), "sample.chunk_id") for row in rows]
    if len(set(ids)) != author.SAMPLE_SIZE:
        raise ReviewError("sample chunk IDs are not distinct")
    return ids


# --- Frozen ann-B sample -----------------------------------------------------


def _chunk_digest(chunk_id: str) -> str:
    payload = ANN_B_DIGEST_NS + b"\x00" + ANN_B_SEED_HEX.encode("utf-8") + b"\x00" + chunk_id.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def freeze_ann_b_sample(paths: author.Paths) -> FrozenSample:
    """Compute the deterministic 24/120 frozen ann-B sample.

    Selection rule (pinned by #95):

        SHA256("ann-b-review-v1\\0<seed-hex>\\0<chunk_id>")

    Sort ascending by ``(digest, chunk_id)`` and select the first 24. The
    seed (``9e3779b1``), algorithm id, and size are non-configurable.
    """
    ids = _sample_chunk_ids(paths)
    ordered = sorted(((_chunk_digest(chunk_id), chunk_id) for chunk_id in ids), key=lambda item: (item[0], item[1]))
    if len({chunk_id for _, chunk_id in ordered}) != len(ids):
        raise ReviewError("digest collisions on the canonical sample; refusing to publish")
    selected = tuple(chunk_id for _, chunk_id in ordered[:ANN_B_SAMPLE_SIZE])
    fingerprint = hashlib.sha256("".join(sorted(selected)).encode("utf-8")).hexdigest()
    source_sha = hashlib.sha256(paths.sample.read_bytes()).hexdigest()
    return FrozenSample(
        selected=selected,
        fingerprint=fingerprint,
        source_sample_sha256=source_sha,
        source_sample_size=len(ids),
    )


def _audit_payload(frozen: FrozenSample, paths: author.Paths) -> dict[str, Any]:
    return {
        "algorithm_id": ANN_B_ALGORITHM_ID,
        "algorithm_version": ANN_B_ALGORITHM_VERSION,
        "seed_hex": ANN_B_SEED_HEX,
        "seed_int": ANN_B_SEED_INT,
        "sample_size": frozen.source_sample_size,
        "selected_count": len(frozen.selected),
        "source_sample_path": os.path.relpath(paths.sample, ROOT).replace(os.sep, "/"),
        "source_sample_sha256": frozen.source_sample_sha256,
        "selected_chunk_ids": list(frozen.selected),
        "selected_chunk_ids_sha256": frozen.fingerprint,
    }


def _audit_body(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _publish_audit(target: Path, payload: Mapping[str, Any]) -> None:
    """Write-once atomic publish: identical -> no-op; divergence -> fail closed."""
    body = _audit_body(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() == body:
            return
        raise ReviewError(
            f"audit preflight failed: existing destination has different bytes; refusing to overwrite: {target}"
        )
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(tmp_fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except OSError as exc:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise ReviewError(f"failed to publish {target}: {exc}") from exc


def publish_ann_b_audit(paths: author.Paths, target: Path | None = None) -> FrozenSample:
    frozen = freeze_ann_b_sample(paths)
    payload = _audit_payload(frozen, paths)
    _publish_audit(target or audit_path(), payload)
    return frozen


def load_ann_b_audit(target: Path, paths: author.Paths) -> FrozenSample:
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ReviewError(f"{target}: expected object")
    if raw.get("algorithm_id") != ANN_B_ALGORITHM_ID:
        raise ReviewError(f"{target}: algorithm_id mismatch")
    if int(raw.get("algorithm_version", -1)) != ANN_B_ALGORITHM_VERSION:
        raise ReviewError(f"{target}: algorithm_version mismatch")
    if raw.get("seed_hex") != ANN_B_SEED_HEX:
        raise ReviewError(f"{target}: seed_hex mismatch")
    if int(raw.get("seed_int", -1)) != ANN_B_SEED_INT:
        raise ReviewError(f"{target}: seed_int mismatch")
    if int(raw.get("sample_size", -1)) != author.SAMPLE_SIZE:
        raise ReviewError(f"{target}: sample_size mismatch")
    if int(raw.get("selected_count", -1)) != ANN_B_SAMPLE_SIZE:
        raise ReviewError(f"{target}: selected_count mismatch")
    selected = raw.get("selected_chunk_ids")
    if not isinstance(selected, list) or len(selected) != ANN_B_SAMPLE_SIZE:
        raise ReviewError(f"{target}: selected_chunk_ids malformed")
    if not all(isinstance(item, str) for item in selected):
        raise ReviewError(f"{target}: selected_chunk_ids must all be strings")
    if len(set(selected)) != ANN_B_SAMPLE_SIZE:
        raise ReviewError(f"{target}: selected_chunk_ids are not distinct")
    fingerprint = raw.get("selected_chunk_ids_sha256")
    if not isinstance(fingerprint, str):
        raise ReviewError(f"{target}: selected_chunk_ids_sha256 missing")
    expected_fp = hashlib.sha256("".join(sorted(selected)).encode("utf-8")).hexdigest()
    if fingerprint != expected_fp:
        raise ReviewError(f"{target}: selected_chunk_ids_sha256 fingerprint mismatch")
    live_sample_sha = hashlib.sha256(paths.sample.read_bytes()).hexdigest()
    if raw.get("source_sample_sha256") != live_sample_sha:
        raise ReviewError(f"{target}: source_sample_sha256 does not match live sample")
    recomputed = freeze_ann_b_sample(paths)
    if list(recomputed.selected) != list(selected):
        raise ReviewError(f"{target}: recomputed selection diverges from audited selection")
    return FrozenSample(
        selected=tuple(selected),
        fingerprint=fingerprint,
        source_sample_sha256=live_sample_sha,
        source_sample_size=int(raw["sample_size"]),
    )


# --- Wash guard --------------------------------------------------------------


_ISO_TZ = re.compile(r"Z$")


def _parse_aware(value: Any, where: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ReviewError(f"{where}: expected non-empty ISO-8601 timestamp")
    candidate = _ISO_TZ.sub("+00:00", value)
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ReviewError(f"{where}: malformed timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise ReviewError(f"{where}: timestamp is naive: {value!r}")
    return parsed.astimezone(timezone.utc)


def _canonical_ann_a_records(paths: author.Paths) -> list[tuple[str, dict[str, Any]]]:
    """Return every canonical ann-A record (validated against the trusted pins).

    Uses ``preflight`` so any divergence in the frozen inputs raises the same
    error the authoring CLI does; we then normalize ``annotated_at`` to
    ``None`` in a copy and run ``validate_record`` so the persisted
    timestamp does not block reading. The original record is never mutated.
    """
    chunk_ids = _sample_chunk_ids(paths)
    if not paths.destination_dir.exists():
        raise ReviewError(f"canonical ann-A directory missing: {paths.destination_dir}")
    records: list[tuple[str, dict[str, Any]]] = []
    for chunk_id in chunk_ids:
        canonical = paths.destination_dir / f"{chunk_id}.json"
        if not canonical.exists():
            raise ReviewError(f"canonical ann-A record missing: {canonical}")
        try:
            payload = json.loads(canonical.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReviewError(f"canonical record malformed: {canonical}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ReviewError(f"canonical record is not an object: {canonical}")
        probe = copy.deepcopy(payload)
        probe["annotated_at"] = None
        author.validate_record(probe, paths)
        records.append((chunk_id, payload))
    if len(records) != author.SAMPLE_SIZE:
        raise ReviewError(f"canonical ann-A coverage is {len(records)}/{author.SAMPLE_SIZE}")
    return records


def _max_annotated_at(paths: author.Paths) -> datetime:
    max_at: datetime | None = None
    for _, record in _canonical_ann_a_records(paths):
        if record.get("annotator_id") != ANN_A_ANNOTATOR:
            raise ReviewError("canonical record annotator_id is not 'ann-A'")
        annotated_at = _parse_aware(record.get("annotated_at"), "annotated_at")
        if max_at is None or annotated_at > max_at:
            max_at = annotated_at
    if max_at is None:
        raise ReviewError("canonical ann-A directory is empty")
    return max_at


def evaluate_wash_guard(paths: author.Paths, now: datetime) -> WashGuard:
    if now.tzinfo is None:
        raise ReviewError("now: must be timezone-aware")
    max_at = _max_annotated_at(paths)
    boundary = max_at + timedelta(days=WASH_DAYS)
    return WashGuard(
        boundary=boundary,
        max_annotated_at=max_at,
        remaining=boundary - now,
        passed=now >= boundary,
    )


def current_utc_now() -> datetime:
    return datetime.now(timezone.utc)


# --- Self-review drafts (ann-A-self) ----------------------------------------


def _validate_review_record(record: Any, paths: author.Paths, *, annotator: str = ANN_A_SELF_ANNOTATOR, status: str = SINGLE_PASS_STATUS) -> dict[str, Any]:
    """Validate a completed review draft against the trusted pins.

    A completed draft carries an aware ``annotated_at`` timestamp and the
    reviewer's ``adjudication_status``/``adjudicated_by`` are not the
    pristine template defaults; normalize only the status/owner fields in a
    copy, let ``validate_record`` pass through, then restore the aware
    timestamp and the canonical ``adjudication_status``/``adjudicated_by``
    in the returned validated record so the sidecar carries both the
    trusted-pin compliance and the human-completion metadata.
    """
    if not isinstance(record, dict):
        raise ReviewError("review-record: expected object")
    if record.get("annotator_id") != annotator:
        raise ReviewError(f"review-record: annotator_id must be {annotator!r}")
    if record.get("adjudication_status") != status:
        raise ReviewError(f"review-record: adjudication_status must be {status!r}")
    aware = _parse_aware(record.get("annotated_at"), "annotated_at")
    probe = copy.deepcopy(record)
    probe["annotated_at"] = None
    probe["adjudication_status"] = SINGLE_PASS_STATUS
    probe["adjudicated_by"] = None
    if not probe.get("note"):
        probe["note"] = f"review by {annotator} (empty note accepted)"
    validated = author.validate_record(probe, paths)
    validated["annotated_at"] = aware.isoformat().replace("+00:00", "Z")
    validated["adjudication_status"] = status
    validated["adjudicated_by"] = None
    return validated


def self_review_template(paths: author.Paths, chunk_id: str) -> dict[str, Any]:
    """Build a fresh, empty ann-A-self draft.

    Uses the existing ``gold-record.v1`` template shape so ``validate_record``
    accepts the draft; the draft starts with ``entities=[]``,
    ``relations=[]``, ``note=None`` and does NOT copy canonical labels or
    notes. ``annotator_id`` is forced to ``ann-A-self``. Provenance for
    the sidecar is recorded via the gold-record pins (canonical chunks
    sha, ontology sha, guideline sha); no top-level ``_meta``.
    """
    template = author.template(paths, chunk_id, ANN_A_SELF_ANNOTATOR)
    template.setdefault('note', None)
    return template


def create_self_review_draft(paths: author.Paths, chunk_id: str, output: Path, *, now: datetime | None = None) -> dict[str, Any]:
    if now is None:
        now = current_utc_now()
    guard = evaluate_wash_guard(paths, now)
    if not guard.passed:
        raise ReviewError(
            f"self-review wash not satisfied: boundary={guard.boundary.isoformat().replace('+00:00', 'Z')} "
            f"now={now.isoformat().replace('+00:00', 'Z')}"
        )
    if not author.CHUNK_ID_RE.match(chunk_id):
        raise ReviewError(f"chunk_id malformed: {chunk_id}")
    author._reject_canonical_destination(paths, output)
    root = reviews_dir().resolve()
    try:
        output.resolve().relative_to(root)
    except ValueError as exc:
        raise ReviewError(f"self-review draft must live under {root}") from exc
    if output.exists():
        raise ReviewError(f"self-review draft already exists: {output}")
    record = self_review_template(paths, chunk_id)
    author._atomic_write_draft(output, author._bytes(record))
    return record


def show_self_review_inputs(paths: author.Paths, chunk_id: str) -> dict[str, Any]:
    chunk, _, _, _ = author.preflight(paths, chunk_id)
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "section_path": chunk.get("section_path"),
        "chunk_index": chunk.get("chunk_index"),
        "text": chunk["text"],
        "ontology_sha256": author._sha(paths.ontology),
        "annotation_guideline_sha256": author._sha(paths.guideline),
        "canonical_chunks_sha256_at_authoring": author._sha(paths.chunks),
    }


# --- Self-review comparison --------------------------------------------------


def _entity_signature(entity: Mapping[str, Any]) -> tuple[str, str, tuple[tuple[int, int, int], ...]]:
    """Stable semantic identity: (entity_type, canonical_form, occurrences).

    Occurrences preserve multiplicity so two distinct mentions of the same
    surface form are not collapsed.
    """
    occurrences = tuple(
        (
            int(item["occurrence_index"]),
            int(item["char_start"]),
            int(item["char_end"]),
        )
        for item in entity.get("occurrences_in_chunk", [])
    )
    return (
        str(entity.get("entity_type", "")),
        str(entity.get("canonical_form", "")),
        occurrences,
    )


def _relation_signature(relation: Mapping[str, Any]) -> tuple[str, tuple[str, str], tuple[str, str], tuple[tuple[int, int], ...]]:
    """Stable semantic identity: (type, subject, object, endpoint_mentions).

    Subject/object are referenced by entity signature (canonical_form +
    occurrence_index), not by local entity_id; endpoint_mentions preserve
    multiplicity.
    """
    subject = relation.get("subject_ref") or {}
    object_ = relation.get("object_ref") or {}
    return (
        str(relation.get("relation_type", "")),
        (str(subject.get("kind", "")), str(subject.get("id", ""))),
        (str(object_.get("kind", "")), str(object_.get("id", ""))),
        tuple(
            (str(item.get("endpoint_role", "")), int(item.get("occurrence_index", -1)))
            for item in relation.get("endpoint_mentions", [])
        ),
    )


def _endpoint_kind(signature: tuple[Any, ...]) -> tuple[str, str]:
    return (signature[1][0], signature[2][0])


def compare_self_review(paths: author.Paths, chunk_id: str, self_record: Mapping[str, Any]) -> dict[str, Any]:
    canonical_path = paths.destination_dir / f"{chunk_id}.json"
    if not canonical_path.exists():
        raise ReviewError(f"canonical ann-A record missing: {canonical_path}")
    canonical_probe = copy.deepcopy(json.loads(canonical_path.read_text(encoding="utf-8")))
    canonical_probe["annotated_at"] = None
    canonical = author.validate_record(canonical_probe, paths)
    self_validated = _validate_review_record(self_record, paths, annotator=ANN_A_SELF_ANNOTATOR)

    canonical_entities = canonical.get("entities", []) or []
    self_entities = self_validated.get("entities", []) or []
    canonical_sigs = [_entity_signature(entity) for entity in canonical_entities]
    self_sigs = [_entity_signature(entity) for entity in self_entities]
    canonical_counter: dict[tuple[str, str, tuple[tuple[int, int, int], ...]], int] = {}
    for sig in canonical_sigs:
        canonical_counter[sig] = canonical_counter.get(sig, 0) + 1
    self_counter: dict[tuple[str, str, tuple[tuple[int, int, int], ...]], int] = {}
    for sig in self_sigs:
        self_counter[sig] = self_counter.get(sig, 0) + 1
    disagreements: list[Disagreement] = []
    canonical_entity_lookup: dict[tuple[str, str, tuple[tuple[int, int, int], ...]], list[dict[str, Any]]] = {}
    for entity in canonical_entities:
        canonical_entity_lookup.setdefault(_entity_signature(entity), []).append(entity)
    self_entity_lookup: dict[tuple[str, str, tuple[tuple[int, int, int], ...]], list[dict[str, Any]]] = {}
    for entity in self_entities:
        self_entity_lookup.setdefault(_entity_signature(entity), []).append(entity)
    for sig in sorted(set(canonical_counter) | set(self_counter)):
        ca = canonical_counter.get(sig, 0)
        sb = self_counter.get(sig, 0)
        if ca != sb:
            disagreements.append(
                Disagreement(
                    kind="entity_count",
                    chunk_id=chunk_id,
                    detail={"signature": list(sig), "canonical": ca, "self": sb},
                )
            )
        pair_count = min(ca, sb)
        for pair_index in range(pair_count):
            canonical_entity = canonical_entity_lookup[sig][pair_index]
            self_entity = self_entity_lookup[sig][pair_index]
            if canonical_entity.get("confidence") != self_entity.get("confidence"):
                disagreements.append(
                    Disagreement(
                        kind="entity_confidence",
                        chunk_id=chunk_id,
                        detail={
                            "signature": list(sig),
                            "canonical": canonical_entity.get("confidence"),
                            "self": self_entity.get("confidence"),
                        },
                    )
                )
            canonical_span = canonical_entity.get("span")
            self_span = self_entity.get("span")
            if canonical_span != self_span:
                disagreements.append(
                    Disagreement(
                        kind="entity_span",
                        chunk_id=chunk_id,
                        detail={"signature": list(sig), "canonical": canonical_span, "self": self_span},
                    )
                )

    canonical_relations = canonical.get("relations", []) or []
    self_relations = self_validated.get("relations", []) or []
    canonical_rel_sigs = [_relation_signature(relation) for relation in canonical_relations]
    self_rel_sigs = [_relation_signature(relation) for relation in self_relations]
    canonical_rel_counter: dict[tuple[Any, ...], int] = {}
    for sig in canonical_rel_sigs:
        canonical_rel_counter[sig] = canonical_rel_counter.get(sig, 0) + 1
    self_rel_counter: dict[tuple[Any, ...], int] = {}
    for sig in self_rel_sigs:
        self_rel_counter[sig] = self_rel_counter.get(sig, 0) + 1
    for sig in sorted(set(canonical_rel_counter) | set(self_rel_counter)):
        ca = canonical_rel_counter.get(sig, 0)
        sb = self_rel_counter.get(sig, 0)
        if ca != sb:
            disagreements.append(
                Disagreement(
                    kind="relation_count",
                    chunk_id=chunk_id,
                    detail={"signature": list(sig), "canonical": ca, "self": sb},
                )
            )

    for relation in canonical_relations + self_relations:
        sig = _relation_signature(relation)
        if sig not in canonical_rel_counter or sig not in self_rel_counter:
            continue
        canonical_rel = next(item for item in canonical_relations if _relation_signature(item) == sig)
        self_rel = next(item for item in self_relations if _relation_signature(item) == sig)
        if canonical_rel.get("confidence") != self_rel.get("confidence"):
            disagreements.append(
                Disagreement(
                    kind="relation_confidence",
                    chunk_id=chunk_id,
                    detail={"signature": list(sig), "canonical": canonical_rel.get("confidence"), "self": self_rel.get("confidence")},
                )
            )
        canonical_kind = _endpoint_kind(_relation_signature(canonical_rel))
        self_kind = _endpoint_kind(_relation_signature(self_rel))
        if canonical_kind != self_kind:
            disagreements.append(
                Disagreement(
                    kind="endpoint_kind",
                    chunk_id=chunk_id,
                    detail={"signature": list(sig), "canonical": list(canonical_kind), "self": list(self_kind)},
                )
            )
        if canonical_rel.get("evidence_quote") != self_rel.get("evidence_quote"):
            disagreements.append(
                Disagreement(
                    kind="evidence",
                    chunk_id=chunk_id,
                    detail={"signature": list(sig), "canonical": canonical_rel.get("evidence_quote"), "self": self_rel.get("evidence_quote")},
                )
            )

    canonical_note = canonical.get("note")
    self_note = self_validated.get("note")
    if (canonical_note or None) != (self_note or None):
        disagreements.append(
            Disagreement(
                kind="note",
                chunk_id=chunk_id,
                detail={"canonical": canonical_note, "self": self_note},
            )
        )

    summary: dict[str, int] = {}
    for disagreement in disagreements:
        summary[disagreement.kind] = summary.get(disagreement.kind, 0) + 1

    sorted_disagreements = [
        disagreement.as_dict()
        for disagreement in sorted(
            disagreements,
            key=lambda item: (item.kind, json.dumps(item.detail, sort_keys=True, ensure_ascii=False)),
        )
    ]
    return {
        "chunk_id": chunk_id,
        "canonical_annotated_at": canonical.get("annotated_at"),
        "disagreements": sorted_disagreements,
        "summary": dict(sorted(summary.items())),
        "agreement": not disagreements,
    }


# --- ann-B union status ------------------------------------------------------


def _completed_self_review_paths(root: Path | None = None) -> list[Path]:
    base = reviews_dir(root)
    if not base.exists():
        return []
    paths: list[Path] = []
    for candidate in sorted(base.iterdir()):
        if not candidate.is_file() or candidate.suffix != ".json":
            continue
        paths.append(candidate)
    return paths


def ann_b_status(paths: author.Paths, frozen: FrozenSample, root: Path | None = None) -> dict[str, Any]:
    completed_disagreements: list[str] = []
    malformed: list[str] = []
    for candidate in _completed_self_review_paths(root):
        chunk_id = candidate.stem
        if not author.CHUNK_ID_RE.match(chunk_id):
            malformed.append(chunk_id)
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            malformed.append(chunk_id)
            continue
        try:
            comparison = compare_self_review(paths, chunk_id, payload)
        except (ReviewError, author.AuthorError):
            malformed.append(chunk_id)
            continue
        if not comparison["agreement"]:
            completed_disagreements.append(chunk_id)
    union = sorted(set(frozen.selected) | set(completed_disagreements))
    membership = {
        chunk_id: (
            "frozen_random"
            if chunk_id in frozen.selected_set
            else "self_disagreement"
        )
        for chunk_id in union
    }
    return {
        "frozen_sample_size": len(frozen.selected),
        "frozen_chunk_ids": list(frozen.selected),
        "frozen_fingerprint": frozen.fingerprint,
        "self_disagreement_chunk_ids": completed_disagreements,
        "self_disagreement_count": len(completed_disagreements),
        "malformed_chunk_ids": sorted(set(malformed)),
        "ann_b_union_chunk_ids": union,
        "ann_b_union_size": len(union),
        "membership": membership,
    }


# --- Sidecar updates (ann-B / ann-C) ----------------------------------------


def load_sidecar(chunk_id: str, root: Path | None = None) -> dict[str, Any] | None:
    sidecar = reviews_dir(root) / f"{chunk_id}.json"
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"sidecar malformed: {sidecar}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewError(f"sidecar is not an object: {sidecar}")
    if payload.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise ReviewError(f"sidecar schema_version mismatch: {sidecar}")
    if payload.get("chunk_id") != chunk_id:
        raise ReviewError(f"sidecar chunk_id mismatch: {sidecar}")
    return payload


def _write_sidecar(sidecar: Path, payload: Mapping[str, Any]) -> None:
    body = author._bytes(dict(payload))  # type: ignore[arg-type]
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    if sidecar.exists():
        raise ReviewError(f"sidecar already exists: {sidecar}")
    author._atomic_write_draft(sidecar, body)


def ann_b_template(paths: author.Paths, chunk_id: str, frozen: FrozenSample, root: Path | None = None) -> dict[str, Any]:
    """Build an ann-B sidecar template.

    Sidecar carries the frozen sample membership reason; ann-B record is
    empty so the first write does not fabricate review cells. Independent
    of canonical ann-A labels / note.
    """
    if chunk_id not in frozen.selected_set:
        raise ReviewError(f"chunk_id {chunk_id} is not in the frozen ann-B sample")
    canonical_path = paths.destination_dir / f"{chunk_id}.json"
    canonical_payload = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical_annotated_at = canonical_payload.get("annotated_at")
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "chunk_id": chunk_id,
        "frozen_sample_membership": "frozen_random",
        "ann_a_self_record": None,
        "self_disagreements": [],
        "ann_b_review": {
            "annotator_id": ANN_B_ANNOTATOR,
            "adjudication_status": DOUBLE_PASS_STATUS,
            "annotated_at": None,
            "entities": [],
            "relations": [],
            "note": None,
        },
        "ann_a_record": None,
        "ann_b_record": None,
        "canonical_annotated_at": canonical_annotated_at,
    }


def complete_self_review(
    paths: author.Paths,
    chunk_id: str,
    self_record: Mapping[str, Any],
    *,
    now: datetime | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Persist a completed ann-A-self draft into the per-chunk sidecar.

    The sidecar uses the same gold-review.v1 wrapper as the ann-B path;
    the ann-A-self record lives at ``ann_a_self_record``, the canonical
    ann-A snapshot at ``ann_a_record``. The ann-B record is left as None
    until ``complete_ann_b_review`` overwrites the same sidecar.
    """
    if now is None:
        now = current_utc_now()
    guard = evaluate_wash_guard(paths, now)
    if not guard.passed:
        raise ReviewError(
            f"self-review wash not satisfied: boundary={guard.boundary.isoformat().replace('+00:00', 'Z')} "
            f"now={now.isoformat().replace('+00:00', 'Z')}"
        )
    if not author.CHUNK_ID_RE.match(chunk_id):
        raise ReviewError(f"chunk_id malformed: {chunk_id}")
    sidecar_path = reviews_dir(root) / f"{chunk_id}.json"
    author._reject_canonical_destination(paths, sidecar_path)
    if sidecar_path.exists():
        raise ReviewError(f"self-review sidecar already exists: {sidecar_path}")
    canonical_payload = json.loads((paths.destination_dir / f"{chunk_id}.json").read_text(encoding="utf-8"))
    self_validated = _validate_review_record(self_record, paths)
    canonical_annotated_at = canonical_payload.get("annotated_at")
    payload = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "chunk_id": chunk_id,
        "frozen_sample_membership": "self_review",
        "ann_a_self_record": self_validated,
        "self_disagreements": [],
        "ann_b_review": None,
        "ann_a_record": canonical_payload,
        "ann_b_record": None,
        "canonical_annotated_at": canonical_annotated_at,
    }
    _write_sidecar(sidecar_path, payload)
    return payload

def complete_ann_b_review(
    paths: author.Paths,
    chunk_id: str,
    review_record: Mapping[str, Any],
    frozen: FrozenSample,
    root: Path | None = None,
) -> dict[str, Any]:
    if not author.CHUNK_ID_RE.match(chunk_id):
        raise ReviewError(f"chunk_id malformed: {chunk_id}")
    sidecar_path = reviews_dir(root) / f"{chunk_id}.json"
    author._reject_canonical_destination(paths, sidecar_path)
    if sidecar_path.exists():
        raise ReviewError(f"sidecar already exists: {sidecar_path}")
    template_payload = ann_b_template(paths, chunk_id, frozen, root=root)
    review_copy = copy.deepcopy(dict(review_record))
    if review_copy.get("annotator_id") != ANN_B_ANNOTATOR:
        raise ReviewError(f"ann-B record annotator_id must be {ANN_B_ANNOTATOR!r}")
    if review_copy.get("adjudication_status") != DOUBLE_PASS_STATUS:
        raise ReviewError("ann-B record adjudication_status must be 'double_pass'")
    _parse_aware(review_copy.get("annotated_at"), "annotated_at")
    # Fill the TOP_LEVEL pins from the trusted template so the reviewer only
    # authors entities/relations/note/annotated_at.
    template_record = author.template(paths, chunk_id, ANN_B_ANNOTATOR)
    for key, value in template_record.items():
        if key in ("entities", "relations", "note", "annotated_at"):
            continue
        review_copy.setdefault(key, value)
    review_validated = _validate_review_record(review_copy, paths, annotator=ANN_B_ANNOTATOR, status=DOUBLE_PASS_STATUS)
    if not review_validated.get("note"):
        review_validated["note"] = "ann-B review (empty note accepted)"
    template_payload["ann_b_review"] = review_validated
    template_payload["ann_a_record"] = json.loads((paths.destination_dir / f"{chunk_id}.json").read_text(encoding="utf-8"))
    template_payload["ann_b_record"] = review_validated
    _write_sidecar(sidecar_path, template_payload)
    return template_payload


# --- Cohen's kappa -----------------------------------------------------------


def _cell_observations(records: Sequence[Mapping[str, Any]], family: str, cell: tuple[Any, ...]) -> list[int]:
    """Build the paired presence/absence observation vector for one concrete cell.

    For each chunk: 1 if the chunk contains any (entity|relation) of the
    requested family/cell, 0 otherwise. Multiplicity is preserved by
    counting; this gives a binary present/absent observation per concrete
    cell across the paired chunks, per the issue contract.
    """
    out: list[int] = []
    if family == "entity_type_confidence":
        entity_type, confidence = cell
        for record in records:
            entities = record.get("entities", []) or []
            present = sum(1 for item in entities if item.get("entity_type") == entity_type and item.get("confidence") == confidence)
            out.append(1 if present > 0 else 0)
        return out
    if family == "relation_type_confidence":
        relation_type, confidence = cell
        for record in records:
            relations = record.get("relations", []) or []
            present = sum(1 for item in relations if item.get("relation_type") == relation_type and item.get("confidence") == confidence)
            out.append(1 if present > 0 else 0)
        return out
    if family == "endpoint_kind":
        subject_kind, object_kind = cell
        for record in records:
            relations = record.get("relations", []) or []
            present = sum(
                1
                for item in relations
                if (item.get("subject_ref") or {}).get("kind") == subject_kind
                and (item.get("object_ref") or {}).get("kind") == object_kind
            )
            out.append(1 if present > 0 else 0)
        return out
    raise ReviewError(f"unknown family: {family}")


def _concrete_cells(records: Sequence[Mapping[str, Any]]) -> dict[str, list[tuple[Any, ...]]]:
    cells: dict[str, list[tuple[Any, ...]]] = {
        "entity_type_confidence": [],
        "relation_type_confidence": [],
        "endpoint_kind": [],
    }
    for record in records:
        for entity in record.get("entities", []) or []:
            cell = (entity.get("entity_type"), entity.get("confidence"))
            if cell not in cells["entity_type_confidence"]:
                cells["entity_type_confidence"].append(cell)
        for relation in record.get("relations", []) or []:
            cell = (relation.get("relation_type"), relation.get("confidence"))
            if cell not in cells["relation_type_confidence"]:
                cells["relation_type_confidence"].append(cell)
            kind_cell = (
                (relation.get("subject_ref") or {}).get("kind"),
                (relation.get("object_ref") or {}).get("kind"),
            )
            if kind_cell not in cells["endpoint_kind"]:
                cells["endpoint_kind"].append(kind_cell)
    cells["entity_type_confidence"] = [
        cell for cell in cells["entity_type_confidence"]
        if cell[0] in ENTITY_TYPES and cell[1] in CONFIDENCE_LABELS
    ]
    return cells


def _binary_kappa(observer_a: Sequence[int], observer_b: Sequence[int]) -> tuple[float | None, dict[str, int]]:
    """Deterministic binary Cohen's κ for paired 0/1 sequences.

    Per the contract:

    - same-uniform observations (both all 0 or both all 1) -> κ undefined
      because chance agreement equals observed agreement;
    - distinct-uniform observations (one all 0, the other all 1) -> κ = 0,
      computed from the closed form, not coerced to undefined.
    """
    if len(observer_a) != len(observer_b):
        raise ReviewError("kappa: observers differ on length")
    if not observer_a:
        raise ReviewError("kappa: empty observation sequences")
    n = len(observer_a)
    a_pos = sum(observer_a)
    b_pos = sum(observer_b)
    agreements = sum(1 for a, b in zip(observer_a, observer_b) if a == b)
    summary = {"n": n, "support_a": a_pos, "support_b": b_pos, "agreements": agreements}
    a_all_zero = a_pos == 0
    a_all_one = a_pos == n
    b_all_zero = b_pos == 0
    b_all_one = b_pos == n
    # Same-uniform on the same category -> κ undefined.
    if (a_all_zero and b_all_zero) or (a_all_one and b_all_one):
        return None, summary
    # Distinct-uniform -> κ=0 (no overlap; expected_agreement=0 too).
    if (a_all_zero and b_all_one) or (a_all_one and b_all_zero):
        return 0.0, summary
    pa_zero = (n - a_pos) / n
    pb_zero = (n - b_pos) / n
    pa_one = a_pos / n
    pb_one = b_pos / n
    expected_agreement = pa_zero * pb_zero + pa_one * pb_one
    observed_agreement = agreements / n
    kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
    return kappa, summary


def _load_observer_records(paths: author.Paths, frozen: FrozenSample, observer_path: Path | None) -> list[dict[str, Any]]:
    """Load paired per-chunk records for observer B from ``observer_path``.

    Observer B records MUST validate under the trusted pins; their
    ``adjudication_status`` is ``double_pass`` and their annotator is
    ``ann-B``. Validation uses the review-specific normalization so the
    actor and aware timestamp are separately enforced.
    """
    if observer_path is None:
        raise ReviewError("observer_b path is required for kappa computation")
    by_chunk: dict[str, dict[str, Any]] = {}
    with observer_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ReviewError(f"{observer_path}:{line_number}: malformed JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ReviewError(f"{observer_path}:{line_number}: expected object")
            chunk_id = payload.get("chunk_id")
            if not isinstance(chunk_id, str):
                raise ReviewError(f"{observer_path}:{line_number}: chunk_id missing")
            by_chunk[chunk_id] = payload
    records: list[dict[str, Any]] = []
    for chunk_id in frozen.selected:
        if chunk_id not in by_chunk:
            raise ReviewError(f"observer_b missing frozen chunk: {chunk_id}")
        payload = by_chunk[chunk_id]
        if payload.get("annotator_id") != ANN_B_ANNOTATOR:
            raise ReviewError(f"{chunk_id}: observer_b annotator_id must be {ANN_B_ANNOTATOR!r}")
        if payload.get("adjudication_status") != DOUBLE_PASS_STATUS:
            raise ReviewError(f"{chunk_id}: observer_b adjudication_status must be 'double_pass'")
        _parse_aware(payload.get("annotated_at"), f"{chunk_id}: annotated_at")
        expected = author.template(paths, chunk_id, ANN_B_ANNOTATOR)
        probe = copy.deepcopy(payload)
        probe["adjudication_status"] = expected["adjudication_status"]
        probe["adjudicated_by"] = expected["adjudicated_by"]
        probe["annotated_at"] = expected["annotated_at"]
        records.append(author.validate_record(probe, paths))
    return records


def compute_kappa(
    paths: author.Paths,
    frozen: FrozenSample,
    observer_b_path: Path,
) -> dict[str, Any]:
    observer_a_records: list[dict[str, Any]] = []
    for chunk_id in frozen.selected:
        canonical_path = paths.destination_dir / f"{chunk_id}.json"
        if not canonical_path.exists():
            raise ReviewError(f"canonical ann-A record missing: {canonical_path}")
        probe = copy.deepcopy(json.loads(canonical_path.read_text(encoding="utf-8")))
        probe["annotated_at"] = None
        observer_a_records.append(author.validate_record(probe, paths))
    observer_b_records = _load_observer_records(paths, frozen, observer_b_path)

    cells = _concrete_cells(observer_a_records + observer_b_records)
    results: dict[str, Any] = {"families": {}, "frozen_chunk_ids": list(frozen.selected), "threshold": KAPPA_THRESHOLD}
    for family in ("entity_type_confidence", "relation_type_confidence", "endpoint_kind"):
        family_results: list[dict[str, Any]] = []
        for cell in sorted(cells[family], key=lambda item: (str(item[0]), str(item[1]))):
            obs_a = _cell_observations(observer_a_records, family, cell)
            obs_b = _cell_observations(observer_b_records, family, cell)
            kappa, summary = _binary_kappa(obs_a, obs_b)
            if kappa is None:
                status = "undefined"
            elif kappa < KAPPA_THRESHOLD:
                status = "escalate"
            else:
                status = "pass"
            entry: dict[str, Any] = {
                "family": family,
                "cell": list(cell),
                "raw_kappa": None if kappa is None else kappa,
                "status": status,
                "support_a": summary["support_a"],
                "support_b": summary["support_b"],
                "n": summary["n"],
                "threshold": KAPPA_THRESHOLD,
            }
            disagreement_chunk_ids = sorted(
                chunk_id
                for chunk_id, a, b in zip(frozen.selected, obs_a, obs_b)
                if a != b
            )
            entry["disagreement_chunk_ids"] = disagreement_chunk_ids
            family_results.append(entry)
        results["families"][family] = family_results
    return results


# --- ann-C handoff -----------------------------------------------------------


def ann_c_handoff(paths: author.Paths, frozen: FrozenSample, root: Path | None = None) -> dict[str, Any]:
    """Build a deterministic ann-C handoff report.

    The handoff is report-only and never mutates canonical records. The
    chunks listed are the union of:

    - self-disagreement chunks (those whose completed sidecar comparison
      disagrees with the canonical ann-A record);
    - frozen-24 chunks whose ann-B sidecar flags at least one concrete κ
      cell below 0.70.

    The original ann-A and ann-B records are preserved in their sidecars
    as ``ann_a_record`` / ``ann_b_record``; this function only reads them
    so the deterministic handoff report can list the snapshots.
    """
    base = reviews_dir(root)
    chunks: list[str] = []
    handoff_sidecars: dict[str, dict[str, Any]] = {}

    # Iterate every sidecar (gold-review.v1 wrapper) under reviews_dir so
    # self-disagreement chunks outside the frozen sample are still surfaced.
    sidecar_ids: list[str] = []
    if base.exists():
        sidecar_ids = sorted(candidate.stem for candidate in base.iterdir() if candidate.is_file() and candidate.suffix == ".json" and candidate.stem != ".kappa")
    sidecar_ids = sorted(set(sidecar_ids) | set(frozen.selected))

    for chunk_id in sidecar_ids:
        try:
            sidecar = load_sidecar(chunk_id, root=root)
        except ReviewError:
            continue
        if sidecar is None:
            continue
        self_record = sidecar.get("ann_a_self_record")
        if self_record is None:
            continue
        try:
            comparison = compare_self_review(paths, chunk_id, self_record)
        except (ReviewError, author.AuthorError):
            continue
        if comparison["agreement"]:
            continue
        if chunk_id not in chunks:
            chunks.append(chunk_id)
        handoff_sidecars[chunk_id] = {
            "ann_a_record": json.loads((paths.destination_dir / f"{chunk_id}.json").read_text(encoding="utf-8")),
            "ann_a_self_record": copy.deepcopy(self_record),
            "self_disagreement_summary": comparison["summary"],
            "ann_b_record": sidecar.get("ann_b_record") or sidecar.get("ann_b_review"),
        }

    # Frozen-24 chunks flagged by κ.
    kappa_path = reviews_dir(root) / ".kappa.json"
    if kappa_path.exists():
        kappa_report = json.loads(kappa_path.read_text(encoding="utf-8"))
        for family, cells in (kappa_report.get("families") or {}).items():
            for entry in cells:
                if entry.get("status") != "escalate":
                    continue
                for chunk_id in entry.get("disagreement_chunk_ids", []) or []:
                    if chunk_id not in chunks:
                        chunks.append(chunk_id)
                    sidecar = load_sidecar(chunk_id, root=root)
                    if sidecar is not None:
                        handoff_sidecars.setdefault(chunk_id, {
                            "ann_a_record": json.loads((paths.destination_dir / f"{chunk_id}.json").read_text(encoding="utf-8")),
                            "ann_a_self_record": sidecar.get("ann_a_self_record"),
                            "self_disagreement_summary": None,
                            "ann_b_record": sidecar.get("ann_b_review"),
                        })
                        handoff_sidecars[chunk_id]["kappa_failing_cells"] = sorted(
                            handoff_sidecars[chunk_id].get("kappa_failing_cells", []) + [{"family": family, "cell": entry.get("cell"), "raw_kappa": entry.get("raw_kappa")}],
                            key=lambda item: (str(item["family"]), json.dumps(item["cell"], sort_keys=True, ensure_ascii=False)),
                        )

    chunks_sorted = sorted(set(chunks))
    return {
        "report_kind": "ann-c-handoff.v1",
        "frozen_sample_size": len(frozen.selected),
        "chunks": chunks_sorted,
        "chunk_count": len(chunks_sorted),
        "sidecars": {chunk_id: handoff_sidecars[chunk_id] for chunk_id in chunks_sorted},
        "invariant": "canonical ann-A records are not mutated by the handoff",
    }


# --- CLI ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed gold-set ann-B review workflow tooling (#95)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze-sample")
    commands.add_parser("wash-guard")
    self_template = commands.add_parser("self-review-template")
    self_template.add_argument("--chunk-id", required=True)
    self_create = commands.add_parser("create-self-review")
    self_create.add_argument("--chunk-id", required=True)
    self_create.add_argument("--output", type=Path, required=True)
    show = commands.add_parser("show-self-review-inputs")
    show.add_argument("--chunk-id", required=True)
    compare = commands.add_parser("compare-self-review")
    compare.add_argument("--chunk-id", required=True)
    compare.add_argument("--input", type=Path, required=True)
    commands.add_parser("ann-b-status")
    ann_b_tpl = commands.add_parser("ann-b-template")
    ann_b_tpl.add_argument("--chunk-id", required=True)
    ann_b_complete = commands.add_parser("complete-ann-b")
    ann_b_complete.add_argument("--chunk-id", required=True)
    ann_b_complete.add_argument("--input", type=Path, required=True)
    kappa = commands.add_parser("kappa")
    kappa.add_argument("--observer-b", type=Path, required=True)
    self_complete = commands.add_parser("complete-self-review")
    self_complete.add_argument("--chunk-id", required=True)
    self_complete.add_argument("--input", type=Path, required=True)
    handoff = commands.add_parser("ann-c-handoff")
    return parser


def _parse_now(value: str | None) -> datetime:
    if value is None:
        return current_utc_now()
    candidate = _ISO_TZ.sub("+00:00", value)
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ReviewError(f"now: malformed timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise ReviewError("now: timestamp is naive")
    return parsed.astimezone(timezone.utc)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = _paths()
    try:
        if args.command == "freeze-sample":
            frozen = publish_ann_b_audit(paths)
            print(json.dumps(_audit_payload(frozen, paths), ensure_ascii=False, sort_keys=True, indent=2))
        elif args.command == "wash-guard":
            now = _parse_now(None)
            guard = evaluate_wash_guard(paths, now)
            print(json.dumps({
                "boundary": guard.boundary.isoformat().replace("+00:00", "Z"),
                "max_annotated_at": guard.max_annotated_at.isoformat().replace("+00:00", "Z"),
                "remaining_seconds": guard.remaining.total_seconds(),
                "passed": guard.passed,
            }, ensure_ascii=False, sort_keys=True, indent=2))
        elif args.command == "self-review-template":
            print(json.dumps(self_review_template(paths, args.chunk_id), ensure_ascii=False, sort_keys=True, indent=2))
        elif args.command == "create-self-review":
            now = _parse_now(None)
            record = create_self_review_draft(paths, args.chunk_id, args.output, now=now)
            print(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2))
        elif args.command == "show-self-review-inputs":
            print(json.dumps(show_self_review_inputs(paths, args.chunk_id), ensure_ascii=False, sort_keys=True, indent=2))
        elif args.command == "compare-self-review":
            payload = json.loads(args.input.read_text(encoding="utf-8"))
            comparison = compare_self_review(paths, args.chunk_id, payload)
            print(json.dumps(comparison, ensure_ascii=False, sort_keys=True, indent=2))
        elif args.command == "ann-b-status":
            frozen = load_ann_b_audit(audit_path(), paths)
            print(json.dumps(ann_b_status(paths, frozen), ensure_ascii=False, sort_keys=True, indent=2))
        elif args.command == "ann-b-template":
            frozen = load_ann_b_audit(audit_path(), paths)
            print(json.dumps(ann_b_template(paths, args.chunk_id, frozen), ensure_ascii=False, sort_keys=True, indent=2))
        elif args.command == "complete-ann-b":
            frozen = load_ann_b_audit(audit_path(), paths)
            payload = json.loads(args.input.read_text(encoding="utf-8"))
            result = complete_ann_b_review(paths, args.chunk_id, payload, frozen)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        elif args.command == "kappa":
            frozen = load_ann_b_audit(audit_path(), paths)
            result = compute_kappa(paths, frozen, args.observer_b)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        elif args.command == "complete-self-review":
            frozen = load_ann_b_audit(audit_path(), paths)
            payload = json.loads(args.input.read_text(encoding="utf-8"))
            result = complete_self_review(paths, args.chunk_id, payload)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        elif args.command == "ann-c-handoff":
            frozen = load_ann_b_audit(audit_path(), paths)
            report = ann_c_handoff(paths, frozen)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        else:
            raise ReviewError(f"unknown command: {args.command}")
    except (ReviewError, author.AuthorError) as exc:
        print(f"review failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
