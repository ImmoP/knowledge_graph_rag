"""Fail-closed human gold-record authoring CLI (#65, #69)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extract.align import _enumerate_occurrences  # noqa: E402

SAMPLE_SIZE = 120
SAMPLE_VERSION = "gold-set-sample.v2"
SCHEMA_VERSION = "gold-record.v1"
ENTITY_TYPES = frozenset(("method", "dataset", "metric", "task"))
CONFIDENCES = frozenset(("stated", "inferred", "hedged"))
TOP_LEVEL = frozenset((
    "schema_version", "chunk_id", "doc_id", "canonical_chunks_sha256_at_authoring",
    "ontology_version", "ontology_sha256", "annotation_guideline_version",
    "annotation_guideline_sha256", "annotator_id", "adjudication_status",
    "adjudicated_by", "annotated_at", "entities", "relations", "note",
))
ENTITY_FIELDS = frozenset(("entity_id", "entity_type", "surface_form", "canonical_form", "span", "occurrences_in_chunk", "confidence"))
RELATION_FIELDS = frozenset(("relation_id", "relation_type", "subject_ref", "object_ref", "evidence_quote", "evidence_span", "endpoint_mentions", "confidence", "note"))
ENTITY_ID_RE = re.compile(r"^ge_([0-9a-f]{8})_(\d+)$")
RELATION_ID_RE = re.compile(r"^gr_([0-9a-f]{8})_(\d+)$")
CHUNK_ID_RE = re.compile(r"^c_([0-9a-f]{16})$")


class AuthorError(RuntimeError):
    """Raised when an input or authored record violates the contract."""


@dataclass(frozen=True)
class Paths:
    chunks: Path
    manifest: Path
    sample: Path
    audit: Path
    leakage: Path
    ontology: Path
    guideline: Path
    destination_dir: Path


def _default_paths() -> Paths:
    return Paths(
        ROOT / "data/derived/chunks.jsonl",
        ROOT / "data/manifest.jsonl",
        ROOT / "data/gold_set/sample.v2.jsonl",
        ROOT / "data/gold_set/sample.v2.audit.json",
        ROOT / "eval/held_out_leakage.v1.json",
        ROOT / "ontology.yaml",
        ROOT / "gold_set/GUIDELINES.md",
        ROOT / "gold_set/chunks",
    )


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise AuthorError(f"cannot read {path}: {exc}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorError(f"malformed JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuthorError(f"{path}: expected JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise AuthorError(f"cannot read JSONL {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuthorError(f"{path}:{line_number}: malformed JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise AuthorError(f"{path}:{line_number}: expected object")
        rows.append(row)
    return rows


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise AuthorError(f"{where}: expected non-empty string")
    return value


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuthorError(f"{where}: expected integer (bool rejected)")
    return value


def _span(value: Any, text: str, literal: str, where: str) -> tuple[int, int]:
    if not isinstance(value, dict) or set(value) != {"char_start", "char_end"}:
        raise AuthorError(f"{where}: expected exactly char_start and char_end")
    start = _integer(value["char_start"], f"{where}.char_start")
    end = _integer(value["char_end"], f"{where}.char_end")
    if start < 0 or end > len(text) or start >= end:
        raise AuthorError(f"{where}: invalid bounds")
    if text[start:end] != literal:
        raise AuthorError(f"{where}: exact literal mismatch")
    return start, end


def _load_ontology(path: Path) -> tuple[int, dict[str, tuple[str, str]]]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise AuthorError(f"malformed ontology {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuthorError("ontology: expected mapping")
    version = _integer(payload.get("version"), "ontology.version")
    relation_types = payload.get("relation_types")
    if not isinstance(relation_types, list):
        raise AuthorError("ontology.relation_types: expected list")
    relations: dict[str, tuple[str, str]] = {}
    for item in relation_types:
        if not isinstance(item, dict):
            raise AuthorError("ontology.relation_types: expected mappings")
        if item.get("population_mode") == "extracted":
            name = _string(item.get("name"), "ontology relation name")
            relations[name] = (_string(item.get("from"), f"{name}.from"), _string(item.get("to"), f"{name}.to"))
    return version, relations


_GUIDELINE_DECL_RE = re.compile(
    r"^[ \t]*annotation_guideline_version\b[^\r\n]*$",
    re.MULTILINE,
)
_GUIDELINE_VALUE_RE = re.compile(
    r"^[ \t]*annotation_guideline_version[ \t]*=[ \t]*(\d+)[ \t]*$"
)


def _guideline_version(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise AuthorError(f"cannot read guideline {path}: {exc}") from exc
    declarations = _GUIDELINE_DECL_RE.findall(content)
    if len(declarations) != 1:
        raise AuthorError("guideline version is not exactly 1")
    match = _GUIDELINE_VALUE_RE.match(declarations[0])
    if match is None or match.group(1) != "1":
        raise AuthorError("guideline version is not exactly 1")
    return "1"


def _reject_canonical_destination(paths: Paths, output: Path) -> None:
    destination = paths.destination_dir.resolve()
    candidate = output.resolve()
    try:
        candidate.relative_to(destination)
    except ValueError:
        return
    raise AuthorError(f"output must not write into canonical destination: {candidate}")


def preflight(paths: Paths, chunk_id: str) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, tuple[str, str]]]:
    """Validate frozen inputs and return the requested canonical chunk."""
    audit = _load_json(paths.audit)
    sample = _jsonl(paths.sample)
    if len(sample) != SAMPLE_SIZE:
        raise AuthorError(f"sample has {len(sample)} rows; expected {SAMPLE_SIZE}")
    ids = [_string(row.get("chunk_id"), "sample.chunk_id") for row in sample]
    if len(set(ids)) != SAMPLE_SIZE:
        raise AuthorError("sample chunk IDs are not distinct")
    if any(row.get("sampler_version") != SAMPLE_VERSION for row in sample):
        raise AuthorError("sample contains a non-v2 sampler record")
    selected_digest = hashlib.sha256("".join(sorted(ids)).encode("utf-8")).hexdigest()
    if audit.get("selected_chunk_ids_sha256") != selected_digest:
        raise AuthorError("sample selected_chunk_ids_sha256 does not match audit")
    for key, path in (("chunks_sha256", paths.chunks), ("manifest_sha256", paths.manifest), ("ontology_sha256", paths.ontology), ("leakage_artifact_sha256", paths.leakage)):
        if audit.get(key) != _sha(path):
            raise AuthorError(f"{key} does not match audit")
    if audit.get("leakage_overlap_count") != 0:
        raise AuthorError("audit leakage_overlap_count is not zero")
    leakage = _load_json(paths.leakage)
    excluded = leakage.get("excluded_chunk_ids")
    if not isinstance(excluded, list) or any(not isinstance(value, str) for value in excluded):
        raise AuthorError("leakage excluded_chunk_ids is malformed")
    if chunk_id not in ids:
        raise AuthorError(f"chunk is not sampled: {chunk_id}")
    if chunk_id in set(excluded):
        raise AuthorError(f"chunk is held-out excluded: {chunk_id}")
    matching_sample = [row for row in sample if row["chunk_id"] == chunk_id]
    if len(matching_sample) != 1:
        raise AuthorError("requested sample chunk is not unique")
    sample_doc_id = _string(matching_sample[0].get("doc_id"), "sample.doc_id")
    chunks = [row for row in _jsonl(paths.chunks) if row.get("chunk_id") == chunk_id]
    if len(chunks) != 1:
        raise AuthorError("canonical chunk row is not unique")
    chunk = chunks[0]
    if _string(chunk.get("doc_id"), "chunk.doc_id") != sample_doc_id:
        raise AuthorError("sample and chunk doc_id differ")
    _string(chunk.get("text"), "chunk.text")
    version, relations = _load_ontology(paths.ontology)
    _guideline_version(paths.guideline)
    return chunk, audit, version, relations


def _annotated_chunk_ids(paths: Paths) -> set[str]:
    """Read every canonical JSON file and return its chunk_id set, fail-closed."""
    if not paths.destination_dir.exists():
        return set()
    annotated: set[str] = set()
    for candidate in paths.destination_dir.iterdir():
        if not candidate.is_file() or candidate.suffix != ".json":
            continue
        # Fail-closed: malformed JSON or wrong filename shape is an error.
        record = _load_json(candidate)
        expected_filename = f"{_string(record.get('chunk_id'), 'record.chunk_id')}.json"
        if candidate.name != expected_filename:
            raise AuthorError(f"canonical filename {candidate.name} does not match record chunk_id {expected_filename}")
        annotated.add(record["chunk_id"])
    return annotated


def status(paths: Paths) -> dict[str, Any]:
    # Validate frozen inputs via the authoritative preflight pipeline.
    sample = _jsonl(paths.sample)
    if len(sample) != SAMPLE_SIZE:
        raise AuthorError(f"sample has {len(sample)} rows; expected {SAMPLE_SIZE}")
    first_chunk_id = _string(sample[0].get("chunk_id"), "sample.chunk_id")
    preflight(paths, first_chunk_id)
    ids = [_string(row.get("chunk_id"), "sample.chunk_id") for row in sample]
    sample_id_set = set(ids)
    annotated = _annotated_chunk_ids(paths)
    if annotated - sample_id_set:
        raise AuthorError(
            f"annotated chunk IDs not in canonical sample: {sorted(annotated - sample_id_set)}"
        )
    next_row: dict[str, Any] | None = None
    for row in sample:
        chunk_id = _string(row.get("chunk_id"), "sample.chunk_id")
        if chunk_id not in annotated:
            next_row = {"chunk_id": chunk_id, "doc_id": _string(row.get("doc_id"), "sample.doc_id")}
            break
    return {
        "annotated": len(annotated),
        "sample_size": SAMPLE_SIZE,
        "remaining": SAMPLE_SIZE - len(annotated),
        "next": next_row,
    }


def template(paths: Paths, chunk_id: str, annotator_id: str) -> dict[str, Any]:
    chunk, _, ontology_version, _ = preflight(paths, chunk_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": chunk_id,
        "doc_id": chunk["doc_id"],
        "canonical_chunks_sha256_at_authoring": _sha(paths.chunks),
        "ontology_version": ontology_version,
        "ontology_sha256": _sha(paths.ontology),
        "annotation_guideline_version": _guideline_version(paths.guideline),
        "annotation_guideline_sha256": _sha(paths.guideline),
        "annotator_id": _string(annotator_id, "annotator_id"),
        "adjudication_status": "single_pass",
        "adjudicated_by": None,
        "annotated_at": None,
        "entities": [],
        "relations": [],
    }


def _validate_entities(value: Any, text: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise AuthorError("entities: expected list")
    entities: dict[str, dict[str, Any]] = {}
    for index, entity in enumerate(value):
        where = f"entities[{index}]"
        if not isinstance(entity, dict) or set(entity) != ENTITY_FIELDS:
            raise AuthorError(f"{where}: fields must be exactly entity schema")
        entity_id = _string(entity.get("entity_id"), f"{where}.entity_id")
        if entity_id in entities:
            raise AuthorError(f"{where}: duplicate entity_id")
        if entity.get("entity_type") not in ENTITY_TYPES:
            raise AuthorError(f"{where}.entity_type: invalid")
        surface = _string(entity.get("surface_form"), f"{where}.surface_form")
        _string(entity.get("canonical_form"), f"{where}.canonical_form")
        if entity.get("confidence") not in CONFIDENCES:
            raise AuthorError(f"{where}.confidence: invalid")
        committed = _span(entity.get("span"), text, surface, f"{where}.span")
        occurrences = entity.get("occurrences_in_chunk")
        if not isinstance(occurrences, list):
            raise AuthorError(f"{where}.occurrences_in_chunk: expected list")
        expected = _enumerate_occurrences(text, surface)
        actual: list[tuple[int, int]] = []
        for occurrence_index, occurrence in enumerate(occurrences):
            if not isinstance(occurrence, dict) or set(occurrence) != {"occurrence_index", "char_start", "char_end"}:
                raise AuthorError(f"{where}.occurrences_in_chunk[{occurrence_index}]: invalid fields")
            if _integer(occurrence["occurrence_index"], f"{where}.occurrence_index") != occurrence_index:
                raise AuthorError(f"{where}.occurrences_in_chunk: indices must be ordered 0..N-1")
            actual.append(_span({"char_start": occurrence["char_start"], "char_end": occurrence["char_end"]}, text, surface, f"{where}.occurrences_in_chunk[{occurrence_index}]"))
        expected_spans = [(start, start + len(surface)) for start in expected]
        if actual != expected_spans:
            raise AuthorError(f"{where}.occurrences_in_chunk: not complete exact occurrence enumeration")
        if committed not in actual:
            raise AuthorError(f"{where}.span: not an enumerated occurrence")
        entities[entity_id] = entity
    return entities


def _validate_endpoint(value: Any, expected_type: str, doc_id: str, entities: dict[str, dict[str, Any]], where: str) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {"kind", "id"}:
        raise AuthorError(f"{where}: expected exactly kind and id")
    kind, endpoint_id = value.get("kind"), value.get("id")
    if expected_type == "document":
        if kind != "document" or endpoint_id != doc_id:
            raise AuthorError(f"{where}: must be current document endpoint")
        return None
    if kind != "mention" or not isinstance(endpoint_id, str) or endpoint_id not in entities:
        raise AuthorError(f"{where}: dangling mention endpoint")
    entity = entities[endpoint_id]
    if entity["entity_type"] != expected_type:
        raise AuthorError(f"{where}: mention endpoint type mismatch")
    return entity


def _validate_relations(value: Any, text: str, doc_id: str, entities: dict[str, dict[str, Any]], signatures: dict[str, tuple[str, str]]) -> None:
    if not isinstance(value, list):
        raise AuthorError("relations: expected list")
    ids: set[str] = set()
    for index, relation in enumerate(value):
        where = f"relations[{index}]"
        if not isinstance(relation, dict) or not ({*relation} == RELATION_FIELDS or set(relation) == RELATION_FIELDS - {"note"}):
            raise AuthorError(f"{where}: fields must be relation schema")
        relation_id = _string(relation.get("relation_id"), f"{where}.relation_id")
        if relation_id in ids:
            raise AuthorError(f"{where}: duplicate relation_id")
        ids.add(relation_id)
        relation_type = relation.get("relation_type")
        if relation_type not in signatures:
            raise AuthorError(f"{where}.relation_type: not extracted ontology relation")
        subject_type, object_type = signatures[relation_type]
        subject = _validate_endpoint(relation.get("subject_ref"), subject_type, doc_id, entities, f"{where}.subject_ref")
        object_ = _validate_endpoint(relation.get("object_ref"), object_type, doc_id, entities, f"{where}.object_ref")
        quote = _string(relation.get("evidence_quote"), f"{where}.evidence_quote")
        evidence = _span(relation.get("evidence_span"), text, quote, f"{where}.evidence_span")
        if relation.get("confidence") not in CONFIDENCES:
            raise AuthorError(f"{where}.confidence: invalid")
        if "note" in relation and not isinstance(relation["note"], str):
            raise AuthorError(f"{where}.note: expected string")
        mentions = relation.get("endpoint_mentions")
        if not isinstance(mentions, list):
            raise AuthorError(f"{where}.endpoint_mentions: expected list")
        expected_mentions: dict[str, int] = {}
        for role, entity in (("subject", subject), ("object", object_)):
            if entity is None:
                continue
            start, end = _span(entity["span"], text, entity["surface_form"], f"{where}.{role} entity span")
            if not (evidence[0] <= start and end <= evidence[1]):
                raise AuthorError(f"{where}: {role} mention is not contained in evidence")
            starts = _enumerate_occurrences(text, entity["surface_form"])
            if len(starts) >= 2:
                expected_mentions[role] = starts.index(start)
        actual_mentions: dict[str, int] = {}
        for mention_index, mention in enumerate(mentions):
            if not isinstance(mention, dict) or set(mention) != {"endpoint_role", "occurrence_index"}:
                raise AuthorError(f"{where}.endpoint_mentions[{mention_index}]: invalid fields")
            role = mention.get("endpoint_role")
            if role not in ("subject", "object") or role in actual_mentions:
                raise AuthorError(f"{where}.endpoint_mentions: duplicate or invalid endpoint role")
            actual_mentions[role] = _integer(mention.get("occurrence_index"), f"{where}.endpoint_mentions[{mention_index}].occurrence_index")
        if actual_mentions != expected_mentions:
            raise AuthorError(f"{where}.endpoint_mentions: must exactly represent multi-occurrence endpoints")


def validate_record(record: Any, paths: Paths) -> dict[str, Any]:
    if not isinstance(record, dict) or not set(record) <= TOP_LEVEL or not (set(record) | {"note"}) >= TOP_LEVEL:
        raise AuthorError("record: unknown or missing top-level fields")
    chunk_id = _string(record.get("chunk_id"), "chunk_id")
    chunk, _, ontology_version, signatures = preflight(paths, chunk_id)
    expected = template(paths, chunk_id, _string(record.get("annotator_id"), "annotator_id"))
    for key in ("schema_version", "chunk_id", "doc_id", "canonical_chunks_sha256_at_authoring", "ontology_version", "ontology_sha256", "annotation_guideline_version", "annotation_guideline_sha256", "adjudication_status", "adjudicated_by", "annotated_at"):
        if record.get(key) != expected[key]:
            raise AuthorError(f"record.{key}: does not match initial-authoring contract")
    if record.get("entities") is None or record.get("relations") is None:
        raise AuthorError("entities and relations are required")
    if "note" in record and (not isinstance(record["note"], str)):
        raise AuthorError("record.note: expected string")
    entities = _validate_entities(record["entities"], chunk["text"])
    if not entities and record["relations"] != []:
        raise AuthorError("empty entity list requires empty relations")
    _validate_relations(record["relations"], chunk["text"], chunk["doc_id"], entities, signatures)
    if not entities and (not isinstance(record.get("note"), str) or not record["note"]):
        raise AuthorError("empty entity list requires non-empty top-level note")
    return record


def _bytes(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _chunk_token(chunk_id: str) -> str:
    """First 8 hex chars of chunk_id after the ``c_`` prefix."""
    match = CHUNK_ID_RE.match(chunk_id)
    if match is None:
        raise AuthorError(f"chunk_id: malformed: {chunk_id}")
    return match.group(1)[:8]


def _next_entity_id(entities: list[dict[str, Any]], chunk_id: str) -> str:
    """Next ``ge_<token>_<seq>`` ID using max existing + 1. Mixed/malformed rejected."""
    token = _chunk_token(chunk_id)
    highest = 0
    for entity in entities:
        identifier = entity.get("entity_id")
        match = ENTITY_ID_RE.match(identifier) if isinstance(identifier, str) else None
        if match is None:
            raise AuthorError(f"entity_id: unexpected format: {identifier!r}")
        existing_token, seq_str = match.group(1), match.group(2)
        if existing_token != token:
            raise AuthorError(f"entity_id: chunk-token mismatch: {identifier!r}")
        try:
            seq = int(seq_str)
        except ValueError as exc:
            raise AuthorError(f"entity_id: unexpected sequence: {identifier!r}") from exc
        if seq < 1:
            raise AuthorError(f"entity_id: sequence must be >= 1: {identifier!r}")
        highest = max(highest, seq)
    return f"ge_{token}_{highest + 1:02d}"


def _next_relation_id(relations: list[dict[str, Any]], chunk_id: str) -> str:
    """Next ``gr_<token>_<seq>`` ID using max existing + 1. Mixed/malformed rejected."""
    token = _chunk_token(chunk_id)
    highest = 0
    for relation in relations:
        identifier = relation.get("relation_id")
        match = RELATION_ID_RE.match(identifier) if isinstance(identifier, str) else None
        if match is None:
            raise AuthorError(f"relation_id: unexpected format: {identifier!r}")
        existing_token, seq_str = match.group(1), match.group(2)
        if existing_token != token:
            raise AuthorError(f"relation_id: chunk-token mismatch: {identifier!r}")
        try:
            seq = int(seq_str)
        except ValueError as exc:
            raise AuthorError(f"relation_id: unexpected sequence: {identifier!r}") from exc
        if seq < 1:
            raise AuthorError(f"relation_id: sequence must be >= 1: {identifier!r}")
        highest = max(highest, seq)
    return f"gr_{token}_{highest + 1:02d}"


def _load_draft(paths: Paths, draft_path: Path) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, tuple[str, str]]]:
    _reject_canonical_destination(paths, draft_path)
    record = _load_json(draft_path)
    chunk_id = _string(record.get("chunk_id"), "chunk_id")
    chunk, _, _, signatures = preflight(paths, chunk_id)
    if record.get("doc_id") != chunk["doc_id"]:
        raise AuthorError("record.doc_id: does not match canonical chunk")
    text = _string(chunk.get("text"), "chunk.text")
    if not isinstance(record.get("entities"), list) or not isinstance(record.get("relations"), list):
        raise AuthorError("record: entities and relations must be lists")
    return record, text, chunk, signatures

def _is_pristine_template(record: dict[str, Any]) -> bool:
    """The legitimate pristine template state: entities and relations empty, no top-level note."""
    if record.get("entities") != [] or record.get("relations") != []:
        return False
    return "note" not in record


def _validate_baseline_draft(record: dict[str, Any], paths: Paths) -> None:
    """Fail-closed validation of the existing draft before any semantic mutation.

    A pristine template (empty entities, empty relations, no top-level note) is not
    yet publish-valid because validate_record requires a non-empty note for empty
    annotation. Inject a temporary non-empty note into a *copy* solely to drive
    validation; the original record and on-disk draft are never touched.
    """
    probe = dict(record)
    if _is_pristine_template(record):
        # Pristine template is intentionally not yet publish-valid; inject a non-empty
        # probe note into the COPY so validate_record can exercise every rule that would
        # otherwise reject the empty/no-note state. The original record and on-disk
        # draft are never touched.
        probe["note"] = "_baseline_validation_probe"
    if not isinstance(probe.get("entities"), list) or not isinstance(probe.get("relations"), list):
        raise AuthorError("record: entities and relations must be lists")
    validate_record(probe, paths)


def _atomic_write_draft(path: Path, payload: bytes) -> None:
    """Same-directory atomic replace: temp file + flush + fsync + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temp_path = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        os.replace(temp_path, path)
    except OSError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise AuthorError(f"failed to write draft {path}: {exc}") from exc


def add_entity(
    paths: Paths,
    draft_path: Path,
    entity_type: str,
    surface_form: str,
    canonical_form: str,
    confidence: str,
    occurrence_index: int | None,
    note: str | None,
) -> dict[str, Any]:
    if entity_type not in ENTITY_TYPES:
        raise AuthorError("--entity-type: invalid")
    if confidence not in CONFIDENCES:
        raise AuthorError("--confidence: invalid")
    record, text, _, _ = _load_draft(paths, draft_path)
    _validate_baseline_draft(record, paths)
    if not surface_form:
        raise AuthorError("--surface-form: expected non-empty string")
    occurrences = _enumerate_occurrences(text, surface_form)
    if not occurrences:
        raise AuthorError("--surface-form: not an exact occurrence in chunk text")
    if occurrence_index is None:
        chosen = 0
    else:
        if occurrence_index < 0 or occurrence_index >= len(occurrences):
            raise AuthorError("--occurrence-index: out of range")
        chosen = occurrence_index
    entities: list[dict[str, Any]] = list(record["entities"])
    for existing in entities:
        if existing["entity_type"] == entity_type and existing["surface_form"] == surface_form:
            raise AuthorError("--surface-form: duplicate entity_type + surface_form already in draft")
    if chosen != 0 and not note:
        raise AuthorError("--note: required when --occurrence-index selects a non-default occurrence")
    committed = occurrences[chosen]
    candidate_entities = list(entities)
    new_entity = {
        "entity_id": _next_entity_id(candidate_entities, record["chunk_id"]),
        "entity_type": entity_type,
        "surface_form": surface_form,
        "canonical_form": canonical_form,
        "span": {"char_start": committed, "char_end": committed + len(surface_form)},
        "occurrences_in_chunk": [
            {"occurrence_index": index, "char_start": start, "char_end": start + len(surface_form)}
            for index, start in enumerate(occurrences)
        ],
        "confidence": confidence,
    }
    candidate_entities.append(new_entity)
    candidate = dict(record)
    candidate["entities"] = candidate_entities
    if note:
        existing_note = record.get("note")
        if isinstance(existing_note, str) and existing_note:
            candidate["note"] = existing_note + "\n" + note
        else:
            candidate["note"] = note
    # Validate the candidate end-to-end before any write.
    validate_record(candidate, paths)
    _atomic_write_draft(draft_path, _bytes(candidate))
    return candidate


def add_relation(
    paths: Paths,
    draft_path: Path,
    relation_type: str,
    subject_ref: dict[str, str],
    object_ref: dict[str, str],
    evidence_quote: str,
    confidence: str,
    evidence_occurrence_index: int | None,
) -> dict[str, Any]:
    if confidence not in CONFIDENCES:
        raise AuthorError("--confidence: invalid")
    # Existing draft must already be a valid authored record before relation mutation.
    record, text, chunk, signatures = _load_draft(paths, draft_path)
    validate_record(record, paths)
    if relation_type not in signatures:
        raise AuthorError("--relation-type: not extracted ontology relation")
    subject_type, object_type = signatures[relation_type]
    entities_by_id: dict[str, dict[str, Any]] = {}
    for entity in record["entities"]:
        entities_by_id[_string(entity.get("entity_id"), "entity_id")] = entity

    def _resolve_endpoint(ref: dict[str, str], expected_type: str, role: str) -> dict[str, Any] | None:
        if not isinstance(ref, dict) or set(ref) != {"kind", "id"}:
            raise AuthorError(f"--{role}-ref: expected exactly kind and id")
        kind, endpoint_id = ref.get("kind"), ref.get("id")
        if expected_type == "document":
            if kind != "document" or endpoint_id != chunk["doc_id"]:
                raise AuthorError(f"--{role}-ref: must be current document endpoint")
            return None
        if kind != "mention" or not isinstance(endpoint_id, str) or endpoint_id not in entities_by_id:
            raise AuthorError(f"--{role}-ref: dangling mention endpoint")
        target = entities_by_id[endpoint_id]
        if target["entity_type"] != expected_type:
            raise AuthorError(f"--{role}-ref: mention endpoint type mismatch")
        return target

    subject_entity = _resolve_endpoint(subject_ref, subject_type, "subject")
    object_entity = _resolve_endpoint(object_ref, object_type, "object")
    if not evidence_quote:
        raise AuthorError("--evidence-quote: expected non-empty string")
    quote_occurrences = _enumerate_occurrences(text, evidence_quote)
    if not quote_occurrences:
        raise AuthorError("--evidence-quote: not an exact occurrence in chunk text")
    if len(quote_occurrences) >= 2 and evidence_occurrence_index is None:
        raise AuthorError("--evidence-occurrence-index: required for repeated evidence quote")
    if evidence_occurrence_index is None:
        chosen_quote = quote_occurrences[0]
    else:
        if evidence_occurrence_index < 0 or evidence_occurrence_index >= len(quote_occurrences):
            raise AuthorError("--evidence-occurrence-index: out of range")
        chosen_quote = quote_occurrences[evidence_occurrence_index]
    evidence_span = (chosen_quote, chosen_quote + len(evidence_quote))
    endpoint_mentions: list[dict[str, Any]] = []
    for role, entity in (("subject", subject_entity), ("object", object_entity)):
        if entity is None:
            continue
        start = _integer(entity["span"]["char_start"], "entity span")
        end = _integer(entity["span"]["char_end"], "entity span")
        if not (evidence_span[0] <= start and end <= evidence_span[1]):
            raise AuthorError(f"--evidence-quote: {role} mention is not contained in evidence")
        starts = _enumerate_occurrences(text, entity["surface_form"])
        if len(starts) >= 2:
            endpoint_mentions.append({"endpoint_role": role, "occurrence_index": starts.index(start)})
    new_relation = {
        "relation_id": _next_relation_id(list(record["relations"]), record["chunk_id"]),
        "relation_type": relation_type,
        "subject_ref": dict(subject_ref),
        "object_ref": dict(object_ref),
        "evidence_quote": evidence_quote,
        "evidence_span": {"char_start": evidence_span[0], "char_end": evidence_span[1]},
        "endpoint_mentions": endpoint_mentions,
        "confidence": confidence,
    }
    candidate = dict(record)
    candidate["relations"] = list(record["relations"]) + [new_relation]
    # Validate the candidate end-to-end before any write.
    validate_record(candidate, paths)
    _atomic_write_draft(draft_path, _bytes(candidate))
    return candidate


def mark_empty(paths: Paths, draft_path: Path, note: str) -> dict[str, Any]:
    if not note:
        raise AuthorError("--note: required for mark-empty")
    record, _, _, _ = _load_draft(paths, draft_path)
    _validate_baseline_draft(record, paths)
    if record["entities"] != [] or record["relations"] != []:
        raise AuthorError("mark-empty: draft must have empty entities and relations")
    candidate = dict(record)
    candidate["note"] = note
    # Validate the candidate end-to-end before any write.
    validate_record(candidate, paths)
    _atomic_write_draft(draft_path, _bytes(candidate))
    return candidate


def validate_draft(paths: Paths, draft_path: Path) -> dict[str, Any]:
    record, _, _, _ = _load_draft(paths, draft_path)
    return validate_record(record, paths)


def _write_once(path: Path, payload: bytes) -> bool:
    if path.exists():
        if path.read_bytes() == payload:
            return False
        raise AuthorError(f"existing canonical record differs; refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = handle.name
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() == payload:
                return False
            raise AuthorError(f"existing canonical record differs; refusing overwrite: {path}") from None
        finally:
            Path(temporary).unlink(missing_ok=True)
        return True
    except OSError as exc:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
        raise AuthorError(f"failed to publish {path}: {exc}") from exc


def publish(paths: Paths, input_path: Path) -> bool:
    record = _load_json(input_path)
    validate_record(record, paths)
    final = dict(record)
    final["annotated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    destination = paths.destination_dir / f"{final['chunk_id']}.json"
    if destination.exists():
        try:
            existing = _load_json(destination)
        except AuthorError:
            raise AuthorError(f"existing canonical record differs; refusing overwrite: {destination}") from None
        if existing.get("annotated_at") is not None:
            final["annotated_at"] = existing["annotated_at"]
    return _write_once(destination, _bytes(final))


def _write_draft(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise AuthorError(
            f"draft destination exists with different bytes: {path}"
        )
    _atomic_write_draft(path, payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed gold-record authoring tool")
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show")
    show.add_argument("--chunk-id", required=True)
    draft = commands.add_parser("template")
    draft.add_argument("--chunk-id", required=True)
    draft.add_argument("--annotator-id", required=True)
    draft.add_argument("--output", type=Path, required=True)
    publish_parser = commands.add_parser("publish")
    publish_parser.add_argument("--input", type=Path, required=True)
    commands.add_parser("status")
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--input", type=Path, required=True)
    add_entity_parser = commands.add_parser("add-entity")
    add_entity_parser.add_argument("--input", type=Path, required=True)
    add_entity_parser.add_argument("--entity-type", required=True)
    add_entity_parser.add_argument("--surface-form", required=True)
    add_entity_parser.add_argument("--canonical-form", required=True)
    add_entity_parser.add_argument("--confidence", required=True)
    add_entity_parser.add_argument("--occurrence-index", type=int, default=None)
    add_entity_parser.add_argument("--note", default=None)
    add_relation_parser = commands.add_parser("add-relation")
    add_relation_parser.add_argument("--input", type=Path, required=True)
    add_relation_parser.add_argument("--relation-type", required=True)
    add_relation_parser.add_argument("--subject-ref", required=True)
    add_relation_parser.add_argument("--object-ref", required=True)
    add_relation_parser.add_argument("--evidence-quote", required=True)
    add_relation_parser.add_argument("--confidence", required=True)
    add_relation_parser.add_argument("--evidence-occurrence-index", type=int, default=None)
    mark_empty_parser = commands.add_parser("mark-empty")
    mark_empty_parser.add_argument("--input", type=Path, required=True)
    mark_empty_parser.add_argument("--note", required=True)
    return parser


def _resolve_endpoint_token(
    record: dict[str, Any],
    signatures: dict[str, tuple[str, str]],
    relation_type: str,
    role: str,
    token: str,
) -> dict[str, str]:
    """Resolve the literal ``document`` or entity_id token to a ``{kind, id}`` dict."""
    if relation_type not in signatures:
        raise AuthorError(f"--relation-type: not extracted ontology relation: {relation_type}")
    expected_type = signatures[relation_type][0 if role == "subject" else 1]
    if token == "document":
        if expected_type != "document":
            raise AuthorError(f"--{role}-ref: ontology expects {expected_type}; 'document' rejected")
        return {"kind": "document", "id": record["doc_id"]}
    entities = record.get("entities", [])
    entities_by_id = {entity.get("entity_id"): entity for entity in entities if isinstance(entity, dict)}
    entity = entities_by_id.get(token)
    if entity is None:
        raise AuthorError(f"--{role}-ref: unknown entity_id in draft: {token}")
    if entity.get("entity_type") != expected_type:
        raise AuthorError(
            f"--{role}-ref: entity_id type {entity.get('entity_type')!r} != ontology expected {expected_type!r}"
        )
    return {"kind": "mention", "id": token}


def _prepare_add_relation(paths: Paths, draft_path: Path, relation_type: str, subject_token: str, object_token: str) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """Load draft and resolve endpoint tokens to ``{kind, id}`` dicts."""
    record, _, _, signatures = _load_draft(paths, draft_path)
    subject_ref = _resolve_endpoint_token(record, signatures, relation_type, "subject", subject_token)
    object_ref = _resolve_endpoint_token(record, signatures, relation_type, "object", object_token)
    return record, subject_ref, object_ref


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = _default_paths()
    try:
        if args.command == "show":
            chunk, _, _, _ = preflight(paths, args.chunk_id)
            print(json.dumps({"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"], "section_path": chunk.get("section_path"), "chunk_index": chunk.get("chunk_index"), "text": chunk["text"]}, ensure_ascii=False, sort_keys=True, indent=2))
        elif args.command == "template":
            _reject_canonical_destination(paths, args.output)
            _write_draft(args.output, _bytes(template(paths, args.chunk_id, args.annotator_id)))
        elif args.command == "publish":
            publish(paths, args.input)
        elif args.command == "status":
            print(json.dumps(status(paths), ensure_ascii=False, sort_keys=True, indent=2))
        elif args.command == "validate":
            validate_draft(paths, args.input)
        elif args.command == "add-entity":
            add_entity(paths, args.input, args.entity_type, args.surface_form, args.canonical_form, args.confidence, args.occurrence_index, args.note)
        elif args.command == "add-relation":
            _, subject_ref, object_ref = _prepare_add_relation(paths, args.input, args.relation_type, args.subject_ref, args.object_ref)
            add_relation(paths, args.input, args.relation_type, subject_ref, object_ref, args.evidence_quote, args.confidence, args.evidence_occurrence_index)
        elif args.command == "mark-empty":
            mark_empty(paths, args.input, args.note)
        else:
            raise AuthorError(f"unknown command: {args.command}")
    except AuthorError as exc:
        print(f"author failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
