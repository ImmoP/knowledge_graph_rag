"""Fail-closed human gold-record authoring CLI (#65)."""
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
    return Paths(ROOT / "data/derived/chunks.jsonl", ROOT / "data/manifest.jsonl", ROOT / "data/gold_set/sample.v2.jsonl", ROOT / "data/gold_set/sample.v2.audit.json", ROOT / "eval/held_out_leakage.v1.json", ROOT / "ontology.yaml", ROOT / "gold_set/GUIDELINES.md", ROOT / "gold_set/chunks")


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


def template(paths: Paths, chunk_id: str, annotator_id: str) -> dict[str, Any]:
    chunk, _, ontology_version, _ = preflight(paths, chunk_id)
    return {"schema_version": SCHEMA_VERSION, "chunk_id": chunk_id, "doc_id": chunk["doc_id"], "canonical_chunks_sha256_at_authoring": _sha(paths.chunks), "ontology_version": ontology_version, "ontology_sha256": _sha(paths.ontology), "annotation_guideline_version": _guideline_version(paths.guideline), "annotation_guideline_sha256": _sha(paths.guideline), "annotator_id": _string(annotator_id, "annotator_id"), "adjudication_status": "single_pass", "adjudicated_by": None, "annotated_at": None, "entities": [], "relations": []}


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
    if path.exists() and path.read_bytes() != payload:
        raise AuthorError(f"draft destination exists with different bytes: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


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
    return parser


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
        else:
            publish(paths, args.input)
    except AuthorError as exc:
        print(f"author failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
