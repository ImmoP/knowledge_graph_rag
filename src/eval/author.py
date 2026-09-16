"""Strict validation and frozen publication for hand-authored eval drafts (#31).

This module validates canonical artifacts and explicit authored paths, then
publishes exact validated draft bytes without generating questions.
"""
from __future__ import annotations

from collections import Counter
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable, Protocol

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = ROOT / "data" / "manifest.jsonl"
DEFAULT_CHUNKS_PATH = ROOT / "data" / "derived" / "chunks.jsonl"
DEFAULT_ONTOLOGY_PATH = ROOT / "ontology.yaml"

_QID_RE = re.compile(r"^q_[0-9]{3}$")
_DOC_ID_RE = re.compile(r"^W[0-9]+$")
_CHUNK_ID_RE = re.compile(r"^c_[0-9a-f]{16}$")
_ENTITY_ID_RE = re.compile(r"^e_[0-9a-f]{16}$")
_REQUIRED_FIELDS = (
    "qid",
    "question",
    "expected_answer",
    "hop_bucket",
    "evidence_path",
    "supporting_doc_ids",
    "supporting_chunk_ids",
    "split",
)


@dataclass(frozen=True)
class Violation:
    """One draft or canonical-artifact contract violation."""

    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message}"


@dataclass(frozen=True)
class ValidationReport:
    """Immutable result of validating an evaluation draft."""

    violations: tuple[Violation, ...]
    record_count: int
    bucket_counts: tuple[tuple[str, int], ...]

    @property
    def ok(self) -> bool:
        return not self.violations


class GraphReader(Protocol):
    """Read-only lookup surface for validating authored graph paths."""

    def node_exists(self, node_id: str) -> bool: ...

    def stored_edge_exists(self, source_id: str, relation: str, target_id: str) -> bool: ...


class _Neo4jGraphReader:
    """Read-only adapter over the official Neo4j driver."""

    def __init__(self, driver: object, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database

    def node_exists(self, node_id: str) -> bool:
        if _DOC_ID_RE.fullmatch(node_id):
            predicate = "n:Document AND n.doc_id = $node_id"
        elif _ENTITY_ID_RE.fullmatch(node_id):
            predicate = "n:Entity AND n.entity_id = $node_id"
        else:
            return False
        return self._exists(f"MATCH (n) WHERE {predicate} RETURN 1 LIMIT 1", {"node_id": node_id})

    def stored_edge_exists(self, source_id: str, relation: str, target_id: str) -> bool:
        return self._exists(
            "MATCH (source)-[r]->(target) "
            "WHERE type(r) = $relation "
            "AND ((source:Document AND source.doc_id = $source_id) "
            "OR (source:Entity AND source.entity_id = $source_id)) "
            "AND ((target:Document AND target.doc_id = $target_id) "
            "OR (target:Entity AND target.entity_id = $target_id)) "
            "RETURN 1 LIMIT 1",
            {"source_id": source_id, "relation": relation, "target_id": target_id},
        )

    def _exists(self, query: str, parameters: dict[str, str]) -> bool:
        try:
            with self._driver.session(  # type: ignore[attr-defined]
                database=self._database, default_access_mode="READ"
            ) as session:
                return session.run(query, **parameters).single() is not None
        except Exception as exc:
            raise RuntimeError("cannot read graph state") from exc

    def close(self) -> None:
        self._driver.close()  # type: ignore[attr-defined]


def default_graph_reader(*, database: str = "neo4j") -> GraphReader:
    """Create the production read-only graph reader using ingestion credentials."""
    from src.build_graph.ingestion import default_driver

    return _Neo4jGraphReader(default_driver(), database)

def _read_jsonl_objects(path: Path, *, artifact: str) -> tuple[list[dict[str, Any]], list[Violation]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [], [Violation(artifact, f"unreadable: {exc}")]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [], [Violation(artifact, f"invalid UTF-8: {exc}")]

    records: list[dict[str, Any]] = []
    violations: list[Violation] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        location = f"{artifact} line {line_number}"
        if not line.strip():
            violations.append(Violation(location, "blank JSONL line"))
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            violations.append(Violation(location, f"malformed JSON: {exc.msg}"))
            continue
        if not isinstance(record, dict):
            violations.append(Violation(location, "record must be a JSON object"))
            continue
        records.append(record)
    return records, violations


def _load_manifest_ids(path: Path) -> tuple[frozenset[str], list[Violation]]:
    records, violations = _read_jsonl_objects(path, artifact="manifest")
    ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        value = record.get("openalex_id")
        if not isinstance(value, str):
            violations.append(Violation(f"manifest record {index}", "openalex_id must be a string"))
        else:
            ids.add(value)
    return frozenset(ids), violations


def _load_chunk_ids(path: Path) -> tuple[frozenset[str], list[Violation]]:
    records, violations = _read_jsonl_objects(path, artifact="chunks")
    ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        value = record.get("chunk_id")
        if not isinstance(value, str):
            violations.append(Violation(f"chunks record {index}", "chunk_id must be a string"))
        else:
            ids.add(value)
    return frozenset(ids), violations


def _load_relation_names(path: Path) -> tuple[frozenset[str], list[Violation]]:
    try:
        payload = yaml.safe_load(path.read_bytes())
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        return frozenset(), [Violation("ontology", f"unreadable or malformed: {exc}")]
    if not isinstance(payload, dict) or not isinstance(payload.get("relation_types"), list):
        return frozenset(), [Violation("ontology", "relation_types must be a list")]
    names: set[str] = set()
    violations: list[Violation] = []
    for index, relation in enumerate(payload["relation_types"], start=1):
        if not isinstance(relation, dict) or not isinstance(relation.get("name"), str):
            violations.append(Violation(f"ontology relation_types[{index}]", "name must be a string"))
        else:
            names.add(relation["name"])
    return frozenset(names), violations


def _record_location(line_number: int, record: dict[str, Any]) -> str:
    qid = record.get("qid")
    return f"record {qid!r}" if isinstance(qid, str) else f"line {line_number}"


def _is_endpoint_id(value: object) -> bool:
    return isinstance(value, str) and bool(
        _DOC_ID_RE.fullmatch(value) or _ENTITY_ID_RE.fullmatch(value) or _CHUNK_ID_RE.fullmatch(value)
    )


def _validate_record(
    record: dict[str, Any],
    *,
    line_number: int,
    manifest_ids: frozenset[str],
    chunk_ids: frozenset[str],
    relation_names: frozenset[str],
) -> list[Violation]:
    location = _record_location(line_number, record)
    out: list[Violation] = []
    for field in _REQUIRED_FIELDS:
        if field not in record:
            out.append(Violation(location, f"missing required field {field!r}"))

    qid = record.get("qid")
    if not isinstance(qid, str) or not _QID_RE.fullmatch(qid):
        out.append(Violation(location, "qid must match ^q_[0-9]{3}$"))
    for field in ("question", "expected_answer"):
        if not isinstance(record.get(field), str):
            out.append(Violation(location, f"{field} must be a string"))
    if record.get("hop_bucket") not in {"1", "2", "3+"}:
        out.append(Violation(location, "hop_bucket must be one of '1', '2', '3+'"))
    if record.get("split") != "held_out":
        out.append(Violation(location, "split must be exactly 'held_out'"))
    if "answer_aliases" in record and (
        not isinstance(record["answer_aliases"], list)
        or not all(isinstance(value, str) for value in record["answer_aliases"])
    ):
        out.append(Violation(location, "answer_aliases must be a list of strings"))
    if "notes" in record and not isinstance(record["notes"], str):
        out.append(Violation(location, "notes must be a string"))

    supporting_docs = record.get("supporting_doc_ids")
    if not isinstance(supporting_docs, list) or not supporting_docs:
        out.append(Violation(location, "supporting_doc_ids must be a non-empty list of strings"))
        supporting_docs = []
    for doc_id in supporting_docs:
        if not isinstance(doc_id, str) or not _DOC_ID_RE.fullmatch(doc_id):
            out.append(Violation(location, "supporting_doc_ids values must match ^W[0-9]+$"))
        elif doc_id not in manifest_ids:
            out.append(Violation(location, f"supporting document {doc_id!r} is absent from manifest"))

    supporting_chunks = record.get("supporting_chunk_ids")
    if not isinstance(supporting_chunks, list) or not supporting_chunks:
        out.append(Violation(location, "supporting_chunk_ids must be a non-empty list of strings"))
        supporting_chunks = []
    for chunk_id in supporting_chunks:
        if not isinstance(chunk_id, str) or not _CHUNK_ID_RE.fullmatch(chunk_id):
            out.append(Violation(location, "supporting_chunk_ids values must match ^c_[0-9a-f]{16}$"))
        elif chunk_id not in chunk_ids:
            out.append(Violation(location, f"supporting chunk {chunk_id!r} is absent from chunks artifact"))

    path = record.get("evidence_path")
    if not isinstance(path, list) or not path:
        out.append(Violation(location, "evidence_path must be a non-empty list"))
        return out
    path_doc_ids: set[str] = set()
    previous_to: str | None = None
    previous_step = 0
    for position, entry in enumerate(path, start=1):
        step_location = f"{location} evidence_path[{position}]"
        if not isinstance(entry, dict):
            out.append(Violation(step_location, "entry must be an object"))
            continue
        step = entry.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            out.append(Violation(step_location, "step must be an int, not bool"))
        elif (position == 1 and step != 1) or (position > 1 and step <= previous_step):
            out.append(Violation(step_location, "steps must start at 1 and strictly increase"))
        else:
            previous_step = step
        for endpoint in ("from", "to"):
            value = entry.get(endpoint)
            if not _is_endpoint_id(value):
                out.append(Violation(step_location, f"{endpoint} must be a canonical W/e_/c_ identifier"))
            elif _DOC_ID_RE.fullmatch(value):
                path_doc_ids.add(value)
                if value not in manifest_ids:
                    out.append(Violation(step_location, f"document endpoint {value!r} is absent from manifest"))
        relation = entry.get("relation")
        if not isinstance(relation, str) or not relation:
            out.append(Violation(step_location, "relation must be a non-empty string"))
        elif relation not in relation_names:
            out.append(Violation(step_location, f"relation {relation!r} is absent from ontology"))
        if entry.get("direction") not in {"forward", "reverse"}:
            out.append(Violation(step_location, "direction must be 'forward' or 'reverse'"))
        current_from = entry.get("from")
        if position > 1 and current_from != previous_to:
            out.append(Violation(step_location, "path is disconnected from the previous step"))
        current_to = entry.get("to")
        previous_to = current_to if isinstance(current_to, str) else None

    if not path_doc_ids.issubset(set(supporting_docs)):
        out.append(Violation(location, "supporting_doc_ids must contain every W ID in evidence_path"))
    length = len(path)
    bucket = record.get("hop_bucket")
    if (bucket == "1" and length != 1) or (bucket == "2" and length != 2) or (bucket == "3+" and length < 3):
        out.append(Violation(location, "evidence_path length does not match hop_bucket"))
    return out




def validate_graph_paths(records: list[dict[str, Any]], graph_reader: GraphReader) -> tuple[Violation, ...]:
    """Validate canonical Document/Entity nodes and exact stored path edges."""
    violations: list[Violation] = []
    for line_number, record in enumerate(records, start=1):
        location = _record_location(line_number, record)
        path = record.get("evidence_path")
        if not isinstance(path, list):
            continue
        for position, entry in enumerate(path, start=1):
            if not isinstance(entry, dict):
                continue
            step_location = f"{location} evidence_path[{position}]"
            endpoints = (entry.get("from"), entry.get("to"))
            for node_id in endpoints:
                if not isinstance(node_id, str) or not (_DOC_ID_RE.fullmatch(node_id) or _ENTITY_ID_RE.fullmatch(node_id)):
                    continue
                try:
                    exists = graph_reader.node_exists(node_id)
                except Exception as exc:
                    violations.append(Violation(step_location, f"graph read failed while checking node {node_id!r}: {exc}"))
                    continue
                if not exists:
                    kind = "Document" if _DOC_ID_RE.fullmatch(node_id) else "Entity"
                    violations.append(Violation(step_location, f"graph {kind} node {node_id!r} is missing"))

            source_id, relation, target_id, direction = (
                entry.get("from"), entry.get("relation"), entry.get("to"), entry.get("direction")
            )
            if not all(isinstance(value, str) for value in (source_id, relation, target_id)) or direction not in {"forward", "reverse"}:
                continue
            if direction == "reverse":
                source_id, target_id = target_id, source_id
            try:
                exists = graph_reader.stored_edge_exists(source_id, relation, target_id)
            except Exception as exc:
                violations.append(Violation(step_location, f"graph read failed while checking stored edge {source_id!r}-[{relation}]->{target_id!r}: {exc}"))
                continue
            if not exists:
                violations.append(Violation(step_location, f"required stored edge {source_id!r}-[{relation}]->{target_id!r} is missing"))
    return tuple(violations)


def validate_draft_with_graph(
    draft_path: Path,
    graph_reader: GraphReader,
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
    require_dataset: bool = False,
) -> ValidationReport:
    """Compose Phase-1 draft validation with Phase-2 graph-path validation."""
    report = validate_draft(
        draft_path,
        manifest_path=manifest_path,
        chunks_path=chunks_path,
        ontology_path=ontology_path,
        require_dataset=require_dataset,
    )
    if not report.ok:
        return report
    records, parse_violations = _read_jsonl_objects(draft_path, artifact="draft")
    violations = [*report.violations, *parse_violations]
    if not parse_violations:
        violations.extend(validate_graph_paths(records, graph_reader))
    return ValidationReport(tuple(violations), report.record_count, report.bucket_counts)


def validate_draft(
    draft_path: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
    require_dataset: bool = False,
) -> ValidationReport:
    """Validate a draft JSONL file without writing or querying a graph.

    ``require_dataset`` additionally pins the canonical 45-record balance.
    """
    records, violations = _read_jsonl_objects(draft_path, artifact="draft")
    manifest_ids, manifest_violations = _load_manifest_ids(manifest_path)
    chunk_ids, chunk_violations = _load_chunk_ids(chunks_path)
    relation_names, ontology_violations = _load_relation_names(ontology_path)
    violations.extend(manifest_violations)
    violations.extend(chunk_violations)
    violations.extend(ontology_violations)

    seen_qids: set[str] = set()
    buckets: Counter[str] = Counter()
    for line_number, record in enumerate(records, start=1):
        violations.extend(_validate_record(
            record,
            line_number=line_number,
            manifest_ids=manifest_ids,
            chunk_ids=chunk_ids,
            relation_names=relation_names,
        ))
        qid = record.get("qid")
        if isinstance(qid, str):
            if qid in seen_qids:
                violations.append(Violation(f"record {qid!r}", "duplicate qid"))
            seen_qids.add(qid)
        bucket = record.get("hop_bucket")
        if isinstance(bucket, str):
            buckets[bucket] += 1

    if require_dataset:
        if len(records) != 45:
            violations.append(Violation("dataset", f"must contain exactly 45 records, found {len(records)}"))
        for bucket in ("1", "2", "3+"):
            if not 12 <= buckets[bucket] <= 18:
                violations.append(Violation("dataset", f"hop bucket {bucket!r} must contain 12–18 records, found {buckets[bucket]}"))
    return ValidationReport(tuple(violations), len(records), tuple((bucket, buckets[bucket]) for bucket in ("1", "2", "3+")))


class PublicationError(RuntimeError):
    """A published evaluation file cannot be safely completed."""


def _atomic_write(path: Path, data: bytes) -> None:
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
            temporary_path = temporary.name
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def _sidecar_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.sha256")


def _publish(output_path: Path, candidate: bytes) -> str:
    """Publish exact candidate bytes once, refusing any later replacement."""
    candidate_sha256 = hashlib.sha256(candidate).hexdigest()
    sidecar_path = _sidecar_path(output_path)
    if output_path.exists():
        try:
            published = output_path.read_bytes()
        except OSError as exc:
            raise PublicationError(f"cannot read published evaluation file {output_path}: {exc}") from exc
        if published != candidate:
            raise PublicationError(f"published evaluation file is frozen; different bytes are refused: {output_path}")
        sha256 = hashlib.sha256(published).hexdigest()
        sidecar = f"{sha256}\n".encode("ascii")
        try:
            if not sidecar_path.exists() or sidecar_path.read_bytes() != sidecar:
                _atomic_write(sidecar_path, sidecar)
        except OSError as exc:
            raise PublicationError(f"cannot record SHA256 sidecar for {output_path}: {exc}") from exc
        return sha256

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(output_path, candidate)
        _atomic_write(sidecar_path, f"{candidate_sha256}\n".encode("ascii"))
    except OSError as exc:
        for path in (sidecar_path, output_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise PublicationError(f"cannot publish evaluation file {output_path}: {exc}") from exc
    return candidate_sha256


def _close_graph_reader(graph_reader: GraphReader) -> None:
    close = getattr(graph_reader, "close", None)
    if callable(close):
        close()


def _output_path_is_supported(output_path: Path) -> bool:
    return bool(re.fullmatch(r"questions(?:\.v[1-9][0-9]*)?\.jsonl", output_path.name))


def main(
    argv: list[str] | None = None,
    *,
    graph_reader_factory: Callable[[], GraphReader] | None = None,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
) -> int:
    """Validate and publish a frozen strict evaluation-set file."""
    parser = argparse.ArgumentParser(description="Validate and publish a strict frozen evaluation set.")
    parser.add_argument("--input", required=True, type=Path, metavar="DRAFT")
    parser.add_argument("--output", required=True, type=Path, metavar="OUTPUT")
    arguments = parser.parse_args(argv)
    if not _output_path_is_supported(arguments.output):
        parser.error("--output must name questions.jsonl or questions.vN.jsonl")

    try:
        candidate = arguments.input.read_bytes()
    except OSError as exc:
        print(f"cannot read draft {arguments.input}: {exc}", file=sys.stderr)
        return 1
    owns_graph_reader = graph_reader_factory is None
    factory = default_graph_reader if owns_graph_reader else graph_reader_factory
    try:
        graph_reader = factory()
    except Exception as exc:
        print(f"graph reader setup failed: {exc}", file=sys.stderr)
        return 1
    shutdown_error: Exception | None = None
    try:
        report = validate_draft_with_graph(
            arguments.input,
            graph_reader,
            manifest_path=manifest_path,
            chunks_path=chunks_path,
            ontology_path=ontology_path,
            require_dataset=True,
        )
    finally:
        if owns_graph_reader:
            try:
                _close_graph_reader(graph_reader)
            except Exception as exc:
                shutdown_error = exc
    if shutdown_error is not None:
        print(f"graph reader shutdown failed: {shutdown_error}", file=sys.stderr)
        return 1
    if not report.ok:
        for violation in report.violations:
            print(violation, file=sys.stderr)
        return 1
    try:
        if arguments.input.read_bytes() != candidate:
            raise PublicationError(f"draft changed during validation: {arguments.input}")
        sha256 = _publish(arguments.output, candidate)
    except (OSError, PublicationError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"published {arguments.output}\nsha256 {sha256}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULT_CHUNKS_PATH", "DEFAULT_MANIFEST_PATH", "DEFAULT_ONTOLOGY_PATH", "GraphReader", "ValidationReport", "Violation", "default_graph_reader", "main", "validate_draft", "validate_draft_with_graph", "validate_graph_paths"]
