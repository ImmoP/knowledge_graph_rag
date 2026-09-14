"""Synthetic canonical-input fixtures for ``tests/test_build_graph.py``.

Builds a small but well-formed on-disk fixture set covering every
preflight rule in ``docs/specs/neo4j-ingestion.md`` §2:

- 3 documents with matching raw OpenAlex ``referenced_works`` records
- 6 chunks across 3 docs
- 4 entities (one per extracted ontology type)
- 4 mentions + 4 assertions resolving to those entities
- A 4th document used to test external-citation filtering

All SHAs and IDs are deterministic. Callers may mutate the on-disk
artifacts after :func:`materialize` returns to simulate drift.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


# Fixed IDs so test references are stable across runs.
DOC_A = "W100000001"
DOC_B = "W100000002"
DOC_C = "W100000003"
DOC_D = "W100000004"  # external — used to verify in-corpus filter

CHUNK_A1 = "c_000000000000a001"
CHUNK_A2 = "c_000000000000a002"
CHUNK_B1 = "c_000000000000b001"
CHUNK_B2 = "c_000000000000b002"
CHUNK_C1 = "c_000000000000c001"
CHUNK_C2 = "c_000000000000c002"

METHOD_EID = "e_0000000000000a01"
DATASET_EID = "e_0000000000000a02"
METRIC_EID = "e_0000000000000a03"
TASK_EID = "e_0000000000000a04"

MENTION_METHOD = "m_" + "a" * 16
MENTION_DATASET = "m_" + "b" * 16
MENTION_METRIC = "m_" + "c" * 16
MENTION_TASK = "m_" + "d" * 16

ASSERTION_PROPOSES = "a_" + "0" * 16
ASSERTION_EVALUATES = "a_" + "1" * 16
ASSERTION_USES_METHOD = "a_" + "2" * 16
ASSERTION_USED_FOR_TASK = "a_" + "3" * 16


def _chunk_text() -> str:
    return "Lorem ipsum dolor sit amet " * 32


def _sha256_hex(payload: str | bytes) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FixturePaths:
    manifest: Path
    chunks: Path
    ontology: Path
    extractions: Path
    extractions_manifest: Path
    entities: Path
    entity_resolutions: Path
    entity_resolution_manifest: Path
    aliases: Path
    openalex_dir: Path
    graph_manifest: Path

def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _build_ontology() -> dict:
    return {
        "version": 1,
        "schema_version": 1,
        "entity_types": [
            {"name": "document", "population_mode": "structural", "definition": "", "examples": []},
            {"name": "chunk", "population_mode": "structural", "definition": "", "examples": []},
            {"name": "method", "population_mode": "extracted", "extraction_prompt": "extract_entities_v1", "definition": "", "examples": []},
            {"name": "dataset", "population_mode": "extracted", "extraction_prompt": "extract_entities_v1", "definition": "", "examples": []},
            {"name": "metric", "population_mode": "extracted", "extraction_prompt": "extract_entities_v1", "definition": "", "examples": []},
            {"name": "task", "population_mode": "extracted", "extraction_prompt": "extract_entities_v1", "definition": "", "examples": []},
        ],
        "relation_types": [
            {"name": "CITES", "from": "document", "to": "document", "population_mode": "structural", "definition": "", "examples": []},
            {"name": "PROPOSES", "from": "document", "to": "method", "population_mode": "extracted", "extraction_prompt": "extract_relations_v1", "definition": "", "examples": []},
            {"name": "INTRODUCES", "from": "document", "to": "dataset", "population_mode": "extracted", "extraction_prompt": "extract_relations_v1", "definition": "", "examples": []},
            {"name": "EVALUATES_ON", "from": "document", "to": "dataset", "population_mode": "extracted", "extraction_prompt": "extract_relations_v1", "definition": "", "examples": []},
            {"name": "USES_METHOD", "from": "document", "to": "method", "population_mode": "extracted", "extraction_prompt": "extract_relations_v1", "definition": "", "examples": []},
            {"name": "REPORTS_METRIC", "from": "document", "to": "metric", "population_mode": "extracted", "extraction_prompt": "extract_relations_v1", "definition": "", "examples": []},
            {"name": "EXTENDS", "from": "method", "to": "method", "population_mode": "extracted", "extraction_prompt": "extract_relations_v1", "definition": "", "examples": []},
            {"name": "EVALUATED_ON", "from": "method", "to": "dataset", "population_mode": "derived", "definition": "", "examples": []},
            {"name": "USED_FOR_TASK", "from": "dataset", "to": "task", "population_mode": "extracted", "extraction_prompt": "extract_relations_v1", "definition": "", "examples": []},
            {"name": "MEASURES", "from": "metric", "to": "task", "population_mode": "extracted", "extraction_prompt": "extract_relations_v1", "definition": "", "examples": []},
        ],
    }


def _build_manifest_rows() -> list[dict]:
    return [
        {"openalex_id": DOC_A, "title": "Doc A", "sha256": "a" * 64},
        {"openalex_id": DOC_B, "title": "Doc B", "sha256": "b" * 64},
        {"openalex_id": DOC_C, "title": "Doc C", "sha256": "c" * 64},
    ]


def _build_chunk_rows() -> list[dict]:
    text = _chunk_text()
    return [
        {"chunk_id": CHUNK_A1, "doc_id": DOC_A, "section_path": ["A"], "chunk_index": 0, "text": text},
        {"chunk_id": CHUNK_A2, "doc_id": DOC_A, "section_path": ["A", "a"], "chunk_index": 1, "text": text},
        {"chunk_id": CHUNK_B1, "doc_id": DOC_B, "section_path": ["B"], "chunk_index": 0, "text": text},
        {"chunk_id": CHUNK_B2, "doc_id": DOC_B, "section_path": ["B", "b"], "chunk_index": 1, "text": text},
        {"chunk_id": CHUNK_C1, "doc_id": DOC_C, "section_path": ["C"], "chunk_index": 0, "text": text},
        {"chunk_id": CHUNK_C2, "doc_id": DOC_C, "section_path": ["C", "c"], "chunk_index": 1, "text": text},
    ]


def _build_openalex_rows() -> dict[str, dict]:
    """OpenAlex raw records. DOC_A cites DOC_B (in-corpus) and itself
    (self-loop); DOC_B cites DOC_A (in-corpus) and DOC_D (external,
    must be filtered). DOC_C cites DOC_A."""
    return {
        DOC_A: {
            "id": f"https://openalex.org/{DOC_A}",
            "referenced_works": [
                f"https://openalex.org/{DOC_B}",
                DOC_A,
            ],
        },
        DOC_B: {
            "id": f"https://openalex.org/{DOC_B}",
            "referenced_works": [
                f"https://openalex.org/{DOC_A}",
                f"https://openalex.org/{DOC_D}",
            ],
        },
        DOC_C: {
            "id": f"https://openalex.org/{DOC_C}",
            "referenced_works": [
                f"https://openalex.org/{DOC_A}",
            ],
        },
    }


def _build_entity_rows() -> list[dict]:
    return [
        {"entity_id": METHOD_EID, "entity_type": "method", "canonical_name": "Foo Method", "aliases": ["Foo Method", "FM"], "resolution_confidence": 1.0},
        {"entity_id": DATASET_EID, "entity_type": "dataset", "canonical_name": "Bar Dataset", "aliases": ["Bar Dataset"], "resolution_confidence": 1.0},
        {"entity_id": METRIC_EID, "entity_type": "metric", "canonical_name": "Baz Metric", "aliases": ["Baz Metric"], "resolution_confidence": 1.0},
        {"entity_id": TASK_EID, "entity_type": "task", "canonical_name": "Qux Task", "aliases": ["Qux Task"], "resolution_confidence": 1.0},
    ]


def _build_resolution_rows() -> list[dict]:
    return [
        {"mention_id": MENTION_METHOD, "entity_id": METHOD_EID, "resolution_method": "exact_normalized", "resolution_score": 1.0},
        {"mention_id": MENTION_DATASET, "entity_id": DATASET_EID, "resolution_method": "exact_normalized", "resolution_score": 1.0},
        {"mention_id": MENTION_METRIC, "entity_id": METRIC_EID, "resolution_method": "exact_normalized", "resolution_score": 1.0},
        {"mention_id": MENTION_TASK, "entity_id": TASK_EID, "resolution_method": "exact_normalized", "resolution_score": 1.0},
    ]


def _build_extraction_rows() -> list[dict]:
    """Returns the sorted-by-(chunk_id, kind, id) JSONL rows.

    Layout:

    - DOC_A1 chunk: PROPOSES doc->method + EVALUATES_ON doc->dataset
      (witnesses the EVALUATED_ON closure).
    - DOC_B1 chunk: USES_METHOD doc->metric
    - DOC_C1 chunk: USED_FOR_TASK dataset->task
    """
    text = _chunk_text()
    sha = _sha256_hex(text)
    rows: list[dict] = []
    rows.append({
        "kind": "mention",
        "mention_id": MENTION_METHOD,
        "chunk_id": CHUNK_A1,
        "doc_id": DOC_A,
        "entity_type": "method",
        "surface_form": "Foo Method",
        "canonical_form": "Foo Method",
        "evidence": {"char_start": 0, "char_end": 10, "source_text_sha256": sha},
        "confidence": "stated",
    })
    rows.append({
        "kind": "assertion",
        "assertion_id": ASSERTION_PROPOSES,
        "chunk_id": CHUNK_A1,
        "doc_id": DOC_A,
        "relation_type": "PROPOSES",
        "subject_ref": {"kind": "document", "id": DOC_A},
        "object_ref": {"kind": "mention", "id": MENTION_METHOD},
        "evidence": {"char_start": 0, "char_end": 10, "source_text_sha256": sha},
        "confidence": "stated",
    })
    rows.append({
        "kind": "mention",
        "mention_id": MENTION_DATASET,
        "chunk_id": CHUNK_A1,
        "doc_id": DOC_A,
        "entity_type": "dataset",
        "surface_form": "Bar Dataset",
        "canonical_form": "Bar Dataset",
        "evidence": {"char_start": 0, "char_end": 10, "source_text_sha256": sha},
        "confidence": "stated",
    })
    rows.append({
        "kind": "assertion",
        "assertion_id": ASSERTION_EVALUATES,
        "chunk_id": CHUNK_A1,
        "doc_id": DOC_A,
        "relation_type": "EVALUATES_ON",
        "subject_ref": {"kind": "document", "id": DOC_A},
        "object_ref": {"kind": "mention", "id": MENTION_DATASET},
        "evidence": {"char_start": 0, "char_end": 10, "source_text_sha256": sha},
        "confidence": "stated",
    })
    rows.append({
        "kind": "mention",
        "mention_id": MENTION_METRIC,
        "chunk_id": CHUNK_B1,
        "doc_id": DOC_B,
        "entity_type": "metric",
        "surface_form": "Baz Metric",
        "canonical_form": "Baz Metric",
        "evidence": {"char_start": 0, "char_end": 10, "source_text_sha256": sha},
        "confidence": "stated",
    })
    rows.append({
        "kind": "assertion",
        "assertion_id": ASSERTION_USES_METHOD,
        "chunk_id": CHUNK_B1,
        "doc_id": DOC_B,
        "relation_type": "USES_METHOD",
        "subject_ref": {"kind": "document", "id": DOC_B},
        "object_ref": {"kind": "mention", "id": MENTION_METRIC},
        "evidence": {"char_start": 0, "char_end": 10, "source_text_sha256": sha},
        "confidence": "stated",
    })
    rows.append({
        "kind": "mention",
        "mention_id": MENTION_TASK,
        "chunk_id": CHUNK_C1,
        "doc_id": DOC_C,
        "entity_type": "task",
        "surface_form": "Qux Task",
        "canonical_form": "Qux Task",
        "evidence": {"char_start": 0, "char_end": 10, "source_text_sha256": sha},
        "confidence": "stated",
    })
    rows.append({
        "kind": "assertion",
        "assertion_id": ASSERTION_USED_FOR_TASK,
        "chunk_id": CHUNK_C1,
        "doc_id": DOC_C,
        "relation_type": "USED_FOR_TASK",
        "subject_ref": {"kind": "document", "id": DOC_C},
        "object_ref": {"kind": "mention", "id": MENTION_TASK},
        "evidence": {"char_start": 0, "char_end": 10, "source_text_sha256": sha},
        "confidence": "stated",
    })
    # chunk_status for every chunk — extraction_complete=true.
    for cid, doc_id in (
        (CHUNK_A1, DOC_A),
        (CHUNK_A2, DOC_A),
        (CHUNK_B1, DOC_B),
        (CHUNK_B2, DOC_B),
        (CHUNK_C1, DOC_C),
        (CHUNK_C2, DOC_C),
    ):
        mention_count = sum(1 for r in rows if r.get("kind") == "mention" and r.get("chunk_id") == cid)
        assertion_count = sum(1 for r in rows if r.get("kind") == "assertion" and r.get("chunk_id") == cid)
        rows.append({
            "kind": "chunk_status",
            "chunk_id": cid,
            "doc_id": doc_id,
            "phases": {
                "entity": {"status": "validated", "attempt_count": 1, "mention_count": mention_count, "failure_count": 0},
                "relation": {"status": "validated", "attempt_count": 1, "assertion_count": assertion_count, "failure_count": 0},
            },
            "extraction_complete": True,
        })
    rows.sort(key=lambda r: (r.get("chunk_id", ""), r.get("kind", ""), r.get("mention_id", r.get("assertion_id", ""))))
    return rows


def materialize(root: Path) -> FixturePaths:
    """Build the on-disk fixture set rooted at ``root`` and return the
    paths. Caller may mutate files (or environment) before invoking
    :func:`tests.test_build_graph.run_preflight_against` or similar
    test code.
    """
    paths = FixturePaths(
        manifest=root / "data" / "manifest.jsonl",
        chunks=root / "data" / "derived" / "chunks.jsonl",
        ontology=root / "ontology.yaml",
        extractions=root / "data" / "extractions.jsonl",
        extractions_manifest=root / "data" / "extractions.manifest.json",
        entities=root / "data" / "entities.jsonl",
        entity_resolutions=root / "data" / "entity_resolutions.jsonl",
        entity_resolution_manifest=root / "data" / "entity_resolution.manifest.json",
        aliases=root / "data" / "aliases.json",
        openalex_dir=root / "data" / "raw" / "openalex",
        graph_manifest=root / "data" / "graph_ingestion.manifest.json",
    )

    ontology = _build_ontology()
    paths.ontology.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    paths.ontology.write_text(
        yaml.safe_dump(ontology, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    _write_jsonl(paths.manifest, _build_manifest_rows())
    _write_jsonl(paths.chunks, _build_chunk_rows())
    _write_jsonl(paths.entities, _build_entity_rows())
    _write_jsonl(paths.entity_resolutions, _build_resolution_rows())

    # entity resolution manifest (minimal valid shape)
    _write_json(paths.entity_resolution_manifest, {
        "schema_version": 1,
        "algorithm": {
            "algorithm_version": "entity_resolution_v1",
            "normalization_version": "resolution_name_v1",
            "similarity_metric": "normalized_levenshtein",
            "threshold": 0.92,
        },
        "counts": {
            "eligible_mentions": 4,
            "entities": 4,
            "extraction_complete_chunks": 6,
            "identical_duplicate_mentions": 0,
        },
        "inputs": {
            "ontology_path": "ontology.yaml",
            "ontology_sha256": _sha256_hex(paths.ontology.read_bytes()),
            "ontology_version": 1,
            "chunks_path": "data/derived/chunks.jsonl",
            "chunks_sha256": _sha256_hex(paths.chunks.read_bytes()),
            "extractions_path": "data/extractions.jsonl",
            "extractions_sha256": "placeholder",  # patched after writing extractions
            "extractions_manifest_path": "data/extractions.manifest.json",
            "extractions_manifest_sha256": "placeholder",
        },
        "outputs": {
            "entities_path": "data/entities.jsonl",
            "entities_records": 4,
            "entities_sha256": _sha256_hex(paths.entities.read_bytes()),
            "entity_resolutions_path": "data/entity_resolutions.jsonl",
            "entity_resolutions_records": 4,
            "entity_resolutions_sha256": _sha256_hex(paths.entity_resolutions.read_bytes()),
        },
    })

    # extractions + manifest
    _write_jsonl(paths.extractions, _build_extraction_rows())
    _write_json(paths.extractions_manifest, _build_extractions_manifest(
        ontology_path=paths.ontology,
        chunks_path=paths.chunks,
        extractions_path=paths.extractions,
    ))

    # OpenAlex raw records.
    paths.openalex_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, record in _build_openalex_rows().items():
        (paths.openalex_dir / f"{doc_id}.json").write_text(
            json.dumps(record, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

    # aliases (empty — every alias key is canonical already).
    paths.aliases.parent.mkdir(parents=True, exist_ok=True)
    paths.aliases.write_text("{}", encoding="utf-8")

    return paths


def _build_extractions_manifest(
    *, ontology_path: Path, chunks_path: Path, extractions_path: Path
) -> dict:
    return {
        "schema_version": 1,
        "provider": "test",
        "model": "test-model",
        "ontology_path": str(ontology_path),
        "ontology_version": 1,
        "ontology_sha256": _sha256_hex(ontology_path.read_bytes()),
        "chunks_path": str(chunks_path),
        "chunks_sha256": _sha256_hex(chunks_path.read_bytes()),
        "extractions_path": str(extractions_path),
        "run_id": "test-run",
        "resumed_from_run_id": None,
        "cache_format_version": 1,
        "prompt_template_files": ["extract_entities_v1.txt", "extract_relations_v1.txt"],
        "prompt_template_sha256": "deadbeef",
        "cumulative_cost_usd": None,
        "cumulative_tokens": None,
        "max_tokens": 2048,
        "temperature": 0.0,
        "wall_clock_started_at": "2026-01-01T00:00:00+00:00",
        "wall_clock_ended_at": "2026-01-01T00:00:01+00:00",
    }


def patch_extractions_manifest(path: Path) -> None:
    """Refresh the extractions manifest's chunk SHA after a chunk-file
    mutation. Tests call this after editing ``chunks.jsonl`` so the
    manifest stays consistent with the on-disk bytes.
    """
    payload = json.loads(path.read_text())
    payload["chunks_sha256"] = _sha256_hex(
        Path(payload["chunks_path"]).read_bytes()
    )
    payload["ontology_sha256"] = _sha256_hex(
        Path(payload["ontology_path"]).read_bytes()
    )
    _write_json(path, payload)
