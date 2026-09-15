"""Synthetic canonical-input fixtures for ``tests/test_build_graph.py``.

Builds a small but well-formed on-disk fixture set covering every
preflight rule in ``docs/specs/neo4j-ingestion.md`` §2:

- 3 documents with matching raw OpenAlex ``referenced_works`` records
- 6 chunks across 3 docs
- 4 entities (one per extracted ontology type)
- 4 mentions + 4 assertions resolving to those entities
- A 4th document used to test external-citation filtering

All chunk_id / mention_id / assertion_id values are content-derived
under the chunk-record §3 / #23 contracts, so the preflight's
re-derivation checks pass and the canonical graph hash is stable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src.build_graph.schema import derive_chunk_id
from src.extract.ids import assertion_id as _assertion_id
from src.extract.ids import mention_id as _mention_id
from src.graph.entity_resolution import entity_id as _entity_id


# Fixed doc IDs.
DOC_A = "W100000001"
DOC_B = "W100000002"
DOC_C = "W100000003"
DOC_D = "W100000004"  # external — used to verify in-corpus filter

# Derived chunk IDs under the chunk-record §3 contract.
CHUNK_A1_ID = derive_chunk_id(doc_id=DOC_A, section_path=["A"], chunk_index=0)
CHUNK_A2_ID = derive_chunk_id(doc_id=DOC_A, section_path=["A", "a"], chunk_index=1)
CHUNK_B1_ID = derive_chunk_id(doc_id=DOC_B, section_path=["B"], chunk_index=0)
CHUNK_B2_ID = derive_chunk_id(doc_id=DOC_B, section_path=["B", "b"], chunk_index=1)
CHUNK_C1_ID = derive_chunk_id(doc_id=DOC_C, section_path=["C"], chunk_index=0)
CHUNK_C2_ID = derive_chunk_id(doc_id=DOC_C, section_path=["C", "c"], chunk_index=1)

METHOD_EID = _entity_id(canonical_name="Foo Method", entity_type="method")
DATASET_EID = _entity_id(canonical_name="Bar Dataset", entity_type="dataset")
METRIC_EID = _entity_id(canonical_name="Baz Metric", entity_type="metric")
TASK_EID = _entity_id(canonical_name="Qux Task", entity_type="task")
METHOD_B_EID = _entity_id(canonical_name="Baz Method", entity_type="method")
DATASET_C_EID = _entity_id(canonical_name="Qux Dataset", entity_type="dataset")


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


def _sha256_hex(payload: str | bytes) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _chunk_text() -> str:
    return "Lorem ipsum dolor sit amet " * 32


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


def _build_chunk_rows() -> list[dict]:
    text = _chunk_text()
    out = []
    for doc_id, section_path, chunk_index, derived_id in (
        (DOC_A, ["A"], 0, CHUNK_A1_ID),
        (DOC_A, ["A", "a"], 1, CHUNK_A2_ID),
        (DOC_B, ["B"], 0, CHUNK_B1_ID),
        (DOC_B, ["B", "b"], 1, CHUNK_B2_ID),
        (DOC_C, ["C"], 0, CHUNK_C1_ID),
        (DOC_C, ["C", "c"], 1, CHUNK_C2_ID),
    ):
        out.append({
            "chunk_id": derived_id,
            "doc_id": doc_id,
            "section_path": list(section_path),
            "chunk_index": chunk_index,
            "text": text,
            "token_estimate": len(text) // 4,
            "char_start": 0,
            "char_end": len(text),
        })
    return out


def _build_openalex_rows() -> dict[str, dict]:
    """OpenAlex raw records.

    DOC_A cites DOC_B (in-corpus) and itself (self-loop removed).
    DOC_B cites DOC_A (in-corpus) and DOC_D (external, filtered).
    DOC_C cites DOC_A.
    """
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


def _build_entity_rows(
    mid_method: str,
    mid_dataset: str,
    mid_method_b: str,
    mid_metric: str,
    mid_dataset_c: str,
    mid_task: str,
) -> list[dict]:
    """Producer-realistic #8 entity records, including the canonical
    mention membership list required by entity-resolution.md §7.1."""
    return [
        {"entity_id": METHOD_EID, "entity_type": "method", "canonical_name": "Foo Method", "mention_ids": [mid_method], "aliases": ["Foo Method", "FM"], "resolution_confidence": 1.0},
        {"entity_id": DATASET_EID, "entity_type": "dataset", "canonical_name": "Bar Dataset", "mention_ids": [mid_dataset], "aliases": ["Bar Dataset"], "resolution_confidence": 1.0},
        {"entity_id": METRIC_EID, "entity_type": "metric", "canonical_name": "Baz Metric", "mention_ids": [mid_metric], "aliases": ["Baz Metric"], "resolution_confidence": 1.0},
        {"entity_id": TASK_EID, "entity_type": "task", "canonical_name": "Qux Task", "mention_ids": [mid_task], "aliases": ["Qux Task"], "resolution_confidence": 1.0},
        {"entity_id": METHOD_B_EID, "entity_type": "method", "canonical_name": "Baz Method", "mention_ids": [mid_method_b], "aliases": ["Baz Method"], "resolution_confidence": 0.95},
        {"entity_id": DATASET_C_EID, "entity_type": "dataset", "canonical_name": "Qux Dataset", "mention_ids": [mid_dataset_c], "aliases": ["Qux Dataset"], "resolution_confidence": 1.0},
    ]


def _build_resolution_rows(mid_method, mid_dataset, mid_method_b, mid_metric, mid_dataset_c, mid_task) -> list[dict]:
    return [
        {"mention_id": mid_method, "entity_id": METHOD_EID, "resolution_method": "exact_normalized", "resolution_score": 1.0},
        {"mention_id": mid_dataset, "entity_id": DATASET_EID, "resolution_method": "exact_normalized", "resolution_score": 1.0},
        {"mention_id": mid_metric, "entity_id": METRIC_EID, "resolution_method": "exact_normalized", "resolution_score": 1.0},
        {"mention_id": mid_task, "entity_id": TASK_EID, "resolution_method": "exact_normalized", "resolution_score": 1.0},
        {"mention_id": mid_method_b, "entity_id": METHOD_B_EID, "resolution_method": "fuzzy", "resolution_score": 0.95},
        {"mention_id": mid_dataset_c, "entity_id": DATASET_C_EID, "resolution_method": "exact_normalized", "resolution_score": 1.0},
    ]


def _build_extraction_rows() -> list[dict]:
    """Returns the sorted-by-(chunk_id, kind, id) JSONL rows.

    Layout:

    - DOC_A1 chunk: PROPOSES doc->method + EVALUATES_ON doc->dataset
      (witnesses the EVALUATED_ON closure).
    - DOC_B1 chunk: USES_METHOD doc->metric
    - DOC_B1 chunk: USES_METHOD doc->method
    - DOC_C1 chunk: USED_FOR_TASK doc->task
    """
    text = _chunk_text()
    sha = _sha256_hex(text)
    mid_method = _mention_id(
        chunk_id=CHUNK_A1_ID, source_text_sha256=sha, entity_type="method",
        char_start=0, char_end=11,
    )
    mid_dataset = _mention_id(
        chunk_id=CHUNK_A1_ID, source_text_sha256=sha, entity_type="dataset",
        char_start=0, char_end=11,
    )
    mid_method_b = _mention_id(
        chunk_id=CHUNK_B1_ID, source_text_sha256=sha, entity_type="method",
        char_start=0, char_end=11,
    )
    mid_metric = _mention_id(
        chunk_id=CHUNK_B1_ID, source_text_sha256=sha, entity_type="metric",
        char_start=0, char_end=11,
    )
    mid_task = _mention_id(
        chunk_id=CHUNK_C1_ID, source_text_sha256=sha, entity_type="task",
        char_start=0, char_end=11,
    )
    aid_proposes = _assertion_id(
        chunk_id=CHUNK_A1_ID, source_text_sha256=sha, relation_type="PROPOSES",
        subject_ref={"kind": "document", "id": DOC_A},
        object_ref={"kind": "mention", "id": mid_method},
        char_start=0, char_end=11,
    )
    aid_evaluates = _assertion_id(
        chunk_id=CHUNK_A1_ID, source_text_sha256=sha, relation_type="EVALUATES_ON",
        subject_ref={"kind": "document", "id": DOC_A},
        object_ref={"kind": "mention", "id": mid_dataset},
        char_start=0, char_end=11,
    )
    aid_uses_method = _assertion_id(
        chunk_id=CHUNK_B1_ID, source_text_sha256=sha, relation_type="USES_METHOD",
        subject_ref={"kind": "document", "id": DOC_B},
        object_ref={"kind": "mention", "id": mid_method_b},
        char_start=0, char_end=11,
    )
    mid_dataset_c = _mention_id(
        chunk_id=CHUNK_C1_ID, source_text_sha256=sha, entity_type="dataset",
        char_start=0, char_end=11,
    )
    aid_used_for_task = _assertion_id(
        chunk_id=CHUNK_C1_ID, source_text_sha256=sha, relation_type="USED_FOR_TASK",
        subject_ref={"kind": "mention", "id": mid_dataset_c},
        object_ref={"kind": "mention", "id": mid_task},
        char_start=0, char_end=11,
    )

    rows: list[dict] = []
    for mid, doc_id, entity_type, surface, canonical in (
        (mid_method, DOC_A, "method", "Lorem ipsum", "Foo Method"),
        (mid_dataset, DOC_A, "dataset", "Lorem ipsum", "Bar Dataset"),
        (mid_method_b, DOC_B, "method", "Lorem ipsum", "Baz Method"),
        (mid_metric, DOC_B, "metric", "Lorem ipsum", "Baz Metric"),
        (mid_dataset_c, DOC_C, "dataset", "Lorem ipsum", "Qux Dataset"),
        (mid_task, DOC_C, "task", "Lorem ipsum", "Qux Task"),
    ):
        chunk_id = {
            mid_method: CHUNK_A1_ID,
            mid_dataset: CHUNK_A1_ID,
            mid_method_b: CHUNK_B1_ID,
            mid_metric: CHUNK_B1_ID,
            mid_dataset_c: CHUNK_C1_ID,
            mid_task: CHUNK_C1_ID,
        }[mid]
        rows.append({
            "kind": "mention",
            "mention_id": mid,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "entity_type": entity_type,
            "surface_form": surface,
            "canonical_form": canonical,
            "evidence": {"char_start": 0, "char_end": 11, "source_text_sha256": sha},
            "confidence": "stated",
        })
    for aid, doc_id, relation_type, subject_kind, subject_id, object_kind, object_id, chunk_id in (
        (aid_proposes, DOC_A, "PROPOSES", "document", DOC_A, "mention", mid_method, CHUNK_A1_ID),
        (aid_evaluates, DOC_A, "EVALUATES_ON", "document", DOC_A, "mention", mid_dataset, CHUNK_A1_ID),
        (aid_uses_method, DOC_B, "USES_METHOD", "document", DOC_B, "mention", mid_method_b, CHUNK_B1_ID),
        (aid_used_for_task, DOC_C, "USED_FOR_TASK", "mention", mid_dataset_c, "mention", mid_task, CHUNK_C1_ID),
    ):
        rows.append({
            "kind": "assertion",
            "assertion_id": aid,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "relation_type": relation_type,
            "subject_ref": {"kind": subject_kind, "id": subject_id},
            "object_ref": {"kind": object_kind, "id": object_id},
            "evidence": {"char_start": 0, "char_end": 11, "source_text_sha256": sha},
            "confidence": "stated",
        })
    # chunk_status for every chunk — extraction_complete=true.
    for cid, doc_id in (
        (CHUNK_A1_ID, DOC_A),
        (CHUNK_A2_ID, DOC_A),
        (CHUNK_B1_ID, DOC_B),
        (CHUNK_B2_ID, DOC_B),
        (CHUNK_C1_ID, DOC_C),
        (CHUNK_C2_ID, DOC_C),
    ):
        mention_count = sum(
            1 for r in rows if r.get("kind") == "mention" and r.get("chunk_id") == cid
        )
        assertion_count = sum(
            1 for r in rows if r.get("kind") == "assertion" and r.get("chunk_id") == cid
        )
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
    rows.sort(key=lambda r: (r.get("chunk_id", ""), r.get("kind", ""),
                             r.get("mention_id", r.get("assertion_id", ""))))
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

    # Write ontology first so we can capture its exact bytes for SHAs.
    ontology = _build_ontology()
    paths.ontology.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    ontology_bytes = yaml.safe_dump(ontology, sort_keys=True, allow_unicode=True).encode("utf-8")
    paths.ontology.write_bytes(ontology_bytes)
    ontology_sha = _sha256_hex(ontology_bytes)

    # Write chunks next; SHAs depend on chunks bytes.
    chunk_rows = _build_chunk_rows()
    _write_jsonl(paths.chunks, chunk_rows)
    chunks_bytes = paths.chunks.read_bytes()
    chunks_sha = _sha256_hex(chunks_bytes)

    # Compute content-derived mention IDs; both producer entity
    # membership and resolution rows depend on the same IDs.
    text = _chunk_text()
    sha = _sha256_hex(text)
    from src.extract.ids import mention_id as _mid
    mid_method = _mid(chunk_id=CHUNK_A1_ID, source_text_sha256=sha,
                      entity_type="method", char_start=0, char_end=11)
    mid_dataset = _mid(chunk_id=CHUNK_A1_ID, source_text_sha256=sha,
                       entity_type="dataset", char_start=0, char_end=11)
    mid_metric = _mid(chunk_id=CHUNK_B1_ID, source_text_sha256=sha,
                      entity_type="metric", char_start=0, char_end=11)
    mid_task = _mid(chunk_id=CHUNK_C1_ID, source_text_sha256=sha,
                    entity_type="task", char_start=0, char_end=11)
    mid_method_b = _mid(chunk_id=CHUNK_B1_ID, source_text_sha256=sha,
                        entity_type="method", char_start=0, char_end=11)
    mid_dataset_c = _mid(chunk_id=CHUNK_C1_ID, source_text_sha256=sha,
                         entity_type="dataset", char_start=0, char_end=11)

    _write_jsonl(paths.entities, _build_entity_rows(
        mid_method, mid_dataset, mid_method_b, mid_metric, mid_dataset_c, mid_task
    ))
    entities_sha = _sha256_hex(paths.entities.read_bytes())

    _write_jsonl(paths.entity_resolutions, _build_resolution_rows(
        mid_method, mid_dataset, mid_method_b, mid_metric, mid_dataset_c, mid_task
    ))
    entity_resolutions_sha = _sha256_hex(paths.entity_resolutions.read_bytes())

    # Write extractions + manifest.
    extraction_rows = _build_extraction_rows()
    _write_jsonl(paths.extractions, extraction_rows)
    extractions_sha = _sha256_hex(paths.extractions.read_bytes())
    extractions_manifest_payload = _build_extractions_manifest(
        ontology_sha=ontology_sha,
        chunks_sha=chunks_sha,
        extractions_sha=extractions_sha,
    )
    _write_json(paths.extractions_manifest, extractions_manifest_payload)
    extractions_manifest_sha = _sha256_hex(paths.extractions_manifest.read_bytes())

    # OpenAlex raw records (bytes determine manifest sha256).
    paths.openalex_dir.mkdir(parents=True, exist_ok=True)
    openalex_payloads = _build_openalex_rows()
    openalex_bytes_by_doc: dict = {}
    for doc_id, record in openalex_payloads.items():
        body = json.dumps(record, sort_keys=True, ensure_ascii=False)
        openalex_bytes_by_doc[doc_id] = body.encode("utf-8")
        (paths.openalex_dir / f"{doc_id}.json").write_bytes(
            openalex_bytes_by_doc[doc_id]
        )

    # Manifest with sha256 = sha256(raw bytes).
    manifest_rows = []
    for doc_id in (DOC_A, DOC_B, DOC_C):
        raw_bytes = openalex_bytes_by_doc[doc_id]
        manifest_rows.append({
            "openalex_id": doc_id,
            "title": f"Doc {doc_id}",
            "sha256": _sha256_hex(raw_bytes),
        })
    _write_jsonl(paths.manifest, manifest_rows)
    manifest_sha = _sha256_hex(paths.manifest.read_bytes())

    # Entity-resolution manifest with matching SHAs.
    er_payload = {
        "schema_version": 1,
        "algorithm": {
            "algorithm_version": "entity_resolution_v1",
            "normalization_version": "resolution_name_v1",
            "similarity_metric": "normalized_levenshtein",
            "threshold": 0.92,
        },
        "counts": {
            "eligible_mentions": 6,
            "entities": 6,
            "extraction_complete_chunks": 6,
            "identical_duplicate_mentions": 0,
        },
        "inputs": {
            "ontology_path": "ontology.yaml",
            "ontology_sha256": ontology_sha,
            "ontology_version": 1,
            "chunks_path": "data/derived/chunks.jsonl",
            "chunks_sha256": chunks_sha,
            "extractions_path": "data/extractions.jsonl",
            "extractions_sha256": extractions_sha,
            "extractions_manifest_path": "data/extractions.manifest.json",
            "extractions_manifest_sha256": extractions_manifest_sha,
        },
        "outputs": {
            "entities_path": "data/entities.jsonl",
            "entities_records": 6,
            "entities_sha256": entities_sha,
            "entity_resolutions_path": "data/entity_resolutions.jsonl",
            "entity_resolutions_records": 6,
            "entity_resolutions_sha256": entity_resolutions_sha,
        },
    }
    _write_json(paths.entity_resolution_manifest, er_payload)

    # aliases (empty — every alias key is canonical already).
    paths.aliases.parent.mkdir(parents=True, exist_ok=True)
    paths.aliases.write_text("{}", encoding="utf-8")

    return paths


def _build_extractions_manifest(
    *, ontology_sha: str, chunks_sha: str, extractions_sha: str
) -> dict:
    del extractions_sha  # #23's producer manifest does not store its output SHA.
    return {
        "schema_version": 1,
        "provider": "test",
        "model": "test-model",
        "ontology_version": 1,
        "ontology_sha256": ontology_sha,
        "chunks_path": "data/derived/chunks.jsonl",
        "chunks_sha256": chunks_sha,
        "run_id": "test-run",
        "resumed_from_run_id": None,
        "cache_format_version": "1",
        "prompt_template_files": [
            "extract_entities_v1.txt",
            "extract_relations_v1.txt",
        ],
        "prompt_template_sha256": ["a" * 64, "b" * 64],
        "cumulative_cost_usd": 0.0,
        "cumulative_tokens": 0,
        "max_tokens": 2048,
        "temperature": 0.0,
        "wall_clock_started_at": "2026-01-01T00:00:00+00:00",
        "wall_clock_ended_at": "2026-01-01T00:00:01+00:00",
    }


def patch_extractions_manifest(
    path: Path, *, chunks_path: Path | None = None, ontology_path: Path | None = None
) -> None:
    """Refresh the extractions manifest's chunk + ontology SHA after a
    fixture mutation.

    ``chunks_path`` / ``ontology_path`` default to the sibling fixture
    layout rooted at the manifest's own parent, so callers do not have
    to reason about the process CWD.
    """
    payload = json.loads(path.read_text())
    data_dir = path.parent
    root = data_dir.parent
    if chunks_path is None:
        chunks_path = data_dir / "derived" / "chunks.jsonl"
    if ontology_path is None:
        ontology_path = root / "ontology.yaml"
    payload["chunks_sha256"] = _sha256_hex(chunks_path.read_bytes())
    payload["ontology_sha256"] = _sha256_hex(ontology_path.read_bytes())
    _write_json(path, payload)
