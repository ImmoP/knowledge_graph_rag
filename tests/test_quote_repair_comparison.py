"""Focused tests for the strictly offline quote-repair comparison."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.extract import pipeline_v2, prompts as prompts_module, schema as schema_module
from src.providers.cache import CachePaths, ProviderCache, extraction_cache_scope

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "compare_quote_repair_v1.py"
_SPEC = importlib.util.spec_from_file_location("compare_quote_repair_v1", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
comparison = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = comparison
_SPEC.loader.exec_module(comparison)

_MINIMAL_ONTOLOGY = """\
version: 1
schema_version: 1
entity_types:
  - name: document
    population_mode: structural
    source: manifest
    examples: []
  - name: dataset
    population_mode: extracted
    extraction_prompt: extract_entities_v2
    definition: A benchmark dataset.
    verified_examples: []
relation_types:
  - name: EVALUATES_ON
    from: document
    to: dataset
    population_mode: extracted
    extraction_prompt: extract_relations_v2
    definition: An evaluation relationship.
    verified_examples: []
"""
_ENTITY_TEMPLATE = (
    "v2-entity: types={allowed_entity_types} defs={entity_type_definitions} "
    "ex={verified_examples} text={chunk_text}\n"
)
_RELATION_TEMPLATE = (
    "v2-relation: rels={allowed_relation_types} defs={relation_definitions} "
    "chunk={chunk_id} doc={doc_id} mentions={mention_list} text={chunk_text}\n"
)


def _world(tmp_path: Path, *, text: str, quote: str) -> tuple[comparison.ComparisonConfig, dict]:
    chunk = {
        "chunk_id": "c_test",
        "doc_id": "W1",
        "section_path": ["results"],
        "chunk_index": 0,
        "text": text,
        "token_estimate": 10,
        "char_start": 0,
        "char_end": len(text),
    }
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    ontology_path = tmp_path / "ontology.yaml"
    ontology_path.write_text(_MINIMAL_ONTOLOGY, encoding="utf-8")
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / prompts_module.ENTITY_PROMPT_FILE_V2).write_text(
        _ENTITY_TEMPLATE, encoding="utf-8"
    )
    (prompts_dir / prompts_module.RELATION_PROMPT_FILE_V2).write_text(
        _RELATION_TEMPLATE, encoding="utf-8"
    )
    cache_dir = tmp_path / "cache"
    baseline_jsonl = tmp_path / "baseline.v2.jsonl"
    baseline_rejects = tmp_path / "baseline.v2.rejects.jsonl"
    config = comparison.ComparisonConfig(
        chunks_path=chunks_path,
        cache_dir=cache_dir,
        prompts_dir=prompts_dir,
        ontology_path=ontology_path,
        baseline_jsonl=baseline_jsonl,
        baseline_rejects=baseline_rejects,
        provider_name="offline-test",
        provider_model="fixture-v1",
    )
    entity_payload = {
        "mentions": [{
            "entity_type": "dataset",
            "surface_form": "NQ",
            "canonical_form": "Natural Questions",
            "confidence": "stated",
            "occurrence_index": 0,
        }]
    }
    from src.extract.ids import mention_id_v2
    from src.extract.validation import source_text_sha256

    start = text.index("NQ")
    mention_id = mention_id_v2(
        chunk_id=chunk["chunk_id"],
        source_text_sha256=source_text_sha256(text),
        entity_type="dataset",
        char_start=start,
        char_end=start + 2,
    )
    relation_payload = {
        "assertions": [{
            "relation_type": "EVALUATES_ON",
            "subject_ref": {"kind": "document", "id": "W1"},
            "object_ref": {"kind": "mention", "id": mention_id},
            "evidence_quote": quote,
            "confidence": "stated",
        }]
    }
    return config, {
        "chunk": chunk,
        "entity_payload": entity_payload,
        "relation_payload": relation_payload,
    }


def _populate_cache(config: comparison.ComparisonConfig, fixture: dict) -> None:
    ontology, chunks, artifacts, report = pipeline_v2._preflight_v2(
        chunks_path=config.chunks_path,
        ontology_path=config.ontology_path,
        prompts_dir=config.prompts_dir,
    )
    cache = ProviderCache(
        CachePaths.from_root(config.cache_dir),
        cache_format_version=config.cache_format_version,
    )
    schemas = {
        "entity": schema_module.entity_pass_v2_schema(ontology),
        "relation": schema_module.relation_pass_v2_schema(ontology),
    }
    allowed = schema_module.build_allowed_name_lists(ontology)
    chunk = chunks[0]
    sha = __import__("src.extract.validation", fromlist=["source_text_sha256"]).source_text_sha256(
        chunk["text"]
    )
    entity_artifact = artifacts[pipeline_v2.V2_ENTITY_PROMPT_FILENAME]
    entity_user = prompts_module.render_entity_prompt(
        template=entity_artifact["text"],
        allowed_entity_types=allowed["entity_types"],
        entity_type_definitions=schema_module.verified_examples_block(ontology),
        verified_examples=schema_module.verified_examples_block(ontology),
        chunk_text=chunk["text"],
    )
    entity_scope = {
        "chunk_id": chunk["chunk_id"],
        "source_text_sha256": sha,
        "ontology_version": str(report.ontology_version),
        "ontology_sha256": report.ontology_sha256,
        "prompt_template_filename": entity_artifact["filename"],
        "prompt_template_content_sha256": entity_artifact["sha256"],
    }
    with extraction_cache_scope(**entity_scope):
        cache.put_generate_json(
            provider_name=config.provider_name,
            provider_model=config.provider_model,
            system=pipeline_v2._render_system_v2(),
            user=entity_user,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            schema=schemas["entity"],
            parsed=fixture["entity_payload"],
        )

    relation_artifact = artifacts[pipeline_v2.V2_RELATION_PROMPT_FILENAME]
    start = chunk["text"].index("NQ")
    mention_lines = [{
        "mention_id": fixture["relation_payload"]["assertions"][0]["object_ref"]["id"],
        "entity_type": "dataset",
        "surface_form": "NQ",
        "evidence": {"char_start": start, "char_end": start + 2},
    }]
    relation_user = prompts_module.render_relation_prompt(
        template=relation_artifact["text"],
        allowed_relation_types=allowed["relation_types"],
        relation_definitions=schema_module.relation_definitions_block(ontology),
        chunk_id=chunk["chunk_id"],
        doc_id=chunk["doc_id"],
        mention_list=prompts_module.render_mention_list(mention_lines),
        chunk_text=chunk["text"],
    )
    relation_scope = {
        "chunk_id": chunk["chunk_id"],
        "source_text_sha256": sha,
        "ontology_version": str(report.ontology_version),
        "ontology_sha256": report.ontology_sha256,
        "prompt_template_filename": relation_artifact["filename"],
        "prompt_template_content_sha256": relation_artifact["sha256"],
    }
    with extraction_cache_scope(**relation_scope):
        cache.put_generate_json(
            provider_name=config.provider_name,
            provider_model=config.provider_model,
            system=pipeline_v2._render_system_v2(),
            user=relation_user,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            schema=schemas["relation"],
            parsed=fixture["relation_payload"],
        )


def _write_stored_baseline(config: comparison.ComparisonConfig) -> None:
    chunks = comparison._select_chunks(config.chunks_path, ("c_test",))
    selected = config.chunks_path.parent / "selected.jsonl"
    comparison._write_selected_chunks(selected, chunks)
    records, rejects, _ = comparison._run_arm(
        config=config,
        chunks=chunks,
        chunks_path=selected,
        output_dir=config.chunks_path.parent,
        arm="stored",
        repair_relations=False,
    )
    config.baseline_jsonl.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    config.baseline_rejects.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in rejects),
        encoding="utf-8",
    )


def _ready_world(tmp_path: Path, *, text: str, quote: str) -> comparison.ComparisonConfig:
    config, fixture = _world(tmp_path, text=text, quote=quote)
    _populate_cache(config, fixture)
    _write_stored_baseline(config)
    return config


def test_missing_cache_fails_closed_without_provider_fallback(tmp_path):
    config, _ = _world(tmp_path, text="DPR evaluates NQ", quote="DPR evaluates NQ")
    config.cache_dir.mkdir()
    config.baseline_jsonl.write_text("", encoding="utf-8")
    config.baseline_rejects.write_text("", encoding="utf-8")

    with pytest.raises(comparison.OfflineCacheMiss, match="required cache entry missing"):
        comparison.run_comparison(config, chunk_ids=("c_test",))


def test_ambiguous_reconstruction_is_refused(tmp_path):
    config = _ready_world(
        tmp_path,
        text="DPR\n evaluates NQ. DPR\n evaluates NQ.",
        quote="DPR evaluates NQ",
    )

    report = comparison.run_comparison(config, chunk_ids=("c_test",))

    decision = report["chunks"][0]["reconstruction_decisions"][0]
    assert decision["status"] == "ambiguous"
    assert decision["replacement_quote"] is None
    assert report["delta"]["accepted_assertion_count"] == 0


def test_exact_quote_is_unchanged(tmp_path):
    config = _ready_world(
        tmp_path,
        text="DPR evaluates NQ",
        quote="DPR evaluates NQ",
    )

    report = comparison.run_comparison(config, chunk_ids=("c_test",))

    decision = report["chunks"][0]["reconstruction_decisions"][0]
    assert decision["status"] == "exact"
    assert decision["replacement_quote"] is None
    assert report["raw_response_identity"]["identical_between_arms"] is True
    assert report["delta"]["accepted_assertion_count"] == 0


def test_non_whitespace_change_is_refused(tmp_path):
    config = _ready_world(
        tmp_path,
        text="DPR evaluates NQ",
        quote="DPR tests NQ",
    )

    report = comparison.run_comparison(config, chunk_ids=("c_test",))

    decision = report["chunks"][0]["reconstruction_decisions"][0]
    assert decision["status"] == "not_found"
    assert decision["replacement_quote"] is None
    assert report["delta"]["accepted_assertion_count"] == 0


def test_unique_whitespace_repair_passes_unchanged_v2_gates(tmp_path):
    config = _ready_world(
        tmp_path,
        text="DPR\n evaluates NQ",
        quote="DPR evaluates NQ",
    )

    report = comparison.run_comparison(config, chunk_ids=("c_test",))

    decision = report["chunks"][0]["reconstruction_decisions"][0]
    assert decision["status"] == "repaired"
    assert decision["replacement_quote"] == "DPR\n evaluates NQ"
    assert report["baseline_summary"]["accepted_assertion_count"] == 0
    assert report["experiment_summary"]["accepted_assertion_count"] == 1
    assertion = report["chunks"][0]["experiment_state"]["accepted_assertions"][0]
    assert assertion["evidence_quote"] == "DPR\n evaluates NQ"
    assert assertion["evidence"]["char_start"] == 0
    assert assertion["evidence"]["char_end"] == len("DPR\n evaluates NQ")
    assert decision["candidate_identity"] == {
        "relation_type": assertion["relation_type"],
        "subject_ref": assertion["subject_ref"],
        "object_ref": assertion["object_ref"],
    }
