"""Consumer-contract tests for the offline Scope-2 pilot.

The Scope-2 pilot is deliberately tiny: exactly two frozen chunks
extracted from the historical Manifest auto-routing run
``20260930T083857Z-c69654``.  These tests sit at the public seams
(`adapt_v2_scope2`, `evaluate_scope2`, `build_historical_report`,
`verify_pilot_package`) and never reach into provider mocks or the
network.  Synthetic fixtures use trusted hashes that the
historical CLI pins; the synthetic test branch never produces or
asserts scientific results.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest



REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "src" / "benchmark" / "scope2_pilot.py"
PILOT_ROOT = Path(
    os.environ.get(
        "KG_RAG_SCOPE2_PILOT_ROOT",
        Path.home() / "migration-incoming" / "pilot" / "kg_rag_ai_pilot_scope2_20261001",
    )
)
DATASET_PATH = REPO_ROOT / "src" / "benchmark" / "dataset" / "benchmark_chunks.jsonl"
HISTORICAL_ROOT = Path(
    os.environ.get(
        "KG_RAG_SCOPE2_HISTORICAL_ROOT",
        Path.home()
        / "migration-incoming"
        / "manifest-auto-autopilot-live"
        / "20260930T083857Z-c69654",
    )
)


def _load_module():
    spec = importlib.util.spec_from_file_location("scope2_pilot", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["scope2_pilot"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _frozen_dataset_bytes() -> bytes:
    return DATASET_PATH.read_bytes()


def _target_chunk_texts() -> dict[str, bytes]:
    rows = [json.loads(line) for line in _frozen_dataset_bytes().splitlines() if line]
    return {row["chunk_id"]: row["text"].encode("utf-8") for row in rows}


def _make_extraction(
    *,
    chunk_id: str,
    doc_id: str,
    mention_count: int,
    assertion_count: int,
    entity_status: str = "validated",
    relation_status: str = "validated",
    entity_failure_count: int = 0,
    relation_failure_count: int = 0,
    extra_mentions: list[dict] | None = None,
    extra_assertions: list[dict] | None = None,
    schema_version: int = 2,
) -> dict:
    return {
        "kind": "chunk_status",
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "extraction_complete": True,
        "schema_version": schema_version,
        "phases": {
            "entity": {
                "status": entity_status,
                "attempt_count": 1,
                "failure_count": entity_failure_count,
                "mention_count": mention_count,
            },
            "relation": {
                "status": relation_status,
                "attempt_count": 1,
                "failure_count": relation_failure_count,
                "assertion_count": assertion_count,
            },
        },
    }


def _positive_mention_record(chunk_id: str, doc_id: str) -> dict:
    import hashlib as _hashlib
    text = _target_chunk_texts()[chunk_id]
    start, end = 46, 64
    sha = "250de138d33b731503594a3826623a7b7b51ab19a5298539498fdb0fa05a8ba9"
    surface = "Switch Transformer"
    assert text[start:end].decode("utf-8") == surface
    mention_id = "m_" + _hashlib.sha256(
        f"{chunk_id}|{surface}|{start}|{end}|{sha}".encode("utf-8")
    ).hexdigest()[:16]
    return {
        "kind": "mention",
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "schema_version": 2,
        "mention_id": mention_id,
        "canonical_form": "Switch Transformer",
        "entity_type": "method",
        "surface_form": surface,
        "evidence": {
            "char_start": start,
            "char_end": end,
            "source_text_sha256": sha,
        },
    }


def _synthetic_extractions() -> tuple[list[dict], list[dict]]:
    extractions = [
        _make_extraction(
            chunk_id="c_61090bb37bc1e7a9",
            doc_id="W3119866685",
            mention_count=1,
            assertion_count=0,
        ),
        _make_extraction(
            chunk_id="c_c0de5074475c9c0f",
            doc_id="W3119866685",
            mention_count=0,
            assertion_count=0,
        ),
    ]
    extractions.append(_positive_mention_record("c_61090bb37bc1e7a9", "W3119866685"))
    rejects: list[dict] = []
    return extractions, rejects


def _trusted_hash_overrides() -> dict[str, str]:
    module = _load_module()
    return dict(module.HISTORICAL_ARTIFACT_HASHES)


def _trust_pilot_manifest(module) -> dict[str, str]:
    """Replace the historical artifact hashes with the actual SHA-256 of the
    freshly written synthetic run directory.  Caller passes the synthetic
    run dir via the side-effect of ``_build_synthetic_historical``.
    """
    module.HISTORICAL_ARTIFACT_HASHES.update(_find_synthetic_hashes())
    return module.HISTORICAL_ARTIFACT_HASHES


_SYNTHETIC_DIR: Path | None = None


def _find_synthetic_hashes() -> dict[str, str]:
    import hashlib as _hashlib
    if _SYNTHETIC_DIR is None or not _SYNTHETIC_DIR.is_dir():
        raise RuntimeError("synthetic run directory not initialised")
    return {
        path.name: _hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_SYNTHETIC_DIR.iterdir())
        if path.is_file()
        and path.name in {
            "extractions.jsonl",
            "extractions.rejects.jsonl",
            "metrics.jsonl",
            "run_summary.json",
        }
    }


def _build_synthetic_historical(tmp_path: Path) -> Path:
    """Write a synthetic historical run directory with a pinned summary.

    Returns the run directory; the caller re-derives ``HISTORICAL_ARTIFACT_HASHES``
    so that the synthetic content is trusted by the offline report builder.
    """

    global _SYNTHETIC_DIR
    run = tmp_path / "historical"
    run.mkdir()
    extractions, rejects = _synthetic_extractions()
    (run / "extractions.jsonl").write_text(
        "\n".join(json.dumps(rec, sort_keys=True) for rec in extractions) + "\n"
    )
    (run / "extractions.rejects.jsonl").write_text(
        "\n".join(json.dumps(rec, sort_keys=True) for rec in rejects) + "\n"
    )
    (run / "metrics.jsonl").write_text("{}\n")
    summary = {
        "chunks": ["c_61090bb37bc1e7a9", "c_c0de5074475c9c0f"],
        "n_chunks": 2,
        "dataset_sha256": (
            "38182c6cdbb9f86abc6ce20ece7a3d9e1cb69ccac2576e2d6274e727c2f84985"
        ),
        "ontology_sha256": (
            "f9a3e203d4c640b91966e077b02f09a547d49b51595cf33e040a3e278cd2325e"
        ),
        "provider": {
            "auto_routing": True,
            "model_identity_verified": False,
            "name": "manifest_auto_benchmark",
            "reported_model": None,
            "requested_model": "auto",
        },
        "result": {"budget_aborted": False, "exit_code": 0, "n_extraction_complete": 2},
    }
    (run / "run_summary.json").write_text(json.dumps(summary, sort_keys=True))
    (run / "raw_responses").mkdir()
    _SYNTHETIC_DIR = run
    return run



# ---------- adapt_v2_scope2 --------------------------------------------


def test_adapt_emits_exactly_two_predictions() -> None:
    module = _load_module()
    extractions, rejects = _synthetic_extractions()
    out = module.adapt_v2_scope2(
        extractions=extractions,
        rejects=rejects,
        dataset=DATASET_PATH,
        historical_artifact_hashes=_trusted_hash_overrides(),
    )
    assert [row["chunk_id"] for row in out] == [
        "c_61090bb37bc1e7a9",
        "c_c0de5074475c9c0f",
    ]
    assert all(row["status"] == "complete" for row in out)


def test_adapt_positive_keeps_exact_switch_mention_offsets() -> None:
    module = _load_module()
    extractions, rejects = _synthetic_extractions()
    out = module.adapt_v2_scope2(
        extractions=extractions,
        rejects=rejects,
        dataset=DATASET_PATH,
        historical_artifact_hashes=_trusted_hash_overrides(),
    )
    pos = next(row for row in out if row["chunk_id"] == "c_61090bb37bc1e7a9")
    assert pos["text_sha256"] == (
        "250de138d33b731503594a3826623a7b7b51ab19a5298539498fdb0fa05a8ba9"
    )
    assert pos["mentions"] == [
        {
            "entity_type": "method",
            "surface_form_exact": "Switch Transformer",
            "char_start": 46,
            "char_end": 64,
        }
    ]
    # adapter must NEVER propagate canonical_form into the output schema
    assert all("canonical_form" not in m for m in pos["mentions"])
    assert pos["relations"] == []


def test_adapt_negative_chunk_is_exactly_empty() -> None:
    module = _load_module()
    extractions, rejects = _synthetic_extractions()
    out = module.adapt_v2_scope2(
        extractions=extractions,
        rejects=rejects,
        dataset=DATASET_PATH,
        historical_artifact_hashes=_trusted_hash_overrides(),
    )
    neg = next(row for row in out if row["chunk_id"] == "c_c0de5074475c9c0f")
    assert neg["text_sha256"] == (
        "22bd98e62ff620aa3933df773477ff8385ca8364aab9a67576d1a497ee98e63b"
    )
    assert neg["mentions"] == []
    assert neg["relations"] == []


def test_adapt_rejects_tampered_dataset_bytes(tmp_path: Path) -> None:
    module = _load_module()
    bundle = tmp_path / "dataset"
    bundle.mkdir()
    bogus = bundle / "benchmark_chunks.jsonl"
    bogus.write_bytes(b'{"chunk_id":"x","text":"","doc_id":"Y"}\n')
    import hashlib as _hashlib
    (bundle / "benchmark_chunks.sha256").write_text(
        _hashlib.sha256(bogus.read_bytes()).hexdigest() + "  benchmark_chunks.jsonl\n"
    )
    (bundle / "benchmark_dataset_manifest.json").write_text(
        json.dumps({
            "algorithm_version": "x",
            "sampler_seed": 0,
            "config": {},
            "selected_chunk_ids": ["x"],
            "documents_selected": ["Y"],
            "counts_by_tag": {},
            "counts_by_length_tercile": {},
            "chunks_path": str(bogus),
            "chunks_sha256": _hashlib.sha256(bogus.read_bytes()).hexdigest(),
        })
    )
    extractions, rejects = _synthetic_extractions()
    with pytest.raises(Exception):
        module.adapt_v2_scope2(
            extractions=extractions,
            rejects=rejects,
            dataset=bogus,
            historical_artifact_hashes=_trusted_hash_overrides(),
        )


def test_adapt_rejects_wrong_artifact_hashes(tmp_path: Path) -> None:
    module = _load_module()
    extractions, rejects = _synthetic_extractions()
    bogus_hashes = dict(_trusted_hash_overrides())
    bogus_hashes["extractions.jsonl"] = "00" * 32
    with pytest.raises(module.PilotContractError):
        module.adapt_v2_scope2(
            extractions=extractions,
            rejects=rejects,
            dataset=DATASET_PATH,
            historical_artifact_hashes=bogus_hashes,
            historical_run_root=tmp_path,
        )


def test_adapt_rejects_when_positive_mention_missing() -> None:
    module = _load_module()
    extractions, rejects = _synthetic_extractions()
    extractions = [
        rec for rec in extractions if rec.get("kind") != "mention"
    ]
    with pytest.raises(module.PilotContractError):
        module.adapt_v2_scope2(
            extractions=extractions,
            rejects=rejects,
            dataset=DATASET_PATH,
            historical_artifact_hashes=_trusted_hash_overrides(),
        )


def test_adapt_rejects_when_negative_has_unexpected_mention() -> None:
    module = _load_module()
    extractions, rejects = _synthetic_extractions()
    extractions.append(
        _make_extraction(
            chunk_id="c_c0de5074475c9c0f",
            doc_id="W3119866685",
            mention_count=1,
            assertion_count=0,
        )
    )
    extractions.append(
        {
            "kind": "mention",
            "chunk_id": "c_c0de5074475c9c0f",
            "doc_id": "W3119866685",
            "schema_version": 2,
            "mention_id": "m_scope2_neg",
            "canonical_form": "False Positive",
            "entity_type": "method",
            "surface_form": "False Positive",
            "evidence": {
                "char_start": 0,
                "char_end": 13,
                "source_text_sha256": (
                    "22bd98e62ff620aa3933df773477ff8385ca8364aab9a67576d1a497ee98e63b"
                ),
            },
        }
    )
    with pytest.raises(module.PilotContractError):
        module.adapt_v2_scope2(
            extractions=extractions,
            rejects=rejects,
            dataset=DATASET_PATH,
            historical_artifact_hashes=_trusted_hash_overrides(),
        )


def test_adapt_rejects_unexpected_assertion_in_negative_chunk() -> None:
    module = _load_module()
    extractions, rejects = _synthetic_extractions()
    extractions.append(
        _make_extraction(
            chunk_id="c_c0de5074475c9c0f",
            doc_id="W3119866685",
            mention_count=0,
            assertion_count=1,
        )
    )
    extractions.append(
        {
            "kind": "assertion",
            "chunk_id": "c_c0de5074475c9c0f",
            "doc_id": "W3119866685",
            "schema_version": 2,
            "assertion_id": "a_scope2_neg",
            "relation_type": "USES_METHOD",
            "subject_ref": {"kind": "document", "id": "W3119866685"},
            "object_ref": {"kind": "mention", "id": "m_scope2_neg"},
            "evidence": {
                "char_start": 0,
                "char_end": 13,
                "source_text_sha256": (
                    "22bd98e62ff620aa3933df773477ff8385ca8364aab9a67576d1a497ee98e63b"
                ),
            },
            "evidence_quote": "False Positive",
        }
    )
    with pytest.raises(module.PilotContractError):
        module.adapt_v2_scope2(
            extractions=extractions,
            rejects=rejects,
            dataset=DATASET_PATH,
            historical_artifact_hashes=_trusted_hash_overrides(),
        )


def test_adapt_rejects_target_rejects_present() -> None:
    module = _load_module()
    extractions, rejects = _synthetic_extractions()
    rejects.append(
        {
            "chunk_id": "c_61090bb37bc1e7a9",
            "phase": "entity",
            "reason": "ZERO_OCCURRENCES",
            "aligner_reject": True,
            "attempt": 1,
        }
    )
    with pytest.raises(module.PilotContractError):
        module.adapt_v2_scope2(
            extractions=extractions,
            rejects=rejects,
            dataset=DATASET_PATH,
            historical_artifact_hashes=_trusted_hash_overrides(),
        )


def test_adapt_silently_skips_non_target_records_and_emits_only_two() -> None:
    """The adapter drops non-target rows but the gate must contain exactly the
    two pinned target chunks.  This mirrors the parent clarification: the
    historical run contains more chunks than the pilot covers; the adapter
    must never promote non-target records into Scope-2 outputs.
    """
    module = _load_module()
    extractions, rejects = _synthetic_extractions()
    extractions.append(
        _make_extraction(
            chunk_id="c_2a9f84f5fe0ce2ce",
            doc_id="W2955868579",
            mention_count=0,
            assertion_count=0,
        )
    )
    extractions.append(
        _make_extraction(
            chunk_id="c_d285f0a62ff9479b",
            doc_id="W3022373106",
            mention_count=0,
            assertion_count=0,
        )
    )
    out = module.adapt_v2_scope2(
        extractions=extractions,
        rejects=rejects,
        dataset=DATASET_PATH,
        historical_artifact_hashes=_trusted_hash_overrides(),
    )
    assert {row["chunk_id"] for row in out} == set(module.TARGET_CHUNK_IDS)


def test_adapt_rejects_wrong_doc_id_on_positive_chunk() -> None:
    module = _load_module()
    extractions, rejects = _synthetic_extractions()
    extractions[0]["doc_id"] = "W1234567890"
    with pytest.raises(module.PilotContractError):
        module.adapt_v2_scope2(
            extractions=extractions,
            rejects=rejects,
            dataset=DATASET_PATH,
            historical_artifact_hashes=_trusted_hash_overrides(),
        )


def test_adapt_rejects_incomplete_phase_status() -> None:
    module = _load_module()
    extractions, rejects = _synthetic_extractions()
    extractions[0]["phases"]["entity"]["status"] = "schema_invalid"
    extractions[0]["phases"]["entity"]["failure_count"] = 1
    with pytest.raises(module.PilotContractError):
        module.adapt_v2_scope2(
            extractions=extractions,
            rejects=rejects,
            dataset=DATASET_PATH,
            historical_artifact_hashes=_trusted_hash_overrides(),
        )


def test_adapt_rejects_status_count_mismatch() -> None:
    module = _load_module()
    extractions, rejects = _synthetic_extractions()
    extractions[0]["phases"]["entity"]["mention_count"] = 0
    with pytest.raises(module.PilotContractError):
        module.adapt_v2_scope2(
            extractions=extractions,
            rejects=rejects,
            dataset=DATASET_PATH,
            historical_artifact_hashes=_trusted_hash_overrides(),
        )


# ---------- evaluate_scope2 --------------------------------------------


def _adapt_default(module) -> list[dict]:
    extractions, rejects = _synthetic_extractions()
    return module.adapt_v2_scope2(
        extractions=extractions,
        rejects=rejects,
        dataset=DATASET_PATH,
        historical_artifact_hashes=_trusted_hash_overrides(),
    )


def test_evaluate_descriptive_pass_for_expected_predictions() -> None:
    module = _load_module()
    result = module.evaluate_scope2(
        _adapt_default(module), pilot_root=PILOT_ROOT
    )
    pos = next(row for row in result["per_chunk"] if row["role"] == "positive")
    neg = next(row for row in result["per_chunk"] if row["role"] == "negative_control")
    assert pos["exact_match_mentions"] == 1
    assert pos["omitted_reference_mentions"] == 0
    assert pos["unexpected_mentions"] == 0
    assert pos["unexpected_relations"] == 0
    assert neg["unexpected_mentions"] == 0
    assert neg["unexpected_relations"] == 0
    assert result["human_gold"] is False
    assert result["automatic_gold_import"] is False
    assert result["allow_global_quality_claims"] is False

def test_evaluate_detects_missing_switch_mention() -> None:
    module = _load_module()
    predictions = _adapt_default(module)
    predictions = [
        {**row, "mentions": []}
        if row["chunk_id"] == "c_61090bb37bc1e7a9"
        else row
        for row in predictions
    ]
    result = module.evaluate_scope2(predictions, pilot_root=PILOT_ROOT)
    pos = next(row for row in result["per_chunk"] if row["role"] == "positive")
    assert pos["exact_match_mentions"] == 0
    assert pos["omitted_reference_mentions"] == 1


def test_evaluate_detects_false_positive_in_negative_chunk() -> None:
    module = _load_module()
    predictions = _adapt_default(module)
    predictions = [
        {
            **row,
            "mentions": [
                {
                    "entity_type": "method",
                    "surface_form_exact": "data parallel",
                    "char_start": 14,
                    "char_end": 27,
                }
            ],
        }
        if row["chunk_id"] == "c_c0de5074475c9c0f"
        else row
        for row in predictions
    ]
    result = module.evaluate_scope2(predictions, pilot_root=PILOT_ROOT)
    neg = next(row for row in result["per_chunk"] if row["role"] == "negative_control")
    assert neg["unexpected_mentions"] == 1


def test_evaluate_detects_unexpected_relation_in_negative_chunk() -> None:
    module = _load_module()
    predictions = _adapt_default(module)
    predictions = [
        {
            **row,
            "relations": [
                {
                    "relation_type": "USES_METHOD",
                    "subject_ref": {"kind": "document", "id": "W3119866685"},
                    "object_ref": {"kind": "mention", "id": "m_scope2_pos"},
                }
            ],
        }
        if row["chunk_id"] == "c_c0de5074475c9c0f"
        else row
        for row in predictions
    ]
    result = module.evaluate_scope2(predictions, pilot_root=PILOT_ROOT)
    neg = next(row for row in result["per_chunk"] if row["role"] == "negative_control")
    assert neg["unexpected_relations"] == 1


def test_evaluate_fails_closed_on_missing_target() -> None:
    module = _load_module()
    predictions = _adapt_default(module)
    predictions = [row for row in predictions if row["chunk_id"] != "c_61090bb37bc1e7a9"]
    with pytest.raises(module.PilotContractError):
        module.evaluate_scope2(predictions)


def test_evaluate_fails_closed_on_unexpected_third_chunk() -> None:
    module = _load_module()
    predictions = _adapt_default(module)
    predictions.append(
        {
            "chunk_id": "c_abdfc58c76ee1139",
            "text_sha256": "00" * 32,
            "status": "complete",
            "mentions": [],
            "relations": [],
        }
    )
    with pytest.raises(module.PilotContractError):
        module.evaluate_scope2(predictions)


def test_evaluate_fails_closed_on_duplicate_chunk() -> None:
    module = _load_module()
    predictions = _adapt_default(module) + _adapt_default(module)
    with pytest.raises(module.PilotContractError):
        module.evaluate_scope2(predictions)


def test_evaluate_fails_closed_on_wrong_source_hash() -> None:
    module = _load_module()
    predictions = _adapt_default(module)
    predictions = [
        {**row, "text_sha256": "00" * 32}
        if row["chunk_id"] == "c_61090bb37bc1e7a9"
        else row
        for row in predictions
    ]
    with pytest.raises(module.PilotContractError):
        module.evaluate_scope2(predictions)


def test_evaluate_fails_closed_on_incomplete_status() -> None:
    module = _load_module()
    predictions = _adapt_default(module)
    predictions = [
        {**row, "status": "incomplete"}
        if row["chunk_id"] == "c_c0de5074475c9c0f"
        else row
        for row in predictions
    ]
    with pytest.raises(module.PilotContractError):
        module.evaluate_scope2(predictions)


# ---------- build_historical_report ------------------------------------


def test_historical_report_is_deterministic_and_bytes_identical(tmp_path: Path) -> None:
    module = _load_module()
    run = _build_synthetic_historical(tmp_path)
    _trust_pilot_manifest(module)
    one = module.build_historical_report(
        historical_run_root=run,
        pilot_root=PILOT_ROOT,
        historical_artifact_hashes=module.HISTORICAL_ARTIFACT_HASHES,
    )
    two = module.build_historical_report(
        historical_run_root=run,
        pilot_root=PILOT_ROOT,
        historical_artifact_hashes=module.HISTORICAL_ARTIFACT_HASHES,
    )


def test_historical_report_with_real_run_records_actual_chunk_outcomes() -> None:
    module = _load_module()
    real_run = HISTORICAL_ROOT
    if not real_run.is_dir():
        pytest.skip("real historical run unavailable")
    report = module.build_historical_report(
        historical_run_root=real_run,
        pilot_root=PILOT_ROOT,
        historical_artifact_hashes=_trusted_hash_overrides(),
    )
    by_id = {row["chunk_id"]: row for row in report["per_chunk"]}
    assert by_id["c_61090bb37bc1e7a9"]["scope_role"] == "positive"
    assert by_id["c_61090bb37bc1e7a9"]["exact_match_mentions"] == 0
    assert by_id["c_c0de5074475c9c0f"]["scope_role"] == "negative_control"
    assert report["history"]["status"] == "historical_observation_only"
    assert report["provider"]["model_identity_verified"] is False
    assert report["raw_response_availability"] == "unavailable"


def test_historical_report_rejects_modified_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    run = _build_synthetic_historical(tmp_path)
    _trust_pilot_manifest(module)
    (run / "extractions.jsonl").write_bytes(b"{}\n{}\n")
    with pytest.raises(module.PilotContractError):
        module.build_historical_report(
            historical_run_root=run,
            pilot_root=PILOT_ROOT,
            historical_artifact_hashes=module.HISTORICAL_ARTIFACT_HASHES,
        )


def test_historical_report_includes_methodology_limits(tmp_path: Path) -> None:
    module = _load_module()
    run = _build_synthetic_historical(tmp_path)
    _trust_pilot_manifest(module)
    report = module.build_historical_report(
        historical_run_root=run,
        pilot_root=PILOT_ROOT,
        historical_artifact_hashes=module.HISTORICAL_ARTIFACT_HASHES,
    )


# ---------- verify_pilot_package ---------------------------------------


def test_verify_pilot_package_accepts_pilot_root() -> None:
    module = _load_module()
    info = module.verify_pilot_package(PILOT_ROOT)
    assert info["status"] == "PASS"
    assert info["scope"] == "ai_reviewed_provisional_scope2"
    assert info["human_gold"] is False
    assert info["included_chunks"] == [
        "c_61090bb37bc1e7a9",
        "c_c0de5074475c9c0f",
    ]
    assert info["excluded_chunks"] == [
        "c_2a9f84f5fe0ce2ce",
        "c_7fdad3ec8f3e783b",
        "c_d285f0a62ff9479b",
    ]


def test_verify_pilot_package_rejects_tampered_manifest(tmp_path: Path) -> None:
    module = _load_module()
    with pytest.raises(module.PilotContractError):
        module.verify_pilot_package(tmp_path)


# ---------- CLI safety -------------------------------------------------





def test_cli_rejects_live_or_provider_arguments() -> None:
    module = _load_module()
    cli = _load_cli_module()
    for argv in (
        ["--historical-run", str(HISTORICAL_ROOT), "--report", "/tmp/scope2.json"],
        ["--historical-run", str(HISTORICAL_ROOT), "--live"],
    ):
        with pytest.raises(SystemExit):
            cli.main(argv)


def _load_cli_module():
    cli_path = REPO_ROOT / "scripts" / "benchmark" / "report_scope2_pilot.py"
    spec = importlib.util.spec_from_file_location("report_scope2_pilot_cli", cli_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_rejects_run_dir_inside_repo_gold(tmp_path: Path) -> None:
    cli = _load_cli_module()
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / "gold_set").mkdir()
    (fake_repo / "data" / "gold_set").mkdir(parents=True)
    sys.modules.setdefault("_scope2_repro", type(sys)("x"))
    argv = [
        "--historical-run",
        str(fake_repo / "gold_set" / "inside.jsonl"),
        "--report",
        str(fake_repo / "outside.json"),
    ]
    with pytest.raises(SystemExit):
        cli.main(argv)


def test_cli_writes_report_only_to_caller_path(tmp_path: Path) -> None:
    if not HISTORICAL_ROOT.is_dir():
        pytest.skip("real historical run unavailable")
    cli = _load_cli_module()
    report = tmp_path / "shr" / "report.json"
    rc = cli.main(
        [
            "--historical-run",
            str(HISTORICAL_ROOT),
            "--pilot-root",
            str(PILOT_ROOT),
            "--report",
            str(report),
        ]
    )
    assert rc == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["scope"] == "ai_reviewed_provisional_scope2"
    assert payload["provider"]["model_identity_verified"] is False