"""Tests for the benchmark evaluator and blinded review generator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from src.benchmark.evaluate import (  # noqa: E402
    ChunkMetrics,
    ProviderMetrics,
    compute_metrics,
    metrics_to_dict,
    write_blinded_review_package,
    write_metrics_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_run(
    run_dir: Path,
    run_id: str,
    *,
    dataset_sha256: str = "deadbeef",
    mentions: list[dict] | None = None,
    asserts: list[dict] | None = None,
    rejects: list[dict] | None = None,
    chunk_statuses: list[dict] | None = None,
    sink: list[dict] | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    if mentions:
        records.extend(mentions)
    if asserts:
        records.extend(asserts)
    if chunk_statuses:
        records.extend(chunk_statuses)
    (run_dir / "extractions.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
        encoding="utf-8",
    )
    (run_dir / "extractions.rejects.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in (rejects or [])) + "\n",
        encoding="utf-8",
    )
    if sink:
        (run_dir / "metrics.jsonl").write_text(
            "\n".join(json.dumps(s, sort_keys=True) for s in sink) + "\n",
            encoding="utf-8",
        )
    summary = {
        "provider": {"name": "stub", "requested_model": "stub-1", "reported_model": "stub-1"},
        "dataset_sha256": dataset_sha256,
        "run_id": run_id,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def _sample_mention(cid: str, mid: str) -> dict:
    return {
        "kind": "mention",
        "chunk_id": cid,
        "mention_id": mid,
        "entity_type": "method",
        "surface_form": "DPR",
        "canonical_form": "Dense Passage Retrieval",
        "evidence": {"char_start": 0, "char_end": 3, "source_text_sha256": "x"},
        "confidence": "stated",
    }


def _sample_assertion(cid: str, aid: str) -> dict:
    return {
        "kind": "assertion",
        "chunk_id": cid,
        "assertion_id": aid,
        "relation_type": "EVALUATES_ON",
        "subject_ref": {"kind": "document", "id": "W1"},
        "object_ref": {"kind": "mention", "id": "m1"},
        "evidence_quote": "evaluates on NQ",
        "evidence": {"char_start": 0, "char_end": 16, "source_text_sha256": "x"},
        "confidence": "stated",
    }


def _sample_status(cid: str, complete: bool = True) -> dict:
    return {
        "kind": "chunk_status",
        "chunk_id": cid,
        "extraction_complete": complete,
        "entity_attempt_count": 1,
        "relation_attempt_count": 1,
        "schema_version": 2,
    }


def _sample_reject(cid: str, gate: str, phase: str = "entity") -> dict:
    return {
        "chunk_id": cid,
        "phase": phase,
        "attempt": 1,
        "gate": gate,
        "reason": f"{gate} error",
        "offending_record": None,
    }


def _sample_sink_call(cid: str, phase: str = "entity", duration: float = 0.1,
                      in_tokens: int = 5, out_tokens: int = 3) -> dict:
    return {
        "chunk_id": cid,
        "phase": phase,
        "requested_model": "stub-1",
        "reported_model": "stub-1",
        "duration_seconds": duration,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "cost_usd": None,
        "cache_hit": False,
        "live_call_index": 1,
        "status": "ok",
    }


def _tiny_chunks(tmp_path: Path) -> Path:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        json.dumps({"chunk_id": "c_a", "doc_id": "W1", "text": "x", "section_path": [], "chunk_index": 0}) + "\n"
        + json.dumps({"chunk_id": "c_b", "doc_id": "W1", "text": "y", "section_path": [], "chunk_index": 1}) + "\n",
        encoding="utf-8",
    )
    return path


def test_compute_metrics_counts_per_chunk_accepts(tmp_path: Path):
    run = _write_run(
        tmp_path / "r1", "r1",
        mentions=[
            _sample_mention("c_a", "m_a1"),
            _sample_mention("c_b", "m_b1"),
        ],
        asserts=[_sample_assertion("c_a", "a_a1")],
        chunk_statuses=[_sample_status("c_a"), _sample_status("c_b", complete=False)],
        sink=[_sample_sink_call("c_a"), _sample_sink_call("c_a"),
              _sample_sink_call("c_b"), _sample_sink_call("c_b")],
    )
    m = compute_metrics(
        output_jsonl=run / "extractions.jsonl",
        output_rejects=run / "extractions.rejects.jsonl",
        metrics_path=run / "metrics.jsonl",
        provider_name="stub",
        requested_model="stub-1",
        reported_model="stub-1",
        dataset_sha256="deadbeef",
    )
    assert m.chunks_total == 2
    assert m.mentions_accepted == 2
    assert m.assertions_accepted == 1
    assert m.chunks_complete == 1
    assert m.chunks_with_relation_ok == 1
    assert m.total_duration_seconds == pytest.approx(0.4)
    assert m.total_input_tokens == 20
    assert m.total_output_tokens == 12
    assert m.model_identity_verified is True


def test_compute_metrics_separates_gate_failures(tmp_path: Path):
    run = _write_run(
        tmp_path / "r1", "r1",
        mentions=[_sample_mention("c_a", "m_a1")],
        asserts=[],
        chunk_statuses=[_sample_status("c_a")],
        rejects=[
            _sample_reject("c_a", "schema", "entity"),
            _sample_reject("c_a", "alignment", "entity"),
            _sample_reject("c_a", "evidence", "relation"),
            _sample_reject("c_a", "ontology", "relation"),
        ],
    )
    m = compute_metrics(
        output_jsonl=run / "extractions.jsonl",
        output_rejects=run / "extractions.rejects.jsonl",
        metrics_path=None,
        provider_name="stub",
        requested_model="stub-1",
        reported_model="stub-1",
        dataset_sha256="x",
    )
    assert m.schema_errors == 1
    assert m.alignment_errors == 1
    assert m.evidence_errors == 1
    assert m.ontology_errors == 1
    assert m.candidates_rejected == 4


def test_compute_metrics_marks_model_mismatch(tmp_path: Path):
    run = _write_run(tmp_path / "r1", "r1", chunk_statuses=[_sample_status("c_a")])
    m = compute_metrics(
        output_jsonl=run / "extractions.jsonl",
        output_rejects=run / "extractions.rejects.jsonl",
        metrics_path=None,
        provider_name="stub",
        requested_model="stub-1",
        reported_model="stub-2",
        dataset_sha256="x",
    )
    assert m.model_identity_verified is False


def test_write_metrics_report_is_deterministic(tmp_path: Path):
    run = _write_run(tmp_path / "r1", "r1", chunk_statuses=[_sample_status("c_a")])
    m = compute_metrics(
        output_jsonl=run / "extractions.jsonl",
        output_rejects=run / "extractions.rejects.jsonl",
        metrics_path=None,
        provider_name="stub",
        requested_model="stub-1",
        reported_model="stub-1",
        dataset_sha256="x",
    )
    a = write_metrics_report(m, output_path=tmp_path / "a.json")
    b = write_metrics_report(m, output_path=tmp_path / "b.json")
    assert a.read_text() == b.read_text()


def test_metrics_to_dict_is_json_serialisable():
    m = ProviderMetrics(
        provider_name="stub",
        requested_model="stub-1",
        reported_model="stub-1",
        model_identity_verified=True,
        dataset_sha256="x",
        per_chunk=(ChunkMetrics(chunk_id="c_a"),),
    )
    d = metrics_to_dict(m)
    json.dumps(d)  # must not raise


def test_write_blinded_review_package_hides_model_identity(tmp_path: Path):
    chunks_path = _tiny_chunks(tmp_path)
    prompts_dir = REPO_ROOT / "prompts"
    run_a = _write_run(
        tmp_path / "run_a", "run_a",
        mentions=[_sample_mention("c_a", "m1")],
        asserts=[_sample_assertion("c_a", "a1")],
        chunk_statuses=[_sample_status("c_a")],
    )
    run_b = _write_run(
        tmp_path / "run_b", "run_b",
        mentions=[_sample_mention("c_a", "m2")],
        asserts=[],
        chunk_statuses=[_sample_status("c_a", complete=False)],
    )
    out = tmp_path / "review"
    rec_path, key_path = write_blinded_review_package(
        runs=[
            ("qwen", run_a / "extractions.jsonl", run_a / "extractions.rejects.jsonl",
             "extract_entities_v2.txt", "extract_relations_v2.txt"),
            ("manifest", run_b / "extractions.jsonl", run_b / "extractions.rejects.jsonl",
             "extract_entities_v2.txt", "extract_relations_v2.txt"),
        ],
        chunks_path=chunks_path,
        prompts_dir=prompts_dir,
        output_dir=out,
    )
    assert rec_path.is_file()
    assert key_path.is_file()
    rec_lines = [json.loads(line) for line in rec_path.read_text().splitlines() if line.strip()]
    # 2 chunks x 2 providers = 4 records
    assert len(rec_lines) == 4
    labels = {rec["review_label"] for rec in rec_lines}
    assert len(labels) == 4
    key_payload = json.loads(key_path.read_text())
    # Each chunk must appear under both providers in the key.
    for cid in ("c_a", "c_b"):
        labels_for_chunk = [e for e in key_payload["mapping"] if e["chunk_id"] == cid]
        assert len(labels_for_chunk) == 2
        assert {e["provider_label"] for e in labels_for_chunk} == {"qwen", "manifest"}


def test_write_blinded_review_package_does_not_emit_gold_labels(tmp_path: Path):
    chunks_path = _tiny_chunks(tmp_path)
    prompts_dir = REPO_ROOT / "prompts"
    run = _write_run(
        tmp_path / "run", "run",
        mentions=[_sample_mention("c_a", "m1")],
        asserts=[_sample_assertion("c_a", "a1")],
        chunk_statuses=[_sample_status("c_a")],
    )
    out = tmp_path / "review"
    rec_path, _ = write_blinded_review_package(
        runs=[("qwen", run / "extractions.jsonl", run / "extractions.rejects.jsonl",
               "extract_entities_v2.txt", "extract_relations_v2.txt")],
        chunks_path=chunks_path,
        prompts_dir=prompts_dir,
        output_dir=out,
    )
    payload = [json.loads(line) for line in rec_path.read_text().splitlines() if line.strip()]
    for rec in payload:
        for k, v in rec.items():
            assert "gold" not in k.lower()
            assert "precision" not in k.lower()
            assert "recall" not in k.lower()
            assert "f1" not in k.lower()


def test_write_blinded_review_package_is_reproducible(tmp_path: Path):
    chunks_path = _tiny_chunks(tmp_path)
    prompts_dir = REPO_ROOT / "prompts"
    run = _write_run(
        tmp_path / "run", "run",
        mentions=[_sample_mention("c_a", "m1")],
        asserts=[_sample_assertion("c_a", "a1")],
        chunk_statuses=[_sample_status("c_a")],
    )
    out1 = tmp_path / "r1"
    out2 = tmp_path / "r2"
    rec1, key1 = write_blinded_review_package(
        runs=[("qwen", run / "extractions.jsonl", run / "extractions.rejects.jsonl",
               "extract_entities_v2.txt", "extract_relations_v2.txt")],
        chunks_path=chunks_path, prompts_dir=prompts_dir, output_dir=out1,
    )
    rec2, key2 = write_blinded_review_package(
        runs=[("qwen", run / "extractions.jsonl", run / "extractions.rejects.jsonl",
               "extract_entities_v2.txt", "extract_relations_v2.txt")],
        chunks_path=chunks_path, prompts_dir=prompts_dir, output_dir=out2,
    )
    assert rec1.read_text() == rec2.read_text()
    assert key1.read_text() == key2.read_text()


def test_evaluate_cli_dry_runs(tmp_path: Path):
    """The evaluator CLI exits 0 with two stub runs."""
    chunks_path = _tiny_chunks(tmp_path)
    run_a = _write_run(tmp_path / "run_a", "run_a", chunk_statuses=[_sample_status("c_a")])
    run_b = _write_run(tmp_path / "run_b", "run_b", chunk_statuses=[_sample_status("c_a")])
    out = tmp_path / "eval"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "benchmark" / "evaluate.py"),
            "--run", str(run_a),
            "--run", str(run_b),
            "--provider-label", "qwen",
            "--provider-label", "manifest",
            "--chunks", str(chunks_path),
            "--output-dir", str(out),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "metrics_qwen.json").is_file()
    assert (out / "metrics_manifest.json").is_file()
    assert (out / "review_records.jsonl").is_file()
    assert (out / "review_key.json").is_file()