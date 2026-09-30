"""Offline end-to-end report tests using only a MockTransport recording."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from src.benchmark.manifest_replay import RecordingManifestError, record_run, replay_run
from src.benchmark.two_chunk_experiment import EXPERIMENT_CHUNK_IDS, load_recording_selection
from src.extract.ids import mention_id_v2
from src.extract.validation import source_text_sha256

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "src/benchmark/dataset/benchmark_chunks.jsonl"
ONTOLOGY = ROOT / "ontology.yaml"
PROMPTS = ROOT / "prompts"
DIAGNOSIS = ROOT / "scripts/experiments/two_chunk_20260930T120000Z/diagnose_dpr.py"


def _diagnoser():
    spec = importlib.util.spec_from_file_location("two_chunk_reports_diagnosis", DIAGNOSIS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _body(payload: dict) -> dict:
    return {"model": "technical-fixture", "choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def _record(tmp_path: Path) -> Path:
    """Frozen inputs plus technical transport responses, not scientific labels."""
    chunks, selection = load_recording_selection(DATASET, EXPERIMENT_CHUNK_IDS)
    dpr = next(chunk for chunk in chunks if chunk["chunk_id"] == EXPERIMENT_CHUNK_IDS[1])
    text = dpr["text"]
    dpr_positions = [text.index("DPR"), text.index("DPR", text.index("DPR") + 3)]
    first_id = mention_id_v2(chunk_id=dpr["chunk_id"], source_text_sha256=source_text_sha256(text), entity_type="method", char_start=dpr_positions[0], char_end=dpr_positions[0] + 3)
    outside_id = mention_id_v2(chunk_id=dpr["chunk_id"], source_text_sha256=source_text_sha256(text), entity_type="method", char_start=dpr_positions[1], char_end=dpr_positions[1] + 3)
    early_quote = text[text.rfind(".", 0, dpr_positions[0]) + 1:text.index(".", dpr_positions[0]) + 1].strip()
    answers = iter([{"mentions": []}, {"assertions": []}, {"mentions": [{"entity_type": "method", "surface_form": "DPR", "canonical_form": "DPR", "confidence": "stated", "occurrence_index": 0}, {"entity_type": "method", "surface_form": "DPR", "canonical_form": "DPR", "confidence": "stated", "occurrence_index": 1}]}, {"assertions": [{"relation_type": "EXTENDS", "subject_ref": {"kind": "mention", "id": first_id}, "object_ref": {"kind": "mention", "id": outside_id}, "evidence_quote": early_quote, "confidence": "stated"}]}])
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_body(next(answers)))
    recording = tmp_path / "recording"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        record_run(output_dir=recording, chunks=chunks, ontology_path=ONTOLOGY, prompts_dir=PROMPTS, max_live_calls=4, api_key="test-only-key", client=client, selection=selection)
    return recording


def test_offline_compare_and_diagnosis_bind_verified_two_chunk_recording(tmp_path: Path) -> None:
    diagnosis = _diagnoser()
    recording = _record(tmp_path)
    replay = tmp_path / "replay"
    report = replay_run(recording_dir=recording, output_dir=replay, whitespace_comparison=True)
    paths = diagnosis.write_dpr_diagnosis(recording_dir=recording, replay_dir=replay, output_dir=replay)
    payload = json.loads(paths["json"].read_text())
    dpr = payload["arms"]["canonical_v2"]["chunks"][EXPERIMENT_CHUNK_IDS[1]]
    assert report["status"] == "matched"
    reject = next(row for row in dpr["rejected_candidates"] if row["reason"] == "ENDPOINT_NOT_CONTAINED")
    assert reject["candidate_references"]["object_ref"]["id"] == "m_28e70c94eb305208"
    decision = dpr["comparison_candidate_references"][0]
    object_reference = decision["original_references"]["object_ref"]
    assert object_reference["id"] == "m_28e70c94eb305208"
    assert object_reference["span"] == [540, 543]
    assert object_reference["inside_quote_occurrence"] is False
    assert object_reference["surface_occurrences"] == [[302, 305], [540, 543]]
    assert decision["original_references"]["subject_ref"]["inside_quote_occurrence"] is True
    assert decision["replacement_references"]["object_ref"]["inside_quote_occurrence"] is None
    assert payload["arms"]["experimental_whitespace_v1"]["chunks"][EXPERIMENT_CHUNK_IDS[1]]["mentions"] == dpr["mentions"]
    assert payload["no_entity_remap"] is True


def test_diagnosis_rejects_tampered_canonical_artifact_without_touching_recording(tmp_path: Path) -> None:
    diagnosis = _diagnoser()
    recording = _record(tmp_path)
    replay = tmp_path / "replay"
    replay_run(recording_dir=recording, output_dir=replay, whitespace_comparison=True)
    before = (recording / "recording.json").read_bytes()
    (replay / "extractions.jsonl").write_bytes(b"tampered\n")
    with pytest.raises(ValueError):
        diagnosis.write_dpr_diagnosis(recording_dir=recording, replay_dir=replay, output_dir=replay)
    assert (recording / "recording.json").read_bytes() == before


def test_diagnosis_rejects_tampered_experimental_artifact(tmp_path: Path) -> None:
    diagnosis = _diagnoser()
    recording = _record(tmp_path)
    replay = tmp_path / "replay"
    replay_run(recording_dir=recording, output_dir=replay, whitespace_comparison=True)
    (replay / "whitespace_comparison" / "extractions.rejects.jsonl").write_bytes(b"tampered\n")
    with pytest.raises(ValueError):
        diagnosis.write_dpr_diagnosis(recording_dir=recording, replay_dir=replay, output_dir=replay)


def test_diagnosis_rejects_tampered_recording_selection(tmp_path: Path) -> None:
    diagnosis = _diagnoser()
    recording = _record(tmp_path)
    replay = tmp_path / "replay"
    replay_run(recording_dir=recording, output_dir=replay, whitespace_comparison=True)
    manifest_path = recording / "recording.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selection"]["chunk_ids"] = list(reversed(manifest["selection"]["chunk_ids"]))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RecordingManifestError):
        diagnosis.write_dpr_diagnosis(recording_dir=recording, replay_dir=replay, output_dir=replay)


def test_diagnosis_rejects_output_overlapping_recording_without_mutation(tmp_path: Path) -> None:
    diagnosis = _diagnoser()
    recording = _record(tmp_path)
    replay = tmp_path / "replay"
    replay_run(recording_dir=recording, output_dir=replay, whitespace_comparison=True)
    before = (recording / "recording.json").read_bytes()
    with pytest.raises(ValueError):
        diagnosis.write_dpr_diagnosis(recording_dir=recording, replay_dir=replay, output_dir=recording)
    assert (recording / "recording.json").read_bytes() == before


def test_diagnosis_rejects_tampered_comparison_decision(tmp_path: Path) -> None:
    diagnosis = _diagnoser()
    recording = _record(tmp_path)
    replay = tmp_path / "replay"
    replay_run(recording_dir=recording, output_dir=replay, whitespace_comparison=True)
    comparison_path = replay / "whitespace_comparison" / "comparison.json"
    comparison = json.loads(comparison_path.read_text())
    comparison["decisions"][0]["candidate_index"] = 99
    comparison_path.write_text(json.dumps(comparison))
    with pytest.raises(ValueError):
        diagnosis.write_dpr_diagnosis(recording_dir=recording, replay_dir=replay, output_dir=replay)
