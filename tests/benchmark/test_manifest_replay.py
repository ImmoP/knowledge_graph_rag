"""End-to-end recording session tests; all HTTP is MockTransport-backed."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import httpx
import pytest

from src.benchmark.manifest_recording import MissingRawResponse
from src.benchmark.manifest_replay import (
    RecordingManifestError,
    load_recording,
    record_run,
    replay_run,
)

ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY = ROOT / "ontology.yaml"
PROMPTS = ROOT / "prompts"
SCRIPT = ROOT / "scripts" / "benchmark" / "manifest_recording.py"


def _chunk(text: str = "DPR evaluates NQ.") -> dict:
    return {
        "chunk_id": "c_recording_test",
        "doc_id": "W1",
        "section_path": ["results"],
        "chunk_index": 0,
        "text": text,
        "token_estimate": 5,
        "char_start": 0,
        "char_end": len(text),
    }


def _http_body(payload: dict, *, model: str = "fixture/model") -> dict:
    return {
        "model": model,
        "choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
    }


def _record(tmp_path: Path, payloads: list[dict], *, max_live_calls: int = 4):
    requests: list[httpx.Request] = []
    answers = iter(payloads)

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_http_body(next(answers)))

    directory = tmp_path / "recording"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        manifest = record_run(
            output_dir=directory,
            chunks=[_chunk()],
            ontology_path=ONTOLOGY,
            prompts_dir=PROMPTS,
            max_live_calls=max_live_calls,
            api_key="synthetic-test-key",
            client=client,
        )
    return directory, manifest, requests


def test_record_and_offline_replay_are_byte_exact_and_send_auth(tmp_path, monkeypatch):
    source, manifest, requests = _record(
        tmp_path, [{"mentions": []}, {"assertions": []}]
    )
    assert manifest["status"] == "complete"
    assert manifest["raw_responses"]["attempt_count"] == 2
    assert requests and all(r.headers["Authorization"] == "Bearer synthetic-test-key" for r in requests)
    persisted = b"".join(p.read_bytes() for p in source.rglob("*") if p.is_file())
    assert b"synthetic-test-key" not in persisted
    assert b"Authorization" not in persisted

    monkeypatch.setenv("MANIFEST_API_KEY", "must-not-be-read")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "forbidden-cache"))
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    output = tmp_path / "replay"
    report = replay_run(recording_dir=source, output_dir=output)
    assert report["status"] == "matched"
    assert report["attempt_count"] == 2
    assert (output / "extractions.jsonl").read_bytes() == (source / "extractions.jsonl").read_bytes()
    assert (output / "extractions.rejects.jsonl").read_bytes() == (source / "extractions.rejects.jsonl").read_bytes()
    assert not (tmp_path / "forbidden-cache").exists()
    assert (output / "empty_chunk_review.json").is_file()


def test_invalid_occurrence_reject_is_replayed_exactly(tmp_path):
    source, _manifest, _ = _record(
        tmp_path,
        [
            {"mentions": [{
                "entity_type": "dataset",
                "surface_form": "NQ",
                "canonical_form": "Natural Questions",
                "confidence": "stated",
                "occurrence_index": 9,
            }]},
            {"assertions": []},
        ],
    )
    rejects = (source / "extractions.rejects.jsonl").read_bytes()
    assert rejects
    reject_rows = [json.loads(line) for line in rejects.splitlines()]
    assert any("INVALID_OCCURRENCE_INDEX" in row["reason"] for row in reject_rows)
    output = tmp_path / "replay"
    replay_run(recording_dir=source, output_dir=output)
    assert (output / "extractions.rejects.jsonl").read_bytes() == rejects


def test_schema_correction_attempt_is_recorded_and_replayed(tmp_path):
    source, manifest, _ = _record(
        tmp_path,
        [{"wrong": []}, {"mentions": []}, {"assertions": []}],
    )
    assert manifest["raw_responses"]["attempt_count"] == 3
    report = replay_run(recording_dir=source, output_dir=tmp_path / "replay")
    assert report["attempt_count"] == 3


def test_repeated_schema_failure_stops_incomplete_and_replays_partial_artifacts(tmp_path):
    source, manifest, requests = _record(
        tmp_path,
        [{"wrong": []}, {"still_wrong": []}, {"unused": []}],
    )
    assert manifest["status"] == "incomplete"
    assert manifest["terminal"] == {"type": "ManifestTerminalInvalidOutput"}
    assert manifest["raw_responses"]["attempt_count"] == 2
    assert len(requests) == 2
    assert set(manifest["artifacts"]) == {"extractions", "rejects"}
    report = replay_run(recording_dir=source, output_dir=tmp_path / "replay")
    assert report["status"] == "matched"
    assert (tmp_path / "replay" / "extractions.jsonl").read_bytes() == (
        source / "extractions.jsonl"
    ).read_bytes()


def test_recording_429_stops_after_exact_response_and_replays_partial_artifacts(
    tmp_path, monkeypatch
):
    raw = b'{"error":{"type":"rate_limit","detail":"retain exactly"}}'
    requests: list[httpx.Request] = []

    def respond(request):
        requests.append(request)
        return httpx.Response(429, content=raw)

    monkeypatch.setattr("src.extract.pipeline_v2.time.sleep", lambda _seconds: None)
    source = tmp_path / "recording"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        manifest = record_run(
            output_dir=source, chunks=[_chunk()], ontology_path=ONTOLOGY,
            prompts_dir=PROMPTS, api_key="synthetic-test-key", client=client,
        )
    assert len(requests) == 1
    assert manifest["status"] == "terminal_error"
    assert manifest["terminal"] == {
        "type": "ManifestTerminalTransportError", "status": 429, "transient": False
    }
    assert (source / "raw_responses" / "000001.bin").read_bytes() == raw
    assert set(manifest["artifacts"]) == {"extractions", "rejects"}
    report = replay_run(recording_dir=source, output_dir=tmp_path / "replay")
    assert report["status"] == "matched"


def test_transport_and_json_retries_share_four_post_cap(tmp_path, monkeypatch):
    responses = iter([
        httpx.Response(503, json={"error": {"type": "overloaded"}}),
        httpx.Response(503, json={"error": {"type": "overloaded"}}),
        httpx.Response(200, json=_http_body({"wrong": []})),
        httpx.Response(200, json=_http_body({"mentions": []})),
    ])
    requests: list[httpx.Request] = []

    def respond(request):
        requests.append(request)
        return next(responses)

    monkeypatch.setattr("src.extract.pipeline_v2.time.sleep", lambda _seconds: None)
    source = tmp_path / "recording"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        manifest = record_run(
            output_dir=source, chunks=[_chunk()], ontology_path=ONTOLOGY,
            prompts_dir=PROMPTS, api_key="synthetic-test-key", client=client,
        )
    assert len(requests) == 4
    assert manifest["raw_responses"]["attempt_count"] == 4
    assert manifest["status"] == "budget_aborted"
    assert manifest["terminal"] == {"type": "CallBudgetExceeded", "cap": 4}
    assert manifest["settings"]["transport_policy"] == {
        "max_total_post_attempts": 4,
        "http_transport_retries": 0,
        "stop_on_http_429": True,
        "stop_on_invalid_output": True,
    }
    assert replay_run(recording_dir=source, output_dir=tmp_path / "replay")["status"] == "matched"


def test_nontransient_http_terminal_is_recorded_and_reproduced(tmp_path):
    def respond(_request):
        return httpx.Response(400, json={"error": {"type": "bad_request"}})

    source = tmp_path / "recording"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        manifest = record_run(
            output_dir=source, chunks=[_chunk()], ontology_path=ONTOLOGY,
            prompts_dir=PROMPTS, api_key="synthetic-test-key", client=client,
        )
    assert manifest["status"] == "terminal_error"
    assert manifest["terminal"]["status"] == 400
    report = replay_run(recording_dir=source, output_dir=tmp_path / "replay")
    assert report["status"] == "matched"




def test_call_cap_produces_replayable_partial_canonical_artifacts(tmp_path):
    source, manifest, _ = _record(
        tmp_path, [{"mentions": []}], max_live_calls=1
    )
    assert manifest["status"] == "budget_aborted"
    assert manifest["terminal"] == {"type": "CallBudgetExceeded", "cap": 1}
    report = replay_run(recording_dir=source, output_dir=tmp_path / "replay")
    assert report["status"] == "matched"
    assert report["result"]["budget_aborted"] is True


def test_tampered_input_and_missing_historical_raw_fail_closed(tmp_path):
    source, _manifest, _ = _record(
        tmp_path, [{"mentions": []}, {"assertions": []}]
    )
    chunks = source / "inputs" / "chunks.jsonl"
    chunks.write_bytes(chunks.read_bytes() + b" ")
    with pytest.raises(RecordingManifestError, match="hash mismatch"):
        load_recording(source)

    source2, _manifest2, _ = _record(
        tmp_path / "second", [{"mentions": []}, {"assertions": []}]
    )
    raw = next((source2 / "raw_responses").glob("*.bin"))
    raw.unlink()
    with pytest.raises(MissingRawResponse):
        replay_run(recording_dir=source2, output_dir=tmp_path / "never")


def test_replay_rejects_output_overlap_without_mutating_source(tmp_path):
    source, _manifest, _ = _record(
        tmp_path, [{"mentions": []}, {"assertions": []}]
    )
    child = source / "child"
    with pytest.raises(RecordingManifestError, match="overlaps"):
        replay_run(recording_dir=source, output_dir=child)
    assert not child.exists()


def test_recording_identity_drift_fails_closed(tmp_path):
    source, _manifest, _ = _record(
        tmp_path, [{"mentions": []}, {"assertions": []}]
    )
    manifest_path = source / "recording.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["implementation"]["dependencies"]["httpx"] = "0.0-tampered"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    with pytest.raises(RecordingManifestError, match="dependency identity drift"):
        load_recording(source)


def test_attempt_index_hash_drift_fails_closed(tmp_path):
    source, _manifest, _ = _record(
        tmp_path, [{"mentions": []}, {"assertions": []}]
    )
    index = source / "attempts.jsonl"
    index.write_bytes(index.read_bytes() + b" ")
    with pytest.raises(RecordingManifestError, match="hash mismatch"):
        load_recording(source)


def _load_cli():
    spec = importlib.util.spec_from_file_location("manifest_recording_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_live_and_cap_guards_are_fail_closed(tmp_path):
    cli = _load_cli()
    with pytest.raises(SystemExit):
        cli._parse_args(["record", "--output-dir", str(tmp_path / "out"), "--allow-manifest-calls", "4"])
    with pytest.raises(SystemExit):
        cli._parse_args([
            "record", "--live", "--output-dir", str(tmp_path / "out"),
            "--allow-manifest-calls", "5",
        ])
    args = cli._parse_args([
        "record", "--live", "--output-dir", str(tmp_path / "out"),
        "--allow-manifest-calls", "1",
    ])
    assert args.live is True and args.allow_manifest_calls == 1


