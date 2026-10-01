"""Offline shadow diagnostics over verified Manifest recording artifacts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from src.benchmark.manifest_recording import MissingRawResponse, sha256_bytes
from src.benchmark.manifest_replay import record_run, replay_run


ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY = ROOT / "ontology.yaml"
PROMPTS = ROOT / "prompts"


SCRIPT = ROOT / "scripts/benchmark/manifest_recording.py"
def _chunks() -> list[dict]:
    texts = (
        "No named retrieval system appears here.",
        "ELMo + improved Bi-DAF",
    )
    return [
        {
            "chunk_id": f"c_shadow_{index}",
            "doc_id": "W-shadow",
            "section_path": ["diagnostics"],
            "chunk_index": index,
            "text": text,
            "token_estimate": 8,
            "char_start": 0,
            "char_end": len(text),
        }
        for index, text in enumerate(texts)
    ]


def _body(payload: dict) -> dict:
    return {
        "model": "technical-fixture",
        "choices": [
            {
                "message": {"content": json.dumps(payload)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _record(tmp_path: Path) -> Path:
    answers = iter(
        [
            {"mentions": []},
            {"assertions": []},
            {
                "mentions": [
                    {
                        "entity_type": "method",
                        "surface_form": "ELMo",
                        "canonical_form": "ELMo",
                        "confidence": "stated",
                        "occurrence_index": 0,
                    },
                    {
                        "entity_type": "method",
                        "surface_form": "ELMo + improved Bi-DAF",
                        "canonical_form": "ELMo + improved Bi-DAF",
                        "confidence": "stated",
                        "occurrence_index": 0,
                    },
                    {
                        "entity_type": "method",
                        "surface_form": "Bi-DAF",
                        "canonical_form": "Bi-DAF",
                        "confidence": "stated",
                        "occurrence_index": 0,
                    },
                    {
                        "entity_type": "method",
                        "surface_form": "Bi-DAF",
                        "canonical_form": "BiDAF",
                        "confidence": "stated",
                        "occurrence_index": 0,
                    },
                ]
            },
            {"assertions": []},
        ]
    )

    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_body(next(answers)))

    recording = tmp_path / "recording"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        manifest = record_run(
            output_dir=recording,
            chunks=_chunks(),
            ontology_path=ONTOLOGY,
            prompts_dir=PROMPTS,
            max_live_calls=4,
            api_key="synthetic-test-key",
            client=client,
        )
    assert manifest["status"] == "complete"
    return recording


def test_shadow_report_is_deterministic_and_preserves_canonical_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = _record(tmp_path)
    canonical_before = {
        name: (recording / name).read_bytes()
        for name in ("extractions.jsonl", "extractions.rejects.jsonl")
    }
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")),
    )

    first = tmp_path / "replay-first"
    second = tmp_path / "replay-second"
    first_replay = replay_run(
        recording_dir=recording,
        output_dir=first,
        offline_diagnostics=True,
    )
    replay_run(
        recording_dir=recording,
        output_dir=second,
        offline_diagnostics=True,
    )

    first_bytes = (first / "offline_diagnostics.json").read_bytes()
    assert first_bytes == (second / "offline_diagnostics.json").read_bytes()
    report = json.loads(first_bytes)
    assert report["format"] == "benchmark-offline-diagnostics-v1"
    assert report["mode"] == "shadow"
    assert report["provider_calls"] == 0
    assert report["retries_enabled"] is False
    assert report["successful_empty_observations"] == [
        {
            "attempt": 1,
            "chunk_id": "c_shadow_0",
            "classification": "technical_observation",
            "domain_conclusion": None,
            "finish_reason": "stop",
            "request_sha256": report["successful_empty_observations"][0][
                "request_sha256"
            ],
            "response_sha256": report["successful_empty_observations"][0][
                "response_sha256"
            ],
        }
    ]
    assert any(
        group["chunk_id"] == "c_shadow_1"
        and group["relation"] == "contains"
        and group["outer"]["surface_form"] == "ELMo + improved Bi-DAF"
        and group["inner"]["surface_form"] == "ELMo"
        for group in report["overlap_groups"]
    )
    assert report["canonical_conflicts"] == [
        {
            "canonical_forms": ["Bi-DAF", "BiDAF"],
            "chunk_id": "c_shadow_1",
            "classification": "technical_anomaly",
            "domain_conclusion": None,
            "entity_type": "method",
            "mention_ids": [report["canonical_conflicts"][0]["mention_ids"][0]],
            "span": [16, 22],
            "surface_form": "Bi-DAF",
        }
    ]
    assert report["provenance"]["canonical_artifacts"] == {
        "extractions": {
            "path": "extractions.jsonl",
            "sha256": sha256_bytes(canonical_before["extractions.jsonl"]),
            "size": len(canonical_before["extractions.jsonl"]),
        },
        "rejects": {
            "path": "extractions.rejects.jsonl",
            "sha256": sha256_bytes(canonical_before["extractions.rejects.jsonl"]),
            "size": len(canonical_before["extractions.rejects.jsonl"]),
        },
    }
    assert first_replay["offline_diagnostics"]["path"] == "offline_diagnostics.json"
    assert first_replay["offline_diagnostics"]["sha256"] == sha256_bytes(first_bytes)
    assert canonical_before == {
        name: (recording / name).read_bytes() for name in canonical_before
    }
    assert canonical_before == {
        name: (first / name).read_bytes() for name in canonical_before
    }


def test_replay_cli_exposes_optional_shadow_mode(tmp_path: Path) -> None:
    recording = _record(tmp_path)
    spec = importlib.util.spec_from_file_location("manifest_recording_shadow", SCRIPT)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output = tmp_path / "cli-replay"

    assert cli.main(
        [
            "replay",
            "--recording-dir",
            str(recording),
            "--output-dir",
            str(output),
            "--offline-diagnostics",
        ]
    ) == 0
    assert (output / "offline_diagnostics.json").is_file()


def test_shadow_report_does_not_copy_surplus_manifest_metadata(tmp_path: Path) -> None:
    recording = _record(tmp_path)
    manifest_path = recording / "recording.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sentinel = "private-host-sentinel-7f3a"
    manifest["inputs"]["chunks"]["source_path"] = f"/{sentinel}/chunks.jsonl"
    manifest["artifacts"]["extractions"]["source_path"] = f"/{sentinel}/output.jsonl"
    manifest["raw_responses"]["attempts_index"]["source_path"] = (
        f"/{sentinel}/attempts.jsonl"
    )
    manifest["_directory"] = f"/{sentinel}/recording"
    manifest["_manifest_sha256"] = sentinel
    manifest["inputs"]["chunks"]["absolute_path"] = f"/{sentinel}/abs/chunks.jsonl"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "allowlisted-provenance"

    replay_run(
        recording_dir=recording,
        output_dir=output,
        offline_diagnostics=True,
    )

    report = json.loads((output / "offline_diagnostics.json").read_text(encoding="utf-8"))
    provenance = report["provenance"]
    inputs = provenance["inputs"]
    canonical = provenance["canonical_artifacts"]
    raw = provenance["raw_attempts"]
    for section in (inputs, canonical, raw):
        for entry in section.values():
            if not isinstance(entry, dict):
                continue
            assert set(entry) <= {"path", "sha256", "size"}
    assert "raw_responses" not in provenance
    top_level = set(report)
    assert sentinel not in json.dumps(report, sort_keys=True)
    assert sentinel not in (output / "offline_diagnostics.json").read_text(encoding="utf-8")
    assert sentinel not in top_level


def test_shadow_report_fails_closed_when_historical_raw_response_is_missing(
    tmp_path: Path,
) -> None:
    recording = _record(tmp_path)
    canonical_before = (recording / "extractions.jsonl").read_bytes()
    (recording / "raw_responses" / "000001.bin").unlink()
    output = tmp_path / "replay"

    with pytest.raises(MissingRawResponse):
        replay_run(
            recording_dir=recording,
            output_dir=output,
            offline_diagnostics=True,
        )

    assert (recording / "extractions.jsonl").read_bytes() == canonical_before
    assert not (output / "offline_diagnostics.json").exists()
