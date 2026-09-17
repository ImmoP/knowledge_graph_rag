"""Tests for the graph-independent held-out leakage freeze (#61)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from src.eval import leakage


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    chunks = [{"chunk_id": f"c_{index:016x}"} for index in range(12)]
    manifest = [{"openalex_id": "W1"}]
    questions: list[dict] = []
    for index in range(45):
        if index < 15:
            bucket = "1"
            endpoints = ["c_0000000000000001", "W1"]
        elif index < 30:
            bucket = "2"
            endpoints = [
                "c_0000000000000001",
                "c_0000000000000002",
                "W1",
            ]
        else:
            bucket = "3+"
            endpoints = [
                "c_0000000000000001",
                "c_0000000000000002",
                "c_0000000000000003",
                "W1",
            ]
        path = [
            {
                "step": step + 1,
                "from": endpoints[step],
                "relation": "CITES",
                "to": endpoints[step + 1],
                "direction": "forward",
            }
            for step in range(len(endpoints) - 1)
        ]
        questions.append(
            {
                "qid": f"q_{index + 1:03d}",
                "question": f"Question {index + 1}",
                "expected_answer": f"Answer {index + 1}",
                "answer_aliases": [f"Alias {index + 1}"],
                "hop_bucket": bucket,
                "evidence_path": path,
                "supporting_doc_ids": ["W1"],
                "supporting_chunk_ids": ["c_0000000000000000"],
                "split": "held_out",
                "notes": f"Private note {index + 1}",
            }
        )

    paths = {
        "draft": tmp_path / "questions.draft.jsonl",
        "chunks": tmp_path / "chunks.jsonl",
        "manifest": tmp_path / "manifest.jsonl",
        "ontology": tmp_path / "ontology.yaml",
        "output": tmp_path / "held_out_leakage.v1.json",
    }
    _write_jsonl(paths["draft"], questions)
    _write_jsonl(paths["chunks"], chunks)
    _write_jsonl(paths["manifest"], manifest)
    paths["ontology"].write_text(
        json.dumps({"relation_types": [{"name": "CITES"}]}) + "\n",
        encoding="utf-8",
    )
    return paths


def _freeze(paths: dict[str, Path]) -> str:
    return leakage.freeze_leakage(
        paths["draft"],
        paths["output"],
        chunks_path=paths["chunks"],
        manifest_path=paths["manifest"],
        ontology_path=paths["ontology"],
    )


def test_freeze_emits_only_leakage_metadata(tmp_path):
    paths = _fixture(tmp_path)
    sha256 = _freeze(paths)
    raw = paths["output"].read_bytes()
    payload = json.loads(raw)

    assert sha256 == hashlib.sha256(raw).hexdigest()
    assert payload["schema_version"] == "held-out-leakage.v1"
    assert payload["record_count"] == 45
    assert payload["all_split_held_out"] is True
    assert payload["source_draft_sha256"] == hashlib.sha256(
        paths["draft"].read_bytes()
    ).hexdigest()
    assert payload["chunks_sha256"] == hashlib.sha256(
        paths["chunks"].read_bytes()
    ).hexdigest()
    assert payload["manifest_sha256"] == hashlib.sha256(
        paths["manifest"].read_bytes()
    ).hexdigest()
    assert payload["ontology_sha256"] == hashlib.sha256(
        paths["ontology"].read_bytes()
    ).hexdigest()
    assert payload["excluded_chunk_ids"] == [
        "c_0000000000000000",
        "c_0000000000000001",
        "c_0000000000000002",
        "c_0000000000000003",
    ]
    assert list(payload["question_chunk_map"]) == sorted(payload["question_chunk_map"])
    assert len(payload["question_chunk_map"]) == 45

    text = raw.decode("utf-8")
    for forbidden in ("Question 1", "Answer 1", "Alias 1", "Private note 1"):
        assert forbidden not in text
    assert raw.endswith(b"\n")


def test_repeated_freeze_is_byte_identical_and_does_not_rewrite(tmp_path):
    paths = _fixture(tmp_path)
    first_sha = _freeze(paths)
    first = paths["output"].read_bytes()
    old_mtime = paths["output"].stat().st_mtime_ns
    time.sleep(0.001)
    second_sha = _freeze(paths)
    assert second_sha == first_sha
    assert paths["output"].read_bytes() == first
    assert paths["output"].stat().st_mtime_ns == old_mtime


def test_frozen_artifact_refuses_different_source_bytes(tmp_path):
    paths = _fixture(tmp_path)
    _freeze(paths)
    rows = [
        json.loads(line)
        for line in paths["draft"].read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["question"] = "Changed but still schema-valid"
    _write_jsonl(paths["draft"], rows)
    with pytest.raises(leakage.LeakageError, match="different bytes are refused"):
        _freeze(paths)


def test_invalid_dataset_does_not_publish(tmp_path):
    paths = _fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["draft"].read_text(encoding="utf-8").splitlines()
    ][:-1]
    _write_jsonl(paths["draft"], rows)
    with pytest.raises(leakage.LeakageError, match="exactly 45"):
        _freeze(paths)
    assert not paths["output"].exists()


def test_unknown_evidence_chunk_does_not_publish(tmp_path):
    paths = _fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["draft"].read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["evidence_path"][0]["from"] = "c_ffffffffffffffff"
    _write_jsonl(paths["draft"], rows)
    with pytest.raises(leakage.LeakageError, match="unknown chunks"):
        _freeze(paths)
    assert not paths["output"].exists()


@pytest.mark.parametrize("input_name", ["chunks", "manifest", "ontology"])
def test_loader_rejects_stale_canonical_identity(tmp_path, input_name):
    paths = _fixture(tmp_path)
    _freeze(paths)
    paths[input_name].write_bytes(paths[input_name].read_bytes() + b"\n")
    with pytest.raises(leakage.LeakageError, match="stale leakage artifact"):
        leakage.load_frozen_exclusions(
            paths["output"],
            chunks_path=paths["chunks"],
            manifest_path=paths["manifest"],
            ontology_path=paths["ontology"],
        )


def test_loader_rejects_extra_fields(tmp_path):
    paths = _fixture(tmp_path)
    _freeze(paths)
    payload = json.loads(paths["output"].read_text(encoding="utf-8"))
    payload["question"] = "must never be accepted"
    paths["output"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(leakage.LeakageError, match="keys mismatch"):
        leakage.load_frozen_exclusions(
            paths["output"],
            chunks_path=paths["chunks"],
            manifest_path=paths["manifest"],
            ontology_path=paths["ontology"],
        )


def test_loader_returns_exact_frozen_union(tmp_path):
    paths = _fixture(tmp_path)
    _freeze(paths)
    assert leakage.load_frozen_exclusions(
        paths["output"],
        chunks_path=paths["chunks"],
        manifest_path=paths["manifest"],
        ontology_path=paths["ontology"],
    ) == {
        "c_0000000000000000",
        "c_0000000000000001",
        "c_0000000000000002",
        "c_0000000000000003",
    }


def test_cli_has_no_force_or_graph_escape_hatch():
    parser = leakage._build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--force" not in options
    assert "--overwrite" not in options
    assert "--skip-validation" not in options
    assert "--graph" not in options
