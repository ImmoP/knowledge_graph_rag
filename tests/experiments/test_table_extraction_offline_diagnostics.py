from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "tests/fixtures/table_extraction_v2_recordings.json"
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts/experiments/table_extraction_offline"
    / "diagnose_recordings.py"
)


def _diagnoser():
    spec = importlib.util.spec_from_file_location(
        "table_extraction_offline_diagnostics", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_diagnosis_reproduces_recorded_empty_overlap_and_canonical_findings() -> None:
    report = _diagnoser().diagnose(_fixture())

    empty_by_id = {
        observation["recording_id"]: observation
        for observation in report["successful_empty_observations"]
    }
    assert set(empty_by_id) == {
        "control-01",
        "control-02",
        "control-03",
        "control-04",
        "control-05",
        "historical-target-empty",
    }
    assert empty_by_id["control-01"]["request_identity"] == "byte_verified"
    assert empty_by_id["historical-target-empty"]["request_identity"] == "not_recorded"
    assert all(item["classification"] == "technical_observation" for item in empty_by_id.values())
    assert all(item["domain_conclusion"] is None for item in empty_by_id.values())

    overlap = next(
        group
        for group in report["overlap_groups"]
        if group["outer_span"] == [210, 232]
        and group["inner_span"] == [210, 214]
    )
    assert overlap["relation"] == "contains"
    assert overlap["recording_ids"] == [
        "target-01",
        "target-02",
        "target-03",
        "target-04",
        "target-05",
    ]
    assert overlap["classification"] == "technical_observation"
    assert overlap["domain_conclusion"] is None

    assert report["canonical_conflicts"] == [
        {
            "entity_type": "method",
            "surface_form": "Bi-DAF",
            "span": [226, 232],
            "canonical_forms": ["Bi-DAF", "BiDAF"],
            "observations": [
                {"recording_id": "target-03", "canonical_form": "Bi-DAF"},
                {"recording_id": "target-04", "canonical_form": "BiDAF"},
            ],
            "classification": "technical_anomaly",
            "domain_conclusion": None,
        }
    ]
    assert {item["question_id"]: item for item in report["domain_questions"]} == {
        item["question_id"]: item for item in _fixture()["domain_questions"]
    }

# The CLI diagnoser above is the offline-only surface for frozen
# historical recordings whose original raw responses are no longer
# available (see docs/experiments/table-extraction-offline-strategy.md §6.1).
# The replay-driven `src/benchmark/offline_diagnostics.py` only consumes
# recordings whose attempt stream has been SHA-verified against the
# recorded manifest, so the `not_recorded` historical empty cannot reach
# it. Both surfaces are tested independently; the canonical replay
# shadow is covered by tests/benchmark/test_offline_diagnostics.py.
    fixture = _fixture()
    reversed_fixture = dict(fixture)
    reversed_fixture["table_recordings"] = [
        {**recording, "mentions": list(reversed(recording["mentions"]))}
        for recording in reversed(fixture["table_recordings"])
    ]
    reversed_fixture["successful_empty_recordings"] = list(
        reversed(fixture["successful_empty_recordings"])
    )

    diagnose = _diagnoser().diagnose
    assert diagnose(reversed_fixture) == diagnose(fixture)




def test_cli_writes_byte_stable_report_without_external_recording_paths(tmp_path: Path) -> None:
    diagnostic = _diagnoser()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    assert diagnostic.main(["--fixture", str(FIXTURE_PATH), "--output", str(first)]) == 0
    assert diagnostic.main(["--fixture", str(FIXTURE_PATH), "--output", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()

    report = json.loads(first.read_text(encoding="utf-8"))
    assert report["format"] == "table-extraction-offline-diagnostics-v1"
    assert "source_artifact" not in first.read_text(encoding="utf-8")
