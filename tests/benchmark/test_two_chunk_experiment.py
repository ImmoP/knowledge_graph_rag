"""Consumer-contract tests for the controlled two-chunk recording experiment.

The real recording surface is deliberately exercised only through injected
``httpx.MockTransport`` clients.  No test here permits a network request.
"""

from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from src.benchmark.manifest_replay import replay_run

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDING_CLI = REPO_ROOT / "scripts" / "benchmark" / "manifest_recording.py"
RUNNER = REPO_ROOT / "scripts" / "experiments" / "two_chunk_20260930T120000Z" / "record_two_chunk.py"
DATASET_PATH = REPO_ROOT / "src" / "benchmark" / "dataset" / "benchmark_chunks.jsonl"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _experiment():
    from src.benchmark import two_chunk_experiment

    return two_chunk_experiment


def _canonical_selection() -> tuple[list[dict], dict]:
    experiment = _experiment()
    return experiment.load_recording_selection(DATASET_PATH, experiment.EXPERIMENT_CHUNK_IDS)


# ---------- selection -------------------------------------------------


def test_selection_default_uses_historical_smoke_chunks() -> None:
    experiment = _experiment()

    chunks, selection = experiment.load_recording_selection(DATASET_PATH)

    assert [row["chunk_id"] for row in chunks] == ["c_0001921855a8156b", "c_001452fd142ce075"]
    assert selection == {
        "source": "smoke_default",
        "dataset_sha256": "38182c6cdbb9f86abc6ce20ece7a3d9e1cb69ccac2576e2d6274e727c2f84985",
        "chunk_ids": ["c_0001921855a8156b", "c_001452fd142ce075"],
    }


def test_selection_preserves_explicit_experiment_order_when_reversed() -> None:
    experiment = _experiment()
    requested = tuple(reversed(experiment.EXPERIMENT_CHUNK_IDS))

    chunks, selection = experiment.load_recording_selection(DATASET_PATH, requested)

    assert [row["chunk_id"] for row in chunks] == list(requested)
    assert selection == {
        "source": "explicit_chunk_id_flag",
        "dataset_sha256": experiment.DATASET_SHA256,
        "chunk_ids": list(requested),
    }


@pytest.mark.parametrize(
    "requested",
    [
        ("c_ca129edd34052284",),
        ("c_ca129edd34052284", "c_8a84177f479aea1f", "c_ca129edd34052284"),
        ("c_ca129edd34052284", "c_ca129edd34052284"),
        ("c_ca129edd34052284", "c_not_an_experiment_chunk"),
    ],
)
def test_selection_rejects_noncanonical_explicit_ids(requested: tuple[str, ...]) -> None:
    experiment = _experiment()

    with pytest.raises(ValueError):
        experiment.load_recording_selection(DATASET_PATH, requested)


def test_selection_rejects_tampered_alternate_dataset(tmp_path: Path) -> None:
    experiment = _experiment()
    alternate = tmp_path / "alternate.jsonl"
    alternate.write_bytes(DATASET_PATH.read_bytes() + b"\n")

    with pytest.raises(RuntimeError):
        experiment.load_recording_selection(alternate, experiment.EXPERIMENT_CHUNK_IDS)


def test_selection_validation_rejects_modified_frozen_row() -> None:
    experiment = _experiment()
    chunks, selection = _canonical_selection()
    altered = [dict(row) for row in chunks]
    altered[0]["text"] = "altered after selection"

    with pytest.raises(ValueError):
        experiment.validate_selection(altered, selection)

# ---------- recording CLI ---------------------------------------------


@pytest.mark.parametrize(
    "chunk_ids",
    [
        ["c_ca129edd34052284"],
        ["c_ca129edd34052284", "c_8a84177f479aea1f", "c_ca129edd34052284"],
    ],
)
def test_recording_cli_rejects_chunk_id_count_other_than_two(chunk_ids: list[str]) -> None:
    cli = _load_module(RECORDING_CLI, f"manifest_recording_count_{len(chunk_ids)}")
    argv = ["record", "--live", "--output-dir", "/tmp/recording", "--allow-manifest-calls", "4"]
    for chunk_id in chunk_ids:
        argv.extend(["--chunk-id", chunk_id])

    assert cli.main(argv) == 1


def test_recording_cli_rejects_foreign_exact_pair() -> None:
    cli = _load_module(RECORDING_CLI, "manifest_recording_foreign_pair")

    assert cli.main([
        "record", "--live", "--output-dir", "/tmp/recording", "--allow-manifest-calls", "4",
        "--chunk-id", "c_ca129edd34052284", "--chunk-id", "c_not_an_experiment_chunk",
    ]) == 1


def test_recording_cli_embeds_verified_selection_and_persists_real_recording(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = _load_module(RECORDING_CLI, "manifest_recording_real_selection")
    original = cli.record_run
    answers = iter([{"mentions": []}, {"assertions": []}, {"mentions": []}, {"assertions": []}])

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "fixture/model", "choices": [{"message": {"content": json.dumps(next(answers))}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    def record_with_mock(**kwargs: object) -> dict:
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            return original(**kwargs, api_key="fixture-only", client=client)

    monkeypatch.setattr(cli, "record_run", record_with_mock)
    output_dir = tmp_path / "recording"
    assert cli.main([
        "record", "--live", "--output-dir", str(output_dir), "--allow-manifest-calls", "4",
        "--chunk-id", "c_ca129edd34052284", "--chunk-id", "c_8a84177f479aea1f",
    ]) == 0
    manifest = json.loads((output_dir / "recording.json").read_text())
    assert manifest["selection"] == _canonical_selection()[1]
    assert manifest["raw_responses"]["attempt_count"] == 4
    assert not (output_dir / "selection.json").exists()
    replay_dir = tmp_path / "replay"
    assert replay_run(recording_dir=output_dir, output_dir=replay_dir)["status"] == "matched"


# ---------- bounded runner --------------------------------------------


def test_runner_preflight_is_offline_and_writes_only_private_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_module(RUNNER, "two_chunk_preflight")
    monkeypatch.delenv("MANIFEST_API_KEY", raising=False)
    def no_client(*_: object, **__: object) -> None:
        raise AssertionError("offline preflight must not construct an HTTP client")

    monkeypatch.setattr(httpx, "Client", no_client)

    def no_recording(**_: object) -> None:
        raise AssertionError("offline preflight must not create a recording client")

    monkeypatch.setattr(runner, "record_run", no_recording)
    root = tmp_path / "recordings"
    assert runner.main(["--recordings-root", str(root), "--run-id", "offline-preflight"]) == 0

    run_dir = root / "offline-preflight"
    preflight = run_dir / "preflight.json"
    assert preflight.is_file()
    assert not (run_dir / "recording.json").exists()
    assert stat.S_IMODE(run_dir.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(preflight.stat().st_mode) & 0o077 == 0
    assert not (run_dir / "selection.json").exists()
    assert not (run_dir / "run_summary.json").exists()
    payload = json.loads(preflight.read_text())
    assert payload["selection"]["chunk_ids"] == list(_experiment().EXPERIMENT_CHUNK_IDS)


@pytest.mark.parametrize("run_id", ["../escape", "nested/name", ".", ""])
def test_runner_rejects_non_single_component_run_id(run_id: str, tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--recordings-root", str(tmp_path), "--run-id", run_id],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("calls", [0, 5])
def test_runner_rejects_call_budget_outside_hard_allowlist(calls: int, tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable, str(RUNNER), "--recordings-root", str(tmp_path), "--run-id", "budget",
            "--allow-manifest-calls", str(calls),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert not any(tmp_path.iterdir())


def test_runner_refuses_existing_or_symlink_run_directory(tmp_path: Path) -> None:
    root = tmp_path / "recordings"
    root.mkdir()
    existing = root / "occupied"
    existing.mkdir()
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--recordings-root", str(root), "--run-id", "occupied"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1

    link = root / "linked"
    link.symlink_to(existing, target_is_directory=True)
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--recordings-root", str(root), "--run-id", "linked"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1


def test_runner_rejects_recording_root_inside_repository(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--recordings-root", str(REPO_ROOT), "--run-id", "outside-only"],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 1
    assert not (REPO_ROOT / "outside-only").exists()


def test_runner_live_writes_selection_with_mocked_http(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_module(RUNNER, "two_chunk_live_runner")
    original = runner.record_run
    answers = iter([{"mentions": []}, {"assertions": []}, {"mentions": []}, {"assertions": []}])

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "fixture/model", "choices": [{"message": {"content": json.dumps(next(answers))}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    def record_with_mock(**kwargs: object) -> dict:
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            return original(**kwargs, api_key="fixture-only", client=client)

    monkeypatch.setattr(runner, "record_run", record_with_mock)
    monkeypatch.setenv("MANIFEST_API_KEY", "fixture-only")
    root = tmp_path / "recordings"
    assert runner.main([
        "--live", "--recordings-root", str(root), "--run-id", "live-run", "--allow-manifest-calls", "4",
    ]) == 0
    manifest = json.loads((root / "live-run" / "recording.json").read_text())
    assert manifest["selection"] == _canonical_selection()[1]
    assert manifest["raw_responses"]["attempt_count"] == 4


def test_runner_checks_frozen_dataset_before_live_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _load_module(RUNNER, "two_chunk_dataset_before_output")
    monkeypatch.setenv("MANIFEST_API_KEY", "test-only-key")
    broken = tmp_path / "broken.jsonl"
    broken.write_bytes(DATASET_PATH.read_bytes() + b"\n")
    root = tmp_path / "recordings"

    assert runner.main([
        "--live", "--dataset", str(broken), "--recordings-root", str(root), "--run-id", "drift",
    ]) == 1
    assert not root.exists()


def test_cli_preserves_smoke_replay_but_refuses_problem_comparison(tmp_path: Path) -> None:
    cli = _load_module(RECORDING_CLI, "manifest_smoke_replay_guard")
    chunks, selection = _experiment().load_recording_selection(DATASET_PATH)
    responses = iter([{"mentions": []}, {"assertions": []}, {"mentions": []}, {"assertions": []}])

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "model": "technical-fixture",
            "choices": [{"message": {"content": json.dumps(next(responses))}}],
        })

    recording = tmp_path / "smoke-recording"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        cli.record_run(
            output_dir=recording, chunks=chunks, selection=selection,
            ontology_path=REPO_ROOT / "ontology.yaml", prompts_dir=REPO_ROOT / "prompts",
            api_key="technical-fixture-key", client=client,
        )
    assert cli.main([
        "replay", "--recording-dir", str(recording), "--output-dir", str(tmp_path / "replay"),
    ]) == 0
    refused = tmp_path / "refused-comparison"
    assert cli.main([
        "replay", "--recording-dir", str(recording), "--output-dir", str(refused),
        "--compare-whitespace",
    ]) == 1
    assert not refused.exists()
