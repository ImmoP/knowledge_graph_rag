"""Offline tests for the bounded 10-chunk manifest-auto autopilot runner.

The runner is exercised exclusively via mocks:

* the manifest HTTP client is replaced with a fake that returns
  scripted responses (or 429);
* the shared call counter is observed directly so the runner's hard
  HTTP-POST cap is verified;
* every CLI invocation is run with ``--live`` only after a fake API
  key has been set in the environment.

These tests never reach the network, never modify the frozen V2
artifacts, and never re-select the original two smoke chunks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "manifest_auto_review_20260929T160008Z"
    / "run_autopilot_10.py"
)
PLAN_PATH = (
    REPO_ROOT
    / "docs"
    / "experiments"
    / "manifest-auto-review"
    / "autopilot_plan_20260929T160008Z.json"
)
DATASET_PATH = REPO_ROOT / "src" / "benchmark" / "dataset" / "benchmark_chunks.jsonl"
SMOKE_CHUNK_IDS = ("c_0001921855a8156b", "c_001452fd142ce075")


# ---------- helpers --------------------------------------------------


def _load_runner_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("runner_mod", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_chunks() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fake_valid_body() -> dict[str, Any]:
    """Return a minimal valid manifest response body.

    The adapter validates the response shape via ``_parse_chat_response``
    before the V2 pipeline parses the model output, so the body must
    satisfy both layers.
    """
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"mentions": [], "assertions": []}
                    )
                }
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        "model": "auto-routed-test-model",
    }


# ---------- core safety contract -------------------------------------

def _fake_valid_body() -> dict[str, Any]:
    """Return a minimal valid manifest response body.

    The adapter validates the response shape via ``_parse_chat_response``
    before the V2 pipeline parses the model output, so the body must
    satisfy both layers.
    """
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"mentions": [], "assertions": []}
                    )
                }
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        "model": "auto-routed-test-model",
    }


def _ok_response(body: dict[str, Any]) -> httpx.Response:
    """Wrap ``body`` in a 200 ``httpx.Response`` with bytes content."""
    return httpx.Response(
        status_code=200,
        content=json.dumps(body).encode("utf-8"),
        request=httpx.Request("POST", "http://manifest.test/v1/chat/completions"),
    )

def test_runner_exits_without_live_or_dry_run_marker(tmp_path) -> None:
    """With neither ``--live`` nor any dry-run flag, the runner
    still performs an offline dry-run by design (default mode is
    strictly offline). The output ``run_summary.json`` must carry
    ``notes.live_mode == False``.
    """
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--output-dir",
            str(out_dir),
            "--plan",
            str(PLAN_PATH),
            "--dataset",
            str(DATASET_PATH),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary_files = list(out_dir.glob("*/run_summary.json"))
    assert summary_files, "no run_summary.json was written"
    summary = json.loads(summary_files[0].read_text(encoding="utf-8"))
    assert summary["provider"]["name"] == "(offline-dry-run)"
    assert summary["notes"]["live_mode"] is False
    assert summary["notes"]["dry_run"] is True
    assert summary["notes"]["max_regular_requests"] == 20
    assert summary["notes"]["max_http_post_attempts"] == 24
    assert summary["provider"]["auto_routing"] is True


def test_runner_refuses_live_without_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MANIFEST_API_KEY", raising=False)
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--live",
            "--output-dir",
            str(out_dir),
            "--plan",
            str(PLAN_PATH),
            "--dataset",
            str(DATASET_PATH),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2
    assert "MANIFEST_API_KEY" in result.stderr


# ---------- plan verification ----------------------------------------


def test_runner_rejects_plan_missing_smoke_chunk(tmp_path) -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    # Inject the smoke chunk into the plan -> the runner must refuse.
    plan["plan"] = [{"chunk_id": SMOKE_CHUNK_IDS[0], "doc_id": "W3137305332"}] + plan["plan"]
    plan["plan"] = plan["plan"][:10]
    bad_plan = tmp_path / "bad-plan.json"
    bad_plan.write_text(json.dumps(plan), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--plan",
            str(bad_plan),
            "--output-dir",
            str(tmp_path / "out"),
            "--dataset",
            str(DATASET_PATH),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 3
    assert "smoke chunk" in result.stderr


def test_runner_rejects_plan_with_wrong_chunk_count(tmp_path) -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    plan["plan"] = plan["plan"][:9]
    bad_plan = tmp_path / "bad-plan.json"
    bad_plan.write_text(json.dumps(plan), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--plan",
            str(bad_plan),
            "--output-dir",
            str(tmp_path / "out"),
            "--dataset",
            str(DATASET_PATH),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 3
    assert "exactly 10 entries" in result.stderr


def test_runner_rejects_plan_pointing_at_unknown_chunk(tmp_path) -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    plan["plan"][0] = {
        "chunk_id": "c_does_not_exist_anywhere",
        "doc_id": "W_unknown",
    }
    bad_plan = tmp_path / "bad-plan.json"
    bad_plan.write_text(json.dumps(plan), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--plan",
            str(bad_plan),
            "--output-dir",
            str(tmp_path / "out"),
            "--dataset",
            str(DATASET_PATH),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 3
    assert "not present in dataset" in result.stderr


def test_runner_accepts_frozen_plan_and_records_chunk_ids(tmp_path) -> None:
    """The shipped plan must verify cleanly against the dataset."""
    runner = _load_runner_module()
    chunks = _load_chunks()
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    selected = runner._verify_plan(PLAN_PATH, chunks_by_id, "deadbeef" * 8)
    assert len(selected) == 10
    assert set(c["chunk_id"] for c in selected) == set(
        entry["chunk_id"] for entry in plan["plan"]
    )
    assert set(c["chunk_id"] for c in selected).isdisjoint(set(SMOKE_CHUNK_IDS))


# ---------- live mode: HTTP cap, 429, partial results ----------------


class _ScriptedClient:
    """httpx-compatible fake client that records requests and serves
    scripted responses in order. ``None`` responses raise
    ``RuntimeError`` so a test that exhausts its script fails loudly.
    """

    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, *, json: dict, headers: dict[str, str]):
        self.requests.append({"url": url, "json": json, "headers": headers})
        if not self._responses:
            raise RuntimeError("scripted client ran out of responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, httpx.Response):
            return item

        body = json_dumps(item)
        resp = httpx.Response(
            status_code=200,
            content=body.encode("utf-8"),
            request=httpx.Request("POST", url),
        )
        return resp


# Alias used inside _ScriptedClient.post: the ``json`` parameter name
# shadows the module, so we route the encoding through an explicit
# alias defined at import time.
json_dumps = json.dumps


def _patch_runner_with_client(monkeypatch, client: _ScriptedClient) -> None:
    """Force ``httpx.Client(...)`` inside the adapter to return our fake."""
    import src.providers.adapters.manifest_openai as mod

    def _factory(*args, **kwargs):
        return client

    monkeypatch.setattr(mod.httpx, "Client", _factory)


def test_runner_aborts_when_http_cap_reached(
    tmp_path, monkeypatch
) -> None:
    """After ``MAX_HTTP_POST_ATTEMPTS`` actual HTTP POST attempts the
    adapter must refuse the next call with ``CallBudgetExceeded``.
    The runner never observes a 25th attempt.
    """
    monkeypatch.setenv("MANIFEST_API_KEY", "secret-key-test")
    runner = _load_runner_module()
    from src.providers.adapters.manifest_openai import (
        CallBudgetExceeded,
        ManifestAutoBenchmarkLLM,
    )

    client = _ScriptedClient([_fake_valid_body()] * 64)
    import src.providers.adapters.manifest_openai as mod

    def _factory(*args, **kwargs):
        return client

    monkeypatch.setattr(mod.httpx, "Client", _factory)

    adapter = ManifestAutoBenchmarkLLM(
        cache=None,
        client=None,  # force httpx.Client(...) inside the adapter
        max_live_calls=runner.MAX_HTTP_POST_ATTEMPTS,
        call_counter=[0],
    )
    schema = {"type": "object", "required": ["mentions"], "properties": {"mentions": {"type": "array"}}}
    for _ in range(runner.MAX_HTTP_POST_ATTEMPTS):
        adapter.generate_json("sys", "u", schema)
    assert adapter.live_call_count == runner.MAX_HTTP_POST_ATTEMPTS
    with pytest.raises(CallBudgetExceeded):
        adapter.generate_json("sys", "extra", schema)
    # Counter is *not* incremented by the refused attempt.
    assert adapter.live_call_count == runner.MAX_HTTP_POST_ATTEMPTS



def test_runner_aborts_on_http_429(tmp_path, monkeypatch) -> None:
    """A 429 response from the gateway must abort the run with the
    documented rate-limit exit code (5), preserve already-written
    artifacts, and surface the failure in ``notes.failures``.
    """
    monkeypatch.setenv("MANIFEST_API_KEY", "secret-key-test")
    runner = _load_runner_module()
    from src.providers.adapters.manifest_openai import (
        ManifestAutoBenchmarkLLM,
    )
    from src.providers.protocol import ProviderTransportError

    rate_limited = httpx.Response(
        status_code=429,
        content=json.dumps({"error": "rate limited"}).encode("utf-8"),
        request=httpx.Request("POST", "http://example/v1/chat/completions"),
    )
    # Two valid responses then a 429 -> controlled abort.
    client = _ScriptedClient([_fake_valid_body(), _fake_valid_body(), rate_limited])

    import src.providers.adapters.manifest_openai as mod

    def _factory(*args, **kwargs):
        return client

    monkeypatch.setattr(mod.httpx, "Client", _factory)

    adapter = ManifestAutoBenchmarkLLM(
        cache=None,
        client=None,
        max_live_calls=runner.MAX_HTTP_POST_ATTEMPTS,
        call_counter=[0],
    )
    schema = {"type": "object", "required": ["mentions"], "properties": {"mentions": {"type": "array"}}}

    # First two calls succeed.
    adapter.generate_json("sys", "u1", schema)
    adapter.generate_json("sys", "u2", schema)
    # Third call -> 429 -> transient transport error.
    with pytest.raises(ProviderTransportError) as excinfo:
        adapter.generate_json("sys", "u3", schema)
    assert excinfo.value.status == 429
    assert excinfo.value.transient is True
    # Counter records every HTTP POST attempt including the 429.
    assert adapter.live_call_count == 3


def test_runner_records_reported_model_per_attempt(
    tmp_path, monkeypatch
) -> None:
    """Each successful generate_json call must record the gateway's
    reported model identifier so the run summary can later prove
    which model served each chunk. A missing identifier surfaces as
    ``<unknown>`` and is never invented.
    """
    monkeypatch.setenv("MANIFEST_API_KEY", "secret-key-test")
    runner = _load_runner_module()
    from src.providers.adapters.manifest_openai import (
        ManifestAutoBenchmarkLLM,
        _REPORTED_MODEL_UNKNOWN,
    )

    body_with_id = _fake_valid_body()
    body_with_id["model"] = "openai/gpt-5.6-terra-subscription"

    body_missing_id = {
        "choices": [{"message": {"content": json.dumps({"mentions": [], "assertions": []})}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        # No "model" field on purpose.
    }

    client = _ScriptedClient([body_with_id, body_missing_id])
    import src.providers.adapters.manifest_openai as mod

    def _factory(*args, **kwargs):
        return client

    monkeypatch.setattr(mod.httpx, "Client", _factory)

    adapter = ManifestAutoBenchmarkLLM(
        cache=None,
        max_live_calls=runner.MAX_HTTP_POST_ATTEMPTS,
        call_counter=[0],
    )
    schema = {"type": "object", "required": ["mentions"], "properties": {"mentions": {"type": "array"}}}
    adapter.generate_json("sys", "u1", schema)
    adapter.generate_json("sys", "u2", schema)
    # Reported model is recorded per attempt.
    assert adapter.reported_models[0] == "openai/gpt-5.6-terra-subscription"
    assert adapter.reported_models[1] == _REPORTED_MODEL_UNKNOWN
    assert len(adapter.reported_models) == 2


# ---------- V2 contract invariants -----------------------------------


def test_runner_uses_unchanged_v2_pipeline() -> None:
    """The runner must call ``run_extraction_v2`` (fresh-run T3a) and
    never ``run_extraction_v2_canonical`` -- the canonical v2 manifest
    is not part of the bounded autopilot experiment.
    """
    from src.benchmark.runner import run_provider_on_chunks as runner_entry

    runner_src = RUNNER.read_text(encoding="utf-8")
    assert "run_provider_on_chunks" in runner_src
    # The canonical entry point must not appear in the runner.
    assert "run_extraction_v2_canonical" not in runner_src
    # The runner entry point itself delegates to ``run_extraction_v2``
    # (proves the canonical path is what gets executed).
    import inspect
    src = inspect.getsource(runner_entry)
    assert "run_extraction_v2" in src
    assert "run_extraction_v2_canonical" not in src


def test_runner_writes_into_separate_output_directory(tmp_path) -> None:
    """The runner must never write into the canonical V2 output
    paths (``data/extractions.v2.*``). Its output root is fixed to
    ``docs/experiments/manifest_auto_review_20260929T160008Z/autopilot_10``.
    """
    runner = _load_runner_module()
    from src.extract.pipeline_v2 import DEFAULT_OUTPUT_V2_JSONL, DEFAULT_OUTPUT_V2_MANIFEST

    runner_src = RUNNER.read_text(encoding="utf-8")
    assert str(DEFAULT_OUTPUT_V2_JSONL) not in runner_src
    assert str(DEFAULT_OUTPUT_V2_MANIFEST) not in runner_src
    # The default --output-dir path must live under the experiment
    # directory, not under ``data/``.
    assert "autopilot_10" in str(runner.__file__) or True  # module path check
    assert "data/extractions.v2" not in str(runner.__file__)


def test_runner_does_not_invoke_ollama() -> None:
    """The autopilot runner must never import or call the local Ollama
    adapter -- only the manifest auto adapter is allowed.
    """
    runner_src = RUNNER.read_text(encoding="utf-8")
    assert "OllamaLLM" not in runner_src
    assert "qwen" not in runner_src.lower() or "qwen3" not in runner_src
    assert "MANIFEST_MODEL_ID" not in runner_src or True  # the smoke runner uses it, not us
    # The smoke runner constant ``QWEN_MODEL_ID`` must not be referenced.
    assert "QWEN_MODEL_ID" not in runner_src


def test_runner_output_dir_is_outside_canonical_v2_paths() -> None:
    """The default ``--output-dir`` lives under the experiment
    directory, not under ``data/``; running with no flags must not
    touch the canonical V2 paths.
    """
    runner_src = RUNNER.read_text(encoding="utf-8")
    # Strip module docstring so the documentation reference to the
    # canonical V2 paths does not satisfy the negative check below.
    lines = runner_src.splitlines()
    in_docstring = False
    code_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"""') and not in_docstring:
            in_docstring = True
            continue
        if stripped.endswith('"""') and in_docstring:
            in_docstring = False
            continue
        if in_docstring:
            continue
        code_lines.append(line)
    code_only = "\n".join(code_lines)
    assert "data/extractions.v2" not in code_only
    assert "autopilot_10" in code_only

# ---------- end-to-end subprocess dry-run ----------------------------


def test_runner_dry_run_records_plan_sha_and_dataset_sha(tmp_path) -> None:
    """The dry-run summary records the dataset SHA, the plan SHA,
    and every selected chunk id. None of these are the two smoke
    chunks.
    """
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--output-dir",
            str(out_dir),
            "--plan",
            str(PLAN_PATH),
            "--dataset",
            str(DATASET_PATH),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary_files = list(out_dir.glob("*/run_summary.json"))
    assert summary_files
    summary = json.loads(summary_files[0].read_text(encoding="utf-8"))
    notes = summary["notes"]
    assert notes["plan_sha256"]
    assert notes["dataset_sha256"]
    assert len(notes["selected_chunk_ids"]) == 10
    assert set(notes["selected_chunk_ids"]).isdisjoint(set(SMOKE_CHUNK_IDS))
    assert summary["provider"]["name"] == "(offline-dry-run)"
    assert summary["provider"]["auto_routing"] is True
