"""Tests for the benchmark runner and the smoke-runner script.

These tests run with stub adapters and never touch the network. They
exercise the V2 pipeline invocation, per-phase measurement recording,
and the smoke run's offline-only paths.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from src.benchmark.dataset.manifest import (  # noqa: E402
    CHUNKS_PATH,
)
from src.benchmark.dataset.smoke_chunks import SMOKE_CHUNK_IDS  # noqa: E402
from src.benchmark.runner import (  # noqa: E402
    RunPaths,
    TimingProvider,
    run_provider_on_chunks,
    new_run_paths,
)


ONTOLOGY_PATH = Path(__file__).resolve().parents[2] / "ontology.yaml"
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


class _StubProvider:
    """Minimal LLMProvider that returns a canned entity/relation
    response for every call. Satisfies the protocol by structural
    typing."""

    def __init__(self) -> None:
        self.name = "stub"
        self.model = "stub-model"
        self.call_log: list[tuple[str, str]] = []

    def generate(self, system: str, user: str, *, max_tokens: int = 1024, temperature: float = 0.0):
        self.call_log.append(("generate", user[:60]))
        return "stub-text", _usage(0, 0)

    def generate_json(self, system: str, user: str, schema: dict, *, max_tokens: int = 1024, temperature: float = 0.0):
        self.call_log.append(("generate_json", user[:60]))
        if "mentions" in schema.get("properties", {}):
            return {"mentions": []}, _usage(5, 4)
        return {"assertions": []}, _usage(7, 3)


def _usage(inp: int, out: int):
    from src.providers.protocol import Usage
    return Usage(input_tokens=inp, output_tokens=out, cost_usd=None)


def test_run_paths_for_run_id_creates_expected_layout(tmp_path: Path):
    paths = RunPaths.for_run_id(tmp_path, "test-run")
    assert paths.run_dir == tmp_path / "test-run"
    assert paths.output_jsonl == tmp_path / "test-run" / "extractions.jsonl"
    assert paths.output_rejects == tmp_path / "test-run" / "extractions.rejects.jsonl"
    assert paths.metrics_path == tmp_path / "test-run" / "metrics.jsonl"
    assert paths.raw_responses_dir == tmp_path / "test-run" / "raw_responses"


def test_timing_provider_records_metrics():
    inner = _StubProvider()
    sink: list = []
    tp = TimingProvider(inner, sink=sink, model_label="stub")
    text, usage = tp.generate("sys", "user", max_tokens=10)
    assert text == "stub-text"
    assert usage.input_tokens == 0
    assert len(sink) == 1
    assert sink[0].cache_hit is False
    assert sink[0].duration_seconds >= 0


def test_timing_provider_records_metrics_for_generate_json():
    inner = _StubProvider()
    sink: list = []
    tp = TimingProvider(inner, sink=sink, model_label="stub")
    schema = {"type": "object", "properties": {"mentions": {"type": "array"}}}
    parsed, usage = tp.generate_json("sys", "user", schema)
    assert "mentions" in parsed
    assert len(sink) == 1
    assert sink[0].input_tokens == 5
    assert sink[0].output_tokens == 4


def test_timing_provider_satisfies_llm_provider_protocol():
    inner = _StubProvider()
    tp = TimingProvider(inner, sink=[], model_label="stub")
    from src.providers.protocol import LLMProvider
    assert isinstance(tp, LLMProvider)
    assert tp.name == "stub"
    assert tp.model == "stub-model"


def test_run_provider_on_chunks_records_sink_and_writes_outputs(tmp_path: Path):
    from src.benchmark.dataset.manifest import load_frozen_chunks
    if not CHUNKS_PATH.is_file():
        pytest.skip("frozen dataset not present (run build_dataset.py first)")
    chunks, _ = load_frozen_chunks()
    smoke_chunks = [c for c in chunks if c["chunk_id"] in SMOKE_CHUNK_IDS]
    paths = new_run_paths(run_root=tmp_path)
    inner = _StubProvider()
    result, sink = run_provider_on_chunks(
        provider=inner,
        chunks_path=CHUNKS_PATH,
        chunks=smoke_chunks,
        ontology_path=ONTOLOGY_PATH,
        prompts_dir=PROMPTS_DIR,
        paths=paths,
    )
    assert len(smoke_chunks) == 2
    # Two chunks × two phases = 4 calls (entity + relation) in the
    # canonical V2 fresh-run semantics.
    assert len(sink) == 4
    assert {s.chunk_id for s in sink} == set(SMOKE_CHUNK_IDS)
    assert {s.phase for s in sink} == {"entity", "relation"}
    # Records are written.
    assert paths.output_jsonl.is_file()
    assert paths.output_rejects.is_file()
    assert paths.metrics_path.is_file()
    # Metrics file is one JSON object per recorded call.
    metrics_lines = [
        json.loads(line) for line in paths.metrics_path.read_text().splitlines() if line.strip()
    ]
    assert len(metrics_lines) == 4


def test_run_provider_on_chunks_preserves_canonical_v2_validation(tmp_path: Path):
    """The runner drives ``run_extraction_v2`` directly, so the
    canonical V2 records (mentions / assertions / chunk_status)
    must be present in the output JSONL."""
    from src.benchmark.dataset.manifest import load_frozen_chunks
    if not CHUNKS_PATH.is_file():
        pytest.skip("frozen dataset not present (run build_dataset.py first)")
    chunks, _ = load_frozen_chunks()
    smoke_chunks = [c for c in chunks if c["chunk_id"] in SMOKE_CHUNK_IDS]
    paths = new_run_paths(run_root=tmp_path)
    inner = _StubProvider()
    result, _ = run_provider_on_chunks(
        provider=inner,
        chunks_path=CHUNKS_PATH,
        chunks=smoke_chunks,
        ontology_path=ONTOLOGY_PATH,
        prompts_dir=PROMPTS_DIR,
        paths=paths,
    )
    kinds = []
    for line in paths.output_jsonl.read_text().splitlines():
        if line.strip():
            kinds.append(json.loads(line).get("kind"))
    # The stub returns empty lists for both phases so no mention /
    # assertion rows are emitted. chunk_status rows must appear once per
    # chunk.
    assert kinds.count("chunk_status") == 2


def test_smoke_chunk_ids_pin_both_expected_chunks():
    assert SMOKE_CHUNK_IDS == ("c_0001921855a8156b", "c_001452fd142ce075")


def test_smoke_chunk_ids_are_first_in_frozen_dataset():
    from src.benchmark.dataset.manifest import load_frozen_chunks
    if not CHUNKS_PATH.is_file():
        pytest.skip("frozen dataset not present")
    chunks, _ = load_frozen_chunks()
    head = [c["chunk_id"] for c in chunks[:2]]
    assert head == list(SMOKE_CHUNK_IDS)


def test_smoke_script_module_loads():
    """The smoke-runner script must import cleanly. It defers
    adapter construction so a missing Ollama or gateway does not
    affect module loading."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_smoke.py"
    spec = importlib.util.spec_from_file_location("run_smoke", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert hasattr(module, "main")


def test_smoke_script_dry_run_emits_summary(tmp_path: Path):
    """End-to-end smoke test that exercises the CLI dry-run path
    and verifies the summary file is written with the two pinned
    chunk ids and that the local-baseline-cache check correctly
    reports ``comparable=False`` when no baseline cache exists."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_smoke.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    summaries = list(tmp_path.glob("*/run_summary.json"))
    assert summaries, f"no summary written; stdout={proc.stdout!r}"
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert payload["chunks"] == list(SMOKE_CHUNK_IDS)
    assert payload["notes"]["local_baseline_cache"]["comparable"] is False

class _AutoFakeResponse:
    def __init__(self, body: dict) -> None:
        self._body = body
        self.status_code = 200

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        return None


class _AutoFakeClient:
    def __init__(self, bodies: list[dict]) -> None:
        self._bodies = list(bodies)
        self.requests: list[dict] = []
        self.timeout = 60.0

    def post(self, url: str, json: dict | None = None, headers: dict | None = None):
        idx = len(self.requests)
        self.requests.append({"url": url, "json": json, "headers": headers or {}})
        return _AutoFakeResponse(self._bodies[idx])


def _auto_body(content: str, reported_model: str | None) -> dict:
    body = {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    if reported_model is not None:
        body["model"] = reported_model
    return body


def test_timing_provider_records_auto_reported_model_per_phase():
    class _AutoStub(_StubProvider):
        def __init__(self) -> None:
            super().__init__()
            self.model = "auto"
            self.reported_models: list[str] = []
            self._reports = iter(["provider/model-a", "provider/model-b"])

        def generate_json(self, system, user, schema, *, max_tokens=1024, temperature=0.0):
            self.reported_models.append(next(self._reports))
            return super().generate_json(
                system, user, schema, max_tokens=max_tokens, temperature=temperature
            )

    inner = _AutoStub()
    sink: list = []
    timed = TimingProvider(inner, sink=sink, model_label="manifest-auto")
    entity_schema = {"type": "object", "properties": {"mentions": {"type": "array"}}}
    relation_schema = {"type": "object", "properties": {"assertions": {"type": "array"}}}

    timed.generate_json("sys", "entity", entity_schema)
    timed.generate_json("sys", "relation", relation_schema)

    assert [call.requested_model for call in sink] == ["auto", "auto"]
    assert [call.reported_model for call in sink] == [
        "provider/model-a",
        "provider/model-b",
    ]
    assert inner.reported_models == []


def test_manifest_auto_cli_saves_partial_report_at_four_post_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Offline end-to-end test: one JSON repair consumes an extra POST.

    The fifth phase attempt is refused before HTTP. The CLI returns 3,
    saves a partial summary, attributes each successful phase to only
    gateway-reported model ids, and marks the refused attempt unknown.
    """
    script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_smoke.py"
    spec = importlib.util.spec_from_file_location("run_smoke_auto_limit", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    from src.providers.adapters.manifest_openai import ManifestAutoBenchmarkLLM

    # POST 1 invalid JSON; POST 2 repair succeeds. Then relation and
    # second-chunk entity consume POSTs 3 and 4. Second-chunk relation
    # is refused before POST 5.
    client = _AutoFakeClient([
        _auto_body("not-json", "provider/model-a"),
        _auto_body(json.dumps({"mentions": []}), "provider/model-b"),
        _auto_body(json.dumps({"assertions": []}), "provider/model-c"),
        _auto_body(json.dumps({"mentions": []}), "provider/model-d"),
    ])

    def build_auto(*, max_live_calls: int, call_counter: list[int]):
        return ManifestAutoBenchmarkLLM(
            model="auto",
            api_key="offline-test-key",
            client=client,
            cache=None,
            max_live_calls=max_live_calls,
            call_counter=call_counter,
        )

    monkeypatch.setattr(module, "_build_manifest_auto_adapter", build_auto)
    monkeypatch.setattr(
        module,
        "_build_qwen_adapter",
        lambda: (_ for _ in ()).throw(AssertionError("Ollama must not be invoked")),
    )
    monkeypatch.setattr(
        module,
        "_detect_local_baseline_cache_comparable",
        lambda: (_ for _ in ()).throw(AssertionError("Ollama cache must not be inspected")),
    )

    exit_code = module.main([
        "--manifest-auto",
        "--allow-manifest-calls",
        "4",
        "--output-dir",
        str(tmp_path),
    ])

    assert exit_code == 3
    assert len(client.requests) == 4
    assert {request["json"]["model"] for request in client.requests} == {"auto"}

    summaries = list(tmp_path.glob("*/run_summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["provider"] == {
        "name": "manifest_auto_benchmark",
        "requested_model": "auto",
        "reported_model": None,
        "model_identity_verified": False,
        "auto_routing": True,
    }
    assert summary["result"]["budget_aborted"] is True
    assert summary["result"]["exit_code"] == 3
    assert summary["result"]["n_extraction_complete"] == 1
    assert summary["notes"]["http_post_attempts"] == 4
    extraction_rows = [
        json.loads(line)
        for line in summaries[0].with_name("extractions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert any(
        row.get("kind") == "chunk_status" and row.get("extraction_complete") is True
        for row in extraction_rows
    )
    assert summary["notes"]["response_cache"]["enabled"] is False

    metrics = summary["metrics"]
    assert [metric["requested_model"] for metric in metrics] == ["auto"] * 4
    assert [metric["reported_model"] for metric in metrics] == [
        "provider/model-b",
        "provider/model-c",
        "provider/model-d",
        "<unknown>",
    ]
    assert metrics[0]["extra"]["attempt_reported_models"] == [
        "provider/model-a",
        "provider/model-b",
    ]
    assert metrics[0]["extra"]["model_changed_during_request"] is True
    assert metrics[-1]["status"] == "error"
    assert metrics[-1]["extra"]["exception_type"] == "CallBudgetExceeded"


def test_manifest_auto_cli_saves_partial_report_after_exhausted_429_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import httpx
    import src.extract.pipeline_v2 as pipeline_v2
    from src.providers.adapters.manifest_openai import ManifestAutoBenchmarkLLM

    script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_smoke.py"
    spec = importlib.util.spec_from_file_location("run_smoke_auto_429", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class RateLimitedResponse:
        status_code = 429

        def json(self):
            return {}

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "HTTP 429",
                request=httpx.Request("POST", "http://manifest.test/v1/chat/completions"),
                response=httpx.Response(429),
            )

    class RateLimitedClient:
        timeout = 60.0

        def __init__(self):
            self.requests: list[dict] = []

        def post(self, url: str, json: dict | None = None, headers: dict | None = None):
            self.requests.append({"url": url, "json": json, "headers": headers or {}})
            return RateLimitedResponse()

    client = RateLimitedClient()

    def build_auto(*, max_live_calls: int, call_counter: list[int]):
        return ManifestAutoBenchmarkLLM(
            model="auto",
            api_key="offline-test-key",
            client=client,
            cache=None,
            max_live_calls=max_live_calls,
            call_counter=call_counter,
        )

    monkeypatch.setattr(module, "_build_manifest_auto_adapter", build_auto)
    monkeypatch.setattr(pipeline_v2.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        module,
        "_detect_local_baseline_cache_comparable",
        lambda: (_ for _ in ()).throw(AssertionError("Ollama cache must not be inspected")),
    )

    exit_code = module.main([
        "--manifest-auto",
        "--allow-manifest-calls",
        "4",
        "--output-dir",
        str(tmp_path),
    ])

    assert exit_code == 4
    assert len(client.requests) == 3
    summaries = list(tmp_path.glob("*/run_summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["notes"]["partial_run"] is True
    assert summary["notes"]["http_post_attempts"] == 3
    assert summary["notes"]["terminal_error"] == {
        "type": "ProviderTransportError",
        "status": 429,
        "transient": True,
    }
    assert summaries[0].with_name("extractions.jsonl").is_file()
    assert summaries[0].with_name("extractions.rejects.jsonl").is_file()
    assert [metric["reported_model"] for metric in summary["metrics"]] == [
        "<unknown>",
        "<unknown>",
        "<unknown>",
    ]
