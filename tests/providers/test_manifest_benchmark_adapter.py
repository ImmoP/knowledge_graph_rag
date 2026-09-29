"""Tests for the experimental ``manifest_benchmark`` adapter.

All tests run with a stub ``httpx.Client``; no live gateway is
required. The adapter has no production registry entry so the tests
import the class directly and confirm it satisfies the structural
``LLMProvider`` protocol.

Coverage:

* successful ``generate`` returns text + Usage,
* successful ``generate_json`` returns parsed dict + Usage,
* schema gate failure surfaces ``LLMValidationError`` after one
  corrective retry,
* HTTP 429 / 503 / 401 are mapped to ``ProviderTransportError`` with
  the right ``transient`` flag and ``status``,
* model mismatch between request and response aborts,
* live-call counter enforces the configured cap,
* cache namespace is isolated from the ollama cache format version,
* bearer auth header carries the manifest secret but tests never log
  it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx  # noqa: E402
import pytest  # noqa: E402

from src.providers.adapters.manifest_openai import (  # noqa: E402
    CACHE_FORMAT_VERSION as MANIFEST_CACHE_FMT,
    PROVIDER_KEY,
    NAME,
    CallBudgetExceeded,
    MANIFEST_AUTO_MODEL,
    ManifestAutoBenchmarkLLM,
    ManifestBenchmarkLLM,
    ManifestConfigError,
    ModelMismatchError,
)
from src.providers.adapters.ollama import CACHE_FORMAT_VERSION as OLLAMA_CACHE_FMT  # noqa: E402
from src.providers.cache import CachePaths, ProviderCache  # noqa: E402
from src.providers.protocol import LLMProvider, LLMValidationError, ProviderTransportError  # noqa: E402


MODEL = "openai/gpt-5.6-terra-subscription"


class _FakeResponse:
    def __init__(
        self,
        body: Any = None,
        *,
        status_code: int = 200,
        raise_exc: BaseException | None = None,
    ) -> None:
        self._body = body if body is not None else {}
        self.status_code = status_code
        self._raise_exc = raise_exc

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self._raise_exc is not None:
            # Adapter catches ``httpx.HTTPStatusError``; wrap arbitrary
            # exceptions so the transport-error mapping fires.
            try:
                raise self._raise_exc
            except httpx.HTTPStatusError:
                raise
            except Exception as exc:
                http_exc = httpx.HTTPStatusError(
                    f"HTTP {self.status_code}",
                    request=None,
                    response=self,  # type: ignore[arg-type]
                )
                raise http_exc from exc
        if self.status_code >= 400:
            err = httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=None,
                response=self,  # type: ignore[arg-type]
            )
            raise err


class _FakeClient:
    """Records every request and returns scripted responses in order."""

    def __init__(self, scripts: list) -> None:
        self._scripts = list(scripts)
        self.requests: list[dict] = []
        self.timeout = 60.0

    def post(self, url: str, json: dict | None = None, headers: dict | None = None):
        self.requests.append({"url": url, "json": json, "headers": headers or {}})
        idx = min(len(self.requests) - 1, len(self._scripts) - 1)
        body = self._scripts[idx]
        if isinstance(body, dict):
            return _FakeResponse(body)
        return body


def _chat_body(content: str, *, prompt_tokens: int = 12, completion_tokens: int = 7) -> dict:
    return {
        "model": MODEL,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@pytest.fixture
def adapter_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MANIFEST_API_KEY", "secret-key-test")
    cache = ProviderCache(CachePaths.from_root(tmp_path / "cache"))
    counter = [0]
    return {
        "cache": cache,
        "counter": counter,
        "max_live_calls": 4,
    }


def test_provider_key_and_name_are_isolated_from_production_registry():
    assert PROVIDER_KEY == "manifest_benchmark"
    assert NAME == "manifest_benchmark"
    from src.providers import make_llm

    with pytest.raises(KeyError):
        make_llm(
            {
                "LLM_PROVIDER": PROVIDER_KEY,
                "LLM_MODEL": MODEL,
                "MANIFEST_API_KEY": "secret-key-test",
            }
        )


def test_cache_namespace_is_isolated_from_ollama():
    assert MANIFEST_CACHE_FMT != OLLAMA_CACHE_FMT
    assert MANIFEST_CACHE_FMT == "manifest-openai-1"


def test_missing_api_key_raises_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MANIFEST_API_KEY", raising=False)
    with pytest.raises(ManifestConfigError):
        ManifestBenchmarkLLM(
            model=MODEL,
            cache=ProviderCache(CachePaths.from_root(tmp_path)),
        )


def test_empty_model_raises_config_error(tmp_path: Path):
    with pytest.raises(ManifestConfigError):
        ManifestBenchmarkLLM(
            model="",
            api_key="k",
            cache=ProviderCache(CachePaths.from_root(tmp_path)),
        )


def test_structural_protocol_satisfied(adapter_env):
    client = _FakeClient([_chat_body("hi")])
    a = ManifestBenchmarkLLM(
        model=MODEL,
        api_key="k",
        client=client,
        cache=adapter_env["cache"],
        call_counter=adapter_env["counter"],
        max_live_calls=adapter_env["max_live_calls"],
    )
    assert isinstance(a, LLMProvider)
    assert a.name == NAME
    assert a.model == MODEL


def test_generate_success(adapter_env):
    body = _chat_body("pong-like response", prompt_tokens=10, completion_tokens=3)
    client = _FakeClient([body])
    a = ManifestBenchmarkLLM(
        model=MODEL,
        api_key="k",
        client=client,
        cache=adapter_env["cache"],
        call_counter=adapter_env["counter"],
        max_live_calls=adapter_env["max_live_calls"],
    )
    text, usage = a.generate("reply with exactly: pong", "ping", max_tokens=64)
    assert "pong" in text
    assert usage.input_tokens == 10
    assert usage.output_tokens == 3
    assert usage.cost_usd is None
    assert client.requests[0]["headers"]["Authorization"] == "Bearer k"
    assert client.requests[0]["json"]["model"] == MODEL
    assert adapter_env["counter"][0] == 1


def test_generate_json_success(adapter_env):
    schema = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "integer"}}}
    payload = json.dumps({"answer": 42})
    body = _chat_body(payload, prompt_tokens=20, completion_tokens=2)
    client = _FakeClient([body])
    a = ManifestBenchmarkLLM(
        model=MODEL,
        api_key="k",
        client=client,
        cache=adapter_env["cache"],
        call_counter=adapter_env["counter"],
        max_live_calls=adapter_env["max_live_calls"],
    )
    parsed, usage = a.generate_json("x", "y", schema)
    assert parsed == {"answer": 42}
    assert usage.input_tokens == 20
    assert usage.output_tokens == 2
    assert client.requests[0]["json"]["response_format"] == {"type": "json_object"}
    assert adapter_env["counter"][0] == 1


def test_generate_json_schema_failure_after_one_retry(adapter_env):
    schema = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "integer"}}}
    bad = _chat_body("not json")
    good = _chat_body(json.dumps({"answer": 7}))
    client = _FakeClient([bad, good])
    a = ManifestBenchmarkLLM(
        model=MODEL,
        api_key="k",
        client=client,
        cache=adapter_env["cache"],
        call_counter=adapter_env["counter"],
        max_live_calls=adapter_env["max_live_calls"],
    )
    parsed, usage = a.generate_json("x", "y", schema)
    assert parsed == {"answer": 7}
    assert adapter_env["counter"][0] == 2
    assert usage.input_tokens is not None and usage.input_tokens > 0


def test_generate_json_schema_invalid_object_retries_once(adapter_env):
    schema = {
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "integer"}},
    }
    invalid = _chat_body(json.dumps({"answer": "wrong-type"}))
    good = _chat_body(json.dumps({"answer": 7}))
    client = _FakeClient([invalid, good])
    adapter = ManifestBenchmarkLLM(
        model=MODEL,
        api_key="k",
        client=client,
        cache=adapter_env["cache"],
        call_counter=adapter_env["counter"],
        max_live_calls=adapter_env["max_live_calls"],
    )

    parsed, _ = adapter.generate_json("x", "y", schema)

    assert parsed == {"answer": 7}
    assert len(client.requests) == 2
    assert adapter.live_call_count == 2


def test_generate_json_two_failures_raise_validation_error(adapter_env):
    schema = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "integer"}}}
    bad1 = _chat_body("not json 1")
    bad2 = _chat_body("not json 2")
    client = _FakeClient([bad1, bad2])
    a = ManifestBenchmarkLLM(
        model=MODEL,
        api_key="k",
        client=client,
        cache=adapter_env["cache"],
        call_counter=adapter_env["counter"],
        max_live_calls=adapter_env["max_live_calls"],
    )
    with pytest.raises(LLMValidationError) as excinfo:
        a.generate_json("x", "y", schema)
    assert "not json 2" in excinfo.value.model_output
    assert adapter_env["counter"][0] == 2


def test_429_maps_to_transient_transport_error(adapter_env):
    err = RuntimeError("rate limited")
    client = _FakeClient([_FakeResponse(status_code=429, raise_exc=err)])
    a = ManifestBenchmarkLLM(
        model=MODEL,
        api_key="k",
        client=client,
        cache=adapter_env["cache"],
        call_counter=adapter_env["counter"],
        max_live_calls=adapter_env["max_live_calls"],
    )
    with pytest.raises(ProviderTransportError) as excinfo:
        a.generate("sys", "user")
    assert excinfo.value.transient is True
    assert excinfo.value.status == 429


def test_503_maps_to_transient_transport_error(adapter_env):
    err = RuntimeError("service unavailable")
    client = _FakeClient([_FakeResponse(status_code=503, raise_exc=err)])
    a = ManifestBenchmarkLLM(
        model=MODEL,
        api_key="k",
        client=client,
        cache=adapter_env["cache"],
        call_counter=adapter_env["counter"],
        max_live_calls=adapter_env["max_live_calls"],
    )
    with pytest.raises(ProviderTransportError) as excinfo:
        a.generate("sys", "user")
    assert excinfo.value.transient is True
    assert excinfo.value.status == 503


def test_401_maps_to_non_transient_transport_error(adapter_env):
    err = RuntimeError("unauthorized")
    client = _FakeClient([_FakeResponse(status_code=401, raise_exc=err)])
    a = ManifestBenchmarkLLM(
        model=MODEL,
        api_key="k",
        client=client,
        cache=adapter_env["cache"],
        call_counter=adapter_env["counter"],
        max_live_calls=adapter_env["max_live_calls"],
    )
    with pytest.raises(ProviderTransportError) as excinfo:
        a.generate("sys", "user")
    assert excinfo.value.transient is False
    assert excinfo.value.status == 401


def test_model_mismatch_raises(adapter_env):
    body = _chat_body("hi")
    body["model"] = "openai/gpt-5.6-luna-subscription"
    client = _FakeClient([body])
    a = ManifestBenchmarkLLM(
        model=MODEL,
        api_key="k",
        client=client,
        cache=adapter_env["cache"],
        call_counter=adapter_env["counter"],
        max_live_calls=adapter_env["max_live_calls"],
    )
    with pytest.raises(ModelMismatchError):
        a.generate("sys", "user")


def test_live_call_counter_enforces_cap(adapter_env):
    # Cache=None so every call actually hits the wire.
    client = _FakeClient([_chat_body("hi")] * 7)
    a = ManifestBenchmarkLLM(
        model=MODEL,
        api_key="k",
        client=client,
        cache=None,
        call_counter=adapter_env["counter"],
        max_live_calls=3,
    )
    a.generate("s", "u")
    a.generate("s", "u")
    a.generate("s", "u")
    assert a.live_call_count == 3
    with pytest.raises(CallBudgetExceeded):
        a.generate("s", "u")
    assert len(client.requests) == 3


def test_call_counter_is_shared_between_adapter_and_runner(adapter_env):
    # Cache=None so both adapters' calls always reach the wire.
    a1 = ManifestBenchmarkLLM(
        model=MODEL,
        api_key="k",
        client=_FakeClient([_chat_body("hi")]),
        cache=None,
        call_counter=adapter_env["counter"],
        max_live_calls=2,
    )
    a2 = ManifestBenchmarkLLM(
        model=MODEL,
        api_key="k",
        client=_FakeClient([_chat_body("hi")]),
        cache=None,
        call_counter=adapter_env["counter"],
        max_live_calls=2,
    )
    a1.generate("s", "u")
    a2.generate("s", "u")
    assert a1.live_call_count == 2
    assert a2.live_call_count == 2
    with pytest.raises(CallBudgetExceeded):
        a1.generate("s", "u")


def test_cache_separates_manifest_from_ollama(adapter_env):
    schema = {"type": "object", "required": ["x"], "properties": {"x": {"type": "string"}}}
    client_m = _FakeClient([_chat_body(json.dumps({"x": "manifest"}))])
    manifest = ManifestBenchmarkLLM(
        model=MODEL,
        api_key="k",
        client=client_m,
        cache=adapter_env["cache"],
        call_counter=adapter_env["counter"],
        max_live_calls=adapter_env["max_live_calls"],
    )
    parsed, _ = manifest.generate_json("sys", "user", schema)
    assert parsed == {"x": "manifest"}

    client_m2 = _FakeClient([])
    manifest2 = ManifestBenchmarkLLM(
        model=MODEL,
        api_key="k",
        client=client_m2,
        cache=adapter_env["cache"],
        call_counter=[0],
    )
    parsed2, _ = manifest2.generate_json("sys", "user", schema)
    assert parsed2 == {"x": "manifest"}
    assert client_m2.requests == []


def test_api_key_is_never_logged(adapter_env, caplog: pytest.LogCaptureFixture):
    body = _chat_body("hi")
    client = _FakeClient([body])
    a = ManifestBenchmarkLLM(
        model=MODEL,
        api_key="topsecret",
        client=client,
        cache=adapter_env["cache"],
        call_counter=adapter_env["counter"],
        max_live_calls=adapter_env["max_live_calls"],
    )
    a.generate("sys", "user")
    combined = caplog.text or ""
    assert "topsecret" not in combined
    assert client.requests[0]["headers"]["Authorization"].endswith("topsecret")

# ---------------------------------------------------------------------------
# Manifest auto-routing benchmark adapter (offline mocks only)
# ---------------------------------------------------------------------------


def _auto_chat_body(content: str, *, reported_model: str | None) -> dict:
    body = {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}}
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 5,
            "total_tokens": 16,
        },
    }
    if reported_model is not None:
        body["model"] = reported_model
    return body


def _auto_adapter(client: _FakeClient, *, max_live_calls: int = 4) -> ManifestAutoBenchmarkLLM:
    return ManifestAutoBenchmarkLLM(
        model=MANIFEST_AUTO_MODEL,
        api_key="test-secret",
        client=client,
        cache=None,
        max_live_calls=max_live_calls,
        call_counter=[0],
    )


def test_auto_mode_accepts_changing_reported_models_and_sends_auto():
    schema = {
        "type": "object",
        "required": ["x"],
        "properties": {"x": {"type": "string"}},
    }
    client = _FakeClient([
        _auto_chat_body(json.dumps({"x": "first"}), reported_model="provider-a/model-1"),
        _auto_chat_body(json.dumps({"x": "second"}), reported_model="provider-b/model-2"),
    ])
    adapter = _auto_adapter(client)

    first, _ = adapter.generate_json("sys", "user-1", schema)
    second, _ = adapter.generate_json("sys", "user-2", schema)

    assert first == {"x": "first"}
    assert second == {"x": "second"}
    assert adapter.reported_models == ["provider-a/model-1", "provider-b/model-2"]
    assert [req["json"]["model"] for req in client.requests] == ["auto", "auto"]
    assert adapter.live_call_count == 2


def test_auto_mode_marks_missing_reported_model_unknown():
    client = _FakeClient([
        _auto_chat_body("plain response", reported_model=None),
    ])
    adapter = _auto_adapter(client)

    text, _ = adapter.generate("sys", "user")

    assert text == "plain response"
    assert adapter.reported_models == ["<unknown>"]
    assert client.requests[0]["json"]["model"] == "auto"


def test_auto_mode_429_is_transient_and_counts_actual_post():
    client = _FakeClient([_FakeResponse(status_code=429)])
    adapter = _auto_adapter(client)

    with pytest.raises(ProviderTransportError) as excinfo:
        adapter.generate("sys", "user")

    assert excinfo.value.transient is True
    assert excinfo.value.status == 429
    assert adapter.live_call_count == 1
    assert len(client.requests) == 1
    # No response body means no verifiable model identifier.
    assert adapter.reported_models == []


def test_auto_mode_call_cap_allows_at_most_four_http_posts():
    client = _FakeClient([
        _auto_chat_body("ok", reported_model=f"provider/model-{idx}")
        for idx in range(4)
    ])
    adapter = _auto_adapter(client, max_live_calls=4)

    for _ in range(4):
        adapter.generate("sys", "user")

    with pytest.raises(CallBudgetExceeded):
        adapter.generate("sys", "fifth-attempt")

    assert adapter.live_call_count == 4
    assert len(client.requests) == 4


def test_auto_mode_rejects_cache_to_prevent_cross_model_contamination(tmp_path: Path):
    cache = ProviderCache(CachePaths.from_root(tmp_path / "cache"))
    with pytest.raises(ManifestConfigError, match="does not allow a ProviderCache"):
        ManifestAutoBenchmarkLLM(
            model="auto",
            api_key="test-secret",
            cache=cache,
        )


def test_auto_mode_rejects_non_auto_model():
    with pytest.raises(ManifestConfigError, match="requires model='auto'"):
        ManifestAutoBenchmarkLLM(
            model=MODEL,
            api_key="test-secret",
            cache=None,
        )
