"""Experimental ``manifest`` adapter for the model-extraction benchmark.

This adapter is **not** part of the production provider registry. It
exists solely so the v2 extraction pipeline can be exercised against
the Manifest gateway's OpenAI-compatible Chat Completions endpoint
during the model-extraction benchmark. The class satisfies the
existing :class:`src.providers.protocol.LLMProvider` protocol
structurally; it is intentionally NOT added to ``LLM_REGISTRY`` so the
default ``make_llm()`` factory cannot silently route a real run
through it.

Design contract (benchmark scope only):

* Auth is the single secret ``MANIFEST_API_KEY`` from the environment.
* Model identifier is passed verbatim to the gateway and echoed back
  in the response; the adapter records the gateway-reported ``model``
  next to the requested one and raises
  :class:`ModelMismatchError` if they do not match. The runner keeps
  no silent fallback path.
* JSON mode uses ``response_format={"type": "json_schema", ...}``
  when the gateway supports it; otherwise it falls back to the same
  inline schema-suffix prompt the Ollama adapter uses. Both modes
  share the same response parser and validation gate.
* Transport errors map to :class:`ProviderTransportError` exactly
  like the Ollama adapter; only HTTP 429 / 408 / 5xx are transient.
* Cache format version is its own namespace; old ollama caches can
  never appear as benchmark hits.

Auto-routing mode (``--manifest-auto``)
--------------------------------------

For the auto-routing benchmark the smoke runner instantiates
:class:`ManifestAutoBenchmarkLLM` instead of
:class:`ManifestBenchmarkLLM`. The auto adapter:

* Sends ``model="auto"`` to the gateway verbatim; the gateway picks
  the real model.
* Does NOT enforce the model-mismatch check; the gateway-reported
  ``model`` may legitimately be any manifest-routed identifier.
* Records the gateway-reported identifier per call on
  ``self.reported_models`` (a list, one entry per ``generate`` /
  ``generate_json`` call, in call order). When the gateway does not
  surface a verifiable identifier the entry is the literal string
  ``"<unknown>"``; the runner never invents a mapping.
* Refuses to read or write the response cache; auto-mode results
  must never be served from a cache entry that could have been
  populated by a different routed model.
* Still honours the per-process ``max_live_calls`` cap; exceeding it
  raises :class:`CallBudgetExceeded` and the runner emits a partial
  report.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from src.providers.cache import ProviderCache
from src.providers.budget import BudgetExceeded

from src.providers.protocol import (
    LLMProvider,
    LLMValidationError,
    ProviderTransportError,
    Usage,
)


PROVIDER_KEY = "manifest_benchmark"
NAME = "manifest_benchmark"

# Adapter-namespace cache version. Bumped if parsing or key shape
# changes; must never collide with the ollama cache namespace.
CACHE_FORMAT_VERSION = "manifest-openai-1"

# Bearer-secret env var the adapter needs at construction time. The
# benchmark runner is the only caller; the production provider
# registry MUST keep this name out of its default-secrets list so the
# ordinary ``make_llm`` factory never accidentally asks for it.
REQUIRED_SECRETS: tuple[str, ...] = ("MANIFEST_API_KEY",)

# Default gateway base URL pinned by the omp ``models.yml``. The runner
# may override this in tests via the constructor argument.
DEFAULT_BASE_URL = "http://100.111.8.20:2100/v1"

# Hard cap on live calls per process. The smoke runner and the
# dataset runner both honour this; exceeding it aborts the run with
# :class:`CallBudgetExceeded` rather than retrying.
DEFAULT_MAX_LIVE_CALLS = 4

# Transient HTTP statuses (matches ollama adapter contract).
_TRANSIENT_HTTP_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

# Sentinel reported-model entry used when the gateway does not
# surface a verifiable model identifier. The runner must never
# invent a mapping; "unknown" is the only honest label.
_REPORTED_MODEL_UNKNOWN = "<unknown>"

# Auto-routing mode token sent to the gateway verbatim. Manifest
# resolves this to a concrete provider/model pair per call. The
# adapter refuses any other value in ``model=`` for the auto class.
MANIFEST_AUTO_MODEL = "auto"




class ManifestConfigError(RuntimeError):
    """Raised when the adapter configuration is invalid."""


class ModelMismatchError(RuntimeError):
    """Raised when the gateway-reported model does not match the
    requested identifier. The runner keeps no silent fallback; a
    mismatch is treated as an undefined-mapping failure and the
    adapter aborts the call."""


class CallBudgetExceeded(BudgetExceeded):
    """Raised before an HTTP POST would exceed the configured cap.

    The unchanged V2 runner recognizes the provider-layer budget base
    class and therefore persists its actual partial extraction state.
    """

    def __init__(self, cap: int) -> None:
        super().__init__(
            cumulative=cap,
            cap=cap,
            unit="HTTP POST attempts",
        )


@dataclass(frozen=True)
class RequestedModel:
    """Captures the requested model id and the gateway-reported id so
    the runner can persist both even when they match."""

    requested: str
    reported: str | None


def _raise_transport_from_http_status(status: int, cause: httpx.HTTPStatusError) -> None:
    """Map an ``httpx.HTTPStatusError`` to :class:`ProviderTransportError`.

    Mirrors the ollama adapter contract: 408 / 429 / 5xx are
    transient; anything else is non-transient.
    """
    transient = status in _TRANSIENT_HTTP_STATUSES
    raise ProviderTransportError(transient=transient, status=status, cause=cause) from cause


def _request_chat(
    client: httpx.Client,
    url: str,
    *,
    payload: dict,
    headers: dict[str, str],
) -> dict:
    """POST ``payload`` to ``url`` and return the parsed JSON body."""
    try:
        response = client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        # Network / timeout / connection refused → transient.
        raise ProviderTransportError(transient=True, status=None, cause=exc) from exc
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            _raise_transport_from_http_status(status, exc)
        raise ProviderTransportError(transient=False, status=None, cause=exc) from exc
    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        raise ProviderTransportError(transient=False, status=None, cause=exc) from exc
    if not isinstance(body, Mapping):
        raise ProviderTransportError(transient=False, status=None)
    return dict(body)


def _parse_chat_response(
    body: Mapping[str, Any],
    *,
    requested_model: str,
    enforce_model_match: bool = True,
    unknown_model: str = "",
) -> tuple[str, Usage, str]:
    """Extract content text, usage, and the gateway-reported model id.

    OpenAI-compatible chat-completions responses carry the assistant
    content at ``choices[0].message.content``. Token counts live at
    ``usage.prompt_tokens`` / ``usage.completion_tokens``. The gateway
    echoes the effective model at ``model``; the adapter requires it
    to match ``requested_model`` exactly and refuses to alias
    "auto" / empty strings to it.
    """
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("manifest response has no 'choices' list")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("manifest response first choice is not a mapping")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("manifest response first choice has no 'message' mapping")
    text = message.get("content")
    if not isinstance(text, str):
        raise ValueError("manifest response message.content is not a string")
    usage_block = body.get("usage")
    in_tokens: int | None = None
    out_tokens: int | None = None
    if isinstance(usage_block, Mapping):
        pt = usage_block.get("prompt_tokens")
        ct = usage_block.get("completion_tokens")
        if isinstance(pt, int):
            in_tokens = pt
        if isinstance(ct, int):
            out_tokens = ct
    reported = body.get("model")
    if not isinstance(reported, str) or not reported:
        reported = unknown_model
    if enforce_model_match and reported and reported != requested_model:
        raise ModelMismatchError(
            f"manifest reported model {reported!r} != requested {requested_model!r}"
        )
    return text, Usage(input_tokens=in_tokens, output_tokens=out_tokens, cost_usd=None), reported


def _parse_model_json(text: str) -> dict:
    """Parse model output as a JSON object. Mirrors ollama contract:
    no scanning, no regex extraction, no silent repair."""
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"model output is not JSON: {e.msg}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"model output is not a JSON object: got {type(obj).__name__}")
    return obj


def _merge_usage(a: Usage, b: Usage) -> Usage:
    """Sum token counts across two ``Usage`` records; cost stays None
    because the manifest gateway is not declared as cost-reporting."""
    in_t = (a.input_tokens or 0) + (b.input_tokens or 0) if (a.input_tokens or b.input_tokens) else None
    out_t = (a.output_tokens or 0) + (b.output_tokens or 0) if (a.output_tokens or b.output_tokens) else None
    return Usage(input_tokens=in_t, output_tokens=out_t, cost_usd=None)


class ManifestBenchmarkLLM(LLMProvider):
    """Adapter for the Manifest gateway's OpenAI-compatible chat API.

    Constructed directly by the benchmark runner; not registered in
    :data:`src.providers.registry.LLM_REGISTRY`. The adapter refuses
    to operate without ``MANIFEST_API_KEY`` (no anon mode, no
    dummy fallback) and counts live calls so the runner can abort
    before exceeding the per-process cap.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        cache: ProviderCache | None = None,
        client: httpx.Client | None = None,
        max_live_calls: int = DEFAULT_MAX_LIVE_CALLS,
        call_counter: list[int] | None = None,
    ) -> None:
        if not model or not isinstance(model, str):
            raise ManifestConfigError("model identifier is required")
        self.name = NAME
        self.model = model
        key = api_key if api_key is not None else os.environ.get("MANIFEST_API_KEY", "")
        if not key:
            raise ManifestConfigError(
                "MANIFEST_API_KEY is required for the manifest benchmark adapter"
            )
        self._api_key = key
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._timeout_s = timeout_s
        self._cache = cache
        self._max_live_calls = int(max_live_calls)
        # Shared counter lets the runner pre-charge known offline
        # replays against the same cap so a partially-precharged
        # process cannot silently exceed it.
        self._counter = call_counter if call_counter is not None else [0]
        self._client = client if client is not None else httpx.Client(timeout=timeout_s)

    # --- live-call accounting --------------------------------------------------

    def _charge_live(self) -> None:
        # Check before increment so ``live_call_count`` is exactly the
        # number of HTTP POST attempts actually made. A rejected fifth
        # attempt leaves the counter at the four-attempt cap.
        if self._counter[0] >= self._max_live_calls:
            raise CallBudgetExceeded(self._max_live_calls)
        self._counter[0] += 1

    @property
    def live_call_count(self) -> int:
        return self._counter[0]

    def _parse_response(self, body: Mapping[str, Any]) -> tuple[str, Usage, str]:
        return _parse_chat_response(body, requested_model=self.model)

    def _record_reported_model(self, reported: str) -> None:
        # Fixed-model mode records the configured identity in the
        # benchmark wrapper; no adapter-side queue is needed.
        return None

    # --- payload assembly --------------------------------------------------

    def _chat_payload(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            # OpenAI-compatible JSON mode. The gateway may or may not
            # accept it; the response parser is identical either way.
            body["response_format"] = {"type": "json_object"}
        return body

    # --- public protocol ----------------------------------------------------

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[str, Usage]:
        cached_text: str | None = None
        if self._cache is not None:
            cached_text = self._cache.get_generate(
                provider_name=self.name,
                provider_model=self.model,
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        if cached_text is not None:
            return cached_text, Usage()

        self._charge_live()
        body = _request_chat(
            self._client,
            f"{self._base_url}/chat/completions",
            payload=self._chat_payload(
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=False,
            ),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        text, usage, reported = self._parse_response(body)
        self._record_reported_model(reported)
        if self._cache is not None:
            self._cache.put_generate(
                provider_name=self.name,
                provider_model=self.model,
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
                text=text,
            )
        return text, usage

    def generate_json(
        self,
        system: str,
        user: str,
        schema: dict,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[dict, Usage]:
        if self._cache is not None:
            cached = self._cache.get_generate_json(
                provider_name=self.name,
                provider_model=self.model,
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
                schema=schema,
            )
            if cached is not None:
                # Re-validate defensively so a tampered cache entry
                # cannot smuggle an invalid payload past the schema
                # gate. Mirrors the ollama adapter contract.
                from src.providers.adapters import _schema

                _schema.validate(cached, schema)
                return cached, Usage()

        # Inline-schema suffix keeps the contract identical to the
        # ollama adapter when the gateway falls back from
        # ``response_format=json_object`` to plain chat.
        json_user = (
            f"{user}\n\n"
            "Respond with valid JSON matching this schema:\n"
            f"{json.dumps(schema, sort_keys=True)}\n"
            "Output only the JSON object, no prose."
        )
        self._charge_live()
        body = _request_chat(
            self._client,
            f"{self._base_url}/chat/completions",
            payload=self._chat_payload(
                system=system,
                user=json_user,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=True,
            ),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        text, usage, reported = self._parse_response(body)
        self._record_reported_model(reported)

        from src.providers.adapters import _schema

        try:
            parsed = _parse_model_json(text)
            _schema.validate(parsed, schema)
        except (ValueError, _schema.SchemaValidationError):
            # Provider contract: malformed JSON and schema-invalid JSON
            # receive one corrective POST. Both attempts consume the
            # shared HTTP cap and record their response-level identities.
            retry_user = (
                f"{user}\n\n"
                "Your previous response was not valid JSON matching the "
                "required schema. Respond again with valid JSON matching "
                f"this schema:\n{json.dumps(schema, sort_keys=True)}\n"
                "Output only the JSON object, no prose."
            )
            self._charge_live()
            retry_body = _request_chat(
                self._client,
                f"{self._base_url}/chat/completions",
                payload=self._chat_payload(
                    system=system,
                    user=retry_user,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_mode=True,
                ),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            retry_text, retry_usage, retry_reported = self._parse_response(retry_body)
            self._record_reported_model(retry_reported)
            usage = _merge_usage(usage, retry_usage)
            try:
                parsed = _parse_model_json(retry_text)
                _schema.validate(parsed, schema)
            except (ValueError, _schema.SchemaValidationError) as second_err:
                raise LLMValidationError(
                    model_output=retry_text,
                    schema=schema,
                    validation_error=second_err,
                ) from second_err
        if self._cache is not None:
            self._cache.put_generate_json(
                provider_name=self.name,
                provider_model=self.model,
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
                schema=schema,
                parsed=parsed,
            )
        return parsed, usage

class ManifestAutoBenchmarkLLM(ManifestBenchmarkLLM):
    """Manifest auto-routing adapter for the extraction system benchmark.

    Sends ``model="auto"`` verbatim, accepts any response-level model
    identifier, records every actual HTTP response identity, and hard-
    disables caching because an ``auto`` key cannot identify the real model.
    """

    def __init__(
        self,
        *,
        model: str = MANIFEST_AUTO_MODEL,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        cache: ProviderCache | None = None,
        client: httpx.Client | None = None,
        max_live_calls: int = DEFAULT_MAX_LIVE_CALLS,
        call_counter: list[int] | None = None,
    ) -> None:
        if model != MANIFEST_AUTO_MODEL:
            raise ManifestConfigError(
                f"ManifestAutoBenchmarkLLM requires model={MANIFEST_AUTO_MODEL!r}; "
                f"got {model!r}"
            )
        if cache is not None:
            raise ManifestConfigError(
                "ManifestAutoBenchmarkLLM does not allow a ProviderCache; "
                "auto-routing results must never be served from a cache."
            )
        super().__init__(
            model=MANIFEST_AUTO_MODEL,
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
            cache=None,
            client=client,
            max_live_calls=max_live_calls,
            call_counter=call_counter,
        )
        self.name = "manifest_auto_benchmark"
        # Drained by TimingProvider after each entity/relation request.
        # JSON correction can add two identities for one phase request.
        self.reported_models: list[str] = []

    def _parse_response(self, body: Mapping[str, Any]) -> tuple[str, Usage, str]:
        # Deliberately no Terra/fixed-model equality check: Manifest owns
        # provider/model selection in this mode.
        return _parse_chat_response(
            body,
            requested_model=MANIFEST_AUTO_MODEL,
            enforce_model_match=False,
            unknown_model=_REPORTED_MODEL_UNKNOWN,
        )

    def _record_reported_model(self, reported: str) -> None:
        self.reported_models.append(reported)
