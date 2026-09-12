"""Ollama LLM adapter (#19).

Local-only LLM provider that talks to a running Ollama server over
HTTP. We deliberately use the existing ``httpx`` dependency rather than
adding the official ``ollama`` SDK — the spec discourages adding
provider SDKs outside the adapter modules and the wire format is small.

Endpoint surface used:

* ``POST {base}/api/chat`` with ``format="json"`` for ``generate_json``
  (Ollama's documented JSON-mode flag); this is the cleanest supported
  mechanism compatible with the pinned public protocol (spec §2).
* ``POST {base}/api/chat`` without ``format`` for ``generate``.

The adapter never imports ``ollama``. Per spec §2 the provider SDK
boundary lives entirely inside this module.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

import httpx

from src.providers.adapters import _schema
from src.providers.cache import ProviderCache
from src.providers.protocol import LLMProvider, LLMValidationError, ProviderTransportError, Usage


DEFAULT_BASE_URL = "http://localhost:11434"
PROVIDER_KEY = "ollama"
NAME = "ollama"

# Ollama reports no cost in USD; cost_usd=None so the budget layer
# skips the USD cap check entirely (spec §6.5, "No cost is not an
# error"). Token counts come from Ollama's ``eval_count`` /
# ``prompt_eval_count`` when present.
#
# Ollama has no API-key requirement (spec §3.3: "Ollama (local; no
# key needed)"). Adapters that need a key declare REQUIRED_SECRETS; we
# declare none.
REQUIRED_SECRETS: tuple[str, ...] = ()


# Cache key namespace; bump on adapter changes that affect parsing or
# output shape (spec §5.3). The constant lives here so old cache files
# never silently appear for the new format.
CACHE_FORMAT_VERSION = "ollama-2"

# --- transport error mapping (spec §6.6, repair #23) ----------------
#
# Transient status codes: connection / timeout failures plus the
# HTTP statuses documented as retry-eligible. Anything outside this
# set (other 4xx — auth, model-not-found, malformed request, 409
# Conflict, 425 Too Early, etc.) is non-transient; the caller's
# bounded retry must NOT retry those.
_TRANSIENT_HTTP_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


def _raise_transport_from_http_status(status: int, cause: httpx.HTTPStatusError) -> None:
    """Map an ``httpx.HTTPStatusError`` to ``ProviderTransportError``.

    ``status`` is the HTTP status code; ``cause`` is the original SDK
    exception (preserved via ``ProviderTransportError.cause`` for
    logging). Spec §6.6 frozen mapping for Ollama: transient on
    ``{408, 429, 500, 502, 503, 504}``; non-transient on every other
    4xx (auth, model-not-found, malformed, ``409``, ``425``). 1xx /
    3xx do not raise ``HTTPStatusError``; only 4xx / 5xx do, so this
    function never sees them.
    """
    transient = status in _TRANSIENT_HTTP_STATUSES
    raise ProviderTransportError(transient=transient, status=status, cause=cause)


def _request_json(client: httpx.Client, url: str, *, payload: dict) -> Any:
    """POST ``payload`` to ``url`` and return the parsed JSON body.

    Transport-layer and HTTP-status failures are translated to
    :class:`ProviderTransportError` per spec §6.6; everything else
    (parse errors, programming errors, unexpected SDK exceptions) is
    intentionally allowed to propagate unchanged — adapters MUST NOT
    use a broad ``except Exception`` and MUST NOT silently leak the
    SDK class.
    """
    try:
        response = client.post(url, json=payload)
    except (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.NetworkError,
        httpx.RemoteProtocolError,
    ) as exc:
        raise ProviderTransportError(transient=True, status=None, cause=exc) from exc

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # ``exc.response.status_code`` is guaranteed on a real
        # response; guard defensively for fakes / mocks that don't
        # set it.
        status = getattr(exc.response, "status_code", None)
        if isinstance(status, int):
            _raise_transport_from_http_status(status, exc)
        raise ProviderTransportError(transient=False, status=None, cause=exc) from exc

    return response.json()


class OllamaLLM(LLMProvider):
    """HTTP adapter for a locally running Ollama server.

    Constructed via :mod:`src.providers.registry` from environment; the
    constructor takes the resolved ``base_url``, ``model``,
    ``timeout_s``, and an injectable cache (``None`` for tests and
    callers that want to opt out of disk caching).
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        cache: ProviderCache | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = NAME
        self.model = model
        self._base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._timeout_s = timeout_s
        self._cache = cache
        # One client per adapter; tests may inject a mock. ``httpx`` is
        # the only outbound HTTP dep we use.
        self._client = client if client is not None else httpx.Client(timeout=timeout_s)

    # --- public protocol --------------------------------------------

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[str, Usage]:
        cached_text = None
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
            usage = Usage()  # cache hit: no fresh accounting
            return cached_text, usage

        payload = self._chat_payload(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            fmt=None,
        )
        body = _request_json(self._client, f"{self._base_url}/api/chat", payload=payload)
        text, usage = _parse_chat_response(body)

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
        # Cache hit short-circuit: a stored dict that already passed
        # validation when it was written; we re-validate defensively
        # so a tampered cache cannot smuggle invalid output.
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
                _schema.validate(cached, schema)
                return cached, Usage()

        # Attempt 1: ask Ollama for JSON mode natively via ``format``.
        text, usage = self._chat_json(system, user, schema, max_tokens, temperature)
        try:
            parsed = _parse_model_json(text)
            _schema.validate(parsed, schema)
        except (ValueError, _schema.SchemaValidationError) as first_err:
            # Attempt 2: retry exactly once with a JSON-corrective
            # suffix. Spec §2: "retries exactly once with a 'respond
            # with valid JSON matching this schema' suffix".
            retry_text, retry_usage = self._chat_json_retry(
                system, user, schema, text, max_tokens, temperature
            )
            total = _merge_usage(usage, retry_usage)
            try:
                parsed = _parse_model_json(retry_text)
                _schema.validate(parsed, schema)
            except (ValueError, _schema.SchemaValidationError) as second_err:
                raise LLMValidationError(
                    model_output=retry_text,
                    schema=schema,
                    validation_error=second_err,
                ) from second_err
            usage = total
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

    # --- internals --------------------------------------------------

    def _chat_payload(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        fmt: Any,
    ) -> dict:
        """Build the ``/api/chat`` request body.

        ``fmt="json"`` enables Ollama's JSON-mode; ``None`` disables.
        ``options.num_predict`` caps token output (Ollama's
        ``num_predict`` is the documented equivalent of
        ``max_tokens``); ``options.temperature`` is forwarded
        unmodified.
        """
        body: dict = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if fmt is not None:
            body["format"] = fmt
        return body

    def _chat_json(
        self,
        system: str,
        user: str,
        schema: dict,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, Usage]:
        # Inject a JSON-mode request into the user prompt so the model
        # has both the structured-output hint and the schema to aim at.
        json_user = (
            f"{user}\n\n"
            "Respond with valid JSON matching this schema:\n"
            f"{json.dumps(schema, sort_keys=True)}\n"
            "Output only the JSON object, no prose."
        )
        payload = self._chat_payload(
            system=system,
            user=json_user,
            max_tokens=max_tokens,
            temperature=temperature,
            fmt="json",
        )
        body = _request_json(self._client, f"{self._base_url}/api/chat", payload=payload)
        return _parse_chat_response(body)

    def _chat_json_retry(
        self,
        system: str,
        user: str,
        schema: dict,
        previous_output: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, Usage]:
        retry_user = (
            f"{user}\n\n"
            "Your previous response was not valid JSON matching the "
            "required schema. Respond again with valid JSON matching "
            "this schema:\n"
            f"{json.dumps(schema, sort_keys=True)}\n"
            "Output only the JSON object, no prose."
        )
        payload = self._chat_payload(
            system=system,
            user=retry_user,
            max_tokens=max_tokens,
            temperature=temperature,
            fmt="json",
        )
        body = _request_json(self._client, f"{self._base_url}/api/chat", payload=payload)
        # The previous output is intentionally NOT cached under any
        # schema key; the retry path bypasses the cache so a poisoned
        # cache entry cannot absorb an invalid response.
        del previous_output  # explicit: unused past this point
        return _parse_chat_response(body)


# --- response parsing ------------------------------------------------


def _parse_chat_response(body: Any) -> tuple[str, Usage]:
    """Extract content text and usage from an Ollama ``/api/chat`` body.

    Ollama returns ``message.content`` for the assistant text and
    optional ``prompt_eval_count`` / ``eval_count`` for token counts.
    ``prompt_eval_count`` = input tokens, ``eval_count`` = output
    tokens. Missing fields become ``None`` so the budget layer can
    fall through gracefully.
    """
    if not isinstance(body, Mapping):
        raise ValueError(f"Ollama /api/chat returned non-mapping body: {type(body).__name__}")
    message = body.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("Ollama /api/chat body has no 'message' mapping")
    text = message.get("content")
    if not isinstance(text, str):
        raise ValueError("Ollama /api/chat message.content is not a string")
    prompt_tokens = body.get("prompt_eval_count")
    eval_tokens = body.get("eval_count")
    usage = Usage(
        input_tokens=int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
        output_tokens=int(eval_tokens) if isinstance(eval_tokens, int) else None,
        cost_usd=None,
    )
    return text, usage


def _merge_usage(a: Usage, b: Usage) -> Usage:
    """Sum token counts across two ``Usage`` records.

    ``cost_usd`` stays ``None`` because neither Ollama call returns a
    cost figure; treating ``None`` as the sum "0" would be a silent
    fabrication (spec §6.5).
    """
    in_t = (a.input_tokens or 0) + (b.input_tokens or 0) if (a.input_tokens or b.input_tokens) else None
    out_t = (a.output_tokens or 0) + (b.output_tokens or 0) if (a.output_tokens or b.output_tokens) else None
    return Usage(input_tokens=in_t, output_tokens=out_t, cost_usd=None)


def _parse_model_json(text: str) -> dict:
    """Parse the model's text output as a JSON object, strictly.

    Spec §2: ``generate_json`` treats the model's response as JSON
    only after normal surrounding whitespace is stripped, and the
    entire response is parsed with ``json.loads``. We never scan for
    embedded ``{...}`` substrings, never regex-extract JSON, and never
    silently repair prose-wrapped or otherwise malformed output —
    invalid JSON is the trigger for the spec-mandated correction
    retry, not something this layer quietly fixes.
    """
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"model output is not JSON: {e.msg}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"model output is not a JSON object: got {type(obj).__name__}")
    return obj