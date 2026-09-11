"""Ollama LLM adapter tests with mocked HTTP (#19).

The adapter is exercised against a stub ``httpx.Client`` that returns
scripted responses. No live Ollama server is required. The cache is
passed in directly (no temp dirs required for ``generate`` /
``generate_json`` correctness).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.providers.adapters.ollama import (  # noqa: E402
    CACHE_FORMAT_VERSION as OLLAMA_CACHE_FMT,
    OllamaLLM,
)
from src.providers.cache import CachePaths, ProviderCache  # noqa: E402
from src.providers.protocol import LLMValidationError, Usage  # noqa: E402


import shutil


class _FakeResponse:
    def __init__(self, body: dict) -> None:
        self._body = body
        self.status_code = 200

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    """Records every request and returns scripted responses."""

    def __init__(self, scripts: list) -> None:
        self._scripts = list(scripts)
        self.requests: list = []
        self.timeout = 60.0

    def post(self, url, json=None):
        self.requests.append({"url": url, "json": json})
        idx = min(len(self.requests) - 1, len(self._scripts) - 1)
        body = self._scripts[idx]
        # If the test passed a raw dict, wrap it in a fake Response so
        # ``raise_for_status()`` and ``.json()`` both work.
        if isinstance(body, dict):
            return _FakeResponse(body)
        return body


def _ollama_chat_body(content: str, *, prompt_eval_count: int = 8, eval_count: int = 4):
    return {
        "model": "llama3.1:8b",
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "done": True,
    }


class TestOllamaGenerate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path("/tmp/ollama-test-cache-XXXX")  # unused
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="ollama-cache-"))
        self.cache = ProviderCache(CachePaths.from_root(self.tmp))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_generate_returns_text_and_usage(self):
        client = _FakeClient([_ollama_chat_body("hello", prompt_eval_count=8, eval_count=4)])
        adapter = OllamaLLM(
            model="llama3.1:8b",
            base_url="http://ollama.test:11434",
            cache=self.cache,
            client=client,
        )
        text, usage = adapter.generate("sys", "user")
        self.assertEqual(text, "hello")
        self.assertEqual(usage.input_tokens, 8)
        self.assertEqual(usage.output_tokens, 4)
        self.assertIsNone(usage.cost_usd)

    def test_generate_uses_chat_endpoint(self):
        client = _FakeClient([_ollama_chat_body("hi")])
        adapter = OllamaLLM(model="m", base_url="http://h:1", cache=self.cache, client=client)
        adapter.generate("s", "u")
        self.assertEqual(client.requests[0]["url"], "http://h:1/api/chat")

    def test_generate_forwards_options(self):
        client = _FakeClient([_ollama_chat_body("hi")])
        adapter = OllamaLLM(model="m", base_url="http://h:1", cache=self.cache, client=client)
        adapter.generate("s", "u", max_tokens=64, temperature=0.5)
        opts = client.requests[0]["json"]["options"]
        self.assertEqual(opts["num_predict"], 64)
        self.assertEqual(opts["temperature"], 0.5)

    def test_generate_cache_hit_skips_http(self):
        client = _FakeClient([_ollama_chat_body("ignored")])
        adapter = OllamaLLM(model="m", base_url="http://h:1", cache=self.cache, client=client)
        adapter.generate("s", "u")  # warms the cache with "ignored"
        # Second call: cache hit, no HTTP — return value is the cached
        # text, not the script body.
        text, usage = adapter.generate("s", "u")
        self.assertEqual(text, "ignored")
        self.assertIsNone(usage.input_tokens)
        self.assertEqual(len(client.requests), 1)


class TestOllamaGenerateJSON(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="ollama-cache-"))
        self.cache = ProviderCache(CachePaths.from_root(self.tmp))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_valid_first_response(self):
        body = _ollama_chat_body('{"answer": 42}')
        client = _FakeClient([body])
        adapter = OllamaLLM(model="m", base_url="http://h:1", cache=self.cache, client=client)
        schema = {"type": "object", "required": ["answer"]}
        parsed, usage = adapter.generate_json("s", "u", schema)
        self.assertEqual(parsed, {"answer": 42})
        # Single HTTP call — no retry.
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(usage.input_tokens, 8)

    def test_invalid_first_response_then_valid_retry(self):
        body_bad = _ollama_chat_body('{"missing": true}')
        body_good = _ollama_chat_body('{"answer": 7}')
        client = _FakeClient([body_bad, body_good])
        adapter = OllamaLLM(model="m", base_url="http://h:1", cache=self.cache, client=client)
        schema = {"type": "object", "required": ["answer"]}
        parsed, usage = adapter.generate_json("s", "u", schema)
        self.assertEqual(parsed, {"answer": 7})
        # Exactly two HTTP calls (one retry).
        self.assertEqual(len(client.requests), 2)
        # Retry payload uses ``format=json`` again (Ollama's documented
        # JSON-mode flag) and a corrective user message.
        retry = client.requests[1]["json"]
        self.assertEqual(retry.get("format"), "json")
        self.assertIn("previous response was not valid JSON", retry["messages"][-1]["content"])

    def test_invalid_twice_raises_llm_validation_error(self):
        body_bad = _ollama_chat_body('{"missing": true}')
        client = _FakeClient([body_bad, body_bad])
        adapter = OllamaLLM(model="m", base_url="http://h:1", cache=self.cache, client=client)
        schema = {"type": "object", "required": ["answer"]}
        with self.assertRaises(LLMValidationError) as ctx:
            adapter.generate_json("s", "u", schema)
        self.assertEqual(ctx.exception.model_output, '{"missing": true}')
        self.assertEqual(ctx.exception.schema, schema)
        # Exactly two HTTP calls: one initial + one retry.
        self.assertEqual(len(client.requests), 2)

    def test_exactly_one_retry(self):
        # Three bad responses: only two HTTP calls should happen
        # because the adapter raises after the retry fails.
        body_bad = _ollama_chat_body('{"missing": true}')
        client = _FakeClient([body_bad, body_bad, body_bad])
        adapter = OllamaLLM(model="m", base_url="http://h:1", cache=self.cache, client=client)
        schema = {"type": "object", "required": ["answer"]}
        with self.assertRaises(LLMValidationError):
            adapter.generate_json("s", "u", schema)
        self.assertEqual(len(client.requests), 2)

    def test_prose_wrapped_json_triggers_retry_then_succeeds(self):
        """Spec §2: invalid JSON (including prose-wrapped) is a retry trigger.

        First response is prose-wrapped JSON \u2014 strict parse fails, the
        retry fires, the retry returns strict valid JSON. Adapter
        returns after exactly two HTTP calls.
        """
        body_prose = _ollama_chat_body('Here you go:\n```json\n{"answer": 1}\n```\nbye')
        body_strict = _ollama_chat_body('{"answer": 1}')
        client = _FakeClient([body_prose, body_strict])
        adapter = OllamaLLM(model="m", base_url="http://h:1", cache=self.cache, client=client)
        schema = {"type": "object", "required": ["answer"]}
        parsed, _ = adapter.generate_json("s", "u", schema)
        self.assertEqual(parsed, {"answer": 1})
        self.assertEqual(len(client.requests), 2)

    def test_prose_wrapped_json_twice_raises(self):
        """Two prose-wrapped (invalid) responses \u2192 LLMValidationError after exactly two HTTP calls."""
        body_prose = _ollama_chat_body('Here you go:\n```json\n{"answer": 1}\n```\nbye')
        client = _FakeClient([body_prose, body_prose])
        adapter = OllamaLLM(model="m", base_url="http://h:1", cache=self.cache, client=client)
        schema = {"type": "object", "required": ["answer"]}
        with self.assertRaises(LLMValidationError) as ctx:
            adapter.generate_json("s", "u", schema)
        self.assertEqual(ctx.exception.model_output, 'Here you go:\n```json\n{"answer": 1}\n```\nbye')
        self.assertEqual(ctx.exception.schema, schema)
        self.assertEqual(len(client.requests), 2)

    def test_invalid_json_twice_raises_with_validation_error(self):
        """Two non-JSON responses \u2192 LLMValidationError, validation_error attached."""
        body_bad_1 = _ollama_chat_body('not json at all')
        body_bad_2 = _ollama_chat_body('still not json')
        client = _FakeClient([body_bad_1, body_bad_2])
        adapter = OllamaLLM(model="m", base_url="http://h:1", cache=self.cache, client=client)
        schema = {"type": "object", "required": ["answer"]}
        with self.assertRaises(LLMValidationError) as ctx:
            adapter.generate_json("s", "u", schema)
        self.assertEqual(ctx.exception.model_output, 'still not json')
        self.assertEqual(len(client.requests), 2)

    def test_strict_parse_only_strips_whitespace(self):
        """Only surrounding whitespace is stripped before json.loads; the entire response must be valid JSON."""
        body = _ollama_chat_body('  \n\t{"answer": 1}\n  ')
        client = _FakeClient([body])
        adapter = OllamaLLM(model="m", base_url="http://h:1", cache=self.cache, client=client)
        parsed, _ = adapter.generate_json("s", "u", {"type": "object", "required": ["answer"]})
        self.assertEqual(parsed, {"answer": 1})
        self.assertEqual(len(client.requests), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()