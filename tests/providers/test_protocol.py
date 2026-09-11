"""Protocol and registry tests (#19).

* ``make_llm`` and ``make_embedding`` resolve the configured adapter
  and construct it with the configured model.
* The runtime-checkable protocols accept structural implementations
  (so adapters don't have to import the spec module to be valid).
* ``Usage`` is frozen; mutating a field raises.
* The adapters shipped in #19 register under the documented keys.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.providers import registry as registry_module  # noqa: E402
from src.providers.protocol import (  # noqa: E402
    EmbeddingProvider,
    LLMProvider,
    LLMValidationError,
    Usage,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestProtocolConformance(unittest.TestCase):
    def test_usage_is_frozen(self):
        u = Usage(input_tokens=1, output_tokens=2, cost_usd=0.0)
        with self.assertRaises(Exception):
            u.input_tokens = 5  # type: ignore[misc]

    def test_usage_default_fields_are_none(self):
        u = Usage()
        self.assertIsNone(u.input_tokens)
        self.assertIsNone(u.output_tokens)
        self.assertIsNone(u.cost_usd)

    def test_llm_validation_error_carries_payload(self):
        err = LLMValidationError("not json", {"type": "object"}, ValueError("bad"))
        self.assertEqual(err.model_output, "not json")
        self.assertEqual(err.schema, {"type": "object"})
        self.assertIsInstance(err.validation_error, ValueError)

    def test_llm_provider_protocol_is_runtime_checkable(self):
        from tests.providers._helpers import FakeLLM

        fake = FakeLLM()
        self.assertIsInstance(fake, LLMProvider)

    def test_embedding_provider_protocol_is_runtime_checkable(self):
        from tests.providers._helpers import FakeEmbedding

        fake = FakeEmbedding()
        self.assertIsInstance(fake, EmbeddingProvider)


class TestBuiltinRegistries(unittest.TestCase):
    def setUp(self):
        registry_module._register_builtins()

    def test_ollama_registered(self):
        self.assertIn("ollama", registry_module.LLM_REGISTRY)
        from src.providers.adapters.ollama import OllamaLLM

        self.assertIs(registry_module.LLM_REGISTRY["ollama"], OllamaLLM)

    def test_sentence_transformers_registered(self):
        self.assertIn("local_sentence_transformers", registry_module.EMBEDDING_REGISTRY)
        from src.providers.adapters.sentence_transformers import (
            SentenceTransformersEmbedding,
        )

        self.assertIs(
            registry_module.EMBEDDING_REGISTRY["local_sentence_transformers"],
            SentenceTransformersEmbedding,
        )

    def test_ollama_declares_no_required_secrets(self):
        from src.providers.adapters.ollama import OllamaLLM

        self.assertEqual(getattr(OllamaLLM, "REQUIRED_SECRETS", ()), ())


class TestMakeLLM(unittest.TestCase):
    def setUp(self):
        registry_module._register_builtins()

    def test_unknown_provider_raises_key_error(self):
        with self.assertRaises(KeyError):
            registry_module.make_llm(env={"LLM_PROVIDER": "missing", "LLM_MODEL": "x"})

    def test_known_provider_constructs(self):
        from src.providers.adapters.ollama import OllamaLLM

        # Pass an env without secrets and with_budget=False so the
        # adapter is returned bare (no HTTP cache directory required).
        adapter = registry_module.make_llm(
            env={"LLM_PROVIDER": "ollama", "LLM_MODEL": "llama3.1:8b"},
            with_budget=False,
        )
        self.assertIsInstance(adapter, OllamaLLM)
        self.assertEqual(adapter.name, "ollama")
        self.assertEqual(adapter.model, "llama3.1:8b")

    def test_make_llm_wraps_with_budget_when_cap_set(self):
        from src.providers.budget import BudgetedLLM

        adapter = registry_module.make_llm(
            env={
                "LLM_PROVIDER": "ollama",
                "LLM_MODEL": "llama3.1:8b",
                "LLM_BUDGET_TOKENS": "1000",
            },
            with_budget=True,
        )
        self.assertIsInstance(adapter, BudgetedLLM)


class TestMakeEmbedding(unittest.TestCase):
    def setUp(self):
        registry_module._register_builtins()

    def test_unknown_provider_raises_key_error(self):
        with self.assertRaises(KeyError):
            registry_module.make_embedding(
                env={"EMBEDDING_PROVIDER": "missing", "EMBEDDING_MODEL": "x"}
            )

    def test_known_provider_constructs(self):
        from tests.providers._helpers import fake_loader

        # Use the adapter class directly with a fake loader — the
        # factory's make_embedding goes through the default
        # sentence-transformers import, which is intentionally not
        # installed in CI.
        cls = registry_module.EMBEDDING_REGISTRY["local_sentence_transformers"]
        adapter = cls(loader=fake_loader)
        self.assertEqual(adapter.name, "local_sentence_transformers")
        self.assertEqual(adapter.dim, 384)
        self.assertTrue(hasattr(adapter, "embed"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()