"""``validate_provider_config`` tests (#19).

Covers spec §3.4:

* ``LLM_PROVIDER`` is registered (else reject).
* Active adapter's required secrets are present.
* ``LLM_MODEL`` / ``EMBEDDING_MODEL`` are non-empty.
* Optional numeric settings parse cleanly.
* Unrelated-provider secrets don't cause failure (spec §3.3).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.providers import registry as registry_module  # noqa: E402
from src.providers import validation  # noqa: E402
from src.providers.validation import validate_provider_config  # noqa: E402

from tests.providers._helpers import FakeLLM  # noqa: E402


def _adapter_for(name: str):
    return registry_module.LLM_REGISTRY[name]


def _embed_adapter_for(name: str):
    return registry_module.EMBEDDING_REGISTRY[name]


def _reset():
    registry_module._register_builtins()


class _OptionalSecretAdapter(FakeLLM):
    REQUIRED_SECRETS = ("REQUIRED_TOKEN_X",)


class TestValidateProviderConfig(unittest.TestCase):
    def setUp(self):
        _reset()

    def _good_env(self, **overrides):
        env = {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "llama3.1:8b",
            "EMBEDDING_PROVIDER": "local_sentence_transformers",
            "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        }
        env.update(overrides)
        return env

    def _report(self, env):
        return validate_provider_config(
            env,
            llm_registry_keys=list(registry_module.LLM_REGISTRY),
            embedding_registry_keys=list(registry_module.EMBEDDING_REGISTRY),
            llm_adapter_for_provider=_adapter_for,
            embedding_adapter_for_provider=_embed_adapter_for,
        )

    def test_clean_env_passes(self):
        report = self._report(self._good_env())
        self.assertTrue(report.ok, msg=str(report.violations))

    def test_missing_llm_provider(self):
        env = self._good_env()
        del env["LLM_PROVIDER"]
        report = self._report(env)
        self.assertFalse(report.ok)
        self.assertTrue(any(v.rule == "llm_provider" for v in report.violations))

    def test_unknown_llm_provider(self):
        report = self._report(self._good_env(LLM_PROVIDER="nonexistent"))
        self.assertFalse(report.ok)
        self.assertTrue(any(v.rule == "llm_provider" for v in report.violations))

    def test_missing_llm_model(self):
        report = self._report(self._good_env(LLM_MODEL=""))
        self.assertFalse(report.ok)
        self.assertTrue(any(v.rule == "llm_model" for v in report.violations))

    def test_whitespace_model_rejected(self):
        report = self._report(self._good_env(LLM_MODEL="   "))
        self.assertFalse(report.ok)

    def test_missing_embedding_provider(self):
        env = self._good_env()
        del env["EMBEDDING_PROVIDER"]
        report = self._report(env)
        self.assertFalse(report.ok)
        self.assertTrue(any(v.rule == "embedding_provider" for v in report.violations))

    def test_unknown_embedding_provider(self):
        report = self._report(self._good_env(EMBEDDING_PROVIDER="nonexistent"))
        self.assertFalse(report.ok)
        self.assertTrue(any(v.rule == "embedding_provider" for v in report.violations))

    def test_missing_embedding_model(self):
        report = self._report(self._good_env(EMBEDDING_MODEL=""))
        self.assertFalse(report.ok)
        self.assertTrue(any(v.rule == "embedding_model" for v in report.violations))

    def test_optional_numeric_parses(self):
        report = self._report(
            self._good_env(
                LLM_BUDGET_USD="1.5",
                LLM_BUDGET_TOKENS="100",
                LLM_PROVIDER_TIMEOUT_S="30",
                EMBEDDING_BATCH_SIZE="32",
            )
        )
        self.assertTrue(report.ok, msg=str(report.violations))

    def test_optional_numeric_invalid_value(self):
        report = self._report(self._good_env(LLM_BUDGET_USD="not-a-number"))
        self.assertFalse(report.ok)
        self.assertTrue(any(v.rule == "optional_numeric" for v in report.violations))

    def test_optional_numeric_negative(self):
        report = self._report(self._good_env(LLM_PROVIDER_TIMEOUT_S="-1"))
        self.assertFalse(report.ok)

    def test_token_cap_must_be_integer(self):
        report = self._report(self._good_env(LLM_BUDGET_TOKENS="3.5"))
        self.assertFalse(report.ok)
        self.assertTrue(any(v.rule == "optional_numeric" for v in report.violations))

    def test_active_adapter_missing_secret_fails(self):
        # Inject a fake LLM adapter that declares REQUIRED_SECRETS.
        registry_module.LLM_REGISTRY["needs_secret"] = _OptionalSecretAdapter
        try:
            env = self._good_env(LLM_PROVIDER="needs_secret")
            report = self._report(env)
            self.assertFalse(report.ok)
            self.assertTrue(any(v.rule == "missing_secret" for v in report.violations))
            # Adding the secret passes.
            env["REQUIRED_TOKEN_X"] = "set"
            report = self._report(env)
            self.assertTrue(report.ok, msg=str(report.violations))
        finally:
            registry_module.LLM_REGISTRY.pop("needs_secret", None)

    def test_unrelated_provider_secret_does_not_fail(self):
        # Spec §3.3: "Mismatched keys are not an error at module load".
        # ANTHROPIC_API_KEY is irrelevant because we selected Ollama.
        env = self._good_env()
        env["ANTHROPIC_API_KEY"] = ""  # empty / unset
        env["OPENAI_API_KEY"] = ""
        report = self._report(env)
        self.assertTrue(report.ok, msg=str(report.violations))

    def test_ollama_declares_no_secrets(self):
        # Direct check: spec §3.3 Ollama has no API-key requirement.
        cls = registry_module.LLM_REGISTRY["ollama"]
        self.assertEqual(getattr(cls, "REQUIRED_SECRETS", ()), ())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()