"""Shared test helpers for ``tests/providers/``.

Lightweight helpers used by multiple test files: a controllable fake
LLM that records its calls and returns scripted responses, plus a
fake embedding adapter. The provider layer uses Protocol typing, so
structural conformance is what matters; these helpers implement the
right method shapes and can be registered into the production
registries (cleared via :func:`reset_for_tests`) to drive
namespace-isolation and cap-crossing tests without mocks.

The sentence-transformers adapter for #19 loads the real model by
default, which would require a network download on first run. All
tests inject a fake ``loader`` so the model object is a plain
stand-in. Tests that touch the real loader live in
``smoke.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.providers import registry as registry_module  # noqa: E402
from src.providers.protocol import (  # noqa: E402
    EmbeddingProvider,
    LLMProvider,
    Usage,
)


# --- Fake adapters ---------------------------------------------------


class FakeLLM(LLMProvider):
    """Scripted LLM that records every call.

    ``generate_scripts`` is a list of (text, Usage) pairs returned in
    order; once exhausted the last pair is repeated. ``generate_json_scripts``
    follows the same pattern but returns dicts. ``calls`` accumulates
    every (system, user, kwargs) tuple so tests can assert on what was
    sent. ``json_attempts`` records how many times ``generate_json``
    was invoked per call site to drive the exactly-once retry test.
    """

    def __init__(
        self,
        name: str = "fake",
        model: str = "fake-1",
        generate_scripts=None,
        generate_json_scripts=None,
    ) -> None:
        self.name = name
        self.model = model
        self._generate_scripts = list(generate_scripts or [("default", Usage())])
        self._generate_json_scripts = list(
            generate_json_scripts or [({}, Usage())]
        )
        self.generate_calls: list = []
        self.generate_json_calls: list = []
        self.json_attempts: list = []

    def generate(self, system, user, *, max_tokens=1024, temperature=0.0):
        self.generate_calls.append(
            (system, user, {"max_tokens": max_tokens, "temperature": temperature})
        )
        return self._generate_scripts[min(len(self.generate_calls) - 1, len(self._generate_scripts) - 1)]

    def generate_json(self, system, user, schema, *, max_tokens=1024, temperature=0.0):
        self.generate_json_calls.append(
            (system, user, schema, {"max_tokens": max_tokens, "temperature": temperature})
        )
        idx = len(self.json_attempts)
        self.json_attempts.append(idx)
        script_idx = min(idx, len(self._generate_json_scripts) - 1)
        return self._generate_json_scripts[script_idx]


class FakeEmbedding(EmbeddingProvider):
    """Deterministic embedding that maps each text to its hash-bucketed vector.

    All returned vectors are L2-normalized (length matches ``dim``) so
    tests can assert on the normalization invariant in isolation.
    """

    def __init__(self, name: str = "fake-embed", model: str = "fake-embed-1", dim: int = 4):
        self.name = name
        self.model = model
        self.dim = dim
        self.embed_calls: list = []

    def embed(self, texts):
        self.embed_calls.append(list(texts))
        vecs = []
        for t in texts:
            seed = abs(hash(t)) % (10 ** 9) or 1
            raw = [(seed >> (i * 4)) & 0xF for i in range(self.dim)]
            # L2-normalize.
            norm = sum(x * x for x in raw) ** 0.5 or 1.0
            vecs.append([x / norm for x in raw])
        return vecs, Usage()


class FakeSentenceTransformer:
    """Stand-in for the real ``SentenceTransformer`` model object.

    Implements ``encode`` with a numpy-free ``list[list[float]]`` shape
    so the unit tests do not require the sentence-transformers /
    numpy dependencies. The adapter's ``_encode`` consumes each row
    via ``list(map(float, row))`` and feeds it through
    :func:`_l2_normalize`, which is what production does on the real
    encoder output.
    """

    def __init__(self, model_id: str, *, revision: str, dim: int = 4):
        self.model_id = model_id
        self.revision = revision
        self.dim = dim

    def encode(self, texts, *, batch_size, convert_to_numpy, normalize_embeddings, show_progress_bar):
        rows = []
        for t in texts:
            seed = abs(hash(t)) % (10 ** 9) or 1
            row = [float((seed >> (i * 4)) & 0xF) for i in range(self.dim)]
            rows.append(row)
        return rows


def fake_loader(model_id: str, *, revision: str):
    return FakeSentenceTransformer(model_id, revision=revision, dim=384)


# --- Registry utilities ---------------------------------------------


def reset_and_register(factories):
    """Clear the registries and register the supplied factories.

    ``factories`` is a dict ``{"llm": {key: cls, ...},
    "embedding": {key: cls, ...}}``. Used by tests that need to inject
    a second LLM/embedding adapter to exercise provider-namespace
    isolation without depending on a second production SDK.
    """
    registry_module.reset_for_tests()
    for k, cls in factories.get("llm", {}).items():
        registry_module.LLM_REGISTRY[k] = cls
    for k, cls in factories.get("embedding", {}).items():
        registry_module.EMBEDDING_REGISTRY[k] = cls


def restore_builtins():
    """Re-register the built-in Ollama / sentence-transformers adapters."""
    registry_module.reset_for_tests()
    # Trigger the lazy registration in registry.
    registry_module._register_builtins()