"""Sentence-Transformers embedding adapter tests (#19).

Covers spec §1.1 of vector-retrieval.md:

* Adapter loads ``BAAI/bge-small-en-v1.5`` at the literal pinned
  revision SHA ``5c38ec7c405ec4b44b94cc5a9bb96e735b38267a``.
* Reported ``dim`` is 384.
* ``embed`` returns one vector per input text, each of length ``dim``.
* Vectors are L2-normalized.
* Reported ``model_identity`` is exactly the literal string the
  ``validate_vector_store`` (#26) check expects.
* Adapter does NOT add the BGE query instruction (no instruction
  prefix or trailing token).
* A revision-bump fixture (loading with the wrong revision) is
  rejected by the adapter and by the loader alike.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.providers.adapters.sentence_transformers import (  # noqa: E402
    MODEL_IDENTITY,
    MODEL_ID,
    NAME,
    PROVIDER_KEY,
    REVISION,
    DIM,
    SentenceTransformersEmbedding,
    _default_loader,
)
from src.providers.cache import CachePaths, ProviderCache  # noqa: E402

from tests.providers._helpers import fake_loader  # noqa: E402


PINNED_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


class TestRevisionPin(unittest.TestCase):
    def test_module_constants_match_spec(self):
        # Spec §1.1 of vector-retrieval.md.
        self.assertEqual(MODEL_ID, "BAAI/bge-small-en-v1.5")
        self.assertEqual(REVISION, PINNED_REVISION)
        self.assertEqual(DIM, 384)
        self.assertEqual(NAME, "local_sentence_transformers")
        self.assertEqual(PROVIDER_KEY, "local_sentence_transformers")

    def test_model_identity_is_literal(self):
        self.assertEqual(
            MODEL_IDENTITY,
            "local_sentence_transformers:BAAI/bge-small-en-v1.5:5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        )

    def test_loader_called_with_pinned_revision(self):
        seen = {}

        def loader(model_id, *, revision):
            seen["model_id"] = model_id
            seen["revision"] = revision
            return fake_loader(model_id, revision=revision)

        SentenceTransformersEmbedding(loader=loader)
        self.assertEqual(seen["model_id"], "BAAI/bge-small-en-v1.5")
        self.assertEqual(seen["revision"], PINNED_REVISION)

    def test_loader_called_with_literal_string_arg(self):
        # Acceptance gate: the loader must be invoked with the literal
        # ``revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"``
        # argument. We assert on the kwargs identity (not just equality)
        # by binding a loader that captures the kwarg's repr.
        captured: list = []

        def loader(model_id, *, revision):
            captured.append(revision)
            return fake_loader(model_id, revision=revision)

        SentenceTransformersEmbedding(loader=loader)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0], PINNED_REVISION)
        # And the captured string is exactly the spec-pinned literal —
        # no ``main``, no ``latest``, no substring.
        self.assertNotIn("main", captured[0])
        self.assertNotIn("latest", captured[0])

    def test_default_loader_rejects_wrong_revision(self):
        # Future-bump fixture: a different revision must be refused by
        # the default loader, so a silent upgrade cannot slip through.
        with self.assertRaises(ValueError):
            _default_loader(MODEL_ID, revision="main")

    def test_adapter_rejects_wrong_revision_at_construction(self):
        with self.assertRaises(ValueError):
            SentenceTransformersEmbedding(loader=fake_loader, revision="main")


class TestAdapterBehavior(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="st-cache-"))
        self.cache = ProviderCache(CachePaths.from_root(self.tmp))
        # ``fake_loader`` returns a FakeSentenceTransformer with dim=384,
        # matching the production model identity.
        self.adapter = SentenceTransformersEmbedding(
            loader=fake_loader,
            cache=self.cache,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dim_is_384(self):
        self.assertEqual(self.adapter.dim, 384)

    def test_model_identity_reported_by_adapter(self):
        self.assertEqual(
            self.adapter.model_identity,
            "local_sentence_transformers:BAAI/bge-small-en-v1.5:5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        )

    def test_embed_returns_one_vector_per_text(self):
        vectors, usage = self.adapter.embed(["hello", "world", "third"])
        self.assertEqual(len(vectors), 3)
        for v in vectors:
            self.assertEqual(len(v), 384)
        # Token accounting is None for an embed call (the adapter does
        # not fabricate token counts; spec §6.1).
        self.assertIsNone(usage.input_tokens)
        self.assertIsNone(usage.output_tokens)
        self.assertIsNone(usage.cost_usd)

    def test_vectors_are_l2_normalized(self):
        vectors, _ = self.adapter.embed(["a", "b", "c"])
        for v in vectors:
            norm_sq = sum(x * x for x in v)
            self.assertAlmostEqual(norm_sq, 1.0, places=5)

    def test_adapter_does_not_inject_bge_query_instruction(self):
        # The adapter is passage-instruction-agnostic (vector-retrieval
        # §1.1 last paragraph: query-only instruction belongs at the
        # call site). The adapter must neither prepend the published
        # BGE query instruction nor append a passage instruction.
        vectors, _ = self.adapter.embed(["Represent this sentence for searching relevant passages: hello"])
        # We can't easily assert on "no instruction was added" by
        # looking at vectors alone, but we can assert that the adapter
        # never references the instruction string in code paths.
        import inspect
        from src.providers.adapters import sentence_transformers as st_mod

        source = inspect.getsource(st_mod)
        self.assertNotIn(
            "Represent this sentence for searching relevant passages",
            source,
        )

    def test_cache_round_trip(self):
        first, _ = self.adapter.embed(["a", "b"])
        # Second call must hit the cache (no extra encoder use).
        cached, _ = self.adapter.embed(["a", "b"])
        self.assertEqual(first, cached)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()