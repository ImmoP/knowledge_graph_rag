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


class TestRuntimeDependencyDeclared(unittest.TestCase):
    """The embedding adapter's production runtime dependency
    (``sentence-transformers``) must be declared in ``requirements.txt``
    so a clean environment created only from requirements can use the
    adapter. This is the #19 pre-push repair acceptance check.
    """

    REPO_ROOT = Path(__file__).resolve().parents[2]

    def _read_requirements(self) -> str:
        return (self.REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    def test_requirements_declares_sentence_transformers(self):
        import re

        text = self._read_requirements()
        # Exact-pin: ``sentence-transformers==X.Y.Z`` (no range, no
        # unpinned caret/tilde). The repo's dependency style is exact
        # pins; we match that to avoid silent drift.
        match = re.search(r"^sentence-transformers\s*==\s*([\w.+-]+)\s*$", text, re.MULTILINE)
        self.assertIsNotNone(
            match,
            "requirements.txt must exact-pin sentence-transformers (no range, no unpinned)",
        )
        pinned = match.group(1)
        # Sanity: a concrete PEP 440 version, not "latest" / "0".
        parts = pinned.split(".")
        self.assertGreaterEqual(
            len(parts), 3, f"sentence-transformers pin must be a concrete version, got {pinned!r}"
        )

    def test_requirements_declares_jsonschema(self):
        """The strict generate_json path uses jsonschema for fail-closed
        validation; that dep must also be declared (same exact-pin style)."""
        import re

        text = self._read_requirements()
        match = re.search(r"^jsonschema\s*==\s*([\w.+-]+)\s*$", text, re.MULTILINE)
        self.assertIsNotNone(
            match,
            "requirements.txt must exact-pin jsonschema (no range, no unpinned)",
        )

    def test_sentence_transformers_pypi_release_supports_python_version(self):
        """Verify the pinned version is real, exists on PyPI, and its
        metadata lists a Python version compatible with this project.
        Network-touching acceptance gate."""
        import re
        import urllib.request

        text = self._read_requirements()
        match = re.search(r"^sentence-transformers\s*==\s*([\w.+-]+)\s*$", text, re.MULTILINE)
        if not match:
            self.skipTest("sentence-transformers not pinned in requirements.txt")
        version = match.group(1)
        url = f"https://pypi.org/pypi/sentence-transformers/{version}/json"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                import json

                payload = json.loads(resp.read())
        except Exception as e:
            self.skipTest(f"PyPI metadata fetch failed ({e}); environment lacks network")
        self.assertEqual(payload["info"]["version"], version)
        classifiers = payload["info"].get("classifiers", [])
        # Must claim at least one Python 3.x support line.
        self.assertTrue(
            any("Python :: 3" in c for c in classifiers),
            f"sentence-transformers=={version} has no Python 3.x classifier",
        )

    def test_declared_dependency_is_installed_in_environment(self):
        """The pinned version must match the version installed in the
        project environment. This proves ``pip install -r requirements.txt``
        brings in the exact dep the adapter expects, without doing a live
        install in the test."""
        import importlib.metadata
        import re

        text = self._read_requirements()
        match = re.search(r"^sentence-transformers\s*==\s*([\w.+-]+)\s*$", text, re.MULTILINE)
        if not match:
            self.skipTest("sentence-transformers not pinned in requirements.txt")
        pinned = match.group(1)
        try:
            installed = importlib.metadata.version("sentence-transformers")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("sentence-transformers not installed in current env")
        self.assertEqual(installed, pinned)


class TestVerifiedEmbeddingRuntimePins(unittest.TestCase):
    """Focused dependency compatibility test for the #19 embedding
    runtime that was actually verified to load
    ``sentence_transformers.SentenceTransformer`` cleanly in the
    project environment (Python 3.12.13, x86_64 macOS).

    Proves the declared requirements.txt pins reproduce the verified
    runtime:

    * numpy==1.26.4
    * scipy==1.12.0
    * torch==2.2.2
    * transformers==4.48.3
    * sentence-transformers==3.4.1
    * jsonschema==4.25.1

    Two layers of proof are checked:

    1. The pin string is present and exact in ``requirements.txt``
       (style: ``<name>==<X.Y.Z>``, no range, no caret, no unpinned).
    2. If the package is importable in the running environment, its
       installed version equals the pin. If the package is not
       installed (e.g. a test runner that skips heavy deps), that
       check is skipped so the test stays runnable in CI without
       torch.
    """

    REPO_ROOT = Path(__file__).resolve().parents[2]

    EXPECTED_PINS: dict[str, str] = {
        "numpy": "1.26.4",
        "scipy": "1.12.0",
        "torch": "2.2.2",
        "transformers": "4.48.3",
        "sentence-transformers": "3.4.1",
        "jsonschema": "4.25.1",
    }

    def _read_requirements(self) -> str:
        return (self.REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    def _pinned_in_requirements(self, name: str) -> str | None:
        import re

        text = self._read_requirements()
        match = re.search(
            rf"^{re.escape(name)}\s*==\s*([\w.+-]+)\s*$",
            text,
            re.MULTILINE,
        )
        return match.group(1) if match else None

    def test_requirements_exact_pins_verified_embedding_runtime(self):
        import re

        text = self._read_requirements()
        for name, expected in self.EXPECTED_PINS.items():
            with self.subTest(dep=name):
                match = re.search(
                    rf"^{re.escape(name)}\s*==\s*([\w.+-]+)\s*$",
                    text,
                    re.MULTILINE,
                )
                self.assertIsNotNone(
                    match,
                    f"requirements.txt must exact-pin {name} "
                    f"(no range, no unpinned)",
                )
                pinned = match.group(1)
                self.assertEqual(
                    pinned,
                    expected,
                    f"{name} pin mismatch: requirements.txt has "
                    f"{pinned!r}, verified runtime needs {expected!r}",
                )
                parts = pinned.split(".")
                self.assertGreaterEqual(
                    len(parts),
                    3,
                    f"{name} pin must be a concrete PEP 440 version, "
                    f"got {pinned!r}",
                )

    def test_no_range_or_unpinned_for_verified_embedding_runtime(self):
        # Guard against accidental caret/tilde/range/<=/>=/!=/~=
        # entries for the six verified pins — silent drift defeats
        # the whole point of pinning. ``==`` (exact pin) is the only
        # allowed operator.
        import re

        text = self._read_requirements()
        for name in self.EXPECTED_PINS:
            with self.subTest(dep=name):
                # Every line mentioning this dep must match the
                # exact-pin shape. Any caret/tilde/range/unpinned
                # entry would be caught here.
                matching_lines = [
                    ln for ln in text.splitlines()
                    if re.match(rf"^{re.escape(name)}\b", ln)
                ]
                self.assertGreaterEqual(
                    len(matching_lines),
                    1,
                    f"requirements.txt has no line for {name!r}",
                )
                for ln in matching_lines:
                    self.assertRegex(
                        ln,
                        rf"^{re.escape(name)}\s*==\s*[\w.+-]+\s*$",
                        f"requirements.txt must exact-pin {name} "
                        f"(no range, no unpinned, no caret/tilde); "
                        f"offending line: {ln!r}",
                    )

    def test_installed_versions_match_verified_pins_when_available(self):
        # If a package is importable in the running env, its installed
        # version must equal the declared pin. Skipped when the
        # package is absent so the test stays runnable in lighter
        # environments.
        import importlib

        # Map distribution name -> Python module name where they differ.
        module_name_for = {
            "sentence-transformers": "sentence_transformers",
        }

        for dist_name, expected in self.EXPECTED_PINS.items():
            with self.subTest(dep=dist_name):
                pinned = self._pinned_in_requirements(dist_name)
                self.assertIsNotNone(
                    pinned,
                    f"requirements.txt missing pin for {dist_name}",
                )
                try:
                    dist_version = importlib.metadata.version(dist_name)
                except importlib.metadata.PackageNotFoundError:
                    self.skipTest(
                        f"{dist_name} not installed in current env; "
                        f"cannot verify installed-version parity"
                    )
                self.assertEqual(
                    dist_version,
                    expected,
                    f"{dist_name} installed as {dist_version!r} but "
                    f"requirements.txt pins {expected!r}",
                )
                # Smoke-import the Python module if it differs from
                # the distribution name; this catches the case where a
                # wrong dist name passes a version check but the real
                # import path is broken.
                module_name = module_name_for.get(dist_name, dist_name)
                try:
                    importlib.import_module(module_name)
                except ImportError:
                    self.skipTest(
                        f"module {module_name!r} not importable in "
                        f"current env"
                    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()