"""Cache-key sensitivity tests (#19).

For each input dimension listed in ``provider-abstractions.md`` §5.1
the test asserts that varying exactly that dimension produces a
distinct cache key while holding all others constant. Also covers:

* ``cache_format_version`` changes the key (per-adapter bump safety);
* switching ``provider.name`` or ``provider.model`` produces disjoint
  namespaces (acceptance gate §2 / spec §9.3);
* ``embed`` is keyed on the ordered full texts list (reordering /
  inserting / deleting / editing any element changes the key);
* ``schema_hash`` participates in ``generate_json`` keys.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.providers.cache import (  # noqa: E402
    CACHE_FORMAT_VERSION,
    CachePaths,
    ProviderCache,
    make_key,
    schema_hash,
)


def _tmp_cache() -> tuple[ProviderCache, Path]:
    td = Path(tempfile.mkdtemp(prefix="provider-cache-"))
    cache = ProviderCache(CachePaths.from_root(td))
    return cache, td


class TestGenerateKeySensitivity(unittest.TestCase):
    """Every input dimension in spec §5.1 must change the cache key."""

    def setUp(self):
        self.cache, self.tmpdir = _tmp_cache()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _key(self, **overrides) -> str:
        base = dict(
            provider_name="ollama",
            provider_model="llama3.1:8b",
            system="sys",
            user="user",
            max_tokens=1024,
            temperature=0.0,
        )
        base.update(overrides)
        return self.cache.key_generate(**base)

    def test_baseline(self):
        # Sanity check: the baseline key is deterministic.
        self.assertEqual(self._key(), self._key())

    def test_provider_name_changes_key(self):
        self.assertNotEqual(self._key(), self._key(provider_name="other"))

    def test_provider_model_changes_key(self):
        self.assertNotEqual(self._key(), self._key(provider_model="other-model"))

    def test_system_changes_key(self):
        self.assertNotEqual(self._key(), self._key(system="different system"))

    def test_user_changes_key(self):
        self.assertNotEqual(self._key(), self._key(user="different user"))

    def test_max_tokens_changes_key(self):
        self.assertNotEqual(self._key(), self._key(max_tokens=512))

    def test_temperature_changes_key(self):
        self.assertNotEqual(self._key(), self._key(temperature=0.5))

    def test_cache_format_version_changes_key(self):
        # The format version is fixed on the cache instance. We build
        # a second instance with a bumped version and confirm the
        # resulting key differs.
        bumped = ProviderCache(self.cache.paths, cache_format_version="2")
        self.assertNotEqual(self.cache.key_generate(
            provider_name="ollama", provider_model="llama3.1:8b",
            system="sys", user="user", max_tokens=1024, temperature=0.0,
        ), bumped.key_generate(
            provider_name="ollama", provider_model="llama3.1:8b",
            system="sys", user="user", max_tokens=1024, temperature=0.0,
        ))


class TestGenerateJSONKeySensitivity(unittest.TestCase):
    def setUp(self):
        self.cache, self.tmpdir = _tmp_cache()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _key(self, **overrides) -> str:
        base = dict(
            provider_name="ollama",
            provider_model="llama3.1:8b",
            system="sys",
            user="user",
            max_tokens=1024,
            temperature=0.0,
            schema={"type": "object", "required": ["answer"]},
        )
        base.update(overrides)
        return self.cache.key_generate_json(**base)

    def test_schema_change_changes_key(self):
        schema_b = {"type": "object", "required": ["answer", "confidence"]}
        self.assertNotEqual(self._key(), self._key(schema=schema_b))

    def test_schema_order_insensitive(self):
        # canonical JSON sort_keys=True → order-independent.
        a = {"type": "object", "required": ["a", "b"], "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
        b = {"properties": {"b": {"type": "integer"}, "a": {"type": "string"}}, "required": ["a", "b"], "type": "object"}
        self.assertEqual(self._key(schema=a), self._key(schema=b))

    def test_schema_hash_is_sha256_of_canonical(self):
        schema = {"type": "object", "required": ["x"]}
        expected = schema_hash(schema)
        # Recompute by hand.
        import hashlib

        canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        self.assertEqual(expected, hashlib.sha256(canonical.encode("utf-8")).hexdigest())


class TestEmbedKeySensitivity(unittest.TestCase):
    def setUp(self):
        self.cache, self.tmpdir = _tmp_cache()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _key(self, **overrides) -> str:
        base = dict(
            provider_name="local_sentence_transformers",
            provider_model="BAAI/bge-small-en-v1.5",
            texts=["hello", "world"],
        )
        base.update(overrides)
        return self.cache.key_embed(**base)

    def test_provider_name_changes_key(self):
        self.assertNotEqual(self._key(), self._key(provider_name="other"))

    def test_provider_model_changes_key(self):
        self.assertNotEqual(self._key(), self._key(provider_model="other-model"))

    def test_texts_change_key(self):
        self.assertNotEqual(self._key(), self._key(texts=["hello", "world!"]))

    def test_texts_reorder_changes_key(self):
        # Embed identity inputs preserve the ordered full texts list;
        # reordering is a meaningful identity change (spec §5.1).
        self.assertNotEqual(self._key(), self._key(texts=["world", "hello"]))

    def test_texts_insert_changes_key(self):
        self.assertNotEqual(self._key(), self._key(texts=["hello", "x", "world"]))

    def test_texts_delete_changes_key(self):
        self.assertNotEqual(self._key(), self._key(texts=["hello"]))

    def test_operation_segregates_namespaces(self):
        # ``operation`` participates in the canonical key blob (the
        # bare make_key constructor takes it explicitly), so the same
        # inputs under different operations cannot collide.
        gen = make_key(
            cache_format_version=CACHE_FORMAT_VERSION,
            operation="generate",
            provider_name="p",
            provider_model="m",
            system="s",
            user="u",
            max_tokens=1,
            temperature=0.0,
        )
        json_ = make_key(
            cache_format_version=CACHE_FORMAT_VERSION,
            operation="generate_json",
            provider_name="p",
            provider_model="m",
            system="s",
            user="u",
            max_tokens=1,
            temperature=0.0,
            schema_canonical_json=schema_hash({"x": 1}),
        )
        embed = make_key(
            cache_format_version=CACHE_FORMAT_VERSION,
            operation="embed",
            provider_name="p",
            provider_model="m",
            texts=["u"],
        )
        self.assertNotEqual(gen, json_)
        self.assertNotEqual(gen, embed)
        self.assertNotEqual(json_, embed)


class TestNamespaceIsolation(unittest.TestCase):
    """Cache namespaces are disjoint across provider / model swaps."""

    def setUp(self):
        self.cache, self.tmpdir = _tmp_cache()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_provider_swap_uses_distinct_files(self):
        cache_a = self.cache
        cache_a.put_generate(
            provider_name="provider_a",
            provider_model="model_a",
            system="sys",
            user="user",
            max_tokens=64,
            temperature=0.0,
            text="alpha",
        )
        cache_a.put_generate(
            provider_name="provider_b",
            provider_model="model_a",
            system="sys",
            user="user",
            max_tokens=64,
            temperature=0.0,
            text="beta",
        )
        files = sorted(p.name for p in (self.tmpdir / "llm").iterdir())
        self.assertEqual(len(files), 2)
        self.assertEqual(len(set(files)), 2)
        self.assertEqual(
            cache_a.get_generate(
                provider_name="provider_a",
                provider_model="model_a",
                system="sys",
                user="user",
                max_tokens=64,
                temperature=0.0,
            ),
            "alpha",
        )
        self.assertEqual(
            cache_a.get_generate(
                provider_name="provider_b",
                provider_model="model_a",
                system="sys",
                user="user",
                max_tokens=64,
                temperature=0.0,
            ),
            "beta",
        )

    def test_model_swap_uses_distinct_files(self):
        cache_a = self.cache
        cache_a.put_generate(
            provider_name="provider_a",
            provider_model="model_a",
            system="sys",
            user="user",
            max_tokens=64,
            temperature=0.0,
            text="alpha",
        )
        cache_a.put_generate(
            provider_name="provider_a",
            provider_model="model_b",
            system="sys",
            user="user",
            max_tokens=64,
            temperature=0.0,
            text="beta",
        )
        keys = set()
        for p in (self.tmpdir / "llm").iterdir():
            keys.add(p.name)
        self.assertEqual(len(keys), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()