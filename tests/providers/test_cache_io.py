"""Cache round-trip and malformed-file handling tests (#19).

Covers:

* ``generate`` / ``generate_json`` / ``embed`` cache write → read
  round-trip.
* Malformed cache entries (non-JSON, wrong shape) are treated as misses
  rather than raising — spec §5.3 prevents a corrupted entry from
  silently poisoning a later run.
* Partial-file atomicity: writes go through ``tempfile`` + ``os.replace``
  so a crash mid-write never leaves a half-written cache file.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.providers.cache import (  # noqa: E402
    CachePaths,
    ProviderCache,
)


def _tmp_cache() -> tuple[ProviderCache, Path]:
    td = Path(tempfile.mkdtemp(prefix="provider-cache-"))
    cache = ProviderCache(CachePaths.from_root(td))
    return cache, td


class TestGenerateRoundTrip(unittest.TestCase):
    def setUp(self):
        self.cache, self.tmpdir = _tmp_cache()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_round_trip(self):
        self.cache.put_generate(
            provider_name="p",
            provider_model="m",
            system="sys",
            user="user",
            max_tokens=64,
            temperature=0.0,
            text="hello",
        )
        self.assertEqual(
            self.cache.get_generate(
                provider_name="p",
                provider_model="m",
                system="sys",
                user="user",
                max_tokens=64,
                temperature=0.0,
            ),
            "hello",
        )

    def test_miss_returns_none(self):
        self.assertIsNone(
            self.cache.get_generate(
                provider_name="p",
                provider_model="m",
                system="sys",
                user="user",
                max_tokens=64,
                temperature=0.0,
            )
        )


class TestGenerateJSONRoundTrip(unittest.TestCase):
    def setUp(self):
        self.cache, self.tmpdir = _tmp_cache()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_round_trip(self):
        schema = {"type": "object", "required": ["answer"]}
        parsed = {"answer": 42}
        self.cache.put_generate_json(
            provider_name="p",
            provider_model="m",
            system="sys",
            user="user",
            max_tokens=64,
            temperature=0.0,
            schema=schema,
            parsed=parsed,
        )
        self.assertEqual(
            self.cache.get_generate_json(
                provider_name="p",
                provider_model="m",
                system="sys",
                user="user",
                max_tokens=64,
                temperature=0.0,
                schema=schema,
            ),
            parsed,
        )

    def test_miss_returns_none(self):
        self.assertIsNone(
            self.cache.get_generate_json(
                provider_name="p",
                provider_model="m",
                system="sys",
                user="user",
                max_tokens=64,
                temperature=0.0,
                schema={"type": "object"},
            )
        )


class TestEmbedRoundTrip(unittest.TestCase):
    def setUp(self):
        self.cache, self.tmpdir = _tmp_cache()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_round_trip(self):
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        self.cache.put_embed(
            provider_name="p",
            provider_model="m",
            texts=["a", "b"],
            vectors=vectors,
        )
        self.assertEqual(
            self.cache.get_embed(
                provider_name="p",
                provider_model="m",
                texts=["a", "b"],
            ),
            vectors,
        )

    def test_miss_returns_none(self):
        self.assertIsNone(
            self.cache.get_embed(
                provider_name="p",
                provider_model="m",
                texts=["a", "b"],
            )
        )


class TestMalformedEntries(unittest.TestCase):
    """Malformed / partial cache files must read as misses, not raise."""

    def setUp(self):
        self.cache, self.tmpdir = _tmp_cache()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_invalid_json_is_treated_as_miss(self):
        # Write a non-JSON blob under the key we'd look up.
        key = self.cache.key_generate(
            provider_name="p",
            provider_model="m",
            system="sys",
            user="user",
            max_tokens=64,
            temperature=0.0,
        )
        target = self.cache.paths.llm / f"{key}.json"
        target.write_text("not json {")
        self.assertIsNone(
            self.cache.get_generate(
                provider_name="p",
                provider_model="m",
                system="sys",
                user="user",
                max_tokens=64,
                temperature=0.0,
            )
        )

    def test_wrong_shape_is_treated_as_miss(self):
        # A JSON file whose ``payload`` is not a string (e.g. a dict)
        # should not be returned as the cache hit for ``generate``.
        key = self.cache.key_generate(
            provider_name="p",
            provider_model="m",
            system="sys",
            user="user",
            max_tokens=64,
            temperature=0.0,
        )
        target = self.cache.paths.llm / f"{key}.json"
        target.write_text(json.dumps({"operation": "generate", "payload": {"oops": True}}))
        self.assertIsNone(
            self.cache.get_generate(
                provider_name="p",
                provider_model="m",
                system="sys",
                user="user",
                max_tokens=64,
                temperature=0.0,
            )
        )

    def test_partial_file_is_treated_as_miss(self):
        key = self.cache.key_embed(
            provider_name="p",
            provider_model="m",
            texts=["a"],
        )
        target = self.cache.paths.embeddings / f"{key}.json"
        target.write_bytes(b'{"payload": [')  # truncated
        self.assertIsNone(
            self.cache.get_embed(
                provider_name="p",
                provider_model="m",
                texts=["a"],
            )
        )


class TestAtomicWrite(unittest.TestCase):
    def setUp(self):
        self.cache, self.tmpdir = _tmp_cache()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_partial_file_left_on_write(self):
        self.cache.put_generate(
            provider_name="p",
            provider_model="m",
            system="sys",
            user="user",
            max_tokens=64,
            temperature=0.0,
            text="hello",
        )
        # Walk the llm subdir: every entry should be a fully-formed
        # ``.json``; no ``.tmp`` artefacts should remain.
        for p in self.cache.paths.llm.iterdir():
            self.assertFalse(p.name.endswith(".tmp"), f"leftover tmp: {p}")
            self.assertTrue(p.name.endswith(".json"))

    def test_put_embed_atomic(self):
        self.cache.put_embed(
            provider_name="p",
            provider_model="m",
            texts=["a"],
            vectors=[[0.1, 0.2]],
        )
        for p in self.cache.paths.embeddings.iterdir():
            self.assertFalse(p.name.endswith(".tmp"))
            self.assertTrue(p.name.endswith(".json"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()