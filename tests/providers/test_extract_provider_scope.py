"""Provider-seam repair (#23) tests.

Covers the two pieces pinned by ``docs/specs/extraction-contract.md``
and ``docs/specs/provider-abstractions.md`` (spec §5.5a + §6.6,
repair #23):

1. ``src.providers.cache.extraction_cache_scope(...)`` integrates into
   the ``generate_json`` cache-key path as a strict superset of the
   canonical key, and produces a byte-identical key when no scope is
   active (spec §5.5 floor is preserved).
2. ``src.providers.adapters.ollama`` maps known ``httpx`` transport
   failures to ``ProviderTransportError(transient=..., status=...,
   cause=...)`` per spec §6.6 — transient on connect/timeout family
   + ``{408, 429, 500, 502, 503, 504}``, non-transient on every other
   4xx (auth, model-not-found, ``409``, ``425``, malformed). No broad
   ``except Exception`` swallowing; non-typed exceptions propagate
   unchanged.

These tests deliberately exercise only the provider seam. They do not
import ``src.extract`` (extraction-side implementation is not part of
this ticket) and they do not touch the prompt-template artifacts
created by the extraction-side implementation.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.providers.cache import (  # noqa: E402
    CachePaths,
    ProviderCache,
    extraction_cache_scope,
    make_key,
    schema_hash,
)
from src.providers.protocol import (  # noqa: E402
    ProviderTransportError,
)


# --- helpers ---------------------------------------------------------


def _six_dims(**overrides) -> dict:
    """Six canonical scope dims; ``overrides`` may replace individual values."""
    base = {
        "chunk_id": "c_0011223344556677",
        "source_text_sha256": "a" * 64,
        "ontology_version": "1",
        "ontology_sha256": "b" * 64,
        "prompt_template_filename": "extract_entities_v1.txt",
        "prompt_template_content_sha256": "c" * 64,
    }
    base.update(overrides)
    return base


# === PART 1: extraction_cache_scope (spec §5.5a) ======================


class TestExtractionScopeKeyIdentity(unittest.TestCase):
    """Spec §5.5a: scope dims mix into ``generate_json`` key; no-scope = canonical."""

    def setUp(self):
        self.cache_root = Path(tempfile.mkdtemp(prefix="provider-cache-scope-"))
        self.cache = ProviderCache(CachePaths.from_root(self.cache_root))

    def tearDown(self):
        shutil.rmtree(self.cache_root, ignore_errors=True)

    def _key(self, **kw) -> str:
        return self.cache.key_generate_json(**kw)

    def _base(self) -> dict:
        return dict(
            provider_name="ollama",
            provider_model="m",
            system="sys",
            user="user",
            max_tokens=64,
            temperature=0.0,
            schema={"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}},
        )

    def test_no_scope_key_matches_canonical_make_key(self):
        # No scope active: the new key path is byte-identical to the
        # canonical ``make_key`` blob, proving the spec §5.5 floor
        # hasn't been weakened.
        base = self._base()
        from_cache = self._key(**base)
        from_make = make_key(
            cache_format_version=self.cache.cache_format_version,
            operation="generate_json",
            provider_name=base["provider_name"],
            provider_model=base["provider_model"],
            system=base["system"],
            user=base["user"],
            max_tokens=base["max_tokens"],
            temperature=base["temperature"],
            schema_canonical_json=schema_hash(base["schema"]),
        )
        self.assertEqual(from_cache, from_make)

    def test_scope_active_changes_key(self):
        base = self._base()
        no_scope = self._key(**base)
        with extraction_cache_scope(**_six_dims()):
            with_scope = self._key(**base)
        self.assertNotEqual(no_scope, with_scope)

    def test_every_six_dim_participates_in_key(self):
        # Changing each of the six dims in turn changes the hash; this
        # proves all six dimensions really are mixed in (not silently
        # dropped by an order-of-iter bug or a typo in the dim key).
        base = self._base()
        with extraction_cache_scope(**_six_dims()):
            anchor = self._key(**base)
        for dim, alt in [
            ("chunk_id", "c_aaaaaaaaaaaaaaaa"),
            ("source_text_sha256", "f" * 64),
            ("ontology_version", "2"),
            ("ontology_sha256", "0" * 64),
            ("prompt_template_filename", "extract_relations_v1.txt"),
            ("prompt_template_content_sha256", "d" * 64),
        ]:
            with extraction_cache_scope(**_six_dims(**{dim: alt})):
                changed = self._key(**base)
            self.assertNotEqual(
                changed,
                anchor,
                f"changing scope dim {dim!r} did not change the cache key",
            )

    def test_scope_dim_order_does_not_change_key(self):
        # Dict insertion order is not significant: two callers passing
        # the same six strings in different orders get the same hash
        # (the helper sorts internally).
        ordered = _six_dims()
        reordered = {k: ordered[k] for k in reversed(list(ordered))}
        with extraction_cache_scope(**ordered):
            k1 = self._key(**self._base())
        with extraction_cache_scope(**reordered):
            k2 = self._key(**self._base())
        self.assertEqual(k1, k2)

    def test_scope_round_trips_through_real_cache(self):
        # The scope-augmented key is what the real cache uses for both
        # get and put — so a put under scope must be a hit on get
        # under the same scope, and a miss on get under a different
        # scope.
        base = self._base()
        parsed = {"a": 42}
        with extraction_cache_scope(**_six_dims()):
            self.cache.put_generate_json(parsed=parsed, **base)
            hit = self.cache.get_generate_json(**base)
            self.assertEqual(hit, parsed)
        # Different scope -> miss.
        with extraction_cache_scope(**_six_dims(chunk_id="c_other")):
            self.assertIsNone(self.cache.get_generate_json(**base))
        # No scope -> miss (proves the scope dims are not folded into
        # the canonical key blob that the prior implementation used).
        self.assertIsNone(self.cache.get_generate_json(**base))

    def test_two_scopes_with_one_dim_different_do_not_collide(self):
        # Two scopes differing in exactly one dim land on different
        # files; this is the namespace-isolation invariant that
        # motivates the strict-superset design.
        base = self._base()
        d1 = _six_dims()
        d2 = _six_dims(chunk_id="c_other")
        with extraction_cache_scope(**d1):
            k1 = self._key(**base)
        with extraction_cache_scope(**d2):
            k2 = self._key(**base)
        self.assertNotEqual(k1, k2)


class TestExtractionScopeSemantics(unittest.TestCase):
    """Context-manager semantics (spec §5.5a, contextvars isolation)."""

    def test_exits_on_exception_and_restores_no_scope(self):
        # Inside the scope the helper must read the dims; after an
        # exception the scope must be popped so subsequent keys revert
        # to the no-scope canonical key.
        base = dict(
            cache_format_version="1",
            provider_name="p",
            provider_model="m",
            system="s",
            user="u",
            max_tokens=1,
            temperature=0.0,
            schema={"type": "object"},
        )
        no_scope = make_key(
            cache_format_version="1",
            operation="generate_json",
            provider_name="p",
            provider_model="m",
            system="s",
            user="u",
            max_tokens=1,
            temperature=0.0,
            schema_canonical_json=schema_hash({"type": "object"}),
        )
        from src.providers.cache import _key_for_generate_json
        with self.assertRaises(RuntimeError):
            with extraction_cache_scope(**_six_dims()):
                # Confirm we are inside the scope.
                inside = _key_for_generate_json(**base)
                self.assertNotEqual(inside, no_scope)
                raise RuntimeError("boom")
        # Outside again: canonical key blob is restored.
        outside = _key_for_generate_json(**base)
        self.assertEqual(outside, no_scope)

    def test_rejects_unknown_key(self):
        with self.assertRaises(ValueError):
            with extraction_cache_scope(
                **_six_dims(unknown_key="x"),
            ):
                pass

    def test_rejects_missing_key(self):
        # Five of the six dims -> missing key.
        dims = _six_dims()
        dims.pop("chunk_id")
        with self.assertRaises(ValueError):
            with extraction_cache_scope(**dims):
                pass

    def test_rejects_non_string_value(self):
        # The six dims are SHA hex / filenames / version; ints are not
        # a valid representation.
        with self.assertRaises(TypeError):
            with extraction_cache_scope(**_six_dims(chunk_id=123)):
                pass

    def test_rejects_empty_kwargs(self):
        with self.assertRaises(ValueError):
            with extraction_cache_scope():
                pass

    def test_nested_scope_raises_runtime_error(self):
        # spec §5.5a: nested extraction scopes are forbidden. Entering
        # a second ``extraction_cache_scope`` while another is active
        # must raise ``RuntimeError`` immediately (fail-closed).
        from src.providers.cache import _key_for_generate_json
        kw = dict(
            cache_format_version="1",
            provider_name="p", provider_model="m", system="s", user="u",
            max_tokens=1, temperature=0.0, schema={"type": "object"},
        )
        outer_dims = _six_dims(chunk_id="c_outer")
        outer_key_inside = None
        with extraction_cache_scope(**outer_dims):
            # Confirm we are inside the outer scope.
            outer_key_inside = _key_for_generate_json(**kw)
            self.assertIsNotNone(outer_key_inside)
            with self.assertRaises(RuntimeError):
                with extraction_cache_scope(**_six_dims(chunk_id="c_inner")):
                    pass

    def test_nested_scope_failure_does_not_disturb_outer(self):
        # After catching the ``RuntimeError`` from the nested attempt,
        # the outer scope must still be active and must still produce
        # exactly the same key it produced before the nested attempt.
        # The nested entry MUST NOT have replaced or mutated the outer
        # scope (spec §5.5a fail-closed).
        from src.providers.cache import _key_for_generate_json
        kw = dict(
            cache_format_version="1",
            provider_name="p", provider_model="m", system="s", user="u",
            max_tokens=1, temperature=0.0, schema={"type": "object"},
        )
        with extraction_cache_scope(**_six_dims(chunk_id="c_outer")):
            outer_before = _key_for_generate_json(**kw)
            try:
                with extraction_cache_scope(**_six_dims(chunk_id="c_inner")):
                    pass
            except RuntimeError:
                pass
            outer_after = _key_for_generate_json(**kw)
            self.assertEqual(outer_before, outer_after)

    def test_outer_scope_resets_after_normal_exit(self):
        # After a normal exit from the (single) scope, no scope is
        # active and the canonical key is restored.
        from src.providers.cache import _key_for_generate_json
        kw = dict(
            cache_format_version="1",
            provider_name="p", provider_model="m", system="s", user="u",
            max_tokens=1, temperature=0.0, schema={"type": "object"},
        )
        no_scope = _key_for_generate_json(**kw)
        with extraction_cache_scope(**_six_dims()):
            self.assertNotEqual(_key_for_generate_json(**kw), no_scope)
        self.assertEqual(_key_for_generate_json(**kw), no_scope)

    def test_outer_scope_resets_after_exception(self):
        # An exception inside the scope body must still pop the scope.
        from src.providers.cache import _key_for_generate_json
        kw = dict(
            cache_format_version="1",
            provider_name="p", provider_model="m", system="s", user="u",
            max_tokens=1, temperature=0.0, schema={"type": "object"},
        )
        no_scope = _key_for_generate_json(**kw)
        with self.assertRaises(RuntimeError):
            with extraction_cache_scope(**_six_dims()):
                raise RuntimeError("boom")
        self.assertEqual(_key_for_generate_json(**kw), no_scope)

    def test_concurrent_contextvars_isolation(self):
        # ``contextvars`` isolation: two independent ``Context.run`` calls
        # each see their own active scope and neither leaks into the
        # other. This is the small deterministic test the spec calls out
        # (no async/network).
        import contextvars
        from src.providers.cache import _key_for_generate_json
        kw = dict(
            cache_format_version="1",
            provider_name="p", provider_model="m", system="s", user="u",
            max_tokens=1, temperature=0.0, schema={"type": "object"},
        )
        ctx_a = contextvars.copy_context()
        ctx_b = contextvars.copy_context()

        def _in_a() -> str:
            with extraction_cache_scope(**_six_dims(chunk_id="c_a")):
                return _key_for_generate_json(**kw)

        def _in_b() -> str:
            with extraction_cache_scope(**_six_dims(chunk_id="c_b")):
                return _key_for_generate_json(**kw)

        k_a = ctx_a.run(_in_a)
        k_b = ctx_b.run(_in_b)
        self.assertNotEqual(k_a, k_b)

        # Outside both contexts the canonical key is produced.
        self.assertNotEqual(_key_for_generate_json(**kw), k_a)
        self.assertNotEqual(_key_for_generate_json(**kw), k_b)


# === PART 3: cache-key floor preservation (spec §5.5) ==============


class TestGenerateKeyUnaffectedByExtractionScope(unittest.TestCase):
    """``generate()`` and ``embed()`` keys ignore the extraction scope.

    Spec §5.5 / §5.5a: the scope is an internal helper for
    ``generate_json`` only. The canonical ``generate`` and ``embed``
    keys must be byte-identical inside and outside an active
    ``extraction_cache_scope`` so the scope never leaks into the
    other adapter paths.
    """

    def _cache(self):
        root = Path(tempfile.mkdtemp(prefix="provider-cache-unaffected-"))
        self.addCleanup(shutil.rmtree, root, True)
        return ProviderCache(CachePaths.from_root(root))

    def _generate_kwargs(self) -> dict:
        return dict(
            provider_name="ollama",
            provider_model="m",
            system="sys",
            user="user",
            max_tokens=64,
            temperature=0.0,
        )

    def _embed_kwargs(self) -> dict:
        return dict(
            provider_name="st",
            provider_model="all-MiniLM-L6-v2",
            texts=["alpha", "beta"],
        )

    def test_generate_key_byte_identical_inside_vs_outside_scope(self):
        cache = self._cache()
        kw = self._generate_kwargs()
        no_scope = cache.key_generate(**kw)
        with extraction_cache_scope(**_six_dims()):
            with_scope = cache.key_generate(**kw)
        self.assertEqual(no_scope, with_scope)

    def test_embed_key_byte_identical_inside_vs_outside_scope(self):
        cache = self._cache()
        kw = self._embed_kwargs()
        no_scope = cache.key_embed(**kw)
        with extraction_cache_scope(**_six_dims()):
            with_scope = cache.key_embed(**kw)
        self.assertEqual(no_scope, with_scope)


class TestCanonicalKeyFloorAndSignatureStability(unittest.TestCase):
    """Spec §5.5 floor + public-signature stability.

    The repair must not weaken the §5.5 canonical identity (no scope
    -> byte-identical pre-#23 key) and must not change the public
    signatures of ``make_key``, ``ProviderCache`` get/put/key, or
    ``LLMProvider``.
    """

    def test_no_scope_generate_json_key_byte_identical_to_make_key(self):
        # ``make_key`` is the single canonical base-key algorithm;
        # ``_key_for_generate_json`` (and therefore the cache) must
        # delegate to it exactly when no scope is active.
        from src.providers.cache import _key_for_generate_json
        kw = dict(
            cache_format_version="1",
            provider_name="p", provider_model="m",
            system="sys", user="user",
            max_tokens=64, temperature=0.0,
            schema={"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}},
        )
        expected = make_key(
            cache_format_version="1",
            operation="generate_json",
            provider_name="p", provider_model="m",
            system="sys", user="user",
            max_tokens=64, temperature=0.0,
            schema_canonical_json=schema_hash(kw["schema"]),
        )
        # Outside any scope the helper key must equal the canonical key.
        self.assertEqual(_key_for_generate_json(**kw), expected)

    def test_identical_scope_round_trips_through_real_cache(self):
        root = Path(tempfile.mkdtemp(prefix="provider-cache-hit-"))
        self.addCleanup(shutil.rmtree, root, True)
        cache = ProviderCache(CachePaths.from_root(root))
        base = dict(
            provider_name="ollama", provider_model="m",
            system="sys", user="user",
            max_tokens=64, temperature=0.0,
            schema={"type": "object"},
        )
        parsed = {"x": 1}
        with extraction_cache_scope(**_six_dims()):
            cache.put_generate_json(parsed=parsed, **base)
            hit = cache.get_generate_json(**base)
            self.assertEqual(hit, parsed)

    def test_each_of_six_dims_changes_key(self):
        # Each one of the six required dims, independently changed,
        # misses the cache.
        from src.providers.cache import _key_for_generate_json
        kw = dict(
            cache_format_version="1",
            provider_name="p", provider_model="m",
            system="s", user="u",
            max_tokens=1, temperature=0.0,
            schema={"type": "object"},
        )
        anchor_dims = _six_dims()
        with extraction_cache_scope(**anchor_dims):
            anchor = _key_for_generate_json(**kw)
        for dim, alt in [
            ("chunk_id", "c_aaaaaaaaaaaaaaaa"),
            ("source_text_sha256", "f" * 64),
            ("ontology_version", "2"),
            ("ontology_sha256", "0" * 64),
            ("prompt_template_filename", "extract_relations_v1.txt"),
            ("prompt_template_content_sha256", "d" * 64),
        ]:
            with extraction_cache_scope(**_six_dims(**{dim: alt})):
                changed = _key_for_generate_json(**kw)
            self.assertNotEqual(
                changed, anchor,
                f"changing scope dim {dim!r} did not change the cache key",
            )

    def test_make_key_public_signature_unchanged(self):
        import inspect
        from src.providers.cache import make_key
        params = list(inspect.signature(make_key).parameters)
        self.assertEqual(
            params,
            [
                "cache_format_version",
                "operation",
                "provider_name",
                "provider_model",
                "system",
                "user",
                "max_tokens",
                "temperature",
                "schema_canonical_json",
                "texts",
            ],
        )

    def test_provider_cache_public_signatures_unchanged(self):
        import inspect
        from src.providers.cache import ProviderCache
        for method, expected_params in [
            ("get_generate", ["self", "provider_name", "provider_model", "system", "user", "max_tokens", "temperature"]),
            ("get_generate_json", ["self", "provider_name", "provider_model", "system", "user", "max_tokens", "temperature", "schema"]),
            ("get_embed", ["self", "provider_name", "provider_model", "texts"]),
            ("put_generate", ["self", "provider_name", "provider_model", "system", "user", "max_tokens", "temperature", "text"]),
            ("put_generate_json", ["self", "provider_name", "provider_model", "system", "user", "max_tokens", "temperature", "schema", "parsed"]),
            ("put_embed", ["self", "provider_name", "provider_model", "texts", "vectors"]),
            ("key_generate", ["self", "provider_name", "provider_model", "system", "user", "max_tokens", "temperature"]),
            ("key_generate_json", ["self", "provider_name", "provider_model", "system", "user", "max_tokens", "temperature", "schema"]),
            ("key_embed", ["self", "provider_name", "provider_model", "texts"]),
        ]:
            params = list(inspect.signature(getattr(ProviderCache, method)).parameters)
            self.assertEqual(params, expected_params, f"{method} signature drifted")

    def test_llm_provider_signatures_unchanged(self):
        import inspect
        from src.providers.protocol import LLMProvider
        for method, expected_params in [
            ("generate", ["self", "system", "user", "max_tokens", "temperature"]),
            ("generate_json", ["self", "system", "user", "schema", "max_tokens", "temperature"]),
        ]:
            params = list(inspect.signature(getattr(LLMProvider, method)).parameters)
            self.assertEqual(params, expected_params, f"LLMProvider.{method} signature drifted")
        # EmbeddingProvider.embed
        from src.providers.protocol import EmbeddingProvider
        params = list(inspect.signature(EmbeddingProvider.embed).parameters)
        self.assertEqual(params, ["self", "texts"])


# === PART 2: Ollama transport error mapping (spec §6.6) ==============


class _FakeResponse:
    """Minimal httpx.Response shape for the adapter's request helper.

    The adapter only needs ``raise_for_status`` and ``json``. We do
    not subclass httpx.Response because the tests want to control the
    status code precisely and exercise the mapping without relying on
    httpx's internal transport machinery.
    """

    def __init__(self, status_code: int, body: dict | None = None):
        self.status_code = status_code
        self._body = body if body is not None else {}

    def raise_for_status(self) -> None:
        # Mirror httpx's contract: 4xx / 5xx raise, 1xx / 2xx / 3xx do not.
        if 400 <= self.status_code:
            request = httpx.Request("POST", "http://test/api/chat")
            response = httpx.Response(self.status_code, json={}, request=request)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=request, response=response
            )

    def json(self):
        return self._body


def _o_chat_response(text: str, prompt_tokens: int = 1, eval_tokens: int = 2):
    return {"message": {"content": text}, "prompt_eval_count": prompt_tokens, "eval_count": eval_tokens}


class TestOllamaTransportErrorMapping(unittest.TestCase):
    """Spec §6.6 frozen mapping for the Ollama adapter."""

    def _adapter(self, client):
        from src.providers.adapters.ollama import OllamaLLM

        return OllamaLLM(
            model="m",
            base_url="http://test",
            timeout_s=1.0,
            client=client,
            cache=None,
        )

    # --- transient transport-layer (no HTTP status) -----------------

    def test_connect_family_exceptions_are_transient(self):
        for exc in [
            httpx.ConnectError("nope", request=httpx.Request("POST", "http://t")),
            httpx.ConnectTimeout("to", request=httpx.Request("POST", "http://t")),
            httpx.ReadTimeout("to", request=httpx.Request("POST", "http://t")),
            httpx.WriteTimeout("to", request=httpx.Request("POST", "http://t")),
            httpx.PoolTimeout("to", request=httpx.Request("POST", "http://t")),
            httpx.NetworkError("net", request=httpx.Request("POST", "http://t")),
            httpx.RemoteProtocolError("proto", request=httpx.Request("POST", "http://t")),
        ]:
            client = MagicMock()
            client.post = MagicMock(side_effect=exc)
            adapter = self._adapter(client)
            with self.assertRaises(ProviderTransportError) as cm:
                adapter.generate("s", "u")
            self.assertTrue(cm.exception.transient, msg=f"{type(exc).__name__} should be transient")
            self.assertIsNone(cm.exception.status)
            self.assertIs(cm.exception.cause, exc)

    def test_transient_http_statuses_are_transient(self):
        for code in (408, 429, 500, 502, 503, 504):
            client = MagicMock()
            client.post = MagicMock(return_value=_FakeResponse(code, _o_chat_response("hi")))
            adapter = self._adapter(client)
            with self.assertRaises(ProviderTransportError) as cm:
                adapter.generate("s", "u")
            self.assertTrue(cm.exception.transient, msg=f"status {code} should be transient")
            self.assertEqual(cm.exception.status, code)
            self.assertIsInstance(cm.exception.cause, httpx.HTTPStatusError)

    def test_other_4xx_are_not_transient(self):
        # 400 / 401 / 403 / 404 / 409 / 418 / 425 — every 4xx outside
        # the transient set — must be non-transient. 409 and 425 are
        # specifically called out by spec §6.6.
        for code in (400, 401, 403, 404, 409, 418, 425):
            client = MagicMock()
            client.post = MagicMock(return_value=_FakeResponse(code, _o_chat_response("hi")))
            adapter = self._adapter(client)
            with self.assertRaises(ProviderTransportError) as cm:
                adapter.generate("s", "u")
            self.assertFalse(cm.exception.transient, msg=f"status {code} must NOT be transient")
            self.assertEqual(cm.exception.status, code)
            self.assertIsInstance(cm.exception.cause, httpx.HTTPStatusError)

    def test_2xx_does_not_raise(self):
        # The mapping is error-path only; 2xx responses must reach the
        # caller as a normal ``(text, usage)`` tuple.
        client = MagicMock()
        client.post = MagicMock(return_value=_FakeResponse(200, _o_chat_response("hello")))
        adapter = self._adapter(client)
        text, usage = adapter.generate("s", "u")
        self.assertEqual(text, "hello")
        self.assertEqual(usage.input_tokens, 1)
        self.assertEqual(usage.output_tokens, 2)

    def test_cause_is_the_original_sdk_exception(self):
        # ``ProviderTransportError.cause`` must be the underlying SDK
        # exception, not a stringified message or wrapped copy.
        exc = httpx.ConnectError("boom", request=httpx.Request("POST", "http://t"))
        client = MagicMock()
        client.post = MagicMock(side_effect=exc)
        adapter = self._adapter(client)
        with self.assertRaises(ProviderTransportError) as cm:
            adapter.generate("s", "u")
        self.assertIsInstance(cm.exception.cause, httpx.ConnectError)
        # Python ``__cause__`` is also wired for traceback chaining.
        self.assertIsInstance(cm.exception.__cause__, httpx.ConnectError)

    def test_non_transport_exception_propagates_unchanged(self):
        # No broad ``except Exception``: a non-httpx exception raised
        # inside ``client.post`` must reach the caller unchanged, not
        # be wrapped in ``ProviderTransportError``.
        class WeirdError(Exception):
            pass

        client = MagicMock()
        client.post = MagicMock(side_effect=WeirdError("nope"))
        adapter = self._adapter(client)
        with self.assertRaises(WeirdError):
            adapter.generate("s", "u")

    def test_generate_json_propagates_transport_errors_without_schema_retry(self):
        # The retry path in ``generate_json`` is owned by the schema
        # layer (spec §2). A transient ``ProviderTransportError`` must
        # reach the caller so the caller's extraction-side bounded
        # retry can decide — not be silently re-issued.
        client = MagicMock()
        client.post = MagicMock(return_value=_FakeResponse(503, _o_chat_response("hi")))
        adapter = self._adapter(client)
        with self.assertRaises(ProviderTransportError) as cm:
            adapter.generate_json("s", "u", {"type": "object"})
        self.assertTrue(cm.exception.transient)
        # Exactly one HTTP attempt — transport errors do NOT trigger
        # the schema-retry path.
        self.assertEqual(client.post.call_count, 1)

    def test_generate_json_non_transient_4xx_reaches_caller(self):
        # Non-transient transport error on attempt 1 must reach the
        # caller without a retry (extraction retry policy is owned
        # outside this module; the adapter MUST NOT silently classify
        # permanent failures as recoverable).
        client = MagicMock()
        client.post = MagicMock(return_value=_FakeResponse(404, _o_chat_response("hi")))
        adapter = self._adapter(client)
        with self.assertRaises(ProviderTransportError) as cm:
            adapter.generate_json("s", "u", {"type": "object"})
        self.assertFalse(cm.exception.transient)
        self.assertEqual(client.post.call_count, 1)

    def test_retry_path_also_maps_transport_errors(self):
        # The internal ``_chat_json_retry`` path is reached only on
        # schema-validation failure, but if attempt 2 itself hits a
        # transport error it must still surface as
        # ``ProviderTransportError`` (not ``LLMValidationError``).
        client = MagicMock()
        # Attempt 1: success but invalid JSON (forces schema retry).
        # Attempt 2: transport failure.
        bad_json_response = _FakeResponse(200, _o_chat_response("not json"))
        transport_fail = httpx.ReadTimeout("to", request=httpx.Request("POST", "http://t"))
        client.post = MagicMock(side_effect=[bad_json_response, transport_fail])
        adapter = self._adapter(client)
        with self.assertRaises(ProviderTransportError) as cm:
            adapter.generate_json("s", "u", {"type": "object"})
        self.assertTrue(cm.exception.transient)
        self.assertIsInstance(cm.exception.cause, httpx.ReadTimeout)


class TestProviderTransportErrorDataclass(unittest.TestCase):
    """The exception object itself behaves correctly."""

    def test_str_includes_transient_and_status(self):
        e = ProviderTransportError(transient=True, status=503)
        self.assertIn("transient=True", str(e))
        self.assertIn("status=503", str(e))

    def test_status_omitted_when_none(self):
        e = ProviderTransportError(transient=True)
        self.assertNotIn("status=", str(e))

    def test_cause_attr_is_explicit(self):
        c = RuntimeError("orig")
        e = ProviderTransportError(transient=False, status=400, cause=c)
        self.assertIs(e.cause, c)

    def test_is_an_exception_subclass(self):
        # Callers may use ``except Exception`` or ``except`` on a
        # specific base; the class must cooperate.
        self.assertTrue(issubclass(ProviderTransportError, Exception))
        with self.assertRaises(ProviderTransportError):
            raise ProviderTransportError(transient=True)
        with self.assertRaises(Exception):
            raise ProviderTransportError(transient=False, status=404)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
