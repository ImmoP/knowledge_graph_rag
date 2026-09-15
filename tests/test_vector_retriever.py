"""Tests for ``src.retrieval.vector`` (#27).

Covers the ``vector-retrieval.md`` §4 contract and the issue's
acceptance gates: result shape, exact-cosine score ordering with
``chunk_id`` tie-break, empty store, under-filled top-k, ``k``
validation ordering (before any database or embedding call), BGE
query-instruction application (and query/passage asymmetry),
stale/mixed embedding-identity rejection, malformed embedding output,
determinism, and the no-ANN / no-threshold search shape.

No network, no model download, no live pgvector: the embedding provider
and the pgvector driver are deterministic in-memory fakes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.providers.protocol import Usage  # noqa: E402
from src.retrieval.vector import (  # noqa: E402
    PINNED_DIM,
    PINNED_MODEL_IDENTITY,
    QUERY_INSTRUCTION,
    SEARCH_SQL,
    STORE_STATS_SQL,
    PgVectorRetriever,
    StaleVectorStoreError,
    VectorRetrievalError,
    VectorRetrievalHit,
    VectorRetriever,
    _PsycopgSearchDriver,
)

PINNED_IDENTITY = (
    "local_sentence_transformers:BAAI/bge-small-en-v1.5:"
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)
DIM = 384


# ---------------------------------------------------------------------------
# Deterministic fakes.
# ---------------------------------------------------------------------------


def _vector_for(text: str, dim: int = DIM) -> list[float]:
    """Deterministic pseudo-embedding: SHA256-seeded, L2-normalized.

    Different input text (e.g. instructed query vs verbatim passage)
    yields a different vector, which is what the BGE asymmetry gate
    needs.
    """
    raw: list[float] = []
    counter = 0
    while len(raw) < dim:
        digest = hashlib.sha256(f"{text}|{counter}".encode("utf-8")).digest()
        raw.extend((b - 127.5) / 127.5 for b in digest)
        counter += 1
    vec = raw[:dim]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class FakeEmbedding:
    """Stands in for the pinned sentence-transformers adapter."""

    name = "local_sentence_transformers"
    model = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        *,
        identity: str = PINNED_IDENTITY,
        dim: int = DIM,
        vectors_override: list[list[float]] | None = None,
    ) -> None:
        self.model_identity = identity
        self.dim = dim
        self._vectors_override = vectors_override
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> tuple[list[list[float]], Usage]:
        self.calls.append(list(texts))
        if self._vectors_override is not None:
            return self._vectors_override, Usage()
        return [_vector_for(t) for t in texts], Usage()


class FakeDriver:
    """In-memory pgvector stand-in implementing the exact-cosine,
    score-desc / ``chunk_id``-asc, ``LIMIT k`` contract the SQL
    encodes."""

    def __init__(self, rows: list[tuple[str, list[float], str]]) -> None:
        self.rows = rows
        self.store_stats_calls: list[str] = []
        self.search_calls: list[tuple[list[float], int]] = []

    def store_stats(self, expected_identity: str) -> tuple[int, tuple[str, str] | None]:
        self.store_stats_calls.append(expected_identity)
        mismatches = sorted(
            (chunk_id, identity)
            for chunk_id, _vec, identity in self.rows
            if identity != expected_identity
        )
        return len(self.rows), (mismatches[0] if mismatches else None)

    def search(self, query_vector, k: int) -> list[tuple[str, float]]:
        self.search_calls.append((list(query_vector), k))
        scored = [
            (chunk_id, _cosine_similarity(query_vector, vec))
            for chunk_id, vec, _identity in self.rows
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]


def _cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _chunk_id(index: int) -> str:
    return "c_" + hashlib.sha256(str(index).encode("utf-8")).hexdigest()[:16]


def _store(texts_by_id: dict[str, str], identity: str = PINNED_IDENTITY):
    return [
        (chunk_id, _vector_for(text), identity)
        for chunk_id, text in texts_by_id.items()
    ]


def _retriever(rows, **provider_kwargs):
    provider = FakeEmbedding(**provider_kwargs)
    driver = FakeDriver(rows)
    return PgVectorRetriever(provider, driver), provider, driver


# ---------------------------------------------------------------------------
# A / B / C / D / E — result shape and ordering.
# ---------------------------------------------------------------------------


class TestResultShape(unittest.TestCase):
    def test_a_hit_is_a_frozen_two_field_dataclass(self):
        hit = VectorRetrievalHit(chunk_id=_chunk_id(1), score=0.5)
        self.assertEqual(
            [f.name for f in dataclasses.fields(VectorRetrievalHit)],
            ["chunk_id", "score"],
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            hit.score = 0.9  # type: ignore[misc]

    def test_a_retrieve_returns_only_hits_with_float_scores(self):
        rows = _store({_chunk_id(i): f"passage {i}" for i in range(3)})
        retriever, _p, _d = _retriever(rows)
        hits = retriever.retrieve("what is a transformer", 3)
        self.assertIsInstance(retriever, VectorRetriever)
        self.assertEqual(len(hits), 3)
        for hit in hits:
            self.assertIsInstance(hit, VectorRetrievalHit)
            self.assertIsInstance(hit.chunk_id, str)
            self.assertIsInstance(hit.score, float)

    def test_a_scores_are_python_floats_even_when_driver_yields_ints(self):
        class IntScoreDriver(FakeDriver):
            def search(self, query_vector, k):
                super().search(query_vector, k)
                return [(_chunk_id(0), 1)]

        provider = FakeEmbedding()
        driver = IntScoreDriver(_store({_chunk_id(0): "only"}))
        hits = PgVectorRetriever(provider, driver).retrieve("q", 1)
        self.assertIsInstance(hits[0].score, float)
        self.assertEqual(hits[0].score, 1.0)


class TestOrdering(unittest.TestCase):
    def test_b_hits_are_ordered_by_score_descending(self):
        query = "graph retrieval baseline"
        target = _vector_for(QUERY_INSTRUCTION + query)
        # Rows crafted so their cosine to the query vector is known and
        # distinct: pure target, a perturbed target, and an opposite.
        near = [x * 0.9 + 0.1 for x in target]
        rows = [
            (_chunk_id(1), [-x for x in target], PINNED_IDENTITY),
            (_chunk_id(2), near, PINNED_IDENTITY),
            (_chunk_id(3), list(target), PINNED_IDENTITY),
        ]
        retriever, _p, _d = _retriever(rows)
        hits = retriever.retrieve(query, 3)
        self.assertEqual(
            [h.chunk_id for h in hits], [_chunk_id(3), _chunk_id(2), _chunk_id(1)]
        )
        self.assertAlmostEqual(hits[0].score, 1.0, places=9)
        self.assertAlmostEqual(hits[2].score, -1.0, places=9)
        self.assertEqual(hits, sorted(hits, key=lambda h: -h.score))

    def test_c_equal_scores_break_ties_on_chunk_id_ascending(self):
        vec = _vector_for("identical passage")
        ids = sorted(_chunk_id(i) for i in range(4))
        rows = [(cid, list(vec), PINNED_IDENTITY) for cid in reversed(ids)]
        retriever, _p, _d = _retriever(rows)
        hits = retriever.retrieve("anything", 4)
        self.assertEqual([h.chunk_id for h in hits], ids)
        self.assertEqual(len({h.score for h in hits}), 1)

    def test_d_synthetic_100_row_harness_matches_reference_ranking(self):
        rows = _store({_chunk_id(i): f"chunk body {i}" for i in range(100)})
        retriever, _p, _d = _retriever(rows)
        query = "exact cosine over one hundred rows"
        hits = retriever.retrieve(query, 10)

        query_vector = _vector_for(QUERY_INSTRUCTION + query)
        reference = sorted(
            (
                (chunk_id, _cosine_similarity(query_vector, vec))
                for chunk_id, vec, _identity in rows
            ),
            key=lambda pair: (-pair[1], pair[0]),
        )[:10]
        self.assertEqual(len(hits), 10)
        self.assertEqual([h.chunk_id for h in hits], [c for c, _s in reference])
        for hit, (_cid, score) in zip(hits, reference):
            self.assertAlmostEqual(hit.score, score, places=12)

    def test_e_k_greater_than_row_count_returns_every_row(self):
        rows = _store({_chunk_id(i): f"chunk {i}" for i in range(7)})
        retriever, _p, _d = _retriever(rows)
        hits = retriever.retrieve("q", 50)
        self.assertEqual(len(hits), 7)
        self.assertEqual(
            sorted(h.chunk_id for h in hits),
            sorted(chunk_id for chunk_id, _v, _i in rows),
        )


# ---------------------------------------------------------------------------
# F / G — empty store.
# ---------------------------------------------------------------------------


class TestEmptyStore(unittest.TestCase):
    def test_f_empty_store_returns_empty_list(self):
        retriever, _p, _d = _retriever([])
        self.assertEqual(retriever.retrieve("anything", 5), [])

    def test_g_empty_store_embeds_nothing_and_runs_no_search(self):
        retriever, provider, driver = _retriever([])
        retriever.retrieve("anything", 5)
        self.assertEqual(provider.calls, [])
        self.assertEqual(driver.search_calls, [])
        self.assertEqual(driver.store_stats_calls, [PINNED_IDENTITY])


# ---------------------------------------------------------------------------
# H / I / J / K / L / M — k validation.
# ---------------------------------------------------------------------------


class TestKValidation(unittest.TestCase):
    def setUp(self):
        rows = _store({_chunk_id(i): f"chunk {i}" for i in range(3)})
        self.retriever, self.provider, self.driver = _retriever(rows)

    def _assert_no_side_effects(self):
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(self.driver.store_stats_calls, [])
        self.assertEqual(self.driver.search_calls, [])

    def test_h_zero_raises_value_error_naming_k(self):
        with self.assertRaises(ValueError) as ctx:
            self.retriever.retrieve("q", 0)
        self.assertIn("k=0", str(ctx.exception))
        self._assert_no_side_effects()

    def test_i_negative_raises_value_error_naming_k(self):
        with self.assertRaises(ValueError) as ctx:
            self.retriever.retrieve("q", -1)
        self.assertIn("k=-1", str(ctx.exception))
        self._assert_no_side_effects()

    def test_j_float_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.retriever.retrieve("q", 5.0)
        self._assert_no_side_effects()

    def test_k_str_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.retriever.retrieve("q", "5")
        self._assert_no_side_effects()

    def test_l_bool_raises_type_error(self):
        for value in (True, False):
            with self.subTest(k=value):
                with self.assertRaises(TypeError):
                    self.retriever.retrieve("q", value)
        self._assert_no_side_effects()

    def test_m_invalid_k_precedes_every_database_and_embedding_call(self):
        for value in (0, -3, 5.0, "5", True, None):
            with self.subTest(k=value):
                with self.assertRaises((TypeError, ValueError)):
                    self.retriever.retrieve("q", value)
        self._assert_no_side_effects()


# ---------------------------------------------------------------------------
# N / O — BGE query preparation.
# ---------------------------------------------------------------------------


class TestQueryPreparation(unittest.TestCase):
    def test_n_instruction_is_prepended_exactly_once(self):
        rows = _store({_chunk_id(0): "a passage"})
        retriever, provider, _d = _retriever(rows)
        query = "how does pgvector compute cosine distance"
        retriever.retrieve(query, 1)

        self.assertEqual(len(provider.calls), 1)
        sent = provider.calls[0]
        self.assertEqual(sent, [QUERY_INSTRUCTION + query])
        self.assertEqual(sent[0].count(QUERY_INSTRUCTION), 1)
        self.assertEqual(sent[0].count(query), 1)

    def test_o_query_and_passage_embedding_inputs_are_asymmetric(self):
        text = "attention is all you need"
        rows = _store({_chunk_id(0): text})
        retriever, provider, _d = _retriever(rows)
        retriever.retrieve(text, 1)

        # #26 embeds passages verbatim; the retriever embeds the
        # instructed query. Same raw text -> different provider input
        # and therefore a different vector.
        passage_vectors, _usage = FakeEmbedding().embed([text])
        query_input = provider.calls[0][0]
        self.assertNotEqual(query_input, text)
        self.assertTrue(query_input.startswith(QUERY_INSTRUCTION))
        self.assertNotEqual(_vector_for(query_input), passage_vectors[0])


# ---------------------------------------------------------------------------
# P / Q — store identity validation.
# ---------------------------------------------------------------------------


class TestStoreIdentity(unittest.TestCase):
    def _mixed_rows(self):
        stale_identity = (
            "local_sentence_transformers:BAAI/bge-small-en-v1.5:"
            "0000000000000000000000000000000000000000"
        )
        rows = _store({_chunk_id(i): f"chunk {i}" for i in range(3)})
        rows.append((_chunk_id(9), _vector_for("stale chunk"), stale_identity))
        return rows, stale_identity

    def test_p_mixed_identity_store_is_rejected_and_names_the_row(self):
        rows, stale_identity = self._mixed_rows()
        retriever, _p, _d = _retriever(rows)
        with self.assertRaises(StaleVectorStoreError) as ctx:
            retriever.retrieve("q", 5)
        exc = ctx.exception
        self.assertEqual(exc.chunk_id, _chunk_id(9))
        self.assertEqual(exc.stored_identity, stale_identity)
        self.assertEqual(exc.expected_identity, PINNED_IDENTITY)
        message = str(exc)
        self.assertIn(_chunk_id(9), message)
        self.assertIn(stale_identity, message)
        self.assertIn(PINNED_IDENTITY, message)
        self.assertIn("re-ingest", message)

    def test_p_wholly_stale_store_is_rejected_not_silently_filtered(self):
        rows = _store(
            {_chunk_id(i): f"chunk {i}" for i in range(3)}, identity="other:model:rev"
        )
        retriever, _p, _d = _retriever(rows)
        with self.assertRaises(StaleVectorStoreError):
            retriever.retrieve("q", 2)

    def test_q_no_retrieval_or_embedding_happens_after_a_stale_failure(self):
        rows, _identity = self._mixed_rows()
        retriever, provider, driver = _retriever(rows)
        with self.assertRaises(StaleVectorStoreError):
            retriever.retrieve("q", 5)
        self.assertEqual(driver.search_calls, [])
        self.assertEqual(provider.calls, [])


# ---------------------------------------------------------------------------
# R / S — malformed embedding output.
# ---------------------------------------------------------------------------


class TestQueryEmbeddingOutput(unittest.TestCase):
    def _retrieve_with(self, vectors):
        rows = _store({_chunk_id(0): "a passage"})
        retriever, _p, driver = _retriever(rows, vectors_override=vectors)
        return retriever, driver

    def test_r_zero_vectors_fails_loudly(self):
        retriever, driver = self._retrieve_with([])
        with self.assertRaises(VectorRetrievalError):
            retriever.retrieve("q", 1)
        self.assertEqual(driver.search_calls, [])

    def test_r_more_than_one_vector_fails_loudly(self):
        retriever, driver = self._retrieve_with([_vector_for("a"), _vector_for("b")])
        with self.assertRaises(VectorRetrievalError):
            retriever.retrieve("q", 1)
        self.assertEqual(driver.search_calls, [])

    def test_s_wrong_dimensionality_fails_loudly(self):
        retriever, driver = self._retrieve_with([_vector_for("a", dim=128)])
        with self.assertRaises(VectorRetrievalError) as ctx:
            retriever.retrieve("q", 1)
        self.assertIn("384", str(ctx.exception))
        self.assertEqual(driver.search_calls, [])


# ---------------------------------------------------------------------------
# T — determinism.
# ---------------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):
    def test_t_two_runs_on_identical_state_return_identical_hits(self):
        rows = _store({_chunk_id(i): f"chunk {i}" for i in range(40)})
        retriever, _p, _d = _retriever(rows)
        first = retriever.retrieve("deterministic query", 5)
        second = retriever.retrieve("deterministic query", 5)
        self.assertEqual(first, second)
        self.assertEqual([h.chunk_id for h in first], [h.chunk_id for h in second])


# ---------------------------------------------------------------------------
# U / V — search-path shape.
# ---------------------------------------------------------------------------


class TestSearchSql(unittest.TestCase):
    def test_u_search_sql_is_exact_cosine_with_deterministic_ordering(self):
        self.assertIn("embedding <=> %s::vector", SEARCH_SQL)
        self.assertIn("1 - (embedding <=> %s::vector) AS score", SEARCH_SQL)
        self.assertIn("ORDER BY score DESC, chunk_id ASC", SEARCH_SQL)
        self.assertIn("LIMIT %s", SEARCH_SQL)

    def test_u_no_ann_index_or_approximate_search_in_the_module(self):
        source = (
            Path(__file__).resolve().parents[1] / "src" / "retrieval" / "vector.py"
        ).read_text(encoding="utf-8").lower()
        for token in ("ivfflat", "hnsw", "ivfflat.probes", "hnsw.ef_search", "create index"):
            self.assertNotIn(token, source)

    def test_u_k_is_pushed_into_the_database_limit(self):
        rows = _store({_chunk_id(i): f"chunk {i}" for i in range(20)})
        retriever, _p, driver = _retriever(rows)
        retriever.retrieve("q", 3)
        self.assertEqual(driver.search_calls[0][1], 3)

    def test_v_no_similarity_threshold_is_applied(self):
        self.assertNotIn("WHERE", SEARCH_SQL)
        self.assertNotIn("identity", SEARCH_SQL)
        # A row pointing away from the query still comes back.
        query = "threshold check"
        opposite = [-x for x in _vector_for(QUERY_INSTRUCTION + query)]
        rows = [(_chunk_id(0), opposite, PINNED_IDENTITY)]
        retriever, _p, _d = _retriever(rows)
        hits = retriever.retrieve(query, 5)
        self.assertEqual(len(hits), 1)
        self.assertLess(hits[0].score, 0.0)

    def test_store_stats_sql_checks_count_and_identity_only(self):
        self.assertIn("count(*)", STORE_STATS_SQL)
        self.assertIn("embedding_model <> %(identity)s", STORE_STATS_SQL)
        self.assertIn("ORDER BY chunk_id LIMIT 1", STORE_STATS_SQL)


# ---------------------------------------------------------------------------
# Production driver plumbing (no live pgvector: a recording fake cursor).
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        self._conn.executed.append((sql, params))

    def fetchone(self):
        return self._conn.one_row

    def fetchall(self):
        return self._conn.all_rows


class _FakeConn:
    def __init__(self, one_row=None, all_rows=()):
        self.one_row = one_row
        self.all_rows = list(all_rows)
        self.executed: list[tuple[str, object]] = []

    def cursor(self):
        return _FakeCursor(self)


class TestPsycopgDriver(unittest.TestCase):
    def test_store_stats_binds_the_identity_and_maps_the_mismatch(self):
        conn = _FakeConn(one_row=(7, _chunk_id(3), "other:model:rev"))
        driver = _PsycopgSearchDriver(conn)
        total, mismatch = driver.store_stats(PINNED_IDENTITY)
        self.assertEqual(total, 7)
        self.assertEqual(mismatch, (_chunk_id(3), "other:model:rev"))
        sql, params = conn.executed[0]
        self.assertEqual(sql, STORE_STATS_SQL)
        self.assertEqual(params, {"identity": PINNED_IDENTITY})

    def test_store_stats_reports_no_mismatch_as_none(self):
        driver = _PsycopgSearchDriver(_FakeConn(one_row=(0, None, None)))
        self.assertEqual(driver.store_stats(PINNED_IDENTITY), (0, None))

    def test_search_sends_a_pgvector_literal_and_k_then_maps_rows(self):
        conn = _FakeConn(all_rows=[(_chunk_id(1), 0.25), (_chunk_id(2), 0)])
        driver = _PsycopgSearchDriver(conn)
        hits = driver.search([0.5, -0.25, 0.0], 2)
        sql, params = conn.executed[0]
        self.assertEqual(sql, SEARCH_SQL)
        self.assertEqual(params, ("[0.5,-0.25,0.0]", 2))
        self.assertEqual(hits, [(_chunk_id(1), 0.25), (_chunk_id(2), 0.0)])
        self.assertIsInstance(hits[1][1], float)


# ---------------------------------------------------------------------------
# Pinned #27 baseline: the injected provider must BE the pinned adapter.
# ---------------------------------------------------------------------------


WRONG_IDENTITY = "local_sentence_transformers:BAAI/bge-base-en-v1.5:deadbeef"


class TestPinnedBaseline(unittest.TestCase):
    def test_pinned_constants_are_the_spec_values(self):
        self.assertEqual(PINNED_MODEL_IDENTITY, PINNED_IDENTITY)
        self.assertEqual(PINNED_DIM, 384)

    def test_wrong_provider_identity_fails_before_any_db_or_embed_call(self):
        rows = _store({_chunk_id(i): f"chunk {i}" for i in range(3)})
        retriever, provider, driver = _retriever(rows, identity=WRONG_IDENTITY)
        with self.assertRaises(VectorRetrievalError) as ctx:
            retriever.retrieve("q", 3)
        message = str(ctx.exception)
        self.assertIn(WRONG_IDENTITY, message)
        self.assertIn(PINNED_MODEL_IDENTITY, message)
        self.assertEqual(driver.store_stats_calls, [])
        self.assertEqual(driver.search_calls, [])
        self.assertEqual(provider.calls, [])

    def test_wrong_provider_dimension_fails_before_any_db_or_embed_call(self):
        rows = _store({_chunk_id(i): f"chunk {i}" for i in range(3)})
        retriever, provider, driver = _retriever(rows, dim=256)
        with self.assertRaises(VectorRetrievalError) as ctx:
            retriever.retrieve("q", 3)
        message = str(ctx.exception)
        self.assertIn("256", message)
        self.assertIn("384", message)
        self.assertEqual(driver.store_stats_calls, [])
        self.assertEqual(driver.search_calls, [])
        self.assertEqual(provider.calls, [])

    def test_store_written_under_the_same_wrong_identity_is_still_refused(self):
        rows = _store(
            {_chunk_id(i): f"chunk {i}" for i in range(3)}, identity=WRONG_IDENTITY
        )
        retriever, provider, driver = _retriever(rows, identity=WRONG_IDENTITY)
        with self.assertRaises(VectorRetrievalError):
            retriever.retrieve("q", 3)
        self.assertEqual(driver.store_stats_calls, [])
        self.assertEqual(driver.search_calls, [])
        self.assertEqual(provider.calls, [])

    def test_store_validation_is_performed_against_the_pinned_identity(self):
        rows = _store({_chunk_id(0): "chunk"})
        retriever, _p, driver = _retriever(rows)
        retriever.retrieve("q", 1)
        self.assertEqual(driver.store_stats_calls, [PINNED_MODEL_IDENTITY])

    def test_returned_query_vector_is_measured_against_pinned_384(self):
        rows = _store({_chunk_id(0): "chunk"})
        # Provider reports the pinned dim but returns a short vector:
        # the canonical 384, not the provider's claim, is the ruler.
        retriever, _p, driver = _retriever(
            rows, vectors_override=[_vector_for("q", dim=383)]
        )
        with self.assertRaises(VectorRetrievalError) as ctx:
            retriever.retrieve("q", 1)
        self.assertIn("383", str(ctx.exception))
        self.assertIn(str(PINNED_DIM), str(ctx.exception))
        self.assertEqual(driver.search_calls, [])

    def test_pinned_provider_path_remains_green(self):
        rows = _store({_chunk_id(i): f"chunk {i}" for i in range(5)})
        retriever, provider, driver = _retriever(rows)
        hits = retriever.retrieve("still works", 3)
        self.assertEqual(len(hits), 3)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(driver.search_calls), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
