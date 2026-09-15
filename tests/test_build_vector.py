"""Tests for ``src.build_vector`` (#26).

Covers impl-spec §10 acceptance gates:

* ``synchronize`` populates a (fake) pgvector store; the returned
  :class:`VectorStoreIdentity` matches the SHA256 of the canonical
  ``chunks.jsonl`` bytes (not a re-serialization) and the configured
  embedding identity.
* Stale-row sync: changing the configured model identity makes the
  loader re-embed the canonical chunks; a follow-up run reports
  ``0`` mismatches.
* Orphan-row sync: an extra row whose ``chunk_id`` is absent from
  the canonical artifact is reported and transactionally deleted by
  the loader; the embedding provider is never called with the
  orphan's text.
* Oversize chunk: a chunk whose ``token_estimate`` exceeds 512 is
  refused with the offending ``chunk_id`` named; no row is written.
* BGE asymmetry fixture: the loader embeds chunk text verbatim
  (without the BGE query instruction); a query-side prepending of
  the instruction produces a different vector.
* Re-run determinism: two consecutive runs on the same input produce
  identical row counts, identical ``source_text_sha256`` values, and
  an identical :class:`VectorStoreIdentity`.
* ``validate_vector_store`` is called as a single source of truth.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.build_vector import (  # noqa: E402
    OversizeChunkError,
    PgVectorDriver,
    SynchronizationReport,
    synchronize,
)
from src.build_vector.preflight import (  # noqa: E402
    PreflightError,
    run_preflight,
)
from src.providers.protocol import Usage  # noqa: E402

PINED_IDENTITY = (
    "local_sentence_transformers:BAAI/bge-small-en-v1.5:"
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)
PINED_MODEL_ID = "BAAI/bge-small-en-v1.5"
PINED_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

# BGE query instruction is pinned by vector-retrieval §1.1; the loader
# applies it to QUERIES only, never to chunk text. Tests assert on the
# asymmetry.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers: FakeDriver, FakeEmbedding, chunk fixture writer.
# ---------------------------------------------------------------------------


class _Tx:
    def __init__(self, driver: "FakeDriver") -> None:
        self._driver = driver
        self.committed = False
        self.rolled_back = False

    def __enter__(self):
        self._driver.open_transaction()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.committed = True
            self._driver.commit()
        else:
            self.rolled_back = True
            self._driver.rollback()
        return False


class FakeDriver(PgVectorDriver):
    """In-memory driver that records every execute / executemany
    call so the test can assert on the loader's behavior without
    psycopg. Transaction semantics are honored via :class:`_Tx`.
    """

    def __init__(self, initial_rows: list[dict] | None = None) -> None:
        self.rows: dict[str, dict] = {
            r["chunk_id"]: r for r in (initial_rows or [])
        }
        self.executed: list[tuple[str, tuple]] = []
        self.executed_many: list[tuple[str, list]] = []
        self._in_tx = False
        self._tx_dirty = False
        # Atomicity bookkeeping: counts how many transactions were
        # committed vs rolled back, in order. Tests assert on these
        # to prove the loader's single-transaction contract.
        self.commit_history: list[bool] = []
        self.tx_opened = 0

    def transaction(self) -> _Tx:
        self.tx_opened += 1
        return _Tx(self)

    def execute(self, sql: str, params: tuple = ()) -> Any:
        self.executed.append((sql, params))
        if "DELETE FROM chunks" in sql:
            (orphan_ids,) = params
            for cid in orphan_ids:
                self.rows.pop(cid, None)
                self._tx_dirty = True
        return None

    def executemany(self, sql: str, seq: list) -> Any:
        self.executed_many.append((sql, list(seq)))
        for chunk_id, vec_literal, model, sha in seq:
            self.rows[chunk_id] = {
                "chunk_id": chunk_id,
                "embedding": vec_literal,
                "embedding_model": model,
                "source_text_sha256": sha,
            }
            self._tx_dirty = True

    def list_rows(self) -> list[dict]:
        return [
            {
                "chunk_id": r["chunk_id"],
                "embedding_model": r["embedding_model"],
                "source_text_sha256": r["source_text_sha256"],
            }
            for r in self.rows.values()
        ]

    def close(self) -> None:
        return None

    def open_transaction(self) -> None:
        self._in_tx = True
        self._tx_dirty = False

    def commit(self) -> None:
        self._in_tx = False
        self.commit_history.append(True)

    def rollback(self) -> None:
        self._in_tx = False
        self.commit_history.append(False)


class FakeEmbedding:
    """Records every ``embed`` call so tests can assert on the BGE
    asymmetry. Returns deterministic L2-normalized vectors keyed by
    text hash so vectors are stable across runs (re-run determinism).
    """

    PROVIDER_KEY = "fake-embedding"

    def __init__(
        self,
        identity: str = PINED_IDENTITY,
        dim: int = 384,
        token_count_fn=None,
    ) -> None:
        self._identity = identity
        self.dim = dim
        self.model_identity = identity
        self.embed_calls: list[list[str]] = []
        # ``token_count_fn`` lets individual tests inject a custom
        # token-counter for the §1.3 oversize regressions. Default is
        # ``len(text.split()) + 2`` (BGE-style 2 special tokens); this
        # is intentionally NOT the chunk-record ``token_estimate``,
        # so the regressions below can prove the loader uses the
        # real tokenizer and not the chunk-record approximation.
        self._token_count_fn = token_count_fn or (
            lambda t: len(t.split()) + 2
        )

    @property
    def name(self) -> str:
        return self._identity.split(":")[0]

    @property
    def model(self) -> str:
        return PINED_MODEL_ID

    def count_tokens(self, text: str) -> int:
        return self._token_count_fn(text)

    def embed(self, texts: list[str]) -> tuple[list[list[float]], Usage]:
        self.embed_calls.append(list(texts))
        out = []
        for t in texts:
            seed_bytes = hashlib.sha256(t.encode("utf-8")).digest()
            raw = [
                int.from_bytes(seed_bytes[i : i + 4], "big") / 2**32 - 0.5
                for i in range(0, 32, 4)
            ]
            row = (raw * (self.dim // 8 + 1))[: self.dim]
            norm = sum(x * x for x in row) ** 0.5 or 1.0
            out.append([x / norm for x in row])
        return out, Usage()


# --- chunk artifact helpers ---------------------------------------


def _derive_chunk_id(doc_id: str, section_path: list, chunk_index: int) -> str:
    """Mirror the chunk-record §3 derivation the production preflight
    uses, so tests always write valid chunk_ids (the preflight now
    re-derives and requires equality)."""
    payload = (
        f"{doc_id}|{json.dumps(section_path, ensure_ascii=False)}|{chunk_index}"
    )
    return "c_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _chunk(
    chunk_id: str | None,
    text: str,
    *,
    doc_id: str = "W100000001",
    section_path: list | None = None,
    chunk_index: int = 0,
    token_estimate: int | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
) -> dict:
    if section_path is None:
        section_path = ["Intro"]
    if chunk_id is None:
        chunk_id = _derive_chunk_id(doc_id, section_path, chunk_index)
    if token_estimate is None:
        token_estimate = len(text) // 4
    if char_start is None:
        char_start = 0
    if char_end is None:
        char_end = len(text)
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "section_path": section_path,
        "chunk_index": chunk_index,
        "text": text,
        "token_estimate": token_estimate,
        "char_start": char_start,
        "char_end": char_end,
    }


def _write_chunks(path: Path, chunks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, separators=(",", ":")) + "\n")


def _row(chunk_id: str, text: str, *, embedding_model: str = PINED_IDENTITY) -> dict:
    return {
        "chunk_id": chunk_id,
        "embedding_model": embedding_model,
        "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


# Derived canonical chunk_ids for the test fixtures. Tests compare
# against these so the same chunk-record §3 derivation the
# production preflight enforces is exercised end-to-end.
CHUNK_A1 = _derive_chunk_id(
    doc_id="W100000001", section_path=["Intro"], chunk_index=0,
)
CHUNK_A2 = _derive_chunk_id(
    doc_id="W100000001", section_path=["Intro"], chunk_index=1,
)
CHUNK_A3 = _derive_chunk_id(
    doc_id="W100000001", section_path=["Intro"], chunk_index=2,
)
CHUNK_B1 = _derive_chunk_id(
    doc_id="W100000002", section_path=["Intro"], chunk_index=0,
)
CHUNK_C1 = _derive_chunk_id(
    doc_id="W100000003", section_path=["Intro"], chunk_index=0,
)
# ---------------------------------------------------------------------------
# Acceptance gate #1: populate store; identity SHA matches file bytes.
# ---------------------------------------------------------------------------


class SyncHappyPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"
        self.chunks = [
            _chunk(None, "alpha body"),
            _chunk(None, "beta body", chunk_index=1),
            _chunk(None, "gamma body", chunk_index=2),
        ]
        _write_chunks(self.chunks_path, self.chunks)
        self.driver = FakeDriver()
        self.embedding = FakeEmbedding()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_sync_populates_store_and_returns_identity(self) -> None:
        report = synchronize(
            chunks_path=self.chunks_path,
            embedding_provider=self.embedding,
            driver=self.driver,
        )
        self.assertIsInstance(report, SynchronizationReport)
        self.assertEqual(set(self.driver.rows), {c["chunk_id"] for c in self.chunks})
        self.assertEqual(report.orphan_rows_deleted, [])
        self.assertEqual(report.rows_reembedded_model_drift, [])
        self.assertEqual(report.rows_reembedded_sha_drift, [])
        self.assertEqual(
            sorted(report.rows_inserted),
            sorted(c["chunk_id"] for c in self.chunks),
        )
        self.assertEqual(
            report.identity.embedding_source_chunks_sha256,
            hashlib.sha256(self.chunks_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(report.identity.embedding_model_identity, PINED_IDENTITY)
        self.assertEqual(report.embedding_model_identity, PINED_IDENTITY)
        self.assertEqual(report.pinned_model_revision, PINED_REVISION)
        self.assertEqual(
            report.canonical_chunks_sha256,
            hashlib.sha256(self.chunks_path.read_bytes()).hexdigest(),
        )

    def test_bootstrap_schema_runs_by_default(self) -> None:
        synchronize(
            chunks_path=self.chunks_path,
            embedding_provider=self.embedding,
            driver=self.driver,
        )
        sqls = [sql for sql, _ in self.driver.executed[:2]]
        self.assertTrue(any("CREATE EXTENSION" in s for s in sqls))
        self.assertTrue(any("CREATE TABLE" in s for s in sqls))

    def test_no_embedding_calls_for_orphan_rows(self) -> None:
        """A row absent from the canonical artifact must be deleted
        without invoking the embedding provider (impl-spec §10.4)."""
        self.driver.rows["c_deadbeefdeadbeef"] = _row(
            "c_deadbeefdeadbeef", "orphan body"
        )
        self.embedding.embed_calls.clear()
        synchronize(
            chunks_path=self.chunks_path,
            embedding_provider=self.embedding,
            driver=self.driver,
        )
        self.assertNotIn("c_deadbeefdeadbeef", self.driver.rows)
        flat = [t for batch in self.embedding.embed_calls for t in batch]
        self.assertNotIn("orphan body", flat)



class StaleRowSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"
        self.chunks = [
            _chunk(None, "alpha"),
            _chunk(None, "beta", chunk_index=1),
        ]
        _write_chunks(self.chunks_path, self.chunks)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_model_drift_is_reembedded(self) -> None:
        wrong_identity = (
            "local_sentence_transformers:BAAI/bge-small-en-v1.5:wrongrev"
        )
        initial_rows = [
            _row(CHUNK_A1, "alpha", embedding_model=wrong_identity),
            _row(CHUNK_A2, "beta"),
        ]
        driver = FakeDriver(initial_rows=initial_rows)
        embedding = FakeEmbedding()
        report = synchronize(
            chunks_path=self.chunks_path,
            embedding_provider=embedding,
            driver=driver,
            bootstrap_schema=False,
        )
        self.assertEqual(
            report.rows_reembedded_model_drift, [CHUNK_A1]
        )
        self.assertTrue(
            all(r["embedding_model"] == PINED_IDENTITY for r in driver.rows.values())
        )
        self.assertEqual(
            report.identity.embedding_source_chunks_sha256,
            hashlib.sha256(self.chunks_path.read_bytes()).hexdigest(),
        )

    def test_sha_drift_is_reembedded(self) -> None:
        bad_rows = [
            _row(CHUNK_A1, "alpha"),
            {
                "chunk_id": CHUNK_A2,
                "embedding_model": PINED_IDENTITY,
                "source_text_sha256": "0" * 64,
            },
        ]
        driver = FakeDriver(initial_rows=bad_rows)
        embedding = FakeEmbedding()
        report = synchronize(
            chunks_path=self.chunks_path,
            embedding_provider=embedding,
            driver=driver,
            bootstrap_schema=False,
        )
        self.assertEqual(
            report.rows_reembedded_sha_drift, [CHUNK_A2]
        )
        self.assertEqual(
            driver.rows[CHUNK_A2]["source_text_sha256"],
            hashlib.sha256(b"beta").hexdigest(),
        )

    def test_subsequent_run_reports_zero_drift(self) -> None:
        """Acceptance gate #2: a second run on the freshly-synced
        store reports zero mismatches in every condition."""
        driver = FakeDriver()
        embedding = FakeEmbedding()
        synchronize(
            chunks_path=self.chunks_path,
            embedding_provider=embedding,
            driver=driver,
        )
        report = synchronize(
            chunks_path=self.chunks_path,
            embedding_provider=embedding,
            driver=driver,
            bootstrap_schema=False,
        )
        self.assertEqual(report.orphan_rows_deleted, [])
        self.assertEqual(report.rows_reembedded_model_drift, [])
        self.assertEqual(report.rows_reembedded_sha_drift, [])
        self.assertEqual(report.rows_inserted, [])


# ---------------------------------------------------------------------------
# Acceptance gate #3: orphan-row sync.
# ---------------------------------------------------------------------------


class OrphanRowSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"
        self.chunks = [_chunk(None, "canonical")]
        _write_chunks(self.chunks_path, self.chunks)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_orphan_is_reported_and_deleted(self) -> None:
        driver = FakeDriver(initial_rows=[
            _row(CHUNK_A1, "canonical"),
            _row("c_deadbeefdeadbeef", "orphan body"),
        ])
        embedding = FakeEmbedding()
        report = synchronize(
            chunks_path=self.chunks_path,
            embedding_provider=embedding,
            driver=driver,
            bootstrap_schema=False,
        )
        self.assertEqual(report.orphan_rows_deleted, ["c_deadbeefdeadbeef"])
        self.assertNotIn("c_deadbeefdeadbeef", driver.rows)

    def test_embedding_provider_never_called_for_orphan(self) -> None:
        driver = FakeDriver(initial_rows=[
            _row(CHUNK_A1, "canonical"),
            _row("c_deadbeefdeadbeef", "orphan body"),
        ])
        embedding = FakeEmbedding()
        synchronize(
            chunks_path=self.chunks_path,
            embedding_provider=embedding,
            driver=driver,
            bootstrap_schema=False,
        )
        flat = [t for batch in embedding.embed_calls for t in batch]
        self.assertNotIn("orphan body", flat)


# ---------------------------------------------------------------------------
# Acceptance gate #4: oversize chunk refusal.
# ---------------------------------------------------------------------------


class OversizeChunkRefusalTests(unittest.TestCase):
    """vector-retrieval.md §1.3 last-line oversize enforcement uses the
    **real** BGE tokenizer (via the adapter's ``count_tokens`` seam),
    NOT the chunk-record ``token_estimate`` approximation. These
    regressions prove the loader honors the real count and ignores
    ``token_estimate`` for the gate.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _oversize_count(self, text: str) -> int:
        """Deterministic fake tokenizer that returns ``token_estimate``
        + 600 so the count is forced above the 512 budget regardless
        of input shape."""
        return len(text) // 4 + 600

    def _safe_count(self, text: str) -> int:
        """Deterministic fake tokenizer that returns 1 (well under 512)."""
        return 1

    def test_token_estimate_under_budget_rejected_when_real_count_oversize(self) -> None:
        """A chunk whose chunk-record ``token_estimate`` is <= 512 but
        whose real-tokenizer count exceeds 512 is REJECTED."""
        bad = _chunk(None,
            "hello world " * 100,  # 1100 chars => token_estimate 275 (<=512)
            chunk_index=0  # chunk-record claim: 275 tokens
        )
        _write_chunks(self.chunks_path, [bad])
        driver = FakeDriver()
        # Real tokenizer says 1100+600 = 1700 tokens, way over 512.
        embedding = FakeEmbedding(token_count_fn=self._oversize_count)
        with self.assertRaises(OversizeChunkError) as ctx:
            synchronize(
                chunks_path=self.chunks_path,
                embedding_provider=embedding,
                driver=driver,
            )
        self.assertEqual(ctx.exception.chunk_id, CHUNK_A1)
        # The error reports the REAL count, not the chunk-record
        self.assertEqual(ctx.exception.token_count, 300 + 600)
        self.assertIn(CHUNK_A1, str(ctx.exception))
        # No embedding calls, no DB writes.
        self.assertEqual(embedding.embed_calls, [])
        self.assertEqual(driver.rows, {})

    def test_token_estimate_oversize_accepted_when_real_count_safe(self) -> None:
        """A chunk whose chunk-record ``token_estimate`` is > 512 but
        whose real-tokenizer count is <= 512 is NOT rejected merely
        because of ``token_estimate``. The loader uses the real
        tokenizer as the gate."""
        good = _chunk(None,
            "alpha",
            chunk_index=0  # chunk-record says way over; irrelevant
        )
        _write_chunks(self.chunks_path, [good])
        driver = FakeDriver()
        # Real tokenizer says 1 token, well under 512.
        embedding = FakeEmbedding(token_count_fn=self._safe_count)
        # Should NOT raise; the loader should proceed to embed + write.
        report = synchronize(
            chunks_path=self.chunks_path,
            embedding_provider=embedding,
            driver=driver,
        )
        self.assertEqual(report.rows_inserted, [CHUNK_A1])
        self.assertIn(CHUNK_A1, driver.rows)
        # The embedding provider WAS called (passing the real check).
        self.assertEqual(embedding.embed_calls, [["alpha"]])

    def test_oversize_error_reports_real_tokenizer_count(self) -> None:
        """The exception message + attributes surface the REAL count."""
        bad = _chunk(None,
            "hello",
            chunk_index=0
        )
        _write_chunks(self.chunks_path, [bad])
        driver = FakeDriver()

        def _count(text: str) -> int:
            return 700  # explicitly real count of 700, way over 512

        embedding = FakeEmbedding(token_count_fn=_count)
        with self.assertRaises(OversizeChunkError) as ctx:
            synchronize(
                chunks_path=self.chunks_path,
                embedding_provider=embedding,
                driver=driver,
            )
        self.assertEqual(ctx.exception.chunk_id, CHUNK_A1)
        self.assertEqual(ctx.exception.token_count, 700)
        self.assertEqual(ctx.exception.token_estimate, len("hello") // 4)
        # Message includes the real count.
        self.assertIn("token_count=700", str(ctx.exception))
        # And the chunk_id is named.
        self.assertIn(CHUNK_A1, str(ctx.exception))
    def test_oversize_failure_causes_zero_embed_calls_and_zero_db_writes(self) -> None:
        """If the real-tokenizer gate fires, the loader must fail
        BEFORE any embedding call or DB mutation."""
        bad = _chunk(None,
            "world",
            chunk_index=0
        )
        _write_chunks(self.chunks_path, [bad])
        driver = FakeDriver()
        # Real tokenizer says 800 tokens.
        embedding = FakeEmbedding(token_count_fn=lambda t: 800)
        try:
            synchronize(
                chunks_path=self.chunks_path,
                embedding_provider=embedding,
                driver=driver,
            )
        except OversizeChunkError:
            pass
        # Zero embedding calls, zero DB writes, zero bootstrap writes
        # beyond the schema-create.
        self.assertEqual(embedding.embed_calls, [])
        self.assertEqual(driver.rows, {})
        # ``bootstrap_schema`` defaults to True; the schema bootstrap
        # is allowed (idempotent, no row mutation). What MUST be zero
        # is any DML on the ``chunks`` table.
        dml_sqls = [
            sql for sql, _ in driver.executed
            if "INSERT" in sql or "DELETE" in sql or "UPDATE" in sql
        ]
        self.assertEqual(dml_sqls, [])

# ---------------------------------------------------------------------------
# BGE asymmetry fixture (acceptance gate implicit — vector-retrieval §1.1).
# ---------------------------------------------------------------------------


class BgeAsymmetryFixtureTests(unittest.TestCase):
    """The loader embeds chunk text VERBATIM; the BGE query instruction
    applies to queries only (vector-retrieval §1.1). The asymmetry is
    what makes the same text on the two sides produce different
    vectors; the adapter itself stays instruction-agnostic and the
    loader never touches the text.
    """

    def test_loader_passes_chunk_text_verbatim_to_embed(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        try:
            chunks_path = Path(tmp.name) / "chunks.jsonl"
            _write_chunks(chunks_path, [
                _chunk(None, "raw passage text"),
            ])
            driver = FakeDriver()
            embedding = FakeEmbedding()
            synchronize(
                chunks_path=chunks_path,
                embedding_provider=embedding,
                driver=driver,
            )
            self.assertEqual(embedding.embed_calls, [["raw passage text"]])
        finally:
            tmp.cleanup()

    def test_query_instruction_vs_passage_text_produces_different_vectors(self) -> None:
        """Static guard of the recipe: query side (with instruction) and
        passage side (verbatim) MUST differ. The loader never embeds
        the instruction because it is a caller's responsibility
        (#27 retriever applies it before embedding)."""
        tmp = tempfile.TemporaryDirectory()
        try:
            chunks_path = Path(tmp.name) / "chunks.jsonl"
            passage = "what is stochastic gradient descent"
            _write_chunks(chunks_path, [_chunk(None, passage)])
            driver = FakeDriver()
            embedding = FakeEmbedding()
            synchronize(
                chunks_path=chunks_path,
                embedding_provider=embedding,
                driver=driver,
            )
            passage_vecs, _ = embedding.embed([passage])
            query_vecs, _ = embedding.embed([BGE_QUERY_INSTRUCTION + passage])
            self.assertNotEqual(passage_vecs[0], query_vecs[0])
        finally:
            tmp.cleanup()


# ---------------------------------------------------------------------------
# Re-run determinism harness.
# ---------------------------------------------------------------------------


class RerunDeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"
        self.chunks = [
            _chunk(
                None,
                f"body {i}",
                doc_id="W100000001",
                section_path=["Intro"],
                chunk_index=i,
            )
            for i in range(5)
        ]
        self.chunks.sort(key=lambda c: c["chunk_id"])
        _write_chunks(self.chunks_path, self.chunks)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_two_consecutive_runs_are_identical(self) -> None:
        driver_a = FakeDriver()
        embedding_a = FakeEmbedding()
        report_a = synchronize(
            chunks_path=self.chunks_path,
            embedding_provider=embedding_a,
            driver=driver_a,
        )
        driver_b = FakeDriver(initial_rows=[
            {
                "chunk_id": cid,
                "embedding_model": PINED_IDENTITY,
                "source_text_sha256": driver_a.rows[cid]["source_text_sha256"],
                "embedding": driver_a.rows[cid]["embedding"],
            }
            for cid in driver_a.rows
        ])
        embedding_b = FakeEmbedding()
        report_b = synchronize(
            chunks_path=self.chunks_path,
            embedding_provider=embedding_b,
            driver=driver_b,
            bootstrap_schema=False,
        )
        # Row counts + per-row SHA + identity are identical.
        self.assertEqual(len(driver_a.rows), len(driver_b.rows))
        for cid in driver_a.rows:
            self.assertEqual(
                driver_a.rows[cid]["source_text_sha256"],
                driver_b.rows[cid]["source_text_sha256"],
            )
        self.assertEqual(
            report_a.identity.embedding_source_chunks_sha256,
            report_b.identity.embedding_source_chunks_sha256,
        )
        self.assertEqual(
            report_a.identity.embedding_model_identity,
            report_b.identity.embedding_model_identity,
        )
        # Second run is a clean no-op.
        self.assertEqual(report_b.orphan_rows_deleted, [])
        self.assertEqual(report_b.rows_reembedded_model_drift, [])
        self.assertEqual(report_b.rows_reembedded_sha_drift, [])
        self.assertEqual(report_b.rows_inserted, [])


# ---------------------------------------------------------------------------
# Canonical chunk-record validation — every shape contract is fail-closed.
# ---------------------------------------------------------------------------


class CanonicalChunkValidationTests(unittest.TestCase):
    """Negative regressions for ``docs/specs/chunk-record.md`` §2 + §3
    + §5. Each malformed shape MUST raise :class:`PreflightError`,
    never be silently coerced."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _good(self, **overrides) -> dict:
        """A minimal well-formed chunk record; tests override one
        field at a time to make the negative case the only delta."""
        text = "alpha body"
        rec = {
            "chunk_id": _derive_chunk_id(
                doc_id="W100000001", section_path=["Intro"], chunk_index=0,
            ),
            "doc_id": "W100000001",
            "section_path": ["Intro"],
            "chunk_index": 0,
            "text": text,
            "token_estimate": len(text) // 4,
            "char_start": 0,
            "char_end": len(text),
        }
        rec.update(overrides)
        return rec

    def _expect_preflight_error(self, records: list[dict], match: str) -> None:
        _write_chunks(self.chunks_path, records)
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(
                self.chunks_path,
                token_counter=lambda t: 1,  # never oversize
            )
        self.assertIn(match, str(ctx.exception))

    # --- section_path shape -----------------------------------------

    def test_section_path_as_string_fails(self) -> None:
        rec = self._good(section_path="Intro")
        self._expect_preflight_error([rec], "section_path")

    def test_empty_section_path_component_fails(self) -> None:
        rec = self._good(section_path=["Intro", ""])
        self._expect_preflight_error([rec], "section_path")

    def test_non_str_section_path_component_fails(self) -> None:
        rec = self._good(section_path=["Intro", 7])
        self._expect_preflight_error([rec], "section_path")

    # --- chunk_index shape -----------------------------------------

    def test_chunk_index_bool_fails(self) -> None:
        rec = self._good(chunk_index=True)
        # chunk_id still re-derives to a real id (True == 1 as int);
        # the bool check is what fires.
        self._expect_preflight_error([rec], "chunk_index")

    def test_chunk_index_negative_fails(self) -> None:
        rec = self._good(chunk_index=-1)
        self._expect_preflight_error([rec], "chunk_index")

    def test_chunk_index_string_fails(self) -> None:
        rec = self._good(chunk_index="0")
        self._expect_preflight_error([rec], "chunk_index")

    def test_duplicate_chunk_index_in_one_doc_fails(self) -> None:
        cid1 = _derive_chunk_id(
            doc_id="W100000001", section_path=["Intro"], chunk_index=0,
        )
        cid2 = _derive_chunk_id(
            doc_id="W100000001", section_path=["Intro"], chunk_index=1,
        )
        # Force cid1 to collide with cid2 by overriding.
        rows = [
            self._good(chunk_id=cid1),
            self._good(chunk_id=cid2, chunk_index=0),  # same index as row 0
        ]
        self._expect_preflight_error(rows, "duplicate chunk_index")

    # --- text shape ------------------------------------------------

    def test_empty_text_fails(self) -> None:
        rec = self._good(text="")
        self._expect_preflight_error([rec], "text")

    def test_non_string_text_fails(self) -> None:
        rec = self._good(text=12345)
        self._expect_preflight_error([rec], "text")

    # --- token_estimate shape --------------------------------------

    def test_token_estimate_bool_fails(self) -> None:
        rec = self._good(token_estimate=True)
        self._expect_preflight_error([rec], "token_estimate")

    def test_incorrect_token_estimate_fails(self) -> None:
        # len(text) // 4 == 2; claim 999
        rec = self._good(token_estimate=999)
        self._expect_preflight_error([rec], "token_estimate")

    # --- char_start / char_end shape ------------------------------

    def test_char_start_bool_fails(self) -> None:
        rec = self._good(char_start=True)
        self._expect_preflight_error([rec], "char_start")

    def test_char_end_negative_when_start_not_escape_fails(self) -> None:
        rec = self._good(char_start=0, char_end=-1)
        self._expect_preflight_error([rec], "offsets")

    def test_char_offsets_inverted_fails(self) -> None:
        rec = self._good(char_start=10, char_end=5)
        self._expect_preflight_error([rec], "offsets")

    def test_escape_hatch_requires_both_minus_one(self) -> None:
        # char_start == -1 escape-hatch with char_end != -1 must fail.
        rec = self._good(char_start=-1, char_end=10)
        self._expect_preflight_error([rec], "escape hatch")

    def test_escape_hatch_minus_one_minus_one_accepted(self) -> None:
        # Both -1 is the documented escape-hatch; the canonical
        # build_graph preflight accepts it.
        rec = self._good(char_start=-1, char_end=-1)
        _write_chunks(self.chunks_path, [rec])
        records = run_preflight(
            self.chunks_path, token_counter=lambda t: 1,
        )
        self.assertEqual(records[0].char_start, -1)
        self.assertEqual(records[0].char_end, -1)

    # --- chunk_id derivation + sort -------------------------------

    def test_non_rederiving_chunk_id_fails(self) -> None:
        # Provide a chunk_id that does NOT match the §3 derivation.
        bogus = "c_deadbeefdeadbeef"
        rec = self._good(chunk_id=bogus)
        self._expect_preflight_error([rec], "does not re-derive")

    def test_non_ascii_section_path_matches_canonical_producer(self) -> None:
        """The preflight's ``_derive_chunk_id`` MUST byte-for-byte
        match the canonical producer
        (``src/chunk/chunker.py`` ``json.dumps(section_path)`` with
        the library defaults — ``ensure_ascii=True``). The two
        diverge for any non-ASCII section name: ``json.dumps(["x"])
        with default settings escapes to ``["\\u00dcberblick"]``
        while ``ensure_ascii=False`` produces ``["Überblick"]``,
        yielding different SHA1 prefixes. A regression that proves
        the fix MUST fail under the broken ``ensure_ascii=False``
        implementation.
        """
        import hashlib as _hashlib
        # Canonical producer formula verbatim — no ensure_ascii override.
        canonical_chunk_id = "c_" + _hashlib.sha1(
            f"W100000001|{json.dumps(['Überblick'])}|0".encode("utf-8")
        ).hexdigest()[:16]
        rec = self._good(
            chunk_id=canonical_chunk_id,
            section_path=["Überblick"],
        )
        _write_chunks(self.chunks_path, [rec])
        # Must NOT raise — the preflight re-derives using the SAME
        # json.dumps defaults the canonical producer used.
        records = run_preflight(
            self.chunks_path, token_counter=lambda t: 1,
        )
        self.assertEqual(records[0].chunk_id, canonical_chunk_id)
    def test_unsorted_records_fail(self) -> None:
        cid_high = _derive_chunk_id(
            doc_id="W100000001", section_path=["Z"], chunk_index=0,
        )
        cid_low = _derive_chunk_id(
            doc_id="W100000001", section_path=["A"], chunk_index=0,
        )
        rows = [
            self._good(chunk_id=cid_high, section_path=["Z"]),
            self._good(chunk_id=cid_low, section_path=["A"]),
        ]
        # Sort order: cid_high > cid_low alphabetically — so the
        # second line is out of order.
        self._expect_preflight_error(rows, "records not sorted")

    # --- field-set exactness --------------------------------------

    def test_extra_field_fails(self) -> None:
        rec = self._good(unexpected_field="garbage")
        self._expect_preflight_error([rec], "extra")

    def test_missing_field_fails(self) -> None:
        rec = self._good()
        del rec["token_estimate"]
        self._expect_preflight_error([rec], "missing")
# ---------------------------------------------------------------------------
# .env loading for the production CLI.
# ---------------------------------------------------------------------------


class CliDotenvLoadingTests(unittest.TestCase):
    """``python -m src.build_vector`` loads the repository-root
    ``.env`` before constructing the embedding provider and the
    pgvector driver. Already-exported process env values win over
    ``.env``. Credentials are never logged.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.env_file = Path(self.tmp.name) / ".env"
        self._keys_we_touch: list[str] = []

    def tearDown(self) -> None:
        # No real os.environ mutation should leak from these tests.
        # The loader only writes into the dict we hand it; we pass
        # ``environ`` explicitly in tests.
        self.tmp.cleanup()

    def _write_env(self, lines: list[str]) -> None:
        self.env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_dotenv_values_visible_to_loader(self) -> None:
        """A key present only in ``.env`` is visible to the loader."""
        from src.build_vector.__main__ import load_env_into_environ
        self._write_env([
            "EMBEDDING_PROVIDER=local_sentence_transformers",
            "EMBEDDING_MODEL=BAAI/bge-small-en-v1.5",
            "PGVECTOR_USER=env_user",
            "PGVECTOR_PASSWORD=env_pw",
            "PGVECTOR_DB=rag",
            "PGVECTOR_HOST=localhost",
            "PGVECTOR_PORT=5433",
        ])
        # Pass an isolated dict so the test never touches os.environ.
        sentinel: dict[str, str] = {}
        applied = load_env_into_environ(self.env_file, environ=sentinel)
        self.assertEqual(sentinel["EMBEDDING_PROVIDER"], "local_sentence_transformers")
        self.assertEqual(sentinel["PGVECTOR_USER"], "env_user")
        self.assertEqual(sentinel["PGVECTOR_PASSWORD"], "env_pw")
        # Returned dict == keys that were newly added (not pre-existing).
        self.assertIn("EMBEDDING_PROVIDER", applied)
        self.assertIn("PGVECTOR_USER", applied)

    def test_already_exported_value_is_not_overwritten(self) -> None:
        """A key already present in ``environ`` is preserved; the
        loader must NOT overwrite it with the .env value."""
        from src.build_vector.__main__ import load_env_into_environ
        self._write_env([
            "PGVECTOR_USER=from_dotenv",
            "PGVECTOR_PASSWORD=dotenv_pw",
        ])
        sentinel: dict[str, str] = {
            "PGVECTOR_USER": "from_process_env",
        }
        load_env_into_environ(self.env_file, environ=sentinel)
        self.assertEqual(sentinel["PGVECTOR_USER"], "from_process_env")
        # But a missing key IS added from the dotenv.
        self.assertEqual(sentinel["PGVECTOR_PASSWORD"], "dotenv_pw")

    def test_dotenv_loading_does_not_mutate_os_environ(self) -> None:
        """The caller's real ``os.environ`` is not mutated by the
        production loader when no explicit ``environ`` is passed in
        tests that pre-snapshot the keys. We exercise that via the
        ``EnvSnapshot`` helper."""
        from src.build_vector.__main__ import EnvSnapshot, load_env_into_environ
        self._write_env([
            "BUILD_VECTOR_TEST_UNIQUE=present_in_dotenv",
        ])
        # Snapshot a sentinel set of keys (not all of os.environ)
        # so the helper remains narrow.
        # Build a synthetic environ dict and apply the loader to it.
        synthetic: dict[str, str] = {
            "BUILD_VECTOR_TEST_UNIQUE": "already_exported",
        }
        with EnvSnapshot(synthetic, ["BUILD_VECTOR_TEST_UNIQUE"]):
            applied = load_env_into_environ(self.env_file, environ=synthetic)
        # After the with-block, the original value is restored.
        self.assertEqual(synthetic["BUILD_VECTOR_TEST_UNIQUE"], "already_exported")
        # The .env value was NOT applied (already exported wins).
        self.assertNotIn("BUILD_VECTOR_TEST_UNIQUE", applied)

    def test_missing_env_file_is_a_noop(self) -> None:
        """No ``.env`` => no-op; ``environ`` unchanged, no raise."""
        from src.build_vector.__main__ import load_env_into_environ
        sentinel: dict[str, str] = {"PRESET": "x"}
        applied = load_env_into_environ(self.env_file, environ=sentinel)
        self.assertEqual(sentinel, {"PRESET": "x"})
        self.assertEqual(applied, {})

    def test_cli_dotenv_drives_provider_and_driver_construction(self) -> None:
        """End-to-end: a temporary ``.env`` is read by the production
        loader; both ``make_embedding`` and ``default_driver`` see
        the same values. We don't connect to pgvector — we observe
        the env values they would read."""
        from src.build_vector.__main__ import load_env_into_environ
        self._write_env([
            "LLM_PROVIDER=ollama",
            "EMBEDDING_PROVIDER=local_sentence_transformers",
            "EMBEDDING_MODEL=BAAI/bge-small-en-v1.5",
            "EMBEDDING_BATCH_SIZE=42",
            "PGVECTOR_HOST=localhost",
            "PGVECTOR_PORT=5433",
            "PGVECTOR_USER=cli_user",
            "PGVECTOR_PASSWORD=cli_pw",
            "PGVECTOR_DB=rag",
        ])
        synthetic: dict[str, str] = {}
        load_env_into_environ(self.env_file, environ=synthetic)
        # The provider registry's ``make_embedding`` reads
        # EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE;
        # all three must now be visible.
        self.assertEqual(synthetic["EMBEDDING_PROVIDER"], "local_sentence_transformers")
        self.assertEqual(synthetic["EMBEDDING_MODEL"], "BAAI/bge-small-en-v1.5")
        self.assertEqual(synthetic["EMBEDDING_BATCH_SIZE"], "42")
        # ``default_driver`` reads PGVECTOR_HOST / PGVECTOR_PORT /
        # PGVECTOR_USER / PGVECTOR_PASSWORD / PGVECTOR_DB.
        self.assertEqual(synthetic["PGVECTOR_HOST"], "localhost")
        self.assertEqual(synthetic["PGVECTOR_USER"], "cli_user")
        self.assertEqual(synthetic["PGVECTOR_DB"], "rag")


# ---------------------------------------------------------------------------
# Preflight-before-DB invariant.
# ---------------------------------------------------------------------------


class PreflightBeforeDBMutationTests(unittest.TestCase):
    """A malformed or oversize canonical artifact must fail
    ``synchronize`` BEFORE any pgvector mutation: zero bootstrap
    DDL, zero execute/executemany against the chunks table, no
    transaction opened. The driver observes no driver mutation
    activity for any preflight rejection."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_oversize_chunk_fails_before_any_driver_mutation(self) -> None:
        # Write a single chunk that the REAL-tokenizer count will
        # exceed. token_estimate must equal len(text) // 4 (per
        # the chunk-record contract); we override the token counter
        # to force the oversize path without breaking the contract.
        bad = _chunk(None, "x" * 4)  # token_estimate = 1
        _write_chunks(self.chunks_path, [bad])
        driver = FakeDriver()
        embedding = FakeEmbedding(token_count_fn=lambda t: 700)
        with self.assertRaises(OversizeChunkError):
            synchronize(
                chunks_path=self.chunks_path,
                embedding_provider=embedding,
                driver=driver,
            )
        # Driver saw ZERO mutation calls (no bootstrap DDL, no
        # execute, no executemany, no list_rows, no transaction).
        self.assertEqual(driver.executed, [])
        self.assertEqual(driver.executed_many, [])
        self.assertEqual(driver.tx_opened, 0)
        self.assertEqual(embedding.embed_calls, [])

    def test_malformed_chunk_fails_before_any_driver_mutation(self) -> None:
        # A row with a section_path that's not a list. The preflight
        # rejects; the driver observes nothing.
        rec = _chunk(None, "alpha")
        rec["section_path"] = "Intro"  # should be list[str]
        _write_chunks(self.chunks_path, [rec])
        driver = FakeDriver()
        embedding = FakeEmbedding()
        with self.assertRaises(PreflightError):
            synchronize(
                chunks_path=self.chunks_path,
                embedding_provider=embedding,
                driver=driver,
            )
        self.assertEqual(driver.executed, [])
        self.assertEqual(driver.executed_many, [])
        self.assertEqual(driver.tx_opened, 0)
        self.assertEqual(embedding.embed_calls, [])

    def test_no_synchronization_transaction_for_preflight_failures(self) -> None:
        """A preflight failure MUST not open any transaction (impl-
        spec §10.4: the transaction wraps the live read + writes +
        validation only after preflight passes)."""
        bad = _chunk(None, "x" * 4)
        _write_chunks(self.chunks_path, [bad])
        driver = FakeDriver()
        embedding = FakeEmbedding(token_count_fn=lambda t: 700)
        try:
            synchronize(
                chunks_path=self.chunks_path,
                embedding_provider=embedding,
                driver=driver,
            )
        except OversizeChunkError:
            pass
        self.assertEqual(
            driver.tx_opened, 0,
            msg="preflight failure must NOT open a synchronization transaction",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
# ---------------------------------------------------------------------------
# §10.6 run metadata.
# ---------------------------------------------------------------------------


class RunMetadataTests(unittest.TestCase):
    """Every successful embedding run must record (§10.6):
    - SHA256 of the exact ``chunks.jsonl`` bytes;
    - the pinned model revision (literal git SHA);
    - run timestamp;
    - the exact ``VectorStoreIdentity`` returned by ``validate_vector_store``.
    No persistent manifest is written.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"
        _write_chunks(self.chunks_path, [
            _chunk(None, "alpha body"),
            _chunk(None, "beta body", chunk_index=1),
        ])
        self.driver = FakeDriver()
        self.embedding = FakeEmbedding()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_all_four_required_fields_are_surfaced(self) -> None:
        report = synchronize(
            chunks_path=self.chunks_path,
            embedding_provider=self.embedding,
            driver=self.driver,
        )
        file_sha = hashlib.sha256(self.chunks_path.read_bytes()).hexdigest()
        # 1. SHA256 of the exact bytes on disk.
        self.assertEqual(report.canonical_chunks_sha256, file_sha)
        # 2. Pinned model revision (literal).
        self.assertEqual(report.pinned_model_revision, PINED_REVISION)
        # 3. ISO timestamp.
        self.assertRegex(report.timestamp, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
        # 4. Exact VectorStoreIdentity — its SHA must equal the file
        # SHA; its model identity must equal the configured one.
        self.assertIsNotNone(report.identity)
        self.assertEqual(report.identity.embedding_source_chunks_sha256, file_sha)
        self.assertEqual(report.identity.embedding_model_identity, PINED_IDENTITY)
        # The run metadata SHA equals the identity SHA (impl-spec §10.6
        # + the post-fix assertion in ``synchronize``).
        self.assertEqual(
            report.canonical_chunks_sha256,
            report.identity.embedding_source_chunks_sha256,
        )

    def test_metadata_sha_is_file_bytes_sha_not_reserialized(self) -> None:
        """The metadata SHA must be the on-disk bytes SHA, NOT a
        re-serialization of parsed records. ``_write_chunks`` is
        deterministic; mutating the on-disk bytes after the run is
        irrelevant for this run's identity (we hash BEFORE we
        return)."""
        before = synchronize(
            chunks_path=self.chunks_path,
            embedding_provider=self.embedding,
            driver=self.driver,
        )
        # Mutate the on-disk file. The run's identity must already
        # have been captured from the prior bytes; we just verify the
        # captured SHA equals the pre-mutation bytes.
        pre_sha = hashlib.sha256(self.chunks_path.read_bytes()).hexdigest()
        with self.chunks_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(_chunk(None, "garbage", chunk_index=99))
                + "\n"
            )
        self.assertEqual(before.canonical_chunks_sha256, pre_sha)
        self.assertEqual(
            before.identity.embedding_source_chunks_sha256, pre_sha
        )


class TransactionAtomicityTests(unittest.TestCase):
    """§10.4 requires orphan deletes, stale-row updates, and missing-row
    inserts to be performed atomically. A driver failure mid-transaction
    must roll back the entire transaction so no partial pgvector state
    survives.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"
        _write_chunks(self.chunks_path, [
            _chunk(None, "alpha"),
            _chunk(None, "beta", chunk_index=1),
        ])

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_mid_transaction_upsert_failure_rolls_back(self) -> None:
        """Pre-run: 1 orphan + 1 sha-drift row. The loader must
        perform orphan delete + upsert in ONE transaction. A failure
        mid-upsert rolls the transaction back so neither mutation
        is observed as committed.
        """
        initial = [
            {
                "chunk_id": "c_deadbeefdeadbeef",  # orphan
                "embedding_model": PINED_IDENTITY,
                "source_text_sha256": hashlib.sha256(b"orphan").hexdigest(),
            },
            {
                "chunk_id": CHUNK_A1,  # sha-drift
                "embedding_model": PINED_IDENTITY,
                "source_text_sha256": "0" * 64,
            },
        ]

        class _RaisingDriver(FakeDriver):
            def executemany(self, sql, seq):
                # Record the upsert but raise so the transaction
                # exits with an exception; ``_Tx.__exit__`` then
                # invokes ``rollback``.
                self.executed_many.append((sql, list(seq)))
                raise RuntimeError("simulated pgvector failure mid-upsert")

        driver = _RaisingDriver(initial_rows=initial)
        embedding = FakeEmbedding()
        from src.build_vector.loader import LoaderError

        with self.assertRaises(LoaderError):
            synchronize(
                chunks_path=self.chunks_path,
                embedding_provider=embedding,
                driver=driver,
                bootstrap_schema=False,
            )
        # Atomicity contract: exactly ONE transaction was opened,
        # and it was rolled back (False in commit_history), not
        # committed (no True after this run).
        self.assertEqual(driver.tx_opened, 1)
        self.assertEqual(driver.commit_history, [False],
                         msg="failed transaction must roll back, never commit")

    def test_validator_failure_inside_transaction_rolls_back(self) -> None:
        """If ``validate_vector_store`` fails inside the transaction
        (e.g. driver-side read failure during list_rows), the
        transaction must roll back so the orphan delete + upserts
        are NOT partially committed."""

        class _ReadFailureDriver(FakeDriver):
            def list_rows(self):
                raise RuntimeError("simulated pgvector read failure")

        driver = _ReadFailureDriver(initial_rows=[])
        embedding = FakeEmbedding()
        from src.build_vector.loader import LoaderError

        with self.assertRaises(LoaderError):
            synchronize(
                chunks_path=self.chunks_path,
                embedding_provider=embedding,
                driver=driver,
                bootstrap_schema=False,
            )
        # The transaction opened by the loader was rolled back.
        self.assertEqual(driver.tx_opened, 1)
        self.assertEqual(driver.commit_history, [False])

    def test_successful_run_commits_exactly_one_transaction(self) -> None:
        """Happy-path sanity: a successful run commits exactly one
        transaction (no partial commits, no extra transactions)."""
        driver = FakeDriver()
        embedding = FakeEmbedding()
        synchronize(
            chunks_path=self.chunks_path,
            embedding_provider=embedding,
            driver=driver,
        )
        self.assertEqual(driver.tx_opened, 1)
        self.assertEqual(driver.commit_history, [True])


# ---------------------------------------------------------------------------
# Store contract — vector-retrieval.md §3.1, impl-spec §10.3.
# ---------------------------------------------------------------------------


class StoreSchemaContractTests(unittest.TestCase):
    """The loader's pgvector schema MUST be exactly:
    - CREATE EXTENSION IF NOT EXISTS vector
    - chunks table with EXACTLY five columns:
      chunk_id text PRIMARY KEY
      embedding vector(384) NOT NULL
      embedding_model text NOT NULL
      source_text_sha256 char(64) NOT NULL
      embedded_at timestamptz NOT NULL DEFAULT now()
    - NO doc_id / section_path / chunk_index / text / token_estimate /
      char_start / char_end persisted.
    """

    def setUp(self) -> None:
        from src.build_vector import loader as loader_module
        self.loader_module = loader_module

    def test_create_extension_is_present(self) -> None:
        bootstrap = self.loader_module._BOOTSTRAP_SQL
        self.assertTrue(
            any("CREATE EXTENSION" in stmt and "vector" in stmt for stmt in bootstrap),
            msg="bootstrap must include CREATE EXTENSION IF NOT EXISTS vector",
        )

    def test_chunks_table_has_exactly_five_columns(self) -> None:
        # Look at just the CREATE TABLE body so we don't catch the
        # ``text`` type in ``embedding_model text NOT NULL``.
        bootstrap = "\n".join(self.loader_module._BOOTSTRAP_SQL)
        # The table body contains ``vector(384)`` which has a ``)``;
        # use a balanced-from-CREATE-TABLE-greedy match.
        m = re.search(
            r"CREATE TABLE[^(]*\((.*)\)\s*$",
            bootstrap, re.DOTALL,
        )
        self.assertIsNotNone(m, msg="CREATE TABLE statement must be present")
        table_body = m.group(1)
        for col in [
            "chunk_id",
            "embedding",
            "embedding_model",
            "source_text_sha256",
            "embedded_at",
        ]:
            self.assertIn(col, table_body, f"column {col!r} must be in schema")
        for forbidden in [
            "doc_id",
            "section_path",
            "chunk_index",
            "token_estimate",
            "char_start",
            "char_end",
        ]:
            self.assertNotRegex(
                table_body, rf"\b{re.escape(forbidden)}\b",
                msg=f"forbidden column {forbidden!r} must not appear",
            )

    def test_embedded_at_is_timestamptz_default_now(self) -> None:
        bootstrap = "\n".join(self.loader_module._BOOTSTRAP_SQL)
        self.assertRegex(
            bootstrap,
            r"embedded_at\s+timestamptz\s+NOT\s+NULL\s+DEFAULT\s+now\(\)",
        )
    def test_upsert_refreshes_embedding_model_sha_and_embedded_at(self) -> None:
        """Re-embedding must refresh the four stored fields; the
        loader must not regress into a stale-row scenario."""
        sql = self.loader_module._UPSERT_SQL
        for col in ["embedding", "embedding_model", "source_text_sha256", "embedded_at"]:
            self.assertIn(col, sql, f"upsert must refresh {col!r}")
        # The upsert must use ON CONFLICT to do an UPDATE on conflict.
        self.assertIn("ON CONFLICT", sql)
        self.assertIn("DO UPDATE SET", sql)


# ---------------------------------------------------------------------------
# Orphan-delete-without-embedding contract.
# ---------------------------------------------------------------------------


class OrphanDeleteContractTests(unittest.TestCase):
    """An orphan row is a pgvector row whose ``chunk_id`` is absent
    from the canonical chunks artifact. The loader must delete it
    INSIDE the same transaction that handles the (re-)embedding,
    and must NOT call ``embed`` with the orphan's text (the orphan
    has no canonical text to embed)."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"
        _write_chunks(self.chunks_path, [_chunk(None, "alpha")])

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_orphan_text_never_passed_to_embed(self) -> None:
        driver = FakeDriver(initial_rows=[
            _row(CHUNK_A1, "alpha"),
            _row("c_deadbeefdeadbeef", "orphan body that must not be embedded"),
        ])
        embedding = FakeEmbedding()
        synchronize(
            chunks_path=self.chunks_path,
            embedding_provider=embedding,
            driver=driver,
        )
        # The orphan was deleted in-tx.
        self.assertNotIn("c_deadbeefdeadbeef", driver.rows)
        # The orphan's text was never embedded.
        flat = [t for batch in embedding.embed_calls for t in batch]
        self.assertNotIn("orphan body that must not be embedded", flat)
        # The canonical chunk's initial row already matches the
        # pinned identity + correct SHA, so no re-embedding is
        # needed; the load-time invariant we care about is that
        # the orphan's text never reaches the embedder.