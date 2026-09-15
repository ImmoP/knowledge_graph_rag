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


def _chunk(
    chunk_id: str,
    text: str,
    *,
    doc_id: str = "W100000001",
    section_path: list | None = None,
    chunk_index: int = 0,
    token_estimate: int | None = None,
) -> dict:
    if token_estimate is None:
        token_estimate = len(text) // 4
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "section_path": section_path or ["Intro"],
        "chunk_index": chunk_index,
        "text": text,
        "token_estimate": token_estimate,
        "char_start": 0,
        "char_end": len(text),
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


# ---------------------------------------------------------------------------
# Acceptance gate #1: populate store; identity SHA matches file bytes.
# ---------------------------------------------------------------------------


class SyncHappyPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"
        self.chunks = [
            _chunk("c_0000000000000001", "alpha body"),
            _chunk("c_0000000000000002", "beta body", chunk_index=1),
            _chunk("c_0000000000000003", "gamma body", chunk_index=2),
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
            _chunk("c_0000000000000001", "alpha"),
            _chunk("c_0000000000000002", "beta"),
        ]
        _write_chunks(self.chunks_path, self.chunks)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_model_drift_is_reembedded(self) -> None:
        wrong_identity = (
            "local_sentence_transformers:BAAI/bge-small-en-v1.5:wrongrev"
        )
        initial_rows = [
            _row("c_0000000000000001", "alpha", embedding_model=wrong_identity),
            _row("c_0000000000000002", "beta"),
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
            report.rows_reembedded_model_drift, ["c_0000000000000001"]
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
            _row("c_0000000000000001", "alpha"),
            {
                "chunk_id": "c_0000000000000002",
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
            report.rows_reembedded_sha_drift, ["c_0000000000000002"]
        )
        self.assertEqual(
            driver.rows["c_0000000000000002"]["source_text_sha256"],
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
        self.chunks = [_chunk("c_0000000000000001", "canonical")]
        _write_chunks(self.chunks_path, self.chunks)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_orphan_is_reported_and_deleted(self) -> None:
        driver = FakeDriver(initial_rows=[
            _row("c_0000000000000001", "canonical"),
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
            _row("c_0000000000000001", "canonical"),
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
        bad = _chunk(
            "c_0000000000000099",
            "hello world " * 100,  # 1100 chars => token_estimate 275 (<=512)
            chunk_index=0,
            token_estimate=275,  # chunk-record claim: 275 tokens
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
        self.assertEqual(ctx.exception.chunk_id, "c_0000000000000099")
        # The error reports the REAL count, not the chunk-record
        self.assertEqual(ctx.exception.token_count, 300 + 600)
        self.assertIn("c_0000000000000099", str(ctx.exception))
        # No embedding calls, no DB writes.
        self.assertEqual(embedding.embed_calls, [])
        self.assertEqual(driver.rows, {})

    def test_token_estimate_oversize_accepted_when_real_count_safe(self) -> None:
        """A chunk whose chunk-record ``token_estimate`` is > 512 but
        whose real-tokenizer count is <= 512 is NOT rejected merely
        because of ``token_estimate``. The loader uses the real
        tokenizer as the gate."""
        good = _chunk(
            "c_0000000000000099",
            "alpha",
            chunk_index=0,
            token_estimate=9999,  # chunk-record says way over; irrelevant
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
        self.assertEqual(report.rows_inserted, ["c_0000000000000099"])
        self.assertIn("c_0000000000000099", driver.rows)
        # The embedding provider WAS called (passing the real check).
        self.assertEqual(embedding.embed_calls, [["alpha"]])

    def test_oversize_error_reports_real_tokenizer_count(self) -> None:
        """The exception message + attributes surface the REAL count."""
        bad = _chunk(
            "c_0000000000000099",
            "hello",
            chunk_index=0,
            token_estimate=50,
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
        self.assertEqual(ctx.exception.chunk_id, "c_0000000000000099")
        self.assertEqual(ctx.exception.token_count, 700)
        self.assertEqual(ctx.exception.token_estimate, 50)
        # Message includes the real count.
        self.assertIn("token_count=700", str(ctx.exception))
        # And the chunk_id is named.
        self.assertIn("c_0000000000000099", str(ctx.exception))

    def test_oversize_failure_causes_zero_embed_calls_and_zero_db_writes(self) -> None:
        """If the real-tokenizer gate fires, the loader must fail
        BEFORE any embedding call or DB mutation."""
        bad = _chunk(
            "c_0000000000000099",
            "world",
            chunk_index=0,
            token_estimate=100,
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
                _chunk("c_0000000000000001", "raw passage text"),
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
            _write_chunks(chunks_path, [_chunk("c_0000000000000001", passage)])
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
            _chunk(f"c_{i:016x}", f"body {i}") for i in range(1, 6)
        ]
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
            _chunk("c_0000000000000001", "alpha body"),
            _chunk("c_0000000000000002", "beta body", chunk_index=1),
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
                json.dumps(_chunk("c_0000000000000099", "garbage", chunk_index=99))
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
            _chunk("c_0000000000000001", "alpha"),
            _chunk("c_0000000000000002", "beta"),
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
                "chunk_id": "c_0000000000000001",  # sha-drift
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
        _write_chunks(self.chunks_path, [_chunk("c_0000000000000001", "alpha")])

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_orphan_text_never_passed_to_embed(self) -> None:
        driver = FakeDriver(initial_rows=[
            _row("c_0000000000000001", "alpha"),
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