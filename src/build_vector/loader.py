"""Transactional pgvector synchronization (#26, impl-spec §10).

Implements ``docs/specs/vector-retrieval.md`` §3.3 and
``docs/specs/implementation-spec.md`` §10.4:

* A pgvector row whose ``chunk_id`` is missing from the canonical
  artifact is an **orphan**: report and delete it in the
  synchronization transaction. It cannot be re-embedded because no
  canonical text exists.
* A row with the wrong ``embedding_model`` is re-embedded from its
  canonical chunk and updated.
* A row whose ``source_text_sha256`` differs from
  ``sha256(canonical_chunk.text)`` is re-embedded from its canonical
  chunk and updated.
* A canonical chunk with no pgvector row is embedded and inserted.

The loader reports counts for all four conditions, performs the
required inserts / updates / deletes atomically (a single driver
transaction covering the live ``list_rows`` read, the writes, and
the post-write validation), and only returns success when
:func:`src.vector_store_validator.validate_vector_store` returns a
:class:`VectorStoreIdentity`.

The driver is **abstract**: production wires a ``psycopg`` connection;
tests inject a fake. The four SQL statements the loader uses are
isolated in module-level constants; tests can swap them out without
forking the loader.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.build_vector.preflight import (
    DEFAULT_CHUNKS_PATH,
    run_preflight,
)
from src.providers.protocol import EmbeddingProvider
from src.vector_store_validator import (
    VectorStoreIdentity,
    VectorStoreValidationError,
    _validate_vector_store,
)


log = logging.getLogger("build_vector")


# ---------------------------------------------------------------------------
# Public error / result types.
# ---------------------------------------------------------------------------


class LoaderError(RuntimeError):
    """Raised when pgvector synchronization fails. The transaction is
    rolled back; no partial state survives."""


@dataclass
class SynchronizationReport:
    """Per-run audit summary emitted next to the logs (impl-spec §10.6).

    The loader returns this so ``__main__`` can print it; ``#33`` does
    not consume it — it consumes the
    :class:`~src.vector_store_validator.VectorStoreIdentity` returned
    by :func:`validate_vector_store`.
    """

    canonical_chunks_sha256: str
    pinned_model_revision: str
    embedding_model_identity: str
    timestamp: str
    orphan_rows_deleted: list[str] = field(default_factory=list)
    rows_reembedded_model_drift: list[str] = field(default_factory=list)
    rows_reembedded_sha_drift: list[str] = field(default_factory=list)
    rows_inserted: list[str] = field(default_factory=list)
    identity: VectorStoreIdentity | None = None


# ---------------------------------------------------------------------------
# Driver port — abstract over psycopg / FakeDriver.
# ---------------------------------------------------------------------------


class PgVectorDriver:
    """Minimal port over a psycopg connection. The loader needs only
    execute / executemany / transaction / row readback.

    Production builds this around a ``psycopg.Connection`` via
    :func:`default_driver`. Tests substitute a fake whose ``execute``
    records the SQL it sees and stores rows in a dict so multiple
    runs are observable.
    """

    def transaction(self) -> Any:
        raise NotImplementedError

    def execute(self, sql: str, params: tuple = ()) -> Any:
        raise NotImplementedError

    def executemany(self, sql: str, seq: Sequence[tuple]) -> Any:
        raise NotImplementedError

    def list_rows(self) -> list[dict]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SQL constants — only used by the production psycopg driver.
# ---------------------------------------------------------------------------


# Schema bootstrap (idempotent). Uses pgvector's ``vector`` type which
# requires ``CREATE EXTENSION IF NOT EXISTS vector``; the pgvector image
# already ships it, but the IF NOT EXISTS keeps it safe.
_BOOTSTRAP_SQL = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    """CREATE TABLE IF NOT EXISTS chunks (
        chunk_id text PRIMARY KEY,
        embedding vector(384) NOT NULL,
        embedding_model text NOT NULL,
        source_text_sha256 char(64) NOT NULL,
        embedded_at timestamptz NOT NULL DEFAULT now()
    )""",
)

_UPSERT_SQL = (
    "INSERT INTO chunks "
    "(chunk_id, embedding, embedding_model, source_text_sha256) "
    "VALUES (%s, %s::vector, %s, %s) "
    "ON CONFLICT (chunk_id) DO UPDATE SET "
    "embedding = EXCLUDED.embedding, "
    "embedding_model = EXCLUDED.embedding_model, "
    "source_text_sha256 = EXCLUDED.source_text_sha256, "
    "embedded_at = now()"
)

_DELETE_ORPHAN_SQL = "DELETE FROM chunks WHERE chunk_id = ANY(%s)"

_LIST_ROWS_SQL = (
    "SELECT chunk_id, embedding_model, source_text_sha256 FROM chunks"
)


# ---------------------------------------------------------------------------
# Loader orchestrator.
# ---------------------------------------------------------------------------


def synchronize(
    chunks_path: Path,
    embedding_provider: EmbeddingProvider,
    driver: PgVectorDriver,
    *,
    bootstrap_schema: bool = True,
) -> SynchronizationReport:
    """Run one pgvector synchronization pass. Returns the report.

    Steps:

    1. Optional schema bootstrap — first run only; subsequent calls
       are idempotent (impl-spec §10.3 table contract).
    2. :func:`run_preflight` — load + validate the canonical
       artifact, using the adapter's
       :meth:`EmbeddingProvider.count_tokens` as the **real
       tokenizer** for the §1.3 last-line oversize guard. No
       ``token_estimate`` is used for the gate (vector-retrieval §1.3
       forbids approximations).
    3. Open ONE transaction that:
       a. reads live ``chunks`` rows;
       b. computes orphans / model-drift / sha-drift / missing sets;
       c. embeds every chunk to (re-)embed (passage text verbatim —
          no BGE query prefix);
       d. deletes orphans;
       e. upserts the inserted + updated rows;
       f. re-reads the rows;
       g. calls :func:`~src.vector_store_validator.validate_vector_store`
          and ONLY commits when it returns a
          :class:`~src.vector_store_validator.VectorStoreIdentity`.
       Any failure between (a) and (g) propagates an exception out
       of the transaction context; the context commits nothing and
       the pgvector state is the pre-run snapshot.
    4. ``#33`` reads the returned :class:`VectorStoreIdentity` from
       the report.
    """
    if bootstrap_schema:
        for stmt in _BOOTSTRAP_SQL:
            driver.execute(stmt)

    # Real-tokenizer seam (vector-retrieval §1.3). The adapter
    # exposes ``count_tokens``; the loader wires it here. Pre-flight
    # never imports any provider SDK — the callable is the seam.
    token_counter = embedding_provider.count_tokens

    records = run_preflight(chunks_path, token_counter=token_counter)
    canonical_by_id = {r.chunk_id: r for r in records}
    canonical_ids = set(canonical_by_id)

    # Single atomic transaction (impl-spec §10.4). The live row
    # read, the four writes, and the post-write validation ALL live
    # inside one driver.transaction() context. A failure anywhere
    # propagates an exception out of the ``with`` block; the
    # ``_Tx.__exit__`` handler then calls ``driver.rollback()``.
    identity: VectorStoreIdentity
    with driver.transaction():
        try:
            rows_before = driver.list_rows()
        except Exception as exc:
            raise LoaderError(
                f"pgvector list_rows failed mid-transaction: {exc}"
            ) from exc
        row_by_id = {r["chunk_id"]: r for r in rows_before}

        orphan_ids = sorted(set(row_by_id) - canonical_ids)
        model_drift_ids = sorted(
            cid for cid, row in row_by_id.items()
            if row["embedding_model"] != embedding_provider.model_identity
        )
        sha_drift_ids = sorted(
            cid for cid, row in row_by_id.items()
            if cid in canonical_by_id
            and row["source_text_sha256"] != canonical_by_id[cid].source_text_sha256
        )
        missing_ids = sorted(canonical_ids - set(row_by_id))

        to_reembed = sorted(
            set(model_drift_ids) | set(sha_drift_ids) | set(missing_ids)
        )

        # Embed the (re-)embedding set verbatim. The adapter is
        # instruction-agnostic (vector-retrieval §1.1); the loader
        # does NOT prepend the BGE query instruction to chunk text.
        # Embedding happens BEFORE the writes but inside the
        # transaction so a driver failure during write rolls back
        # the entire run; the embedding cost is wasted on failure
        # but the spec forbids partial pgvector state.
        new_vectors: dict[str, list[float]] = {}
        if to_reembed:
            texts = [canonical_by_id[cid].text for cid in to_reembed]
            try:
                vectors, _usage = embedding_provider.embed(texts)
            except Exception as exc:
                raise LoaderError(
                    f"embedding adapter failed mid-transaction: {exc}"
                ) from exc
            if len(vectors) != len(texts):
                raise LoaderError(
                    f"embedding adapter returned {len(vectors)} vectors "
                    f"for {len(texts)} texts"
                )
            new_vectors = dict(zip(to_reembed, vectors))

        # Orphan delete. Failure propagates out of ``with``;
        # transaction rolls back.
        if orphan_ids:
            driver.execute(_DELETE_ORPHAN_SQL, (orphan_ids,))

        upsert_rows: list[tuple] = []
        for cid, vec in new_vectors.items():
            rec = canonical_by_id[cid]
            upsert_rows.append(
                (
                    cid,
                    _format_vector_literal(vec),
                    embedding_provider.model_identity,
                    rec.source_text_sha256,
                )
            )
        if upsert_rows:
            try:
                driver.executemany(_UPSERT_SQL, upsert_rows)
            except Exception as exc:
                raise LoaderError(
                    f"pgvector upsert failed mid-transaction: {exc}"
                ) from exc

        # Re-read live rows for the validator. ``list_rows`` is the
        # contract; ``PgVectorDriver`` subclasses (psycopg + Fake)
        # return the current row shape the validator consumes. We
        # use the private ``_validate_vector_store`` worker so the
        # production-public ``validate_vector_store(chunks_path,
        # configured_embedding_identity)`` stays pure and connection-
        # agnostic; the loader owns its own pgvector connection.
        rows_after = driver.list_rows()
        try:
            identity = _validate_vector_store(
                chunks_path=chunks_path,
                configured_embedding_identity=embedding_provider.model_identity,
                rows_provider=lambda: rows_after,
            )
        except VectorStoreValidationError as exc:
            raise LoaderError(
                f"validation failed after sync: {exc}"
            ) from exc

    # Metadata is recorded AFTER the transaction commits. The
    # canonical-file SHA is computed from the same on-disk bytes the
    # validator hashed; both must agree byte-for-byte (§10.6 + the
    # single-source-of-truth guarantee).
    canonical_bytes = chunks_path.read_bytes()
    canonical_sha = hashlib.sha256(canonical_bytes).hexdigest()
    assert identity.embedding_source_chunks_sha256 == canonical_sha, (
        "validator SHA must equal the canonical chunks-file bytes"
    )
    pinned_revision = _extract_revision(embedding_provider.model_identity)
    report = SynchronizationReport(
        canonical_chunks_sha256=canonical_sha,
        pinned_model_revision=pinned_revision,
        embedding_model_identity=embedding_provider.model_identity,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        orphan_rows_deleted=orphan_ids,
        rows_reembedded_model_drift=model_drift_ids,
        rows_reembedded_sha_drift=sha_drift_ids,
        rows_inserted=missing_ids,
        identity=identity,
    )
    log.info(
        "build_vector sync: orphans=%d model_drift=%d sha_drift=%d "
        "inserted=%d identity_sha=%s",
        len(orphan_ids), len(model_drift_ids), len(sha_drift_ids),
        len(missing_ids), identity.embedding_source_chunks_sha256,
    )
    return report


# ---------------------------------------------------------------------------
# Production psycopg driver.
# ---------------------------------------------------------------------------


def default_driver(env_file: Path | None = None) -> PgVectorDriver:
    """Build a ``psycopg``-backed driver from the repository ``.env``.

    Required env vars (per local-infra.md §5.1, docker-compose.yml):
    ``PGVECTOR_HOST``, ``PGVECTOR_PORT``, ``PGVECTOR_USER``,
    ``PGVECTOR_PASSWORD``, ``PGVECTOR_DB``.

    Lazy import keeps ``psycopg`` out of the module top — the test
    suite never pays the import cost unless the loader is exercised
    for real.
    """
    if env_file is not None:
        from dotenv import dotenv_values

        env = {k: v for k, v in dotenv_values(env_file).items() if v is not None}
    else:
        env = dict(os.environ)

    host = env.get("PGVECTOR_HOST", "localhost")
    port = int(env.get("PGVECTOR_PORT", "5433"))
    user = env.get("PGVECTOR_USER")
    password = env.get("PGVECTOR_PASSWORD")
    database = env.get("PGVECTOR_DB", "rag")
    if not user or not password:
        raise LoaderError("PGVECTOR_USER and PGVECTOR_PASSWORD must be set in .env")

    import psycopg  # local import; tests don't pay the cost

    conn = psycopg.connect(
        host=host, port=port, user=user, password=password, dbname=database,
        autocommit=False,
    )
    return _PsycopgDriver(conn)


class _PsycopgDriver(PgVectorDriver):
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def transaction(self) -> Any:
        # psycopg with autocommit=False is already inside a
        # transaction; the context manager just gates commit/rollback
        # on whether the body raised.
        conn = self._conn

        class _Tx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                if exc_type is None:
                    conn.commit()
                else:
                    conn.rollback()
                return False

        return _Tx()

    def execute(self, sql: str, params: tuple = ()) -> Any:
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return cur

    def executemany(self, sql: str, seq: Sequence[tuple]) -> Any:
        with self._conn.cursor() as cur:
            cur.executemany(sql, seq)
            return cur

    def list_rows(self) -> list[dict]:
        with self._conn.cursor() as cur:
            cur.execute(_LIST_ROWS_SQL)
            out: list[dict] = []
            for chunk_id, embedding_model, source_text_sha256 in cur.fetchall():
                out.append(
                    {
                        "chunk_id": chunk_id,
                        "embedding_model": embedding_model,
                        "source_text_sha256": source_text_sha256,
                    }
                )
            return out

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _format_vector_literal(vec: list[float]) -> str:
    """Render a 384-dim L2-normalized vector as pgvector's ``[f,f,...]``
    text literal. Embedding values are JSON-safe floats.
    """
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _extract_revision(identity: str) -> str:
    """``<provider>:<model>:<revision>`` -> ``<revision>``. Best-effort;
    falls back to the full identity if it does not look like the
    triple form.
    """
    parts = identity.split(":")
    return parts[-1] if len(parts) == 3 else identity


__all__ = [
    "DEFAULT_CHUNKS_PATH",
    "LoaderError",
    "PgVectorDriver",
    "SynchronizationReport",
    "default_driver",
    "synchronize",
]