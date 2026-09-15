"""Vector-only retriever (#27).

Implements ``docs/specs/vector-retrieval.md`` §4 and impl-spec §11.1:
exact pgvector cosine search over the pinned BGE-small embeddings
written by #26.

Shape:

* :class:`VectorRetrievalHit` — the entire result type (§4.1).
* :class:`VectorRetriever` — the protocol callers depend on (§4.2).
* :class:`PgVectorRetriever` — the one concrete implementation; it
  owns query preparation, store identity validation and the score
  contract, and delegates SQL to a :class:`VectorSearchDriver`.
* :func:`default_driver` — production psycopg driver built from the
  ``PGVECTOR_*`` environment convention shared with #26.

The retriever never imports a provider SDK: the embedding adapter is
injected, and the #27 baseline (``MODEL_IDENTITY`` / ``DIM``, §1.1) is
reused from the adapter module's canonical constants rather than
copied as literals into retrieval logic. Importing those constants
does not pull in ``sentence_transformers`` — the adapter loads its SDK
lazily. The injected provider is checked against that baseline, so a
wrong provider (even one paired with a store written under the same
wrong identity) fails closed instead of silently retrieving.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from src.providers.adapters.sentence_transformers import (
    DIM as PINNED_DIM,
    MODEL_IDENTITY as PINNED_MODEL_IDENTITY,
)
from src.providers.protocol import EmbeddingProvider


#: BGE short-query instruction, pinned by vector-retrieval §1.1. Applied
#: to queries only — stored passages are embedded verbatim by #26, and
#: the adapter itself stays instruction-agnostic.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

#: Exact cosine search (§2: no ANN index, no approximate search). The
#: ``score DESC, chunk_id ASC`` ordering is the §4.3 determinism
#: contract and is enforced by the database, not by client-side
#: reranking. No WHERE clause: §4.3 forbids a similarity threshold.
SEARCH_SQL = (
    "SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score "
    "FROM chunks "
    "ORDER BY score DESC, chunk_id ASC "
    "LIMIT %s"
)

#: Row count plus the lexicographically first row whose
#: ``embedding_model`` differs from the active identity. One round trip
#: answers both §4.3 questions asked before retrieval: is the store
#: empty, and is it a mixed/stale-revision store?
STORE_STATS_SQL = (
    "WITH mismatch AS ("
    "SELECT chunk_id, embedding_model FROM chunks "
    "WHERE embedding_model <> %(identity)s ORDER BY chunk_id LIMIT 1"
    ") "
    "SELECT (SELECT count(*) FROM chunks), "
    "(SELECT chunk_id FROM mismatch), "
    "(SELECT embedding_model FROM mismatch)"
)


# ---------------------------------------------------------------------------
# Public result / error types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorRetrievalHit:
    """A single vector hit (spec §4.1). Serving resolves ``chunk_id``
    against ``data/derived/chunks.jsonl``; no text or metadata is
    carried here."""

    chunk_id: str
    score: float


@runtime_checkable
class VectorRetriever(Protocol):
    """Vector-only retrieval surface (spec §4.2)."""

    def retrieve(self, query: str, k: int) -> list[VectorRetrievalHit]: ...


class VectorRetrievalError(RuntimeError):
    """Retrieval could not be performed as specified (malformed
    embedding-provider output, and the base of the stale-store
    failure)."""


class StaleVectorStoreError(VectorRetrievalError):
    """A stored row was embedded with a different model identity.

    The store is not usable at retrieval time and cannot be repaired by
    filtering, translating revisions, or re-embedding on the fly: the
    corpus must be re-ingested with the active provider (#26).
    """

    def __init__(
        self, *, chunk_id: str, stored_identity: str, expected_identity: str
    ) -> None:
        super().__init__(
            f"pgvector chunks row chunk_id={chunk_id} was embedded with "
            f"embedding_model={stored_identity!r}, but the active embedding "
            f"provider identity is {expected_identity!r}; retrieval against a "
            "mixed/stale-revision store is refused — re-ingest the corpus "
            "(src.build_vector) with the active provider"
        )
        self.chunk_id = chunk_id
        self.stored_identity = stored_identity
        self.expected_identity = expected_identity


# ---------------------------------------------------------------------------
# Database port.
# ---------------------------------------------------------------------------


class VectorSearchDriver(Protocol):
    """The two reads the retriever needs from pgvector.

    Production is :func:`default_driver`; tests inject a deterministic
    in-memory stand-in.
    """

    def store_stats(self, expected_identity: str) -> tuple[int, tuple[str, str] | None]:
        """Return ``(row_count, mismatch)`` where ``mismatch`` is
        ``None`` or ``(chunk_id, stored_embedding_model)`` for the
        lexicographically first row whose identity differs."""

    def search(
        self, query_vector: Sequence[float], k: int
    ) -> list[tuple[str, float]]:
        """Exact cosine top-``k`` as ``(chunk_id, score)`` already
        ordered by score desc, ``chunk_id`` asc."""


# ---------------------------------------------------------------------------
# Concrete retriever.
# ---------------------------------------------------------------------------


class PinnedEmbeddingProvider(EmbeddingProvider, Protocol):
    """#27-local narrowing of :class:`EmbeddingProvider`.

    The vector baseline needs the ``model_identity`` seam the pinned
    sentence-transformers adapter exposes. It is declared here rather
    than on #19's generic contract so the provider layer stays
    unbroadened.
    """

    model_identity: str


class PgVectorRetriever(VectorRetriever):
    """Exact pgvector cosine retrieval over the pinned BGE embeddings."""

    def __init__(
        self, embedding_provider: PinnedEmbeddingProvider, driver: VectorSearchDriver
    ) -> None:
        self._embedding = embedding_provider
        self._driver = driver

    def retrieve(self, query: str, k: int) -> list[VectorRetrievalHit]:
        """Top-``k`` chunks for ``query``, score desc / ``chunk_id`` asc.

        Order of operations is part of the contract: ``k`` is validated
        first, then the injected provider is checked against the #27
        pinned baseline — both before any database or embedding call.
        The store's identity and row count are then checked before the
        query is embedded, so an empty store costs zero embedding calls
        and a stale store never reaches the search SQL.
        """
        _validate_k(k)
        self._validate_active_provider()

        total, mismatch = self._driver.store_stats(PINNED_MODEL_IDENTITY)
        if mismatch is not None:
            chunk_id, stored_identity = mismatch
            raise StaleVectorStoreError(
                chunk_id=chunk_id,
                stored_identity=stored_identity,
                expected_identity=PINNED_MODEL_IDENTITY,
            )
        if total == 0:
            return []

        query_vector = self._embed_query(query)
        return [
            VectorRetrievalHit(chunk_id=str(chunk_id), score=float(score))
            for chunk_id, score in self._driver.search(query_vector, k)
        ]

    def _validate_active_provider(self) -> None:
        """Fail closed unless the injected provider *is* the #27
        baseline (§1.1). Matching the store against whatever provider
        happens to be injected would accept a wrong provider paired
        with a store written under the same wrong identity."""
        identity = getattr(self._embedding, "model_identity", None)
        if identity != PINNED_MODEL_IDENTITY:
            raise VectorRetrievalError(
                f"active embedding provider identity is {identity!r}, but "
                f"vector retrieval is pinned to {PINNED_MODEL_IDENTITY!r}; "
                "retrieval with any other provider/revision is refused"
            )
        if self._embedding.dim != PINNED_DIM:
            raise VectorRetrievalError(
                f"active embedding provider reports dim="
                f"{self._embedding.dim}, but vector retrieval is pinned to "
                f"{PINNED_DIM} dimensions"
            )

    def _embed_query(self, query: str) -> list[float]:
        """Embed the BGE-instructed query exactly once, rejecting
        malformed provider output rather than continuing with it."""
        vectors, _usage = self._embedding.embed([QUERY_INSTRUCTION + query])
        if len(vectors) != 1:
            raise VectorRetrievalError(
                f"embedding provider returned {len(vectors)} vectors for one "
                "query; exactly one is required"
            )
        vector = vectors[0]
        if len(vector) != PINNED_DIM:
            raise VectorRetrievalError(
                f"embedding provider returned a {len(vector)}-dimensional "
                f"query vector; the pinned dimension is {PINNED_DIM}"
            )
        return vector


def _validate_k(k: int) -> None:
    """Spec §4.3: non-``int`` ``k`` raises ``TypeError``; ``k <= 0``
    raises ``ValueError`` naming the offending value. ``bool`` is
    rejected despite subclassing ``int`` — ``retrieve(q, True)`` is a
    caller bug, not a request for one hit."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if k <= 0:
        raise ValueError(f"k must be > 0, got k={k}")


# ---------------------------------------------------------------------------
# Production psycopg driver.
# ---------------------------------------------------------------------------


def default_driver() -> "_PsycopgSearchDriver":
    """Build a psycopg-backed driver from the ``PGVECTOR_*`` environment
    convention shared with #26 (local-infra.md §5.1). Credentials are
    read, never logged; ``psycopg`` is imported lazily so the test suite
    does not pay for it.
    """
    host = os.environ.get("PGVECTOR_HOST", "localhost")
    port = int(os.environ.get("PGVECTOR_PORT", "5433"))
    user = os.environ.get("PGVECTOR_USER")
    password = os.environ.get("PGVECTOR_PASSWORD")
    database = os.environ.get("PGVECTOR_DB", "rag")
    if not user or not password:
        raise RuntimeError(
            "PGVECTOR_USER and PGVECTOR_PASSWORD must be set in the environment"
        )

    import psycopg  # local import; tests do not pay the cost

    conn = psycopg.connect(
        host=host, port=port, user=user, password=password, dbname=database
    )
    return _PsycopgSearchDriver(conn)


class _PsycopgSearchDriver(VectorSearchDriver):
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def store_stats(self, expected_identity: str) -> tuple[int, tuple[str, str] | None]:
        with self._conn.cursor() as cur:
            cur.execute(STORE_STATS_SQL, {"identity": expected_identity})
            total, offender_id, offender_model = cur.fetchone()
        if offender_id is None:
            return int(total), None
        return int(total), (offender_id, offender_model)

    def search(
        self, query_vector: Sequence[float], k: int
    ) -> list[tuple[str, float]]:
        literal = "[" + ",".join(repr(float(x)) for x in query_vector) + "]"
        with self._conn.cursor() as cur:
            cur.execute(SEARCH_SQL, (literal, k))
            return [(chunk_id, float(score)) for chunk_id, score in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


__all__ = [
    "PINNED_DIM",
    "PINNED_MODEL_IDENTITY",
    "QUERY_INSTRUCTION",
    "PgVectorRetriever",
    "PinnedEmbeddingProvider",
    "StaleVectorStoreError",
    "VectorRetrievalError",
    "VectorRetrievalHit",
    "VectorRetriever",
    "VectorSearchDriver",
    "default_driver",
]
