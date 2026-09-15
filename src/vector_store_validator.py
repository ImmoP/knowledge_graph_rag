"""Vector-store identity validator (#26, impl-spec §10).

The single source of truth for vector-store freshness. Used by:

* :mod:`src.build_vector.loader` — the synchronization loader calls
  ``validate_vector_store`` after every insert / update / delete and
  before returning the run result.
* ``#27`` — vector retriever (``VectorRetriever``) calls it as a
  freshness precondition at query-load time so a stale row never
  feeds a similarity score into serving.
* ``#33`` — benchmark preflight calls it and records the returned
  :class:`VectorStoreIdentity` as audit-only evidence; it does not
  re-implement the predicates here.

This module never reads provider SDKs, never constructs an
``EmbeddingProvider``, never connects to pgvector itself. It is a pure
function over the canonical chunks artifact and the live ``chunks``
table rows.

Public API (the required downstream seam):

    validate_vector_store(
        chunks_path,
        configured_embedding_identity,
    ) -> VectorStoreIdentity

Production wiring (the loader, the retriever, the benchmark preflight)
reads the live ``chunks`` rows through the normal repository pgvector
connection path (:func:`_read_pgvector_rows`). Tests inject a
deterministic stand-in via the private ``_rows_provider`` keyword
argument; the keyword is underscore-prefixed so it does not appear in
the public signature and ``#27`` / ``#33`` cannot accidentally
depend on it.

The four predicates (one-row-per-chunk, no orphan rows, matching
``embedding_model``, matching ``source_text_sha256``) are defined
exactly once, here. ``#27`` and ``#33`` import this module; the static
guard in :mod:`tests.test_vector_store_validator` enforces that no
duplication creeps in.

Spec: ``docs/specs/vector-retrieval.md`` §3.3, ``docs/specs/implementation-spec.md`` §10.4, §10.6.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


#: Pinned BGE input budget from ``vector-retrieval.md`` §1.1. Re-exported
#: here so the loader and the token-budget guard share one constant.
MAX_TOKENS = 512

#: Chunk-record §1 contract: ``^c_[0-9a-f]{16}$``.
CHUNK_ID_PATTERN = r"^c_[0-9a-f]{16}$"


# ---------------------------------------------------------------------------
# Public result / error types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorStoreIdentity:
    """What the loader / preflight / benchmark record after a successful
    validation.

    Two fields are exposed to callers; everything else the validator
    computes is internal. The frozen dataclass prevents downstream code
    from accidentally rewriting the audit fields (#33 records them
    byte-for-byte).
    """

    #: SHA256 of the EXACT ``data/derived/chunks.jsonl`` bytes the
    #: validator hashed. Computed via :func:`hashlib.sha256` against the
    #: on-disk file; never via re-serialization of parsed records.
    embedding_source_chunks_sha256: str

    #: The literal ``<provider>:<model>:<revision>`` string the
    #: configuration asked for. Captured here so #33's preflight can
    #: record the same value the loader used, without scraping logs.
    embedding_model_identity: str


class VectorStoreValidationError(RuntimeError):
    """Raised when one of the four predicates fails. The exception
    message names the offending ``chunk_id``(s) so the operator can
    find the row in pgvector without re-running validation.

    Predicates:

    * an orphan row (its ``chunk_id`` is absent from the canonical
      chunks artifact);
    * a row whose ``embedding_model`` differs from the configured
      identity;
    * a row whose ``source_text_sha256`` differs from
      ``sha256(canonical_chunk.text)``;
    * a canonical chunk with no row.
    """

    def __init__(
        self,
        *,
        predicate: str,
        offender_ids: Iterable[str],
        detail: str = "",
    ) -> None:
        ids = sorted(set(offender_ids))
        suffix = f" ({detail})" if detail else ""
        super().__init__(
            f"{predicate}: {len(ids)} offender(s): {ids[:10]}"
            + ("..." if len(ids) > 10 else "")
            + suffix
        )
        self.predicate = predicate
        self.offender_ids = ids


# ---------------------------------------------------------------------------
# Public function.
# ---------------------------------------------------------------------------


def validate_vector_store(
    chunks_path: Path,
    configured_embedding_identity: str,
) -> VectorStoreIdentity:
    """Validate the live pgvector ``chunks`` rows against the canonical
    ``data/derived/chunks.jsonl`` artifact and the configured embedding
    identity. Returns the :class:`VectorStoreIdentity` on success;
    raises :class:`VectorStoreValidationError` (with the offending
    ``chunk_id``s named) on any of the four predicate failures.

    :param chunks_path: the exact on-disk path to the canonical chunks
        artifact. The validator hashes the raw bytes of this file and
        parses the same artifact for membership / per-row checks. It
        never substitutes a re-serialization of parsed records.
    :param configured_embedding_identity: the literal
        ``<provider>:<model>:<revision>`` string the loader / preflight
        / benchmark is configured for. Usually the value of
        ``EmbeddingProvider.model_identity`` from the #19 adapter.

    Order of checks (early-exit on first failure):

    1. orphan-row check — every pgvector ``chunk_id`` must appear in
       the canonical artifact;
    2. ``embedding_model`` check — every row's value must equal
       ``configured_embedding_identity``;
    3. ``source_text_sha256`` check — every row's value must equal
       ``sha256(canonical_chunk.text)``;
    4. one-row-per-chunk check — every canonical chunk must have
       exactly one pgvector row.

    Production reads the live ``chunks`` rows through
    :func:`_read_pgvector_rows`; tests inject a deterministic stand-in
    via the private ``_rows_provider`` keyword.
    """
    return _validate_vector_store(
        chunks_path=Path(chunks_path),
        configured_embedding_identity=configured_embedding_identity,
        rows_provider=_read_pgvector_rows,
    )


def _validate_vector_store(
    *,
    chunks_path: Path,
    configured_embedding_identity: str,
    rows_provider: Callable[[], list[dict]],
) -> VectorStoreIdentity:
    """Private worker. Public API is :func:`validate_vector_store`; this
    function takes the rows provider as a keyword argument so unit
    tests can inject a deterministic stand-in without touching the
    public seam ``#27`` / ``#33`` will call."""

    raw_bytes = chunks_path.read_bytes()
    file_sha = hashlib.sha256(raw_bytes).hexdigest()

    canonical = _parse_chunks(raw_bytes)
    canonical_by_id: dict[str, dict] = {c["chunk_id"]: c for c in canonical}
    canonical_ids = set(canonical_by_id)

    rows = list(rows_provider())
    row_by_id: dict[str, dict] = {}
    for row in rows:
        cid = row["chunk_id"]
        if cid in row_by_id:
            raise VectorStoreValidationError(
                predicate="duplicate-row",
                offender_ids=[cid],
                detail=f"two rows share chunk_id={cid}",
            )
        row_by_id[cid] = row

    # 1. orphan-row check
    orphans = sorted(set(row_by_id) - canonical_ids)
    if orphans:
        raise VectorStoreValidationError(
            predicate="orphan-row",
            offender_ids=orphans,
        )

    # 2. embedding_model equality
    model_drift = sorted(
        cid
        for cid, row in row_by_id.items()
        if row["embedding_model"] != configured_embedding_identity
    )
    if model_drift:
        raise VectorStoreValidationError(
            predicate="embedding_model_mismatch",
            offender_ids=model_drift,
            detail=f"configured={configured_embedding_identity!r}",
        )

    # 3. source_text_sha256 equality
    sha_drift = sorted(
        cid
        for cid, row in row_by_id.items()
        if row["source_text_sha256"] != _text_sha256(canonical_by_id[cid]["text"])
    )
    if sha_drift:
        raise VectorStoreValidationError(
            predicate="source_text_sha256_mismatch",
            offender_ids=sha_drift,
        )

    # 4. one-row-per-chunk (every canonical chunk must be present)
    missing = sorted(canonical_ids - set(row_by_id))
    if missing:
        raise VectorStoreValidationError(
            predicate="missing-row",
            offender_ids=missing,
        )

    return VectorStoreIdentity(
        embedding_source_chunks_sha256=file_sha,
        embedding_model_identity=configured_embedding_identity,
    )


# ---------------------------------------------------------------------------
# Production pgvector row readback. Lives here so the public function
# does not need to know about ``psycopg`` or any caller-specific
# connection lifecycle.
# ---------------------------------------------------------------------------


_LIST_ROWS_SQL = (
    "SELECT chunk_id, embedding_model, source_text_sha256 FROM chunks"
)


def _read_pgvector_rows() -> list[dict]:
    """Read live ``chunks`` rows from the repository's pgvector.

    Opens a fresh psycopg connection per call (validator is
    short-lived: preflight at query-load time, once per retriever
    process). Reads ``PGVECTOR_HOST`` / ``PGVECTOR_PORT`` /
    ``PGVECTOR_USER`` / ``PGVECTOR_PASSWORD`` / ``PGVECTOR_DB`` from
    the environment (per local-infra.md §5.1). Returns a list of
    dicts with keys ``chunk_id``, ``embedding_model``,
    ``source_text_sha256``; ``chunk_id`` deduplication is performed
    by :func:`_validate_vector_store`.

    Lazy-imported so module import never pays the ``psycopg`` cost for
    callers that wire their own ``_rows_provider`` (the loader,
    tests).
    """
    import psycopg  # local import; tests / loader do not pay the cost

    host = os.environ.get("PGVECTOR_HOST", "localhost")
    port = int(os.environ.get("PGVECTOR_PORT", "5433"))
    user = os.environ.get("PGVECTOR_USER")
    password = os.environ.get("PGVECTOR_PASSWORD")
    database = os.environ.get("PGVECTOR_DB", "rag")
    if not user or not password:
        raise RuntimeError(
            "PGVECTOR_USER and PGVECTOR_PASSWORD must be set in the environment"
        )

    with psycopg.connect(
        host=host, port=port, user=user, password=password, dbname=database
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(_LIST_ROWS_SQL)
            return [
                {
                    "chunk_id": chunk_id,
                    "embedding_model": embedding_model,
                    "source_text_sha256": source_text_sha256,
                }
                for chunk_id, embedding_model, source_text_sha256 in cur.fetchall()
            ]


# ---------------------------------------------------------------------------
# Helpers — internal to this module on purpose.
# ---------------------------------------------------------------------------


_REQUIRED_CHUNK_FIELDS = frozenset(
    {"chunk_id", "doc_id", "section_path", "chunk_index", "text",
     "token_estimate", "char_start", "char_end"}
)


def _parse_chunks(raw_bytes: bytes) -> list[dict]:
    """Parse the canonical chunks JSONL byte stream. Each record must
    carry the chunk-record §3 required field set exactly; missing /
    extra fields raise :class:`ValueError` so the loader / benchmark
    preflight surfaces a structural defect instead of silently
    misclassifying.
    """
    out: list[dict] = []
    for line_no, line in enumerate(raw_bytes.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"chunks.jsonl line {line_no}: invalid JSON: {exc}"
            ) from exc
        missing = _REQUIRED_CHUNK_FIELDS - set(rec)
        if missing:
            raise ValueError(
                f"chunks.jsonl line {line_no}: missing fields {sorted(missing)}"
            )
        out.append(rec)
    return out


def _text_sha256(text: str) -> str:
    """SHA256 of the chunk text bytes. UTF-8 encoded; the spec does not
    pin an encoding, but Python's json module writes UTF-8 by default
    and the chunk-record schema treats ``text`` as a JSON string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()