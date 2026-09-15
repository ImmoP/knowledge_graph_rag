"""pgvector embedding ingestion (#26, impl-spec §10).

Implements ``docs/specs/vector-retrieval.md`` and ``docs/specs/implementation-spec.md`` §10.

Pipeline:

1. :func:`src.build_vector.preflight.run_preflight` — loads
   ``data/derived/chunks.jsonl``, validates every record against the
   §3 chunk-shape, fails closed on any oversize chunk (impl-spec §10.5
   last-line enforcement). Raises :class:`PreflightError`.
2. :func:`src.build_vector.loader.synchronize` — single-transaction
   synchronization (impl-spec §10.4). Computes the four canonical
   conditions (orphan / wrong-model / wrong-sha / missing), performs
   the required inserts, updates, and orphan deletions, and reports
   counts. Then calls :func:`src.vector_store_validator.validate_vector_store`
   so the run only returns success when the four predicates hold.
3. CLI entry :mod:`src.build_vector.__main__` wires the
   ``EmbeddingProvider`` from :func:`src.providers.registry.make_embedding`
   and a ``psycopg`` connection to the loader.

This module never reads LLMProvider, neo4j, or any provider SDK outside
``src/providers/``. The embedding adapter is the only provider
dependency; ``sentence_transformers`` is imported inside the adapter's
loader, never at the top of this module.
"""

from __future__ import annotations

from src.build_vector.loader import (
    LoaderError,
    PgVectorDriver,
    SynchronizationReport,
    default_driver,
    synchronize,
)
from src.build_vector.preflight import (
    OversizeChunkError,
    PreflightError,
    run_preflight,
)

__all__ = [
    "LoaderError",
    "OversizeChunkError",
    "PgVectorDriver",
    "PreflightError",
    "SynchronizationReport",
    "default_driver",
    "run_preflight",
    "synchronize",
]