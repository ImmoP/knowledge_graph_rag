"""CLI entry point for pgvector embedding ingestion (#26).

Usage::

    python -m src.build_vector

Reads the pgvector connection (``PGVECTOR_HOST``,
``PGVECTOR_PORT``, ``PGVECTOR_USER``, ``PGVECTOR_PASSWORD``,
``PGVECTOR_DB``) and the embedding provider configuration from the
repository ``.env`` (gitignored). Uses
:func:`src.providers.registry.make_embedding` so the adapter stays
the single source of truth for ``BAAI/bge-small-en-v1.5`` at revision
``5c38ec7c405ec4b44b94cc5a9bb96e735b38267a``.

Exit codes:

* ``0`` — success; the validator's
  :class:`~src.vector_store_validator.VectorStoreIdentity` is printed
  so #33's preflight can record the same values without scraping logs.
* ``1`` — pgvector infrastructure error (driver / network / schema).
* ``2`` — preflight error (oversize chunk, schema drift, missing
  canonical artifact).
"""

from __future__ import annotations

import logging
import sys

from src.build_vector.loader import (
    LoaderError,
    PgVectorDriver,
    default_driver,
    synchronize,
)
from src.build_vector.preflight import OversizeChunkError, PreflightError
from src.providers.registry import make_embedding


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    driver: PgVectorDriver | None = None
    try:
        embedding_provider = make_embedding()
        driver = default_driver()
        report = synchronize(
            chunks_path=_chunks_path(),
            embedding_provider=embedding_provider,
            driver=driver,
        )
    except OversizeChunkError as exc:
        print(f"build_vector preflight: {exc}", file=sys.stderr)
        return 2
    except PreflightError as exc:
        print(f"build_vector preflight: {exc}", file=sys.stderr)
        return 2
    except LoaderError as exc:
        print(f"build_vector loader: {exc}", file=sys.stderr)
        return 1
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:  # pragma: no cover - already errored
                pass

    assert report.identity is not None  # synchronize returns identity on success
    print(
        f"build_vector sync ok: "
        f"orphan_rows_deleted={len(report.orphan_rows_deleted)} "
        f"rows_reembedded_model_drift={len(report.rows_reembedded_model_drift)} "
        f"rows_reembedded_sha_drift={len(report.rows_reembedded_sha_drift)} "
        f"rows_inserted={len(report.rows_inserted)} "
        f"embedding_source_chunks_sha256={report.identity.embedding_source_chunks_sha256} "
        f"embedding_model_identity={report.identity.embedding_model_identity} "
        f"timestamp={report.timestamp}"
    )
    return 0


def _chunks_path():
    """Resolve the canonical chunks path. Mirrors the preflight's
    module constant; kept as a thin wrapper so the CLI can evolve
    (e.g. ``--chunks`` override) without touching the preflight
    import."""
    from src.build_vector.preflight import DEFAULT_CHUNKS_PATH

    return DEFAULT_CHUNKS_PATH


if __name__ == "__main__":
    sys.exit(main())