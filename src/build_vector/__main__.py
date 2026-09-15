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
  :class:`~src.vector_store_validator.VectorStoreIdentity` and the
  full §10.6 run metadata are printed so ``#33``'s preflight can
  record the same values without scraping logs.
* ``1`` — pgvector infrastructure error (driver / network / schema).
* ``2`` — preflight error (oversize chunk, malformed canonical
  artifact, missing chunks file).

CLI environment loading
-----------------------

The CLI loads the repository-root ``.env`` before constructing the
embedding provider and the pgvector driver. The loader honors the
"already-exported wins" rule: values present in ``os.environ`` are
NOT overwritten by ``.env`` (``override=False`` on
``dotenv_values``). The loader does NOT log any credentials.

Tests inject the loader directly via :func:`load_env_into_environ`
plus a saved/restored environment snapshot, so the caller's real
``os.environ`` is never mutated permanently.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterable

from src.build_vector.loader import (
    LoaderError,
    PgVectorDriver,
    default_driver,
    synchronize,
)
from src.build_vector.preflight import OversizeChunkError, PreflightError
from src.providers.registry import make_embedding


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOTENV_PATH = REPO_ROOT / ".env"


def load_env_into_environ(
    env_file: Path = DEFAULT_DOTENV_PATH,
    *,
    environ: dict[str, str] = os.environ,
) -> dict[str, str]:
    """Load ``env_file`` into ``environ``. Already-exported keys win:
    values in ``environ`` are not overwritten.

    Returns the parsed dotenv contents as a dict (without
    interpolating ``$VAR`` references — the production env has
    nothing referencing other vars). Uses ``python-dotenv``'s
    ``dotenv_values`` for parsing and then assigns only the keys
    missing from ``environ``.

    The function does NOT log credentials. ``env_file`` may not
    exist; in that case the loader returns an empty dict and leaves
    ``environ`` untouched.
    """
    try:
        from dotenv import dotenv_values
    except ImportError as exc:  # pragma: no cover - dependency pinned
        raise LoaderError(
            "python-dotenv is required for the build_vector CLI; "
            "install python-dotenv>=1.0 (already in requirements.txt)"
        ) from exc

    if not env_file.is_file():
        return {}

    parsed = dotenv_values(env_file)
    if parsed is None:
        return {}

    applied: dict[str, str] = {}
    for key, value in parsed.items():
        if value is None:
            continue
        # Already-exported wins: do NOT overwrite os.environ keys.
        if key not in environ:
            environ[key] = value
            applied[key] = value
    return applied


def _parse_existing_env(env_file: Path) -> dict[str, str]:
    """Public seam for tests: parse the dotenv file WITHOUT mutating
    any environment. Returns the dict the loader would have applied,
    with already-exported keys filtered out."""
    sentinel = {}
    return load_env_into_environ(env_file, environ=sentinel)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_env_into_environ()
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

    assert report.identity is not None
    # Full §10.6 run metadata, printed so #33 can record it
    # directly without parsing logs.
    print(
        f"build_vector sync ok: "
        f"orphan_rows_deleted={len(report.orphan_rows_deleted)} "
        f"rows_reembedded_model_drift={len(report.rows_reembedded_model_drift)} "
        f"rows_reembedded_sha_drift={len(report.rows_reembedded_sha_drift)} "
        f"rows_inserted={len(report.rows_inserted)} "
        f"canonical_chunks_sha256={report.canonical_chunks_sha256} "
        f"embedding_source_chunks_sha256={report.identity.embedding_source_chunks_sha256} "
        f"pinned_model_revision={report.pinned_model_revision} "
        f"embedding_model_identity={report.identity.embedding_model_identity} "
        f"timestamp={report.timestamp}"
    )
    return 0


def _chunks_path():
    from src.build_vector.preflight import DEFAULT_CHUNKS_PATH

    return DEFAULT_CHUNKS_PATH


# Public seam: snapshot + restore helper used by tests to verify
# already-exported-wins without mutating the caller's real env.
class EnvSnapshot:
    """Save a copy of the supplied env keys, run a callable, restore.

    Used by the .env-loading CLI tests: snapshot the keys the loader
    is about to set, run the loader into a sentinel dict, then
    restore the originals so the caller's real ``os.environ`` is
    untouched.
    """

    def __init__(
        self,
        env: dict[str, str],
        keys: Iterable[str],
    ) -> None:
        self._env = env
        self._keys = list(keys)
        self._saved: dict[str, str | None] = {}

    def __enter__(self) -> "EnvSnapshot":
        for k in self._keys:
            self._saved[k] = self._env.get(k)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for k, original in self._saved.items():
            if original is None:
                self._env.pop(k, None)
            else:
                self._env[k] = original


if __name__ == "__main__":
    sys.exit(main())