"""CLI entry point for Neo4j graph ingestion (#25).

Usage::

    python -m src.build_graph

Reads ``NEO4J_URI``, ``NEO4J_USER``, ``NEO4J_PASSWORD`` from the
repository ``.env`` (gitignored) per §9 and writes
``data/graph_ingestion.manifest.json`` per §8.2.

Exit codes:

- ``0`` — success.
- ``1`` — Neo4j infrastructure error (driver failure, auth, network).
- ``2`` — preflight error (input integrity, ontology drift, missing
  raw OpenAlex record, etc.).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.build_graph.ingestion import IngestionError, run_ingestion
from src.build_graph.preflight import PreflightError


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = run_ingestion()
    except PreflightError as exc:
        print(f"build_graph preflight: {exc}", file=sys.stderr)
        return 2
    except IngestionError as exc:
        print(f"build_graph ingestion: {exc}", file=sys.stderr)
        return 1

    print(
        f"graph_content_sha256={result.graph_content_sha256} "
        f"citation_source_sha256={result.citation_source_sha256} "
        f"manifest={result.manifest_path} "
        f"nodes={sum(result.node_counts.values())} "
        f"relationships={sum(result.relationship_counts.values())}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())