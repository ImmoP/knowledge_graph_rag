"""CLI entry point for the deterministic two-stage entity resolver (#24).

Usage::

    python -m src.entity_resolve

Reads ``data/derived/chunks.jsonl``, ``data/extractions.jsonl``,
``data/extractions.manifest.json``, and ``ontology.yaml``; writes
``data/entities.jsonl``, ``data/entity_resolutions.jsonl``, and
``data/entity_resolution.manifest.json`` per §7.

The threshold, algorithm version, and normalisation version are
**module-scope constants** in
:mod:`src.graph.entity_resolution` and are **not** configurable
through CLI flags or environment variables (spec §4.1). This
entry point exists for ergonomic invocation only.
"""

from __future__ import annotations

import logging
import sys

from src.graph import entity_resolution as er


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = er.run_resolution()
    except er.PreflightError as exc:
        print(f"entity resolution preflight: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"entity resolution preflight: {exc}", file=sys.stderr)
        return 2
    print(
        f"entities={result.entities_records} "
        f"resolutions={result.entity_resolutions_records} "
        f"eligible_mentions={result.eligible_mentions} "
        f"identical_duplicate_mentions={result.identical_duplicate_mentions} "
        f"extraction_complete_chunks={result.extraction_complete_chunks}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
