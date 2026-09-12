"""CLI entry point for the extraction pipeline (#23).

Usage::

    python -m src.extract

Reads ``LLM_PROVIDER`` and ``LLM_MODEL`` from the environment (per
spec §3.1) and produces ``data/extractions.jsonl``,
``data/extractions.manifest.json``, and ``data/extractions.rejects.jsonl``.

Re-running the entry point against existing artifacts is the
resumability path (spec §13.1): chunks whose
``chunk_status.extraction_complete == true`` are skipped; incomplete
chunks are re-run and their records replaced.
"""

from __future__ import annotations

import logging
import sys

from src.extract import pipeline
from src.providers import BudgetExceeded, make_llm


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        provider = make_llm()
    except Exception as exc:  # noqa: BLE001 — preflight fatal
        print(f"extraction preflight: provider config invalid: {exc}", file=sys.stderr)
        return 2
    try:
        result = pipeline.run_extraction(provider=provider)
    except pipeline.PreflightError as exc:
        print(f"extraction preflight: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"extraction preflight: {exc}", file=sys.stderr)
        return 2
    except BudgetExceeded as exc:
        print(f"extraction budget exceeded: {exc}", file=sys.stderr)
        return 1

    print(
        f"run_id={result.run_id} "
        f"resumed_from={result.resumed_from_run_id or '<none>'} "
        f"budget_aborted={result.budget_aborted} "
        f"records_written={result.records_written} "
        f"mentions={result.mention_count} "
        f"assertions={result.assertion_count} "
        f"complete_chunks={result.complete_chunk_count} "
        f"rejects={result.reject_count}"
    )
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())