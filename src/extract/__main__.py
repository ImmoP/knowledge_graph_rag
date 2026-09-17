"""CLI entry point for the extraction pipeline (#23, #55).

Usage::

    python -m src.extract            # v1 fresh-run with resume (spec §13.1)
    python -m src.extract v2         # canonical v2 with semantic resume (#55)

Reads ``LLM_PROVIDER`` and ``LLM_MODEL`` from the environment (per
spec §3.1) and produces the artifact set named by the chosen
entry. The bare-invocation v1 path is byte-for-byte unchanged from
the pre-T3b release (#55 §"CLI dispatch"); the new ``v2`` token
selects :func:`src.extract.pipeline_v2.run_extraction_v2_canonical`.
"""

from __future__ import annotations

import logging
import sys

from src.extract import pipeline, pipeline_v2
from src.providers import BudgetExceeded, make_llm


def main() -> int:
    if len(sys.argv) == 1:
        return _main_v1()
    if len(sys.argv) == 2 and sys.argv[1] == "v2":
        return _main_canonical_v2()
    print("usage: python -m src.extract [v2]", file=sys.stderr)
    return 2


def _main_v1() -> int:
    """Bare ``python -m src.extract`` — byte-for-byte v1 entry."""
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


def _main_canonical_v2() -> int:
    """``python -m src.extract v2`` — canonical v2 entry (#55)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        provider = make_llm()
    except Exception as exc:  # noqa: BLE001 — preflight fatal
        print(f"extraction preflight: provider config invalid: {exc}", file=sys.stderr)
        return 2
    try:
        result = pipeline_v2.run_extraction_v2_canonical(provider=provider)
    except pipeline_v2.PreflightErrorV2 as exc:
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
        f"rejects={result.reject_count} "
        f"cumulative_tokens={result.cumulative_tokens} "
        f"cumulative_cost_usd={result.cumulative_cost_usd:.6f}"
    )
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
