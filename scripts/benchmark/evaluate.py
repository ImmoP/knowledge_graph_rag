#!/usr/bin/env python3
"""Evaluate one or more benchmark runs.

Usage::

    python scripts/benchmark/evaluate.py \
        --run /tmp/bench-runs/<run-id> \
        --chunks src/benchmark/dataset/benchmark_chunks.jsonl \
        --prompts-dir prompts/ \
        --output-dir /tmp/bench-runs/<run-id>/eval

The evaluator consumes the canonical V2 extractions and rejects
JSONL produced by the runner and emits:

* ``metrics_<label>.json`` — per-provider technical metrics.
* ``review_records.jsonl`` — blinded review package (no model id).
* ``review_key.json`` — the label-to-provider mapping (sealed
  until the reviewer is done).

The evaluator never calls the network, never invents gold labels,
and never claims scientific correctness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.evaluate import (  # noqa: E402
    compute_metrics,
    write_blinded_review_package,
    write_metrics_report,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        action="append",
        help="Run directory produced by scripts/benchmark/run_smoke.py. May be passed multiple times to compare runs.",
    )
    parser.add_argument(
        "--provider-label",
        action="append",
        default=[],
        help="Optional human-readable label for each --run (in order). Defaults to the run id basename.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=REPO_ROOT / "src" / "benchmark" / "dataset" / "benchmark_chunks.jsonl",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=REPO_ROOT / "prompts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--review-seed",
        type=int,
        default=0xB11ADEED,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_for_metrics: list[tuple] = []
    runs_for_review: list[tuple] = []
    for i, run_dir in enumerate(args.run):
        provider_label = (
            args.provider_label[i]
            if i < len(args.provider_label)
            else run_dir.name
        )
        run_summary_path = run_dir / "run_summary.json"
        summary = (
            json.loads(run_summary_path.read_text(encoding="utf-8"))
            if run_summary_path.is_file()
            else {}
        )
        runs_for_metrics.append((
            provider_label,
            run_dir / "extractions.jsonl",
            run_dir / "extractions.rejects.jsonl",
            "extract_entities_v2.txt",
            "extract_relations_v2.txt",
            summary,
        ))
        runs_for_review.append((
            provider_label,
            run_dir / "extractions.jsonl",
            run_dir / "extractions.rejects.jsonl",
            "extract_entities_v2.txt",
            "extract_relations_v2.txt",
        ))

    for provider_label, jsonl_path, rejects_path, _, _, summary in runs_for_metrics:
        m = compute_metrics(
            output_jsonl=jsonl_path,
            output_rejects=rejects_path,
            metrics_path=jsonl_path.parent / "metrics.jsonl",
            provider_name=summary.get("provider", {}).get("name", provider_label),
            requested_model=summary.get("provider", {}).get("requested_model", "?"),
            reported_model=summary.get("provider", {}).get("reported_model"),
            dataset_sha256=summary.get("dataset_sha256", ""),
        )
        out = args.output_dir / f"metrics_{provider_label}.json"
        write_metrics_report(m, output_path=out)
        print(f"metrics: {out}")

    rec_path, key_path = write_blinded_review_package(
        runs=runs_for_review,
        chunks_path=args.chunks,
        prompts_dir=args.prompts_dir,
        output_dir=args.output_dir,
        seed=args.review_seed,
    )
    print(f"review records: {rec_path}")
    print(f"review key:     {key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())