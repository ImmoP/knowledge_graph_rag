#!/usr/bin/env python3
"""Build the frozen benchmark dataset.

Usage::

    python scripts/benchmark/build_dataset.py \
        --chunks /path/to/chunks.jsonl \
        --output-dir src/benchmark/dataset/

Builds the deterministic chunk set for the model-extraction
benchmark, writes the JSONL + SHA-256 + manifest into ``--output-dir``.
Re-running with the same arguments produces a byte-identical result
(any drift fails closed).

The two smoke chunks ``c_0001921855a8156b`` and
``c_001452fd142ce075`` are required to appear at the head of the
selection so the smoke runner can address them deterministically.

This script is offline-only. It never reads caches, never makes
network calls, and never inspects model output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.dataset.select import (  # noqa: E402
    SelectionConfig,
    select_chunks,
    write_selection_artifacts,
)
from src.benchmark.dataset.smoke_chunks import SMOKE_CHUNK_IDS  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        required=True,
        help="Path to the corpus chunks.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "src" / "benchmark" / "dataset",
        help="Where to write the dataset files.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=SelectionConfig.target_size,
    )
    parser.add_argument(
        "--min-documents",
        type=int,
        default=SelectionConfig.min_documents,
    )
    parser.add_argument(
        "--per-doc-cap",
        type=int,
        default=SelectionConfig.per_doc_cap,
    )
    parser.add_argument(
        "--no-required",
        action="store_true",
        help="Skip pinning the smoke chunks at the head of the set.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = SelectionConfig(
        target_size=args.target_size,
        min_documents=args.min_documents,
        per_doc_cap=args.per_doc_cap,
    )
    required: tuple[str, ...] = () if args.no_required else SMOKE_CHUNK_IDS
    selected, report = select_chunks(
        args.chunks,
        config=cfg,
        required_chunk_ids=required,
    )
    out_jsonl = args.output_dir / "benchmark_chunks.jsonl"
    out_sha = args.output_dir / "benchmark_chunks.sha256"
    out_manifest = args.output_dir / "benchmark_dataset_manifest.json"
    write_selection_artifacts(
        selected,
        report,
        output_jsonl=out_jsonl,
        output_sha256=out_sha,
        output_manifest=out_manifest,
    )
    print(
        f"dataset: {len(selected)} chunks from {len(report.documents_selected)} "
        f"documents; algorithm={report.algorithm_version}; seed={report.sampler_seed:#x}; "
        f"sha256={report.chunks_sha256}"
    )
    print(f"  jsonl:   {out_jsonl}")
    print(f"  sha256:  {out_sha}")
    print(f"  manifest:{out_manifest}")
    print(f"  tags:    {report.counts_by_tag}")
    print(f"  lengths: {report.counts_by_length_tercile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())