#!/usr/bin/env python3
"""Offline-only Scope-2 historical report CLI.

Writes a deterministic report at the caller-selected path.  Refuses to
import provider code, perform network calls, or write to gold/productive
paths.  The frozen dataset defaults to the repository location but the
historical run and external pilot root are mandatory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src.benchmark.scope2_pilot import (
    HISTORICAL_ARTIFACT_HASHES,
    PilotContractError,
    build_historical_report,
    verify_pilot_package,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    REPO_ROOT / "src" / "benchmark" / "dataset" / "benchmark_chunks.jsonl"
)
FORBIDDEN_RUN_PATH_PARTS: tuple[str, ...] = (
    "/gold_set/",
    "/data/gold_set/",
    "/data/manifest.jsonl",
)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"CLI REFUSED: {message}")


def _resolve_run_path(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    _ensure(path.is_dir(), f"historical run directory not found: {raw}")
    parts = "/" + str(path).replace("\\", "/") + "/"
    for forbidden in FORBIDDEN_RUN_PATH_PARTS:
        _ensure(
            forbidden not in parts,
            f"historical run path overlaps a gold/productive location: {raw}",
        )
    return path


def _resolve_pilot_root(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    _ensure(path.is_dir(), f"pilot root directory not found: {raw}")
    return path


def _resolve_report_path(raw: str, *, run_path: Path) -> Path:
    path = Path(raw).expanduser().resolve()
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    parts = "/" + str(path).replace("\\", "/") + "/"
    for forbidden in FORBIDDEN_RUN_PATH_PARTS:
        _ensure(
            forbidden not in parts,
            f"report path overlaps a gold/productive location: {raw}",
        )
    _ensure(
        path != run_path,
        "report path coincides with historical run root",
    )
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="report_scope2_pilot",
        description="Offline Scope-2 historical observation report.",
    )
    parser.add_argument(
        "--historical-run",
        required=True,
        help="External directory containing the historical Manifest run artifacts.",
    )
    parser.add_argument(
        "--pilot-root",
        required=True,
        help="External Scope-2 pilot root containing pilot_manifest.json.",
    )
    parser.add_argument(
        "--frozen-dataset",
        default=str(DEFAULT_DATASET),
        help=(
            "Frozen benchmark_chunks.jsonl path (default: repository copy)."
        ),
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Caller-selected output JSON path; must live outside gold/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_path = _resolve_run_path(args.historical_run)
    pilot_root = _resolve_pilot_root(args.pilot_root)
    report_path = _resolve_report_path(args.report, run_path=run_path)

    try:
        verify_pilot_package(pilot_root)
        report = build_historical_report(
            historical_run_root=run_path,
            pilot_root=pilot_root,
            historical_artifact_hashes=HISTORICAL_ARTIFACT_HASHES,
        )
    except PilotContractError as exc:
        print(f"SCOPE-2 REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover -- safety net
        print(f"SCOPE-2 REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp = report_path.with_suffix(report_path.suffix + ".partial")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(report_path)
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())