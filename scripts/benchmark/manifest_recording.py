#!/usr/bin/env python3
"""Record a bounded Manifest Auto smoke run or replay it fully offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.dataset.manifest import CHUNKS_PATH, load_frozen_chunks  # noqa: E402
from src.benchmark.dataset.smoke_chunks import SMOKE_CHUNK_IDS  # noqa: E402
from src.benchmark.manifest_replay import record_run, replay_run  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    record = subcommands.add_parser("record", help="perform an explicitly enabled live recording")
    record.add_argument("--live", action="store_true", help="required acknowledgement of live API calls")
    record.add_argument("--dataset", type=Path, default=CHUNKS_PATH)
    record.add_argument("--ontology", type=Path, default=REPO_ROOT / "ontology.yaml")
    record.add_argument("--prompts-dir", type=Path, default=REPO_ROOT / "prompts")
    record.add_argument("--output-dir", type=Path, required=True)
    record.add_argument("--allow-manifest-calls", type=int, required=True)
    record.add_argument("--max-tokens", type=int, default=2048)
    record.add_argument("--temperature", type=float, default=0.0)

    replay = subcommands.add_parser("replay", help="replay a recording without network access")
    replay.add_argument("--recording-dir", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.add_argument("--compare-whitespace", action="store_true")
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "record":
        if not args.live:
            parser.error("record requires --live; live calls are never enabled by default")
        if not 1 <= args.allow_manifest_calls <= 4:
            parser.error("--allow-manifest-calls must be between 1 and 4")
    return args


def _load_pinned_chunks(dataset: Path) -> list[dict]:
    # The committed frozen dataset gets its full manifest/hash verification.
    if dataset.resolve() == CHUNKS_PATH.resolve():
        chunks, _manifest = load_frozen_chunks()
    else:
        # An explicit alternate dataset is useful only for isolated tests; the
        # exact two pinned IDs and order remain non-negotiable.
        chunks = []
        for line in dataset.read_text(encoding="utf-8").splitlines():
            if line.strip():
                import json

                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("dataset rows must be objects")
                chunks.append(row)
    by_id = {chunk.get("chunk_id"): chunk for chunk in chunks}
    if any(chunk_id not in by_id for chunk_id in SMOKE_CHUNK_IDS):
        raise ValueError("dataset does not contain both pinned smoke chunks")
    selected = [by_id[chunk_id] for chunk_id in SMOKE_CHUNK_IDS]
    if [chunk["chunk_id"] for chunk in selected] != list(SMOKE_CHUNK_IDS):
        raise ValueError("smoke chunk order mismatch")
    return selected


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        if args.command == "record":
            chunks = _load_pinned_chunks(args.dataset)
            manifest = record_run(
                output_dir=args.output_dir,
                chunks=chunks,
                ontology_path=args.ontology,
                prompts_dir=args.prompts_dir,
                max_live_calls=args.allow_manifest_calls,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            count = manifest["raw_responses"]["attempt_count"]
            print(f"record status={manifest['status']} attempts={count} manifest=recording.json")
            if manifest["status"] == "budget_aborted":
                return 3
            if manifest["status"] != "complete":
                return 4
        else:
            report = replay_run(
                recording_dir=args.recording_dir,
                output_dir=args.output_dir,
                whitespace_comparison=args.compare_whitespace,
            )
            if report["status"] == "matched":
                extractions = report["exact"]["files"]["extractions"]["actual_sha256"]
                rejects = report["exact"]["files"]["rejects"]["actual_sha256"]
                print(
                    f"replay status=matched attempts={report['attempt_count']} "
                    f"extractions_sha256={extractions} rejects_sha256={rejects}"
                )
            else:
                print(
                    f"replay status={report['status']} attempts={report['attempt_count']}"
                )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        # Recording data, remote errors, paths, and secrets are intentionally
        # omitted. The exception class is sufficient for automation.
        print(f"manifest recording failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
