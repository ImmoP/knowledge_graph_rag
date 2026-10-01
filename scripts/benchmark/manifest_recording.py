#!/usr/bin/env python3
"""Record a bounded Manifest Auto run or replay it fully offline.

By default the recording uses the two pinned smoke chunks. Explicit
``--chunk-id`` selection accepts only the two controlled problem chunks,
each once, in caller order. The frozen dataset digest and selection are
validated and stored together in ``recording.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.dataset.manifest import CHUNKS_PATH  # noqa: E402
from src.benchmark.two_chunk_experiment import (  # noqa: E402
    load_recording_selection,
    recording_exit_code,
    verify_experiment_recording,
)
from src.benchmark.manifest_replay import load_recording, record_run, replay_run  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    record = subcommands.add_parser("record", help="perform an explicitly enabled live recording")
    record.add_argument("--live", action="store_true", help="required acknowledgement of live API calls")
    record.add_argument(
        "--chunk-id",
        action="append",
        default=None,
        dest="chunk_ids",
        metavar="CHUNK_ID",
        help=(
            "repeat for each of c_ca129edd34052284 and c_8a84177f479aea1f; "
            "caller order is preserved. Omit both flags for the pinned smoke chunks."
        ),
    )
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
    replay.add_argument(
        "--offline-diagnostics",
        action="store_true",
        help="write a deterministic shadow report from verified recording artifacts",
    )
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


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        if args.command == "record":
            chunks, selection = load_recording_selection(
                args.dataset,
                tuple(args.chunk_ids) if args.chunk_ids is not None else None,
            )
            manifest = record_run(
                output_dir=args.output_dir,
                chunks=chunks,
                ontology_path=args.ontology,
                prompts_dir=args.prompts_dir,
                max_live_calls=args.allow_manifest_calls,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                selection=selection,
            )
            count = manifest["raw_responses"]["attempt_count"]
            print(f"record status={manifest['status']} attempts={count} manifest=recording.json")
            return recording_exit_code(manifest)
        else:
            if args.compare_whitespace:
                verify_experiment_recording(load_recording(args.recording_dir))
            report = replay_run(
                recording_dir=args.recording_dir,
                output_dir=args.output_dir,
                whitespace_comparison=args.compare_whitespace,
                offline_diagnostics=args.offline_diagnostics,
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