#!/usr/bin/env python3
"""Replay one verified two-chunk recording and compare its whitespace arm offline."""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.benchmark.manifest_recording import RecordingError
from src.benchmark.manifest_replay import load_recording, replay_run
from src.benchmark.two_chunk_experiment import verify_experiment_recording


def _diagnoser():
    spec = importlib.util.spec_from_file_location("two_chunk_diagnose_dpr", Path(__file__).with_name("diagnose_dpr.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("diagnosis unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = load_recording(args.recording_dir)
        verify_experiment_recording(manifest)
        source = manifest["_directory"]
        output = args.output_dir.expanduser().absolute().resolve(strict=False)
        if output == source or source in output.parents or output in source.parents:
            raise RecordingError("comparison output overlaps recording")
        report = replay_run(recording_dir=args.recording_dir, output_dir=args.output_dir, whitespace_comparison=True)
        if report.get("status") != "matched":
            print("error: replay did not produce an exact result", file=sys.stderr)
            return 1
        paths = _diagnoser().write_dpr_diagnosis(recording_dir=args.recording_dir, replay_dir=args.output_dir, output_dir=args.output_dir)
    except Exception:
        print("error: verified offline comparison failed", file=sys.stderr)
        return 1
    print(f"comparison: {args.output_dir / 'whitespace_comparison' / 'comparison.json'}")
    print(f"DPR diagnostic JSON: {paths['json']}")
    print(f"DPR diagnostic Markdown: {paths['markdown']}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
