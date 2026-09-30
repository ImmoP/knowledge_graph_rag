#!/usr/bin/env python3
"""Verify the two problem chunks offline, or explicitly record at most four POSTs.

Without --live this writes only preflight.json, never a simulated recording.
Live and offline outputs must be fresh private directories outside every Git tree.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.dataset.manifest import CHUNKS_PATH  # noqa: E402
from src.benchmark.manifest_recording import (  # noqa: E402
    assert_safe_bytes,
    external_directory,
    private_write,
    sha256_bytes,
)
from src.benchmark.manifest_replay import _implementation_identity, record_run  # noqa: E402
from src.benchmark.two_chunk_experiment import (  # noqa: E402
    EXPERIMENT_CHUNK_IDS,
    load_recording_selection,
    recording_exit_code,
)
from src.extract import prompts, schema  # noqa: E402

MAX_HTTP_POST_ATTEMPTS = 4


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--ontology", type=Path, default=REPO_ROOT / "ontology.yaml")
    parser.add_argument("--prompts-dir", type=Path, default=REPO_ROOT / "prompts")
    parser.add_argument(
        "--recordings-root", type=Path,
        default=Path(os.environ.get("BENCHMARK_RECORDINGS_DIR", "/tmp/manifest-recordings")),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--allow-manifest-calls", type=int, default=MAX_HTTP_POST_ATTEMPTS)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--live", action="store_true", help="explicitly enable real HTTP requests")
    args = parser.parse_args(argv)
    if not 1 <= args.allow_manifest_calls <= MAX_HTTP_POST_ATTEMPTS:
        parser.error("--allow-manifest-calls must be between 1 and 4")
    if args.max_tokens <= 0 or not math.isfinite(args.temperature):
        parser.error("generation settings require positive max-tokens and finite temperature")
    if args.run_id is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.run_id = f"{stamp}-{'recording' if args.live else 'preflight'}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("--run-id must be a single directory name starting with a letter or digit")
    if args.live and not os.environ.get("MANIFEST_API_KEY"):
        parser.error("--live requires MANIFEST_API_KEY")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        chunks, selection = load_recording_selection(args.dataset, EXPERIMENT_CHUNK_IDS)
        prompt_artifacts = prompts.verify_prompt_artifacts_v2(args.prompts_dir)
        ontology_bytes = args.ontology.read_bytes()
        ontology = schema.load_ontology(args.ontology)
        schemas = {
            "entity": schema.entity_pass_v2_schema(ontology),
            "relation": schema.relation_pass_v2_schema(ontology),
        }
        for content in [ontology_bytes, *(item["bytes"] for item in prompt_artifacts.values())]:
            assert_safe_bytes(content)
        output = args.recordings_root.expanduser() / args.run_id
        if output.exists() or output.is_symlink():
            raise ValueError("output directory must be fresh")
        if args.live:
            manifest = record_run(
                output_dir=output,
                chunks=chunks,
                ontology_path=args.ontology,
                prompts_dir=args.prompts_dir,
                max_live_calls=args.allow_manifest_calls,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                selection=selection,
            )
            print(
                f"record status={manifest['status']} "
                f"attempts={manifest['raw_responses']['attempt_count']} "
                "manifest=recording.json"
            )
            return recording_exit_code(manifest)
        payload = {
            "format": "manifest-two-chunk-preflight-v1",
            "status": "verified_offline",
            "selection": selection,
            "ontology_sha256": sha256_bytes(ontology_bytes),
            "prompts_sha256": {name: item["sha256"] for name, item in prompt_artifacts.items()},
            "implementation": _implementation_identity(),
            "schemas_sha256": {
                name: sha256_bytes(json.dumps(value, sort_keys=True).encode("utf-8"))
                for name, value in schemas.items()
            },
            "max_live_calls": args.allow_manifest_calls,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "http_post_attempts": 0,
            "response_cache_enabled": False,
        }
        content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        assert_safe_bytes(content)
        output = external_directory(output, create=True)
        private_write(output / "preflight.json", content)
        print(json.dumps(payload, sort_keys=True))
        return 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        # Never echo remote content, credential-bearing filenames or exception text.
        print(f"two-chunk experiment failed: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
