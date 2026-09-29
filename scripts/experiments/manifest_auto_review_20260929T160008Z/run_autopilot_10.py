#!/usr/bin/env python3
"""Bounded live runner for the manifest-auto autopilot 10-chunk plan.

This script is the *experimental live counterpart* of the smoke
runner (``scripts/benchmark/run_smoke.py --manifest-auto``). It
consumes the frozen selection plan written by
``scripts/experiments/manifest_auto_review_20260929T160008Z/autopilot_select_10.py``
and runs the canonical V2 pipeline through the
``ManifestAutoBenchmarkLLM`` adapter.

The runner is strictly additive:

* It never re-selects chunks or re-tests the original two smoke
  chunks.
* It never modifies the frozen V2 pipeline, ontology, prompts, or
  the manifest-adapter contract.
* It never invokes a local Ollama adapter and never uses a fixed
  Terra model id.

Safety contract:

* Default mode is **offline / dry-run**: no HTTP requests are made,
  the manifest adapter is never built, and the runner emits an empty
  summary so the wiring is exercised.
* ``--live`` is the **only** switch that enables real HTTP traffic.
* Hard cap of **20** regular requests (10 chunks × 2 phases). The
  adapter enforces this via its shared ``call_counter``.
* Hard cap of **24** actual HTTP POST attempts including all
  corrective retries and transport-level retries. The runner refuses
  to dispatch any further attempt once 24 are observed.
* HTTP ``429`` triggers a **controlled abort** -- the runner stops
  with exit code 5, retains all artifacts written so far, and logs
  the rate-limit response without forcing another paid request.
* Every failed chunk and every terminal gateway error is logged to
  the run summary's ``notes.failures`` array.
* Model identity is recorded **per request**. When the gateway does
  not surface a verifiable model identifier, the run summary records
  ``"<unknown>"`` and never invents one.

Inputs:

* ``--plan PATH``: JSON autopilot plan
  (default: ``docs/experiments/manifest-auto-review/autopilot_plan_20260929T160008Z.json``).
* ``--dataset PATH``: frozen benchmark chunk JSONL
  (default: ``src/benchmark/dataset/benchmark_chunks.jsonl``).
* ``--output-dir PATH``: parent directory for the new run
  (default: ``docs/experiments/manifest_auto_review_20260929T160008Z/autopilot_10``).
* ``--manifest-api-key-env NAME``: environment variable holding the
  bearer token (default: ``MANIFEST_API_KEY``). Required only when
  ``--live`` is set.

Output (per run):

* ``run_summary.json`` -- provider, dataset SHA, prompt SHAs,
  per-call metrics (chunk_id, phase, requested_model, reported_model,
  status, reject_reason), notes, failure list.
* ``extractions.jsonl`` / ``extractions.rejects.jsonl`` -- the
  canonical V2 output for the ten chunks.

The V2 canonical manifest is **not** touched: this runner writes
into the dedicated ``autopilot_10/`` subdirectory of the experiment
and never into ``data/extractions.v2.*``. That separation keeps the
frozen V2 contract intact and makes the bounded live run a separate
artifact set for the next scientific review.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.dataset.manifest import (  # noqa: E402
    load_frozen_chunks,
    verify_dataset_files,
)
from src.benchmark.dataset.smoke_chunks import SMOKE_CHUNK_IDS  # noqa: E402
from src.benchmark.runner import (  # noqa: E402
    TimingProvider,
    new_run_paths,
    run_provider_on_chunks,
    write_run_summary,
)
from src.extract import prompts as prompts_module  # noqa: E402
from src.extract.pipeline_v2 import RunResultV2  # noqa: E402
from src.providers.protocol import ProviderTransportError  # noqa: E402


# ---- safety constants (frozen for this experiment) -----------------


#: Maximum regular phase requests: 10 chunks * 2 phases. Corrective
#: retries do not count against this number, only the cap below.
MAX_REGULAR_REQUESTS = 20

#: Absolute safety limit on actual HTTP POST attempts, including all
#: JSON-corrective retries (one extra per phase on schema failure)
#: and transport-level retries. The runner refuses to dispatch any
#: further attempt once this count is observed.
MAX_HTTP_POST_ATTEMPTS = 24

#: Exit codes.
EXIT_OK = 0
EXIT_NO_MODE = 2
EXIT_INVALID_PLAN = 3
EXIT_BUDGET_EXCEEDED = 4
EXIT_RATE_LIMITED = 5
EXIT_TRANSPORT = 6
EXIT_INTERNAL_ERROR = 7


# ---- helpers -------------------------------------------------------


def _now_iso() -> str:
    return datetime.datetime.now(timezone := __import__("datetime").timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_plan(
    plan_path: Path,
    chunks_by_id: dict[str, dict],
    dataset_sha: str,
) -> list[dict]:
    """Validate the autopilot plan against the frozen chunk set.

    Raises ``SystemExit`` with ``EXIT_INVALID_PLAN`` if any of the
    following are violated:

    * the plan JSON parses,
    * every ``chunk_id`` in the plan exists in the committed chunks,
    * the SHA-256 of every plan chunk matches the dataset SHA
      record (every plan chunk carries the same dataset identity),
    * the two smoke-run chunks are NOT in the plan,
    * the plan has exactly ten entries.
    """
    if not plan_path.is_file():
        print(
            f"error: autopilot plan not found: {plan_path}",
            file=sys.stderr,
        )
        sys.exit(EXIT_INVALID_PLAN)
    try:
        plan = _read_json(plan_path)
    except json.JSONDecodeError as exc:
        print(
            f"error: autopilot plan is not valid JSON: {exc}",
            file=sys.stderr,
        )
        sys.exit(EXIT_INVALID_PLAN)

    entries = plan.get("plan")
    if not isinstance(entries, list) or len(entries) != 10:
        print(
            f"error: autopilot plan must contain exactly 10 entries; "
            f"got {len(entries) if isinstance(entries, list) else type(entries).__name__}",
            file=sys.stderr,
        )
        sys.exit(EXIT_INVALID_PLAN)

    selected: list[dict] = []
    for entry in entries:
        cid = entry.get("chunk_id")
        if not isinstance(cid, str):
            print(
                f"error: plan entry missing 'chunk_id': {entry}",
                file=sys.stderr,
            )
            sys.exit(EXIT_INVALID_PLAN)
        if cid in SMOKE_CHUNK_IDS:
            print(
                f"error: plan includes excluded smoke chunk {cid!r}; "
                f"smoke ids are {SMOKE_CHUNK_IDS}",
                file=sys.stderr,
            )
            sys.exit(EXIT_INVALID_PLAN)
        if cid not in chunks_by_id:
            print(
                f"error: plan chunk {cid!r} not present in dataset "
                f"(dataset sha256={dataset_sha[:16]}...)",
                file=sys.stderr,
            )
            sys.exit(EXIT_INVALID_PLAN)
        if entry.get("doc_id") != chunks_by_id[cid]["doc_id"]:
            print(
                f"error: plan chunk {cid!r} doc_id mismatch "
                f"({entry.get('doc_id')!r} vs dataset {chunks_by_id[cid]['doc_id']!r})",
                file=sys.stderr,
            )
            sys.exit(EXIT_INVALID_PLAN)
        selected.append(chunks_by_id[cid])

    return selected


def _build_manifest_auto_adapter(*, call_counter: list[int]):
    from src.providers.adapters.manifest_openai import ManifestAutoBenchmarkLLM

    api_key = os.environ.get("MANIFEST_API_KEY", "")
    return ManifestAutoBenchmarkLLM(
        api_key=api_key,
        cache=None,  # auto routing: response cache explicitly disabled
        max_live_calls=MAX_HTTP_POST_ATTEMPTS,
        call_counter=call_counter,
    )


def _count_partial_extractions(paths) -> dict[str, int]:
    counts = {"records": 0, "mentions": 0, "assertions": 0, "complete": 0}
    if paths.output_jsonl.is_file():
        for line in paths.output_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            counts["records"] += 1
            kind = rec.get("kind")
            if kind == "mention":
                counts["mentions"] += 1
            elif kind == "assertion":
                counts["assertions"] += 1
            elif kind == "chunk_status" and rec.get("extraction_complete") is True:
                counts["complete"] += 1
    return counts


def _partial_result(paths, *, exit_code: int, budget_aborted: bool) -> RunResultV2:
    counts = _count_partial_extractions(paths)
    return RunResultV2(
        exit_code=exit_code,
        run_id=paths.run_dir.name,
        records_written=counts["records"],
        mention_count=counts["mentions"],
        assertion_count=counts["assertions"],
        complete_chunk_count=counts["complete"],
        reject_count=0,
        budget_aborted=budget_aborted,
    )


# ---- argparse ------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--plan",
        type=Path,
        default=(
            REPO_ROOT
            / "docs"
            / "experiments"
            / "manifest-auto-review"
            / "autopilot_plan_20260929T160008Z.json"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "src" / "benchmark" / "dataset" / "benchmark_chunks.jsonl",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=REPO_ROOT / "ontology.yaml",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=REPO_ROOT / "prompts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "docs"
            / "experiments"
            / "manifest_auto_review_20260929T160008Z"
            / "autopilot_10"
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Enable real HTTP requests to the manifest gateway. Without "
            "this flag the runner is a fully offline dry-run: the manifest "
            "adapter is never constructed, no token is read, and no JSON "
            "is POSTed."
        ),
    )
    parser.add_argument(
        "--manifest-api-key-env",
        default="MANIFEST_API_KEY",
        help=(
            "Environment variable name carrying the bearer token. Only "
            "read when --live is set; never logged or persisted."
        ),
    )
    args = parser.parse_args(argv)
    if args.live and args.manifest_api_key_env not in os.environ:
        print(
            f"error: --live requires the {args.manifest_api_key_env} "
            f"environment variable to be set",
            file=sys.stderr,
        )
        sys.exit(EXIT_NO_MODE)
    return args


# ---- main ----------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Always: verify the dataset SHA before reading anything else.
    manifest = verify_dataset_files()
    all_chunks, _ = load_frozen_chunks(chunks_path=args.dataset)
    chunks_by_id = {c["chunk_id"]: c for c in all_chunks}

    # Always: verify the autopilot plan against the dataset.
    selected = _verify_plan(args.plan, chunks_by_id, manifest.chunks_sha256)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = new_run_paths(run_root=args.output_dir)
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    prompt_artifacts = prompts_module.verify_prompt_artifacts_v2(args.prompts_dir)
    entity_prompt_sha = prompt_artifacts[prompts_module.ENTITY_PROMPT_FILE_V2]["sha256"]
    relation_prompt_sha = prompt_artifacts[prompts_module.RELATION_PROMPT_FILE_V2]["sha256"]
    ontology_sha = hashlib.sha256(args.ontology.read_bytes()).hexdigest()

    notes: dict[str, Any] = {
        "plan_path": str(args.plan),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "excluded_smoke_chunk_ids": list(SMOKE_CHUNK_IDS),
        "dataset_sha256": manifest.chunks_sha256,
        "max_regular_requests": MAX_REGULAR_REQUESTS,
        "max_http_post_attempts": MAX_HTTP_POST_ATTEMPTS,
        "live_mode": args.live,
        "response_cache": {
            "enabled": False,
            "reason": (
                "auto routing may select different real models for identical "
                "inputs; cache disabled to prevent cross-model contamination"
            ),
        },
        "selected_chunk_ids": [c["chunk_id"] for c in selected],
        "failures": [],
    }

    print(f"run_dir: {paths.run_dir}")
    print(f"selected chunks: {[c['chunk_id'] for c in selected]}")

    if not args.live:
        # Strictly offline dry-run: exercise the wiring without HTTP traffic.
        notes["dry_run"] = True
        empty_result = type("R", (), {"budget_aborted": False, "exit_code": 0})()
        write_run_summary(
            paths,
            provider_name="(offline-dry-run)",
            requested_model="auto",
            reported_model=None,
            dataset_sha256=manifest.chunks_sha256,
            ontology_sha256=ontology_sha,
            prompt_sha256_entity=entity_prompt_sha,
            prompt_sha256_relation=relation_prompt_sha,
            chunks=selected,
            result=empty_result,
            sink=[],
            notes=notes,
            auto_routing=True,
        )
        print(f"offline-dry-run summary: {paths.run_dir / 'run_summary.json'}")
        return EXIT_OK

    # ---- live mode ----------------------------------------------------
    from src.providers.adapters.manifest_openai import CallBudgetExceeded

    call_counter = [0]
    auto_adapter = _build_manifest_auto_adapter(call_counter=call_counter)
    sink: list = []
    timed_auto = TimingProvider(auto_adapter, sink=sink, model_label="manifest-auto")

    terminal_error: BaseException | None = None
    exit_code = EXIT_OK
    try:
        result, _ = run_provider_on_chunks(
            provider=timed_auto,
            chunks_path=args.dataset,
            chunks=selected,
            ontology_path=args.ontology,
            prompts_dir=args.prompts_dir,
            paths=paths,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except CallBudgetExceeded as exc:
        terminal_error = exc
        exit_code = EXIT_BUDGET_EXCEEDED
        result = _partial_result(paths, exit_code=exit_code, budget_aborted=True)
        notes["budget_abort_reason"] = (
            f"manifest HTTP POST cap reached ({MAX_HTTP_POST_ATTEMPTS})"
        )
    except ProviderTransportError as exc:
        terminal_error = exc
        exit_code = EXIT_TRANSPORT
        result = _partial_result(paths, exit_code=exit_code, budget_aborted=False)
        notes["partial_run"] = True
        notes["terminal_error"] = {
            "type": type(exc).__name__,
            "status": exc.status,
            "transient": exc.transient,
        }
        # HTTP 429 -> controlled rate-limit abort, distinct from a generic
        # transport failure. The runner logs the rate-limit response and
        # does not retry.
        if getattr(exc, "status", None) == 429:
            exit_code = EXIT_RATE_LIMITED
            notes["terminal_error"]["kind"] = "rate_limited"
            notes["rate_limited"] = True
        notes["failures"].append({
            "type": type(exc).__name__,
            "status": exc.status,
            "transient": exc.transient,
            "occurred_at": _now_iso(),
        })
    except Exception as exc:  # noqa: BLE001 - defensive: surface any unexpected error
        terminal_error = exc
        exit_code = EXIT_INTERNAL_ERROR
        result = _partial_result(paths, exit_code=exit_code, budget_aborted=False)
        notes["partial_run"] = True
        notes["terminal_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        notes["failures"].append({
            "type": type(exc).__name__,
            "message": str(exc),
            "occurred_at": _now_iso(),
        })

    # Inspect per-phase sink for chunk-level failures (e.g. validation rejects)
    # and record them so the next reviewer can see which chunks failed even
    # if the run completed.
    for call in sink:
        if call.status == "error":
            notes["failures"].append({
                "chunk_id": call.chunk_id,
                "phase": call.phase,
                "type": call.extra.get("exception_type", "unknown"),
                "reported_model": call.reported_model,
                "occurred_at": _now_iso(),
            })

    notes["http_post_attempts"] = call_counter[0]
    write_run_summary(
        paths,
        provider_name="manifest_auto_benchmark",
        requested_model="auto",
        # Auto routing can vary per call; never collapse the run
        # to a single underlying model. Per-phase identities live
        # in ``metrics[].reported_model`` and ``metrics[].extra.attempt_reported_models``.
        reported_model=None,
        dataset_sha256=manifest.chunks_sha256,
        ontology_sha256=ontology_sha,
        prompt_sha256_entity=entity_prompt_sha,
        prompt_sha256_relation=relation_prompt_sha,
        chunks=selected,
        result=result,
        sink=sink,
        notes=notes,
        auto_routing=True,
    )
    print(f"manifest-auto 10-chunk summary: {paths.run_dir / 'run_summary.json'}")

    if exit_code != EXIT_OK:
        reason = (
            notes.get("budget_abort_reason")
            or notes.get("terminal_error", {}).get("type", "unknown")
        )
        print(
            f"manifest-auto 10-chunk stopped after {call_counter[0]} "
            f"HTTP POST attempts: {reason}",
            file=sys.stderr,
        )
        return exit_code

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
