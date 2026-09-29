#!/usr/bin/env python3
"""Initial limited live comparison smoke run.

Usage::

    # Offline dry-run; emits an empty summary so the wiring is
    # exercised end-to-end without network calls.
    python scripts/benchmark/run_smoke.py --dry-run \
        --dataset src/benchmark/dataset/benchmark_chunks.jsonl \
        --output-dir /tmp/bench-runs/

    # Online local Qwen run (no manifest calls).
    python scripts/benchmark/run_smoke.py --qwen \
        --dataset src/benchmark/dataset/benchmark_chunks.jsonl \
        --output-dir /tmp/bench-runs/

    # Online fixed-Terra manifest run (controlled comparison).
    python scripts/benchmark/run_smoke.py --manifest \
        --dataset src/benchmark/dataset/benchmark_chunks.jsonl \
        --output-dir /tmp/bench-runs/

    # Online Manifest auto-routing system benchmark. Sends model='auto',
    # never invokes Ollama, disables the response cache, and allows at
    # most four actual HTTP POST attempts including retries / JSON repair.
    python scripts/benchmark/run_smoke.py --manifest-auto \
        --allow-manifest-calls 4 \
        --dataset src/benchmark/dataset/benchmark_chunks.jsonl \
        --output-dir /tmp/bench-runs/

The smoke run processes exactly two chunks
(``c_0001921855a8156b`` and ``c_001452fd142ce075``) and refuses
to execute without ``--dry-run``, ``--qwen``, ``--manifest``, or
``--manifest-auto``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.dataset.manifest import (  # noqa: E402
    load_frozen_chunks,
    verify_dataset_files,
)
from src.benchmark.dataset.smoke_chunks import (  # noqa: E402
    SMOKE_CHUNK_IDS,
    SMOKE_MAX_LIVE_CALLS,
    SMOKE_MAX_LOCAL_CALLS,
)
from src.benchmark.runner import (  # noqa: E402
    TimingProvider,
    new_run_paths,
    run_provider_on_chunks,
    write_run_summary,
)
from src.extract import prompts as prompts_module  # noqa: E402
from src.extract.pipeline_v2 import RunResultV2  # noqa: E402
from src.providers.protocol import ProviderTransportError  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(os.environ.get("BENCHMARK_RUNS_DIR", "/tmp/bench-runs"))
QWEN_MODEL_ID = "qwen3:8b"
MANIFEST_MODEL_ID = "openai/gpt-5.6-terra-subscription"
MANIFEST_AUTO_MODEL_ID = "auto"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "src" / "benchmark" / "dataset" / "benchmark_chunks.jsonl",
    )
    parser.add_argument("--ontology", type=Path, default=REPO_ROOT / "ontology.yaml")
    parser.add_argument("--prompts-dir", type=Path, default=REPO_ROOT / "prompts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip both adapters; emit an empty summary so wiring is exercised.",
    )
    parser.add_argument(
        "--qwen",
        action="store_true",
        help="Drive the local Qwen3 8B adapter on the smoke chunks.",
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help=(
            "Drive the manifest adapter on the smoke chunks with the "
            "fixed Terra model (controlled-comparison mode)."
        ),
    )
    parser.add_argument(
        "--manifest-auto",
        action="store_true",
        help=(
            "Drive Manifest routing with model='auto'. No Ollama call; "
            "response cache disabled; at most --allow-manifest-calls HTTP POSTs."
        ),
    )
    parser.add_argument(
        "--allow-local-qwen-calls",
        type=int,
        default=SMOKE_MAX_LOCAL_CALLS,
        help="Cap on local Qwen live calls.",
    )
    parser.add_argument(
        "--allow-manifest-calls",
        type=int,
        default=SMOKE_MAX_LIVE_CALLS,
        help="Cap on manifest live calls.",
    )
    args = parser.parse_args(argv)
    if args.manifest_auto and (args.qwen or args.manifest):
        parser.error("--manifest-auto cannot be combined with --qwen or --manifest")
    if args.manifest_auto and args.allow_manifest_calls > SMOKE_MAX_LIVE_CALLS:
        parser.error(
            f"--manifest-auto allows at most {SMOKE_MAX_LIVE_CALLS} HTTP POST attempts"
        )
    return args


def _select_smoke_chunks(all_chunks: list[dict]) -> list[dict]:
    by_id = {c["chunk_id"]: c for c in all_chunks}
    out: list[dict] = []
    for cid in SMOKE_CHUNK_IDS:
        if cid not in by_id:
            raise KeyError(f"smoke chunk {cid!r} not present in dataset")
        out.append(by_id[cid])
    return out


def _build_qwen_adapter():
    from src.providers.adapters.ollama import OllamaLLM
    from src.providers.registry import _build_cache

    cache = _build_cache(os.environ.copy())
    return OllamaLLM(
        model=QWEN_MODEL_ID,
        cache=cache,
    )


def _build_manifest_adapter(*, max_live_calls: int, call_counter: list[int]):
    from src.providers.adapters.manifest_openai import (
        ManifestBenchmarkLLM,
    )
    from src.providers.cache import CachePaths, ProviderCache

    # Isolated cache namespace so the manifest cache never collides
    # with the production ollama cache.
    manifest_cache_root = (
        Path(os.environ["LLM_CACHE_DIR"])
        if os.environ.get("LLM_CACHE_DIR")
        else REPO_ROOT / "data" / "derived" / "llm_cache"
    ) / "manifest_openai"
    cache = ProviderCache(
        CachePaths.from_root(manifest_cache_root),
        cache_format_version="manifest-openai-1",
    )
    return ManifestBenchmarkLLM(
        model=MANIFEST_MODEL_ID,
        cache=cache,
        max_live_calls=max_live_calls,
        call_counter=call_counter,
    )


def _build_manifest_auto_adapter(*, max_live_calls: int, call_counter: list[int]):
    """Build the dedicated Manifest auto-routing benchmark adapter.

    ``model="auto"`` is sent verbatim. The adapter disables the
    response cache and accepts whatever concrete model identifier
    Manifest reports for each request; missing identifiers remain
    explicitly unknown.
    """
    from src.providers.adapters.manifest_openai import ManifestAutoBenchmarkLLM

    return ManifestAutoBenchmarkLLM(
        model=MANIFEST_AUTO_MODEL_ID,
        cache=None,
        max_live_calls=max_live_calls,
        call_counter=call_counter,
    )


def _detect_local_baseline_cache_comparable() -> dict:
    """Inspect the local Ollama cache for prior Qwen responses on
    the two smoke chunks. The local cache is comparable to the
    benchmark dataset only if the model id, prompt sha, schema, and
    chunk text are byte-identical. Otherwise we mark the cached
    responses as not-comparable and require a fresh local call.
    """
    from src.providers.registry import _build_cache
    try:
        cache = _build_cache(os.environ.copy())
    except Exception as e:
        return {"comparable": False, "reason": f"could not resolve local cache: {e}"}
    paths = cache.paths
    if not paths.llm.exists():
        return {"comparable": False, "reason": f"no local LLM cache at {paths.llm}"}

    prompt_artifacts = prompts_module.verify_prompt_artifacts_v2()
    current_entity_sha = prompt_artifacts[prompts_module.ENTITY_PROMPT_FILE_V2]["sha256"]
    current_relation_sha = prompt_artifacts[prompts_module.RELATION_PROMPT_FILE_V2]["sha256"]

    found: dict[str, dict] = {}
    for cache_file in paths.llm.glob("*.json"):
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("provider_model") != QWEN_MODEL_ID:
            continue
        if payload.get("operation") != "generate_json":
            continue
        prov_payload = payload.get("payload", {})
        user_prompt = (
            prov_payload.get("user", "") if isinstance(prov_payload, dict) else ""
        )
        for cid in SMOKE_CHUNK_IDS:
            if cid in user_prompt:
                found[cid] = {
                    "cache_file": str(cache_file),
                    "cache_sha256": hashlib.sha256(cache_file.read_bytes()).hexdigest(),
                }
                break

    return {
        "comparable": len(found) == len(SMOKE_CHUNK_IDS),
        "current_entity_prompt_sha256": current_entity_sha,
        "current_relation_prompt_sha256": current_relation_sha,
        "found_entries": found,
        "note": (
            "comparability requires model id, prompt sha, schema, "
            "and chunk text to match exactly. The current dataset "
            "and the prior pilot cache may differ in any of these. "
            "A live local call is required unless all four "
            "dimensions are proven byte-identical."
        ),
    }


def _count_partial_extractions(paths) -> dict[str, int]:
    counts = {"records": 0, "mentions": 0, "assertions": 0, "complete": 0, "rejects": 0}
    if paths.output_jsonl.is_file():
        for line in paths.output_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            counts["records"] += 1
            kind = record.get("kind")
            if kind == "mention":
                counts["mentions"] += 1
            elif kind == "assertion":
                counts["assertions"] += 1
            elif kind == "chunk_status" and record.get("extraction_complete") is True:
                counts["complete"] += 1
    if paths.output_rejects.is_file():
        counts["rejects"] = sum(
            1
            for line in paths.output_rejects.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
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
        reject_count=counts["rejects"],
        budget_aborted=budget_aborted,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = verify_dataset_files()
    chunks, _ = load_frozen_chunks(chunks_path=args.dataset)
    smoke_chunks = _select_smoke_chunks(chunks)

    if not (args.dry_run or args.qwen or args.manifest or args.manifest_auto):
        print(
            "refusing to run: pass --dry-run, --qwen, --manifest, or --manifest-auto",
            file=sys.stderr,
        )
        return 2

    paths = new_run_paths(run_root=args.output_dir)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir: {paths.run_dir}")
    print(f"smoke chunks: {[c['chunk_id'] for c in smoke_chunks]}")

    notes: dict = {
        "smoke_chunk_ids": list(SMOKE_CHUNK_IDS),
        "dataset_manifest_sha256": manifest.chunks_sha256,
        "max_local_qwen_calls": args.allow_local_qwen_calls,
        "max_manifest_calls": args.allow_manifest_calls,
        "qwen_model_id": QWEN_MODEL_ID,
        "manifest_model_id": MANIFEST_MODEL_ID,
        "manifest_auto_model_id": MANIFEST_AUTO_MODEL_ID,
    }
    if args.manifest_auto:
        notes["local_baseline_cache"] = {
            "checked": False,
            "reason": "manifest-auto mode never invokes Ollama",
        }
        notes["response_cache"] = {
            "enabled": False,
            "reason": (
                "auto routing may select different real models for identical inputs; "
                "cache disabled to prevent cross-model contamination"
            ),
        }
    else:
        notes["local_baseline_cache"] = _detect_local_baseline_cache_comparable()

    prompt_artifacts = prompts_module.verify_prompt_artifacts_v2(args.prompts_dir)
    entity_prompt_sha = prompt_artifacts["extract_entities_v2.txt"]["sha256"]
    relation_prompt_sha = prompt_artifacts["extract_relations_v2.txt"]["sha256"]
    ontology_sha = hashlib.sha256(args.ontology.read_bytes()).hexdigest()

    call_counter = [0]

    if args.qwen:
        qwen_adapter = _build_qwen_adapter()
        result, sink = run_provider_on_chunks(
            provider=qwen_adapter,
            chunks_path=args.dataset,
            chunks=smoke_chunks,
            ontology_path=args.ontology,
            prompts_dir=args.prompts_dir,
            paths=paths,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        write_run_summary(
            paths,
            provider_name="qwen_ollama",
            requested_model=QWEN_MODEL_ID,
            reported_model=QWEN_MODEL_ID,
            dataset_sha256=manifest.chunks_sha256,
            ontology_sha256=ontology_sha,
            prompt_sha256_entity=entity_prompt_sha,
            prompt_sha256_relation=relation_prompt_sha,
            chunks=smoke_chunks,
            result=result,
            sink=sink,
            notes=notes,
        )
        print(f"qwen summary: {paths.run_dir / 'run_summary.json'}")

    if args.manifest:
        manifest_adapter = _build_manifest_adapter(
            max_live_calls=args.allow_manifest_calls,
            call_counter=call_counter,
        )
        result, sink = run_provider_on_chunks(
            provider=manifest_adapter,
            chunks_path=args.dataset,
            chunks=smoke_chunks,
            ontology_path=args.ontology,
            prompts_dir=args.prompts_dir,
            paths=paths,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        write_run_summary(
            paths,
            provider_name="manifest_benchmark",
            requested_model=MANIFEST_MODEL_ID,
            reported_model=MANIFEST_MODEL_ID,
            dataset_sha256=manifest.chunks_sha256,
            ontology_sha256=ontology_sha,
            prompt_sha256_entity=entity_prompt_sha,
            prompt_sha256_relation=relation_prompt_sha,
            chunks=smoke_chunks,
            result=result,
            sink=sink,
            notes=notes,
        )
        print(f"manifest summary: {paths.run_dir / 'run_summary.json'}")

    if args.manifest_auto:
        from src.providers.adapters.manifest_openai import CallBudgetExceeded

        auto_adapter = _build_manifest_auto_adapter(
            max_live_calls=args.allow_manifest_calls,
            call_counter=call_counter,
        )
        sink: list = []
        timed_auto = TimingProvider(
            auto_adapter,
            sink=sink,
            model_label="manifest-auto",
        )
        budget_abort_reason: str | None = None
        terminal_error: BaseException | None = None
        try:
            result, _ = run_provider_on_chunks(
                provider=timed_auto,
                chunks_path=args.dataset,
                chunks=smoke_chunks,
                ontology_path=args.ontology,
                prompts_dir=args.prompts_dir,
                paths=paths,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except CallBudgetExceeded as exc:
            # Defensive preflight fallback; in extraction V2 catches this
            # BudgetExceeded subclass and returns its durable partial result.
            terminal_error = exc
            budget_abort_reason = str(exc)
            result = _partial_result(paths, exit_code=3, budget_aborted=True)
        except ProviderTransportError as exc:
            # V2 propagates terminal HTTP/transport errors. The timing
            # wrapper has already persisted per-attempt metrics; retain any
            # extraction artifacts that reached disk and always emit summary.
            terminal_error = exc
            result = _partial_result(paths, exit_code=4, budget_aborted=False)
            notes["partial_run"] = True
            notes["terminal_error"] = {
                "type": type(exc).__name__,
                "status": exc.status,
                "transient": exc.transient,
            }

        terminal_transport = auto_adapter.last_terminal_transport_error
        if result.budget_aborted and terminal_transport is not None:
            terminal_error = terminal_transport
            result.budget_aborted = False
            result.exit_code = 4
            notes["partial_run"] = True
            notes["terminal_error"] = {
                "type": "ProviderTransportError",
                "status": terminal_transport.status,
                "transient": terminal_transport.transient,
            }
        elif result.budget_aborted:
            budget_abort_reason = budget_abort_reason or (
                f"manifest HTTP POST cap reached ({args.allow_manifest_calls})"
            )
            result.exit_code = 3
            notes["partial_run"] = True
            notes["budget_abort_reason"] = budget_abort_reason
        notes["http_post_attempts"] = call_counter[0]
        write_run_summary(
            paths,
            provider_name="manifest_auto_benchmark",
            requested_model=MANIFEST_AUTO_MODEL_ID,
            # Auto routing can vary per call; never collapse the run
            # to a single underlying model. Per-phase identities live
            # in ``metrics[].reported_model``.
            reported_model=None,
            dataset_sha256=manifest.chunks_sha256,
            ontology_sha256=ontology_sha,
            prompt_sha256_entity=entity_prompt_sha,
            prompt_sha256_relation=relation_prompt_sha,
            chunks=smoke_chunks,
            result=result,
            sink=sink,
            notes=notes,
            auto_routing=True,
        )
        print(f"manifest-auto summary: {paths.run_dir / 'run_summary.json'}")
        if result.exit_code != 0:
            reason = budget_abort_reason or (
                f"{type(terminal_error).__name__}: {terminal_error}"
                if terminal_error is not None
                else "partial run"
            )
            print(
                f"manifest-auto stopped after {call_counter[0]} HTTP POST attempts: "
                f"{reason}",
                file=sys.stderr,
            )
            return result.exit_code

    if args.dry_run and not (args.qwen or args.manifest or args.manifest_auto):
        notes["dry_run"] = True
        empty_result = type("R", (), {"budget_aborted": False, "exit_code": 0})()
        write_run_summary(
            paths,
            provider_name="(dry-run)",
            requested_model="(none)",
            reported_model=None,
            dataset_sha256=manifest.chunks_sha256,
            ontology_sha256=ontology_sha,
            prompt_sha256_entity=entity_prompt_sha,
            prompt_sha256_relation=relation_prompt_sha,
            chunks=smoke_chunks,
            result=empty_result,
            sink=[],
            notes=notes,
        )
        print(f"dry-run summary: {paths.run_dir / 'run_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())