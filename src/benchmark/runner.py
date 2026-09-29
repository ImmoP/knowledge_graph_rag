"""Benchmark runner: drive the V2 pipeline with a benchmark adapter.

The runner is intentionally a thin wrapper around
:func:`src.extract.pipeline_v2.run_extraction_v2` plus a structured
metrics layer. It does NOT introduce a parallel extraction pipeline.
Every chunk is processed by the canonical V2 runner so the
technical-validation gates are byte-identical to the production
extraction.

The runner records per-phase measurements:

* the chunk id and phase,
* the requested model identifier and the gateway-reported
* response duration in seconds,


* response duration in seconds,
* token counts returned by the provider,
* the canonical v2 status (``validated`` / ``rejected``),
* the canonical v2 reject reason (if any),
* the live-call counter at the time of the call,
* the cache hit / miss flag.

Reports are written outside the repository
(``$BENCHMARK_RUNS_DIR``, default ``/tmp/bench-runs/<run-id>``)
alongside the raw responses. Nothing about a run lands in git
unless an operator explicitly copies a report there.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.extract.pipeline_v2 import RunResultV2, run_extraction_v2


@dataclass
class RunPaths:
    """On-disk layout for one benchmark run."""

    run_dir: Path
    output_jsonl: Path
    output_rejects: Path
    metrics_path: Path
    raw_responses_dir: Path

    @classmethod
    def for_run_id(cls, run_root: Path, run_id: str) -> "RunPaths":
        run_dir = run_root / run_id
        return cls(
            run_dir=run_dir,
            output_jsonl=run_dir / "extractions.jsonl",
            output_rejects=run_dir / "extractions.rejects.jsonl",
            metrics_path=run_dir / "metrics.jsonl",
            raw_responses_dir=run_dir / "raw_responses",
        )


@dataclass
class PhaseCall:
    """One provider-call measurement."""

    chunk_id: str
    phase: str
    requested_model: str
    reported_model: str | None
    duration_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    cache_hit: bool
    live_call_index: int
    status: str
    reject_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def default_run_root() -> Path:
    env = os.environ.get("BENCHMARK_RUNS_DIR")
    if env:
        return Path(env)
    return Path("/tmp/bench-runs")


def _utc_run_id() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class TimingProvider:
    """Wrapper that records per-call measurements around a benchmark
    adapter without altering its behaviour. The wrapped adapter is
    expected to satisfy the :class:`LLMProvider` protocol.
    """

    def __init__(self, inner: Any, *, sink: list[PhaseCall], model_label: str) -> None:
        self._inner = inner
        self._sink = sink
        self._model_label = model_label

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def model(self) -> str:
        return self._inner.model

    def _cache_hit(self, args: dict[str, Any]) -> bool:
        cache = getattr(self._inner, "_cache", None)
        if cache is None:
            return False
        if "schema" in args:
            cached = cache.get_generate_json(
                provider_name=self.name,
                provider_model=self.model,
                system=args["system"],
                user=args["user"],
                max_tokens=args.get("max_tokens", 1024),
                temperature=args.get("temperature", 0.0),
                schema=args["schema"],
            )
        else:
            cached = cache.get_generate(
                provider_name=self.name,
                provider_model=self.model,
                system=args["system"],
                user=args["user"],
                max_tokens=args.get("max_tokens", 1024),
                temperature=args.get("temperature", 0.0),
            )
        return cached is not None

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[str, Any]:
        cache_hit = self._cache_hit({
            "system": system, "user": user,
            "max_tokens": max_tokens, "temperature": temperature,
        })
        start = time.perf_counter()
        text, usage = self._inner.generate(
            system, user, max_tokens=max_tokens, temperature=temperature
        )
        duration = time.perf_counter() - start
        live_idx = getattr(self._inner, "live_call_count", 0)
        self._sink.append(PhaseCall(
            chunk_id="(unknown)",
            phase="entity_or_relation",
            requested_model=self._inner.model,
            reported_model=self._inner.model,
            duration_seconds=duration,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
            cache_hit=cache_hit,
            live_call_index=live_idx,
            status="ok",
        ))
        return text, usage

    def generate_json(
        self,
        system: str,
        user: str,
        schema: dict,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[dict, Any]:
        cache_hit = self._cache_hit({
            "system": system, "user": user,
            "max_tokens": max_tokens, "temperature": temperature,
            "schema": schema,
        })
        start = time.perf_counter()
        parsed, usage = self._inner.generate_json(
            system, user, schema, max_tokens=max_tokens, temperature=temperature
        )
        duration = time.perf_counter() - start
        live_idx = getattr(self._inner, "live_call_count", 0)
        self._sink.append(PhaseCall(
            chunk_id="(unknown)",
            phase="entity_or_relation",
            requested_model=self._inner.model,
            reported_model=self._inner.model,
            duration_seconds=duration,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
            cache_hit=cache_hit,
            live_call_index=live_idx,
            status="ok",
        ))
        return parsed, usage


def _attach_chunk_id(call: PhaseCall, chunk_id: str, phase: str, status: str, reject_reason: str | None = None) -> None:
    """Patch a recorded call with the chunk/phase metadata that the
    V2 pipeline knows about post-hoc."""
    call.chunk_id = chunk_id
    call.phase = phase
    call.status = status
    call.reject_reason = reject_reason

def run_provider_on_chunks(
    *,
    provider: Any,
    chunks_path: Path,
    chunks: list[dict],
    ontology_path: Path,
    prompts_dir: Path,
    paths: RunPaths,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> tuple[RunResultV2, list[PhaseCall]]:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.raw_responses_dir.mkdir(parents=True, exist_ok=True)

    fd, tmp_chunks = tempfile.mkstemp(prefix="chunks.", suffix=".jsonl")
    import os as _os

    try:
        with _os.fdopen(fd, "w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c, sort_keys=True, ensure_ascii=False))
                fh.write("\n")
        sink: list[PhaseCall] = []
        if isinstance(provider, TimingProvider):
            timed = provider
        else:
            timed = TimingProvider(
                provider,
                sink=sink,
                model_label=getattr(provider, "name", "?"),
            )
        result = run_extraction_v2(
            provider=timed,
            output_jsonl=paths.output_jsonl,
            output_rejects=paths.output_rejects,
            chunks_path=Path(tmp_chunks),
            ontology_path=ontology_path,
            prompts_dir=prompts_dir,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        _decorate_sink_with_chunk_metadata(sink, paths, chunks)
        _write_metrics(paths.metrics_path, sink)
        return result, sink
    finally:
        try:
            _os.unlink(tmp_chunks)
        except FileNotFoundError:
            pass



def _decorate_sink_with_chunk_metadata(
    sink: list[PhaseCall],
    paths: RunPaths,
    chunks: list[dict],
) -> None:
    """Backfill chunk_id, phase, status, reject_reason on the sink.

    The V2 pipeline reports ``chunk_status`` records with
    ``extraction_complete`` and ``attempt_count`` per chunk. The
    rejects jsonl carries the gate reason. We map each sink entry
    (in arrival order) to the chunk whose phase it served; the
    sink order is determined by the V2 runner (entity then relation
    per chunk, chunks iterated in input order).
    """
    statuses_by_chunk: dict[str, dict] = {}
    rejects_by_chunk: dict[str, list[dict]] = {}
    if paths.output_jsonl.is_file():
        for line in paths.output_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") == "chunk_status":
                statuses_by_chunk[rec["chunk_id"]] = rec
    if paths.output_rejects.is_file():
        for line in paths.output_rejects.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rejects_by_chunk.setdefault(rec["chunk_id"], []).append(rec)

    # The V2 runner iterates chunks in input order; for each chunk
    # it makes one entity call and then one relation call. Sink
    # entries arrive in the same order.
    chunk_ids = [c["chunk_id"] for c in chunks]
    pos = 0
    for cid in chunk_ids:
        for phase in ("entity", "relation"):
            if pos >= len(sink):
                return
            call = sink[pos]
            pos += 1
            call.chunk_id = cid
            call.phase = phase
            status_rec = statuses_by_chunk.get(cid, {})
            call.status = (
                "validated"
                if status_rec.get("extraction_complete") is True
                else "rejected"
            )
            phase_rejects = [r for r in rejects_by_chunk.get(cid, []) if r.get("phase") == phase]
            if phase_rejects:
                call.reject_reason = phase_rejects[0].get("reason")


def _write_metrics(path: Path, sink: list[PhaseCall]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for call in sink:
            fh.write(json.dumps(asdict(call), ensure_ascii=False, sort_keys=True))
            fh.write("\n")


def new_run_paths(*, run_root: Path | None = None) -> RunPaths:
    root = run_root or default_run_root()
    run_id = _utc_run_id() + "-" + os.urandom(3).hex()
    return RunPaths.for_run_id(root, run_id)


def write_run_summary(
    paths: RunPaths,
    *,
    provider_name: str,
    requested_model: str,
    reported_model: str | None,
    dataset_sha256: str,
    ontology_sha256: str,
    prompt_sha256_entity: str,
    prompt_sha256_relation: str,
    chunks: list[dict],
    result: RunResultV2,
    sink: list[PhaseCall],
    notes: dict[str, Any] | None = None,
) -> Path:
    """Write a deterministic, content-free run summary as JSON."""
    payload = {
        "provider": {
            "name": provider_name,
            "requested_model": requested_model,
            "reported_model": reported_model,
            "model_identity_verified": requested_model == reported_model,
        },
        "dataset_sha256": dataset_sha256,
        "ontology_sha256": ontology_sha256,
        "prompts": {
            "entity_sha256": prompt_sha256_entity,
            "relation_sha256": prompt_sha256_relation,
        },
        "chunks": [c["chunk_id"] for c in chunks],
        "n_chunks": len(chunks),
        "result": {
            "n_extraction_complete": sum(
                1 for r in (paths.output_jsonl.read_text().splitlines() if paths.output_jsonl.is_file() else [])
                if r.strip()
                for rec in [json.loads(r)]
                if rec.get("kind") == "chunk_status" and rec.get("extraction_complete") is True
            ),
            "budget_aborted": result.budget_aborted,
            "exit_code": result.exit_code,
        },
        "metrics": [asdict(c) for c in sink],
        "notes": notes or {},
    }
    out = paths.run_dir / "run_summary.json"
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out