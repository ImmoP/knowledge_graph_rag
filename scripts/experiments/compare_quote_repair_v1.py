#!/usr/bin/env python3
"""Strictly offline A/B replay for experimental whitespace quote repair.

The baseline and experiment consume the same content-addressed cache entries.
The experiment changes only relation ``evidence_quote`` values for which
``experimental_quote_repair.repair_quote`` returns ``repaired``. Both arms
then execute the unchanged V2 pipeline gates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.extract import pipeline_v2
from src.extract.experimental_quote_repair import REPAIR_VERSION, repair_quote
from src.providers.adapters import _schema
from src.providers.cache import _key_for_generate_json
from src.providers.protocol import Usage

DEFAULT_CHUNK_IDS = (
    "c_001452fd142ce075",
    "c_0001921855a8156b",
)


class OfflineReplayError(RuntimeError):
    """Base error for a replay that cannot be proven offline and identical."""


class OfflineCacheMiss(OfflineReplayError):
    """Raised instead of falling through to any provider on a cache miss."""


class BaselineMismatch(OfflineReplayError):
    """Raised when replayed baseline records differ from the stored baseline."""


@dataclass(frozen=True)
class ComparisonConfig:
    chunks_path: Path
    cache_dir: Path
    prompts_dir: Path
    ontology_path: Path
    baseline_jsonl: Path
    baseline_rejects: Path
    provider_name: str = "ollama"
    provider_model: str = "qwen3:8b"
    cache_format_version: str = "1"
    max_tokens: int = 2048
    temperature: float = 0.0


@dataclass(frozen=True)
class ScheduledCall:
    chunk_id: str
    phase: str
    source_text: str


@dataclass(frozen=True)
class ReplayCall:
    chunk_id: str
    phase: str
    cache_key: str
    cache_record_sha256: str
    raw_payload_sha256: str


class ReadOnlyReplayProvider:
    """LLMProvider-shaped cache reader with no network or write path."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        provider_name: str,
        provider_model: str,
        cache_format_version: str,
        schedule: Sequence[ScheduledCall],
        repair_relations: bool,
    ) -> None:
        self.name = provider_name
        self.model = provider_model
        self._llm_dir = cache_dir / "llm"
        self._cache_format_version = cache_format_version
        self._schedule = list(schedule)
        self._repair_relations = repair_relations
        self._call_index = 0
        self.calls: list[ReplayCall] = []
        self.decisions: dict[str, list[dict]] = {
            call.chunk_id: [] for call in schedule if call.phase == "relation"
        }

    def generate(self, *args, **kwargs):
        del args, kwargs
        raise OfflineReplayError("offline replay does not support generate()")

    def generate_json(
        self,
        system: str,
        user: str,
        schema: dict,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[dict, Usage]:
        if self._call_index >= len(self._schedule):
            raise OfflineReplayError("V2 issued more provider calls than scheduled")

        scheduled = self._schedule[self._call_index]
        observed_phase = _phase_from_schema(schema)
        if observed_phase != scheduled.phase:
            raise OfflineReplayError(
                "provider call order changed: "
                f"expected {scheduled.chunk_id}/{scheduled.phase}, got {observed_phase}"
            )

        key = _key_for_generate_json(
            cache_format_version=self._cache_format_version,
            provider_name=self.name,
            provider_model=self.model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            schema=schema,
        )
        cache_path = self._llm_dir / f"{key}.json"
        record, record_sha = _read_cache_record(
            cache_path,
            expected_provider_name=self.name,
            expected_provider_model=self.model,
        )
        raw_payload = record["payload"]
        try:
            _schema.validate(raw_payload, schema)
        except _schema.SchemaValidationError as exc:
            raise OfflineReplayError(
                f"cached payload fails current {observed_phase} schema: {cache_path.name}: {exc}"
            ) from exc

        raw_digest = _canonical_sha256(raw_payload)
        payload = copy.deepcopy(raw_payload)
        if self._repair_relations and observed_phase == "relation":
            payload = self._repair_payload(
                payload,
                chunk_id=scheduled.chunk_id,
                source_text=scheduled.source_text,
            )
            try:
                _schema.validate(payload, schema)
            except _schema.SchemaValidationError as exc:
                raise OfflineReplayError(
                    f"repaired payload fails relation schema for {scheduled.chunk_id}: {exc}"
                ) from exc

        self.calls.append(
            ReplayCall(
                chunk_id=scheduled.chunk_id,
                phase=observed_phase,
                cache_key=key,
                cache_record_sha256=record_sha,
                raw_payload_sha256=raw_digest,
            )
        )
        self._call_index += 1
        return payload, Usage()

    def assert_complete(self) -> None:
        if self._call_index != len(self._schedule):
            remaining = self._schedule[self._call_index :]
            labels = [f"{call.chunk_id}/{call.phase}" for call in remaining]
            raise OfflineReplayError(f"V2 did not consume scheduled cache entries: {labels}")

    def _repair_payload(self, payload: dict, *, chunk_id: str, source_text: str) -> dict:
        assertions = payload.get("assertions")
        if not isinstance(assertions, list):
            return payload

        for index, candidate in enumerate(assertions):
            if not isinstance(candidate, dict):
                continue
            quote = candidate.get("evidence_quote")
            result = repair_quote(source_text, quote)
            decision = {
                "candidate_index": index,
                "status": result.status,
                "original_quote": quote,
                "candidate_identity": {
                    "relation_type": candidate.get("relation_type"),
                    "subject_ref": copy.deepcopy(candidate.get("subject_ref")),
                    "object_ref": copy.deepcopy(candidate.get("object_ref")),
                },
                "replacement_quote": result.quote if result.status == "repaired" else None,
                "char_start": result.char_start,
                "char_end": result.char_end,
            }
            self.decisions[chunk_id].append(decision)
            if result.status == "repaired":
                candidate["evidence_quote"] = result.quote
        return payload


def _phase_from_schema(schema: dict) -> str:
    properties = schema.get("properties")
    if isinstance(properties, dict) and "mentions" in properties:
        return "entity"
    if isinstance(properties, dict) and "assertions" in properties:
        return "relation"
    raise OfflineReplayError("unrecognized generate_json schema")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_cache_record(
    path: Path,
    *,
    expected_provider_name: str,
    expected_provider_model: str,
) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise OfflineCacheMiss(f"required cache entry missing: {path.name}") from exc
    except OSError as exc:
        raise OfflineReplayError(f"cannot read cache entry {path.name}: {exc}") from exc

    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OfflineReplayError(f"invalid cache entry {path.name}: {exc}") from exc

    if not isinstance(record, dict):
        raise OfflineReplayError(f"cache entry is not an object: {path.name}")
    expected_metadata = {
        "operation": "generate_json",
        "provider_name": expected_provider_name,
        "provider_model": expected_provider_model,
    }
    for field, expected in expected_metadata.items():
        if record.get(field) != expected:
            raise OfflineReplayError(
                f"cache entry {path.name} has {field}={record.get(field)!r}, expected {expected!r}"
            )
    if not isinstance(record.get("payload"), dict):
        raise OfflineReplayError(f"cache entry has no object payload: {path.name}")
    return record, hashlib.sha256(raw).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise OfflineReplayError(f"cannot read JSONL {path}: {exc}") from exc
    records: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OfflineReplayError(
                f"invalid JSONL in {path} at line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise OfflineReplayError(f"non-object JSONL record in {path} at line {line_number}")
        records.append(value)
    return records


def _select_chunks(path: Path, chunk_ids: Sequence[str]) -> list[dict]:
    if len(set(chunk_ids)) != len(chunk_ids):
        raise OfflineReplayError("chunk IDs must be unique")
    wanted = set(chunk_ids)
    selected = [record for record in _read_jsonl(path) if record.get("chunk_id") in wanted]
    observed = [record.get("chunk_id") for record in selected]
    missing = wanted.difference(observed)
    if missing:
        raise OfflineReplayError(f"requested chunks not found: {sorted(missing)}")
    if len(selected) != len(chunk_ids):
        raise OfflineReplayError("chunk input contains duplicate requested chunk IDs")
    for chunk in selected:
        if not isinstance(chunk.get("text"), str):
            raise OfflineReplayError(f"chunk has no string text: {chunk.get('chunk_id')}")
    return selected


def _write_selected_chunks(path: Path, chunks: Sequence[dict]) -> None:
    path.write_text(
        "".join(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n" for chunk in chunks),
        encoding="utf-8",
    )


def _schedule(chunks: Sequence[dict]) -> list[ScheduledCall]:
    return [
        ScheduledCall(chunk_id=chunk["chunk_id"], phase=phase, source_text=chunk["text"])
        for chunk in chunks
        for phase in ("entity", "relation")
    ]


def _run_arm(
    *,
    config: ComparisonConfig,
    chunks: Sequence[dict],
    chunks_path: Path,
    output_dir: Path,
    arm: str,
    repair_relations: bool,
) -> tuple[list[dict], list[dict], ReadOnlyReplayProvider]:
    provider = ReadOnlyReplayProvider(
        cache_dir=config.cache_dir,
        provider_name=config.provider_name,
        provider_model=config.provider_model,
        cache_format_version=config.cache_format_version,
        schedule=_schedule(chunks),
        repair_relations=repair_relations,
    )
    output_jsonl = output_dir / f"{arm}.v2.jsonl"
    output_rejects = output_dir / f"{arm}.v2.rejects.jsonl"
    result = pipeline_v2.run_extraction_v2(
        provider=provider,
        output_jsonl=output_jsonl,
        output_rejects=output_rejects,
        chunks_path=chunks_path,
        ontology_path=config.ontology_path,
        prompts_dir=config.prompts_dir,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    provider.assert_complete()
    if result.exit_code != 0 or result.budget_aborted:
        raise OfflineReplayError(f"{arm} V2 replay aborted unexpectedly")
    return _read_jsonl(output_jsonl), _read_jsonl(output_rejects), provider


def _records_for_chunk(records: Sequence[dict], chunk_id: str) -> dict:
    status = next(
        (record for record in records if record.get("chunk_id") == chunk_id and record.get("kind") == "chunk_status"),
        None,
    )
    if status is None:
        raise OfflineReplayError(f"missing chunk_status for {chunk_id}")
    return {
        "accepted_mentions": [
            record for record in records
            if record.get("chunk_id") == chunk_id and record.get("kind") == "mention"
        ],
        "accepted_assertions": [
            record for record in records
            if record.get("chunk_id") == chunk_id and record.get("kind") == "assertion"
        ],
        "chunk_status": status,
        "extraction_complete": status.get("extraction_complete") is True,
    }


def _summary(records: Sequence[dict], rejects: Sequence[dict]) -> dict:
    return {
        "chunk_count": sum(record.get("kind") == "chunk_status" for record in records),
        "complete_chunk_count": sum(
            record.get("kind") == "chunk_status" and record.get("extraction_complete") is True
            for record in records
        ),
        "accepted_mention_count": sum(record.get("kind") == "mention" for record in records),
        "accepted_assertion_count": sum(record.get("kind") == "assertion" for record in records),
        "rejected_candidate_count": len(rejects),
    }


def _assert_stored_baseline(
    *,
    generated_records: Sequence[dict],
    generated_rejects: Sequence[dict],
    config: ComparisonConfig,
    chunk_ids: Sequence[str],
) -> dict:
    wanted = set(chunk_ids)
    stored_records = [
        record for record in _read_jsonl(config.baseline_jsonl)
        if record.get("chunk_id") in wanted
    ]
    stored_rejects = [
        record for record in _read_jsonl(config.baseline_rejects)
        if record.get("chunk_id") in wanted
    ]
    if list(generated_records) != stored_records:
        raise BaselineMismatch("replayed baseline records differ from stored baseline subset")
    if list(generated_rejects) != stored_rejects:
        raise BaselineMismatch("replayed baseline rejects differ from stored baseline subset")
    return {
        "records_sha256": _canonical_sha256(stored_records),
        "rejects_sha256": _canonical_sha256(stored_rejects),
        "exact_match": True,
    }


def run_comparison(
    config: ComparisonConfig,
    *,
    chunk_ids: Sequence[str] = DEFAULT_CHUNK_IDS,
) -> dict:
    """Replay both arms and return a deterministic, audit-oriented report."""
    chunks = _select_chunks(config.chunks_path, chunk_ids)

    with tempfile.TemporaryDirectory(prefix="quote-repair-v1-") as tmp:
        root = Path(tmp)
        selected_chunks_path = root / "selected_chunks.jsonl"
        _write_selected_chunks(selected_chunks_path, chunks)
        baseline_records, baseline_rejects, baseline_provider = _run_arm(
            config=config,
            chunks=chunks,
            chunks_path=selected_chunks_path,
            output_dir=root,
            arm="baseline",
            repair_relations=False,
        )
        experiment_records, experiment_rejects, experiment_provider = _run_arm(
            config=config,
            chunks=chunks,
            chunks_path=selected_chunks_path,
            output_dir=root,
            arm="experiment",
            repair_relations=True,
        )

    baseline_calls = [asdict(call) for call in baseline_provider.calls]
    experiment_calls = [asdict(call) for call in experiment_provider.calls]
    if baseline_calls != experiment_calls:
        raise OfflineReplayError("baseline and experiment did not consume identical raw cache entries")

    stored_baseline = _assert_stored_baseline(
        generated_records=baseline_records,
        generated_rejects=baseline_rejects,
        config=config,
        chunk_ids=chunk_ids,
    )

    chunk_reports: list[dict] = []
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        baseline_state = _records_for_chunk(baseline_records, chunk_id)
        experiment_state = _records_for_chunk(experiment_records, chunk_id)
        baseline_state["rejected_candidates"] = [
            record for record in baseline_rejects if record.get("chunk_id") == chunk_id
        ]
        experiment_state["rejected_candidates"] = [
            record for record in experiment_rejects if record.get("chunk_id") == chunk_id
        ]
        chunk_reports.append(
            {
                "chunk_id": chunk_id,
                "baseline_starting_state": baseline_state,
                "reconstruction_decisions": experiment_provider.decisions.get(chunk_id, []),
                "experiment_state": experiment_state,
            }
        )

    baseline_summary = _summary(baseline_records, baseline_rejects)
    experiment_summary = _summary(experiment_records, experiment_rejects)
    decision_counts = Counter(
        decision["status"]
        for decisions in experiment_provider.decisions.values()
        for decision in decisions
    )
    return {
        "experiment": "quote-repair-v1",
        "repair_version": REPAIR_VERSION,
        "offline_guarantee": (
            "Read-only content-addressed cache replay; a missing or malformed entry raises and "
            "there is no provider fallback or network-capable adapter."
        ),
        "validation_scope": {
            "technical": (
                "Both arms use identical stored parsed responses and the unchanged official V2 "
                "alignment, ontology, endpoint-containment, and evidence checks after the optional repair."
            ),
            "scientific": (
                "Not established by this replay. Accepted assertions still require independent "
                "domain review or a labeled gold set for semantic correctness and usefulness."
            ),
        },
        "inputs": {
            "selected_chunk_ids": [chunk["chunk_id"] for chunk in chunks],
            "chunks_sha256": hashlib.sha256(config.chunks_path.read_bytes()).hexdigest(),
            "ontology_sha256": hashlib.sha256(config.ontology_path.read_bytes()).hexdigest(),
            "prompt_entity_sha256": hashlib.sha256(
                (config.prompts_dir / pipeline_v2.V2_ENTITY_PROMPT_FILENAME).read_bytes()
            ).hexdigest(),
            "prompt_relation_sha256": hashlib.sha256(
                (config.prompts_dir / pipeline_v2.V2_RELATION_PROMPT_FILENAME).read_bytes()
            ).hexdigest(),
            "provider_name": config.provider_name,
            "provider_model": config.provider_model,
            "cache_format_version": config.cache_format_version,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        },
        "raw_response_identity": {
            "identical_between_arms": True,
            "calls": baseline_calls,
        },
        "stored_baseline_verification": stored_baseline,
        "baseline_summary": baseline_summary,
        "experiment_summary": experiment_summary,
        "delta": {
            key: experiment_summary[key] - baseline_summary[key]
            for key in baseline_summary
        },
        "reconstruction_status_counts": dict(sorted(decision_counts.items())),
        "chunks": chunk_reports,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--prompts-dir", required=True, type=Path)
    parser.add_argument("--baseline-jsonl", required=True, type=Path)
    parser.add_argument("--baseline-rejects", required=True, type=Path)
    parser.add_argument("--ontology", type=Path, default=REPO_ROOT / "ontology.yaml")
    parser.add_argument("--output", type=Path, help="Write JSON report here; default: stdout")
    parser.add_argument("--provider-name", default="ollama")
    parser.add_argument("--provider-model", default="qwen3:8b")
    parser.add_argument("--cache-format-version", default="1")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ComparisonConfig(
        chunks_path=args.chunks,
        cache_dir=args.cache_dir,
        prompts_dir=args.prompts_dir,
        ontology_path=args.ontology,
        baseline_jsonl=args.baseline_jsonl,
        baseline_rejects=args.baseline_rejects,
        provider_name=args.provider_name,
        provider_model=args.provider_model,
        cache_format_version=args.cache_format_version,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    try:
        report = run_comparison(config)
    except OfflineReplayError as exc:
        print(f"offline comparison failed closed: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
