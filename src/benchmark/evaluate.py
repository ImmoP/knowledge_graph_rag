"""Benchmark evaluation: per-chunk/per-phase metrics.

The evaluator reads the canonical V2 extraction JSONL and rejects
JSONL produced by :func:`src.benchmark.runner.run_provider_on_chunks`
and computes per-chunk metrics that explicitly separate **technical
acceptance** (the V2 gates: schema, alignment, evidence, ontology)
from **scientific correctness** (a property that only human review
can determine).

Technical metrics are deterministic and reproducible. Scientific
metrics are *not* computed by this module; instead, it produces a
**blinded review package** that a human reviewer can label without
seeing which model produced which record.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChunkMetrics:
    chunk_id: str
    entity_call_count: int = 0
    entity_accepted_mention_count: int = 0
    entity_reject_count: int = 0
    entity_attempt_count: int = 0
    entity_validation_ok: bool = False
    entity_reject_reasons: tuple[str, ...] = ()
    relation_call_count: int = 0
    relation_accepted_assertion_count: int = 0
    relation_reject_count: int = 0
    relation_attempt_count: int = 0
    relation_validation_ok: bool = False
    relation_reject_reasons: tuple[str, ...] = ()
    complete: bool = False
    abort_reason: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderMetrics:
    provider_name: str
    requested_model: str
    reported_model: str | None
    model_identity_verified: bool
    dataset_sha256: str
    chunks_total: int = 0
    chunks_complete: int = 0
    chunks_with_entity_ok: int = 0
    chunks_with_relation_ok: int = 0
    mentions_accepted: int = 0
    assertions_accepted: int = 0
    candidates_rejected: int = 0
    schema_errors: int = 0
    evidence_errors: int = 0
    ontology_errors: int = 0
    alignment_errors: int = 0
    total_duration_seconds: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    per_chunk: tuple[ChunkMetrics, ...] = ()
    abort_reasons: tuple[str, ...] = ()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def _index_records(records: list[dict], rejects: list[dict]):
    """Build the per-chunk index used for metrics."""
    chunks_index: dict[str, dict] = {}
    for rec in records:
        kind = rec.get("kind")
        cid = rec.get("chunk_id")
        if not cid:
            continue
        if kind == "chunk_status":
            chunks_index.setdefault(cid, {}).update(rec)
        elif kind == "mention":
            chunks_index.setdefault(cid, {}).setdefault("_mentions", []).append(rec)
        elif kind == "assertion":
            chunks_index.setdefault(cid, {}).setdefault("_assertions", []).append(rec)

    reject_index: dict[str, list[dict]] = {}
    for rec in rejects:
        reject_index.setdefault(rec["chunk_id"], []).append(rec)

    return chunks_index, reject_index


def compute_metrics(
    *,
    output_jsonl: Path,
    output_rejects: Path,
    metrics_path: Path | None,
    provider_name: str,
    requested_model: str,
    reported_model: str | None,
    dataset_sha256: str,
) -> ProviderMetrics:
    """Compute per-chunk and aggregate technical metrics."""
    records = _read_jsonl(output_jsonl)
    rejects = _read_jsonl(output_rejects)
    chunks_index, reject_index = _index_records(records, rejects)
    sink = _read_jsonl(metrics_path) if metrics_path else []

    per_chunk: list[ChunkMetrics] = []
    for chunk_id, status in chunks_index.items():
        if chunk_id.startswith("_"):
            continue
        chunk_rejects = reject_index.get(chunk_id, [])
        entity_rejects = [r for r in chunk_rejects if r.get("phase") == "entity"]
        relation_rejects = [r for r in chunk_rejects if r.get("phase") == "relation"]
        mentions = status.get("_mentions", []) if isinstance(status, dict) else []
        assertions = status.get("_assertions", []) if isinstance(status, dict) else []
        per_chunk.append(ChunkMetrics(
            chunk_id=chunk_id,
            entity_call_count=1,
            entity_accepted_mention_count=len(mentions),
            entity_reject_count=len(entity_rejects),
            entity_attempt_count=int(status.get("entity_attempt_count", 1) or 1)
                if isinstance(status, dict) else 1,
            entity_validation_ok=len(mentions) > 0 or len(entity_rejects) == 0,
            entity_reject_reasons=tuple(sorted({
                r.get("reason") for r in entity_rejects if r.get("reason")
            })),
            relation_call_count=1,
            relation_accepted_assertion_count=len(assertions),
            relation_reject_count=len(relation_rejects),
            relation_attempt_count=int(status.get("relation_attempt_count", 1) or 1)
                if isinstance(status, dict) else 1,
            relation_validation_ok=bool(status.get("extraction_complete", False))
                if isinstance(status, dict) else False,
            relation_reject_reasons=tuple(sorted({
                r.get("reason") for r in relation_rejects if r.get("reason")
            })),
            complete=bool(status.get("extraction_complete", False))
                if isinstance(status, dict) else False,
        ))

    total_mentions = sum(c.entity_accepted_mention_count for c in per_chunk)
    total_assertions = sum(c.relation_accepted_assertion_count for c in per_chunk)
    schema_errors = sum(
        sum(1 for r in reject_index.get(c.chunk_id, [])
            if r.get("phase") == "entity" and r.get("gate") == "schema")
        for c in per_chunk
    )
    evidence_errors = sum(
        sum(1 for r in reject_index.get(c.chunk_id, []) if r.get("gate") == "evidence")
        for c in per_chunk
    )
    ontology_errors = sum(
        sum(1 for r in reject_index.get(c.chunk_id, []) if r.get("gate") == "ontology")
        for c in per_chunk
    )
    alignment_errors = sum(
        sum(1 for r in reject_index.get(c.chunk_id, []) if r.get("gate") == "alignment")
        for c in per_chunk
    )
    candidates_rejected = sum(c.entity_reject_count + c.relation_reject_count for c in per_chunk)
    total_duration = sum(call.get("duration_seconds", 0.0) for call in sink)
    total_input_tokens = sum(call.get("input_tokens") or 0 for call in sink)
    total_output_tokens = sum(call.get("output_tokens") or 0 for call in sink)

    return ProviderMetrics(
        provider_name=provider_name,
        requested_model=requested_model,
        reported_model=reported_model,
        model_identity_verified=requested_model == reported_model,
        dataset_sha256=dataset_sha256,
        chunks_total=len(per_chunk),
        chunks_complete=sum(1 for c in per_chunk if c.complete),
        chunks_with_entity_ok=sum(1 for c in per_chunk if c.entity_validation_ok),
        chunks_with_relation_ok=sum(1 for c in per_chunk if c.relation_validation_ok),
        mentions_accepted=total_mentions,
        assertions_accepted=total_assertions,
        candidates_rejected=candidates_rejected,
        schema_errors=schema_errors,
        evidence_errors=evidence_errors,
        ontology_errors=ontology_errors,
        alignment_errors=alignment_errors,
        total_duration_seconds=total_duration,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        per_chunk=tuple(per_chunk),
    )


def metrics_to_dict(m: ProviderMetrics) -> dict:
    """Convert a ProviderMetrics to a JSON-safe dict."""
    return dataclasses.asdict(m)


def write_metrics_report(m: ProviderMetrics, *, output_path: Path) -> Path:
    payload = {
        "report_version": "benchmark-eval/v1",
        "metrics": metrics_to_dict(m),
        "notes": (
            "Technical-acceptance metrics derived from the canonical V2 gates "
            "(schema, alignment, evidence, ontology). Scientific correctness "
            "is NOT computed; see the blinded review package for the human "
            "review scaffold."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _sanitize_label(s: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", s.lower())


def write_blinded_review_package(
    *,
    runs: list[tuple[str, Path, Path, str, str]],
    chunks_path: Path,
    prompts_dir: Path,
    output_dir: Path,
    seed: int = 0xB11ADEED,
) -> tuple[Path, Path]:
    """Generate a blinded review package.

    Parameters
    ----------
    runs:
        List of ``(provider_label, output_jsonl, output_rejects,
        prompt_filename_entity, prompt_filename_relation)`` tuples.
    chunks_path:
        Path to the frozen chunks JSONL.
    prompts_dir:
        Path to the prompts directory.
    output_dir:
        Where to write the review package. Created if missing.
    seed:
        RNG seed; pin so the blinding is reproducible.
    """
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = _read_jsonl(chunks_path)
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    entity_prompt = (prompts_dir / "extract_entities_v2.txt").read_text(encoding="utf-8")
    relation_prompt = (prompts_dir / "extract_relations_v2.txt").read_text(encoding="utf-8")

    review_records: list[dict] = []
    review_key: list[dict] = []
    for provider_label, jsonl_path, rejects_path, _, _ in runs:
        records = _read_jsonl(jsonl_path)
        rejects = _read_jsonl(rejects_path)
        records_by_chunk: dict[str, list[dict]] = {}
        chunk_status_by_chunk: dict[str, dict] = {}
        for rec in records:
            kind = rec.get("kind")
            cid = rec.get("chunk_id")
            if not cid:
                continue
            if kind == "chunk_status":
                chunk_status_by_chunk[cid] = rec
            elif kind in {"mention", "assertion"}:
                records_by_chunk.setdefault(cid, []).append(rec)
        rejects_by_chunk: dict[str, list[dict]] = {}
        for rec in rejects:
            rejects_by_chunk.setdefault(rec["chunk_id"], []).append(rec)

        for chunk_id, chunk in chunks_by_id.items():
            label = "rev_" + _sanitize_label(provider_label) + "_" + hashlib.sha256(
                (provider_label + "|" + chunk_id + "|" + str(seed)).encode("utf-8")
            ).hexdigest()[:8]
            record = {
                "review_label": label,
                "chunk_id": chunk_id,
                "doc_id": chunk.get("doc_id"),
                "text": chunk.get("text"),
                "provider_label": provider_label,
                "mentions": records_by_chunk.get(chunk_id, []),
                "assertions": [
                    a for a in records_by_chunk.get(chunk_id, [])
                    if a.get("kind") == "assertion"
                ],
                "rejects": rejects_by_chunk.get(chunk_id, []),
                "chunk_status": chunk_status_by_chunk.get(chunk_id, {}),
                "prompts": {
                    "entity": entity_prompt,
                    "relation": relation_prompt,
                },
                "reviewer_prompt": (
                    "For each (mention, assertion) record above, decide "
                    "whether the evidence supports the claim. Do not "
                    "compare to a gold set; the reviewer's job is "
                    "plausibility, evidence faithfulness, and "
                    "schema/ontology compliance. Use the labels "
                    "'supported', 'unsupported', 'unclear', or "
                    "'schema_error'. The provider identity is hidden "
                    "behind the review_label; reveal it only after "
                    "all labels are recorded."
                ),
            }
            review_records.append(record)
            review_key.append({
                "review_label": label,
                "provider_label": provider_label,
                "chunk_id": chunk_id,
            })
    rng.shuffle(review_records)
    rng.shuffle(review_key)

    rec_path = output_dir / "review_records.jsonl"
    with rec_path.open("w", encoding="utf-8") as fh:
        for rec in review_records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True))
            fh.write("\n")
    key_path = output_dir / "review_key.json"
    key_payload = {
        "review_key_version": "benchmark-eval/v1",
        "seed": seed,
        "mapping": review_key,
    }
    key_path.write_text(
        json.dumps(key_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rec_path, key_path