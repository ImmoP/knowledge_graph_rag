"""Offline-only whitespace arm; never changes recorded bytes or canonical output."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from src.benchmark.manifest_recording import RecordingError, private_write, sha256_bytes
from src.extract.experimental_quote_repair import REPAIR_VERSION, repair_quote
from src.providers.cache import current_extraction_cache_scope


class WhitespaceComparisonProvider:
    """Repair only relation quotes after the unchanged Manifest parser.

    Entity payloads and occurrence indices remain untouched. Relation output
    cannot affect later request construction, so both arms consume exactly the
    same recorded requests/responses even when a repaired quote passes V2.
    """

    def __init__(self, inner, chunks: list[dict]) -> None:
        self._inner = inner
        self._texts = {chunk["chunk_id"]: chunk["text"] for chunk in chunks}
        self.name = inner.name
        self.model = inner.model
        self.decisions: list[dict] = []

    def generate_json(self, system, user, schema, *, max_tokens=1024, temperature=0.0):
        original, usage = self._inner.generate_json(
            system, user, schema, max_tokens=max_tokens, temperature=temperature
        )
        if "assertions" not in schema.get("properties", {}):
            return original, usage
        scope = current_extraction_cache_scope() or {}
        chunk_id = scope["chunk_id"]
        payload = deepcopy(original)
        for index, candidate in enumerate(payload.get("assertions", [])):
            quote = candidate.get("evidence_quote")
            result = repair_quote(self._texts[chunk_id], quote)
            self.decisions.append({
                "chunk_id": chunk_id,
                "candidate_index": index,
                "candidate_identity": {
                    "relation_type": candidate.get("relation_type"),
                    "subject_ref": deepcopy(candidate.get("subject_ref")),
                    "object_ref": deepcopy(candidate.get("object_ref")),
                },
                "original_quote": quote,
                "status": result.status,
                "replacement_quote": result.quote if result.status == "repaired" else None,
                "char_start": result.char_start,
                "char_end": result.char_end,
            })
            if result.status == "repaired":
                candidate["evidence_quote"] = result.quote
        return payload, usage


def _counts(records: list[dict], rejects: list[dict]) -> dict:
    kinds = Counter(record.get("kind") for record in records)
    return {
        "accepted_mentions": kinds["mention"],
        "accepted_assertions": kinds["assertion"],
        "rejects": len(rejects),
        "complete_chunks": sum(
            record.get("kind") == "chunk_status" and record.get("extraction_complete") is True
            for record in records
        ),
    }


def compare_whitespace(*, recording_dir: Path, output_dir: Path) -> dict:
    """Create a separate relation-quote arm after exact canonical replay."""
    from src.benchmark.manifest_replay import load_recording, run_replay_pipeline

    manifest = load_recording(recording_dir)
    source = manifest["_directory"]
    output = Path(output_dir)
    baseline = {}
    for name in ("extractions.jsonl", "extractions.rejects.jsonl"):
        recorded = (source / name).read_bytes()
        if (output / name).read_bytes() != recorded:
            raise RecordingError("whitespace comparison requires an exact canonical replay")
        baseline[name] = recorded
    arm = output / "whitespace_comparison"
    result, client, provider = run_replay_pipeline(
        source, arm,
        provider_wrapper=lambda inner: WhitespaceComparisonProvider(inner, manifest["_chunks"]),
    )
    if client.attempt_count != manifest["raw_responses"]["attempt_count"]:
        raise RecordingError("whitespace arm consumed a different HTTP attempt stream")

    def rows(raw: bytes) -> list[dict]:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    baseline_counts = _counts(rows(baseline["extractions.jsonl"]), rows(baseline["extractions.rejects.jsonl"]))
    experimental_bytes = (arm / "extractions.jsonl").read_bytes()
    experimental_rejects = (arm / "extractions.rejects.jsonl").read_bytes()
    experiment_counts = _counts(rows(experimental_bytes), rows(experimental_rejects))
    report = {
        "format": "manifest-whitespace-comparison-v1",
        "canonical_v2": False,
        "repair_version": REPAIR_VERSION,
        "scope": "relation evidence_quote only; entities and occurrence indices unchanged",
        "recording_manifest_sha256": manifest["_manifest_sha256"],
        "identical_http_attempt_stream": True,
        "attempt_count": client.attempt_count,
        "budget_aborted": result.budget_aborted,
        "baseline": baseline_counts,
        "experiment": experiment_counts,
        "delta": {key: experiment_counts[key] - value for key, value in baseline_counts.items()},
        "decisions": provider.decisions,
        "sha256": {
            "baseline_extractions": sha256_bytes(baseline["extractions.jsonl"]),
            "baseline_rejects": sha256_bytes(baseline["extractions.rejects.jsonl"]),
            "experimental_extractions": sha256_bytes(experimental_bytes),
            "experimental_rejects": sha256_bytes(experimental_rejects),
        },
        "scientific_conclusion": None,
        "note": "Technical acceptance after whitespace repair is not a semantic quality or gold-label judgment.",
    }
    private_write(arm / "comparison.json", (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return report
