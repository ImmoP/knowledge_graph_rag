"""Deterministic shadow diagnostics over verified benchmark recordings."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.benchmark.manifest_recording import MissingRawResponse, sha256_bytes


REPORT_FORMAT = "benchmark-offline-diagnostics-v1"
REPORT_FILENAME = "offline_diagnostics.json"


def _jsonl(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise MissingRawResponse(f"invalid {label} encoding") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MissingRawResponse(f"invalid {label} line {number}") from exc
        if not isinstance(row, dict):
            raise MissingRawResponse(f"invalid {label} line {number}")
        rows.append(row)
    return rows


def _entity_empty_payload(raw_response: bytes) -> bool:
    try:
        response = json.loads(raw_response)
        choices = response["choices"]
        content = choices[0]["message"]["content"]
        payload = json.loads(content.strip())
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError):
        return False
    return isinstance(payload, dict) and payload.get("mentions") == []


def _verified_raw_response(manifest: dict[str, Any], attempt: dict[str, Any]) -> bytes:
    number = attempt.get("attempt")
    expected_hash = attempt.get("response_sha256")
    expected_size = attempt.get("response_bytes")
    if type(number) is not int or number < 1:
        raise MissingRawResponse("invalid diagnostic attempt number")
    if not isinstance(expected_hash, str) or type(expected_size) is not int:
        raise MissingRawResponse(f"attempt {number} has no verifiable raw response")
    path = manifest["_directory"] / "raw_responses" / f"{number:06d}.bin"
    if path.is_symlink() or not path.is_file():
        raise MissingRawResponse(f"raw response {number} is missing or symlinked")
    raw = path.read_bytes()
    if len(raw) != expected_size or sha256_bytes(raw) != expected_hash:
        raise MissingRawResponse(f"raw response {number} hash mismatch")
    return raw


def _successful_empty_observations(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mentions_by_chunk: dict[str, int] = defaultdict(int)
    statuses: dict[str, dict[str, Any]] = {}
    for record in records:
        chunk_id = record.get("chunk_id")
        if not isinstance(chunk_id, str):
            continue
        if record.get("kind") == "mention":
            mentions_by_chunk[chunk_id] += 1
        elif record.get("kind") == "chunk_status":
            statuses[chunk_id] = record

    observations: list[dict[str, Any]] = []
    for attempt in attempts:
        scope = attempt.get("scope")
        if not isinstance(scope, dict):
            continue
        chunk_id = scope.get("chunk_id")
        prompt = scope.get("prompt_template_filename")
        if not isinstance(chunk_id, str) or prompt != "extract_entities_v2.txt":
            continue
        entity_status = statuses.get(chunk_id, {}).get("phases", {}).get("entity", {})
        successful_canonical_empty = (
            entity_status.get("status") == "validated"
            and entity_status.get("mention_count") == 0
            and entity_status.get("failure_count") == 0
            and mentions_by_chunk.get(chunk_id, 0) == 0
        )
        successful_attempt = (
            attempt.get("complete") is True
            and attempt.get("status") == 200
            and attempt.get("finish_reason") == "stop"
            and attempt.get("error_type") is None
        )
        if not successful_canonical_empty or not successful_attempt:
            continue
        raw_response = _verified_raw_response(manifest, attempt)
        if not _entity_empty_payload(raw_response):
            continue
        observations.append(
            {
                "attempt": attempt["attempt"],
                "chunk_id": chunk_id,
                "request_sha256": attempt["request_sha256"],
                "response_sha256": attempt["response_sha256"],
                "finish_reason": "stop",
                "classification": "technical_observation",
                "domain_conclusion": None,
            }
        )
    return sorted(
        observations,
        key=lambda item: (item["chunk_id"], item["attempt"]),
    )


def _span_relation(
    left: dict[str, Any], right: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], str] | None:
    left_start = left["evidence"]["char_start"]
    left_end = left["evidence"]["char_end"]
    right_start = right["evidence"]["char_start"]
    right_end = right["evidence"]["char_end"]
    if left_start == right_start and left_end == right_end:
        return None
    if left_start <= right_start and right_end <= left_end:
        return left, right, "contains"
    if right_start <= left_start and left_end <= right_end:
        return right, left, "contains"
    if max(left_start, right_start) < min(left_end, right_end):
        outer, inner = sorted(
            (left, right),
            key=lambda mention: (
                mention["evidence"]["char_start"],
                -mention["evidence"]["char_end"],
                mention["surface_form"],
            ),
        )
        return outer, inner, "partial_overlap"
    return None


def _overlap_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mentions_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("kind") == "mention":
            mentions_by_chunk[record["chunk_id"]].append(record)

    groups: list[dict[str, Any]] = []
    for chunk_id, mentions in sorted(mentions_by_chunk.items()):
        ordered = sorted(
            mentions,
            key=lambda item: (
                item["evidence"]["char_start"],
                item["evidence"]["char_end"],
                item["entity_type"],
                item["surface_form"],
                item["canonical_form"],
                item["mention_id"],
            ),
        )
        seen: set[tuple[Any, ...]] = set()
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                relation = _span_relation(left, right)
                if relation is None:
                    continue
                outer, inner, kind = relation
                key = (
                    outer["mention_id"],
                    inner["mention_id"],
                    outer["surface_form"],
                    inner["surface_form"],
                    kind,
                )
                if key in seen:
                    continue
                seen.add(key)
                groups.append(
                    {
                        "chunk_id": chunk_id,
                        "outer": {
                            "mention_id": outer["mention_id"],
                            "entity_type": outer["entity_type"],
                            "surface_form": outer["surface_form"],
                        },
                        "outer_span": [
                            outer["evidence"]["char_start"],
                            outer["evidence"]["char_end"],
                        ],
                        "inner": {
                            "mention_id": inner["mention_id"],
                            "entity_type": inner["entity_type"],
                            "surface_form": inner["surface_form"],
                        },
                        "inner_span": [
                            inner["evidence"]["char_start"],
                            inner["evidence"]["char_end"],
                        ],
                        "relation": kind,
                        "classification": "technical_observation",
                        "domain_conclusion": None,
                    }
                )
    return sorted(
        groups,
        key=lambda item: (
            item["chunk_id"],
            item["outer_span"],
            item["inner_span"],
            item["outer"]["surface_form"],
            item["inner"]["surface_form"],
        ),
    )


def _canonical_conflicts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("kind") != "mention":
            continue
        evidence = record["evidence"]
        key = (
            record["chunk_id"],
            record["entity_type"],
            record["surface_form"],
            evidence["char_start"],
            evidence["char_end"],
        )
        observations[key].append(record)

    conflicts: list[dict[str, Any]] = []
    for key, mentions in sorted(observations.items()):
        canonical_forms = sorted({mention["canonical_form"] for mention in mentions})
        if len(canonical_forms) < 2:
            continue
        chunk_id, entity_type, surface_form, start, end = key
        conflicts.append(
            {
                "chunk_id": chunk_id,
                "entity_type": entity_type,
                "surface_form": surface_form,
                "span": [start, end],
                "canonical_forms": canonical_forms,
                "mention_ids": sorted({mention["mention_id"] for mention in mentions}),
                "classification": "technical_anomaly",
                "domain_conclusion": None,
            }
        )
    return conflicts


def _file_identity(entry: dict[str, Any]) -> dict[str, Any]:
    """Project only verified, non-sensitive file identity fields.

    The allowlist is intentional: any new field the recorder starts
    writing must be evaluated here before it can leak into a shadow
    report. ``source_path``/``absolute_path`` are deliberately rejected
    to prevent host-side paths from propagating alongside SHA-256
    identities.
    """
    return {name: entry[name] for name in ("path", "sha256", "size") if name in entry}

def build_offline_diagnostics(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build a side-effect-free report from artifacts verified by replay loading."""
    artifacts = manifest.get("_verified_artifacts")
    attempts_raw = manifest.get("_verified_attempts")
    if not isinstance(artifacts, dict) or not isinstance(attempts_raw, bytes):
        raise MissingRawResponse("offline diagnostics require a verified recording")
    extraction_raw = artifacts.get("extractions")
    if not isinstance(extraction_raw, bytes):
        raise MissingRawResponse("offline diagnostics require canonical extractions")
    records = _jsonl(extraction_raw, label="canonical extractions")
    attempts = _jsonl(attempts_raw, label="attempt index")

    return {
        "format": REPORT_FORMAT,
        "mode": "shadow",
        "provider_calls": 0,
        "retries_enabled": False,
        "automatic_gold_labels": False,
        "provenance": {
            "recording_format": manifest["format"],
            "recording_manifest_sha256": manifest["_manifest_sha256"],
            "inputs": {
                name: _file_identity(manifest["inputs"][name])
                for name in sorted(manifest["inputs"])
            },
            "canonical_artifacts": {
                name: _file_identity(manifest["artifacts"][name])
                for name in sorted(manifest["artifacts"])
            },
            "raw_attempts": {
                "attempt_count": manifest["raw_responses"]["attempt_count"],
                "attempts_index": _file_identity(
                    manifest["raw_responses"]["attempts_index"]
                ),
            },
        },
        "successful_empty_observations": _successful_empty_observations(
            manifest, records, attempts
        ),
        "overlap_groups": _overlap_groups(records),
        "canonical_conflicts": _canonical_conflicts(records),
    }
