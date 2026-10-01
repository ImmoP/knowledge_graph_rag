#!/usr/bin/env python3
"""Diagnose recorded V2 entity extraction variation without provider calls."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


Mention = tuple[str, str, str, int, int, str]


def _mention(raw: list[Any]) -> Mention:
    if len(raw) != 6:
        raise ValueError("mention fixture must contain exactly six fields")
    entity_type, surface, canonical, start, end, confidence = raw
    if not all(isinstance(value, str) for value in (entity_type, surface, canonical, confidence)):
        raise TypeError("mention text fields must be strings")
    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        raise ValueError("mention span must be a valid half-open range")
    return entity_type, surface, canonical, start, end, confidence


def _span_relation(left: Mention, right: Mention) -> tuple[Mention, Mention, str] | None:
    left_start, left_end = left[3], left[4]
    right_start, right_end = right[3], right[4]
    if left_start == right_start and left_end == right_end:
        return None
    if left_start <= right_start and right_end <= left_end:
        return left, right, "contains"
    if right_start <= left_start and left_end <= right_end:
        return right, left, "contains"
    if max(left_start, right_start) < min(left_end, right_end):
        outer, inner = sorted((left, right), key=lambda m: (m[3], -m[4], m[1]))
        return outer, inner, "partial_overlap"
    return None


def _diagnose_overlaps(recordings: list[dict]) -> list[dict]:
    observed: dict[tuple, set[str]] = defaultdict(set)
    for recording in recordings:
        mentions = sorted((_mention(raw) for raw in recording["mentions"]), key=lambda m: (m[3], m[4], m[0], m[1], m[2], m[5]))
        for index, left in enumerate(mentions):
            for right in mentions[index + 1 :]:
                relation = _span_relation(left, right)
                if relation is None:
                    continue
                outer, inner, kind = relation
                key = (
                    outer[0], outer[1], outer[3], outer[4],
                    inner[0], inner[1], inner[3], inner[4], kind,
                )
                observed[key].add(recording["recording_id"])

    groups = []
    for key, recording_ids in observed.items():
        (
            outer_type, outer_surface, outer_start, outer_end,
            inner_type, inner_surface, inner_start, inner_end, relation,
        ) = key
        groups.append({
            "outer": {"entity_type": outer_type, "surface_form": outer_surface},
            "outer_span": [outer_start, outer_end],
            "inner": {"entity_type": inner_type, "surface_form": inner_surface},
            "inner_span": [inner_start, inner_end],
            "relation": relation,
            "recording_ids": sorted(recording_ids),
            "classification": "technical_observation",
            "domain_conclusion": None,
        })
    return sorted(groups, key=lambda item: (item["outer_span"], item["inner_span"], item["outer"]["surface_form"], item["inner"]["surface_form"]))


def _diagnose_canonical_conflicts(recordings: list[dict]) -> list[dict]:
    by_identity: dict[tuple[str, str, int, int], list[tuple[str, str]]] = defaultdict(list)
    for recording in recordings:
        for raw in recording["mentions"]:
            entity_type, surface, canonical, start, end, _confidence = _mention(raw)
            by_identity[(entity_type, surface, start, end)].append((recording["recording_id"], canonical))

    conflicts = []
    for (entity_type, surface, start, end), observations in sorted(by_identity.items()):
        canonical_forms = sorted({canonical for _, canonical in observations})
        if len(canonical_forms) < 2:
            continue
        conflicts.append({
            "entity_type": entity_type,
            "surface_form": surface,
            "span": [start, end],
            "canonical_forms": canonical_forms,
            "observations": [
                {"recording_id": recording_id, "canonical_form": canonical}
                for recording_id, canonical in sorted(observations)
            ],
            "classification": "technical_anomaly",
            "domain_conclusion": None,
        })
    return conflicts


def _diagnose_successful_empty(recordings: list[dict]) -> list[dict]:
    observations = []
    for recording in sorted(recordings, key=lambda item: item["recording_id"]):
        is_successful_empty = (
            recording.get("http_status") == 200
            and recording.get("finish_reason") == "stop"
            and recording.get("parsed_ok") is True
            and recording.get("reject_count") == 0
            and recording.get("mentions") == []
        )
        if not is_successful_empty:
            continue
        observations.append({
            "recording_id": recording["recording_id"],
            "chunk_id": recording["chunk_id"],
            "request_identity": recording["request_identity"],
            "attempt_count": recording["attempt_count"],
            "request_sha256": recording.get("request_sha256"),
            "classification": "technical_observation",
            "domain_conclusion": None,
        })
    return observations


def diagnose(fixture: dict) -> dict:
    """Return order-independent technical diagnostics from a frozen fixture."""
    table_recordings = fixture["table_recordings"]
    if len(table_recordings) != 5:
        raise ValueError("fixture must contain exactly five table recordings")
    return {
        "format": "table-extraction-offline-diagnostics-v1",
        "table_chunk": fixture["table_chunk"],
        "recording_ids": sorted(item["recording_id"] for item in table_recordings),
        "successful_empty_observations": _diagnose_successful_empty(
            fixture["successful_empty_recordings"]
        ),
        "overlap_groups": _diagnose_overlaps(table_recordings),
        "canonical_conflicts": _diagnose_canonical_conflicts(table_recordings),
        "domain_questions": sorted(
            fixture["domain_questions"], key=lambda item: item["question_id"]
        ),
        "automatic_gold_labels": False,
    }




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    report = diagnose(fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
