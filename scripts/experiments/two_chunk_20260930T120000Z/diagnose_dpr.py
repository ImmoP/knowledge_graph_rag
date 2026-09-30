#!/usr/bin/env python3
"""Render an offline, no-remapping DPR diagnostic from a verified recording."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.manifest_recording import ReplayClient, assert_safe_bytes, external_directory, private_write, sha256_bytes
from src.benchmark.manifest_replay import (
    EXTRACTIONS_FILENAME,
    REJECTS_FILENAME,
    load_recording,
)
from src.benchmark.two_chunk_experiment import DPR_CHUNK_ID, EXPERIMENT_CHUNK_IDS, verify_experiment_recording
from src.extract.experimental_quote_repair import REPAIR_VERSION


def _rows(raw: bytes, label: str) -> list[dict]:
    try:
        result = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if not all(isinstance(row, dict) for row in result):
        raise ValueError(f"invalid {label} row")
    return result


def _verified_arm(path: Path, expected: dict[str, str] | None, label: str) -> tuple[list[dict], list[dict], dict[str, str]]:
    path = external_directory(path, create=False)
    contents: dict[str, bytes] = {}
    for filename in (EXTRACTIONS_FILENAME, REJECTS_FILENAME):
        candidate = path / filename
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"{label} artifact missing: {candidate}")
        raw = candidate.read_bytes()
        assert_safe_bytes(raw)
        contents[filename] = raw
    hashes = {name: sha256_bytes(raw) for name, raw in contents.items()}
    if expected is not None:
        for filename, actual in hashes.items():
            expected_hash = expected.get(filename)
            if expected_hash != actual:
                raise ValueError(f"{label} artifact checksum mismatch: {filename}")
    return _rows(contents[EXTRACTIONS_FILENAME], f"{label} extractions"), _rows(contents[REJECTS_FILENAME], f"{label} rejects"), hashes


def _occurrences(text: str, literal: str) -> list[list[int]]:
    if not literal:
        return []
    starts: list[list[int]] = []
    at = 0
    while True:
        at = text.find(literal, at)
        if at < 0:
            return starts
        starts.append([at, at + len(literal)])
        at += 1


def _mentions(records: list[dict], chunk_id: str, text: str) -> list[dict]:
    result = []
    for row in records:
        if row.get("kind") != "mention" or row.get("chunk_id") != chunk_id:
            continue
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        start, end = evidence.get("char_start"), evidence.get("char_end")
        valid = isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(text)
        literal = text[start:end] if valid else None
        surface = row.get("surface_form")
        result.append({
            "mention_id": row.get("mention_id"), "surface_form": surface,
            "evidence_span": [start, end] if valid else None,
            "evidence_literal_in_text": literal,
            "literal_matches_surface_form": literal == surface if literal is not None else None,
            "surface_occurrences": _occurrences(text, surface) if isinstance(surface, str) else [],
        })
    return result


def _relations(records: list[dict], chunk_id: str, mentions: list[dict], text: str) -> list[dict]:
    index = {row.get("mention_id"): row for row in mentions}
    result = []
    for row in records:
        if row.get("kind") != "assertion" or row.get("chunk_id") != chunk_id:
            continue
        refs = {key: (row.get(key) if isinstance(row.get(key), dict) else {}) for key in ("subject_ref", "object_ref")}
        quote = row.get("evidence_quote")
        occurrences = _occurrences(text, quote) if isinstance(quote, str) else []
        result.append({
            "assertion_id": row.get("assertion_id"), "relation_type": row.get("relation_type"),
            "evidence_quote": quote, "evidence_quote_occurrences": occurrences,
            "referenced_mentions": {
                key: {"id": ref.get("id"), "span": index.get(ref.get("id"), {}).get("evidence_span"),
                      "inside_quote_occurrence": any(span[0] >= occurrence[0] and span[1] <= occurrence[1] for occurrence in occurrences)
                      if (span := index.get(ref.get("id"), {}).get("evidence_span")) else None}
                for key, ref in refs.items()
            },
        })
    return result


def _decision_references(decision: dict, mentions: list[dict], text: str) -> dict:
    """Describe raw comparison references against frozen mention evidence only."""
    identity = decision.get("candidate_identity")
    identity = identity if isinstance(identity, dict) else {}
    index = {mention.get("mention_id"): mention for mention in mentions}
    original = decision.get("original_quote")
    repaired = decision.get("replacement_quote")
    def refs(quote: object) -> dict:
        occurrences = _occurrences(text, quote) if isinstance(quote, str) else []
        return {key: {"id": ref.get("id"), "span": index.get(ref.get("id"), {}).get("evidence_span"),
                      "surface_occurrences": index.get(ref.get("id"), {}).get("surface_occurrences", []),
                      "inside_quote_occurrence": any(span[0] >= occurrence[0] and span[1] <= occurrence[1] for occurrence in occurrences)
                      if isinstance(quote, str) and (span := index.get(ref.get("id"), {}).get("evidence_span")) else None}
                for key, ref in ((key, identity.get(key) if isinstance(identity.get(key), dict) else {})
                                 for key in ("subject_ref", "object_ref"))}
    return {"candidate_index": decision.get("candidate_index"), "status": decision.get("status"),
            "original_quote": original, "original_quote_occurrences": _occurrences(text, original) if isinstance(original, str) else [],
            "original_references": refs(original), "replacement_quote": repaired,
            "replacement_quote_occurrences": _occurrences(text, repaired) if isinstance(repaired, str) else [],
            "replacement_references": refs(repaired)}


def _rejected(rows: list[dict], chunk_id: str) -> list[dict]:
    """Retain rejected candidate references verbatim; never infer a replacement."""
    result = []
    for row in rows:
        if row.get("chunk_id") != chunk_id:
            continue
        offending = row.get("offending_record")
        candidate = offending.get("offending_value", offending) if isinstance(offending, dict) else {}
        candidate = candidate if isinstance(candidate, dict) else {}
        result.append({
            "phase": row.get("phase"), "gate": row.get("gate"), "reason": row.get("reason"),
            "aligner_reason": row.get("aligner_reason"), "offending_record": offending,
            "evidence_quote": candidate.get("evidence_quote"),
            "candidate_references": {key: candidate.get(key) for key in ("subject_ref", "object_ref")},
        })
    return result


def _render_markdown(report: dict) -> str:
    lines = [
        "# DPR diagnostic report", "",
        f"Source recording status: {report['source_recording']['status']}.",
        "No entity IDs are remapped by this report.", "",
    ]
    for arm, payload in report["arms"].items():
        lines.extend([f"## {arm}", ""])
        for chunk_id, chunk in payload["chunks"].items():
            lines.extend([f"### {chunk_id}", "", "#### Mentions", ""])
            for mention in chunk["mentions"]:
                lines.append(f"- `{mention['mention_id']}` span={mention['evidence_span']} literal={mention['evidence_literal_in_text']!r} alternatives={mention['surface_occurrences']}")
            lines.extend(["", "#### Relations", ""])
            for relation in chunk["relations"]:
                lines.append(f"- `{relation['assertion_id']}` quote={relation['evidence_quote']!r} occurrences={relation['evidence_quote_occurrences']} references={relation['referenced_mentions']}")
            lines.extend(["", "#### Rejected candidates", ""])
            for reject in chunk["rejected_candidates"]:
                lines.append(f"- `{reject['phase']}/{reject['reason'] or reject['aligner_reason']}` quote={reject['evidence_quote']!r} refs={reject['candidate_references']}")
            lines.extend(["", "#### Relation evidence and referenced mention spans", ""])
            for candidate in chunk["comparison_candidate_references"]:
                lines.append(f"- Candidate {candidate['candidate_index']}: {candidate['status']}")
                lines.append(f"  - Original quote: {candidate['original_quote']!r}; occurrences={candidate['original_quote_occurrences']}")
                lines.append(f"  - Original references: {candidate['original_references']}")
                lines.append(f"  - Whitespace replacement: {candidate['replacement_quote']!r}; occurrences={candidate['replacement_quote_occurrences']}")
                lines.append(f"  - Unchanged references: {candidate['replacement_references']}")
            lines.append("")
    return "\n".join(lines)


def write_dpr_diagnosis(*, recording_dir: Path, replay_dir: Path, output_dir: Path) -> dict[str, Path]:
    manifest = load_recording(recording_dir)
    verify_experiment_recording(manifest)
    ReplayClient(manifest["_directory"]).close()
    replay_dir = external_directory(Path(replay_dir), create=False)
    output_dir = Path(output_dir).expanduser().absolute()
    source = manifest["_directory"]
    if replay_dir == source or source in replay_dir.parents or replay_dir in source.parents:
        raise ValueError("replay output overlaps recording")
    resolved_output = output_dir.resolve(strict=False)
    if resolved_output == source or source in resolved_output.parents or resolved_output in source.parents:
        raise ValueError("diagnostic output overlaps recording")
    external_directory(replay_dir / "whitespace_comparison", create=False)
    source_artifacts = manifest["_verified_artifacts"]
    canonical, canonical_rejects, canonical_hashes = _verified_arm(
        replay_dir,
        {EXTRACTIONS_FILENAME: sha256_bytes(source_artifacts["extractions"]), REJECTS_FILENAME: sha256_bytes(source_artifacts["rejects"])},
        "canonical replay",
    )
    comparison_path = replay_dir / "whitespace_comparison" / "comparison.json"
    if comparison_path.is_symlink() or not comparison_path.is_file():
        raise ValueError("whitespace comparison report missing")
    comparison_raw = comparison_path.read_bytes()
    assert_safe_bytes(comparison_raw)
    try:
        comparison = json.loads(comparison_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid whitespace comparison report") from exc
    hashes = comparison.get("sha256") if isinstance(comparison, dict) else None
    if (not isinstance(hashes, dict) or comparison.get("format") != "manifest-whitespace-comparison-v1"
            or comparison.get("repair_version") != REPAIR_VERSION
            or comparison.get("recording_manifest_sha256") != manifest["_manifest_sha256"]):
        raise ValueError("whitespace comparison is not bound to recording")
    if hashes.get("baseline_extractions") != sha256_bytes(source_artifacts["extractions"]) or hashes.get("baseline_rejects") != sha256_bytes(source_artifacts["rejects"]):
        raise ValueError("whitespace comparison baseline checksum mismatch")
    experimental, experimental_rejects, experimental_hashes = _verified_arm(
        replay_dir / "whitespace_comparison",
        {EXTRACTIONS_FILENAME: hashes.get("experimental_extractions"), REJECTS_FILENAME: hashes.get("experimental_rejects")},
        "whitespace arm",
    )
    decisions = comparison.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("invalid whitespace comparison decisions")
    replay_report_path = replay_dir / "replay_report.json"
    if replay_report_path.is_symlink() or not replay_report_path.is_file():
        raise ValueError("canonical replay report missing")
    replay_report_raw = replay_report_path.read_bytes()
    assert_safe_bytes(replay_report_raw)
    replay_report = json.loads(replay_report_raw)
    if (replay_report.get("status") != "matched"
            or replay_report.get("recording_manifest_sha256") != manifest["_manifest_sha256"]
            or replay_report.get("whitespace_comparison") != comparison):
        raise ValueError("comparison decisions differ from verified replay report")
    texts = {row["chunk_id"]: row.get("text", "") for row in manifest["_chunks"]}
    def arm_payload(records: list[dict], rejects: list[dict], artifact_hashes: dict[str, str]) -> dict:
        chunks: dict[str, dict] = {}
        for chunk_id in EXPERIMENT_CHUNK_IDS:
            mentions = _mentions(records, chunk_id, texts[chunk_id])
            chunks[chunk_id] = {
                "chunk_status": next((row for row in records if row.get("kind") == "chunk_status" and row.get("chunk_id") == chunk_id), None),
                "mentions": mentions,
                "relations": _relations(records, chunk_id, mentions, texts[chunk_id]),
                "rejected_candidates": _rejected(rejects, chunk_id),
                "comparison_candidate_references": [
                    _decision_references(decision, mentions, texts[chunk_id])
                    for decision in decisions if isinstance(decision, dict) and decision.get("chunk_id") == chunk_id
                ],
            }
        return {"sha256": artifact_hashes, "chunks": chunks}
    report = {"format": "manifest-two-chunk-dpr-diagnostic-v2", "recording_manifest_sha256": manifest["_manifest_sha256"],
              "focus_chunk_id": DPR_CHUNK_ID, "no_entity_remap": True,
              "source_recording": {"status": manifest["status"], "complete": manifest["status"] == "complete"},
              "arms": {"canonical_v2": arm_payload(canonical, canonical_rejects, canonical_hashes),
                       "experimental_whitespace_v1": arm_payload(experimental, experimental_rejects, experimental_hashes)}}
    output = external_directory(output_dir, create=True)
    json_path, markdown_path = output / "dpr_diagnostic.json", output / "dpr_diagnostic.md"
    json_bytes = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    markdown_bytes = _render_markdown(report).encode()
    assert_safe_bytes(json_bytes)
    assert_safe_bytes(markdown_bytes)
    private_write(json_path, json_bytes)
    private_write(markdown_path, markdown_bytes)
    return {"json": json_path, "markdown": markdown_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", required=True, type=Path)
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        paths = write_dpr_diagnosis(recording_dir=args.recording_dir, replay_dir=args.replay_dir,
                                    output_dir=args.output_dir or args.replay_dir)
    except Exception:
        print("error: verified DPR diagnosis failed", file=sys.stderr)
        return 1
    print(f"DPR diagnostic JSON: {paths['json']}")
    print(f"DPR diagnostic Markdown: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
