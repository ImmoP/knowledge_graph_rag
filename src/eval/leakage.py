"""Graph-independent held-out leakage freeze for the v2 gold-set sampler (#61)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from src.eval.author import validate_draft

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRAFT_PATH = ROOT / "eval" / "questions.draft.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "eval" / "held_out_leakage.v1.json"
DEFAULT_CHUNKS_PATH = ROOT / "data" / "derived" / "chunks.jsonl"
DEFAULT_MANIFEST_PATH = ROOT / "data" / "manifest.jsonl"
DEFAULT_ONTOLOGY_PATH = ROOT / "ontology.yaml"

SCHEMA_VERSION = "held-out-leakage.v1"
EXPECTED_RECORD_COUNT = 45

_QID_RE = re.compile(r"^q_[0-9]{3}$")
_CHUNK_ID_RE = re.compile(r"^c_[0-9a-f]{16}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "record_count",
        "all_split_held_out",
        "source_draft_sha256",
        "chunks_sha256",
        "manifest_sha256",
        "ontology_sha256",
        "excluded_chunk_ids",
        "question_chunk_map",
    }
)


class LeakageError(RuntimeError):
    """Raised when leakage freeze or validation fails closed."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise LeakageError(f"cannot read {label} {path}: {exc}") from exc


def _parse_jsonl_bytes(data: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LeakageError(f"{label} is not valid UTF-8: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise LeakageError(f"{label} line {line_number}: blank JSONL line")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LeakageError(
                f"{label} line {line_number}: malformed JSON: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise LeakageError(f"{label} line {line_number}: record must be an object")
        records.append(record)
    return records


def _chunk_ids_from_bytes(data: bytes) -> set[str]:
    records = _parse_jsonl_bytes(data, "chunks")
    chunk_ids: set[str] = set()
    for index, record in enumerate(records, 1):
        value = record.get("chunk_id")
        if not isinstance(value, str) or not _CHUNK_ID_RE.fullmatch(value):
            raise LeakageError(f"chunks record {index}: invalid chunk_id {value!r}")
        if value in chunk_ids:
            raise LeakageError(f"chunks record {index}: duplicate chunk_id {value}")
        chunk_ids.add(value)
    return chunk_ids


def _question_chunk_map(records: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for record in records:
        qid = record["qid"]
        chunks = set(record["supporting_chunk_ids"])
        for step in record["evidence_path"]:
            for endpoint_name in ("from", "to"):
                endpoint = step[endpoint_name]
                if isinstance(endpoint, str) and _CHUNK_ID_RE.fullmatch(endpoint):
                    chunks.add(endpoint)
        mapping[qid] = sorted(chunks)
    return {qid: mapping[qid] for qid in sorted(mapping)}


def _payload(
    draft_bytes: bytes,
    chunks_bytes: bytes,
    manifest_bytes: bytes,
    ontology_bytes: bytes,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    question_map = _question_chunk_map(records)
    excluded = sorted({chunk_id for ids in question_map.values() for chunk_id in ids})
    return {
        "all_split_held_out": all(record.get("split") == "held_out" for record in records),
        "chunks_sha256": _sha256_bytes(chunks_bytes),
        "excluded_chunk_ids": excluded,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "ontology_sha256": _sha256_bytes(ontology_bytes),
        "question_chunk_map": question_map,
        "record_count": len(records),
        "schema_version": SCHEMA_VERSION,
        "source_draft_sha256": _sha256_bytes(draft_bytes),
    }


def _serialize(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    temporary_name: str | None = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _publish(output_path: Path, candidate: bytes) -> str:
    sha256 = _sha256_bytes(candidate)
    if output_path.exists():
        existing = _read_bytes(output_path, "published leakage artifact")
        if existing != candidate:
            raise LeakageError(
                f"published leakage artifact is frozen; different bytes are refused: {output_path}"
            )
        return sha256
    try:
        _atomic_write(output_path, candidate)
    except OSError as exc:
        raise LeakageError(f"cannot publish leakage artifact {output_path}: {exc}") from exc
    return sha256


def freeze_leakage(
    draft_path: Path,
    output_path: Path,
    *,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
) -> str:
    """Validate a human draft without Neo4j and freeze only leakage metadata."""
    initial = {
        "draft": _read_bytes(draft_path, "draft"),
        "chunks": _read_bytes(chunks_path, "chunks"),
        "manifest": _read_bytes(manifest_path, "manifest"),
        "ontology": _read_bytes(ontology_path, "ontology"),
    }
    report = validate_draft(
        draft_path,
        manifest_path=manifest_path,
        chunks_path=chunks_path,
        ontology_path=ontology_path,
        require_dataset=True,
    )
    if not report.ok:
        raise LeakageError("\n".join(str(violation) for violation in report.violations))

    current = {
        "draft": _read_bytes(draft_path, "draft"),
        "chunks": _read_bytes(chunks_path, "chunks"),
        "manifest": _read_bytes(manifest_path, "manifest"),
        "ontology": _read_bytes(ontology_path, "ontology"),
    }
    changed = [name for name in initial if initial[name] != current[name]]
    if changed:
        raise LeakageError(
            "input changed during leakage validation: " + ", ".join(sorted(changed))
        )

    records = _parse_jsonl_bytes(initial["draft"], "draft")
    payload = _payload(
        initial["draft"],
        initial["chunks"],
        initial["manifest"],
        initial["ontology"],
        records,
    )
    missing = sorted(
        set(payload["excluded_chunk_ids"]) - _chunk_ids_from_bytes(initial["chunks"])
    )
    if missing:
        raise LeakageError(
            f"draft evidence/support references unknown chunks: {missing[:5]}"
        )
    candidate = _serialize(payload)
    return _publish(output_path, candidate)


def _require_sha256(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise LeakageError(f"leakage artifact: {key} must be a lowercase SHA256")
    return value


def load_frozen_exclusions(
    leakage_path: Path,
    *,
    chunks_path: Path,
    manifest_path: Path,
    ontology_path: Path,
) -> set[str]:
    """Validate a frozen leakage artifact against current canonical identities."""
    raw = _read_bytes(leakage_path, "leakage artifact")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LeakageError(f"malformed leakage artifact {leakage_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LeakageError("leakage artifact must be a JSON object")
    if set(payload) != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - set(payload))
        extra = sorted(set(payload) - _REQUIRED_KEYS)
        raise LeakageError(
            f"leakage artifact keys mismatch; missing={missing}, extra={extra}"
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise LeakageError(
            f"leakage artifact schema_version must be {SCHEMA_VERSION!r}"
        )
    if payload.get("record_count") != EXPECTED_RECORD_COUNT:
        raise LeakageError(
            f"leakage artifact record_count must be {EXPECTED_RECORD_COUNT}"
        )
    if payload.get("all_split_held_out") is not True:
        raise LeakageError("leakage artifact all_split_held_out must be true")

    _require_sha256(payload, "source_draft_sha256")
    expected_identities = {
        "chunks_sha256": _sha256_bytes(_read_bytes(chunks_path, "chunks")),
        "manifest_sha256": _sha256_bytes(_read_bytes(manifest_path, "manifest")),
        "ontology_sha256": _sha256_bytes(_read_bytes(ontology_path, "ontology")),
    }
    for key, expected in expected_identities.items():
        actual = _require_sha256(payload, key)
        if actual != expected:
            raise LeakageError(
                f"stale leakage artifact: {key}={actual}, current={expected}"
            )

    question_map = payload.get("question_chunk_map")
    if not isinstance(question_map, dict) or len(question_map) != EXPECTED_RECORD_COUNT:
        raise LeakageError(
            f"leakage artifact question_chunk_map must contain {EXPECTED_RECORD_COUNT} qids"
        )
    union: set[str] = set()
    for qid in sorted(question_map):
        if not isinstance(qid, str) or not _QID_RE.fullmatch(qid):
            raise LeakageError(f"leakage artifact has invalid qid {qid!r}")
        values = question_map[qid]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not _CHUNK_ID_RE.fullmatch(value) for value in values)
        ):
            raise LeakageError(
                f"leakage artifact {qid}: chunk IDs must be a non-empty canonical list"
            )
        if values != sorted(set(values)):
            raise LeakageError(
                f"leakage artifact {qid}: chunk IDs must be sorted and unique"
            )
        union.update(values)

    excluded = payload.get("excluded_chunk_ids")
    if (
        not isinstance(excluded, list)
        or any(not isinstance(value, str) or not _CHUNK_ID_RE.fullmatch(value) for value in excluded)
    ):
        raise LeakageError(
            "leakage artifact excluded_chunk_ids must be a canonical chunk-ID list"
        )
    if excluded != sorted(set(excluded)):
        raise LeakageError(
            "leakage artifact excluded_chunk_ids must be sorted and unique"
        )
    if set(excluded) != union:
        raise LeakageError(
            "leakage artifact excluded_chunk_ids must equal union(question_chunk_map)"
        )
    if raw != _serialize(payload):
        raise LeakageError("leakage artifact is not in canonical serialized form")
    return set(excluded)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze graph-independent held-out leakage metadata."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DRAFT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        sha256 = freeze_leakage(
            args.input,
            args.output,
            chunks_path=args.chunks,
            manifest_path=args.manifest,
            ontology_path=args.ontology,
        )
    except LeakageError as exc:
        print(f"leakage freeze failed: {exc}", file=sys.stderr)
        return 2
    print(f"published {args.output}\nsha256 {sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CHUNKS_PATH",
    "DEFAULT_DRAFT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_ONTOLOGY_PATH",
    "DEFAULT_OUTPUT_PATH",
    "EXPECTED_RECORD_COUNT",
    "LeakageError",
    "SCHEMA_VERSION",
    "freeze_leakage",
    "load_frozen_exclusions",
    "main",
]
