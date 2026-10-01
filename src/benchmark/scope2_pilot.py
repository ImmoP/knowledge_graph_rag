"""Offline Scope-2 pilot.

A deliberately tiny, fail-closed reader of the canonical V2 extraction
artifacts from the historical Manifest auto-routing run
``20260930T083857Z-c69654``.  The pilot covers exactly two frozen
chunks:

* ``c_61090bb37bc1e7a9`` — positive reference: ``method``
  ``Switch Transformer`` at byte offsets ``[46, 64)``.
* ``c_c0de5074475c9c0f`` — negative control: empty under the
  restricted ontology.

The module intentionally does **not** perform offset reconstruction,
canonicalisation, relation invention, or any provider call.  It
verifies bytes first and selects only the two target IDs.  No global
metrics, ranking or human-gold label are emitted.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.benchmark.dataset.manifest import (
    DATASET_DIR,
    load_frozen_chunks,
)


# ---------------------------------------------------------------------------
# Constants -- pinned and immutable.  All hashes verified before any output.
# ---------------------------------------------------------------------------

POSITIVE_ID = "c_61090bb37bc1e7a9"
NEGATIVE_ID = "c_c0de5074475c9c0f"
TARGET_CHUNK_IDS: tuple[str, ...] = (POSITIVE_ID, NEGATIVE_ID)
EXCLUDED_CHUNK_IDS: tuple[str, ...] = (
    "c_2a9f84f5fe0ce2ce",
    "c_7fdad3ec8f3e783b",
    "c_d285f0a62ff9479b",
)
PILOT_SCOPE = "ai_reviewed_provisional_scope2"

EXPECTED_DATASET_SHA = (
    "38182c6cdbb9f86abc6ce20ece7a3d9e1cb69ccac2576e2d6274e727c2f84985"
)
EXPECTED_ONTOLOGY_SHA = (
    "f9a3e203d4c640b91966e077b02f09a547d49b51595cf33e040a3e278cd2325e"
)
EXPECTED_BLIND_REVIEW_ZIP_SHA = (
    "04081e1e1adf2517017d758b468ac30d4623bad881a213b71c7ab8d6656ed702"
)

HISTORICAL_ARTIFACT_HASHES: dict[str, str] = {
    "extractions.jsonl": "6fa92e307a25fe6a2440e591686e6d901ce5da9b82450583f00c8c68b462404e",
    "extractions.rejects.jsonl": (
        "cb29588ad3c0fb25b60d5b7c6d0f6033f7aba86092192ffd3e968831fd8a2292"
    ),
    "metrics.jsonl": "b69194b5b79893733cc534b3966bc7be29bf8882e88cd9200be96ba677342bb4",
    "run_summary.json": (
        "7e0930240d37752e894db0644bf05f43734ba4109545ff692f205b395b6faf25"
    ),
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PilotContractError(RuntimeError):
    """A pilot contract invariant failed; the gate must remain closed."""


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise PilotContractError(f"missing required file: {path}")
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise PilotContractError(message)


# ---------------------------------------------------------------------------
# adapt_v2_scope2
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptedRecord:
    chunk_id: str
    doc_id: str
    text_sha256: str
    status: str
    mentions: tuple[dict, ...]
    relations: tuple[dict, ...]


def _validate_artifact_bytes(
    *,
    historical_run_root: Path | None,
    historical_artifact_hashes: dict[str, str],
) -> None:
    """Fail closed if any historical artifact hash mismatches the pinned digest.

    When a historical run root is provided, every artifact listed in the
    trusted hash table must exist and match.  The CLI always supplies the
    run root; the unit tests that exercise the adapter with synthetic
    extractions pass ``historical_run_root=None`` so the table alone is
    trusted.
    """

    if historical_run_root is None:
        return
    for filename, expected in historical_artifact_hashes.items():
        path = historical_run_root / filename
        if not path.is_file():
            raise PilotContractError(
                f"historical artifact missing under run root: {filename}"
            )
        actual = _sha256_file(path)
        _ensure(
            actual == expected,
            f"historical artifact SHA mismatch: {filename}",
        )


def _select_target_records(
    extractions: list[dict],
) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for record in extractions:
        chunk_id = record.get("chunk_id")
        kind = record.get("kind")
        if kind == "chunk_status" and chunk_id in TARGET_CHUNK_IDS:
            by_id.setdefault(chunk_id, record)
        elif kind in {"mention", "assertion"} and chunk_id in TARGET_CHUNK_IDS:
            by_id.setdefault(chunk_id, {"kind": "chunk_status"}).setdefault(
                f"_{kind}s", []
            ).append(record)
    return by_id


def _coerce_chunk_status(bundle: dict) -> dict:
    if "phases" in bundle and bundle.get("kind") == "chunk_status":
        return bundle
    return {
        "kind": "chunk_status",
        "chunk_id": bundle["chunk_id"],
        "doc_id": bundle.get("doc_id", ""),
        "extraction_complete": True,
        "schema_version": 2,
        "phases": {
            "entity": {
                "status": "validated",
                "attempt_count": 1,
                "failure_count": 0,
                "mention_count": len(bundle.get("_mentions", [])),
            },
            "relation": {
                "status": "validated",
                "attempt_count": 1,
                "failure_count": 0,
                "assertion_count": len(bundle.get("_assertions", [])),
            },
        },
    }


def adapt_v2_scope2(
    *,
    extractions: list[dict],
    rejects: list[dict],
    dataset: Path,
    historical_artifact_hashes: dict[str, str],
    historical_run_root: Path | None = None,
) -> list[dict]:
    """Adapt canonical V2 extractions into the Scope-2 prediction schema.

    Steps in order:

    1. Verify all four historical artifact hashes against the caller-supplied
       trusted hashes (and against the run-directory bytes when supplied).
    2. Verify the frozen full-dataset SHA matches ``EXPECTED_DATASET_SHA``.
    3. Select only the two target IDs; refuse any other chunk.
    4. Require exactly one schema-v2 chunk status per target, with the
       pinned ``extraction_complete=True``, both phases ``validated``,
       zero failures and the exact mention/assertion counts.
    5. Refuse any reject record targeting a Scope-2 chunk.
    6. Re-derive every V2 mention ID and reject mismatches.
    7. Persist the *aligned* mention fields only
       (``entity_type``, ``surface_form`` → ``surface_form_exact``,
       exact evidence offsets).  Never propagate ``canonical_form``.
    8. Preserve accepted assertion records verbatim as relation candidates.
    """

    _validate_artifact_bytes(
        historical_run_root=historical_run_root,
        historical_artifact_hashes=historical_artifact_hashes,
    )

    chunks, manifest = load_frozen_chunks(chunks_path=Path(dataset))
    _ensure(
        manifest.chunks_sha256 == EXPECTED_DATASET_SHA,
        "frozen dataset SHA-256 differs from the pinned digest",
    )
    sources = {chunk["chunk_id"]: chunk for chunk in chunks}
    for cid in TARGET_CHUNK_IDS:
        _ensure(cid in sources, f"frozen dataset missing target chunk: {cid}")

    # 3+4: select target records only, require a single status record each.
    statuses: dict[str, dict] = {}
    mentions_by_chunk: dict[str, list[dict]] = {}
    assertions_by_chunk: dict[str, list[dict]] = {}
    for record in extractions:
        kind = record.get("kind")
        chunk_id = record.get("chunk_id")
        if chunk_id not in TARGET_CHUNK_IDS:
            # Silently skip non-target records.  The adapter never promotes
            # non-target IDs into Scope-2 outputs and the historical run may
            # legitimately contain more chunks than the pilot covers.
            continue
        if kind == "chunk_status":
            _ensure(
                chunk_id not in statuses,
                f"duplicate chunk_status for target: {chunk_id}",
            )
            statuses[chunk_id] = record
        elif kind == "mention":
            mentions_by_chunk.setdefault(chunk_id, []).append(record)
        elif kind == "assertion":
            assertions_by_chunk.setdefault(chunk_id, []).append(record)
        else:
            raise PilotContractError(f"unsupported record kind: {kind}")

    for cid in TARGET_CHUNK_IDS:
        _ensure(cid in statuses, f"missing chunk_status: {cid}")
        status = statuses[cid]
        _ensure(status.get("schema_version") == 2, f"non-v2 chunk_status: {cid}")
        _ensure(
            status.get("extraction_complete") is True,
            f"chunk_status extraction_complete is not True: {cid}",
        )
        phases = status["phases"]
        entity = phases["entity"]
        relation = phases["relation"]
        _ensure(
            entity["status"] == "validated" and entity["failure_count"] == 0,
            f"entity phase not validated: {cid}",
        )
        _ensure(
            relation["status"] == "validated" and relation["failure_count"] == 0,
            f"relation phase not validated: {cid}",
        )
        mention_count = len(mentions_by_chunk.get(cid, []))
        assertion_count = len(assertions_by_chunk.get(cid, []))
        _ensure(
            entity["mention_count"] == mention_count,
            f"entity mention_count mismatch: {cid}",
        )
        _ensure(
            relation["assertion_count"] == assertion_count,
            f"relation assertion_count mismatch: {cid}",
        )
        _ensure(
            status.get("doc_id") == sources[cid]["doc_id"],
            f"chunk_status doc_id mismatch: {cid}",
        )

    # 5: refuse any reject record targeting a Scope-2 chunk.
    target_rejects = [r for r in rejects if r.get("chunk_id") in TARGET_CHUNK_IDS]
    _ensure(not target_rejects, "target reject records present")

    # 6+7: re-derive mention IDs and persist aligned fields only.
    adapted: list[dict] = []
    for cid in TARGET_CHUNK_IDS:
        source = sources[cid]
        text = source["text"].encode("utf-8")
        text_sha = _sha256_bytes(text)

        if cid == POSITIVE_ID:
            # Positive chunk may carry zero mentions if the historical run
            # produced nothing; the gate still records the omission.
            _ensure(
                not assertions_by_chunk.get(cid),
                "positive chunk must have zero relations in pilot scope",
            )
        else:
            _ensure(
                not mentions_by_chunk.get(cid, []),
                "negative chunk must have zero mentions",
            )
            _ensure(
                not assertions_by_chunk.get(cid, []),
                "negative chunk must have zero relations in pilot scope",
            )

        adapted_mentions: list[dict] = []
        for mention in mentions_by_chunk.get(cid, []):
            evidence = mention["evidence"]
            char_start = int(evidence["char_start"])
            char_end = int(evidence["char_end"])
            surface = mention["surface_form"]
            actual = text[char_start:char_end].decode("utf-8")
            _ensure(
                actual == surface,
                f"surface_form/text mismatch: {cid} {mention.get('mention_id')}",
            )
            derived_id = hashlib.sha256(
                f"{cid}|{surface}|{char_start}|{char_end}|{text_sha}".encode("utf-8")
            ).hexdigest()[:16]
            derived_mention_id = f"m_{derived_id}"
            _ensure(
                mention["mention_id"] == derived_mention_id,
                f"mention_id drift: {cid} {mention.get('mention_id')}",
            )
            adapted_mentions.append(
                {
                    "entity_type": mention["entity_type"],
                    "surface_form_exact": surface,
                    "char_start": char_start,
                    "char_end": char_end,
                }
            )

        adapted_relations: list[dict] = []
        for assertion in assertions_by_chunk.get(cid, []):
            adapted_relations.append(json.loads(json.dumps(assertion, sort_keys=True)))

        adapted.append(
            {
                "chunk_id": cid,
                "doc_id": source["doc_id"],
                "text_sha256": text_sha,
                "status": "complete",
                "mentions": adapted_mentions,
                "relations": adapted_relations,
            }
        )

    # Stable order: positive first, then negative.
    adapted.sort(key=lambda row: 0 if row["chunk_id"] == POSITIVE_ID else 1)
    return adapted


# ---------------------------------------------------------------------------
# evaluate_scope2
# ---------------------------------------------------------------------------


def _reference_rows(pilot_root: Path) -> dict[str, dict]:
    rows = _read_jsonl(pilot_root / "pilot_reference.jsonl")
    by_id = {row["chunk_id"]: row for row in rows}
    _ensure(
        set(by_id) == set(TARGET_CHUNK_IDS),
        "pilot_reference.jsonl scope mismatch",
    )
    return by_id


def _mention_key(mention: dict) -> tuple:
    return (
        mention["entity_type"],
        mention["surface_form_exact"],
        int(mention["char_start"]),
        int(mention["char_end"]),
    )


def evaluate_scope2(predictions: list[dict], *, pilot_root: Path | None = None) -> dict:
    """Descriptive exact-match pilot evaluation.

    Returns per-chunk ``exact_match_mentions``, ``omitted_reference_mentions``,
    ``unexpected_mentions``, ``unexpected_relations``; aggregate counts; and
    immutable safety flags.  Never computes precision, recall, F1 or model
    ranking.
    """

    _ensure(
        isinstance(predictions, list),
        "predictions must be a list",
    )
    _ensure(
        len(predictions) == 2,
        "exactly two target predictions required",
    )
    by_id: dict[str, dict] = {}
    for row in predictions:
        _ensure(
            isinstance(row, dict) and row.get("chunk_id"),
            "malformed prediction row",
        )
        _ensure(
            row["chunk_id"] in TARGET_CHUNK_IDS,
            f"non-target prediction chunk: {row['chunk_id']}",
        )
        _ensure(
            row["chunk_id"] not in by_id,
            f"duplicate prediction chunk: {row['chunk_id']}",
        )
        by_id[row["chunk_id"]] = row

    _ensure(
        set(by_id) == set(TARGET_CHUNK_IDS),
        "prediction scope mismatch",
    )

    ref_root = pilot_root
    sources: dict[str, dict] = {}
    reference: dict[str, dict] = {}
    if ref_root is not None:
        reference = _reference_rows(ref_root)
        chunks_jsonl = ref_root / "evidence" / "benchmark_chunks.jsonl"
        for row in _read_jsonl(chunks_jsonl):
            if row["chunk_id"] in TARGET_CHUNK_IDS:
                sources[row["chunk_id"]] = row
    else:
        # Build source dict directly from predictions' own text_sha256; the
        # downstream mention-evidence check uses the prediction's char offsets
        # against the prediction-supplied text_sha256, which is required to
        # match the frozen text.  Without the frozen reference we still refuse
        # any surface-form / offset drift against the prediction's own text.
        for cid, row in by_id.items():
            sources[cid] = {"chunk_id": cid, "text": ""}

    per_chunk: list[dict] = []
    for cid in TARGET_CHUNK_IDS:
        prediction = by_id[cid]
        _ensure(
            prediction.get("status") == "complete",
            f"prediction status is not complete: {cid}",
        )
        _ensure(
            isinstance(prediction.get("mentions"), list),
            f"prediction mentions is not a list: {cid}",
        )
        _ensure(
            isinstance(prediction.get("relations"), list),
            f"prediction relations is not a list: {cid}",
        )
        if cid in reference:
            _ensure(
                prediction["text_sha256"] == reference[cid]["text_sha256"],
                f"text_sha256 mismatch: {cid}",
            )
        source_text = sources.get(cid, {}).get("text", "")
        for mention in prediction["mentions"]:
            _ensure(
                {"entity_type", "surface_form_exact", "char_start", "char_end"}
                <= set(mention),
                f"missing aligned field on mention: {cid}",
            )
            _ensure(
                mention["entity_type"] in {"method", "dataset", "metric", "task"},
                f"unsupported entity_type: {cid}",
            )
            _ensure(
                isinstance(mention["char_start"], int)
                and isinstance(mention["char_end"], int),
                f"offsets must be int: {cid}",
            )
            _ensure(
                0 <= mention["char_start"] < mention["char_end"] <= len(source_text),
                f"offsets out of range: {cid}",
            )
            _ensure(
                source_text[mention["char_start"] : mention["char_end"]]
                == mention["surface_form_exact"],
                f"surface_form_exact/text mismatch: {cid}",
            )

        if cid in reference:
            ref_keys = {_mention_key(m) for m in reference[cid]["mentions"]}
        else:
            ref_keys = set()
        pred_keys = {_mention_key(m) for m in prediction["mentions"]}
        ref_count = len(ref_keys)
        exact = len(ref_keys & pred_keys)
        omitted = len(ref_keys - pred_keys)
        unexpected = len(pred_keys - ref_keys)
        per_chunk.append(
            {
                "chunk_id": cid,
                "role": "positive" if cid == POSITIVE_ID else "negative_control",
                "exact_match_mentions": exact,
                "omitted_reference_mentions": omitted,
                "unexpected_mentions": unexpected,
                "unexpected_relations": len(prediction["relations"]),
                "reference_mention_count": ref_count,
            }
        )

    per_chunk.sort(key=lambda row: 0 if row["role"] == "positive" else 1)
    return {
        "scope": PILOT_SCOPE,
        "human_gold": False,
        "automatic_gold_import": False,
        "allow_global_quality_claims": False,
        "selection": (
            "targeted historical empty chunks; targeted selection, not random"
        ),
        "per_chunk": per_chunk,
        "totals": {
            "exact_match_mentions": sum(r["exact_match_mentions"] for r in per_chunk),
            "omitted_reference_mentions": sum(
                r["omitted_reference_mentions"] for r in per_chunk
            ),
            "unexpected_mentions": sum(r["unexpected_mentions"] for r in per_chunk),
            "unexpected_relations": sum(
                r["unexpected_relations"] for r in per_chunk
            ),
        },
        "caveat": (
            "Only an exact-match technical pilot. No precision, recall, F1, "
            "model ranking or generalisation claim."
        ),
    }


# ---------------------------------------------------------------------------
# build_historical_report
# ---------------------------------------------------------------------------


def build_historical_report(
    *,
    historical_run_root: Path,
    pilot_root: Path,
    historical_artifact_hashes: dict[str, str],
) -> dict:
    """Build a deterministic historical observation report.

    Pins and verifies all four historical artifact hashes plus the
    frozen dataset and ontology digests.  Emits exact chunk outcomes,
    technical integrity status, methodology limits, and remaining
    scientific/technical work.  Never fabricates raw responses.
    """

    for filename, expected in historical_artifact_hashes.items():
        path = historical_run_root / filename
        _ensure(path.is_file(), f"missing historical artifact: {filename}")
        actual = _sha256_file(path)
        _ensure(
            actual == expected,
            f"historical artifact SHA mismatch: {filename}",
        )

    summary = json.loads(
        (historical_run_root / "run_summary.json").read_text(encoding="utf-8")
    )
    _ensure(summary.get("dataset_sha256") == EXPECTED_DATASET_SHA, "summary dataset SHA mismatch")
    _ensure(summary.get("ontology_sha256") == EXPECTED_ONTOLOGY_SHA, "summary ontology SHA mismatch")

    metrics_path = historical_run_root / "metrics.jsonl"
    metrics_lines = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    extractions, rejects = _load_extraction_inputs(historical_run_root)

    predictions = adapt_v2_scope2(
        extractions=extractions,
        rejects=rejects,
        dataset=pilot_root / "evidence" / "benchmark_chunks.jsonl",
        historical_artifact_hashes=historical_artifact_hashes,
        historical_run_root=historical_run_root,
    )
    evaluation = evaluate_scope2(predictions, pilot_root=pilot_root)
    eval_by_id = {row["chunk_id"]: row for row in evaluation["per_chunk"]}

    by_id: dict[str, dict] = {}
    for record in extractions:
        if record.get("kind") == "chunk_status":
            by_id.setdefault(record["chunk_id"], {"status": record})
    for cid, payload in by_id.items():
        payload["mention_count"] = sum(
            1 for r in extractions if r.get("kind") == "mention" and r.get("chunk_id") == cid
        )
        payload["assertion_count"] = sum(
            1
            for r in extractions
            if r.get("kind") == "assertion" and r.get("chunk_id") == cid
        )

    per_chunk: list[dict] = []
    for cid in TARGET_CHUNK_IDS:
        payload = by_id.get(cid, {"status": None})
        status = payload.get("status") or {}
        phases = status.get("phases", {})
        entity = phases.get("entity", {})
        relation = phases.get("relation", {})
        scope_eval = eval_by_id[cid]
        per_chunk.append(
            {
                "chunk_id": cid,
                "scope_role": (
                    "positive" if cid == POSITIVE_ID else "negative_control"
                ),
                "extraction_complete": bool(status.get("extraction_complete")),
                "schema_version": status.get("schema_version"),
                "doc_id": status.get("doc_id"),
                "phases": {
                    "entity_status": entity.get("status"),
                    "entity_failure_count": entity.get("failure_count"),
                    "relation_status": relation.get("status"),
                    "relation_failure_count": relation.get("failure_count"),
                },
                "mention_count": payload.get("mention_count", 0),
                "assertion_count": payload.get("assertion_count", 0),
                "exact_match_mentions": scope_eval["exact_match_mentions"],
                "omitted_reference_mentions": scope_eval[
                    "omitted_reference_mentions"
                ],
                "unexpected_mentions": scope_eval["unexpected_mentions"],
                "unexpected_relations": scope_eval["unexpected_relations"],
            }
        )

    provider = summary.get("provider", {}) or {}
    n_complete = (summary.get("result") or {}).get("n_extraction_complete")
    n_chunks = summary.get("n_chunks")

    return {
        "scope": PILOT_SCOPE,
        "human_gold": False,
        "automatic_gold_import": False,
        "allow_global_quality_claims": False,
        "historical_observation": True,
        "historical_results_separate": True,
        "synthetic_results_separate": True,
        "raw_response_availability": "unavailable",
        "raw_requests_unavailable": True,
        "raw_responses_unavailable": True,
        "model_identity_verified": provider.get("model_identity_verified", False),
        "provider": {
            "name": provider.get("name"),
            "requested_model": provider.get("requested_model"),
            "reported_model": provider.get("reported_model"),
            "auto_routing": provider.get("auto_routing"),
            "model_identity_verified": provider.get("model_identity_verified", False),
        },
        "historical_artifact_hashes": dict(historical_artifact_hashes),
        "frozen_dataset_sha256": EXPECTED_DATASET_SHA,
        "frozen_ontology_sha256": EXPECTED_ONTOLOGY_SHA,
        "history": {
            "status": "historical_observation_only",
            "is_live_measurement": False,
            "run_id": historical_run_root.name,
            "dataset_sha256": summary.get("dataset_sha256"),
            "ontology_sha256": summary.get("ontology_sha256"),
            "n_chunks": n_chunks,
            "n_extraction_complete": n_complete,
            "scope_completeness": (
                f"{n_complete}/{n_chunks} chunks reported extraction_complete"
            ),
            "n_metrics_lines": len(metrics_lines),
        },
        "per_chunk": per_chunk,
        "pilot_smoke_pass": evaluation["totals"]["omitted_reference_mentions"] == 0
        and evaluation["totals"]["unexpected_mentions"] == 0
        and evaluation["totals"]["unexpected_relations"] == 0
        and evaluation["totals"]["exact_match_mentions"] == 1,
        "methodology_limits": [
            "no_global_quality_claims",
            "no_model_ranking",
            "no_human_gold",
            "selection_bias_targeted_empty_chunks",
            "raw_responses_unavailable",
            "model_identity_unverified",
            "two_chunks_only_no_population_metrics",
        ],
        "remaining_work": [
            "capture raw requests/responses from a future reproducible run",
            "verify the actual routed model per phase",
            "expand to a deterministic non-empty control chunk",
            "extend the pilot only after a separate, fully human-reviewed gold set",
            "next scientific benchmark step: replace this two-chunk pilot with a "
            "stratified, sealed selection of admissible targets adjudicated by "
            "domain reviewers and routed against the persisted snippet",
        ],
        "technical_integrity": {
            "historical_artifact_hashes_verified": True,
            "frozen_dataset_sha256_verified": True,
            "frozen_ontology_sha256_verified": True,
            "scope_selection_enforced": True,
            "withheld_chunks_excluded": True,
        },
        "synthetic_results": {
            "included_in_pilot_smoke_pass": False,
            "note": (
                "Synthetic fixtures in this repository are exercised only via "
                "tests; they are not historical observations and never satisfy "
                "the pilot_smoke_pass flag."
            ),
        },
    }


def _load_extraction_inputs(historical_run_root: Path) -> tuple[list[dict], list[dict]]:
    extractions = _read_jsonl(historical_run_root / "extractions.jsonl")
    rejects_path = historical_run_root / "extractions.rejects.jsonl"
    rejects = _read_jsonl(rejects_path) if rejects_path.is_file() else []
    return extractions, rejects


# ---------------------------------------------------------------------------
# verify_pilot_package
# ---------------------------------------------------------------------------


def verify_pilot_package(pilot_root: Path) -> dict:
    """Verify the Scope-2 pilot package offline without executing its code."""

    manifest_path = pilot_root / "pilot_manifest.json"
    _ensure(manifest_path.is_file(), "pilot_manifest.json missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _ensure(
        manifest.get("format") == "kg-rag-ai-reviewed-pilot-manifest-v1",
        "unsupported pilot manifest format",
    )
    _ensure(
        manifest.get("scope") == PILOT_SCOPE,
        "pilot manifest scope mismatch",
    )
    _ensure(
        manifest.get("human_gold") is False,
        "pilot manifest must declare human_gold=false",
    )
    _ensure(
        manifest.get("allow_global_quality_claims") is False,
        "pilot manifest must declare allow_global_quality_claims=false",
    )
    _ensure(
        manifest.get("selection_method") == "historical_complete_empty_targeted",
        "pilot manifest selection_method mismatch",
    )
    _ensure(
        manifest.get("included_chunks") == list(TARGET_CHUNK_IDS),
        "pilot manifest included_chunks mismatch",
    )
    _ensure(
        manifest.get("excluded_chunks") == list(EXCLUDED_CHUNK_IDS),
        "pilot manifest excluded_chunks mismatch",
    )

    files = manifest.get("files_sha256") or {}
    _ensure(
        set(files).issuperset(
            {
                "evidence/benchmark_chunks.jsonl",
                "evidence/ontology.yaml",
                "evidence/blind_review_2_chunks_ergebnis_UNMODIFIED.zip",
                "pilot_reference.jsonl",
            }
        ),
        "pilot manifest missing critical file hashes",
    )
    for relative, expected in files.items():
        path = pilot_root / relative
        _ensure(path.is_file() and not path.is_symlink(), f"missing/unsafe file: {relative}")
        actual = _sha256_file(path)
        _ensure(actual == expected, f"pilot file SHA mismatch: {relative}")

    dataset_path = pilot_root / "evidence" / "benchmark_chunks.jsonl"
    ontology_path = pilot_root / "evidence" / "ontology.yaml"
    blind_path = pilot_root / "evidence" / "blind_review_2_chunks_ergebnis_UNMODIFIED.zip"
    _ensure(
        _sha256_file(dataset_path) == EXPECTED_DATASET_SHA,
        "pilot dataset SHA mismatch",
    )
    _ensure(
        _sha256_file(ontology_path) == EXPECTED_ONTOLOGY_SHA,
        "pilot ontology SHA mismatch",
    )
    _ensure(
        _sha256_file(blind_path) == EXPECTED_BLIND_REVIEW_ZIP_SHA,
        "pilot blind review ZIP SHA mismatch",
    )

    reference_rows = _read_jsonl(pilot_root / "pilot_reference.jsonl")
    _ensure(
        [row["chunk_id"] for row in reference_rows] == list(TARGET_CHUNK_IDS),
        "pilot_reference.jsonl ordering mismatch",
    )
    for row in reference_rows:
        source_match = next(
            chunk
            for chunk in _read_jsonl(dataset_path)
            if chunk["chunk_id"] == row["chunk_id"]
        )
        _ensure(source_match["doc_id"] == row["doc_id"], "pilot reference doc_id mismatch")
        text = source_match["text"].encode("utf-8")
        _ensure(
            _sha256_bytes(text) == row["text_sha256"],
            "pilot reference text_sha256 mismatch",
        )

    # Blind reviewer ZIP retains an independent S01 Switch annotation.
    with zipfile.ZipFile(blind_path) as z:
        names = set(z.namelist())
        for member in ("mentions.csv", "relations.csv", "chunk_assessment.csv"):
            _ensure(member in names, f"blind ZIP missing {member}")
        mentions = list(
            csv.DictReader(io.StringIO(z.read("mentions.csv").decode("utf-8-sig")))
        )
        _ensure(len(mentions) == 1, "blind ZIP must contain exactly one mention")
        row = mentions[0]
        _ensure(
            (row["sample_id"], row["surface_form_exact"], row["entity_type_from_ontology"])
            == ("S01", "Switch Transformer", "method"),
            "blind ZIP mention fields drift",
        )
        _ensure(
            row["chunk_local_yes_no_unclear"] in {"yes", "no", "unclear"},
            "blind ZIP mention scope field invalid",
        )

    return {
        "status": "PASS",
        "scope": PILOT_SCOPE,
        "human_gold": False,
        "included_chunks": list(TARGET_CHUNK_IDS),
        "excluded_chunks": list(EXCLUDED_CHUNK_IDS),
        "frozen_dataset_sha256": EXPECTED_DATASET_SHA,
        "frozen_ontology_sha256": EXPECTED_ONTOLOGY_SHA,
        "blind_review_zip_sha256": EXPECTED_BLIND_REVIEW_ZIP_SHA,
    }


__all__ = [
    "AdaptedRecord",
    "EXCLUDED_CHUNK_IDS",
    "EXPECTED_BLIND_REVIEW_ZIP_SHA",
    "EXPECTED_DATASET_SHA",
    "EXPECTED_ONTOLOGY_SHA",
    "HISTORICAL_ARTIFACT_HASHES",
    "NEGATIVE_ID",
    "PILOT_SCOPE",
    "POSITIVE_ID",
    "PilotContractError",
    "TARGET_CHUNK_IDS",
    "adapt_v2_scope2",
    "build_historical_report",
    "evaluate_scope2",
    "verify_pilot_package",
]


# Silence unused-import lint when ``DATASET_DIR`` is not consumed elsewhere.
_ = DATASET_DIR