"""Deterministic acceptance tests for the v1 difficulty classifier (#97 / #48 Refinement 5).

These tests build synthetic gold records + sample rows under ``tmp_path``,
run ``scripts.gold_set.classify_difficulty`` against them, and assert that
the resulting classification matches the Spec verbatim. They never read the
canonical 120 gold records under ``gold_set/chunks/``; canonical records
are read-only inputs and may not be modified by this PR.

Tests cover:

* Easy / hard / medium classification per Refinement 5.
* Hard-easy-medium precedence (partition invariant).
* Spec gap 1: ``COMMON_NOUNS = {"attention", "model", "score"}`` is the
  v1 minimum; a literal that matches trips the easy / hard rules.
* Spec gap 2: sentence-boundary heuristic.
* Spec gap 3: easy rule applies across all four entity types.
* Spec gap 4: hard rule on common-noun method endpoint applies only when
  ``entity_type == "method"``.
* Per-record SHA256 pin reproducibility (byte-identical re-runs).
* Audit fields (classifier version, paper-tier counts, hard-trigger counts).
* Fail-closed on missing sample row.
* Skipped-on-missing-sample behavior is recorded in the audit.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIFFICULTY = ROOT / "scripts" / "gold_set" / "classify_difficulty.py"


# ---------------------------------------------------------------------------
# Module import (classify_difficulty is a script, not a package module).
# ---------------------------------------------------------------------------


_spec = importlib.util.spec_from_file_location(
    "scripts.gold_set.classify_difficulty", DIFFICULTY
)
difficulty = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = difficulty
_spec.loader.exec_module(difficulty)


# ---------------------------------------------------------------------------
# Test fixture builders.
# ---------------------------------------------------------------------------


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _enumerate_occurrences(text: str, literal: str) -> list[int]:
    starts: list[int] = []
    pos = 0
    while True:
        idx = text.find(literal, pos)
        if idx < 0:
            break
        starts.append(idx)
        pos = idx + len(literal)
    return starts


def _make_entity(
    chunk_id: str,
    seq: int,
    entity_type: str,
    text: str,
    literal: str,
    *,
    confidence: str = "stated",
) -> dict:
    token = chunk_id[2:10]
    starts = _enumerate_occurrences(text, literal)
    return {
        "entity_id": f"ge_{token}_{seq:02d}",
        "entity_type": entity_type,
        "surface_form": literal,
        "canonical_form": literal,
        "span": {"char_start": starts[0], "char_end": starts[0] + len(literal)},
        "occurrences_in_chunk": [
            {"occurrence_index": i, "char_start": s, "char_end": s + len(literal)}
            for i, s in enumerate(starts)
        ],
        "confidence": confidence,
    }


def _make_record(
    chunk_id: str,
    doc_id: str,
    text: str,
    *,
    entities: list[dict],
    relations: list[dict],
    paper_tier: str,
    note: str | None = None,
    chunk_sha: str = "fake-canonical-chunks-sha",
    ontology_sha: str = "fake-ontology-sha",
    guideline_sha: str = "fake-guideline-sha",
) -> tuple[dict, dict]:
    """Return ``(sample_row, gold_record)`` ready to be written."""
    sample_row = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "sampler_version": "gold-set-sample.v2",
        "sampler_seed": "9e3779b1",
        "strata": {
            "paper_tier": paper_tier,
            "section_depth": 1,
            "length_tercile": "medium",
            "density_post_stratum": "non_empty",
        },
    }
    record: dict = {
        "schema_version": "gold-record.v1",
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "canonical_chunks_sha256_at_authoring": chunk_sha,
        "ontology_version": 1,
        "ontology_sha256": ontology_sha,
        "annotation_guideline_version": "1",
        "annotation_guideline_sha256": guideline_sha,
        "annotator_id": "ann-A",
        "adjudication_status": "single_pass",
        "adjudicated_by": None,
        "annotated_at": "2026-09-22T00:00:00Z",
        "entities": entities,
        "relations": relations,
    }
    if note is not None:
        record["note"] = note
    return sample_row, record


def _setup_paths(
    tmp_path: Path,
    *,
    sample_rows: list[dict],
    records: dict[str, dict],
    sample_audit: dict | None = None,
) -> dict:
    sample_path = tmp_path / "sample.v2.jsonl"
    gold_dir = tmp_path / "chunks"
    gold_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = tmp_path / "difficulty.v1.jsonl"
    output_audit = tmp_path / "difficulty.v1.audit.json"
    sample_audit_path = tmp_path / "sample.v2.audit.json"

    _jsonl(sample_path, sample_rows)
    for chunk_id, record in records.items():
        (gold_dir / f"{chunk_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    if sample_audit is None:
        sample_audit = {
            "selected_chunk_ids_sha256": hashlib.sha256(
                "".join(sorted(row["chunk_id"] for row in sample_rows)).encode("utf-8")
            ).hexdigest(),
            "chunks_sha256": "fake",
            "manifest_sha256": "fake",
            "ontology_sha256": "fake",
            "leakage_artifact_sha256": "fake",
            "leakage_overlap_count": 0,
        }
    _json(sample_audit_path, sample_audit)

    return {
        "sample": sample_path,
        "gold_dir": gold_dir,
        "output_jsonl": output_jsonl,
        "output_audit": output_audit,
        "sample_audit": sample_audit_path,
    }


# A handful of pre-built textual fixtures used across the tests below.


TEXT_EASY = (
    "We propose DPR as a retrieval method that combines BERT and BM25 to "
    "answer questions on the SQuAD benchmark. Experiments report F1 and EM."
)
TEXT_HARD_MULTI = (
    "We compare model A and model B on retrieval. Model A scores 41.5 F1, "
    "while model B scores 39.2 F1. Both use BM25 and DPR."
)
TEXT_HARD_HEDGED = (
    "We propose DPR as a method that may improve over BM25 on SQuAD, and "
    "speculate that a larger batch could yield better F1."
)
TEXT_HARD_BOUNDARY = (
    "DPR is a retrieval method. We evaluate it on the SQuAD benchmark with "
    "F1 and EM as the headline metrics."
)
TEXT_LONG_TAIL_EASY = (
    "We propose DPR and BM25 to evaluate on SQuAD with F1. This paper is "
    "in the long tail of citations."
)
TEXT_COMMON_NOUN_ENDPOINT = (
    "We propose attention as a method to improve DPR. Experiments on SQuAD "
    "report F1 and EM as headline metrics."
)


# ---------------------------------------------------------------------------
# Triggers / decide.
# ---------------------------------------------------------------------------


def test_decide_hard_dominates_easy():
    """A chunk with any hard trigger is ``hard``, even if all easy conditions hold."""
    triggers = difficulty.Triggers(
        has_multi_occurrence_disambiguation=True,
        has_evidence_spanning_sentence_boundary=False,
        has_inferred_or_hedged_relation=False,
        has_common_noun_method_endpoint=False,
        paper_tier_long_tail=True,
        easy_conditions=(True, True, True, True),
    )
    assert difficulty.decide(triggers) == "hard"


def test_decide_easy_requires_all_conditions():
    triggers = difficulty.Triggers(
        has_multi_occurrence_disambiguation=False,
        has_evidence_spanning_sentence_boundary=False,
        has_inferred_or_hedged_relation=False,
        has_common_noun_method_endpoint=False,
        paper_tier_long_tail=False,
        easy_conditions=(True, True, True, False),
    )
    assert difficulty.decide(triggers) == "medium"

    triggers_easy = difficulty.Triggers(
        has_multi_occurrence_disambiguation=False,
        has_evidence_spanning_sentence_boundary=False,
        has_inferred_or_hedged_relation=False,
        has_common_noun_method_endpoint=False,
        paper_tier_long_tail=False,
        easy_conditions=(True, True, True, True),
    )
    assert difficulty.decide(triggers_easy) == "easy"


def test_partition_invariant_under_random_combinations():
    """No trigger combination may produce an undefined label, and
    every classification must be exactly one of easy/medium/hard."""
    from itertools import product

    for hard_combo in product([False, True], repeat=5):
        for easy_combo in product([False, True], repeat=4):
            triggers = difficulty.Triggers(
                has_multi_occurrence_disambiguation=hard_combo[0],
                has_evidence_spanning_sentence_boundary=hard_combo[1],
                has_inferred_or_hedged_relation=hard_combo[2],
                has_common_noun_method_endpoint=hard_combo[3],
                paper_tier_long_tail=hard_combo[4],
                easy_conditions=easy_combo,
            )
            label = difficulty.decide(triggers)
            assert label in ("easy", "medium", "hard")
            # Hard always wins when present.
            if any(hard_combo):
                assert label == "hard"
            elif all(easy_combo):
                assert label == "easy"
            else:
                assert label == "medium"


# ---------------------------------------------------------------------------
# Classification against synthetic gold records.
# ---------------------------------------------------------------------------


def _record_id(test_name: str, suffix: str = "01") -> str:
    """Deterministic synthetic chunk_id for a test name + suffix.

    Hashes the test name AND suffix so two calls with the same name but
    different suffixes (e.g. ``missing_sample`` vs
    ``missing_sample``/suffix=02) produce distinct IDs.
    """
    token = hashlib.sha256(f"{test_name}|{suffix}".encode("utf-8")).hexdigest()[:16]
    return f"c_{token}"

def _doc_id() -> str:
    return "W1"


def test_easy_chunk(tmp_path):
    text = TEXT_EASY
    chunk_id = _record_id("easy")
    entities = [
        _make_entity(chunk_id, 1, "method", text, "DPR"),
        _make_entity(chunk_id, 2, "method", text, "BM25"),
        _make_entity(chunk_id, 3, "method", text, "BERT"),
        _make_entity(chunk_id, 4, "dataset", text, "SQuAD"),
        _make_entity(chunk_id, 5, "metric", text, "F1"),
        _make_entity(chunk_id, 6, "metric", text, "EM"),
    ]
    relations = [
        {
            "relation_id": "gr_01",
            "relation_type": "EVALUATES_ON",
            "subject_ref": {"kind": "mention", "id": entities[0]["entity_id"]},
            "object_ref": {"kind": "mention", "id": entities[3]["entity_id"]},
            "evidence_quote": "DPR as a retrieval method that combines BERT",
            "evidence_span": {"char_start": 10, "char_end": 53},
            "endpoint_mentions": [],
            "confidence": "stated",
        }
    ]
    sample_row, record = _make_record(
        chunk_id, _doc_id(), text,
        entities=entities, relations=relations,
        paper_tier="benchmark_central",
    )
    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record})
    audit = difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    rows = [json.loads(line) for line in paths["output_jsonl"].read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["difficulty"] == "easy"
    assert rows[0]["classifier_version"] == difficulty.CLASSIFIER_VERSION
    assert rows[0]["classifier_id"] == difficulty.CLASSIFIER_ID
    assert audit["difficulty_counts"]["easy"]["count"] == 1
    assert audit["difficulty_counts"]["hard"]["count"] == 0


def test_hard_chunk_long_tail_paper(tmp_path):
    """Paper tier ``long_tail`` alone is a hard trigger, even when all easy conditions
    would otherwise hold (here all easy conditions DO hold too, demonstrating
    hard-dominates-easy precedence)."""
    text = TEXT_LONG_TAIL_EASY
    chunk_id = _record_id("hard_long_tail")
    entities = [
        _make_entity(chunk_id, 1, "method", text, "DPR"),
        _make_entity(chunk_id, 2, "method", text, "BM25"),
        _make_entity(chunk_id, 3, "dataset", text, "SQuAD"),
        _make_entity(chunk_id, 4, "metric", text, "F1"),
    ]
    relations = [
        {
            "relation_id": "gr_01",
            "relation_type": "EVALUATES_ON",
            "subject_ref": {"kind": "mention", "id": entities[0]["entity_id"]},
            "object_ref": {"kind": "mention", "id": entities[2]["entity_id"]},
            "evidence_quote": "DPR and BM25 to evaluate on SQuAD",
            "evidence_span": {"char_start": 10, "char_end": 40},
            "endpoint_mentions": [],
            "confidence": "stated",
        }
    ]
    sample_row, record = _make_record(
        chunk_id, _doc_id(), text,
        entities=entities, relations=relations,
        paper_tier="long_tail",
    )
    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record})
    difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    rows = [json.loads(line) for line in paths["output_jsonl"].read_text().splitlines() if line]
    assert rows[0]["difficulty"] == "hard"
    assert "long_tail_paper" in rows[0]["triggers"]["hard_triggers"]


def test_hard_chunk_inferred_relation(tmp_path):
    text = TEXT_HARD_HEDGED
    chunk_id = _record_id("hard_inferred")
    entities = [
        _make_entity(chunk_id, 1, "method", text, "DPR"),
        _make_entity(chunk_id, 2, "method", text, "BM25"),
        _make_entity(chunk_id, 3, "dataset", text, "SQuAD"),
        _make_entity(chunk_id, 4, "metric", text, "F1"),
    ]
    relations = [
        {
            "relation_id": "gr_01",
            "relation_type": "EVALUATES_ON",
            "subject_ref": {"kind": "mention", "id": entities[0]["entity_id"]},
            "object_ref": {"kind": "mention", "id": entities[2]["entity_id"]},
            "evidence_quote": "DPR as a method that may improve over BM25",
            "evidence_span": {"char_start": 10, "char_end": 47},
            "endpoint_mentions": [],
            "confidence": "inferred",
        }
    ]
    sample_row, record = _make_record(
        chunk_id, _doc_id(), text,
        entities=entities, relations=relations,
        paper_tier="mid_density",
    )
    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record})
    difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    rows = [json.loads(line) for line in paths["output_jsonl"].read_text().splitlines() if line]
    assert rows[0]["difficulty"] == "hard"
    assert "inferred_or_hedged_relation" in rows[0]["triggers"]["hard_triggers"]


def test_hard_chunk_sentence_boundary_heuristic(tmp_path):
    text = TEXT_HARD_BOUNDARY
    chunk_id = _record_id("hard_boundary")
    entities = [
        _make_entity(chunk_id, 1, "method", text, "DPR"),
        _make_entity(chunk_id, 2, "dataset", text, "SQuAD"),
        _make_entity(chunk_id, 3, "metric", text, "F1"),
        _make_entity(chunk_id, 4, "metric", text, "EM"),
    ]
    relations = [
        {
            "relation_id": "gr_01",
            "relation_type": "EVALUATES_ON",
            "subject_ref": {"kind": "mention", "id": entities[0]["entity_id"]},
            "object_ref": {"kind": "mention", "id": entities[1]["entity_id"]},
            "evidence_quote": "DPR is a retrieval method. We evaluate it on the SQuAD benchmark",
            "evidence_span": {"char_start": 0, "char_end": 70},
            "endpoint_mentions": [],
            "confidence": "stated",
        }
    ]
    sample_row, record = _make_record(
        chunk_id, _doc_id(), text,
        entities=entities, relations=relations,
        paper_tier="benchmark_central",
    )
    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record})
    difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    rows = [json.loads(line) for line in paths["output_jsonl"].read_text().splitlines() if line]
    assert rows[0]["difficulty"] == "hard"
    assert "evidence_spans_sentence_boundary" in rows[0]["triggers"]["hard_triggers"]


def test_hard_chunk_common_noun_method_endpoint(tmp_path):
    """Spec gap 4: common-noun method endpoint triggers hard."""
    text = TEXT_COMMON_NOUN_ENDPOINT
    chunk_id = _record_id("hard_common_noun_endpoint")
    entities = [
        _make_entity(chunk_id, 1, "method", text, "attention"),
        _make_entity(chunk_id, 2, "method", text, "DPR"),
        _make_entity(chunk_id, 3, "dataset", text, "SQuAD"),
        _make_entity(chunk_id, 4, "metric", text, "F1"),
        _make_entity(chunk_id, 5, "metric", text, "EM"),
    ]
    relations = [
        {
            "relation_id": "gr_01",
            "relation_type": "USES_METHOD",
            "subject_ref": {"kind": "mention", "id": entities[1]["entity_id"]},
            "object_ref": {"kind": "mention", "id": entities[0]["entity_id"]},
            "evidence_quote": "attention as a method to improve DPR",
            "evidence_span": {"char_start": 14, "char_end": 44},
            "endpoint_mentions": [],
            "confidence": "stated",
        }
    ]
    sample_row, record = _make_record(
        chunk_id, _doc_id(), text,
        entities=entities, relations=relations,
        paper_tier="mid_density",
    )
    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record})
    difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    rows = [json.loads(line) for line in paths["output_jsonl"].read_text().splitlines() if line]
    assert rows[0]["difficulty"] == "hard"
    assert "common_noun_method_endpoint" in rows[0]["triggers"]["hard_triggers"]


def test_hard_chunk_multi_occurrence_disambiguation(tmp_path):
    """``model`` appears >= 2 times AND is the surface form of an entity referenced
    by a relation's endpoint mention → hard trigger."""
    text = TEXT_HARD_MULTI
    chunk_id = _record_id("hard_multi_occ")
    # model appears 4 times; both occurrences are spelled identically.
    entities = [
        _make_entity(chunk_id, 1, "method", text, "model"),
        _make_entity(chunk_id, 2, "method", text, "BM25"),
        _make_entity(chunk_id, 3, "metric", text, "F1"),
        _make_entity(chunk_id, 4, "method", text, "DPR"),
    ]
    relations = [
        {
            "relation_id": "gr_01",
            "relation_type": "USES_METHOD",
            "subject_ref": {"kind": "document", "id": _doc_id()},
            "object_ref": {"kind": "mention", "id": entities[0]["entity_id"]},
            "evidence_quote": "model A and model B",
            "evidence_span": {"char_start": 10, "char_end": 28},
            "endpoint_mentions": [],
            "confidence": "stated",
        }
    ]
    sample_row, record = _make_record(
        chunk_id, _doc_id(), text,
        entities=entities, relations=relations,
        paper_tier="mid_density",
    )
    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record})
    difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    rows = [json.loads(line) for line in paths["output_jsonl"].read_text().splitlines() if line]
    assert rows[0]["difficulty"] == "hard"
    assert "multi_occurrence_disambiguation" in rows[0]["triggers"]["hard_triggers"]


def test_medium_chunk_missing_easy_condition(tmp_path):
    """All four easy conditions must hold for ``easy``. Here ``< 2`` unique
    surface forms trips one condition → medium."""
    text = "DPR achieves F1 of 41.5 on SQuAD."
    chunk_id = _record_id("medium_too_few")
    entities = [
        _make_entity(chunk_id, 1, "method", text, "DPR"),
        _make_entity(chunk_id, 2, "metric", text, "F1"),
        _make_entity(chunk_id, 3, "dataset", text, "SQuAD"),
    ]
    # Use a relation between document and single mention so the
    # at-least-one-mention-mention-relation easy condition fails.
    relations = [
        {
            "relation_id": "gr_01",
            "subject_ref": {"kind": "document", "id": _doc_id()},
            "object_ref": {"kind": "mention", "id": entities[0]["entity_id"]},
            "evidence_quote": "DPR achieves F1",
            "evidence_span": {"char_start": 0, "char_end": 15},
            "endpoint_mentions": [],
            "confidence": "stated",
        }
    ]
    sample_row, record = _make_record(
        chunk_id, _doc_id(), text,
        entities=entities, relations=relations,
        paper_tier="benchmark_central",
    )
    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record})
    difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    rows = [json.loads(line) for line in paths["output_jsonl"].read_text().splitlines() if line]
    assert rows[0]["difficulty"] == "medium"
    assert rows[0]["triggers"]["easy_conditions"]["at_least_one_mention_mention_relation"] is False


def test_empty_chunk_is_medium(tmp_path):
    """A chunk with no entities and no relations has neither easy nor hard triggers
    from the Spec (paper tier may still trip hard). With benchmark_central tier and
    empty record, it is medium."""
    text = "This chunk mentions no admissible method, dataset, metric, or task."
    chunk_id = _record_id("medium_empty")
    sample_row, record = _make_record(
        chunk_id, _doc_id(), text,
        entities=[], relations=[],
        paper_tier="benchmark_central",
        note="no admissible entity",
    )
    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record})
    difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    rows = [json.loads(line) for line in paths["output_jsonl"].read_text().splitlines() if line]
    assert rows[0]["difficulty"] == "medium"


def test_empty_long_tail_chunk_is_hard(tmp_path):
    """Empty chunk from a long_tail paper is hard (paper tier alone trips hard)."""
    text = "Empty chunk."
    chunk_id = _record_id("hard_empty_long_tail")
    sample_row, record = _make_record(
        chunk_id, _doc_id(), text,
        entities=[], relations=[],
        paper_tier="long_tail",
        note="no admissible entity",
    )
    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record})
    difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    rows = [json.loads(line) for line in paths["output_jsonl"].read_text().splitlines() if line]
    assert rows[0]["difficulty"] == "hard"
    assert "long_tail_paper" in rows[0]["triggers"]["hard_triggers"]


# ---------------------------------------------------------------------------
# Reproducibility / versioning / audit.
# ---------------------------------------------------------------------------


def test_byte_identical_reruns(tmp_path):
    text = TEXT_EASY
    chunk_id = _record_id("repro")
    entities = [
        _make_entity(chunk_id, 1, "method", text, "DPR"),
        _make_entity(chunk_id, 2, "method", text, "BM25"),
        _make_entity(chunk_id, 3, "dataset", text, "SQuAD"),
        _make_entity(chunk_id, 4, "metric", text, "F1"),
    ]
    relations = [
        {
            "relation_id": "gr_01",
            "relation_type": "EVALUATES_ON",
            "subject_ref": {"kind": "mention", "id": entities[0]["entity_id"]},
            "object_ref": {"kind": "mention", "id": entities[2]["entity_id"]},
            "evidence_quote": "DPR as a retrieval method",
            "evidence_span": {"char_start": 10, "char_end": 35},
            "endpoint_mentions": [],
            "confidence": "stated",
        }
    ]
    sample_row, record = _make_record(
        chunk_id, _doc_id(), text,
        entities=entities, relations=relations,
        paper_tier="benchmark_central",
    )
    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record})
    difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    first_jsonl = paths["output_jsonl"].read_bytes()
    first_audit = paths["output_audit"].read_bytes()
    difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    second_jsonl = paths["output_jsonl"].read_bytes()
    second_audit = paths["output_audit"].read_bytes()
    assert first_jsonl == second_jsonl
    assert first_audit == second_audit


def test_gold_record_sha256_excludes_note(tmp_path):
    """A change to the diagnostic ``note`` field MUST NOT change the difficulty
    classification or the gold_record_sha256 pin (per #48 Refinement 9)."""
    text = TEXT_EASY
    chunk_id = _record_id("sha_no_note")
    entities = [
        _make_entity(chunk_id, 1, "method", text, "DPR"),
        _make_entity(chunk_id, 2, "method", text, "BM25"),
        _make_entity(chunk_id, 3, "dataset", text, "SQuAD"),
        _make_entity(chunk_id, 4, "metric", text, "F1"),
    ]
    relations = [
        {
            "relation_id": "gr_01",
            "relation_type": "EVALUATES_ON",
            "subject_ref": {"kind": "mention", "id": entities[0]["entity_id"]},
            "object_ref": {"kind": "mention", "id": entities[2]["entity_id"]},
            "evidence_quote": "DPR as a retrieval method",
            "evidence_span": {"char_start": 10, "char_end": 35},
            "endpoint_mentions": [],
            "confidence": "stated",
        }
    ]
    sample_row, record_a = _make_record(
        chunk_id, _doc_id(), text,
        entities=entities, relations=relations,
        paper_tier="benchmark_central",
        note="first version",
    )
    sample_row, record_b = _make_record(
        chunk_id, _doc_id(), text,
        entities=entities, relations=relations,
        paper_tier="benchmark_central",
        note="second version, longer explanation",
    )

    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record_a})
    difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    first_pin = json.loads(paths["output_jsonl"].read_text().splitlines()[0])["gold_record_sha256"]

    # Rewrite the record with a different note, re-run.
    (paths["gold_dir"] / f"{chunk_id}.json").write_text(
        json.dumps(record_b, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    second_pin = json.loads(paths["output_jsonl"].read_text().splitlines()[0])["gold_record_sha256"]
    assert first_pin == second_pin


def test_audit_records_script_sha_and_version(tmp_path):
    text = TEXT_EASY
    chunk_id = _record_id("audit_fields")
    entities = [
        _make_entity(chunk_id, 1, "method", text, "DPR"),
        _make_entity(chunk_id, 2, "dataset", text, "SQuAD"),
    ]
    sample_row, record = _make_record(
        chunk_id, _doc_id(), text,
        entities=entities, relations=[],
        paper_tier="benchmark_central",
    )
    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record})
    audit = difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    assert audit["classifier_id"] == difficulty.CLASSIFIER_ID
    assert audit["classifier_version"] == difficulty.CLASSIFIER_VERSION
    assert audit["classifier_script_sha256"] == hashlib.sha256(
        DIFFICULTY.read_bytes()
    ).hexdigest()
    assert audit["difficulty_counts"]["medium"]["count"] == 1
    assert audit["spec_gaps_observed"]  # non-empty list
    assert any(
        "common_noun" in gap for gap in audit["spec_gaps_observed"]
    )


def test_audit_marks_contract_authoritative_when_gaps_closed(tmp_path):
    """Refinement 10 closes the five v1 gaps without changing the heuristic.

    The classifier may run authoritatively; selection-bound freezing remains
    gated on the 7-day wash and ann-B / ann-C adjudication.
    """
    text = TEXT_EASY
    chunk_id = _record_id("audit_contract")
    entities = [
        _make_entity(chunk_id, 1, "method", text, "DPR"),
        _make_entity(chunk_id, 2, "dataset", text, "SQuAD"),
    ]
    sample_row, record = _make_record(
        chunk_id, _doc_id(), text,
        entities=entities, relations=[],
        paper_tier="benchmark_central",
    )
    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record})
    audit = difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    assert difficulty.SPEC_GAPS_UNRESOLVED == frozenset()
    assert audit["spec_gaps_unresolved"] == []
    assert audit["spec_gaps_unresolved_count"] == 0
    assert audit["contract_status"] == difficulty.CONTRACT_AUTHORITATIVE
    assert audit["contract_blocker_issue"] is None
    assert audit["contract_authority_ref"] == (
        "ImmoP/knowledge_graph_rag#48 (Refinement 10, issuecomment-5783801175)"
    )


def test_fail_closed_guard_fires_under_synthetic_gap(monkeypatch, tmp_path, capsys):
    """A future reopened gap still makes ``--require-authoritative`` exit 2."""
    text = TEXT_EASY
    chunk_id = _record_id("audit_guard")
    entities = [
        _make_entity(chunk_id, 1, "method", text, "DPR"),
        _make_entity(chunk_id, 2, "dataset", text, "SQuAD"),
    ]
    sample_row, record = _make_record(
        chunk_id, _doc_id(), text,
        entities=entities, relations=[],
        paper_tier="benchmark_central",
    )
    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record})
    monkeypatch.setattr(difficulty, "SPEC_GAPS_UNRESOLVED", frozenset({"synthetic_gap"}))
    rc = difficulty.main([
        "--gold-dir", str(paths["gold_dir"]),
        "--sample", str(paths["sample"]),
        "--sample-audit", str(paths["sample_audit"]),
        "--output-jsonl", str(paths["output_jsonl"]),
        "--output-audit", str(paths["output_audit"]),
        "--quiet",
        "--require-authoritative",
    ])
    assert rc == 2
    captured = capsys.readouterr()
    assert "non_authoritative_pending_resolution" in captured.err
    assert "ImmoP/knowledge_graph_rag#99" in captured.err


def test_require_authoritative_flag_passes_when_contract_authoritative(tmp_path, capsys):
    """The closed v1 contract makes ``--require-authoritative`` exit 0."""
    text = TEXT_EASY
    chunk_id = _record_id("audit_guard_pass")
    entities = [
        _make_entity(chunk_id, 1, "method", text, "DPR"),
        _make_entity(chunk_id, 2, "dataset", text, "SQuAD"),
    ]
    sample_row, record = _make_record(
        chunk_id, _doc_id(), text,
        entities=entities, relations=[],
        paper_tier="benchmark_central",
    )
    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record})
    rc = difficulty.main([
        "--gold-dir", str(paths["gold_dir"]),
        "--sample", str(paths["sample"]),
        "--sample-audit", str(paths["sample_audit"]),
        "--output-jsonl", str(paths["output_jsonl"]),
        "--output-audit", str(paths["output_audit"]),
        "--quiet",
        "--require-authoritative",
    ])
    assert rc == 0
    assert capsys.readouterr().err == ""

# ---------------------------------------------------------------------------
# Fail-closed behavior.
# ---------------------------------------------------------------------------


def test_missing_sample_row_is_recorded_as_skipped(tmp_path):
    """A gold record whose chunk_id is not in the sample is skipped (not crashing)."""
    text = TEXT_EASY
    chunk_id = _record_id("missing_sample")
    sample_chunk_id = _record_id("missing_sample", suffix="02")
    entities = [
        _make_entity(chunk_id, 1, "method", text, "DPR"),
        _make_entity(chunk_id, 2, "dataset", text, "SQuAD"),
    ]
    sample_row = {
        "chunk_id": sample_chunk_id,
        "doc_id": _doc_id(),
        "sampler_version": "gold-set-sample.v2",
        "sampler_seed": "9e3779b1",
        "strata": {
            "paper_tier": "benchmark_central",
            "section_depth": 1,
            "length_tercile": "medium",
            "density_post_stratum": "non_empty",
        },
    }
    record = {
        "schema_version": "gold-record.v1",
        "chunk_id": chunk_id,
        "doc_id": _doc_id(),
        "canonical_chunks_sha256_at_authoring": "fake",
        "ontology_version": 1,
        "ontology_sha256": "fake",
        "annotation_guideline_version": "1",
        "annotation_guideline_sha256": "fake",
        "annotator_id": "ann-A",
        "adjudication_status": "single_pass",
        "adjudicated_by": None,
        "annotated_at": "2026-09-22T00:00:00Z",
        "entities": entities,
        "relations": [],
    }
    paths = _setup_paths(
        tmp_path,
        sample_rows=[sample_row],
        records={chunk_id: record},
    )
    audit = difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    assert audit["chunk_count_skipped"] == 1
    assert audit["skipped"] == [{"chunk_id": chunk_id, "reason": "missing_from_sample"}]
    assert audit["chunk_count_classified"] == 0
    assert paths["output_jsonl"].read_text().strip() == ""


def test_malformed_sample_fails_closed(tmp_path):
    """A sample row missing ``strata.paper_tier`` raises DifficultyError."""
    sample_path = tmp_path / "sample.v2.jsonl"
    sample_path.write_text(
        json.dumps({"chunk_id": "c_xxx", "doc_id": "W1"}) + "\n",
        encoding="utf-8",
    )
    gold_dir = tmp_path / "chunks"
    gold_dir.mkdir()
    with pytest.raises(difficulty.DifficultyError):
        difficulty.run(
            gold_dir=gold_dir,
            sample_path=sample_path,
            output_jsonl=tmp_path / "out.jsonl",
            output_audit=tmp_path / "out.audit.json",
        )


# ---------------------------------------------------------------------------
# Spec gap behavior (NOT silently filled).
# ---------------------------------------------------------------------------


def test_common_noun_list_is_exactly_the_three_examples():
    """Spec gap 1: the v1 common-noun list is exactly the three examples from
    Refinement 5. ``attention``, ``model``, ``score`` trip the easy rule;
    any other noun (e.g. ``retrieval``) does not. This test pins the v1
    minimum; bumping COMMON_NOUNS requires bumping CLASSIFIER_VERSION.
    """
    assert difficulty.COMMON_NOUNS == frozenset({"attention", "model", "score"})


def test_easy_fails_when_surface_form_is_common_noun(tmp_path):
    """A chunk otherwise easy except for ``score`` as a surface form fails the
    ``no_common_noun_surface_form`` easy condition."""
    text = (
        "We propose DPR as a retrieval method that combines BM25 to answer "
        "questions on SQuAD. Experiments report score and recall."
    )
    chunk_id = _record_id("medium_common_noun")
    entities = [
        _make_entity(chunk_id, 1, "method", text, "DPR"),
        _make_entity(chunk_id, 2, "method", text, "BM25"),
        _make_entity(chunk_id, 3, "dataset", text, "SQuAD"),
        _make_entity(chunk_id, 4, "metric", text, "score"),
    ]
    relations = [
        {
            "relation_id": "gr_01",
            "relation_type": "EVALUATES_ON",
            "subject_ref": {"kind": "mention", "id": entities[0]["entity_id"]},
            "object_ref": {"kind": "mention", "id": entities[2]["entity_id"]},
            "evidence_quote": "DPR as a retrieval method",
            "evidence_span": {"char_start": 10, "char_end": 35},
            "endpoint_mentions": [],
            "confidence": "stated",
        }
    ]
    sample_row, record = _make_record(
        chunk_id, _doc_id(), text,
        entities=entities, relations=relations,
        paper_tier="benchmark_central",
    )
    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record})
    difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    rows = [json.loads(line) for line in paths["output_jsonl"].read_text().splitlines() if line]
    assert rows[0]["difficulty"] == "medium"
    assert rows[0]["triggers"]["easy_conditions"]["no_common_noun_surface_form"] is False


def test_hard_common_noun_endpoint_only_for_method(tmp_path):
    """Spec gap 4: the ``common_noun_method_endpoint`` trigger requires
    ``entity_type == "method"``. A dataset called ``score`` (entity_type
    ``dataset``) does NOT trip the trigger."""
    text = "We propose score as a dataset to evaluate DPR on SQuAD with F1."
    chunk_id = _record_id("dataset_common_noun")
    entities = [
        _make_entity(chunk_id, 1, "dataset", text, "score"),
        _make_entity(chunk_id, 2, "method", text, "DPR"),
        _make_entity(chunk_id, 3, "dataset", text, "SQuAD"),
        _make_entity(chunk_id, 4, "metric", text, "F1"),
    ]
    relations = [
        {
            "relation_id": "gr_01",
            "relation_type": "EVALUATES_ON",
            "subject_ref": {"kind": "document", "id": _doc_id()},
            "object_ref": {"kind": "mention", "id": entities[0]["entity_id"]},
            "evidence_quote": "We propose score",
            "evidence_span": {"char_start": 0, "char_end": 16},
            "endpoint_mentions": [],
            "confidence": "stated",
        }
    ]
    sample_row, record = _make_record(
        chunk_id, _doc_id(), text,
        entities=entities, relations=relations,
        paper_tier="benchmark_central",
    )
    paths = _setup_paths(tmp_path, sample_rows=[sample_row], records={chunk_id: record})
    difficulty.run(
        gold_dir=paths["gold_dir"],
        sample_path=paths["sample"],
        output_jsonl=paths["output_jsonl"],
        output_audit=paths["output_audit"],
        sample_audit_path=paths["sample_audit"],
    )
    rows = [json.loads(line) for line in paths["output_jsonl"].read_text().splitlines() if line]
    # No hard triggers (dataset endpoint does NOT trip the common-noun-method rule).
    assert "common_noun_method_endpoint" not in rows[0]["triggers"]["hard_triggers"]
    # No mention-mention relation either (subject is document), so easy fails.
    assert rows[0]["difficulty"] == "medium"