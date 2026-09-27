"""Synthetic acceptance tests for the ann-B review workflow (#95).

Tests use only ``tmp_path`` and the public author/review APIs so they can
run anywhere. The fixture builds a 120-chunk canonical sample + canonical
ann-A records backed by a minimal ontology/guideline/chunks layer.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKTREE = ROOT
REVIEW = WORKTREE / "scripts" / "gold_set" / "review.py"
AUTHOR = WORKTREE / "scripts" / "gold_set" / "author.py"


# Import the real author package so internal review.py imports resolve to the same
# module object (the test loader would otherwise register two copies of author and
# review.py's ``from scripts.gold_set import author`` would not see it).
import scripts.gold_set.author as _author_pkg  # noqa: E402
author = _author_pkg

# Replace the review module with the worktree copy so we exercise #95's code.
spec = importlib.util.spec_from_file_location("scripts.gold_set.review", REVIEW)
review = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = review
spec.loader.exec_module(review)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_TEXT = "The corpus contains several retrieval experiments with reproducible evaluation pipelines. Authors propose and extend methods across benchmark suites."


def _build_paths(tmp_path: Path, *, canonical_annotated_at: str = "2026-09-21T13:44:14.559542Z") -> tuple[author.Paths, str]:
    chunks_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    sample_path = tmp_path / "sample.jsonl"
    audit_path = tmp_path / "sample.v2.audit.json"
    leakage_path = tmp_path / "leakage.json"
    ontology_path = tmp_path / "ontology.yaml"
    guideline_path = tmp_path / "GUIDELINES.md"
    destination = tmp_path / "chunks_dir"
    destination.mkdir(parents=True, exist_ok=True)

    chunk_id = "c_0000000000000001"
    sample_rows = [{"chunk_id": chunk_id, "doc_id": "W1", "sampler_version": "gold-set-sample.v2", "sampler_seed": "9e3779b1"}]
    sample_rows += [
        {"chunk_id": f"c_{i:016x}", "doc_id": "W1", "sampler_version": "gold-set-sample.v2", "sampler_seed": "9e3779b1"}
        for i in range(2, 121)
    ]
    assert len(sample_rows) == 120
    chunk_rows = [{"chunk_id": row["chunk_id"], "doc_id": "W1", "section_path": ["s"], "chunk_index": 0, "text": _TEXT} for row in sample_rows]
    _jsonl(chunks_path, chunk_rows)
    _jsonl(manifest_path, [{"openalex_id": "W1"}])
    _jsonl(sample_path, sample_rows)
    ontology_path.write_text(
        """version: 1
relation_types:
  - {name: PROPOSES, from: document, to: method, population_mode: extracted}
  - {name: INTRODUCES, from: document, to: dataset, population_mode: extracted}
  - {name: EVALUATES_ON, from: document, to: dataset, population_mode: extracted}
  - {name: USES_METHOD, from: document, to: method, population_mode: extracted}
  - {name: REPORTS_METRIC, from: document, to: metric, population_mode: extracted}
  - {name: EXTENDS, from: method, to: method, population_mode: extracted}
  - {name: USED_FOR_TASK, from: dataset, to: task, population_mode: extracted}
  - {name: MEASURES, from: metric, to: task, population_mode: extracted}
  - {name: CITES, from: document, to: document, population_mode: structural}
  - {name: EVALUATED_ON, from: method, to: dataset, population_mode: derived}
""",
        encoding="utf-8",
    )
    guideline_path.write_text("annotation_guideline_version = 1\n", encoding="utf-8")
    _json(leakage_path, {"excluded_chunk_ids": []})
    _json(
        audit_path,
        {
            "selected_chunk_ids_sha256": hashlib.sha256("".join(sorted(row["chunk_id"] for row in sample_rows)).encode("utf-8")).hexdigest(),
            "chunks_sha256": _sha(chunks_path),
            "manifest_sha256": _sha(manifest_path),
            "ontology_sha256": _sha(ontology_path),
            "leakage_artifact_sha256": _sha(leakage_path),
            "leakage_overlap_count": 0,
        },
    )

    canonical = author.template(
        author.Paths(chunks_path, manifest_path, sample_path, audit_path, leakage_path, ontology_path, guideline_path, destination),
        chunk_id,
        "ann-A",
    )
    canonical["entities"] = [
        _entity(chunk_id, 1, "method", _TEXT, "retrieval"),
        _entity(chunk_id, 2, "dataset", _TEXT, "benchmark"),
        _entity(chunk_id, 3, "task", _TEXT, "experiments"),
        _entity(chunk_id, 4, "metric", _TEXT, "evaluation"),
    ]
    canonical["note"] = "non-empty"
    author.validate_record(canonical, author.Paths(chunks_path, manifest_path, sample_path, audit_path, leakage_path, ontology_path, guideline_path, destination))
    canonical["annotated_at"] = canonical_annotated_at
    canonical_bytes = (json.dumps(canonical, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    for row in sample_rows:
        (destination / f"{row['chunk_id']}.json").write_bytes(canonical_bytes)

    return (
        author.Paths(chunks_path, manifest_path, sample_path, audit_path, leakage_path, ontology_path, guideline_path, destination),
        chunk_id,
    )


def _entity(chunk_id: str, seq: int, entity_type: str, text: str, literal: str) -> dict:
    token = chunk_id[2:10]
    starts = author._enumerate_occurrences(text, literal)
    return {
        "entity_id": f"ge_{token}_{seq:02d}",
        "entity_type": entity_type,
        "surface_form": literal,
        "canonical_form": literal,
        "span": {"char_start": starts[0], "char_end": starts[0] + len(literal)},
        "occurrences_in_chunk": [
            {"occurrence_index": index, "char_start": start, "char_end": start + len(literal)}
            for index, start in enumerate(starts)
        ],
        "confidence": "stated",
    }


def _canonical_record(paths: author.Paths, chunk_id: str) -> dict:
    return json.loads((paths.destination_dir / f"{chunk_id}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Frozen sample
# ---------------------------------------------------------------------------


def test_frozen_sample_deterministic_and_byte_identical(tmp_path):
    paths, _ = _build_paths(tmp_path)
    first = review.freeze_ann_b_sample(paths)
    second = review.freeze_ann_b_sample(paths)
    assert first == second
    assert len(first.selected) == 24
    assert first.selected_set == set(first.selected)
    digest_input = (
        review.ANN_B_DIGEST_NS + b"\x00" + review.ANN_B_SEED_HEX.encode("utf-8") + b"\x00" + first.selected[0].encode("utf-8")
    )
    expected_first_digest = hashlib.sha256(digest_input).hexdigest()
    sample_ids = review._sample_chunk_ids(paths)
    expected_first = min(
        ((hashlib.sha256(review.ANN_B_DIGEST_NS + b"\x00" + review.ANN_B_SEED_HEX.encode("utf-8") + b"\x00" + cid.encode("utf-8")).hexdigest(), cid) for cid in sample_ids),
        key=lambda item: (item[0], item[1]),
    )
    assert first.selected[0] == expected_first[1]
    assert expected_first_digest == expected_first[0]
    expected_fp = hashlib.sha256("".join(sorted(first.selected)).encode("utf-8")).hexdigest()
    assert first.fingerprint == expected_fp


def test_publish_audit_write_once_and_preflight(tmp_path):
    paths, _ = _build_paths(tmp_path)
    target = tmp_path / "audit.json"
    frozen = review.publish_ann_b_audit(paths, target=target)
    assert target.exists()
    body = target.read_bytes()
    review.publish_ann_b_audit(paths, target=target)
    assert target.read_bytes() == body
    target.write_bytes(b"different")
    with pytest.raises(review.ReviewError, match="different bytes"):
        review.publish_ann_b_audit(paths, target=target)


def test_load_audit_recomputes_and_rejects_divergence(tmp_path):
    paths, _ = _build_paths(tmp_path)
    target = tmp_path / "audit.json"
    review.publish_ann_b_audit(paths, target=target)
    loaded = review.load_ann_b_audit(target, paths)
    assert loaded.selected == review.freeze_ann_b_sample(paths).selected
    raw = json.loads(target.read_text(encoding="utf-8"))
    # Pick a sample chunk_id that is not currently selected and swap it in
    # while keeping all entries distinct. The recompute check then fails.
    sample_ids = review._sample_chunk_ids(paths)
    selected_set = set(raw["selected_chunk_ids"])
    replacement = next(cid for cid in sample_ids if cid not in selected_set)
    raw["selected_chunk_ids"][0] = replacement
    target.write_text(json.dumps(raw, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(review.ReviewError, match="fingerprint mismatch|recomputed"):
        review.load_ann_b_audit(target, paths)


# ---------------------------------------------------------------------------
# Wash guard
# ---------------------------------------------------------------------------


def test_wash_guard_lt_seven_days_fails(tmp_path):
    paths, _ = _build_paths(tmp_path)
    canonical = _canonical_record(paths, "c_0000000000000001")
    max_at = _dt.datetime.fromisoformat(canonical["annotated_at"].replace("Z", "+00:00"))
    just_before = max_at + _dt.timedelta(days=7) - _dt.timedelta(seconds=1)
    guard = review.evaluate_wash_guard(paths, just_before.astimezone(_dt.timezone.utc))
    assert not guard.passed
    assert guard.boundary == max_at + _dt.timedelta(days=7)


def test_wash_guard_eq_seven_days_passes(tmp_path):
    paths, _ = _build_paths(tmp_path)
    canonical = _canonical_record(paths, "c_0000000000000001")
    max_at = _dt.datetime.fromisoformat(canonical["annotated_at"].replace("Z", "+00:00"))
    exactly = max_at + _dt.timedelta(days=7)
    guard = review.evaluate_wash_guard(paths, exactly.astimezone(_dt.timezone.utc))
    assert guard.passed


def test_wash_guard_gt_seven_days_passes(tmp_path):
    paths, _ = _build_paths(tmp_path)
    canonical = _canonical_record(paths, "c_0000000000000001")
    max_at = _dt.datetime.fromisoformat(canonical["annotated_at"].replace("Z", "+00:00"))
    after = max_at + _dt.timedelta(days=8)
    guard = review.evaluate_wash_guard(paths, after.astimezone(_dt.timezone.utc))
    assert guard.passed


def test_create_self_review_wash_fail_closed(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    canonical = _canonical_record(paths, chunk_id)
    max_at = _dt.datetime.fromisoformat(canonical["annotated_at"].replace("Z", "+00:00"))
    before = max_at + _dt.timedelta(days=2)
    reviews_root = tmp_path / "gold_set" / "reviews"
    out = reviews_root / f"{chunk_id}.json"
    with pytest.raises(review.ReviewError, match="wash not satisfied"):
        review.create_self_review_draft(paths, chunk_id, out, now=before.astimezone(_dt.timezone.utc))
    assert not out.exists()


# ---------------------------------------------------------------------------
# Self-review templates / show
# ---------------------------------------------------------------------------


def test_self_template_does_not_copy_canonical_labels_or_note(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    canonical = _canonical_record(paths, chunk_id)
    template = review.self_review_template(paths, chunk_id)
    assert template["entities"] == []
    assert template["relations"] == []
    assert template["note"] is None
    assert template.get("annotated_at") is None
    assert template["annotator_id"] == review.ANN_A_SELF_ANNOTATOR
    assert template["adjudication_status"] == review.SINGLE_PASS_STATUS
    for entity in canonical["entities"]:
        for key in ("entity_type", "surface_form", "canonical_form", "confidence"):
            assert entity[key] not in json.dumps(template["entities"])
    assert "note" in template or template.get("note") is None


def test_show_inputs_excludes_canonical_labels(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    payload = review.show_self_review_inputs(paths, chunk_id)
    assert payload["chunk_id"] == chunk_id
    # The sidecar must not serialise canonical labels as structured fields.
    assert "entities" not in payload
    assert "relations" not in payload
    assert "note" not in payload
    canonical = _canonical_record(paths, chunk_id)
    for entity in canonical["entities"]:
        # Surface forms may legitimately appear inside chunk text; the
        # requirement is that the self-review rendering does NOT carry
        # them as a separate labelled record alongside the chunk text.
        assert "entity_type" not in json.dumps(payload.get("text", ""))


# ---------------------------------------------------------------------------
# Self comparison
# ---------------------------------------------------------------------------


def _build_self_record(paths: author.Paths, chunk_id: str, *, mutate_confidence: str | None = None) -> dict:
    canonical = _canonical_record(paths, chunk_id)
    record = json.loads(json.dumps(canonical))
    record["annotator_id"] = review.ANN_A_SELF_ANNOTATOR
    record["adjudication_status"] = review.SINGLE_PASS_STATUS
    record["annotated_at"] = "2026-09-30T12:00:00Z"
    record["entities"] = list(canonical["entities"])
    record["relations"] = list(canonical["relations"])
    if mutate_confidence is not None:
        record["entities"][0]["confidence"] = mutate_confidence
    return record


def test_compare_self_review_perfect_match(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    self_record = _build_self_record(paths, chunk_id)
    comparison = review.compare_self_review(paths, chunk_id, self_record)
    assert comparison["agreement"] is True
    assert comparison["disagreements"] == []


def test_compare_self_review_reports_exact_cells(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    self_record = _build_self_record(paths, chunk_id, mutate_confidence="inferred")
    comparison = review.compare_self_review(paths, chunk_id, self_record)
    kinds = {item["kind"] for item in comparison["disagreements"]}
    assert "entity_confidence" in kinds


def test_compare_self_review_preserves_multiplicity(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    self_record = _build_self_record(paths, chunk_id)
    original = self_record["entities"][1]
    extra = json.loads(json.dumps(original))
    extra["entity_id"] = f"ge_{chunk_id[2:10]}_99"
    self_record["entities"].append(extra)
    comparison = review.compare_self_review(paths, chunk_id, self_record)
    kinds = {item["kind"] for item in comparison["disagreements"]}
    assert "entity_count" in kinds


def test_compare_self_review_refuses_naive_timestamp(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    self_record = _build_self_record(paths, chunk_id)
    self_record["annotated_at"] = "2026-09-30T12:00:00"
    with pytest.raises(review.ReviewError, match="naive"):
        review.compare_self_review(paths, chunk_id, self_record)


def test_compare_self_review_refuses_wrong_actor(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    self_record = _build_self_record(paths, chunk_id)
    self_record["annotator_id"] = review.ANN_A_ANNOTATOR
    with pytest.raises(review.ReviewError, match="annotator_id"):
        review.compare_self_review(paths, chunk_id, self_record)


# ---------------------------------------------------------------------------
# ann-B union status
# ---------------------------------------------------------------------------


def test_ann_b_status_union_and_membership(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    frozen = review.freeze_ann_b_sample(paths)
    root = tmp_path
    (root / "gold_set" / "reviews").mkdir(parents=True)
    self_record = _build_self_record(paths, chunk_id, mutate_confidence="inferred")
    reviews_root = root / "gold_set" / "reviews"
    (reviews_root / f"{chunk_id}.json").write_text(json.dumps(self_record, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    status = review.ann_b_status(paths, frozen, root=root)
    assert status["frozen_sample_size"] == 24
    assert chunk_id in status["self_disagreement_chunk_ids"]
    assert chunk_id in status["ann_b_union_chunk_ids"]
    assert status["membership"][chunk_id] == "self_disagreement"
    for frozen_chunk in frozen.selected:
        assert status["membership"][frozen_chunk] == "frozen_random"


# ---------------------------------------------------------------------------
# Sidecar (ann-B)
# ---------------------------------------------------------------------------


def test_ann_b_template_does_not_copy_canonical_labels(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    frozen = review.freeze_ann_b_sample(paths)
    target = tmp_path / "reviews"
    target.mkdir()
    frozen_chunk = frozen.selected[0]
    template = review.ann_b_template(paths, frozen_chunk, frozen, root=target)
    canonical = _canonical_record(paths, frozen_chunk)
    assert template["schema_version"] == review.SIDECAR_SCHEMA_VERSION
    assert template["ann_b_review"]["annotator_id"] == review.ANN_B_ANNOTATOR
    assert template["ann_b_review"]["adjudication_status"] == review.DOUBLE_PASS_STATUS
    assert template["ann_b_review"]["entities"] == []
    assert template["ann_b_review"]["relations"] == []
    assert template["ann_b_review"]["note"] is None
    for entity in canonical["entities"]:
        for key in ("entity_type", "surface_form", "canonical_form", "confidence"):
            assert entity[key] not in json.dumps(template["ann_b_review"]["entities"])


def test_complete_ann_b_rejects_wrong_status(tmp_path):
    paths, _ = _build_paths(tmp_path)
    frozen = review.freeze_ann_b_sample(paths)
    root = tmp_path
    (root / "gold_set" / "reviews").mkdir(parents=True)
    frozen_chunk = frozen.selected[0]
    bad = {
        "annotator_id": review.ANN_B_ANNOTATOR,
        "adjudication_status": "review",
        "annotated_at": "2026-09-30T12:00:00Z",
        "entities": [],
        "relations": [],
        "note": "",
    }
    template_record = author.template(paths, frozen_chunk, "ann-B")
    for key in ("schema_version", "chunk_id", "doc_id", "canonical_chunks_sha256_at_authoring", "ontology_version", "ontology_sha256", "annotation_guideline_version", "annotation_guideline_sha256"):
        bad[key] = template_record[key]
    with pytest.raises(review.ReviewError, match="double_pass"):
        review.complete_ann_b_review(paths, frozen_chunk, bad, frozen, root=root)


def test_complete_ann_b_rejects_naive_timestamp(tmp_path):
    paths, _ = _build_paths(tmp_path)
    frozen = review.freeze_ann_b_sample(paths)
    root = tmp_path
    (root / "gold_set" / "reviews").mkdir(parents=True)
    frozen_chunk = frozen.selected[0]
    bad = {
        "annotator_id": review.ANN_B_ANNOTATOR,
        "adjudication_status": review.DOUBLE_PASS_STATUS,
        "annotated_at": "2026-09-30T12:00:00",
        "entities": [],
        "relations": [],
        "note": "",
    }
    with pytest.raises(review.ReviewError, match="naive"):
        review.complete_ann_b_review(paths, frozen_chunk, bad, frozen, root=root)


def test_complete_ann_b_refuses_canonical_destination(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    frozen = review.freeze_ann_b_sample(paths)
    target = paths.destination_dir / f"{chunk_id}.json"
    payload = {
        "annotator_id": review.ANN_B_ANNOTATOR,
        "adjudication_status": review.DOUBLE_PASS_STATUS,
        "annotated_at": "2026-09-30T12:00:00Z",
        "entities": [],
        "relations": [],
        "note": "",
    }
    with pytest.raises((review.ReviewError, author.AuthorError)):
        review.complete_ann_b_review(paths, chunk_id, payload, frozen, root=tmp_path)


def test_complete_ann_b_atomic_and_refuses_existing(tmp_path):
    paths, _ = _build_paths(tmp_path)
    frozen = review.freeze_ann_b_sample(paths)
    root = tmp_path
    (root / "gold_set" / "reviews").mkdir(parents=True)
    frozen_chunk = frozen.selected[0]
    payload = {
        "annotator_id": review.ANN_B_ANNOTATOR,
        "adjudication_status": review.DOUBLE_PASS_STATUS,
        "annotated_at": "2026-09-30T12:00:00Z",
        "entities": [],
        "relations": [],
        "note": "",
    }
    sidecar = root / "gold_set" / "reviews" / f"{frozen_chunk}.json"
    review.complete_ann_b_review(paths, frozen_chunk, payload, frozen, root=root)
    assert sidecar.exists()
    with pytest.raises(review.ReviewError, match="already exists"):
        review.complete_ann_b_review(paths, frozen_chunk, payload, frozen, root=root)


# ---------------------------------------------------------------------------
# Cohen's κ
# ---------------------------------------------------------------------------


def test_kappa_perfect_agreement(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    frozen = review.freeze_ann_b_sample(paths)
    canonical = _canonical_record(paths, chunk_id)
    observer_b = [json.loads(json.dumps(canonical)) for cid in frozen.selected]
    for record, cid in zip(observer_b, frozen.selected):
        record["chunk_id"] = cid
        record["annotator_id"] = review.ANN_B_ANNOTATOR
        record["adjudication_status"] = review.DOUBLE_PASS_STATUS
        record["annotated_at"] = "2026-09-30T12:00:00Z"
    observer_b_path = tmp_path / "observer_b.jsonl"
    observer_b_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in observer_b) + "\n",
        encoding="utf-8",
    )
    result = review.compute_kappa(paths, frozen, observer_b_path)
    for family, cells in result["families"].items():
        for cell in cells:
            if cell["status"] == "undefined":
                # Same-uniform cells: kappa is undefined per contract; raw_kappa may be None.
                continue
            assert cell["raw_kappa"] == 1.0
            assert cell["status"] == "pass"


def test_kappa_perfect_disagreement(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    frozen = review.freeze_ann_b_sample(paths)
    observer_b: list[dict] = []
    half = len(frozen.selected) // 2
    for index, cid in enumerate(frozen.selected):
        canonical = _canonical_record(paths, cid)
        row = json.loads(json.dumps(canonical))
        row["chunk_id"] = cid
        row["annotator_id"] = review.ANN_B_ANNOTATOR
        row["adjudication_status"] = review.DOUBLE_PASS_STATUS
        row["annotated_at"] = "2026-09-30T12:00:00Z"
        if index < half:
            row["entities"] = []
        else:
            row["relations"] = []
        observer_b.append(row)
    observer_b_path = tmp_path / "observer_b.jsonl"
    observer_b_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in observer_b) + "\n",
        encoding="utf-8",
    )
    result = review.compute_kappa(paths, frozen, observer_b_path)
    saw_disagreement = False
    for family, cells in result["families"].items():
        for cell in cells:
            if cell["support_a"] == cell["support_b"] == 0:
                continue
            if cell["status"] == "escalate":
                saw_disagreement = True
                assert cell["raw_kappa"] < 0.70
                assert isinstance(cell["disagreement_chunk_ids"], list)
    assert saw_disagreement, "expected at least one escalate cell"


def test_kappa_undefined_when_same_uniform(tmp_path):
    a = [1, 1, 1, 1]
    b = [1, 1, 1, 1]
    kappa, summary = review._binary_kappa(a, b)
    assert kappa is None


def test_kappa_undefined_when_both_zero(tmp_path):
    a = [0, 0, 0, 0]
    b = [0, 0, 0, 0]
    kappa, summary = review._binary_kappa(a, b)
    assert kappa is None


def test_kappa_distinct_uniform_yields_zero(tmp_path):
    a = [1, 1, 1, 1]
    b = [0, 0, 0, 0]
    kappa, summary = review._binary_kappa(a, b)
    assert kappa == 0.0
    assert summary["n"] == 4
    assert summary["support_a"] == 4
    assert summary["support_b"] == 0


def test_kappa_observations_scoped_to_concrete_cell(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    frozen = review.freeze_ann_b_sample(paths)
    canonical = _canonical_record(paths, chunk_id)
    obs_a = review._cell_observations([canonical], "entity_type_confidence", ("method", "stated"))
    assert obs_a == [1]
    obs_other = review._cell_observations([canonical], "entity_type_confidence", ("metric", "inferred"))
    assert obs_other == [0]


# ---------------------------------------------------------------------------
# ann-C handoff
# ---------------------------------------------------------------------------


def test_ann_c_handoff_is_deterministic_and_report_only(tmp_path):
    paths, _ = _build_paths(tmp_path)
    frozen = review.freeze_ann_b_sample(paths)
    chunk_id = frozen.selected[0]
    canonical = json.loads((paths.destination_dir / f"{chunk_id}.json").read_text(encoding="utf-8"))
    root = tmp_path
    (root / "gold_set" / "reviews").mkdir(parents=True)
    self_record = _build_self_record(paths, chunk_id, mutate_confidence="inferred")
    sidecar_payload = {
        "schema_version": review.SIDECAR_SCHEMA_VERSION,
        "chunk_id": chunk_id,
        "frozen_sample_membership": "frozen_random",
        "ann_a_self_record": self_record,
        "self_disagreements": [],
        "ann_b_review": None,
        "ann_a_record": canonical,
        "ann_b_record": None,
        "canonical_annotated_at": canonical.get("annotated_at"),
    }
    (root / "gold_set" / "reviews" / f"{chunk_id}.json").write_text(json.dumps(sidecar_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    a = review.ann_c_handoff(paths, frozen, root=root)
    b = review.ann_c_handoff(paths, frozen, root=root)
    assert a == b
    assert a["invariant"] == "canonical ann-A records are not mutated by the handoff"
    assert chunk_id in a["chunks"]
    assert a["sidecars"][chunk_id]["ann_a_record"]["annotator_id"] == review.ANN_A_ANNOTATOR
    assert a["sidecars"][chunk_id]["ann_a_self_record"]["annotator_id"] == review.ANN_A_SELF_ANNOTATOR


def test_ann_c_handoff_includes_kappa_failing_chunks(tmp_path):
    paths, _ = _build_paths(tmp_path)
    frozen = review.freeze_ann_b_sample(paths)
    chunk_id = frozen.selected[0]
    root = tmp_path
    (root / "gold_set" / "reviews").mkdir(parents=True)
    kappa_path = root / "gold_set" / "reviews" / ".kappa.json"
    kappa_path.write_text(
        json.dumps({
            "families": {
                "entity_type_confidence": [
                    {"family": "entity_type_confidence", "cell": ["method", "stated"], "status": "escalate", "raw_kappa": 0.3, "disagreement_chunk_ids": [chunk_id]}
                ]
            }
        }, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    canonical = json.loads((paths.destination_dir / f"{chunk_id}.json").read_text(encoding="utf-8"))
    sidecar = {
        "schema_version": review.SIDECAR_SCHEMA_VERSION,
        "chunk_id": chunk_id,
        "frozen_sample_membership": "frozen_random",
        "ann_a_self_record": None,
        "self_disagreements": [],
        "ann_b_review": None,
        "ann_a_record": canonical,
        "ann_b_record": None,
        "canonical_annotated_at": canonical.get("annotated_at"),
    }
    (root / "gold_set" / "reviews" / f"{chunk_id}.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    a = review.ann_c_handoff(paths, frozen, root=root)
    assert chunk_id in a["chunks"]


def test_complete_self_review_writes_wrapper_sidecar_with_canonical_snapshot(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    max_at = _dt.datetime.fromisoformat("2026-09-21T13:44:14.559542Z".replace("Z", "+00:00"))
    later = max_at + _dt.timedelta(days=8)
    root = tmp_path
    (root / "gold_set" / "reviews").mkdir(parents=True)
    self_record = _build_self_record(paths, chunk_id, mutate_confidence="inferred")
    payload = review.complete_self_review(paths, chunk_id, self_record, now=later.astimezone(_dt.timezone.utc), root=root)
    assert payload["schema_version"] == review.SIDECAR_SCHEMA_VERSION
    assert payload["ann_a_self_record"]["annotator_id"] == review.ANN_A_SELF_ANNOTATOR
    assert payload["ann_a_record"]["annotator_id"] == review.ANN_A_ANNOTATOR
    assert payload["ann_b_review"] is None
    assert payload["ann_b_record"] is None


def test_complete_self_review_wash_blocks_early(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    max_at = _dt.datetime.fromisoformat("2026-09-21T13:44:14.559542Z".replace("Z", "+00:00"))
    before = max_at + _dt.timedelta(days=2)
    root = tmp_path
    (root / "gold_set" / "reviews").mkdir(parents=True)
    self_record = _build_self_record(paths, chunk_id)
    with pytest.raises(review.ReviewError, match="wash not satisfied"):
        review.complete_self_review(paths, chunk_id, self_record, now=before.astimezone(_dt.timezone.utc))


def test_ann_c_handoff_lists_self_disagreement_with_sidecar_snapshots(tmp_path):
    paths, _ = _build_paths(tmp_path)
    frozen = review.freeze_ann_b_sample(paths)
    chunk_id = frozen.selected[0]
    max_at = _dt.datetime.fromisoformat("2026-09-21T13:44:14.559542Z".replace("Z", "+00:00"))
    later = max_at + _dt.timedelta(days=8)
    root = tmp_path
    (root / "gold_set" / "reviews").mkdir(parents=True)
    self_record = _build_self_record(paths, chunk_id, mutate_confidence="inferred")
    review.complete_self_review(paths, chunk_id, self_record, now=later.astimezone(_dt.timezone.utc), root=root)
    report = review.ann_c_handoff(paths, frozen, root=root)
    assert chunk_id in report["chunks"]
    entry = report["sidecars"][chunk_id]
    assert entry["ann_a_record"]["annotator_id"] == review.ANN_A_ANNOTATOR
    assert entry["ann_a_self_record"]["annotator_id"] == review.ANN_A_SELF_ANNOTATOR
    assert entry["ann_b_record"] is None
    assert entry["self_disagreement_summary"]


# ---------------------------------------------------------------------------
# Sidecar atomicity / canonical rejection
# ---------------------------------------------------------------------------


def test_self_review_draft_refuses_canonical_destination(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    canonical_annotated_at = "2026-09-21T13:44:14.559542Z"
    max_at = _dt.datetime.fromisoformat(canonical_annotated_at.replace("Z", "+00:00"))
    later = max_at + _dt.timedelta(days=8)
    target = paths.destination_dir / f"{chunk_id}.json"
    with pytest.raises((review.ReviewError, author.AuthorError)):
        review.create_self_review_draft(paths, chunk_id, target, now=later.astimezone(_dt.timezone.utc))
    assert not target.exists() or target.read_text(encoding="utf-8").startswith("{")


def test_sidecar_invalid_payload_fails_closed(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    frozen = review.freeze_ann_b_sample(paths)
    root = tmp_path
    (root / "gold_set" / "reviews").mkdir(parents=True)
    bad = {
        "schema_version": "wrong",
        "chunk_id": chunk_id,
    }
    sidecar = root / "gold_set" / "reviews" / f"{chunk_id}.json"
    sidecar.write_text(json.dumps(bad, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(review.ReviewError):
        review.load_sidecar(chunk_id, root=root)


def test_partial_sidecar_fails_closed(tmp_path):
    paths, chunk_id = _build_paths(tmp_path)
    frozen = review.freeze_ann_b_sample(paths)
    root = tmp_path
    (root / "gold_set" / "reviews").mkdir(parents=True)
    sidecar = root / "gold_set" / "reviews" / f"{chunk_id}.json"
    sidecar.write_text("{not-json", encoding="utf-8")
    with pytest.raises(review.ReviewError):
        review.load_sidecar(chunk_id, root=root)
