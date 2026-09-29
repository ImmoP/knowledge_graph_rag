"""Tests for the offline manual-annotation scorer.

These tests lock the contract of
``scripts/experiments/manifest_auto_review_20260929T160008Z/score_manual_annotation.py``:

* precision is computed only from labelled records (no auto-labels);
* recall is reported as a lower bound, with the upper bound being
  trivially 1.0, and no point estimate is ever published;
* the lower bound can never exceed the upper bound;
* unknown labels raise a clear error;
* the two real annotation JSONs shipped with the review are
  consistent with the per-chunk review markdowns.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "manifest_auto_review_20260929T160008Z"
    / "score_manual_annotation.py"
)
ANNOTATIONS_CHUNK2 = (
    REPO_ROOT
    / "docs"
    / "experiments"
    / "manifest_auto_review_20260929T160008Z"
    / "annotations.json"
)
ANNOTATIONS_CHUNK1 = (
    REPO_ROOT
    / "docs"
    / "experiments"
    / "manifest_auto_review_20260929T160008Z"
    / "annotations_chunk1.json"
)


# ---------- helper imports (run the script in-process) --------------


def _load_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "score_manual_annotation", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scorer():
    return _load_module()


# ---------- core scoring contract -----------------------------------


def test_scorer_emits_supported_precision_only(scorer) -> None:
    annotation = {
        "review_label": "rev_x",
        "chunk_id": "c_x",
        "doc_id": "W_x",
        "mention_labels": [
            {"mention_id": "m1", "label": "supported"},
            {"mention_id": "m2", "label": "unsupported"},
        ],
        "assertion_labels": [
            {"assertion_id": "a1", "label": "supported"},
            {"assertion_id": "a2", "label": "unsupported"},
            {"assertion_id": "a3", "label": "supported"},
        ],
        "possible_false_negatives": [],
    }
    out = scorer.score(annotation)
    assert out["summary"]["mention_precision"] == 0.5
    assert out["summary"]["assertion_precision"] == round(2 / 3, 4)
    # No FNs -> recall lower bound undefined.
    assert out["summary"]["recall_estimate_supported"] is False
    assert out["summary"]["recall_estimate_lower_bound"] is None
    assert out["summary"]["recall_estimate_upper_bound"] == 1.0
    assert out["summary"]["assertion_F1_lower_bound"] is None


def test_scorer_recall_lower_bound_from_fns(scorer) -> None:
    annotation = {
        "review_label": "rev_x",
        "chunk_id": "c_x",
        "doc_id": "W_x",
        "mention_labels": [],
        "assertion_labels": [
            {"assertion_id": "a1", "label": "supported"},
            {"assertion_id": "a2", "label": "supported"},
        ],
        "possible_false_negatives": [
            {"kind": "assertion", "missing": {"x": 1}},
        ],
    }
    out = scorer.score(annotation)
    assert out["summary"]["recall_estimate_lower_bound"] == round(2 / 3, 4)
    # precision = 1.0, recall_lower = 2/3
    assert out["summary"]["assertion_precision"] == 1.0
    assert out["summary"]["assertion_F1_lower_bound"] == round(
        2 * 1.0 * (2 / 3) / (1.0 + (2 / 3)), 4
    )


def test_scorer_rejects_unknown_label(scorer) -> None:
    annotation = {
        "review_label": "rev_x",
        "chunk_id": "c_x",
        "doc_id": "W_x",
        "mention_labels": [
            {"mention_id": "m1", "label": "definitely_yes"},
        ],
        "assertion_labels": [],
        "possible_false_negatives": [],
    }
    with pytest.raises(ValueError, match="unknown label"):
        scorer.score(annotation)


def test_scorer_recall_lower_bound_never_exceeds_upper(scorer) -> None:
    # Pathological input that would imply >100% recall if mishandled.
    annotation = {
        "review_label": "rev_x",
        "chunk_id": "c_x",
        "doc_id": "W_x",
        "mention_labels": [],
        "assertion_labels": [
            {"assertion_id": "a1", "label": "supported"},
        ],
        "possible_false_negatives": [],  # no FNs -> recall = 1.0
    }
    out = scorer.score(annotation)
    assert out["summary"]["recall_estimate_lower_bound"] is None
    assert out["summary"]["recall_estimate_upper_bound"] == 1.0


def test_scorer_counts_unclear_and_schema_error(scorer) -> None:
    annotation = {
        "review_label": "rev_x",
        "chunk_id": "c_x",
        "doc_id": "W_x",
        "mention_labels": [
            {"mention_id": "m1", "label": "supported"},
            {"mention_id": "m2", "label": "unclear"},
            {"mention_id": "m3", "label": "schema_error"},
        ],
        "assertion_labels": [
            {"assertion_id": "a1", "label": "supported"},
            {"assertion_id": "a2", "label": "unsupported"},
        ],
        "possible_false_negatives": [],
    }
    out = scorer.score(annotation)
    assert out["summary"]["mentions_supported"] == 1
    assert out["summary"]["mentions_unclear"] == 1
    assert out["summary"]["mentions_schema_error"] == 1
    assert out["summary"]["assertions_unsupported"] == 1
    # precision counts only "supported" over "labelled"
    assert out["summary"]["mention_precision"] == round(1 / 3, 4)
    assert out["summary"]["assertion_precision"] == 0.5



# ---------- corrected recall-interval math ---------------------------


def test_scorer_recall_interval_with_unclear_collapses_when_empty(
    scorer,
) -> None:
    """When ``unclear`` is zero, the strict interval collapses to the
    legacy lower bound and the helper reports
    ``recall_interval_includes_unclear = False``. This is the
    shipped chunk-2 scenario.
    """
    annotation = {
        "review_label": "rev_x",
        "chunk_id": "c_x",
        "doc_id": "W_x",
        "mention_labels": [],
        "assertion_labels": [
            {"assertion_id": "a1", "label": "supported"},
            {"assertion_id": "a2", "label": "supported"},
            {"assertion_id": "a3", "label": "unsupported"},
        ],
        "possible_false_negatives": [
            {"kind": "assertion", "missing": {"x": 1}},
        ],
    }
    out = scorer.score(annotation)
    summary = out["summary"]
    # No ``unclear`` items -> strict interval collapses to a point.
    assert summary["recall_interval_includes_unclear"] is False
    assert summary["recall_lower_bound_strict"] == round(2 / 3, 4)
    assert summary["recall_upper_bound_strict"] == round(2 / 3, 4)
    # Legacy alias still equals the strict lower bound.
    assert summary["recall_estimate_lower_bound"] == round(2 / 3, 4)
    assert summary["recall_estimate_upper_bound"] == 1.0
    # F1 lower == F1 upper when the interval collapses.
    assert summary["f1_lower_bound_strict"] == summary["f1_upper_bound_strict"]


def test_scorer_recall_interval_brackets_when_unclear_present(
    scorer,
) -> None:
    """When ``unclear`` items exist, the strict recall interval widens
    and the upper bound is strictly greater than the lower bound.
    This is the *corrected* interpretation: items the reviewer could
    not verify sit inside the universe of true assertions and shift
    the upper bound upward.
    """
    annotation = {
        "review_label": "rev_x",
        "chunk_id": "c_x",
        "doc_id": "W_x",
        "mention_labels": [],
        "assertion_labels": [
            {"assertion_id": "a1", "label": "supported"},
            {"assertion_id": "a2", "label": "supported"},
            {"assertion_id": "a3", "label": "unclear"},
        ],
        "possible_false_negatives": [
            {"kind": "assertion", "missing": {"x": 1}},
        ],
    }
    out = scorer.score(annotation)
    summary = out["summary"]
    # Lower bound (assume all unclear are FP): 2/(2+1) = 2/3.
    assert summary["recall_lower_bound_strict"] == round(2 / 3, 4)
    # Upper bound (assume all unclear are TP): 3/(3+1) = 3/4.
    assert summary["recall_upper_bound_strict"] == round(3 / 4, 4)
    # Flag is set so consumers know the interval is wider than a point.
    assert summary["recall_interval_includes_unclear"] is True
    # F1 lower uses the lower recall, F1 upper uses the upper recall.
    precision = 2 / 3  # 2 supported / 3 labelled
    expected_f1_lower = 2 * precision * (2 / 3) / (precision + (2 / 3))
    expected_f1_upper = 2 * precision * (3 / 4) / (precision + (3 / 4))
    assert summary["f1_lower_bound_strict"] == round(expected_f1_lower, 4)
    assert summary["f1_upper_bound_strict"] == round(expected_f1_upper, 4)
    # Sanity: lower <= upper on both recall and F1.
    assert (
        summary["recall_lower_bound_strict"]
        <= summary["recall_upper_bound_strict"]
    )
    assert (
        summary["f1_lower_bound_strict"]
        <= summary["f1_upper_bound_strict"]
    )


def test_scorer_recall_interval_undefined_when_no_supported(
    scorer,
) -> None:
    """If ``supported`` is zero (only ``unsupported`` / ``unclear`` /
    ``schema_error`` items in the accepted set), the recall interval
    is undefined -- there are no true positives to measure.
    """
    annotation = {
        "review_label": "rev_x",
        "chunk_id": "c_x",
        "doc_id": "W_x",
        "mention_labels": [],
        "assertion_labels": [
            {"assertion_id": "a1", "label": "unsupported"},
            {"assertion_id": "a2", "label": "unclear"},
        ],
        "possible_false_negatives": [
            {"kind": "assertion", "missing": {"x": 1}},
        ],
    }
    out = scorer.score(annotation)
    summary = out["summary"]
    assert summary["recall_estimate_supported"] is False
    assert summary["recall_lower_bound_strict"] is None
    assert summary["recall_upper_bound_strict"] is None
    assert summary["f1_lower_bound_strict"] is None
    assert summary["f1_upper_bound_strict"] is None


def test_scorer_recall_interval_all_unclear_does_not_inflate(
    scorer,
) -> None:
    """When ``unclear`` items dominate and ``supported`` is zero,
    the strict interval is reported as ``None`` for both bounds --
    there are no ``supported`` items to anchor the lower bound, and
    the upper bound is not published alone because it would be
    meaningless without a paired lower bound.
    """
    annotation = {
        "review_label": "rev_x",
        "chunk_id": "c_x",
        "doc_id": "W_x",
        "mention_labels": [],
        "assertion_labels": [
            {"assertion_id": "a1", "label": "unclear"},
            {"assertion_id": "a2", "label": "unclear"},
        ],
        "possible_false_negatives": [
            {"kind": "assertion", "missing": {"x": 1}},
        ],
    }
    out = scorer.score(annotation)
    summary = out["summary"]
    assert summary["recall_estimate_supported"] is False
    assert summary["recall_lower_bound_strict"] is None
    assert summary["recall_upper_bound_strict"] is None


def test_scorer_chunk2_interval_collapses_to_point(scorer) -> None:
    """The shipped chunk-2 annotation has zero ``unclear`` items, so
    the corrected interval collapses to the legacy lower bound
    (``supported / (supported + FN)`` = ``2 / 3``).
    """
    data = json.loads(ANNOTATIONS_CHUNK2.read_text(encoding="utf-8"))
    out = scorer.score(data)
    summary = out["summary"]
    assert summary["assertions_unclear"] == 0
    assert summary["recall_interval_includes_unclear"] is False
    assert summary["recall_lower_bound_strict"] == round(2 / 3, 4)
    assert summary["recall_upper_bound_strict"] == round(2 / 3, 4)
    # Legacy alias unchanged.
    assert summary["recall_estimate_lower_bound"] == round(2 / 3, 4)
    assert summary["recall_estimate_upper_bound"] == 1.0


# ---------- shipped annotations match the review --------------------



def test_chunk2_annotations_match_review_markdown() -> None:
    """The two unsupported assertions in the shipped annotations file
    must correspond to the two ``USES_METHOD`` edges flagged in
    ``review_chunk_c_001452fd142ce075.md``.
    """
    data = json.loads(ANNOTATIONS_CHUNK2.read_text(encoding="utf-8"))
    assert data["chunk_id"] == "c_001452fd142ce075"
    assert data["doc_id"] == "W3038572442"
    assert len(data["mention_labels"]) == 4
    assert len(data["assertion_labels"]) == 4
    unsupported = [a for a in data["assertion_labels"] if a["label"] == "unsupported"]
    assert len(unsupported) == 2
    # Both unsupported assertions are USES_METHOD (per the review).
    ids = {a["assertion_id"] for a in unsupported}
    assert ids == {"a_186ebfe9b1cd3758", "a_48f1cfb9f80bca7b"}

    # The possible_false_negatives list must include the Answer Coverage
    # mention and the REPORTS_METRIC assertion that the review flagged.
    kinds = [fn["kind"] for fn in data["possible_false_negatives"]]
    assert "mention" in kinds
    assert "assertion" in kinds
    metric_fns = [
        fn for fn in data["possible_false_negatives"]
        if fn["kind"] == "mention"
        and fn["missing"]["canonical_form"] == "Answer Coverage"
    ]
    assert len(metric_fns) == 1
    relation_fns = [
        fn for fn in data["possible_false_negatives"]
        if fn["kind"] == "assertion"
        and fn["missing"]["relation_type"] == "REPORTS_METRIC"
    ]
    assert len(relation_fns) == 1


def test_chunk1_annotations_match_review_markdown() -> None:
    """Chunk 1 has zero accepted assertions; the annotations must
    reflect that, and the possible_false_negatives are explicitly
    flagged as borderline."""
    data = json.loads(ANNOTATIONS_CHUNK1.read_text(encoding="utf-8"))
    assert data["chunk_id"] == "c_0001921855a8156b"
    assert data["doc_id"] == "W3137305332"
    assert len(data["mention_labels"]) == 2
    assert data["assertion_labels"] == []
    fns = data["possible_false_negatives"]
    assert all(fn["kind"] == "assertion" for fn in fns)
    assert all(fn["missing"]["relation_type"] == "USES_METHOD" for fn in fns)


# ---------- end-to-end script invocation -----------------------------


def test_script_runs_against_chunk2_annotations(tmp_path) -> None:
    """The CLI helper runs offline and emits the documented numbers."""
    out = tmp_path / "scored.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--annotations",
            str(ANNOTATIONS_CHUNK2),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    scored = json.loads(out.read_text(encoding="utf-8"))
    assert scored["chunk_id"] == "c_001452fd142ce075"
    summary = scored["summary"]
    assert summary["mentions_supported"] == 4
    assert summary["assertions_supported"] == 2
    assert summary["assertions_unsupported"] == 2
    assert summary["mention_precision"] == 1.0
    assert summary["assertion_precision"] == 0.5
    # One FN assertion (REPORTS_METRIC) flagged -> recall_lower = 2/3.
    assert summary["recall_estimate_lower_bound"] == round(2 / 3, 4)
    assert summary["recall_estimate_upper_bound"] == 1.0


def test_script_runs_against_chunk1_annotations(tmp_path) -> None:
    out = tmp_path / "scored.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--annotations",
            str(ANNOTATIONS_CHUNK1),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    scored = json.loads(out.read_text(encoding="utf-8"))
    summary = scored["summary"]
    assert summary["mentions_supported"] == 2
    assert summary["assertions_supported"] == 0
    assert summary["mention_precision"] == 1.0
    # No labelled claims, so assertion_precision is 0.0 by convention
    # (0 / 0 falls back to 0.0 in the helper) and recall is undefined.
    assert summary["assertion_precision"] == 0.0
    assert summary["recall_estimate_supported"] is False
    assert summary["possible_false_negative_assertions"] == 2


def test_script_refuses_missing_input(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--annotations",
            str(tmp_path / "does-not-exist.json"),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2
    assert "annotations file not found" in result.stderr