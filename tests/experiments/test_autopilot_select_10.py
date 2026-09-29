"""Offline tests for the deterministic 10-chunk autopilot selector.

These tests are offline (no LLM calls, no network) and lock the
selection contract documented in
``docs/experiments/manifest-auto-review/autopilot_plan_20260929T160008Z.md``:

* the selector picks exactly 10 chunks;
* the two smoke-run chunks are excluded;
* the length-tercile balance matches the documented targets
  (3 long, 3 medium, 4 short);
* per-doc cap is respected (<=2 chunks per doc);
* at least one prose chunk and one table chunk are picked;
* the same seed reproduces the same plan.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "manifest_auto_review_20260929T160008Z"
    / "autopilot_select_10.py"
)
CHUNKS_JSONL = REPO_ROOT / "src" / "benchmark" / "dataset" / "benchmark_chunks.jsonl"


def _load_module():
    spec = importlib.util.spec_from_file_location("autopilot_select_10", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def selector():
    return _load_module()


@pytest.fixture(scope="module")
def chunks():
    out = []
    with CHUNKS_JSONL.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---------- core selection contract -----------------------------------


def test_selector_picks_exactly_ten_chunks(selector, chunks) -> None:
    plan = selector.select_plan(chunks)
    assert len(plan["plan"]) == 10


def test_selector_excludes_smoke_run_chunks(selector, chunks) -> None:
    plan = selector.select_plan(chunks)
    excluded = set(plan["excluded_chunk_ids"])
    picked_ids = {c["chunk_id"] for c in plan["plan"]}
    # The two smoke-run chunks are excluded and must not appear in the plan.
    assert "c_0001921855a8156b" in excluded
    assert "c_001452fd142ce075" in excluded
    assert picked_ids.isdisjoint(excluded)


def test_selector_tercile_balance_matches_targets(selector, chunks) -> None:
    plan = selector.select_plan(chunks)
    totals = plan["totals"]
    assert totals["long"] == 3
    assert totals["medium"] == 3
    assert totals["short"] == 4


def test_selector_per_doc_cap_respected(selector, chunks) -> None:
    plan = selector.select_plan(chunks)
    counts: dict[str, int] = {}
    for c in plan["plan"]:
        counts[c["doc_id"]] = counts.get(c["doc_id"], 0) + 1
    assert all(v <= 2 for v in counts.values())


def test_selector_includes_prose_and_table_chunks(selector, chunks) -> None:
    plan = selector.select_plan(chunks)
    totals = plan["totals"]
    assert totals["prose_chunks"] >= 1
    assert totals["table_chunks"] >= 1


def test_selector_plan_records_carry_required_metadata(
    selector, chunks
) -> None:
    plan = selector.select_plan(chunks)
    for c in plan["plan"]:
        assert c["chunk_id"]
        assert c["doc_id"]
        assert c["length_bucket"] in {"short", "medium", "long"}
        assert isinstance(c["tags"], dict)
        # Tag keys are booleans.
        assert set(c["tags"].keys()) == {"prose", "latex", "table"}
        assert isinstance(c["char_len"], int)
        assert c["char_len"] > 0


def test_selector_same_seed_reproduces_same_plan(selector, chunks) -> None:
    plan_a = selector.select_plan(chunks, seed=0xFEEDFACE)
    plan_b = selector.select_plan(chunks, seed=0xFEEDFACE)
    assert [c["chunk_id"] for c in plan_a["plan"]] == [
        c["chunk_id"] for c in plan_b["plan"]
    ]


def test_selector_uses_documented_algorithm_version(selector, chunks) -> None:
    plan = selector.select_plan(chunks)
    assert plan["algorithm_version"] == "manifest-auto-autopilot-selector/v1"
    assert plan["seed"] == 0xFEEDFACE


def test_selector_emits_canonical_selection_criteria(
    selector, chunks
) -> None:
    plan = selector.select_plan(chunks)
    criteria = plan["selection_criteria"]
    # All five documented criteria are present and ordered for readability.
    assert any("exclude_chunks_in_smoke_run" in c for c in criteria)
    assert any("length_tercile_balance" in c for c in criteria)
    assert any("per_doc_cap" in c for c in criteria)
    assert any("tag_minimums" in c for c in criteria)
    assert any("deterministic_order" in c for c in criteria)


# ---------- end-to-end CLI -------------------------------------------


def test_cli_runs_offline_and_writes_plan(tmp_path, selector) -> None:
    out = tmp_path / "plan.json"
    result = __import__("subprocess").run(
        [
            sys.executable,
            str(SCRIPT),
            "--chunks",
            str(CHUNKS_JSONL),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": __import__("os").environ["PATH"]},
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert len(plan["plan"]) == 10
    assert plan["totals"]["long"] == 3
    assert plan["totals"]["medium"] == 3
    assert plan["totals"]["short"] == 4