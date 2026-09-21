"""Tests for the deterministic nearest-feasible gold-set sampler v2 (#63)."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import pytest

from scripts.gold_set import sample_v2


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ontology(path: Path) -> None:
    path.write_text('{"relation_types":[{"name":"CITES"}]}\n', encoding="utf-8")


def _write_leakage(
    path: Path,
    *,
    chunks_path: Path,
    manifest_path: Path,
    ontology_path: Path,
    per_question: dict[str, list[str]],
    source_draft_sha256: str = "0" * 64,
    record_count: int = 45,
    all_split_held_out: bool = True,
) -> None:
    question_map = {qid: sorted(set(ids)) for qid, ids in sorted(per_question.items())}
    excluded = sorted({chunk_id for ids in question_map.values() for chunk_id in ids})
    payload = {
        "all_split_held_out": all_split_held_out,
        "chunks_sha256": _sha(chunks_path),
        "excluded_chunk_ids": excluded,
        "manifest_sha256": _sha(manifest_path),
        "ontology_sha256": _sha(ontology_path),
        "question_chunk_map": question_map,
        "record_count": record_count,
        "schema_version": "held-out-leakage.v1",
        "source_draft_sha256": source_draft_sha256,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _default_question_map(excluded: list[str]) -> dict[str, list[str]]:
    return {f"q_{index:03d}": excluded[:] for index in range(1, 46)}

def _v2_corpus() -> tuple[list[dict], list[dict]]:
    """Synthetic corpus modeled on the canonical DPR/RAG/ColBERT-proxy setup."""
    papers = [
        *sample_v2.CENTRAL_ANCHORS,
        "W3038831159",
        *(f"W{1000 + index}" for index in range(20)),
    ]
    manifest = [
        {"openalex_id": paper_id, "cited_by_count": index}
        for index, paper_id in enumerate(papers)
    ]
    chunks: list[dict] = []
    serial = 0
    dpr_cells = {(1, 0): 10, (1, 1): 3, (1, 2): 5, (2, 0): 11, (2, 1): 7, (2, 2): 6, (0, 2): 2}
    for (depth, length), count in dpr_cells.items():
        for _ in range(count):
            size = [1000, 1500, 1800][length]
            chunks.append(
                {
                    "chunk_id": f"c_{serial:016x}",
                    "doc_id": "W3015883388",
                    "section_path": ["section"] * depth,
                    "chunk_index": serial,
                    "text": "x" * size,
                    "token_estimate": size // 4,
                    "char_start": 0,
                    "char_end": size,
                }
            )
            serial += 1
    rag_cells = {(0, 1): 1, (1, 0): 4, (1, 1): 3}
    for (depth, length), count in rag_cells.items():
        for _ in range(count):
            size = [1000, 1500, 1800][length]
            chunks.append(
                {
                    "chunk_id": f"c_{serial:016x}",
                    "doc_id": "W3039017601",
                    "section_path": ["section"] * depth,
                    "chunk_index": serial,
                    "text": "x" * size,
                    "token_estimate": size // 4,
                    "char_start": 0,
                    "char_end": size,
                }
            )
            serial += 1
    colbert_cells = {(0, 0): 1, (0, 1): 1, (1, 0): 3, (1, 1): 5, (1, 2): 4, (2, 0): 9, (2, 1): 12, (2, 2): 7}
    for (depth, length), count in colbert_cells.items():
        for _ in range(count):
            size = [1000, 1500, 1800][length]
            chunks.append(
                {
                    "chunk_id": f"c_{serial:016x}",
                    "doc_id": "W3038831159",
                    "section_path": ["section"] * depth,
                    "chunk_index": serial,
                    "text": "x" * size,
                    "token_estimate": size // 4,
                    "char_start": 0,
                    "char_end": size,
                }
            )
            serial += 1
    for paper_id in papers[3:]:
        for depth in range(3):
            for length in range(3):
                for repeat in range(2):
                    size = [1000, 1500, 1800][length]
                    chunks.append(
                        {
                            "chunk_id": f"c_{serial:016x}",
                            "doc_id": paper_id,
                            "section_path": ["section"] * depth,
                            "chunk_index": serial,
                            "text": "x" * size,
                            "token_estimate": size // 4,
                            "char_start": 0,
                            "char_end": size,
                        }
                    )
                    serial += 1
    return chunks, manifest


def _default_exclusions(chunks: list[dict]) -> list[str]:
    return [
        row["chunk_id"]
        for row in chunks
        if row["doc_id"] == "W3015883388"
    ][:6]



@pytest.fixture(scope="module")
def production_result(tmp_path_factory):
    root = tmp_path_factory.mktemp("v2_production")
    chunks, manifest = _v2_corpus()
    excluded = _default_exclusions(chunks)
    paths = {
        "chunks": root / "chunks.jsonl",
        "manifest": root / "manifest.jsonl",
        "leakage": root / "held_out_leakage.v1.json",
        "ontology": root / "ontology.yaml",
        "output": root / "sample.v2.jsonl",
        "audit": root / "sample.v2.audit.json",
    }
    _write(paths["chunks"], chunks)
    _write(paths["manifest"], manifest)
    _write_ontology(paths["ontology"])
    _write_leakage(
        paths["leakage"],
        chunks_path=paths["chunks"],
        manifest_path=paths["manifest"],
        ontology_path=paths["ontology"],
        per_question=_default_question_map(excluded),
    )
    count = sample_v2.run_sample(
        paths["chunks"],
        paths["manifest"],
        paths["leakage"],
        paths["output"],
        ontology_path=paths["ontology"],
        audit_path=paths["audit"],
    )
    assert count == 120
    payload = paths["output"].read_bytes()
    rows = [json.loads(line) for line in payload.splitlines()]
    audit = json.loads(paths["audit"].read_text())
    return chunks, manifest, excluded, paths, payload, rows, audit


def test_frozen_constants():
    assert sample_v2.GOLD_SET_SEED == 0x9E3779B1
    assert sample_v2.SAMPLER_SEED_TEXT == "9e3779b1"
    assert sample_v2.SAMPLER_VERSION == "gold-set-sample.v2"
    assert sample_v2.SAMPLE_SIZE == 120
    assert sample_v2.NON_CENTRAL_QUOTA == 12
    assert sample_v2.CENTRAL_TOTAL == 60
    assert sample_v2.CENTRAL_NOMINAL_QUOTA == 20
    assert sample_v2.CENTRAL_ANCHORS == ("W3015883388", "W3039017601")
    assert sample_v2.COLBERT_PROXY_CANDIDATES == (
        "W3021397474",
        "W3038831159",
        "W3217305727",
    )
    assert sample_v2.SECTION_DEPTH_TARGETS == (6, 24, 88, 2)
    assert sample_v2.LENGTH_TARGETS == (40, 40, 40)


def test_primary_colbert_proxy_selection():
    proxy_id, safe = sample_v2._resolve_colbert_proxy({"W3021397474": 40})
    assert proxy_id == "W3021397474"
    assert safe == 40


def test_first_colbert_fallback_to_w3038831159():
    proxy_id, safe = sample_v2._resolve_colbert_proxy({"W3038831159": 42})
    assert proxy_id == "W3038831159"
    assert safe == 42


def test_second_colbert_fallback_to_w3217305727():
    proxy_id, safe = sample_v2._resolve_colbert_proxy({"W3217305727": 5})
    assert proxy_id == "W3217305727"
    assert safe == 5


def test_no_colbert_proxy_fails_closed():
    with pytest.raises(sample_v2.SampleError, match="no ColBERT-family candidate is available"):
        sample_v2._resolve_colbert_proxy({})


def test_central_quota_capacity_32_8_42_yields_26_8_26():
    quotas = sample_v2._central_quotas(
        {"W3015883388": 32, "W3039017601": 8, "W3038831159": 42}
    )
    assert quotas == {"W3015883388": 26, "W3039017601": 8, "W3038831159": 26}


def test_infeasible_central_total_fails_closed():
    with pytest.raises(sample_v2.SampleError, match="central total infeasible"):
        sample_v2._central_quotas(
            {"W3015883388": 5, "W3039017601": 5, "W3038831159": 42}
        )


def test_all_deficient_central_fails_closed():
    with pytest.raises(sample_v2.SampleError, match="central total infeasible"):
        sample_v2._central_quotas(
            {"W3015883388": 5, "W3039017601": 8, "W3038831159": 10}
        )


def test_central_anchor_zero_capacity_fails_closed():
    with pytest.raises(sample_v2.SampleError, match="zero leakage-safe chunks"):
        sample_v2._central_quotas(
            {"W3015883388": 0, "W3039017601": 8, "W3038831159": 42}
        )


def test_production_contract(production_result):
    chunks, _, excluded, _, payload, rows, audit = production_result
    ids = [row["chunk_id"] for row in rows]
    paper_counts = Counter(row["doc_id"] for row in rows)
    tier_counts = Counter(row["strata"]["paper_tier"] for row in rows)
    assert len(rows) == len(set(ids)) == 120
    assert ids == sorted(ids)
    assert payload.endswith(b"\n")
    central_in_audit = sorted(audit["central_paper_ids"])
    non_central_in_audit = sorted(audit["non_central_paper_ids"])
    assert sum(audit["paper_quotas"][p] for p in central_in_audit) == 60
    assert sum(audit["paper_quotas"][p] for p in non_central_in_audit) == 60
    for paper_id in non_central_in_audit:
        assert audit["paper_quotas"][paper_id] == 12
    assert tier_counts == {"benchmark_central": 60, "mid_density": 36, "long_tail": 24}
    assert paper_counts["W3039017601"] == 8
    assert paper_counts["W3015883388"] == 26
    assert paper_counts["W3038831159"] == 26
    assert set(excluded).isdisjoint(ids)
    assert audit["leakage_overlap_count"] == 0
    actual_depth = tuple(audit["achieved_depth_marginals"])
    actual_length = tuple(audit["achieved_length_marginals"])
    assert sum(actual_depth) == 120
    assert sum(actual_length) == 120
    assert all(
        set(row) == {"chunk_id", "doc_id", "strata", "sampler_seed", "sampler_version"}
        for row in rows
    )
    assert all(
        set(row["strata"])
        == {"paper_tier", "section_depth", "length_tercile", "density_post_stratum"}
        for row in rows
    )
    assert {row["sampler_seed"] for row in rows} == {"9e3779b1"}
    assert {row["sampler_version"] for row in rows} == {"gold-set-sample.v2"}
    assert all(
        "text" not in row and "title" not in row and "question" not in row
        for row in rows
    )


def test_repeated_run_is_byte_identical_and_does_not_rewrite(production_result):
    _, _, _, paths, payload, _, _ = production_result
    old_mtime = paths["output"].stat().st_mtime_ns
    time.sleep(0.001)
    assert sample_v2.run_sample(
        paths["chunks"],
        paths["manifest"],
        paths["leakage"],
        paths["output"],
        ontology_path=paths["ontology"],
        audit_path=paths["audit"],
    ) == 120
    assert paths["output"].read_bytes() == payload
    assert paths["output"].stat().st_mtime_ns == old_mtime


def test_shuffled_inputs_with_refrozen_identity_are_byte_identical(
    production_result, tmp_path
):
    chunks, manifest, excluded, _, payload, _, _ = production_result
    rng = random.Random(20260918)
    shuffled_chunks = chunks.copy()
    shuffled_manifest = manifest.copy()
    rng.shuffle(shuffled_chunks)
    rng.shuffle(shuffled_manifest)
    chunks_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    ontology_path = tmp_path / "ontology.yaml"
    leakage_path = tmp_path / "held_out_leakage.v1.json"
    output = tmp_path / "sample.v2.jsonl"
    audit_path = tmp_path / "sample.v2.audit.json"
    _write(chunks_path, shuffled_chunks)
    _write(manifest_path, shuffled_manifest)
    _write_ontology(ontology_path)
    _write_leakage(
        leakage_path,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        ontology_path=ontology_path,
        per_question=_default_question_map(excluded),
    )
    sample_v2.run_sample(
        chunks_path,
        manifest_path,
        leakage_path,
        output,
        ontology_path=ontology_path,
        audit_path=audit_path,
    )
    assert output.read_bytes() == payload


def test_audit_corresponds_exactly_to_selected_rows(production_result):
    _, _, _, paths, _, rows, audit = production_result
    selected_chunk_ids = sorted(row["chunk_id"] for row in rows)
    assert audit["chunks_sha256"] == _sha(paths["chunks"])
    assert audit["manifest_sha256"] == _sha(paths["manifest"])
    assert audit["ontology_sha256"] == _sha(paths["ontology"])
    assert audit["leakage_artifact_sha256"] == _sha(paths["leakage"])
    assert audit["selected_chunk_ids_sha256"] == hashlib.sha256(
        "".join(selected_chunk_ids).encode("utf-8")
    ).hexdigest()
    quota_total = sum(audit["paper_quotas"].values())
    assert quota_total == 120
    for paper_id in audit["selected_paper_ids"]:
        assert paper_id in audit["paper_quotas"]
        assert paper_id in audit["leakage_safe_capacity"]
        assert audit["paper_quotas"][paper_id] <= audit["leakage_safe_capacity"][paper_id]
    assert audit["sampler_version"] == "gold-set-sample.v2"
    assert audit["sampler_seed"] == "9e3779b1"
    for field in (
        "selected_paper_ids",
        "paper_tiers",
        "paper_quotas",
        "leakage_safe_capacity",
        "reference_depth_marginals",
        "achieved_depth_marginals",
        "reference_length_marginals",
        "achieved_length_marginals",
        "objective",
        "leakage_overlap_count",
        "chunks_sha256",
        "manifest_sha256",
        "selected_chunk_ids_sha256",
        "ontology_sha256",
        "leakage_artifact_sha256",
        "sampler_seed",
        "sampler_version",
    ):
        assert field in audit, f"audit missing {field}"


def test_different_existing_output_is_refused(tmp_path):
    output = tmp_path / "sample.v2.jsonl"
    output.write_bytes(b"different\n")
    with pytest.raises(sample_v2.SampleError, match="refusing to overwrite"):
        sample_v2._publish(output, b"expected\n")
    assert output.read_bytes() == b"different\n"


def test_no_force_or_legacy_questions_escape_hatch():
    parser = sample_v2._build_parser()
    option_strings = {
        option for action in parser._actions for option in action.option_strings
    }
    assert "--leakage" in option_strings
    assert "--ontology" in option_strings
    assert "--questions" not in option_strings
    assert "--force" not in option_strings
    assert "--overwrite" not in option_strings
    assert "--skip-leakage" not in option_strings
    assert "--allow-empty-held-out" not in option_strings


def test_sampler_does_not_read_question_content_or_candidate_extraction_files(
    monkeypatch, production_result, tmp_path
):
    chunks, manifest, excluded, _, _, _, _ = production_result
    chunks_path = tmp_path / "chunks"
    manifest_path = tmp_path / "manifest"
    ontology_path = tmp_path / "ontology"
    leakage_path = tmp_path / "leakage"
    _write(chunks_path, chunks)
    _write(manifest_path, manifest)
    _write_ontology(ontology_path)
    _write_leakage(
        leakage_path,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        ontology_path=ontology_path,
        per_question=_default_question_map(excluded),
    )
    original = Path.read_text

    forbidden_substrings = ("extractions", "questions.jsonl", "questions.draft")

    def guarded(self, *args, **kwargs):
        lowered = str(self).lower()
        for forbidden in forbidden_substrings:
            assert forbidden not in lowered, (
                f"sampler touched a forbidden path: {self}"
            )
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    sample_v2.run_sample(
        chunks_path,
        manifest_path,
        leakage_path,
        tmp_path / "output",
        ontology_path=ontology_path,
        audit_path=tmp_path / "audit",
    )


def test_globally_optimal_nearest_feasible_allocation_small_case(monkeypatch):
    """Tiny synthetic case: verify the sampler matches the brute-force optimum."""
    import scripts.gold_set.sample_v2 as mod
    monkeypatch.setattr(mod, "SECTION_DEPTH_TARGETS", (2, 8, 8, 2))
    monkeypatch.setattr(mod, "LENGTH_TARGETS", (8, 6, 6))
    papers = ("W_A", "W_B", "W_C", "W_D")
    quotas = {"W_A": 8, "W_B": 4, "W_C": 4, "W_D": 4}
    cells_by_paper = {
        "W_A": {
            (1, 0): [sample_v2.Chunk(f"c_a{i:016x}", "W_A", 1, 0) for i in range(8)],
            (2, 2): [sample_v2.Chunk(f"c_b{i:016x}", "W_A", 2, 2) for i in range(4)],
        },
        "W_B": {
            (1, 1): [sample_v2.Chunk(f"c_b{i:016x}", "W_B", 1, 1) for i in range(4)],
        },
        "W_C": {
            (2, 1): [sample_v2.Chunk(f"c_c{i:016x}", "W_C", 2, 1) for i in range(4)],
        },
        "W_D": {
            (1, 0): [sample_v2.Chunk(f"c_d{i:016x}", "W_D", 1, 0) for i in range(4)],
        },
    }
    # W_B/C/D are forced to a single cell. Only W_A has a choice: x chunks at (1,0),
    # (8-x) chunks at (2,2). Brute-force the optimum over x in [0..8]:
    best_obj = None
    best_alloc = None
    for x in range(9):
        if (8 - x) > 4:
            continue
        depth = (0, x + 4 + 4, 8 - x + 4, 0)
        length = (x + 4, 4 + 4, 8 - x)
        struct_obj = sample_v2._structure_objective(depth, length)
        if best_obj is None or struct_obj < best_obj:
            best_obj = struct_obj
            best_alloc = (depth, length)
    allocation = sample_v2._allocate(papers, quotas, cells_by_paper)
    actual_depth = (0, 0, 0, 0)
    actual_length = (0, 0, 0)
    for chunk in allocation.selected:
        actual_depth = (
            actual_depth[0] + (1 if chunk.section_depth == 0 else 0),
            actual_depth[1] + (1 if chunk.section_depth == 1 else 0),
            actual_depth[2] + (1 if chunk.section_depth == 2 else 0),
            actual_depth[3] + (1 if chunk.section_depth == 3 else 0),
        )
        actual_length = (
            actual_length[0] + (1 if chunk.length_index == 0 else 0),
            actual_length[1] + (1 if chunk.length_index == 1 else 0),
            actual_length[2] + (1 if chunk.length_index == 2 else 0),
        )
    assert best_alloc is not None
    assert actual_depth == best_alloc[0]
    assert actual_length == best_alloc[1]

def test_unsound_pareto_pruning_regression():
    """Regression for the unsound per-paper Pareto pruning bug (#63).

    For a paper with quota 12, the spec's marginals
    A = (d=(0,1,9,2), l=(4,4,4)) and B = (d=(0,0,10,2), l=(4,4,4)) were both
    reachable, yet the old ``_pareto_minimal`` compared each paper against a
    quota-scaled LOCAL ideal and marked B dominated by A. B must survive: when
    the remaining papers contribute depth (6,24,78,0), only B reaches the exact
    global target (6,24,88,2), while A overshoots to (6,25,87,2).
    """
    cells = {
        (1, 0): [sample_v2.Chunk(f"c_xa{i:016x}", "W_X", 1, 0) for i in range(3)],
        (2, 0): [sample_v2.Chunk(f"c_xb{i:016x}", "W_X", 2, 0) for i in range(6)],
        (2, 1): [sample_v2.Chunk(f"c_xc{i:016x}", "W_X", 2, 1) for i in range(6)],
        (2, 2): [sample_v2.Chunk(f"c_xd{i:016x}", "W_X", 2, 2) for i in range(4)],
        (3, 2): [sample_v2.Chunk(f"c_xe{i:016x}", "W_X", 3, 2) for i in range(3)],
    }
    marginals = sample_v2._enumerate_marginals(cells, 12)
    # Both marginals must survive enumeration; unsound pruning discarded B.
    assert ((0, 1, 9, 2), (4, 4, 4)) in marginals
    assert ((0, 0, 10, 2), (4, 4, 4)) in marginals
    # B reaches the exact global target; A does not.
    rest_depth = (6, 24, 78, 0)
    rest_length = (36, 36, 36)
    a_depth = tuple(rest_depth[i] + (0, 1, 9, 2)[i] for i in range(4))
    b_depth = tuple(rest_depth[i] + (0, 0, 10, 2)[i] for i in range(4))
    total_length = tuple(rest_length[j] + 4 for j in range(3))
    assert b_depth == sample_v2.SECTION_DEPTH_TARGETS
    a_obj = sample_v2._structure_objective(a_depth, total_length)
    b_obj = sample_v2._structure_objective(b_depth, total_length)
    assert b_obj < a_obj, (a_obj, b_obj)


def test_primary_in_manifest_with_chunks_remains_resolved_proxy():
    """Regression: even if the primary's leakage-safe capacity differs from
    its canonical chunk count, the primary remains the resolved proxy as
    long as it has canonical chunks.
    """
    import scripts.gold_set.sample_v2 as mod
    canonical_counts = {
        "W3021397474": 40,
        "W3038831159": 50,
        "W3217305727": 30,
    }
    proxy_id, safe = mod._resolve_colbert_proxy(canonical_counts)
    assert proxy_id == "W3021397474"
    # The returned count is the canonical-chunk count (pre-exclusion), NOT
    # the leakage-safe capacity. Leakage-safe capacity is computed separately.
    assert safe == 40


def test_proxy_resolution_is_chunks_only_no_manifest_required():
    """Regression for #48 Refinement 8: proxy availability is chunks-only.

    The primary candidate has canonical chunks but is absent from the
    manifest. Proxy resolution must still pick it as the resolved proxy
    because availability is decided by the canonical-chunk artifact alone;
    manifest presence is NOT a prerequisite for proxy resolution.
    """
    import scripts.gold_set.sample_v2 as mod
    canonical_counts = {
        "W3021397474": 40,
        "W3038831159": 50,
        "W3217305727": 30,
    }
    # Manifest deliberately empty / absent for all candidates.
    proxy_id, safe = mod._resolve_colbert_proxy(canonical_counts)
    assert proxy_id == "W3021397474"
    assert safe == 40


def test_proxy_resolution_falls_through_when_primary_absent_from_chunks():
    """Regression for #48 Refinement 8: when the primary has NO canonical
    chunks, the proxy resolver evaluates the next candidate in the pinned
    order (W3038831159, then W3217305727). Manifest is not consulted.
    """
    import scripts.gold_set.sample_v2 as mod
    # Primary has zero canonical chunks; secondary has 42.
    proxy_id, safe = mod._resolve_colbert_proxy({"W3021397474": 0, "W3038831159": 42})
    assert proxy_id == "W3038831159"
    assert safe == 42


def test_central_dpr_missing_from_manifest_fails_closed():
    """Central anchor DPR missing from the manifest must fail closed."""
    with pytest.raises(
        sample_v2.SampleError,
        match="central anchor W3015883388 missing from manifest",
    ):
        sample_v2._central_quotas(
            {"W3015883388": 32, "W3039017601": 8, "W3038831159": 26},
            canonical_counts={"W3038831159": 40},
            manifest_ids={"W3039017601", "W3038831159"},
        )


def test_central_rag_missing_from_manifest_fails_closed():
    """Central anchor RAG missing from the manifest must fail closed."""
    with pytest.raises(
        sample_v2.SampleError,
        match="central anchor W3039017601 missing from manifest",
    ):
        sample_v2._central_quotas(
            {"W3015883388": 32, "W3039017601": 8, "W3038831159": 26},
            canonical_counts={"W3038831159": 40},
            manifest_ids={"W3015883388", "W3038831159"},
        )


def test_audit_body_is_deterministic_canonical_json():
    """Audit serialization is canonical: sorted keys, indent 2, trailing \n."""
    body1 = sample_v2._audit_body({"b": 2, "a": 1})
    body2 = sample_v2._audit_body({"a": 1, "b": 2})
    assert body1 == body2
    assert body1.endswith(b"\n")
    assert json.loads(body1.decode("utf-8")) == {"a": 1, "b": 2}


def test_audit_write_once_repeated_publish_does_not_rewrite(tmp_path):
    """Repeated publish with identical bytes is an idempotent no-op."""
    audit_path = tmp_path / "audit.json"
    body = sample_v2._audit_body({"x": 1})
    sample_v2._publish_audit(audit_path, body)
    old_mtime = audit_path.stat().st_mtime_ns
    time.sleep(0.01)
    sample_v2._publish_audit(audit_path, body)
    assert audit_path.stat().st_mtime_ns == old_mtime
    assert audit_path.read_bytes() == body


def test_audit_write_once_different_bytes_refused(tmp_path):
    """An existing audit with different bytes is hard-refused (no force)."""
    audit_path = tmp_path / "audit.json"
    sample_v2._publish_audit(audit_path, sample_v2._audit_body({"x": 1}))
    with pytest.raises(
        sample_v2.SampleError, match="publish preflight failed"
    ):
        sample_v2._publish_audit(audit_path, sample_v2._audit_body({"x": 2}))


def test_parser_has_no_no_audit_escape_hatch():
    """The --no-audit escape hatch is removed; audit is mandatory."""
    parser = sample_v2._build_parser()
    option_strings = {
        opt for action in parser._actions for opt in action.option_strings
    }
    assert "--no-audit" not in option_strings
    assert "--force" not in option_strings
    assert "--overwrite" not in option_strings


def test_audit_signature_includes_resolved_colbert_proxy():
    """The audit record carries the resolved ColBERT proxy explicitly."""
    import inspect

    sig = inspect.signature(sample_v2._audit)
    assert "resolved_colbert_proxy" in sig.parameters


def test_tie_break_is_deterministic_through_rank_and_chunk_id_lex(monkeypatch):
    """Two structurally identical allocations resolve deterministically.

    Objective 5 (seeded allocation rank) and objective 6 (stable chunk-ID
    lex order) must produce the same allocation on repeated runs, and the
    reported objective-6 component must equal the sorted selected chunk IDs.
    """
    monkeypatch.setattr(sample_v2, "SECTION_DEPTH_TARGETS", (0, 0, 12, 0))
    monkeypatch.setattr(sample_v2, "LENGTH_TARGETS", (4, 4, 4))
    papers = ("W_A", "W_B")
    quotas = {"W_A": 6, "W_B": 6}
    cells_by_paper = {
        "W_A": {
            (2, 0): [sample_v2.Chunk(f"c_aa{i:016x}", "W_A", 2, 0) for i in range(6)]
        },
        "W_B": {
            (2, 0): [sample_v2.Chunk(f"c_bb{i:016x}", "W_B", 2, 0) for i in range(6)]
        },
    }
    first = sample_v2._allocate(papers, quotas, cells_by_paper)
    second = sample_v2._allocate(papers, quotas, cells_by_paper)
    assert first.objective == second.objective
    assert first.objective[5] == tuple(
        sorted(chunk.chunk_id for chunk in first.selected)
    )


def test_same_marginal_witness_uses_stable_chunk_id_tie_break():
    """Equivalent contingency witnesses resolve by objective 6, not cell counts."""
    cells = {
        (1, 0): [sample_v2.Chunk("c_a000000000000000", "W_A", 1, 0)],
        (1, 1): [sample_v2.Chunk("c_z000000000000000", "W_A", 1, 1)],
        (2, 0): [sample_v2.Chunk("c_y000000000000000", "W_A", 2, 0)],
        (2, 1): [sample_v2.Chunk("c_b000000000000000", "W_A", 2, 1)],
    }

    witness = sample_v2._enumerate_witness(cells, 2)[((0, 1, 1, 0), (1, 1, 0))]

    assert sample_v2._witness_chunk_ids(cells, witness) == (
        "c_a000000000000000",
        "c_b000000000000000",
    )


def test_two_artifact_preflight_blocks_when_audit_differs(production_result, tmp_path):
    """Audit preflight conflict MUST prevent the output from being written.

    Setup:
    - output path absent
    - audit path exists with conflicting bytes

    Expected:
    - run_sample raises SampleError
    - output remains absent (no partial publication)
    - audit remains exactly as it was (unchanged bytes)
    """
    chunks, manifest, excluded, _, _, _, audit = production_result
    output = tmp_path / "sample.v2.jsonl"
    audit_path = tmp_path / "sample.v2.audit.json"
    audit_path.write_bytes(b'{"unrelated": "audit"}' + b"\n")
    audit_before = audit_path.read_bytes()
    chunks_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    ontology_path = tmp_path / "ontology.yaml"
    leakage_path = tmp_path / "held_out_leakage.v1.json"
    _write(chunks_path, chunks)
    _write(manifest_path, manifest)
    _write_ontology(ontology_path)
    _write_leakage(
        leakage_path,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        ontology_path=ontology_path,
        per_question=_default_question_map(excluded),
    )
    with pytest.raises(sample_v2.SampleError):
        sample_v2.run_sample(
            chunks_path,
            manifest_path,
            leakage_path,
            output,
            ontology_path=ontology_path,
            audit_path=audit_path,
        )
    assert not output.exists(), "output must not be created when audit preflight fails"
    assert audit_path.read_bytes() == audit_before, "audit must not be touched"


def test_two_artifact_preflight_blocks_when_output_differs(production_result, tmp_path):
    """Output preflight conflict MUST prevent the audit from being written.

    Setup:
    - output path exists with conflicting bytes
    - audit path absent

    Expected:
    - run_sample raises SampleError
    - output remains exactly as it was (unchanged bytes)
    - audit remains absent (no partial publication)
    """
    chunks, manifest, excluded, _, _, _, _ = production_result
    output = tmp_path / "sample.v2.jsonl"
    output.write_bytes(b"different\n")
    output_before = output.read_bytes()
    audit_path = tmp_path / "sample.v2.audit.json"
    chunks_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    ontology_path = tmp_path / "ontology.yaml"
    leakage_path = tmp_path / "held_out_leakage.v1.json"
    _write(chunks_path, chunks)
    _write(manifest_path, manifest)
    _write_ontology(ontology_path)
    _write_leakage(
        leakage_path,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        ontology_path=ontology_path,
        per_question=_default_question_map(excluded),
    )
    with pytest.raises(sample_v2.SampleError):
        sample_v2.run_sample(
            chunks_path,
            manifest_path,
            leakage_path,
            output,
            ontology_path=ontology_path,
            audit_path=audit_path,
        )
    assert output.read_bytes() == output_before, "output must not be touched"
    assert not audit_path.exists(), "audit must not be created when output preflight fails"


def test_two_artifact_preflight_writes_both_when_both_absent(production_result, tmp_path):
    """Sanity check: when both destinations are absent, both are written.

    The pair preflight must NOT regress the happy path.
    """
    chunks, manifest, excluded, _, payload, _, _ = production_result
    output = tmp_path / "sample.v2.jsonl"
    audit_path = tmp_path / "sample.v2.audit.json"
    chunks_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    ontology_path = tmp_path / "ontology.yaml"
    leakage_path = tmp_path / "held_out_leakage.v1.json"
    _write(chunks_path, chunks)
    _write(manifest_path, manifest)
    _write_ontology(ontology_path)
    _write_leakage(
        leakage_path,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        ontology_path=ontology_path,
        per_question=_default_question_map(excluded),
    )
    count = sample_v2.run_sample(
        chunks_path,
        manifest_path,
        leakage_path,
        output,
        ontology_path=ontology_path,
        audit_path=audit_path,
    )
    assert count == 120
    assert output.read_bytes() == payload
    assert audit_path.exists()


def test_two_artifact_preflight_is_idempotent_when_bytes_match(production_result, tmp_path):
    """When both destinations already hold the expected bytes, publication
    is a no-op for both (no rewrite, no mtime change).
    """
    chunks, manifest, excluded, _, payload, _, _ = production_result
    output = tmp_path / "sample.v2.jsonl"
    audit_path = tmp_path / "sample.v2.audit.json"
    chunks_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    ontology_path = tmp_path / "ontology.yaml"
    leakage_path = tmp_path / "held_out_leakage.v1.json"
    _write(chunks_path, chunks)
    _write(manifest_path, manifest)
    _write_ontology(ontology_path)
    _write_leakage(
        leakage_path,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        ontology_path=ontology_path,
        per_question=_default_question_map(excluded),
    )
    sample_v2.run_sample(
        chunks_path,
        manifest_path,
        leakage_path,
        output,
        ontology_path=ontology_path,
        audit_path=audit_path,
    )
    output_bytes = output.read_bytes()
    audit_bytes = audit_path.read_bytes()
    output_mtime = output.stat().st_mtime_ns
    audit_mtime = audit_path.stat().st_mtime_ns
    time.sleep(0.01)
    count = sample_v2.run_sample(
        chunks_path,
        manifest_path,
        leakage_path,
        output,
        ontology_path=ontology_path,
        audit_path=audit_path,
    )
    assert count == 120
    assert output.read_bytes() == output_bytes
    assert audit_path.read_bytes() == audit_bytes
    assert output.stat().st_mtime_ns == output_mtime
    assert audit_path.stat().st_mtime_ns == audit_mtime


def test_objective_5_formula_matches_specified_definition():
    """Direct unit test for the objective-5 allocation-rank formula.

    Per-paper rank = int(SHA256("9e3779b1|<paper_id>|<marginal>"))
    Allocation rank = sum(per-paper ranks) over the chosen paper marginals.
    Objective 6 = lex-sorted selected chunk IDs.

    The formula is exercised directly (without DP) so that any drift in
    the byte sequence is caught even when no allocation ties occur.
    """
    paper_id = "W3015883388"
    marginal = ((6, 24, 88, 2), (40, 40, 40))
    expected_per_paper = int.from_bytes(
        hashlib.sha256(b"9e3779b1|W3015883388|((6, 24, 88, 2), (40, 40, 40))").digest(),
        "big",
    )
    assert sample_v2._marginal_rank(paper_id, marginal) == expected_per_paper
    # Cross-check via raw hashlib SHA-256 to pin the exact encoding.
    raw_digest = hashlib.sha256(
        f"9e3779b1|{paper_id}|{marginal}".encode("utf-8")
    ).hexdigest()
    assert hex(sample_v2._marginal_rank(paper_id, marginal))[2:] == raw_digest
    # Allocation rank is the sum of per-paper ranks; verify the sum equals
    # int.from_bytes(...).big for an aggregation across two papers.
    second_marginal = ((2, 6, 4, 0), (4, 4, 4))
    second_rank = sample_v2._marginal_rank("W3039017601", second_marginal)
    allocation_rank = (
        sample_v2._marginal_rank(paper_id, marginal)
        + sample_v2._marginal_rank("W3039017601", second_marginal)
    )
    expected_total = expected_per_paper + second_rank
    assert allocation_rank == expected_total


def test_objective_6_is_lex_sorted_chunk_ids():
    """Objective 6 must equal the lex-sorted sequence of selected chunk IDs."""
    chunks = [
        sample_v2.Chunk("c_b000000000000000", "W_A", 1, 0),
        sample_v2.Chunk("c_a000000000000000", "W_A", 2, 1),
        sample_v2.Chunk("c_c000000000000000", "W_A", 2, 0),
    ]
    allocation = type("A", (), {"selected": tuple(chunks)})()
    objective6_value = tuple(sorted(chunk.chunk_id for chunk in allocation.selected))
    assert objective6_value == (
        "c_a000000000000000",
        "c_b000000000000000",
        "c_c000000000000000",
    )
    # Mirror the rule used by _selected_ids / _global_optimum: sort then
    # deduplicate through a stable merge (no duplicates here).
    merged = sample_v2._merge_sorted_ids(
        tuple(sorted(c.chunk_id for c in chunks[:2])),
        tuple(sorted(c.chunk_id for c in chunks[2:])),
    )
    assert merged == objective6_value
