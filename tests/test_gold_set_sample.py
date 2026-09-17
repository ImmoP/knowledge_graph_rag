from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from itertools import product
from pathlib import Path

import pytest

from scripts.gold_set import sample


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


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
    question_map = {
        qid: sorted(set(ids))
        for qid, ids in sorted(per_question.items())
    }
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


def _corpus() -> tuple[list[dict], list[dict]]:
    papers = [*sample.CENTRAL_PAPERS, *(f"W{1000 + index}" for index in range(30))]
    manifest = [
        {"openalex_id": paper_id, "cited_by_count": index}
        for index, paper_id in enumerate(papers)
    ]
    chunks: list[dict] = []
    serial = 0
    for paper_id in papers:
        for depth in range(4):
            for size in (1000, 1500, 1800):
                for _ in range(4):
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
    central_chunks = [
        row["chunk_id"]
        for row in chunks
        if row["doc_id"] == sample.CENTRAL_PAPERS[0]
    ]
    return central_chunks[:2]


@pytest.fixture(scope="module")
def production_result(tmp_path_factory):
    root = tmp_path_factory.mktemp("production")
    chunks, manifest = _corpus()
    excluded = _default_exclusions(chunks)
    paths = {
        "chunks": root / "chunks.jsonl",
        "manifest": root / "manifest.jsonl",
        "leakage": root / "held_out_leakage.v1.json",
        "ontology": root / "ontology.yaml",
        "output": root / "sample.jsonl",
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
    assert sample.run_sample(
        paths["chunks"],
        paths["manifest"],
        paths["leakage"],
        paths["output"],
        ontology_path=paths["ontology"],
    ) == 120
    payload = paths["output"].read_bytes()
    rows = [json.loads(line) for line in payload.splitlines()]
    return chunks, manifest, excluded, paths, payload, rows


def test_frozen_constants():
    assert sample.GOLD_SET_SEED == 0x9E3779B1
    assert sample.SAMPLER_SEED_TEXT == "9e3779b1"
    assert sample.SAMPLER_VERSION == "gold-set-sample.v1"
    assert sample.SAMPLE_SIZE == 120
    assert sample.CENTRAL_PAPERS == ("W3015883388", "W3039017601", "W3021397474")


def test_production_contract(production_result):
    chunks, _, excluded, _, payload, rows = production_result
    ids = [row["chunk_id"] for row in rows]
    paper_counts = Counter(row["doc_id"] for row in rows)
    tier_counts = Counter(row["strata"]["paper_tier"] for row in rows)
    assert len(rows) == len(set(ids)) == 120
    assert ids == sorted(ids)
    assert payload.endswith(b"\n")
    assert [paper_counts[paper_id] for paper_id in sample.CENTRAL_PAPERS] == [20, 20, 20]
    assert sorted(
        count
        for paper_id, count in paper_counts.items()
        if paper_id not in sample.CENTRAL_PAPERS
    ) == [12] * 5
    assert tier_counts == {"benchmark_central": 60, "mid_density": 36, "long_tail": 24}
    assert Counter(row["strata"]["section_depth"] for row in rows) == {
        0: 6,
        1: 24,
        2: 88,
        3: 2,
    }
    assert Counter(row["strata"]["length_tercile"] for row in rows) == {
        "short": 40,
        "medium": 40,
        "long": 40,
    }
    assert {row["strata"]["density_post_stratum"] for row in rows} == {"pending"}
    assert all(
        set(row) == {"chunk_id", "doc_id", "strata", "sampler_seed", "sampler_version"}
        for row in rows
    )
    assert all(
        set(row["strata"])
        == {"paper_tier", "section_depth", "length_tercile", "density_post_stratum"}
        for row in rows
    )
    assert set(excluded).isdisjoint(ids)
    assert paper_counts[sample.CENTRAL_PAPERS[0]] == 20
    assert all(
        "text" not in row and "title" not in row and "question" not in row
        for row in rows
    )
    assert {row["sampler_seed"] for row in rows} == {"9e3779b1"}
    assert {row["sampler_version"] for row in rows} == {"gold-set-sample.v1"}
    assert {row["chunk_id"] for row in rows} <= {row["chunk_id"] for row in chunks}


def test_repeated_run_is_byte_identical_and_does_not_rewrite(production_result):
    _, _, _, paths, payload, _ = production_result
    old_mtime = paths["output"].stat().st_mtime_ns
    time.sleep(0.001)
    assert sample.run_sample(
        paths["chunks"],
        paths["manifest"],
        paths["leakage"],
        paths["output"],
        ontology_path=paths["ontology"],
    ) == 120
    assert paths["output"].read_bytes() == payload
    assert paths["output"].stat().st_mtime_ns == old_mtime


def test_shuffled_inputs_with_refrozen_identity_are_byte_identical(
    production_result, tmp_path
):
    chunks, manifest, excluded, _, payload, _ = production_result
    rng = random.Random(20260917)
    shuffled_chunks = chunks.copy()
    shuffled_manifest = manifest.copy()
    rng.shuffle(shuffled_chunks)
    rng.shuffle(shuffled_manifest)

    chunks_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    ontology_path = tmp_path / "ontology.yaml"
    leakage_path = tmp_path / "held_out_leakage.v1.json"
    output = tmp_path / "sample.jsonl"
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
    sample.run_sample(
        chunks_path,
        manifest_path,
        leakage_path,
        output,
        ontology_path=ontology_path,
    )
    assert output.read_bytes() == payload


def test_source_draft_identity_does_not_affect_sampling(production_result, tmp_path):
    chunks, manifest, excluded, _, payload, _ = production_result
    chunks_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    ontology_path = tmp_path / "ontology.yaml"
    leakage_path = tmp_path / "held_out_leakage.v1.json"
    output = tmp_path / "sample.jsonl"
    _write(chunks_path, chunks)
    _write(manifest_path, manifest)
    _write_ontology(ontology_path)
    _write_leakage(
        leakage_path,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        ontology_path=ontology_path,
        per_question=_default_question_map(excluded),
        source_draft_sha256="f" * 64,
    )
    sample.run_sample(
        chunks_path,
        manifest_path,
        leakage_path,
        output,
        ontology_path=ontology_path,
    )
    assert output.read_bytes() == payload


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload.update(record_count=44), "record_count"),
        (lambda payload: payload.update(all_split_held_out=False), "all_split_held_out"),
        (lambda payload: payload.update(schema_version="held-out-leakage.v2"), "schema_version"),
    ],
)
def test_leakage_preflight_failures(production_result, tmp_path, mutator, message):
    chunks, manifest, excluded, _, _, _ = production_result
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
    payload = json.loads(leakage_path.read_text(encoding="utf-8"))
    mutator(payload)
    leakage_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(sample.SampleError, match=message):
        sample.run_sample(
            chunks_path,
            manifest_path,
            leakage_path,
            tmp_path / "out",
            ontology_path=ontology_path,
        )


def test_unknown_frozen_chunk_fails(tmp_path):
    chunks, manifest = _corpus()
    chunks_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    ontology_path = tmp_path / "ontology.yaml"
    leakage_path = tmp_path / "held_out_leakage.v1.json"
    _write(chunks_path, chunks)
    _write(manifest_path, manifest)
    _write_ontology(ontology_path)
    mapping = _default_question_map(_default_exclusions(chunks))
    mapping["q_001"] = ["c_ffffffffffffffff"]
    _write_leakage(
        leakage_path,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        ontology_path=ontology_path,
        per_question=mapping,
    )
    with pytest.raises(sample.SampleError, match="unknown chunks"):
        sample.run_sample(
            chunks_path,
            manifest_path,
            leakage_path,
            tmp_path / "out",
            ontology_path=ontology_path,
        )


def test_stale_leakage_identity_fails(tmp_path):
    chunks, manifest = _corpus()
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
        per_question=_default_question_map(_default_exclusions(chunks)),
    )
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    with pytest.raises(sample.SampleError, match="stale leakage artifact"):
        sample.run_sample(
            chunks_path,
            manifest_path,
            leakage_path,
            tmp_path / "out",
            ontology_path=ontology_path,
        )


def test_questions_jsonl_is_not_an_accepted_sampler_input(tmp_path):
    chunks, manifest = _corpus()
    chunks_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    ontology_path = tmp_path / "ontology.yaml"
    questions_path = tmp_path / "questions.jsonl"
    _write(chunks_path, chunks)
    _write(manifest_path, manifest)
    _write_ontology(ontology_path)
    _write(
        questions_path,
        [
            {
                "qid": f"q_{index:03d}",
                "supporting_chunk_ids": [_default_exclusions(chunks)[0]],
                "split": "held_out",
            }
            for index in range(1, 46)
        ],
    )
    with pytest.raises(sample.SampleError, match="malformed leakage artifact"):
        sample.run_sample(
            chunks_path,
            manifest_path,
            questions_path,
            tmp_path / "out",
            ontology_path=ontology_path,
        )


def test_missing_central_paper_fails(tmp_path):
    chunks, manifest = _corpus()
    manifest = [
        row for row in manifest if row["openalex_id"] != sample.CENTRAL_PAPERS[0]
    ]
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
        per_question=_default_question_map(_default_exclusions(chunks)),
    )
    with pytest.raises(sample.SampleError, match="central paper missing"):
        sample.run_sample(
            chunks_path,
            manifest_path,
            leakage_path,
            tmp_path / "out",
            ontology_path=ontology_path,
        )


def test_central_paper_below_quota_fails(tmp_path):
    chunks, manifest = _corpus()
    central = sample.CENTRAL_PAPERS[0]
    kept = [row for row in chunks if row["doc_id"] != central][:]
    central_rows = [row for row in chunks if row["doc_id"] == central][:20]
    kept += central_rows
    excluded = [row["chunk_id"] for row in central_rows[:2]]
    chunks_path = tmp_path / "chunks"
    manifest_path = tmp_path / "manifest"
    ontology_path = tmp_path / "ontology"
    leakage_path = tmp_path / "leakage"
    _write(chunks_path, kept)
    _write(manifest_path, manifest)
    _write_ontology(ontology_path)
    _write_leakage(
        leakage_path,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        ontology_path=ontology_path,
        per_question=_default_question_map(excluded),
    )
    with pytest.raises(sample.SampleError, match="leakage-safe chunks"):
        sample.run_sample(
            chunks_path,
            manifest_path,
            leakage_path,
            tmp_path / "out",
            ontology_path=ontology_path,
        )


def _run_pool_failure(tmp_path, chunks, manifest, message):
    chunks_path = tmp_path / "chunks"
    manifest_path = tmp_path / "manifest"
    ontology_path = tmp_path / "ontology"
    leakage_path = tmp_path / "leakage"
    _write(chunks_path, chunks)
    _write(manifest_path, manifest)
    _write_ontology(ontology_path)
    central_chunks = [
        row["chunk_id"]
        for row in chunks
        if row["doc_id"] == sample.CENTRAL_PAPERS[0]
    ]
    _write_leakage(
        leakage_path,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        ontology_path=ontology_path,
        per_question=_default_question_map(central_chunks[:1]),
    )
    with pytest.raises(sample.SampleError, match=message):
        sample.run_sample(
            chunks_path,
            manifest_path,
            leakage_path,
            tmp_path / "out",
            ontology_path=ontology_path,
        )


def test_insufficient_mid_density_pool_fails(tmp_path):
    chunks, manifest = _corpus()
    keep = {*sample.CENTRAL_PAPERS, "W1000", "W1001", "W1002", "W1003"}
    chunks = [row for row in chunks if row["doc_id"] in keep]
    manifest = [row for row in manifest if row["openalex_id"] in keep]
    _run_pool_failure(tmp_path, chunks, manifest, "insufficient mid-density")


def test_insufficient_long_tail_pool_fails(tmp_path):
    chunks, manifest = _corpus()
    keep = {*sample.CENTRAL_PAPERS, *(f"W{1000 + index}" for index in range(10))}
    chunks = [row for row in chunks if row["doc_id"] in keep]
    manifest = [row for row in manifest if row["openalex_id"] in keep]
    _run_pool_failure(tmp_path, chunks, manifest, "insufficient long-tail")


def test_paper_options_seededly_resolves_same_marginal_witnesses():
    paper_id = "W59"
    cells = {
        (0, 0): [sample.Chunk("c_0000000000000000", paper_id, 0, 0)],
        (0, 1): [sample.Chunk("c_0000000000000001", paper_id, 0, 1)],
        (1, 0): [sample.Chunk("c_0000000000000002", paper_id, 1, 0)],
        (1, 1): [sample.Chunk("c_0000000000000003", paper_id, 1, 1)],
    }
    target_marginal = (1, 1, 0, 0, 1, 1, 0)
    witnesses = []
    for counts in product(range(2), repeat=4):
        witness = (
            counts[0],
            counts[1],
            0,
            counts[2],
            counts[3],
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        marginal = (
            sum(witness[0:3]),
            sum(witness[3:6]),
            0,
            0,
            sum(witness[0::3]),
            sum(witness[1::3]),
            sum(witness[2::3]),
        )
        if sum(witness) == 2 and marginal == target_marginal:
            witnesses.append(witness)

    assert len(witnesses) > 1
    expected = min(
        witnesses,
        key=lambda witness: sample._rank(
            "paper-cell-witness", f"{paper_id}|{witness}"
        ),
    )
    assert expected != min(witnesses)
    options = dict(sample._paper_options(paper_id, 2, cells))
    assert options[target_marginal] == expected


def test_impossible_joint_allocation_has_diagnostics():
    selected = [*sample.CENTRAL_PAPERS, "W1", "W2", "W3", "W4", "W5"]
    tiers = {
        paper: "benchmark_central"
        if paper in sample.CENTRAL_PAPERS
        else "mid_density"
        for paper in selected
    }
    chunks = []
    serial = 0
    for paper in selected:
        quota = 20 if paper in sample.CENTRAL_PAPERS else 12
        for index in range(quota):
            chunks.append(
                sample.Chunk(
                    f"c_{serial:016x}",
                    paper,
                    2,
                    index % 3,
                )
            )
            serial += 1
    with pytest.raises(sample.SampleError) as error:
        sample._allocate(selected, tiers, chunks, set())
    message = str(error.value)
    assert "selected papers" in message
    assert "available section-depth totals" in message
    assert "required length totals" in message
    assert "tier=" in message and "quota=" in message and "eligible=" in message


def test_different_existing_output_is_refused(tmp_path):
    output = tmp_path / "sample.jsonl"
    output.write_bytes(b"different\n")
    with pytest.raises(sample.SampleError, match="refusing to overwrite"):
        sample._publish(output, b"expected\n")
    assert output.read_bytes() == b"different\n"


def test_no_force_or_legacy_questions_escape_hatch():
    parser = sample._build_parser()
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


def test_bool_rejected_where_integer_required(tmp_path):
    _, manifest = _corpus()
    manifest[0]["cited_by_count"] = True
    _write(tmp_path / "manifest", manifest)
    with pytest.raises(sample.SampleError, match="bool rejected"):
        sample._load_manifest(tmp_path / "manifest")


def test_malformed_jsonl_fails(tmp_path):
    path = tmp_path / "chunks"
    path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(sample.SampleError, match="malformed JSONL"):
        sample._load_chunks(path)


def test_missing_and_empty_canonical_inputs_fail(tmp_path):
    with pytest.raises(sample.SampleError, match="required input missing"):
        sample._load_chunks(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(sample.SampleError, match="input is empty"):
        sample._load_manifest(empty)


def test_sampler_does_not_read_candidate_extractions(
    monkeypatch, production_result, tmp_path
):
    chunks, manifest, excluded, _, _, _ = production_result
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

    def guarded(self, *args, **kwargs):
        assert "extractions" not in self.name
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    sample.run_sample(
        chunks_path,
        manifest_path,
        leakage_path,
        tmp_path / "output",
        ontology_path=ontology_path,
    )
