from __future__ import annotations

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


def _questions(chunks: list[dict], count: int = 45) -> list[dict]:
    supporting = next(
        row["chunk_id"] for row in chunks if row["doc_id"] == sample.CENTRAL_PAPERS[0]
    )
    evidence = next(
        row["chunk_id"]
        for row in chunks
        if row["doc_id"] == sample.CENTRAL_PAPERS[0] and row["chunk_id"] != supporting
    )
    return [
        {
            "qid": f"q_{index:03d}",
            "question": "Synthetic held-out question",
            "expected_answer": "Synthetic answer",
            "hop_bucket": "1",
            "evidence_path": [
                {
                    "step": 1,
                    "from": evidence,
                    "relation": "CITES",
                    "to": sample.CENTRAL_PAPERS[0],
                    "direction": "forward",
                }
            ],
            # Deliberately names a selected paper: document-level exclusion is forbidden.
            "supporting_doc_ids": [sample.CENTRAL_PAPERS[0]],
            "supporting_chunk_ids": [supporting],
            "split": "held_out",
        }
        for index in range(count)
    ]


@pytest.fixture(scope="module")
def production_result(tmp_path_factory):
    root = tmp_path_factory.mktemp("production")
    chunks, manifest = _corpus()
    questions = _questions(chunks)
    paths = {
        "chunks": root / "chunks.jsonl",
        "manifest": root / "manifest.jsonl",
        "questions": root / "questions.jsonl",
        "output": root / "sample.jsonl",
    }
    _write(paths["chunks"], chunks)
    _write(paths["manifest"], manifest)
    _write(paths["questions"], questions)
    assert sample.run_sample(*paths.values()) == 120
    payload = paths["output"].read_bytes()
    rows = [json.loads(line) for line in payload.splitlines()]
    return chunks, manifest, questions, paths, payload, rows


def test_frozen_constants():
    assert sample.GOLD_SET_SEED == 0x9E3779B1
    assert sample.SAMPLER_SEED_TEXT == "9e3779b1"
    assert sample.SAMPLER_VERSION == "gold-set-sample.v1"
    assert sample.SAMPLE_SIZE == 120
    assert sample.CENTRAL_PAPERS == ("W3015883388", "W3039017601", "W3021397474")


def test_production_contract(production_result):
    chunks, _, questions, _, payload, rows = production_result
    ids = [row["chunk_id"] for row in rows]
    paper_counts = Counter(row["doc_id"] for row in rows)
    tier_counts = Counter(row["strata"]["paper_tier"] for row in rows)
    assert len(rows) == len(set(ids)) == 120
    assert ids == sorted(ids)
    assert payload.endswith(b"\n")
    assert [paper_counts[paper_id] for paper_id in sample.CENTRAL_PAPERS] == [20, 20, 20]
    assert sorted(count for paper_id, count in paper_counts.items() if paper_id not in sample.CENTRAL_PAPERS) == [12] * 5
    assert tier_counts == {"benchmark_central": 60, "mid_density": 36, "long_tail": 24}
    assert Counter(row["strata"]["section_depth"] for row in rows) == {0: 6, 1: 24, 2: 88, 3: 2}
    assert Counter(row["strata"]["length_tercile"] for row in rows) == {"short": 40, "medium": 40, "long": 40}
    assert {row["strata"]["density_post_stratum"] for row in rows} == {"pending"}
    assert all(set(row) == {"chunk_id", "doc_id", "strata", "sampler_seed", "sampler_version"} for row in rows)
    assert all(set(row["strata"]) == {"paper_tier", "section_depth", "length_tercile", "density_post_stratum"} for row in rows)
    excluded = {
        questions[0]["supporting_chunk_ids"][0],
        questions[0]["evidence_path"][0]["from"],
    }
    assert excluded.isdisjoint(ids)
    assert paper_counts[sample.CENTRAL_PAPERS[0]] == 20
    assert all("text" not in row and "title" not in row and "question" not in row for row in rows)
    assert {row["sampler_seed"] for row in rows} == {"9e3779b1"}
    assert {row["sampler_version"] for row in rows} == {"gold-set-sample.v1"}
    assert {row["chunk_id"] for row in rows} <= {row["chunk_id"] for row in chunks}


def test_repeated_run_is_byte_identical_and_does_not_rewrite(production_result):
    _, _, _, paths, payload, _ = production_result
    old_mtime = paths["output"].stat().st_mtime_ns
    time.sleep(0.001)
    assert sample.run_sample(*paths.values()) == 120
    assert paths["output"].read_bytes() == payload
    assert paths["output"].stat().st_mtime_ns == old_mtime


def test_shuffled_inputs_are_byte_identical(production_result, tmp_path):
    chunks, manifest, questions, _, payload, _ = production_result
    rng = random.Random(20260917)
    for rows, name in ((chunks, "chunks"), (manifest, "manifest"), (questions, "questions")):
        shuffled = rows.copy()
        rng.shuffle(shuffled)
        _write(tmp_path / f"{name}.jsonl", shuffled)
    output = tmp_path / "sample.jsonl"
    sample.run_sample(
        tmp_path / "chunks.jsonl",
        tmp_path / "manifest.jsonl",
        tmp_path / "questions.jsonl",
        output,
    )
    assert output.read_bytes() == payload

def test_irrelevant_held_out_content_does_not_change_sample(tmp_path):
    chunks, manifest = _corpus()
    questions = _questions(chunks)
    altered_questions = json.loads(json.dumps(questions))
    for index, question in enumerate(altered_questions):
        question.update(
            qid={"not": f"a-qid-{index}"},
            question=["not", "a", "question"],
            expected_answer={"not": "an answer"},
            answer_aliases=None,
            hop_bucket={"not": "a bucket"},
            supporting_doc_ids="not-a-document-list",
        )
        question["evidence_path"][0].update(
            step=["not", "a", "step"],
            relation={"not": "a relation"},
            direction=None,
        )

    _write(tmp_path / "chunks.jsonl", chunks)
    _write(tmp_path / "manifest.jsonl", manifest)
    _write(tmp_path / "questions.jsonl", questions)
    _write(tmp_path / "altered-questions.jsonl", altered_questions)
    original_output = tmp_path / "original.jsonl"
    altered_output = tmp_path / "altered.jsonl"
    sample.run_sample(
        tmp_path / "chunks.jsonl",
        tmp_path / "manifest.jsonl",
        tmp_path / "questions.jsonl",
        original_output,
    )
    sample.run_sample(
        tmp_path / "chunks.jsonl",
        tmp_path / "manifest.jsonl",
        tmp_path / "altered-questions.jsonl",
        altered_output,
    )
    assert altered_output.read_bytes() == original_output.read_bytes()



@pytest.mark.parametrize("count", [0, 44, 46])
def test_wrong_question_count_fails_without_writing(tmp_path, count):
    chunks, manifest = _corpus()
    _write(tmp_path / "chunks.jsonl", chunks)
    _write(tmp_path / "manifest.jsonl", manifest)
    _write(tmp_path / "questions.jsonl", _questions(chunks, count))
    output = tmp_path / "sample.jsonl"
    with pytest.raises(sample.SampleError, match="exactly 45"):
        sample.run_sample(tmp_path / "chunks.jsonl", tmp_path / "manifest.jsonl", tmp_path / "questions.jsonl", output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda questions: questions[0].update(split="dev"), "split"),
        (lambda questions: questions[0].update(supporting_chunk_ids=[]), "must be non-empty"),
        (lambda questions: questions[0].update(supporting_chunk_ids=["c_ffffffffffffffff"]), "unknown chunks"),
    ],
)
def test_question_preflight_failures(tmp_path, mutation, message):
    chunks, manifest = _corpus()
    questions = _questions(chunks)
    mutation(questions)
    _write(tmp_path / "chunks.jsonl", chunks)
    _write(tmp_path / "manifest.jsonl", manifest)
    _write(tmp_path / "questions.jsonl", questions)
    with pytest.raises(sample.SampleError, match=message):
        sample.run_sample(tmp_path / "chunks.jsonl", tmp_path / "manifest.jsonl", tmp_path / "questions.jsonl", tmp_path / "out")


def test_unknown_evidence_chunk_fails(tmp_path):
    chunks, manifest = _corpus()
    questions = _questions(chunks)
    questions[0]["evidence_path"][0]["from"] = "c_ffffffffffffffff"
    for path, rows in ((tmp_path / "chunks", chunks), (tmp_path / "manifest", manifest), (tmp_path / "questions", questions)):
        _write(path, rows)
    with pytest.raises(sample.SampleError, match="unknown chunks"):
        sample.run_sample(tmp_path / "chunks", tmp_path / "manifest", tmp_path / "questions", tmp_path / "out")


def test_missing_central_paper_fails(tmp_path):
    chunks, manifest = _corpus()
    manifest = [row for row in manifest if row["openalex_id"] != sample.CENTRAL_PAPERS[0]]
    for path, rows in ((tmp_path / "chunks", chunks), (tmp_path / "manifest", manifest), (tmp_path / "questions", _questions(chunks))):
        _write(path, rows)
    with pytest.raises(sample.SampleError, match="central paper missing"):
        sample.run_sample(tmp_path / "chunks", tmp_path / "manifest", tmp_path / "questions", tmp_path / "out")


def test_central_paper_below_quota_fails(tmp_path):
    chunks, manifest = _corpus()
    central = sample.CENTRAL_PAPERS[0]
    kept = [row for row in chunks if row["doc_id"] != central][:]
    kept += [row for row in chunks if row["doc_id"] == central][:20]
    questions = _questions(chunks)
    # One support and one evidence exclusion reduce 20 to 18.
    for path, rows in ((tmp_path / "chunks", kept), (tmp_path / "manifest", manifest), (tmp_path / "questions", questions)):
        _write(path, rows)
    with pytest.raises(sample.SampleError, match="leakage-safe chunks"):
        sample.run_sample(tmp_path / "chunks", tmp_path / "manifest", tmp_path / "questions", tmp_path / "out")


def test_insufficient_mid_density_pool_fails(tmp_path):
    chunks, manifest = _corpus()
    keep = {*sample.CENTRAL_PAPERS, "W1000", "W1001", "W1002", "W1003"}
    chunks = [row for row in chunks if row["doc_id"] in keep]
    manifest = [row for row in manifest if row["openalex_id"] in keep]
    questions = _questions(chunks)
    for path, rows in ((tmp_path / "chunks", chunks), (tmp_path / "manifest", manifest), (tmp_path / "questions", questions)):
        _write(path, rows)
    with pytest.raises(sample.SampleError, match="insufficient mid-density"):
        sample.run_sample(tmp_path / "chunks", tmp_path / "manifest", tmp_path / "questions", tmp_path / "out")




def test_insufficient_long_tail_pool_fails(tmp_path):
    chunks, manifest = _corpus()
    keep = {*sample.CENTRAL_PAPERS, *(f"W{1000 + index}" for index in range(10))}
    chunks = [row for row in chunks if row["doc_id"] in keep]
    manifest = [row for row in manifest if row["openalex_id"] in keep]
    questions = _questions(chunks)
    for path, rows in ((tmp_path / "chunks", chunks), (tmp_path / "manifest", manifest), (tmp_path / "questions", questions)):
        _write(path, rows)
    with pytest.raises(sample.SampleError, match="insufficient long-tail"):
        sample.run_sample(tmp_path / "chunks", tmp_path / "manifest", tmp_path / "questions", tmp_path / "out")


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
        witness = (counts[0], counts[1], 0, counts[2], counts[3], 0, 0, 0, 0, 0, 0, 0)
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
        key=lambda witness: sample._rank("paper-cell-witness", f"{paper_id}|{witness}"),
    )
    assert expected != min(witnesses)
    options = dict(sample._paper_options(paper_id, 2, cells))
    assert options[target_marginal] == expected


def test_impossible_joint_allocation_has_diagnostics():
    selected = [*sample.CENTRAL_PAPERS, "W1", "W2", "W3", "W4", "W5"]
    tiers = {paper: "benchmark_central" if paper in sample.CENTRAL_PAPERS else "mid_density" for paper in selected}
    chunks = []
    serial = 0
    # Enough chunks per paper, but no depth-0 chunks anywhere.
    for paper in selected:
        quota = 20 if paper in sample.CENTRAL_PAPERS else 12
        for index in range(quota):
            chunks.append(sample.Chunk(f"c_{serial:016x}", paper, 2, index % 3))
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


def test_no_force_or_overwrite_escape_hatch():
    parser = sample._build_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}
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


def test_sampler_does_not_read_candidate_extractions(monkeypatch, production_result, tmp_path):
    chunks, manifest, questions, _, _, _ = production_result
    paths = [tmp_path / name for name in ("chunks", "manifest", "questions")]
    for path, rows in zip(paths, (chunks, manifest, questions)):
        _write(path, rows)
    original = Path.read_text

    def guarded(self, *args, **kwargs):
        assert "extractions" not in self.name
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    sample.run_sample(*paths, tmp_path / "output")
