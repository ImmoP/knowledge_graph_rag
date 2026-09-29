"""Tests for the deterministic benchmark dataset selector and loader."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from src.benchmark.dataset.manifest import (  # noqa: E402
    CHUNKS_PATH,
    SHA256_PATH,
    MANIFEST_PATH,
    load_frozen_chunks,
    verify_dataset_files,
)
from src.benchmark.dataset.select import (  # noqa: E402
    DEFAULT_TARGET_SIZE,
    SAMPLER_SEED,
    SELECTION_ALGORITHM_VERSION,
    SelectionConfig,
    select_chunks,
    write_selection_artifacts,
)
from src.benchmark.dataset.smoke_chunks import SMOKE_CHUNK_IDS, SMOKE_MAX_LIVE_CALLS  # noqa: E402


def _write_synthetic_corpus(path: Path, *, n_docs: int = 12, per_doc: int = 60) -> None:
    """Write a tiny but realistic synthetic chunk corpus to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    seq = 0
    for d in range(n_docs):
        doc_id = f"W{d:09d}"
        for i in range(per_doc):
            seq += 1
            cid = f"c_{seq:016x}"
            if i % 7 == 0:
                # prose chunk (no LaTeX commands at all)
                text = (
                    "This is a sentence of plain prose without any LaTeX commands. "
                    * 4
                )
            elif i % 5 == 0:
                # table chunk (carries table LaTeX)
                text = (
                    "\\begin{table}[t]\n\\centering\n\\caption{Results.}\n"
                    "\\begin{tabular}{rr}\n\\hline\nA & B \\\\\n\\hline\n1 & 2 \\\\\n"
                    "\\hline\n\\end{tabular}\n\\end{table}\n"
                )
            else:
                # latex prose
                text = (
                    "We use \\cite{foo2020} and \\begin{equation} x = 1 \\end{equation} "
                    * 4
                )
            lines.append(json.dumps({
                "chunk_id": cid,
                "doc_id": doc_id,
                "section_path": [f"sec_{i}"],
                "chunk_index": i,
                "text": text,
            }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def synthetic_corpus(tmp_path: Path) -> Path:
    p = tmp_path / "chunks.jsonl"
    _write_synthetic_corpus(p, n_docs=12, per_doc=60)
    return p


def _default_cfg() -> SelectionConfig:
    return SelectionConfig(target_size=30, min_documents=6, per_doc_cap=8)


def test_default_target_is_within_required_range():
    assert 30 <= DEFAULT_TARGET_SIZE <= 50


def test_sampler_seed_is_pinned():
    assert SAMPLER_SEED == 0x5EEDF00D


def test_algorithm_version_is_pinned():
    assert SELECTION_ALGORITHM_VERSION.startswith("benchmark-dataset-selector/")


def test_smoke_chunks_are_pinned():
    assert SMOKE_CHUNK_IDS == ("c_0001921855a8156b", "c_001452fd142ce075")
    assert SMOKE_MAX_LIVE_CALLS == 4


def test_selector_is_deterministic(synthetic_corpus: Path):
    cfg = _default_cfg()
    a, ra = select_chunks(synthetic_corpus, config=cfg)
    b, rb = select_chunks(synthetic_corpus, config=cfg)
    assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]
    assert ra.chunks_sha256 == rb.chunks_sha256


def test_selector_respects_required_ids(synthetic_corpus: Path):
    cfg = _default_cfg()
    chunks, _ = select_chunks(synthetic_corpus, config=cfg)
    head = [c["chunk_id"] for c in chunks[:2]]
    pinned, _ = select_chunks(synthetic_corpus, config=cfg, required_chunk_ids=head)
    pinned_ids = [c["chunk_id"] for c in pinned[:2]]
    assert pinned_ids == head


def test_selector_rejects_missing_required_id(synthetic_corpus: Path):
    with pytest.raises(KeyError):
        select_chunks(synthetic_corpus, required_chunk_ids=("c_does_not_exist",))


def test_selector_enforces_min_documents(synthetic_corpus: Path):
    with pytest.raises(RuntimeError):
        select_chunks(
            synthetic_corpus,
            config=SelectionConfig(target_size=30, min_documents=200, per_doc_cap=10),
        )


def test_selector_enforces_tag_floors(synthetic_corpus: Path):
    """Demanding 100% prose is rejected against a corpus that is
    only ~14% prose by construction."""
    with pytest.raises(RuntimeError):
        select_chunks(
            synthetic_corpus,
            config=SelectionConfig(
                target_size=30,
                min_documents=4,
                per_doc_cap=8,
                min_prose_fraction=1.0,
            ),
        )


def test_selector_enforces_per_doc_cap(synthetic_corpus: Path):
    chunks, report = select_chunks(
        synthetic_corpus,
        config=SelectionConfig(target_size=30, min_documents=10, per_doc_cap=3),
    )
    from collections import Counter
    counts = Counter(c["doc_id"] for c in chunks)
    assert all(v <= 3 for v in counts.values())
    assert len(report.documents_selected) >= 10


def test_selector_emits_table_latex_prose_mix(synthetic_corpus: Path):
    chunks, report = select_chunks(synthetic_corpus, config=_default_cfg())
    tags = report.counts_by_tag
    assert tags["latex"] >= int(0.2 * len(chunks))
    assert tags["table"] >= int(0.1 * len(chunks))


def test_write_artifacts_produces_verifiable_dataset(synthetic_corpus: Path, tmp_path: Path):
    chunks, report = select_chunks(synthetic_corpus, config=_default_cfg())
    out_jsonl = tmp_path / "dataset" / "benchmark_chunks.jsonl"
    out_sha = tmp_path / "dataset" / "benchmark_chunks.sha256"
    out_manifest = tmp_path / "dataset" / "benchmark_dataset_manifest.json"
    write_selection_artifacts(
        chunks, report,
        output_jsonl=out_jsonl, output_sha256=out_sha, output_manifest=out_manifest,
    )
    sha_written = out_sha.read_text().strip().split()[0]
    assert sha_written == report.chunks_sha256
    manifest_payload = json.loads(out_manifest.read_text())
    assert manifest_payload["chunks_sha256"] == report.chunks_sha256
    import hashlib
    assert hashlib.sha256(out_jsonl.read_bytes()).hexdigest() == sha_written
    on_disk = [json.loads(line) for line in out_jsonl.read_text().splitlines() if line.strip()]
    assert [c["chunk_id"] for c in on_disk] == list(report.selected_chunk_ids)


def test_write_artifacts_detects_sha_drift(synthetic_corpus: Path, tmp_path: Path):
    chunks, report = select_chunks(synthetic_corpus, config=_default_cfg())
    from dataclasses import replace
    bad_report = replace(report, chunks_sha256="0" * 64)
    with pytest.raises(RuntimeError):
        write_selection_artifacts(
            chunks, bad_report,
            output_jsonl=tmp_path / "d.jsonl",
            output_sha256=tmp_path / "d.sha256",
            output_manifest=tmp_path / "d.manifest.json",
        )


def test_frozen_dataset_loads_and_verifies():
    """The committed benchmark dataset must load and verify."""
    if not (CHUNKS_PATH.is_file() and SHA256_PATH.is_file() and MANIFEST_PATH.is_file()):
        pytest.skip("frozen dataset not present (run build_dataset.py first)")
    chunks, manifest = load_frozen_chunks()
    assert 30 <= len(chunks) <= 50
    assert manifest.algorithm_version == SELECTION_ALGORITHM_VERSION
    assert SMOKE_CHUNK_IDS[0] in {c["chunk_id"] for c in chunks[:2]}
    assert SMOKE_CHUNK_IDS[1] in {c["chunk_id"] for c in chunks[:2]}
    verify_dataset_files()