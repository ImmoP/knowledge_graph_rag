"""Deterministic, leakage-safe 120-chunk gold-set sampler (#59)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.leakage import (  # noqa: E402
    DEFAULT_ONTOLOGY_PATH,
    LeakageError,
    load_frozen_exclusions,
)

GOLD_SET_SEED = 0x9E3779B1
SAMPLER_SEED_TEXT = "9e3779b1"
SAMPLER_VERSION = "gold-set-sample.v1"
SAMPLE_SIZE = 120

CENTRAL_PAPERS = ("W3015883388", "W3039017601", "W3021397474")
CENTRAL_QUOTA = 20
NON_CENTRAL_QUOTA = 12
SECTION_DEPTH_TARGETS = (6, 24, 88, 2)
LENGTH_TARGETS = (40, 40, 40)
LENGTH_LABELS = ("short", "medium", "long")

CHUNK_ID_RE = re.compile(r"^c_[0-9a-f]{16}$")
DOC_ID_RE = re.compile(r"^W\d+$")


class SampleError(RuntimeError):
    """Raised when preflight, selection, allocation, or publication fails."""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    section_depth: int
    length_index: int

    @property
    def length_label(self) -> str:
        return LENGTH_LABELS[self.length_index]


@dataclass(frozen=True)
class ManifestPaper:
    openalex_id: str
    cited_by_count: int


def _rank(namespace: str, stable_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{SAMPLER_SEED_TEXT}|{namespace}|{stable_id}".encode("utf-8")
    ).hexdigest()
    return digest, stable_id


def _read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict]:
    if not path.exists():
        raise SampleError(f"required input missing: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SampleError(f"non-UTF-8 input {path}: {exc}") from exc
    records: list[dict] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SampleError(f"malformed JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise SampleError(f"{path}:{line_number}: record must be an object")
        records.append(value)
    if not records and not allow_empty:
        raise SampleError(f"input is empty: {path}")
    return records


def _string(record: dict, key: str, where: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise SampleError(f"{where}: {key} must be a non-empty string")
    return value


def _integer(record: dict, key: str, where: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SampleError(f"{where}: {key} must be an integer (bool rejected)")
    return value




def _load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for line_number, record in enumerate(_read_jsonl(path), 1):
        where = f"{path}:{line_number}"
        chunk_id = _string(record, "chunk_id", where)
        doc_id = _string(record, "doc_id", where)
        if not CHUNK_ID_RE.fullmatch(chunk_id):
            raise SampleError(f"{where}: invalid chunk_id {chunk_id!r}")
        if not DOC_ID_RE.fullmatch(doc_id):
            raise SampleError(f"{where}: invalid doc_id {doc_id!r}")
        if chunk_id in seen:
            raise SampleError(f"{where}: duplicate chunk_id {chunk_id}")
        seen.add(chunk_id)
        section_path = record.get("section_path")
        if not isinstance(section_path, list) or any(
            not isinstance(part, str) or not part for part in section_path
        ):
            raise SampleError(f"{where}: section_path must be a list of non-empty strings")
        text = _string(record, "text", where)
        _integer(record, "chunk_index", where)
        _integer(record, "token_estimate", where)
        _integer(record, "char_start", where)
        _integer(record, "char_end", where)
        length_index = 0 if len(text) < 1200 else 1 if len(text) <= 1700 else 2
        chunks.append(Chunk(chunk_id, doc_id, min(len(section_path), 3), length_index))
    return chunks


def _load_manifest(path: Path) -> list[ManifestPaper]:
    papers: list[ManifestPaper] = []
    seen: set[str] = set()
    for line_number, record in enumerate(_read_jsonl(path), 1):
        where = f"{path}:{line_number}"
        openalex_id = _string(record, "openalex_id", where)
        if not DOC_ID_RE.fullmatch(openalex_id):
            raise SampleError(f"{where}: invalid openalex_id {openalex_id!r}")
        if openalex_id in seen:
            raise SampleError(f"{where}: duplicate openalex_id {openalex_id}")
        seen.add(openalex_id)
        cited_by_count = _integer(record, "cited_by_count", where)
        papers.append(ManifestPaper(openalex_id, cited_by_count))
    return papers



def _select_papers(
    manifest: Sequence[ManifestPaper],
    chunks: Sequence[Chunk],
    excluded: set[str],
) -> tuple[list[str], list[str]]:
    manifest_ids = {paper.openalex_id for paper in manifest}
    counts: dict[str, int] = {}
    for chunk in chunks:
        if chunk.chunk_id not in excluded:
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
    for paper_id in CENTRAL_PAPERS:
        if paper_id not in manifest_ids:
            raise SampleError(f"central paper missing from manifest: {paper_id}")
        if counts.get(paper_id, 0) < CENTRAL_QUOTA:
            raise SampleError(
                f"central paper {paper_id} has {counts.get(paper_id, 0)} leakage-safe chunks; "
                f"need {CENTRAL_QUOTA}"
            )

    eligible = sorted(
        (
            paper
            for paper in manifest
            if paper.openalex_id not in CENTRAL_PAPERS
            and counts.get(paper.openalex_id, 0) >= NON_CENTRAL_QUOTA
        ),
        key=lambda paper: (paper.cited_by_count, paper.openalex_id),
    )

    mid_pool = [
        paper
        for index, paper in enumerate(eligible)
        if 3 * index // len(eligible) in (1, 2)
    ] if eligible else []
    mid = [
        paper.openalex_id
        for paper in sorted(
            mid_pool,
            key=lambda paper: _rank("non-central-mid", paper.openalex_id),
        )[:3]
    ]
    if len(mid) != 3:
        raise SampleError(f"insufficient mid-density pool: required 3 papers; found {len(mid)}")

    long_pool = [
        paper
        for index, paper in enumerate(eligible)
        if 10 * index // len(eligible) == 0 and paper.openalex_id not in mid
    ]
    long = [
        paper.openalex_id
        for paper in sorted(
            long_pool,
            key=lambda paper: _rank("non-central-long", paper.openalex_id),
        )[:2]
    ]
    if len(long) != 2:
        raise SampleError(f"insufficient long-tail pool: required 2 papers; found {len(long)}")
    return mid, long


def _paper_options(
    paper_id: str,
    quota: int,
    cells: dict[tuple[int, int], list[Chunk]],
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return (marginal state, 12-cell witness) options for one paper."""
    # State: (used, depth0..3, length0..2) -> cell-count witness.
    states: dict[tuple[int, ...], tuple[int, ...]] = {(0, 0, 0, 0, 0, 0, 0, 0): ()}
    cell_keys = [(depth, length) for depth in range(4) for length in range(3)]
    for depth, length in cell_keys:
        capacity = len(cells.get((depth, length), ()))
        next_states: dict[tuple[int, ...], tuple[int, ...]] = {}
        for state, witness in states.items():
            used = state[0]
            maximum = min(
                capacity,
                quota - used,
                SECTION_DEPTH_TARGETS[depth] - state[1 + depth],
                LENGTH_TARGETS[length] - state[5 + length],
            )
            for count in range(maximum + 1):
                values = list(state)
                values[0] += count
                values[1 + depth] += count
                values[5 + length] += count
                key = tuple(values)
                candidate = witness + (count,)
                previous = next_states.get(key)
                if previous is None or _rank(
                    "paper-cell-witness", f"{paper_id}|{candidate}"
                ) < _rank("paper-cell-witness", f"{paper_id}|{previous}"):
                    next_states[key] = candidate
        states = next_states
    options = []
    for state, witness in states.items():
        if state[0] == quota:
            marginal = state[1:]
            options.append((marginal, witness))
    options.sort(key=lambda option: _rank("paper-allocation", f"{paper_id}|{option[0]}|{option[1]}"))
    return options


def _allocate(
    selected_papers: Sequence[str],
    tiers: dict[str, str],
    chunks: Sequence[Chunk],
    excluded: set[str],
) -> list[Chunk]:
    quotas = {
        paper_id: CENTRAL_QUOTA if paper_id in CENTRAL_PAPERS else NON_CENTRAL_QUOTA
        for paper_id in selected_papers
    }
    cells_by_paper: dict[str, dict[tuple[int, int], list[Chunk]]] = {
        paper_id: {} for paper_id in selected_papers
    }
    for chunk in chunks:
        if chunk.doc_id in cells_by_paper and chunk.chunk_id not in excluded:
            cells_by_paper[chunk.doc_id].setdefault(
                (chunk.section_depth, chunk.length_index), []
            ).append(chunk)
    for cells in cells_by_paper.values():
        for cell in cells.values():
            cell.sort(key=lambda chunk: _rank("chunk", chunk.chunk_id))

    paper_options = {
        paper_id: _paper_options(paper_id, quotas[paper_id], cells_by_paper[paper_id])
        for paper_id in selected_papers
    }
    ordered_papers = sorted(
        selected_papers,
        key=lambda paper_id: (
            len(paper_options[paper_id]),
            _rank("paper-search-order", paper_id),
        ),
    )
    suffix_depth = [[0, 0, 0, 0] for _ in range(len(ordered_papers) + 1)]
    suffix_length = [[0, 0, 0] for _ in range(len(ordered_papers) + 1)]
    for index in range(len(ordered_papers) - 1, -1, -1):
        paper_id = ordered_papers[index]
        cells = cells_by_paper[paper_id]
        suffix_depth[index] = suffix_depth[index + 1].copy()
        suffix_length[index] = suffix_length[index + 1].copy()
        for (depth, length), cell in cells.items():
            suffix_depth[index][depth] += len(cell)
            suffix_length[index][length] += len(cell)

    memo: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()

    def search(
        index: int,
        depth_left: tuple[int, ...],
        length_left: tuple[int, ...],
    ) -> list[tuple[str, tuple[int, ...]]] | None:
        if index == len(ordered_papers):
            return [] if not any(depth_left) and not any(length_left) else None
        key = (index, depth_left, length_left)
        if key in memo:
            return None
        if any(depth_left[i] > suffix_depth[index][i] for i in range(4)) or any(
            length_left[i] > suffix_length[index][i] for i in range(3)
        ):
            memo.add(key)
            return None
        paper_id = ordered_papers[index]
        for marginal, witness in paper_options[paper_id]:
            depth = marginal[:4]
            length = marginal[4:]
            if any(depth[i] > depth_left[i] for i in range(4)) or any(
                length[i] > length_left[i] for i in range(3)
            ):
                continue
            result = search(
                index + 1,
                tuple(depth_left[i] - depth[i] for i in range(4)),
                tuple(length_left[i] - length[i] for i in range(3)),
            )
            if result is not None:
                return [(paper_id, witness), *result]
        memo.add(key)
        return None

    allocation = search(0, SECTION_DEPTH_TARGETS, LENGTH_TARGETS)
    if allocation is None:
        lines = ["joint allocation failed:", "selected papers:"]
        for paper_id in selected_papers:
            eligible = sum(len(cell) for cell in cells_by_paper[paper_id].values())
            lines.append(
                f"  {paper_id} tier={tiers[paper_id]} quota={quotas[paper_id]} eligible={eligible}"
            )
        available_depth = [0, 0, 0, 0]
        available_length = [0, 0, 0]
        for cells in cells_by_paper.values():
            for (depth, length), cell in cells.items():
                available_depth[depth] += len(cell)
                available_length[length] += len(cell)
        lines.extend(
            (
                f"available section-depth totals: {available_depth}",
                f"available length totals: {available_length}",
                f"required section-depth totals: {list(SECTION_DEPTH_TARGETS)}",
                f"required length totals: {list(LENGTH_TARGETS)}",
            )
        )
        raise SampleError("\n".join(lines))

    selected: list[Chunk] = []
    cell_keys = [(depth, length) for depth in range(4) for length in range(3)]
    for paper_id, witness in allocation:
        for cell_key, count in zip(cell_keys, witness):
            selected.extend(cells_by_paper[paper_id].get(cell_key, ())[:count])
    return selected


def _record(chunk: Chunk, tier: str) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "strata": {
            "paper_tier": tier,
            "section_depth": chunk.section_depth,
            "length_tercile": chunk.length_label,
            "density_post_stratum": "pending",
        },
        "sampler_seed": SAMPLER_SEED_TEXT,
        "sampler_version": SAMPLER_VERSION,
    }


def _serialize(records: Sequence[dict]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def _publish(output_path: Path, payload: bytes) -> None:
    if output_path.exists():
        if output_path.read_bytes() == payload:
            return
        raise SampleError(f"output exists with different bytes; refusing to overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, output_path)
    except OSError as exc:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        raise SampleError(f"failed to publish {output_path}: {exc}") from exc


def run_sample(
    chunks_path: Path,
    manifest_path: Path,
    leakage_path: Path,
    output_path: Path,
    *,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
) -> int:
    chunks = _load_chunks(chunks_path)
    manifest = _load_manifest(manifest_path)
    try:
        exclusions = load_frozen_exclusions(
            leakage_path,
            chunks_path=chunks_path,
            manifest_path=manifest_path,
            ontology_path=ontology_path,
        )
    except LeakageError as exc:
        raise SampleError(str(exc)) from exc
    missing = sorted(exclusions - {chunk.chunk_id for chunk in chunks})
    if missing:
        raise SampleError(f"leakage artifact references unknown chunks: {missing[:5]}")
    mid, long = _select_papers(manifest, chunks, exclusions)
    selected_papers = [*CENTRAL_PAPERS, *mid, *long]
    tiers = {
        **{paper_id: "benchmark_central" for paper_id in CENTRAL_PAPERS},
        **{paper_id: "mid_density" for paper_id in mid},
        **{paper_id: "long_tail" for paper_id in long},
    }
    selected = _allocate(selected_papers, tiers, chunks, exclusions)
    records = sorted((_record(chunk, tiers[chunk.doc_id]) for chunk in selected), key=lambda r: r["chunk_id"])
    if len(records) != SAMPLE_SIZE or len({record["chunk_id"] for record in records}) != SAMPLE_SIZE:
        raise SampleError(f"allocator returned {len(records)} records; required {SAMPLE_SIZE} distinct records")
    _publish(output_path, _serialize(records))
    return len(records)


def _default_paths() -> dict[str, Path]:
    return {
        "chunks": ROOT / "data" / "derived" / "chunks.jsonl",
        "manifest": ROOT / "data" / "manifest.jsonl",
        "leakage": ROOT / "eval" / "held_out_leakage.v1.json",
        "ontology": ROOT / "ontology.yaml",
        "output": ROOT / "data" / "gold_set" / "sample.v1.jsonl",
    }


def _build_parser() -> argparse.ArgumentParser:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description="Deterministic 120-chunk gold-set sampler")
    parser.add_argument("--chunks", type=Path, default=defaults["chunks"])
    parser.add_argument("--manifest", type=Path, default=defaults["manifest"])
    parser.add_argument("--leakage", type=Path, default=defaults["leakage"])
    parser.add_argument("--ontology", type=Path, default=defaults["ontology"])
    parser.add_argument("--output", type=Path, default=defaults["output"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        count = run_sample(
            args.chunks,
            args.manifest,
            args.leakage,
            args.output,
            ontology_path=args.ontology,
        )
    except SampleError as exc:
        print(f"sample failed: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {count} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
