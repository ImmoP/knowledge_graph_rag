"""Deterministic, leakage-safe 120-chunk gold-set sampler v2 (#63).

Preserves the v1 hard invariants and replaces the impossible exact structural
marginals with a deterministic nearest-feasible allocation. Implements the
central proxy resolution and the predeclared central quota rule from #48
Refinement 8 / 8a.

Structural-allocation tie-break (objectives 5 and 6):

    per-paper rank =
        int(SHA256("9e3779b1|<paper_id>|<marginal>"))

    allocation rank =
        sum(per-paper ranks)  over the chosen paper marginals

    objective 6 =
        lexicographically sorted selected chunk IDs

The DP enforces: first structural (depth/length) objective, then objective 5,
then objective 6.
"""

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
SAMPLER_VERSION = "gold-set-sample.v2"
SAMPLE_SIZE = 120
NON_CENTRAL_QUOTA = 12
CENTRAL_TOTAL = 60
CENTRAL_NOMINAL_QUOTA = 20

CENTRAL_ANCHORS = ("W3015883388", "W3039017601")
COLBERT_PROXY_CANDIDATES = ("W3021397474", "W3038831159", "W3217305727")

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


def _safe_capacities(
    chunks: Sequence[Chunk], papers: Sequence[str], excluded: set[str]
) -> dict[str, int]:
    paper_set = set(papers)
    counts: dict[str, int] = {}
    for chunk in chunks:
        if chunk.doc_id in paper_set and chunk.chunk_id not in excluded:
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
    return counts


def _resolve_colbert_proxy(
    canonical_counts: dict[str, int],
) -> tuple[str, int]:
    """Resolve the ColBERT-family proxy from canonical chunks only (#48 Refinement 8).

    Candidates are evaluated in the pinned order
    ``(W3021397474, W3038831159, W3217305727)``. A candidate is *available*
    iff it has at least one canonical chunk in the chunks artifact, computed
    BEFORE held-out exclusions. Manifest presence is intentionally NOT a
    prerequisite for proxy resolution.

    The returned count is the canonical-chunk count (pre-exclusion); the
    leakage-safe capacity is computed separately by the caller.
    """
    for candidate in COLBERT_PROXY_CANDIDATES:
        canonical = canonical_counts.get(candidate, 0)
        if canonical > 0:
            return candidate, canonical
    raise SampleError(
        "no ColBERT-family candidate is available; fail closed"
    )

def _central_quotas(
    capacities: dict[str, int],
    *,
    canonical_counts: dict[str, int] | None = None,
    manifest_ids: set[str] | None = None,
) -> dict[str, int]:
    """Apply the #48 Refinement 8a central quota rule.

    Deficient anchors (< nominal 20) are fixed at full safe capacity.
    Remaining deficit is distributed among non-deficient anchors subject to
    capacity, minimizing lexicographically:
        1. maximum surplus over 20,
        2. sum of squared surpluses over 20,
        3. stable anchor-ID deterministic tie-break.
    Fail closed if the 60-central total is not capacity-feasible.

    The DPR/RAG central anchors MUST still appear in the manifest; the
    ColBERT-family proxy is resolved by canonical-chunk availability only
    (#48 Refinement 8). When ``canonical_counts`` is provided, the proxy
    resolution uses those pre-exclusion canonical counts. Otherwise the
    proxy is resolved from the leakage-safe capacities already passed in
    (unit-test convenience).
    """
    deficient: dict[str, int] = {}
    non_deficient: list[tuple[str, int]] = []
    for anchor_id in CENTRAL_ANCHORS:
        if manifest_ids is not None and anchor_id not in manifest_ids:
            raise SampleError(
                f"central anchor {anchor_id} missing from manifest; fail closed"
            )
        safe = capacities.get(anchor_id, 0)
        if safe == 0:
            raise SampleError(
                f"central anchor {anchor_id} has zero leakage-safe chunks; fail closed"
            )
        if safe < CENTRAL_NOMINAL_QUOTA:
            deficient[anchor_id] = safe
        else:
            non_deficient.append((anchor_id, safe))

    if canonical_counts is not None:
        proxy_id, proxy_safe = _resolve_colbert_proxy(canonical_counts)
        proxy_capacity_for_quota = capacities.get(proxy_id, 0)
    else:
        proxy_id, proxy_safe = _resolve_colbert_proxy(capacities)
        proxy_capacity_for_quota = proxy_safe
    if proxy_id in deficient:
        raise SampleError(
            f"ColBERT-family proxy {proxy_id} duplicates deficient anchor"
        )
    if proxy_capacity_for_quota < CENTRAL_NOMINAL_QUOTA:
        deficient[proxy_id] = proxy_capacity_for_quota
    else:
        non_deficient.append((proxy_id, proxy_capacity_for_quota))

    fixed_total = sum(deficient.values())
    if fixed_total > CENTRAL_TOTAL:
        raise SampleError(
            f"central total infeasible: deficient anchors ({fixed_total}) "
            f"exceed {CENTRAL_TOTAL}"
        )

    remaining = CENTRAL_TOTAL - fixed_total
    quotas = dict(deficient)
    if remaining == 0:
        return quotas

    non_deficient.sort(key=lambda entry: entry[0])
    feasible_total = sum(cap for _, cap in non_deficient)
    if remaining > feasible_total:
        raise SampleError(
            f"central total infeasible: need {remaining} more chunks; "
            f"non-deficient anchors offer {feasible_total}"
        )

    # Enumerate every capacity-feasible distribution among non-deficient anchors.
    options: list[tuple[int, int, str, tuple[tuple[str, int], ...]]] = []

    def enumerate_split(position: int, leftover: int, current: list[tuple[str, int]]) -> None:
        if position == len(non_deficient):
            if leftover == 0:
                surpluses = [
                    max(0, take - CENTRAL_NOMINAL_QUOTA) for _, take in current
                ]
                max_surplus = max(surpluses)
                sum_sq = sum(s * s for s in surpluses)
                canonical = "|".join(
                    f"{anchor}:{take}" for anchor, take in current
                )
                options.append((max_surplus, sum_sq, canonical, tuple(current)))
            return
        anchor_id, capacity = non_deficient[position]
        upper = min(capacity, leftover)
        for take in range(upper + 1):
            current.append((anchor_id, take))
            enumerate_split(position + 1, leftover - take, current)
            current.pop()

    enumerate_split(0, remaining, [])
    if not options:
        raise SampleError("central total infeasible: no capacity-feasible split found")
    options.sort(key=lambda option: (option[0], option[1], option[2]))
    best_split = options[0][3]
    quotas.update(dict(best_split))
    return quotas


def _select_papers(
    manifest: Sequence[ManifestPaper],
    chunks: Sequence[Chunk],
    excluded: set[str],
    central_ids: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Five non-central papers via v1 methodology, excluding the resolved central IDs."""
    counts = _safe_capacities(chunks, [paper.openalex_id for paper in manifest], excluded)
    central_set = set(central_ids)
    eligible = sorted(
        (
            paper
            for paper in manifest
            if paper.openalex_id not in central_set
            and counts.get(paper.openalex_id, 0) >= NON_CENTRAL_QUOTA
        ),
        key=lambda paper: (paper.cited_by_count, paper.openalex_id),
    )

    mid_pool = [
        paper
        for index, paper in enumerate(eligible)
        if eligible and 3 * index // len(eligible) in (1, 2)
    ]
    mid = [
        paper.openalex_id
        for paper in sorted(
            mid_pool,
            key=lambda paper: _rank("non-central-mid", paper.openalex_id),
        )[:3]
    ]
    if len(mid) != 3:
        raise SampleError(
            f"insufficient mid-density pool: required 3 papers; found {len(mid)}"
        )

    long_pool = [
        paper
        for index, paper in enumerate(eligible)
        if eligible and 10 * index // len(eligible) == 0 and paper.openalex_id not in mid
    ]
    long = [
        paper.openalex_id
        for paper in sorted(
            long_pool,
            key=lambda paper: _rank("non-central-long", paper.openalex_id),
        )[:2]
    ]
    if len(long) != 2:
        raise SampleError(
            f"insufficient long-tail pool: required 2 papers; found {len(long)}"
        )
    return mid, long


def _cells_by_paper(
    selected_papers: Sequence[str],
    chunks: Sequence[Chunk],
    excluded: set[str],
) -> dict[str, dict[tuple[int, int], list[Chunk]]]:
    cells: dict[str, dict[tuple[int, int], list[Chunk]]] = {
        paper_id: {} for paper_id in selected_papers
    }
    for chunk in chunks:
        if chunk.doc_id in cells and chunk.chunk_id not in excluded:
            cells[chunk.doc_id].setdefault(
                (chunk.section_depth, chunk.length_index), []
            ).append(chunk)
    for paper_cells in cells.values():
        for cell in paper_cells.values():
            cell.sort(key=lambda chunk: chunk.chunk_id)
    return cells


def _cell_keys() -> list[tuple[int, int]]:
    return [(depth, length) for depth in range(4) for length in range(3)]


def _enumerate_marginals(
    cells: dict[tuple[int, int], list[Chunk]],
    quota: int,
) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Enumerate exact-quota (depth, length) marginals for one paper.

    Witnesses (which specific chunks fill each cell) are NOT tracked here.
    Witnesses are resolved later for the globally chosen marginal(s) only.
    """
    cell_keys = _cell_keys()
    capacities = [len(cells.get(key, ())) for key in cell_keys]
    states: set[tuple[int, int, int, int, int, int]] = {(0, 0, 0, 0, 0, 0)}
    for cell_idx in range(12):
        depth, length = cell_keys[cell_idx]
        capacity = capacities[cell_idx]
        next_states: set[tuple[int, int, int, int, int, int]] = set()
        for used, d0, d1, d2, l0, l1 in states:
            d3 = used - d0 - d1 - d2
            l2 = used - l0 - l1
            if d3 < 0 or l2 < 0:
                continue
            max_count = min(capacity, quota - used)
            for count in range(max_count + 1):
                next_states.add(
                    (
                        used + count,
                        d0 + (count if depth == 0 else 0),
                        d1 + (count if depth == 1 else 0),
                        d2 + (count if depth == 2 else 0),
                        l0 + (count if length == 0 else 0),
                        l1 + (count if length == 1 else 0),
                    )
                )
        states = next_states
        if not states:
            break
    marginals: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for used, d0, d1, d2, l0, l1 in states:
        d3 = used - d0 - d1 - d2
        l2 = used - l0 - l1
        if used != quota or d3 < 0 or l2 < 0:
            continue
        marginals.add(((d0, d1, d2, d3), (l0, l1, l2)))
    return marginals


def _witness_chunk_ids(
    cells: dict[tuple[int, int], list[Chunk]], witness: tuple[int, ...]
) -> tuple[str, ...]:
    """Return a witness's deterministic, stable-sorted selected chunk IDs."""
    return tuple(
        sorted(
            chunk.chunk_id
            for cell, count in zip(_cell_keys(), witness)
            for chunk in cells.get(cell, ())[:count]
        )
    )


def _enumerate_witness(
    cells: dict[tuple[int, int], list[Chunk]],
    quota: int,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[int, ...]]:
    """Enumerate exact-quota marginals with their chunk-ID-minimal witness.

    Different 12-cell contingency tables can have the same depth and length
    marginals. Their objective-5 rank is consequently identical, so retain
    the witness with the lexicographically smallest stable-sorted chunk IDs:
    objective 6, not a cell-count ordering.
    """
    cell_keys = _cell_keys()
    states: dict[tuple[int, int, int, int, int, int], tuple[tuple[int, ...], tuple[str, ...]]] = {
        (0, 0, 0, 0, 0, 0): ((), ())
    }
    for depth, length in cell_keys:
        cell = cells.get((depth, length), ())
        next_states: dict[
            tuple[int, int, int, int, int, int], tuple[tuple[int, ...], tuple[str, ...]]
        ] = {}
        for (used, d0, d1, d2, l0, l1), (witness, selected_ids) in states.items():
            for count in range(min(len(cell), quota - used) + 1):
                key = (
                    used + count,
                    d0 + (count if depth == 0 else 0),
                    d1 + (count if depth == 1 else 0),
                    d2 + (count if depth == 2 else 0),
                    l0 + (count if length == 0 else 0),
                    l1 + (count if length == 1 else 0),
                )
                candidate = (*witness, count), tuple(
                    sorted((*selected_ids, *(chunk.chunk_id for chunk in cell[:count])))
                )
                existing = next_states.get(key)
                if existing is None or candidate[1] < existing[1]:
                    next_states[key] = candidate
        states = next_states
    witnesses: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[str, ...]]] = {}
    for (used, d0, d1, d2, l0, l1), (witness, selected_ids) in states.items():
        d3, l2 = used - d0 - d1 - d2, used - l0 - l1
        if used != quota or d3 < 0 or l2 < 0:
            continue
        marginal = ((d0, d1, d2, d3), (l0, l1, l2))
        existing = witnesses.get(marginal)
        if existing is None or selected_ids < existing[1]:
            witnesses[marginal] = (witness, selected_ids)
    return {marginal: witness for marginal, (witness, _) in witnesses.items()}


def _structure_objective(
    actual_depth: tuple[int, ...], actual_length: tuple[int, ...]
) -> tuple[int, int, int, int]:
    depth_l1 = sum(abs(actual_depth[i] - SECTION_DEPTH_TARGETS[i]) for i in range(4))
    length_l1 = sum(abs(actual_length[j] - LENGTH_TARGETS[j]) for j in range(3))
    combined_l1 = depth_l1 + length_l1
    max_dev = max(
        *(abs(actual_depth[i] - SECTION_DEPTH_TARGETS[i]) for i in range(4)),
        *(abs(actual_length[j] - LENGTH_TARGETS[j]) for j in range(3)),
    )
    return (combined_l1, max_dev, depth_l1, length_l1)


@dataclass(frozen=True)
class _PaperOption:
    """Precomputed immutable DP payload for one feasible paper marginal."""

    marginal: tuple[tuple[int, ...], tuple[int, ...]]
    witness: tuple[int, ...]
    chunk_ids: tuple[str, ...]
    rank: int
    delta: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class _DPValue:
    seeded_rank: int
    previous: _DPValue | None
    paper_id: str | None
    option: _PaperOption | None


def _selected_ids(value: _DPValue) -> tuple[str, ...]:
    """Materialize objective 6 only when objective 5 has tied."""
    option_ids: list[tuple[str, ...]] = []
    while value.option is not None:
        option_ids.append(value.option.chunk_ids)
        assert value.previous is not None
        value = value.previous
    selected_ids: tuple[str, ...] = ()
    for chunk_ids in reversed(option_ids):
        selected_ids = _merge_sorted_ids(selected_ids, chunk_ids)
    return selected_ids


def _better_value(candidate: _DPValue, existing: _DPValue | None) -> bool:
    """Compare the objective-5/6 representative for one aggregate state."""
    return existing is None or candidate.seeded_rank < existing.seeded_rank or (
        candidate.seeded_rank == existing.seeded_rank
        and _selected_ids(candidate) < _selected_ids(existing)
    )




def _merge_sorted_ids(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    """Merge disjoint sorted chunk-ID sequences without re-sorting them."""
    merged: list[str] = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] < right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1
    merged.extend(left[left_index:])
    merged.extend(right[right_index:])
    return tuple(merged)


def _full_marginals(
    state: tuple[int, int, int, int, int], processed_quota: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Restore dependent depth-3 and length-2 coordinates for a DP state."""
    d0, d1, d2, l0, l1 = state
    return (d0, d1, d2, processed_quota - d0 - d1 - d2), (l0, l1, processed_quota - l0 - l1)


def _suffix_lower_bound(
    state: tuple[int, int, int, int, int],
    processed_quota: int,
    suffix_bound: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> tuple[int, int, int, int]:
    """Admissible structural objective lower bound for all completions."""
    depth, length = _full_marginals(state, processed_quota)
    minimum_depth, minimum_length, maximum_depth, maximum_length = suffix_bound
    depth_deviations = tuple(
        max(
            depth[i] + minimum_depth[i] - SECTION_DEPTH_TARGETS[i],
            SECTION_DEPTH_TARGETS[i] - (depth[i] + maximum_depth[i]),
            0,
        )
        for i in range(4)
    )
    length_deviations = tuple(
        max(
            length[i] + minimum_length[i] - LENGTH_TARGETS[i],
            LENGTH_TARGETS[i] - (length[i] + maximum_length[i]),
            0,
        )
        for i in range(3)
    )
    return (
        sum(depth_deviations) + sum(length_deviations),
        max(*depth_deviations, *length_deviations),
        sum(depth_deviations),
        sum(length_deviations),
    )


def _marginal_rank(paper_id: str, marginal: tuple[tuple[int, ...], tuple[int, ...]]) -> int:
    """Return objective 5's per-paper rank for one feasible marginal.

    Objective 5 (allocation-rank tie-break) is defined as:

        per-paper rank =
            int(SHA256("9e3779b1" + "|" + paper_id + "|" + str(marginal)))

    The allocation rank for an allocation is the SUM of per-paper ranks
    across every selected paper marginal. Objective 6 is the lex-sorted
    sequence of selected chunk IDs. The DP minimizes (struct obj,
    allocation rank, lex-sorted chunk IDs) in that order.
    """
    payload = f"{SAMPLER_SEED_TEXT}|{paper_id}|{marginal}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _global_optimum(
    ordered_papers: Sequence[str],
    cells_by_paper: dict[str, dict[tuple[int, int], list[Chunk]]],
    quotas: dict[str, int],
) -> tuple[
    tuple[int, int, int, int, int, tuple[str, ...]],
    dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
    dict[str, tuple[int, ...]],
]:
    """Find the exact global optimum with aggregate-marginal dynamic programming."""
    paper_options: dict[str, tuple[_PaperOption, ...]] = {}
    for paper_id in ordered_papers:
        options = tuple(
            _PaperOption(
                marginal=marginal,
                witness=witness,
                chunk_ids=_witness_chunk_ids(cells_by_paper[paper_id], witness),
                rank=_marginal_rank(paper_id, marginal),
                delta=(*marginal[0][:3], *marginal[1][:2]),
            )
            for marginal, witness in _enumerate_witness(
                cells_by_paper[paper_id], quotas[paper_id]
            ).items()
        )
        if not options:
            raise SampleError(f"paper {paper_id} has no quota-feasible marginals")
        paper_options[paper_id] = options

    dp_order = sorted(ordered_papers, key=lambda paper_id: (len(paper_options[paper_id]), paper_id))
    print("marginal options:", file=sys.stderr)
    for paper_id in ordered_papers:
        print(
            f"  {paper_id}: quota={quotas[paper_id]} options={len(paper_options[paper_id])}",
            file=sys.stderr,
        )
    print(f"DP processing order: {', '.join(dp_order)}", file=sys.stderr)

    suffix_bounds: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = [
        ((0, 0, 0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0))
        for _ in range(len(dp_order) + 1)
    ]
    for position in range(len(dp_order) - 1, -1, -1):
        minimum_depth, minimum_length, maximum_depth, maximum_length = suffix_bounds[position + 1]
        options = paper_options[dp_order[position]]
        suffix_bounds[position] = (
            tuple(minimum_depth[i] + min(option.marginal[0][i] for option in options) for i in range(4)),
            tuple(minimum_length[i] + min(option.marginal[1][i] for option in options) for i in range(3)),
            tuple(maximum_depth[i] + max(option.marginal[0][i] for option in options) for i in range(4)),
            tuple(maximum_length[i] + max(option.marginal[1][i] for option in options) for i in range(3)),
        )

    cheap_depth = tuple(sum(paper_options[paper_id][0].marginal[0][i] for paper_id in ordered_papers) for i in range(4))
    cheap_length = tuple(sum(paper_options[paper_id][0].marginal[1][i] for paper_id in ordered_papers) for i in range(3))
    cheap_incumbent = _structure_objective(cheap_depth, cheap_length)
    beam: dict[tuple[int, int, int, int, int], _DPValue] = {
        (0, 0, 0, 0, 0): _DPValue(0, None, None, None)
    }
    processed_quota = 0
    for index, paper_id in enumerate(dp_order, 1):
        processed_quota += quotas[paper_id]
        candidates: dict[tuple[int, int, int, int, int], _DPValue] = {}
        for state, value in beam.items():
            for option in paper_options[paper_id]:
                candidate_state = tuple(state[i] + option.delta[i] for i in range(5))
                candidate = _DPValue(value.seeded_rank + option.rank, value, paper_id, option)
                if _better_value(candidate, candidates.get(candidate_state)):
                    candidates[candidate_state] = candidate
        beam = dict(sorted(
            candidates.items(),
            key=lambda item: (
                _suffix_lower_bound(item[0], processed_quota, suffix_bounds[index]),
                item[1].seeded_rank,
                _selected_ids(item[1]),
            ),
        )[:256])
    improved_state = min(beam, key=lambda state: _structure_objective(*_full_marginals(state, processed_quota)))
    incumbent = _structure_objective(*_full_marginals(improved_state, processed_quota))
    print(f"initial cheap incumbent: {cheap_incumbent}", file=sys.stderr)
    print(f"improved feasible incumbent: {incumbent}", file=sys.stderr)

    states: dict[tuple[int, int, int, int, int], _DPValue] = {
        (0, 0, 0, 0, 0): _DPValue(0, None, None, None)
    }
    processed_quota = 0
    transitions = 0
    for index, paper_id in enumerate(dp_order, 1):
        incoming_states = len(states)
        processed_quota += quotas[paper_id]
        candidates: dict[tuple[int, int, int, int, int], _DPValue] = {}
        for state, value in states.items():
            for option in paper_options[paper_id]:
                transitions += 1
                candidate_state = tuple(state[i] + option.delta[i] for i in range(5))
                candidate = _DPValue(value.seeded_rank + option.rank, value, paper_id, option)
                if _better_value(candidate, candidates.get(candidate_state)):
                    candidates[candidate_state] = candidate
        states = {
            state: value
            for state, value in candidates.items()
            if _suffix_lower_bound(state, processed_quota, suffix_bounds[index]) <= incumbent
        }
        print(
            f"paper {index} {paper_id}: options={len(paper_options[paper_id])} "
            f"incoming_states={incoming_states} unique_candidate_states={len(candidates)} "
            f"surviving_states={len(states)}",
            file=sys.stderr,
        )
    print(f"candidate transitions evaluated: {transitions}", file=sys.stderr)
    if not states:
        raise SampleError("safe DP pruning removed every complete allocation")

    best_state, best_value = min(
        states.items(),
        key=lambda item: (
            *_structure_objective(*_full_marginals(item[0], processed_quota)),
            item[1].seeded_rank,
            _selected_ids(item[1]),
        ),
    )
    objective = (
        *_structure_objective(*_full_marginals(best_state, processed_quota)),
        best_value.seeded_rank,
        _selected_ids(best_value),
    )
    choices: list[tuple[str, tuple[tuple[int, ...], tuple[int, ...]], tuple[int, ...]]] = []
    while best_value.paper_id is not None:
        assert best_value.option is not None
        choices.append((best_value.paper_id, best_value.option.marginal, best_value.option.witness))
        assert best_value.previous is not None
        best_value = best_value.previous
    return (
        objective,
        {paper_id: marginal for paper_id, marginal, _ in choices},
        {paper_id: witness for paper_id, _, witness in choices},
    )


@dataclass(frozen=True)
class AllocationResult:
    selected: tuple[Chunk, ...]
    objective: tuple[int, int, int, int, int, tuple[str, ...]]
    marginals: dict[str, tuple[tuple[int, ...], tuple[int, ...]]]
    witnesses: dict[str, tuple[int, ...]]


def _allocate(
    selected_papers: Sequence[str],
    quotas: dict[str, int],
    cells_by_paper: dict[str, dict[tuple[int, int], list[Chunk]]],
) -> AllocationResult:
    """Globally optimal structural allocation under pinned per-paper quotas."""
    for paper_id in selected_papers:
        if not _enumerate_marginals(cells_by_paper[paper_id], quotas[paper_id]):
            raise SampleError(
                f"paper {paper_id} cannot satisfy quota {quotas[paper_id]} from its cells"
            )
    ordered_papers = sorted(selected_papers, key=lambda paper_id: paper_id)
    objective, marginals_map, witnesses_map = _global_optimum(
        ordered_papers,
        cells_by_paper,
        quotas,
    )
    cell_keys = _cell_keys()
    selected_chunks: list[Chunk] = []
    for paper_id in ordered_papers:
        witness = witnesses_map[paper_id]
        for cell_key, count in zip(cell_keys, witness):
            if count > 0:
                selected_chunks.extend(
                    cells_by_paper[paper_id].get(cell_key, ())[:count]
                )
    return AllocationResult(
        selected=tuple(selected_chunks),
        objective=objective,
        marginals=marginals_map,
        witnesses=witnesses_map,
    )


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


def _preflight_destination(path: Path, payload: bytes) -> None:
    """Inspect-only preflight of one destination.

    Outcomes:
    - absent -> publishable
    - existing identical bytes -> publishable (idempotent no-op)
    - existing different bytes -> hard fail

    Never writes to the destination.
    """
    if not path.exists():
        return
    if path.read_bytes() == payload:
        return
    raise SampleError(
        f"publish preflight failed: existing destination has different bytes; "
        f"refusing to overwrite: {path}"
    )


def _write_once(path: Path, payload: bytes) -> None:
    """Atomic, write-once publication (assumes preflight has already passed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
    except OSError as exc:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        raise SampleError(f"failed to publish {path}: {exc}") from exc


def _preflight_publish_pair(
    output_path: Path,
    output_payload: bytes,
    audit_path: Path,
    audit_payload: bytes,
) -> None:
    """Preflight both artifacts before writing either.

    Both byte payloads are inspected against their destinations. If EITHER
    destination has conflicting existing bytes, the preflight fails BEFORE
    any write is performed, so partial publication is impossible.

    Outcomes for each destination:

    - absent -> publishable
    - identical bytes -> publishable / idempotent
    - different bytes -> hard fail
    """
    _preflight_destination(output_path, output_payload)
    _preflight_destination(audit_path, audit_payload)


def _publish(output_path: Path, payload: bytes) -> None:
    _preflight_destination(output_path, payload)
    if output_path.exists() and output_path.read_bytes() == payload:
        return
    _write_once(output_path, payload)

def _audit(
    selected_papers: Sequence[str],
    tiers: dict[str, str],
    capacities: dict[str, int],
    quotas: dict[str, int],
    objective: tuple[int, int, int, int, int, tuple[str, ...]],
    allocation: AllocationResult,
    *,
    leakage_overlap: int,
    chunks_sha256: str,
    manifest_sha256: str,
    ontology_sha256: str,
    leakage_sha256: str,
    resolved_colbert_proxy: str,
) -> dict[str, object]:
    """Build the deterministic audit record.

    ``chunks_sha256``, ``manifest_sha256``, ``ontology_sha256`` and
    ``leakage_artifact_sha256`` are SHA-256 hashes of the exact input bytes
    that produced this sample. ``selected_chunk_ids_sha256`` is the SHA-256 of
    the sorted selected chunk IDs (objective 6 fingerprint).
    """
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
    selected_chunk_ids = sorted(chunk.chunk_id for chunk in allocation.selected)
    selected_chunk_ids_sha256 = hashlib.sha256(
        "".join(selected_chunk_ids).encode("utf-8")
    ).hexdigest()
    objective_list = list(objective)
    return {
        "selected_paper_ids": sorted(selected_papers),
        "paper_tiers": {paper_id: tiers[paper_id] for paper_id in selected_papers},
        "central_paper_ids": sorted(
            paper_id for paper_id in selected_papers if tiers[paper_id] == "benchmark_central"
        ),
        "non_central_paper_ids": sorted(
            paper_id for paper_id in selected_papers if tiers[paper_id] != "benchmark_central"
        ),
        "resolved_colbert_proxy": resolved_colbert_proxy,
        "leakage_safe_capacity": {
            paper_id: capacities.get(paper_id, 0) for paper_id in selected_papers
        },
        "paper_quotas": {paper_id: quotas[paper_id] for paper_id in selected_papers},
        "reference_depth_marginals": list(SECTION_DEPTH_TARGETS),
        "achieved_depth_marginals": list(actual_depth),
        "reference_length_marginals": list(LENGTH_TARGETS),
        "achieved_length_marginals": list(actual_length),
        "objective": objective_list,
        "leakage_overlap_count": leakage_overlap,
        "chunks_sha256": chunks_sha256,
        "manifest_sha256": manifest_sha256,
        "ontology_sha256": ontology_sha256,
        "leakage_artifact_sha256": leakage_sha256,
        "selected_chunk_ids_sha256": selected_chunk_ids_sha256,
        "sampler_seed": SAMPLER_SEED_TEXT,
        "sampler_version": SAMPLER_VERSION,
    }


def _audit_body(payload: dict[str, object]) -> bytes:
    """Canonical deterministic JSON serialization of an audit record."""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return text.encode("utf-8")


def _publish_audit(audit_path: Path, body: bytes) -> None:
    """Write-once audit publication (same per-file semantics as the sample).

    Used only for unit testing the audit publish path in isolation. The
    canonical sampler publication path uses ``_preflight_publish_pair`` and
    ``_write_once`` so that BOTH artifacts are preflighted together before
    either is written.
    """
    _preflight_destination(audit_path, body)
    if audit_path.exists() and audit_path.read_bytes() == body:
        return
    _write_once(audit_path, body)

def run_sample(
    chunks_path: Path,
    manifest_path: Path,
    leakage_path: Path,
    output_path: Path,
    audit_path: Path,
    *,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
) -> int:
    chunks = _load_chunks(chunks_path)
    manifest = _load_manifest(manifest_path)
    manifest_ids = {paper.openalex_id for paper in manifest}
    leakage_bytes = leakage_path.read_bytes()
    leakage_sha256 = hashlib.sha256(leakage_bytes).hexdigest()
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    ontology_sha256 = hashlib.sha256(ontology_path.read_bytes()).hexdigest()
    chunks_sha256 = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
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
    # ColBERT-family availability (#48 Refinement 8): a candidate is available
    # iff it has at least one canonical chunk BEFORE held-out exclusions. The
    # candidates are evaluated in the pinned order
    # ``(W3021397474, W3038831159, W3217305727)``. Manifest presence is
    # intentionally NOT required for proxy resolution; the canonical-chunk
    # artifact alone decides availability.
    canonical_counts = {
        candidate: sum(
            1 for chunk in chunks if chunk.doc_id == candidate
        )
        for candidate in COLBERT_PROXY_CANDIDATES
    }
    proxy_id, _ = _resolve_colbert_proxy(canonical_counts)
    central_ids = [*CENTRAL_ANCHORS, proxy_id]
    # Leakage-safe capacity is computed separately from availability.
    capacities = _safe_capacities(chunks, central_ids, exclusions)
    central_quotas = _central_quotas(
        capacities,
        canonical_counts=canonical_counts,
        manifest_ids=manifest_ids,
    )
    if set(central_quotas) != set(central_ids):
        raise SampleError("central quota split is malformed")
    if sum(central_quotas.values()) != CENTRAL_TOTAL:
        raise SampleError(
            f"central total {sum(central_quotas.values())} != {CENTRAL_TOTAL}"
        )

    mid, long = _select_papers(manifest, chunks, exclusions, central_ids)
    non_central_ids = [*mid, *long]
    selected_papers = [*central_ids, *non_central_ids]
    full_capacities = _safe_capacities(chunks, selected_papers, exclusions)
    tiers: dict[str, str] = {
        **{paper_id: "benchmark_central" for paper_id in central_ids},
        **{paper_id: "mid_density" for paper_id in mid},
        **{paper_id: "long_tail" for paper_id in long},
    }
    quotas = {
        **{paper_id: central_quotas[paper_id] for paper_id in central_ids},
        **{paper_id: NON_CENTRAL_QUOTA for paper_id in non_central_ids},
    }
    if sum(quotas.values()) != SAMPLE_SIZE:
        raise SampleError(
            f"per-paper quotas total {sum(quotas.values())} != {SAMPLE_SIZE}"
        )

    cells_by_paper = _cells_by_paper(selected_papers, chunks, exclusions)
    allocation = _allocate(selected_papers, quotas, cells_by_paper)
    selected = list(allocation.selected)
    if len(selected) != SAMPLE_SIZE or len({chunk.chunk_id for chunk in selected}) != SAMPLE_SIZE:
        raise SampleError(
            f"allocator returned {len(selected)} chunks; required {SAMPLE_SIZE} distinct chunks"
        )

    leakage_overlap = len(exclusions & {chunk.chunk_id for chunk in selected})
    if leakage_overlap != 0:
        raise SampleError(
            f"selected sample has {leakage_overlap} overlapping leakage chunks"
        )

    records = sorted(
        (_record(chunk, tiers[chunk.doc_id]) for chunk in selected),
        key=lambda record: record["chunk_id"],
    )
    if len(records) != SAMPLE_SIZE:
        raise SampleError(f"records {len(records)} != {SAMPLE_SIZE}")
    output_payload = _serialize(records)

    audit = _audit(
        selected_papers,
        tiers,
        full_capacities,
        quotas,
        allocation.objective,
        allocation,
        leakage_overlap=leakage_overlap,
        chunks_sha256=chunks_sha256,
        manifest_sha256=manifest_sha256,
        ontology_sha256=ontology_sha256,
        leakage_sha256=leakage_sha256,
        resolved_colbert_proxy=proxy_id,
    )
    audit_payload = _audit_body(audit)

    # Two-artifact publication preflight: inspect BOTH destinations against
    # the freshly-computed byte payloads BEFORE writing either. If either
    # destination has conflicting bytes, fail closed without touching disk.
    _preflight_publish_pair(output_path, output_payload, audit_path, audit_payload)
    # Honor the idempotent no-op contract: identical bytes already on disk
    # must not be rewritten (mtime must not change for repeated runs).
    if not (output_path.exists() and output_path.read_bytes() == output_payload):
        _write_once(output_path, output_payload)
    if not (audit_path.exists() and audit_path.read_bytes() == audit_payload):
        _write_once(audit_path, audit_payload)
    return SAMPLE_SIZE


def _default_paths() -> dict[str, Path]:
    return {
        "chunks": ROOT / "data" / "derived" / "chunks.jsonl",
        "manifest": ROOT / "data" / "manifest.jsonl",
        "leakage": ROOT / "eval" / "held_out_leakage.v1.json",
        "ontology": ROOT / "ontology.yaml",
        "output": ROOT / "data" / "gold_set" / "sample.v2.jsonl",
        "audit": ROOT / "data" / "gold_set" / "sample.v2.audit.json",
    }


def _build_parser() -> argparse.ArgumentParser:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(
        description="Deterministic 120-chunk gold-set sampler v2 (exact global optimum)"
    )
    parser.add_argument("--chunks", type=Path, default=defaults["chunks"])
    parser.add_argument("--manifest", type=Path, default=defaults["manifest"])
    parser.add_argument("--leakage", type=Path, default=defaults["leakage"])
    parser.add_argument("--ontology", type=Path, default=defaults["ontology"])
    parser.add_argument("--output", type=Path, default=defaults["output"])
    parser.add_argument("--audit", type=Path, default=defaults["audit"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        count = run_sample(
            args.chunks,
            args.manifest,
            args.leakage,
            args.output,
            args.audit,
            ontology_path=args.ontology,
        )
    except SampleError as exc:
        print(f"sample failed: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {count} records to {args.output}")
    print(f"wrote audit to {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
