"""Deterministic chunk-set selection for the model-extraction benchmark.

The benchmark requires a frozen, reproducible chunk set spanning:

* multiple independent documents,
* prose, LaTeX, table-environment content,
* short, medium, and long chunks.

The selector is **deterministic**: given the same corpus file and
the same configuration it produces the same byte-identical chunk
list. The selection algorithm version is committed in
:mod:`src.benchmark.dataset.manifest` and bumped on any change to
the rules.

Algorithm (version ``v1``):

1. Read every chunk from the corpus JSONL; preserve order.
2. Bucket each chunk by length tercile (``short``, ``medium``,
   ``long``) computed over the full corpus.
3. Tag each chunk with a content-kind hint based on cheap textual
   signals: ``latex`` (any ``\\`` / ``\\begin`` / ``\\cite``),
   ``table`` (any ``\\begin{table}`` / ``\\begin{tabular}`` /
   ``\\caption``), ``prose`` (LaTeX-command density below 5%).
4. Sort chunks by a deterministic key that prefers table-heavy,
   then latex-heavy, then per-doc spread, then corpus order.
5. Walk chunks in that order and accept until the target size is
   reached, enforcing the per-document cap and required ids.
6. Verify the tag floors and minimum-document count; abort on any
   violation.
7. Return the selected chunk dicts in deterministic order.

The algorithm is offline-only: it never touches the network, never
reads any cache, never inspects model output.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

SELECTION_ALGORITHM_VERSION = "benchmark-dataset-selector/v1"

# Frozen sampler seed. Must never change without an algorithm bump.
SAMPLER_SEED = 0x5EEDF00D

DEFAULT_TARGET_SIZE = 40
DEFAULT_MIN_DOCUMENTS = 6
DEFAULT_PER_DOC_CAP = 10
DEFAULT_MIN_TABLE_FRACTION = 0.10
DEFAULT_MIN_LATEX_FRACTION = 0.20
DEFAULT_MIN_PROSE_FRACTION = 0.05

@dataclass(frozen=True)
class SelectionConfig:
    target_size: int = DEFAULT_TARGET_SIZE
    min_documents: int = DEFAULT_MIN_DOCUMENTS
    per_doc_cap: int = DEFAULT_PER_DOC_CAP
    min_table_fraction: float = DEFAULT_MIN_TABLE_FRACTION
    min_latex_fraction: float = DEFAULT_MIN_LATEX_FRACTION
    min_prose_fraction: float = DEFAULT_MIN_PROSE_FRACTION

@dataclass(frozen=True)
class ChunkTag:
    """Content-kind hints derived from cheap textual signals."""

    prose: bool
    latex: bool
    table: bool

    def as_set(self) -> frozenset[str]:
        tags: set[str] = set()
        if self.prose:
            tags.add("prose")
        if self.latex:
            tags.add("latex")
        if self.table:
            tags.add("table")
        return frozenset(tags)

def _compute_tercile_bounds(lengths: Sequence[int]) -> tuple[int, int]:
    if not lengths:
        raise ValueError("cannot compute terciles on empty length list")
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    lo_idx = max(0, n // 3 - 1)
    hi_idx = max(lo_idx + 1, (2 * n) // 3 - 1)
    return sorted_lengths[lo_idx], sorted_lengths[hi_idx]

def _length_bucket(text: str, tercile_bounds: tuple[int, int]) -> str:
    n = len(text)
    if n < tercile_bounds[0]:
        return "short"
    if n < tercile_bounds[1]:
        return "medium"
    return "long"

_LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+")

def _tag_chunk(text: str) -> ChunkTag:
    """Derive cheap content-kind hints without parsing the chunk.

    A chunk is tagged ``latex`` whenever it carries any LaTeX
    command (``\\``, ``\\begin``, ``\\cite``); ``table`` when it
    carries a table or caption command. ``prose`` is set when the
    chunk contains enough non-LaTeX text to be considered
    prose-heavy (LaTeX-command density below 5% of characters).
    """
    has_backslash = "\\" in text
    has_begin = "\\begin" in text
    has_cite = "\\cite" in text
    has_table = "\\begin{table" in text or "\\begin{tabular" in text
    has_caption = "\\caption" in text
    latex = has_backslash or has_begin or has_cite
    table = has_table or has_caption
    if not text:
        return ChunkTag(prose=False, latex=latex, table=table)
    latex_chars = sum(len(m.group(0)) for m in _LATEX_CMD_RE.finditer(text))
    prose = latex_chars < 0.05 * len(text)
    return ChunkTag(prose=prose, latex=latex, table=table)

def _doc_id_stable_rank(doc_id: str) -> int:
    """Stable, content-free per-document rank used to break ties."""
    return int.from_bytes(hashlib.sha256(doc_id.encode("utf-8")).digest()[:4], "big")

def _read_chunks_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"chunks file not found: {path}")
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out

def _sort_key(idx: int, length: int, sorted_lengths: Sequence[int], doc_rank: int,
              tag: ChunkTag) -> tuple[int, int, int, int, int]:
    """Deterministic sort key. Lower sorts first.

    Bucket table-heavy and latex-heavy chunks early so the tag
    floors are met before the bulk prose fills the rest. Then
    per-document spread (rare docs first), then length-balanced,
    then corpus order.
    """
    table_score = 0 if tag.table else 1
    latex_score = 0 if tag.latex else 1
    length_rank = bisect.bisect_left(sorted_lengths, length)
    return (table_score, latex_score, length_rank, idx, doc_rank)

@dataclass(frozen=True)
class SelectionReport:
    algorithm_version: str
    sampler_seed: int
    config: SelectionConfig
    selected_chunk_ids: tuple[str, ...]
    documents_selected: tuple[str, ...]
    counts_by_tag: dict[str, int]
    counts_by_length_tercile: dict[str, int]
    chunks_path: Path
    chunks_sha256: str

def select_chunks(
    chunks_path: Path,
    *,
    config: SelectionConfig | None = None,
    required_chunk_ids: Sequence[str] = (),
) -> tuple[list[dict], SelectionReport]:
    """Select a deterministic chunk set for the benchmark."""
    cfg = config or SelectionConfig()
    chunks = _read_chunks_jsonl(chunks_path)
    if not chunks:
        raise ValueError("chunks corpus is empty")

    by_id = {c["chunk_id"]: c for c in chunks}
    for required in required_chunk_ids:
        if required not in by_id:
            raise KeyError(
                f"required chunk {required!r} not present in corpus {chunks_path}"
            )

    lengths = [len(c["text"]) for c in chunks]
    sorted_lengths = sorted(lengths)
    tercile_bounds = _compute_tercile_bounds(lengths)
    tags = {c["chunk_id"]: _tag_chunk(c["text"]) for c in chunks}
    bucket = {
        c["chunk_id"]: _length_bucket(c["text"], tercile_bounds)
        for c in chunks
    }

    sortable = []
    for idx, c in enumerate(chunks):
        sortable.append(
            (
                _sort_key(
                    idx=idx,
                    length=len(c["text"]),
                    sorted_lengths=sorted_lengths,
                    doc_rank=_doc_id_stable_rank(c["doc_id"]),
                    tag=tags[c["chunk_id"]],
                ),
                c["chunk_id"],
            )
        )
    sortable.sort()

    selected_ids: list[str] = []
    selected_set: set[str] = set()
    per_doc_count: Counter[str] = Counter()
    documents_selected: set[str] = set()

    # Required first, in given order, at the head of the result.
    for required in required_chunk_ids:
        c = by_id[required]
        selected_ids.append(required)
        selected_set.add(required)
        per_doc_count[c["doc_id"]] += 1
        documents_selected.add(c["doc_id"])

    # Group remaining candidates by document. Within each doc we
    # keep three ordered sub-queues: tables, prose, latex-prose.
    # The selector first takes one pass over the table queues
    # (round-robin across docs), one pass over the prose queues,
    # then fills the remainder from the latex-prose queues. This
    # keeps the per-document floor satisfied and honours the tag
    # floors without a single bucket starving the others.
    doc_table_queues: dict[str, list[str]] = {}
    doc_prose_queues: dict[str, list[str]] = {}
    doc_latex_queues: dict[str, list[str]] = {}
    for _key, cid in sortable:
        if cid in selected_set:
            continue
        if len(by_id[cid]["text"]) < 80:
            continue
        tag = tags[cid]
        doc = by_id[cid]["doc_id"]
        if tag.table:
            doc_table_queues.setdefault(doc, []).append(cid)
        elif tag.prose:
            doc_prose_queues.setdefault(doc, []).append(cid)
        elif tag.latex:
            doc_latex_queues.setdefault(doc, []).append(cid)
    doc_order = sorted(
        set(doc_table_queues) | set(doc_prose_queues) | set(doc_latex_queues),
        key=_doc_id_stable_rank,
    )
    table_index = {d: 0 for d in doc_order}
    prose_index = {d: 0 for d in doc_order}
    latex_index = {d: 0 for d in doc_order}

    # Per-doc balanced fill. Each doc contributes up to its per-doc
    # cap of chunks in this priority order:
    #   1. up to 1 table (if the doc has any),
    #   2. up to 1 prose chunk (if the doc has any),
    #   3. the remainder from the doc's latex-prose queue.
    # Docs are walked in deterministic stable-rank order so the
    # result is reproducible. Required ids are honoured at the head
    # of the selection so the smoke chunks are always addressable.
    per_doc_fill = cfg.target_size // max(1, cfg.min_documents)
    if per_doc_fill > cfg.per_doc_cap:
        per_doc_fill = cfg.per_doc_cap
    if per_doc_fill < 1:
        per_doc_fill = 1

    def _take_balanced(d: str) -> None:
        if per_doc_count.get(d, 0) >= cfg.per_doc_cap:
            return
        if len(selected_ids) >= cfg.target_size:
            return
        # 1 table if available.
        if per_doc_count.get(d, 0) < per_doc_fill:
            for cid in doc_table_queues.get(d, []):
                if cid in selected_set:
                    continue
                if len(selected_ids) >= cfg.target_size:
                    return
                selected_ids.append(cid)
                selected_set.add(cid)
                per_doc_count[d] = per_doc_count.get(d, 0) + 1
                documents_selected.add(d)
                break
        # 1 prose if available and not at cap.
        if per_doc_count.get(d, 0) < per_doc_fill:
            if len(selected_ids) >= cfg.target_size:
                return
            for cid in doc_prose_queues.get(d, []):
                if cid in selected_set:
                    continue
                if len(selected_ids) >= cfg.target_size:
                    return
                selected_ids.append(cid)
                selected_set.add(cid)
                per_doc_count[d] = per_doc_count.get(d, 0) + 1
                documents_selected.add(d)
                break
        # Latex prose up to per-doc fill.
        for cid in doc_latex_queues.get(d, []):
            if per_doc_count.get(d, 0) >= per_doc_fill:
                break
            if cid in selected_set:
                continue
            if len(selected_ids) >= cfg.target_size:
                return
            selected_ids.append(cid)
            selected_set.add(cid)
            per_doc_count[d] = per_doc_count.get(d, 0) + 1
            documents_selected.add(d)
            if per_doc_count.get(d, 0) >= cfg.per_doc_cap:
                break

    # Walk docs in stable-rank order until target is reached.
    for d in doc_order:
        if len(selected_ids) >= cfg.target_size:
            break
        _take_balanced(d)
    # Second wave: any doc still under its cap.
    if len(selected_ids) < cfg.target_size:
        for d in doc_order:
            _take_balanced(d)
    # Third wave: top up prose where missing.
            for cid in doc_prose_queues.get(d, []):
                if cid in selected_set:
                    continue
                if per_doc_count.get(d, 0) >= cfg.per_doc_cap:
                    break
                selected_ids.append(cid)
                selected_set.add(cid)
                per_doc_count[d] = per_doc_count.get(d, 0) + 1
                documents_selected.add(d)
                break

    if len(selected_ids) < cfg.target_size:
        raise RuntimeError(
            f"selection produced {len(selected_ids)} chunks; target {cfg.target_size}"
        )
    if len(documents_selected) < cfg.min_documents:
        raise RuntimeError(
            f"selection only used {len(documents_selected)} documents; "
            f"minimum is {cfg.min_documents}"
        )

    tag_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    for cid in selected_ids:
        for tag in tags[cid].as_set():
            tag_counts[tag] += 1
        length_counts[bucket[cid]] += 1

    total = len(selected_ids)
    if tag_counts["table"] < cfg.min_table_fraction * total:
        raise RuntimeError(
            f"selection has {tag_counts['table']} table chunks; "
            f"floor is {cfg.min_table_fraction * total:.0f}"
        )
    if tag_counts["latex"] < cfg.min_latex_fraction * total:
        raise RuntimeError(
            f"selection has {tag_counts['latex']} latex chunks; "
            f"floor is {cfg.min_latex_fraction * total:.0f}"
        )
    if tag_counts["prose"] < cfg.min_prose_fraction * total:
        raise RuntimeError(
            f"selection has {tag_counts['prose']} prose chunks; "
            f"floor is {cfg.min_prose_fraction * total:.0f}"
        )

    canonical_chunks = [
        json.dumps(by_id[cid], sort_keys=True, ensure_ascii=False)
        for cid in selected_ids
    ]
    chunks_sha256 = hashlib.sha256(
        ("\n".join(canonical_chunks) + "\n").encode("utf-8")
    ).hexdigest()

    selected_chunks = [by_id[cid] for cid in selected_ids]
    report = SelectionReport(
        algorithm_version=SELECTION_ALGORITHM_VERSION,
        sampler_seed=SAMPLER_SEED,
        config=cfg,
        selected_chunk_ids=tuple(selected_ids),
        documents_selected=tuple(sorted(documents_selected)),
        counts_by_tag=dict(tag_counts),
        counts_by_length_tercile=dict(length_counts),
        chunks_path=chunks_path,
        chunks_sha256=chunks_sha256,
    )
    return selected_chunks, report

def write_selection_artifacts(
    selected_chunks: Iterable[dict],
    report: SelectionReport,
    *,
    output_jsonl: Path,
    output_sha256: Path,
    output_manifest: Path,
) -> None:
    """Atomically write the frozen selection JSONL, its SHA-256, and
    the selection manifest. Overwrites any prior content. Never
    writes partial files."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=output_jsonl.name + ".", suffix=".tmp", dir=str(output_jsonl.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for c in selected_chunks:
                fh.write(json.dumps(c, sort_keys=True, ensure_ascii=False))
                fh.write("\n")
        os.replace(tmp, output_jsonl)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise

    raw = output_jsonl.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != report.chunks_sha256:
        raise RuntimeError(
            f"selected-chunks sha256 drift: computed {sha} != recorded {report.chunks_sha256}"
        )
    output_sha256.write_text(sha + "\n", encoding="utf-8")

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "algorithm_version": report.algorithm_version,
        "sampler_seed": report.sampler_seed,
        "config": {
            "target_size": report.config.target_size,
            "min_documents": report.config.min_documents,
            "per_doc_cap": report.config.per_doc_cap,
            "min_table_fraction": report.config.min_table_fraction,
            "min_latex_fraction": report.config.min_latex_fraction,
            "min_prose_fraction": report.config.min_prose_fraction,
        },
        "selected_chunk_ids": list(report.selected_chunk_ids),
        "documents_selected": list(report.documents_selected),
        "counts_by_tag": report.counts_by_tag,
        "counts_by_length_tercile": report.counts_by_length_tercile,
        "chunks_path": str(report.chunks_path),
        "chunks_sha256": report.chunks_sha256,
    }
    output_manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )