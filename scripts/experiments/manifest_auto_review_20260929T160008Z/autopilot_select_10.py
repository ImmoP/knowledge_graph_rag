#!/usr/bin/env python3
"""Deterministic 10-chunk autopilot selection for the next manifest-auto run.

This script picks the next 10 chunks the bounded autopilot would dispatch
to the manifest provider. **It does not make any LLM calls.** It only
materialises a JSON plan with the chosen ``chunk_id`` s, their tags,
length terciles, and the per-doc allocation; nothing is sent to the
provider.

Selection criteria (documented; order of precedence):

1. Exclude the two chunks already processed by the smoke run
   (``c_0001921855a8156b``, ``c_001452fd142ce075``).
2. Length-tercile balance across the 40-chunk benchmark dataset. The
   terciles are computed in characters using the same algorithm as
   ``src/benchmark/dataset/select.py::_compute_tercile_bounds``.
   Target allocation: 3 long, 3 medium, 4 short.
3. Per-doc cap of 2 chunks (the smoke run already touches 2 docs; we
   expand coverage by sampling at most 2 chunks per remaining doc).
4. Tag diversity: include at least one prose-heavy chunk and one
   table-heavy chunk if available in the candidate pool.
5. Deterministic order: a frozen seed (``AUTOPILOT_SEED`` below) plus
   the per-doc rank and chunk-index tie-breakers.

Inputs:

* ``--chunks PATH``: JSONL of benchmark chunks (default
  ``src/benchmark/dataset/benchmark_chunks.jsonl``).
* ``--exclude ID,ID,...``: comma-separated chunk ids to skip (default:
  the two smoke-run ids).
* ``--output PATH``: write the plan JSON.

Output JSON shape::

    {
      "algorithm_version": "manifest-auto-autopilot-selector/v1",
      "seed": <int>,
      "excluded_chunk_ids": [...],
      "selection_criteria": [...],
      "length_tercile_bounds_chars": [lo, hi],
      "tag_targets": {"prose_min": 1, "table_min": 1},
      "per_doc_cap": 2,
      "tercile_target": {"short": 4, "medium": 3, "long": 3},
      "plan": [
        {"chunk_id": "...", "doc_id": "...", "char_len": N,
         "token_estimate": N, "length_bucket": "long|medium|short",
         "tags": {"prose": bool, "latex": bool, "table": bool},
         "selection_rank": N}
      ],
      "totals": {"long": N, "medium": N, "short": N,
                 "prose_chunks": N, "table_chunks": N, "unique_docs": N}
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

# Make the canonical length/tercile helpers reusable without copying.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.benchmark.dataset.select import (  # noqa: E402
    _compute_tercile_bounds,
    _length_bucket,
    _tag_chunk,
)

ALGORITHM_VERSION = "manifest-auto-autopilot-selector/v1"
AUTOPILOT_SEED = 0xFEEDFACE  # frozen; any change is an algorithm bump

DEFAULT_EXCLUDED = ("c_0001921855a8156b", "c_001452fd142ce075")

TERCILE_TARGET = {"short": 4, "medium": 3, "long": 3}
PER_DOC_CAP = 2
TAG_TARGETS = {"prose_min": 1, "table_min": 1}


def _stable_doc_rank(doc_id: str) -> int:
    """Stable per-doc rank used to break ties; mirrors select.py."""
    return int.from_bytes(
        hashlib.sha256(doc_id.encode("utf-8")).digest()[:4], "big"
    )


def _candidate_pool(chunks: list[dict], excluded: set[str]) -> list[dict]:
    return [c for c in chunks if c["chunk_id"] not in excluded]


def _bucket_counts(candidates: list[dict], bounds: tuple[int, int]) -> Counter:
    counts: Counter = Counter()
    for c in candidates:
        counts[_length_bucket(c["text"], bounds)] += 1
    return counts


def _select_for_bucket(
    bucket: str,
    candidates: list[dict],
    bounds: tuple[int, int],
    *,
    per_doc_taken: Counter,
    seed_rng: random.Random,
    k: int,
) -> list[dict]:
    """Pick up to k chunks from a single length-bucket, respecting the
    per-doc cap and the seed RNG ordering.
    """
    pool = [c for c in candidates if _length_bucket(c["text"], bounds) == bucket]
    # Deterministic order: per-doc rank, then chunk_index, then chunk_id.
    pool.sort(
        key=lambda c: (
            _stable_doc_rank(c["doc_id"]),
            c.get("chunk_index", 0),
            c["chunk_id"],
        )
    )
    # Stable shuffle with the seeded RNG so future ties reproduce.
    seed_rng.shuffle(pool)

    picked: list[dict] = []
    for c in pool:
        if len(picked) >= k:
            break
        if per_doc_taken[c["doc_id"]] >= PER_DOC_CAP:
            continue
        picked.append(c)
        per_doc_taken[c["doc_id"]] += 1
    return picked


def _backfill_tag_targets(
    missing_tags: set[str],
    picked: list[dict],
    candidates: list[dict],
    bounds: tuple[int, int],
    *,
    per_doc_taken: Counter,
    seed_rng: random.Random,
) -> None:
    """If the picked set does not include a required tag, swap in (or
    append) candidates that carry that tag, breaking the per-doc cap
    only when strictly necessary. The swap keeps the length balance
    honest by preferring same-bucket candidates first.
    """
    if not missing_tags:
        return

    picked_ids = {c["chunk_id"]: c for c in picked}

    def carries_tag(c: dict, tag: str) -> bool:
        t = _tag_chunk(c["text"])
        return getattr(t, tag)

    def candidates_with(tag: str) -> list[dict]:
        out = [c for c in candidates if carries_tag(c, tag)]
        out.sort(
            key=lambda c: (
                _length_bucket(c["text"], bounds) == "long",
                _length_bucket(c["text"], bounds) == "medium",
                _stable_doc_rank(c["doc_id"]),
                c.get("chunk_index", 0),
            ),
            reverse=True,
        )
        return out

    for tag in sorted(missing_tags):
        # Try to swap a picked chunk for a same-bucket candidate that
        # carries the missing tag. Prefer drops that do not violate the
        # per-doc cap in a way that removes a *unique* doc from the
        # final plan.
        swapped = False
        for tag_carrier in candidates_with(tag):
            if tag_carrier["chunk_id"] in picked_ids:
                continue
            # Look for a drop target among already-picked chunks that
            # are in the same bucket as ``tag_carrier`` so the tercile
            # balance is preserved.
            carrier_bucket = _length_bucket(tag_carrier["text"], bounds)
            for drop in list(picked):
                if _length_bucket(drop["text"], bounds) != carrier_bucket:
                    continue
                # Swapping keeps per-doc balance neutral iff the drop's
                # doc is represented elsewhere in the plan, or the
                # cap was already exceeded.
                drop_doc = drop["doc_id"]
                drop_doc_count = sum(
                    1 for c in picked if c["doc_id"] == drop_doc
                )
                if drop_doc_count <= 1:
                    # Dropping would lose a doc; only proceed if the
                    # carrier doc is new.
                    if per_doc_taken.get(tag_carrier["doc_id"], 0) >= 1:
                        continue
                picked.remove(drop)
                picked.append(tag_carrier)
                picked_ids.pop(drop["chunk_id"], None)
                picked_ids[tag_carrier["chunk_id"]] = tag_carrier
                per_doc_taken[drop_doc] -= 1
                per_doc_taken[tag_carrier["doc_id"]] = (
                    per_doc_taken.get(tag_carrier["doc_id"], 0) + 1
                )
                swapped = True
                break
            if swapped:
                break
        if swapped:
            continue
        # No same-bucket swap available; append the tag-carrier if the
        # per-doc cap allows.
        for tag_carrier in candidates_with(tag):
            if tag_carrier["chunk_id"] in picked_ids:
                continue
            if per_doc_taken.get(tag_carrier["doc_id"], 0) >= PER_DOC_CAP:
                continue
            picked.append(tag_carrier)
            picked_ids[tag_carrier["chunk_id"]] = tag_carrier
            per_doc_taken[tag_carrier["doc_id"]] = (
                per_doc_taken.get(tag_carrier["doc_id"], 0) + 1
            )
            break


def select_plan(
    chunks: list[dict],
    *,
    excluded_ids: tuple[str, ...] = DEFAULT_EXCLUDED,
    seed: int = AUTOPILOT_SEED,
) -> dict:
    excluded_set = set(excluded_ids)
    lengths = [len(c["text"]) for c in chunks]
    bounds = _compute_tercile_bounds(lengths)
    candidates = _candidate_pool(chunks, excluded_set)
    seed_rng = random.Random(seed)

    per_doc_taken: Counter = Counter()
    picked: list[dict] = []

    # Pick by tercile target. Order: long first (highest information
    # density), then medium, then short.
    for bucket in ("long", "medium", "short"):
        picked.extend(
            _select_for_bucket(
                bucket,
                candidates,
                bounds,
                per_doc_taken=per_doc_taken,
                seed_rng=seed_rng,
                k=TERCILE_TARGET[bucket],
            )
        )

    # Tag backfill.
    missing_tags = set()
    has_prose = any(_tag_chunk(c["text"]).prose for c in picked)
    has_table = any(_tag_chunk(c["text"]).table for c in picked)
    if not has_prose:
        missing_tags.add("prose")
    if not has_table:
        missing_tags.add("table")
    _backfill_tag_targets(
        missing_tags,
        picked,
        candidates,
        bounds,
        per_doc_taken=per_doc_taken,
        seed_rng=seed_rng,
    )

    # Final deterministic order for the plan JSON.
    picked.sort(
        key=lambda c: (
            {"long": 0, "medium": 1, "short": 2}[
                _length_bucket(c["text"], bounds)
            ],
            _stable_doc_rank(c["doc_id"]),
            c.get("chunk_index", 0),
            c["chunk_id"],
        )
    )

    plan = []
    for rank, c in enumerate(picked, start=1):
        t = _tag_chunk(c["text"])
        plan.append(
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "section_path": c.get("section_path", []),
                "char_len": len(c["text"]),
                "token_estimate": c.get("token_estimate"),
                "length_bucket": _length_bucket(c["text"], bounds),
                "tags": {
                    "prose": t.prose,
                    "latex": t.latex,
                    "table": t.table,
                },
                "selection_rank": rank,
            }
        )

    totals = {
        "short": sum(
            1 for c in picked if _length_bucket(c["text"], bounds) == "short"
        ),
        "medium": sum(
            1 for c in picked if _length_bucket(c["text"], bounds) == "medium"
        ),
        "long": sum(
            1 for c in picked if _length_bucket(c["text"], bounds) == "long"
        ),
        "prose_chunks": sum(1 for c in picked if _tag_chunk(c["text"]).prose),
        "table_chunks": sum(1 for c in picked if _tag_chunk(c["text"]).table),
        "unique_docs": len({c["doc_id"] for c in picked}),
    }

    return {
        "algorithm_version": ALGORITHM_VERSION,
        "seed": seed,
        "excluded_chunk_ids": list(excluded_ids),
        "selection_criteria": [
            "exclude_chunks_in_smoke_run",
            f"length_tercile_balance={TERCILE_TARGET}",
            f"per_doc_cap={PER_DOC_CAP}",
            f"tag_minimums={TAG_TARGETS}",
            "deterministic_order_by_doc_rank_then_chunk_index",
        ],
        "length_tercile_bounds_chars": list(bounds),
        "tag_targets": TAG_TARGETS,
        "per_doc_cap": PER_DOC_CAP,
        "tercile_target": TERCILE_TARGET,
        "plan": plan,
        "totals": totals,
    }


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("src/benchmark/dataset/benchmark_chunks.jsonl"),
        help="JSONL of benchmark chunks",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=",".join(DEFAULT_EXCLUDED),
        help="comma-separated chunk ids to skip",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=AUTOPILOT_SEED,
        help="seed for the deterministic shuffle (algorithm-version bump if changed)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the plan JSON to this path",
    )
    args = parser.parse_args(argv)

    if not args.chunks.is_file():
        print(
            f"error: chunks file not found: {args.chunks}",
            file=sys.stderr,
        )
        return 2
    chunks = _read_jsonl(args.chunks)
    excluded = tuple(
        e.strip() for e in args.exclude.split(",") if e.strip()
    )
    plan = select_plan(chunks, excluded_ids=excluded, seed=args.seed)
    payload = json.dumps(plan, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())