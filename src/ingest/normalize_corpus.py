"""Collapse duplicate records, apply title corrections, remeasure density.

OpenAlex holds some papers under two work IDs. Left alone they split the
citation graph: works citing identity A never connect to works citing
identity B, and every edge pointing at the discarded identity is lost.

This step resolves each paper to one canonical record and keeps the other IDs
as aliases, so incoming edges can be remapped instead of dropped. That is the
same operation the project performs later on entity names, applied here to the
document layer.

Duplicates are detected two ways: a shared arXiv ID, or an identical
normalised title in the same publication year. Every merge is printed, so the
decisions stay auditable.

Local only. No network access.

    python -m src.ingest.normalize_corpus --dry-run
    python -m src.ingest.normalize_corpus
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "openalex"
MANIFEST = ROOT / "data" / "manifest.jsonl"
OVERRIDES = ROOT / "data" / "title_overrides.json"
ALIASES = ROOT / "data" / "aliases.json"


def short_id(openalex_id: str) -> str:
    return openalex_id.rsplit("/", 1)[-1]


def normalize_title(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return " ".join(text.split())


def load_rows() -> list[dict]:
    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} not found")
    return [json.loads(line) for line in MANIFEST.read_text().splitlines() if line]


def load_raw() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in RAW.glob("*.json")}


def load_overrides() -> dict[str, dict]:
    if not OVERRIDES.exists():
        print(f"No {OVERRIDES.name}, skipping title corrections.")
        return {}
    return json.loads(OVERRIDES.read_text(encoding="utf-8"))


def group_duplicates(rows: list[dict]) -> list[list[dict]]:
    """Group rows that describe the same paper. Returns groups of size >= 2."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("arxiv_id"):
            groups[f"arxiv:{row['arxiv_id']}"].append(row)
        else:
            key = normalize_title(row.get("title"))
            if key:
                groups[f"title:{key}:{row.get('year')}"].append(row)
            else:
                groups[f"id:{row['openalex_id']}"].append(row)
    return [g for g in groups.values() if len(g) > 1]


def pick_canonical(group: list[dict]) -> dict:
    """Most citations wins; a non-empty title breaks ties."""
    return max(group, key=lambda r: (r.get("cited_by_count") or 0,
                                     bool(r.get("title"))))


def density(ids: set[str], raw: dict[str, dict], alias: dict[str, str],
            label: str) -> float:
    """Internal citation density, resolving references through the alias map."""
    total = internal = isolated = 0
    per_work = []
    for wid in ids:
        record = raw.get(wid)
        if not record:
            continue
        refs = {alias.get(short_id(r), short_id(r))
                for r in record.get("referenced_works") or []}
        hits = sum(1 for r in refs if r in ids and r != wid)
        total += len(refs)
        internal += hits
        if refs:
            per_work.append(hits / len(refs))
        if hits == 0:
            isolated += 1

    ratio = internal / total if total else 0.0
    median = statistics.median(per_work) if per_work else 0.0
    print(f"  {label:<22} {len(ids):>4} works  {ratio:>6.1%} density  "
          f"{median:>6.1%} median  {isolated:>3} isolated")
    return ratio


def main() -> None:
    ap = argparse.ArgumentParser(description="Deduplicate and correct the corpus")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the merges but write nothing")
    args = ap.parse_args()

    rows = load_rows()
    raw = load_raw()
    overrides = load_overrides()
    by_id = {r["openalex_id"]: r for r in rows}

    print(f"Manifest rows: {len(rows)}, raw records on disk: {len(raw)}\n")

    groups = group_duplicates(rows)
    alias: dict[str, str] = {}

    print("=" * 72)
    print("MERGES")
    print("=" * 72)
    if not groups:
        print("  none")
    for group in groups:
        canonical = pick_canonical(group)
        cid = canonical["openalex_id"]
        dropped = [r for r in group if r["openalex_id"] != cid]
        key = canonical.get("arxiv_id") or normalize_title(canonical.get("title"))[:34]
        print(f"\n  {key}")
        print(f"    keep {cid:<12} {canonical.get('cited_by_count', 0):>6} cit")
        for r in dropped:
            alias[r["openalex_id"]] = cid
            print(f"    drop {r['openalex_id']:<12} "
                  f"{r.get('cited_by_count', 0):>6} cit")

    kept_ids = {r["openalex_id"] for r in rows if r["openalex_id"] not in alias}

    print("\n" + "=" * 72)
    print("DENSITY")
    print("=" * 72)
    before = density({r["openalex_id"] for r in rows}, raw, {}, "before merge")
    after = density(kept_ids, raw, alias, "after merge")
    delta = after - before
    print(f"\n  change: {delta:+.1%}")
    print("=" * 72)

    if args.dry_run:
        print("\nDry run, nothing written.")
        return

    applied = 0
    out = []
    for wid in sorted(kept_ids):
        row = dict(by_id[wid])
        merged = sorted(k for k, v in alias.items() if v == wid)
        row["merged_ids"] = merged
        row["title_source"] = "openalex"

        # The most-cited record is not always the most complete one. If it has
        # no title, take one from a record merged into it.
        if not row.get("title"):
            for other in merged:
                if by_id[other].get("title"):
                    row["title"] = by_id[other]["title"]
                    row["title_source"] = "merged_duplicate"
                    break
        if wid in overrides:
            row["title"] = overrides[wid]["title"]
            row["title_source"] = "manual_override"
            row["title_override_reason"] = overrides[wid]["reason"]
            applied += 1
        out.append(json.dumps(row, ensure_ascii=False))

    MANIFEST.write_text("\n".join(out) + "\n", encoding="utf-8")
    ALIASES.write_text(json.dumps(alias, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")

    unused = [k for k in overrides if k not in kept_ids]
    print(f"\nManifest rewritten: {len(out)} works "
          f"({len(rows)} before, {len(alias)} merged away)")
    print(f"Title overrides applied: {applied} of {len(overrides)}")
    if unused:
        print(f"  not found in corpus: {', '.join(unused)}")
    print(f"Alias map written: {ALIASES.name}")


if __name__ == "__main__":
    main()