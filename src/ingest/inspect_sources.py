"""Inspect which full-text sources the corpus actually offers.

Reads only what is already on disk under data/raw/openalex/, so it costs
nothing and cannot change the snapshot.

    python -m src.ingest.inspect_sources
        Report the breakdown of available sources.

    python -m src.ingest.inspect_sources --fix-manifest
        Rewrite data/manifest.jsonl with improved arXiv detection and an
        added full_text_url field. Does not refetch anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "openalex"
MANIFEST = ROOT / "data" / "manifest.jsonl"

# Matches both landing pages (arxiv.org/abs/2106.09685) and direct PDF links
# (arxiv.org/pdf/2106.09685v2), with or without a version suffix.
ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.I)
ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", re.I)


def short_id(openalex_id: str) -> str:
    return openalex_id.rsplit("/", 1)[-1]


def iter_locations(work: dict):
    """Every location on the record, including the ones outside `locations`."""
    for key in ("primary_location", "best_oa_location"):
        loc = work.get(key)
        if loc:
            yield loc
    for loc in work.get("locations") or []:
        yield loc


def arxiv_id_of(work: dict) -> str | None:
    """Find an arXiv ID in the DOI, any location URL, or the ids block."""
    m = ARXIV_DOI.search(work.get("doi") or "")
    if m:
        return m.group(1)
    for loc in iter_locations(work):
        for field in ("landing_page_url", "pdf_url"):
            m = ARXIV_URL.search(loc.get(field) or "")
            if m:
                return m.group(1)
    for value in (work.get("ids") or {}).values():
        m = ARXIV_URL.search(str(value)) or ARXIV_DOI.search(str(value))
        if m:
            return m.group(1)
    return None


def classify(work: dict) -> tuple[str, str | None]:
    """Return (source kind, best URL) for full-text retrieval."""
    if arxiv_id_of(work):
        return "arxiv", f"https://arxiv.org/abs/{arxiv_id_of(work)}"
    for loc in iter_locations(work):
        url = loc.get("pdf_url") or ""
        if "aclanthology.org" in url:
            return "acl_anthology", url
        if url:
            return "other_pdf", url
    for loc in iter_locations(work):
        if loc.get("landing_page_url"):
            return "landing_only", loc["landing_page_url"]
    return "none", None


def load_works() -> list[dict]:
    files = sorted(RAW.glob("*.json"))
    if not files:
        raise SystemExit(f"No records in {RAW}. Run the fetch step first.")
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def report(works: list[dict]) -> None:
    kinds = Counter()
    oa = 0
    for w in works:
        kind, _ = classify(w)
        kinds[kind] += 1
        if w.get("open_access", {}).get("is_oa"):
            oa += 1

    print("=" * 60)
    print("FULL-TEXT SOURCE BREAKDOWN")
    print("=" * 60)
    print(f"  works on disk:        {len(works)}")
    print(f"  flagged open access:  {oa}")
    print()
    labels = {
        "arxiv": "arXiv (LaTeX source available)",
        "acl_anthology": "ACL Anthology PDF",
        "other_pdf": "other open PDF",
        "landing_only": "landing page only, no PDF",
        "none": "no location at all",
    }
    for kind, label in labels.items():
        count = kinds.get(kind, 0)
        share = count / len(works) if works else 0
        print(f"  {label:<34} {count:>4}  ({share:.0%})")

    reachable = kinds["arxiv"] + kinds["acl_anthology"] + kinds["other_pdf"]
    print()
    print(f"  reachable full text:  {reachable} of {len(works)} "
          f"({reachable / len(works):.0%})")
    print("=" * 60)


def fix_manifest(works: list[dict]) -> None:
    """Rewrite the manifest using the improved detection. No network access."""
    now = datetime.now(timezone.utc).isoformat()
    lines = []
    for w in sorted(works, key=lambda x: x["id"]):
        wid = short_id(w["id"])
        raw = (RAW / f"{wid}.json").read_bytes()
        kind, url = classify(w)
        lines.append(json.dumps({
            "openalex_id": wid,
            "doi": w.get("doi"),
            "arxiv_id": arxiv_id_of(w),
            "title": w.get("title"),
            "year": w.get("publication_year"),
            "cited_by_count": w.get("cited_by_count"),
            "source_kind": kind,
            "full_text_url": url,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "fetched_at": now,
            "source_url": w["id"],
        }, ensure_ascii=False))
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nManifest rewritten: {len(lines)} lines")
    print("Note: fetched_at now reflects this rewrite, not the original download.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect available full-text sources")
    ap.add_argument("--fix-manifest", action="store_true",
                    help="rewrite the manifest with improved detection")
    args = ap.parse_args()

    works = load_works()
    report(works)
    if args.fix_manifest:
        fix_manifest(works)


if __name__ == "__main__":
    main()