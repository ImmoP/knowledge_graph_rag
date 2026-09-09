"""Corpus builder: expand a set of anchor papers into a citation-dense corpus.

Two modes:

    python -m src.ingest.fetch --resolve
        Look up each line in anchors.txt and show matching works, so you can
        pick the merged record rather than an arXiv preprint duplicate.

    python -m src.ingest.fetch --limit 150
        Build the corpus. Requires every anchor to be an OpenAlex work ID.

anchors.txt accepts one of these per line: a W-ID, a DOI, an arXiv ID, or a
title. Lines starting with # are ignored.

Outputs: one JSON file per work under data/raw/openalex/, a line per work in
data/manifest.jsonl, and an internal citation density report on stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

BASE = "https://api.openalex.org"
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "openalex"
MANIFEST = ROOT / "data" / "manifest.jsonl"
ANCHORS = ROOT / "anchors.txt"

# Slim projection used while exploring. Full records are only fetched for the
# works that actually make it into the corpus, since payload size drives cost.
SLIM = "id,doi,title,publication_year,cited_by_count,referenced_works"
ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}$")
WID_RE = re.compile(r"^W\d+$")


class Budget:
    """Tracks the cost OpenAlex reports per response and stops before overrun."""

    def __init__(self, limit_usd: float) -> None:
        self.limit = limit_usd
        self.spent = 0.0
        self.calls = 0

    def charge(self, meta: dict) -> None:
        self.spent += float(meta.get("cost_usd") or 0.0)
        self.calls += 1
        if self.spent > self.limit:
            raise SystemExit(
                f"\nBudget exhausted: {self.spent:.4f} USD after {self.calls} calls "
                f"(limit {self.limit:.2f}). Raise it with --max-cost."
            )


class Client:
    """Thin OpenAlex wrapper with retries, cursor paging and cost accounting."""

    def __init__(self, api_key: str, budget: Budget) -> None:
        self.key = api_key
        self.budget = budget
        self.http = httpx.Client(timeout=30.0, follow_redirects=True)

    def get(self, path: str, **params) -> dict:
        params["api_key"] = self.key
        for attempt in range(5):
            try:
                r = self.http.get(f"{BASE}{path}", params=params)
            except httpx.RequestError as exc:
                if attempt == 4:
                    raise
                print(f"  network error ({exc.__class__.__name__}), "
                      f"attempt {attempt + 2}/5")
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            data = r.json()
            self.budget.charge(data.get("meta", {}))
            return data
        raise SystemExit(f"Gave up after 5 attempts: {path} {params}")

    def paginate(self, path: str, max_items: int, **params):
        cursor, seen = "*", 0
        while cursor and seen < max_items:
            page = self.get(path, cursor=cursor, per_page=100, **params)
            for item in page.get("results", []):
                yield item
                seen += 1
                if seen >= max_items:
                    return
            cursor = page.get("meta", {}).get("next_cursor")


def short_id(openalex_id: str) -> str:
    """https://openalex.org/W123 -> W123"""
    return openalex_id.rsplit("/", 1)[-1]


def read_anchor_lines() -> list[str]:
    if not ANCHORS.exists():
        sys.exit(f"{ANCHORS} not found")
    return [ln.strip() for ln in ANCHORS.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]


def title_lookup(client: Client, title: str) -> list[dict]:
    """Search by title, relevance-ranked.

    Deliberately no sort parameter: any explicit sort overrides OpenAlex's
    relevance ranking, which turns a title lookup into "every work containing
    these words, most-cited first". title.search also beats plain search,
    which matches abstract and full text too. Commas and pipes are stripped
    because the filter grammar treats them as separators.
    """
    query = re.sub(r"[,|]", " ", title).strip()
    found = client.get("/works", filter=f"title.search:{query}",
                       per_page=8, select=SLIM)
    results = found.get("results", [])
    if not results:
        found = client.get("/works", search=query, per_page=8, select=SLIM)
        results = found.get("results", [])
    return results


def resolve_mode(client: Client, lines: list[str]) -> None:
    """Show candidate records per anchor so the user can pick the merged one.

    OpenAlex often holds both an arXiv preprint and the published version of
    the same paper. The preprint record usually carries few citations and no
    references at all, and picking it silently disconnects the graph.
    """
    for line in lines:
        print("\n" + "=" * 72)
        print(f"ANCHOR: {line}")
        print("=" * 72)

        if WID_RE.match(line):
            print("  Already a W-ID, nothing to do.")
            continue

        if line.lower().startswith("10.") or ARXIV_RE.match(line):
            doi = line.lower() if line.lower().startswith("10.") \
                else f"10.48550/arxiv.{line}"
            hit = client.get("/works", filter=f"doi:{doi}", select=SLIM)
            results = hit.get("results", [])
            if not results:
                print(f"  No DOI match for {doi}")
                continue
            title = results[0].get("title") or ""
            print(f"  Found via DOI: {title}")
            print("  Searching that title to surface duplicates:")
            line = title

        results = title_lookup(client, line)
        if not results:
            print("  No matches. Try rephrasing the title.")
            continue

        for w in results:
            refs = len(w.get("referenced_works") or [])
            warn = "   <-- no references, likely a duplicate" if refs == 0 else ""
            print(f"\n  {short_id(w['id'])}")
            print(f"    {(w.get('title') or '')[:66]}")
            print(f"    {w.get('publication_year')}  "
                  f"{w.get('cited_by_count', 0)} citations  "
                  f"{refs} references{warn}")

        print("\n  Pick the row whose title actually matches.")
        print("  If two rows share a title, take the one with references > 0.")

    print("\n" + "=" * 72)
    print("Update anchors.txt, then run again without --resolve.")


def load_anchors(client: Client, lines: list[str]) -> list[dict]:
    bad = [ln for ln in lines if not WID_RE.match(ln)]
    if bad:
        sys.exit(
            "These lines are not W-IDs:\n  " + "\n  ".join(bad) +
            "\n\nRun 'python -m src.ingest.fetch --resolve' first."
        )
    data = client.get("/works", filter=f"openalex:{'|'.join(lines)}",
                      per_page=100, select=SLIM)
    anchors = data.get("results", [])

    print(f"\nAnchors loaded: {len(anchors)} of {len(lines)}")
    print("-" * 72)
    for w in anchors:
        refs = len(w.get("referenced_works") or [])
        flag = "  <-- no references!" if refs == 0 else ""
        print(f"  {short_id(w['id']):<12} {w.get('cited_by_count', 0):>7} cit  "
              f"{refs:>4} refs  {(w.get('title') or '')[:38]}{flag}")
    return anchors


def expand(client: Client, anchors: list[dict], citing_per_anchor: int
           ) -> tuple[Counter, dict[str, dict]]:
    """Collect the two neighbourhoods of every anchor.

    Backward: what the anchor cites, read straight off referenced_works.
    Forward: what cites the anchor, which needs a separate cites: query.
    Returns candidate counts plus every slim record seen along the way.
    """
    votes: Counter = Counter()
    known: dict[str, dict] = {}

    for w in anchors:
        wid = short_id(w["id"])
        known[wid] = w
        refs = [short_id(r) for r in w.get("referenced_works", [])]
        votes.update(refs)

        citing = list(client.paginate(
            "/works", max_items=citing_per_anchor,
            filter=f"cites:{wid}", sort="cited_by_count:desc", select=SLIM))
        for c in citing:
            known[short_id(c["id"])] = c
            votes[short_id(c["id"])] += 1
        print(f"  {wid}: {len(refs)} references, {len(citing)} citing works")

    for w in anchors:
        votes[short_id(w["id"])] += 1000
    return votes, known


def fill_slim(client: Client, ids: list[str], known: dict[str, dict]) -> None:
    """Backfill slim records so every candidate has a reference list."""
    missing = [i for i in ids if i not in known]
    if not missing:
        return
    print(f"\nBackfilling slim records: {len(missing)}")
    for start in range(0, len(missing), 100):
        block = missing[start:start + 100]
        data = client.get("/works", filter=f"openalex:{'|'.join(block)}",
                          per_page=100, select=SLIM)
        for w in data.get("results", []):
            known[short_id(w["id"])] = w


def build_adjacency(candidates: set[str], known: dict[str, dict]
                    ) -> dict[str, set[str]]:
    """Undirected neighbourhood: an edge whenever one candidate cites another."""
    adj: dict[str, set[str]] = defaultdict(set)
    for cid in candidates:
        rec = known.get(cid)
        if not rec:
            continue
        for ref in rec.get("referenced_works") or []:
            rid = short_id(ref)
            if rid in candidates and rid != cid:
                adj[cid].add(rid)
                adj[rid].add(cid)
    for cid in candidates:
        adj.setdefault(cid, set())
    return adj


def k_core(adj: dict[str, set[str]], k: int, keep: set[str]) -> set[str]:
    """Repeatedly drop nodes with fewer than k neighbours until stable.

    Dropping one node can push its neighbours below the threshold, so this
    has to iterate. Nodes in `keep` (the anchors) are never removed.
    """
    nodes = set(adj)
    while True:
        drop = {n for n in nodes
                if n not in keep and sum(1 for m in adj[n] if m in nodes) < k}
        if not drop:
            return nodes
        nodes -= drop


def select(votes: Counter, known: dict[str, dict], anchor_ids: set[str],
           limit: int, min_degree: int) -> list[str]:
    """Keep the densely connected core, ranked by internal degree."""
    candidates = set(votes)
    adj = build_adjacency(candidates, known)

    print(f"\nCandidates: {len(candidates)}")
    for k in (1, 2, 3):
        print(f"  {k}-core: {len(k_core(adj, k, anchor_ids))} works")

    core = k_core(adj, min_degree, anchor_ids)
    ranked = sorted(
        core,
        key=lambda n: (-sum(1 for m in adj[n] if m in core),
                       -(known.get(n, {}).get("cited_by_count") or 0)),
    )
    chosen = ranked[:limit]
    print(f"\nSelected from the {min_degree}-core: {len(chosen)}")
    if len(chosen) < limit:
        print("  Note: core is smaller than --limit. Raise --citing-per-anchor "
              "or lower --min-degree.")
    return chosen


def fetch_full(client: Client, ids: list[str], force: bool) -> list[dict]:
    """Fetch complete records for the selected works, skipping ones on disk."""
    RAW.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if force or not (RAW / f"{i}.json").exists()]
    print(f"\nFull records to fetch: {len(todo)} (already on disk: "
          f"{len(ids) - len(todo)})")

    got: set[str] = set()
    for start in range(0, len(todo), 100):
        block = todo[start:start + 100]
        data = client.get("/works", filter=f"openalex:{'|'.join(block)}", per_page=100)
        for w in data.get("results", []):
            wid = short_id(w["id"])
            (RAW / f"{wid}.json").write_text(
                json.dumps(w, ensure_ascii=False, indent=2), encoding="utf-8")
            got.add(wid)

    silent = [i for i in todo if i not in got]
    if silent:
        print(f"  No response for {len(silent)} ids (merged or deleted upstream)")

    works = []
    for wid in ids:
        path = RAW / f"{wid}.json"
        if path.exists():
            works.append(json.loads(path.read_text(encoding="utf-8")))
    return works


def arxiv_id_of(work: dict) -> str | None:
    doi = (work.get("doi") or "").lower()
    if "arxiv." in doi:
        return doi.split("arxiv.", 1)[1]
    for loc in work.get("locations") or []:
        url = ((loc.get("landing_page_url") or "") + (loc.get("pdf_url") or "")).lower()
        if "arxiv.org/abs/" in url:
            return url.split("arxiv.org/abs/", 1)[1].split("v")[0].strip("/ ")
    return None


def write_manifest(works: list[dict]) -> None:
    """One line per work, hashed so a later run can prove the data is unchanged."""
    now = datetime.now(timezone.utc).isoformat()
    lines = []
    for w in sorted(works, key=lambda x: x["id"]):
        wid = short_id(w["id"])
        raw = (RAW / f"{wid}.json").read_bytes()
        lines.append(json.dumps({
            "openalex_id": wid,
            "doi": w.get("doi"),
            "arxiv_id": arxiv_id_of(w),
            "title": w.get("title"),
            "year": w.get("publication_year"),
            "cited_by_count": w.get("cited_by_count"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "fetched_at": now,
            "source_url": w["id"],
        }, ensure_ascii=False))
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nManifest written: {len(lines)} lines")


def density(works: list[dict]) -> None:
    """Share of reference edges that stay inside the corpus.

    This is the go/no-go number: below roughly 15 percent there are not enough
    internal paths for multi-hop questions to have an answer.
    """
    inside = {short_id(w["id"]) for w in works}
    total = internal = isolated = 0
    per_work = []
    for w in works:
        refs = [short_id(r) for r in w.get("referenced_works", [])]
        hits = sum(1 for r in refs if r in inside)
        total += len(refs)
        internal += hits
        if refs:
            per_work.append(hits / len(refs))
        if hits == 0:
            isolated += 1

    ratio = internal / total if total else 0.0
    print("\n" + "=" * 72)
    print("CITATION DENSITY")
    print("=" * 72)
    print(f"  works in corpus:                {len(works)}")
    print(f"  reference edges total:          {total}")
    print(f"  of those inside the corpus:     {internal} ({ratio:.1%})")
    if per_work:
        print(f"  median per work:                {statistics.median(per_work):.1%}")
    print(f"  works with no internal ref:     {isolated}")
    print()
    print("  Dense enough." if ratio >= 0.15 else "  Too sparse (under 15%).")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a citation-dense corpus from anchor papers via OpenAlex")
    ap.add_argument("--resolve", action="store_true",
                    help="only resolve anchors and print candidate records")
    ap.add_argument("--limit", type=int, default=150,
                    help="target corpus size")
    ap.add_argument("--citing-per-anchor", type=int, default=400,
                    help="how many citing works to pull per anchor")
    ap.add_argument("--min-degree", type=int, default=2,
                    help="minimum internal neighbours a work must keep")
    ap.add_argument("--max-cost", type=float, default=0.40,
                    help="spend limit in USD (free daily allowance: 1.00)")
    ap.add_argument("--force", action="store_true",
                    help="refetch works already on disk")
    ap.add_argument("--dry-run", action="store_true",
                    help="explore and select, but write nothing")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENALEX_API_KEY")
    if not key:
        sys.exit("OPENALEX_API_KEY missing from .env")

    lines = read_anchor_lines()
    budget = Budget(args.max_cost)
    client = Client(key, budget)

    if args.resolve:
        resolve_mode(client, lines)
        print(f"\nCost: {budget.spent:.4f} USD in {budget.calls} calls.")
        return

    anchors = load_anchors(client, lines)
    if not anchors:
        sys.exit("No anchors loaded, aborting.")

    print("\nExpanding:")
    votes, known = expand(client, anchors, args.citing_per_anchor)
    fill_slim(client, list(votes), known)

    anchor_ids = {short_id(w["id"]) for w in anchors}
    chosen = select(votes, known, anchor_ids, args.limit, args.min_degree)

    if args.dry_run:
        print(f"\nDry run, nothing written. Cost: {budget.spent:.4f} USD "
              f"in {budget.calls} calls.")
        return

    works = fetch_full(client, chosen, args.force)
    write_manifest(works)
    density(works)
    print(f"\nCost: {budget.spent:.4f} USD in {budget.calls} calls.")


if __name__ == "__main__":
    main()