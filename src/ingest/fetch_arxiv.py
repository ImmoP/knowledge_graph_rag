"""Download arXiv source packages for the arXiv subset of the corpus.

Reads data/manifest.jsonl, takes every entry with source_kind == "arxiv", and
fetches its e-print package from export.arxiv.org — the host arXiv asks
programmatic clients to use, so interactive readers are not affected.

    python -m src.ingest.fetch_arxiv --limit 3     # try a few first
    python -m src.ingest.fetch_arxiv               # the whole subset

Downloads land in data/raw/latex/ (or data/raw/pdf/ for submissions with no
source package) and are recorded in data/fulltext_manifest.jsonl.

Rate limiting: arXiv documents one request every three seconds. That is the
default here. Do not lower it — a blocked IP costs far more than the wait.

Licence note: arXiv e-prints stay under their authors' copyright. These files
are inputs for local processing only and must never be committed or
redistributed. data/raw/ is gitignored for exactly this reason.
"""



from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest.jsonl"
LATEX = ROOT / "data" / "raw" / "latex"
PDF = ROOT / "data" / "raw" / "pdf"
FULLTEXT_MANIFEST = ROOT / "data" / "fulltext_manifest.jsonl"
USER_AGENT = "kg-rag-corpus/0.1 (research project; contact: Immo.Primus@gmx.de)"

EPRINT = "https://export.arxiv.org/e-print/{arxiv_id}"

# Identify the client. arXiv asks for a contact address so they can reach you
# instead of silently blocking, if a script misbehaves.
USER_AGENT = "kg-rag-corpus/0.1 (research project; contact: you@example.com)"

GZIP_MAGIC = b"\x1f\x8b"
PDF_MAGIC = b"%PDF"


def load_targets() -> list[dict]:
    if not MANIFEST.exists():
        sys.exit(f"{MANIFEST} not found. Run the corpus build first.")
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line]
    if rows and "source_kind" not in rows[0]:
        sys.exit("Manifest has no source_kind field. Run:\n"
                 "  python -m src.ingest.inspect_sources --fix-manifest")
    return [r for r in rows if r.get("source_kind") == "arxiv" and r.get("arxiv_id")]


def already_have(arxiv_id: str) -> Path | None:
    for path in (LATEX / f"{arxiv_id}.tar.gz", PDF / f"{arxiv_id}.pdf"):
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def classify_payload(blob: bytes) -> tuple[str, Path]:
    """Decide what arXiv actually sent, by magic bytes rather than headers."""
    if blob.startswith(GZIP_MAGIC):
        return "gzip", LATEX
    if blob.startswith(PDF_MAGIC):
        return "pdf", PDF
    return "unknown", LATEX


def fetch_one(http: httpx.Client, arxiv_id: str, delay: float) -> dict | None:
    """Download one e-print with backoff. Returns a manifest row, or None."""
    url = EPRINT.format(arxiv_id=arxiv_id)

    for attempt in range(4):
        try:
            r = http.get(url)
        except httpx.RequestError as exc:
            print(f"    network error ({exc.__class__.__name__}), retrying")
            time.sleep(delay * (attempt + 2))
            continue

        if r.status_code == 429:
            wait = delay * (attempt + 2) * 2
            print(f"    rate limited, waiting {wait:.0f}s")
            time.sleep(wait)
            continue
        if r.status_code == 404:
            print("    404 — no e-print package for this id")
            return None
        if r.status_code >= 500:
            time.sleep(delay * (attempt + 2))
            continue
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}, skipping")
            return None

        blob = r.content
        if not blob:
            print("    empty response, skipping")
            return None

        kind, folder = classify_payload(blob)
        if kind == "unknown":
            print(f"    unrecognised payload ({blob[:8]!r}), storing anyway")

        folder.mkdir(parents=True, exist_ok=True)
        suffix = ".pdf" if kind == "pdf" else ".tar.gz"
        path = folder / f"{arxiv_id}{suffix}"
        path.write_bytes(blob)

        return {
            "arxiv_id": arxiv_id,
            "path": str(path.relative_to(ROOT)),
            "payload": kind,
            "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    print("    gave up after 4 attempts")
    return None


def write_fulltext_manifest(rows: list[dict]) -> None:
    """Merge with anything already recorded, keyed by arXiv ID."""
    existing: dict[str, dict] = {}
    if FULLTEXT_MANIFEST.exists():
        for line in FULLTEXT_MANIFEST.read_text().splitlines():
            if line:
                row = json.loads(line)
                existing[row["arxiv_id"]] = row
    for row in rows:
        existing[row["arxiv_id"]] = row

    lines = [json.dumps(existing[k], ensure_ascii=False) for k in sorted(existing)]
    FULLTEXT_MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nFull-text manifest: {len(lines)} entries")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download arXiv sources for the corpus")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N downloads (0 = no limit)")
    ap.add_argument("--delay", type=float, default=3.0,
                    help="seconds between requests (arXiv documents 3)")
    ap.add_argument("--force", action="store_true",
                    help="refetch files already on disk")
    args = ap.parse_args()

    if args.delay < 3.0:
        sys.exit("Refusing to run below 3s between requests — see arXiv's "
                 "bulk access guidance.")

    targets = load_targets()
    print(f"arXiv entries in manifest: {len(targets)}")

    todo = targets if args.force else [
        t for t in targets if not already_have(t["arxiv_id"])]
    if args.limit:
        todo = todo[:args.limit]
    print(f"To download now: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    estimate = len(todo) * args.delay / 60
    print(f"Estimated wall time at {args.delay}s per request: {estimate:.1f} min\n")

    rows, failed = [], []
    with httpx.Client(timeout=120.0, follow_redirects=True,
                      headers={"User-Agent": USER_AGENT}) as http:
        for i, target in enumerate(todo, 1):
            arxiv_id = target["arxiv_id"]
            title = (target.get("title") or "")[:48]
            print(f"[{i}/{len(todo)}] {arxiv_id}  {title}")

            row = fetch_one(http, arxiv_id, args.delay)
            if row:
                rows.append(row)
                print(f"    {row['payload']}, {row['bytes'] / 1024:.0f} KB")
            else:
                failed.append(arxiv_id)

            if i < len(todo):
                time.sleep(args.delay)

    if rows:
        write_fulltext_manifest(rows)

    kinds = {"gzip": 0, "pdf": 0, "unknown": 0}
    for row in rows:
        kinds[row["payload"]] = kinds.get(row["payload"], 0) + 1

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  succeeded:            {len(rows)}")
    print(f"    source packages:    {kinds['gzip']}")
    print(f"    PDF only:           {kinds['pdf']}")
    if kinds.get("unknown"):
        print(f"    unrecognised:       {kinds['unknown']}")
    print(f"  failed:               {len(failed)}")
    if failed:
        print("  " + ", ".join(failed))
    print("=" * 60)


if __name__ == "__main__":
    main()