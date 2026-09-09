"""Verify corpus metadata against the downloaded LaTeX sources.

OpenAlex records can carry a correct DOI and arXiv link but a title scraped
from somewhere else. The downloaded source package is the ground truth: it
contains the paper's own \\title{...}. This script compares the two and also
reports works that share an arXiv ID, which means one paper is held under two
OpenAlex records.

Picking the right .tex file matters more than it sounds. A source package
routinely contains the conference style template (whose \\title is
"Instructions for ACL 2019 Proceedings") and sometimes leftover files from the
authors' previous paper. Taking the first \\title found gives the wrong answer
often enough to be useless, so candidate files are ranked: a file must have
\\begin{document} and a \\title, must not look like a template, and among what
survives, the one with the most \\cite commands wins — the real paper cites,
the template does not.

Reads only local files. No network access, no cost.

    python -m src.ingest.verify_titles
    python -m src.ingest.verify_titles --show-all
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import tarfile
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest.jsonl"
LATEX = ROOT / "data" / "raw" / "latex"

TITLE_RE = re.compile(r"\\title\s*(?:\[[^\]]*\])?\s*\{", re.I)
COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.M)
LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+\s*")
CITE_RE = re.compile(r"\\cite[a-zA-Z]*\s*[\[{]")

# Phrases that identify a conference template rather than a paper.
TEMPLATE_MARKERS = (
    "instructions", "formatting", "template", "bare demo", "sample paper",
    "your paper", "paper title", "guidelines for", "author guidelines",
    "style file", "latex example",
)


def balanced_group(text: str, start: int) -> str:
    """Read a brace-balanced group starting at the opening brace index."""
    depth, out = 0, []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out)


def normalize(text: str) -> str:
    """Strip LaTeX markup and punctuation so two titles can be compared."""
    text = COMMENT_RE.sub("", text)
    text = text.replace("\\\\", " ").replace("~", " ")
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = LATEX_CMD_RE.sub(" ", text)
    text = re.sub(r"[{}\\]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.lower().split())


def tex_members(path: Path):
    """Yield (name, text) for every .tex file, tar archive or bare gzip."""
    blob = path.read_bytes()
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(".tex"):
                    handle = tar.extractfile(member)
                    if handle:
                        yield member.name, handle.read().decode("utf-8",
                                                                errors="replace")
            return
    except tarfile.TarError:
        pass
    try:
        yield path.name, gzip.decompress(blob).decode("utf-8", errors="replace")
    except OSError:
        return


def looks_like_template(title: str) -> bool:
    return any(marker in title for marker in TEMPLATE_MARKERS)


def title_from_source(path: Path) -> tuple[str | None, str | None]:
    """Return (title, source filename) for the most plausible main file."""
    candidates = []
    for name, text in tex_members(path):
        if "\\begin{document}" not in text:
            continue
        match = TITLE_RE.search(text)
        if not match:
            continue
        title = normalize(balanced_group(text, match.end() - 1))
        if len(title) <= 8 or looks_like_template(title):
            continue
        candidates.append((len(CITE_RE.findall(text)), title, name))

    if not candidates:
        return None, None
    candidates.sort(key=lambda c: -c[0])
    _, title, name = candidates[0]
    return title, name


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def main() -> None:
    ap = argparse.ArgumentParser(description="Check manifest titles against sources")
    ap.add_argument("--show-all", action="store_true",
                    help="list every comparison, not just the problems")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="similarity below this counts as a mismatch")
    args = ap.parse_args()

    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line]

    by_arxiv: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("arxiv_id"):
            by_arxiv[row["arxiv_id"]].append(row)

    duplicates = {k: v for k, v in by_arxiv.items() if len(v) > 1}

    mismatches, checked, unreadable = [], 0, []
    for arxiv_id, entries in sorted(by_arxiv.items()):
        path = LATEX / f"{arxiv_id}.tar.gz"
        if not path.exists():
            continue
        source_title, source_file = title_from_source(path)
        if not source_title:
            unreadable.append(arxiv_id)
            continue
        checked += 1
        manifest_title = normalize(entries[0].get("title") or "")
        score = similarity(manifest_title, source_title)
        if args.show_all:
            print(f"{score:.2f}  {arxiv_id}  {source_title[:52]}  [{source_file}]")
        if score < args.threshold:
            mismatches.append((arxiv_id, entries[0], source_title,
                               source_file, score))

    print("\n" + "=" * 72)
    print("DUPLICATE ARXIV IDS")
    print("=" * 72)
    if not duplicates:
        print("  none")
    for arxiv_id, entries in sorted(duplicates.items()):
        print(f"\n  {arxiv_id}")
        for e in entries:
            title = (e.get("title") or "(empty title)")[:44]
            print(f"    {e['openalex_id']:<12} {e.get('cited_by_count', 0):>6} cit  "
                  f"{title}")

    print("\n" + "=" * 72)
    print("TITLE MISMATCHES")
    print("=" * 72)
    if not mismatches:
        print("  none")
    for arxiv_id, entry, source_title, source_file, score in mismatches:
        print(f"\n  {arxiv_id}  ({entry['openalex_id']}, similarity {score:.2f})")
        print(f"    manifest: {(entry.get('title') or '')[:60]}")
        print(f"    source:   {source_title[:60]}")
        print(f"    from:     {source_file}")

    print("\n" + "=" * 72)
    print(f"  packages compared:   {checked}")
    print(f"  title mismatches:    {len(mismatches)}")
    print(f"  duplicate ids:       {len(duplicates)} "
          f"({sum(len(v) for v in duplicates.values())} records)")
    print(f"  no usable title:     {len(unreadable)}")
    print("=" * 72)


if __name__ == "__main__":
    main()