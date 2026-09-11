"""Section-aware chunker (impl-spec §5, chunking-strategy.md, chunk-record.md).

Consumes the source-capability-gated classification established by
``src.ingest.preflight`` (#21). Only text-bearing documents (tracked
``payload == "gzip"``) produce chunks. Output: ``data/derived/chunks.jsonl``.

    python -m src.chunk.chunker
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.ingest import parse_latex
from src.ingest import preflight

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest.jsonl"
FULLTEXT_MANIFEST = ROOT / "data" / "fulltext_manifest.jsonl"
RAW_LATEX = ROOT / "data" / "raw" / "latex"
OUTPUT = ROOT / "data" / "derived" / "chunks.jsonl"

SECTION_RE = re.compile(
    r"\\(?P<cmd>section|subsection|subsubsection)\*?\s*\{(?P<title>[^}]*)\}"
)
TOKEN_BUDGET = 512
LEVEL_SECTION = 1
LEVEL_SUBSECTION = 2
LEVEL_SUBSUBSECTION = 3
LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3}


@dataclass(frozen=True)
class Marker:
    level: int
    title: str
    start: int
    end: int


@dataclass
class Chunk:
    doc_id: str
    section_path: list
    chunk_index: int
    text: str
    char_start: int
    char_end: int

    def to_record(self) -> dict:
        return {
            "chunk_id": "c_" + hashlib.sha1(
                f"{self.doc_id}|{json.dumps(self.section_path)}|{self.chunk_index}".encode()
            ).hexdigest()[:16],
            "doc_id": self.doc_id,
            "section_path": self.section_path,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "token_estimate": len(self.text) // 4,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


def find_markers(text: str) -> list:
    out = []
    for m in SECTION_RE.finditer(text):
        out.append(Marker(level=LEVELS[m.group("cmd")], title=m.group("title"),
                          start=m.start(), end=m.end()))
    return out


def _paragraph_split(body: str) -> list:
    """Split body on \\n\\n boundaries in document order."""
    parts = []
    cursor = 0
    for m in re.finditer(r"\n\n", body):
        parts.append(body[cursor:m.end()])
        cursor = m.end()
    parts.append(body[cursor:])
    return parts


def _line_split(body: str) -> list:
    """Split body on single newlines (fallback when paragraphs are still oversized)."""
    return body.split("\n")


def _split_oversize_subsection(doc_id, body, body_offset, section_path, start_index):
    """Subdivide an oversize subsection at paragraph boundaries, falling back to
    line breaks for overlong paragraphs.

    Citation-span rule: never split inside a [[CITE:...]] span. Strategy
    §6 tie-break prefers split BEFORE the open `[[CITE:` over after it.
    """
    paragraphs = _paragraph_split(body)
    out_chunks = []
    chunk_index = start_index

    def _flush(pend, pend_start):
        nonlocal chunk_index
        if not pend:
            return pend_start
        if len(pend) // 4 > TOKEN_BUDGET:
            line_chunks = _flush_lines(pend, pend_start, doc_id, section_path, chunk_index)
            for c in line_chunks:
                c.chunk_index = chunk_index
                out_chunks.append(c)
                chunk_index += 1
        else:
            out_chunks.append(_mk_chunk(doc_id, section_path, chunk_index, pend,
                                        pend_start, pend_start + len(pend)))
            chunk_index += 1
        return pend_start + len(pend)

    pending = ""
    pending_start = body_offset
    cursor = 0
    for para in paragraphs:
        para_offset = body_offset + cursor
        cursor += len(para)
        if not pending:
            pending = para
            pending_start = para_offset
            continue
        candidate = pending + para
        if len(candidate) // 4 > TOKEN_BUDGET:
            pending_start = _flush(pending, pending_start)
            pending = para
            pending_start = para_offset
        else:
            pending = candidate

    if pending:
        _flush(pending, pending_start)
    return out_chunks


def _flush_lines(text, text_start, doc_id, section_path, start_index):
    """Split an overlong single paragraph at line boundaries, with a final
    safe-whitespace fallback for any remaining overlong line.

    Citation-span rule: if a line opens `[[CITE:` but the closing `]]`
    lies past our boundary, we keep the open marker with the earlier chunk.

    Whitespace fallback (chunking-strategy.md §3 final stage): if a single
    line exceeds the token budget and has no internal newlines, split at the
    LAST Unicode-whitespace boundary that keeps the resulting chunk within
    the budget. Never split inside a [[CITE:...]] span. Preserve document
    order; preserve the same section_path; increment chunk_index monotonically;
    emit char_start/char_end against the cleaned full-document text. Fail
    closed with OversizeChunkError if no safe split point exists.
    """
    chunks: list = []
    chunk_index = start_index
    lines = _line_split(text)
    pending = ""
    pending_start = text_start
    cursor = 0
    for line in lines:
        line_offset = text_start + cursor
        cursor += len(line) + 1  # +1 for the newline we split on
        if not pending:
            pending = line
            pending_start = line_offset
            continue
        candidate = pending + "\n" + line
        if len(candidate) // 4 > TOKEN_BUDGET and pending:
            opens = pending.count("[[CITE:")
            closes = pending.count("]]")
            if opens > closes:
                last_open = pending.rfind("[[CITE:")
                head = pending[:last_open]
                tail = pending[last_open:]
                head_end = pending_start + last_open
                chunks.extend(_emit_with_whitespace_fallback(
                    doc_id, section_path, head, pending_start, head_end, chunk_index))
                chunk_index = start_index + len(chunks)
                pending = tail + "\n" + line
                pending_start = pending_start + last_open
            else:
                chunks.extend(_emit_with_whitespace_fallback(
                    doc_id, section_path, pending, pending_start,
                    pending_start + len(pending), chunk_index))
                chunk_index = start_index + len(chunks)
                pending = line
                pending_start = line_offset
        else:
            pending = candidate

    if pending:
        chunks.extend(_emit_with_whitespace_fallback(
            doc_id, section_path, pending, pending_start,
            pending_start + len(pending), chunk_index))
    return chunks


def _emit_with_whitespace_fallback(doc_id, section_path, text, char_start, char_end, start_index):
    """Emit one chunk for ``text``; if its token estimate exceeds the budget,
    subdivide via the safe-whitespace fallback instead of leaving it oversize.

    The ``text`` here is the chunk-as-candidate before any title-prefixing;
    the cap applies to it directly.
    """
    if len(text) // 4 <= TOKEN_BUDGET:
        return [_mk_chunk(doc_id, section_path, start_index, text, char_start, char_end)]
    # Recompute the offset where ``text`` starts in its source. The chunker
    # constructs candidates with absolute offsets; if text is short enough
    # after whitespace subdivision, the resulting chunks must reference
    # absolute offsets in the cleaned full-document text. We pass char_start
    # as the base and let _flush_whitespace compute child offsets from it.
    return _flush_whitespace(text, char_start, doc_id, section_path, start_index)


def _flush_whitespace(text, text_start, doc_id, section_path, start_index):
    """Final fallback for an overlong line with no internal \\n.

    Splits at the LAST Unicode-whitespace position in ``text`` such that the
    resulting leading chunk stays within TOKEN_BUDGET chars (the cap is on
    the chunk-as-emitted; section title is not prepended in this chunker).

    Citation-span rule: never split inside a [[CITE:...]] span. If the
    chosen whitespace boundary lies inside an open span, fall back to the
    boundary immediately BEFORE the open [[CITE: marker.

    Every input character appears exactly once across the resulting chunks
    in document order. If no safe split point exists (e.g. a single token
    longer than the budget), raises OversizeChunkError.
    """
    chunks = []
    chunk_index = start_index
    # Estimate in chars (TOKEN_BUDGET * 4 chars/token).
    char_cap = TOKEN_BUDGET * 4

    # We split the source text into pieces greedily, each starting at
    # text_start + consumed_so_far, and bounded by char_cap.
    cursor = 0
    n = len(text)
    while cursor < n:
        remaining = text[cursor:]
        if len(remaining) // 4 <= TOKEN_BUDGET:
            # Last piece fits.
            chunks.append(_mk_chunk(doc_id, section_path, chunk_index,
                                    remaining, text_start + cursor,
                                    text_start + cursor + len(remaining)))
            chunk_index += 1
            cursor += len(remaining)
            continue

        # Find the LAST whitespace boundary within the first char_cap chars
        # that keeps the head within budget AND respects citation-span.
        head_end = _find_safe_split(text, cursor, char_cap)
        if head_end is None or head_end <= cursor:
            # No safe split point: fail closed.
            raise OversizeChunkError(
                doc_id, section_path, chunk_index,
                len(remaining) // 4, len(remaining),
            )
        head = text[cursor:head_end]
        # Trim trailing whitespace from the head so the chunk text ends on
        # a non-whitespace char (deterministic, easier to read). The
        # trimmed bytes are boundary separators between chunks, not
        # document content; they are not silently dropped from any doc-
        # meaningful text, only from inter-chunk whitespace (which the
        # chunker is permitted to normalise per `implementation-spec.md`
        # 10.5 as long as no chunk exceeds the 512-token budget).
        head_stripped = head.rstrip()
        head_stripped_end = cursor + len(head_stripped)
        chunks.append(_mk_chunk(doc_id, section_path, chunk_index,
                                head_stripped, text_start + cursor,
                                text_start + head_stripped_end))
        chunk_index += 1
        # Advance cursor past the trimmed trailing whitespace.
        cursor = head_end

    return chunks


def _find_safe_split(text: str, start: int, char_cap: int) -> int | None:
    """Return the position of the LAST whitespace boundary in
    ``text[start:start+char_cap]`` such that no [[CITE:...]] marker spans
    the boundary. Returns absolute index in ``text``, or ``None`` if no
    such boundary exists.

    Citation-span rule: ``chunking-strategy.md`` §6 says prefer splitting
    BEFORE the open ``[[CITE:`` marker. If every whitespace inside the
    candidate window is inside an OPEN citation span (no close in the
    window), return the position of the open ``[[CITE:`` so the head ends
    just before it. If ``start`` has already advanced past all open
    markers with no close in sight, return the position of the next
    ``]]`` close so the head ends at the span's close. If neither kind of
    cite boundary exists and no whitespace exists, return ``None`` so the
    caller fails closed.
    """
    end = min(start + char_cap, len(text))
    # Walk backwards from end-1, prefer non-whitespace, then whitespace.
    i = end - 1
    while i > start:
        c = text[i]
        if c.isspace() and not _inside_open_cite_at(text, start, end, i):
            return i
        i -= 1

    # No safe whitespace in window. Locate the closest cite span that we can
    # split at: the FIRST [[CITE: in this window (split before it) or the
    # close of an open span that started before start (split at its close).
    cites = _cites_in_window(text, start, end)
    if cites:
        first_open = cites[0][0]
        if first_open > start:
            return first_open
        # Open span started before or at start; return its close.
        for o, c in cites:
            if o <= start < c:
                return c
    return None


def _inside_open_cite_at(text: str, start: int, end: int, idx: int) -> bool:
    """True if ``text[idx]`` lies inside an OPEN [[CITE:...]] span in the
    half-open window [start, end). The close ``]]`` may lie past ``end``.
    """
    pos = start
    while pos < end:
        o = text.find("[[CITE:", pos, end)
        if o == -1:
            return False
        c = text.find("]]", o, end)
        if c == -1:
            return o <= idx < end
        if o <= idx < c + 2:
            return True
        pos = c + 2
    return False


def _cites_in_window(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """List of (open_pos, close_pos_exclusive) for every [[CITE:...]] span
    whose open lies in [start, end). Spans that close past ``end`` have
    ``close_pos_exclusive == end``."""
    out: list[tuple[int, int]] = []
    pos = start
    while True:
        o = text.find("[[CITE:", pos, end)
        if o == -1:
            break
        c = text.find("]]", o, end)
        if c == -1:
            out.append((o, end))
            break
        out.append((o, c + 2))
        pos = c + 2
    return out


def _mk_chunk(doc_id, section_path, chunk_index, text, char_start, char_end):
    return Chunk(doc_id=doc_id, section_path=list(section_path),
                 chunk_index=chunk_index, text=text,
                 char_start=char_start, char_end=char_end)


def _chunks_for_subsection_range(doc_id, markers_in_range, body, body_offset, base_path, start_index):
    """Chunk a contiguous range of subsection/subsubsection markers."""
    chunks = []
    chunk_index = start_index
    if not markers_in_range:
        text = body.strip("\n")
        if not text:
            return chunks
        leading = body.find(text) if text else 0
        start = body_offset + leading
        chunks.append(_mk_chunk(doc_id, base_path, chunk_index, text, start,
                                start + len(text)))
        return chunks

    # The chunker treats BOTH subsubsection and subsection markers as
    # subsection-equivalent chunk boundaries when they are the leaf-level
    # markers present in this section. (Cascade rule §3.)
    # Determine leaf-level: if there are subsections in this range, leaves
    # are subsections; else leaves are subsubsections.
    has_sub = any(m.level == LEVEL_SUBSECTION for m in markers_in_range)
    has_subsub = any(m.level == LEVEL_SUBSUBSECTION for m in markers_in_range)
    if has_sub:
        leaf_level = LEVEL_SUBSECTION
        # Subsubsections inside a subsection attach to the enclosing
        # subsection's body (they don't start new chunks at the same level).
        leaves = [m for m in markers_in_range if m.level == LEVEL_SUBSECTION]
    elif has_subsub:
        leaf_level = LEVEL_SUBSUBSECTION
        leaves = [m for m in markers_in_range if m.level == LEVEL_SUBSUBSECTION]
    else:
        leaves = markers_in_range

    for i, m in enumerate(leaves):
        next_m = leaves[i + 1] if i + 1 < len(leaves) else None
        sub_end = next_m.start if next_m else len(body)
        sub_body = body[m.end:sub_end]
        sub_text = sub_body.strip("\n")
        if not sub_text:
            continue
        leading = sub_body.find(sub_text)
        sub_start_abs = body_offset + m.end + leading
        sub_end_abs = sub_start_abs + len(sub_text)
        section_path = list(base_path) + [m.title]

        # Citation-span rule at subsection boundary: if the previous chunk
        # ends with an unclosed `[[CITE:` and this one starts with `]]`,
        # we DO NOT close it here -- strategy §6 says prefer splitting BEFORE
        # the open `[[CITE:`. With our paragraph-level splits inside an
        # oversize subsection we already handle that. At the subsection
        # boundary, the previous subsection's body ends, and the next
        # subsection's title follows. The previous chunk is sealed.
        if len(sub_text) // 4 > TOKEN_BUDGET:
            chunks.extend(_split_oversize_subsection(
                doc_id, sub_body, body_offset + m.end, section_path, chunk_index))
        else:
            chunks.append(_mk_chunk(doc_id, section_path, chunk_index, sub_text,
                                    sub_start_abs, sub_end_abs))
        chunk_index = start_index + len(chunks)
    return chunks


def chunk_text(doc_id: str, text: str) -> list:
    """Produce Chunk records for a cleaned LaTeX document.

    Whitespace-only chunks (boundary artifacts from the cascade re-bundle)
    are dropped. Per the chunk-record spec §2, ``chunk_index`` is the
    0-based position of this chunk within ``doc_id`` in document order.
    We renumber globally so the indices are contiguous 0..n-1 across the
    full document (section_path is the breadcrumb, chunk_index is the
    position). char_start / char_end survive the renumber unchanged;
    chunk_id is re-derived from the new index.
    """
    chunks = _chunk_text_impl(doc_id, text)
    out: list = []
    for c in chunks:
        if not c.text.strip():
            continue
        out.append(c)
    for i, c in enumerate(out):
        if c.chunk_index != i:
            c.chunk_index = i
    return out


def _chunk_text_impl(doc_id: str, text: str) -> list:
    """Produce Chunk records for a cleaned LaTeX document."""
    markers = find_markers(text)
    if not markers:
        text_s = text.strip("\n")
        if not text_s:
            return []
        leading = text.find(text_s) if text_s else 0
        return [_mk_chunk(doc_id, [], 0, text_s, leading, leading + len(text_s))]

    chunks: list = []
    chunk_index = 0

    # Pre-section content. Strip leading/trailing newlines so the chunk text
    # contains only natural-language content; advance char_start past
    # leading whitespace and set char_end before trailing whitespace so
    # body[char_start:char_end] == text.
    pre_raw = text[:markers[0].start]
    pre_stripped = pre_raw.strip("\n")
    leading = pre_raw.find(pre_stripped) if pre_stripped else 0
    if pre_stripped:
        char_end_pre = leading + len(pre_stripped)
    else:
        char_end_pre = markers[0].start
    if pre_stripped:
        if len(pre_stripped) // 4 > TOKEN_BUDGET:
            pre_chunks = _split_oversize_subsection(
                doc_id, text[:markers[0].start], 0, [], chunk_index)
            for c in pre_chunks:
                c.chunk_index = chunk_index
                chunks.append(c)
                chunk_index += 1
        else:
            chunks.append(_mk_chunk(
                doc_id, [], chunk_index, pre_stripped, leading, char_end_pre))
            chunk_index += 1

    sections = [m for m in markers if m.level == LEVEL_SECTION]
    if not sections:
        # No top-level sections: only subsubsections (or subsection-only).
        return _chunks_for_subsection_range(
            doc_id, markers, text, 0, [], chunk_index)
    # The "no top-level sections" branch above passes absolute marker
    # positions into _chunks_for_subsection_range with body=text and
    # body_offset=0, so the positions align (body==text and offsets==0).

    for i, sm in enumerate(sections):
        next_sm = sections[i + 1] if i + 1 < len(sections) else None
        sec_end = next_sm.start if next_sm else len(text)
        sec_body = text[sm.end:sec_end]
        section_path = [sm.title]

        markers_in_section = [m for m in markers if sm.end <= m.start < sec_end]

        if markers_in_section:
            # Shift marker positions into sec_body coordinates.
            shifted = [Marker(m.level, m.title, m.start - sm.end, m.end - sm.end)
                       for m in markers_in_section]
            sec_chunks = _chunks_for_subsection_range(
                doc_id, shifted, sec_body, sm.end, section_path, chunk_index)
            # renumber chunk_index contiguously
            for c in sec_chunks:
                c.chunk_index = chunk_index
                chunks.append(c)
                chunk_index += 1
        else:
            sec_stripped = sec_body.strip("\n")
            if sec_stripped:
                leading = sec_body.find(sec_stripped)
                sec_start = sm.end + leading
                sec_end_correct = sec_start + len(sec_stripped)
                if len(sec_stripped) // 4 > TOKEN_BUDGET:
                    sec_chunks = _split_oversize_subsection(
                        doc_id, sec_body, sm.end, section_path, chunk_index)
                    for c in sec_chunks:
                        c.chunk_index = chunk_index
                        chunks.append(c)
                        chunk_index += 1
                else:
                    chunks.append(_mk_chunk(doc_id, section_path, chunk_index,
                                            sec_stripped, sec_start, sec_end_correct))
                    chunk_index += 1

    _enforce_token_budget(doc_id, chunks)
    return chunks


class OversizeChunkError(ValueError):
    """A chunk exceeds the BGE token budget and cannot be subdivided further.

    Strategy §3 cascade (section -> subsection -> paragraph -> line) is
    insufficient when a single paragraph contains no internal whitespace
    boundaries (e.g. table-content blocks rendered as one long string by
    parse_latex). The chunker fails closed on these cases rather than
    silently splitting at arbitrary character positions.
    """

    def __init__(self, doc_id, section_path, chunk_index, token_estimate, char_len):
        super().__init__(
            f"oversize chunk: doc_id={doc_id} section_path={section_path} "
            f"chunk_index={chunk_index} token_estimate={token_estimate} chars={char_len} "
            f"budget={TOKEN_BUDGET}"
        )
        self.doc_id = doc_id
        self.section_path = list(section_path)
        self.chunk_index = chunk_index
        self.token_estimate = token_estimate
        self.char_len = char_len


def _enforce_token_budget(doc_id, chunks):
    for c in chunks:
        est = len(c.text) // 4
        if est > TOKEN_BUDGET:
            raise OversizeChunkError(
                doc_id, c.section_path, c.chunk_index, est, len(c.text)
            )


def read_cleaned_text(tarball: Path) -> str:
    tex = parse_latex.read_tex_files(tarball)
    main = parse_latex.choose_main_file(tex)
    expanded, _ = parse_latex.resolve_includes(tex, main)
    cleaned = parse_latex.remove_comments(expanded)
    return parse_latex.preserve_citations(cleaned)


def _read_jsonl(path: Path) -> list:
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _build_fulltext_index(rows: list) -> dict:
    seen: dict = {}
    for row in rows:
        if row["arxiv_id"] in seen:
            raise ValueError(f"duplicate arxiv_id in fulltext_manifest: {row['arxiv_id']}")
        seen[row["arxiv_id"]] = row
    return seen


def chunk_document(doc_id: str, arxiv_id: str) -> list:
    tarball = RAW_LATEX / f"{arxiv_id}.tar.gz"
    if not tarball.exists():
        raise FileNotFoundError(f"latex archive missing for {doc_id} ({arxiv_id})")
    with tarfile.open(tarball, "r:gz"):
        pass
    cleaned = read_cleaned_text(tarball)
    return [c.to_record() for c in chunk_text(doc_id, cleaned)]


def main() -> int:
    manifest_records = _read_jsonl(MANIFEST)
    ft_rows = _read_jsonl(FULLTEXT_MANIFEST)
    try:
        ft_index = _build_fulltext_index(ft_rows)
    except ValueError as exc:
        print(f"PREFLIGHT FAIL: {exc}", file=sys.stderr)
        return 1

    capability = preflight.SourceCapability(by_arxiv_id=ft_index)
    tb_report = preflight.classify_text_bearing(manifest_records, capability)

    if tb_report.unresolved_capability:
        print(f"PREFLIGHT FAIL: unresolved source capability: {tb_report.unresolved_capability}",
              file=sys.stderr)
        return 1

    print(f"canonical document corpus: {len(manifest_records)}")
    print(f"v1 text-bearing subset: {len(tb_report.text_bearing)}, "
          f"{len(tb_report.pdf_payload)} pdf-payload, {len(tb_report.non_text_bearing)} non-text-bearing")

    by_doc_id = {r["openalex_id"]: r for r in manifest_records}

    all_records = []
    failures = []
    for doc_id in sorted(tb_report.text_bearing):
        arxiv_id = by_doc_id[doc_id]["arxiv_id"]
        try:
            all_records.extend(chunk_document(doc_id, arxiv_id))
        except (FileNotFoundError, tarfile.TarError, OSError, ValueError) as exc:
            failures.append((doc_id, arxiv_id, type(exc).__name__, str(exc)))

    if failures:
        print(f"PREFLIGHT FAIL: {len(failures)} text-bearing document(s) failed to chunk-build.",
              file=sys.stderr)
        for doc_id, arxiv_id, stage, msg in failures:
            print(f"  - doc_id={doc_id} arxiv_id={arxiv_id} stage={stage} error={msg}",
                  file=sys.stderr)
        print(f"  canonical chunks artifact left untouched: {OUTPUT}", file=sys.stderr)
        return 1

    all_records.sort(key=lambda r: r["chunk_id"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="chunks.", suffix=".jsonl", dir=str(OUTPUT.parent))
    try:
        with io.open(fd, "w", encoding="utf-8", newline="\n") as fh:
            for record in all_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        Path(tmp_path).replace(OUTPUT)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    output_sha = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    chunk_doc_ids = {r["doc_id"] for r in all_records}
    print(f"chunk_count: {len(all_records)}")
    print(f"chunk_doc_count: {len(chunk_doc_ids)}")
    print(f"output_sha256: {output_sha}")
    # Set equality gate.
    assert chunk_doc_ids == set(tb_report.text_bearing), \
        f"doc-id coverage gate failed: missing={set(tb_report.text_bearing)-chunk_doc_ids}, " \
        f"extra={chunk_doc_ids-set(tb_report.text_bearing)}"
    assert len(tb_report.text_bearing) == 69, \
        f"expected 69 text-bearing docs, got {len(tb_report.text_bearing)}"
    print("PREFLIGHT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())