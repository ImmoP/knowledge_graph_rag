# Chunking Strategy Specification

Resolution of [#6 — Decide the section-aware chunking strategy](https://github.com/ImmoP/knowledge_graph_rag/issues/6). This document is the **contract** the implementing chunker reads. It is not the implementing ticket — that goes to `/to-tickets` once the map is cleared.

This spec assumes the [`chunk-record` spec](chunk-record.md) is in force. In particular: the chunk manifest lives at `data/derived/chunks.jsonl`, records are JSONL sorted by `chunk_id`, and `chunk_id` is the positional hash pinned in §3 of the chunk-record spec. This document does not redefine those; it pins the *strategy* that fills them in.

The chunking strategy is the small set of rules by which a parsed document becomes a sequence of chunk records. It fixes which boundaries are honored, what precedence rules apply, and how the output is guaranteed to be deterministic. Numerical thresholds and parser quirks belong to the implementing ticket; the strategy names the rules and the precedence, not the numbers.

## 1. Inputs and outputs

**Inputs:**
- `data/manifest.jsonl` — the canonical document corpus (182 rows); every record defines one canonical `Document` identity.
- `data/raw/latex/<arxiv_id>.tar.gz` — arXiv LaTeX source packages (read-only; never modified).
- The parsed-and-cleaned text produced by `src/ingest/parse_latex.py` — specifically the output of `read_tex_files` → `choose_main_file` → `resolve_includes` → `remove_comments` → `preserve_citations`. The chunker does **not** re-parse the raw `.tex`.

**Output:**
- `data/derived/chunks.jsonl` — one chunk record per line, conforming to the chunk-record spec, one record per `(doc_id, section_path, chunk_index)` triple, sorted by `chunk_id`. The chunker produces chunks only for documents in the **v1 text-bearing corpus subset** defined in §12. Documents outside that subset produce **zero** chunk records; they remain canonical `Document` identities for graph, citation, and seed-lift purposes but contribute nothing to the chunk namespace, the embedding store, the entity-resolution mention set, or the benchmark.

**Inputs the chunker does NOT read:**
- `data/aliases.json` — alias merging is a manifest concern, already applied in the manifest's `merged_ids`. The chunker uses `openalex_id` directly.
- `data/title_overrides.json` — metadata only.
- `ontology.yaml` — extraction uses it, not the chunker.

## 12. v1 canonical-document corpus vs v1 text-bearing document set

The canonical document corpus is `data/manifest.jsonl`; **every** record there defines one canonical `Document` identity. Not every record in that canonical corpus yields chunks in v1.

The **v1 text-bearing document set** is determined by canonical manifest metadata **and the tracked source-capability artifact**:

```python
# tracked, committed source-capability artifact
fulltext_manifest_by_id = {row["arxiv_id"]: row for row in data/fulltext_manifest.jsonl}

text_bearing_doc_ids = {
    record["openalex_id"]
    for record in data/manifest.jsonl
    if record["arxiv_id"] is not None
    and fulltext_manifest_by_id.get(record["arxiv_id"], {}).get("payload") == "gzip"
}
```

Membership is a function of the **tracked metadata** alone (the committed `data/manifest.jsonl` plus the committed `data/fulltext_manifest.jsonl`). It is **byte-for-byte deterministic**, identical across machines given the same committed artifacts, and **independent of which raw files happen to be present locally**. A `payload == "pdf"` arXiv item is a valid canonical `Document` identity but is **not** in the v1 text-bearing set; the architecture intentionally narrows v1 text coverage to LaTeX (`payload == "gzip"`) and does **not** add PDF parsing, OCR, or ACL Anthology ingestion.

For every member of `text_bearing_doc_ids`, the v1 full-text source is the LaTeX package:

```
data/raw/latex/<arxiv_id>.tar.gz
```

A text-bearing document's archive is a **precondition** for downstream success, not part of the membership predicate. A record satisfies membership the moment `record["arxiv_id"] is not None` AND the tracked `data/fulltext_manifest.jsonl` records that arXiv id with `payload == "gzip"`; the preflight is responsible for confirming the archive exists. The two checks must NOT be conflated:

- `arxiv_id != null` AND tracked `payload == "gzip"` AND archive exists: valid text-bearing document.
- `arxiv_id != null` AND tracked `payload == "gzip"` AND archive missing/unreadable: input-integrity failure; the preflight (#21) exits non-zero. The pipeline does **not** silently reclassify such a record as non-text-bearing.
- `arxiv_id != null` AND tracked `payload == "pdf"`: canonical non-text-bearing Document in v1; zero chunks required; this is the valid terminal state, not an error. (The PDF on disk is real corpus content; v1 just does not parse it.)
- `arxiv_id != null` AND no tracked `data/fulltext_manifest.jsonl` row for that id: preflight failure — unresolved source capability; #21 exits non-zero. This means the source-capability probe (#4 / fetch_arxiv pipeline) was not run for that arXiv id on the current checkout.
- duplicate tracked rows for one `arxiv_id` in `data/fulltext_manifest.jsonl`: preflight failure.
- `arxiv_id == null`: canonical non-text-bearing Document; zero chunks required; this is the valid terminal state, not an error.

A prepared checkout is **never** allowed to silently shrink the text-bearing corpus because an archive is missing. Membership is derived from the tracked artifacts; archive presence is a downstream runtime check.

In v1, the LaTeX source package is the only full-text source. Records with `source_kind ∈ {"other_pdf", "landing_only", "acl_anthology"}` are valid canonical `Document` identities (for graph nodes, `CITES`, and seed lifting) but are **outside the v1 text-bearing set**. v1 does **not** parse PDF, OCR, ACL Anthology HTML, or any non-LaTeX source; the architecture intentionally narrows v1 text coverage to the LaTeX subset.

Properties the implementing ticket pins:

- The text-bearing `doc_id` set is reproducible from the tracked `data/manifest.jsonl` and `data/fulltext_manifest.jsonl` artifacts and equals the set of chunk `doc_id` values in `data/derived/chunks.jsonl` (set equality; see §11 Acceptance gate).
- Every manifest record is a canonical `Document` identity (the manifest is not shrunk).
- Only text-bearing records produce `Chunk` records, `Mention` records, `Assertion` records, and rows in the embedding store. Any other use of a `chunk_id` is invalid.
- A record that satisfies the membership predicate (tracked `payload == "gzip"`) but whose `data/raw/latex/<arxiv_id>.tar.gz` is missing or unreadable fails the preflight (#21). It is **not** silently removed from the text-bearing set.
- The non-text-bearing set is reported in run traces (count and `doc_id`s) but the manifest itself is unchanged.
- The classifier is a deterministic predicate over tracked manifest and source-capability metadata. It is not an LLM call.
- No PDF parsing, no OCR, no ACL Anthology ingestion, and no new ingestion stage are introduced in v1.

Downstream invariants this pin enables:

- `#25` may place a `Document` node with zero `HAS_CHUNK` relationships (a valid non-text-bearing canonical identity); the graph contract forbids inventing `HAS_CHUNK` only on the basis of corpus presence.
- For a non-text-bearing terminal Document, a pure-CITES path that reaches it produces no `HAS_CHUNK` fallback chunk (the document has no chunks) and therefore yields no hit from that path. This is valid, not stale graph state.
- `#31` / `#35` require every `supporting_chunk_ids` to point to a chunk in `data/derived/chunks.jsonl`. Non-text-bearing documents therefore cannot supply `supporting_chunk_ids` directly — their `gold_evidence_path` must reach a text-bearing terminal Document whose chunks carry the load.
- `#26` embeds every canonical chunk; it does not embed documents.
- `#33` runs against the canonical chunk namespace and never assumes every manifest document has chunks.

The numbers in §9 remain the corpus-characterizing numbers for the v1 **text-bearing** corpus (not the full manifest).

## 2. Default granularity: subsection

The default chunk boundary is the `\subsection{...}` command. A chunk contains the section title that introduces it, the subsection title, and all text between this subsection's start and the next subsection (or section) start.

**Rationale:** this is the empirical sweet spot in RAG literature (one claim ≈ one chunk; fits typical embedding context windows; preserves topical coherence). It also matches the natural structure of the corpus, where sections typically have 2–6 subsections each.

## 3. Cascading fallback rules

Real LaTeX is not uniform. The strategy names the rules that handle non-uniform structure:

| Situation | Rule |
|---|---|
| Section with one or more `\subsection` markers | Each subsection is its own chunk. |
| Section with **no** `\subsection` markers | The whole section (title + body) is one chunk. |
| Subsection long enough to exceed the chunk-size cap (number pinned by implementing ticket) | Subdivide at paragraph breaks (`\n\n`). Each subdivision stays inside the original subsection's `section_path` and increments `chunk_index` monotonically. |
| Paragraph long enough that paragraph-only subdivision leaves a still-overlong fragment | Subdivide further at line breaks (`\n`). |
| Line long enough that line-boundary subdivision leaves a still-overlong fragment (no `\n` boundaries left) | Final safe-whitespace subdivision: split at the LAST Unicode-whitespace boundary in the next `chunk-token-budget * 4` chars that respects the citation-span rule (§6). Never split inside a `[[CITE:...]]` span. If a single token exceeds the budget and there is no safe split point, fail closed with `OversizeChunkError` rather than silently truncate. |
| Document with no `\section` markers at all | The entire body is one chunk with `section_path = []`. |
| Document with `\subsubsection` but no `\subsection` | Treat `\subsubsection` as the subsection-equivalent boundary. |
| Section titles containing nested braces (`\section{X {Y} Z}`) | Section title is truncated at the first inner `}`; chunk continues to use whatever text followed. Implementing ticket emits a warning per occurrence. |

The cascade is not "first matching rule wins"; it is "section → subsection → paragraph → line → safe-whitespace subdivision of overlong lines." When a section has subsections, paragraph and finer subdivision *inside* a subsection only triggers if that subsection is overlong. The whitespace-fallback stage preserves every character (excluding inter-chunk whitespace separators) and respects the citation-span rule; it never silently drops or truncates chunk content. A chunk that cannot be emitted within the budget under these rules causes the whole build to fail closed (no partial chunks.jsonl written).

## 4. Pre-section content (abstract, front matter)

Text that appears before the first `\section{...}` (typically the abstract, sometimes title-block fragments) forms a single chunk with `section_path = []` and `chunk_index = 0`. The implementing ticket must not let this text bleed into the first section's chunk.

If the document has no `\section` markers at all, the entire document is one chunk with `section_path = []` and `chunk_index = 0`.

## 5. Section detection — pinned regex

The chunker detects section boundaries from the cleaned text using this pinned regex:

```
\\(section|subsection|subsubsection)\*?\s*\{([^}]*)\}
```

Properties this regex guarantees:
- It matches `\section`, `\section*`, `\subsection`, `\subsection*`, `\subsubsection`, `\subsubsection*`. Starred variants are treated identically to non-starred.
- It relies on the fact that `parse_latex.py`'s `preserve_citations` substitutes citation commands with `[[CITE:...]]` markers *before* the chunker runs, and citations do not appear inside `\section{...}` titles in any well-formed LaTeX in this corpus (verified on the sample papers during charting).
- It does **not** match `\paragraph{...}` or `\subparagraph{...}`; these are paragraph-level commands whose text content stays *inside* the enclosing subsection's chunk.
- It does **not** match `\part`, `\chapter`, `\appendix`. This is a paper corpus, not a thesis corpus; the implementing ticket verifies this assumption empirically before shipping (see §8).

When the regex matches, the chunker extracts:
- The command name (`section` / `subsection` / `subsubsection`) → drives chunk granularity (per §3).
- The captured title text → goes into `section_path` as one element.

If a section command has no title text (`\section{}`), the `section_path` element is the empty string `""`. The implementing ticket may flag this as a data-quality issue but does not skip the section.

## 6. Citation-span rule

`parse_latex.py` injects `[[CITE:key1,key2]]` markers into the cleaned text. The strategy pins the following rule:

**A chunk boundary is never placed inside a `[[CITE:...]]` span.**

Boundary-precedence tie-break (in order):

1. Section boundary (`\section`, `\subsection`, `\subsubsection`) — preferred.
2. Paragraph boundary (`\n\n`) — used for subdivision inside an overlong subsection.
3. Closure of the citation group (the position immediately after the `]]` of the open `[[CITE:` that the boundary would otherwise split) — last resort.

If, after applying rule 3, a chunk would still exceed the size cap, the implementing ticket prefers to split **before** the open `[[CITE:` rather than after it (so the citation group stays in the earlier chunk). This avoids leaving dangling references in chunk text.

The strategy deliberately does **not** pin a citation-group-collapse rule: multiple `[[CITE:key1]] [[CITE:key2]]` in the same sentence remain two separate spans. Citation *resolution* (mapping `key` to an OpenAlex work ID) is an extraction concern, not a chunking concern.

## 7. Section path construction

`section_path` is built incrementally as the chunker walks the document:

```
section_path = [section_title, subsection_title, subsubsection_title (if any)]
```

- Length is 0 (pre-section chunk), 1 (section without subsections), 2 (section with subsections), or 3 (subsubsection subdivision).
- The chunker maintains a small stack: when a `\section{...}` is seen, the stack is reset to `[section_title]`; `\subsection{...}` appends; `\subsubsection{...}` appends a third level (capped at 3).
- The same `section_path` is used for every chunk produced by subdividing one subsection — i.e. `chunk_index` increases but `section_path` does not.

The chunk-record spec §4 example shows `section_path = []`, `["Introduction"]`, and `["Introduction", "Background"]`. The empty list (`[]`) is reserved for the pre-section chunk (§4).

## 8. Idempotency guarantees

The chunk manifest is reproducible. The strategy contributes these guarantees:

- **Section processing order:** sections in document order, never re-ordered.
- **Chunk index assignment:** monotonic 0-based within a document; subsections enumerated in document order; subdivision chunks of an overlong subsection enumerated in document order (paragraphs in document order, not by length or any other metric).
- **No timestamps, random IDs, or process-derived state in the output.** Restated here because the chunking strategy is the most likely place for accidental non-determinism (e.g. dict iteration order).
- **Starred variants are treated identically to non-starred.** A `\section*{Acknowledgements}` produces the same kind of chunk as `\section{Methods}`.

The implementing ticket verifies idempotency by running the chunker twice on a non-trivial subset of the corpus and asserting SHA256 equivalence of `data/derived/chunks.jsonl`.

## 9. Corpus-characterizing numbers

These are the *corpus-characterizing* numbers, not chunking rules. They tell the implementing ticket what it's sizing chunks to:

- 182 documents in the canonical document corpus; the v1 text-bearing corpus subset is the records that carry an `arxiv_id` AND for which `data/fulltext_manifest.jsonl` records `payload == "gzip"`. The remaining records are non-text-bearing canonical `Document` identities in v1 (which includes PDF-payload arXiv items, non-arXiv source_kinds, and arXiv items with no resolved capability row).
- Cleaned-document sizes observed: 5k–120k characters (mean ≈ 40k).
- Sections per document observed: 5–12 typical.
- Subsections per document observed: 5–30 typical.
- Citation markers per document observed: 20–200 typical, mostly inline within paragraphs.
- Target chunk count for the corpus: a few thousand chunks, well within the 64-bit `chunk_id` namespace.

The implementing ticket picks concrete size targets (e.g. "target 256–512 tokens, hard cap 1024 tokens") and verifies the resulting chunk count stays in the few-thousands range. If the implementation drifts into the tens of thousands (over-fragmenting) or hundreds (under-fragmenting), the strategy says that is a sizing problem and the implementing ticket should re-tune the numbers without changing the rules.

## 10. What this spec leaves open (implementing ticket owns these)

These are explicitly the implementing ticket's responsibility and not part of the strategy:

- **Numerical size thresholds** for "overlong subsection," "overshort subsection," etc.
- **How overshort chunks are merged** with adjacent chunks of the same subsection.
- **Handling of malformed LaTeX** — unbalanced braces, malformed `\input` chains, multi-file documents where `choose_main_file` returns the wrong main.
- **Starred-section deduplication** — repeated `\section*{...}` titles produce repeated `section_path` values; `chunk_index` distinguishes them. Acceptable per §8.
- **Atomic-write protocol** for `data/derived/chunks.jsonl` (write to a tempfile, rename).
- **Tokenizer** used for the size measurements. The chunk-record spec pins `len(text) // 4` for `token_estimate`; the implementing ticket may use a precise tokenizer for sizing decisions during development.

## 11. Acceptance gate

The implementing ticket closes only when all of these pass:

1. **Idempotency check:** two consecutive chunker runs on the full corpus produce byte-identical `data/derived/chunks.jsonl`. SHA256 of the file is the gate.
2. **Chunk-record spec compliance:** the runnable script from [`chunk-record.md` §9](chunk-record.md) passes against the produced file.
3. **Citation-span rule:** spot-check three papers (one short, one medium, one with dense citations) and assert no `chunk.text` ends with an unclosed `[[CITE:` or starts with a `]]`. The script can search the file for any chunk whose `text` matches `\[\[CITE:.*$` without a corresponding `]]` before the next `\n\n` boundary.
4. **Corpus size sanity:** the chunk count is in the few-thousands range (per §9). If it is not, the implementing ticket re-tunes size thresholds per §9 without changing the strategy rules.

---

This spec resolves ticket #6. It is referenced by #7 (extraction operates on chunks produced by this strategy), #10 (vector retrieval chunks are sized by these rules), and the benchmark runner in #14 (which records the chunk-manifest SHA produced by an idempotent run of this strategy).