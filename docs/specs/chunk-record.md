# Chunk Record Specification

Resolution of [#2 — Define chunk record shape and stable shared IDs](https://github.com/ImmoP/knowledge_graph_rag/issues/2). Every later stage of the pipeline (chunking, extraction, both ingestions, both retrievals) reads from this spec. Anything ambiguous here forces those tickets to re-litigate it; this spec is meant to remove that ambiguity.

The chunking *rule* (which section boundaries split where, paragraph- vs section-level chunks, etc.) is decided in [#6](https://github.com/ImmoP/knowledge_graph_rag/issues/6). This spec only fixes the *shape* of the chunk record and the *contract* for its identity.

## 1. Canonical on-disk artifact

- **Path:** `data/derived/chunks.jsonl`
- **Format:** JSONL. One chunk record per line. UTF-8. Trailing newline.
- **Ordering:** records are sorted by `chunk_id` lexicographically. Sorting is the responsibility of the chunking pipeline (the implementing ticket); the file as written is sorted, so consumers can rely on it.
- **Versioning:** the chunk manifest is a derived artifact computed from `data/manifest.jsonl` + `data/raw/latex/`. It is *not* checked into git (`data/derived/` is gitignored). Reproducibility is provided by determinism (re-running produces a byte-identical file) and by the benchmark runner recording the SHA256 of this file in its report metadata (per [#14](https://github.com/ImmoP/knowledge_graph_rag/issues/14)).

## 2. Required fields per chunk record

Each line is a JSON object with **exactly** these fields. No extras; no omissions.

| Field           | Type   | Description                                                                                                                                                                  |
|-----------------|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `chunk_id`      | string | Primary key. Format: `c_` followed by exactly 16 lowercase hex characters. Stable, content-independent, derived per §3.                                                       |
| `doc_id`        | string | Document identifier. Equals the `openalex_id` from `data/manifest.jsonl`, including its `W[0-9]+` form (no `https://` prefix, no `/W...` path).                              |
| `section_path`  | list   | Ordered list of section names from root to leaf, each a non-empty string. The empty list `[]` denotes a chunk that precedes the first `\section` (e.g. the abstract).       |
| `chunk_index`   | int    | 0-based index of this chunk within `doc_id`, counting in document order. Two chunks in the same document always have distinct `chunk_index` values.                          |
| `text`          | string | The cleaned chunk text. Built from the cleaned-and-citation-preserved text that `src/ingest/parse_latex.py` already produces. No re-citation-marker substitution is allowed downstream: `[[CITE:...]]` markers appear in `text` exactly as the parser emitted them. |
| `token_estimate`| int    | `len(text) // 4`. This is the pinned estimator. "Estimate" is part of the name; do not promote it to a precise token count without changing the field name.                 |
| `char_start`    | int    | 0-based character offset (inclusive) of this chunk's first character within the cleaned full-document text. Required field, but may be `-1` if the chunker cannot compute it (the rule permits this escape hatch but the default chunker computes it). |
| `char_end`      | int    | 0-based character offset (exclusive) of the chunk's last character in the cleaned full-document text. Same escape-hatch rule as `char_start`.                              |

### 2.1 Field notes

- **`section_path` is a JSON list, not a slash-joined string.** This makes every level of the path individually addressable in Cypher queries (`n.section_path[0] = "Introduction"`) and in retrieval results. Implementers may render it as ` / `-joined for display; the on-disk form is a list.
- **`text` is the only field that contains the chunk's natural-language content.** Retrieval returns chunk_ids; consumers fetch `text` from the chunk manifest by ID. Embeddings live in pgvector, not on the chunk record (see §6).
- **`char_start` / `char_end` use the *cleaned* full-document text** produced by `parse_latex.py` after `remove_comments` + `preserve_citations`. They are not offsets into the raw LaTeX. This makes them stable against LaTeX comment-stripping changes that don't affect cleaned output.
- **The 4-chars-per-token estimator** is intentionally crude. The local embedding model (per [#10](https://github.com/ImmoP/knowledge_graph_rag/issues/10)) will read `text` directly and produce precise token counts at embedding time; `token_estimate` exists for retrieval-budget decisions and coarse generation-context budgeting where running the tokenizer is too expensive.

## 3. `chunk_id` derivation

```
chunk_id = "c_" + sha1(f"{doc_id}|{json.dumps(section_path)}|{chunk_index}")[:16]
```

Properties guaranteed by this derivation:

- **Deterministic** for fixed `(doc_id, section_path, chunk_index)`. The same input triple always produces the same ID.
- **Independent of `text` content.** A cosmetic edit to a paper's body does not invalidate IDs (chunk_index and section_path are stable against such edits as long as section boundaries and chunk ordering don't move).
- **Independent across documents.** Two chunks in different documents with the same `section_path` and `chunk_index` get different IDs because `doc_id` is mixed in.
- **Collision-resistant.** 64 bits of SHA-1 prefix over a population of roughly 36 000 chunks (182 documents × ~200 chunks) makes a collision astronomically unlikely; if one ever occurs, the implementing ticket surfaces it as an error rather than papering over it.

### 3.1 Why this derivation and not a content hash

- A content hash changes whenever the cleaning pipeline changes (e.g. comments stripped slightly differently, citation markers reformatted). That invalidates every downstream artifact tied to chunk IDs — embeddings, extractions, the eval set's ground truth — every time the parser is improved. Positional derivation decouples IDs from text-content changes.
- A purely-sequential ID (e.g. `c_000001`) is order-dependent and fragile: inserting a chunk invalidates every later ID in the document. The derivation above makes IDs stable under any local edit.
- A pure SHA-1 of `text` would conflate two different chunks that happen to contain the same paragraph (e.g. repeated templates). The `(section_path, chunk_index)` coordinates disambiguate.

## 4. JSONL line shape (worked example)

A paper with `openalex_id = "W1522301498"` ("Adam: A Method for Stochastic Optimization"), after parsing and chunking, produces records like:

```json
{"chunk_id": "c_a1b2c3d4e5f60718", "doc_id": "W1522301498", "section_path": [], "chunk_index": 0, "text": "We propose Adam, a method for stochastic optimization...", "token_estimate": 87, "char_start": 0, "char_end": 348}
{"chunk_id": "c_2f3e4d5c6b7a8091", "doc_id": "W1522301498", "section_path": ["Introduction"], "chunk_index": 1, "text": "The success of deep learning...", "token_estimate": 142, "char_start": 349, "char_end": 917}
{"chunk_id": "c_9a8b7c6d5e4f3021", "doc_id": "W1522301498", "section_path": ["Introduction", "Background"], "chunk_index": 2, "text": "Stochastic gradient descent...", "token_estimate": 96, "char_start": 918, "char_end": 1302}
```

This is illustrative — actual IDs depend on the SHA-1 of the input triple. The example exists to show that `section_path` carries the full breadcrumb from the document root to the leaf, that `chunk_index` is sequential within the document, and that the abstract (`section_path = []`) is a regular chunk like any other.

## 5. Idempotency contract

Re-running the chunking pipeline on unchanged inputs (`data/manifest.jsonl`, `data/raw/latex/`, the chunking rule from [#6](https://github.com/ImmoP/knowledge_graph_rag/issues/6), and the on-disk chunking code) produces a byte-identical `data/derived/chunks.jsonl`. Required for the chunk manifest to function as a versioned artifact.

To achieve this, the implementing ticket must:

- Sort output lines by `chunk_id` before writing.
- Use the deterministic `chunk_id` derivation from §3.
- Avoid timestamps, hashes of the running process, or any non-deterministic field in the record.

If the chunking rule from [#6](https://github.com/ImmoP/knowledge_graph_rag/issues/6) changes in a way that affects `(section_path, chunk_index)` for any chunk, IDs change for those chunks and downstream artifacts must be regenerated. That is expected and acceptable; the rule version is the unit of change, not the cleaning pipeline.

## 6. What does **not** live on the chunk record

- **Embeddings.** Live in pgvector, keyed by `chunk_id`. The chunk record carries the text; the embedding is a separate artifact produced by the chunking/embedding ticket.
- **Document metadata** (title, year, authors, arxiv_id, doi). Lives in `data/manifest.jsonl`. Consumers join `chunk.doc_id` → `manifest.openalex_id` to fetch document metadata.
- **Extraction results.** Live in `data/extractions.jsonl` (per [#7](https://github.com/ImmoP/knowledge_graph_rag/issues/7)). Consumers join `extraction.chunk_id` → `chunk.chunk_id`.
- **Citation resolution.** The `[[CITE:key]]` markers appear in `text` unchanged; resolving them to OpenAlex work IDs is the citation-resolution's job, not the chunk record's.
- **Provider/embedding-model identity.** Belongs in the embedding artifact, not on the chunk record.

This separation is what makes the chunk manifest portable: a single artifact that doesn't depend on which model embedded it or which provider ran the extraction.

## 7. Cross-store ID contract

This is the contract that lets the vector store (pgvector) and the graph store (Neo4j) reference the same chunks without drift.

- **pgvector:** the `chunks` table has a primary key column `chunk_id` (TEXT, NOT NULL, UNIQUE) matching the regex `^c_[0-9a-f]{16}$`. Every row's `chunk_id` must appear in `data/derived/chunks.jsonl`.
- **Neo4j:** every `(:Chunk)` node has a `chunk_id` property matching the same regex. A uniqueness constraint `(:Chunk {chunk_id})` is created at ingestion. The chunk manifest is the source of truth; the graph and vector stores are derived from it.
- **Retrieval:** every retrieval mode returns `chunk_id` values from this namespace. The benchmark runner validates that every returned `chunk_id` exists in the current chunk manifest before scoring.

## 8. What this spec leaves open

These are decisions the *implementing* tickets make and refine; the spec only fixes the record shape and identity contract.

- The exact chunking rule (section vs paragraph vs hybrid, what counts as a subsection, how `[[CITE:...]]` spans are preserved across chunk boundaries) — [#6](https://github.com/ImmoP/knowledge_graph_rag/issues/6).
- The exact tokenizer used for embedding-time precise token counts — [#10](https://github.com/ImmoP/knowledge_graph_rag/issues/10).
- Whether `char_start`/`char_end` are computed from the cleaned text or from the original LaTeX (the spec commits to cleaned-text offsets; the chunker must match this) — [#6](https://github.com/ImmoP/knowledge_graph_rag/issues/6).
- How `section_path` handles sections with empty titles or titles containing LaTeX commands — [#6](https://github.com/ImmoP/knowledge_graph_rag/issues/6).

## 9. Acceptance test (runnable)

This spec is verifiable by a single small script (no test framework needed). It must succeed before #2 is considered closed:

```python
# pseudo-test, not committed
import json, hashlib, re

lines = Path("data/derived/chunks.jsonl").read_text().splitlines()
assert lines, "chunk manifest is empty"

ids = set()
seen_doc = set()
for ln in lines:
    r = json.loads(ln)
    assert re.fullmatch(r"c_[0-9a-f]{16}", r["chunk_id"]), r["chunk_id"]
    assert re.fullmatch(r"W\d+", r["doc_id"]), r["doc_id"]
    assert isinstance(r["section_path"], list)
    assert all(isinstance(s, str) and s for s in r["section_path"])
    assert isinstance(r["chunk_index"], int) and r["chunk_index"] >= 0
    assert isinstance(r["text"], str) and r["text"]
    assert r["token_estimate"] == len(r["text"]) // 4
    assert r["char_start"] == -1 or r["char_end"] > r["char_start"] >= 0
    assert r["chunk_id"] not in ids, f"duplicate chunk_id {r['chunk_id']}"
    ids.add(r["chunk_id"])

# Sort order: file is sorted by chunk_id.
file_ids = [json.loads(ln)["chunk_id"] for ln in lines]
assert file_ids == sorted(file_ids), "records not sorted by chunk_id"

# ID derivation is the spec's formula.
def derive(doc_id, section_path, chunk_index):
    payload = f"{doc_id}|{json.dumps(section_path)}|{chunk_index}"
    return "c_" + hashlib.sha1(payload.encode()).hexdigest()[:16]

# Spot-check at least one record's id matches the derivation.
sample = json.loads(lines[len(lines)//2])
expected = derive(sample["doc_id"], sample["section_path"], sample["chunk_index"])
assert sample["chunk_id"] == expected, "id derivation mismatch"

# Every doc_id appears in the manifest.
manifest = {json.loads(l)["openalex_id"] for l in Path("data/manifest.jsonl").read_text().splitlines()}
doc_ids_in_chunks = {json.loads(l)["doc_id"] for l in lines}
missing = doc_ids_in_chunks - manifest
assert not missing, f"chunks reference unknown docs: {sorted(missing)[:5]}"
```

The implementing ticket runs this script against its output and ships only when it passes.

---

This spec resolves ticket #2. It is referenced by #6 (chunking rule must produce records matching §2), #7 (extraction operates per `chunk_id` per §2), #9 (Neo4j ingestion must use this `chunk_id` namespace per §7), #10 (vector store uses the same `chunk_id` PK per §7), and #11 (graph retrieval returns chunk_ids from this namespace).