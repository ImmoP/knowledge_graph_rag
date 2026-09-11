# Vector-Only Retrieval Baseline Specification

Resolution of [#10 — Decide the vector-only retrieval baseline shape](https://github.com/ImmoP/knowledge_graph_rag/issues/10). This document is the **contract** the vector-only retrieval baseline must satisfy. It is not the implementing code — the implementing ticket goes to `/to-tickets`.

The vector retriever is the first of three retrieval shapes that ship together as benchmark conditions: vector-only (this spec), graph-only ([#11](https://github.com/ImmoP/knowledge_graph_rag/issues/11)), hybrid/routed ([#12](https://github.com/ImmoP/knowledge_graph_rag/issues/12)). This spec is scoped to vector-only and defines only vector-specific behaviour. The benchmark runner is responsible for any common result envelope (per #14) and may wrap or adapt per-retriever results there.

## 1. Embedding model pin

**Selected model:** `BAAI/bge-small-en-v1.5`.

### 1.1 Pinned values

| Property              | Value                                                                |
|-----------------------|----------------------------------------------------------------------|
| Hugging Face id       | `BAAI/bge-small-en-v1.5`                                             |
| Model revision        | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` (literal git revision SHA of `BAAI/bge-small-en-v1.5`; pinned across `implementation-spec.md`, `#19`, `#26`, `#27`, `#33`, and the closed #10 resolution thread) |
| Embedding dimension   | `384`                                                                |
| Max model input length | `512` tokens                                                         |
| Pooling               | mean-pooling, as supplied by the pinned `SentenceTransformer` model (default for `BAAI/bge-small-en-v1.5`) |
| Normalization         | L2-normalize embeddings before returning from the adapter; `score = 1 − cosine_distance` is then equivalent to inner-product similarity over normalized vectors |
| Query instruction     | `"Represent this sentence for searching relevant passages: "` (the published BGE query instruction for short-query-to-passage retrieval) — **applied to queries only** |
| Passage instruction   | none — stored passage / chunk text is embedded verbatim, with no prefix |
| Provider adapter      | `sentence-transformers` adapter registered under the `EmbeddingProvider` protocol (per [#16](https://github.com/ImmoP/knowledge_graph_rag/issues/16)) |

The query instruction and the no-instruction-for-passages rule are pinned together: any future change must update both fields and re-embed the corpus. The asymmetry is required by BGE's published short-query retrieval recipe.

### 1.2 Query / passage normalization parity

The embedding adapter L2-normalizes both query and passage embeddings before returning them from `embed(...)`. With normalization, `cosine_distance` reduces to `1 − inner_product`, and pgvector's `<=>` operator returns the cosine distance directly. Storing and querying normalized vectors means the loader does not need to track a "is this normalized?" flag per row.

### 1.3 Chunk-size guarantee (no silent truncation)

Silent truncation is forbidden in the vector baseline. The implementing ticket MUST guarantee, before embedding begins, that every chunk in `data/derived/chunks.jsonl` fits within the model's 512-token input limit. The implementing ticket does this by:

- measuring chunk-token counts (using the same tokenizer the embedding model ships with, not an approximation);
- if any chunk exceeds 512 tokens, refusing to embed and surfacing the offending `chunk_id`s and their token counts.

The chunking strategy ticket ([#6](https://github.com/ImmoP/knowledge_graph_rag/issues/6)) is responsible for picking chunk-size thresholds that guarantee this property at chunking time; this spec only records that the embedding contract rejects oversized chunks.

If the chunking strategy ever changes in a way that re-introduces oversize chunks, the embedding run fails loudly rather than silently truncating. Silent truncation would make the vector baseline retrieve from only part of a supposedly canonical chunk, which violates the contract that `data/derived/chunks.jsonl` is the unit of retrieval.

## 2. Index choice

**Selected index:** exact pgvector cosine search, no ANN index. Vectors are stored in a plain `vector(384)` column and retrieval computes exact cosine distance over all rows.

### 2.1 Why exact, not ANN

The corpus is small (182 papers × tens of chunks each, on the order of 10⁴ chunks × 384 floats ≈ 14 MB of vector data). Exact cosine search over this size is fast enough that ANN's recall savings are not justified. The vector baseline therefore commits to exact cosine as the selected baseline.

### 2.2 If ANN is later considered

ANN can never "beat exact search on recall" when exact search is the reference. If an ANN index is later considered, it must demonstrate:

- a **pinned recall@k target** relative to exact cosine search (e.g. recall@10 ≥ 0.99 over a representative chunk sample), AND
- a **meaningful latency improvement** over exact search that justifies the recall risk.

The representative chunk sample is a sample of `data/derived/chunks.jsonl` chunks used for index parameter selection. It is distinct from the held-out benchmark; this terminology avoids any overlap with the held-out-benchmark vocabulary (the held-out benchmark is the 45 questions in `eval/questions.jsonl`).

ANN parameter selection MUST NOT touch the 45 held-out benchmark questions. If the comparison procedure reveals ANN parameters that meet the recall target and the latency bar, a follow-up ticket amends this spec and adds the index type and parameters to §2.

Until that follow-up lands, the vector baseline ships with no index — only the `vector(384)` column and exact cosine search.

## 3. Store contract

### 3.1 pgvector table

Table name: `chunks`.

| Column              | Type           | Notes                                                                                                                |
|---------------------|----------------|----------------------------------------------------------------------------------------------------------------------|
| `chunk_id`          | `text`         | PRIMARY KEY. Matches `^c_[0-9a-f]{16}$` per the chunk-record spec ([#2](https://github.com/ImmoP/knowledge_graph_rag/issues/2)). |
| `embedding`         | `vector(384)`  | L2-normalized, dim pinned by the embedding model.                                                                   |
| `embedding_model`   | `text`         | `"<provider>:<model>:<revision>"` — the full identity used to produce the embedding. Required for audit and stale-detection. |
| `source_text_sha256`| `char(64)`     | SHA256 of the chunk's `text` field at the time of embedding. Required for stale-embedding detection (see §3.3).     |
| `embedded_at`       | `timestamptz`  | Default `now()`. Audit only; not used for staleness.                                                                |

### 3.2 Derived-store contract

`data/derived/chunks.jsonl` is the source of truth for chunks (per [#2](https://github.com/ImmoP/knowledge_graph_rag/issues/2)). pgvector is a **derived store**: it holds embeddings keyed by `chunk_id`, and its contents may be rebuilt from scratch at any time by re-embedding `data/derived/chunks.jsonl`. The implementing pipeline reads the canonical artifact and populates pgvector; it never writes embeddings for chunks that do not appear there.

Every `chunk_id` stored in pgvector MUST exist in `data/derived/chunks.jsonl`. A `chunk_id` present in pgvector but missing from the canonical artifact is a defect to surface, not to silently drop.

### 3.3 Stale-embedding detection

`chunk_id` is intentionally content-independent (per [#2](https://github.com/ImmoP/knowledge_graph_rag/issues/2)), so validating only `chunk_id` and `embedding_model` is insufficient: the same `chunk_id` can survive while its text changes after parser or cleaning changes. The `source_text_sha256` column makes stale embeddings detectable.

A row is **stale** and MUST be re-embedded if any of the following is true on load:

| Condition | Detection |
|-----------|-----------|
| `chunk_id` missing from `data/derived/chunks.jsonl` | orphan-row check: `SELECT chunk_id FROM chunks WHERE chunk_id NOT IN (<canonical chunk_ids>)` returns non-empty |
| Embedding model/revision differs | row's `embedding_model` ≠ currently-configured `<provider>:<model>:<revision>` |
| Source-text fingerprint differs | row's `source_text_sha256` ≠ `sha256(canonical_chunk.text)` for that `chunk_id` |
| Embedding is missing | canonical artifact has a chunk with no row in pgvector |

When stale rows are detected, the loader re-embeds the affected chunks. The loader does not silently drop or skip stale rows; it reports the count and proceeds to re-embed.

### 3.4 Run metadata

The implementing ticket records, alongside each embedding run's logs:

- the SHA256 of `data/derived/chunks.jsonl` at the time of the run;
- the pinned model revision;
- the timestamp.

This metadata is audit-only; it does not gate correctness (the per-row `source_text_sha256` is the authoritative staleness signal). Keeping the canonical artifact's SHA256 lets an operator reconstruct which version of the corpus a given pgvector snapshot was produced from.

## 4. Retrieval interface

### 4.1 Result type

```python
@dataclass(frozen=True)
class VectorRetrievalHit:
    chunk_id: str
    score: float
```

That is the entire result shape for vector-only retrieval. Graph-only ([#11](https://github.com/ImmoP/knowledge_graph_rag/issues/11)) and hybrid ([#12](https://github.com/ImmoP/knowledge_graph_rag/issues/12)) define their own result types if they need different fields. A later retrieval/hybrid decision may introduce a common result envelope if the benchmark runner needs one (per #14); this spec does not pre-empt that.

### 4.2 Retriever protocol

```python
class VectorRetriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[VectorRetrievalHit]: ...
```

### 4.3 Score semantics

- `score = 1 − pgvector_cosine_distance`, where `pgvector_cosine_distance` is the value returned by pgvector's `<=>` operator on L2-normalized vectors.
- Results are ordered by `score` descending.
- **Tie-break:** if two hits have identical `score` (possible because cosine distance on `double precision` floats can collide), they are ordered by `chunk_id` ascending (lexicographic). The ordering is therefore deterministic.
- **No similarity threshold.** The baseline returns the top-k by score regardless of how low the score is.
- **Empty store:** if pgvector contains zero rows, `retrieve` returns `[]`.
- **Under-filled top-k:** if `k` exceeds the number of rows in the store, `retrieve` returns all rows.
- **Invalid `k`:** `k <= 0` raises `ValueError` before any database call. The error message includes the offending `k` value.
- **`k` type:** `int`. Non-int inputs (`float`, `str`) raise `TypeError`.

### 4.4 Query preparation

The `retrieve(query, k)` implementation prepends the BGE query instruction (per §1.1) to the query string before embedding. The passage instruction is `""`. The implementing ticket does not expose this asymmetry to callers — callers pass natural-language queries and the retriever applies the BGE recipe internally.

## 5. Interaction with other tickets

- **#16 provider abstractions:** the embedding adapter for `BAAI/bge-small-en-v1.5` is registered under the `EmbeddingProvider` Protocol. The adapter L2-normalizes before returning.
- **#3 local infrastructure:** pgvector runs in the local Postgres container; `chunks` table is created by the implementing ticket.
- **#4 evaluation set:** retrieval parameters (model revision, embedding dimension, indexing choice) MUST NOT be tuned on the 45 held-out benchmark questions. Tuning is forbidden on the 45; if development material is needed it lives in a separate out-of-band artifact (e.g. `eval/dev_questions.jsonl`), not among the 45.
- **#11 graph-only and #12 hybrid:** this spec defines only vector-specific behaviour. Any cross-retriever result envelope is the benchmark runner's concern (per #14), not this ticket's.

## 6. Acceptance gate

This spec closes when committed to `docs/specs/vector-retrieval.md`. The implementing ticket closes when:

1. `chunks` table exists in pgvector with the columns and types in §3.1.
2. Every `chunk_id` in pgvector exists in `data/derived/chunks.jsonl` (orphan-row check passes).
3. Every row's `embedding_model` equals the pinned `<provider>:<model>:<revision>`.
4. Every row's `source_text_sha256` equals `sha256(canonical_chunk.text)` (stale-row check passes).
5. Every chunk in `data/derived/chunks.jsonl` has a row in pgvector (no missing embeddings).
6. Every chunk in `data/derived/chunks.jsonl` has a `text` length that fits the model's 512-token limit after tokenization (no silent-truncation check passes; failing chunks are surfaced, not embedded).
7. A `retrieve(query, k)` implementation exists matching the §4 contract, including the §4.3 score semantics and the §4.4 query preparation.
8. A small self-check (per ponytail rule: one runnable check that fails if the logic breaks) exercises `retrieve` against a populated store and asserts: top-k ordering by `score` desc, deterministic tie-break by `chunk_id`, `[]` on empty store, all rows when `k` exceeds count, `ValueError` on `k <= 0`.

The first six gates validate the store and the embedding; the last two validate the retrieval interface.

---

This spec resolves ticket #10. It is referenced by #14 (benchmark runner consumes vector-retrieval results) and by #12 (hybrid may consult the vector retriever as a sub-component). The implementing ticket for #10 ships the embedding adapter, the loader, the `chunks` table, and the `VectorRetriever` implementation.
