# Hybrid / Routed Retrieval Specification

Resolution of [#12 — Decide the hybrid/routed retrieval shape](https://github.com/ImmoP/knowledge_graph_rag/issues/12). This document is the **contract** the hybrid/routed retrieval baseline must satisfy. It does not implement retrieval.

The hybrid retriever is the third of three retrieval shapes that ship together as benchmark conditions: vector-only ([#10](https://github.com/ImmoP/knowledge_graph_rag/issues/10)), graph-only ([#11](https://github.com/ImmoP/knowledge_graph_rag/issues/11)), and hybrid/routed (this spec). Hybrid is a **first-class benchmark condition**, not an opportunistic combination. The router is a small deterministic rule, never a learned router in v1. The spec composes the #10 vector retriever and the #11 graph retriever unchanged; it adds only routing, fusion, and a hybrid result type that records per-retriever provenance.

## 1. Standing constraints

These are the architectural properties hybrid inherits from upstream specs and the parent map. Each is named so the implementing ticket can audit against it.

1. **First-class benchmark condition.** Hybrid results must be comparable across runs and benchmarkable per hop bucket alongside vector-only and graph-only. The hybrid shape does not block this.
2. **Composes #10 and #11 unchanged.** `VectorRetriever.retrieve(query, k)` and `GraphRetriever.retrieve(query, k, hop_budget=3)` are called directly with the signatures already pinned. Neither retriever is reimplemented.
3. **No router rule, deterministic invocation.** The hybrid invokes both retrievers with the caller's query and the frozen constants in §3; there is no query-side router, no learned model, no embedding similarity for routing, no `LLMProvider` call in the hot path. A measured-failure fallback exists on paper (§8) but is **not enabled**.
4. **No `LLMProvider` or `EmbeddingProvider` in the hot path.** Hybrid routing and fusion are computed offline; only the local BGE adapter from #10 sees an embedding at query time, via `VectorRetriever.retrieve`. Graph retrieval sees neither.
5. **Held-out governance.** The 45 canonical questions in `eval/questions.jsonl` are `split: "held_out"` and **must not be used to tune any hybrid parameter** (router rule, fusion constant, candidate depth, tie-break). Tuning, if any, lives in `eval/dev_questions.jsonl` strictly out-of-band (per eval-schema §6).
6. **Canonical chunk identity.** `chunk_id` is the unit of retrieval and the unit of fusion. Dedup is structural (one fused entry per `chunk_id`). Source chunk text matches via the canonical artifact `data/derived/chunks.jsonl` only.
7. **Provenance preserved.** Graph hits carry the connected five-field `evidence_path` from #11 unchanged. Hybrid never rewrites or normalises evidence paths.
8. **Per-retriever auditability.** Every merged hit records the originating retriever(s), per-retriever rank (1-based, `None` when absent), per-retriever score, and for graph hits the full evidence path. The benchmark runner (#14) can attribute every hit to `vector`, `graph`, or both without guessing.
9. **Smallest deterministic v1.** No learned reranker, no LLM judge, no per-question weight learning, no SVM, no gradient. A standard list-fusion algorithm with one frozen knob.
10. **No README change. No implementation code in this spec.**

The frozen signature is:

```python
class HybridRetriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[HybridRetrievalHit]: ...
```

`k` semantics, types, and error behaviour match #10 §4.3 (non-int ⇒ `TypeError`, `k <= 0` ⇒ `ValueError` before either retriever is called). An empty or whitespace-only query returns `[]` without invoking either retriever.

## 2. Routing

The hybrid retriever has **no router**. For every valid hybrid query, both retrievers are invoked with the same query string and the same globally-frozen graph hop budget; their results are fused by RRF (§4) and ordered (§6). No query-side rule inspects either retriever's output to decide whether (or how) to call the other.

```python
vector_candidates = vector_retriever.retrieve(
    query,
    k = HYBRID_K_VEC_MULTIPLIER * k,
)

graph_candidates = graph_retriever.retrieve(
    query,
    k = HYBRID_K_GRAPH_MULTIPLIER * k,
    hop_budget = HYBRID_HOP_BUDGET,
)
```

with the constants frozen in §3. The `GraphRetriever` owns its own deterministic query seed lifting (per #11 §2) and returns `[]` when no Entity or Document seed can be lifted from the query. When that happens, `graph_candidates == []` and fusion reduces naturally to the vector candidate list — no special-case branch, no fallback, no externally-supplied seed.

### 2.1 What the hybrid does **not** do

The v1 hybrid composes the two retrievers without inventing a new retrieval primitive. Concretely, the hybrid does **not**:

- derive a `doc_id` from the vector top hit and pass it back to `GraphRetriever` as a Document seed;
- add a `retrieve_from_seed(...)` or any equivalent externally-seeded traversal entry point;
- synthesise a query string containing a `doc_id` and re-run graph retrieval on it;
- bootstrap from chunk to Document to seed graph traversal inside the hybrid;
- add a second graph multiplier, a second graph hop budget, or a graph-tail call of any kind;
- read `hop_bucket` from the question to alter `hop_budget`.

Both retrievers are invoked for every valid hybrid query, exactly once, with the call shapes above. No path between them is permitted beyond RRF fusion.

### 2.2 Why no router rule

Earlier drafting considered a two-case "graph-led / vector-led" rule keyed off #11's seed-lift output. That was abandoned because:

- the only meaningful graph contribution to questions whose query yields zero seeds would have required passing a vector-derived `doc_id` back into `GraphRetriever`, which is exactly the externally-seeded primitive v1 forbids;
- a hybrid that always calls both retrievers with the same frozen parameters has no threshold to tune, no switch to flip, and nothing to mislead an operator about which retriever did the work.

An LLM-classifier fallback is recorded as a disabled v2 option in §8; it is **not enabled** in v1.

## 3. Candidate depth

Both retrievers are invoked with the caller's query and a frozen `k` multiplier. The `k` value passed to each retriever is larger than the caller's `k` to give fusion room to interleave the two ranked lists.

```text
HYBRID_K_VEC_MULTIPLIER   = 2
HYBRID_K_GRAPH_MULTIPLIER = 2
HYBRID_HOP_BUDGET         = 3
```

The two retriever calls are exactly:

```python
vector_candidates = vector_retriever.retrieve(
    query,
    k = HYBRID_K_VEC_MULTIPLIER * k,        # = 2 * k
)

graph_candidates = graph_retriever.retrieve(
    query,
    k = HYBRID_K_GRAPH_MULTIPLIER * k,      # = 2 * k
    hop_budget = HYBRID_HOP_BUDGET,         # = 3, the #11 frozen benchmark value
)
```

Bounds:

- the vector `k` value never exceeds `min(2k, store_row_count)`;
- the graph `k` value never exceeds the number of unique `chunk_id`s the graph retriever admits;
- `hop_budget` is the globally-frozen #11 benchmark value `3` for the hybrid and **is never altered** by the hybrid. The held-out policy forbids reading `hop_bucket` from a question to alter depth; `3+` is a reporting category, not a query-time hint.

No other multiplier, hop budget, or depth adjustment exists in the v1 hybrid.


## 4. Merge: Reciprocal Rank Fusion (RRF)

RRF is the standard Cormack/Clarke/Buettcher/Soberón list-fusion algorithm. For each unique `chunk_id` that appears in at least one of the two retrievers' top lists:

```text
rrf_score(chunk_id) = Σ_r  1 / (k_rrf + rank_r(chunk_id))
                       over r ∈ {vector, graph}
                       where rank_r is the 1-based position in r's returned list
                       (skipped when retriever r did not return chunk_id)
```

The single knob is the RRF damping constant `k_rrf`. It is frozen:

```text
RRF_K = 60
```

`60` is the value used in the original Cormack et al. (2009) formulation and the value most commonly cited as a robust default across hybrid retrieval literature. There is no per-corpus search over this value; the v1 spec pins it.

### 4.1 Properties of RRF that earn it here

- **Score-invariant to scale.** RRF only uses rank, not raw score. Vector's `1 − cosine_distance` and graph's `1 / hop_count` have incompatible ranges, which makes weighted sum unstable. Rank fusion ignores ranges.
- **Bidirectional contribution.** A `chunk_id` returned by both retrievers contributes twice; a `chunk_id` returned by only one contributes once. This is the property we want: "evidence from two sources is stronger than evidence from one" without learned weights.
- **Deterministic.** No randomness, no model state. The same inputs always fuse to the same list.
- **Costless to audit.** The benchmark runner can reproduce the fused order from the two retriever outputs alone.

### 4.2 Why not weighted sum / linear combination

Weighted-sum fusion requires per-source weights. Those weights would be tuned on the held-out set (forbidden) or arbitrarily pinned (worse — defeats the point of fusion). RRF has no weights.

### 4.3 Why not cross-encoder rerank

Cross-encoder reranking requires a learned model. Forbidden in v1. A future v2 may add it; this spec does not pre-empt or preclude that.

## 5. Duplicate chunk handling

Dedup is structural in RRF: every `chunk_id` is computed exactly once as `Σ_r 1/(k_rrf + rank_r(c))`. Two retrievers contributing the same `chunk_id` add the two reciprocals; that is the point of fusion, not a defect.

Two physically distinct `chunk_id`s whose underlying text is near-identical (different `chunk_index`, similar text) are **not** deduplicated. The canonical `chunk_id` is the unit of retrieval and the eval ground truth references `chunk_id`s by that exact value. Semantic-text deduplication across `chunk_id`s is not the hybrid retriever's responsibility.

## 6. Ordering and tie-breaking

The fused list is ordered by exactly three deterministic keys, in this order:

1. `rrf_score` descending.
2. **`len(sources)` descending** — a chunk attributed to *both* retrievers sorts ahead of any chunk attributed to only one at the same RRF score. Rationale: dual-source attribution is the zero-cost signal fusion exists to amplify, and promoting it costs no learned weights.
3. `chunk_id` ascending (lexicographic, UTF-8 byte order). Final deterministic tie-break.

The hybrid does **not** rank a graph-only hit ahead of a vector-only hit merely because its source is graph. At `(rrf_score, len(sources))` equal, `sources == ("vector",)` and `sources == ("graph",)` are equal, and only `chunk_id` decides the tie. Per-source modality preference is forbidden in v1.

Tie-break is a deterministic, recorded rule. It is **not** a learned preference and is not tunable.


## 7. Hybrid retrieval interface

### 7.1 Result types

```python
from dataclasses import dataclass
from typing import Literal, Protocol

# Re-exported from graph-retrieval.md for type completeness:
@dataclass(frozen=True)
class GraphEvidenceStep:
    step: int
    from_id: str
    relation: str
    to_id: str
    direction: Literal["forward", "reverse"]

@dataclass(frozen=True)
class HybridHitProvenance:
    in_vector: bool
    in_graph: bool
    vector_rank: int | None           # 1-based; None when not in vector's top
    vector_score: float | None        # 1 − cosine_distance, verbatim from #10
    graph_rank: int | None            # 1-based; None when not in graph's top
    graph_score: float | None         # 1 / hop_count, verbatim from #11
    graph_evidence_path: tuple[GraphEvidenceStep, ...] | None

@dataclass(frozen=True)
class HybridRetrievalHit:
    chunk_id: str
    score: float                      # RRF fused score, higher-is-better
    sources: tuple[Literal["vector", "graph"], ...]  # deduped, sorted
    provenance: HybridHitProvenance
```

`sources` is a small sorted tuple. The benchmark runner can attribute every hit to its contributing retriever(s) without re-walking `provenance`. `graph_evidence_path` is the verbatim #11 canonical serialisation (with `from_id`/`to_id` mapped to JSON keys `from`/`to`); #11 §5 applies unchanged.

### 7.2 Protocol

```python
class HybridRetriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[HybridRetrievalHit]: ...
```

Composition: `HybridRetriever` is constructed with a `VectorRetriever` (per #10) and a `GraphRetriever` (per #11) and the frozen constants from §3–§4. It performs routing per §2, calls both retrievers with the depths from §3, fuses per §4, orders per §6, and returns `list[HybridRetrievalHit]`.

### 7.3 Empty / under-filled / zero-result behaviour

- **Empty store** in either retriever ⇒ that retriever's `[]`. Hybrid fuses the other side; if both are `[]`, hybrid returns `[]`.
- **Under-filled:** if fewer than `k` unique `chunk_id`s survive fusion, return all of them (no padding).
- **Invalid input** (`k` non-int or ≤ 0, non-string or empty query): same errors as #10 §4.3 and #11 §7. Both retrievers must be checked; in particular a non-int `k` (including `bool`) raises `TypeError` before any database call. The hybrid does not introduce new error classes.
- **Both empty:** `[]`. No fallback to LLM, fuzzy retrieval, embedding-similarity rerank, or any other adaptive behaviour.

## 8. Disabled fallback (recorded, not enabled)

A measured-failure fallback would let the hybrid switch to an LLM classifier (via `LLMProvider.generate`) when the deterministic rule measurably fails on the out-of-band dev slice. The fallback exists on paper as a v2 option, with the following contract:

- **Trigger condition:** measured recall@10 of the deterministic hybrid on `eval/dev_questions.jsonl` falls below the recall of a trivial vector-only baseline by ≥ X%. The threshold X is set by the user post-Wayfinder in the implementing ticket, not by the agent and not from the held-out set.
- **Mechanism:** replace `route = "graph-led" | "vector-led"` with a single-shot `LLMProvider.generate(...)` call whose prompt enumerates the ontology's entity-type names and asks for `{"route": "graph-led"|"vector-led", "reason": "..."}`.
- **Hot-path impact:** enables one `LLMProvider` call per query. Budget caps from #16 §6 govern it.

**The fallback is not enabled in v1.** The implementing ticket may enable it later **only after** the dev slice exists and only if the trigger condition fires. The held-out benchmark never uses this fallback; if enabled, it is benchmarked as a separate condition (e.g. `hybrid-routed-classifier`), not as `hybrid`.


## 9. Upstream consistency

- **#4 (eval-schema §6, no-tuning-on-held-out):** every hybrid parameter (`RRF_K`, `HYBRID_K_VEC_MULTIPLIER`, `HYBRID_K_GRAPH_MULTIPLIER`, `HYBRID_HOP_BUDGET`, and the three-key tie-break rule) is frozen before benchmark numbers are recorded. Tuning lives only in `eval/dev_questions.jsonl` or any out-of-band file.
- **#5 (ontology):** hybrid reuses #11's seed-lift over the frozen ontology entity-type names. Routing does not introduce new ontology types or relation types.
- **#7/#8 (extraction, resolution):** hybrid consumes #11's lift output unchanged. The lift uses #8 normalisation policy and #7 mention validation, both already settled.
- **#9 (Neo4j):** hybrid never writes to the graph. It reads the same #11 Cypher queries (through `GraphRetriever`) on the same projection.
- **#10 (vector):** `VectorRetriever.retrieve(query, k)` is composed unchanged. Hybrid's `k_vec = k * 2` is the only contract surface that interacts with the vector retriever, and #10 §4.3 already permits `k` values up to the store row count.
- **#11 (graph):** `GraphRetriever.retrieve(query, k, hop_budget=3)` is composed unchanged. The hybrid calls it at `hop_budget = HYBRID_HOP_BUDGET = 3`, which equals the #11 frozen benchmark value. Hybrid never alters the graph retriever's per-hit ordering or evidence path, and never re-invokes it with a different `hop_budget` or an externally-supplied seed.
- **#13 (grounded generation):** generation consumes `HybridRetrievalHit` exactly like `VectorRetrievalHit` / `GraphRetrievalHit` — it reads `chunk_id`, looks up `text` from `data/derived/chunks.jsonl`, and uses the provenance to decide which citations to surface.
- **#14 (benchmark runner):** the runner treats `hybrid` as a third condition with its own column in the headline table. Per-retriever metrics are derivable from each hit's `provenance.vector_*` / `graph_*` fields; the runner records the version of this spec in `report.json`.
- **#15 (serving surface):** `POST /ask` with `mode: "hybrid"` enters this contract unchanged. The serving layer does not learn any hybrid-specific fields.

No upstream contract is altered by this spec. `VectorRetriever`, `GraphRetriever`, and their result types are unchanged.

## 10. Out of scope

- **Retrieval implementation or fusion code.** This spec pins the contract; the implementing ticket writes the body.
- **Learned rerankers, cross-encoders, or any model that would require training data.** Forbidden in v1.
- **LLM, fuzzy, embedding-similarity, or vector-frontier fallback inside the hybrid.** All are out of scope; the disabled fallback is recorded in §8 only as a future v2 mechanism.
- **Per-question dynamic weight learning, gradient-based fusion, or bandit-style router adaptation.** Out of scope.
- **Changes to the graph retriever, vector retriever, ontology, extraction contract, entity resolution policy, or Neo4j schema.** Hybrid composes them; it does not edit them.
- **`README.md`.** User-owned; never modified.

## 11. Acceptance gate

This spec closes when committed to `docs/specs/hybrid-retrieval.md`. The implementing ticket closes only when:

1. `HybridRetriever.retrieve(query, k)` is implemented with the §1 signature, §2 routing, §3 candidate depth, §4 RRF, §5 deduplication, §6 ordering, and §7 result types.
2. A deterministic test fixture exercises both retrievers being invoked for every valid query with the frozen constants:
    - a query for which `GraphRetriever.retrieve` lifts ≥ 1 Entity/Document seed produces a non-empty `graph_candidates` list at `hop_budget = 3`;
    - a query for which `GraphRetriever.retrieve` lifts no seed returns `[]` from graph and the hybrid still completes; no fallback, no tail-call, no chunk→Document bootstrapping is invoked;
    - a static inspection of the hybrid module confirms exactly the two `retrieve(...)` calls in §2 and no `retrieve_from_seed(...)`, no query re-writing with a derived `doc_id`, and no second `hop_budget` value.
3. A second fixture exercises RRF: a `chunk_id` returned by both retrievers at ranks 3 and 2 outranks a `chunk_id` returned by vector at rank 1 alone; an empty retriever contributes nothing without disturbing the other side; under-filled results return every unique chunk.
4. A third fixture exercises the result envelope: every emitted `HybridRetrievalHit` carries `chunk_id`, `score`, non-empty `sources` from `{"vector","graph"}`, and a `provenance` whose `in_vector` / `in_graph` flags agree with `sources`. Hits with `in_graph == True` carry a non-empty `graph_evidence_path` whose `from`/`to` connectivity holds per #11 §5.
5. A self-check (per ponytail rule: one runnable check that fails if the logic breaks) iterates a small fixed set of fixture queries through `retrieve(query, k)` and asserts byte-identical output across repeated runs and across perturbed retriever return orders.
6. Static inspection finds no `LLMProvider`, `EmbeddingProvider`, learned-rerank library, provider SDK, or held-out-tuning dependency in the hybrid module. (`VectorRetriever` and `GraphRetriever` themselves remain subject to their own #10/#11 acceptance gates; this gate only checks the hybrid code does not add a new provider dependency.)
7. The frozen constants (`RRF_K = 60`, `HYBRID_K_VEC_MULTIPLIER = 2`, `HYBRID_K_GRAPH_MULTIPLIER = 2`, `HYBRID_HOP_BUDGET = 3`) appear in code as named constants; the implementing ticket records them with the spec version in `report.json` (per #14).

The first four gates validate the router, fusion, and result envelope. Gates five and six validate determinism and the hot-path constraint. Gate seven validates that the frozen parameters are not silently re-tuned.


---

This spec resolves ticket #12. It is referenced by #13 (grounded generation consumes `HybridRetrievalHit` like the other retriever hits), #14 (benchmark runner adds a `hybrid` column to the headline table and records the spec version in `report.json`), and #15 (serving surface routes `mode: "hybrid"` to this contract). The implementing ticket for #12 ships the `HybridRetriever` implementation, the deterministic test fixtures, and the small self-check that validates byte-identical output across repeated runs.
