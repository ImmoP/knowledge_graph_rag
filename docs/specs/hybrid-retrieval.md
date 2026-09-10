# Hybrid / Routed Retrieval Specification

Resolution of [#12 — Decide the hybrid/routed retrieval shape](https://github.com/ImmoP/knowledge_graph_rag/issues/12). This document is the **contract** the hybrid/routed retrieval baseline must satisfy. It does not implement retrieval.

The hybrid retriever is the third of three retrieval shapes that ship together as benchmark conditions: vector-only ([#10](https://github.com/ImmoP/knowledge_graph_rag/issues/10)), graph-only ([#11](https://github.com/ImmoP/knowledge_graph_rag/issues/11)), and hybrid/routed (this spec). Hybrid is a **first-class benchmark condition**, not an opportunistic combination. The router is a small deterministic rule, never a learned router in v1. The spec composes the #10 vector retriever and the #11 graph retriever unchanged; it adds only routing, fusion, and a hybrid result type that records per-retriever provenance.

## 1. Standing constraints

These are the architectural properties hybrid inherits from upstream specs and the parent map. Each is named so the implementing ticket can audit against it.

1. **First-class benchmark condition.** Hybrid results must be comparable across runs and benchmarkable per hop bucket alongside vector-only and graph-only. The hybrid shape does not block this.
2. **Composes #10 and #11 unchanged.** `VectorRetriever.retrieve(query, k)` and `GraphRetriever.retrieve(query, k, hop_budget=3)` are called directly with the signatures already pinned. Neither retriever is reimplemented.
3. **Deterministic routing.** Routing is a pure function of the query string, the ontology, and the canonical artifacts. No learned model, no embedding similarity for routing, no `LLMProvider` call in the hot path. A measured-failure fallback exists on paper (§8) but is **not enabled**.
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

Routing is the deterministic rule that picks which retriever leads the merge for a given query. It is computed *before* either retriever is called and uses the same seed-lifting machinery as #11 §2, with the same normalisation rules (NFKC + Unicode `casefold()` + whitespace collapse, retention of all boundary-valid matches, maximal-span suppression only among proper overlapping matches, ambiguity retention, deduplication by canonical node identity, lexicographic order by `(node_kind, canonical_id)` with `Document` before `Entity`).

### 2.1 The rule, in two cases

| Case | Seed-lift result | Lead retriever | Tail-call |
|---|---|---|---|
| **graph-led** | ≥ 1 Entity or Document seed lifted | graph at `hop_budget = 3`, `k_g = k * 2` | vector at `k_vec = k * 2` |
| **vector-led** | 0 seeds lifted | vector at `k_vec = k * 2` | graph at `hop_budget = 1`, `k_g_tail = k`, seeded with the **vector top-1 hit's `doc_id`** as a Document seed |

Both retrievers always run. The router only chooses the *order* (which goes first) and the *depth* of the secondary call. It does **not** short-circuit either retriever.

The vector-led tail-call is the only piece of routing that is not a direct reprojection of #11's seed-lift machinery. Its purpose is to keep pure `CITES` 1-hop questions alive — the user names neither the cited paper nor the cited-by paper, the lift is empty, but the vector retriever often surfaces a chunk whose `doc_id` is the right anchor. Using the vector top-1 `doc_id` as a Document seed at `hop_budget = 1` enumerates `CITES` paths in and out of that document. At `hop_budget = 1` only direct-citation document chunks (or their citations) surface, which is the correct shape for a 1-hop citation question and is bounded in cost. The tail-call **only fires when the graph-led branch's seed-lift was empty**; if the lift was non-empty, the graph has already traversed `hop_budget = 3` from the real seeds.

If the vector top-1 hit is a chunk whose `doc_id` is not a Document node (impossible under the chunk-record spec, which forces `doc_id = openalex_id` ∈ `data/manifest.jsonl` — the Document node), the tail-call seeds with no Document, the graph-lift yields `[]`, and the tail-call returns `[]`. The tail-call never raises on a malformed seed.

### 2.2 Why this rule and not a learned router

- A learned router requires either labels (which the held-out rules forbid tuning on) or unsupervised contrastive training (which the v1 scope explicitly excludes).
- A rule based on the existing deterministic seed-lift is **zero new mechanism** — it reuses the exact machinery #11 already specified and validated.
- A threshold on vector confidence (e.g. "if top score > X then vector-led") depends on a number tuned on the held-out set; forbidden.
- Always-both is cheaper than running two parallel services with conditional branches: one process, two retrievers, deterministic order, easy audit.

### 2.3 No embedding-based routing

Embedding similarity between the query and any stored entity / document summary is **not used for routing**. It would require a learned embedding space tuned against retrieval-quality labels, which the held-out policy forbids, and it would silently make the routing decision depend on the `EmbeddingProvider` adapter.

## 3. Candidate depth

The `k` values passed to the two retrievers are not the caller's `k`. They are larger to give fusion room to interleave the two ranked lists.

| Retriever | When | `k` requested | Bound |
|---|---|---|---|
| Vector (#10) | always | `k_vec = k * 2` | never exceeds `min(2k, store_row_count)` |
| Graph (#11) | graph-led | `k_g = k * 2` | never exceeds the number of unique `chunk_id`s graph admits |
| Graph (#11) | vector-led tail-call | `k_g_tail = k` | never exceeds the number of unique `chunk_id`s graph admits |

These multipliers are **frozen**:

```text
HYBRID_K_VEC_MULTIPLIER        = 2
HYBRID_K_GRAPH_MULTIPLIER      = 2   # graph-led only
HYBRID_K_GRAPH_TAIL_MULTIPLIER = 1   # vector-led tail-call: k * 1
```

`hop_budget` for graph is **never tuned**: graph-led uses the #11 frozen benchmark value `3`, vector-led tail-call uses `1`. The held-out policy forbids reading `hop_bucket` from a question to alter depth; `3+` is a reporting category, not a query-time hint.


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

Sort the fused list by:

1. `rrf_score` descending.
2. **sources length descending** — a chunk attributed to *both* retrievers sorts ahead of a chunk attributed to only one, at the same RRF score. Rationale: dual-source attribution is the signal fusion exists to amplify; promoting it is the cheapest zero-cost way to honour that without learned weights.
3. **`sources == ("graph",)` first** at the same `(score, sources_len)` — a graph-only hit leads a vector-only hit when both have length 1, because graph hits carry semantic evidence the vector retriever cannot fabricate on a paraphrase query.
4. `chunk_id` ascending (lexicographic, UTF-8 byte order). Final deterministic tie-break.

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

- **#4 (eval-schema §6, no-tuning-on-held-out):** every hybrid parameter (`RRF_K`, the three multipliers, the tie-break rule, the tail-call presence) is frozen before benchmark numbers are recorded. Tuning lives only in `eval/dev_questions.jsonl` or any out-of-band file.
- **#5 (ontology):** hybrid reuses #11's seed-lift over the frozen ontology entity-type names. Routing does not introduce new ontology types or relation types.
- **#7/#8 (extraction, resolution):** hybrid consumes #11's lift output unchanged. The lift uses #8 normalisation policy and #7 mention validation, both already settled.
- **#9 (Neo4j):** hybrid never writes to the graph. It reads the same #11 Cypher queries (through `GraphRetriever`) on the same projection.
- **#10 (vector):** `VectorRetriever.retrieve(query, k)` is composed unchanged. Hybrid's `k_vec = k * 2` is the only contract surface that interacts with the vector retriever, and #10 §4.3 already permits `k` values up to the store row count.
- **#11 (graph):** `GraphRetriever.retrieve(query, k, hop_budget=3)` is composed unchanged. The vector-led tail-call uses `hop_budget=1` (a valid integer ≥ 1 per #11 §3.1). Hybrid never alters the graph retriever's per-hit ordering or evidence path.
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
2. A deterministic test fixture exercises the router rule:
    - A query whose lifted seeds are non-empty lands in the graph-led branch and the graph runs at `hop_budget = 3`.
    - A query whose lifted seeds are empty lands in the vector-led branch and the tail-call graph runs at `hop_budget = 1` using the vector top-1 `doc_id`.
    - A query whose vector top-1 hit has no resolvable Document node (synthetic stress case) still completes and returns `[]` from the tail-call, never raising.
3. A second fixture exercises RRF: a `chunk_id` returned by both retrievers at ranks 3 and 2 outranks a `chunk_id` returned by vector at rank 1 alone; an empty retriever contributes nothing without disturbing the other side; under-filled results return every unique chunk.
4. A third fixture exercises the result envelope: every emitted `HybridRetrievalHit` carries `chunk_id`, `score`, non-empty `sources` from `{"vector","graph"}`, and a `provenance` whose `in_vector` / `in_graph` flags agree with `sources`. Hits with `in_graph == True` carry a non-empty `graph_evidence_path` whose `from`/`to` connectivity holds per #11 §5.
5. A self-check (per ponytail rule: one runnable check that fails if the logic breaks) iterates a small fixed set of fixture queries through `retrieve(query, k)` and asserts byte-identical output across repeated runs and across perturbed retriever return orders.
6. Static inspection finds no `LLMProvider`, `EmbeddingProvider`, learned-rerank library, provider SDK, or held-out-tuning dependency in the hybrid module. (`VectorRetriever` and `GraphRetriever` themselves remain subject to their own #10/#11 acceptance gates; this gate only checks the hybrid code does not add a new provider dependency.)
7. The frozen constants (`RRF_K = 60`, three multipliers, `hop_budget = 3` for graph-led, `hop_budget = 1` for tail-call) appear in code as named constants; the implementing ticket records them with the spec version in `report.json` (per #14).

The first four gates validate the router, fusion, and result envelope. Gates five and six validate determinism and the hot-path constraint. Gate seven validates that the frozen parameters are not silently re-tuned.


---

This spec resolves ticket #12. It is referenced by #13 (grounded generation consumes `HybridRetrievalHit` like the other retriever hits), #14 (benchmark runner adds a `hybrid` column to the headline table and records the spec version in `report.json`), and #15 (serving surface routes `mode: "hybrid"` to this contract). The implementing ticket for #12 ships the `HybridRetriever` implementation, the deterministic test fixtures, and the small self-check that validates byte-identical output across repeated runs.
