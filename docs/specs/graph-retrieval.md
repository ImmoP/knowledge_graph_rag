# Graph-Only Retrieval Specification

Draft resolution of [#11 — Decide the graph-only retrieval shape](https://github.com/ImmoP/knowledge_graph_rag/issues/11). This document fixes deterministic seed lifting, semantic traversal, evidence collection, scoring, and the graph-only return contract. It does not implement retrieval.

The retriever reads the derived Neo4j projection fixed by [#9](neo4j-ingestion.md). Canonical JSONL artifacts remain authoritative. Graph retrieval does not call `LLMProvider` or `EmbeddingProvider`, import a provider SDK, query pgvector, or fall back to vector retrieval.

## 1. Preconditions

Before accepting queries, the graph retriever verifies that `data/graph_ingestion.manifest.json` exists and matches the graph and canonical inputs under #9. Missing or stale graph state is an error, not an empty result.

The retriever reads:

- `data/entities.jsonl` for Entity IDs, canonical names, and aliases.
- `data/manifest.jsonl` for Document IDs and titles.
- `ontology.yaml` for the allowed semantic relationship types and stored directions.
- the #9 Neo4j graph for traversal and relationship provenance.

`README.md`, extraction artifacts, entity-resolution artifacts, raw OpenAlex records, and chunk text are not mutated.

## 2. Deterministic query seed lifting

Seed lifting covers two graph-node namespaces:

| Namespace | Candidate strings | Canonical identity |
|---|---|---|
| Entity | `entity_id`, `canonical_name`, every `aliases` value from `data/entities.jsonl` | `("Entity", entity_id)` |
| Document | `doc_id`, `title` from `data/manifest.jsonl` | `("Document", doc_id)` |

An explicit `W...` document ID is therefore a valid candidate when that literal ID occurs in the query. Entity IDs are valid literal candidates for the same reason.

### 2.1 Matching normalization and boundary rule

Apply the #8 resolution-name normalization policy to candidate strings: Unicode NFKC, Unicode `casefold()`, whitespace collapse/trim, and only the pinned removal of irrelevant surrounding wrapper quotation punctuation. Digits, version/decimal separators, `@k`, internal hyphens, and other meaningful internal punctuation remain significant.

For query matching, apply NFKC, `casefold()`, and whitespace collapse to the complete query without stripping internal or whole-query punctuation. Search for each normalized candidate as an exact substring of that normalized query.

A substring match is boundary-valid exactly when:

- the character immediately before the match is absent or is not Unicode alphanumeric; and
- the character immediately after the match is absent or is not Unicode alphanumeric.

Thus a candidate cannot match inside a larger Unicode alphanumeric token. Punctuation inside the candidate remains part of the exact matched span; punctuation immediately outside it is a permitted boundary. There is no tokeniser-specific stemming, fuzzy matching, acronym generation, edit-distance threshold, embedding comparison, or learned type classifier.

### 2.2 Maximal spans and ambiguity

Represent every boundary-valid occurrence as its half-open normalized-query interval plus the candidate graph node. Maximal-span suppression applies only among overlapping candidate matches: discard a matched interval only when another valid overlapping interval strictly contains it. Matches sharing the same maximal interval are all retained, including when one name maps to several graph nodes. Partially overlapping intervals that do not contain one another are both maximal and remain eligible. Non-overlapping matches are independent and all remain eligible regardless of their relative lengths; for example, `RAG on Natural Questions` retains both `RAG` and `Natural Questions`.

De-duplicate retained seeds by canonical graph-node identity. Order them lexicographically by `(node_kind, canonical_id)`, where `Document` sorts before `Entity`. Multiple aliases or occurrences that identify the same node do not give it extra weight.

If no Entity or Document seed matches, return `[]`. Graph-only retrieval performs no internal fuzzy, vector, LLM, full-corpus, or provider-backed fallback.

## 3. Semantic traversal

Starting from every retained seed, enumerate paths containing between one and `hop_budget` semantic relationships. A semantic relationship is a relationship type declared in `ontology.yaml`: `CITES`, the eight extracted relationship types, or derived `EVALUATED_ON` under ontology v1.

Traversal may follow a stored ontology relationship forward or reverse. The canonical ontology name is unchanged in either direction. Each traversed semantic relationship consumes exactly one hop.

`HAS_CHUNK` and `MENTIONS` are #9 infrastructure relationships. They may be used only by the evidence-collection rules in §4, consume zero semantic hops, and never appear in `evidence_path`.

### 3.1 Hop budget

`hop_budget` is an integer inclusive maximum, not an exact requested path length. `bool` is not accepted as an integer. Values below one raise `ValueError`; other non-integer values raise `TypeError` before a database query.

The frozen v1 benchmark value is:

```text
GRAPH_HOP_BUDGET = 3
```

The benchmark uses this same value for all 45 questions. It never reads a held-out question's `hop_bucket` to select or alter retrieval depth. The `3+` bucket is a reporting category, not a query-time hint; a question whose minimum path exceeds the v1 ceiling may miss under this baseline.

Callers outside the canonical benchmark may supply another valid integer, but benchmark results are comparable only when their recorded graph-retrieval policy and hop budget match.

### 3.2 Path uniqueness and bounded expansion

A path must not repeat the same logical Neo4j relationship identity fixed by #9. Parallel extracted assertions remain distinct because their `assertion_id` values differ. Node revisits are permitted only when no relationship identity repeats. Enumeration and downstream selection use logical IDs and canonical values, never Neo4j internal IDs or query return order.

Traversal filters relationship types from the loaded ontology; it does not hard-code or dynamically admit arbitrary database relationship types. Paths of length zero do not produce hits.

## 4. Supporting-chunk provenance

Chunk collection is path-specific. A chunk is not an ordinary candidate merely because it mentions a reached Entity or belongs to a reached Document.

### 4.1 Assertion evidence

For each extracted ontology relationship in a path, collect the canonical `chunk_id` stored on that exact assertion relationship.

For each derived `EVALUATED_ON` relationship in a path, resolve every value in its `supporting_assertion_ids` to the corresponding participating extracted assertion relationship and collect those relationships' canonical `chunk_id` values. A missing, dangling, or inconsistent supporting assertion is stale graph state and fails retrieval rather than being skipped.

If a path contains at least one extracted or derived assertion-evidence chunk, that path returns only the sorted unique set of those assertion-evidence chunks. It does not additionally expand `MENTIONS` relationships or all `HAS_CHUNK` relationships for path nodes.

### 4.2 Pure structural `CITES` fallback

A path composed only of `CITES` relationships has metadata provenance, not assertion-text provenance. Because such a path contains no assertion-evidence chunk, it may use exactly one structural-context fallback: collect the canonical chunks belonging to the terminal `Document` through `HAS_CHUNK`.

Those chunks are context attached to the structurally reached document. They are not textual proof that a `CITES` relationship exists. `HAS_CHUNK` remains zero-hop plumbing and is omitted from `evidence_path`.

No other path uses this fallback. In particular, a mixed path containing `CITES` plus any extracted or derived assertion evidence returns only its assertion-evidence chunks.

## 5. Canonical evidence paths

Each semantic traversal serializes to the exact five-field #4 record shape:

```json
{"step":1,"from":"e_...","relation":"EVALUATED_ON","to":"e_...","direction":"forward"}
```

The fields are exactly:

- `step`: consecutive 1-based traversal position.
- `from`: canonical traversal-order node ID.
- `relation`: canonical ontology relationship name.
- `to`: canonical traversal-order node ID.
- `direction`: `forward` when `from -> to` follows the stored ontology edge; `reverse` when traversal goes opposite it.

For every step after the first, `previous.to == current.from`. Both directions consume one semantic hop. `HAS_CHUNK` and `MENTIONS` never serialize as steps.

An implementation may use Python attributes such as `from_id` and `to_id`, but canonical evidence serialization uses the field names `from` and `to`. Canonical path serialization is JSON over the ordered array of step objects with `sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=False`, and UTF-8.

## 6. Scoring and candidate reduction

The score uses only semantic path length:

```text
score = 1.0 / semantic_hop_count
```

There is no popularity, seed-frequency, path-count, evidence-count, relation-type, direction, confidence, or learned bonus. All paths of the same semantic length receive the same score.

For each unique `chunk_id`:

1. Keep the candidate with the highest score, equivalently the shortest semantic path.
2. If several candidates have the same score, keep the one whose canonical evidence-path serialization is lexicographically smallest by UTF-8 byte order.

Order final hits by score descending and then `chunk_id` ascending. Return at most `k` unique chunks. Query-seed ordering, database row order, Neo4j internal IDs, and the number of matching aliases or paths cannot affect the result.

## 7. Retrieval interface

```python
from dataclasses import dataclass
from typing import Literal, Protocol

@dataclass(frozen=True)
class GraphEvidenceStep:
    step: int
    from_id: str
    relation: str
    to_id: str
    direction: Literal["forward", "reverse"]

    # Canonical serialization maps from_id -> "from" and to_id -> "to".

@dataclass(frozen=True)
class GraphRetrievalHit:
    chunk_id: str
    score: float
    evidence_path: tuple[GraphEvidenceStep, ...]

class GraphRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        k: int,
        hop_budget: int = 3,
    ) -> list[GraphRetrievalHit]: ...
```

`chunk_id` and higher-is-better `score` deliberately align with `VectorRetrievalHit` from #10. Graph retrieval adds only its required semantic provenance.

Input and empty-result behavior:

- Non-string `query` raises `TypeError`; an empty or whitespace-only query returns `[]`.
- Non-integer `k` (including `bool`) raises `TypeError`; `k <= 0` raises `ValueError` before a database query.
- Non-integer `hop_budget` (including `bool`) raises `TypeError`; `hop_budget < 1` raises `ValueError` before a database query. The default remains the frozen v1 benchmark value `3`.
- No lifted seeds, an empty valid graph, no semantic path, or no eligible evidence chunk returns `[]`.
- If fewer than `k` unique chunks qualify, return all qualifying chunks.
- Multiple seeds are traversed together; candidates are unioned and reduced by §6.

## 8. Held-out governance

The 45 canonical questions in `eval/questions.jsonl` are held out. They may never be used to select, validate, adjust, or justify:

- seed-normalization or boundary rules;
- fuzzy thresholds, because v1 has no fuzzy lifting;
- maximal-span or ambiguity behavior;
- the hop budget;
- scoring rules or weights;
- fallback behavior or tie-breaks.

The v1 rules and benchmark hop budget are frozen before benchmark authoring/evaluation. Separate manually authored resolver/retrieval development evidence may motivate a future version, but it lives outside the canonical 45 and any changed contract is versioned before evaluation.

## 9. Upstream consistency

- **#4:** evidence paths use the repaired five-field traversal record with explicit direction. Hop count remains the number of semantic relationship records; benchmark size and split policy do not change.
- **#5:** only ontology-declared relationships are traversed; their canonical names, stored directions, endpoints, and population modes remain unchanged.
- **#8:** Entity lifting reuses the deterministic name normalization policy but performs exact query-span lookup only. It does not rerun or alter entity resolution.
- **#9:** canonical graph IDs, relationship provenance, `HAS_CHUNK`/`MENTIONS` plumbing classification, and the ban on internal IDs are preserved. Retrieval rejects missing/stale ingestion state.
- **#10:** result types share `chunk_id` and higher-is-better `score`; graph-only adds `evidence_path` and takes an explicit hop budget. The benchmark/hybrid layer may adapt the two protocols without changing either retriever's semantics.

No unresolved contradiction remains after the #4 direction-field repair.

## 10. Implementing-ticket acceptance gate

The later implementation ticket closes only when a deterministic test fixture demonstrates:

1. Exact normalized Entity and Document lifting, explicit `W...` matching, boundary rejection inside larger alphanumeric tokens, maximal-span suppression, ambiguity retention, de-duplication, and zero-match `[]` behavior.
2. Forward and reverse traversal restricted to ontology relationships, with `HAS_CHUNK` and `MENTIONS` absent from paths and hop counts.
3. One-, two-, and three-hop paths respect the inclusive budget, while the benchmark configuration remains globally fixed at three and does not inspect `hop_bucket`.
4. Extracted edges yield only their exact assertion chunks; derived edges yield participating assertion chunks; mixed paths add no arbitrary node-context chunks.
5. Pure `CITES` paths alone use terminal-Document chunks and identify them as structural context rather than citation proof.
6. Scoring, best-candidate reduction, canonical-path ties, and final `chunk_id` ties reproduce byte-identical ordering across repeated runs and perturbed database return order.
7. Every emitted path serializes with exactly `step`, `from`, `relation`, `to`, and `direction`, remains connected, and uses no Neo4j internal ID.
8. Invalid query, `k`, and hop-budget inputs follow §7/§3.1; under-filled and empty results follow §7.
9. Static inspection finds no LangChain, provider SDK, `LLMProvider`, `EmbeddingProvider`, pgvector, fuzzy matcher, learned ranker/router, or held-out-tuning dependency in graph retrieval.

## 11. Out of scope

- Retrieval implementation or Cypher optimization.
- Query-time LLM, embedding, fuzzy, or vector fallback.
- Learned ranking, routing, seed disambiguation, or relation weighting.
- Full-text indexes, vector indexes, APOC, or Graph Data Science.
- Changes to graph ingestion, entity resolution, ontology vocabulary, or the 45-question benchmark.
- Hybrid routing/fusion, grounded generation, benchmark scoring, or serving.

---

Once approved and committed, this spec resolves #11. Implementation work belongs to the post-Wayfinder implementation tickets.
