# Ontology Specification

Resolution of [#5 — Curate the ontology (5–10 entities, 8–15 relations)](https://github.com/ImmoP/knowledge_graph_rag/issues/5). This document pins the **`ontology.yaml` schema** and the **recommended entity/relation vocabulary**. The actual instances (which `Method` is which, which `Dataset` is which) are populated in the implementing authoring ticket — the user authors the ontology.

`ontology.yaml` is the authoritative machine-readable schema. The extraction prompt template and the Neo4j ingestion contract both read from it; the schema is not duplicated in Python code. Per the standing preferences, the ontology is hand-curated, driven by what the eval set needs to ask about (not completeness), and Author-entity type is excluded unless a benchmark path requires it.

## 1. Scope discipline

Every entity type and every relation type MUST satisfy two gates before being added:

1. **Eval-set gate:** at least one category of held-out question (1-hop, 2-hop, or 3+) becomes expressible as a `evidence_path` traversal because this type exists. Adding a type that no question would ever traverse is a defect.
2. **Extraction gate:** at least one feasible extraction prompt template (per [#7](https://github.com/ImmoP/knowledge_graph_rag/issues/7)) can identify instances of this type from paper text. Adding a type that no extraction prompt can reliably populate is a defect.

The implementing authoring ticket records, for each entity and relation type, the eval-set question pattern that motivated it and the extraction prompt that populates it. The validation tool (see §4) refuses to ship an `ontology.yaml` whose types don't satisfy both gates.

## 2. Recommended entity types

The recommended starter set, sized to fit the 5–10 entity-type budget:

| # | Type        | Definition                                                                                       | Example instances from corpus (illustrative)                          | Eval-set motivation                                                      |
|---|-------------|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|--------------------------------------------------------------------------|
| 1 | `Document`  | A paper in the corpus. Identity is the manifest `openalex_id` (e.g. `W1522301498`).             | every paper in `data/manifest.jsonl`                                | Source node for citation traversal and per-paper relations               |
| 2 | `Chunk`     | A derived chunk from a document, per the chunk-record spec ([#2](https://github.com/ImmoP/knowledge_graph_rag/issues/2)). Identity is `chunk_id` (e.g. `c_a1b2c3d4e5f60718`). | every chunk in `data/derived/chunks.jsonl`                           | Evidence-text carrier; every retrieval result is a `Chunk`               |
| 3 | `Method`    | A named retrieval / generation / representation method discussed in the corpus.                 | `BM25`, `DPR`, `ColBERT`, `RAG`, `REALM`, `Self-RAG`, `FLARE`         | 1-hop and 2-hop questions about which paper proposes which method        |
| 4 | `Dataset`   | A named evaluation dataset used or introduced in the corpus.                                     | `Natural Questions`, `TriviaQA`, `MS-MARCO`, `HotpotQA`, `SQuAD`, `BEIR` | 2-hop questions about which paper evaluates on which dataset              |
| 5 | `Metric`    | A named retrieval or QA evaluation metric.                                                       | `Recall@k`, `MRR`, `NDCG`, `Exact Match`, `F1`                       | 1-hop questions about which metric a paper reports; 3-hop compositional  |
| 6 | `Task`      | A named retrieval / QA / generation task.                                                        | `Open-Domain QA`, `Multi-Hop QA`, `Passage Retrieval`, `Re-ranking`  | 2-hop / 3-hop questions linking datasets → tasks, methods → tasks        |

**Total: 6 entity types**, inside the 5–10 budget with room for one or two more if the implementing authoring ticket finds a real need.

### 2.1 Excluded types and why

- **`Author`:** excluded per user instruction. Adding Author would be easy to extract but no benchmark path requires traversing authors in this corpus.
- **`Concept`, `Finding`, `Claim`:** excluded as junk-drawer risk. Anything that would land in `Concept` should be classified more specifically as `Method`, `Task`, or `Dataset`. Anything that would land in `Finding` is expressible as a `(Method) -[EVALUATED_ON]-> (Dataset)` triple plus a numeric score attached to the relation.
- **`Benchmark`:** not added. A "benchmark" in this corpus is a `(Task, Dataset, Metric)` composition; modelling it as a separate entity would duplicate structure and complicate retrieval. Implementations may compose these three types at query time.

### 2.2 Adding a 7th or 8th type

If during implementing authoring the user identifies a category that no current type captures and that supports a real benchmark question, they may add a type. The validation tool (per §4) enforces both gates and refuses types that fail them.

## 3. Recommended relation types

The recommended starter set, sized to fit the 8–15 relation-type budget:

| #  | Relation                          | From → To              | Definition                                                                                  | Example instances from corpus (illustrative)                                                                                                  |
|----|-----------------------------------|------------------------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | `CITES`                           | `Document` → `Document` | Document A references Document B in its bibliography. Direction: citing → cited.            | `(W1522301498) -[CITES]-> (W_HISTORICAL_RMSPROP)`                                                                                            |
| 2  | `PROPOSES`                        | `Document` → `Method`  | Document introduces or originates a method.                                                  | `(W1522301498) -[PROPOSES]-> (Adam Optimizer)`                                                                                                |
| 3  | `INTRODUCES`                      | `Document` → `Dataset` | Document releases or proposes a dataset.                                                    | `(W_NQ_PAPER) -[INTRODUCES]-> (Natural Questions)`                                                                                            |
| 4  | `EVALUATES_ON`                    | `Document` → `Dataset` | Document reports experimental results on a dataset.                                         | `(W_SELF_RAG) -[EVALUATES_ON]-> (PopQA)`                                                                                                      |
| 5  | `USES_METHOD`                     | `Document` → `Method`  | Document applies or builds on a method without originating it. Distinct from `PROPOSES`.     | `(W_RETRO) -[USES_METHOD]-> (ColBERT)`                                                                                                        |
| 6  | `REPORTS_METRIC`                  | `Document` → `Metric`  | Document reports a metric in its results.                                                   | `(W_RAG_PAPER) -[REPORTS_METRIC]-> (Exact Match)`                                                                                            |
| 7  | `EXTENDS`                         | `Method` → `Method`    | Method A builds on or generalises method B.                                                 | `(Adam) -[EXTENDS]-> (RMSProp)`                                                                                                               |
| 8  | `EVALUATED_ON`                    | `Method` → `Dataset`   | A method's empirical evaluation is on a dataset. (Derivable from `Document EVALUATES_ON Dataset` when the document proposes the method.) | `(RAG) -[EVALUATED_ON]-> (Natural Questions)`                                                                                                 |
| 9  | `USED_FOR_TASK`                   | `Dataset` → `Task`     | A dataset supports or is standard for a task.                                               | `(HotpotQA) -[USED_FOR_TASK]-> (Multi-Hop QA)`                                                                                                |
| 10 | `MEASURES`                        | `Metric` → `Task`      | A metric evaluates performance on a task.                                                   | `(Recall@k) -[MEASURES]-> (Passage Retrieval)`                                                                                                |

**Total: 10 relation types**, inside the 8–15 budget.

### 3.1 Why each relation earns a slot

- `CITES` is the citation-graph backbone. Without it, document-to-document traversal collapses to "any paper that mentions another paper's title," which is noisier than citations.
- `PROPOSES` vs `USES_METHOD`: a 1-hop question like "Which paper first proposed method X?" requires distinguishing origin from application. Without both, the answer collapses to whatever the retriever finds first.
- `EVALUATES_ON` (Document → Dataset) vs `EVALUATED_ON` (Method → Dataset): the document-level relation is what the extraction prompt naturally emits ("In this paper we evaluate on NQ"); the method-level relation is a derived closure used by retrieval. The implementing ingestion contract (per #9) may either store both or derive the method-level one at query time, but the ontology exposes both so the retrieval contract (#11) can reason about either path.
- `EXTENDS` is the only method-to-method relation; it captures the lineage that motivates 3+ hop questions like "what earlier method did Adam generalise, and on what dataset was the earlier method first evaluated?"
- `USED_FOR_TASK` and `MEASURES` give dataset→task and metric→task closures. Together with `EVALUATED_ON` they support 3-hop questions of the form "what dataset, task, and metric does method X involve?" — the kind of compositional question the eval set needs.

### 3.2 Excluded relations and why

- **`AUTHORED_BY` (Document → Author):** excluded because Author is excluded.
- **`CO_OCCURS_WITH` (Entity → Entity):** excluded as too noisy. Co-occurrence is what an LLM would emit by default; the ontology refuses it.
- **`SIMILAR_TO` (Method → Method):** excluded as vague. Method-to-method relations should be specific (`EXTENDS`); "similar" is what retrieval explores, not what the graph asserts.

### 3.3 Adding a relation

Same gates as §2: any added relation must support a `evidence_path` traversal in the eval set AND be expressible in an extraction prompt template.

## 4. File schema: `ontology.yaml`

The file is YAML. Top-level keys:

```yaml
version: 1
entity_types:
  - name: <snake_case string>
    definition: <one-sentence definition>
    examples:
      - <example instance>
      - <example instance>
    eval_pattern: <which eval question category motivates this type>
    extraction_prompt: <which prompt template populates instances of this type>

relation_types:
  - name: <SCREAMING_SNAKE_CASE string>
    from: <entity type name>
    to: <entity type name>
    definition: <one-sentence definition>
    examples:
      - from: <example source instance>
        relation: <relation name>
        to: <example target instance>
    eval_pattern: <which eval question category motivates this relation>
    extraction_prompt: <which prompt template populates instances of this relation>
```

### 4.1 Field rules

- `version` is an int, currently `1`. Bumped when the schema shape changes.
- `entity_types[].name` and `relation_types[].name` are unique within their list.
- `entity_types[].examples` is non-empty.
- `relation_types[].from` and `.to` are valid `entity_types[].name` values.
- `relation_types[].examples` is non-empty; each example's `from` and `to` are example instances of the corresponding entity type.
- `eval_pattern` is free-form text but must reference one of the hop buckets (`1-hop`, `2-hop`, `3+`) or state "structural" if the relation is needed for traversal plumbing rather than direct retrieval.
- `extraction_prompt` references a named prompt template (the per-type templates are authored in the extraction contract per #7).

### 4.2 Author attribution

`ontology.yaml` may include a top-level `notes` field (free-form string) for the author's rationale comments. The validation tool ignores this field.

## 5. Validation rules (implementing authoring tool)

The implementing authoring tool enforces, before writing `ontology.yaml`:

1. **Schema conformance** against §4. Type checks, required-field presence, list non-emptiness.
2. **Type-count gate:** 5 ≤ `len(entity_types)` ≤ 10; 8 ≤ `len(relation_types)` ≤ 15. The tool refuses to write outside these ranges.
3. **Unique names:** `entity_types[].name` and `relation_types[].name` are unique within their lists.
4. **Endpoint conformance:** every `relation_types[].from` and `.to` is a valid entity type name.
5. **Examples non-empty** for every type.
6. **Author exclusion:** no entity type named `Author`. (If a real benchmark path ever requires authors, the schema bump and this rule update are joint work.)
7. **No junk-drawer type names:** `Concept`, `Finding`, `Claim`, `Benchmark`, `Topic`, `Theme` are forbidden at the type-name level. The validation tool refuses them.

The tool also runs a "appears in extraction prompt template" cross-check against the extraction contract (#7): every `entity_types[].extraction_prompt` and `relation_types[].extraction_prompt` must reference a prompt template that exists in the extraction pipeline. The implementing extraction ticket (#7) defines these templates; the authoring tool reads them.

## 6. Acceptance gate

This spec closes when committed to `docs/specs/ontology.md`. The implementing authoring ticket closes when:

1. `ontology.yaml` conforms to §4.
2. Every entity type and every relation type satisfies the eval-set gate and the extraction gate (§1).
3. Every type's `examples` are drawn from the corpus (papers in `data/manifest.jsonl`, methods/datasets/metrics/tasks that actually appear in those papers).
4. The 5 ≤ entity_types ≤ 10 and 8 ≤ relation_types ≤ 15 size constraints hold.
5. The validation tool's checks (§5) pass on the file.

The first three gates validate substance; the last two validate shape.

## 7. Downstream coupling

- **#7 extraction contract** reads `entity_types[]` and `relation_types[]` to construct per-type prompt templates. The extraction prompt for each entity type references the type's `examples` as few-shot exemplars.
- **#8 entity resolution** uses `entity_types[].examples` as known-candidate seeds for the resolution policy.
- **#9 Neo4j ingestion** reads the relation types as the edge vocabulary; Cypher constraints and uniqueness rules derive from `relation_types[].from` and `.to`.
- **#11 graph retrieval** uses the relation types as the traversal relation-filter. Cypher queries are parameterised on the ontology.
- **#12 hybrid retrieval** consults the ontology for routing decisions ("is the query about a `Method`?").
- **#4 evaluation schema** uses the entity-type names as the namespace for `e_<id>` references in evidence paths.

The ontology is therefore a single load-bearing artifact: every retrieval / extraction / ingestion decision reads from it. Any change to `ontology.yaml` requires re-running extraction and re-ingestion.

---

This spec resolves ticket #5. The implementing authoring ticket populates `ontology.yaml` with the recommended starter content from §2 and §3 (or a user-refined version that still satisfies §4–§5), populates the `examples` fields with real corpus instances, and ships the validation tool.
