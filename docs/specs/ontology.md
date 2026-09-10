# Ontology Specification

Resolution of [#5 — Curate the ontology (5–10 entities, 8–15 relations)](https://github.com/ImmoP/knowledge_graph_rag/issues/5). This document pins the **`ontology.yaml` schema**, the **frozen v1 vocabulary**, and the **validation rules**. The actual instances (which `Method` is which, which `Dataset` is which) and the real-corpus `examples` are populated in the implementing authoring ticket.

`ontology.yaml` is the authoritative machine-readable schema. The extraction contract ([#7](https://github.com/ImmoP/knowledge_graph_rag/issues/7)) and the Neo4j ingestion contract ([#9](https://github.com/ImmoP/knowledge_graph_rag/issues/9)) read vocabulary, endpoints, and population modes from it. The schema is **not** duplicated in Python code.

## 1. Frozen v1 vocabulary

The following type names and relation names are the authoritative v1 vocabulary. Implementation tickets may populate instances and improve real examples, but **MUST NOT** silently add, remove, rename, or change endpoint types.

### 1.1 Entity types (6, inside the 5–10 budget)

| Name      | Population mode | Definition                                                                                       |
|-----------|-----------------|--------------------------------------------------------------------------------------------------|
| `document`| structural      | A paper in the corpus. Identity is the manifest `openalex_id` (e.g. `W1522301498`).              |
| `chunk`   | structural      | A derived chunk from a document, per the chunk-record spec ([#2](https://github.com/ImmoP/knowledge_graph_rag/issues/2)). Identity is `chunk_id` (e.g. `c_a1b2c3d4e5f60718`). |
| `method`  | extracted       | A named retrieval / generation / representation method discussed in the corpus.                  |
| `dataset` | extracted       | A named evaluation dataset used or introduced in the corpus.                                     |
| `metric`  | extracted       | A named retrieval or QA evaluation metric.                                                       |
| `task`    | extracted       | A named retrieval / QA / generation task.                                                        |

### 1.2 Relation types (10, inside the 8–15 budget)

| Name             | From → To            | Population mode | Definition                                                                                |
|------------------|----------------------|-----------------|-------------------------------------------------------------------------------------------|
| `CITES`          | `document` → `document` | structural   | Document A references Document B in its bibliography. Direction: citing → cited.          |
| `PROPOSES`       | `document` → `method`   | extracted    | Document introduces or originates a method.                                                |
| `INTRODUCES`     | `document` → `dataset`  | extracted    | Document releases or proposes a dataset.                                                  |
| `EVALUATES_ON`   | `document` → `dataset`  | extracted    | Document reports experimental results on a dataset.                                       |
| `USES_METHOD`    | `document` → `method`   | extracted    | Document applies or builds on a method without originating it.                            |
| `REPORTS_METRIC` | `document` → `metric`   | extracted    | Document reports a metric in its results.                                                 |
| `EXTENDS`        | `method` → `method`     | extracted    | Method A builds on or generalises method B.                                               |
| `EVALUATED_ON`   | `method` → `dataset`    | derived      | A method's empirical evaluation is on a dataset. Derived from `Document EVALUATES_ON Dataset` when the proposing document evaluates on the dataset. |
| `USED_FOR_TASK`  | `dataset` → `task`      | extracted    | A dataset supports or is standard for a task.                                             |
| `MEASURES`       | `metric` → `task`       | extracted    | A metric evaluates performance on a task.                                                 |

### 1.3 Excluded types and why

- **`author`:** excluded per user standing preference. Adding author would be easy to extract but no benchmark path requires traversing authors in this corpus.
- **`concept`, `finding`, `claim`, `benchmark`, `topic`, `theme`:** excluded as junk-drawer risks. Anything that would land in `concept` should be classified more specifically as `method`, `task`, or `dataset`. Anything that would land in `finding` is expressible as a `(method) -[EVALUATED_ON]-> (dataset)` triple plus a numeric score attached to the relation. The validation tool forbids these names.

### 1.4 Vocabulary change rule

Any change to the v1 vocabulary (adding a type, removing a type, renaming, changing endpoints, changing population mode) requires:

1. An **ontology version bump** in `ontology.yaml` (`version: 1` → `version: 2`, …).
2. **Regeneration of dependent extraction and graph artifacts** — the extraction pipeline must be re-run, the Neo4j graph must be re-ingested, and the chunk–entity / doc–entity maps must be rebuilt.
3. **Benchmark metadata** recording the ontology version and the SHA256 of `ontology.yaml` at the time of the benchmark run.

Silent vocabulary changes are forbidden. This rule is what makes benchmark reports comparable across runs.

## 2. Population modes

Every entity and relation type carries exactly one `population_mode`:

| Mode         | Meaning                                                                                                    | Required field                                  |
|--------------|------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| `structural` | Produced deterministically from canonical project artifacts or citation-resolution data, not by an LLM.   | `source` (deterministic source / procedure)     |
| `extracted`  | Populated by an LLM via a named prompt template; the only mode that may call the `LLMProvider`.            | `extraction_prompt` (named template reference)  |
| `derived`    | Computed deterministically from other graph facts at query time or at ingestion time, never by an LLM.     | `derivation` (the derivation rule)              |

Only `extracted` modes require an extraction prompt template. The earlier spec's blanket requirement that every type has an `extraction_prompt` was wrong; `document`, `chunk`, `CITES`, and `EVALUATED_ON` are not LLM-extracted.

### 2.1 Why each v1 type has the mode it does

- **`document` is structural** because paper identity comes from `data/manifest.jsonl` (a canonical artifact), not from extraction.
- **`chunk` is structural** because chunk identity and text come from `data/derived/chunks.jsonl` (a canonical artifact, per [#2](https://github.com/ImmoP/knowledge_graph_rag/issues/2)).
- **`CITES` is structural** because v1 derives the in-corpus citation graph deterministically from `data/raw/openalex/<work>.json` `referenced_works`, canonicalises IDs through `data/aliases.json`, and retains only canonical targets in `data/manifest.jsonl`. No LLM is involved.
- **`EVALUATED_ON` is derived** because it is the closure of `document PROPOSES method` × `document EVALUATES_ON dataset`: when the same document both proposes a method and evaluates on a dataset, the corresponding `(method, dataset)` pair gets an `EVALUATED_ON` edge. The LLM does not independently emit `EVALUATED_ON` triples.

Under the #9 Neo4j label projection, the illustrative witness match is `MATCH (d:Document)-[:PROPOSES]->(m:Entity:Method), (d)-[:EVALUATES_ON]->(ds:Entity:Dataset)`. This is only the label-consistent expression of the existing derivation; it does not change the vocabulary, endpoints, population mode, semantics, or ontology version.

### 2.2 Why no type is `extracted` for non-named entities

Methods, datasets, metrics, and tasks are the four `extracted` entity types — they are the things that appear in paper text under named, surface-form mentions. `document`, `chunk`, and `CITES` come from canonical artifacts; `EVALUATED_ON` is a derivation closure. No extracted prompt template is needed for any of those.

## 3. Scope discipline

Every `extracted` entity and relation type MUST satisfy two gates before being added to the vocabulary:

1. **Eval-set gate:** at least one category of held-out question (1-hop, 2-hop, or 3+) becomes expressible as an `evidence_path` traversal because this type exists. Adding a type that no question would ever traverse is a defect.
2. **Extraction gate:** at least one feasible extraction prompt template (per #7) can identify instances of this type from paper text, with measured precision/recall that meets the implementing pipeline's quality bar.

`structural` types satisfy the eval-set gate by construction (they exist because the canonical artifact needs them) and need no extraction gate. `derived` types satisfy the eval-set gate by being a useful closure, and need no extraction gate because they are not LLM-extracted.

## 4. File schema: `ontology.yaml`

Top-level keys:

```yaml
version: 1
schema_version: 1

entity_types:
  - name: <snake_case string>
    definition: <one-sentence definition>
    population_mode: structural | extracted | derived
    # Required iff population_mode = extracted:
    extraction_prompt: <named prompt template>
    # Required iff population_mode = structural:
    source: <deterministic source or procedure>
    # Required iff population_mode = derived:
    derivation: <derivation rule, expressed as a small graph query or pseudocode>
    examples:
      - status: verified | pending_verification
        instance: <example instance identifier or name>
        note: <optional, short context for why pending>

relation_types:
  - name: <SCREAMING_SNAKE_CASE string>
    from: <entity type name>
    to: <entity type name>
    definition: <one-sentence definition>
    population_mode: structural | extracted | derived
    extraction_prompt: <named prompt template>      # if extracted
    source: <deterministic source or procedure>     # if structural
    derivation: <derivation rule>                   # if derived
    examples:
      - status: verified | pending_verification
        from: <example source instance>
        to: <example target instance>
        note: <optional, short context for why pending>
```

### 4.1 Field rules

- `version` is the **ontology version** (the vocabulary identity). Bumped only when types are added, removed, renamed, or have endpoints or population modes changed. Bumped even if examples change (well, no — examples don't bump the version; only structural changes do).
- `schema_version` is the **file-shape version**. Bumped when field names or required-fields change.
- `entity_types[].name` and `relation_types[].name` are unique within their list.
- `relation_types[].from` and `.to` are valid `entity_types[].name` values.
- Each type's mode-specific field is present iff the type's `population_mode` matches. The validation tool enforces this.
- `examples` is non-empty for every type.
- Each example carries a `status` of `verified` or `pending_verification`. A `verified` example is one that the author has confirmed appears in the corpus (paper is in `data/manifest.jsonl`, method/dataset/metric/task is named in that paper's text). A `pending_verification` example is a placeholder the author intends to confirm against the corpus. The validation tool refuses `verified` examples that don't actually exist in the corpus; it allows `pending_verification` examples with a `note` explaining what's pending.

### 4.2 Author attribution

`ontology.yaml` may include a top-level `notes` field (free-form string) for the author's rationale comments. The validation tool ignores this field.

### 4.3 Authority rule

Vocabulary, endpoints, and population modes are defined **only** in `ontology.yaml`. Code (#7 extraction, #9 ingestion, #11 retrieval) reads these properties from the file at startup. Duplicating any of these in Python is a defect; the validation tooling can lint for it by reading the file.

## 5. Validation rules (implementing authoring tool)

The implementing authoring tool enforces, before writing `ontology.yaml`:

1. **Schema conformance** against §4. Type checks, required-field presence, mode-specific field presence.
2. **Size gates:** 5 ≤ `len(entity_types)` ≤ 10; 8 ≤ `len(relation_types)` ≤ 15. Refuses to write outside these ranges.
3. **Unique names** within each list.
4. **Endpoint conformance:** every `relation_types[].from` and `.to` is a valid entity type name.
5. **Population-mode field presence:** every entity and relation type has exactly one mode-specific field (extraction_prompt if extracted, source if structural, derivation if derived). The mode must match the mode-specific field.
6. **Examples non-empty** for every type.
7. **Author exclusion:** no entity type named `author` (case-insensitive).
8. **Junk-drawer exclusion:** `concept`, `finding`, `claim`, `benchmark`, `topic`, `theme` are forbidden at the type-name level (case-insensitive).
9. **Verified-example corpus check:** every `examples[].status == "verified"` entry must point to corpus data that actually exists. For `document` examples, the openalex_id must be in `data/manifest.jsonl`. For other types, the example's instance name must appear in at least one paper's title or abstract (string match is sufficient for validation; semantic correctness is the implementing extraction ticket's responsibility).
10. **Pending-verification marker:** every `examples[].status == "pending_verification"` entry MUST carry a non-empty `note` explaining what needs verifying. The authoring tool surfaces these as a to-do list at write time.

The tool also runs an "appears in extraction prompt template" cross-check against the extraction contract (#7): every `extracted` type's `extraction_prompt` value must reference a prompt template that exists in the extraction pipeline. Per #7 there are exactly two grouped templates — `extract_entities_v1` and `extract_relations_v1` — and every `extracted` entity type references the former; every `extracted` relation type references the latter. The authoring tool reads the two template files and verifies the references resolve.

## 6. Acceptance gate

This spec closes when committed to `docs/specs/ontology.md`. The implementing authoring ticket closes when:

1. `ontology.yaml` conforms to §4.
2. Every `extracted` type satisfies the eval-set gate and the extraction gate (§3).
3. Every `structural` type has a `source` that names a canonical artifact or procedure that exists.
4. Every `derived` type has a `derivation` that is implementable as a deterministic graph query or pseudocode.
5. Every type's `examples` are drawn from the corpus, marked `verified`, with the corpus-instance verification check (§5.9) passing. Pending-verification entries are allowed only with a `note` and must be resolved before the file ships as the canonical benchmark input.
6. The 5 ≤ entity_types ≤ 10 and 8 ≤ relation_types ≤ 15 size constraints hold.
7. The validation tool's checks (§5) pass on the file.

## 7. Downstream coupling

- **#7 extraction contract** reads `entity_types[]` (filtered to `population_mode == extracted`) and `relation_types[]` (filtered similarly) to render the two grouped prompt templates (`extract_entities_v1` and `extract_relations_v1`). The templates are grouped — one template per phase, not one per type — so each chunk is sent through the LLM exactly twice. The per-type allowed-name lists and the per-type `verified` examples are rendered into the templates at run time from this file; an ontology edit is the only change needed to keep the templates honest.
- **#8 entity resolution** may use `entity_types[].examples` (verified) when curating a separate human-labelled development fixture for threshold evaluation. Production resolution clusters only validated mentions from extraction-complete chunks; ontology examples are never production seeds, aliases, or synthetic mentions.
- **#9 Neo4j ingestion** reads the relation types as the edge vocabulary; Cypher constraints and uniqueness rules derive from `relation_types[].from`, `.to`, and `.population_mode`. `derived` relations are computed at ingestion (or at query time, depending on the implementing choice); they are never LLM-extracted.
- **#11 graph retrieval** uses the relation types as the traversal relation-filter. Cypher queries are parameterised on the ontology.
- **#12 hybrid retrieval** consults the ontology's entity-type names for routing decisions ("is the query about a `method`?").
- **#4 evaluation schema** uses the entity-type names as the namespace for `e_<id>` references in evidence paths.

The ontology is therefore a single load-bearing artifact. The vocabulary-change rule (§1.4) is what protects the benchmark's reproducibility.

---

This spec resolves ticket #5. The implementing authoring ticket populates `ontology.yaml` with verified corpus examples against §4, applies the population-mode-specific fields per §2, and ships the validation tool that enforces §5.
