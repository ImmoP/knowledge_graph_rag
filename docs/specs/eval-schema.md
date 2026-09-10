# Evaluation Set Schema Specification

Resolution of [#4 — Decide the evaluation-set schema and authoring protocol](https://github.com/ImmoP/knowledge_graph_rag/issues/4). This document is the **schema and rules** the eventual eval set conforms to. It is not the 45 questions themselves — those are authored post-Wayfinder, in an implementing ticket produced by `/to-tickets` once the chunk manifest exists and the ontology (#5) is known.

The split between schema (Wayfinder) and questions (post-Wayfinder) is structural: the schema fixes the shape every downstream ticket (retrieval, generation, benchmark runner) consumes; the questions depend on (a) concrete `chunk_id` values from a chunker that does not yet exist and (b) concrete entity/relation types from an ontology that does not yet exist. The schema is what Wayfinder can produce today; the questions are what the implementing pipeline produces once those upstream artifacts land.

## 1. Dataset size and bucket balance

- **Total:** 45 questions.
- **Buckets:** three hop-count buckets — `1`, `2`, `3+`.
- **Per-bucket target:** ~15 each (45 / 3 = 15).
- **Acceptable range:** 12–18 per bucket. Outside this range is permitted but flagged at authoring time as a balance warning.
- **Distribution enforcement:** the implementing authoring tool records the bucket counts and refuses to write the JSONL file unless 45 valid records exist across the three buckets.

The `3+` bucket accommodates 3, 4, or more graph relationship traversals. The corpus is small so 4+ is rare; the `+` keeps the schema open without forcing an additional bucket.

## 2. Operational hop-count definition

**Hop count = the minimum number of graph-relationship traversals required to derive the answer from the annotated `evidence_path`.**

This is the single, pinned definition. Other intuitive notions are explicitly rejected:

- NOT the number of papers mentioned in the question text.
- NOT the number of chunks cited in `supporting_chunk_ids`.
- NOT a heuristic about how complex the question "feels" to a reader.

Every consumer of the eval set (authoring tool, benchmark runner, scoring engine) uses this definition. A question whose `evidence_path` has `n` relation steps is `hop_bucket = "n"` for `n ∈ {1, 2}`, or `"3+"` for `n ≥ 3`.

The `evidence_path` MUST NOT contain degenerate "chunk-only" steps with no relation traversal; every step is one directed traversal record. A 1-hop question has exactly one such record; the smallest legal `evidence_path` length is therefore 1.

## 3. JSONL record schema

Each line of `eval/questions.jsonl` (and future `eval/questions.vN.jsonl` files) is a JSON object with these fields:

### 3.1 Required fields

| Field                | Type           | Description                                                                                                       |
|----------------------|----------------|-------------------------------------------------------------------------------------------------------------------|
| `qid`                | string         | Stable identifier, format `q_NNN` where NNN is zero-padded to three digits (e.g. `q_001`, `q_045`). Unique within a version file. |
| `question`           | string         | Natural-language question as presented to the retriever / user.                                                  |
| `expected_answer`    | string         | Canonical free-text answer. The benchmark scorer treats substring matches against this or against `answer_aliases` (if present) as hits. |
| `hop_bucket`         | string         | One of `"1"`, `"2"`, `"3+"`.                                                                                      |
| `evidence_path`      | list           | Ordered list of evidence steps (see §4). Each step is a relation traversal. The list length equals the hop count for `hop_bucket ∈ {"1", "2"}` and is ≥3 for `hop_bucket = "3+"`. |
| `supporting_doc_ids` | list of string | Manifest `openalex_id` values (`W[0-9]+`) covering every document the answer depends on. Non-empty.              |
| `supporting_chunk_ids` | list of string | `chunk_id` values (per chunk-record spec: `^c_[0-9a-f]{16}$`) covering every chunk that asserts a claim required by the answer. May be `[]` during initial authoring (before the chunk manifest exists); MUST be non-empty by the time the file ships. |
| `split`              | string         | One of `"held_out"` (default), `"dev"`, `"train"`. See §6.                                                       |

### 3.2 Optional fields

| Field             | Type           | Description                                                                                                       |
|-------------------|----------------|-------------------------------------------------------------------------------------------------------------------|
| `answer_aliases`  | list of string | Surface-form variants of `expected_answer` that the scorer also treats as hits (e.g. `"Adam"`, `"the Adam optimizer"`). |
| `notes`           | string         | Author's free-form notes (rationale, authoring difficulty, scoring caveats). The benchmark runner ignores this field. |

### 3.3 Field rules

- `qid` is required and unique within a version file. Re-numbering across versions is permitted; benchmark reports record `qid` together with the dataset version they came from.
- `expected_answer` is a single string, not a list. Multi-part answers use `answer_aliases` for variants and use plain prose for the canonical form.
- `supporting_doc_ids` is non-empty. A question that can be answered without any document is not a benchmark question.
- `supporting_chunk_ids` is non-empty by ship time. During initial authoring (before the chunk manifest exists) the author may write `[]` and the authoring tool accepts it; once the chunk manifest exists, the tool refuses to ship a file with `[]` in any record.

## 4. Evidence-path representation

Each `evidence_path` entry is a JSON object:

```json
{"step": 1, "from": "<id>", "relation": "<relation_type>", "to": "<id>", "direction": "forward"}
```

Field-by-field:

- `step` — int, 1-based, monotonically increasing.
- `from`, `to` — string identifiers in one of three namespaces:
  - `W[0-9]+` — Document (manifest `openalex_id`).
  - `^e_[0-9a-f]{16}$` — Entity node, using the canonical entity-ID shape pinned by #8 and projected by #9.
  - `c_[0-9a-f]{16}` — Chunk (per chunk-record spec).
- `relation` — string, a relation type from the ontology (per #5). The benchmark runner rejects any `relation` value not in the ontology.
- `direction` — string, exactly `"forward"` or `"reverse"`. `forward` means traversal from `from` to `to` follows the stored ontology edge; `reverse` means traversal from `from` to `to` goes opposite the stored ontology edge. `relation` retains the canonical ontology name in both cases.

`from` and `to` are always traversal-order node IDs. Every record consumes exactly one semantic ontology hop regardless of `direction`. Infrastructure relationships `HAS_CHUNK` and `MENTIONS` never appear in `evidence_path` and consume no semantic hops.

**Path connectivity rule:** for any step `n ≥ 2`, the step's `from` MUST equal step `n-1`'s `to`. The path is a connected walk over the graph. A disconnected step is a defect the authoring tool surfaces, not silently accepts.

**Reuse rule:** `supporting_doc_ids` ⊇ {all `W[0-9]+` values appearing in `evidence_path`}. The author does not need to repeat them in `supporting_doc_ids` for path-validity purposes, but every document mentioned anywhere in the question must appear in `supporting_doc_ids`.

## 5. Authoring protocol

The eval set is human-authored. No LLM-generated questions. The author's role is to write the 45 records. The agent's role is to provide tooling that catches errors before the file ships.

### 5.1 Authoring tool responsibilities

The implementing ticket (post-Wayfinder) ships a small CLI that:

1. **Schema-validates** every record against §3. Type checks, regex checks on IDs, required-field presence.
2. **Bucket-balances:** counts `hop_bucket` distribution and warns (or refuses, in strict mode) outside the 12–18 range.
3. **Reference-validates:**
   - Every `supporting_doc_ids` value appears in `data/manifest.jsonl`.
   - Every `supporting_chunk_ids` value appears in `data/derived/chunks.jsonl` (when the chunk manifest exists).
   - Every `evidence_path.from` and `evidence_path.to` is a valid ID in one of the three namespaces and (for `W` and `e` types) exists in the manifest / graph.
4. **Path-validates:**
   - Steps are connected (per §4).
   - The `relation` value of every step appears in the ontology (per #5).
   - Every step's `direction` is `forward` or `reverse`, and the graph contains the named ontology edge in the indicated stored direction.
   - `evidence_path` length matches `hop_bucket` per §2.
5. **Writes** `eval/questions.jsonl` only if all checks pass in strict mode. The tool reports warnings otherwise.

### 5.2 Author workflow

1. Read the schema (this spec) and the operational hop-count definition (§2).
2. For each of 45 questions:
   - Pick a target `hop_bucket`.
   - Pick a real corpus question whose answer requires that hop count.
   - Read the relevant paper(s) to identify the supporting documents and the expected graph path.
   - Fill in `qid`, `question`, `expected_answer`, `hop_bucket`, `evidence_path`, `supporting_doc_ids`, and — if the chunk manifest already exists — `supporting_chunk_ids`.
3. Run the authoring tool after every batch of new records. Fix errors before adding more.
4. When 45 valid records exist and all checks pass, the tool writes `eval/questions.jsonl` (or `eval/questions.vN.jsonl`).

### 5.3 Worked example

A 2-hop question starting from a method and traversing through the document that proposed it might look like:

```json
{
  "qid": "q_017",
  "question": "Which dataset did the ColBERT paper evaluate on?",
  "expected_answer": "MS MARCO",
  "hop_bucket": "2",
  "evidence_path": [
    {"step": 1, "from": "e_1111111111111111", "relation": "PROPOSES", "to": "W3021397474", "direction": "reverse"},
    {"step": 2, "from": "W3021397474", "relation": "EVALUATES_ON", "to": "e_2222222222222222", "direction": "forward"}
  ],
  "supporting_doc_ids": ["W3021397474"],
  "supporting_chunk_ids": [],
  "split": "held_out",
  "notes": "Authored before the chunk manifest existed; chunk_ids to be filled in once the chunker ships."
}
```

The Entity IDs are illustrative placeholders for the resolved ColBERT and MS MARCO entities; the authoring tool validates them against the real graph at run time. The example's relation names, endpoint types, traversal directions, and Document ID follow ontology v1 and the canonical manifest.

## 6. Held-out policy (no split — selected operating policy)

**Project operating policy (selected):** the canonical eval set is **not split**. All 45 questions in `eval/questions.jsonl` are `split: "held_out"`. None of them may be used for tuning.

- Every record has `split: "held_out"`. The `split` field remains in the schema (because the schema is general) but for this project every shipped record uses the `held_out` value.
- The benchmark runner treats `held_out` records as the canonical evaluation set and refuses to expose them to retrieval-tuning loops.
- **Tuning is forbidden on these 45 questions.** Retrieval parameters — chunk sizes, embedding-model choices, hybrid-router rules, generation prompts — must be finalized *before* the benchmark numbers are recorded. If a measured run produces a number that depends on a parameter tuned on the held-out set, the run is invalid and the parameter must be re-tuned on out-of-band data (see below).
- **Development questions, if needed, do not live among the 45.** A separate dev artifact (e.g. `eval/dev_questions.jsonl` or any file outside the canonical benchmark) may be authored at any time for router-rule tuning or retrieval-shape exploration. Dev questions are not part of the benchmark and are not counted in the 45. The benchmark runner does not read them by default.
- **No silent re-tagging.** A record's `split` field is part of its schema. Promoting a `held_out` record to a different status is forbidden in any version file that is also used as benchmark input. (Dev questions are a separate file, not a re-tagged record.)

This is a deliberate trade-off: the user accepts fewer iterations on the retrieval pipeline before the benchmark numbers are recorded, in exchange for a strictly-clean held-out benchmark. Out-of-band tuning uses any of (a) the user's judgment, (b) a small dev artifact kept out of `eval/questions.jsonl`, or (c) qualitative inspection of representative queries. None of those touch the 45.

## 7. Freeze and versioning policy

The dataset is **fixed before retrieval tuning**. This rule is structural, not stylistic:

- The first `eval/questions.jsonl` that the authoring tool produces is the canonical frozen set.
- Retrieval-tuning tickets (#10, #11, #12) reference this frozen set as a *consumed* artifact, not as a mutable input.
- If new questions need to be added (e.g. after running the benchmark reveals a gap), the additions go to `eval/questions.v2.jsonl`. The original `eval/questions.jsonl` is unchanged.
- **Never** silently merge new questions into the canonical file. The benchmark's reproducibility depends on this.

**File naming convention:** `eval/questions.jsonl` is the current canonical file. Older versions are kept as `eval/questions.v1.jsonl`, `eval/questions.v2.jsonl`, etc. The benchmark runner takes a `--questions-file` argument (default `eval/questions.jsonl`) and records:

- the file path
- the file's SHA256
- the number of records
- the bucket distribution
- the per-split count (held_out / dev / train)

in `report.json`. These metadata are part of the benchmark record; comparing numbers across runs is only valid when the recorded metadata match.

Because the project elects no split (§6), the per-split count is always `(held_out: 45, dev: 0, train: 0)` for the canonical benchmark file. The runner records it anyway because the schema is general; future projects or forks that reintroduce splits will get correct counts without further config.

## 8. What this spec leaves open (post-Wayfinder / implementing)

These are explicitly not part of the schema decision:

- **The actual 45 questions.** Authored by the user, post-Wayfinder, once the chunk manifest exists and the ontology (#5) is known.
- **The exact cited chunk_ids.** Filled in at authoring time against the real chunk manifest.
- **The benchmark runner's scoring math.** That is [#14](https://github.com/ImmoP/knowledge_graph_rag/issues/14)'s decision. This spec only fixes the *input* shape.
- **Authoring tool implementation details.** The CLI command name, output format, error message wording, exact JSON-schema validator library. The implementing ticket picks these.
- **LLM-judge as a scorer.** If substring / alias matching is insufficient and an LLM judge is later added, the schema does not change (free-text `expected_answer` accommodates both). The judge's behavior is part of #14.

## 9. Acceptance gate

This spec closes when committed to `docs/specs/eval-schema.md`. The implementing authoring ticket closes when:

1. 45 records exist across the three buckets with 12–18 per bucket.
2. The authoring tool's strict-mode checks (§5.1) pass on every record.
3. Every `supporting_doc_ids` value exists in `data/manifest.jsonl`.
4. Every `supporting_chunk_ids` value exists in `data/derived/chunks.jsonl`.
5. `eval/questions.jsonl` is committed with its SHA256 recorded alongside.

The first three checks gate the schema's *correctness*; the last two gate its *completeness* (chunks are filled in only after the chunker ships).

---

This spec resolves ticket #4. It is referenced by #14 (benchmark runner consumes `eval/questions.jsonl` against this schema), by #12 (hybrid retrieval may tune its router rule on `split: "dev"` records but not on `held_out`), and by the post-Wayfinder implementing ticket that authors the actual 45 records.

The five-field evidence-step shape is a representational consistency repair made while resolving #11. It keeps #4 closed and does not change benchmark size, split policy, hop-count semantics, or ontology vocabulary. It makes reverse semantic traversal explicit while preserving connected traversal-order paths.
