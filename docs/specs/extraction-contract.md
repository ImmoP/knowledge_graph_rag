# Extraction Pipeline Contract

Resolution of [#7 — Decide the entity and relation extraction pipeline contract](https://github.com/ImmoP/knowledge_graph_rag/issues/7). This document is the **contract** extraction must satisfy, not the implementing code. The implementing ticket goes to `/to-tickets` once the map is cleared.

This spec assumes the [`chunk-record` spec](chunk-record.md), the [`chunking-strategy` spec](chunking-strategy.md), the [`ontology` spec](ontology.md), and the [`provider-abstractions` spec](provider-abstractions.md) are in force. It does not redefine any of them; it pins only what extraction must do on top of them.

## 1. Role in the pipeline

Extraction is the stage that turns chunked text into structured records on disk. It is the **only** stage that calls the `LLMProvider` for text generation. Its inputs are canonical artifacts produced upstream; its outputs feed downstream stages that do not call the LLM at all (canonical entity resolution lives in [#8](https://github.com/ImmoP/knowledge_graph_rag/issues/8); Neo4j ingestion in [#9](https://github.com/ImmoP/knowledge_graph_rag/issues/9)).

```
chunks.jsonl  ──┐
                ├──►  EXTRACTION (this spec)  ──►  extractions.jsonl  ──►  #8 / #9
ontology.yaml  ──┘
```

The extraction pipeline is the seam between "unstructured paper text" and "structured facts the rest of the pipeline reasons over." Everything downstream consumes only its validated output.

## 2. Inputs

| Input | Source | What extraction reads from it |
|---|---|---|
| Chunk records | `data/derived/chunks.jsonl` (per [#2](https://github.com/ImmoP/knowledge_graph_rag/issues/2)) | `chunk_id`, `doc_id`, `section_path`, `chunk_index`, `text`, `token_estimate` |
| Ontology | `ontology.yaml` (per [#5](https://github.com/ImmoP/knowledge_graph_rag/issues/5)) | `entity_types[]` filtered to `population_mode == extracted`; `relation_types[]` filtered the same way; per-type `examples` (the `verified` subset) for few-shot exemplars |
| Provider config | environment (per [#16](https://github.com/ImmoP/knowledge_graph_rag/issues/16)) | `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BUDGET_USD`, `LLM_BUDGET_TOKENS`, `LLM_CACHE_DIR`, `LLM_PROVIDER_TIMEOUT_S` |
| Prompt template text | `prompts/extract_entities_v1.txt` and `prompts/extract_relations_v1.txt` (this spec) | the verbatim text the pipeline sends to the LLM |

Extraction does **not** read `data/raw/`, the embeddings table, or any downstream artifact.

## 3. Output: `data/extractions.jsonl`

- **Path:** `data/extractions.jsonl`. Gitignored (per `.gitignore`; `data/derived/` is gitignored, and this artifact lives in the same directory). Reproducibility is provided by determinism + the SHA256 record in the benchmark report.
- **Format:** JSONL, one record per line, UTF-8, trailing newline.
- **Ordering:** records are sorted by `(chunk_id, kind, mention_id or assertion_id)` lexicographically. Sorting is the implementing pipeline's responsibility; the file as written is sorted.
- **Sibling manifest:** `data/extractions.manifest.json` records the run identity: provider, model, ontology version + SHA256, prompt template versions + SHA256, schema version, chunk-manifest SHA256, wall-clock timestamp. The JSONL is content; the manifest is metadata.

### 3.1 Record shapes

A line is one of three record kinds, distinguished by a `kind` field. The pipeline writes a `mention` record for each surfaced entity instance, an `assertion` record for each surfaced relation instance, and a `chunk_status` record once per chunk after both phases finish.

#### 3.1.1 `mention` record (entity pass output)

```json
{
  "kind": "mention",
  "mention_id": "m_3f9a0b2c8d1e4f7a",
  "chunk_id": "c_a1b2c3d4e5f60718",
  "doc_id": "W3015883388",
  "entity_type": "method",
  "surface_form": "Dense Passage Retrieval",
  "canonical_form": "DPR",
  "evidence": {
    "char_start": 1024,
    "char_end": 1051,
    "source_text_sha256": "..."
  },
  "confidence": "stated"
}
```

Field rules:

- `mention_id` is `m_` + the first 16 hex characters of `sha256` over a canonical JSON payload (see §6). It is content-addressed, deterministic, and assigned by pipeline code after validation. The 16-hex prefix is the full ID; there is no chunk suffix, run suffix, or sequence number.
- `entity_type` must be one of `method`, `dataset`, `metric`, `task` — the four `extracted` entity types in `ontology.yaml` v1.
- `surface_form` is the literal text span the LLM cited as evidence.
- `canonical_form` is the LLM's normalised name (e.g. `"DPR"` for `"Dense Passage Retrieval"`); canonical resolution to a graph-node key is #8's job.
- `evidence.char_start` / `char_end` are 0-based character offsets into the canonical chunk `text` (`char_start` inclusive, `char_end` exclusive). Required for the benchmark runner to cite evidence paths.
- `evidence.source_text_sha256` is `sha256(chunk.text)`. Required to detect parser-driven text drift between runs.
- `confidence` is `"stated"`, `"inferred"`, or `"hedged"`, reflecting the LLM's hedged-certainty classification of its own mention. Extraction does not interpret this further; downstream stages may.

#### 3.1.2 `assertion` record (relation pass output)

```json
{
  "kind": "assertion",
  "assertion_id": "a_7c1d8e5b2a09f4e6",
  "chunk_id": "c_a1b2c3d4e5f60718",
  "doc_id": "W3015883388",
  "relation_type": "PROPOSES",
  "subject_ref": {"kind": "document", "id": "W3015883388"},
  "object_ref": {"kind": "mention", "id": "m_b2e7f4a190c8d356"},
  "evidence": {
    "char_start": 2048,
    "char_end": 2401,
    "source_text_sha256": "..."
  },
  "confidence": "stated"
}
```

Field rules:

- `assertion_id` is `a_` + the first 16 hex characters of `sha256` over a canonical JSON payload (see §6). It is content-addressed, deterministic, and assigned by pipeline code after validation.
- `relation_type` must be one of the eight `extracted` relation types in `ontology.yaml` v1 (`PROPOSES`, `INTRODUCES`, `EVALUATES_ON`, `USES_METHOD`, `REPORTS_METRIC`, `EXTENDS`, `USED_FOR_TASK`, `MEASURES`).
- `subject_ref` and `object_ref` are typed endpoint references. Each is exactly `{"kind":"document","id":"<doc_id>"}` or `{"kind":"mention","id":"<mention_id>"}`; no other keys or endpoint kinds are allowed.
- Endpoint kind is derived from the relation endpoint type in `ontology.yaml`. In ontology v1, a structural `document` endpoint must use `kind: "document"` and its `id` must equal the current chunk's `doc_id` byte-for-byte. An extracted endpoint must use `kind: "mention"`, reference a validated mention from the same chunk, and match the endpoint's ontology entity type.
- The LLM never invents structural document identity. Pipeline code injects the current `chunk.doc_id` or validates an emitted document ref against it. **Cross-chunk mention references and cross-document inference are forbidden in v1** (see §5.3).
- `evidence` carries the asserted relation's text span and the chunk-text SHA256, exactly as for mentions.
- `confidence` mirrors the mention-record convention.

#### 3.1.3 `chunk_status` record (audit)

```json
{
  "kind": "chunk_status",
  "chunk_id": "c_a1b2c3d4e5f60718",
  "doc_id": "W3015883388",
  "phases": {
    "entity":   {"status": "validated", "attempt_count": 1, "mention_count":   4, "failure_count": 0},
    "relation": {"status": "validated", "attempt_count": 1, "assertion_count": 2, "failure_count": 0}
  },
  "extraction_complete": true
}
```

A `chunk_status` record is written **once per chunk** after both phases finish for that chunk. It is the auditable record of "this chunk is extraction-complete" (see §9). Resumability (§13) reads from these records.

`phases.<entity|relation>.failure_count` (additive, repair #23) is the number of candidate records rejected at Gate 2 (ontology conformity) or Gate 3 (evidence integrity) for that phase. A phase-level Gate 1 (schema) failure is reflected by `status="rejected"` and is **not** added to `failure_count`, because Gate 1 fails the entire phase rather than a per-record reject. `status="validated"` means the phase call ran to completion; it may still carry `failure_count > 0`, in which case the chunk is **not** extraction-complete per §9 rule 3.

### 3.2 Forbidden fields

- No `provider`, `model`, or `cost_usd` on individual records. Provider identity is recorded once in the run manifest; per-record provider fields would bloat the file and complicate re-runs against a different provider.
- No timestamps on individual records. The run manifest records the run timestamp; per-record timestamps would break deterministic sorting.
- No internal IDs not listed above (no UUIDs). The `m_*` and `a_*` IDs are content-addressed hash prefixes (§6), not UUIDs. They are deterministic and human-inspectable: anyone can reproduce them from the public inputs (chunk identity, evidence offsets, ontology type).

## 4. Provider boundary

Extraction calls the LLM exclusively through `LLMProvider.generate_json` (per [#16](https://github.com/ImmoP/knowledge_graph_rag/issues/16) §1 and §2). It does not import any provider SDK.

- **No direct provider SDK imports** appear in the extraction module. This is verifiable with `grep -rE "import (anthropic|openai|ollama)" src/extraction/` returning empty.
- All schema-conformant JSON parsing happens inside the adapter. Extraction code receives a parsed dict (already schema-validated) or an `LLMValidationError`.
- `Usage` is the only cross-call accounting signal the extraction wrapper consumes.

### 4.1 Default provider and model

This spec **does not pin a default provider or model**. The default is whichever values `LLM_PROVIDER` and `LLM_MODEL` carry at first-run time, per [#16](https://github.com/ImmoP/knowledge_graph_rag/issues/16) §7. This is the same "provider is a deployment concern, not an architecture concern" rule that the provider spec applied to vector retrieval.

What the implementing ticket must ship:

1. One working `LLMProvider` adapter already shipped (per #16's acceptance gate).
2. An `extract.py` entry point that calls `make_llm()` (per #16 §4), constructs the two prompt templates, and produces `data/extractions.jsonl` and `data/extractions.manifest.json`.
3. `.env.example` updated to document the recommended provider choices for extraction. Anthropic + `claude-haiku-4-5` is a reasonable first pick for cost/speed; a local Ollama model is a reasonable first pick for cost-zero reproducibility. Both are interchangeable — that is the architecture property under test.

### 4.2 Default generation parameters

For both prompts:

| Parameter | Value | Why |
|---|---|---|
| `temperature` | `0.0` | Extraction is meant to be reproducible across runs; non-zero temperature would invalidate the cache contract and inject noise into evidence offsets. |
| `max_tokens` | `2048` | Sized to accommodate a typical chunk with several mentions and a few relations as JSON. The implementing ticket pins a hard ceiling and refuses inputs whose validated output would exceed it. |

These are the *default* parameters; `.env` may override them per-call. The cache key (§10) mixes the actual values into the hash, so changing them produces a new cache namespace — the cache contract tolerates parameter changes without contamination.

## 5. Prompt template strategy

There are exactly **two named prompt templates**, and they are **grouped**, not per-type:

| Template file | Phase | Output |
|---|---|---|
| `prompts/extract_entities_v1.txt` | entity pass | a JSON object holding all `extracted` entity types surfaced in the chunk |
| `prompts/extract_relations_v1.txt` | relation pass | a JSON object holding all `extracted` relation types surfaced in the chunk, with typed document/mention endpoint references |

The grouped design is deliberate: it sends each chunk through the LLM exactly twice (once per phase), not 4 + 8 = 12 times. The cost saving matters because the cache contract (§10) and the budget cap (§11) both interact with per-call cost.

### 5.1 Template file location and versioning

- **Location:** `prompts/` at the repo root (a directory the implementing ticket creates; not currently present).
- **Naming:** the template filename's `v1` suffix is the **template version**. Any edit to the template text bumps the suffix (`v1` → `v2`). Bumps are the unit of cache invalidation; the cache key (§10) hashes the file content, so unintentional edits would change the key automatically — but the version suffix is the human-visible signal of intentional change.
- **Immutability:** committed template text is frozen. Edit history is git; the live text is whatever HEAD resolves to. The cache key hashes the file content so that any non-empty diff invalidates derived cache entries.
- **No inlining in code.** Templates are read from disk at run time; the Python code does not contain the prompt text. This keeps prompts diffable, reviewable, and editable by people who do not read Python.

### 5.2 Per-type schema inside the entity-pass template

`extract_entities_v1.txt` instructs the LLM to emit a JSON object whose top-level shape is:

```json
{
  "mentions": [
    {"entity_type": "<one of method|dataset|metric|task>",
     "surface_form": "...",
     "canonical_form": "...",
     "char_start": <int>,
     "char_end": <int>,
     "confidence": "stated|inferred|hedged"}
  ]
}
```

`entity_type` is constrained to **exactly** the four `extracted` names from `ontology.yaml` (the implementing ticket reads `ontology.yaml` at startup and validates the LLM output against that list). The prompt's instruction text names them by their canonical names (`method`, `dataset`, `metric`, `task`); the implementing ticket renders the list from the ontology file rather than hard-coding it, so a curated ontology change is the only edit needed to keep the prompt honest. Examples in the prompt come from `ontology.yaml`'s per-type `examples[].status == "verified"` set.

### 5.3 Per-type schema inside the relation-pass template

`extract_relations_v1.txt` instructs the LLM to emit:

```json
{
  "assertions": [
    {"relation_type": "<one of PROPOSES|INTRODUCES|EVALUATES_ON|USES_METHOD|REPORTS_METRIC|EXTENDS|USED_FOR_TASK|MEASURES>",
     "subject_ref": {"kind": "document|mention", "id": "<current doc_id or supplied mention_id>"},
     "object_ref": {"kind": "document|mention", "id": "<current doc_id or supplied mention_id>"},
     "char_start": <int>,
     "char_end": <int>,
     "confidence": "stated|inferred|hedged"}
  ]
}
```

The relation prompt receives the current `chunk.doc_id` and the validated same-chunk mention records after their `m_*` IDs have been assigned. It instructs the LLM that:

- It may assert only ontology-defined relations grounded in the current chunk.
- For an endpoint whose ontology entity type is `population_mode: extracted`, the ref must be `kind: "mention"`; `id` must be one of the supplied validated same-chunk mention IDs, and that mention's `entity_type` must match the relation endpoint type.
- For an endpoint whose ontology entity type is the structural `document` type, the ref must be `kind: "document"`; the only permitted `id` is the supplied current `chunk.doc_id`. Pipeline code injects that known value or validates it byte-for-byte rather than trusting model-generated identity.
- **Cross-chunk mention references and cross-document inference are forbidden in v1.** If a required extracted entity is not a supplied same-chunk mention, the LLM must omit the assertion.
- **Span selection obligation (§5.3, repair #23).** The relation prompt must explicitly instruct the model that the cited span `chunk.text[char_start:char_end]` must contain or directly express the asserted relation. The model is the authority on span selection; the validator only checks span existence and source-text SHA (per §7.1). No additional mechanical check (no embeddings, no similarity threshold, no LLM judge, no regex semantic heuristic, no extra LLM call) is applied to the span.

This makes all eight extracted v1 relations representable: the five `document → extracted` relations (`PROPOSES`, `INTRODUCES`, `EVALUATES_ON`, `USES_METHOD`, `REPORTS_METRIC`) use a document source ref and mention target ref; `EXTENDS`, `USED_FOR_TASK`, and `MEASURES` use mention refs at both endpoints. Structural `CITES` is not emitted by this prompt, and derived `EVALUATED_ON` is not independently extracted.

### 5.4 Schema enforcement

`generate_json` enforces schema validity per its one validation retry (per [#16](https://github.com/ImmoP/knowledge_graph_rag/issues/16) §2.2 and §2.3). The implementing ticket passes the per-template JSON Schema to `generate_json`. Schemas are generated from `ontology.yaml` at startup, not maintained separately.

## 6. Mention and assertion ID generation

Mention IDs (`m_*`) and assertion IDs (`a_*`) are **generated by pipeline code, never by the LLM**. The LLM provides the mention list (entity pass) or typed endpoint references constrained to the supplied current-document and same-chunk mention IDs (relation pass); pipeline code validates/injects refs and assigns IDs after schema, ontology, and evidence validation succeeds.

IDs are **content-addressed**: each ID is a hash prefix over a canonical-JSON payload of the inputs that make the record unique. The same validated record always hashes to the same ID, regardless of when or in what order the pipeline emits it. These are pre-resolution IDs — #8 produces the canonical `e_*` entity IDs; this spec produces only the IDs that pin a particular surfaced mention or assertion to its evidence.

### 6.1 ID derivation

For a validated mention record:

```
mention_id = "m_" + sha256(canonical_mention_payload).hexdigest()[:16]
```

where `canonical_mention_payload` is the UTF-8 JSON encoding of an object containing exactly these fields, in this fixed shape:

```json
{
  "chunk_id":           "<chunk_id>",
  "source_text_sha256": "<source_text_sha256>",
  "entity_type":        "<entity_type>",
  "char_start":         <int>,
  "char_end":           <int>
}
```

For a validated assertion record:

```
assertion_id = "a_" + sha256(canonical_assertion_payload).hexdigest()[:16]
```

where `canonical_assertion_payload` is the UTF-8 JSON encoding of:

```json
{
  "chunk_id":           "<chunk_id>",
  "source_text_sha256": "<source_text_sha256>",
  "relation_type":      "<relation_type>",
  "subject_ref":        {"kind": "<document|mention>", "id": "<doc_id|mention_id>"},
  "object_ref":         {"kind": "<document|mention>", "id": "<doc_id|mention_id>"},
  "char_start":         <int>,
  "char_end":           <int>
}
```

Deterministic JSON serialisation is mandatory. The pipeline MUST serialise the payload with:

- `sort_keys=True` (field names sorted lexicographically; this also fixes their iteration order),
- `separators=(",", ":")` (no incidental whitespace),
- UTF-8 encoding (`ensure_ascii=False`), and
- integer fields serialised as JSON numbers (not strings).

No other fields, comments, or whitespace may appear in the payload. Each nested endpoint object contains exactly `kind` and `id` and is serialised under the same canonical-JSON rules. `source_text_sha256` is the 64-hex `sha256(chunk.text)` string (not the raw bytes); the same string format the record's `evidence.source_text_sha256` field carries. The 16-hex prefix is the full ID; there is no chunk suffix, run suffix, or sequence number.

### 6.2 Determinism invariants

The ID derivation produces identical IDs for the same validated record regardless of pipeline-level choices, so:

1. **Same extraction ⇒ same ID.** Two runs that validate the same mention (same `chunk_id`, `source_text_sha256`, `entity_type`, `char_start`, `char_end`) compute the same `mention_id`. Same for assertions.
2. **Clean full run ≡ resumed run.** Processing order (full-run ordering vs. resume-after-failure ordering) does not enter the payload, so the IDs are identical.
3. **Processing order cannot change IDs.** IDs are derived from the record's own fields, not from position in any stream or run.
4. **Failure in another chunk cannot renumber IDs.** No global counter, no chunk index, no "previous chunk count" enters the payload.
5. **Resumability does not change IDs.** A mention validated on the first run and an equivalent mention validated on a resumed run hash to the same ID.

These are the properties the benchmark runner's reproducibility metadata and the resumability gate (§13) depend on.

### 6.3 Why the LLM does not generate IDs

- LLM-generated IDs are non-deterministic in length and case even when asked for a specific shape.
- LLM-generated IDs would couple ID assignment to LLM output, defeating the cache contract's whole point: the cache lets two runs with identical inputs produce identical IDs even if the LLM is replaced.
- Content-addressed IDs make re-runs diff-clean: a re-run against the same provider produces a byte-identical `data/extractions.jsonl` for the records that validate identically, with no need to renumber between attempts.
- Content-addressed IDs also make the JSONL safe to merge or diff across branches / runs without an ID-remapping step: any record with the same `mention_id` or `assertion_id` is the same record.

### 6.4 Pre-resolution status

`m_*` and `a_*` IDs pin a mention or assertion to a specific evidence location; they do **not** identify the underlying entity or relation across mentions/assertions. Canonical entity resolution (which produces the `e_*` IDs) is #8's job. This spec deliberately does not collapse mentions across chunks or across surface forms — two mentions of `"DPR"` in the same chunk with different `char_start` get different `m_*` IDs by construction.

## 7. Canonical hashing and evidence offsets

Every evidence-bearing field is deterministic against the canonical chunk text.

- **`source_text_sha256`** is `sha256(chunk.text)` where `chunk.text` is the exact `text` field from `data/derived/chunks.jsonl`. The pipeline reads `text` directly; it never re-derives it from raw LaTeX.
- **`char_start` / `char_end`** are 0-based character offsets into `chunk.text` (`char_start` inclusive, `char_end` exclusive).

The pipeline must use the **exact** `text` string the chunk record carries. Any normalisation (whitespace collapsing, citation-marker rewriting, etc.) at extraction time would invalidate downstream evidence citations. The chunk record's `text` is canonical; extraction is a reader, not an editor.

### 7.1 Validation rules for evidence

For every mention and every assertion record:

1. `0 <= evidence.char_start < evidence.char_end <= len(chunk.text)` for the relevant chunk. Offsets outside this range fail validation.
2. For **mentions**, the substring `chunk.text[char_start:char_end]` after whitespace normalisation on both sides (collapse runs, trim leading/trailing whitespace) contains the `surface_form`. For **assertions**, the same normalised substring is non-empty. The mechanical evidence gate does not compare the assertion span against a "paraphrased form" of the relation, because the model-facing assertion shape (§3.1.2 / §5.3) carries no relation-text field. Whether the cited span directly hosts or expresses the asserted relation is the model's responsibility (§5.3, repair #23) and is not a post-condition of this gate. No embeddings, similarity threshold, LLM judge, regex semantic heuristic, or additional LLM call is introduced for assertion evidence. Whitespace normalisation is limited to collapsing runs of whitespace and trimming leading/trailing whitespace; no other edits.
3. `evidence.source_text_sha256 == sha256(chunk.text)` for the chunk in question. A mismatch fails validation; this is the drift detector.

## 8. Validation contract

Each phase's LLM output goes through three gates before it is written to `data/extractions.jsonl`. A failure at any gate sends the record to `data/extractions.rejects.jsonl` and increments the phase's failure counter (recorded in `chunk_status`); it does not crash the run.

### 8.1 Gate 1 — schema validity

Owned by `generate_json` (per [#16](https://github.com/ImmoP/knowledge_graph_rag/issues/16) §2.2). One retry on validation failure. If validation still fails after the retry, the adapter raises `LLMValidationError` and the pipeline routes the input to the reject log and proceeds to the next chunk.

### 8.2 Gate 2 — ontology conformity

- For every `mention` record: `entity_type` ∈ `{method, dataset, metric, task}` (the four `extracted` entity types in `ontology.yaml` v1).
- For every `assertion` record: `relation_type` ∈ `{PROPOSES, INTRODUCES, EVALUATES_ON, USES_METHOD, REPORTS_METRIC, EXTENDS, USED_FOR_TASK, MEASURES}` (the eight `extracted` relation types in `ontology.yaml` v1).
- For every `assertion` record: `subject_ref.kind` and `object_ref.kind` match the population modes of the relation's `from` and `to` entity types in `ontology.yaml`.
- A structural `document` endpoint has `kind: "document"` and `id == assertion.doc_id == chunk.doc_id` byte-for-byte. No other structural endpoint kind is defined in ontology v1.
- An extracted endpoint has `kind: "mention"`; its ID resolves to a validated mention from the same chunk, and that mention's `entity_type` matches the relation endpoint type. Dangling and cross-chunk mention refs fail validation.
- There is no blanket typed-endpoint inequality rule. A self-loop is rejected only when the ontology or a relation-specific contract explicitly forbids it; ontology v1 does not pin a general self-loop prohibition for the extracted relations.

The implementing ticket renders the allowed `entity_type` and `relation_type` sets from `ontology.yaml` filtered to `population_mode == extracted`, so a curated ontology change automatically updates the validation surface. **Code does not hard-code any ontology name.**

### 8.3 Gate 3 — evidence integrity

Per §7.1: offsets in range (`0 <= evidence.char_start < evidence.char_end <= len(chunk.text)`), the span contains the expected text (mentions) or is non-empty after whitespace normalisation (assertions), and `evidence.source_text_sha256 == sha256(chunk.text.encode("utf-8")).hexdigest()`. Failures are reject-log entries; the affected phase is recorded as not having passed Gate 3 and the chunk is not extraction-complete (§9).

### 8.4 Reject log

`data/extractions.rejects.jsonl` carries one record per failed gate-2 or gate-3 occurrence:

```json
{
  "chunk_id": "c_a1b2c3d4e5f60718",
  "phase": "entity | relation",
  "attempt": 1,
  "gate": "ontology | evidence",
  "reason": "...",
  "offending_record": { ... the rejected mention or assertion ... }
}
```

A schema-validation failure (§8.1) is also recorded, with `gate: "schema"`. Reject-log records do not consume IDs.

The implementing ticket surfaces reject counts in the run's stdout summary so an operator notices them without reading the JSONL. A non-zero reject count does not by itself invalidate the run; the run is invalidated only if a chunk is not extraction-complete (§9) for a reason that cannot be retried cleanly.

## 9. Extraction-complete gate

A chunk is **extraction-complete** when and only when all of the following hold:

1. The entity phase produced a validated `mention` list for the chunk (zero or more records; zero is acceptable when the chunk contains no surfaced entities).
2. The relation phase produced a validated `assertion` list for the chunk (zero or more records; zero is acceptable).
3. Both phases passed all three validation gates. **Strict reading (repair #23):** a per-record Gate 2 or Gate 3 rejection in either phase means that phase did not pass all three gates, even if the provider call itself returned a parseable result. Zero validated records is acceptable only when the phase ran to completion with zero Gate 1, Gate 2, and Gate 3 failures (the LLM returned an empty `mentions` / `assertions` array and every gate ran without raising on a candidate).
4. A `chunk_status` record with `extraction_complete: true` exists in `data/extractions.jsonl` for this `chunk_id`.

A chunk is **never** extraction-complete on a single phase. Partial completion (entity-only or relation-only) is an incomplete state: the chunk is not benchmark-eligible, but its validated phase output is retained and resumable (§13).

The downstream Neo4j ingestion ([#9](https://github.com/ImmoP/knowledge_graph_rag/issues/9)) consumes **only** chunks whose `chunk_status.extraction_complete == true`. The benchmark runner ([#14](https://github.com/ImmoP/knowledge_graph_rag/issues/14)) records in its report metadata the count of extraction-complete chunks at run time.

## 10. Cache contract

The cache is a property of `LLMProvider.generate_json`, per [#16](https://github.com/ImmoP/knowledge_graph_rag/issues/16) §5. This spec pins what extraction feeds into it and what it must guarantee for extraction specifically.

### 10.1 Key composition (extraction call to `generate_json`)

```
key = sha256(
    provider.name
  + "|" + provider.model
  + "|" + sha256(system_prompt)
  + "|" + sha256(user_prompt)
  + "|" + sha256(json.dumps(schema, sort_keys=True))
  + "|" + str(temperature)
  + "|" + str(max_tokens)
  + "|" + chunk_id
  + "|" + source_text_sha256
  + "|" + ontology_version
  + "|" + ontology_sha256
  + "|" + prompt_template_filename
  + "|" + sha256(prompt_template_file_content)
)
```

Where:

- `provider.name`, `provider.model` come from the active adapter (per #16 §1).
- `system_prompt` is the rendered system message (extraction's two templates each carry a small system preamble explaining the role and the schema).
- `user_prompt` is the rendered user message: the chunk text + the few-shot examples + the relation prompt's mention-list appendix.
- `schema` is the JSON Schema dict passed to `generate_json`.
- `temperature`, `max_tokens` are the actual values used for this call.
- `chunk_id` is the chunk's primary key.
- `source_text_sha256` is `sha256(chunk.text)`. Mixing this into the key guarantees that parser-driven text drift (even if the resulting chunk manifest is byte-identical because the cleaning pipeline re-derives the same content) does not silently reuse a stale cached response.
- `ontology_version` is the integer from `ontology.yaml`'s top-level `version:` field.
- `ontology_sha256` is `sha256(open("ontology.yaml", "rb").read()).hexdigest()`.
- `prompt_template_filename` is the basename (`extract_entities_v1.txt` or `extract_relations_v1.txt`).
- `prompt_template_file_content` is the verbatim template file content. Mixing the file hash (not just the filename) means an unversioned edit to the template invalidates the cache; the version suffix is the human-visible signal, the hash is the authoritative one.

This composition is a strict superset of the #16 §5.1 key, which is composed of `(provider.name, provider.model, sha256(system + 0x1f + user))`. Extraction adds the schema, the generation parameters, the chunk identity, the ontology identity, and the prompt-template identity on top. The #16 base key composition remains valid; this is a refinement specific to extraction.

### 10.2 Cache directory layout

```
${LLM_CACHE_DIR}/
└── llm/
    ├── anthropic/
    │   └── claude-haiku-4-5/
    │       └── <key>.json
    ├── openai/
    │   └── gpt-4o-mini/
    │       └── <key>.json
    └── ollama/
        └── llama3.1:8b/
            └── <key>.json
```

Layout matches #16 §5.2, scoped per provider/model. Extraction does not add new substructure.

### 10.3 Stored value

Per #16 §5.2: the parsed dict from `generate_json`. Cache files are content-addressed JSON.

### 10.4 Invalidation

Cache invalidation is automatic because every identity component is mixed into the key. There is no manual "purge cache" step. Concretely:

| Change | Effect |
|---|---|
| Switch `LLM_PROVIDER` | new namespace (new top-level dir). |
| Switch `LLM_MODEL` | new namespace (new model subdir). |
| Edit a prompt template file | new key (prompt template content hash differs). |
| Bump `ontology.yaml`'s `version:` | new key (the integer is mixed). |
| Edit any byte of `ontology.yaml` | new key (`ontology_sha256` differs). |
| Edit any field of a chunk record in `chunks.jsonl` | new key (`source_text_sha256` differs, or `chunk_id` differs). |
| Change `temperature` or `max_tokens` | new key (parameter mixed). |
| Change the rendered schema (driven by ontology change or template change) | new key (schema hash differs). |

The implementing ticket does not need an `invalidate_cache` CLI. Operators who want to force a clean re-run delete the `LLM_CACHE_DIR`.

### 10.5 What is NOT cached by extraction

Per #16 §5.4:

- Budget accounting state (per-process counters).
- `chunk_status` records (extraction computes these; the cache is only for LLM call responses).
- Reject-log records (extraction computes these; the cache is only for LLM call responses).

## 11. Budget control

The budget caps from #16 §6 apply without modification. Extraction's specific additions:

- `LLM_BUDGET_USD` and `LLM_BUDGET_TOKENS` are the only caps. Extraction does not introduce a third cap.
- The wrapper checks the cap **after every `generate_json` call**, before writing the validated records for that call. The check uses the cumulative `Usage` maintained by the #16 wrapper.
- **Clean abort:** when the cap is exceeded, the wrapper raises `BudgetExceeded`. The extraction entry point catches it, finishes writing the in-progress chunk's records (if validation already passed), and exits with non-zero status. It does not delete or rewrite existing records on disk; it leaves the partial run as-is.
- **Partial results policy:** an incomplete run produces a `data/extractions.jsonl` that contains every extraction-complete chunk's records plus every successfully-validated phase of every in-progress chunk. The `chunk_status` records make the partial state inspectable. Resumability (§13) reads from the chunk_status records.

The run manifest (`data/extractions.manifest.json`) records the cumulative cost and token count at the moment the run terminated, regardless of whether termination was clean (cap) or crash (operator SIGINT, etc.).

## 12. Retry policy

The retry policy is a strict subset of what `LLMProvider` permits. Extraction retries exactly one category of failure, with bounded backoff:

### 12.1 Transient transport failures

A failure classified as transient is retried with **bounded backoff** up to **3 attempts** total. Backoff schedule (pinned constant; see §12.3):

| Attempt | Action |
|---|---|
| 1 | call immediately (no `sleep(0)`) |
| (transient failure of attempt 1) | `sleep(2)` seconds |
| 2 | call |
| (transient failure of attempt 2) | `sleep(4)` seconds |
| 3 | call |
| (transient failure of attempt 3) | re-raise the final `ProviderTransportError` unchanged |

Total elapsed cap: 6 s of sleep across the run. No `sleep(0)` is invoked; no sleep follows attempt 3; maximum three attempts; no `RetryExhausted` wrapper is introduced.

**Transient classification (repair #23).** Adapters raise `ProviderTransportError(*, transient: bool, status: int | None = None, cause: BaseException | None = None)` (see `provider-abstractions §6.6`) from their outbound HTTP boundary. The Ollama adapter wraps only known transport/status failures: `httpx.TransportError` subclasses (including `ConnectError`, `ConnectTimeout`, `ReadTimeout`, `WriteTimeout`, `PoolTimeout`, `NetworkError`, `RemoteProtocolError`) and `httpx.HTTPStatusError`. Transient HTTP statuses are pinned to `{408, 429, 500, 502, 503, 504}`; any other 4xx (auth, model-not-found, malformed request) is non-transient. `409` and `425` are NOT transient. Unexpected parsing, programming, and internal exceptions propagate unchanged; `LLMValidationError` and `BudgetExceeded` propagate unchanged. The implementing ticket does not classify by exception class-name string match.

### 12.2 What is NOT retried

- **Schema-validation failures.** These are `LLMValidationError` (per #16 §2.3). The single retry owned by `generate_json` has already been spent; if validation still fails, the record goes to the reject log and the chunk is not extraction-complete. The pipeline never retries with a different schema or a different prompt.
- **Budget-exceeded.** `BudgetExceeded` is the abort signal, not a retry condition.
- **Ontology or evidence gate failures.** These are post-LLM validation issues, not transport failures.
- **Non-transient provider errors** (HTTP 4xx other than 408/429, auth failures, model-not-found). These are surfaced as `LLMProviderError` and routed to the reject log.

### 12.3 Why this is a strict subset

The #16 §8 explicitly defers "Retry policy beyond the single validation retry" to the implementing ticket. Extraction pins the subset: 3 attempts total, 2s/4s exponential schedule. The implementing ticket does not invent a different policy for a "real" run; the constant is committed to this spec.

### 12.4 Effect on the cache

Retries hit the cache normally. If the first attempt's `generate_json` call succeeded and wrote a cache entry, the retry finds it on the next call and returns immediately. Retries that do not touch the LLM (cache hits) have no delay.

## 13. Resumability and benchmark eligibility

### 13.1 Resumability

A re-run of the entry point against an existing `data/extractions.jsonl` and `data/extractions.manifest.json` is idempotent in the strong sense:

- For every `chunk_id` whose `chunk_status.extraction_complete == true` in the existing file: skip both phases entirely.
- For every `chunk_id` whose `chunk_status.extraction_complete == false` (or that has no `chunk_status` record): re-run entity, then relation. The validated records from the existing file are kept; the new records replace them.

**Identity preconditions for carry-forward.** Carrying a prior `chunk_status.extraction_complete == true` forward is permitted only when the previous run's manifest matches the current run across every one of the following fields. Any mismatch, or any missing field, or any unparseable previous manifest, invalidates carry-forward for every canonical chunk and forces reprocessing under the current identity.

| Field | Source in the previous manifest | Source in the current run |
|---|---|---|
| `chunks_sha256` | `data/extractions.manifest.json` | freshly computed by the current `PreflightReport` |
| `ontology_sha256` | same | same |
| `prompt_template_sha256` (list equality) | same | same |
| `provider` | same | same |
| `model` | same | same |
| `max_tokens` | same | same |
| `temperature` | same | same |

An unparseable previous manifest (corrupt JSON, missing required fields, non-dict shape) is treated as identity-mismatch for every chunk.

This is implemented by:

1. On startup, load the existing `data/extractions.jsonl` (if present) into an in-memory map keyed by `(chunk_id, kind)`.
2. On startup, parse the existing `data/extractions.manifest.json`. If the file is missing, unparseable, or any required field above is missing or wrong-typed, treat the previous run as having no identity.
3. Compute the current identity from the freshly read canonical artifacts and provider configuration. Compare every field in the table above against the parsed previous manifest.
4. If every field matches, carry forward prior `chunk_status.extraction_complete == true` records. If any field differs or the previous manifest cannot be trusted, do not carry forward; every canonical chunk is reprocessed under the current identity.
5. Before processing a chunk, check whether a `chunk_status` record with `extraction_complete == true` exists in the carried-forward set.
6. If yes, skip. If no, run both phases, then write all records for that chunk (mention, assertion, and chunk_status).

The run manifest records `resumed_from_run_id` (the prior run's timestamp) so the benchmark report can trace the lineage.

### 13.2 Benchmark eligibility

The benchmark runner ([#14](https://github.com/ImmoP/knowledge_graph_rag/issues/14)) treats extraction as benchmark-eligible **only** when all of the following hold:

1. The run exited with status 0.
2. `data/extractions.manifest.json` is present.
3. For every `chunk_id` in the corpus's chunk manifest: a `chunk_status` record with `extraction_complete == true` exists in `data/extractions.jsonl`.
4. The ontology version recorded in the manifest equals the `version` field in `ontology.yaml` at run time.
5. The `chunks.jsonl` SHA256 recorded in the manifest equals the SHA256 of `data/derived/chunks.jsonl` at run time.

A run that violates any of these is recorded in the benchmark report as **incomplete** and is not used for benchmark numbers. The contract is strict: a benchmark that consumed an incomplete extraction is unreproducible, and the whole point of the cache + budget + resumability story is to make complete runs achievable within the budget.

The map's no-tuning-on-held-out policy ([#4](https://github.com/ImmoP/knowledge_graph_rag/issues/4)) implies that benchmark numbers must be reproducible. Incomplete extraction is one of the two documented ways a benchmark can be invalid; the other (the eval set or its evidence path being out of date) is #14's responsibility, not this spec's.

## 14. Cross-ticket boundaries

### 14.1 What this spec hands off

- **To #8 (entity resolution):** the `mention` records in `data/extractions.jsonl`. #8 reads `mention.canonical_form`, normalises, and produces canonical entity keys. #8 owns the `surface_form → canonical_form` cluster membership and the gold entity IDs; this spec produces only the pre-resolution mentions.
- **To #9 (Neo4j ingestion):** the `assertion` records, filtered to extraction-complete chunks. #9 maps `{"kind":"document","id":doc_id}` directly to `Document(doc_id)` and resolves `{"kind":"mention","id":m_*}` through `data/entity_resolutions.jsonl` to a canonical entity `e_*`; missing, duplicate, dangling, or type-incompatible refs fail ingestion. The cross-store ID contract (#2 §7) means `chunk_id` and `doc_id` flow through unchanged.
- **To #14 (benchmark runner):** the `chunk_status` records (to compute completeness), the run manifest (for metadata), and the extraction output (for the benchmark input if a path goes through extracted facts).

### 14.2 What this spec explicitly does NOT do

- **Canonical entity resolution.** Owned by #8. This spec produces pre-resolution mentions and assertions; the LLM's `canonical_form` is one input to #8, not its output.
- **Graph ingestion.** Owned by #9.
- **Retrieval.** Owned by #11 and #12.
- **Grounded answer generation.** Owned by #13.
- **Citation key resolution** (`[[CITE:...]]` → OpenAlex work IDs). That is a separate pipeline, post-extraction; the citation markers live in `chunk.text` and are passed through unchanged.

### 14.3 What this spec assumes from upstream

- `data/derived/chunks.jsonl` exists and conforms to the chunk-record spec (§9 acceptance gate).
- `ontology.yaml` exists, validates, and is `version: 1`.
- A working `LLMProvider` adapter is registered (per #16's acceptance gate).
- Required env vars are set (or unset-and-defaulted).

### 14.4 What this spec asks downstream to consume

The downstream stages (#8, #9, #14) must:

- Read `data/extractions.jsonl` as the sole source of extraction output. They MUST NOT call the LLM directly.
- Validate every record against the JSON Schema for its kind before using it. Validation failures are bugs in this spec or in upstream; downstream surfaces them, does not paper over them.
- Use the `chunk_status` records to determine extraction-completeness. They MUST NOT treat absence-of-record as "this chunk has no extractions."

## 15. Acceptance gate

This spec closes when committed to `docs/specs/extraction-contract.md`. The implementing extraction ticket closes when all of the following pass:

1. **`data/extractions.jsonl` and `data/extractions.manifest.json` are produced** by running `extract.py` against the full chunk manifest, on a real LLMProvider adapter.
2. **All `chunk_id` values in `chunks.jsonl` have a `chunk_status` record** with `extraction_complete: true` in `extractions.jsonl`. The completeness check from §13.2 passes.
3. **Schema validation** of every record in `extractions.jsonl` against the JSON Schemas for the three record kinds passes. The implementing ticket ships a small validation script (no test framework) and runs it.
4. **Cache key uniqueness check:** switching `LLM_PROVIDER` between two registered adapters produces disjoint cache namespaces. Switching `LLM_MODEL` does likewise. Switching ontology versions produces disjoint cache namespaces.
5. **Budget cap triggers:** with `LLM_BUDGET_TOKENS` set to a value that the second call would exceed, the run aborts after the second call with `BudgetExceeded`, exits non-zero, and leaves `data/extractions.jsonl` containing the first chunk's records plus the in-progress chunk's records.
6. **Retry behaviour:** a forced transient failure (simulated by an adapter test double) is retried per §12.1's schedule; a forced schema-validation failure is sent to the reject log without retrying beyond `generate_json`'s built-in single retry.
7. **Resumability:** a partial run followed by a re-run produces the same `extractions.jsonl` as a single uninterrupted run would have (byte-identical, given a deterministic provider). The ID-stability property is also asserted explicitly: the set of `mention_id` and `assertion_id` values in the resumed run equals the set the uninterrupted run would have produced, regardless of which chunks were processed first or which chunks failed.
7a. **Resume identity gate (impl-spec §13.1):** a re-run against an existing `data/extractions.manifest.json` carries forward prior `chunk_status.extraction_complete == true` records **only** when every one of the seven identity fields listed in §13.1 (`chunks_sha256`, `ontology_sha256`, `prompt_template_sha256`, `provider`, `model`, `max_tokens`, `temperature`) matches the freshly computed current value. A change in any one field, a missing field, an unparseable manifest, or a missing manifest file invalidates carry-forward for every canonical chunk and forces reprocessing under the current identity. The implementing ticket ships tests covering (a) identity match making zero new LLM calls for previously completed chunks, (b) each of the seven field drifts forcing reprocessing of the previously completed chunks, and (c) an unparseable prior manifest failing closed.
8. **No provider SDK import** in `src/extraction/` (verifiable with `grep`).
9. **Idempotency:** two consecutive runs against the same chunk manifest, ontology, prompt templates, and provider produce byte-identical `data/extractions.jsonl` (assuming a deterministic provider). SHA256 of the file is the gate. The implementing ticket additionally ships a tiny harness that re-derives every `mention_id` and `assertion_id` from its record fields per §6.1 and asserts the recomputed ID equals the stored one for every record in the file; this catches any future regression that re-introduces a position-dependent ID rule.
10. **`ontology.yaml` is updated** in lock-step with this spec: every `extracted` entity type's `extraction_prompt` references `extract_entities_v1`; every `extracted` relation type's `extraction_prompt` references `extract_relations_v1`. The ontology spec's "appears in extraction prompt template" cross-check (§5.10) passes against the two grouped templates.

## 16. What this spec leaves open (implementing ticket owns)

- **The exact text** of `extract_entities_v1.txt` and `extract_relations_v1.txt` (the prompt wording, the few-shot example ordering, the system preambles). This spec pins the *shape* of what they must instruct; the implementing ticket writes the prose.
- **The exact JSON Schema** for the two output shapes. Schemas are generated from `ontology.yaml`, but the implementing ticket writes the generator.
- **Concurrency model.** Sequential is the default; parallel chunk processing requires the implementing ticket to add a flag and reason about cache ordering. Until that flag lands, chunks are processed one at a time.
- **The reject-log inspector tool** (a small CLI to summarise reject counts and types). The reject log itself is part of this spec; the inspector is implementing-ticket convenience.
- **Adapter-level retry parameters** within the bounds set by §12 (the adapter picks the implementation of "exponential backoff"; extraction pins the schedule and attempt count).
- **A small end-to-end smoke test** (e.g. extract the first 5 chunks, validate completeness, verify expected entities). This spec requires the smoke test; the implementing ticket writes it.

---

This spec resolves ticket #7. It is referenced by #8 (entity resolution consumes pre-resolution mentions), #9 (Neo4j ingestion consumes validated assertions for extraction-complete chunks), #14 (benchmark runner validates extraction completeness), and the implementing extraction ticket (which `extract.py` ships). The cross-store ID contract from #2 §7 is the only ID-format dependency downstream of this spec.
