# Benchmark Runner Contract Specification

Resolution of [#14 — Decide the benchmark-runner shape and metrics](https://github.com/ImmoP/knowledge_graph_rag/issues/14). This document is the **benchmark execution, scoring, and reporting contract** that consumes the eval set (#4) against the three retrieval conditions (#10 vector, #11 graph, #12 hybrid) and the grounded generator (#13), under the provider abstractions (#16). It is not the implementing code; that ticket is produced by `/to-tickets` once this spec lands.

The benchmark runner answers one question: *given the frozen 45-question held-out eval set, how much of the required evidence does each retrieval condition recover, and how well does the grounded generator use that evidence?* Everything in this spec exists to make that comparison legible, reproducible, and honest.

## 1. Standing constraints

- **Held-out canonical eval.** Input is `eval/questions.jsonl` per #4 §3 and §7. Default `--questions-file` is `eval/questions.jsonl`; any override is recorded in `report.json`. The shipped canonical file holds exactly 45 records, all `split: "held_out"`. The runner does not consume `eval/dev_questions.jsonl` and refuses to evaluate on any record whose `split != "held_out"`.
- **Three retrieval conditions, kept separate.** `vector`, `graph`, `hybrid` are independent conditions. The runner calls each retriever unchanged against its own pinned contract (#10, #11, #12). No condition is composed, weighted, or rerouted at the benchmark layer.
- **One generation per `(qid, mode)` against `k=5`.** The generator (#13) is called exactly once per `(qid, mode)` using the ordered `k=5` retrieval result for that mode. Generation never sees `k=20`. There is no provider-nondeterminism probe: a second call would be a #16 cache hit after the first miss and is not a valid probe; the runner never bypasses the cache.
- **Deterministic v1 scoring.** No LLM judge, no embedding judge, no learned NLI. Every metric in this spec is a closed-form function of the recorded inputs. A future judge, if ever added, is a separate versioned follow-up.
- **No feedback from gold to runtime.** `expected_answer`, `supporting_chunk_ids`, `gold_evidence_path`, and `hop_bucket` never enter retrieval, generation, validation, or retry. They are scoring inputs only.

## 2. Scope and non-goals

### 2.1 In scope

- Benchmark execution orchestration across the three retrieval conditions.
- Gold scoring (answer-content, citation, claim-support).
- Aggregation across `hop_bucket` × `mode` × question.
- Report serialization (`report.json`, `records.jsonl`, headline/secondary tables).
- Run validity and failure policy (`status = refused | completed | invalid`).

### 2.2 Out of scope

The runner does not modify retriever behavior (#10, #11, #12), generator behavior (#13), the #13 runtime validator, the #16 cache, eval records, held-out parameters, `README.md`, or implement any code in this ticket.

## 3. Inputs

### 3.1 Eval set

A path to a JSONL file shaped per #4 §3. The runner preflights the file before any retrieval call (§10). For each record the runner reads:

- `qid` — pinned format `q_NNN` (zero-padded to three digits; e.g. `q_001`, `q_045`). Unique within the file.
- `question`
- `expected_answer` and (optional) `answer_aliases`
- `hop_bucket` ∈ `{"1", "2", "3+"}`
- `evidence_path` (the eval field) — copied into the per-question record under the renamed key `gold_evidence_path` (§7) to disambiguate from retrieval provenance.
- `supporting_chunk_ids` — required non-empty in the shipped file.
- `supporting_doc_ids`
- `split` — must be `"held_out"` for every record the runner consumes.

Records with empty `supporting_chunk_ids` cause a hard refusal (§10.1), not silent skip. This is consistent with #4 §3.3: `[]` is authoring-transient and the shipped file requires non-empty gold.

### 3.2 Retrieval artifacts

- `data/derived/chunks.jsonl` — canonical chunk manifest (#2).
- `data/manifest.jsonl` — corpus manifest.
- `ontology.yaml` — frozen v1 ontology (#5).
- Graph ingestion outputs per #9, including `graph_content_sha256` and `graph_ingestion_manifest_sha256`.
- Vector store state per #10, including `embedding_source_chunks_sha256` (§10.2).
- Per-mode retrieval spec SHAs (§8.2): `vector_retrieval_spec_sha256`, `graph_retrieval_spec_sha256`, `hybrid_retrieval_spec_sha256`.

### 3.3 Generation inputs

- The pinned grounded prompt at `prompts/grounded_answer_v1.txt` per #13, recorded as `prompt_template_sha256` and `grounded_generation_spec_sha256`.
- `LLMProvider` and `EmbeddingProvider` instances per #16; `LLM_PROVIDER`, `LLM_MODEL`, `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` are runtime configuration.

### 3.4 Pinned retrieval constants

- `GENERATION_K = 5`
- `RETRIEVAL_EVAL_K_VALUES = (5, 20)`
- Graph invocation: `graph_retriever.retrieve(query, k, hop_budget=3)` for both `k=5` and `k=20`.
- Vector and hybrid invocations use their own already-pinned public interfaces unchanged.

Because `retrieve(query, k)` may legitimately depend on `k` (especially hybrid, which calls vector and graph with `HYBRID_K_VEC_MULTIPLIER = 2` and `HYBRID_K_GRAPH_MULTIPLIER = 2` multiplied by `k`), the runner does not assume `retrieve(query, 20)[:5] == retrieve(query, 5)`. The two `k` values are called independently.

## 4. Per-question run

### 4.1 What the runner does, per `(qid, mode)`

For each `(qid, mode)`:

1. Call `vector_retriever.retrieve(query, 5)` and `vector_retriever.retrieve(query, 20)`.
2. Call `graph_retriever.retrieve(query, 5, hop_budget=3)` and `graph_retriever.retrieve(query, 20, hop_budget=3)`.
3. Call `hybrid_retriever.retrieve(query, 5)` and `hybrid_retriever.retrieve(query, 20)`.
4. Feed the ordered `k=5` hits to the orchestrator exactly once. The orchestrator calls `LLMProvider.generate` once via #13 and produces either a `GenerationResult` (which may be `abstained`) or raises `GenerationParseError`.

That is six retrieval calls and one generation per `(qid, mode)`. The benchmark unit is `(qid, mode)`.

### 4.2 Generation evidence is the ordered `k=5` retrieval result

The runner takes the ordered `k=5` hit list from the appropriate retriever unchanged and hands it to the orchestrator as the `EvidenceBlock` tuple per #13. The orchestrator is responsible for assembling the `GenerationRequest` from that ordered list; the runner does not re-rank, trim, or supplement it. Generation against `k=20` is not performed.

## 5. Retrieval metrics

Metrics operate on the **ordered chunk hits actually returned by the retriever** for the requested `k`, not on a flattened set derived from `gold_evidence_path`. For each `(qid, mode, k)` the runner stores the ordered hit list under `retrieval.k5.hits` / `retrieval.k20.hits` (§7). Metrics are computed per-question against that array.

### 5.1 Required core family

- **`retrieval_recall@k`** = `|top_k_chunk_ids ∩ supporting_chunk_ids| / |supporting_chunk_ids|`. With `supporting_chunk_ids == []` (which the runner never allows in the shipped file): undefined → not computed.
- **`hit_rate@k`** = `1` if `top_k_chunk_ids ∩ supporting_chunk_ids != ∅`, else `0`.
- **`MRR@k`** = `1 / r` where `r` is the smallest 1-based rank ≤ `k` whose `chunk_id ∈ supporting_chunk_ids`; `0` if no supporting chunk occurs in the top `k`. When multiple gold chunks appear, the first relevant rank wins; later relevant chunks do not double-count.

### 5.2 Optional diagnostic

- **`retrieval_precision@k`** may be reported as a secondary diagnostic. It is not a headline candidate and is not required to be present in v1. It is not called "citation precision" — that name is reserved for the gold-citation metric in §6.2.

### 5.3 Retrieval hit record shape

Each entry in `retrieval.k5.hits` and `retrieval.k20.hits` is:

```json
{
  "rank": 1,
  "chunk_id": "c_<16 hex>",
  "score": <retriever-native numeric score>,
  "provenance": <object or null>
}
```

`score` is the retriever-native score and **must not be compared numerically across vector, graph, and hybrid**; the runner does not aggregate scores across modes. For vector, `provenance` is `null` or minimal. For graph and hybrid, `provenance` is the per-retriever provenance contract from #11 / #12 — verbatim, including the connected five-field directional `graph_evidence_path` for graph and the per-retriever rank/score/provenance bundle for hybrid.

## 6. Answer-quality and grounding metrics

These metrics operate on the single recorded `GenerationResult` (or `GenerationParseError`) for each `(qid, mode)` at `k=5`.

### 6.1 Deterministic v1 answer-content metrics

- **`answer_hit`** — binary. `1` iff the normalized `expected_answer` (or any `answer_aliases[i]`) appears as a substring of the normalized `answer_text`. Normalization for this rule is the same NFKC + casefold + collapse-whitespace pipeline used for `normalized_f1` below. Otherwise `0`. `ABSTAIN` yields `answer_hit = 0`. `GenerationParseError` yields `answer_hit = 0`. `answer_hit` always uses the **full question denominator** (n includes abstentions and parse errors as zero).
- **`normalized_f1`** — secondary deterministic content metric. Always enabled. Never a headline-selection candidate.

  Tokenization (pinned, applied identically to prediction and target):
  1. Remove canonical `[[c_<16 hex>]]` citation tokens from `answer_text`.
  2. Unicode NFKC.
  3. Casefold.
  4. Tokenize into maximal contiguous Unicode-alphanumeric sequences (numeric tokens retained).

  Apply the same tokenization to `expected_answer` and each `answer_alias`. Compute standard multiset token precision and recall, then F1, against each candidate target. The reported `normalized_f1` is the **maximum F1** over `{expected_answer} ∪ answer_aliases`. Edge cases: both prediction and target token lists empty → `F1 = 1`; exactly one empty → `F1 = 0`.

  `ABSTAIN` and `GenerationParseError` both yield `normalized_f1 = 0`. `normalized_f1` always uses the full question denominator.

  Normalization is fixed and is not tuned on the 45 held-out questions.

### 6.2 Gold citation metrics

These compare the generator's cited chunks against `supporting_chunk_ids`. They are #14 benchmark metrics only and never feed back into retrieval, generation, validation, or retry.

Let:

- `C_all` = unique syntactically valid `cited_chunk_ids` in the generated answer (after #13 parsing; `ABSTAIN` has `C_all = ∅`; `GenerationParseError` makes these metrics undefined).
- `C_in` = unique cited chunk_ids whose #13 citation status is **not** `OUT_OF_EVIDENCE`.
- `G` = `supporting_chunk_ids`.

Definitions:

- **`gold_citation_precision`** = `|C_in ∩ G| / |C_all|` if `C_all` is non-empty, else `0.0`.
- **`gold_citation_recall`** = `|C_in ∩ G| / |G|`.

Properties:

- An `OUT_OF_EVIDENCE` citation can never receive gold-citation credit.
- An `OUT_OF_EVIDENCE` citation still penalizes `gold_citation_precision` through `C_all`.
- An `OUT_OF_EVIDENCE` citation never increases `gold_citation_recall`.

For a parsed answer with no citations and non-empty `G`: both metrics are `0.0`. For `ABSTAIN`: both are `0` and `claim_support_coverage = None` (see §6.4).

### 6.3 Additional grounding diagnostics

Reported as secondary metrics. None of these collapse `UNCITED`, `CITED_UNSUPPORTED`, and `OUT_OF_EVIDENCE` into a single value:

- **`claim_support_coverage`** — from #13's per-claim verdict set: `(#SUPPORTED claims) / (#claims)`. `0/0 → 1.0` only when there are zero claims AND the answer is not `ABSTAIN` and not a parse error; `None` otherwise (ABSTAIN has no claims; parse error has no parsed structure).
- **`cited_unsupported_claim_rate`** = `(claims with verdict CITED_UNSUPPORTED) / (#claims)`. Same edge-case handling as `claim_support_coverage`.
- **`uncited_claim_rate`** = `(claims with verdict UNCITED) / (#claims)`. Same edge-case handling.
- **`out_of_evidence_citation_rate`** = `(#unique OUT_OF_EVIDENCE cited chunk_ids) / (#unique syntactically valid cited chunk_ids)`. With `|C_all| == 0` → `0.0` (citationless answers and `ABSTAIN` both yield `0.0` here).

A model that emits no citations must not appear to have zero grounding problems: `uncited_claim_rate` will typically be high for such a model, and `cited_unsupported_claim_rate` is computed only over the claims that actually exist.

### 6.4 ABSTAIN

`ABSTAIN` is a deterministic protocol output per #13 §4.2 — a `GenerationResult` with `abstained == True`, `answer_text == "ABSTAIN"`, and all collections empty. It is not an error.

Scoring:

- `answer_hit = 0`
- `normalized_f1 = 0`
- `gold_citation_precision = 0`
- `gold_citation_recall = 0`
- `claim_support_coverage = None`
- `cited_unsupported_claim_rate = None`
- `uncited_claim_rate = None`
- `out_of_evidence_citation_rate = 0.0`
- `abstained = true`
- `parse_error = false`

ABSTAIN's absence of claims is **not** interpreted as perfect grounding.

### 6.5 `GenerationParseError`

`GenerationParseError` is a protocol failure per #13 §5, distinct from `ABSTAIN`. It must remain explicitly visible.

Recorded as `generation.status = "parse_error"`. It contributes:

- `answer_hit = 0`
- `normalized_f1 = 0`
- `parse_error = true`
- `abstained = false`

Grounding metrics that require a parsed claim/citation structure (`claim_support_coverage`, `cited_unsupported_claim_rate`, `uncited_claim_rate`, `out_of_evidence_citation_rate`, `gold_citation_precision`, `gold_citation_recall`) are `null` on this path.

The runner never silently drops a parse-error question from `answer_hit` or `normalized_f1` denominators — both metrics are zero over those questions, contributing to the full-denominator mean.

When aggregating those grounding metrics, the runner reports:

- the **conditional mean over defined values** (the parseable subset), and
- `n_defined` alongside the conditional mean (so a cell built on 7 parseable answers does not appear equivalent to one built on 15).

This is reported in addition to `parse_error_rate`, not in place of it.

## 7. Per-question record

One record per `(qid, mode)`. The record contains both `k=5` and `k=20` retrieval evaluations under `retrieval.k5` and `retrieval.k20`, plus exactly one `generation` object produced from `k=5`. The experimental unit is unambiguous: one record = one question + one mode + one generation.

```json
{
  "schema_version": "benchmark-record.v1",
  "qid": "q_001",
  "hop_bucket": "2",
  "mode": "hybrid",
  "question": "...",
  "expected_answer": "...",
  "answer_aliases": [],
  "supporting_chunk_ids": ["c_<16 hex>", "..."],
  "supporting_doc_ids": ["W..."],
  "gold_evidence_path": [
    {"step": 1, "from": "e_...", "relation": "...", "to": "W...", "direction": "forward"}
  ],
  "retrieval": {
    "k5":  {"hits": [], "metrics": {}},
    "k20": {"hits": [], "metrics": {}}
  },
  "generation": {
    "status": "ok | abstained | parse_error",
    "answer_text": "...",
    "cited_chunk_ids": ["c_<16 hex>"],
    "claims": [],
    "citation_statuses": [],
    "usage_calls": [
      {"input_tokens": 0, "output_tokens": 0, "cost_usd": null}
    ]
  },
  "answer_hit": 0,
  "normalized_f1": 0.0,
  "gold_citation_precision": 0.0,
  "gold_citation_recall": 0.0,
  "claim_support_coverage": null,
  "cited_unsupported_claim_rate": null,
  "uncited_claim_rate": null,
  "out_of_evidence_citation_rate": 0.0,
  "abstained": false,
  "parse_error": false
}
```

The shape above shows the field set, not every field is required to be present on every record; `null` and absence are used where §6 specifies.

### 7.1 Eval-record field rename

The runner copies `evidence_path` from the eval record into the per-question record under the renamed key `gold_evidence_path`. Retrieval provenance (per-hit `provenance` on each retrieval hit, including the #11 / #12 graph/hybrid fields) is attached to the retrieval hits, not to `gold_evidence_path`. The two are never conflated. Gold evidence lives in `gold_evidence_path`; per-retrieval provenance lives on each retrieval hit.

### 7.2 `usage_calls` discipline

`usage_calls` is a tuple of #16 `Usage` values per #13 §3 / §6, with fields `input_tokens`, `output_tokens`, `cost_usd` (monetary cost may be `null` for providers that do not expose it). The runner does not place `cache_key`, `provider`, `model`, `cache_hit`, or timestamps inside `usage_calls` unless an upstream contract explicitly exposes those fields. Provider/model identity is recorded in `report.json.config` (§8). Optional provider/cache observability is at most an implementation log outside the `GenerationResult` contract.

## 8. Reproducibility and report metadata

### 8.1 Two distinct identifiers

- **`run_fingerprint`** — a content-addressed identity over the semantic benchmark inputs. Same fingerprint ⇒ same retrieval conditions, same scoring, same expected outputs (subject to §11). Independent of execution time.
- **`run_id`** — a unique execution identifier. Different `run_id` values may share the same `run_fingerprint`.

### 8.2 `run_fingerprint`

`run_fingerprint = first 16 hex of sha256(canonical_json(run_inputs))`, where `canonical_json` is the canonical-JSON serialization pinned by #16 §5.1 (sort_keys, separators `(",",":")`, UTF-8). `run_inputs` is a JSON object with the following required fields. `started_at`, `finished_at`, wall-clock durations, cache-hit state, generated output, and timestamps are excluded.

- `git_sha` — `git rev-parse HEAD` at the moment preflight begins.
- `eval_file` — `{ "path": <absolute path>, "sha256": <hex> }`.
- `chunks_sha256` — SHA256 of `data/derived/chunks.jsonl`.
- `ontology_sha256` — SHA256 of `ontology.yaml`.
- `graph_ingestion_manifest_sha256` — from #9 ingestion manifest.
- `graph_content_sha256` — from #9 ingestion manifest.
- `embedding_source_chunks_sha256` — SHA256 of `data/derived/chunks.jsonl` against which the current vector-store embeddings were produced/validated (§10.2).
- `embedding_model` — `{ "provider": <string>, "model": <string>, "revision": <string> }` for the embedding model identity used by the vector store at run time.
- `retrieval` — `{ "k_values": [5, 20], "graph_hop_budget": 3, "hybrid_policy": <frozen #12 policy reference> }`.
- `llm` — `{ "provider": <string>, "model": <string>, "temperature": <float>, "max_tokens": <int> }`.
- `prompt` — `{ "path": "prompts/grounded_answer_v1.txt", "sha256": <hex> }`.
- `cache_format_version` — from #16.
- `spec_shas` — `{ "vector_retrieval_spec_sha256", "graph_retrieval_spec_sha256", "hybrid_retrieval_spec_sha256", "grounded_generation_spec_sha256", "provider_abstractions_spec_sha256" }`, all pinned.
- `report` — `{ "schema_version": "benchmark-report.v1", "scoring_version": <pinned id or sha> }`.

The implementing ticket may add additional semantic-input fields, but it may not weaken or remove any field above.

### 8.3 `run_id`

`run_id = <UTC compact>-<run_fingerprint>`, where `<UTC compact>` is `YYYYMMDDTHHMMSSZ` (filesystem-safe; no `:`). Two executions that share every semantic input but run at different wall-clock times have different `run_id` values and the same `run_fingerprint`.

### 8.4 `report.json`

One file per run, at `data/runs/<run_id>/report.json`. Fields:

```json
{
  "schema_version": "benchmark-report.v1",
  "scoring_version": "<pinned id or sha>",
  "run_id": "YYYYMMDDTHHMMSSZ-<16hex>",
  "run_fingerprint": "<16 hex>",
  "started_at": "UTC ISO-8601",
  "finished_at": "UTC ISO-8601",
  "git_sha": "<hex>",
  "status": "refused | completed | invalid",
  "eval_set": {
    "path": "<absolute path>",
    "sha256": "<hex>",
    "n_questions": 45,
    "n_by_bucket": {"1": 15, "2": 15, "3+": 15},
    "n_by_split": {"held_out": 45, "dev": 0, "train": 0}
  },
  "chunks_sha256": "<hex>",
  "ontology_sha256": "<hex>",
  "graph_ingestion_manifest_sha256": "<hex>",
  "graph_content_sha256": "<hex>",
  "embedding_source_chunks_sha256": "<hex>",
  "vector_retrieval_spec_sha256": "<hex>",
  "graph_retrieval_spec_sha256": "<hex>",
  "hybrid_retrieval_spec_sha256": "<hex>",
  "grounded_generation_spec_sha256": "<hex>",
  "provider_abstractions_spec_sha256": "<hex>",
  "prompt_template_sha256": "<hex>",
  "config": {
    "retrieval_k_values": [5, 20],
    "embedding_provider": "<string>",
    "embedding_model": "<string>",
    "embedding_model_revision": "<string>",
    "llm_provider": "<string>",
    "llm_model": "<string>",
    "cache_format_version": "<string>",
    "temperature": 0.0,
    "max_tokens": <int>,
    "prompt_template_path": "prompts/grounded_answer_v1.txt"
  },
  "modes": ["vector", "graph", "hybrid"]
}
```

`n_questions`, `n_by_bucket`, `n_by_split` are derived from the validated eval file actually consumed. The canonical project requires `n_questions == 45` and all records `held_out`. Per-bucket counts are expected to fall in the #4 12–18 range but are not assumed to be exactly 15 / 15 / 15. The primary table's `n` column contains the actual number of valid questions in that bucket.

The runner does not dump environment variables or secrets. Provider credentials, environment-variable dumps, and live database state are not part of the report.

### 8.5 Per-record observations file

Alongside `report.json`, the runner writes `data/runs/<run_id>/records.jsonl` — one JSON object per `(qid, mode)` per §7. This file is derived benchmark output, not source data; the implementing ticket must ensure this runtime directory is excluded from Git tracking.

## 9. Aggregation and headline layout

### 9.1 Aggregation rule

**Macro mean within a bucket, primary.** For each `hop_bucket` and each `mode`, every metric in §5 / §6 is reported as the **macro mean of per-question values** — each authored question has equal weight. Buckets are never weighted against each other in primary tables.

If a micro set metric is useful for a particular column (e.g. one weighted by `|supporting_chunk_ids|`), it appears only as a clearly labeled secondary diagnostic. It is never a primary table value.

For metrics whose values can be `null` (grounding metrics over `ABSTAIN` or parse error), the primary cell value is the **conditional mean over defined values** with `n_defined` written alongside, per §6.5. `answer_hit` and `normalized_f1` are exceptions: they always use the full question denominator, including abstentions and parse errors as zero (§6.1, §6.5).

### 9.2 Primary headline table

One table. Rows = `hop_bucket`. Columns = conditions. Cells = mean `retrieval_recall@5`.

| hop_bucket | n | vector | graph | hybrid |
|:----------|:-:|:------:|:-----:|:------:|
| 1          |   |        |       |        |
| 2          |   |        |       |        |
| 3+         |   |        |       |        |
| Total      |   |        |       |        |

`Total` is a clearly secondary summary row. Per-bucket rows are the signal: the comparison across modes is the experiment.

### 9.3 Secondary tables

Same shape (`rows = buckets`, `columns = conditions`), one per metric:

- `hit_rate@5`
- `MRR@20`
- `answer_hit`
- `normalized_f1`
- `gold_citation_precision`
- `gold_citation_recall`
- `claim_support_coverage`
- `cited_unsupported_claim_rate`
- `uncited_claim_rate`
- `out_of_evidence_citation_rate`
- `abstain_rate`
- `parse_error_rate`

Grounding-metric secondary tables additionally print `n_defined` per cell per §6.5.

## 10. Preflight, failure policy, and run status

Three statuses. They are not interchangeable. A run reports exactly one.

### 10.1 `status = "refused"` — fatal preflight failure

The benchmark never begins because a required precondition failed. The runner writes `report.json` with `status = "refused"` (and as much of the metadata section as can be filled) and exits non-zero. Headline metrics are not published. Conditions that cause refusal:

- Invalid or incomplete eval file: schema failure, any record with `supporting_chunk_ids == []`, any record with `split != "held_out"`, `qid` outside `q_NNN`, file unreadable.
- Dirty benchmark-relevant repository state (see §10.4).
- Stale or invalid vector store (§10.2).
- Invalid configuration (missing provider credentials, missing retrieval constants, invalid eval set path).
- Missing required artifacts (chunks manifest, ontology, graph ingestion outputs) for the configured retrieval modes.

The offending `qid`s are listed in the refusal block of `report.json` so the operator can fix the source.

### 10.2 Vector store preflight

The runner requires:

- `chunks_sha256 == embedding_source_chunks_sha256` (the SHA256 of `data/derived/chunks.jsonl` against which the current vector-store embeddings were produced/validated).
- The #10 stale-row validation must pass:
  - every canonical chunk present,
  - no orphan rows,
  - matching embedding model and revision on every row,
  - matching `source_text_sha256` per row.

The runner does not invent a hash over live database row order. If the implementing vector-store ticket later introduces a canonical embedding manifest, its SHA may be added as an additional `report.json` metadata field.

### 10.3 `status = "completed"` — full valid execution

All three retrieval conditions have complete valid retrieval results on all 45 questions. `GenerationParseError` and `ABSTAIN` may still appear; they are recorded per §6. Headline and secondary tables are published.

### 10.4 Repository-state preflight

`git_sha` participates in `run_fingerprint`. Benchmark preflight refuses an execution whose benchmark-relevant repository state differs from `HEAD`:

- no staged tracked modifications,
- no unstaged tracked modifications,
- no untracked files under benchmark-relevant source/config paths that could be imported or consumed by the run.

Local agent-control files such as `AGENTS.md` and `docs/agents/**` are not benchmark inputs and do not invalidate a run. `data/runs/**` is derived output (§12) and does not invalidate a run.

The contract is: **the executed benchmark code and tracked configuration correspond to the recorded `git_sha`.** The runner does not silently run against dirty benchmark code. The implementing ticket may define the exact benchmark-relevant path allowlist/denylist; the contract above is what must hold.

### 10.5 `status = "invalid"` — partial execution, infrastructure failure mid-run

Execution began, but at least one retrieval condition suffered a retrieval infrastructure error (e.g. Neo4j unreachable) on at least one of its 45 questions. The affected condition is **invalid for official aggregation and headline reporting** even if only one of its 45 questions is missing a valid retrieval result: a 44/45 partial result is **not** directly comparable to a complete condition. Conditions that completed cleanly on all 45 questions are still reported as such.

`report.json` records:

- `status = "invalid"`,
- per-mode `mode_status ∈ {"valid", "invalid"}` derived from retrieval-result completeness,
- invalid-condition diagnostics preserved for inspection,
- valid conditions' headline and secondary tables published normally.

There is no percentage threshold. There is no configurable threshold. The rule is the binary "all 45 valid retrieval results" vs "not all 45 valid retrieval results".

### 10.6 Generation-side failure handling

- **`GenerationParseError`** is a recorded model/protocol outcome and the run continues. It is not an infrastructure failure and does not invalidate its condition. It is reported via `parse_error_rate` and the §6.5 null-handling.
- **`ABSTAIN`** is a valid model outcome and the run continues. Reported via `abstain_rate` and §6.4.

### 10.7 Status transition diagram

- Preflight failure → `refused`. No execution. Headline metrics not published.
- Preflight passes, all retrieval conditions complete cleanly on 45/45 → `completed`. Headline metrics published.
- Preflight passes, ≥1 retrieval condition has any incomplete retrieval result → `invalid`. Valid conditions' headline metrics published; invalid conditions excluded.

## 11. Reproducibility claims

The runner does not promise `records.jsonl` is byte-identical across fresh executions; it contains wall-clock durations, timestamps, cache-hit state, and provider-generated outputs. Instead, the spec distinguishes four reproducibility regimes:

- **A. Deterministic retrieval reproducibility.** Same validated retrieval artifacts and config → same ordered retrieval hits and retrieval metrics. The retrievers (#10, #11, #12) are required to be deterministic over their inputs; the benchmark runner records `run_fingerprint` to make "same config" checkable.
- **B. Deterministic scoring reproducibility.** The same recorded per-question inputs/results → byte-identical metric outputs under the same `scoring_version`. All scoring functions in §5 / §6 are closed-form and pure.
- **C. Provider-conditioned generation reproducibility.** Fresh uncached provider calls may differ even with `temperature = 0`. The runner pins `temperature`, `max_tokens`, prompt template SHA, model identity, and `cache_format_version` so a future uncached provider-variance experiment has the inputs it would need; running such an experiment is out of scope for v1.
- **D. Cached generation replay.** Reusing the same valid #16 cache entry yields the recorded response again. Replay is a property of #16; the benchmark runner does not own it.

If a semantic-record hash is desired, it is computed over a canonical projection that explicitly excludes timing/cache-observability fields. The runner does not promise byte-equality of `records.jsonl` across runs.

## 12. Run output and filesystem policy

- Run output lives at `data/runs/<run_id>/` and contains `report.json` and `records.jsonl`. This directory is **benchmark output, not source data**.
- The implementing ticket must ensure this runtime directory is excluded from Git tracking.
- `report.json` and `records.jsonl` carry the run's full provenance (§8.4). Two `run_id`s with identical `run_fingerprint` may be compared cell-for-cell under §11 B; two `run_id`s with different `run_fingerprint` are different experiments.
- The runner does not edit `data/manifest.jsonl`, `data/derived/chunks.jsonl`, `ontology.yaml`, the graph, or the vector store. It only reads.

## 13. What this spec leaves open

These are explicitly post-Wayfinder / implementing:

- **The implementing code.** Pickled retriever/graph/embedding client construction, async/sync choice, CLI shape (`--questions-file`, `--output-dir`, `--mode` overrides, etc.), error-message wording, JSON encoder choice. The implementing ticket picks these within the contract above.
- **A future LLM judge.** If substring + `normalized_f1` prove insufficient and an LLM judge is later added, it is a separate versioned follow-up. The eval-schema `expected_answer` shape already accommodates this; the #14 contract is not reopened.
- **A future uncached provider-variance experiment.** Would consume the pinned config (§8.2) and the recorded `usage_calls`. Not part of v1.
- **A canonical vector-store embedding manifest.** If the implementing vector-store ticket introduces one, its SHA may be added as additional `report.json` metadata (§10.2).

## 14. Acceptance gate

This spec closes when committed to `docs/specs/benchmark-runner.md`. The implementing benchmark-runner ticket closes when:

1. The runner preflights pass on the shipped `eval/questions.jsonl` (§10.1, §10.2, §10.4).
2. All 45 questions complete across all three retrieval conditions with valid retrieval results (§10.3).
3. `report.json` is written with every field in §8.4 populated, including all five spec SHAs and the embedding `revision`.
4. `records.jsonl` contains 135 records (45 × 3 modes) matching the §7 shape.
5. The §9 headline table and all §9.3 secondary tables are published in `report.json`.
6. Two consecutive runs against the same `eval/questions.jsonl` and same config produce identical `run_fingerprint` and identical headline-table cell values per §11 A / B.

---

This spec resolves ticket #14. It is referenced by the post-Wayfinder implementing ticket that builds the runner, and by #1 (the map) which records its resolution. It is the canonical contract for benchmark execution, scoring, and reporting against the eval set (#4) under the three retrieval conditions (#10, #11, #12) and the grounded generator (#13), with the provider abstractions (#16) fixing the cache identity that makes §11 D work.
