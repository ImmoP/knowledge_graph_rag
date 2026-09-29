# Bounded Plan: Targeted Model Trial for ``c_012d2ff7efe77f40``

**Status:** plan only — no model call has been issued.
**Scope:** the third chunk from the v2-multichunk-01 replay whose entity
phase was rejected for a hard mid-string truncation and whose relation
phase returned ``{}`` in violation of the v2 schema.

## What the evidence proves vs. what remains a hypothesis

| Claim | Status | Source |
|---|---|---|
| Entity output was truncated mid-token inside the 6th surface form | **Proven** | `baseline.v2.rejects.jsonl` for `c_012d2ff7efe77f40` (phase `entity`, `reason` starts with `"LLM output failed schema validation after one retry: model output is not JSON: Unterminated string"`; stored `model_output` ends with the literal substring `"REAL` and carries zero closing brackets) |
| The truncation occurred on the OUTPUT side of the API call | **Proven** | Chunk `token_estimate=158`, chunk text 634 chars; Ollama request carries `options.num_predict=2048`; 832 chars of model output ≈200 tokens, far below the cap |
| The Ollama adapter's single JSON-retry cannot recover from truncation | **Proven by code path** | `src/providers/adapters/ollama.py` lines 213–235 — `_chat_json_retry` uses the **same** `num_predict`; the retry is text-level ("respond with valid JSON") and inherits the same token cap, so a model that stops emitting after the 5th mention stops the same way twice |
| The transport-retry layer was never invoked | **Proven** | `src/extract/retry.py` — only `ProviderTransportError(transient=True)` qualifies; `_parse_model_json` raises `ValueError` (a non-transient exception) which the ollama adapter translates to `LLMValidationError` |
| The relation phase emitted `{}` because the model ignored the prompt's "return `{assertions: []}` for an empty answer" rule | **Hypothesis** | The cached output is `{}`. The relation prompt *does* tell the model to emit `{"assertions": []}`. We have no raw response-side log (no token counts, no `done_reason`, no model-side reasoning trace) to discriminate prompt-following failure from a downstream effect of the entity failure |
| The relation phase failure is downstream of the entity failure | **Hypothesis, partial evidence** | Pipeline code calls relation phase **after** entity phase unconditionally (`src/extract/pipeline_v2.py` `_run_entity_phase_v2` returns a `PhaseOutcomeV2`; the caller does not gate on `outcome.status == "validated"`). The relation prompt receives an empty mention list when entity yielded zero validated mentions. The cached `chunk_status.phases.relation.attempt_count=1` confirms the call was made; the cached relation output being `{}` is consistent with a model that, given zero mention endpoints, declined to emit the explicit empty-assertion form. **We do not have a second cached run that establishes causality.** |
| Token-limit (num_predict) was the cause of the truncation | **Refuted as the sole cause** | 832 chars ≈ 200 output tokens, well below `num_predict=2048`. A pure token-cap cause would require a much longer output; the truncation happens after only 5 complete mentions |

## What a bounded targeted trial MUST do

The next trial is bounded to ``c_012d2ff7efe77f40`` only, single chunk,
single provider (``ollama`` ``qwen3:8b``), and a fixed budget. The
hypothesis under test is the *primary* one we cannot resolve offline:

> *The cached relation output ``{}`` reflects a model-side prompt-following
> failure when the entity phase returns zero validated mentions, not a
> schema or transport failure.*

The trial is **only allowed** if:

1. The reviewer explicitly opts in to a single ``POST /api/chat`` call
   per phase (entity, then relation) for this one chunk.
2. The output of every call is captured verbatim with `prompt_eval_count`
   and `eval_count` and the raw body (so we finally have the
   `done_reason` field that the cached rejects lack).
3. The trial does NOT mutate any committed artefact under
   ``data/``, ``docs/``, ``src/``, or ``prompts/``.
4. The trial does NOT feed any result into the existing baseline or
   experimental record. Its output is written to a fresh, timestamped
   path and referenced only from the report it produces.
5. No new model or network call is permitted beyond the two above.

### Trial shape (when opted in)

For each phase, capture:

* the verbatim response body (so we can see whether `done_reason` is
  `"stop"` or `"length"`; offline analysis cannot recover this);
* `prompt_eval_count` and `eval_count` (offline rejects carry neither);
* the post-validation trace, including whether the model emits
  `{"assertions": []}` for the relation phase when given an empty
  mention surface;
* the same `num_predict` (2048) as the original pilot, to keep the
  comparison meaningful;
* the identical prompt templates (no prompt edits) — only the same
  provider, model, parameters.

### Variables the trial can isolate

| Variable | What it answers |
|---|---|
| Pass `format=null` (drop Ollama JSON-mode) | Was JSON-mode the cause of premature stop? |
| Pass `format="json"` (current setting) | Reproduces the cached failure mode |
| Run relation phase with an explicit empty mention list | Does the model emit `{"assertions": []}` when prompted, or fall back to `{}`? |

### Variables the trial CANNOT isolate without scope expansion

* Chunk content — cannot be changed in a single-chunk offline replay.
* Provider/model — pinned to match the cache key for cross-artefact
  identity.
* Prompt text — would break the deterministic cache key.

### Acceptance criterion for the trial

* If the model emits a complete entity JSON under the same conditions
  → the prior failure was non-deterministic and the trial may continue
  to the relation phase.
* If the model emits the same truncated entity JSON AND a `done_reason`
  of `"stop"` → the cause is a model-side stop token emitted mid-list,
  not a transport or token-cap issue. The relation phase must then be
  run with an empty validated mention list to test the prompt-following
  hypothesis.
* If the model emits the same `{}` for the relation phase when given
  an empty mention list → the prompt-following hypothesis is
  **rejected**: the failure is structural (the model ignores the empty
  answer rule). The relation prompt would need a small clarification
  or the relation phase must be skipped when the entity phase yields
  zero mentions.
* Any other outcome must be reported verbatim and stop the trial.

## What this plan does NOT propose

* No edit to the prompts, schema, ontology, extraction contract, or
  V2 gates.
* No edit to the existing ``baseline.v2.jsonl``,
  ``baseline.v2.rejects.jsonl``, or any cached record.
* No human review labels on any of the three existing
  ``experimental_whitespace_v1`` review entries.
* No generalisation claim: a single chunk under ``c_012d2ff7efe77f40``
  cannot establish chunking, prompt, or model behaviour for the wider
  corpus.

## Reviewer's checklist before opting in

- [ ] Two API calls is acceptable (one entity, one relation).
- [ ] No reuse of any output as gold, baseline, or experimental record.
- [ ] The trial report will be written next to this plan and will
      reference only the cached evidence plus the fresh capture.
- [ ] If the trial surfaces a non-deterministic stop, the plan will
      *not* silently widen into a chunking change.