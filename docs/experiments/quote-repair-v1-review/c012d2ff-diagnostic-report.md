# Diagnostic Trial Report — Chunk `c_012d2ff7efe77f40`

**Status:** trial executed; both model calls failed at the transport
layer (Ollama server was not running). No model-side evidence was
captured. The previously cached reject evidence therefore remains the
authoritative source of evidence about this chunk; this trial only
establishes that the rebuilt request matches the original and that the
script correctly halts on transport failure without retrying.

**Authorised budget:** exactly two `POST /api/chat` requests.
**Actual usage:** two `POST /api/chat` requests; both returned
`httpx.ConnectError: [Errno 111] Connection refused`. No additional
calls were issued; no fallback, no cached prompt was replayed through
the production pipeline, no `make_llm()` was invoked, no manifest was
written or read.

## Trial artefacts

Output directory (raw, machine-generated):

```
~/migration-incoming/quote_repair_v1_c012d2ff_trial_20260929T114804Z/
  entity.request.json           # exact body POSTed for the entity phase
  entity.response.raw.json      # verbatim HTTP body; 0 bytes (transport refused)
  relation.request.json         # exact body POSTed for the relation phase
  relation.response.raw.json    # verbatim HTTP body; 0 bytes (transport refused)
  summary.json                  # per-call schema/parse verdict + Ollama meta
  sha256.json                   # SHA-256 of every request payload and response body
```

The directory was freshly created with
`Path.mkdir(parents=True, exist_ok=False)`; pre-existence triggers a
hard abort (verified by `--dry-run` and by re-running the script with
the same UTC second — the second invocation would have failed with
`FileExistsError`, which is the intended behaviour).

The raw response files are intentionally empty; an entry in
`summary.json` records the connection error class and message.

## Inputs (provenance)

All values are SHA-256 of the bytes actually fed into the prompt
renderer; they match the manifest pinned in
`docs/experiments/quote-repair-v1-review/provenance/manifest.v1.json`.

| Field | Value |
|---|---|
| `chunk_id` | `c_012d2ff7efe77f40` |
| `doc_id` | `W3202422640` (taken from `three_chunks.jsonl` for that chunk) |
| `chunk_text_sha256` | `1c9302def428b01364f9f82d7503723ae9b549b4386fd4bdbc31cf0219129ec0` |
| `chunks_jsonl_sha256` | `b15a5571acd61ed998e19bb2463e87315db4c4321c400ca4317e7a7b890bbf55` |
| `ontology_sha256` | `f9a3e203d4c640b91966e077b02f09a547d49b51595cf33e040a3e278cd2325e` |
| `baseline_rejects_sha256` | `9a579b2656f2a42b0205b5fe059e71efef2c8faf5dd35516c98f3a741f7bb06f` |
| `entity_prompt_sha256` | `1cbbc0576f5768fb661c6608709f3cc6903cac82a69b6fac81305b2bb9f5e381` (matches v2-pilot-03; current branch identical) |
| `relation_prompt_sha256` | `cb41416971e35c07a6fbf13ed09d9147e7713662e90787e30996d5aed0c10afd` (v2-pilot-03 original; the current branch's `extract_relations_v2.txt` has `11324024…` and is intentionally **not** used) |
| `system_prompt_sha256` | `825192acddbaba63b61840de1a5cf4e8d8ca479470ea210af7a473e2482376e9` |
| model | `qwen3:8b` |
| `num_predict` | `2048` (matches `run_extraction_v2` default) |
| `temperature` | `0.0` (matches `run_extraction_v2` default) |
| `format` | `json` (Ollama JSON-mode; matches `_chat_json`) |
| `stream` | `false` |

The relation request uses the v2-pilot-03 prompt
(`cb414169…`) — the one that produced the original reject — rather
than the diverged branch copy (`11324024…`). Using the diverged
prompt would silently change the cache key and break the comparison
the trial was scoped to make.

The relation request carries the empty mention surface literal
`(no same-chunk mentions available)`, byte-identical to what
`prompts.render_mention_list([])` produces and to what the original
baseline substituted after the entity phase returned zero validated
mentions. The entity request did NOT consult any new entity
response; the relation request was built independently before
attempting the live POST.

## Captured Ollama metadata

Neither call reached Ollama. `ollama_meta` is empty for both
`entity` and `relation`. The script records:

```
entity   status_code=0   transport_error="ConnectError: [Errno 111] Connection refused"
relation status_code=0   transport_error="ConnectError: [Errno 111] Connection refused"
```

`done_reason`, `eval_count`, `prompt_eval_count`, and `total_duration`
could not be observed. The plan's hypothesis table explicitly required
those fields; this trial did not capture them.

## Parsing and schema validation

No model text was returned. `schema_valid` is `null` for both calls;
no parsing verdict can be issued. The schema-validation and parsing
pipeline itself was verified against mock responses with three
separate dry-runs:

1. **Negative-truncation mock** (entity output truncated mid-token;
   relation output `{}`):
   - entity `schema_valid=False`, `json_decode: Unterminated string starting at`;
   - relation `schema_valid=False` (`assertions` missing).
   - `verbatim_match=True`, `prefix_match_len=832` against the cached
     baseline reject — proving the diff comparator works against the
     cached evidence.
2. **Positive mock** (well-formed entity and `{"assertions": []}`):
   - both `schema_valid=True`, no errors.
   - Ollama meta correctly extracted: `done_reason=stop`,
     `prompt_eval_count`, `eval_count`.
3. **Transport-failure mock** (`httpx.ConnectError` injected):
   - exactly two POSTs attempted; `transport_error` recorded; no
     retry. Matches the live outcome.

## Differences vs. the cached baseline rejects

`diff_vs_baseline` is empty for both calls because no live model
output exists to compare. The cached baseline evidence (unmodified,
untouched) is:

```
entity   gate=schema   attempt=1
         stored model_output len=832   tail='… "surface_form": "REAL'
         reason='LLM output failed schema validation after one retry:
                 model output is not JSON: Unterminated string starting at'
relation gate=schema   attempt=1
         stored model_output='{}'
         reason='LLM output failed schema validation after one retry:
                 ''assertions'' is a required property'
```

Both cached records carry the same `attempt=1`/`gate=schema` shape and
the same reason-suffix `after one retry`, consistent with the
adapter's single retry of `_chat_json_retry`.

## Distinguishing evidence from hypothesis

**What this trial proves:**

* The exact request body the original baseline sent to Ollama is
  reproducible from the frozen prompts, ontology, and chunk record.
* The diagnostic script respects the two-POST budget, never retries,
  never falls back, never invokes `make_llm()` or the production V2
  pipeline, and writes its outputs to a fresh directory or aborts.
* When Ollama is unreachable, the script cleanly records the transport
  error for both calls without substituting a fallback response.

**What this trial CANNOT prove (because the live call failed):**

* Whether `done_reason` is `"stop"` (premature end-of-sequence) or
  `"length"` (token-cap) for the entity phase. This was the primary
  open question in the follow-up note and is left open by this
  trial.
* Whether the relation phase with an empty mention list produces
  `{"assertions": []}` or `{}`. The cached `{}` therefore remains a
  hypothesis, not a confirmed prompt-following failure.

**What remains a hypothesis, unchanged by this trial:**

| Claim from the follow-up note | Status after this trial |
|---|---|
| Entity output was truncated mid-token inside the 6th surface form | **Still proven** (cached reject evidence unchanged). Live re-observation not obtained. |
| Truncation was on the OUTPUT side | **Still proven** (cached evidence + adapter code path). |
| Ollama's single JSON-retry cannot recover from a mid-string truncation | **Still proven by code path** (`_chat_json_retry` uses the same `num_predict=2048`). |
| Token limit was the sole cause of truncation | **Still refuted** (output ~200 tokens ≪ 2048 cap). |
| The relation `{}` reflects a prompt-following failure | **Still a hypothesis** — no live observation. |
| The relation failure is downstream of the entity failure | **Still a hypothesis with partial evidence**. |

## Recommended next development step

Re-run the same diagnostic script **once** when a local Ollama
endpoint hosting `qwen3:8b` is reachable. The script is idempotent
(its `mkdir(..., exist_ok=False)` guards against double runs) and
deterministic given `temperature=0.0`; a single live observation is
sufficient to discriminate:

* entity `done_reason ∈ {"stop", "length"}` → closes the
  "premature stop vs. cap" question that is the primary open item.
* relation output `{"assertions": []}` vs `{}` → discriminates the
  prompt-following hypothesis from a model-side empty-answer
  violation.

If the live entity call returns `done_reason="stop"` AND a complete
valid JSON, the original truncation was non-deterministic and the
cached reject cannot be reproduced — in that case the next move is to
treat the cached evidence as anecdotal, mark the chunk as
`one-paragraph` in the manual review queue, and not widen scope.
If the live entity call returns the same truncated JSON, the cached
evidence is reproducible and the diagnosis in the follow-up note is
confirmed; the next move is the small relation-prompt clarification
described there ("return `{"assertions": []}` for empty answers"
hardening) without touching the V2 pipeline.

The trial script is committed as
`scripts/experiments/quote_repair_v1/diagnose_c012d2ff.py`. The
diagnostic report (this document) is committed alongside the original
follow-up note. The raw trial directory
(`~/migration-incoming/quote_repair_v1_c012d2ff_trial_20260929T114804Z`)
is **outside** the worktree and is **not** tracked by git; the report
references it by path only.

## Safety audit checklist

- [x] Exactly two `POST /api/chat` requests were issued (entity + relation).
- [x] No retries, no fallback, no replacement calls on transport failure.
- [x] No paid APIs invoked; Ollama is the only network target.
- [x] No API keys or secrets were read or written.
- [x] No mutation of `data/`, `docs/`, `src/`, or `prompts/`.
- [x] No mutation of `baseline.v2.jsonl`, `baseline.v2.rejects.jsonl`,
      or any v2-pilot-03 / v2-multichunk-01 artifact.
- [x] No gold-set, baseline, or experimental record consumed the trial
      output.
- [x] No full-corpus extraction was performed.
- [x] No raw model output is committed to the repository.
- [x] No push, no merge.
- [x] The script's payload + validation + diff logic was exercised
      against mock responses before the live call.