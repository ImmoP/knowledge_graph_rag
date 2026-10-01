# Model-Extraction Benchmark

Reproducible extraction benchmark on the canonical V2 entity/relation
pipeline. It supports the existing fixed-model comparison (`--manifest`
for GPT-5.6 Terra, plus the legacy local `--qwen` mode) and a separate
Manifest routing-system benchmark (`--manifest-auto`). The benchmark is
offline-first; live calls require an explicit mode switch and remain
subject to a hard per-process call cap.

## What it compares

* Same chunks, prompts, ontology, schema, parameters.
* Two pinned model identifiers:

  * `qwen3:8b` — local Ollama (no provider key needed).
  * `openai/gpt-5.6-terra-subscription` — Manifest gateway
    (`MANIFEST_API_KEY` only).
* Fixed `--manifest`: the gateway-reported model is captured next to
  the requested Terra identifier; a mismatch aborts the call
  (`ModelMismatchError`).
* `--manifest-auto`: sends `model="auto"` verbatim and lets Manifest
  choose provider/model independently for every request. Each entity and
  relation metric records `requested_model: "auto"`; `reported_model` is
  the response-level identifier when present, otherwise `"<unknown>"`.
  JSON-correction attempts are retained under
  `extra.attempt_reported_models`. Auto results are system-routing
  results, not single-model results.
* Auto mode disables the response cache to prevent responses produced by
  different routed models sharing an `auto` cache key.
* Each call is charged against a shared live-call counter; the
  smoke comparison refuses to exceed four manifest calls and two
  local Qwen calls.

Neither mode silently falls back in the adapter. Fixed mode retains
strict Terra identity matching; only auto mode accepts Manifest's routed
identity. The benchmark never logs the bearer secret and never writes
raw responses into git.

## Layout

| Path | Purpose |
|---|---|
| `src/benchmark/dataset/benchmark_chunks.jsonl` | Frozen 40-chunk set over 9 corpus documents (prose / LaTeX / tables; short / medium / long). |
| `src/benchmark/dataset/benchmark_chunks.sha256` | SHA-256 of the JSONL bytes; verified by every load. |
| `src/benchmark/dataset/benchmark_dataset_manifest.json` | Selection algorithm version, seed, configuration, per-tag and per-tercile counts. |
| `src/benchmark/dataset/smoke_chunks.py` | Pinned smoke chunk ids + call caps. |
| `src/benchmark/runner.py` | Thin wrapper that drives the canonical V2 pipeline and records per-phase metrics. |
| `src/benchmark/evaluate.py` | Per-chunk / aggregate technical metrics + blinded review-package generator. |
| `scripts/benchmark/build_dataset.py` | Builds the frozen dataset deterministically from a corpus JSONL. |
| `scripts/benchmark/run_smoke.py` | Two-chunk limited live comparison; refuses without an explicit switch. |
| `scripts/benchmark/evaluate.py` | Consumes a run directory and emits metrics + blinded review. |
| `src/providers/adapters/manifest_openai.py` | Experimental adapter for the Manifest gateway. NOT registered in the production provider registry. |
| `src/benchmark/offline_diagnostics.py` | Optionaler, rein lesender Shadow-Bericht über verifizierte Recording-Artefakte. |
| `tests/providers/test_manifest_benchmark_adapter.py` | Adapter mock tests (success / 401 / 429 / 503 / schema fail / model-id mismatch / call cap / cache isolation / no-secret-log). |
| `tests/benchmark/test_dataset.py` | Deterministic selector tests. |
| `tests/benchmark/test_runner_smoke.py` | Runner and smoke-script tests. |
| `tests/benchmark/test_evaluate.py` | Evaluator and blinded-review tests. |

## Reproducing

### 0. Pull the frozen dataset

The dataset is committed under `src/benchmark/dataset/`. Re-running
the build from the same corpus must produce a byte-identical
selection:

```bash
PYTHONPATH=. python scripts/benchmark/build_dataset.py \
    --chunks /path/to/chunks.jsonl \
    --output-dir src/benchmark/dataset/
```

The output is checked in; subsequent runs must reproduce the same
SHA-256 (`38182c6cdbb9f86abc6ce20ece7a3d9e1cb69ccac2576e2d6274e727c2f84985`)
or fail closed.

### 1. Mock tests (offline, no network)

```bash
PYTHONPATH=. python -m pytest \
    tests/providers/test_manifest_benchmark_adapter.py \
    tests/benchmark/ \
    -v
```

This runs 17 + 33 = 50 unit tests and proves the entire wiring is
correct without any live model call.

### 2. Dry-run smoke (offline)

```bash
PYTHONPATH=. python scripts/benchmark/run_smoke.py \
    --dry-run \
    --output-dir /tmp/bench-runs/
```

Produces an empty summary under `/tmp/bench-runs/<run-id>/`. No
network. No local Ollama call. No `MANIFEST_API_KEY` required.

### 3. Local Qwen smoke (live, capped)

Requires a running local Ollama daemon with `qwen3:8b` available.

```bash
PYTHONPATH=. python scripts/benchmark/run_smoke.py \
    --qwen \
    --output-dir /tmp/bench-runs/
```

Refuses to exceed `--allow-local-qwen-calls` live calls (default 2
for the two-chunk smoke: one entity + one relation per chunk, two
chunks).

### 4. Live manifest smoke (live, capped, requires key)

```bash
export MANIFEST_API_KEY=...
PYTHONPATH=. python scripts/benchmark/run_smoke.py \
    --manifest \
    --output-dir /tmp/bench-runs/
```

Refuses to exceed `--allow-manifest-calls` live calls (default 4).
The adapter enforces the cap independently via `CallBudgetExceeded`.

### 5. Manifest auto-routing smoke (live, capped, no Ollama)

```bash
MANIFEST_API_KEY="$MANIFEST_API_KEY" PYTHONPATH=. \
python scripts/benchmark/run_smoke.py \
    --manifest-auto \
    --allow-manifest-calls 4 \
    --output-dir /tmp/bench-runs/manifest-auto
```

This mode sends exactly `model="auto"`, disables the model-response
cache, and never constructs the Ollama adapter. The cap counts actual
HTTP POST attempts, including V2 transport retries and the adapter's JSON
correction attempt. A refused fifth attempt exits with code `3` after
writing `metrics.jsonl` and `run_summary.json` as a partial report.

Evaluate the produced run without assigning unknown routes to a model:

```bash
PYTHONPATH=. python scripts/benchmark/evaluate.py \
    --run /tmp/bench-runs/manifest-auto/<run-id> \
    --provider-label manifest-auto-routing \
    --chunks src/benchmark/dataset/benchmark_chunks.jsonl \
    --output-dir /tmp/bench-runs/manifest-auto-eval/
```

Inspect `run_summary.json` first. `provider.reported_model` intentionally
stays `null` because one run may span multiple routed models; per-phase
identities live in `metrics[].reported_model`. Treat `"<unknown>"` as
unattributed and report the run only as `manifest-auto-routing`.

### 6. Combined Qwen + manifest smoke (fixed-model comparison)

```bash
export MANIFEST_API_KEY=...
PYTHONPATH=. python scripts/benchmark/run_smoke.py \
    --qwen \
    --manifest \
    --output-dir /tmp/bench-runs/
```

This is the first limited live comparison the task brief describes.
Total live cap is 6 (2 local + 4 manifest).

### 7. Evaluate a run

```bash
PYTHONPATH=. python scripts/benchmark/evaluate.py \
    --run /tmp/bench-runs/<run-qwen-id> \
    --run /tmp/bench-runs/<run-manifest-id> \
    --provider-label qwen \
    --provider-label manifest \
    --chunks src/benchmark/dataset/benchmark_chunks.jsonl \
    --output-dir /tmp/bench-runs/eval/
```

Produces:

* `metrics_qwen.json` and `metrics_manifest.json` — per-provider
  technical-acceptance metrics.
* `review_records.jsonl` — blinded review package; provider
  identities are hidden behind `review_label`s.
* `review_key.json` — the label-to-provider mapping. The reviewer
  keeps this sealed until the review is complete.

The evaluator never assigns gold labels and never computes
precision/recall; a human review pass is required for scientific
correctness.

### 8. Controlled two-problem-chunk recording and offline diagnosis

The recording CLI keeps the original two smoke chunks when `--chunk-id` is
omitted. An explicit selection must contain **each** of
`c_ca129edd34052284` and `c_8a84177f479aea1f` exactly once; either order is
accepted and retained. Other IDs, duplicates, subsets and dataset drift fail
before HTTP. Even `--dataset` pointing elsewhere must contain the same frozen
dataset bytes with SHA-256
`38182c6cdbb9f86abc6ce20ece7a3d9e1cb69ccac2576e2d6274e727c2f84985`.

`recording.json` embeds `selection.source`, `selection.dataset_sha256` and
ordered `selection.chunk_ids`; `settings.chunk_ids` and the checksummed input
snapshot must agree. The dataset digest identifies the **whole** frozen
dataset, not the selected two-row snapshot. Replay verifies both identities.

Use the same Python environment for recording and replay: dependency versions
and implementation file hashes are frozen. On the Ubuntu workstation the
existing environment below contains the required packages; no installation or
download is required. Run A, B and C in the same Bash shell:

```bash
cd /home/immo/migration-incoming/model-benchmark-exp
PY=/home/immo/projects/knowledge_graph_rag/.venv/bin/python
EXP=scripts/experiments/two_chunk_20260930T120000Z
umask 077
RUN_ROOT=$(mktemp -d "$HOME/manifest-two-chunk.XXXXXXXX")
printf 'Private experiment root: %s\n' "$RUN_ROOT"
```

**A — offline preflight and focused tests (no key needed):**

```bash
env -u MANIFEST_API_KEY PYTHONPATH=. "$PY" "$EXP/record_two_chunk.py" \
  --recordings-root "$RUN_ROOT" --run-id preflight \
  --allow-manifest-calls 4
env -u MANIFEST_API_KEY PYTHONPATH=. "$PY" -m pytest -q \
  tests/benchmark tests/providers/test_manifest_benchmark_adapter.py \
  tests/test_extract_v2_pipeline.py tests/test_extract_align.py \
  tests/test_experimental_quote_repair.py tests/test_quote_repair_comparison.py
```

Preflight verifies dataset, chunk selection, ontology, prompt artifacts,
schemas and recording dependency identity. It creates only private
`preflight/preflight.json`, with zero POSTs; never a fake `recording.json`.
The tests use technical mock transport responses, not scientific gold labels.

**B — explicitly enabled live recording, at most four actual POST attempts:**

```bash
read -rsp 'MANIFEST_API_KEY: ' MANIFEST_API_KEY; printf '\n'
export MANIFEST_API_KEY
if PYTHONPATH=. "$PY" scripts/benchmark/manifest_recording.py record \
  --live --allow-manifest-calls 4 \
  --chunk-id c_ca129edd34052284 \
  --chunk-id c_8a84177f479aea1f \
  --output-dir "$RUN_ROOT/recording"; then
  LIVE_RC=0
else
  LIVE_RC=$?
fi
unset MANIFEST_API_KEY
printf 'Recording exit code: %s\n' "$LIVE_RC"
```

The dedicated runner also supports `--live --recordings-root "$RUN_ROOT"
--run-id recording --allow-manifest-calls 4` and fixes the same ordered pair.
Do **not** run both live commands: each invocation has its own budget. No
automatic retry, resume or cache exists between invocations.

New recordings freeze their stop policy: the first HTTP 429 stops immediately;
other eligible transport retries and one JSON-correction attempt share the
four-POST budget. The owned HTTP transport has zero hidden retries and no
redirects. Exhausted corrective/schema-invalid output stops as `incomplete`.
A nonzero result or incomplete chunk count is never labelled `complete`.
Canonical V2 validation, evidence alignment and entity IDs are unchanged.

Recording exit codes: `0` complete; `1` validation/setup/integrity failure;
`2` CLI misuse; `3` POST budget exhausted; `4` incomplete/other terminal run;
`5` HTTP 429. Partial raw attempts and available canonical artifacts are kept;
do not rerun automatically after a nonzero exit. A secret-bearing response is
never persisted, and an incomplete durable attempt cannot be replayed.

**C — offline exact replay, isolated whitespace comparison and DPR diagnosis:**

```bash
env -u MANIFEST_API_KEY PYTHONPATH=. "$PY" "$EXP/compare_v2_whitespace.py" \
  --recording-dir "$RUN_ROOT/recording" \
  --output-dir "$RUN_ROOT/replay"
```

This command consumes an existing verified recording; it never generates
model responses. Outputs:

* `replay/extractions.jsonl`, `replay/extractions.rejects.jsonl` and
  `replay/replay_report.json`: exact canonical V2 replay and byte/hash proof.
* `replay/whitespace_comparison/`: separate `experimental-whitespace-v1`
  artifacts and `comparison.json`, using the identical HTTP attempt stream.
* `replay/dpr_diagnostic.json` and `replay/dpr_diagnostic.md`: both arms'
  mentions/spans, literal occurrences, raw referenced IDs, relation quotes,
  rejected candidates and quote containment. Textual alternatives are
  displayed, **never assigned** to an entity.


Optional kann derselbe verifizierte Replay-Lauf einen separaten
Shadow-Diagnosebericht schreiben:

```bash
env -u MANIFEST_API_KEY PYTHONPATH=. "$PY" scripts/benchmark/manifest_recording.py replay \
  --recording-dir "$RUN_ROOT/recording" \
  --output-dir "$RUN_ROOT/replay-with-diagnostics" \
  --offline-diagnostics
```

`offline_diagnostics.json` erfasst erfolgreiche Entity-Leerantworten,
überlappende kanonische Mentions, `canonical_form`-Konflikte sowie die
Provenienz und SHA-256-Identitäten der verifizierten Inputs, Attempts und
kanonischen Artefakte. Der Modus erzeugt keine Provider-Aufrufe, plant keine
Retries und verändert weder Recording noch `extractions.jsonl` oder
`extractions.rejects.jsonl`. Fehlende oder nicht passende Raw-Antworten führen
zum Abbruch ohne Diagnosebericht.

**Historische Recordings bleiben an ihre Originalimplementierung
gebunden.** Replay verifiziert Eingaben, Prompt- und Schema-Snapshots,
Implementierungs-Hashes und Plattform-Provenienz gegen den im
Recording-Manifest eingefrorenen Stand. Bei Abweichung schlägt
`load_recording(...)` mit `RecordingManifestError` fehl. Eine
automatische Migration auf neue Schema-, Prompt- oder
Ontologie-Versionen findet **nicht** statt; Anpassungen sind nur
durch eine neue, separat aufgezeichnete Aufzeichnung möglich. Der
Schattenbericht darf nur eigene technische Beobachtungen liefern,
nie eine nachträgliche Bewertung der historischen Extraktion.

For separate replay and diagnosis commands instead of C:

```bash
env -u MANIFEST_API_KEY PYTHONPATH=. "$PY" scripts/benchmark/manifest_recording.py replay \
  --recording-dir "$RUN_ROOT/recording" --output-dir "$RUN_ROOT/replay-separate" \
  --compare-whitespace
env -u MANIFEST_API_KEY PYTHONPATH=. "$PY" "$EXP/diagnose_dpr.py" \
  --recording-dir "$RUN_ROOT/recording" --replay-dir "$RUN_ROOT/replay-separate"
```

`matched` means faithful replay, **not** a successful original extraction.
Comparison and diagnosis retain `source_recording.status` and `complete`;
missing phases remain missing. Interrupted, missing or tampered raw data fails
closed. All recording/report paths are outside every Git tree, reject symlink
components, and use private files. Replay/comparison require fresh directories;
diagnosis cannot write into its source recording. Historical recordings and
results are not rewritten: replay of an older implementation requires that
older code and recorded environment.

Development verification used socket-blocked Python processes and mock HTTP
only. The broader historical provenance test also requires the existing local
`data/raw/latex/2007.00808.tar.gz`; it is not fetched or manufactured.


## Safety and cost constraints

* **Default mode is strictly offline.** The runner only makes a
  network call when `--manifest` (or `--qwen`) is passed.
* **Live-call caps.** Both adapters enforce a per-process cap and
  raise `CallBudgetExceeded` on overflow. The smoke runner defaults
  are 2 local Qwen and 4 manifest calls; raise or lower these via
  CLI flags.
* **Separated routing modes.** `--manifest` pins Terra and enforces
  response-model equality. `--manifest-auto` sends `auto`, accepts only
  the identity Manifest actually reports, and marks absence as unknown.
  The runner reads `MANIFEST_API_KEY` exclusively; neither benchmark
  adapter is registered in production `make_llm`.
* **No key in logs.** The adapter never logs its bearer token; the
  test suite has a guard for this.
* **No raw responses in git.** All runs land under
  `BENCHMARK_RUNS_DIR` (default `/tmp/bench-runs/`); gitignore
  excludes `data/raw/`, `data/derived/`, and `data/runs/` already.
* **429 / 401 abort.** Transient HTTP errors are retried only by
  the bounded V2 retry policy; every retry consumes the same four-POST
  cap. Non-transient (401) and exhausted-cap errors abort cleanly with a
  non-zero exit code; auto mode saves the partial report first.

## What is not in scope yet

* No gold labels are produced. The blinded review package is the
  scaffold for a later human pass.
* No corpus-wide run. The smoke comparison is bounded to the two
  pinned chunks; the 40-chunk dataset is committed for a later,
  operator-initiated run.
* GPT-5.6 Sol is not wired in yet; the adapter and runner are ready
  for it (`openai/gpt-5.6-sol-subscription` from `~/.omp/agent/models.yml`)
  but no smoke comparison has been written.