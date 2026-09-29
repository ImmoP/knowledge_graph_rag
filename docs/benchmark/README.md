# Model-Extraction Benchmark

Reproducible comparison of the locally-running **Qwen3 8B** against
the Manifest-gateway-hosted **GPT-5.6 Terra** on the canonical V2
entity/relation extraction pipeline. The benchmark is offline-first;
live model calls are gated behind an explicit `--manifest` / `--qwen`
switch and a hard per-process call cap.

## What it compares

* Same chunks, prompts, ontology, schema, parameters.
* Two pinned model identifiers:

  * `qwen3:8b` — local Ollama (no provider key needed).
  * `openai/gpt-5.6-terra-subscription` — Manifest gateway
    (`MANIFEST_API_KEY` only).
* The gateway-reported model is captured next to the requested one;
  a mismatch aborts the call (`ModelMismatchError`).
* Each call is charged against a shared live-call counter; the
  smoke comparison refuses to exceed four manifest calls and two
  local Qwen calls.

The benchmark never uses `auto`-routing, never silently falls back to
a different model, never logs the bearer secret, and never writes raw
responses into git.

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

### 5. Combined Qwen + manifest smoke (first limited live comparison)

```bash
export MANIFEST_API_KEY=...
PYTHONPATH=. python scripts/benchmark/run_smoke.py \
    --qwen \
    --manifest \
    --output-dir /tmp/bench-runs/
```

This is the first limited live comparison the task brief describes.
Total live cap is 6 (2 local + 4 manifest).

### 6. Evaluate a run

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

## Safety and cost constraints

* **Default mode is strictly offline.** The runner only makes a
  network call when `--manifest` (or `--qwen`) is passed.
* **Live-call caps.** Both adapters enforce a per-process cap and
  raise `CallBudgetExceeded` on overflow. The smoke runner defaults
  are 2 local Qwen and 4 manifest calls; raise or lower these via
  CLI flags.
* **No auto / silent fallback.** The adapter pins the model id;
  the runner reads `MANIFEST_API_KEY` exclusively; the registry's
  default `make_llm` still rejects `manifest_benchmark`.
* **No key in logs.** The adapter never logs its bearer token; the
  test suite has a guard for this.
* **No raw responses in git.** All runs land under
  `BENCHMARK_RUNS_DIR` (default `/tmp/bench-runs/`); gitignore
  excludes `data/raw/`, `data/derived/`, and `data/runs/` already.
* **429 / 401 abort.** Transient HTTP errors are retried only by
  the bounded extraction-cache retry; non-transient (401) or
  budget errors abort the run cleanly with a non-zero exit code.

## What is not in scope yet

* No gold labels are produced. The blinded review package is the
  scaffold for a later human pass.
* No corpus-wide run. The smoke comparison is bounded to the two
  pinned chunks; the 40-chunk dataset is committed for a later,
  operator-initiated run.
* GPT-5.6 Sol is not wired in yet; the adapter and runner are ready
  for it (`openai/gpt-5.6-sol-subscription` from `~/.omp/agent/models.yml`)
  but no smoke comparison has been written.