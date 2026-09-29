# Quality metrics derivable from the manual annotation

This document records the quality metrics that can be computed
**after** a human review pass over the two chunks in the
`manifest-auto` smoke run, the formulas used, and the limitations of
each estimate. It complements the per-chunk review markdowns in this
directory and is the contract that
`scripts/experiments/manifest_auto_review_20260929T160008Z/score_manual_annotation.py`
implements.

## 1. What the manual annotation contains

The reviewer writes a JSON file (see `annotations.json` /
`annotations_chunk1.json`) with the following blocks per chunk:

* `mention_labels` — one record per accepted mention from the run,
  each tagged with one of `supported`, `unsupported`, `unclear`,
  `schema_error`.
* `assertion_labels` — one record per accepted assertion from the
  run, same four labels.
* `possible_false_negatives` — a list of mentions or assertions the
  reviewer believes the run *missed* but which are in-scope for the
  ontology. Each entry has a free-text `rationale`.

No gold labels are produced and no gold set is mutated; the
reviewer is the only source of truth.

## 2. Directly derivable precision

For each chunk and for the run's totals, the following precision
numbers are computable from the labelled sets alone (no FNs needed):

```
mention_precision_chunk    = mentions_supported_chunk / mentions_labelled_chunk
assertion_precision_chunk  = assertions_supported_chunk / claims_labelled_chunk

mention_precision_run      = sum(mentions_supported_run) / sum(mentions_labelled_run)
assertion_precision_run    = sum(assertions_supported_run) / sum(claims_labelled_run)
```

The two chunks in this run produce:

| Chunk | mentions_labelled | mentions_supported | mention_precision | assertions_labelled | assertions_supported | assertion_precision |
|---|---|---|---|---|---|---|
| `c_0001921855a8156b` (ColBERT/DC-BERT) | 2 | 2 | **1.000** | 0 | 0 | n/a (no labelled claims) |
| `c_001452fd142ce075` (ANCE table caption) | 4 | 4 | **1.000** | 4 | 2 | **0.500** |
| **Run total** | 6 | 6 | **1.000** | 4 | 2 | **0.500** |

Precision is reported to four decimals; the run-total
`assertion_precision = 0.5` reflects the two
`USES_METHOD` false-positive relation types (DR, ANCE) flagged in the
review of `c_001452fd142ce075`.

## 3. Recall: what is and is not estimable

The run produces a finite accepted set; recall is the share of the
*true* mentions/assertions that landed in that set. Without a held-out
gold set, **recall is not directly observable.** A bounded interval
*can* be derived from the reviewer's labelled set and the flagged
`possible_false_negatives`, under two assumptions:

1. The flagged FN list is **complete** (the reviewer caught every
   miss).
2. `unsupported` and `schema_error` items are **definitively outside**
   the universe of true assertions — they are not in the recall
   denominator.

Under these assumptions, the only remaining uncertainty is how to
treat `unclear` items: each `unclear` may be a true positive (in the
universe) or a false positive (out of the universe). The two
extremes bracket the true recall:

```
# Worst case (every unclear is a false positive; out of the universe):
recall_lower_bound_strict = supported / (supported + |flagged_FNs|)

# Best case (every unclear is a true positive; in the universe):
recall_upper_bound_strict = (supported + |unclear|) / (supported + |unclear| + |flagged_FNs|)
```

If the FN list itself may be incomplete (the reviewer may have
missed some FNs), the absolute ceiling on recall is the trivial
`recall_estimate_upper_bound = 1.0`. The true recall therefore lies
in the intersection of the strict interval and `[0, 1]`. **No
point estimate is published**; downstream consumers must read the
interval and decide how to handle `unclear` items in their own
context.

For this run's `c_001452fd142ce075` chunk there are no `unclear`
items in the labelled set, so the strict interval collapses to a
single point:

```
supported_assertions      = 2
unclear_assertions        = 0
possible_FN_assertions    = 1   (REPORTS_METRIC(W3038572442 -> Answer Coverage))

recall_lower_bound_strict = 2 / (2 + 0 + 1) = 0.6667
recall_upper_bound_strict = 2 / (2 + 0 + 1) = 0.6667   (collapses; no unclear items)
recall_estimate_supported = True
recall_interval_includes_unclear = False
recall_estimate_upper_bound      = 1.0   (FN list may be incomplete)
```

F1 is reported the same way, as `f1_lower_bound_strict` and
`f1_upper_bound_strict`, using the lower / upper recall bounds with
the assertion precision.

For `c_0001921855a8156b` there are no labelled claims in the run
(the accepted assertions list is empty), so `supported == 0` and the
recall interval is undefined: there are no true positives to anchor
the lower bound, and the helper cannot publish a paired upper
bound. The scorer reports `recall_estimate_supported: false` and
leaves both strict bounds as `null`. The flagged
`possible_false_negatives` are *still recorded* in the JSON output
so the reviewer can later revisit the borderline `USES_METHOD`
edges against a future gold set; they do not contribute to a
numeric recall estimate here.

### Why a recall *point estimate* is still not justified

* The reviewer may have *missed* a false negative. The FN list is
  not guaranteed complete. This is why
  `recall_estimate_upper_bound` stays at `1.0` even when the
  strict interval is narrower.
* The reviewer may have *hallucinated* a false negative. The FN
  list is not guaranteed pure. This is why the strict lower bound
  assumes only `supported` items are TP.
* The reviewer may have over- or under-classified an item as
  `unclear`. The strict interval's width encodes this uncertainty.
* Without an independently-labelled gold set, there is no way to
  distinguish "missed" from "hallucinated" entries in the FN list
  or to pin down the true recall within the strict interval.

## 4. Other metrics computable from the manual annotation


* **Per-relation-type precision**: break the `assertion_precision`
  computation down by `relation_type`. For this run:
    * `EVALUATES_ON` (2 claims, both supported) — **1.000**.
    * `USES_METHOD` (2 claims, both unsupported) — **0.000**.
* **Per-entity-type precision**: same shape for `mention_precision`
  split by `entity_type`.
    * `dataset` (2 mentions, both supported) — **1.000**.
    * `method` (2 mentions, both supported) — **1.000**.
    * `metric` (0 mentions in accepted set, 1 missed) — undefined
      on accepted-only; FNR-eligible.
* **Mention FN rate per chunk**: `|flagged_FN_mentions| /
  (mentions_supported + |flagged_FN_mentions|)`. For
  `c_001452fd142ce075`: `1 / (4 + 1) = 0.20`. For
  `c_0001921855a8156b`: `0 / 2 = 0.00`.
* **Unsupported-by-type distribution**: for each label in
  `{unsupported, unclear, schema_error}`, count by
  `relation_type` / `entity_type`. This surfaces which ontology
  edges the provider is over-using. For this run: both unsupported
  assertions are `USES_METHOD`.

## 5. What cannot be derived from one annotated run

* **Cross-chunk or cross-run recall** — needs a gold set.
* **Calibration of the reviewer's FN list** — needs an independent
  audit pass.
* **Provider identity effects** — this run used only the
  `manifest-auto` provider; comparison against `qwen` / `manifest`
  would need their own runs.
* **Stability across chunks** — one chunk per doc-id is not a
  statistical sample.

## 6. Reproducing these numbers

```
PYTHONPATH=. python \
  scripts/experiments/manifest_auto_review_20260929T160008Z/score_manual_annotation.py \
  --annotations docs/experiments/manifest_auto_review_20260929T160008Z/annotations.json \
  --output /tmp/manifest-auto-scores-chunk2.json
```

```
PYTHONPATH=. python \
  scripts/experiments/manifest_auto_review_20260929T160008Z/score_manual_annotation.py \
  --annotations docs/experiments/manifest_auto_review_20260929T160008Z/annotations_chunk1.json \
  --output /tmp/manifest-auto-scores-chunk1.json
```

Both runs are offline; the helper does no I/O beyond reading the
input JSON and writing the output JSON. No LLM calls, no gold-set
mutations, no auto-labels.

## 7. Companion files

| File | Purpose |
|---|---|
| `review_chunk_c_0001921855a8156b.md` | Per-mention / per-assertion review for the ColBERT/DC-BERT figure caption chunk. |
| `review_chunk_c_001452fd142ce075.md` | Per-mention / per-assertion review for the ANCE table caption chunk (where the FN on `Answer Coverage` and the two `USES_METHOD` false-positive relation types are documented). |
| `annotations.json` | Human-labelled `supported`/`unsupported`/`unclear`/`schema_error` records for chunk `c_001452fd142ce075` plus the `possible_false_negatives` list. |
| `annotations_chunk1.json` | Same shape for chunk `c_0001921855a8156b`. |
| `quality_metrics.md` | This file. |
| `../scripts/.../score_manual_annotation.py` | Offline scorer that consumes the two annotation files and emits the precision / recall-bound numbers above. |