#!/usr/bin/env python3
"""Offline scorer for the manual annotation of the manifest-auto run.

Consumes the ``annotations.json`` written by a human reviewer (one
record per mention/assertion plus an optional list of "possible false
negative" mentions/assertions the reviewer thinks the run missed) and
emits the precision / F1 numbers that the review markdown documents
as derivable.

This script is **deliberately offline**:
* no LLM calls
* no edits to gold sets
* no labels assigned by the script -- every label in the input was
  written by a human reviewer
* no automated scientific labels -- the scorer only aggregates labels
  the human has already committed

Usage::

    python scripts/experiments/manifest_auto_review_20260929T160008Z/score_manual_annotation.py \
        --annotations docs/experiments/manifest_auto_review_20260929T160008Z/annotations.json \
        --output /tmp/scores.json

Input schema (JSON object)::

    {
      "review_label": "rev_manifest_auto_54dfd813",   # any opaque string
      "chunk_id": "c_001452fd142ce075",
      "doc_id": "W3038572442",
      "mention_labels": [
        {"mention_id": "m_...", "label": "supported"},
        ...
      ],
      "assertion_labels": [
        {"assertion_id": "a_...", "label": "supported"},
        ...
      ],
      "possible_false_negatives": [
        {"kind": "mention", "missing": {"canonical_form": "...",
                                         "entity_type": "..."},
         "rationale": "..."},
        {"kind": "assertion", "missing": {"relation_type": "...",
                                          "subject_doc_id": "...",
                                          "object_kind": "mention|metric",
                                          "object_id": "..."},
         "rationale": "..."}
      ]
    }

Output schema (JSON object)::

    {
      "summary": {
        "mentions_supported": <int>,
        "mentions_unsupported": <int>,
        "mentions_unclear": <int>,
        "mentions_schema_error": <int>,
        "mention_precision": <float>,
        "assertion_precision": <float>,
        "recall_estimate_supported": <bool>,
        "recall_estimate_lower_bound": <float or null>,
        "recall_estimate_upper_bound": <float>,
        "assertion_F1_lower_bound": <float or null>,
        # NEW (correction of the recall-interval math, see below):
        "recall_lower_bound_strict": <float or null>,
        "recall_upper_bound_strict": <float or null>,
        "recall_interval_includes_unclear": <bool>,
        "f1_lower_bound_strict": <float or null>,
        "f1_upper_bound_strict": <float or null>
      },
      "possible_false_negatives": [...],
      "labelset_version": "manual-review/v1"
    }

Recall is intrinsically unobservable from a single annotated run.
Given the labelled sets and the reviewer's flagged FN list, the true
recall value lies in the *interval*

    [recall_lower_bound_strict, recall_upper_bound_strict]

where both bounds assume the FN list is complete and ``unsupported``
/ ``schema_error`` items are definitively outside the universe of
true assertions. The bounds differ only in how they treat
``unclear`` items, which the reviewer could not fully verify:

* ``recall_lower_bound_strict = supported / (supported + flagged_FNs)``
  -- assumes every ``unclear`` item is a false positive (not in the
  true universe of assertions). This is the smallest the recall can
  be under the completeness assumption.
* ``recall_upper_bound_strict = (supported + unclear) /
  (supported + unclear + flagged_FNs)`` -- assumes every ``unclear``
  item is a true positive (in the universe). This is the largest the
  recall can be under the completeness assumption.
* ``recall_estimate_lower_bound`` and ``recall_estimate_upper_bound``
  are the legacy fields. ``recall_estimate_lower_bound`` is kept as
  an alias for ``recall_lower_bound_strict`` for downstream
  consumers that read the original output schema; ``recall_estimate_upper_bound``
  is kept at ``1.0`` because the FN list may itself be incomplete,
  so the absolute ceiling is the trivial bound.

Without an independently-labelled gold set, neither bound is a
point estimate -- the script reports both bounds and refuses to
publish a single number. F1 is reported the same way, as
``f1_lower_bound_strict`` / ``f1_upper_bound_strict`` using the
two recall bounds.

The script distinguishes **three recall scenarios** explicitly,
because the prior output schema conflated them under the label
``recall_lower_bound``:

1. **Conditional recall (FN-list completeness assumed).** Both
   ``recall_lower_bound_strict`` and ``recall_upper_bound_strict``
   hold only if *every* false-negative assertion in the chunk has
   been flagged in ``possible_false_negatives``. They differ only
   in how they treat ``unclear`` items -- the lower bound assumes
   they are FP, the upper bound assumes they are TP.

2. **Independent-review-confirmed metrics.** ``mention_precision``
   and ``assertion_precision`` are computed only from labelled
   records and are therefore point estimates, not bounds. The
   strict F1 interval ``[f1_lower_bound_strict, f1_upper_bound_strict]``
   is conditional on the same FN-list completeness as scenario 1
   and therefore inherits the same caveat.

3. **Recall with an unknown-completeness FN list.** As soon as the
   reviewer flags at least one false negative, additional unflagged
   FNs cannot be ruled out. The script therefore publishes a
   second interval ``[recall_with_unknown_fn_lower_bound,
   recall_with_unknown_fn_upper_bound]`` whose **only** assumptions
   are non-negativity of counts and the trivial recall upper bound
   ``1.0``. The lower bound collapses to ``0.0`` and the upper bound
   stays at ``1.0`` -- both are the only values defensible without
   a complete reference annotation.

A boolean ``recall_lower_bound_is_conditional_on_complete_fn_list``
is published alongside the legacy fields so consumers can detect
when ``recall_lower_bound_strict`` is unsafe to interpret as a
generic lower bound. The new fields never overwrite the legacy
ones; existing dashboards continue to work unchanged.

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ALLOWED_LABELS = {"supported", "unsupported", "unclear", "schema_error"}


def _label_counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = {label: 0 for label in ALLOWED_LABELS}
    for rec in records:
        label = rec.get("label")
        if label not in ALLOWED_LABELS:
            raise ValueError(
                f"unknown label {label!r} for {key}={rec.get(key)!r}; "
                f"allowed={sorted(ALLOWED_LABELS)}"
            )
        counts[label] += 1
    return counts


def _recall_bounds(
    *,
    supported: int,
    unclear: int,
    fn_assertions: int,
) -> tuple[float | None, float | None, bool]:
    """Return the strict recall interval given a labelling.

    The true recall lies in
    ``[supported/(supported + fn), (supported+unclear)/(supported+unclear+fn)]``
    under the assumption that the FN list is complete and that
    ``unsupported`` / ``schema_error`` items are definitively outside
    the universe of true assertions. The two bounds differ only in
    how they treat ``unclear`` items -- the lower bound assumes they
    are FP, the upper bound assumes they are TP.

    Returns ``(lower, upper, well_defined)``: ``well_defined`` is True
    iff the FN list is non-empty and at least one TP (``supported``)
    is present. When ``well_defined`` is False, both bounds are
    ``None`` -- the denominator collapses or the numerator is zero.
    """
    if fn_assertions <= 0 or supported <= 0:
        return None, None, False
    lower = supported / (supported + fn_assertions)
    upper_denom = supported + unclear + fn_assertions
    upper = (supported + unclear) / upper_denom if upper_denom > 0 else None
    includes_unclear = unclear > 0
    return lower, upper, includes_unclear


def _f1(precision: float, recall: float | None) -> float | None:
    if recall is None or precision <= 0:
        return None
    return 2 * precision * recall / (precision + recall)


def score(annotation: dict[str, Any]) -> dict[str, Any]:
    """Score a single annotation file."""
    mention_labels = annotation.get("mention_labels", [])
    assertion_labels = annotation.get("assertion_labels", [])
    fns = annotation.get("possible_false_negatives", [])

    mc = _label_counts(mention_labels, "mention_id")
    ac = _label_counts(assertion_labels, "assertion_id")

    mentions_labelled = sum(mc.values())
    assertions_labelled = sum(ac.values())
    mention_precision = (
        mc["supported"] / mentions_labelled if mentions_labelled else 0.0
    )
    assertion_precision = (
        ac["supported"] / assertions_labelled if assertions_labelled else 0.0
    )

    supported_assertions = ac["supported"]
    unclear_assertions = ac["unclear"]
    fn_assertions = sum(1 for fn in fns if fn.get("kind") == "assertion")
    fn_mentions = sum(1 for fn in fns if fn.get("kind") == "mention")

    recall_lower, recall_upper, recall_interval_includes_unclear = (
        _recall_bounds(
            supported=supported_assertions,
            unclear=unclear_assertions,
            fn_assertions=fn_assertions,
        )
    )
    recall_supported = recall_lower is not None

    # Legacy alias kept for downstream consumers that read the original
    # output schema: ``recall_estimate_lower_bound`` is identical to
    # ``recall_lower_bound_strict`` (the worst-case bound assuming all
    # ``unclear`` are FP). ``recall_estimate_upper_bound`` stays at
    # ``1.0`` because the FN list may be incomplete -- see the module
    # docstring for the full interpretation.
    f1_lower = _f1(assertion_precision, recall_lower)
    f1_upper = _f1(assertion_precision, recall_upper)

    summary = {
        "mentions_supported": mc["supported"],
        "mentions_unsupported": mc["unsupported"],
        "mentions_unclear": mc["unclear"],
        "mentions_schema_error": mc["schema_error"],
        "mentions_labelled_total": mentions_labelled,
        "mention_precision": round(mention_precision, 4),
        "assertions_supported": ac["supported"],
        "assertions_unsupported": ac["unsupported"],
        "assertions_unclear": ac["unclear"],
        "assertions_schema_error": ac["schema_error"],
        "assertions_labelled_total": assertions_labelled,
        "assertion_precision": round(assertion_precision, 4),
        "possible_false_negatives_total": len(fns),
        "possible_false_negative_assertions": fn_assertions,
        "possible_false_negative_mentions": fn_mentions,
        "recall_estimate_supported": recall_supported,
        "recall_estimate_lower_bound": (
            round(recall_lower, 4) if recall_lower is not None else None
        ),
        "recall_estimate_upper_bound": 1.0,
        "assertion_F1_lower_bound": (
            round(f1_lower, 4) if f1_lower is not None else None
        ),
        # Corrected recall-interval math: the true recall is bracketed
        # by these two strict bounds (see module docstring). The lower
        # bound is the legacy formula, the upper bound is the missing
        # twin. ``recall_interval_includes_unclear`` is True when the
        # labelling has any ``unclear`` items (so the bounds differ).
        "recall_lower_bound_strict": (
            round(recall_lower, 4) if recall_lower is not None else None
        ),
        "recall_upper_bound_strict": (
            round(recall_upper, 4) if recall_upper is not None else None
        ),
        "recall_interval_includes_unclear": recall_interval_includes_unclear,
        "f1_lower_bound_strict": (
            round(f1_lower, 4) if f1_lower is not None else None
        ),
        "f1_upper_bound_strict": (
            round(f1_upper, 4) if f1_upper is not None else None
        ),
    }

    # ---- scenario 3: unknown-completeness FN list ---------------------
    # The strict bounds above assume the reviewer's FN list is
    # exhaustive. The instant at least one FN is flagged, additional
    # unflagged FNs cannot be ruled out, so the only defensible
    # *unconditional* recall interval collapses to [0.0, 1.0]. The
    # numeric values below are derived from the same counts as the
    # strict interval; they are *not* legacy aliases and are
    # published alongside the strict fields so downstream code can
    # decide which scenario to display.
    summary["recall_with_unknown_fn_lower_bound"] = 0.0
    summary["recall_with_unknown_fn_upper_bound"] = 1.0
    summary["recall_lower_bound_is_conditional_on_complete_fn_list"] = True
    # The FN list is *never* provably complete from a single annotation
    # pass; if no FN is flagged, that only means the reviewer did not
    # see one, not that none exist. We therefore always report
    # ``"unknown"`` -- the value is a sentinel, not a state change.
    summary["fn_list_completeness"] = "unknown"
    summary["recall_point_estimate_published"] = False

    # Sanity: lower bound cannot exceed upper bound, and the strict
    # interval must be within the absolute [0, 1] ceiling.
    if (
        summary["recall_lower_bound_strict"] is not None
        and summary["recall_lower_bound_strict"] > 1.0
    ):
        raise ValueError(
            "recall lower bound > 1.0; check annotation counts"
        )
    if (
        summary["recall_upper_bound_strict"] is not None
        and summary["recall_upper_bound_strict"] > 1.0
    ):
        raise ValueError(
            "recall upper bound > 1.0; check annotation counts"
        )
    if (
        summary["recall_lower_bound_strict"] is not None
        and summary["recall_upper_bound_strict"] is not None
        and summary["recall_lower_bound_strict"]
        > summary["recall_upper_bound_strict"] + 1e-9
    ):
        raise ValueError(
            "recall lower bound exceeds strict upper bound; check "
            "annotation counts"
        )

    return {
        "summary": summary,
        "possible_false_negatives": fns,
        "review_label": annotation.get("review_label"),
        "chunk_id": annotation.get("chunk_id"),
        "doc_id": annotation.get("doc_id"),
        "labelset_version": "manual-review/v1",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="path to the human-written annotations JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path to write the scored output JSON",
    )
    args = parser.parse_args(argv)

    if not args.annotations.is_file():
        print(
            f"error: annotations file not found: {args.annotations}",
            file=sys.stderr,
        )
        return 2

    annotation = json.loads(args.annotations.read_text(encoding="utf-8"))
    scored = score(annotation)

    if args.output:
        args.output.write_text(
            json.dumps(scored, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(scored, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
