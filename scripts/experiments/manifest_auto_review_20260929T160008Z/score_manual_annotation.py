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
        "assertion_F1_lower_bound": <float or null>
      },
      "possible_false_negatives": [...],
      "labelset_version": "manual-review/v1"
    }

Recall is intrinsically unobservable from a single annotated run:
we can lower-bound it (assuming every flagged false negative is a
true miss and the list is complete) and upper-bound it (assuming
the flagged list is pure but possibly incomplete). The script
reports both bounds and refuses to publish a point estimate.
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
    fn_assertions = sum(1 for fn in fns if fn.get("kind") == "assertion")
    fn_mentions = sum(1 for fn in fns if fn.get("kind") == "mention")

    # Recall is only lower-bounded when *both* the run produced some
    # supported assertions *and* the reviewer flagged some possible
    # false negatives. If either is missing, the denominator is zero
    # or the ratio is vacuous, and the script reports
    # ``recall_estimate_supported = False`` to make that explicit.
    if fn_assertions > 0 and supported_assertions > 0:
        recall_lower = supported_assertions / (
            supported_assertions + fn_assertions
        )
        recall_supported = True
    else:
        recall_lower = None
        recall_supported = False

    if (
        recall_lower is not None
        and supported_assertions > 0
        and assertion_precision > 0
    ):
        f1 = (
            2 * assertion_precision * recall_lower
            / (assertion_precision + recall_lower)
        )
    else:
        f1 = None

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
            round(f1, 4) if f1 is not None else None
        ),
    }

    # Sanity: lower bound cannot exceed upper bound.
    if (
        summary["recall_estimate_lower_bound"] is not None
        and summary["recall_estimate_lower_bound"] > 1.0
    ):
        raise ValueError(
            "recall lower bound > 1.0; check annotation counts"
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