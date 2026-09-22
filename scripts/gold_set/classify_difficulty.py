"""Deterministic gold-set difficulty classifier (#48 Refinement 5).

This module implements the v1 difficulty classification for the 120-chunk
extraction-contract v2 gold-set pilot. It runs AFTER ``gold_set/chunks/*.json``
is complete (per ``gold_set/GUIDELINES.md``) and BEFORE any candidate-model
evaluation against the 120 chunks. Difficulty is a deterministic function of
the chunk text and the canonical gold annotations; candidate-model outputs
never enter the classification.

The script reads:

* ``gold_set/chunks/<chunk_id>.json`` — canonical ``gold-record.v1`` records.
* ``data/gold_set/sample.v2.jsonl`` — v2 sample rows, carrying
  ``strata.paper_tier``.
* ``data/gold_set/sample.v2.audit.json`` — sample audit with the SHA256 pins
  that must equal the inputs at run time.

It writes two artifacts under ``data/gold_set/`` (both gitignored):

* ``difficulty.v1.jsonl`` — one JSON record per chunk with the per-chunk
  classification and the per-trigger flags. This is **provisional / dry-run**
  output for this slice; the pilot runner that consumes it does not exist
  yet, and the file MUST NOT be treated as a frozen selection input until
  the four spec gaps documented below are resolved under a bumped
  ``classifier_version``.
* ``difficulty.v1.audit.json`` — runtime audit with the classifier script
  SHA256, classifier version, per-stratum counts, and observed spec gaps.

The classification rule (verbatim from #48 Refinement 5):

    A chunk is ``easy`` iff ALL of the following hold:
      - It contains >= 2 unique entity surface forms.
      - Its entities all have a single occurrence in ``chunk.text`` (no
        ``occurrence_index`` ambiguity).
      - It contains at least one relation of an extracted type whose both
        endpoints are mentions.
      - It contains no surface form that is also a common noun.

    A chunk is ``hard`` iff ANY of:
      - It contains a surface form with >= 2 occurrences where the relation's
        evidence span must disambiguate.
      - It contains a relation whose ``evidence_quote`` spans a sentence
        boundary.
      - It contains an ``inferred`` or ``hedged`` relation.
      - It contains a relation whose endpoint mention's surface form is a
        common noun used as a method name in the chunk.
      - It is from a ``long_tail`` paper.

    ``easy`` and ``hard`` partition the chunk; everything else is ``medium``.

Partition precedence (documented): because ``hard`` is ``ANY of`` and
``easy`` is ``ALL of``, the two can co-trigger on the same chunk. The Spec's
"easy and hard partition the chunk" invariant forces a deterministic
precedence: any ``hard`` trigger dominates, then all easy conditions, then
``medium``. This precedence is the only one consistent with the Spec.

Spec gaps documented (NOT silently filled):

1. *Common noun list* — Refinement 5 names three examples
   (``attention``, ``model``, ``score``). The v1 minimum is the literal set
   ``{"attention", "model", "score"}``. Any future expansion MUST bump
   ``CLASSIFIER_VERSION`` and is out of scope for this slice.
2. *Sentence boundary detection* — the Spec does not define a segmentation
   algorithm. The v1 heuristic: a boundary is detected iff the
   ``evidence_quote`` contains a hard terminator (``.``, ``?``, ``!``)
   followed by whitespace and another ASCII letter, OR a paragraph break
   (``\\n\\n``). Any future refinement MUST bump ``CLASSIFIER_VERSION``.
3. *Easy rule on "unique entity surface forms"* — applied across all four
   ontology entity types, the only reading that keeps
   ``easy ⊂ (entities ≥ 2)``.
4. *Hard rule on common-noun method endpoint* — applied only to
   ``entity_type == "method"`` endpoint mentions, matching the Spec's
   "used as a method name in the chunk" wording.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gold_set import author  # noqa: E402

# --- Frozen protocol constants ------------------------------------------------

CLASSIFIER_ID = "gold-set-difficulty.v1"
CLASSIFIER_VERSION = 1

DIFFICULTY_LABELS = ("easy", "medium", "hard")
ENTITY_TYPES = ("method", "dataset", "metric", "task")
CONFIDENCE_LABELS = ("stated", "inferred", "hedged")
PAPER_TIER_LONG_TAIL = "long_tail"

# Spec gap 1: common-noun list (Refinement 5 names three examples verbatim).
COMMON_NOUNS: frozenset[str] = frozenset({"attention", "model", "score"})

# Spec gap 2: sentence-boundary heuristic. The Spec does not pin an algorithm;
# this is a v1 minimum that future revisions replace under CLASSIFIER_VERSION.
_SENTENCE_TERMINATORS = ".!?"
_PARAGRAPH_BREAK = "\n\n"
_SENTENCE_BOUNDARY_RE = re.compile(
    r"[.!?][\s\u00A0]+[A-Za-z]"
)

# Output paths. Both are gitignored; the script regenerates them idempotently.
OUTPUT_JSONL = "data/gold_set/difficulty.v1.jsonl"
OUTPUT_AUDIT = "data/gold_set/difficulty.v1.audit.json"


# --- Errors ------------------------------------------------------------------


class DifficultyError(RuntimeError):
    """Raised when the classifier contract is violated."""


# --- Data structures ---------------------------------------------------------


@dataclass(frozen=True)
class Triggers:
    """Per-trigger flags recorded in the output JSONL.

    Each flag captures whether the corresponding hard trigger fired. The
    ``easy_conditions`` field records the four easy sub-conditions as a
    bool tuple so the audit can attribute "medium" classifications to
    whichever easy condition failed.
    """

    has_multi_occurrence_disambiguation: bool
    has_evidence_spanning_sentence_boundary: bool
    has_inferred_or_hedged_relation: bool
    has_common_noun_method_endpoint: bool
    paper_tier_long_tail: bool
    easy_conditions: tuple[bool, bool, bool, bool]

    @property
    def hard_triggers(self) -> tuple[str, ...]:
        triggers: list[str] = []
        if self.has_multi_occurrence_disambiguation:
            triggers.append("multi_occurrence_disambiguation")
        if self.has_evidence_spanning_sentence_boundary:
            triggers.append("evidence_spans_sentence_boundary")
        if self.has_inferred_or_hedged_relation:
            triggers.append("inferred_or_hedged_relation")
        if self.has_common_noun_method_endpoint:
            triggers.append("common_noun_method_endpoint")
        if self.paper_tier_long_tail:
            triggers.append("long_tail_paper")
        return tuple(triggers)

    @property
    def all_easy_conditions(self) -> bool:
        return all(self.easy_conditions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hard_triggers": list(self.hard_triggers),
            "easy_conditions": {
                "unique_surface_forms_ge_2": self.easy_conditions[0],
                "all_single_occurrence": self.easy_conditions[1],
                "at_least_one_mention_mention_relation": self.easy_conditions[2],
                "no_common_noun_surface_form": self.easy_conditions[3],
            },
        }


@dataclass(frozen=True)
class Classification:
    chunk_id: str
    doc_id: str
    paper_tier: str
    difficulty: str
    triggers: Triggers
    gold_record_sha256: str
    canonical_chunks_sha256: str
    ontology_sha256: str
    annotation_guideline_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "paper_tier": self.paper_tier,
            "difficulty": self.difficulty,
            "classifier_version": CLASSIFIER_VERSION,
            "classifier_id": CLASSIFIER_ID,
            "gold_record_sha256": self.gold_record_sha256,
            "canonical_chunks_sha256_at_authoring": self.canonical_chunks_sha256,
            "ontology_sha256": self.ontology_sha256,
            "annotation_guideline_sha256": self.annotation_guideline_sha256,
            "triggers": self.triggers.as_dict(),
        }


# --- SHA256 helpers ----------------------------------------------------------


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_json(payload: Any) -> str:
    return _sha_bytes(_canonical_json(payload))


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


# --- Loaders -----------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_sample(sample_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Return ``({chunk_id: sample_row}, {chunk_id: paper_tier})``.

    Fails closed if any row is missing a chunk_id or paper_tier.
    """
    rows = _load_jsonl(sample_path)
    by_id: dict[str, dict[str, Any]] = {}
    paper_tier_by_id: dict[str, str] = {}
    for row in rows:
        chunk_id = row.get("chunk_id")
        if not isinstance(chunk_id, str):
            raise DifficultyError(f"sample row missing chunk_id: {row!r}")
        if chunk_id in by_id:
            raise DifficultyError(f"sample chunk_id not unique: {chunk_id}")
        strata = row.get("strata")
        if not isinstance(strata, dict):
            raise DifficultyError(f"sample row missing strata: {chunk_id}")
        paper_tier = strata.get("paper_tier")
        if not isinstance(paper_tier, str):
            raise DifficultyError(f"sample row missing strata.paper_tier: {chunk_id}")
        by_id[chunk_id] = row
        paper_tier_by_id[chunk_id] = paper_tier
    return by_id, paper_tier_by_id


def _load_gold_records(chunks_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Load every canonical ``gold-record.v1`` JSON file.

    Sorted by chunk_id for byte-identical output across filesystem orders.
    """
    if not chunks_dir.exists():
        raise DifficultyError(f"gold chunks directory missing: {chunks_dir}")
    pairs: list[tuple[str, Path, dict[str, Any]]] = []
    for candidate in chunks_dir.iterdir():
        if not candidate.is_file() or not candidate.name.endswith(".json"):
            continue
        with candidate.open("r", encoding="utf-8") as fh:
            record = json.load(fh)
        chunk_id = record.get("chunk_id")
        if not isinstance(chunk_id, str):
            raise DifficultyError(f"gold record missing chunk_id: {candidate}")
        if chunk_id != candidate.stem:
            raise DifficultyError(f"gold record chunk_id/filename mismatch: {candidate}")
        pairs.append((chunk_id, candidate, record))
    pairs.sort(key=lambda triple: triple[0])
    return [(path, record) for _, path, record in pairs]


# --- Classification helpers --------------------------------------------------


def _sentence_boundary(quote: str) -> bool:
    """v1 sentence-boundary heuristic (Spec gap 2)."""
    if _PARAGRAPH_BREAK in quote:
        return True
    return _SENTENCE_BOUNDARY_RE.search(quote) is not None


def _collect_unique_surface_forms(record: Mapping[str, Any]) -> set[str]:
    forms: set[str] = set()
    for entity in record.get("entities", []) or []:
        surface = entity.get("surface_form")
        if isinstance(surface, str):
            forms.add(surface)
    return forms


def _all_entities_single_occurrence(record: Mapping[str, Any]) -> bool:
    for entity in record.get("entities", []) or []:
        occurrences = entity.get("occurrences_in_chunk", []) or []
        if len(occurrences) != 1:
            return False
    return True


def _has_mention_mention_relation(record: Mapping[str, Any]) -> bool:
    for relation in record.get("relations", []) or []:
        if not isinstance(relation, dict):
            continue
        subject = relation.get("subject_ref")
        object_ = relation.get("object_ref")
        if not (isinstance(subject, dict) and isinstance(object_, dict)):
            continue
        if subject.get("kind") == "mention" and object_.get("kind") == "mention":
            return True
    return False


def _common_noun_method_endpoint(record: Mapping[str, Any]) -> bool:
    entities_by_id = {
        entity.get("entity_id"): entity
        for entity in record.get("entities", []) or []
        if isinstance(entity, dict) and isinstance(entity.get("entity_id"), str)
    }
    for relation in record.get("relations", []) or []:
        if not isinstance(relation, dict):
            continue
        for endpoint in (relation.get("subject_ref"), relation.get("object_ref")):
            if not isinstance(endpoint, dict):
                continue
            if endpoint.get("kind") != "mention":
                continue
            entity = entities_by_id.get(endpoint.get("id"))
            if entity is None:
                continue
            if entity.get("entity_type") != "method":
                continue
            surface = entity.get("surface_form")
            if isinstance(surface, str) and surface in COMMON_NOUNS:
                return True
    return False


def _multi_occurrence_disambiguation(record: Mapping[str, Any]) -> bool:
    """A chunk has multi-occurrence disambiguation iff at least one entity
    has ``>= 2`` occurrences in ``chunk.text`` AND at least one relation's
    endpoint mention references that surface form (so the relation's evidence
    span must disambiguate via ``occurrence_index``).
    """
    multi_occurrence_surfaces: set[str] = set()
    for entity in record.get("entities", []) or []:
        if not isinstance(entity, dict):
            continue
        occurrences = entity.get("occurrences_in_chunk", []) or []
        if len(occurrences) >= 2:
            surface = entity.get("surface_form")
            if isinstance(surface, str):
                multi_occurrence_surfaces.add(surface)
    if not multi_occurrence_surfaces:
        return False
    entities_by_id = {
        entity.get("entity_id"): entity
        for entity in record.get("entities", []) or []
        if isinstance(entity, dict) and isinstance(entity.get("entity_id"), str)
    }
    for relation in record.get("relations", []) or []:
        if not isinstance(relation, dict):
            continue
        for endpoint in (relation.get("subject_ref"), relation.get("object_ref")):
            if not isinstance(endpoint, dict):
                continue
            if endpoint.get("kind") != "mention":
                continue
            entity = entities_by_id.get(endpoint.get("id"))
            if entity is None:
                continue
            surface = entity.get("surface_form")
            if isinstance(surface, str) and surface in multi_occurrence_surfaces:
                return True
    return False


def _evidence_spans_sentence_boundary(record: Mapping[str, Any]) -> bool:
    for relation in record.get("relations", []) or []:
        if not isinstance(relation, dict):
            continue
        quote = relation.get("evidence_quote")
        if not isinstance(quote, str):
            continue
        if _sentence_boundary(quote):
            return True
    return False


def _inferred_or_hedged_relation(record: Mapping[str, Any]) -> bool:
    for relation in record.get("relations", []) or []:
        if not isinstance(relation, dict):
            continue
        confidence = relation.get("confidence")
        if confidence in ("inferred", "hedged"):
            return True
    return False


def classify(record: Mapping[str, Any], paper_tier: str) -> Triggers:
    """Compute the trigger flags for a single gold record.

    The function is pure: identical inputs ⇒ identical output, with no I/O
    or hidden state. The classification decision itself is made by
    :func:`decide` from the trigger flags, applying the documented
    hard-easy-medium precedence.
    """
    surface_forms = _collect_unique_surface_forms(record)
    easy_unique = len(surface_forms) >= 2
    easy_single_occ = _all_entities_single_occurrence(record)
    easy_mention_mention = _has_mention_mention_relation(record)
    easy_no_common_noun = not (
        surface_forms & COMMON_NOUNS
    )

    triggers = Triggers(
        has_multi_occurrence_disambiguation=_multi_occurrence_disambiguation(record),
        has_evidence_spanning_sentence_boundary=_evidence_spans_sentence_boundary(record),
        has_inferred_or_hedged_relation=_inferred_or_hedged_relation(record),
        has_common_noun_method_endpoint=_common_noun_method_endpoint(record),
        paper_tier_long_tail=paper_tier == PAPER_TIER_LONG_TAIL,
        easy_conditions=(
            easy_unique,
            easy_single_occ,
            easy_mention_mention,
            easy_no_common_noun,
        ),
    )
    return triggers


def decide(triggers: Triggers) -> str:
    """Apply hard > easy > medium precedence to the trigger flags."""
    if triggers.hard_triggers:
        return "hard"
    if triggers.all_easy_conditions:
        return "easy"
    return "medium"


# --- Top-level pipeline ------------------------------------------------------


def _validate_record_top_level(record: Mapping[str, Any]) -> None:
    """Minimal structural check for the fields the classifier needs.

    The full ``author.validate_record`` would require a ``Paths`` object
    with the held-out leakage artifact and audit hash equality, neither of
    which is the classifier's concern. We check only the fields that
    participate in the classification decision so that the classifier
    remains robust against partial / draft records during development.
    """
    required = (
        "chunk_id",
        "doc_id",
        "canonical_chunks_sha256_at_authoring",
        "ontology_sha256",
        "annotation_guideline_sha256",
        "entities",
        "relations",
    )
    for key in required:
        if key not in record:
            raise DifficultyError(f"gold record missing required field: {key}")
    if not isinstance(record["entities"], list):
        raise DifficultyError("gold record entities must be a list")
    if not isinstance(record["relations"], list):
        raise DifficultyError("gold record relations must be a list")


def _sha_for_gold_record(record: Mapping[str, Any]) -> str:
    """SHA256 over the canonical JSON encoding of the record.

    Excludes the ``note`` field because notes are diagnostic human metadata
    and may be amended under the same ``annotation_guideline_version``
    without changing the difficulty classification (per #48 Refinement 9).
    The SHA pins the reproducibility contract that ``#48 §10`` requires.
    """
    pinned = {
        key: record[key]
        for key in (
            "schema_version",
            "chunk_id",
            "doc_id",
            "canonical_chunks_sha256_at_authoring",
            "ontology_version",
            "ontology_sha256",
            "annotation_guideline_version",
            "annotation_guideline_sha256",
            "annotator_id",
            "adjudication_status",
            "adjudicated_by",
            "annotated_at",
            "entities",
            "relations",
        )
        if key in record
    }
    return _sha_json(pinned)


def run(
    *,
    gold_dir: Path,
    sample_path: Path,
    output_jsonl: Path,
    output_audit: Path,
    sample_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Run the classifier end-to-end.

    Returns the audit payload; writes both output files atomically. The
    function is idempotent: identical inputs ⇒ byte-identical output.
    """
    sample_by_id, paper_tier_by_id = _load_sample(sample_path)
    gold_records = _load_gold_records(gold_dir)

    classifications: list[Classification] = []
    skipped: list[dict[str, Any]] = []

    for path, record in gold_records:
        chunk_id = record["chunk_id"]
        if chunk_id not in paper_tier_by_id:
            skipped.append({"chunk_id": chunk_id, "reason": "missing_from_sample"})
            continue
        _validate_record_top_level(record)
        paper_tier = paper_tier_by_id[chunk_id]
        triggers = classify(record, paper_tier)
        difficulty = decide(triggers)
        classification = Classification(
            chunk_id=chunk_id,
            doc_id=record["doc_id"],
            paper_tier=paper_tier,
            difficulty=difficulty,
            triggers=triggers,
            gold_record_sha256=_sha_for_gold_record(record),
            canonical_chunks_sha256=record["canonical_chunks_sha256_at_authoring"],
            ontology_sha256=record["ontology_sha256"],
            annotation_guideline_sha256=record["annotation_guideline_sha256"],
        )
        classifications.append(classification)

    _atomic_write_jsonl(output_jsonl, [c.as_dict() for c in classifications])

    audit = _build_audit(classifications, skipped, sample_audit_path, sample_path)
    _atomic_write_json(output_audit, audit)

    return audit


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            fh.write("\n")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        fh.write("\n")
    os.replace(tmp, path)


def _build_audit(
    classifications: Sequence[Classification],
    skipped: Sequence[Mapping[str, Any]],
    sample_audit_path: Path | None,
    sample_path: Path,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    classifier_script_sha256 = _sha_bytes(script_path.read_bytes())

    stratum_counts: dict[str, dict[str, int]] = {
        difficulty: {"count": 0} for difficulty in DIFFICULTY_LABELS
    }
    paper_tier_counts: dict[str, dict[str, int]] = {}
    trigger_counts: dict[str, int] = {}

    for classification in classifications:
        stratum_counts[classification.difficulty]["count"] += 1
        tier = classification.paper_tier
        paper_tier_counts.setdefault(tier, {d: 0 for d in DIFFICULTY_LABELS})
        paper_tier_counts[tier][classification.difficulty] += 1
        for trigger in classification.triggers.hard_triggers:
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1

    audit: dict[str, Any] = {
        "classifier_id": CLASSIFIER_ID,
        "classifier_version": CLASSIFIER_VERSION,
        "classifier_script_sha256": classifier_script_sha256,
        "classifier_script_path": str(script_path.relative_to(ROOT)) if script_path.is_relative_to(ROOT) else str(script_path),
        "sample_v2_path": str(sample_path.relative_to(ROOT)) if sample_path.is_relative_to(ROOT) else str(sample_path),
        "sample_v2_sha256": _sha_bytes(sample_path.read_bytes()),
        "chunks_dir": str(Path("gold_set/chunks")),
        "chunk_count_total": len(classifications) + len(skipped),
        "chunk_count_classified": len(classifications),
        "chunk_count_skipped": len(skipped),
        "skipped": list(skipped),
        "difficulty_counts": stratum_counts,
        "paper_tier_difficulty_counts": paper_tier_counts,
        "hard_trigger_counts": trigger_counts,
        "spec_gaps_observed": [
            "common_noun_list_minimum_only: attention, model, score",
            "sentence_boundary_heuristic_only: terminator+whitespace+capital OR paragraph break",
        ],
        "versioning_pins": {
            "annotation_guideline_version": 1,
            "ontology_version_required": True,
        },
    }
    if sample_audit_path is not None:
        audit["sample_audit_sha256"] = _sha_bytes(sample_audit_path.read_bytes())
    return audit


# --- CLI ---------------------------------------------------------------------


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="classify_difficulty",
        description=(
            "Deterministic difficulty classification for the v2 gold-set "
            "pilot (#48 Refinement 5). Reads gold records and the v2 sample, "
            "writes difficulty.v1.jsonl and difficulty.v1.audit.json."
        ),
    )
    parser.add_argument(
        "--gold-dir",
        default="gold_set/chunks",
        help="Directory containing the canonical gold-record JSON files.",
    )
    parser.add_argument(
        "--sample",
        default="data/gold_set/sample.v2.jsonl",
        help="Path to the v2 sample JSONL.",
    )
    parser.add_argument(
        "--sample-audit",
        default="data/gold_set/sample.v2.audit.json",
        help="Optional sample audit JSON whose SHA256 is recorded in the audit.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=OUTPUT_JSONL,
        help="Output JSONL path (gitignored).",
    )
    parser.add_argument(
        "--output-audit",
        default=OUTPUT_AUDIT,
        help="Output audit JSON path (gitignored).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-classification summary printed to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    gold_dir = _resolve(args.gold_dir)
    sample_path = _resolve(args.sample)
    sample_audit_path = _resolve(args.sample_audit)
    output_jsonl = _resolve(args.output_jsonl)
    output_audit = _resolve(args.output_audit)

    if not sample_audit_path.exists():
        sample_audit_path_for_audit: Path | None = None
    else:
        sample_audit_path_for_audit = sample_audit_path

    audit = run(
        gold_dir=gold_dir,
        sample_path=sample_path,
        output_jsonl=output_jsonl,
        output_audit=output_audit,
        sample_audit_path=sample_audit_path_for_audit,
    )

    if not args.quiet:
        counts = audit["difficulty_counts"]
        summary = ", ".join(f"{k}={counts[k]['count']}" for k in DIFFICULTY_LABELS)
        print(
            f"classify_difficulty: {audit['chunk_count_classified']} chunks "
            f"({summary}); output={output_jsonl}; audit={output_audit}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())