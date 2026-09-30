"""Unlabelled human-review packets; zero accepted records are not gold negatives."""

from __future__ import annotations

from collections import Counter

from src.benchmark.manifest_recording import sha256_bytes


def build_empty_chunk_review(
    *, chunks: list[dict], records: list[dict], rejects: list[dict], provenance: dict
) -> dict:
    """Select output-empty chunks, retaining failures and unknown completion.

    This deliberately does not infer model-empty arrays from canonical output.
    Empty model output, alignment failures, and incomplete calls are different
    observations. Only a human can judge scientific absence or omissions.
    """
    entries = []
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        chunk_records = [r for r in records if r.get("chunk_id") == chunk_id]
        counts = Counter(r.get("kind") for r in chunk_records)
        if counts["mention"] or counts["assertion"]:
            continue
        statuses = [r for r in chunk_records if r.get("kind") == "chunk_status"]
        entries.append({
            "chunk_id": chunk_id,
            "doc_id": chunk["doc_id"],
            "section_path": chunk.get("section_path", []),
            "source_text": chunk["text"],
            "source_text_sha256": sha256_bytes(chunk["text"].encode("utf-8")),
            "observation": {
                "accepted_mentions": 0,
                "accepted_assertions": 0,
                "canonical_status": statuses[0] if statuses else None,
                "canonical_rejects": [r for r in rejects if r.get("chunk_id") == chunk_id],
                "model_returned_empty_arrays": None,
                "model_empty_note": "Inspect original HTTP responses; not inferred from accepted records.",
            },
            "human_review": {
                "reviewer": None,
                "reviewed_at": None,
                "source_context_checked": None,
                "raw_response_references_checked": None,
                "section_assessment": None,
                "assessment_choices": [
                    "truly_empty_for_ontology", "possible_missed_entities",
                    "possible_missed_relations", "possible_missed_entities_and_relations",
                    "insufficient_context_or_technical_failure",
                ],
                "rationale_and_exact_evidence": None,
                "possible_missed_entities": None,
                "possible_missed_relations": None,
                "context_limits": None,
                "second_reviewer": None,
                "adjudication": None,
            },
        })
    return {
        "format": "manifest-empty-chunk-review-v1",
        "provenance": provenance,
        "selection": "zero canonical accepted mentions AND assertions; includes incomplete/rejected chunks",
        "automatic_gold_labels": False,
        "scientific_conclusion": None,
        "instructions": (
            "Use docs/benchmark/empty-chunk-review-template.md. Review the source independently, "
            "then inspect raw model answers and technical rejects. Leave all decisions unset until "
            "human review. Completed review is not automatically imported as gold data."
        ),
        "chunks": entries,
    }
