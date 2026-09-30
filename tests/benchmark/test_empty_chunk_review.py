from src.benchmark.empty_chunk_review import build_empty_chunk_review


def test_empty_review_keeps_rejects_and_incomplete_chunks_without_gold():
    chunks = [
        {"chunk_id": name, "doc_id": "W1", "text": "NQ is evaluated."}
        for name in ("empty", "rejected", "aborted", "mention", "assertion")
    ]
    records = [
        {"chunk_id": "empty", "kind": "chunk_status", "extraction_complete": True},
        {"chunk_id": "rejected", "kind": "chunk_status", "extraction_complete": False},
        {"chunk_id": "mention", "kind": "mention"},
        {"chunk_id": "assertion", "kind": "assertion"},
    ]
    rejects = [{"chunk_id": "rejected", "reason": "INVALID_OCCURRENCE_INDEX"}]
    report = build_empty_chunk_review(
        chunks=chunks, records=records, rejects=rejects, provenance={"recording_sha256": "fixture"}
    )
    entries = {entry["chunk_id"]: entry for entry in report["chunks"]}
    assert set(entries) == {"empty", "rejected", "aborted"}
    assert entries["rejected"]["observation"]["canonical_rejects"] == rejects
    assert entries["aborted"]["observation"]["canonical_status"] is None
    assert report["automatic_gold_labels"] is False
    assert report["scientific_conclusion"] is None
    for entry in entries.values():
        assert entry["human_review"]["section_assessment"] is None
        assert entry["human_review"]["possible_missed_entities"] is None
        assert entry["human_review"]["possible_missed_relations"] is None
        assert entry["observation"]["model_returned_empty_arrays"] is None
