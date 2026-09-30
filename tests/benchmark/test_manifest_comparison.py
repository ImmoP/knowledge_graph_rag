"""Consumer-visible A/B invariants over recorded synthetic HTTP responses."""

import json
from pathlib import Path

import httpx
import pytest

from src.benchmark.manifest_replay import record_run, replay_run
from src.extract.ids import mention_id_v2
from src.extract.validation import source_text_sha256

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("text", "quote", "status", "assertions", "rejects"),
    [
        ("DPR\n evaluates NQ.", "DPR evaluates NQ.", "repaired", 1, 0),
        ("DPR evaluates NQ.", "DPR evaluates NQ.", "exact", 1, 0),
        ("DPR\n evaluates NQ. DPR\n evaluates NQ.", "DPR evaluates NQ.", "ambiguous", 0, 1),
        ("DPR\n\n evaluates NQ.", "DPR evaluates NQ.", "paragraph_boundary", 0, 1),
        ("DPR evaluates NQ.", "DPR evaluates nq.", "not_found", 0, 1),
    ],
)
def test_comparison_preserves_raw_and_canonical_outputs(
    tmp_path, monkeypatch, text, quote, status, assertions, rejects
):
    chunk = {
        "chunk_id": "c_comparison", "doc_id": "W1", "section_path": ["results"],
        "chunk_index": 0, "text": text, "token_estimate": 10,
        "char_start": 0, "char_end": len(text),
    }
    start = text.index("NQ")
    mention_id = mention_id_v2(
        chunk_id=chunk["chunk_id"], source_text_sha256=source_text_sha256(text),
        entity_type="dataset", char_start=start, char_end=start + 2,
    )
    payloads = iter([
        {"mentions": [{
            "entity_type": "dataset", "surface_form": "NQ", "canonical_form": "Natural Questions",
            "confidence": "stated", "occurrence_index": 0,
        }]},
        {"assertions": [{
            "relation_type": "EVALUATES_ON", "subject_ref": {"kind": "document", "id": "W1"},
            "object_ref": {"kind": "mention", "id": mention_id},
            "evidence_quote": quote, "confidence": "stated",
        }]},
    ])

    def respond(request):
        return httpx.Response(200, json={
            "model": "synthetic/offline-fixture",
            "choices": [{"message": {"content": json.dumps(next(payloads))}, "finish_reason": "stop"}],
        })

    source = tmp_path / "recording"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        record_run(
            output_dir=source, chunks=[chunk], ontology_path=ROOT / "ontology.yaml",
            prompts_dir=ROOT / "prompts", max_live_calls=4, api_key="synthetic-test-key", client=client,
        )
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}

    def no_network(*args, **kwargs):
        raise AssertionError("offline replay created network client")

    monkeypatch.setattr(httpx, "Client", no_network)
    output = tmp_path / "replay"
    replay_run(recording_dir=source, output_dir=output, whitespace_comparison=True)
    report = json.loads((output / "whitespace_comparison" / "comparison.json").read_text())
    assert report["canonical_v2"] is False
    assert report["decisions"][0]["status"] == status
    assert report["experiment"]["accepted_assertions"] == assertions
    assert report["experiment"]["rejects"] == rejects
    for relative, content in before.items():
        assert (source / relative).read_bytes() == content
    assert (output / "extractions.jsonl").read_bytes() == (source / "extractions.jsonl").read_bytes()
    assert (output / "extractions.rejects.jsonl").read_bytes() == (source / "extractions.rejects.jsonl").read_bytes()
