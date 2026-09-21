"""Synthetic acceptance tests for the fail-closed gold-record authoring CLI (#65)."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/gold_set/author.py"
SPEC = importlib.util.spec_from_file_location("gold_author", MODULE)
author = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = author
SPEC.loader.exec_module(author)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, text: str = "M uses D to measure T with K.", excluded: bool = False) -> tuple[author.Paths, str]:
    chunk_id = "c_0000000000000001"
    chunks = tmp_path / "chunks.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    sample = tmp_path / "sample.jsonl"
    audit = tmp_path / "audit.json"
    leakage = tmp_path / "leakage.json"
    ontology = tmp_path / "ontology.yaml"
    guideline = tmp_path / "GUIDELINES.md"
    _jsonl(chunks, [{"chunk_id": chunk_id, "doc_id": "W1", "text": text}])
    _jsonl(manifest, [{"openalex_id": "W1"}])
    _jsonl(sample, [{"chunk_id": chunk_id, "doc_id": "W1", "sampler_version": "gold-set-sample.v2"}] * 120)
    # Replace with 120 distinct IDs while retaining the requested canonical row.
    sample_rows = [{"chunk_id": chunk_id, "doc_id": "W1", "sampler_version": "gold-set-sample.v2"}] + [{"chunk_id": f"c_{index:016x}", "doc_id": "W1", "sampler_version": "gold-set-sample.v2"} for index in range(2, 121)]
    _jsonl(sample, sample_rows)
    ontology.write_text("""version: 1
relation_types:
  - {name: PROPOSES, from: document, to: method, population_mode: extracted}
  - {name: INTRODUCES, from: document, to: dataset, population_mode: extracted}
  - {name: EVALUATES_ON, from: document, to: dataset, population_mode: extracted}
  - {name: USES_METHOD, from: document, to: method, population_mode: extracted}
  - {name: REPORTS_METRIC, from: document, to: metric, population_mode: extracted}
  - {name: EXTENDS, from: method, to: method, population_mode: extracted}
  - {name: USED_FOR_TASK, from: dataset, to: task, population_mode: extracted}
  - {name: MEASURES, from: metric, to: task, population_mode: extracted}
  - {name: CITES, from: document, to: document, population_mode: structural}
  - {name: EVALUATED_ON, from: method, to: dataset, population_mode: derived}
""", encoding="utf-8")
    guideline.write_text("annotation_guideline_version = 1\n", encoding="utf-8")
    _json(leakage, {"excluded_chunk_ids": [chunk_id] if excluded else []})
    _json(audit, {"selected_chunk_ids_sha256": hashlib.sha256("".join(sorted(row["chunk_id"] for row in sample_rows)).encode()).hexdigest(), "chunks_sha256": _sha(chunks), "manifest_sha256": _sha(manifest), "ontology_sha256": _sha(ontology), "leakage_artifact_sha256": _sha(leakage), "leakage_overlap_count": 0})
    return author.Paths(chunks, manifest, sample, audit, leakage, ontology, guideline, tmp_path / "published"), chunk_id


def _entity(entity_id: str, entity_type: str, text: str, literal: str) -> dict:
    starts = author._enumerate_occurrences(text, literal)
    occurrences = [{"occurrence_index": index, "char_start": start, "char_end": start + len(literal)} for index, start in enumerate(starts)]
    return {"entity_id": entity_id, "entity_type": entity_type, "surface_form": literal, "canonical_form": literal, "span": {"char_start": starts[0], "char_end": starts[0] + len(literal)}, "occurrences_in_chunk": occurrences, "confidence": "stated"}


def _record(paths: author.Paths, chunk_id: str, text: str) -> dict:
    record = author.template(paths, chunk_id, "ann-A")
    record["entities"] = [_entity("e_m", "method", text, "M"), _entity("e_d", "dataset", text, "D"), _entity("e_t", "task", text, "T"), _entity("e_k", "metric", text, "K")]
    return record


def test_show_and_template_valid_path(tmp_path, capsys):
    paths, chunk_id = _fixture(tmp_path)
    chunk, *_ = author.preflight(paths, chunk_id)
    assert chunk["text"].startswith("M uses")
    draft = author.template(paths, chunk_id, "ann-A")
    assert draft["annotated_at"] is None
    assert draft["entities"] == [] and draft["relations"] == []
    assert "note" not in draft
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("kind", ["unsampled", "excluded"])
def test_unsampled_and_leakage_excluded_rejected(tmp_path, kind):
    paths, chunk_id = _fixture(tmp_path, excluded=kind == "excluded")
    with pytest.raises(author.AuthorError, match="not sampled|held-out"):
        author.preflight(paths, "c_ffffffffffffffff" if kind == "unsampled" else chunk_id)


@pytest.mark.parametrize("target", ["chunks", "manifest", "ontology", "leakage"])
def test_stale_input_identity_rejected(tmp_path, target):
    paths, chunk_id = _fixture(tmp_path)
    getattr(paths, target).write_bytes(getattr(paths, target).read_bytes() + b" ")
    with pytest.raises(author.AuthorError, match="does not match audit"):
        author.preflight(paths, chunk_id)


def test_malformed_sample_and_question_independence(tmp_path, monkeypatch):
    paths, chunk_id = _fixture(tmp_path)
    _jsonl(paths.sample, [{"chunk_id": chunk_id, "doc_id": "W1", "sampler_version": "gold-set-sample.v2"}])
    with pytest.raises(author.AuthorError, match="120"):
        author.preflight(paths, chunk_id)
    paths, chunk_id = _fixture(tmp_path)
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("question accessed")) if "questions" in self.name else original(self))
    author.preflight(paths, chunk_id)


def test_entity_exactness_and_fail_closed_shapes(tmp_path):
    text = "DPR then DPR"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    record["entities"] = [_entity("e", "method", text, "DPR")]
    record["note"] = "non-empty"
    author.validate_record(record, paths)
    record["entities"][0]["occurrences_in_chunk"] = record["entities"][0]["occurrences_in_chunk"][:1]
    with pytest.raises(author.AuthorError, match="enumeration"):
        author.validate_record(record, paths)
    record["entities"][0] = _entity("e", "method", text, "DPR")
    record["entities"][0]["span"]["char_start"] = True
    with pytest.raises(author.AuthorError, match="bool"):
        author.validate_record(record, paths)
    record["entities"][0] = _entity("e", "method", text, "DPR")
    record["entities"][0]["surface_form"] = "dpr"
    with pytest.raises(author.AuthorError, match="mismatch"):
        author.validate_record(record, paths)


@pytest.mark.parametrize("relation_type,subject,object", [("PROPOSES", "document", "method"), ("INTRODUCES", "document", "dataset"), ("EVALUATES_ON", "document", "dataset"), ("USES_METHOD", "document", "method"), ("REPORTS_METRIC", "document", "metric"), ("EXTENDS", "method", "method"), ("USED_FOR_TASK", "dataset", "task"), ("MEASURES", "metric", "task")])
def test_all_extracted_relation_signatures(tmp_path, relation_type, subject, object):
    text = "M uses D to measure T with K."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = _record(paths, chunk_id, text)
    ids = {"method": "e_m", "dataset": "e_d", "task": "e_t", "metric": "e_k"}
    endpoint = lambda type_: {"kind": "document", "id": "W1"} if type_ == "document" else {"kind": "mention", "id": ids[type_]}
    record["relations"] = [{"relation_id": "r", "relation_type": relation_type, "subject_ref": endpoint(subject), "object_ref": endpoint(object), "evidence_quote": text, "evidence_span": {"char_start": 0, "char_end": len(text)}, "endpoint_mentions": [], "confidence": "stated"}]
    author.validate_record(record, paths)


def test_relations_endpoint_evidence_and_empty_rules(tmp_path):
    text = "M M uses D."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    record["entities"] = [_entity("e_m", "method", text, "M"), _entity("e_d", "dataset", text, "D")]
    relation = {"relation_id": "r", "relation_type": "PROPOSES", "subject_ref": {"kind": "document", "id": "W1"}, "object_ref": {"kind": "mention", "id": "e_m"}, "evidence_quote": text, "evidence_span": {"char_start": 0, "char_end": len(text)}, "endpoint_mentions": [{"endpoint_role": "object", "occurrence_index": 0}], "confidence": "stated"}
    record["relations"] = [relation]
    author.validate_record(record, paths)
    relation["relation_type"] = "CITES"
    with pytest.raises(author.AuthorError, match="not extracted"):
        author.validate_record(record, paths)
    relation["relation_type"] = "PROPOSES"
    relation["endpoint_mentions"] = []
    with pytest.raises(author.AuthorError, match="exactly represent"):
        author.validate_record(record, paths)
    record = author.template(paths, chunk_id, "ann-A")
    record["note"] = ""
    with pytest.raises(author.AuthorError, match="non-empty"):
        author.validate_record(record, paths)
    record["note"] = "true negative"
    record["relations"] = [relation]
    with pytest.raises(author.AuthorError, match="empty entity"):
        author.validate_record(record, paths)

def test_unknown_fields_and_write_once_publication(tmp_path, monkeypatch):
    text = "M uses D to measure T with K."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = _record(paths, chunk_id, text)
    input_path = tmp_path / "draft.json"
    _json(input_path, record)
    class _Clock:
        def __init__(self) -> None:
            self.values = iter(("2026-01-01T00:00:00+00:00", "2026-02-02T00:00:00+00:00"))
            self.fallback = datetime.datetime.fromisoformat("2026-03-03T00:00:00+00:00")

        def now(self, _tz):
            try:
                return datetime.datetime.fromisoformat(next(self.values))
            except StopIteration:
                return self.fallback

    monkeypatch.setattr(author, "datetime", _Clock())
    assert author.publish(paths, input_path)
    destination = paths.destination_dir / f"{chunk_id}.json"
    first_bytes = destination.read_bytes()
    first_mtime = destination.stat().st_mtime_ns
    first_timestamp = json.loads(first_bytes)["annotated_at"]
    assert first_timestamp == "2026-01-01T00:00:00Z"
    assert not author.publish(paths, input_path)
    assert destination.read_bytes() == first_bytes
    second_timestamp = json.loads(destination.read_bytes())["annotated_at"]
    assert second_timestamp == first_timestamp
    assert destination.stat().st_mtime_ns == first_mtime
    destination.write_bytes(b"different\n")
    with pytest.raises(author.AuthorError, match="differs"):
        author.publish(paths, input_path)
    assert destination.read_bytes() == b"different\n"
    record["unknown"] = 1
    with pytest.raises(author.AuthorError, match="unknown"):
        author.validate_record(record, paths)

def test_parser_has_no_overwrite_escape_hatches():
    def _options_for(parser: argparse.ArgumentParser) -> set[str]:
        seen: set[str] = set()
        stack = [parser]
        while stack:
            current = stack.pop()
            for action in current._actions:
                seen.update(action.option_strings)
                sub_actions = getattr(action, "choices", None)
                if isinstance(sub_actions, dict):
                    stack.extend(sub_actions.values())
        return seen

    options = _options_for(author._build_parser())
    assert "--force" not in options and "--overwrite" not in options

@pytest.mark.parametrize("invalid_type", ["Method", "dataset ", "tool", ""])
def test_invalid_entity_type_rejected(tmp_path, invalid_type):
    text = "M uses D to measure T with K."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = _record(paths, chunk_id, text)
    record["entities"][0]["entity_type"] = invalid_type
    with pytest.raises(author.AuthorError, match="entity_type"):
        author.validate_record(record, paths)


@pytest.mark.parametrize("invalid_confidence", ["STATED", "guessed", "", 1])
def test_invalid_entity_confidence_rejected(tmp_path, invalid_confidence):
    text = "M uses D to measure T with K."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = _record(paths, chunk_id, text)
    record["entities"][0]["confidence"] = invalid_confidence
    with pytest.raises(author.AuthorError, match="confidence"):
        author.validate_record(record, paths)


def test_relation_type_must_be_extracted(tmp_path):
    text = "M uses D to measure T with K."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = _record(paths, chunk_id, text)
    ids = {"method": "e_m", "dataset": "e_d", "task": "e_t", "metric": "e_k"}
    base = {"relation_id": "r", "subject_ref": {"kind": "document", "id": "W1"}, "object_ref": {"kind": "mention", "id": ids["dataset"]}, "evidence_quote": text, "evidence_span": {"char_start": 0, "char_end": len(text)}, "endpoint_mentions": [], "confidence": "stated"}
    for relation_type in ("CITES", "EVALUATED_ON"):
        relation = dict(base)
        relation["relation_type"] = relation_type
        record["relations"] = [relation]
        with pytest.raises(author.AuthorError, match="not extracted"):
            author.validate_record(record, paths)


def test_dangling_mention_endpoint_rejected(tmp_path):
    text = "M uses D to measure T with K."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = _record(paths, chunk_id, text)
    record["relations"] = [{"relation_id": "r", "relation_type": "PROPOSES", "subject_ref": {"kind": "document", "id": "W1"}, "object_ref": {"kind": "mention", "id": "ghost"}, "evidence_quote": text, "evidence_span": {"char_start": 0, "char_end": len(text)}, "endpoint_mentions": [], "confidence": "stated"}]
    with pytest.raises(author.AuthorError, match="dangling"):
        author.validate_record(record, paths)

def test_wrong_type_mention_endpoint_rejected(tmp_path):
    text = "M uses D to measure T with K."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = _record(paths, chunk_id, text)
    record["relations"] = [{"relation_id": "r", "relation_type": "PROPOSES", "subject_ref": {"kind": "document", "id": "W1"}, "object_ref": {"kind": "mention", "id": "e_d"}, "evidence_quote": text, "evidence_span": {"char_start": 0, "char_end": len(text)}, "endpoint_mentions": [], "confidence": "stated"}]
    with pytest.raises(author.AuthorError, match="type mismatch"):
        author.validate_record(record, paths)


def test_evidence_span_outside_mention_endpoint_rejected(tmp_path):
    text = "M uses D to measure T with K."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = _record(paths, chunk_id, text)
    evidence_quote = "uses D"
    record["relations"] = [{"relation_id": "r", "relation_type": "PROPOSES", "subject_ref": {"kind": "document", "id": "W1"}, "object_ref": {"kind": "mention", "id": "e_m"}, "evidence_quote": evidence_quote, "evidence_span": {"char_start": text.find(evidence_quote), "char_end": text.find(evidence_quote) + len(evidence_quote)}, "endpoint_mentions": [], "confidence": "stated"}]
    with pytest.raises(author.AuthorError, match="not contained"):
        author.validate_record(record, paths)


def test_malformed_sample_jsonl_rejected(tmp_path):
    paths, chunk_id = _fixture(tmp_path)
    paths.sample.write_text("{not valid json\n", encoding="utf-8")
    with pytest.raises(author.AuthorError, match="malformed JSON"):
        author.preflight(paths, chunk_id)

@pytest.mark.parametrize("bad_occurrences", [
    [{"occurrence_index": 1, "char_start": 9, "char_end": 12}],
    [{"occurrence_index": 0, "char_start": 0, "char_end": 3}],
])
def test_incomplete_or_out_of_order_occurrences_rejected(tmp_path, bad_occurrences):
    text = "DPR then DPR"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    entity = _entity("e", "method", text, "DPR")
    entity["occurrences_in_chunk"] = bad_occurrences
    entity["span"] = {"char_start": text.find("DPR"), "char_end": text.find("DPR") + 3}
    record["entities"] = [entity]
    record["note"] = "non-empty"
    with pytest.raises(author.AuthorError, match="enumeration|indices"):
        author.validate_record(record, paths)


def test_no_access_to_eval_questions(tmp_path, monkeypatch):
    paths, chunk_id = _fixture(tmp_path)
    accessed: list[str] = []
    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes
    original_open = Path.open

    def watching_read_text(self, *args, **kwargs):
        if "questions" in self.name:
            accessed.append(self.name)
        return original_read_text(self, *args, **kwargs)

    def watching_read_bytes(self, *args, **kwargs):
        if "questions" in self.name:
            accessed.append(self.name)
        return original_read_bytes(self, *args, **kwargs)

    def watching_open(self, *args, **kwargs):
        if "questions" in self.name:
            accessed.append(self.name)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", watching_read_text)
    monkeypatch.setattr(Path, "read_bytes", watching_read_bytes)
    monkeypatch.setattr(Path, "open", watching_open)
    author.preflight(paths, chunk_id)
    assert not accessed, f"preflight touched eval/questions files: {accessed}"


def test_template_destination_inside_canonical_rejected(tmp_path):
    paths, chunk_id = _fixture(tmp_path)
    target = paths.destination_dir / f"{chunk_id}.json"
    with pytest.raises(author.AuthorError, match="canonical"):
        author._reject_canonical_destination(paths, target)
    nested = paths.destination_dir / "nested" / "draft.json"
    with pytest.raises(author.AuthorError, match="canonical"):
        author._reject_canonical_destination(paths, nested)
    outside = tmp_path / "draft.json"
    author._reject_canonical_destination(paths, outside)


def test_template_cli_rejects_canonical_destination(tmp_path, capsys, monkeypatch):
    paths, chunk_id = _fixture(tmp_path)
    monkeypatch.setattr(author, "_default_paths", lambda: paths)
    target = paths.destination_dir / f"{chunk_id}.json"
    code = author.main(["template", "--chunk-id", chunk_id, "--annotator-id", "ann-A", "--output", str(target)])
    captured = capsys.readouterr()
    assert code == 2
    assert "canonical" in captured.err
    assert not target.exists()

def test_guideline_version_exact_one_accepted(tmp_path):
    paths, _chunk_id = _fixture(tmp_path)
    paths.guideline.write_text("annotation_guideline_version = 1\n", encoding="utf-8")
    assert author._guideline_version(paths.guideline) == "1"


@pytest.mark.parametrize("version_line", ["annotation_guideline_version = 10\n", "annotation_guideline_version = 11\n", "annotation_guideline_version = 1.0\n"])
def test_guideline_version_other_numbers_rejected(tmp_path, version_line):
    paths, _chunk_id = _fixture(tmp_path)
    paths.guideline.write_text(version_line, encoding="utf-8")
    with pytest.raises(author.AuthorError, match="exactly 1"):
        author._guideline_version(paths.guideline)


def test_guideline_version_missing_rejected(tmp_path):
    paths, _chunk_id = _fixture(tmp_path)
    paths.guideline.write_text("# header\nNo version here.\n", encoding="utf-8")
    with pytest.raises(author.AuthorError, match="exactly 1"):
        author._guideline_version(paths.guideline)


def test_guideline_version_conflicting_rejected(tmp_path):
    paths, _chunk_id = _fixture(tmp_path)
    paths.guideline.write_text("annotation_guideline_version = 1\nannotation_guideline_version = 2\n", encoding="utf-8")
    with pytest.raises(author.AuthorError, match="exactly 1"):
        author._guideline_version(paths.guideline)


def test_guideline_version_multiline_assignment_rejected(tmp_path):
    paths, _chunk_id = _fixture(tmp_path)
    paths.guideline.write_text("annotation_guideline_version =\n1\n", encoding="utf-8")
    with pytest.raises(author.AuthorError, match="exactly 1"):
        author._guideline_version(paths.guideline)


def test_guideline_version_valid_plus_invalid_float_rejected(tmp_path):
    paths, _chunk_id = _fixture(tmp_path)
    paths.guideline.write_text("annotation_guideline_version = 1\nannotation_guideline_version = 1.0\n", encoding="utf-8")
    with pytest.raises(author.AuthorError, match="exactly 1"):
        author._guideline_version(paths.guideline)


def test_guideline_version_valid_plus_non_numeric_rejected(tmp_path):
    paths, _chunk_id = _fixture(tmp_path)
    paths.guideline.write_text("annotation_guideline_version = 1\nannotation_guideline_version = foo\n", encoding="utf-8")
    with pytest.raises(author.AuthorError, match="exactly 1"):
        author._guideline_version(paths.guideline)
