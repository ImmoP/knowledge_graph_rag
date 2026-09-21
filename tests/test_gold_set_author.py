"""Synthetic acceptance tests for the fail-closed gold-record authoring CLI (#65, #69)."""
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


def _entity(chunk_id: str, seq: int, entity_type: str, text: str, literal: str) -> dict:
    token = chunk_id[2:10]
    starts = author._enumerate_occurrences(text, literal)
    occurrences = [{"occurrence_index": index, "char_start": start, "char_end": start + len(literal)} for index, start in enumerate(starts)]
    return {
        "entity_id": f"ge_{token}_{seq:02d}",
        "entity_type": entity_type,
        "surface_form": literal,
        "canonical_form": literal,
        "span": {"char_start": starts[0], "char_end": starts[0] + len(literal)},
        "occurrences_in_chunk": occurrences,
        "confidence": "stated",
    }


def _record(paths: author.Paths, chunk_id: str, text: str) -> dict:
    record = author.template(paths, chunk_id, "ann-A")
    record["entities"] = [
        _entity(chunk_id, 1, "method", text, "M"),
        _entity(chunk_id, 2, "dataset", text, "D"),
        _entity(chunk_id, 3, "task", text, "T"),
        _entity(chunk_id, 4, "metric", text, "K"),
    ]
    return record


def _seed_draft(tmp_path: Path, paths: author.Paths, chunk_id: str, record: dict) -> Path:
    draft = tmp_path / f"draft_{chunk_id}.json"
    _json(draft, record)
    return draft


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
    record["entities"] = [_entity(chunk_id, 1, "method", text, "DPR")]
    record["note"] = "non-empty"
    author.validate_record(record, paths)
    record["entities"][0]["occurrences_in_chunk"] = record["entities"][0]["occurrences_in_chunk"][:1]
    with pytest.raises(author.AuthorError, match="enumeration"):
        author.validate_record(record, paths)
    record["entities"][0] = _entity(chunk_id, 1, "method", text, "DPR")
    record["entities"][0]["span"]["char_start"] = True
    with pytest.raises(author.AuthorError, match="bool"):
        author.validate_record(record, paths)
    record["entities"][0] = _entity(chunk_id, 1, "method", text, "DPR")
    record["entities"][0]["surface_form"] = "dpr"
    with pytest.raises(author.AuthorError, match="mismatch"):
        author.validate_record(record, paths)


@pytest.mark.parametrize("relation_type,subject,object", [("PROPOSES", "document", "method"), ("INTRODUCES", "document", "dataset"), ("EVALUATES_ON", "document", "dataset"), ("USES_METHOD", "document", "method"), ("REPORTS_METRIC", "document", "metric"), ("EXTENDS", "method", "method"), ("USED_FOR_TASK", "dataset", "task"), ("MEASURES", "metric", "task")])
def test_all_extracted_relation_signatures(tmp_path, relation_type, subject, object):
    text = "M uses D to measure T with K."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = _record(paths, chunk_id, text)
    token = chunk_id[2:10]
    ids = {"method": f"ge_{token}_01", "dataset": f"ge_{token}_02", "task": f"ge_{token}_03", "metric": f"ge_{token}_04"}
    endpoint = lambda type_: {"kind": "document", "id": "W1"} if type_ == "document" else {"kind": "mention", "id": ids[type_]}
    record["relations"] = [{"relation_id": f"gr_{token}_01", "relation_type": relation_type, "subject_ref": endpoint(subject), "object_ref": endpoint(object), "evidence_quote": text, "evidence_span": {"char_start": 0, "char_end": len(text)}, "endpoint_mentions": [], "confidence": "stated"}]
    author.validate_record(record, paths)


def test_relations_endpoint_evidence_and_empty_rules(tmp_path):
    text = "M M uses D."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    token = chunk_id[2:10]
    record["entities"] = [_entity(chunk_id, 1, "method", text, "M"), _entity(chunk_id, 2, "dataset", text, "D")]
    relation = {"relation_id": f"gr_{token}_01", "relation_type": "PROPOSES", "subject_ref": {"kind": "document", "id": "W1"}, "object_ref": {"kind": "mention", "id": f"ge_{token}_01"}, "evidence_quote": text, "evidence_span": {"char_start": 0, "char_end": len(text)}, "endpoint_mentions": [{"endpoint_role": "object", "occurrence_index": 0}], "confidence": "stated"}
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
    token = chunk_id[2:10]
    base = {"relation_id": f"gr_{token}_01", "subject_ref": {"kind": "document", "id": "W1"}, "object_ref": {"kind": "mention", "id": f"ge_{token}_02"}, "evidence_quote": text, "evidence_span": {"char_start": 0, "char_end": len(text)}, "endpoint_mentions": [], "confidence": "stated"}
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
    token = chunk_id[2:10]
    record["relations"] = [{"relation_id": f"gr_{token}_01", "relation_type": "PROPOSES", "subject_ref": {"kind": "document", "id": "W1"}, "object_ref": {"kind": "mention", "id": "ghost"}, "evidence_quote": text, "evidence_span": {"char_start": 0, "char_end": len(text)}, "endpoint_mentions": [], "confidence": "stated"}]
    with pytest.raises(author.AuthorError, match="dangling"):
        author.validate_record(record, paths)


def test_wrong_type_mention_endpoint_rejected(tmp_path):
    text = "M uses D to measure T with K."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = _record(paths, chunk_id, text)
    token = chunk_id[2:10]
    record["relations"] = [{"relation_id": f"gr_{token}_01", "relation_type": "PROPOSES", "subject_ref": {"kind": "document", "id": "W1"}, "object_ref": {"kind": "mention", "id": f"ge_{token}_02"}, "evidence_quote": text, "evidence_span": {"char_start": 0, "char_end": len(text)}, "endpoint_mentions": [], "confidence": "stated"}]
    with pytest.raises(author.AuthorError, match="type mismatch"):
        author.validate_record(record, paths)


def test_evidence_span_outside_mention_endpoint_rejected(tmp_path):
    text = "M uses D to measure T with K."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = _record(paths, chunk_id, text)
    token = chunk_id[2:10]
    evidence_quote = "uses D"
    record["relations"] = [{"relation_id": f"gr_{token}_01", "relation_type": "PROPOSES", "subject_ref": {"kind": "document", "id": "W1"}, "object_ref": {"kind": "mention", "id": f"ge_{token}_01"}, "evidence_quote": evidence_quote, "evidence_span": {"char_start": text.find(evidence_quote), "char_end": text.find(evidence_quote) + len(evidence_quote)}, "endpoint_mentions": [], "confidence": "stated"}]
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
    entity = _entity(chunk_id, 1, "method", text, "DPR")
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


# ---------- #69 status fail-closed ----------


def _publish_canonical(paths: author.Paths, chunk_id: str) -> None:
    """Write a publishable record directly under the canonical destination.

    Bypasses preflight because the synthetic fixture's chunks.jsonl contains only one
    canonical chunk row. status() only consumes the chunk_id field.
    """
    paths.destination_dir.mkdir(parents=True, exist_ok=True)
    _json(paths.destination_dir / f"{chunk_id}.json", {"chunk_id": chunk_id, "doc_id": "W1", "entities": [], "relations": [], "annotated_at": "2026-01-01T00:00:00Z"})


def test_status_reports_three_annotated_in_synthetic_fixture(tmp_path):
    paths, chunk_id = _fixture(tmp_path)
    for identifier in ["c_0000000000000002", "c_0000000000000003", "c_0000000000000004"]:
        _publish_canonical(paths, identifier)
    result = author.status(paths)
    assert result["annotated"] == 3
    assert result["sample_size"] == 120
    assert result["remaining"] == 117
    assert result["next"] == {"chunk_id": "c_0000000000000001", "doc_id": "W1"}


def test_status_reports_zero_when_no_canonical_records(tmp_path):
    paths, _chunk_id = _fixture(tmp_path)
    result = author.status(paths)
    assert result["annotated"] == 0
    assert result["remaining"] == 120
    assert result["next"] == {"chunk_id": "c_0000000000000001", "doc_id": "W1"}


def test_status_orders_by_sample_v2_order(tmp_path):
    paths, _chunk_id = _fixture(tmp_path)
    _publish_canonical(paths, "c_0000000000000005")
    result = author.status(paths)
    assert result["next"] == {"chunk_id": "c_0000000000000001", "doc_id": "W1"}


def test_status_next_is_fourth_missing_in_canonical_order(tmp_path):
    """Valid 3/120: next must be the canonical fourth missing sample ID."""
    paths, chunk_id = _fixture(tmp_path)
    # Annotate c_0000000000000001..3 and c_...05, leaving c_...04 as the next missing.
    for identifier in [chunk_id, "c_0000000000000002", "c_0000000000000003", "c_0000000000000005"]:
        _publish_canonical(paths, identifier)
    result = author.status(paths)
    assert result["annotated"] == 4
    assert result["remaining"] == 116
    assert result["next"]["chunk_id"] == "c_0000000000000004"


def test_status_malformed_canonical_json_fails_closed(tmp_path):
    paths, _chunk_id = _fixture(tmp_path)
    paths.destination_dir.mkdir(parents=True, exist_ok=True)
    (paths.destination_dir / "c_0000000000000002.json").write_text("{not valid json\n", encoding="utf-8")
    with pytest.raises(author.AuthorError, match="malformed JSON"):
        author.status(paths)


def test_status_foreign_chunk_id_fails_closed(tmp_path):
    paths, _chunk_id = _fixture(tmp_path)
    paths.destination_dir.mkdir(parents=True, exist_ok=True)
    foreign_id = "c_deadbeefdeadbeef"
    _json(paths.destination_dir / f"{foreign_id}.json", {"chunk_id": foreign_id, "doc_id": "W1", "entities": [], "relations": []})
    with pytest.raises(author.AuthorError, match="not in canonical sample"):
        author.status(paths)


def test_status_filename_chunk_id_mismatch_fails_closed(tmp_path):
    paths, _chunk_id = _fixture(tmp_path)
    paths.destination_dir.mkdir(parents=True, exist_ok=True)
    _json(paths.destination_dir / "c_0000000000000002.json", {"chunk_id": "c_0000000000000003", "doc_id": "W1", "entities": [], "relations": []})
    with pytest.raises(author.AuthorError, match="does not match record"):
        author.status(paths)


def test_status_duplicate_logical_chunk_ids_fail_closed(tmp_path):
    paths, _chunk_id = _fixture(tmp_path)
    paths.destination_dir.mkdir(parents=True, exist_ok=True)
    identifier = "c_0000000000000002"
    _publish_canonical(paths, identifier)
    _json(paths.destination_dir / "c_0000000000000099.json", {"chunk_id": identifier, "doc_id": "W1", "entities": [], "relations": []})
    with pytest.raises(author.AuthorError, match="does not match record"):
        author.status(paths)


def test_status_frozen_input_mismatch_fails_closed(tmp_path):
    paths, _chunk_id = _fixture(tmp_path)
    paths.destination_dir.mkdir(parents=True, exist_ok=True)
    paths.chunks.write_bytes(paths.chunks.read_bytes() + b" ")
    with pytest.raises(author.AuthorError, match="does not match audit"):
        author.status(paths)


def test_status_cli_does_not_access_eval_questions(tmp_path, capsys, monkeypatch):
    paths, _chunk_id = _fixture(tmp_path)
    monkeypatch.setattr(author, "_default_paths", lambda: paths)
    accessed: list[str] = []
    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes

    def watching_read_text(self, *args, **kwargs):
        if "questions" in self.name:
            accessed.append(self.name)
        return original_read_text(self, *args, **kwargs)

    def watching_read_bytes(self, *args, **kwargs):
        if "questions" in self.name:
            accessed.append(self.name)
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", watching_read_text)
    monkeypatch.setattr(Path, "read_bytes", watching_read_bytes)
    code = author.main(["status"])
    captured = capsys.readouterr()
    assert code == 0
    assert "annotated" in captured.out
    assert not accessed, f"status touched eval/questions files: {accessed}"


# ---------- #69 ID convention ----------


def test_next_entity_id_first(tmp_path):
    chunk_id = "c_02e12cb05aeca236"
    assert author._next_entity_id([], chunk_id) == "ge_02e12cb0_01"


def test_next_entity_id_second(tmp_path):
    chunk_id = "c_02e12cb05aeca236"
    existing = [{
        "entity_id": "ge_02e12cb0_01",
        "entity_type": "method",
        "surface_form": "x", "canonical_form": "x",
        "span": {"char_start": 0, "char_end": 1},
        "occurrences_in_chunk": [{"occurrence_index": 0, "char_start": 0, "char_end": 1}],
        "confidence": "stated",
    }]
    assert author._next_entity_id(existing, chunk_id) == "ge_02e12cb0_02"


def test_next_entity_id_gap_uses_max_plus_one(tmp_path):
    chunk_id = "c_02e12cb05aeca236"
    existing = [
        {"entity_id": "ge_02e12cb0_01", "entity_type": "method", "surface_form": "x", "canonical_form": "x", "span": {"char_start": 0, "char_end": 1}, "occurrences_in_chunk": [{"occurrence_index": 0, "char_start": 0, "char_end": 1}], "confidence": "stated"},
        {"entity_id": "ge_02e12cb0_03", "entity_type": "method", "surface_form": "x", "canonical_form": "x", "span": {"char_start": 0, "char_end": 1}, "occurrences_in_chunk": [{"occurrence_index": 0, "char_start": 0, "char_end": 1}], "confidence": "stated"},
    ]
    assert author._next_entity_id(existing, chunk_id) == "ge_02e12cb0_04"


def test_next_entity_id_rejects_malformed_id(tmp_path):
    chunk_id = "c_02e12cb05aeca236"
    existing = [{"entity_id": "e_1", "entity_type": "method", "surface_form": "x", "canonical_form": "x", "span": {"char_start": 0, "char_end": 1}, "occurrences_in_chunk": [{"occurrence_index": 0, "char_start": 0, "char_end": 1}], "confidence": "stated"}]
    with pytest.raises(author.AuthorError, match="unexpected format"):
        author._next_entity_id(existing, chunk_id)


def test_next_entity_id_rejects_mixed_token(tmp_path):
    chunk_id = "c_02e12cb05aeca236"
    existing = [{"entity_id": "ge_ffffffff_01", "entity_type": "method", "surface_form": "x", "canonical_form": "x", "span": {"char_start": 0, "char_end": 1}, "occurrences_in_chunk": [{"occurrence_index": 0, "char_start": 0, "char_end": 1}], "confidence": "stated"}]
    with pytest.raises(author.AuthorError, match="chunk-token mismatch"):
        author._next_entity_id(existing, chunk_id)


def test_next_relation_id_first_and_gap_and_malformed(tmp_path):
    chunk_id = "c_02e12cb05aeca236"
    assert author._next_relation_id([], chunk_id) == "gr_02e12cb0_01"
    existing = [
        {"relation_id": "gr_02e12cb0_01", "relation_type": "PROPOSES", "subject_ref": {"kind": "document", "id": "W1"}, "object_ref": {"kind": "mention", "id": "ge_02e12cb0_01"}, "evidence_quote": "x", "evidence_span": {"char_start": 0, "char_end": 1}, "endpoint_mentions": [], "confidence": "stated"},
        {"relation_id": "gr_02e12cb0_03", "relation_type": "PROPOSES", "subject_ref": {"kind": "document", "id": "W1"}, "object_ref": {"kind": "mention", "id": "ge_02e12cb0_01"}, "evidence_quote": "x", "evidence_span": {"char_start": 0, "char_end": 1}, "endpoint_mentions": [], "confidence": "stated"},
    ]
    assert author._next_relation_id(existing, chunk_id) == "gr_02e12cb0_04"
    malformed = [{"relation_id": "r_1", "relation_type": "PROPOSES", "subject_ref": {"kind": "document", "id": "W1"}, "object_ref": {"kind": "mention", "id": "ge_02e12cb0_01"}, "evidence_quote": "x", "evidence_span": {"char_start": 0, "char_end": 1}, "endpoint_mentions": [], "confidence": "stated"}]
    with pytest.raises(author.AuthorError, match="unexpected format"):
        author._next_relation_id(malformed, chunk_id)


# ---------- #69 add-entity / add-relation / mark-empty ----------


def test_add_entity_exact_literal_default_occurrence(tmp_path):
    text = "M uses D to measure T with K."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    author.add_entity(paths, draft, "method", "M", "method M", "stated", None, None)
    written = json.loads(draft.read_text(encoding="utf-8"))
    assert written["entities"][0]["entity_id"] == "ge_00000000_01"
    assert written["entities"][0]["span"] == {"char_start": 0, "char_end": 1}
    assert len(written["entities"][0]["occurrences_in_chunk"]) == 1


def test_add_entity_repeated_literal_enumerates_all(tmp_path):
    text = "DPR then DPR"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    author.add_entity(paths, draft, "method", "DPR", "DPR", "stated", None, None)
    written = json.loads(draft.read_text(encoding="utf-8"))
    occurrences = written["entities"][0]["occurrences_in_chunk"]
    assert [item["occurrence_index"] for item in occurrences] == [0, 1]
    assert written["entities"][0]["span"]["char_start"] == 0


def test_add_entity_explicit_occurrence_index_requires_note(tmp_path):
    text = "DPR then DPR"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    with pytest.raises(author.AuthorError, match="--note"):
        author.add_entity(paths, draft, "method", "DPR", "DPR", "stated", 1, None)
    author.add_entity(paths, draft, "method", "DPR", "DPR", "stated", 1, "human note")
    written = json.loads(draft.read_text(encoding="utf-8"))
    assert written["entities"][0]["span"]["char_start"] == text.find("DPR", 1)
    assert written["note"] == "human note"


def test_add_entity_invalid_occurrence_index_rejected(tmp_path):
    text = "DPR then DPR"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    with pytest.raises(author.AuthorError, match="out of range"):
        author.add_entity(paths, draft, "method", "DPR", "DPR", "stated", 5, None)


def test_add_entity_duplicate_type_surface_rejected(tmp_path):
    text = "M uses M"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    author.add_entity(paths, draft, "method", "M", "M", "stated", None, None)
    with pytest.raises(author.AuthorError, match="duplicate"):
        author.add_entity(paths, draft, "method", "M", "M", "stated", None, None)


@pytest.mark.parametrize("invalid_type", ["Method", "tool", ""])
def test_add_entity_invalid_type_rejected(tmp_path, invalid_type):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    with pytest.raises(author.AuthorError):
        author.add_entity(paths, draft, invalid_type, "M", "M", "stated", None, None)


@pytest.mark.parametrize("invalid_confidence", ["STATED", "guessed", ""])
def test_add_entity_invalid_confidence_rejected(tmp_path, invalid_confidence):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    with pytest.raises(author.AuthorError):
        author.add_entity(paths, draft, "method", "M", "M", invalid_confidence, None, None)


def test_add_entity_invalid_literal_not_in_text_rejected(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    with pytest.raises(author.AuthorError, match="not an exact occurrence"):
        author.add_entity(paths, draft, "method", "NotHere", "NotHere", "stated", None, None)


def test_add_entity_canonical_form_does_not_normalize(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    author.add_entity(paths, draft, "method", "M", "Canonical-FoRm", "stated", None, None)
    written = json.loads(draft.read_text(encoding="utf-8"))
    assert written["entities"][0]["canonical_form"] == "Canonical-FoRm"
    assert written["entities"][0]["surface_form"] == "M"


def test_add_entity_deterministic_id_sequence(tmp_path):
    text = "M D K"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    author.add_entity(paths, draft, "method", "M", "M", "stated", None, None)
    author.add_entity(paths, draft, "dataset", "D", "D", "stated", None, None)
    written = json.loads(draft.read_text(encoding="utf-8"))
    assert [entity["entity_id"] for entity in written["entities"]] == ["ge_00000000_01", "ge_00000000_02"]


def test_add_entity_empty_canonical_form_fails_and_leaves_draft_unchanged(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    before = draft.read_bytes()
    with pytest.raises(author.AuthorError, match="canonical_form"):
        author.add_entity(paths, draft, "method", "M", "", "stated", None, None)
    assert draft.read_bytes() == before


def test_add_entity_corrupted_draft_fails_without_write(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    draft = tmp_path / "draft.json"
    _json(draft, {"chunk_id": chunk_id, "doc_id": "W1", "entities": "not-a-list", "relations": "not-a-list"})
    before = draft.read_bytes()
    with pytest.raises(author.AuthorError, match="entities and relations must be lists"):
        author.add_entity(paths, draft, "method", "M", "M", "stated", None, None)
    assert draft.read_bytes() == before


def test_add_entity_cli_writes_draft(tmp_path, capsys, monkeypatch):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = tmp_path / "draft.json"
    _json(draft, record)
    monkeypatch.setattr(author, "_default_paths", lambda: paths)
    code = author.main(["add-entity", "--input", str(draft), "--entity-type", "method", "--surface-form", "M", "--canonical-form", "M", "--confidence", "stated"])
    assert code == 0
    written = json.loads(draft.read_text(encoding="utf-8"))
    assert written["entities"][0]["surface_form"] == "M"
    assert written["entities"][0]["entity_id"] == "ge_00000000_01"


def test_add_relation_cli_resolves_document_token(tmp_path, capsys, monkeypatch):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = tmp_path / "draft.json"
    _json(draft, record)
    author.add_entity(paths, draft, "method", "M", "M", "stated", None, None)
    written = json.loads(draft.read_text(encoding="utf-8"))
    method_id = written["entities"][0]["entity_id"]
    monkeypatch.setattr(author, "_default_paths", lambda: paths)
    code = author.main([
        "add-relation", "--input", str(draft),
        "--relation-type", "PROPOSES",
        "--subject-ref", "document",
        "--object-ref", method_id,
        "--evidence-quote", text,
        "--confidence", "stated",
    ])
    assert code == 0
    final = json.loads(draft.read_text(encoding="utf-8"))
    assert final["relations"][0]["subject_ref"] == {"kind": "document", "id": "W1"}
    assert final["relations"][0]["object_ref"] == {"kind": "mention", "id": method_id}


def test_add_relation_cli_rejects_document_for_mention_only_type(tmp_path, capsys, monkeypatch):
    text = "alpha and beta."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = tmp_path / "draft.json"
    _json(draft, record)
    author.add_entity(paths, draft, "method", "alpha", "alpha", "stated", None, None)
    written = json.loads(draft.read_text(encoding="utf-8"))
    a_id = next(e["entity_id"] for e in written["entities"] if e["surface_form"] == "alpha")
    monkeypatch.setattr(author, "_default_paths", lambda: paths)
    code = author.main([
        "add-relation", "--input", str(draft),
        "--relation-type", "EXTENDS",
        "--subject-ref", "document",
        "--object-ref", a_id,
        "--evidence-quote", text,
        "--confidence", "stated",
    ])
    assert code == 2
    assert "ontology expects" in capsys.readouterr().err


def test_add_relation_cli_rejects_unknown_entity_token(tmp_path, capsys, monkeypatch):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = tmp_path / "draft.json"
    _json(draft, record)
    monkeypatch.setattr(author, "_default_paths", lambda: paths)
    code = author.main([
        "add-relation", "--input", str(draft),
        "--relation-type", "PROPOSES",
        "--subject-ref", "document",
        "--object-ref", "ge_00000000_99",
        "--evidence-quote", text,
        "--confidence", "stated",
    ])
    assert code == 2
    assert "unknown entity_id" in capsys.readouterr().err


def test_add_relation_invalid_existing_record_fails_without_write(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    draft = tmp_path / "draft.json"
    bad_record = author.template(paths, chunk_id, "ann-A")
    bad_record["entities"] = [{
        "entity_id": "ge_00000000_01",
        "entity_type": "method",
        "surface_form": "M",
        "span": {"char_start": 0, "char_end": 1},
        "occurrences_in_chunk": [{"occurrence_index": 0, "char_start": 0, "char_end": 1}],
        "confidence": "stated",
    }]
    _json(draft, bad_record)
    before = draft.read_bytes()
    with pytest.raises(author.AuthorError, match="entity schema"):
        author.add_relation(paths, draft, "PROPOSES", {"kind": "document", "id": "W1"}, {"kind": "mention", "id": "ge_00000000_01"}, text, "stated", None)
    assert draft.read_bytes() == before


def test_add_relation_repeated_evidence_requires_index(tmp_path):
    text = "D uses M uses M"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    author.add_entity(paths, draft, "method", "uses", "uses", "stated", 1, "human note")
    written = json.loads(draft.read_text(encoding="utf-8"))
    method_id = written["entities"][0]["entity_id"]
    with pytest.raises(author.AuthorError, match="evidence-occurrence-index"):
        author.add_relation(paths, draft, "PROPOSES", {"kind": "document", "id": "W1"}, {"kind": "mention", "id": method_id}, "uses M", "stated", None)
    author.add_relation(paths, draft, "PROPOSES", {"kind": "document", "id": "W1"}, {"kind": "mention", "id": method_id}, "uses M", "stated", 1)
    final = json.loads(draft.read_text(encoding="utf-8"))
    second = text.find("uses M")
    second = text.find("uses M", second + 1)
    assert final["relations"][0]["evidence_span"]["char_start"] == second


def test_add_relation_evidence_containment_required(tmp_path):
    text = "M uses D to measure T with K."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    author.add_entity(paths, draft, "method", "M", "M", "stated", None, None)
    written = json.loads(draft.read_text(encoding="utf-8"))
    method_id = written["entities"][0]["entity_id"]
    with pytest.raises(author.AuthorError, match="contained in evidence"):
        author.add_relation(paths, draft, "PROPOSES", {"kind": "document", "id": "W1"}, {"kind": "mention", "id": method_id}, "measure T with K.", "stated", None)


def test_add_relation_evidence_out_of_range_rejected(tmp_path):
    text = "M uses D to measure T with K."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    author.add_entity(paths, draft, "method", "M", "M", "stated", None, None)
    written = json.loads(draft.read_text(encoding="utf-8"))
    method_id = written["entities"][0]["entity_id"]
    with pytest.raises(author.AuthorError, match="out of range"):
        author.add_relation(paths, draft, "PROPOSES", {"kind": "document", "id": "W1"}, {"kind": "mention", "id": method_id}, "M uses D", "stated", 9)


def test_add_relation_endpoint_must_be_current_doc(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    author.add_entity(paths, draft, "method", "M", "M", "stated", None, None)
    written = json.loads(draft.read_text(encoding="utf-8"))
    method_id = written["entities"][0]["entity_id"]
    with pytest.raises(author.AuthorError, match="document"):
        author.add_relation(paths, draft, "PROPOSES", {"kind": "document", "id": "OTHER"}, {"kind": "mention", "id": method_id}, text, "stated", None)


def test_add_relation_unknown_type_rejected(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    author.add_entity(paths, draft, "method", "M", "M", "stated", None, None)
    written = json.loads(draft.read_text(encoding="utf-8"))
    method_id = written["entities"][0]["entity_id"]
    with pytest.raises(author.AuthorError, match="not extracted"):
        author.add_relation(paths, draft, "CITES", {"kind": "document", "id": "W1"}, {"kind": "mention", "id": method_id}, text, "stated", None)


def test_add_relation_mention_endpoint_type_mismatch(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    author.add_entity(paths, draft, "dataset", "D", "D", "stated", None, None)
    written = json.loads(draft.read_text(encoding="utf-8"))
    dataset_id = written["entities"][0]["entity_id"]
    with pytest.raises(author.AuthorError, match="type mismatch"):
        author.add_relation(paths, draft, "PROPOSES", {"kind": "document", "id": "W1"}, {"kind": "mention", "id": dataset_id}, text, "stated", None)


def test_add_relation_deterministic_id_sequence(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    author.add_entity(paths, draft, "method", "M", "M", "stated", None, None)
    author.add_entity(paths, draft, "dataset", "D", "D", "stated", None, None)
    written = json.loads(draft.read_text(encoding="utf-8"))
    method_id = next(entity["entity_id"] for entity in written["entities"] if entity["entity_type"] == "method")
    dataset_id = next(entity["entity_id"] for entity in written["entities"] if entity["entity_type"] == "dataset")
    author.add_relation(paths, draft, "INTRODUCES", {"kind": "document", "id": "W1"}, {"kind": "mention", "id": dataset_id}, text, "stated", None)
    author.add_relation(paths, draft, "PROPOSES", {"kind": "document", "id": "W1"}, {"kind": "mention", "id": method_id}, text, "stated", None)
    final = json.loads(draft.read_text(encoding="utf-8"))
    assert [relation["relation_id"] for relation in final["relations"]] == ["gr_00000000_01", "gr_00000000_02"]


def test_add_relation_repeated_endpoint_mention_index(tmp_path):
    text = "M M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    author.add_entity(paths, draft, "method", "M", "M", "stated", 1, "human note")
    written = json.loads(draft.read_text(encoding="utf-8"))
    method_id = written["entities"][0]["entity_id"]
    author.add_relation(paths, draft, "PROPOSES", {"kind": "document", "id": "W1"}, {"kind": "mention", "id": method_id}, text, "stated", None)
    final = json.loads(draft.read_text(encoding="utf-8"))
    mentions = final["relations"][0]["endpoint_mentions"]
    assert mentions == [{"endpoint_role": "object", "occurrence_index": 1}]


def test_add_relation_cli_writes_draft(tmp_path, capsys, monkeypatch):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = tmp_path / "draft.json"
    _json(draft, record)
    author.add_entity(paths, draft, "method", "M", "M", "stated", None, None)
    written = json.loads(draft.read_text(encoding="utf-8"))
    method_id = written["entities"][0]["entity_id"]
    monkeypatch.setattr(author, "_default_paths", lambda: paths)
    code = author.main([
        "add-relation", "--input", str(draft),
        "--relation-type", "PROPOSES",
        "--subject-ref", "document",
        "--object-ref", method_id,
        "--evidence-quote", text,
        "--confidence", "stated",
    ])
    assert code == 0
    final = json.loads(draft.read_text(encoding="utf-8"))
    assert final["relations"][0]["relation_type"] == "PROPOSES"


def test_add_entity_rejects_canonical_destination(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    canonical = paths.destination_dir / f"{chunk_id}.json"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(author.AuthorError, match="canonical"):
        author.add_entity(paths, canonical, "method", "M", "M", "stated", None, None)


def test_add_relation_rejects_canonical_destination(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    canonical = paths.destination_dir / f"{chunk_id}.json"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(author.AuthorError, match="canonical"):
        author.add_relation(paths, canonical, "PROPOSES", {"kind": "document", "id": "W1"}, {"kind": "mention", "id": "ge_00000000_01"}, text, "stated", None)


def test_mark_empty_rejects_non_empty_arrays(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    record["entities"] = [_entity(chunk_id, 1, "method", text, "M")]
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    with pytest.raises(author.AuthorError, match="empty entities and relations"):
        author.mark_empty(paths, draft, "human reason")


def test_mark_empty_rejects_empty_note(tmp_path):
    paths, chunk_id = _fixture(tmp_path)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    with pytest.raises(author.AuthorError, match="--note"):
        author.mark_empty(paths, draft, "")


def test_mark_empty_sets_note(tmp_path):
    paths, chunk_id = _fixture(tmp_path)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    author.mark_empty(paths, draft, "human reason")
    written = json.loads(draft.read_text(encoding="utf-8"))
    assert written["entities"] == []
    assert written["relations"] == []
    assert written["note"] == "human reason"


def test_mark_empty_invalid_metadata_fails_without_write(tmp_path):
    paths, chunk_id = _fixture(tmp_path)
    draft = tmp_path / "draft.json"
    bad_record = author.template(paths, chunk_id, "ann-A")
    bad_record["unknown_field"] = 1
    _json(draft, bad_record)
    before = draft.read_bytes()
    with pytest.raises(author.AuthorError, match="unknown or missing"):
        author.mark_empty(paths, draft, "human reason")
    assert draft.read_bytes() == before


def test_validate_cli_succeeds_for_valid_record(tmp_path, capsys, monkeypatch):
    text = "M uses D to measure T with K."
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = _record(paths, chunk_id, text)
    record["note"] = "no"
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    monkeypatch.setattr(author, "_default_paths", lambda: paths)
    code = author.main(["validate", "--input", str(draft)])
    assert code == 0


def test_validate_cli_fails_for_invalid_record(tmp_path, capsys, monkeypatch):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    record["entities"] = [_entity(chunk_id, 1, "method", text, "M")]
    record["unknown_field"] = 1
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    monkeypatch.setattr(author, "_default_paths", lambda: paths)
    code = author.main(["validate", "--input", str(draft)])
    assert code == 2


def test_validate_cli_rejects_canonical_destination(tmp_path, capsys, monkeypatch):
    paths, chunk_id = _fixture(tmp_path)
    canonical = paths.destination_dir / f"{chunk_id}.json"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(author, "_default_paths", lambda: paths)
    code = author.main(["validate", "--input", str(canonical)])
    assert code == 2


def test_mark_empty_rejects_canonical_destination(tmp_path):
    paths, chunk_id = _fixture(tmp_path)
    canonical = paths.destination_dir / f"{chunk_id}.json"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(author.AuthorError, match="canonical"):
        author.mark_empty(paths, canonical, "human reason")


# ---------- #69 atomic draft-write ----------


def test_atomic_write_draft_leaves_existing_on_replace_failure(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = tmp_path / "draft.json"
    _json(draft, record)
    initial_bytes = draft.read_bytes()
    original_replace = author.os.replace
    def boom(*args, **kwargs):
        raise OSError("simulated")
    author.os.replace = boom
    try:
        with pytest.raises(author.AuthorError, match="failed to write draft"):
            author._atomic_write_draft(draft, b"different payload\n")
    finally:
        author.os.replace = original_replace
    assert draft.read_bytes() == initial_bytes



# ---------- #69 final-recovery _write_draft no-op semantics ----------


def test_write_draft_creates_when_absent(tmp_path):
    paths, chunk_id = _fixture(tmp_path)
    target = tmp_path / "draft.json"
    payload = b"hello\n"
    author._write_draft(target, payload)
    assert target.read_bytes() == payload


def test_write_draft_identical_bytes_is_noop(tmp_path):
    paths, chunk_id = _fixture(tmp_path)
    target = tmp_path / "draft.json"
    payload = b"hello\n"
    author._write_draft(target, payload)
    mtime_before = target.stat().st_mtime_ns
    before = target.read_bytes()
    author._write_draft(target, payload)
    assert target.read_bytes() == before
    assert target.stat().st_mtime_ns == mtime_before


def test_write_draft_different_bytes_rejected(tmp_path):
    paths, chunk_id = _fixture(tmp_path)
    target = tmp_path / "draft.json"
    target.write_bytes(b"original\n")
    before = target.read_bytes()
    with pytest.raises(author.AuthorError, match="different bytes"):
        author._write_draft(target, b"different\n")
    assert target.read_bytes() == before


def test_template_cli_first_call_writes_draft(tmp_path, capsys, monkeypatch):
    paths, chunk_id = _fixture(tmp_path)
    monkeypatch.setattr(author, "_default_paths", lambda: paths)
    target = tmp_path / "draft.json"
    code = author.main(["template", "--chunk-id", chunk_id, "--annotator-id", "ann-A", "--output", str(target)])
    assert code == 0
    assert target.exists()
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["chunk_id"] == chunk_id
    assert written["entities"] == []


def test_template_cli_second_identical_call_is_noop(tmp_path, capsys, monkeypatch):
    paths, chunk_id = _fixture(tmp_path)
    monkeypatch.setattr(author, "_default_paths", lambda: paths)
    target = tmp_path / "draft.json"
    author.main(["template", "--chunk-id", chunk_id, "--annotator-id", "ann-A", "--output", str(target)])
    before = target.read_bytes()
    code = author.main(["template", "--chunk-id", chunk_id, "--annotator-id", "ann-A", "--output", str(target)])
    assert code == 0
    assert target.read_bytes() == before


def test_template_cli_different_bytes_rejected(tmp_path, capsys, monkeypatch):
    paths, chunk_id = _fixture(tmp_path)
    monkeypatch.setattr(author, "_default_paths", lambda: paths)
    target = tmp_path / "draft.json"
    author.main(["template", "--chunk-id", chunk_id, "--annotator-id", "ann-A", "--output", str(target)])
    target.write_bytes(b"tampered\n")
    code = author.main(["template", "--chunk-id", chunk_id, "--annotator-id", "ann-A", "--output", str(target)])
    captured = capsys.readouterr()
    assert code == 2
    assert "different bytes" in captured.err
    assert target.read_bytes() == b"tampered\n"


def test_template_cli_canonical_destination_still_rejected(tmp_path, capsys, monkeypatch):
    paths, chunk_id = _fixture(tmp_path)
    monkeypatch.setattr(author, "_default_paths", lambda: paths)
    target = paths.destination_dir / f"{chunk_id}.json"
    code = author.main(["template", "--chunk-id", chunk_id, "--annotator-id", "ann-A", "--output", str(target)])
    captured = capsys.readouterr()
    assert code == 2
    assert "canonical" in captured.err
    assert not target.exists()


# ---------- #69 final-recovery baseline-draft validation ----------


def test_pristine_template_allows_add_entity(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    written = author.add_entity(paths, draft, "method", "M", "M", "stated", None, None)
    assert len(written["entities"]) == 1


def test_pristine_template_allows_mark_empty(tmp_path):
    paths, chunk_id = _fixture(tmp_path)
    record = author.template(paths, chunk_id, "ann-A")
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    written = author.mark_empty(paths, draft, "human reason")
    assert written["note"] == "human reason"


def test_add_entity_malformed_entity_object_fails_closed(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    record["entities"] = [{"not_a_valid_entity": True}]
    record["relations"] = []
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    before = draft.read_bytes()
    with pytest.raises(author.AuthorError, match="entity schema|entity_id"):
        author.add_entity(paths, draft, "method", "M", "M", "stated", None, None)
    assert draft.read_bytes() == before


def test_add_entity_invalid_top_level_note_fails_closed(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    record["entities"] = []
    record["relations"] = []
    record["note"] = 123
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    before = draft.read_bytes()
    with pytest.raises(author.AuthorError):
        author.add_entity(paths, draft, "method", "M", "M", "stated", None, "human note")
    assert draft.read_bytes() == before


def test_add_entity_invalid_top_level_note_fails_even_without_user_note(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    record["entities"] = []
    record["relations"] = []
    record["note"] = 123
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    before = draft.read_bytes()
    with pytest.raises(author.AuthorError):
        author.add_entity(paths, draft, "method", "M", "M", "stated", None, None)
    assert draft.read_bytes() == before


def test_mark_empty_invalid_top_level_note_fails_closed(tmp_path):
    paths, chunk_id = _fixture(tmp_path)
    record = author.template(paths, chunk_id, "ann-A")
    record["entities"] = []
    record["relations"] = []
    record["note"] = 123
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    before = draft.read_bytes()
    with pytest.raises(author.AuthorError):
        author.mark_empty(paths, draft, "human reason")
    assert draft.read_bytes() == before


def test_add_entity_invalid_note_is_not_repaired(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    record["entities"] = []
    record["relations"] = []
    record["note"] = 123
    draft = _seed_draft(tmp_path, paths, chunk_id, record)
    before = draft.read_bytes()
    with pytest.raises(author.AuthorError):
        author.add_entity(paths, draft, "method", "M", "M", "stated", None, None)
    on_disk = json.loads(draft.read_text(encoding="utf-8"))
    assert on_disk["note"] == 123


def test_validate_baseline_draft_accepts_pristine_template(tmp_path):
    paths, chunk_id = _fixture(tmp_path)
    record = author.template(paths, chunk_id, "ann-A")
    author._validate_baseline_draft(record, paths)


def test_validate_baseline_draft_accepts_non_pristine_valid_record(tmp_path):
    text = "M uses D"
    paths, chunk_id = _fixture(tmp_path, text=text)
    record = author.template(paths, chunk_id, "ann-A")
    record["entities"] = [{
        "entity_id": "ge_00000000_01",
        "entity_type": "method",
        "surface_form": "M",
        "canonical_form": "M",
        "span": {"char_start": 0, "char_end": 1},
        "occurrences_in_chunk": [{"occurrence_index": 0, "char_start": 0, "char_end": 1}],
        "confidence": "stated",
    }]
    author._validate_baseline_draft(record, paths)


def test_validate_baseline_draft_rejects_malformed_entities(tmp_path):
    paths, chunk_id = _fixture(tmp_path)
    record = author.template(paths, chunk_id, "ann-A")
    record["entities"] = [{"oops": True}]
    record["relations"] = []
    with pytest.raises(author.AuthorError, match="entity schema|entity_id"):
        author._validate_baseline_draft(record, paths)


def test_validate_baseline_draft_rejects_invalid_note_type(tmp_path):
    paths, chunk_id = _fixture(tmp_path)
    record = author.template(paths, chunk_id, "ann-A")
    record["note"] = {"nested": "wrong"}
    with pytest.raises(author.AuthorError):
        author._validate_baseline_draft(record, paths)
