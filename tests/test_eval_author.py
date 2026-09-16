"""Synthetic Phase-1 tests for ``src.eval.author`` (#31)."""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.author import validate_draft, validate_draft_with_graph  # noqa: E402
import src.eval.author as author  # noqa: E402


DOC = "W100"
OTHER_DOC = "W101"
CHUNK = "c_0123456789abcdef"
ENTITY = "e_abcdef0123456789"
OTHER_ENTITY = "e_1111111111111111"


def record(qid="q_001", bucket="1"):
    path = [{"step": 1, "from": DOC, "relation": "CITES", "to": ENTITY, "direction": "forward"}]
    if bucket == "2":
        path.append({"step": 2, "from": ENTITY, "relation": "PROPOSES", "to": DOC, "direction": "reverse"})
    if bucket == "3+":
        path.extend([
            {"step": 2, "from": ENTITY, "relation": "PROPOSES", "to": DOC, "direction": "reverse"},
            {"step": 3, "from": DOC, "relation": "CITES", "to": ENTITY, "direction": "forward"},
        ])
    return {"qid": qid, "question": "Synthetic question", "expected_answer": "Synthetic answer", "hop_bucket": bucket, "evidence_path": path, "supporting_doc_ids": [DOC], "supporting_chunk_ids": [CHUNK], "split": "held_out"}


class FakeGraph:
    def __init__(self, *, nodes=(), edges=(), fail=False):
        self.nodes = set(nodes)
        self.edges = set(edges)
        self.fail = fail

    def node_exists(self, node_id):
        if self.fail:
            raise RuntimeError("offline")
        return node_id in self.nodes

    def stored_edge_exists(self, source_id, relation, target_id):
        if self.fail:
            raise RuntimeError("offline")
        return (source_id, relation, target_id) in self.edges


class EvalFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.jsonl"
        self.chunks = self.root / "chunks.jsonl"
        self.ontology = self.root / "ontology.yaml"
        self.manifest.write_text(json.dumps({"openalex_id": DOC}) + "\n", encoding="utf-8")
        self.chunks.write_text(json.dumps({"chunk_id": CHUNK}) + "\n", encoding="utf-8")
        self.ontology.write_text("relation_types:\n  - name: CITES\n  - name: PROPOSES\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def validate(self, records, *, dataset=False):
        draft = self.root / "draft.jsonl"
        draft.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
        return validate_draft(draft, manifest_path=self.manifest, chunks_path=self.chunks, ontology_path=self.ontology, require_dataset=dataset)

    def assert_invalid(self, records, text, *, dataset=False):
        report = self.validate(records, dataset=dataset)
        self.assertFalse(report.ok, [str(item) for item in report.violations])
        self.assertTrue(any(text in item.message for item in report.violations), [str(item) for item in report.violations])


class TestRecordValidation(EvalFixture):
    def test_valid_schema_record_accepted(self):
        self.assertTrue(self.validate([record()]).ok)

    def test_missing_required_field_rejected(self):
        value = record(); del value["question"]
        self.assert_invalid([value], "missing required field")

    def test_malformed_qid_and_non_string_field_rejected(self):
        value = record(qid="question-1"); value["question"] = 1
        self.assert_invalid([value], "qid must match")
        self.assert_invalid([value], "question must be a string")

    def test_duplicate_qids_rejected(self):
        self.assert_invalid([record(), record()], "duplicate qid")

    def test_dev_and_train_splits_rejected(self):
        for split in ("dev", "train"):
            with self.subTest(split=split):
                value = record(); value["split"] = split
                self.assert_invalid([value], "split must be exactly")

    def test_empty_and_malformed_supporting_ids_rejected(self):
        for field, value, text in (
            ("supporting_doc_ids", [], "supporting_doc_ids must be a non-empty"),
            ("supporting_chunk_ids", [], "supporting_chunk_ids must be a non-empty"),
            ("supporting_doc_ids", ["bad"], "supporting_doc_ids values must match"),
            ("supporting_chunk_ids", ["c_bad"], "supporting_chunk_ids values must match"),
        ):
            with self.subTest(field=field, value=value):
                item = record(); item[field] = value
                self.assert_invalid([item], text)

    def test_unknown_supporting_references_rejected(self):
        value = record(); value["supporting_doc_ids"] = ["W999"]
        self.assert_invalid([value], "absent from manifest")
        value = record(); value["supporting_chunk_ids"] = ["c_1111111111111111"]
        self.assert_invalid([value], "absent from chunks artifact")

    def test_optional_fields_are_typed(self):
        value = record(); value["answer_aliases"] = ["answer", 1]
        self.assert_invalid([value], "answer_aliases must be")
        value = record(); value["notes"] = []
        self.assert_invalid([value], "notes must be a string")


class TestPathValidation(EvalFixture):
    def test_boolean_step_rejected(self):
        value = record(); value["evidence_path"][0]["step"] = True
        self.assert_invalid([value], "step must be an int")

    def test_document_path_id_must_be_supported_and_canonical(self):
        value = record(); value["evidence_path"][0]["from"] = "W999"; value["supporting_doc_ids"] = ["W999"]
        self.assert_invalid([value], "absent from manifest")
        value = record(); value["evidence_path"][0]["from"] = "entity"
        self.assert_invalid([value], "must be a canonical")
        value = record(); value["evidence_path"][0]["to"] = "W101"
        self.assert_invalid([value], "must contain every W ID")

    def test_disconnected_path_invalid_direction_and_unknown_relation_rejected(self):
        value = record(bucket="2"); value["evidence_path"][1]["from"] = "e_1111111111111111"
        self.assert_invalid([value], "path is disconnected")
        value = record(); value["evidence_path"][0]["direction"] = "sideways"
        self.assert_invalid([value], "direction must be")
        value = record(); value["evidence_path"][0]["relation"] = "UNKNOWN"
        self.assert_invalid([value], "absent from ontology")

    def test_hop_lengths_rejected_when_mismatched(self):
        for bucket, path in (("1", record("q_001", "2")["evidence_path"]), ("2", record()["evidence_path"]), ("3+", record("q_001", "2")["evidence_path"])):
            with self.subTest(bucket=bucket):
                value = record(); value["hop_bucket"] = bucket; value["evidence_path"] = path
                self.assert_invalid([value], "length does not match")


class TestArtifactFailures(EvalFixture):
    def test_malformed_artifacts_fail_closed(self):
        originals = {
            self.manifest: self.manifest.read_text(encoding="utf-8"),
            self.chunks: self.chunks.read_text(encoding="utf-8"),
            self.ontology: self.ontology.read_text(encoding="utf-8"),
        }
        for path, contents, text in ((self.manifest, "{\n", "manifest"), (self.chunks, "{\n", "chunks"), (self.ontology, "relation_types: [\n", "ontology")):
            with self.subTest(path=path):
                path.write_text(contents, encoding="utf-8")
                self.assert_invalid([record()], text)
                path.write_text(originals[path], encoding="utf-8")
    def test_invalid_utf8_artifacts_fail_closed(self):
        for path, text in ((self.manifest, "manifest"), (self.chunks, "chunks"), (self.ontology, "ontology")):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(b"\x80")
                self.assert_invalid([record()], text)
                path.write_bytes(original)

    def test_malformed_draft_json_fails_closed(self):
        draft = self.root / "draft.jsonl"
        draft.write_text("{\n", encoding="utf-8")
        report = validate_draft(draft, manifest_path=self.manifest, chunks_path=self.chunks, ontology_path=self.ontology)
        self.assertFalse(report.ok)
        self.assertIn("malformed JSON", report.violations[0].message)
    def test_invalid_utf8_draft_fails_closed(self):
        draft = self.root / "draft.jsonl"
        draft.write_bytes(b"\x80")
        report = validate_draft(draft, manifest_path=self.manifest, chunks_path=self.chunks, ontology_path=self.ontology)
        self.assertFalse(report.ok)
        self.assertIn("invalid UTF-8", report.violations[0].message)


    def test_missing_artifacts_fail_closed(self):
        self.manifest.unlink()
        self.assert_invalid([record()], "manifest")
    def test_every_missing_canonical_artifact_fails_closed(self):
        for path, text in ((self.manifest, "manifest"), (self.chunks, "chunks"), (self.ontology, "ontology")):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.unlink()
                self.assert_invalid([record()], text)
                path.write_bytes(original)

    def test_draft_blank_and_non_object_lines_fail_closed(self):
        draft = self.root / "draft.jsonl"
        draft.write_text("\n[]\n", encoding="utf-8")
        report = validate_draft(draft, manifest_path=self.manifest, chunks_path=self.chunks, ontology_path=self.ontology)
        self.assertFalse(report.ok)
        self.assertEqual(2, len(report.violations))


class TestDatasetValidation(EvalFixture):
    def test_44_and_46_records_rejected(self):
        for count in (44, 46):
            with self.subTest(count=count):
                records = [record(f"q_{index:03}", "1") for index in range(count)]
                self.assert_invalid(records, "exactly 45 records", dataset=True)

    def test_unbalanced_distribution_rejected(self):
        records = [record(f"q_{index:03}", "1" if index < 5 else "2") for index in range(45)]
        self.assert_invalid(records, "hop bucket", dataset=True)

    def test_boundary_distribution_and_valid_dataset_pass(self):
        records = ([record(f"q_{index:03}", "1") for index in range(12)] + [record(f"q_{index:03}", "2") for index in range(12, 30)] + [record(f"q_{index:03}", "3+") for index in range(30, 45)])
        report = self.validate(records, dataset=True)
        self.assertTrue(report.ok, [str(item) for item in report.violations])
        self.assertEqual((("1", 12), ("2", 18), ("3+", 15)), report.bucket_counts)

    def test_every_record_must_be_held_out(self):
        records = [record(f"q_{index:03}", "1") for index in range(45)]
        records[0]["split"] = "dev"
        self.assert_invalid(records, "split must be exactly", dataset=True)




class TestGraphPathValidation(EvalFixture):
    def validate_graph(self, records, graph):
        draft = self.root / "draft.jsonl"
        draft.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
        return validate_draft_with_graph(draft, graph, manifest_path=self.manifest, chunks_path=self.chunks, ontology_path=self.ontology)

    def assert_graph_invalid(self, records, graph, text):
        report = self.validate_graph(records, graph)
        self.assertFalse(report.ok, [str(item) for item in report.violations])
        self.assertTrue(any(text in item.message for item in report.violations), [str(item) for item in report.violations])

    def test_existing_document_and_entity_nodes_are_accepted(self):
        graph = FakeGraph(nodes={DOC, ENTITY}, edges={(DOC, "CITES", ENTITY)})
        self.assertTrue(self.validate_graph([record()], graph).ok)

    def test_missing_document_or_entity_node_is_rejected(self):
        self.assert_graph_invalid([record()], FakeGraph(nodes={ENTITY}, edges={(DOC, "CITES", ENTITY)}), "graph Document node")
        self.assert_graph_invalid([record()], FakeGraph(nodes={DOC}, edges={(DOC, "CITES", ENTITY)}), "graph Entity node")

    def test_exact_forward_direction_and_relation_are_required(self):
        graph = FakeGraph(nodes={DOC, ENTITY}, edges={(ENTITY, "CITES", DOC)})
        self.assert_graph_invalid([record()], graph, "required stored edge")
        graph = FakeGraph(nodes={DOC, ENTITY}, edges={(DOC, "PROPOSES", ENTITY)})
        self.assert_graph_invalid([record()], graph, "required stored edge")
        graph = FakeGraph(nodes={DOC, ENTITY}, edges={(DOC, "CITES", OTHER_ENTITY)})

    def test_exact_reverse_direction_is_required(self):
        value = record()
        value["evidence_path"][0].update(from_=ENTITY)
        value["evidence_path"][0]["from"] = ENTITY
        value["evidence_path"][0]["to"] = DOC
        value["evidence_path"][0]["relation"] = "PROPOSES"
        value["evidence_path"][0]["direction"] = "reverse"
        graph = FakeGraph(nodes={DOC, ENTITY}, edges={(DOC, "PROPOSES", ENTITY)})
        self.assertTrue(self.validate_graph([value], graph).ok)
        self.assert_graph_invalid([value], FakeGraph(nodes={DOC, ENTITY}, edges={(ENTITY, "PROPOSES", DOC)}), "required stored edge")

    def test_every_step_in_mixed_direction_path_is_checked(self):
        value = record(bucket="2")
        graph = FakeGraph(nodes={DOC, ENTITY}, edges={(DOC, "CITES", ENTITY), (DOC, "PROPOSES", ENTITY)})
        self.assertTrue(self.validate_graph([value], graph).ok)
        self.assert_graph_invalid([value], FakeGraph(nodes={DOC, ENTITY}, edges={(DOC, "CITES", ENTITY)}), "required stored edge")

    def test_phase_one_path_defects_prevent_graph_acceptance(self):
        value = record(bucket="2")
        value["evidence_path"][1]["from"] = OTHER_ENTITY
        graph = FakeGraph(nodes={DOC, ENTITY, OTHER_ENTITY}, edges={(DOC, "CITES", ENTITY), (DOC, "PROPOSES", OTHER_ENTITY)})
        self.assert_graph_invalid([value], graph, "path is disconnected")

    def test_graph_reader_failure_fails_closed(self):
        self.assert_graph_invalid([record()], FakeGraph(fail=True), "graph read failed")


class TestNeo4jGraphReader(unittest.TestCase):
    def test_adapter_uses_read_only_session(self):
        sessions = []
        class Result:
            def single(self): return {"found": 1}
        class Session:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def run(self, *_args, **_kwargs): return Result()
        class Driver:
            def session(self, **kwargs):
                sessions.append(kwargs)
                return Session()
        reader = author._Neo4jGraphReader(Driver())
        self.assertTrue(reader.node_exists(DOC))
        self.assertTrue(reader.stored_edge_exists(DOC, "anything", ENTITY))
        self.assertEqual(["READ", "READ"], [session["default_access_mode"] for session in sessions])

    def test_author_module_has_no_writes_retrieval_or_relation_vocabulary(self):
        source = inspect.getsource(author)
        for forbidden in ("CREATE ", "MERGE ", "DELETE ", "DETACH DELETE", " SET ", " REMOVE ", " DROP ", "GraphRetriever", "retrieve("):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        for relation in ("CITES", "PROPOSES", "INTRODUCES", "EVALUATES_ON", "USES_METHOD", "REPORTS_METRIC", "EXTENDS", "EVALUATED_ON", "USED_FOR_TASK", "MEASURES"):
            with self.subTest(relation=relation):
                self.assertNotIn(relation, source)
class TestAuthorshipGuards(unittest.TestCase):
    def test_tool_remains_validation_only(self):
        source = inspect.getsource(author)
        for forbidden in ("--force", "--allow-overwrite", "OVERWRITE_ALLOWED", "openai", "anthropic", "ollama", "benchmark"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source.lower())
        self.assertNotIn("q_045", source)



class TestStrictPublication(EvalFixture):
    def setUp(self):
        super().setUp()
        self.output = self.root / "eval" / "questions.jsonl"
        self.versioned_output = self.root / "eval" / "questions.v2.jsonl"
        self.graph = FakeGraph(
            nodes={DOC, ENTITY},
            edges={(DOC, "CITES", ENTITY), (DOC, "PROPOSES", ENTITY)},
        )

    def draft_bytes(self, *, whitespace=False, final_newline=True):
        records = (
            [record(f"q_{index:03}", "1") for index in range(15)]
            + [record(f"q_{index:03}", "2") for index in range(15, 30)]
            + [record(f"q_{index:03}", "3+") for index in range(30, 45)]
        )
        separator = ", " if whitespace else ","
        data = b"\n".join(json.dumps(item, separators=(separator, ":")).encode("utf-8") for item in records)
        return data + (b"\n" if final_newline else b"")

    def run_cli(self, draft_bytes, *, output=None, graph=None, extra=()):
        draft = self.root / "draft.jsonl"
        draft.write_bytes(draft_bytes)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = author.main(
                ["--input", str(draft), "--output", str(output or self.output), *extra],
                graph_reader_factory=lambda: self.graph if graph is None else graph,
                manifest_path=self.manifest,
                chunks_path=self.chunks,
                ontology_path=self.ontology,
            )
        return result, stdout.getvalue(), stderr.getvalue()

    def test_valid_cli_publishes_exact_bytes_and_sidecar(self):
        candidate = self.draft_bytes(final_newline=False)
        result, stdout, stderr = self.run_cli(candidate)
        self.assertEqual(0, result, stderr)
        self.assertEqual(candidate, self.output.read_bytes())
        digest = hashlib.sha256(candidate).hexdigest()
        self.assertEqual(f"{digest}\n", self.output.with_name("questions.jsonl.sha256").read_text(encoding="ascii"))
        self.assertIn(str(self.output), stdout)
        self.assertIn(digest, stdout)

    def test_strict_cli_rejects_invalid_drafts_without_publication(self):
        invalid = self.draft_bytes().replace(b'"split":"held_out"', b'"split":"dev"', 1)
        result, _, stderr = self.run_cli(invalid)
        self.assertEqual(1, result)
        self.assertIn("split must be exactly", stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name("questions.jsonl.sha256").exists())
        self.assertFalse(list((self.root / "eval").glob(".*")) if (self.root / "eval").exists() else [])

    def test_strict_cli_requires_45_records_and_graph_validity(self):
        forty_four = b"\n".join(self.draft_bytes().splitlines()[:-1]) + b"\n"
        result, _, stderr = self.run_cli(forty_four)
        self.assertEqual(1, result)
        self.assertIn("exactly 45", stderr)
        result, _, stderr = self.run_cli(self.draft_bytes(), graph=FakeGraph(nodes={DOC, ENTITY}))
        self.assertEqual(1, result)
        self.assertIn("required stored edge", stderr)

    def test_canonical_is_frozen_but_repairs_missing_or_stale_sidecar(self):
        candidate = self.draft_bytes()
        self.assertEqual(0, self.run_cli(candidate)[0])
        before_mtime = self.output.stat().st_mtime_ns
        sidecar = self.output.with_name("questions.jsonl.sha256")
        sidecar.unlink()
        self.assertEqual(0, self.run_cli(candidate)[0])
        self.assertEqual(before_mtime, self.output.stat().st_mtime_ns)
        sidecar.write_text("0" * 64 + "\n", encoding="ascii")
        self.assertEqual(0, self.run_cli(candidate)[0])
        self.assertEqual(before_mtime, self.output.stat().st_mtime_ns)
        self.assertEqual(f"{hashlib.sha256(candidate).hexdigest()}\n", sidecar.read_text(encoding="ascii"))

    def test_byte_distinct_canonical_and_versioned_files_are_refused(self):
        candidate = self.draft_bytes()
        changed = self.draft_bytes(whitespace=True)
        self.assertEqual(0, self.run_cli(candidate)[0])
        before_bytes, before_mtime = self.output.read_bytes(), self.output.stat().st_mtime_ns
        before_sidecar = self.output.with_name("questions.jsonl.sha256").read_bytes()
        result, _, stderr = self.run_cli(changed)
        self.assertEqual(1, result)
        self.assertIn("frozen", stderr)
        self.assertEqual(before_bytes, self.output.read_bytes())
        self.assertEqual(before_mtime, self.output.stat().st_mtime_ns)
        self.assertEqual(before_sidecar, self.output.with_name("questions.jsonl.sha256").read_bytes())
        self.assertEqual(0, self.run_cli(candidate, output=self.versioned_output)[0])
        version_mtime = self.versioned_output.stat().st_mtime_ns
        self.assertEqual(0, self.run_cli(candidate, output=self.versioned_output)[0])
        self.assertEqual(version_mtime, self.versioned_output.stat().st_mtime_ns)
        self.assertEqual(1, self.run_cli(changed, output=self.versioned_output)[0])

    def test_version_file_preserves_canonical_and_gets_own_sidecar(self):
        canonical = self.draft_bytes()
        versioned = self.draft_bytes(whitespace=True)
        self.assertEqual(0, self.run_cli(canonical)[0])
        canonical_bytes, canonical_mtime = self.output.read_bytes(), self.output.stat().st_mtime_ns
        canonical_sha = self.output.with_name("questions.jsonl.sha256").read_bytes()
        self.assertEqual(0, self.run_cli(versioned, output=self.versioned_output)[0])
        self.assertEqual(canonical_bytes, self.output.read_bytes())
        self.assertEqual(canonical_mtime, self.output.stat().st_mtime_ns)
        self.assertEqual(canonical_sha, self.output.with_name("questions.jsonl.sha256").read_bytes())
        self.assertEqual(f"{hashlib.sha256(versioned).hexdigest()}\n", self.versioned_output.with_name("questions.v2.jsonl.sha256").read_text(encoding="ascii"))

    def test_cli_exposes_no_bypass_options_or_output_names(self):
        source = inspect.getsource(author)
        for forbidden in ("--force", "--allow-overwrite", "OVERWRITE_ALLOWED", "ALLOW_OVERWRITE", "SKIP_GRAPH", "NO_GRAPH", "NON_STRICT", "ALLOW_EMPTY", "ALLOW_DEV"):
            self.assertNotIn(forbidden, source)
        for option in ("--force", "--allow-overwrite"):
            with self.subTest(option=option):
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    author.main([option])
        for name in ("foo.jsonl", "questions-new.jsonl", "copy.jsonl", "questions.v0.jsonl"):
            with self.subTest(name=name):
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    author.main(["--input", "draft.jsonl", "--output", str(self.root / "eval" / name)])

    def test_injected_graph_reader_is_not_closed(self):
        class ClosingGraph(FakeGraph):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.closed = False
            def close(self):
                self.closed = True
        graph = ClosingGraph(nodes={DOC, ENTITY}, edges={(DOC, "CITES", ENTITY), (DOC, "PROPOSES", ENTITY)})
        self.assertEqual(0, self.run_cli(self.draft_bytes(), graph=graph)[0])
        self.assertFalse(graph.closed)

    def test_production_graph_reader_is_closed_on_success_validation_and_publication_failures(self):
        class ClosingGraph(FakeGraph):
            def __init__(self):
                super().__init__(nodes={DOC, ENTITY}, edges={(DOC, "CITES", ENTITY), (DOC, "PROPOSES", ENTITY)})
                self.closed = False
            def close(self):
                self.closed = True
        draft = self.root / "draft.jsonl"
        for candidate, publication_failure in ((self.draft_bytes(), False), (self.draft_bytes().replace(b'"split":"held_out"', b'"split":"dev"', 1), False), (self.draft_bytes(), True)):
            with self.subTest(publication_failure=publication_failure):
                graph = ClosingGraph()
                draft.write_bytes(candidate)
                patches = [mock.patch.object(author, "default_graph_reader", return_value=graph)]
                if publication_failure:
                    patches.append(mock.patch.object(author, "_atomic_write", side_effect=OSError("disk full")))
                with contextlib.ExitStack() as stack:
                    for patch in patches:
                        stack.enter_context(patch)
                    with contextlib.redirect_stderr(io.StringIO()):
                        author.main(["--input", str(draft), "--output", str(self.output)], manifest_path=self.manifest, chunks_path=self.chunks, ontology_path=self.ontology)
                self.assertTrue(graph.closed)

    def test_new_output_failures_leave_no_publication(self):
        candidate = self.draft_bytes()
        with mock.patch.object(author, "_atomic_write", side_effect=OSError("disk full")):
            result, _, stderr = self.run_cli(candidate)
        self.assertEqual(1, result)
        self.assertIn("cannot publish", stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name("questions.jsonl.sha256").exists())

        calls = 0
        real_atomic_write = author._atomic_write
        def fail_sidecar(path, data):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("disk full")
            real_atomic_write(path, data)
        with mock.patch.object(author, "_atomic_write", side_effect=fail_sidecar):
            result, _, stderr = self.run_cli(candidate)
        self.assertEqual(1, result)
        self.assertIn("cannot publish", stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name("questions.jsonl.sha256").exists())

    def test_existing_output_failure_and_freeze_refusal_preserve_sidecar(self):
        candidate = self.draft_bytes()
        self.assertEqual(0, self.run_cli(candidate)[0])
        sidecar = self.output.with_name("questions.jsonl.sha256")
        before_bytes, before_mtime, before_sidecar = self.output.read_bytes(), self.output.stat().st_mtime_ns, sidecar.read_bytes()
        sidecar.unlink()
        with mock.patch.object(author, "_atomic_write", side_effect=OSError("disk full")):
            self.assertEqual(1, self.run_cli(candidate)[0])
        self.assertEqual(before_bytes, self.output.read_bytes())
        self.assertEqual(before_mtime, self.output.stat().st_mtime_ns)
        self.assertFalse(sidecar.exists())
        sidecar.write_bytes(before_sidecar)
        changed = self.draft_bytes(whitespace=True)
        self.assertEqual(1, self.run_cli(changed)[0])
        sidecar.write_bytes(b"stale\n")
        changed = self.draft_bytes(whitespace=True)
        self.assertEqual(1, self.run_cli(changed)[0])
        self.assertEqual(before_bytes, self.output.read_bytes())
        self.assertEqual(before_mtime, self.output.stat().st_mtime_ns)
        self.assertEqual(b"stale\n", sidecar.read_bytes())

    def test_identical_retry_revalidates_artifacts_and_graph(self):
        candidate = self.draft_bytes()
        self.assertEqual(0, self.run_cli(candidate)[0])
        self.chunks.unlink()
        result, _, stderr = self.run_cli(candidate)
        self.assertEqual(1, result)
        self.assertIn("chunks", stderr)
        self.chunks.write_text(json.dumps({"chunk_id": CHUNK}) + "\n", encoding="utf-8")
        result, _, stderr = self.run_cli(candidate, graph=FakeGraph(nodes={DOC, ENTITY}))
        self.assertEqual(1, result)
        self.assertIn("required stored edge", stderr)

    def test_semantically_equivalent_raw_bytes_are_refused(self):
        candidate = self.draft_bytes()
        self.assertEqual(0, self.run_cli(candidate)[0])
        for changed in (
            self.draft_bytes(whitespace=True),
            self.draft_bytes(final_newline=False),
            self.draft_bytes().replace(b"\n", b"\r\n"),
            self.draft_bytes().replace(b"Synthetic question", "Synthetic que\u0301stion".encode("utf-8")),
        ):
            with self.subTest(candidate=changed[:32]):
                self.assertNotEqual(candidate, changed)
                result, _, stderr = self.run_cli(changed)
                self.assertEqual(1, result)
                self.assertIn("frozen", stderr)

    def test_production_entrypoint_requires_strict_dataset_validation(self):
        source = inspect.getsource(author.main)
        self.assertIn("require_dataset=True", source)

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
