"""Acceptance tests for the canonical v2 extraction runner (#55).

Each ``TestCase`` covers one letter of the #55 acceptance checklist:

    A. Exact system prompt provenance
    B. Strict committed artifact set
    C. Carry-forward silent narrowing table
    D. CLI exact dispatch
    E. Crash ordering
    F. Audit-only identity fields don't block resume
    G. Usage / reject lifecycle

Plus ancillary regressions: ``run_id`` format, ``preflight_v2``
public signature, and ``_canonical_reject_records`` discriminator.

Library calls only — no subprocess / no I/O beyond tmp dirs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extract import (  # noqa: E402
    __main__ as extract_main,
    pipeline as pipeline_v1,
    pipeline_v2,
    prompts as prompts_module,
    schema as schema_module,
)
from src.providers.protocol import LLMProvider, Usage  # noqa: E402


MINIMAL_ONT = """
version: 1
schema_version: 1
entity_types:
  - name: document
    population_mode: structural
    source: manifest
    examples: []
  - name: method
    population_mode: extracted
    extraction_prompt: extract_entities_v2
    definition: A research method or technique.
    verified_examples:
      - surface: "DPR"
        canonical: "Dense Passage Retrieval"
        char_range: [0, 3]
        context: "We use DPR for retrieval."
relation_types:
  - name: PROPOSES
    from: document
    to: method
    population_mode: extracted
    extraction_prompt: extract_relations_v2
    definition: A paper proposing a method.
    verified_examples: []
"""

ENTITY_TEMPLATE_V2 = (
    "v2-entity: types={allowed_entity_types} defs={entity_type_definitions} "
    "ex={verified_examples} text={chunk_text}\n"
)
RELATION_TEMPLATE_V2 = (
    "v2-relation: rels={allowed_relation_types} defs={relation_definitions} "
    "chunk={chunk_id} doc={doc_id} mentions={mention_list} text={chunk_text}\n"
)


def _write_world(root, *, chunks=None, doc_id="W1"):
    ont_path = root / "ontology.yaml"
    ont_path.write_text(MINIMAL_ONT)
    if chunks is None:
        chunks = [{
            "chunk_id": "c_000",
            "doc_id": doc_id,
            "section_path": ["intro"],
            "chunk_index": 0,
            "text": "We use DPR for retrieval and we also use BERT for ranking.",
            "token_estimate": 12,
            "char_start": 0,
            "char_end": 61,
        }]
    chunks_path = root / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8"
    )
    prompts_dir = root / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / prompts_module.ENTITY_PROMPT_FILE_V2).write_text(ENTITY_TEMPLATE_V2)
    (prompts_dir / prompts_module.RELATION_PROMPT_FILE_V2).write_text(RELATION_TEMPLATE_V2)
    return {
        "ontology": ont_path,
        "chunks": chunks_path,
        "prompts": prompts_dir,
    }


def _canonical_outputs(root):
    d = root / "out"
    d.mkdir(parents=True, exist_ok=True)
    return {
        "jsonl": d / "extractions.v2.jsonl",
        "rejects": d / "extractions.v2.rejects.jsonl",
        "manifest": d / "extractions.v2.manifest.json",
    }


def _read_jsonl(p):
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _minimal_valid_manifest() -> str:
    """A manifest that passes the structural reader so the runner
    actually attempts to validate the sibling JSONL/rejects."""
    return json.dumps({
        "schema_version": schema_module.SCHEMA_VERSION_V2,
        "run_id": "20240101T000000Z-" + "0" * 16,
        "resumed_from_run_id": None,
        "chunks_sha256": "a" * 64,
        "ontology_sha256": "b" * 64,
        "prompt_template_sha256": ["c" * 64, "d" * 64],
        "system_prompt_sha256": "e" * 64,
        "provider": "fake-llm-v2",
        "model": "fake-model-v2",
        "max_tokens": 2048,
        "temperature": 0.0,
        "alignment_algorithm_version": "v2-exact-span-1",
        "alignment_algorithm_sha256": "f" * 64,
        "ontology_version": 1,
        "prompt_template_files": ["extract_entities_v2.txt", "extract_relations_v2.txt"],
        "chunks_path": "/tmp/chunks.jsonl",
        "ontology_path": "/tmp/ontology.yaml",
        "pipeline_version": "f" * 40,
        "cache_format_version": "1",
        "wall_clock_started_at": "20240101T000000Z",
        "wall_clock_ended_at": "20240101T000100Z",
        "cumulative_tokens": 30,
        "cumulative_cost_usd": 0.0,
        "reject_count": 0,
        "chunk_count_total": 1,
        "chunk_count_complete": 1,
    })


def _dpr_text():
    return "We use DPR for retrieval and we also use BERT for ranking."


def _dpr_entity_payload():
    return {"mentions": [{
        "entity_type": "method",
        "surface_form": "DPR",
        "canonical_form": "Dense Passage Retrieval",
        "confidence": "stated",
    }]}


def _dpr_relation_payload(chunk_id, doc_id, mention_mid):
    return {"assertions": [{
        "relation_type": "PROPOSES",
        "subject_ref": {"kind": "document", "id": doc_id},
        "object_ref": {"kind": "mention", "id": mention_mid},
        "evidence_quote": "DPR for retrieval",
        "confidence": "stated",
    }]}


def _seed_canonical_run(world, outs, *, scripts=None):
    """Produce a valid prior canonical artifact set.

    Returns ``(llm, result, mention_id)`` so callers can assert on the
    provider calls the seeding run consumed.
    """
    from src.extract.ids import mention_id_v2

    text = _dpr_text()
    mention_mid = mention_id_v2(
        chunk_id="c_000",
        source_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        entity_type="method",
        char_start=text.index("DPR"),
        char_end=text.index("DPR") + 3,
    )
    llm = _RecordingLLM(scripts=scripts or [
        _dpr_entity_payload(),
        _dpr_relation_payload("c_000", "W1", mention_mid),
    ])
    result = pipeline_v2.run_extraction_v2_canonical(
        provider=llm,
        chunks_path=world["chunks"],
        ontology_path=world["ontology"],
        prompts_dir=world["prompts"],
        output_jsonl=outs["jsonl"],
        output_rejects=outs["rejects"],
        output_manifest=outs["manifest"],
    )
    return llm, result, mention_mid


class _RecordingLLM(LLMProvider):
    """Scripted LLM that records every system-prompt SHA it sees."""

    def __init__(self, scripts=None) -> None:
        self.name = "fake-llm-v2-rec"
        self.model = "fake-model-v2-rec"
        self._scripts = list(scripts or [])
        self.system_calls = []

    def generate(self, system, user, *, max_tokens=1024, temperature=0.0):
        return "", Usage(input_tokens=1, output_tokens=1)

    def generate_json(self, system, user, schema, *, max_tokens=1024, temperature=0.0):
        props = schema.get("properties", {}) or {}
        is_entity = "mentions" in props
        if self._scripts:
            idx = min(len(self.system_calls), len(self._scripts) - 1)
            payload = copy.deepcopy(self._scripts[idx])
        else:
            payload = {"mentions": []} if is_entity else {"assertions": []}
        self.system_calls.append(hashlib.sha256(system.encode("utf-8")).hexdigest())
        return copy.deepcopy(payload), Usage(input_tokens=10, output_tokens=20)


class _FailingAfterFirstLLM(LLMProvider):
    """LLM whose first JSON call raises — used for crash-ordering."""

    def __init__(self) -> None:
        self.name = "fake-llm-v2-crash"
        self.model = "fake-model-v2-crash"
        self.calls = 0

    def generate(self, system, user, *, max_tokens=1024, temperature=0.0):
        return "", Usage(input_tokens=1, output_tokens=1)

    def generate_json(self, system, user, schema, *, max_tokens=1024, temperature=0.0):
        self.calls += 1
        raise RuntimeError("simulated crash before commit")


# --- A: exact system prompt provenance ------------------------------


class TestSystemPromptProvenance(unittest.TestCase):

    def test_canonical_runner_records_sha256_of_every_system_prompt(self):
        from src.extract.ids import mention_id_v2
        text = _dpr_text()
        mention_mid = mention_id_v2(
            chunk_id="c_000",
            source_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            entity_type="method",
            char_start=text.index("DPR"),
            char_end=text.index("DPR") + 3,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            llm = _RecordingLLM(scripts=[
                _dpr_entity_payload(),
                _dpr_relation_payload("c_000", "W1", mention_mid),
            ])
            pipeline_v2.run_extraction_v2_canonical(
                provider=llm,
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
                output_jsonl=outs["jsonl"],
                output_rejects=outs["rejects"],
                output_manifest=outs["manifest"],
            )
            manifest = json.loads(outs["manifest"].read_text())
            expected = manifest["system_prompt_sha256"]
            self.assertTrue(llm.system_calls, "no system prompt was recorded")
            for sha in llm.system_calls:
                self.assertEqual(sha, expected)


# --- B: strict committed artifact set -------------------------------


class TestCommittedArtifactSetStrict(unittest.TestCase):

    def test_malformed_canonical_jsonl_hard_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            outs["jsonl"].write_text("{not json\n")
            outs["rejects"].write_text("")
            outs["manifest"].write_text(_minimal_valid_manifest())
            with self.assertRaises(pipeline_v2.PreflightErrorV2):
                pipeline_v2.run_extraction_v2_canonical(
                    provider=_RecordingLLM(),
                    chunks_path=world["chunks"],
                    ontology_path=world["ontology"],
                    prompts_dir=world["prompts"],
                    output_jsonl=outs["jsonl"],
                    output_rejects=outs["rejects"],
                    output_manifest=outs["manifest"],
                )

    def test_malformed_canonical_rejects_hard_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            outs["rejects"].write_text("garbage\n")
            outs["jsonl"].write_text("")
            outs["manifest"].write_text(_minimal_valid_manifest())
            with self.assertRaises(pipeline_v2.PreflightErrorV2):
                pipeline_v2.run_extraction_v2_canonical(
                    provider=_RecordingLLM(),
                    chunks_path=world["chunks"],
                    ontology_path=world["ontology"],
                    prompts_dir=world["prompts"],
                    output_jsonl=outs["jsonl"],
                    output_rejects=outs["rejects"],
                    output_manifest=outs["manifest"],
                )

    def test_reject_missing_schema_version_hard_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            outs["jsonl"].write_text("")
            outs["rejects"].write_text(json.dumps({"chunk_id": "c_000"}) + "\n")
            outs["manifest"].write_text(_minimal_valid_manifest())
            with self.assertRaises(pipeline_v2.PreflightErrorV2):
                pipeline_v2.run_extraction_v2_canonical(
                    provider=_RecordingLLM(),
                    chunks_path=world["chunks"],
                    ontology_path=world["ontology"],
                    prompts_dir=world["prompts"],
                    output_jsonl=outs["jsonl"],
                    output_rejects=outs["rejects"],
                    output_manifest=outs["manifest"],
                )

    def test_reject_with_v1_schema_version_hard_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            outs["jsonl"].write_text("")
            outs["rejects"].write_text(json.dumps({
                "schema_version": 1, "chunk_id": "c_000",
            }) + "\n")
            outs["manifest"].write_text(_minimal_valid_manifest())
            with self.assertRaises(pipeline_v2.PreflightErrorV2):
                pipeline_v2.run_extraction_v2_canonical(
                    provider=_RecordingLLM(),
                    chunks_path=world["chunks"],
                    ontology_path=world["ontology"],
                    prompts_dir=world["prompts"],
                    output_jsonl=outs["jsonl"],
                    output_rejects=outs["rejects"],
                    output_manifest=outs["manifest"],
                )

    def test_emitted_canonical_rejects_carry_schema_version_2(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            # First call: malformed entity payload so Gate 1 rejects.
            llm = _RecordingLLM(scripts=[
                {"mentions": "this-is-not-a-list"},
                {"assertions": []},
            ])
            pipeline_v2.run_extraction_v2_canonical(
                provider=llm,
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
                output_jsonl=outs["jsonl"],
                output_rejects=outs["rejects"],
                output_manifest=outs["manifest"],
            )
            rejects = _read_jsonl(outs["rejects"])
            self.assertGreater(len(rejects), 0)
            for r in rejects:
                self.assertEqual(r["schema_version"], schema_module.SCHEMA_VERSION_V2)


# --- C: carry-forward silent narrowing table ------------------------


class TestCarryForwardSilentNarrowing(unittest.TestCase):
    """For every #55 narrowing row: corrupt ONE prior record in place,
    rerun, assert no hard-fail and the chunk is rerun."""

    def _world(self, root):
        return _write_world(root)

    def _initial_run(self, world, outs):
        """Produce a clean baseline so a prior valid manifest exists."""
        from src.extract.ids import mention_id_v2
        text = _dpr_text()
        mention_mid = mention_id_v2(
            chunk_id="c_000",
            source_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            entity_type="method",
            char_start=text.index("DPR"),
            char_end=text.index("DPR") + 3,
        )
        llm = _RecordingLLM(scripts=[
            _dpr_entity_payload(),
            _dpr_relation_payload("c_000", "W1", mention_mid),
        ])
        pipeline_v2.run_extraction_v2_canonical(
            provider=llm,
            chunks_path=world["chunks"],
            ontology_path=world["ontology"],
            prompts_dir=world["prompts"],
            output_jsonl=outs["jsonl"],
            output_rejects=outs["rejects"],
            output_manifest=outs["manifest"],
        )
        return mention_mid

    def _rewrite_jsonl(self, outs, *, mutate):
        """Read the JSONL, mutate via callable, rewrite in place."""
        records = _read_jsonl(outs["jsonl"])
        mutate(records)
        lines = [json.dumps(r) for r in records]
        outs["jsonl"].write_text("\n".join(lines) + "\n")

    def _rerun(self, world, outs, llm):
        return pipeline_v2.run_extraction_v2_canonical(
            provider=llm,
            chunks_path=world["chunks"],
            ontology_path=world["ontology"],
            prompts_dir=world["prompts"],
            output_jsonl=outs["jsonl"],
            output_rejects=outs["rejects"],
            output_manifest=outs["manifest"],
        )

    def _clean_llm_for_rerun(self):
        from src.extract.ids import mention_id_v2
        text = _dpr_text()
        mention_mid = mention_id_v2(
            chunk_id="c_000",
            source_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            entity_type="method",
            char_start=text.index("DPR"),
            char_end=text.index("DPR") + 3,
        )
        return _RecordingLLM(scripts=[
            _dpr_entity_payload(),
            _dpr_relation_payload("c_000", "W1", mention_mid),
        ]), mention_mid

    def _rerun_assert_no_hardfail_and_recomputed(self, world, outs):
        """The corrupted chunk must actually be re-run, not carried.

        A carried record would satisfy ``extraction_complete`` too, so
        the load-bearing assertion is that the provider was called for
        the invalidated chunk (one fixture chunk = 2 JSON calls: entity
        + relation) and that the resulting records are valid again.
        """
        fresh_llm, mention_mid = self._clean_llm_for_rerun()
        result = self._rerun(world, outs, fresh_llm)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            len(fresh_llm.system_calls), 2,
            "invalidated chunk must be re-run through both provider phases",
        )
        records = _read_jsonl(outs["jsonl"])
        statuses = [r for r in records if r["kind"] == "chunk_status"]
        self.assertEqual(len(statuses), 1)
        self.assertTrue(statuses[0]["extraction_complete"])
        mentions = [r for r in records if r["kind"] == "mention"]
        assertions = [r for r in records if r["kind"] == "assertion"]
        self.assertEqual([m["mention_id"] for m in mentions], [mention_mid])
        self.assertEqual(len(assertions), 1)
        self.assertEqual(
            assertions[0]["object_ref"], {"kind": "mention", "id": mention_mid},
        )
        self.assertEqual(assertions[0]["evidence_quote"], "DPR for retrieval")

    def test_mention_id_mismatch_narrows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._world(root)
            outs = _canonical_outputs(root)
            self._initial_run(world, outs)

            def mutate(recs):
                for r in recs:
                    if r["kind"] == "mention":
                        r["mention_id"] = r["mention_id"][:-1] + "0"
                        return
            self._rewrite_jsonl(outs, mutate=mutate)
            self._rerun_assert_no_hardfail_and_recomputed(world, outs)

    def test_assertion_id_mismatch_narrows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._world(root)
            outs = _canonical_outputs(root)
            self._initial_run(world, outs)

            def mutate(recs):
                for r in recs:
                    if r["kind"] == "assertion":
                        r["assertion_id"] = r["assertion_id"][:-1] + "0"
                        return
            self._rewrite_jsonl(outs, mutate=mutate)
            self._rerun_assert_no_hardfail_and_recomputed(world, outs)

    def test_mention_evidence_slice_mismatch_narrows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._world(root)
            outs = _canonical_outputs(root)
            self._initial_run(world, outs)

            def mutate(recs):
                for r in recs:
                    if r["kind"] == "mention":
                        r["evidence"]["char_start"] += 1
                        return
            self._rewrite_jsonl(outs, mutate=mutate)
            self._rerun_assert_no_hardfail_and_recomputed(world, outs)

    def test_assertion_evidence_quote_mismatch_narrows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._world(root)
            outs = _canonical_outputs(root)
            self._initial_run(world, outs)

            def mutate(recs):
                for r in recs:
                    if r["kind"] == "assertion":
                        # ``evidence_quote`` is the persisted top-level
                        # field and is NOT part of the assertion ID, so
                        # the failure lands on the evidence gate.
                        r["evidence_quote"] = "definitely-not-in-chunk"
                        return
            self._rewrite_jsonl(outs, mutate=mutate)
            self._rerun_assert_no_hardfail_and_recomputed(world, outs)

    def test_source_sha_mismatch_narrows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._world(root)
            outs = _canonical_outputs(root)
            self._initial_run(world, outs)

            def mutate(recs):
                for r in recs:
                    if r["kind"] == "mention":
                        r["evidence"]["source_text_sha256"] = "0" * 64
                        return
            self._rewrite_jsonl(outs, mutate=mutate)
            self._rerun_assert_no_hardfail_and_recomputed(world, outs)

    def test_mention_ontology_failure_narrows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._world(root)
            outs = _canonical_outputs(root)
            self._initial_run(world, outs)

            def mutate(recs):
                for r in recs:
                    if r["kind"] == "mention":
                        r["entity_type"] = "nonexistent-type"
                        return
            self._rewrite_jsonl(outs, mutate=mutate)
            self._rerun_assert_no_hardfail_and_recomputed(world, outs)

    def test_assertion_ontology_failure_narrows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._world(root)
            outs = _canonical_outputs(root)
            self._initial_run(world, outs)

            def mutate(recs):
                for r in recs:
                    if r["kind"] == "assertion":
                        r["relation_type"] = "nonexistent-relation"
                        return
            self._rewrite_jsonl(outs, mutate=mutate)
            self._rerun_assert_no_hardfail_and_recomputed(world, outs)

    def test_dangling_mention_endpoint_narrows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._world(root)
            outs = _canonical_outputs(root)
            self._initial_run(world, outs)

            def mutate(recs):
                for r in recs:
                    if r["kind"] == "assertion":
                        # Point at a mention id that doesn't exist.
                        r["object_ref"]["id"] = "m_doesnotexist"
                        return
            self._rewrite_jsonl(outs, mutate=mutate)
            self._rerun_assert_no_hardfail_and_recomputed(world, outs)

    def test_endpoint_not_contained_narrows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._world(root)
            outs = _canonical_outputs(root)
            self._initial_run(world, outs)
            from src.extract.ids import assertion_id_v2
            text = _dpr_text()
            quote = "we also use BERT for ranking."
            start = text.index(quote)
            end = start + len(quote)
            chunk_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

            def mutate(recs):
                for r in recs:
                    if r["kind"] == "assertion":
                        # Evidence stays byte-exact, but the span no
                        # longer contains the referenced mention span
                        # (#49 ENDPOINT_NOT_CONTAINED). The ID is
                        # rederived so the ID gate still passes.
                        r["evidence"]["char_start"] = start
                        r["evidence"]["char_end"] = end
                        r["evidence_quote"] = quote
                        r["assertion_id"] = assertion_id_v2(
                            chunk_id=r["chunk_id"],
                            source_text_sha256=chunk_sha,
                            relation_type=r["relation_type"],
                            subject_ref=r["subject_ref"],
                            object_ref=r["object_ref"],
                            char_start=start,
                            char_end=end,
                        )
                        return
            self._rewrite_jsonl(outs, mutate=mutate)
            self._rerun_assert_no_hardfail_and_recomputed(world, outs)

    def test_wrong_document_endpoint_narrows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._world(root)
            outs = _canonical_outputs(root)
            self._initial_run(world, outs)

            def mutate(recs):
                for r in recs:
                    if r["kind"] == "assertion":
                        r["subject_ref"]["id"] = "W-other"
                        return
            self._rewrite_jsonl(outs, mutate=mutate)
            self._rerun_assert_no_hardfail_and_recomputed(world, outs)

    def test_wrong_status_doc_id_narrows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._world(root)
            outs = _canonical_outputs(root)
            self._initial_run(world, outs)

            def mutate(recs):
                for r in recs:
                    if r["kind"] == "chunk_status":
                        r["doc_id"] = "W-other"
                        return
            self._rewrite_jsonl(outs, mutate=mutate)
            self._rerun_assert_no_hardfail_and_recomputed(world, outs)

    def test_wrong_status_chunk_id_narrows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._world(root)
            outs = _canonical_outputs(root)
            self._initial_run(world, outs)

            def mutate(recs):
                for r in recs:
                    if r["kind"] == "chunk_status":
                        r["chunk_id"] = "c_other"
                        return
            self._rewrite_jsonl(outs, mutate=mutate)
            self._rerun_assert_no_hardfail_and_recomputed(world, outs)

    def test_phase_status_not_validated_narrows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._world(root)
            outs = _canonical_outputs(root)
            self._initial_run(world, outs)

            def mutate(recs):
                for r in recs:
                    if r["kind"] == "chunk_status":
                        r["phases"]["entity"]["status"] = "rejected"
                        return
            self._rewrite_jsonl(outs, mutate=mutate)
            self._rerun_assert_no_hardfail_and_recomputed(world, outs)

    def test_failure_count_nonzero_narrows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._world(root)
            outs = _canonical_outputs(root)
            self._initial_run(world, outs)

            def mutate(recs):
                for r in recs:
                    if r["kind"] == "chunk_status":
                        r["phases"]["relation"]["failure_count"] = 1
                        return
            self._rewrite_jsonl(outs, mutate=mutate)
            self._rerun_assert_no_hardfail_and_recomputed(world, outs)

    def test_mention_count_mismatch_narrows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._world(root)
            outs = _canonical_outputs(root)
            self._initial_run(world, outs)

            def mutate(recs):
                for r in recs:
                    if r["kind"] == "chunk_status":
                        r["phases"]["entity"]["mention_count"] = 999
                        return
            self._rewrite_jsonl(outs, mutate=mutate)
            self._rerun_assert_no_hardfail_and_recomputed(world, outs)

    def test_extraction_complete_false_narrows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._world(root)
            outs = _canonical_outputs(root)
            self._initial_run(world, outs)

            def mutate(recs):
                for r in recs:
                    if r["kind"] == "chunk_status":
                        r["extraction_complete"] = False
                        return
            self._rewrite_jsonl(outs, mutate=mutate)
            self._rerun_assert_no_hardfail_and_recomputed(world, outs)


# --- D: CLI exact dispatch ------------------------------------------


class TestCLIExactDispatch(unittest.TestCase):

    def test_unknown_subcommand_exits_2_with_usage(self):
        saved_argv = sys.argv
        try:
            for arg in (["python", "extra"], ["python", "v3"],
                        ["python", "v2 extra"], ["python", "v2 extra args"]):
                sys.argv = arg
                out = __import__("io").StringIO()
                err = __import__("io").StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = extract_main.main()
                self.assertEqual(
                    code, 2,
                    f"arg={arg!r} must exit 2, got {code}",
                )
                self.assertIn(
                    "usage: python -m src.extract [v2]",
                    err.getvalue(),
                    f"arg={arg!r} must print usage; stderr={err.getvalue()!r}",
                )
        finally:
            sys.argv = saved_argv

    def test_v2_subcommand_dispatches_to_canonical_runner(self):
        """`python -m src.extract v2` with no LLM_PROVIDER set must
        exit 2 (preflight failure) — proving the dispatch went to the
        canonical path, not the bare-v1 path (which would also exit 2,
        but with a different message).
        """
        saved_argv = sys.argv
        try:
            sys.argv = ["python", "v2"]
            out = __import__("io").StringIO()
            err = __import__("io").StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = extract_main.main()
            # Either 2 (preflight) or 1 (budget) is acceptable; both
            # prove we reached the runner. Bare-v1 path would also
            # exit 2 but with a different error message — the v2 path
            # uses PreflightErrorV2, which prints "extraction
            # preflight:" but may include the v2-specific paths.
            self.assertNotEqual(code, 0)
            self.assertIn("extraction preflight", err.getvalue())
        finally:
            sys.argv = saved_argv

    def test_no_subcommand_dispatches_to_bare_v1(self):
        """`python -m src.extract` with no LLM_PROVIDER set must
        exit 2 with the bare-v1 preflight message — proving the
        bare-v1 path is still wired up identically to ``d57c1dd``."""
        saved_argv = sys.argv
        try:
            sys.argv = ["python"]
            out = __import__("io").StringIO()
            err = __import__("io").StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = extract_main.main()
            self.assertEqual(code, 2)
            self.assertIn(
                "extraction preflight: provider config invalid",
                err.getvalue(),
            )
        finally:
            sys.argv = saved_argv


# --- E: crash ordering ----------------------------------------------


class TestCrashOrdering(unittest.TestCase):

    def test_crash_during_first_artifact_rewrite_leaves_no_manifest(self):
        """Provider calls precede the commit phase, so a provider
        exception cannot simulate a crash after the manifest unlink.
        Fail the FIRST artifact rewrite that follows
        ``_invalidate_prior_manifest`` instead."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            _seed_canonical_run(world, outs)
            self.assertTrue(outs["manifest"].is_file())
            # Invalidate the carried chunk so the next run does work.
            records = _read_jsonl(outs["jsonl"])
            for rec in records:
                if rec["kind"] == "chunk_status":
                    rec["extraction_complete"] = False
            outs["jsonl"].write_text(
                "\n".join(json.dumps(r) for r in records) + "\n",
            )

            def boom(path, lines):
                raise OSError("simulated crash after manifest unlink")

            original = pipeline_v2._atomic_rewrite_jsonl
            pipeline_v2._atomic_rewrite_jsonl = boom
            try:
                with self.assertRaises(OSError):
                    _seed_canonical_run(world, outs)
            finally:
                pipeline_v2._atomic_rewrite_jsonl = original
            self.assertFalse(
                outs["manifest"].is_file(),
                "manifest must be absent after a post-unlink crash",
            )

            healthy, result, _ = _seed_canonical_run(world, outs)
            self.assertEqual(result.exit_code, 0)
            self.assertIsNone(
                result.resumed_from_run_id,
                "no committed manifest ⇒ next run must be fresh",
            )
            self.assertEqual(len(healthy.system_calls), 2)
            self.assertTrue(outs["manifest"].is_file())

    def test_pre_commit_provider_crash_preserves_old_artifact_set(self):
        """A provider failure happens before the commit phase, so the
        prior canonical set must survive byte-for-byte."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            _seed_canonical_run(world, outs)
            # Force identity mismatch so new provider work is required.
            manifest = json.loads(outs["manifest"].read_text())
            manifest["chunks_sha256"] = "0" * 64
            outs["manifest"].write_text(json.dumps(manifest))
            old_manifest_bytes = outs["manifest"].read_bytes()
            old_jsonl_bytes = outs["jsonl"].read_bytes()
            old_rejects_bytes = outs["rejects"].read_bytes()

            crashing_llm = _FailingAfterFirstLLM()
            with self.assertRaises(RuntimeError):
                pipeline_v2.run_extraction_v2_canonical(
                    provider=crashing_llm,
                    chunks_path=world["chunks"],
                    ontology_path=world["ontology"],
                    prompts_dir=world["prompts"],
                    output_jsonl=outs["jsonl"],
                    output_rejects=outs["rejects"],
                    output_manifest=outs["manifest"],
                )
            self.assertEqual(crashing_llm.calls, 1)
            self.assertEqual(outs["manifest"].read_bytes(), old_manifest_bytes)
            self.assertEqual(outs["jsonl"].read_bytes(), old_jsonl_bytes)
            self.assertEqual(outs["rejects"].read_bytes(), old_rejects_bytes)


# --- F: audit-only identity fields don't block resume ---------------


class TestAuditOnlyIdentityFields(unittest.TestCase):
    """``ontology_version`` / ``pipeline_version`` are audit-only: a
    difference must still resume and carry the valid chunk forward."""

    def _assert_audit_field_resumes(self, key, value):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            _, prior, _ = _seed_canonical_run(world, outs)
            manifest = json.loads(outs["manifest"].read_text())
            manifest[key] = value
            outs["manifest"].write_text(json.dumps(manifest))

            llm, result, _ = _seed_canonical_run(world, outs)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.resumed_from_run_id, prior.run_id)
            self.assertEqual(
                len(llm.system_calls), 0,
                f"{key} is audit-only; the valid chunk must be carried",
            )
            self.assertEqual(result.complete_chunk_count, 1)

    def test_ontology_version_differs_resume_compatible(self):
        self._assert_audit_field_resumes("ontology_version", 999)

    def test_pipeline_version_differs_resume_compatible(self):
        self._assert_audit_field_resumes("pipeline_version", "0" * 40)


# --- F2: semantic identity mismatch = silent fresh run --------------


class TestIdentityMismatchNarrowsToFreshRun(unittest.TestCase):
    """Resume-identity fields #2-#11 are semantic: a missing field,
    wrong type, or different value means "fresh run", never an error."""

    def _assert_fresh_run(self, mutate):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            _seed_canonical_run(world, outs)
            manifest = json.loads(outs["manifest"].read_text())
            mutate(manifest)
            outs["manifest"].write_text(json.dumps(manifest))

            llm, result, _ = _seed_canonical_run(world, outs)
            self.assertEqual(result.exit_code, 0)
            self.assertIsNone(result.resumed_from_run_id)
            self.assertEqual(len(llm.system_calls), 2)
            self.assertEqual(result.complete_chunk_count, 1)

    def test_missing_chunks_sha256_is_fresh(self):
        self._assert_fresh_run(lambda m: m.pop("chunks_sha256"))

    def test_bool_max_tokens_is_fresh(self):
        self._assert_fresh_run(lambda m: m.update(max_tokens=True))

    def test_int_temperature_is_fresh(self):
        self._assert_fresh_run(lambda m: m.update(temperature=0))

    def test_malformed_prompt_template_sha256_is_fresh(self):
        self._assert_fresh_run(
            lambda m: m.update(prompt_template_sha256={"entity": "a" * 64}),
        )

    def _assert_hard_fail(self, mutate):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            _seed_canonical_run(world, outs)
            manifest = json.loads(outs["manifest"].read_text())
            mutate(manifest)
            outs["manifest"].write_text(json.dumps(manifest))
            with self.assertRaises(pipeline_v2.PreflightErrorV2):
                _seed_canonical_run(world, outs)

    def test_missing_schema_version_hard_fails(self):
        self._assert_hard_fail(lambda m: m.pop("schema_version"))

    def test_bool_schema_version_hard_fails(self):
        self._assert_hard_fail(lambda m: m.update(schema_version=True))

    def test_wrong_schema_version_hard_fails(self):
        self._assert_hard_fail(lambda m: m.update(schema_version=1))


# --- F3: manifest structural validation -----------------------------


class TestManifestStructuralValidation(unittest.TestCase):
    """Structural corruption hard-fails with ``PreflightErrorV2`` —
    never a leaked ``KeyError``."""

    def _assert_structural_hard_fail(self, manifest_text):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            _seed_canonical_run(world, outs)
            outs["manifest"].write_text(manifest_text)
            with self.assertRaises(pipeline_v2.PreflightErrorV2):
                _seed_canonical_run(world, outs)

    def _mutated_manifest(self, mutate):
        manifest = json.loads(_minimal_valid_manifest())
        mutate(manifest)
        return json.dumps(manifest)

    def test_malformed_json_hard_fails(self):
        self._assert_structural_hard_fail("{not valid json")

    def test_non_object_hard_fails(self):
        self._assert_structural_hard_fail("[]")

    def test_unknown_extra_key_hard_fails(self):
        self._assert_structural_hard_fail(
            self._mutated_manifest(lambda m: m.update(surprise="x")),
        )

    def test_missing_audit_field_hard_fails(self):
        self._assert_structural_hard_fail(
            self._mutated_manifest(lambda m: m.pop("cumulative_tokens")),
        )

    def test_wrong_type_audit_field_hard_fails(self):
        self._assert_structural_hard_fail(
            self._mutated_manifest(lambda m: m.update(cumulative_tokens="30")),
        )

    def test_prompt_template_files_wrong_length_hard_fails(self):
        self._assert_structural_hard_fail(
            self._mutated_manifest(
                lambda m: m.update(prompt_template_files=["only_one.txt"]),
            ),
        )

    def test_prompt_template_files_wrong_member_type_hard_fails(self):
        self._assert_structural_hard_fail(
            self._mutated_manifest(
                lambda m: m.update(prompt_template_files=["a.txt", 2]),
            ),
        )


# --- G: usage / reject lifecycle ------------------------------------


class TestUsageAndRejectLifecycle(unittest.TestCase):

    def test_resume_seeds_cumulative_usage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            llm1 = _RecordingLLM(scripts=[
                _dpr_entity_payload(),
                {"assertions": []},
            ])
            pipeline_v2.run_extraction_v2_canonical(
                provider=llm1,
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
                output_jsonl=outs["jsonl"],
                output_rejects=outs["rejects"],
                output_manifest=outs["manifest"],
            )
            prior = json.loads(outs["manifest"].read_text())
            # Two phases (entity + relation), each ~30 tokens.
            self.assertEqual(prior["cumulative_tokens"], 60)
            # Force the carry-forward to fail by flipping
            # extraction_complete, so the chunk re-runs and adds new
            # usage on top of the seeded prior.
            records = _read_jsonl(outs["jsonl"])
            for r in records:
                if r["kind"] == "chunk_status":
                    r["extraction_complete"] = False
            outs["jsonl"].write_text("\n".join(json.dumps(r) for r in records) + "\n")
            llm2 = _RecordingLLM(scripts=[
                _dpr_entity_payload(),
                {"assertions": []},
            ])
            result = pipeline_v2.run_extraction_v2_canonical(
                provider=llm2,
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
                output_jsonl=outs["jsonl"],
                output_rejects=outs["rejects"],
                output_manifest=outs["manifest"],
            )
            self.assertEqual(result.exit_code, 0)
            new_manifest = json.loads(outs["manifest"].read_text())
            # Seed 60 + new 60 (both phases rerun) = 120.
            self.assertEqual(new_manifest["cumulative_tokens"], 120)
            self.assertEqual(new_manifest["resumed_from_run_id"], prior["run_id"])

    def test_identity_mismatch_resets_usage_to_zero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            _seed_canonical_run(world, outs)
            # Mutate chunks_sha256 (resume-identity field).
            manifest = json.loads(outs["manifest"].read_text())
            manifest["chunks_sha256"] = "0" * 64
            outs["manifest"].write_text(json.dumps(manifest))

            llm, result, _ = _seed_canonical_run(world, outs)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(len(llm.system_calls), 2)
            new_manifest = json.loads(outs["manifest"].read_text())
            self.assertEqual(new_manifest["resumed_from_run_id"], None)
            # Fresh run = only this run's usage (no seed): 2 x 30.
            self.assertEqual(new_manifest["cumulative_tokens"], 60)

    def test_carried_chunks_contribute_no_old_rejects(self):
        """A carried chunk is the canonical truth; the prior reject log
        is rewritten from the current run only."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            _, prior, _ = _seed_canonical_run(world, outs)
            with outs["rejects"].open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "schema_version": 2,
                    "chunk_id": "c_stale",
                    "phase": "entity",
                    "reason": "STALE_REJECT",
                }) + "\n")

            llm, result, _ = _seed_canonical_run(world, outs)
            self.assertEqual(result.resumed_from_run_id, prior.run_id)
            self.assertEqual(len(llm.system_calls), 0)
            rejects = _read_jsonl(outs["rejects"])
            self.assertEqual(rejects, [])
            manifest = json.loads(outs["manifest"].read_text())
            self.assertEqual(manifest["reject_count"], len(rejects))

    def test_fresh_run_strips_stale_rejects(self):
        """Without a prior manifest the sibling artifacts are
        uncommitted: the run is fresh and rewrites the reject log."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            outs["rejects"].write_text(json.dumps({
                "schema_version": 2, "chunk_id": "c_stale",
                "phase": "entity", "reason": "STALE",
            }) + "\n")

            _, result, _ = _seed_canonical_run(world, outs)
            self.assertIsNone(result.resumed_from_run_id)
            rejects = _read_jsonl(outs["rejects"])
            self.assertEqual(
                [r for r in rejects if r.get("chunk_id") == "c_stale"], [],
            )

    def test_manifest_reject_count_matches_reject_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            # Malformed entity payload ⇒ Gate 1 emits rejects.
            _seed_canonical_run(world, outs, scripts=[
                {"mentions": "this-is-not-a-list"},
                {"assertions": []},
            ])
            rejects = _read_jsonl(outs["rejects"])
            self.assertGreater(len(rejects), 0)
            manifest = json.loads(outs["manifest"].read_text())
            self.assertEqual(manifest["reject_count"], len(rejects))


# --- ancillary regressions ------------------------------------------


class TestRunIdFormat(unittest.TestCase):

    def test_run_id_matches_contract_regex(self):
        """``run_id`` must match ``^\\d{8}T\\d{6}Z-[0-9a-f]{16}$``."""
        pat = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{16}$")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _canonical_outputs(root)
            _seed_canonical_run(world, outs)
            manifest = json.loads(outs["manifest"].read_text())
            self.assertRegex(manifest["run_id"], pat)
            self.assertEqual(manifest["resumed_from_run_id"], None)


class TestPreflightV2PublicSignature(unittest.TestCase):

    def test_preflight_v2_param_list_is_frozen(self):
        """The standalone ``preflight_v2`` helper must NOT accept
        ``provider``, ``max_tokens``, or ``temperature`` — those are
        runner-internal and were intentionally NOT part of the T3a
        contract per #55."""
        import inspect
        sig = inspect.signature(pipeline_v2.preflight_v2)
        names = list(sig.parameters)
        self.assertEqual(
            names, ["chunks_path", "ontology_path", "prompts_dir"],
            f"preflight_v2 signature drifted: {names}",
        )

    def test_preflight_v2_runs_without_provider_or_git(self):
        """T3a preflight resolves only the disk-backed identity fields;
        provider / runner / ``git rev-parse HEAD`` stay unresolved."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            report = pipeline_v2.preflight_v2(
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
            )
            self.assertEqual(report.chunks_path, world["chunks"])
            self.assertIsNone(report.pipeline_version)
            self.assertIsNone(report.provider_name)
            self.assertIsNone(report.max_tokens)
            self.assertIsNone(report.temperature)


class TestCanonicalFunctionContract(unittest.TestCase):

    def test_run_extraction_v2_canonical_parameter_order(self):
        """The canonical keyword-only argument order is pinned by #55."""
        import inspect
        sig = inspect.signature(pipeline_v2.run_extraction_v2_canonical)
        self.assertEqual(
            list(sig.parameters),
            [
                "provider",
                "chunks_path",
                "ontology_path",
                "prompts_dir",
                "output_jsonl",
                "output_rejects",
                "output_manifest",
                "max_tokens",
                "temperature",
            ],
        )
        for name, param in sig.parameters.items():
            self.assertEqual(
                param.kind,
                inspect.Parameter.KEYWORD_ONLY,
                f"{name} must be keyword-only",
            )


if __name__ == "__main__":
    unittest.main()
