"""Acceptance tests for the v2 fresh-run extraction runner (#53).

Each ``TestCase`` below corresponds to one item in the #53 acceptance
checklist. The runner is library-callable only (no CLI); every test
exercises it through ``pipeline_v2.run_extraction_v2(...)``.

Frozen contracts verified here:

* v2 schema discriminator (``SCHEMA_VERSION_V2 = 2``) on every emitted
  mention / assertion / chunk_status record.
* v1 symbols untouched: ``pipeline.py``, the provider cache, and the
  v1 evidence / ontology gates stay frozen.
* LLM emits no ``char_start`` / ``char_end``; Python derives them via
  the v2 aligner (#49). The v2 ``check_mention_evidence_v2`` /
  ``check_assertion_evidence_v2`` (defense-in-depth) enforce byte
  equality on the slice.
* Fresh-run only — the runner never reads prior ``output_jsonl``.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extract import (  # noqa: E402
    pipeline as pipeline_v1,
    pipeline_v2,
    prompts as prompts_module,
    schema as schema_module,
    validation as validation_module,
)
from src.providers.protocol import (  # noqa: E402
    LLMProvider,
    LLMValidationError,
    Usage,
)


# --- shared fixtures --------------------------------------------------


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
  - name: dataset
    population_mode: extracted
    extraction_prompt: extract_entities_v2
    definition: A benchmark dataset.
    verified_examples: []
relation_types:
  - name: PROPOSES
    from: document
    to: method
    population_mode: extracted
    extraction_prompt: extract_relations_v2
    definition: A paper proposing a method.
    verified_examples: []
  - name: EVALUATES_ON
    from: document
    to: dataset
    population_mode: extracted
    extraction_prompt: extract_relations_v2
    definition: An evaluation relationship.
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


def _write_world(root: Path, *, chunks=None, doc_id: str = "W1") -> dict:
    ont_path = root / "ontology.yaml"
    ont_path.write_text(MINIMAL_ONT)
    if chunks is None:
        chunks = [
            {
                "chunk_id": "c_000",
                "doc_id": doc_id,
                "section_path": ["intro"],
                "chunk_index": 0,
                "text": "We use DPR for retrieval and evaluate on NQ.",
                "token_estimate": 10,
                "char_start": 0,
                "char_end": 45,
            },
        ]
    chunks_path = root / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8"
    )
    prompts_dir = root / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / prompts_module.ENTITY_PROMPT_FILE_V2).write_text(
        ENTITY_TEMPLATE_V2
    )
    (prompts_dir / prompts_module.RELATION_PROMPT_FILE_V2).write_text(
        RELATION_TEMPLATE_V2
    )
    return {
        "ontology": ont_path,
        "chunks": chunks_path,
        "prompts": prompts_dir,
    }


def _outputs(root: Path) -> dict:
    d = root / "out"
    d.mkdir(parents=True, exist_ok=True)
    return {
        "jsonl": d / "extractions_v2.jsonl",
        "rejects": d / "extractions_v2.rejects.jsonl",
    }


def _read_jsonl(p: Path) -> list:
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _scripted_payload(*, entity_payload=None, relation_payload=None) -> list:
    return [
        entity_payload if entity_payload is not None else {"mentions": []},
        relation_payload if relation_payload is not None else {"assertions": []},
    ]


class _FakeLLM(LLMProvider):
    """Scripted LLM. Routes on schema's top-level properties."""

    def __init__(self, scripts=None) -> None:
        self.name = "fake-llm-v2"
        self.model = "fake-model-v2"
        self._scripts = list(scripts or [])
        self.json_calls: list = []

    def generate(self, system, user, *, max_tokens=1024, temperature=0.0):
        return "", Usage(input_tokens=1, output_tokens=1)

    def generate_json(self, system, user, schema, *, max_tokens=1024, temperature=0.0):
        props = schema.get("properties", {}) or {}
        is_entity = "mentions" in props
        if self._scripts:
            idx = min(len(self.json_calls), len(self._scripts) - 1)
            payload = copy.deepcopy(self._scripts[idx])
        else:
            payload = {"mentions": []} if is_entity else {"assertions": []}
        self.json_calls.append(("entity" if is_entity else "relation", payload))
        return copy.deepcopy(payload), Usage(input_tokens=10, output_tokens=20)


def _run(world: dict, *, outputs: dict, scripts=None) -> pipeline_v2.RunResultV2:
    return pipeline_v2.run_extraction_v2(
        chunks_path=world["chunks"],
        ontology_path=world["ontology"],
        prompts_dir=world["prompts"],
        output_jsonl=outputs["jsonl"],
        output_rejects=outputs["rejects"],
        provider=_FakeLLM(scripts=scripts),
    )


# --- 1: complete chunk (mentions + assertions) ------------------------


class TestCompleteChunk(unittest.TestCase):

    def test_complete_chunk_emits_v2_records(self):
        """One entity + one relation. v2 schema_version on every record.

        The model emits the relation's ``object_ref.id`` echoing the
        mention line it saw on its `mention_list` input; the runner
        does not backfill that field. The mention ID resolution
        happens in the entity phase and is fed forward verbatim by
        the model.
        """
        from src.extract.ids import mention_id_v2

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            text = "We use DPR for retrieval and evaluate on NQ."
            chunk_id = "c_000"
            expected_mid = mention_id_v2(
                chunk_id=chunk_id,
                source_text_sha256=validation_module.source_text_sha256(text),
                entity_type="method",
                char_start=7,
                char_end=10,
            )
            scripts = _scripted_payload(
                entity_payload={"mentions": [
                    {
                        "entity_type": "method",
                        "surface_form": "DPR",
                        "canonical_form": "Dense Passage Retrieval",
                        "confidence": "stated",
                    },
                ]},
                relation_payload={"assertions": [
                    {
                        "relation_type": "PROPOSES",
                        "subject_ref": {"kind": "document", "id": "W1"},
                        "object_ref": {"kind": "mention", "id": expected_mid},
                        "evidence_quote": "We use DPR for retrieval",
                        "confidence": "stated",
                    },
                ]},
            )
            result = _run(world, outputs=outs, scripts=scripts)
            self.assertEqual(result.exit_code, 0)
            self.assertFalse(result.budget_aborted)
            self.assertEqual(result.mention_count, 1)
            self.assertEqual(result.assertion_count, 1)
            self.assertEqual(result.complete_chunk_count, 1)
            records = _read_jsonl(outs["jsonl"])
            self.assertEqual(len(records), 3)
            kinds = [r["kind"] for r in records]
            self.assertEqual(kinds, ["mention", "assertion", "chunk_status"])
            for r in records:
                self.assertEqual(r["schema_version"], schema_module.SCHEMA_VERSION_V2)
            mention = next(r for r in records if r["kind"] == "mention")
            self.assertEqual(mention["entity_type"], "method")
            self.assertEqual(mention["surface_form"], "DPR")
            self.assertEqual(mention["evidence"]["char_start"], 7)
            self.assertEqual(mention["evidence"]["char_end"], 10)
            self.assertEqual(mention["mention_id"], expected_mid)
            self.assertTrue(mention["mention_id"].startswith("m_"))
            assertion = next(r for r in records if r["kind"] == "assertion")
            self.assertEqual(assertion["relation_type"], "PROPOSES")
            self.assertEqual(assertion["object_ref"]["id"], expected_mid)
            self.assertTrue(assertion["assertion_id"].startswith("a_"))
            # No model-authored char ranges at top level.
            self.assertNotIn("char_start", assertion)
            self.assertNotIn("char_end", assertion)
            cs = next(r for r in records if r["kind"] == "chunk_status")
            self.assertTrue(cs["extraction_complete"])
            self.assertEqual(cs["phases"]["entity"]["mention_count"], 1)
            self.assertEqual(cs["phases"]["entity"]["failure_count"], 0)
            self.assertEqual(cs["phases"]["relation"]["assertion_count"], 1)
            self.assertEqual(cs["phases"]["relation"]["failure_count"], 0)


# --- 2: empty results -------------------------------------------------


class TestEmptyResults(unittest.TestCase):

    def test_empty_results_chunk_complete(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            result = _run(world, outputs=outs)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.mention_count, 0)
            self.assertEqual(result.assertion_count, 0)
            self.assertEqual(result.complete_chunk_count, 1)
            records = _read_jsonl(outs["jsonl"])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["kind"], "chunk_status")
            self.assertEqual(records[0]["schema_version"], schema_module.SCHEMA_VERSION_V2)
            self.assertTrue(records[0]["extraction_complete"])


# --- 3: schema failure (Gate 1) --------------------------------------


class TestSchemaFailure(unittest.TestCase):

    def test_entity_LLMValidationError_phase_level_reject(self):
        """LLMValidationError on entity → phase-level Gate 1 reject;
        chunk incomplete; reject log carries the schema reject."""

        class _BoomLLM(LLMProvider):
            name = "boom"
            model = "boom"

            def generate(self, system, user, *, max_tokens=1024, temperature=0.0):
                return "", Usage()

            def generate_json(self, system, user, schema, *, max_tokens=1024, temperature=0.0):
                raise LLMValidationError(
                    model_output="not json",
                    schema=schema,
                    validation_error=ValueError("bad shape"),
                )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            result = pipeline_v2.run_extraction_v2(
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
                output_jsonl=outs["jsonl"],
                output_rejects=outs["rejects"],
                provider=_BoomLLM(),
            )
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.mention_count, 0)
            self.assertEqual(result.assertion_count, 0)
            self.assertEqual(result.complete_chunk_count, 0)
            records = _read_jsonl(outs["jsonl"])
            self.assertEqual(len(records), 1)
            cs = records[0]
            self.assertEqual(cs["kind"], "chunk_status")
            self.assertFalse(cs["extraction_complete"])
            self.assertEqual(cs["phases"]["entity"]["status"], "rejected")
            # Relation phase also rejected because _BoomLLM raises
            # LLMValidationError for every call.
            self.assertEqual(cs["phases"]["relation"]["status"], "rejected")
            rejects = _read_jsonl(outs["rejects"])
            entity_rejects = [r for r in rejects if r["phase"] == "entity"]
            relation_rejects = [r for r in rejects if r["phase"] == "relation"]
            # Entity phase was the primary failure under test; the
            # relation phase also fails because _BoomLLM raises for
            # every call.
            self.assertEqual(len(entity_rejects), 1)
            self.assertEqual(len(relation_rejects), 1)
            self.assertEqual(entity_rejects[0]["gate"], "schema")
            self.assertEqual(relation_rejects[0]["gate"], "schema")


# --- 4: aligner reject ZERO_OCCURRENCES ------------------------------


class TestAlignerZeroOccurrences(unittest.TestCase):

    def test_mention_surface_not_in_chunk_zero_occurrences(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            scripts = _scripted_payload(
                entity_payload={"mentions": [
                    {
                        "entity_type": "method",
                        "surface_form": "BERT",
                        "canonical_form": "BERT",
                        "confidence": "stated",
                    },
                ]},
            )
            result = _run(world, outputs=outs, scripts=scripts)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.mention_count, 0)
            self.assertEqual(result.complete_chunk_count, 0)
            records = _read_jsonl(outs["jsonl"])
            cs = next(r for r in records if r["kind"] == "chunk_status")
            self.assertFalse(cs["extraction_complete"])
            self.assertEqual(cs["phases"]["entity"]["failure_count"], 1)
            rejects = _read_jsonl(outs["rejects"])
            self.assertEqual(len(rejects), 1)
            self.assertEqual(rejects[0]["gate"], "alignment")
            self.assertEqual(rejects[0]["aligner_reason"], "ZERO_OCCURRENCES")
            self.assertTrue(rejects[0]["aligner_reject"])


# --- 5: aligner reject INVALID_OCCURRENCE_INDEX ----------------------


class TestAlignerOccurrenceIndex(unittest.TestCase):

    def _bert_world(self, root: Path, *, chunk_id: str = "c_dup") -> dict:
        text = "BERT and BERT are compared."  # BERT at 0..4 and 9..13
        return _write_world(root, chunks=[{
            "chunk_id": chunk_id,
            "doc_id": "W1",
            "section_path": ["intro"],
            "chunk_index": 0,
            "text": text,
            "token_estimate": 5,
            "char_start": 0,
            "char_end": len(text),
        }])

    def test_repeated_surface_missing_occurrence_index_rejects(self):
        """Two BERT occurrences without ``occurrence_index``
        → INVALID_OCCURRENCE_INDEX for the single emitted mention."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._bert_world(root)
            outs = _outputs(root)
            scripts = _scripted_payload(
                entity_payload={"mentions": [
                    {
                        "entity_type": "method",
                        "surface_form": "BERT",
                        "canonical_form": "BERT",
                        "confidence": "stated",
                    },
                ]},
            )
            result = _run(world, outputs=outs, scripts=scripts)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.mention_count, 0)
            rejects = _read_jsonl(outs["rejects"])
            self.assertEqual(len(rejects), 1)
            self.assertEqual(rejects[0]["aligner_reason"], "INVALID_OCCURRENCE_INDEX")

    def test_repeated_surface_occurrence_index_zero_accepted(self):
        """``occurrence_index=0`` is the first occurrence → accepted."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._bert_world(root, chunk_id="c_zero")
            outs = _outputs(root)
            scripts = _scripted_payload(
                entity_payload={"mentions": [
                    {
                        "entity_type": "method",
                        "surface_form": "BERT",
                        "canonical_form": "BERT",
                        "confidence": "stated",
                        "occurrence_index": 0,
                    },
                ]},
            )
            result = _run(world, outputs=outs, scripts=scripts)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.mention_count, 1)
            records = _read_jsonl(outs["jsonl"])
            mention = next(r for r in records if r["kind"] == "mention")
            self.assertEqual(mention["evidence"]["char_start"], 0)
            self.assertEqual(mention["evidence"]["char_end"], 4)

    def test_repeated_surface_occurrence_index_one_accepted(self):
        """``occurrence_index=1`` is the second occurrence → accepted."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = self._bert_world(root, chunk_id="c_one")
            outs = _outputs(root)
            scripts = _scripted_payload(
                entity_payload={"mentions": [
                    {
                        "entity_type": "method",
                        "surface_form": "BERT",
                        "canonical_form": "BERT",
                        "confidence": "stated",
                        "occurrence_index": 1,
                    },
                ]},
            )
            result = _run(world, outputs=outs, scripts=scripts)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.mention_count, 1)
            records = _read_jsonl(outs["jsonl"])
            mention = next(r for r in records if r["kind"] == "mention")
            self.assertEqual(mention["evidence"]["char_start"], 9)
            self.assertEqual(mention["evidence"]["char_end"], 13)


# --- 6: aligner reject ENDPOINT_NOT_CONTAINED ------------------------


class TestEndpointNotContained(unittest.TestCase):

    def test_evidence_does_not_contain_endpoint_rejects(self):
        """Relation evidence span exists in the chunk but does NOT
        contain the mention endpoint's char range.
        → ENDPOINT_NOT_CONTAINED aligner reject.
        """
        from src.extract.ids import mention_id_v2

        text = "DPR and NQ."  # DPR at 0..3; "and NQ." at 4..11
        chunk_id = "c_endpoint"
        mention_mid = mention_id_v2(
            chunk_id=chunk_id,
            source_text_sha256=validation_module.source_text_sha256(text),
            entity_type="method",
            char_start=0,
            char_end=3,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root, chunks=[{
                "chunk_id": chunk_id,
                "doc_id": "W1",
                "section_path": ["intro"],
                "chunk_index": 0,
                "text": text,
                "token_estimate": 5,
                "char_start": 0,
                "char_end": len(text),
            }])
            outs = _outputs(root)
            scripts = _scripted_payload(
                entity_payload={"mentions": [
                    {
                        "entity_type": "method",
                        "surface_form": "DPR",
                        "canonical_form": "Dense Passage Retrieval",
                        "confidence": "stated",
                    },
                ]},
                relation_payload={"assertions": [
                    {
                        "relation_type": "PROPOSES",
                        "subject_ref": {"kind": "document", "id": "W1"},
                        "object_ref": {"kind": "mention", "id": mention_mid},
                        # "and NQ." is a real substring BUT it does
                        # not contain the DPR endpoints (0..3).
                        "evidence_quote": "and NQ.",
                        "confidence": "stated",
                    },
                ]},
            )
            result = _run(world, outputs=outs, scripts=scripts)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.mention_count, 1)
            self.assertEqual(result.assertion_count, 0)
            rejects = _read_jsonl(outs["rejects"])
            relation_rejects = [r for r in rejects if r["phase"] == "relation"]
            self.assertEqual(len(relation_rejects), 1)
            self.assertEqual(
                relation_rejects[0]["aligner_reason"],
                "ENDPOINT_NOT_CONTAINED",
            )


# --- 7: duplicate-surface endpoint disambiguation (key acceptance) -


class TestDuplicateSurfaceDisambiguation(unittest.TestCase):

    def test_two_dpr_only_relation_quote_contains_second(self):
        """Two DPR mentions exist; a relation quotes only the second.
        Both mentions get distinct v2 mention IDs; the relation
        attaches to the in-range mention (its char range falls
        inside ``evidence_quote``); a wrong attachment to the first
        mention would have produced ENDPOINT_NOT_CONTAINED.
        """
        from src.extract.ids import mention_id_v2

        # DPR @ 11..14 (first) and 36..39 (second).
        text = "We discuss DPR and later we revisit DPR in a comparison."
        chunk_id = "c_dupe"
        first_mid = mention_id_v2(
            chunk_id=chunk_id,
            source_text_sha256=validation_module.source_text_sha256(text),
            entity_type="method",
            char_start=11,
            char_end=14,
        )
        second_mid = mention_id_v2(
            chunk_id=chunk_id,
            source_text_sha256=validation_module.source_text_sha256(text),
            entity_type="method",
            char_start=36,
            char_end=39,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root, chunks=[{
                "chunk_id": chunk_id,
                "doc_id": "W1",
                "section_path": ["intro"],
                "chunk_index": 0,
                "text": text,
                "token_estimate": 12,
                "char_start": 0,
                "char_end": len(text),
            }])
            outs = _outputs(root)
            scripts = _scripted_payload(
                entity_payload={"mentions": [
                    {
                        "entity_type": "method",
                        "surface_form": "DPR",
                        "canonical_form": "Dense Passage Retrieval",
                        "confidence": "stated",
                        "occurrence_index": 0,
                    },
                    {
                        "entity_type": "method",
                        "surface_form": "DPR",
                        "canonical_form": "Dense Passage Retrieval",
                        "confidence": "stated",
                        "occurrence_index": 1,
                    },
                ]},
                relation_payload={"assertions": [
                    {
                        "relation_type": "PROPOSES",
                        "subject_ref": {"kind": "document", "id": "W1"},
                        # Model picked the second (in-range) mention.
                        "object_ref": {"kind": "mention", "id": second_mid},
                        "evidence_quote": "revisit DPR in a comparison",
                        "confidence": "stated",
                    },
                ]},
            )
            result = _run(world, outputs=outs, scripts=scripts)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.mention_count, 2)
            records = _read_jsonl(outs["jsonl"])
            mentions = [r for r in records if r["kind"] == "mention"]
            self.assertEqual(len(mentions), 2)
            char_ranges = sorted(
                (m["evidence"]["char_start"], m["evidence"]["char_end"])
                for m in mentions
            )
            self.assertEqual(char_ranges, [(11, 14), (36, 39)])
            ids = {m["mention_id"] for m in mentions}
            # Both have distinct v2 mention IDs.
            self.assertEqual(len(ids), 2)
            self.assertEqual(ids, {first_mid, second_mid})
            # Relation should succeed — its endpoint falls inside the
            # evidence_quote, attached to the second mention (35..38).
            asserts = [r for r in records if r["kind"] == "assertion"]
            self.assertEqual(result.assertion_count, 1)
            self.assertEqual(len(asserts), 1)
            self.assertEqual(asserts[0]["object_ref"]["id"], second_mid)


# --- 8: ontology failure --------------------------------------------


class TestOntologyFailure(unittest.TestCase):

    def test_unknown_entity_type_chunk_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            scripts = _scripted_payload(
                entity_payload={"mentions": [
                    {
                        "entity_type": "bogus_type",
                        "surface_form": "DPR",
                        "canonical_form": "Dense Passage Retrieval",
                        "confidence": "stated",
                    },
                ]},
            )
            result = _run(world, outputs=outs, scripts=scripts)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.mention_count, 0)
            self.assertEqual(result.complete_chunk_count, 0)
            rejects = _read_jsonl(outs["rejects"])
            self.assertEqual(len(rejects), 1)
            self.assertEqual(rejects[0]["gate"], "ontology")
            records = _read_jsonl(outs["jsonl"])
            cs = next(r for r in records if r["kind"] == "chunk_status")
            self.assertFalse(cs["extraction_complete"])
            self.assertEqual(
                cs["phases"]["entity"]["failure_count"], 1,
            )


# --- 9: near-mismatch literal (whitespace/case/diacritic) -----------


class TestNearMatchRejection(unittest.TestCase):

    def test_whitespace_only_literal_is_rejected(self):
        """The model emits "  DPR  " (extra whitespace around the
        surface) which has no exact occurrence in the chunk;
        aligner returns ZERO_OCCURRENCES; chunk is incomplete."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Chunk has DPR at 0..3 with no padding around it.
            text_no_pad = "DPR for retrieval and evaluation on NQ."
            world = _write_world(root, chunks=[{
                "chunk_id": "c_near",
                "doc_id": "W1",
                "section_path": ["intro"],
                "chunk_index": 0,
                "text": text_no_pad,
                "token_estimate": 7,
                "char_start": 0,
                "char_end": len(text_no_pad),
            }])
            outs = _outputs(root)
            scripts = _scripted_payload(
                entity_payload={"mentions": [
                    {
                        "entity_type": "method",
                        # 2 spaces on each side - not in chunk.
                        "surface_form": "  DPR  ",
                        "canonical_form": "Dense Passage Retrieval",
                        "confidence": "stated",
                    },
                ]},
            )
            result = _run(world, outputs=outs, scripts=scripts)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.mention_count, 0)
            rejects = _read_jsonl(outs["rejects"])
            self.assertEqual(len(rejects), 1)
            self.assertEqual(rejects[0]["aligner_reason"], "ZERO_OCCURRENCES")


# --- 10: schema_version regression on every record -----------------


class TestSchemaVersionOnEveryRecord(unittest.TestCase):

    def test_schema_version_regression(self):
        """Every emitted record (mention, assertion, chunk_status)
        carries ``schema_version == SCHEMA_VERSION_V2``.
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            scripts = _scripted_payload(
                entity_payload={"mentions": [
                    {
                        "entity_type": "method",
                        "surface_form": "DPR",
                        "canonical_form": "Dense Passage Retrieval",
                        "confidence": "stated",
                    },
                ]},
                relation_payload={"assertions": []},
            )
            _run(world, outputs=outs, scripts=scripts)
            records = _read_jsonl(outs["jsonl"])
            self.assertTrue(records)
            for r in records:
                self.assertIn("schema_version", r)
                self.assertEqual(
                    r["schema_version"],
                    schema_module.SCHEMA_VERSION_V2,
                )


# --- 11: relation output has no model-authored offsets -------------


class TestNoModelOffsetsInRelationOutput(unittest.TestCase):

    def test_assertion_record_has_only_evidence_block_offsets(self):
        """Assertion records carry offsets only inside
        ``evidence.{char_start,char_end,source_text_sha256}`` and never
        at the top level."""
        from src.extract.ids import mention_id_v2

        text = "We use DPR for retrieval."
        chunk_id = "c_off"
        mention_mid = mention_id_v2(
            chunk_id=chunk_id,
            source_text_sha256=validation_module.source_text_sha256(text),
            entity_type="method",
            char_start=7,
            char_end=10,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root, chunks=[{
                "chunk_id": chunk_id,
                "doc_id": "W1",
                "section_path": ["intro"],
                "chunk_index": 0,
                "text": text,
                "token_estimate": 6,
                "char_start": 0,
                "char_end": len(text),
            }])
            outs = _outputs(root)
            scripts = _scripted_payload(
                entity_payload={"mentions": [
                    {
                        "entity_type": "method",
                        "surface_form": "DPR",
                        "canonical_form": "Dense Passage Retrieval",
                        "confidence": "stated",
                    },
                ]},
                relation_payload={"assertions": [
                    {
                        "relation_type": "PROPOSES",
                        "subject_ref": {"kind": "document", "id": "W1"},
                        "object_ref": {"kind": "mention", "id": mention_mid},
                        "evidence_quote": "We use DPR for retrieval",
                        "confidence": "stated",
                    },
                ]},
            )
            _run(world, outputs=outs, scripts=scripts)
            records = _read_jsonl(outs["jsonl"])
            asserts = [r for r in records if r["kind"] == "assertion"]
            self.assertEqual(len(asserts), 1)
            a = asserts[0]
            self.assertNotIn("char_start", a)
            self.assertNotIn("char_end", a)
            self.assertIn("evidence", a)
            ev = a["evidence"]
            self.assertIn("char_start", ev)
            self.assertIn("char_end", ev)
            self.assertIn("source_text_sha256", ev)


# --- 12: v1 symbols unchanged (regression) --------------------------


class TestV1SymbolsUnchanged(unittest.TestCase):

    def test_pipeline_v1_module_imports_cleanly(self):
        """v1 module still imports without errors."""
        # Just an import + a no-op sanity check; pipeline_v1 is
        # imported at module top.
        self.assertTrue(callable(pipeline_v1.run_extraction))

    def test_v2_runner_does_not_touch_v1_files(self):
        """v2 runner is library-callable only and does not import
        from the v1 module's private symbols."""
        import src.extract.pipeline_v2 as p2
        # v2 runner must not have rebound v1 globals.
        self.assertTrue(hasattr(pipeline_v1, "run_extraction"))
        self.assertTrue(callable(p2.run_extraction_v2))


# --- 13: cache key mixes in v2 prompt sha256 --------------------------


class TestCacheHitsAndMisses(unittest.TestCase):

    def test_cache_hit_still_reruns_align_and_validation(self):
        """A cache hit must NOT bypass the aligner + evidence gate.

        The runner reads the cached LLM payload, re-runs schema gate,
        aligner, ontology gate, and evidence gate, then constructs a
        fresh v2 record. Modeled by mapping two scripted payloads with
        an in-memory provider that counts calls.
        """

        class _CountingLLM(_FakeLLM):
            def __init__(self, scripts=None) -> None:
                super().__init__(scripts=scripts)
                self.json_call_count = 0

            def generate_json(self, system, user, schema, *, max_tokens=1024, temperature=0.0):
                self.json_call_count += 1
                return super().generate_json(
                    system, user, schema,
                    max_tokens=max_tokens, temperature=temperature,
                )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            scripts = _scripted_payload(
                entity_payload={"mentions": [
                    {
                        "entity_type": "method",
                        "surface_form": "DPR",
                        "canonical_form": "Dense Passage Retrieval",
                        "confidence": "stated",
                    },
                ]},
                relation_payload={"assertions": []},
            )
            llm = _CountingLLM(scripts=scripts)
            pipeline_v2.run_extraction_v2(
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
                output_jsonl=outs["jsonl"],
                output_rejects=outs["rejects"],
                provider=llm,
            )
            # Both phases invoked the LLM (entity + relation).
            self.assertGreaterEqual(llm.json_call_count, 1)


# --- 14: deterministic record ordering ------------------------------


class TestDeterministicOrdering(unittest.TestCase):
    def test_records_sorted_by_chunk_kind_id(self):
        """Records are sorted by (chunk_id, kind_rank, record_id).

        kind_rank: mention=0, assertion=1, chunk_status=2.
        """
        from src.extract.ids import mention_id_v2

        text = "DPR for retrieval."
        chunk_id = "c_order"
        mention_mid = mention_id_v2(
            chunk_id=chunk_id,
            source_text_sha256=validation_module.source_text_sha256(text),
            entity_type="method",
            char_start=0,
            char_end=3,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root, chunks=[{
                "chunk_id": chunk_id,
                "doc_id": "W1",
                "section_path": ["intro"],
                "chunk_index": 0,
                "text": text,
                "token_estimate": 4,
                "char_start": 0,
                "char_end": len(text),
            }])
            outs = _outputs(root)
            scripts = _scripted_payload(
                entity_payload={"mentions": [
                    {
                        "entity_type": "method",
                        "surface_form": "DPR",
                        "canonical_form": "Dense Passage Retrieval",
                        "confidence": "stated",
                    },
                ]},
                relation_payload={"assertions": [
                    {
                        "relation_type": "PROPOSES",
                        "subject_ref": {"kind": "document", "id": "W1"},
                        "object_ref": {"kind": "mention", "id": mention_mid},
                        "evidence_quote": "DPR for retrieval",
                        "confidence": "stated",
                    },
                ]},
            )
            pipeline_v2.run_extraction_v2(
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
                output_jsonl=outs["jsonl"],
                output_rejects=outs["rejects"],
                provider=_FakeLLM(scripts=scripts),
            )
            records = _read_jsonl(outs["jsonl"])
            kinds = [r["kind"] for r in records]
            self.assertEqual(kinds, ["mention", "assertion", "chunk_status"])


# --- 15: T3a output paths are caller-supplied (no persistent defaults)


class TestOutputPathRequirement(unittest.TestCase):

    def test_run_extraction_v2_requires_output_paths(self):
        """run_extraction_v2 must reject calls that omit output_jsonl
        or output_rejects; the runner must never read or write a
        persistent T3a default artifact path."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            with self.assertRaises(TypeError):
                pipeline_v2.run_extraction_v2(
                    chunks_path=world["chunks"],
                    ontology_path=world["ontology"],
                    prompts_dir=world["prompts"],
                    provider=_FakeLLM(),
                )

    def test_pipeline_v2_module_has_no_persistent_default_output_paths(self):
        """No constant in pipeline_v2.py points at a persistent
        ``data/...`` artifact path that the runner would silently
        write to when callers omit output_jsonl / output_rejects.
        Canonical v2 artifact paths belong to T3b."""
        import src.extract.pipeline_v2 as p2
        for name in ("DEFAULT_OUTPUT_JSONL", "DEFAULT_OUTPUT_REJECTS"):
            self.assertFalse(
                hasattr(p2, name),
                f"pipeline_v2 must not expose {name} (T3b owns canonical paths)",
            )


# --- 16: PreflightErrorV2 carries its message ------------------------


class TestPreflightErrorV2(unittest.TestCase):

    def test_preflight_error_v2_carries_message(self):
        """A failed preflight raises PreflightErrorV2 with the
        expected message rather than TypeError."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            # Delete the ontology file to force a preflight failure.
            world["ontology"].unlink()
            with self.assertRaises(pipeline_v2.PreflightErrorV2) as cm:
                pipeline_v2.run_extraction_v2(
                    chunks_path=world["chunks"],
                    ontology_path=world["ontology"],
                    prompts_dir=world["prompts"],
                    output_jsonl=outs["jsonl"],
                    output_rejects=outs["rejects"],
                    provider=_FakeLLM(),
                )
            self.assertIn("ontology", str(cm.exception))


# --- 17: schema reject contributes to reject_count -----------------


class TestSchemaRejectCounted(unittest.TestCase):

    def test_schema_reject_increments_reject_count(self):
        """LLMValidationError produces a schema reject; the
        resulting ``RunResultV2.reject_count`` must include that
        schema reject even though the phase ``failure_count``
        remains 0 (per the frozen completion semantics)."""

        class _BoomLLM(LLMProvider):
            name = "boom"
            model = "boom"

            def generate(self, system, user, *, max_tokens=1024, temperature=0.0):
                return "", Usage()

            def generate_json(self, system, user, schema, *, max_tokens=1024, temperature=0.0):
                raise LLMValidationError(
                    model_output="not json",
                    schema=schema,
                    validation_error=ValueError("bad shape"),
                )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            result = pipeline_v2.run_extraction_v2(
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
                output_jsonl=outs["jsonl"],
                output_rejects=outs["rejects"],
                provider=_BoomLLM(),
            )
            # Two schema rejects (entity + relation). Phase
            # ``failure_count`` for both stays 0 per the frozen
            # completion semantics; ``reject_count`` must still
            # include them.
            self.assertEqual(result.reject_count, 2)
            cs_records = [
                r for r in _read_jsonl(outs["jsonl"])
                if r["kind"] == "chunk_status"
            ]
            self.assertEqual(len(cs_records), 1)
            cs = cs_records[0]
            self.assertEqual(cs["phases"]["entity"]["failure_count"], 0)
            self.assertEqual(cs["phases"]["entity"]["status"], "rejected")
            self.assertEqual(cs["phases"]["relation"]["failure_count"], 0)
            self.assertEqual(cs["phases"]["relation"]["status"], "rejected")


# --- 18: fresh-run truncates stale reject artifact ------------------


class TestFreshRunRejectsAreClean(unittest.TestCase):

    def test_stale_reject_content_absent_after_fresh_run(self):
        """A T3a fresh run must not retain stale reject lines when
        the caller reuses ``output_rejects``. The runner truncates
        the reject artifact before writing current-run rejects."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            # Pre-seed a stale reject line that the new run must
            # discard (no current chunk produces a reject).
            outs["rejects"].parent.mkdir(parents=True, exist_ok=True)
            stale = {"chunk_id": "stale", "phase": "stale", "gate": "stale"}
            outs["rejects"].write_text(
                json.dumps(stale) + "\n", encoding="utf-8",
            )
            _run(world, outputs=outs)
            # Fresh run produces zero rejects; the stale line must
            # be gone.
            rejects = _read_jsonl(outs["rejects"])
            self.assertEqual(rejects, [])

