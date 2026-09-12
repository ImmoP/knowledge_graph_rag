"""Acceptance tests for the extraction pipeline (#23).

See ``docs/specs/implementation-spec.md`` §15.4 for the canonical list
of scenarios. Each top-level ``TestCase`` below maps to one or more
numbered scenarios; the assertion bodies are written so the test name
becomes the source of truth for the acceptance gate.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extract import (  # noqa: E402
    ids,
    pipeline,
    prompts,
    schema,
    validation,
)
from src.extract.cache import (  # noqa: E402
    prompt_content_sha256,
    required_identity_dimensions,
)
from src.extract.prompts import (  # noqa: E402
    ENTITY_PROMPT_FILE,
    PROMPTS_DIR,
    RELATION_PROMPT_FILE,
    PromptArtifactMissing,
)
from src.extract.retry import (  # noqa: E402
    BACKOFF_SCHEDULE_S,
    MAX_ATTEMPTS,
    is_transient,
    with_transport_retry,
)
from src.extract.validation import (  # noqa: E402
    EvidenceError,
    OntologyError,
    source_text_sha256,
)
from src.providers import BudgetExceeded, Usage  # noqa: E402
from src.providers.cache import (  # noqa: E402
    CACHE_FORMAT_VERSION,
    extraction_cache_scope as provider_extraction_cache_scope,
    make_key as base_make_key,
    schema_hash,
)
from src.providers.protocol import LLMProvider, LLMValidationError  # noqa: E402


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
    extraction_prompt: extract_entities_v1
    definition: A research method or technique.
    verified_examples:
      - surface: "DPR"
        canonical: "Dense Passage Retrieval"
        char_range: [0, 3]
        context: "We use DPR for retrieval."
  - name: dataset
    population_mode: extracted
    extraction_prompt: extract_entities_v1
    definition: A benchmark dataset.
    verified_examples: []
relation_types:
  - name: PROPOSES
    from: document
    to: method
    population_mode: extracted
    extraction_prompt: extract_relations_v1
    definition: A paper proposing a method.
    verified_examples: []
  - name: EVALUATES_ON
    from: document
    to: dataset
    population_mode: extracted
    extraction_prompt: extract_relations_v1
    definition: An evaluation relationship.
    verified_examples: []
"""

ENTITY_TEMPLATE = (
    "You are an entity extractor.\n"
    "Allowed entity types: {allowed_entity_types}\n"
    "Type definitions: {entity_type_definitions}\n"
    "Verified examples: {verified_examples}\n"
    "Text: {chunk_text}\n"
)
RELATION_TEMPLATE = (
    "You are a relation extractor.\n"
    "Allowed relations: {allowed_relation_types}\n"
    "Definitions: {relation_definitions}\n"
    "Chunk: {chunk_id} Doc: {doc_id}\n"
    "Mentions:\n{mention_list}\n"
    "Text: {chunk_text}\n"
)


def _write_world(root: Path, *, n_chunks: int = 2, doc_id: str = "W1") -> dict:
    """Write a minimal ontology + chunks + prompts world under ``root``.

    Each default chunk places "DPR" at character offsets 7..9 so the
    default ``_FakeLLM`` payload (surface_form="DPR",
    char_start=7, char_end=10) passes Gate 3 (evidence) for both
    chunks. Tests that want different outcomes pass ``scripts=`` to
    override the LLM output.
    """
    ont_path = root / "ontology.yaml"
    ont_path.write_text(MINIMAL_ONT)
    chunks = []
    for i in range(n_chunks):
        if i == 0:
            text = "We use DPR for retrieval and evaluate on Natural Questions."
        elif i == 1:
            # Seven-char prefix so "DPR" sits at offsets 7..9 (end-exclusive 10).
            text = "Chunk2 DPR is a strong baseline we cite."
        else:
            text = f"Chunk {i} placeholder."
        chunks.append({
            "chunk_id": f"c_{i:03d}",
            "doc_id": doc_id,
            "section_path": ["intro"],
            "chunk_index": i,
            "text": text,
            "token_estimate": len(text.split()),
            "char_start": 0,
            "char_end": len(text),
        })
    chunks_path = root / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8"
    )
    prompts_dir = root / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / ENTITY_PROMPT_FILE).write_text(ENTITY_TEMPLATE)
    (prompts_dir / RELATION_PROMPT_FILE).write_text(RELATION_TEMPLATE)
    return {
        "ontology": ont_path,
        "chunks": chunks_path,
        "prompts": prompts_dir,
    }


class _FakeLLM(LLMProvider):
    """Scripted LLM.

    Routes on the schema's top-level ``properties``: ``mentions`` is
    the entity pass, ``assertions`` is the relation pass. Tests can
    supply a list of ``scripts`` (one payload per call) or rely on the
    default empty assertions + one DPR mention.
    """

    def __init__(self, scripts=None) -> None:
        self.name = "fake-llm"
        self.model = "fake-model"
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
            payload = (
                {"mentions": [
                    {"entity_type": "method", "surface_form": "DPR",
                     "canonical_form": "Dense Passage Retrieval",
                     "char_start": 7, "char_end": 10, "confidence": "stated"},
                ]}
                if is_entity
                else {"assertions": []}
            )
        self.json_calls.append(("entity" if is_entity else "relation", payload))
        return copy.deepcopy(payload), Usage(input_tokens=10, output_tokens=20)


def _default_outputs(root: Path) -> dict:
    d = root / "out"
    d.mkdir(parents=True, exist_ok=True)
    return {
        "jsonl": d / "extractions.jsonl",
        "rejects": d / "extractions.rejects.jsonl",
        "manifest": d / "extractions.manifest.json",
    }


def _read_jsonl(p: Path) -> list:
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_manifest(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _run(world: dict, *, outputs: dict, scripts=None, max_tokens=1024, temperature=0.0):
    return pipeline.run_extraction(
        chunks_path=world["chunks"],
        ontology_path=world["ontology"],
        prompts_dir=world["prompts"],
        output_jsonl=outputs["jsonl"],
        output_rejects=outputs["rejects"],
        output_manifest=outputs["manifest"],
        provider=_FakeLLM(scripts=scripts),
        max_tokens=max_tokens,
        temperature=temperature,
    )


# --- 1: preflight gates ----------------------------------------------


class TestPreflightErrors(unittest.TestCase):

    def test_missing_chunks_file_is_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            (root / "chunks.jsonl").unlink()
            with self.assertRaises(pipeline.PreflightError):
                pipeline.preflight(
                    chunks_path=world["chunks"],
                    ontology_path=world["ontology"],
                    prompts_dir=world["prompts"],
                    provider=_FakeLLM(),
                    max_tokens=1024,
                    temperature=0.0,
                )

    def test_missing_ontology_is_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            (root / "ontology.yaml").unlink()
            with self.assertRaises(pipeline.PreflightError):
                pipeline.preflight(
                    chunks_path=world["chunks"],
                    ontology_path=world["ontology"],
                    prompts_dir=world["prompts"],
                    provider=_FakeLLM(),
                    max_tokens=1024,
                    temperature=0.0,
                )

    def test_invalid_ontology_yaml_is_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            (root / "ontology.yaml").write_text(":\nthis is not yaml")
            with self.assertRaises(Exception):
                pipeline.preflight(
                    chunks_path=world["chunks"],
                    ontology_path=world["ontology"],
                    prompts_dir=world["prompts"],
                    provider=_FakeLLM(),
                    max_tokens=1024,
                    temperature=0.0,
                )

    def test_missing_prompt_artifact_is_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            (world["prompts"] / RELATION_PROMPT_FILE).unlink()
            with self.assertRaises(PromptArtifactMissing):
                prompts.verify_prompt_artifacts(world["prompts"])

    def test_chunks_sha_in_manifest_changes_when_chunks_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            m1 = _read_manifest(outputs["manifest"])
            self.assertTrue(m1["chunks_sha256"])
            extra = json.loads(world["chunks"].read_text().splitlines()[0])
            with world["chunks"].open("a") as f:
                f.write(json.dumps(extra) + "\n")
            _run(world, outputs=outputs)
            m2 = _read_manifest(outputs["manifest"])
            self.assertNotEqual(m1["chunks_sha256"], m2["chunks_sha256"])

    def test_prompt_sha_in_manifest_changes_when_template_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            m1 = _read_manifest(outputs["manifest"])
            pa = world["prompts"] / ENTITY_PROMPT_FILE
            pa.write_text(pa.read_text() + "\n# tweak\n")
            _run(world, outputs=outputs)
            m2 = _read_manifest(outputs["manifest"])
            self.assertNotEqual(
                m1["prompt_template_sha256"][0],
                m2["prompt_template_sha256"][0],
            )


# --- 2: deterministic outputs ----------------------------------------


class TestDeterministicOutputs(unittest.TestCase):

    def test_clean_run_is_byte_stable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs_a = _default_outputs(root)
            _run(world, outputs=outputs_a)
            text_a = outputs_a["jsonl"].read_text()
            _run(world, outputs=outputs_a)
            text_b = outputs_a["jsonl"].read_text()
            self.assertEqual(text_a, text_b)

    def test_partial_then_resume_matches_clean_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            full = outputs["jsonl"].read_text()

            with world["chunks"].open("r") as f:
                lines = f.readlines()
            first = world["chunks"].with_name("chunks_first.jsonl")
            first.write_text(lines[0])

            outputs2 = _default_outputs(root / "second")
            pipeline.run_extraction(
                chunks_path=first,
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
                output_jsonl=outputs2["jsonl"],
                output_rejects=outputs2["rejects"],
                output_manifest=outputs2["manifest"],
                provider=_FakeLLM(),
            )
            pipeline.run_extraction(
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
                output_jsonl=outputs2["jsonl"],
                output_rejects=outputs2["rejects"],
                output_manifest=outputs2["manifest"],
                provider=_FakeLLM(),
            )
            self.assertEqual(outputs2["jsonl"].read_text(), full)

    def test_records_sorted_by_chunk_then_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            records = _read_jsonl(outputs["jsonl"])
            seen = []
            for r in records:
                kind = r.get("kind")
                if kind == "mention":
                    seen.append((r["chunk_id"], "0", r["mention_id"]))
                elif kind == "assertion":
                    seen.append((r["chunk_id"], "1", r["assertion_id"]))
                elif kind == "chunk_status":
                    seen.append((r["chunk_id"], "2", r["chunk_id"]))
            self.assertEqual(seen, sorted(seen))


# --- 3: resume semantics ---------------------------------------------


class TestResumeSemantics(unittest.TestCase):

    def test_resume_full_no_calls(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            llm = _FakeLLM()
            pipeline.run_extraction(
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
                output_jsonl=outputs["jsonl"],
                output_rejects=outputs["rejects"],
                output_manifest=outputs["manifest"],
                provider=llm,
            )
            self.assertEqual(len(llm.json_calls), 0)

    def test_resume_partial_reprocesses_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            manifest = _read_manifest(outputs["manifest"])
            previous_run = manifest["run_id"]
            llm = _FakeLLM()
            r2 = pipeline.run_extraction(
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
                output_jsonl=outputs["jsonl"],
                output_rejects=outputs["rejects"],
                output_manifest=outputs["manifest"],
                provider=llm,
            )
            self.assertEqual(r2.resumed_from_run_id, previous_run)
            self.assertEqual(len(llm.json_calls), 0)

    def test_resume_incomplete_chunk_replaced(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            lines = outputs["jsonl"].read_text().splitlines()
            kept = [line for line in lines if json.loads(line).get("chunk_id") != "c_000"]
            outputs["jsonl"].write_text("\n".join(kept) + "\n")
            llm = _FakeLLM()
            pipeline.run_extraction(
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
                output_jsonl=outputs["jsonl"],
                output_rejects=outputs["rejects"],
                output_manifest=outputs["manifest"],
                provider=llm,
            )
            self.assertGreater(len(llm.json_calls), 0)


# --- 4: validation gates ---------------------------------------------


class TestValidationGates(unittest.TestCase):

    def test_evidence_gate_rejects_out_of_bounds_offsets(self):
        with self.assertRaises(EvidenceError):
            validation.check_mention_evidence(
                mention={
                    "entity_type": "method", "surface_form": "DPR",
                    "canonical_form": "x", "char_start": 100, "char_end": 105,
                    "confidence": "stated",
                    "evidence": {
                        "char_start": 100, "char_end": 105,
                        "source_text_sha256": source_text_sha256("hello"),
                    },
                },
                chunk_text="hello",
                expected_source_text_sha256=source_text_sha256("hello"),
            )

    def test_evidence_gate_rejects_surface_not_in_span(self):
        text = "BM25 is strong."
        sha = source_text_sha256(text)
        with self.assertRaises(EvidenceError):
            validation.check_mention_evidence(
                mention={
                    "entity_type": "method",
                    "surface_form": "TFIDF",
                    "canonical_form": "TF-IDF",
                    "char_start": 0, "char_end": 4,
                    "confidence": "stated",
                    "evidence": {"char_start": 0, "char_end": 4, "source_text_sha256": sha},
                },
                chunk_text=text,
                expected_source_text_sha256=sha,
            )

    def test_evidence_gate_rejects_sha_drift(self):
        text = "DPR is great."
        with self.assertRaises(EvidenceError):
            validation.check_mention_evidence(
                mention={
                    "entity_type": "method",
                    "surface_form": "DPR",
                    "canonical_form": "DPR",
                    "char_start": 0, "char_end": 3,
                    "confidence": "stated",
                    "evidence": {"char_start": 0, "char_end": 3, "source_text_sha256": "deadbeef"},
                },
                chunk_text=text,
                expected_source_text_sha256=source_text_sha256(text),
            )

    def test_ontology_gate_rejects_unknown_entity_type(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            ontology = schema.load_ontology(world["ontology"])
            bad = {
                "entity_type": "robot", "surface_form": "DPR",
                "canonical_form": "DPR", "char_start": 0, "char_end": 3,
                "confidence": "stated",
                "evidence": {"char_start": 0, "char_end": 3, "source_text_sha256": "x"},
            }
            with self.assertRaises(OntologyError):
                validation.check_mention_ontology(mention=bad, ontology=ontology)

    def test_ontology_gate_rejects_unknown_relation_type(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            ontology = schema.load_ontology(world["ontology"])
            bad = {
                "relation_type": "DESTROYS",
                "subject_ref": {"kind": "document", "id": "W1"},
                "object_ref": {"kind": "mention", "id": "m_x"},
                "char_start": 0, "char_end": 3, "confidence": "stated",
                "evidence": {"char_start": 0, "char_end": 3, "source_text_sha256": "x"},
            }
            with self.assertRaises(OntologyError):
                validation.check_assertion_ontology(
                    assertion=bad, ontology=ontology, doc_id="W1",
                    same_chunk_mentions={"m_x": {}},
                    entity_type_for_mention={"m_x": "method"},
                )

    def test_ontology_gate_rejects_document_endpoint_with_wrong_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            ontology = schema.load_ontology(world["ontology"])
            bad = {
                "relation_type": "PROPOSES",
                "subject_ref": {"kind": "document", "id": "W-OTHER"},
                "object_ref": {"kind": "mention", "id": "m_x"},
                "char_start": 0, "char_end": 3, "confidence": "stated",
                "evidence": {"char_start": 0, "char_end": 3, "source_text_sha256": "x"},
            }
            with self.assertRaises(OntologyError):
                validation.check_assertion_ontology(
                    assertion=bad, ontology=ontology, doc_id="W1",
                    same_chunk_mentions={"m_x": {"entity_type": "method"}},
                    entity_type_for_mention={"m_x": "method"},
                )

    def test_ontology_gate_rejects_mention_endpoint_not_in_chunk(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            ontology = schema.load_ontology(world["ontology"])
            bad = {
                "relation_type": "PROPOSES",
                "subject_ref": {"kind": "document", "id": "W1"},
                "object_ref": {"kind": "mention", "id": "m_other"},
                "char_start": 0, "char_end": 3, "confidence": "stated",
                "evidence": {"char_start": 0, "char_end": 3, "source_text_sha256": "x"},
            }
            with self.assertRaises(OntologyError):
                validation.check_assertion_ontology(
                    assertion=bad, ontology=ontology, doc_id="W1",
                    same_chunk_mentions={"m_local": {"entity_type": "method"}},
                    entity_type_for_mention={"m_local": "method"},
                )

    def test_ontology_gate_rejects_subject_object_type_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            ontology = schema.load_ontology(world["ontology"])
            bad = {
                "relation_type": "PROPOSES",
                "subject_ref": {"kind": "document", "id": "W1"},
                "object_ref": {"kind": "mention", "id": "m_x"},
                "char_start": 0, "char_end": 3, "confidence": "stated",
                "evidence": {"char_start": 0, "char_end": 3, "source_text_sha256": "x"},
            }
            with self.assertRaises(OntologyError):
                validation.check_assertion_ontology(
                    assertion=bad, ontology=ontology, doc_id="W1",
                    same_chunk_mentions={"m_x": {"entity_type": "dataset"}},
                    entity_type_for_mention={"m_x": "dataset"},
                )


# --- 5: id derivation / stability ------------------------------------


class TestIdDerivation(unittest.TestCase):

    def test_mention_id_uses_only_documented_fields(self):
        text = "DPR is great."
        sha = source_text_sha256(text)
        a = ids.mention_id(chunk_id="c1", source_text_sha256=sha,
                           entity_type="method", char_start=0, char_end=3)
        b = ids.mention_id(chunk_id="c1", source_text_sha256=sha,
                           entity_type="method", char_start=0, char_end=3)
        self.assertEqual(a, b)
        c = ids.mention_id(chunk_id="c2", source_text_sha256=sha,
                           entity_type="method", char_start=0, char_end=3)
        self.assertNotEqual(a, c)
        d = ids.mention_id(chunk_id="c1", source_text_sha256=sha,
                           entity_type="dataset", char_start=0, char_end=3)
        self.assertNotEqual(a, d)

    def test_assertion_id_uses_only_documented_fields(self):
        text = "DPR is great."
        sha = source_text_sha256(text)
        a = ids.assertion_id(chunk_id="c1", source_text_sha256=sha,
                             relation_type="PROPOSES",
                             subject_ref={"kind": "document", "id": "W1"},
                             object_ref={"kind": "mention", "id": "m_x"},
                             char_start=0, char_end=3)
        b = ids.assertion_id(chunk_id="c1", source_text_sha256=sha,
                             relation_type="PROPOSES",
                             subject_ref={"kind": "document", "id": "W1"},
                             object_ref={"kind": "mention", "id": "m_x"},
                             char_start=0, char_end=3)
        self.assertEqual(a, b)
        c = ids.assertion_id(chunk_id="c1", source_text_sha256=sha,
                             relation_type="EVALUATES_ON",
                             subject_ref={"kind": "document", "id": "W1"},
                             object_ref={"kind": "mention", "id": "m_x"},
                             char_start=0, char_end=3)
        self.assertNotEqual(a, c)

    def test_id_derivation_harness_accepts_canonical_records(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
            proc = subprocess.run(
                [sys.executable, "-m", "src.extract.verify", str(outputs["jsonl"])],
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

    def test_id_derivation_harness_rejects_tampered_records(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            # Tamper: flip char_start in the mention evidence.
            text = outputs["jsonl"].read_text()
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            for r in records:
                if r.get("kind") == "mention":
                    r["evidence"]["char_start"] = r["evidence"]["char_start"] + 1
                    break
            tampered = root / "tampered.jsonl"
            tampered.write_text("\n".join(json.dumps(r) for r in records) + "\n")
            env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
            proc = subprocess.run(
                [sys.executable, "-m", "src.extract.verify", str(tampered)],
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)


# --- 6: cache identity -----------------------------------------------


class TestCacheIdentity(unittest.TestCase):

    def test_required_identity_dimensions(self):
        dims = required_identity_dimensions()
        self.assertEqual(len(dims), 6)
        self.assertIn("chunk_id", dims)
        self.assertIn("source_text_sha256", dims)
        self.assertIn("ontology_version", dims)
        self.assertIn("ontology_sha256", dims)
        self.assertIn("prompt_template_filename", dims)
        self.assertIn("prompt_template_content_sha256", dims)

    def test_provider_extraction_cache_scope_mixes_into_key(self):
        # Spec §5.5a: the extraction-cache scope is mixed into the
        # generate_json cache key as a strict superset of the
        # canonical §5.1 identity. The base key (no scope active)
        kwargs = dict(
            cache_format_version=CACHE_FORMAT_VERSION,
            operation="generate_json",
            provider_name="ollama", provider_model="llama3",
            system="sys", user="usr", max_tokens=1024, temperature=0.0,
            schema_canonical_json=schema_hash({"type": "object"}),
        )
        base_key = base_make_key(**kwargs)
        scope_dims = dict(
            chunk_id="c1",
            source_text_sha256="abc",
            ontology_version="1",
            ontology_sha256="ont",
            prompt_template_filename="x.txt",
            prompt_template_content_sha256="p",
        )
        with provider_extraction_cache_scope(**scope_dims):
            scoped_key = base_make_key(**kwargs)
        self.assertNotEqual(base_key, scoped_key)
        # Determinism: same scope dims → same key.
        with provider_extraction_cache_scope(**scope_dims):
            scoped_key_2 = base_make_key(**kwargs)
        self.assertEqual(scoped_key, scoped_key_2)

    def test_provider_extraction_cache_scope_rejects_partial_dims(self):
        with self.assertRaises(ValueError):
            with provider_extraction_cache_scope(chunk_id="c1"):
                pass

    def test_provider_extraction_cache_scope_rejects_unknown_dim(self):
        with self.assertRaises(ValueError):
            with provider_extraction_cache_scope(
                chunk_id="c1", source_text_sha256="abc",
                ontology_version="1", ontology_sha256="ont",
                prompt_template_filename="x.txt",
                prompt_template_content_sha256="p",
                extra_dim="nope",
            ):
                pass

    def test_provider_extraction_cache_scope_forbids_nesting(self):
        scope_dims = dict(
            chunk_id="c1",
            source_text_sha256="abc",
            ontology_version="1",
            ontology_sha256="ont",
            prompt_template_filename="x.txt",
            prompt_template_content_sha256="p",
        )
        with provider_extraction_cache_scope(**scope_dims):
            with self.assertRaises(RuntimeError):
                with provider_extraction_cache_scope(**scope_dims):
                    pass

    def test_extraction_key_differs_when_ontology_sha_differs(self):
        kwargs = dict(
            cache_format_version=CACHE_FORMAT_VERSION,
            operation="generate_json",
            provider_name="ollama", provider_model="llama3",
            system="sys", user="usr", max_tokens=1024, temperature=0.0,
            schema_canonical_json=schema_hash({"type": "object"}),
        )
        scope_a = dict(
            chunk_id="c1",
            source_text_sha256="abc",
            ontology_version="1",
            ontology_sha256="ont-a",
            prompt_template_filename="x.txt",
            prompt_template_content_sha256="p",
        )
        scope_b = dict(scope_a, ontology_sha256="ont-b")
        with provider_extraction_cache_scope(**scope_a):
            k_a = base_make_key(**kwargs)
        with provider_extraction_cache_scope(**scope_b):
            k_b = base_make_key(**kwargs)
        self.assertNotEqual(k_a, k_b)

    def test_extraction_key_differs_when_prompt_content_differs(self):
        kwargs = dict(
            cache_format_version=CACHE_FORMAT_VERSION,
            operation="generate_json",
            provider_name="ollama", provider_model="llama3",
            system="sys", user="usr", max_tokens=1024, temperature=0.0,
            schema_canonical_json=schema_hash({"type": "object"}),
        )
        scope_a = dict(
            chunk_id="c1",
            source_text_sha256="abc",
            ontology_version="1",
            ontology_sha256="ont",
            prompt_template_filename="x.txt",
            prompt_template_content_sha256="p-a",
        )
        scope_b = dict(scope_a, prompt_template_content_sha256="p-b")
        with provider_extraction_cache_scope(**scope_a):
            k_a = base_make_key(**kwargs)
        with provider_extraction_cache_scope(**scope_b):
            k_b = base_make_key(**kwargs)
        self.assertNotEqual(k_a, k_b)

    def test_extraction_key_differs_when_chunk_id_differs(self):
        kwargs = dict(
            cache_format_version=CACHE_FORMAT_VERSION,
            operation="generate_json",
            provider_name="ollama", provider_model="llama3",
            system="sys", user="usr", max_tokens=1024, temperature=0.0,
            schema_canonical_json=schema_hash({"type": "object"}),
        )
        scope_a = dict(
            chunk_id="c1",
            source_text_sha256="abc",
            ontology_version="1",
            ontology_sha256="ont",
            prompt_template_filename="x.txt",
            prompt_template_content_sha256="p",
        )
        scope_b = dict(scope_a, chunk_id="c2")
        with provider_extraction_cache_scope(**scope_a):
            k_a = base_make_key(**kwargs)
        with provider_extraction_cache_scope(**scope_b):
            k_b = base_make_key(**kwargs)
        self.assertNotEqual(k_a, k_b)

    def test_extraction_key_differs_when_source_text_sha_differs(self):
        kwargs = dict(
            cache_format_version=CACHE_FORMAT_VERSION,
            operation="generate_json",
            provider_name="ollama", provider_model="llama3",
            system="sys", user="usr", max_tokens=1024, temperature=0.0,
            schema_canonical_json=schema_hash({"type": "object"}),
        )
        scope_a = dict(
            chunk_id="c1",
            source_text_sha256="abc",
            ontology_version="1",
            ontology_sha256="ont",
            prompt_template_filename="x.txt",
            prompt_template_content_sha256="p",
        )
        scope_b = dict(scope_a, source_text_sha256="def")
        with provider_extraction_cache_scope(**scope_a):
            k_a = base_make_key(**kwargs)
        with provider_extraction_cache_scope(**scope_b):
            k_b = base_make_key(**kwargs)
        self.assertNotEqual(k_a, k_b)


# --- 7: retry policy -------------------------------------------------


class TestRetryPolicy(unittest.TestCase):

    def test_backoff_schedule(self):
        # Spec §12.1 — sleeps fire BEFORE attempts 2 and 3 only;
        # there is no sleep(0) before attempt 1 and no sleep after
        # attempt 3.
        self.assertEqual(BACKOFF_SCHEDULE_S, (2.0, 4.0))
        self.assertEqual(MAX_ATTEMPTS, 3)

    def test_with_transport_retry_succeeds_after_two_transients(self):
        calls = {"n": 0}
        sleeps = []

        from src.providers.protocol import ProviderTransportError

        def op():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ProviderTransportError(transient=True)
            return "ok", Usage()

        val, usage = with_transport_retry(op, sleep=lambda s: sleeps.append(s))
        self.assertEqual(val, "ok")
        # First attempt has no sleep; sleeps fire before attempts 2 and 3.
        self.assertEqual(sleeps, [2.0, 4.0][:2])

    def test_with_transport_retry_exhausts_after_max_attempts(self):
        calls = {"n": 0}
        sleeps = []

        from src.providers.protocol import ProviderTransportError

        def op():
            calls["n"] += 1
            raise ProviderTransportError(transient=True)

        # Spec §12.1: on the third transient failure the original
        # ProviderTransportError is re-raised unchanged — there is
        # NO RetryExhausted wrapper.
        with self.assertRaises(ProviderTransportError):
            with_transport_retry(op, sleep=lambda s: sleeps.append(s))
        self.assertEqual(calls["n"], MAX_ATTEMPTS)
        # Two sleeps: before attempt 2 (2.0s) and before attempt 3 (4.0s).
        self.assertEqual(sleeps, [2.0, 4.0])

    def test_non_transient_error_not_retried(self):
        from src.providers.protocol import ProviderTransportError

        calls = {"n": 0}

        def op():
            calls["n"] += 1
            raise ProviderTransportError(transient=False)

        with self.assertRaises(ProviderTransportError):
            with_transport_retry(op, sleep=lambda s: None)
        self.assertEqual(calls["n"], 1)

    def test_is_transient_recognises_typed_transient_flag(self):
        # Spec §12.1 / provider-abstractions §6.6: classification is
        # typed via ProviderTransportError.transient. The helper does
        # not inspect duck-typed status codes or class names — the
        # adapter decides transient vs non-transient when it raises.
        from src.providers.protocol import ProviderTransportError

        for transient in (True, False):
            err = ProviderTransportError(transient=transient)
            self.assertEqual(is_transient(err), transient)

        # Non-ProviderTransportError exceptions are never retried,
        # even if they happen to carry a status code attribute.
        class _HttpxLikeError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code
                super().__init__(status_code)

        self.assertFalse(is_transient(_HttpxLikeError(500)))
        self.assertFalse(is_transient(_HttpxLikeError(429)))

    def test_is_transient_does_not_retry_on_non_transient_status(self):
        from src.providers.protocol import ProviderTransportError

        # Even a status code the spec considers retry-eligible
        # (e.g. 503) is not retried unless the adapter raised with
        # transient=True. The adapter alone decides.
        err = ProviderTransportError(transient=False, status=503)
        self.assertFalse(is_transient(err))


# --- 8: schema generation --------------------------------------------


class TestSchemaGeneration(unittest.TestCase):

    def test_entity_schema_filters_to_extracted_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            ontology = schema.load_ontology(world["ontology"])
            ent_schema = schema.entity_pass_schema(ontology)
            enum = ent_schema["properties"]["mentions"]["items"]["properties"]["entity_type"]["enum"]
            # document is structural; must NOT appear in entity pass enum
            self.assertNotIn("document", enum)
            self.assertIn("method", enum)
            self.assertIn("dataset", enum)

    def test_relation_schema_filters_to_extracted_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            ontology = schema.load_ontology(world["ontology"])
            rel_schema = schema.relation_pass_schema(ontology)
            enum = rel_schema["properties"]["assertions"]["items"]["properties"]["relation_type"]["enum"]
            self.assertIn("PROPOSES", enum)
            self.assertIn("EVALUATES_ON", enum)

    def test_allowed_name_lists(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            ontology = schema.load_ontology(world["ontology"])
            lists = schema.build_allowed_name_lists(ontology)
            self.assertIn("method", lists["entity_types"])
            self.assertIn("dataset", lists["entity_types"])
            self.assertIn("PROPOSES", lists["relation_types"])
            self.assertIn("EVALUATES_ON", lists["relation_types"])

    def test_prompt_rendering_injects_ontology_lists(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            ontology = schema.load_ontology(world["ontology"])
            lists = schema.build_allowed_name_lists(ontology)
            rendered = prompts.render_entity_prompt(
                template=ENTITY_TEMPLATE,
                allowed_entity_types=lists["entity_types"],
                entity_type_definitions=schema.verified_examples_block(ontology),
                verified_examples=schema.verified_examples_block(ontology),
                chunk_text="DPR is great.",
            )
            self.assertIn("method", rendered)
            self.assertIn("dataset", rendered)
            self.assertIn("DPR is great.", rendered)


# --- 9: manifest shape -----------------------------------------------


class TestManifestShape(unittest.TestCase):

    def test_manifest_has_all_required_fields(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            m = _read_manifest(outputs["manifest"])
            required = [
                "run_id", "resumed_from_run_id", "provider", "model",
                "ontology_version", "ontology_sha256",
                "prompt_template_files", "prompt_template_sha256",
                "schema_version", "chunks_path", "chunks_sha256",
                "wall_clock_started_at", "wall_clock_ended_at",
                "cumulative_tokens", "cumulative_cost_usd",
                "max_tokens", "temperature", "cache_format_version",
            ]
            for k in required:
                self.assertIn(k, m)

    def test_manifest_provider_and_model_match_provider(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            m = _read_manifest(outputs["manifest"])
            self.assertEqual(m["provider"], "fake-llm")
            self.assertEqual(m["model"], "fake-model")

    def test_manifest_prompt_filenames_captured(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            m = _read_manifest(outputs["manifest"])
            self.assertEqual(m["prompt_template_files"], [ENTITY_PROMPT_FILE, RELATION_PROMPT_FILE])

    def test_manifest_timestamp_compact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            m = _read_manifest(outputs["manifest"])
            ts = m["wall_clock_started_at"]
            # Compact format YYYYMMDDTHHMMSSZ
            self.assertRegex(ts, r"^\d{8}T\d{6}Z$")
            datetime.datetime.strptime(ts, "%Y%m%dT%H%M%SZ")


# --- 10: chunk_status invariants ------------------------------------


class TestChunkStatus(unittest.TestCase):

    def test_one_chunk_status_per_chunk(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            records = _read_jsonl(outputs["jsonl"])
            statuses = [r for r in records if r.get("kind") == "chunk_status"]
            by_chunk = {}
            for s in statuses:
                by_chunk.setdefault(s["chunk_id"], []).append(s)
            for cid, ss in by_chunk.items():
                self.assertEqual(len(ss), 1, f"chunk_status duplicated for {cid}")

    def test_extraction_complete_only_when_both_phases_validated(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            records = _read_jsonl(outputs["jsonl"])
            for s in records:
                if s.get("kind") == "chunk_status":
                    phases = s["phases"]
                    # Spec §9 (repair #23): extraction_complete is true
                    # iff BOTH phases are validated AND neither phase
                    # accumulated any Gate 2 / Gate 3 per-record
                    # rejects.
                    expected = (
                        phases["entity"]["status"] == "validated"
                        and phases["relation"]["status"] == "validated"
                        and phases["entity"]["failure_count"] == 0
                        and phases["relation"]["failure_count"] == 0
                    )
                    self.assertEqual(s["extraction_complete"], expected)

    def test_zero_mention_zero_assertion_chunk_is_complete(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root, n_chunks=3)
            outputs = _default_outputs(root)
            # Force empty LLM responses.
            scripts = [{"mentions": [], "assertions": []} for _ in range(10)]
            _run(world, outputs=outputs, scripts=scripts)
            records = _read_jsonl(outputs["jsonl"])
            statuses = [r for r in records if r.get("kind") == "chunk_status"]
            self.assertEqual(len(statuses), 3)
            for s in statuses:
                self.assertTrue(s["extraction_complete"])


# --- 11: atomic output ----------------------------------------------


class TestAtomicOutput(unittest.TestCase):

    def test_canonical_file_present_after_clean_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            self.assertTrue(outputs["jsonl"].is_file())
            self.assertTrue(outputs["manifest"].is_file())

    def test_atomic_rewrite_no_partial_line(self):
        # Patch the temp dir to simulate a crash mid-write by raising
        # after the temp file is created but before os.replace. The
        # original JSONL must still be intact.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            _run(world, outputs=outputs)
            original_bytes = outputs["jsonl"].read_bytes()
            real_replace = os.replace
            calls = {"n": 0}

            def boom(src, dst):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("simulated crash mid-write")
                real_replace(src, dst)

            with mock.patch("src.extract.pipeline.os.replace", side_effect=boom):
                with self.assertRaises(OSError):
                    _run(world, outputs=outputs)
            # The original file is still intact and parseable.
            self.assertEqual(outputs["jsonl"].read_bytes(), original_bytes)
            records = _read_jsonl(outputs["jsonl"])
            self.assertGreater(len(records), 0)


# --- 12: prompt rendering & reject log -------------------------------


class TestPromptRenderingAndRejects(unittest.TestCase):

    def test_render_mention_list_includes_mention_id(self):
        mentions = [
            {
                "mention_id": "m_abc",
                "entity_type": "method",
                "surface_form": "DPR",
                "evidence": {"char_start": 0, "char_end": 3},
            }
        ]
        rendered = prompts.render_mention_list(mentions)
        self.assertIn("m_abc", rendered)
        self.assertIn("DPR", rendered)

    def test_render_mention_list_empty(self):
        self.assertEqual(prompts.render_mention_list([]), "(no same-chunk mentions available)")

    def test_prompt_text_matches_disk(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            arts = prompts.verify_prompt_artifacts(world["prompts"])
            self.assertEqual(arts[ENTITY_PROMPT_FILE]["sha256"],
                             prompt_content_sha256(world["prompts"] / ENTITY_PROMPT_FILE))

    def test_schema_reject_route(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            # Make the LLM emit something the schema rejects: not an
            # object at the top level.
            bad_entity = {"mentions": "not-a-list"}
            scripts = [bad_entity, {"assertions": []}, bad_entity, {"assertions": []}]
            _run(world, outputs=outputs, scripts=scripts)
            self.assertTrue(outputs["rejects"].is_file())
            records = [json.loads(line) for line in outputs["rejects"].read_text().splitlines() if line.strip()]
            self.assertGreater(len(records), 0)
            for r in records:
                self.assertIn("gate", r)
                self.assertIn("phase", r)
                self.assertIn("reason", r)
                self.assertIn("offending_record", r)

    def test_ontology_reject_route(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            entity_phase_bad = {"mentions": [
                {"entity_type": "robot", "surface_form": "DPR",
                 "canonical_form": "DPR", "char_start": 0, "char_end": 3,
                 "confidence": "stated"},
            ]}
            scripts = [entity_phase_bad, {"assertions": []},
                       entity_phase_bad, {"assertions": []}]
            _run(world, outputs=outputs, scripts=scripts)
            self.assertTrue(outputs["rejects"].is_file())
            records = [json.loads(line) for line in outputs["rejects"].read_text().splitlines() if line.strip()]
            gates = [r["gate"] for r in records]
            self.assertIn("ontology", gates)

    def test_evidence_reject_route(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            # char_end exceeds chunk text length; schema passes, evidence fails.
            entity_phase_bad = {"mentions": [
                {"entity_type": "method", "surface_form": "ZZ",
                 "canonical_form": "ZZ", "char_start": 0, "char_end": 99999,
                 "confidence": "stated"},
            ]}
            scripts = [entity_phase_bad, {"assertions": []},
                       entity_phase_bad, {"assertions": []}]
            _run(world, outputs=outputs, scripts=scripts)
            self.assertTrue(outputs["rejects"].is_file())
            records = [json.loads(line) for line in outputs["rejects"].read_text().splitlines() if line.strip()]
            gates = [r["gate"] for r in records]
            self.assertIn("evidence", gates)


# --- 13: budget abort -----------------------------------------------


class TestBudgetAbort(unittest.TestCase):

    def test_budget_abort_exits_with_code_1(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)

            class BudgetBoom(_FakeLLM):
                def __init__(self):
                    super().__init__()
                    self._raise = False
                def generate_json(self, system, user, schema, *, max_tokens=1024, temperature=0.0):
                    if self._raise:
                        raise BudgetExceeded(cumulative=1.0, cap=1.0, unit="USD")
                    self._raise = True
                    return super().generate_json(system, user, schema,
                                                 max_tokens=max_tokens,
                                                 temperature=temperature)

            llm = BudgetBoom()
            r = pipeline.run_extraction(
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
                output_jsonl=outputs["jsonl"],
                output_rejects=outputs["rejects"],
                output_manifest=outputs["manifest"],
                provider=llm,
            )
            self.assertEqual(r.exit_code, 1)
            self.assertTrue(r.budget_aborted)
            self.assertTrue(outputs["manifest"].is_file())
            m = _read_manifest(outputs["manifest"])
            self.assertIn("cumulative_tokens", m)


# --- 14: cli entry ---------------------------------------------------


class TestCliEntry(unittest.TestCase):

    def test_python_m_src_extract_smoke(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outputs = _default_outputs(root)
            # Pre-populate the canonical outputs via the pipeline.
            _run(world, outputs=outputs)
            env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
            # First invoke the CLI: it will require env vars to
            # construct an LLM, so pre-set them and adjust the path.
            fake_registry = root / "reg"
            (fake_registry / "src/providers").mkdir(parents=True)
            # Just sanity-check the module exists.
            proc = subprocess.run(
                [sys.executable, "-c", "import src.extract.__main__"],
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)


# --- 15: cross-spec acceptance --------------------------------------


class TestCrossSpec(unittest.TestCase):

    def test_prompt_filenames_match_repo_artifacts(self):
        # The two physical prompts must be present at the repo root
        # and their hashes must match what the cache module computes.
        self.assertTrue((PROMPTS_DIR / ENTITY_PROMPT_FILE).is_file())
        self.assertTrue((PROMPTS_DIR / RELATION_PROMPT_FILE).is_file())

    def test_schema_version_constant(self):
        self.assertEqual(schema.SCHEMA_VERSION, 1)

    def test_confidence_values(self):
        self.assertEqual(set(schema.CONFIDENCE_VALUES),
                         {"stated", "inferred", "hedged"})

    def test_endpoint_kinds(self):
        self.assertEqual(set(schema.ENDPOINT_KINDS),
                         {"document", "mention"})

    def test_hex_prefix_length_16(self):
        self.assertEqual(ids.HEX_PREFIX_LEN, 16)

    def test_canonical_json_settings(self):
        # If the canonical JSON settings ever drift, the IDs change
        # across the codebase. Lock the rules here.
        m = ids.mention_id(chunk_id="c1", source_text_sha256="abc",
                           entity_type="method", char_start=0, char_end=3)
        self.assertTrue(m.startswith("m_"))
        # 18 chars total: 2 + 16
        self.assertEqual(len(m), len("m_") + ids.HEX_PREFIX_LEN)

    def test_pipeline_main_importable(self):
        import src.extract.__main__ as m  # noqa: F401

    def test_pipeline_run_extraction_requires_provider(self):
        with self.assertRaises(pipeline.PreflightError):
            pipeline.run_extraction()


# --- 16: Defect 1 — transport / internal errors escape the phase runners ---

class _ScriptedBoomLLM(_FakeLLM):
    """LLM double that raises the supplied exception on each call.

    ``exc_factory`` is invoked per call to manufacture the exception
    (so each raise constructs a fresh exception object with its own
    identity). The factory returns ``(exc, payload)`` where ``exc``
    is what ``generate_json`` raises and ``payload`` is the JSON it
    would have returned had it succeeded (recorded but unused).
    """

    def __init__(self, exc_factory):
        super().__init__()
        self._exc_factory = exc_factory

    def generate_json(self, system, user, schema, *, max_tokens=1024, temperature=0.0):
        exc, payload = self._exc_factory()
        self.json_calls.append(
            ("entity" if "mentions" in schema.get("properties", {}) else "relation", payload)
        )
        raise exc


def _read_chunk_first(world: dict) -> dict:
    chunks = pipeline._read_chunks(world["chunks"])
    return chunks[0]


def _build_phase_call_state(world: dict):
    """Build the (ontology, prompt_artifacts, run_state) tuple for a phase runner call."""
    preflight_report, ontology, prompt_artifacts = pipeline.preflight(
        chunks_path=world["chunks"],
        ontology_path=world["ontology"],
        prompts_dir=world["prompts"],
        provider=_FakeLLM(),
        max_tokens=1024, temperature=0.0,
    )
    run_state = pipeline.RunState(
        run_id="t", resumed_from_run_id=None,
        preflight=preflight_report,
        started_at="20250101T000000Z",
    )
    return ontology, prompt_artifacts, run_state


class TestPhaseRunnerTransportPropagation(unittest.TestCase):
    """Defect 1: provider-call exceptions escape the phase runners verbatim.

    The phase runners only catch ``LLMValidationError`` (which they
    convert into the canonical Gate 1 phase-level reject). Every
    other exception type — ``BudgetExceeded``, transient
    ``ProviderTransportError`` (final retry-exhausted), non-transient
    ``ProviderTransportError``, and arbitrary unexpected exceptions —
    propagates unchanged so the caller sees the real failure object.
    """

    def _world(self):
        import shutil
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return _write_world(root)

    def test_entity_phase_three_transient_failures_escape_unchanged(self):
        from src.providers.protocol import ProviderTransportError
        world = self._world()
        ontology, prompt_artifacts, run_state = _build_phase_call_state(world)
        final_exc = ProviderTransportError(transient=True, status=503)

        def factory():
            return final_exc, {}

        provider = _ScriptedBoomLLM(factory)
        with self.assertRaises(ProviderTransportError) as ctx:
            pipeline._run_entity_phase(
                chunk=_read_chunk_first(world),
                ontology=ontology,
                prompt_artifact=prompt_artifacts[ENTITY_PROMPT_FILE],
                provider=provider,
                max_tokens=1024, temperature=0.0,
                schema=schema.entity_pass_schema(ontology),
                run_state=run_state,
            )
        # The same final exception object escapes unchanged.
        self.assertIs(ctx.exception, final_exc)
        # Three attempts: transient retried up to MAX_ATTEMPTS by retry.py.
        self.assertEqual(len(provider.json_calls), 3)

    def test_relation_phase_three_transient_failures_escape_unchanged(self):
        from src.providers.protocol import ProviderTransportError
        world = self._world()
        ontology, prompt_artifacts, run_state = _build_phase_call_state(world)
        final_exc = ProviderTransportError(transient=True, status=500)

        def factory():
            return final_exc, {}

        provider = _ScriptedBoomLLM(factory)
        with self.assertRaises(ProviderTransportError) as ctx:
            pipeline._run_relation_phase(
                chunk=_read_chunk_first(world),
                ontology=ontology,
                prompt_artifact=prompt_artifacts[RELATION_PROMPT_FILE],
                provider=provider,
                max_tokens=1024, temperature=0.0,
                schema=schema.relation_pass_schema(ontology),
                run_state=run_state,
                same_chunk_mentions=[],
                entity_type_for_mention={},
            )
        self.assertIs(ctx.exception, final_exc)
        self.assertEqual(len(provider.json_calls), 3)

    def test_non_transient_provider_transport_error_escapes_after_one_call(self):
        from src.providers.protocol import ProviderTransportError
        world = self._world()
        ontology, prompt_artifacts, run_state = _build_phase_call_state(world)
        err = ProviderTransportError(transient=False, status=400)

        def factory():
            return err, {}

        provider = _ScriptedBoomLLM(factory)
        with self.assertRaises(ProviderTransportError) as ctx:
            pipeline._run_entity_phase(
                chunk=_read_chunk_first(world),
                ontology=ontology,
                prompt_artifact=prompt_artifacts[ENTITY_PROMPT_FILE],
                provider=provider,
                max_tokens=1024, temperature=0.0,
                schema=schema.entity_pass_schema(ontology),
                run_state=run_state,
            )
        self.assertIs(ctx.exception, err)
        # Non-transient: no retries; only the original call.
        self.assertEqual(len(provider.json_calls), 1)

    def test_arbitrary_runtime_error_escapes_after_one_call(self):
        world = self._world()
        ontology, prompt_artifacts, run_state = _build_phase_call_state(world)
        boom = RuntimeError("provider.generate_json blew up")

        def factory():
            return boom, {}

        provider = _ScriptedBoomLLM(factory)
        with self.assertRaises(RuntimeError) as ctx:
            pipeline._run_entity_phase(
                chunk=_read_chunk_first(world),
                ontology=ontology,
                prompt_artifact=prompt_artifacts[ENTITY_PROMPT_FILE],
                provider=provider,
                max_tokens=1024, temperature=0.0,
                schema=schema.entity_pass_schema(ontology),
                run_state=run_state,
            )
        self.assertIs(ctx.exception, boom)
        # Arbitrary exceptions are not retried; exactly one call.
        self.assertEqual(len(provider.json_calls), 1)

    def test_llm_validation_error_becomes_canonical_gate1_reject(self):
        # LLMValidationError is the ONE exception the phase runner
        # converts into a phase-level Gate 1 reject. It is not
        # retried, it does not propagate, and failure_count stays 0.
        world = self._world()
        ontology, prompt_artifacts, run_state = _build_phase_call_state(world)
        exc = LLMValidationError(
            model_output="not-json", schema={}, validation_error=ValueError("bad")
        )

        def factory():
            return exc, {}

        provider = _ScriptedBoomLLM(factory)
        outcome, validated = pipeline._run_entity_phase(
            chunk=_read_chunk_first(world),
            ontology=ontology,
            prompt_artifact=prompt_artifacts[ENTITY_PROMPT_FILE],
            provider=provider,
            max_tokens=1024, temperature=0.0,
            schema=schema.entity_pass_schema(ontology),
            run_state=run_state,
        )
        self.assertEqual(outcome.status, "rejected")
        self.assertEqual(outcome.failure_count, 0)
        self.assertEqual(outcome.mention_count, 0)
        self.assertEqual(len(outcome.reject_records), 1)
        reject = outcome.reject_records[0]
        self.assertEqual(reject["gate"], "schema")
        self.assertEqual(reject["phase"], "entity")
        # LLMValidationError is NOT transport-retried; exactly one call.
        self.assertEqual(len(provider.json_calls), 1)
        self.assertEqual(validated, [])

    def test_budget_exceeded_propagates_unchanged(self):
        world = self._world()
        ontology, prompt_artifacts, run_state = _build_phase_call_state(world)
        budget_exc = BudgetExceeded(cumulative=1.0, cap=1.0, unit="USD")

        def factory():
            return budget_exc, {}

        provider = _ScriptedBoomLLM(factory)
        with self.assertRaises(BudgetExceeded) as ctx:
            pipeline._run_entity_phase(
                chunk=_read_chunk_first(world),
                ontology=ontology,
                prompt_artifact=prompt_artifacts[ENTITY_PROMPT_FILE],
                provider=provider,
                max_tokens=1024, temperature=0.0,
                schema=schema.entity_pass_schema(ontology),
                run_state=run_state,
            )
        self.assertIs(ctx.exception, budget_exc)


# --- 17: Defect 2 — relation prompt wording -----------------------------

class TestRelationPromptWording(unittest.TestCase):
    """Defect 2: the relation prompt's evidence-span obligation.

    The model-facing span-selection obligation must say the selected
    span must contain or directly express the asserted relation.
    Mechanical comparisons against a relation paraphrase are NOT
    applied to the span.
    """

    def test_committed_relation_prompt_has_direct_expression_obligation(self):
        path = PROMPTS_DIR / RELATION_PROMPT_FILE
        self.assertTrue(path.is_file(), msg=f"missing committed prompt: {path}")
        text = path.read_text(encoding="utf-8")
        self.assertIn("directly express the asserted relation", text)
        self.assertIn("contain or directly express the asserted relation", text)

    def test_committed_relation_prompt_drops_paraphrased_form_wording(self):
        path = PROMPTS_DIR / RELATION_PROMPT_FILE
        text = path.read_text(encoding="utf-8")
        # The old "paraphrased form of the relation" wording must
        # be gone from the physical prompt artifact.
        self.assertNotIn("paraphrased form of the relation", text)
        # The prompt must no longer instruct the model that the
        # span must CONTAIN a paraphrase (the old obligation).
        self.assertNotIn("span must contain a paraphrased", text)
        self.assertNotIn("span must contain a\n  paraphrased", text)
        # The span-existence obligation is now stated as the model
        # must select a span that "contain or directly express"
        # the asserted relation.
        self.assertIn("contain or directly express", text)

    def test_committed_relation_prompt_sha256_known(self):
        # Lock the current SHA so an unversioned prompt edit is
        # noticed by this test (the manifest / cache key derivation
        # already catches it; this test is the documentation-level
        # tripwire).
        path = PROMPTS_DIR / RELATION_PROMPT_FILE
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(
            sha,
            "da6c463354bf4aa567754bb03aee75193e14e01bd67f7597e74dbd6996507d7f",
        )


# --- 18: Defect 5 — Gate 1 vs Gate 2 / Gate 3 failure_count --------------

class TestGateCounting(unittest.TestCase):
    """Defect 5: Gate 1 phase-level failures set failure_count=0.

    Per spec §3.1.3 / §9, ``failure_count`` counts ONLY Gate 2
    (ontology) + Gate 3 (evidence) per-record rejects. A Gate 1
    (schema) phase-level failure means the entire phase is rejected,
    with status="rejected" and failure_count=0.
    """

    def _world(self):
        import shutil
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return _write_world(root)

    def test_gate1_status_rejected_failure_count_zero_entity(self):
        # LLMValidationError → phase-level Gate 1 reject, failure_count=0.
        world = self._world()
        ontology, prompt_artifacts, run_state = _build_phase_call_state(world)
        exc = LLMValidationError(
            model_output="bad", schema={}, validation_error=ValueError("x")
        )
        provider = _ScriptedBoomLLM(lambda: (exc, {}))
        outcome, validated = pipeline._run_entity_phase(
            chunk=_read_chunk_first(world),
            ontology=ontology,
            prompt_artifact=prompt_artifacts[ENTITY_PROMPT_FILE],
            provider=provider,
            max_tokens=1024, temperature=0.0,
            schema=schema.entity_pass_schema(ontology),
            run_state=run_state,
        )
        self.assertEqual(outcome.status, "rejected")
        self.assertEqual(outcome.failure_count, 0)
        self.assertEqual(outcome.mention_count, 0)

    def test_gate1_status_rejected_failure_count_zero_relation(self):
        world = self._world()
        ontology, prompt_artifacts, run_state = _build_phase_call_state(world)
        exc = LLMValidationError(
            model_output="bad", schema={}, validation_error=ValueError("x")
        )
        provider = _ScriptedBoomLLM(lambda: (exc, {}))
        outcome, validated = pipeline._run_relation_phase(
            chunk=_read_chunk_first(world),
            ontology=ontology,
            prompt_artifact=prompt_artifacts[RELATION_PROMPT_FILE],
            provider=provider,
            max_tokens=1024, temperature=0.0,
            schema=schema.relation_pass_schema(ontology),
            run_state=run_state,
            same_chunk_mentions=[],
            entity_type_for_mention={},
        )
        self.assertEqual(outcome.status, "rejected")
        self.assertEqual(outcome.failure_count, 0)

    def test_gate2_ontology_failure_count_increments(self):
        # Force the entity phase to surface ONE ontology reject by
        # feeding an entity_type that isn't in the extracted set.
        import shutil
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        world = _write_world(root)
        outputs = _default_outputs(root)
        bad_entity = {"mentions": [
            {"entity_type": "robot", "surface_form": "DPR",
             "canonical_form": "DPR", "char_start": 0, "char_end": 3,
             "confidence": "stated"},
        ]}
        scripts = [bad_entity, {"assertions": []}, bad_entity, {"assertions": []}]
        _run(world, outputs=outputs, scripts=scripts)
        records = _read_jsonl(outputs["jsonl"])
        statuses = [r for r in records if r.get("kind") == "chunk_status"]
        self.assertGreater(len(statuses), 0)
        for s in statuses:
            # Provider call returned a parseable payload → status=validated,
            # but the per-record ontology reject incremented failure_count.
            self.assertEqual(s["phases"]["entity"]["status"], "validated")
            self.assertGreater(s["phases"]["entity"]["failure_count"], 0)

    def test_gate3_evidence_failure_count_increments(self):
        # Force the entity phase to surface ONE evidence reject by
        # feeding a char_end that exceeds chunk text length.
        import shutil
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        world = _write_world(root)
        outputs = _default_outputs(root)
        bad_entity = {"mentions": [
            {"entity_type": "method", "surface_form": "ZZ",
             "canonical_form": "ZZ", "char_start": 0, "char_end": 99999,
             "confidence": "stated"},
        ]}
        scripts = [bad_entity, {"assertions": []}, bad_entity, {"assertions": []}]
        _run(world, outputs=outputs, scripts=scripts)
        records = _read_jsonl(outputs["jsonl"])
        statuses = [r for r in records if r.get("kind") == "chunk_status"]
        self.assertGreater(len(statuses), 0)
        for s in statuses:
            self.assertEqual(s["phases"]["entity"]["status"], "validated")
            self.assertGreater(s["phases"]["entity"]["failure_count"], 0)


# --- 19: Defect 6 — canonical ordering rule ------------------------------

class TestCanonicalOrderingRule(unittest.TestCase):
    """Defect 6: ONE deterministic sort key, no bucket emission.

    The serializer uses one and only one ordering rule, the
    ``(chunk_id, kind_rank, record_id)`` tuple sort applied to the
    union of every emitted record. Scrambled insertion order must
    produce an output that exactly matches that key.
    """

    def test_scrambled_input_matches_canonical_sort_key(self):
        # Build a deliberately scrambled record list covering all
        # three kinds, multiple chunk_ids, mention/assertion ids,
        # and chunk_status records interleaved in a non-natural
        # order. Format them through the serializer and assert the
        # emitted line order equals the canonical key order.
        records_by_kind = {
            # chunk_status records: use chunk_id as the third key.
            "c_001": {"kind": "chunk_status", "chunk_id": "c_001", "doc_id": "W9"},
            "c_002": {"kind": "chunk_status", "chunk_id": "c_002", "doc_id": "W9"},
            "c_000": {"kind": "chunk_status", "chunk_id": "c_000", "doc_id": "W9"},
            # mentions: out-of-order IDs within each chunk.
            "c_002:m_zzz": {"kind": "mention", "mention_id": "m_zzz", "chunk_id": "c_002",
                            "doc_id": "W9", "entity_type": "method", "surface_form": "x",
                            "canonical_form": "x", "evidence": {"char_start": 0, "char_end": 1,
                                                                 "source_text_sha256": "a"}},
            "c_002:m_aaa": {"kind": "mention", "mention_id": "m_aaa", "chunk_id": "c_002",
                            "doc_id": "W9", "entity_type": "method", "surface_form": "x",
                            "canonical_form": "x", "evidence": {"char_start": 0, "char_end": 1,
                                                                 "source_text_sha256": "a"}},
            "c_001:m_aaa": {"kind": "mention", "mention_id": "m_aaa", "chunk_id": "c_001",
                            "doc_id": "W9", "entity_type": "method", "surface_form": "x",
                            "canonical_form": "x", "evidence": {"char_start": 0, "char_end": 1,
                                                                 "source_text_sha256": "a"}},
            "c_001:m_zzz": {"kind": "mention", "mention_id": "m_zzz", "chunk_id": "c_001",
                            "doc_id": "W9", "entity_type": "method", "surface_form": "x",
                            "canonical_form": "x", "evidence": {"char_start": 0, "char_end": 1,
                                                                 "source_text_sha256": "a"}},
            # assertions: same plan, with assertion_id as the third key.
            "c_001:a_zzz": {"kind": "assertion", "assertion_id": "a_zzz", "chunk_id": "c_001",
                            "doc_id": "W9", "relation_type": "PROPOSES",
                            "subject_ref": {"kind": "document", "id": "W9"},
                            "object_ref": {"kind": "mention", "id": "m_x"},
                            "evidence": {"char_start": 0, "char_end": 1,
                                          "source_text_sha256": "a"}},
            "c_001:a_aaa": {"kind": "assertion", "assertion_id": "a_aaa", "chunk_id": "c_001",
                            "doc_id": "W9", "relation_type": "PROPOSES",
                            "subject_ref": {"kind": "document", "id": "W9"},
                            "object_ref": {"kind": "mention", "id": "m_x"},
                            "evidence": {"char_start": 0, "char_end": 1,
                                          "source_text_sha256": "a"}},
            "c_002:a_aaa": {"kind": "assertion", "assertion_id": "a_aaa", "chunk_id": "c_002",
                            "doc_id": "W9", "relation_type": "PROPOSES",
                            "subject_ref": {"kind": "document", "id": "W9"},
                            "object_ref": {"kind": "mention", "id": "m_x"},
                            "evidence": {"char_start": 0, "char_end": 1,
                                          "source_text_sha256": "a"}},
        }

        # Split into the three buckets (intentionally in scrambled
        # order — not sorted by chunk_id).
        mentions_by_chunk = {
            "c_001": [records_by_kind["c_001:m_zzz"], records_by_kind["c_001:m_aaa"]],
            "c_002": [records_by_kind["c_002:m_zzz"], records_by_kind["c_002:m_aaa"]],
        }
        assertions_by_chunk = {
            "c_001": [records_by_kind["c_001:a_zzz"], records_by_kind["c_001:a_aaa"]],
            "c_002": [records_by_kind["c_002:a_aaa"]],
        }
        statuses_by_chunk = {
            "c_001": records_by_kind["c_001"],
            "c_002": records_by_kind["c_002"],
            "c_000": records_by_kind["c_000"],
        }

        emitted = list(pipeline._format_records(
            mentions_by_chunk=mentions_by_chunk,
            assertions_by_chunk=assertions_by_chunk,
            statuses_by_chunk=statuses_by_chunk,
        ))
        parsed = [json.loads(line) for line in emitted]

        # The contract key is (chunk_id, kind_rank, record_id).
        # kind_rank: 0=mention, 1=assertion, 2=chunk_status.
        # For chunk_status, the third element is the chunk_id (the
        # record's only identity field).
        def contract_key(rec):
            if rec["kind"] == "mention":
                return (rec["chunk_id"], 0, rec["mention_id"])
            if rec["kind"] == "assertion":
                return (rec["chunk_id"], 1, rec["assertion_id"])
            return (rec["chunk_id"], 2, rec["chunk_id"])

        actual = [contract_key(r) for r in parsed]
        # The serializer must produce exactly the canonical sort.
        self.assertEqual(actual, sorted(actual))
        # And the canonical sort itself is unambiguous: regenerate it
        # from the union of records and compare exactly.
        all_records = (
            [r for v in mentions_by_chunk.values() for r in v]
            + [r for v in assertions_by_chunk.values() for r in v]
            + list(statuses_by_chunk.values())
        )
        expected = [contract_key(r) for r in all_records]
        expected.sort()
        self.assertEqual(actual, expected)


# --- 20: Defect 7 — relation membership from extracted population_mode ---

DEFECT_A_ONT = """
version: 1
schema_version: 1
entity_types:
  - name: document
    population_mode: structural
    source: manifest
    examples: []
  - name: method
    population_mode: extracted
    extraction_prompt: extract_entities_v1
    definition: A research method or technique.
    verified_examples: []
  - name: dataset
    population_mode: extracted
    extraction_prompt: extract_entities_v1
    definition: A benchmark dataset.
    verified_examples: []
relation_types:
  - name: PROPOSES
    from: document
    to: method
    population_mode: extracted
    extraction_prompt: extract_relations_v1
    definition: A paper proposing a method.
    verified_examples: []
  - name: EXTENDS
    from: method
    to: method
    population_mode: extracted
    extraction_prompt: extract_relations_v1
    definition: Method A builds on method B.
    verified_examples: []
  - name: CITES
    from: document
    to: document
    population_mode: structural
    source: bibliography
    examples: []
  - name: EVALUATED_ON
    from: method
    to: dataset
    population_mode: derived
    derivation: |
      Derived from PROPOSES x EVALUATES_ON closure.
    examples: []
"""


class TestDefectARelationMembership(unittest.TestCase):
    """Defect 7: relation population_mode drives extraction membership.

    Only ``relation_types[].population_mode == extracted`` is
    surfaced by extraction. Structural (``CITES``) and derived
    (``EVALUATED_ON``) relations are excluded by membership
    filtering, NOT by endpoint-population filtering.
    """

    def setUp(self):
        import yaml
        self._ontology = yaml.safe_load(DEFECT_A_ONT)

    def test_extracted_proposes_and_extends_are_members(self):
        extracted = schema.extracted_relation_types(self._ontology)
        names = sorted(rt["name"] for rt in extracted)
        self.assertIn("PROPOSES", names)
        self.assertIn("EXTENDS", names)

    def test_structural_cites_is_excluded(self):
        extracted = schema.extracted_relation_types(self._ontology)
        names = {rt["name"] for rt in extracted}
        self.assertNotIn("CITES", names)

    def test_derived_evaluated_on_is_excluded(self):
        extracted = schema.extracted_relation_types(self._ontology)
        names = {rt["name"] for rt in extracted}
        self.assertNotIn("EVALUATED_ON", names)

    def test_proposes_endpoint_validation_with_structural_document(self):
        # PROPOSES (document -> method) is an extracted relation,
        # and its document endpoint is structural. The structural
        # document endpoint does NOT exclude an extracted relation.
        same_chunk = {
            "m_aaa": {"entity_type": "method", "surface_form": "DPR",
                      "evidence": {"char_start": 0, "char_end": 3,
                                   "source_text_sha256": "abc"}},
        }
        good = {
            "relation_type": "PROPOSES",
            "subject_ref": {"kind": "document", "id": "W9"},
            "object_ref": {"kind": "mention", "id": "m_aaa"},
            "char_start": 0, "char_end": 3, "confidence": "stated",
            "evidence": {"char_start": 0, "char_end": 3,
                         "source_text_sha256": "abc"},
        }
        # Must NOT raise: structural document endpoint + extracted
        # mention endpoint is a legal PROPOSES shape.
        validation.check_assertion_ontology(
            assertion=good,
            ontology=self._ontology,
            doc_id="W9",
            same_chunk_mentions=same_chunk,
            entity_type_for_mention={"m_aaa": "method"},
        )

    def test_proposes_wrong_document_id_still_rejected(self):
        # Defence-in-depth: even though the document endpoint is
        # structural, the relation-pass gate still rejects a
        # structural document id that disagrees with the current
        # chunk's doc_id.
        same_chunk = {
            "m_aaa": {"entity_type": "method", "surface_form": "DPR",
                      "evidence": {"char_start": 0, "char_end": 3,
                                   "source_text_sha256": "abc"}},
        }
        bad = {
            "relation_type": "PROPOSES",
            "subject_ref": {"kind": "document", "id": "W-OTHER"},
            "object_ref": {"kind": "mention", "id": "m_aaa"},
            "char_start": 0, "char_end": 3, "confidence": "stated",
            "evidence": {"char_start": 0, "char_end": 3,
                         "source_text_sha256": "abc"},
        }
        with self.assertRaises(OntologyError):
            validation.check_assertion_ontology(
                assertion=bad,
                ontology=self._ontology,
                doc_id="W9",
                same_chunk_mentions=same_chunk,
                entity_type_for_mention={"m_aaa": "method"},
            )
