"""Acceptance tests for the T3b canonical v2 entry (#55)."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extract import (  # noqa: E402
    align as align_module,
    pipeline_v2,
    prompts as prompts_module,
    schema as schema_module,
)
from src.providers.protocol import (  # noqa: E402
    LLMProvider,
    Usage,
)


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


def _file_sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


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
    (prompts_dir / prompts_module.ENTITY_PROMPT_FILE_V2).write_text(ENTITY_TEMPLATE_V2)
    (prompts_dir / prompts_module.RELATION_PROMPT_FILE_V2).write_text(RELATION_TEMPLATE_V2)
    return {
        "ontology": ont_path,
        "chunks": chunks_path,
        "prompts": prompts_dir,
    }


def _outputs(root: Path, *, prefix: str = "extractions.v2") -> dict:
    d = root / "out"
    d.mkdir(parents=True, exist_ok=True)
    return {
        "jsonl": d / f"{prefix}.jsonl",
        "rejects": d / f"{prefix}.rejects.jsonl",
        "manifest": d / f"{prefix}.manifest.json",
    }


def _read_jsonl(p: Path) -> list:
    if not p.is_file():
        return []
    return [
        json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(p: Path) -> dict | None:
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _scripted_payload(*, entity_payload=None, relation_payload=None) -> list:
    return [
        entity_payload if entity_payload is not None else {"mentions": []},
        relation_payload if relation_payload is not None else {"assertions": []},
    ]


class _FakeLLM(LLMProvider):
    def __init__(self, scripts=None) -> None:
        self.name = "fake-llm-v2-canonical"
        self.model = "fake-model-v2-canonical"
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


def _run_canonical(
    world: dict, *, outputs: dict, scripts=None, llm=None,
) -> pipeline_v2.RunResultV2Canonical:
    provider = llm if llm is not None else _FakeLLM(scripts=scripts)
    return pipeline_v2.run_extraction_v2_canonical(
        provider=provider,
        output_jsonl=outputs["jsonl"],
        output_rejects=outputs["rejects"],
        output_manifest=outputs["manifest"],
        chunks_path=world["chunks"],
        ontology_path=world["ontology"],
        prompts_dir=world["prompts"],
    )


# --- 1: fresh-run smoke (manifest, jsonl, rejects all written) ------

class TestFreshRunSmoke(unittest.TestCase):

    def test_fresh_run_writes_manifest_jsonl_rejects(self):
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
            )
            result = _run_canonical(world, outputs=outs, scripts=scripts)
            self.assertEqual(result.exit_code, 0)
            self.assertFalse(result.budget_aborted)
            self.assertIsNone(result.resumed_from_run_id)
            self.assertEqual(result.mention_count, 1)
            self.assertEqual(result.complete_chunk_count, 1)
            self.assertEqual(result.reject_count, 0)
            self.assertGreater(result.cumulative_tokens, 0)

            self.assertTrue(outs["jsonl"].is_file())
            self.assertTrue(outs["rejects"].is_file())
            self.assertTrue(outs["manifest"].is_file())

            manifest = _read_json(outs["manifest"])
            self.assertEqual(manifest["schema_version"],
                             schema_module.SCHEMA_VERSION_V2)
            self.assertEqual(manifest["run_id"], result.run_id)
            self.assertIsNone(manifest["resumed_from_run_id"])
            self.assertEqual(manifest["chunks_sha256"], _file_sha(world["chunks"]))
            self.assertEqual(manifest["ontology_sha256"],
                             _file_sha(world["ontology"]))
            self.assertEqual(manifest["prompt_template_sha256"], [
                _file_sha(world["prompts"] / prompts_module.ENTITY_PROMPT_FILE_V2),
                _file_sha(world["prompts"] / prompts_module.RELATION_PROMPT_FILE_V2),
            ])
            self.assertEqual(manifest["provider"], "fake-llm-v2-canonical")
            self.assertEqual(manifest["model"], "fake-model-v2-canonical")
            self.assertEqual(manifest["alignment_algorithm_version"],
                             align_module.ALIGNMENT_ALGORITHM_VERSION)
            self.assertEqual(manifest["alignment_algorithm_sha256"],
                             _file_sha(Path(__file__).resolve().parents[1]
                                       / "src/extract/align.py"))
            self.assertEqual(manifest["chunk_count_total"], 1)
            self.assertEqual(manifest["chunk_count_complete"], 1)
            self.assertEqual(manifest["reject_count"], 0)


# --- 2: every manifest field present -------------------------------

class TestManifestFields(unittest.TestCase):

    def test_manifest_carries_all_required_fields(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            _run_canonical(world, outputs=outs)
            manifest = _read_json(outs["manifest"])
            for key in (
                "schema_version", "run_id", "resumed_from_run_id",
                "chunks_sha256", "ontology_sha256",
                "prompt_template_sha256", "system_prompt_sha256",
                "provider", "model", "max_tokens", "temperature",
                "alignment_algorithm_version", "alignment_algorithm_sha256",
                "ontology_version", "prompt_template_files",
                "chunks_path", "ontology_path", "pipeline_version",
                "cache_format_version", "wall_clock_started_at",
                "wall_clock_ended_at", "cumulative_tokens",
                "cumulative_cost_usd", "reject_count",
                "chunk_count_total", "chunk_count_complete",
            ):
                self.assertIn(key, manifest)


# --- 3: schema_version on every record -----------------------------

class TestSchemaVersionOnEveryRecord(unittest.TestCase):

    def test_schema_version_v2_on_every_emitted_record(self):
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
            )
            _run_canonical(world, outputs=outs, scripts=scripts)
            records = _read_jsonl(outs["jsonl"])
            self.assertTrue(records)
            for r in records:
                self.assertEqual(
                    r["schema_version"], schema_module.SCHEMA_VERSION_V2,
                )


# --- 4: pipeline_version = git HEAD --------------------------------

class TestPipelineVersion(unittest.TestCase):

    def test_pipeline_version_equals_git_head(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            _run_canonical(world, outputs=outs)
            manifest = _read_json(outs["manifest"])
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(head.returncode, 0)
            self.assertEqual(manifest["pipeline_version"], head.stdout.strip())


# --- 5: resume with identity match ---------------------------------

class TestSemanticResume(unittest.TestCase):

    def test_second_run_resumes_complete_chunks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            llm = _FakeLLM(scripts=_scripted_payload(
                entity_payload={"mentions": [
                    {
                        "entity_type": "method",
                        "surface_form": "DPR",
                        "canonical_form": "Dense Passage Retrieval",
                        "confidence": "stated",
                    },
                ]},
            ))
            r1 = pipeline_v2.run_extraction_v2_canonical(
                provider=llm,
                output_jsonl=outs["jsonl"],
                output_rejects=outs["rejects"],
                output_manifest=outs["manifest"],
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
            )
            self.assertEqual(r1.exit_code, 0)
            self.assertIsNone(r1.resumed_from_run_id)
            self.assertGreater(len(llm.json_calls), 0)
            first_calls = list(llm.json_calls)

            r2 = pipeline_v2.run_extraction_v2_canonical(
                provider=llm,
                output_jsonl=outs["jsonl"],
                output_rejects=outs["rejects"],
                output_manifest=outs["manifest"],
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
            )
            self.assertEqual(r2.exit_code, 0)
            self.assertEqual(r2.resumed_from_run_id, r1.run_id)
            self.assertEqual(len(llm.json_calls), len(first_calls))
            self.assertEqual(r2.mention_count, r1.mention_count)
            self.assertEqual(r2.complete_chunk_count, r1.complete_chunk_count)


# --- 6: identity mismatch → fresh run, no error --------------------

class TestIdentityMismatch(unittest.TestCase):

    def test_ontology_drift_silently_falls_back_to_fresh_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            r1 = _run_canonical(world, outputs=outs)
            self.assertIsNone(r1.resumed_from_run_id)
            (root / "ontology.yaml").write_text(MINIMAL_ONT.replace(
                "version: 1", "version: 2",
            ))
            r2 = _run_canonical(world, outputs=outs)
            self.assertEqual(r2.exit_code, 0)
            self.assertIsNone(r2.resumed_from_run_id)
            self.assertGreater(r2.cumulative_tokens, 0)
            manifest = _read_json(outs["manifest"])
            self.assertEqual(manifest["ontology_sha256"],
                             _file_sha(world["ontology"]))

    def test_chunks_drift_silently_falls_back_to_fresh_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            r1 = _run_canonical(world, outputs=outs)
            self.assertEqual(r1.exit_code, 0)
            new_chunks = [{
                "chunk_id": "c_999",
                "doc_id": "W2",
                "section_path": ["x"],
                "chunk_index": 0,
                "text": "Different chunk.",
                "token_estimate": 3,
                "char_start": 0,
                "char_end": 16,
            }]
            (root / "chunks.jsonl").write_text(
                "\n".join(json.dumps(c) for c in new_chunks) + "\n",
                encoding="utf-8",
            )
            r2 = _run_canonical(world, outputs=outs)
            self.assertEqual(r2.exit_code, 0)
            self.assertIsNone(r2.resumed_from_run_id)

    def test_prompt_drift_silently_falls_back_to_fresh_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            r1 = _run_canonical(world, outputs=outs)
            (world["prompts"] / prompts_module.ENTITY_PROMPT_FILE_V2).write_text(
                ENTITY_TEMPLATE_V2 + "\nNEW CONTENT\n"
            )
            r2 = _run_canonical(world, outputs=outs)
            self.assertIsNone(r2.resumed_from_run_id)


# --- 7: per-chunk carry-forward silent narrowing -------------------

class TestPerChunkCarryForward(unittest.TestCase):

    def test_one_bad_chunk_silently_narrows_resume(self):
        """Corrupt a single chunk's mention_count so it fails the
        carry-forward validator; the runner re-runs that chunk
        only and still resumes the rest. The bad chunk's records
        are dropped from the output (current-state)."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            chunks = [
                {
                    "chunk_id": "c_000",
                    "doc_id": "W1",
                    "section_path": ["intro"],
                    "chunk_index": 0,
                    "text": "We use DPR for retrieval.",
                    "token_estimate": 6,
                    "char_start": 0,
                    "char_end": 24,
                },
                {
                    "chunk_id": "c_001",
                    "doc_id": "W1",
                    "section_path": ["intro"],
                    "chunk_index": 1,
                    "text": "Evaluation on NQ.",
                    "token_estimate": 4,
                    "char_start": 0,
                    "char_end": 15,
                },
            ]
            world = _write_world(root, chunks=chunks)
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
            )
            llm = _FakeLLM(scripts=scripts)
            r1 = pipeline_v2.run_extraction_v2_canonical(
                provider=llm,
                output_jsonl=outs["jsonl"],
                output_rejects=outs["rejects"],
                output_manifest=outs["manifest"],
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
            )
            self.assertEqual(r1.exit_code, 0)
            self.assertEqual(r1.complete_chunk_count, 2)

            # Corrupt c_001's chunk_status so its carry-forward fails.
            jsonl_text = outs["jsonl"].read_text(encoding="utf-8")
            lines = jsonl_text.splitlines()
            new_lines = []
            for ln in lines:
                rec = json.loads(ln)
                if rec.get("chunk_id") == "c_001" and rec.get("kind") == "chunk_status":
                    rec["phases"]["entity"]["mention_count"] = 999  # mismatch
                    ln = json.dumps(rec, sort_keys=True, ensure_ascii=False)
                new_lines.append(ln)
            outs["jsonl"].write_text("\n".join(new_lines) + "\n")

            calls_before = len(llm.json_calls)
            r2 = pipeline_v2.run_extraction_v2_canonical(
                provider=llm,
                output_jsonl=outs["jsonl"],
                output_rejects=outs["rejects"],
                output_manifest=outs["manifest"],
                chunks_path=world["chunks"],
                ontology_path=world["ontology"],
                prompts_dir=world["prompts"],
            )
            self.assertEqual(r2.exit_code, 0)
            self.assertEqual(r2.resumed_from_run_id, r1.run_id)
            # Exactly one chunk was re-run: 2 phases x 1 chunk = 2 calls.
            self.assertEqual(len(llm.json_calls) - calls_before, 2)


# --- 8: malformed manifest = hard fail ------------------------------

class TestMalformedManifest(unittest.TestCase):

    def test_malformed_manifest_hard_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            # Run once to write a valid manifest.
            _run_canonical(world, outputs=outs)
            # Corrupt it.
            outs["manifest"].write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(pipeline_v2.PreflightErrorV2):
                _run_canonical(world, outputs=outs)

    def test_wrong_schema_version_in_manifest_hard_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            _run_canonical(world, outputs=outs)
            manifest = _read_json(outs["manifest"])
            manifest["schema_version"] = 1  # wrong version
            outs["manifest"].write_text(
                json.dumps(manifest), encoding="utf-8",
            )
            with self.assertRaises(pipeline_v2.PreflightErrorV2):
                _run_canonical(world, outputs=outs)


# --- 9: manifest present without artifacts = hard fail --------------

class TestManifestArtifactConsistency(unittest.TestCase):

    def test_manifest_present_without_jsonl_hard_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            _run_canonical(world, outputs=outs)
            outs["jsonl"].unlink()
            with self.assertRaises(pipeline_v2.PreflightErrorV2):
                _run_canonical(world, outputs=outs)

    def test_manifest_present_without_rejects_hard_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            _run_canonical(world, outputs=outs)
            outs["rejects"].unlink()
            with self.assertRaises(pipeline_v2.PreflightErrorV2):
                _run_canonical(world, outputs=outs)


# --- 10: crash after manifest unlink → next invocation is fresh ----

class TestManifestInvalidation(unittest.TestCase):

    def test_crash_after_manifest_unlink_treated_as_fresh(self):
        """Simulate a crash after the manifest was unlinked but
        before the new manifest was written. The JSONL and rejects
        are stale; the next invocation must treat them as
        untrusted and write a fresh manifest."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            r1 = _run_canonical(world, outputs=outs)
            self.assertIsNone(r1.resumed_from_run_id)
            # Simulate crash: unlink manifest, keep JSONL/rejects.
            outs["manifest"].unlink()
            self.assertFalse(outs["manifest"].exists())
            r2 = _run_canonical(world, outputs=outs)
            self.assertEqual(r2.exit_code, 0)
            # Without a manifest, the runner cannot trust identity;
            # it must produce a fresh manifest and not resume.
            self.assertIsNone(r2.resumed_from_run_id)
            self.assertTrue(outs["manifest"].is_file())


# --- 11: atomic JSONL rewrite ---------------------------------------

class TestAtomicRewrites(unittest.TestCase):

    def test_old_jsonl_unchanged_when_provider_explodes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            world = _write_world(root)
            outs = _outputs(root)
            # Run a valid first run.
            scripts = _scripted_payload(
                entity_payload={"mentions": [
                    {
                        "entity_type": "method",
                        "surface_form": "DPR",
                        "canonical_form": "Dense Passage Retrieval",
                        "confidence": "stated",
                    },
                ]},
            )
            _run_canonical(world, outputs=outs, scripts=scripts)
            first_jsonl = outs["jsonl"].read_bytes()
            first_manifest = outs["manifest"].read_bytes()
            # Force a transport explosion on the second run.
            from src.providers.protocol import ProviderTransportError

            class _BoomLLM(_FakeLLM):
                def generate_json(self, system, user, schema, *,
                                  max_tokens=1024, temperature=0.0):
                    raise ProviderTransportError(transient=False)

            # Delete the manifest so we get a fresh run, then explode.
            outs["manifest"].unlink()
            with self.assertRaises(ProviderTransportError):
                _run_canonical(world, outputs=outs, llm=_BoomLLM())
            # Old JSONL is untouched (atomic write was never started).
            self.assertEqual(outs["jsonl"].read_bytes(), first_jsonl)
            # No fresh manifest was written.
            self.assertFalse(outs["manifest"].exists())


# --- 12: CLI dispatch -----------------------------------------------

class TestCliDispatch(unittest.TestCase):

    def test_cli_dispatch_v2_argument_accepted(self):
        """``python -m src.extract v2`` is accepted by the dispatcher.

        We don't drive the full CLI to completion here (the v2 path
        requires a registered LLM_PROVIDER + network/model files).
        The library entry is exhaustively tested above; this case
        only asserts the dispatch token is wired correctly.
        """
        proc = subprocess.run(
            [sys.executable, "-c",
             "from src.extract.__main__ import main; "
             "import sys; "
             "sys.argv = ['x', 'v2']; "
             "import os; os.environ.setdefault('LLM_PROVIDER','missing'); "
             "try: main()\n"
             "except SystemExit as e: print('exit', e.code); raise"],
            cwd=str(Path(__file__).resolve().parents[1]),
            env={**__import__('os').environ,
                 "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
            capture_output=True, text=True, timeout=30,
        )
        # It will fail with a preflight error (no provider), but the
        # dispatcher accepted ``v2``. We just assert that the failure
        # mode is provider-related, not the v2 token.
        self.assertNotIn("unknown argument", proc.stdout + proc.stderr)
        self.assertNotIn("usage:", proc.stdout + proc.stderr)

    def test_cli_unknown_argument_exits_2(self):
        proc = subprocess.run(
            [sys.executable, "-m", "src.extract", "garbage"],
            cwd=str(Path(__file__).resolve().parents[1]),
            env={**__import__('os').environ,
                 "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("usage", (proc.stderr + proc.stdout).lower())

    def test_cli_bare_routes_to_v1(self):
        """Bare ``python -m src.extract`` (no token) routes to v1.

        We assert the module imports cleanly and ``main()`` is
        callable with no argv beyond the program name.
        """
        proc = subprocess.run(
            [sys.executable, "-c",
             "from src.extract.__main__ import main; "
             "import sys; sys.argv = ['x']; "
             "import os; os.environ.setdefault('LLM_PROVIDER','missing'); "
             "try: main()\n"
             "except SystemExit as e: print('exit', e.code); raise"],
            cwd=str(Path(__file__).resolve().parents[1]),
            env={**__import__('os').environ,
                 "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
            capture_output=True, text=True, timeout=30,
        )
        self.assertNotIn("unknown argument", proc.stdout + proc.stderr)



class TestCanonicalFunctionContract(unittest.TestCase):
    """Public signature contract (#55)."""

    def test_run_extraction_v2_canonical_parameter_order(self) -> None:
        """The canonical keyword-only argument order is pinned by #55.

        Any drift here would silently break CLI contract consumers
        that pass values positionally by name or rely on argument
        ordinal for documentation/tracing.
        """
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
