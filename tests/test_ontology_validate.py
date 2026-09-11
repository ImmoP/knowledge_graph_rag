"""Tests for ``src.ontology.validate`` (#20).

Uses stdlib ``unittest`` only; no pytest dependency. The validator is
generic over the supplied YAML — every fixture builds its own
``ontology.yaml`` and corpus surface under a temp dir so the tests do
not need the canonical ``data/`` tree to exist. The repo's committed
``ontology.yaml`` is exercised in :class:`TestCommittedOntology`.
"""

from __future__ import annotations

import ast
import json
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ontology import validate as v  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]


# --- helpers ----------------------------------------------------------


def _write_minimal_ontology(
    path: Path,
    *,
    entity_names=None,
    relation_names=None,
    entity_overrides=None,
    relation_overrides=None,
) -> Path:
    """Write a minimal valid v1-shape ontology YAML to ``path``.

    ``entity_overrides``/``relation_overrides`` mutate generated record
    dicts by their ``name``. The output is intentionally lenient
    (every example is ``pending_verification``) so a generic-shape
    fixture passes; tests that need a different shape override
    examples by writing the whole ontology inline.
    """
    ent_names = entity_names or [
        "document", "chunk", "method", "dataset", "metric", "task",
    ]
    rel_names = relation_names or [
        "CITES", "PROPOSES", "INTRODUCES", "EVALUATES_ON", "USES_METHOD",
        "REPORTS_METRIC", "EXTENDS", "EVALUATED_ON", "USED_FOR_TASK", "MEASURES",
    ]
    lines: list = ["version: 1", "schema_version: 1", "", "entity_types:"]
    overrides_e = entity_overrides or {}
    overrides_r = relation_overrides or {}
    for n in ent_names:
        spec = overrides_e.get(n, {})
        mode = spec.get("population_mode", "structural")
        rec = [
            f"  - name: {n}",
            f"    definition: definition-for-{n}",
            f"    population_mode: {mode}",
        ]
        if mode == "structural":
            rec.append(f"    source: data/manifest.jsonl (synthetic).")
        elif mode == "extracted":
            rec.append(f"    extraction_prompt: extract_entities_v1")
        else:
            rec.append(f"    derivation: synthetic derivation for {n}.")
        rec.append("    examples:")
        rec.append("      - status: pending_verification")
        rec.append(f"        instance: {n}-placeholder")
        rec.append("        note: synthetic placeholder for fixture.")
        lines.extend(rec)
        lines.append("")
    lines.append("relation_types:")
    for n in rel_names:
        spec = overrides_r.get(n, {})
        mode = spec.get("population_mode", "extracted")
        from_e = spec.get("from", ent_names[0])
        to_e = spec.get("to", ent_names[2])
        rec = [
            f"  - name: {n}",
            f"    from: {from_e}",
            f"    to: {to_e}",
            f"    definition: definition-for-{n}",
            f"    population_mode: {mode}",
        ]
        if mode == "structural":
            rec.append(f"    source: data/manifest.jsonl (synthetic).")
        elif mode == "extracted":
            rec.append(f"    extraction_prompt: extract_relations_v1")
        else:
            rec.append(f"    derivation: synthetic closure for {n}.")
        rec.append("    examples:")
        rec.append("      - status: pending_verification")
        rec.append(f"        from: {from_e}")
        rec.append(f"        to: {to_e}")
        rec.append(f"        note: synthetic placeholder for {n}.")
        lines.extend(rec)
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def _write_manifest(path: Path, ids=("W100", "W200")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for oid in ids:
            f.write(json.dumps({"openalex_id": oid, "title": f"title-{oid}"}) + "\n")
    return path


def _write_openalex_dir(path: Path, papers=()) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for oid, title, abstract_tokens in papers:
        aii: dict = {}
        for idx, token in enumerate(abstract_tokens):
            aii.setdefault(token, []).append(idx)
        meta = {
            "id": f"https://openalex.org/{oid}",
            "title": title,
            "display_name": title,
            "abstract_inverted_index": aii,
        }
        (path / f"{oid}.json").write_text(json.dumps(meta))
    return path


def _build_minimal_corpus(tmp: Path, *, manifest_ids=("W100", "W200"),
                          papers=()):
    mp = tmp / "manifest.jsonl"
    _write_manifest(mp, ids=manifest_ids)
    oa = tmp / "openalex"
    _write_openalex_dir(oa, papers=papers)
    return mp, oa


# --- committed-ontology + prompt-file-existence tests ----------------


class TestCommittedOntology(unittest.TestCase):
    """The committed ``ontology.yaml`` must validate cleanly."""

    def test_committed_ontology_passes(self):
        report = v.validate_ontology()
        self.assertTrue(
            report.ok,
            msg=(
                "committed ontology has violations: "
                + str([x.format() for x in report.violations])
            ),
        )
        self.assertEqual(report.version, 1)
        self.assertEqual(report.schema_version, 1)
        self.assertEqual(report.yaml_sha256,
            "f9a3e203d4c640b91966e077b02f09a547d49b51595cf33e040a3e278cd2325e",
        )
        self.assertEqual(v.ONTOLOGY_PATH, REPO_ROOT / "ontology.yaml")


class TestPromptFileExistence(unittest.TestCase):
    """#20 has no dependency on #23 — physical prompt files need not exist."""

    def test_validator_passes_without_prompt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            onp = tmp / "ontology.yaml"
            _write_minimal_ontology(onp)
            mp, oa = _build_minimal_corpus(tmp)
            self.assertFalse((tmp / "prompts").exists())
            report = v.validate_ontology(
                onp, manifest_path=mp, openalex_dir=oa
            )
            self.assertTrue(
                report.ok,
                msg=f"unexpected violations: "
                    f"{[x.format() for x in report.violations]}",
            )


# --- shape-violation fixtures ----------------------------------------


def _run_with(tmpdir, ontology_text, *, manifest_ids=("W100", "W200"),
              papers=(), on_vocab_bump="warn"):
    tmp = Path(tmpdir)
    onp = tmp / "ontology.yaml"
    onp.write_text(textwrap.dedent(ontology_text).lstrip("\n"))
    mp, oa = _build_minimal_corpus(tmp, manifest_ids=manifest_ids,
                                   papers=papers)
    return v.validate_ontology(
        onp, manifest_path=mp, openalex_dir=oa, on_vocab_bump=on_vocab_bump
    )


class TestTopLevelShape(unittest.TestCase):
    def test_extra_top_level_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _run_with(
                tmp,
                """
                version: 1
                schema_version: 1
                mystery: 1
                entity_types: []
                relation_types: []
                """,
            )
            self.assertFalse(report.ok)
            self.assertIn("top_level_keys",
                          {x.rule for x in report.violations})

    def test_missing_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _run_with(
                tmp,
                """
                schema_version: 1
                entity_types: []
                relation_types: []
                """,
            )
            self.assertFalse(report.ok)
            self.assertIn("version", {x.rule for x in report.violations})

    def test_missing_schema_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _run_with(
                tmp,
                """
                version: 1
                entity_types: []
                relation_types: []
                """,
            )
            self.assertFalse(report.ok)
            self.assertIn("schema_version", {x.rule for x in report.violations})

    def test_size_below_entity_min(self):
        # 4 entities, 8 relations (entity count under floor of 5).
        ent_lines: list = []
        for i in range(4):
            ent_lines.append(f"  - name: e{i}")
            ent_lines.append("    definition: d")
            if i == 0:
                ent_lines.append("    population_mode: structural")
                ent_lines.append("    source: data/manifest.jsonl")
                ent_lines.append("    examples:")
                ent_lines.append("      - status: verified")
                ent_lines.append("        instance: W100")
                ent_lines.append("        note: ok")
            else:
                ent_lines.append("    population_mode: extracted")
                ent_lines.append("    extraction_prompt: extract_entities_v1")
                ent_lines.append("    examples:")
                ent_lines.append("      - status: pending_verification")
                ent_lines.append(f"        instance: p{i}")
                ent_lines.append("        note: n")
        rel_lines: list = []
        for i in range(8):
            rel_lines.append(f"  - name: R{i}")
            rel_lines.append("    from: e0")
            rel_lines.append("    to: e0")
            rel_lines.append("    definition: d")
            rel_lines.append("    population_mode: extracted")
            rel_lines.append("    extraction_prompt: extract_relations_v1")
            rel_lines.append("    examples:")
            rel_lines.append("      - status: pending_verification")
            rel_lines.append("        from: e0")
            rel_lines.append("        to: e0")
            rel_lines.append("        note: n")
        yaml_text = (
            "version: 1\n"
            "schema_version: 1\n"
            "entity_types:\n"
            + "\n".join(ent_lines) + "\n"
            "relation_types:\n"
            + "\n".join(rel_lines) + "\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            (tmpp / "ontology.yaml").write_text(yaml_text)
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "title", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn("1.1", {x.rule for x in report.violations})

    def test_size_above_relation_max(self):
        # 6 entities (in range), 16 relations (above ceiling of 15).
        ent_lines: list = []
        for i in range(6):
            ent_lines.append(f"  - name: e{i}")
            ent_lines.append("    definition: d")
            ent_lines.append("    population_mode: extracted")
            ent_lines.append("    extraction_prompt: extract_entities_v1")
            ent_lines.append("    examples:")
            ent_lines.append("      - status: pending_verification")
            ent_lines.append(f"        instance: p{i}")
            ent_lines.append("        note: n")
        rel_lines: list = []
        for i in range(16):
            rel_lines.append(f"  - name: R{i:02d}")
            rel_lines.append("    from: e0")
            rel_lines.append("    to: e1")
            rel_lines.append("    definition: d")
            rel_lines.append("    population_mode: extracted")
            rel_lines.append("    extraction_prompt: extract_relations_v1")
            rel_lines.append("    examples:")
            rel_lines.append("      - status: pending_verification")
            rel_lines.append("        from: e0")
            rel_lines.append("        to: e1")
            rel_lines.append("        note: n")
        yaml_text = (
            "version: 1\n"
            "schema_version: 1\n"
            "entity_types:\n"
            + "\n".join(ent_lines) + "\n"
            "relation_types:\n"
            + "\n".join(rel_lines) + "\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            (tmpp / "ontology.yaml").write_text(yaml_text)
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "title", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn("1.2", {x.rule for x in report.violations})


class TestEntityDiscipline(unittest.TestCase):
    def test_reserved_name_author(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(
                tmpp / "ontology.yaml",
                entity_names=[
                    "author",
                    "alpha", "beta", "gamma", "delta", "epsilon",
                ],
                entity_overrides={
                    "author": {"population_mode": "extracted"},
                    "alpha": {"population_mode": "structural"},
                },
            )
            mp, oa = _build_minimal_corpus(tmpp, manifest_ids=("W100",),
                                            papers=(("W100", "t", ["t"]),))
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn("1.3", {x.rule for x in report.violations})

    def test_reserved_name_author_case_insensitive(self):
        # §1.3 forbids `author` case-insensitively. Construct an entity
        # whose name is `Author` and confirm the violation surfaces.
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(
                tmpp / "ontology.yaml",
                entity_names=[
                    "Author",  # mixed case; forbidden
                    "alpha", "beta", "gamma", "delta", "epsilon",
                ],
                entity_overrides={
                    "Author": {"population_mode": "extracted"},
                    "alpha": {"population_mode": "structural"},
                },
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn("1.3", {x.rule for x in report.violations})

    def test_junk_drawer_name_concept_finding_claim_benchmark_topic_theme(self):
        for forbidden in ["concept", "finding", "claim", "benchmark", "topic", "theme"]:
            with self.subTest(forbidden=forbidden):
                with tempfile.TemporaryDirectory() as tmp:
                    tmpp = Path(tmp)
                    other = [n for n in [
                        "alpha", "beta", "gamma", "delta", "epsilon", "zeta"
                    ] if n != forbidden]
                    names = [forbidden] + other[:5]
                    _write_minimal_ontology(
                        tmpp / "ontology.yaml",
                        entity_names=names,
                        entity_overrides={
                            names[1]: {"population_mode": "structural"},
                        },
                    )
                    mp, oa = _build_minimal_corpus(
                        tmpp, manifest_ids=("W100",),
                        papers=(("W100", "t", ["t"]),),
                    )
                    report = v.validate_ontology(
                        tmpp / "ontology.yaml",
                        manifest_path=mp, openalex_dir=oa,
                    )
                    self.assertFalse(report.ok, msg=f"{forbidden}: expected violation")
                    self.assertIn(
                        "1.3", {x.rule for x in report.violations},
                        msg=f"{forbidden}: expected 1.3 violation in {[x.rule for x in report.violations]}",
                    )

    def test_bad_entity_name_regex(self):
        # Uppercase entity name — fails ^[a-z][a-z0-9_]*$.
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(
                tmpp / "ontology.yaml",
                entity_names=[
                    "Document",
                    "alpha", "beta", "gamma", "delta", "epsilon",
                ],
                entity_overrides={
                    "Document": {"population_mode": "structural"},
                },
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn(
                "entity_name",
                {x.rule for x in report.violations},
            )

    def test_bad_relation_name_regex(self):
        # Lowercase relation name — fails ^[A-Z][A-Z0-9_]*$.
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(
                tmpp / "ontology.yaml",
                relation_names=[
                    "cites",  # bad regex
                    "PROPOSES", "INTRODUCES", "EVALUATES_ON",
                    "USES_METHOD", "REPORTS_METRIC", "EXTENDS",
                    "EVALUATED_ON", "USED_FOR_TASK", "MEASURES",
                ],
                relation_overrides={"cites": {"population_mode": "extracted"}},
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn(
                "relation_name",
                {x.rule for x in report.violations},
            )

    def test_duplicate_entity_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(
                tmpp / "ontology.yaml",
                entity_names=[
                    "alpha", "alpha",  # duplicate
                    "beta", "gamma", "delta", "epsilon",
                ],
                entity_overrides={"alpha": {"population_mode": "structural"}},
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn(
                "entity_unique",
                {x.rule for x in report.violations},
            )

    def test_duplicate_relation_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(
                tmpp / "ontology.yaml",
                relation_names=[
                    "CITES", "CITES",  # duplicate
                    "PROPOSES", "INTRODUCES", "EVALUATES_ON",
                    "USES_METHOD", "REPORTS_METRIC", "EXTENDS",
                    "EVALUATED_ON",
                ],
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn(
                "relation_unique",
                {x.rule for x in report.violations},
            )


class TestPopulationModeRules(unittest.TestCase):
    def test_extraction_prompt_on_structural_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(
                tmpp / "ontology.yaml",
                entity_overrides={
                    "document": {"population_mode": "structural"},
                    "chunk": {"population_mode": "structural"},
                    "method": {"population_mode": "extracted"},
                    "dataset": {"population_mode": "extracted"},
                    "metric": {"population_mode": "extracted"},
                    "task": {"population_mode": "extracted"},
                },
            )
            # Inject a mode-incompatible field by editing the YAML directly.
            txt = (tmpp / "ontology.yaml").read_text()
            (tmpp / "ontology.yaml").write_text(
                txt.replace(
                    "    population_mode: structural\n    source:",
                    "    population_mode: structural\n    extraction_prompt: extract_entities_v1\n    source:",
                    1,
                )
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn("5.5", {x.rule for x in report.violations})

    def test_derivation_on_extracted_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(
                tmpp / "ontology.yaml",
                entity_overrides={
                    "method": {"population_mode": "extracted"},
                },
            )
            txt = (tmpp / "ontology.yaml").read_text()
            (tmpp / "ontology.yaml").write_text(
                txt.replace(
                    "    population_mode: extracted\n    extraction_prompt: extract_entities_v1",
                    "    population_mode: extracted\n    derivation: wut\n    extraction_prompt: extract_entities_v1",
                    1,
                )
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn("5.5", {x.rule for x in report.violations})

    def test_source_on_derived_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(
                tmpp / "ontology.yaml",
                entity_overrides={
                    "method": {"population_mode": "derived"},
                },
            )
            txt = (tmpp / "ontology.yaml").read_text()
            (tmpp / "ontology.yaml").write_text(
                txt.replace(
                    "    population_mode: derived\n    derivation:",
                    "    population_mode: derived\n    source: bogus\n    derivation:",
                    1,
                )
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn("5.5", {x.rule for x in report.violations})

    def test_population_mode_outside_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(
                tmpp / "ontology.yaml",
                entity_overrides={
                    "method": {"population_mode": "extracted"},
                },
            )
            txt = (tmpp / "ontology.yaml").read_text()
            (tmpp / "ontology.yaml").write_text(
                txt.replace(
                    "    population_mode: extracted",
                    "    population_mode: fuzzy",
                    1,
                )
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn(
                "population_mode",
                {x.rule for x in report.violations},
            )


class TestExtractionPromptRules(unittest.TestCase):
    """Phase-specific prompt resolution (ontology.md §5.10) and edge cases."""

    def test_phase_specific_prompt_swap_on_entity(self):
        # Extracted entity-type whose extraction_prompt is the relations-pass
        # template. Violates §5.10 — the grouped templates are not
        # interchangeable between the two categories.
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(
                tmpp / "ontology.yaml",
                entity_overrides={
                    "method": {"population_mode": "extracted"},
                },
            )
            txt = (tmpp / "ontology.yaml").read_text()
            (tmpp / "ontology.yaml").write_text(
                txt.replace(
                    "extraction_prompt: extract_entities_v1",
                    "extraction_prompt: extract_relations_v1",
                    1,
                )
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn("5.10", {x.rule for x in report.violations})

    def test_phase_specific_prompt_swap_on_relation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(tmpp / "ontology.yaml")
            txt = (tmpp / "ontology.yaml").read_text()
            (tmpp / "ontology.yaml").write_text(
                txt.replace(
                    "extraction_prompt: extract_relations_v1",
                    "extraction_prompt: extract_entities_v1",
                    1,
                )
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn("5.10", {x.rule for x in report.violations})

    def test_unknown_extraction_prompt_name(self):
        # An extracted entity whose extraction_prompt is something other
        # than the two grouped-template names.
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(
                tmpp / "ontology.yaml",
                entity_overrides={
                    "method": {"population_mode": "extracted"},
                },
            )
            txt = (tmpp / "ontology.yaml").read_text()
            (tmpp / "ontology.yaml").write_text(
                txt.replace(
                    "extraction_prompt: extract_entities_v1",
                    "extraction_prompt: extract_things_v1",
                    1,
                )
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn("5.10", {x.rule for x in report.violations})


class TestEndpointReferences(unittest.TestCase):
    def test_relation_from_to_undeclared_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(
                tmpp / "ontology.yaml",
                relation_overrides={
                    "PROPOSES": {
                        "population_mode": "extracted",
                        "from": "ghost",
                        "to": "method",
                    },
                },
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn("5.4", {x.rule for x in report.violations})


class TestExampleRules(unittest.TestCase):
    def test_status_outside_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(tmpp / "ontology.yaml")
            txt = (tmpp / "ontology.yaml").read_text()
            (tmpp / "ontology.yaml").write_text(
                txt.replace(
                    "      - status: pending_verification\n        instance:",
                    "      - status: unverified\n        instance:",
                    1,
                )
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn(
                "example_status",
                {x.rule for x in report.violations},
            )

    def test_empty_examples_on_entity(self):
        # Construct an ontology whose first entity has empty examples.
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            ent_lines: list = []
            for i in range(6):
                ent_lines.append(f"  - name: e{i}")
                ent_lines.append(f"    definition: d{i}")
                if i == 0:
                    ent_lines.append("    population_mode: structural")
                    ent_lines.append("    source: data/manifest.jsonl")
                    ent_lines.append("    examples: []")
                elif i == 1:
                    ent_lines.append("    population_mode: structural")
                    ent_lines.append("    source: data/chunks.jsonl")
                    ent_lines.append("    examples:")
                    ent_lines.append("      - status: pending_verification")
                    ent_lines.append("        instance: c_xx")
                    ent_lines.append("        note: n")
                else:
                    ent_lines.append("    population_mode: extracted")
                    ent_lines.append("    extraction_prompt: extract_entities_v1")
                    ent_lines.append("    examples:")
                    ent_lines.append("      - status: pending_verification")
                    ent_lines.append(f"        instance: p{i}")
                    ent_lines.append("        note: n")
            rel_lines: list = []
            for i in range(8):
                rel_lines.append(f"  - name: R{i}")
                rel_lines.append("    from: e0")
                rel_lines.append("    to: e2")
                rel_lines.append("    definition: d")
                rel_lines.append("    population_mode: extracted")
                rel_lines.append("    extraction_prompt: extract_relations_v1")
                rel_lines.append("    examples:")
                rel_lines.append("      - status: pending_verification")
                rel_lines.append("        from: e0")
                rel_lines.append("        to: e2")
                rel_lines.append("        note: n")
            yaml_text = (
                "version: 1\n"
                "schema_version: 1\n"
                "entity_types:\n"
                + "\n".join(ent_lines) + "\n"
                "relation_types:\n"
                + "\n".join(rel_lines) + "\n"
            )
            (tmpp / "ontology.yaml").write_text(yaml_text)
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn("examples", {x.rule for x in report.violations})

    def test_verified_document_example_with_unknown_id(self):
        # Document-like entity-type (must be named "document" for the
        # structural check to fire under the genericity heuristic that
        # treats W-prefixed instances as manifest-anchored IDs).
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(tmpp / "ontology.yaml")
            txt = (tmpp / "ontology.yaml").read_text()
            # Replace the "document" entity's example with a verified one
            # whose instance is not in the manifest.
            new = (
                "version: 1\n"
                "schema_version: 1\n"
                "entity_types:\n"
                "  - name: document\n"
                "    definition: d\n"
                "    population_mode: structural\n"
                "    source: data/manifest.jsonl\n"
                "    examples:\n"
                "      - status: verified\n"
                "        instance: W9999999999\n"
                "        note: bogus\n"
                "  - name: chunk\n"
                "    definition: d\n"
                "    population_mode: structural\n"
                "    source: data/chunks.jsonl\n"
                "    examples:\n"
                "      - status: pending_verification\n"
                "        instance: c_xx\n"
                "        note: n\n"
                "  - name: a\n"
                "    definition: d\n"
                "    population_mode: extracted\n"
                "    extraction_prompt: extract_entities_v1\n"
                "    examples:\n"
                "      - status: pending_verification\n"
                "        instance: p\n"
                "        note: n\n"
                "  - name: b\n"
                "    definition: d\n"
                "    population_mode: extracted\n"
                "    extraction_prompt: extract_entities_v1\n"
                "    examples:\n"
                "      - status: pending_verification\n"
                "        instance: p\n"
                "        note: n\n"
                "  - name: c\n"
                "    definition: d\n"
                "    population_mode: extracted\n"
                "    extraction_prompt: extract_entities_v1\n"
                "    examples:\n"
                "      - status: pending_verification\n"
                "        instance: p\n"
                "        note: n\n"
                "relation_types:\n"
                + "\n".join(
                    f"  - name: R{i}\n"
                    f"    from: document\n"
                    f"    to: a\n"
                    f"    definition: d\n"
                    f"    population_mode: structural\n"
                    f"    source: data/manifest.jsonl\n"
                    f"    examples:\n"
                    f"      - status: pending_verification\n"
                    f"        from: document\n"
                    f"        to: a\n"
                    f"        note: n"
                    for i in range(8)
                )
                + "\n"
            )
            (tmpp / "ontology.yaml").write_text(new)
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn("5.9", {x.rule for x in report.violations})

    def test_verified_non_document_example_absent_from_surface(self):
        # A verified entity-type (extracted) example whose instance does
        # NOT appear in any corpus paper's title/abstract surface.
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            new = (
                "version: 1\n"
                "schema_version: 1\n"
                "entity_types:\n"
                "  - name: document\n"
                "    definition: d\n"
                "    population_mode: structural\n"
                "    source: data/manifest.jsonl\n"
                "    examples:\n"
                "      - status: verified\n"
                "        instance: W100\n"
                "        note: ok\n"
                "  - name: alpha\n"
                "    definition: d\n"
                "    population_mode: extracted\n"
                "    extraction_prompt: extract_entities_v1\n"
                "    examples:\n"
                "      - status: verified\n"
                "        instance: nonexistentthing\n"
                "        note: bogus\n"
                "  - name: beta\n"
                "    definition: d\n"
                "    population_mode: extracted\n"
                "    extraction_prompt: extract_entities_v1\n"
                "    examples:\n"
                "      - status: pending_verification\n"
                "        instance: p\n"
                "        note: n\n"
                "  - name: gamma\n"
                "    definition: d\n"
                "    population_mode: extracted\n"
                "    extraction_prompt: extract_entities_v1\n"
                "    examples:\n"
                "      - status: pending_verification\n"
                "        instance: p\n"
                "        note: n\n"
                "  - name: delta\n"
                "    definition: d\n"
                "    population_mode: extracted\n"
                "    extraction_prompt: extract_entities_v1\n"
                "    examples:\n"
                "      - status: pending_verification\n"
                "        instance: p\n"
                "        note: n\n"
                "relation_types:\n"
                + "\n".join(
                    f"  - name: R{i}\n"
                    f"    from: document\n"
                    f"    to: alpha\n"
                    f"    definition: d\n"
                    f"    population_mode: extracted\n"
                    f"    extraction_prompt: extract_relations_v1\n"
                    f"    examples:\n"
                    f"      - status: pending_verification\n"
                    f"        from: document\n"
                    f"        to: alpha\n"
                    f"        note: n"
                    for i in range(8)
                )
                + "\n"
            )
            (tmpp / "ontology.yaml").write_text(new)
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "alpha title", ["alpha", "body"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn("5.9", {x.rule for x in report.violations})

    def test_pending_verification_with_empty_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(tmpp / "ontology.yaml")
            txt = (tmpp / "ontology.yaml").read_text()
            new = txt.replace(
                "        note: synthetic placeholder for fixture.",
                "        note:",
                1,
            )
            (tmpp / "ontology.yaml").write_text(new)
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn("5.10", {x.rule for x in report.violations})

    def test_empty_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(tmpp / "ontology.yaml")
            txt = (tmpp / "ontology.yaml").read_text()
            new = txt.replace(
                "definition: definition-for-document",
                "definition:",
                1,
            )
            (tmpp / "ontology.yaml").write_text(new)
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn(
                "definition", {x.rule for x in report.violations}
            )

    def test_non_string_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(tmpp / "ontology.yaml")
            txt = (tmpp / "ontology.yaml").read_text()
            new = txt.replace(
                "definition: definition-for-document",
                "definition: 42",
                1,
            )
            (tmpp / "ontology.yaml").write_text(new)
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertIn(
                "definition", {x.rule for x in report.violations}
            )


# --- Generic / vocab-duplication / bump / genericity behavioural ------


def _build_widget_ontology(
    tmp: Path, *, enames, rnames, papers
) -> Path:
    """Build a *generic* valid v1-shape ontology with completely different
    entity/relation names and an unrelated synthetic corpus. The validator
    must accept it unchanged.
    """
    onp = tmp / "ontology.yaml"
    lines = ["version: 1", "schema_version: 1", "", "entity_types:"]
    for n in enames:
        lines.append(f"  - name: {n}")
        lines.append(f"    definition: definition-for-{n}")
        lines.append(f"    population_mode: structural")
        lines.append(f"    source: data/manifest.jsonl")
        lines.append(f"    examples:")
        lines.append(f"      - status: verified")
        lines.append(f"        instance: {papers[0][0]}")
        lines.append(f"        note: korrect")
    lines.append("")
    lines.append("relation_types:")
    for n in rnames:
        lines.append(f"  - name: {n}")
        lines.append(f"    from: {enames[0]}")
        lines.append(f"    to: {enames[0]}")
        lines.append(f"    definition: definition-for-{n}")
        lines.append(f"    population_mode: structural")
        lines.append(f"    source: data/manifest.jsonl")
        lines.append(f"    examples:")
        lines.append(f"      - status: pending_verification")
        lines.append(f"        from: {enames[0]}")
        lines.append(f"        to: {enames[0]}")
        lines.append(f"        note: synth")
    onp.write_text("\n".join(lines) + "\n")
    return onp


class TestGenericityBehavioural(unittest.TestCase):
    """The genericity guard: a synthetic ontology with completely different
    names must pass the validator unchanged. This proves the structural
    checks operate from the supplied YAML, not from a frozen v1 table.
    """

    def test_widget_ontology_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            enames = [
                "widget_a", "widget_b", "widget_c",
                "widget_d", "widget_e",
            ]
            rnames = [
                "WIDGET_A_TO_B", "WIDGET_B_TO_C", "WIDGET_C_TO_D",
                "WIDGET_D_TO_E", "WIDGET_A_TO_C", "WIDGET_B_TO_D",
                "WIDGET_C_TO_E", "WIDGET_A_TO_D",
            ]
            papers = (
                ("W12345", "widget_a title", ["body", "tokens"]),
            )
            onp = _build_widget_ontology(
                tmpp, enames=enames, rnames=rnames, papers=papers
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W12345",),
                papers=papers,
            )
            report = v.validate_ontology(
                onp, manifest_path=mp, openalex_dir=oa,
            )
            self.assertTrue(
                report.ok,
                msg=(
                    "widget ontology should pass generic validator; got "
                    f"{[x.format() for x in report.violations]}"
                ),
            )


# A non-trivial sanity check: the validator source must not define any
# constant/list/dict/module-level literal whose contents equal the v1
# vocabulary, the ontology endpoints, or the population modes (other than
# the policy names listed in the ticket — population modes and the two
# symbolic prompt names). We do this via a static AST scan.


V1_ENTITY_VOCAB = frozenset({
    "document", "chunk", "method", "dataset", "metric", "task",
})
V1_RELATION_VOCAB = frozenset({
    "CITES", "PROPOSES", "INTRODUCES", "EVALUATES_ON", "USES_METHOD",
    "REPORTS_METRIC", "EXTENDS", "EVALUATED_ON", "USED_FOR_TASK", "MEASURES",
})
V1_POPULATION_MODES = frozenset({"structural", "extracted", "derived"})
V1_PROMPTS = frozenset({
    "extract_entities_v1", "extract_relations_v1",
})


# Whitelist of identifiers that legitimately appear in the validator
# for non-duplication reasons (e.g. shape helpers, regex patterns,
# per-section comments about the v1 vocabulary).
V1_VOCAB_WHITELIST_IDS = frozenset({
    # constants names themselves (count as policy, not vocabulary duplication)
    "ENTITY_EXTRACTION_PROMPT",
    "RELATION_EXTRACTION_PROMPT",
    "EXTRACTION_PROMPTS",
    "FORBIDDEN_ENTITY_NAMES",
    "POPULATION_MODES",
    # _MODE_FIELDS is a per-mode field-rule lookup table — its keys are
    # the population-mode policy values themselves. The contents are
    # mode-specific field requirements (source/extraction_prompt/
    # derivation), not vocabulary duplication.
    "_MODE_FIELDS",
})


def _string_constants_in_module(tree: ast.Module) -> list:
    """Collect all string constants assigned at module top level."""
    out: list = []
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
            value = node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        # Only collect names that ARE NOT in the whitelist. The whitelist
        # names legitimately contain v1 vocabulary strings.
        if target.id in V1_VOCAB_WHITELIST_IDS:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            out.append((target.id, value.value))
        elif isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    out.append((target.id, elt.value))
        elif isinstance(value, ast.Dict):
            for k in value.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    out.append((target.id, k.value))
            for v in value.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    out.append((target.id, v.value))
    return out


class TestNoPythonVocabDuplication(unittest.TestCase):
    """AST guard: validator source must not define any constant/list/dict/
    module-level literal whose contents equal the v1 vocabulary or
    ontology endpoints.

    Only the named validation-policy constants
    (``FORBIDDEN_ENTITY_NAMES``, ``POPULATION_MODES``,
    ``ENTITY_EXTRACTION_PROMPT``, ``RELATION_EXTRACTION_PROMPT``,
    ``EXTRACTION_PROMPTS``) are allowed to mention v1 vocabulary
    strings — those are policy literals, per ontology.md §4.3.
    """

    def test_no_entity_vocab_duplication(self):
        src = (REPO_ROOT / "src" / "ontology" / "validate.py").read_text()
        tree = ast.parse(src)
        for name, value in _string_constants_in_module(tree):
            if value in V1_ENTITY_VOCAB:
                self.fail(
                    f"module-level literal {name!r} = {value!r} duplicates a "
                    "v1 entity vocabulary name; move to validation-policy "
                    "rules or ontology.yaml."
                )

    def test_no_relation_vocab_duplication(self):
        src = (REPO_ROOT / "src" / "ontology" / "validate.py").read_text()
        tree = ast.parse(src)
        for name, value in _string_constants_in_module(tree):
            if value in V1_RELATION_VOCAB:
                self.fail(
                    f"module-level literal {name!r} = {value!r} duplicates a "
                    "v1 relation vocabulary name."
                )

    def test_no_population_mode_duplication_outside_policy(self):
        # The `POPULATION_MODES` constant is itself the policy tuple. We
        # only flag module-level strings that equal a population-mode
        # value if their holding name isn't the policy one.
        src = (REPO_ROOT / "src" / "ontology" / "validate.py").read_text()
        tree = ast.parse(src)
        for name, value in _string_constants_in_module(tree):
            if value in V1_POPULATION_MODES:
                self.fail(
                    f"module-level literal {name!r} = {value!r} duplicates a "
                    "population-mode value; allowed only inside "
                    "POPULATION_MODES."
                )

    def test_no_extraction_prompt_duplication_outside_policy(self):
        src = (REPO_ROOT / "src" / "ontology" / "validate.py").read_text()
        tree = ast.parse(src)
        for name, value in _string_constants_in_module(tree):
            if value in V1_PROMPTS:
                self.fail(
                    f"module-level literal {name!r} = {value!r} duplicates a "
                    "phase-specific prompt name; allowed only inside "
                    "ENTITY_EXTRACTION_PROMPT / RELATION_EXTRACTION_PROMPT / "
                    "EXTRACTION_PROMPTS."
                )


class TestVersionBumpGuard(unittest.TestCase):
    def test_version_bump_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(tmpp / "ontology.yaml")
            txt = (tmpp / "ontology.yaml").read_text()
            (tmpp / "ontology.yaml").write_text(
                txt.replace("version: 1", "version: 2", 1)
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
                on_vocab_bump="warn",
            )
            self.assertTrue(report.ok)
            self.assertTrue(any("bump" in w.lower() for w in report.warnings))

    def test_version_bump_errors_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(tmpp / "ontology.yaml")
            txt = (tmpp / "ontology.yaml").read_text()
            (tmpp / "ontology.yaml").write_text(
                txt.replace("version: 1", "version: 2", 1)
            )
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )
            report = v.validate_ontology(
                tmpp / "ontology.yaml",
                manifest_path=mp, openalex_dir=oa,
                on_vocab_bump="error",
            )
            self.assertFalse(report.ok)
            self.assertEqual(
                {x.rule for x in report.violations},
                {"1.4"},
            )


class TestCLIExitBehaviour(unittest.TestCase):
    def test_cli_exits_nonzero_on_violation(self):
        from src.ontology.validate import main
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            _write_minimal_ontology(tmpp / "ontology.yaml")
            txt = (tmpp / "ontology.yaml").read_text()
            new = txt.replace("version: 1", "version: 0", 1)
            (tmpp / "ontology.yaml").write_text(new)
            import os
            old1, old2 = os.environ.get("ONTOLOGY_MANIFEST_PATH"), os.environ.get("ONTOLOGY_OPENALEX_DIR")
            os.environ["ONTOLOGY_MANIFEST_PATH"] = str(tmpp / "manifest.jsonl")
            os.environ["ONTOLOGY_OPENALEX_DIR"] = str(tmpp / "openalex")
            _write_manifest(tmpp / "manifest.jsonl", ids=("W100",))
            _write_openalex_dir(
                tmpp / "openalex", papers=(("W100", "t", ["t"]),)
            )
            try:
                rc = main([str(tmpp / "ontology.yaml")])
            finally:
                if old1 is None: os.environ.pop("ONTOLOGY_MANIFEST_PATH", None)
                else: os.environ["ONTOLOGY_MANIFEST_PATH"] = old1
                if old2 is None: os.environ.pop("ONTOLOGY_OPENALEX_DIR", None)
                else: os.environ["ONTOLOGY_OPENALEX_DIR"] = old2
            self.assertEqual(rc, 1)


class TestYAMLParseContract(unittest.TestCase):
    """The validator reads ``ontology.yaml`` exactly as committed.

    There is **no** pre-processing or auto-repair of malformed scalars.
    A file that ``yaml.safe_load`` cannot parse must surface a
    ``yaml_parse`` violation (not be silently fixed). This is the
    contract the lenient loader used to violate.
    """

    def test_committed_ontology_parses_with_standard_yaml(self):
        # Direct standard-library parse of the on-disk file must succeed.
        # If this fails, the validator was relying on a repair that
        # masked a real syntax issue in the committed file.
        import yaml as _yaml
        doc = _yaml.safe_load(Path("ontology.yaml").read_text())
        self.assertIsInstance(doc, dict)
        self.assertEqual(doc["version"], 1)
        self.assertEqual(doc["schema_version"], 1)

    def test_malformed_yaml_is_rejected_not_repaired(self):
        # Regression: an unquoted scalar with ": " mid-value is the
        # canonical #20 contract failure mode. The validator must surface
        # a `yaml_parse` violation; it must NOT silently rewrite the file.
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            onp = tmpp / "ontology.yaml"
            # `definition:` value contains `Direction: ` mid-scalar.
            bad = (
                "version: 1\n"
                "schema_version: 1\n"
                "entity_types:\n"
                "  - name: document\n"
                "    definition: A. Direction: cited in body.\n"
                "    population_mode: structural\n"
                "    source: data/manifest.jsonl\n"
                "    examples:\n"
                "      - status: pending_verification\n"
                "        instance: x\n"
                "        note: n\n"
                "  - name: alpha\n"
                "    definition: d\n"
                "    population_mode: structural\n"
                "    source: data/manifest.jsonl\n"
                "    examples:\n"
                "      - status: pending_verification\n"
                "        instance: y\n"
                "        note: n\n"
                "  - name: beta\n"
                "    definition: d\n"
                "    population_mode: extracted\n"
                "    extraction_prompt: extract_entities_v1\n"
                "    examples:\n"
                "      - status: pending_verification\n"
                "        instance: z\n"
                "        note: n\n"
                "  - name: gamma\n"
                "    definition: d\n"
                "    population_mode: extracted\n"
                "    extraction_prompt: extract_entities_v1\n"
                "    examples:\n"
                "      - status: pending_verification\n"
                "        instance: z\n"
                "        note: n\n"
                "  - name: delta\n"
                "    definition: d\n"
                "    population_mode: extracted\n"
                "    extraction_prompt: extract_entities_v1\n"
                "    examples:\n"
                "      - status: pending_verification\n"
                "        instance: z\n"
                "        note: n\n"
                "relation_types:\n"
                + "\n".join(
                    f"  - name: R{i}\n"
                    f"    from: document\n"
                    f"    to: alpha\n"
                    f"    definition: d\n"
                    f"    population_mode: structural\n"
                    f"    source: data/manifest.jsonl\n"
                    f"    examples:\n"
                    f"      - status: pending_verification\n"
                    f"        from: document\n"
                    f"        to: alpha\n"
                    f"        note: n"
                    for i in range(8)
                )
                + "\n"
            )
            onp.write_text(bad)
            mp, oa = _build_minimal_corpus(
                tmpp, manifest_ids=("W100",),
                papers=(("W100", "t", ["t"]),),
            )

            import yaml as _yaml
            with self.assertRaises(_yaml.YAMLError):
                _yaml.safe_load(onp.read_text())

            report = v.validate_ontology(
                onp, manifest_path=mp, openalex_dir=oa,
            )
            self.assertFalse(report.ok)
            self.assertEqual(
                {x.rule for x in report.violations},
                {"yaml_parse"},
                msg=f"expected only yaml_parse violation; got "
                    f"{[x.format() for x in report.violations]}",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
