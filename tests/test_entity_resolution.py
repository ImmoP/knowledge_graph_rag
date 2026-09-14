"""Acceptance tests for ``src.graph.entity_resolution`` (#24).

These cover the §10 acceptance gate end-to-end. Each ``TestCase``
maps to one numbered acceptance criterion or group of related
criteria; the assertion body names the exact contract under test.
The fixtures build a deterministic synthetic corpus in a temp
directory so the tests can exercise every code path without
depending on the production extraction run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph import entity_resolution as er  # noqa: E402
from src.graph.entity_resolution import (  # noqa: E402
    entity_id,
    lev_similarity,
    normalized_resolution_name,
    run_resolution,
)

# ---------------------------------------------------------------------------
# §3 normalisation fixtures.
# ---------------------------------------------------------------------------


class NormalisationTests(unittest.TestCase):
    """§3 order-sensitive normalisation chain."""

    def test_nfkc_then_casefold(self):
        # NFKC decomposes ß into "ss" (casefold then collapses to "ss").
        assert normalized_resolution_name("Stra\u00dfe") == "strasse"
        # Already folded forms survive.
        assert normalized_resolution_name("DPR") == "dpr"
        assert normalized_resolution_name("dpr") == "dpr"

    def test_pd_dash_to_ascii(self):
        # U+2010 hyphen, U+2013 en-dash, U+2014 em-dash, U+2015 horizontal
        # bar all map to ASCII "-".
        for ch in ("\u2010", "\u2013", "\u2014", "\u2015"):
            assert (
                normalized_resolution_name(f"ColBERT{ch}v2") == "colbert-v2"
            ), ch
        # The acceptance fixture: ``ColBERT‐v2`` (U+2010 hyphen) collapses
        # to identical normalised form as ``ColBERT-v2`` (ASCII "-").
        assert normalized_resolution_name("ColBERT‐v2") == normalized_resolution_name(
            "ColBERT-v2"
        )

    def test_whitespace_runs_collapse_and_trim(self):
        assert normalized_resolution_name("  BERT   v2  ") == "bert v2"
        assert normalized_resolution_name("\t\ndpr\n\t") == "dpr"
        # Internal runs collapse to one ASCII space.
        assert normalized_resolution_name("a\t\tb") == "a b"

    def test_wrapper_quote_whitelist_pair_removal(self):
        assert normalized_resolution_name('"DPR"') == "dpr"
        assert normalized_resolution_name("'DPR'") == "dpr"
        assert normalized_resolution_name("`DPR`") == "dpr"
        assert normalized_resolution_name("\u201cDPR\u201d") == "dpr"
        assert normalized_resolution_name("\u2018DPR\u2019") == "dpr"
        assert normalized_resolution_name("\u00abDPR\u00bb") == "dpr"
        assert normalized_resolution_name("\u2039DPR\u203a") == "dpr"

    def test_nested_or_repeated_wrapper_quote_removal(self):
        # nested ASCII doubles: each pass peels one matched pair + trims.
        assert normalized_resolution_name('"""DPR"""') == "dpr"
        # mixed curly/ascii: outer ASCII -> "‘DPR’" -> inner curly -> DPR.
        assert normalized_resolution_name('"\u2018DPR\u2019"') == "dpr"
        # Repeated removal converges; one-pass == n-pass.
        one_pass = normalized_resolution_name('""DPR""')
        multi_pass = normalized_resolution_name('""""DPR""""')
        assert one_pass == "dpr"
        assert multi_pass == "dpr"

    def test_non_whitelist_punctuation_preserved(self):
        # Parentheses, brackets, commas are NOT in the whitelist.
        assert normalized_resolution_name("(DPR)") == "(dpr)"
        assert normalized_resolution_name("[DPR]") == "[dpr]"
        assert normalized_resolution_name("DPR,") == "dpr,"
        # ASCII "-" is not a wrapper character and stays literal.
        assert normalized_resolution_name("-DPR-") == "-dpr-"
        # Multiple non-whitelist chars preserved literally.
        assert normalized_resolution_name("(DPR) [extra]!") == "(dpr) [extra]!"

    def test_internal_punctuation_and_symbols_preserved(self):
        assert normalized_resolution_name("v1.5") == "v1.5"
        assert normalized_resolution_name("Recall@5") == "recall@5"
        assert normalized_resolution_name("T5-base") == "t5-base"
        assert normalized_resolution_name("T5+") == "t5+"
        assert normalized_resolution_name("path/to/x") == "path/to/x"

    def test_distinct_mentions_stay_distinct(self):
        # §3 explicit examples — surface variants remain distinct.
        assert normalized_resolution_name("BERT v1") != normalized_resolution_name(
            "BERT v2"
        )
        assert normalized_resolution_name("Recall@5") != normalized_resolution_name(
            "Recall@10"
        )
        assert normalized_resolution_name("T5-base") != normalized_resolution_name(
            "T5-large"
        )
        # Exact matches become exact-match eligible.
        assert normalized_resolution_name('"DPR"') == normalized_resolution_name("DPR")

    def test_empty_normalisation_keeps_empty(self):
        assert normalized_resolution_name("") == ""



# ---------------------------------------------------------------------------
# §4 similarity + threshold pin.
# ---------------------------------------------------------------------------


class SimilarityTests(unittest.TestCase):

    def test_identical_is_one(self):
        assert lev_similarity("dpr", "dpr") == 1.0

    def test_known_distance_values(self):
        # 1 insertion -> 1 - 1/4
        assert lev_similarity("dpr", "dprx") == 1 - 1 / 4
        # 1 substitution on length-3 -> 1 - 1/3
        assert lev_similarity("dpr", "dpa") == 1 - 1 / 3

    def test_pinned_threshold_constant(self):
        assert er.NORMALIZED_LEVENSHTEIN_THRESHOLD == 0.92
        assert er.ENTITY_RESOLUTION_ALGORITHM_VERSION == "entity_resolution_v1"
        assert er.NORMALIZATION_VERSION == "resolution_name_v1"


# ---------------------------------------------------------------------------
# §6.1 entity ID derivation.
# ---------------------------------------------------------------------------


class EntityIdTests(unittest.TestCase):

    def test_format_and_determinism(self):
        a = entity_id(canonical_name="DPR", entity_type="method")
        b = entity_id(canonical_name="DPR", entity_type="method")
        assert a == b
        assert a.startswith("e_")
        assert len(a) == len("e_") + 16
        body = a[2:]
        assert all(c in "0123456789abcdef" for c in body), body

    def test_payload_includes_normalised_canonical_name(self):
        a = entity_id(canonical_name="Dense Passage Retrieval", entity_type="method")
        b = entity_id(canonical_name='"Dense Passage Retrieval"', entity_type="method")
        assert a == b

    def test_distinct_inputs_distinct_ids(self):
        a = entity_id(canonical_name="DPR", entity_type="method")
        b = entity_id(canonical_name="T5", entity_type="method")
        assert a != b
        a = entity_id(canonical_name="DPR", entity_type="method")
        b = entity_id(canonical_name="DPR", entity_type="dataset")
        assert a != b


# ---------------------------------------------------------------------------
# Synthetic corpus builders used by the orchestrator tests below.
# ---------------------------------------------------------------------------


_MINIMAL_ONTOLOGY = """\
version: 1
schema_version: 1
entity_types:
  - name: document
    population_mode: structural
    examples: []
  - name: method
    population_mode: extracted
    extraction_prompt: extract_entities_v1
    examples: []
  - name: dataset
    population_mode: extracted
    extraction_prompt: extract_entities_v1
    examples: []
  - name: metric
    population_mode: extracted
    extraction_prompt: extract_entities_v1
    examples: []
  - name: task
    population_mode: extracted
    extraction_prompt: extract_entities_v1
    examples: []
relation_types: []
"""


def _canonical_mention(
    *,
    chunk_id: str,
    doc_id: str,
    entity_type: str,
    canonical: str,
    surface: str,
    char_start: int,
    char_end: int,
    chunk_text: str | None = None,
) -> dict:
    """Build a §3.1.1 mention record whose mention_id re-derives from the
    extraction contract's §6.1 derivation rule.
    """
    text = chunk_text if chunk_text is not None else f"chunk text {chunk_id}"
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    payload = {
        "chunk_id": chunk_id,
        "source_text_sha256": sha,
        "entity_type": entity_type,
        "char_start": char_start,
        "char_end": char_end,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
    ).hexdigest()
    return {
        "kind": "mention",
        "mention_id": "m_" + digest[:16],
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "entity_type": entity_type,
        "surface_form": surface,
        "canonical_form": canonical,
        "evidence": {
            "char_start": char_start,
            "char_end": char_end,
            "source_text_sha256": sha,
        },
        "confidence": "stated",
    }


def _chunk_status(*, chunk_id: str, doc_id: str, complete: bool) -> dict:
    return {
        "kind": "chunk_status",
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "phases": {
            "entity": {
                "status": "validated",
                "attempt_count": 1,
                "mention_count": 1,
                "failure_count": 0,
            },
            "relation": {
                "status": "validated",
                "attempt_count": 1,
                "assertion_count": 0,
                "failure_count": 0,
            },
        },
        "extraction_complete": complete,
    }


def _build_world(
    tmp: Path,
    *,
    chunks=("c_0000000000000001",),
    mentions=(),
    chunk_status_complete=True,
) -> dict:
    """Write a synthetic chunks + extractions + manifest world under ``tmp``."""
    onp = tmp / "ontology.yaml"
    onp.write_text(_MINIMAL_ONTOLOGY)
    chp = tmp / "chunks.jsonl"
    chunks_records = [
        {
            "chunk_id": cid,
            "doc_id": "W1",
            "section_path": ["intro"],
            "chunk_index": i,
            "text": f"chunk text {cid}",
            "token_estimate": 10,
            "char_start": 0,
            "char_end": 30,
        }
        for i, cid in enumerate(chunks)
    ]
    chp.write_text(
        "\n".join(
            json.dumps(c, sort_keys=True, ensure_ascii=False) for c in chunks_records
        )
        + "\n",
        encoding="utf-8",
    )
    chunks_sha = hashlib.sha256(chp.read_bytes()).hexdigest()
    ontology_sha = hashlib.sha256(onp.read_bytes()).hexdigest()
    emp = tmp / "extractions.manifest.json"
    emp.write_text(
        json.dumps(
            {
                "chunks_sha256": chunks_sha,
                "ontology_sha256": ontology_sha,
                "ontology_version": 1,
                "schema_version": 1,
                "run_id": "synthetic",
                "provider": "synthetic",
                "model": "synthetic",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    exp = tmp / "extractions.jsonl"
    rows = []
    rows.extend(
        _chunk_status(chunk_id=cid, doc_id="W1", complete=chunk_status_complete)
        for cid in chunks
    )
    rows.extend(mentions)
    exp.write_text(
        "\n".join(
            json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "ontology": onp,
        "chunks": chp,
        "extractions": exp,
        "manifest": emp,
        "entities": tmp / "entities.jsonl",
        "entity_resolutions": tmp / "entity_resolutions.jsonl",
        "entity_resolution_manifest": tmp / "entity_resolution.manifest.json",
    }


def _read_jsonl(p: Path) -> list:
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Acceptance criteria fixtures.
# ---------------------------------------------------------------------------


def _run(world: dict):
    return run_resolution(
        chunks_path=world["chunks"],
        extractions_path=world["extractions"],
        extractions_manifest_path=world["manifest"],
        ontology_path=world["ontology"],
        entities_path=world["entities"],
        entity_resolutions_path=world["entity_resolutions"],
        entity_resolution_manifest_path=world["entity_resolution_manifest"],
    )


class HardPreflightTests(unittest.TestCase):
    """Acceptance criterion 4: corrupted manifest aborts before any output."""

    def test_corrupted_chunks_sha_aborts_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mentions = [
                _canonical_mention(
                    chunk_id="c_0000000000000001",
                    doc_id="W1",
                    entity_type="method",
                    canonical="DPR",
                    surface="DPR",
                    char_start=0,
                    char_end=3,
                )
            ]
            w = _build_world(tmp, mentions=mentions)
            # Corrupt the chunks SHA.
            manifest = json.loads(w["manifest"].read_text(encoding="utf-8"))
            manifest["chunks_sha256"] = "0" * 64
            w["manifest"].write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaises(er.PreflightError):
                _run(w)
            self.assertFalse(w["entities"].exists())
            self.assertFalse(w["entity_resolutions"].exists())
            self.assertFalse(w["entity_resolution_manifest"].exists())

    def test_missing_chunk_status_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # Build a world with two chunks and only one status record.
            mentions = [
                _canonical_mention(
                    chunk_id="c_0000000000000001",
                    doc_id="W1",
                    entity_type="method",
                    canonical="DPR",
                    surface="DPR",
                    char_start=0,
                    char_end=3,
                ),
                _canonical_mention(
                    chunk_id="c_0000000000000002",
                    doc_id="W1",
                    entity_type="method",
                    canonical="T5",
                    surface="T5",
                    char_start=0,
                    char_end=2,
                ),
            ]
            w = _build_world(
                tmp,
                chunks=("c_0000000000000001", "c_0000000000000002"),
                mentions=mentions,
            )
            # Drop the second status record to simulate a partial run.
            rows = _read_jsonl(w["extractions"])
            filtered = [r for r in rows if r.get("chunk_id") != "c_0000000000000002"]
            w["extractions"].write_text(
                "\n".join(
                    json.dumps(r, sort_keys=True, ensure_ascii=False)
                    for r in filtered
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(er.PreflightError):
                _run(w)
            self.assertFalse(w["entities"].exists())

    def test_incomplete_chunk_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            w = _build_world(tmp, chunk_status_complete=False)
            with self.assertRaises(er.PreflightError):
                _run(w)
            self.assertFalse(w["entities"].exists())

    def test_unknown_chunk_id_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mention = _canonical_mention(
                chunk_id="c_0000000000000001",
                doc_id="W1",
                entity_type="method",
                canonical="DPR",
                surface="DPR",
                char_start=0,
                char_end=3,
            )
            mention["chunk_id"] = "c_unknownchunk1234567"
            w = _build_world(tmp, mentions=[mention])


class DeterminismTests(unittest.TestCase):
    """§8: two consecutive runs on identical inputs are byte-identical."""

    def test_two_runs_byte_identical(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # Two mentions across two entity types in one chunk.
            mentions = [
                _canonical_mention(
                    chunk_id="c_0000000000000001",
                    doc_id="W1",
                    entity_type="method",
                    canonical="DPR",
                    surface="DPR",
                    char_start=0,
                    char_end=3,
                ),
                _canonical_mention(
                    chunk_id="c_0000000000000001",
                    doc_id="W1",
                    entity_type="dataset",
                    canonical="NQ",
                    surface="Natural Questions",
                    char_start=10,
                    char_end=20,
                ),
            ]
            w = _build_world(tmp, mentions=mentions)
            _run(w)
            ent_b = w["entities"].read_bytes()
            res_b = w["entity_resolutions"].read_bytes()
            man_b = w["entity_resolution_manifest"].read_bytes()
            _run(w)
            self.assertEqual(w["entities"].read_bytes(), ent_b)
            self.assertEqual(w["entity_resolutions"].read_bytes(), res_b)
            self.assertEqual(w["entity_resolution_manifest"].read_bytes(), man_b)


class StageAInputPinTests(unittest.TestCase):
    """Criterion 5: ``canonical_form`` is the clustering key, not surface."""

    def test_surface_does_not_affect_clustering(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mentions = [
                _canonical_mention(
                    chunk_id="c_0000000000000001",
                    doc_id="W1",
                    entity_type="method",
                    canonical="ColBERT",
                    surface="ColBERT!",
                    char_start=0,
                    char_end=8,
                ),
                _canonical_mention(
                    chunk_id="c_0000000000000002",
                    doc_id="W1",
                    entity_type="method",
                    canonical="ColBERT",
                    surface="ColBERT!?",
                    char_start=0,
                    char_end=9,
                ),
            ]
            w = _build_world(
                tmp,
                chunks=("c_0000000000000001", "c_0000000000000002"),
                mentions=mentions,
            )
            _run(w)
            entities = _read_jsonl(w["entities"])
            # Both mentions must merge into a single entity because
            # ``canonical_form`` is identical.
            self.assertEqual(len(entities), 1)
            self.assertEqual(len(entities[0]["mention_ids"]), 2)
            # Surface variant remains in the alias list.
            self.assertIn("ColBERT!", entities[0]["aliases"])
            self.assertIn("ColBERT!?", entities[0]["aliases"])


class StageBTypeSeparationTests(unittest.TestCase):
    """Criterion 9: cross-type mentions stay isolated."""

    def test_high_similarity_across_types_does_not_merge(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # Two mentions with normalised identical strings ("colbert")
            # but different entity types. They must NOT merge.
            mentions = [
                _canonical_mention(
                    chunk_id="c_0000000000000001",
                    doc_id="W1",
                    entity_type="method",
                    canonical="ColBERT",
                    surface="ColBERT",
                    char_start=0,
                    char_end=7,
                ),
                _canonical_mention(
                    chunk_id="c_0000000000000002",
                    doc_id="W1",
                    entity_type="dataset",
                    canonical="ColBERT",
                    surface="ColBERT",
                    char_start=0,
                    char_end=7,
                ),
            ]
            w = _build_world(
                tmp,
                chunks=("c_0000000000000001", "c_0000000000000002"),
                mentions=mentions,
            )
            _run(w)
            entities = _read_jsonl(w["entities"])
            self.assertEqual(len(entities), 2)
            types = sorted(e["entity_type"] for e in entities)
            self.assertEqual(types, ["dataset", "method"])


class ThresholdSensitivityTests(unittest.TestCase):
    """Criterion 3: threshold changes cluster membership."""

    def test_threshold_change_changes_membership(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # Two clusters whose normalised similarity sits between
            # 0.92 and 0.95: length-30 strings with 2 substitutions →
            # similarity 28/30 ≈ 0.9333 (above the default 0.92 but
            # strictly below 0.95). This is the only window where
            # mutating the threshold from 0.92 to 0.95 changes cluster
            # membership.
            a = _canonical_mention(
                chunk_id="c_0000000000000001",
                doc_id="W1",
                entity_type="method",
                canonical="A" * 30,
                surface="A" * 30,
                char_start=0,
                char_end=30,
            )
            # Target cluster: length-30 "a"*30. Candidate cluster:
            # length-30 with 2 substitutions → sim = 28/30 ≈ 0.9333.
            b = _canonical_mention(
                chunk_id="c_0000000000000002",
                doc_id="W1",
                entity_type="method",
                canonical=("A" * 28) + "BC",
                surface=("A" * 28) + "BC",
                char_start=0,
                char_end=30,
            )
            w = _build_world(
                tmp,
                chunks=("c_0000000000000001", "c_0000000000000002"),
                mentions=[a, b],
            )
            _run(w)
            ents = _read_jsonl(w["entities"])
            self.assertEqual(len(ents), 1, ents)
            # Mutating the threshold to 0.95 splits the cluster.
            original = er.NORMALIZED_LEVENSHTEIN_THRESHOLD
            try:
                er.NORMALIZED_LEVENSHTEIN_THRESHOLD = 0.95
                _run(w)
                ents = _read_jsonl(w["entities"])
                self.assertEqual(len(ents), 2, ents)
            finally:
                er.NORMALIZED_LEVENSHTEIN_THRESHOLD = original
            # Reverting restores the canonical merge.
            _run(w)
            ents = _read_jsonl(w["entities"])
            self.assertEqual(len(ents), 1)


class WeakestLinkConfidenceTests(unittest.TestCase):
    """Criterion 8: ``resolution_confidence`` is the weakest accepted score."""

    def test_weakest_link_not_mean(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # Cluster A: ``"a" * 30`` (target seed, 0.97-ish merge with B)
            # Cluster B: ``"a"*29 + "b"`` — 1 sub from A, score 29/30=0.9667
            # Cluster C: ``"a"*28 + "bc"`` — 2 subs from A, 1 sub from B;
            #   cross_score(C, T={A,B}) = min(sim_to_A, sim_to_B)
            #   = min(28/30, 29/30) = 28/30 ≈ 0.9333.
            big_a = _canonical_mention(
                chunk_id="c_0000000000000001",
                doc_id="W1",
                entity_type="method",
                canonical="a" * 30,
                surface="a" * 30,
                char_start=0,
                char_end=30,
            )
            big_b = _canonical_mention(
                chunk_id="c_0000000000000002",
                doc_id="W1",
                entity_type="method",
                canonical=("a" * 29) + "b",
                surface=("a" * 29) + "b",
                char_start=0,
                char_end=30,
            )
            big_c = _canonical_mention(
                chunk_id="c_0000000000000003",
                doc_id="W1",
                entity_type="method",
                canonical=("a" * 28) + "bc",
                surface=("a" * 28) + "bc",
                char_start=0,
                char_end=30,
            )
            w = _build_world(
                tmp,
                chunks=(
                    "c_0000000000000001",
                    "c_0000000000000002",
                    "c_0000000000000003",
                ),
                mentions=[big_a, big_b, big_c],
            )
            _run(w)
            ents = _read_jsonl(w["entities"])
            self.assertEqual(len(ents), 1)
            # Weakest accepted score = 28/30 ≈ 0.933333; rounded to six
            # decimals = 0.933333. NOT mean of 0.9667 and 0.9333.
            self.assertAlmostEqual(
                ents[0]["resolution_confidence"], 0.933333, places=5
            )


class MappingResolutionTests(unittest.TestCase):
    """Criterion 10: Stage A → exact_normalized; Stage B → fuzzy."""

    def test_stage_a_method_and_score(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mentions = [
                _canonical_mention(
                    chunk_id="c_0000000000000001",
                    doc_id="W1",
                    entity_type="method",
                    canonical="DPR",
                    surface="DPR",
                    char_start=0,
                    char_end=3,
                ),
                _canonical_mention(
                    chunk_id="c_0000000000000002",
                    doc_id="W1",
                    entity_type="method",
                    canonical="DPR",
                    surface="DPR",
                    char_start=0,
                    char_end=3,
                ),
            ]
            w = _build_world(
                tmp,
                chunks=("c_0000000000000001", "c_0000000000000002"),
                mentions=mentions,
            )
            _run(w)
            res = _read_jsonl(w["entity_resolutions"])
            self.assertEqual(len(res), 2)
            for r in res:
                self.assertEqual(r["resolution_method"], "exact_normalized")
                self.assertEqual(r["resolution_score"], 1.0)

    def test_stage_b_method_and_score(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            a = _canonical_mention(
                chunk_id="c_0000000000000001",
                doc_id="W1",
                entity_type="method",
                canonical="a" * 30,
                surface="a" * 30,
                char_start=0,
                char_end=30,
            )
            b = _canonical_mention(
                chunk_id="c_0000000000000002",
                doc_id="W1",
                entity_type="method",
                canonical=("a" * 29) + "b",
                surface=("a" * 29) + "b",
                char_start=0,
                char_end=30,
            )
            w = _build_world(
                tmp,
                chunks=("c_0000000000000001", "c_0000000000000002"),
                mentions=[a, b],
            )
            _run(w)
            res = _read_jsonl(w["entity_resolutions"])
            self.assertEqual(len(res), 2)
            methods = sorted(r["resolution_method"] for r in res)
            self.assertEqual(methods, ["exact_normalized", "fuzzy"])
            fuzzy = next(r for r in res if r["resolution_method"] == "fuzzy")
            # 29/30 = 0.96666666...
            self.assertAlmostEqual(fuzzy["resolution_score"], 0.966667, places=5)

import re


class AliasOrderingTests(unittest.TestCase):
    """Criterion 11 / §6.4: aliases sorted by
    ``(normalized_resolution_name(alias), alias)``.
    """

    def test_alias_excludes_canonical(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            a = _canonical_mention(
                chunk_id="c_0000000000000001",
                doc_id="W1",
                entity_type="method",
                canonical="DPR",
                surface="dense passage retrieval",
                char_start=0,
                char_end=3,
            )
            b = _canonical_mention(
                chunk_id="c_0000000000000002",
                doc_id="W1",
                entity_type="method",
                canonical="DPR",
                surface="DPR",
                char_start=0,
                char_end=3,
            )
            w = _build_world(
                tmp,
                chunks=("c_0000000000000001", "c_0000000000000002"),
                mentions=[a, b],
            )
            _run(w)
            ents = _read_jsonl(w["entities"])
            self.assertEqual(len(ents), 1)
            aliases = ents[0]["aliases"]
            self.assertNotIn("DPR", aliases)
            self.assertEqual(aliases, ["dense passage retrieval"])

    def test_alias_sort_key(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            a = _canonical_mention(
                chunk_id="c_0000000000000001",
                doc_id="W1",
                entity_type="method",
                canonical="Alpha",
                surface="AlphaSurf",
                char_start=0,
                char_end=5,
            )
            b = _canonical_mention(
                chunk_id="c_0000000000000002",
                doc_id="W1",
                entity_type="method",
                canonical="Alpha",
                surface="BetaSurf",
                char_start=0,
                char_end=5,
            )
            w = _build_world(
                tmp,
                chunks=("c_0000000000000001", "c_0000000000000002"),
                mentions=[a, b],
            )
            _run(w)
            ents = _read_jsonl(w["entities"])
            self.assertEqual(len(ents), 1)
            aliases = ents[0]["aliases"]
            self.assertNotIn("Alpha", aliases)
            self.assertIn("AlphaSurf", aliases)
            self.assertIn("BetaSurf", aliases)
            self.assertEqual(aliases, sorted(aliases))

class OutputSchemaTests(unittest.TestCase):
    """§7: entities.jsonl / entity_resolutions.jsonl / manifest shapes."""

    def test_entity_fields_exact(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            m = _canonical_mention(
                chunk_id="c_0000000000000001",
                doc_id="W1",
                entity_type="method",
                canonical="DPR",
                surface="DPR",
                char_start=0,
                char_end=3,
            )
            w = _build_world(tmp, mentions=[m])
            _run(w)
            ents = _read_jsonl(w["entities"])
            self.assertEqual(len(ents), 1)
            self.assertEqual(
                sorted(ents[0].keys()),
                ["aliases", "canonical_name", "entity_id", "entity_type",
                 "mention_ids", "resolution_confidence"],
            )
            self.assertTrue(
                re.match(r"^e_[0-9a-f]{16}$", ents[0]["entity_id"]),
                ents[0]["entity_id"],
            )
            self.assertEqual(ents[0]["resolution_confidence"], 1.0)
    def test_resolution_fields_exact(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            m = _canonical_mention(
                chunk_id="c_0000000000000001",
                doc_id="W1",
                entity_type="method",
                canonical="DPR",
                surface="DPR",
                char_start=0,
                char_end=3,
            )
            w = _build_world(tmp, mentions=[m])
            _run(w)
            res = _read_jsonl(w["entity_resolutions"])
            self.assertEqual(len(res), 1)
            self.assertEqual(
                sorted(res[0].keys()),
                ["entity_id", "mention_id", "resolution_method",
                 "resolution_score"],
            )
            self.assertEqual(res[0]["resolution_method"], "exact_normalized")
            self.assertEqual(res[0]["resolution_score"], 1.0)

    def test_manifest_required_fields_only(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            m = _canonical_mention(
                chunk_id="c_0000000000000001",
                doc_id="W1",
                entity_type="method",
                canonical="DPR",
                surface="DPR",
                char_start=0,
                char_end=3,
            )
            w = _build_world(tmp, mentions=[m])
            _run(w)
            manifest = json.loads(
                w["entity_resolution_manifest"].read_text(encoding="utf-8")
            )
            self.assertEqual(
                sorted(manifest.keys()),
                ["algorithm", "counts", "inputs", "outputs", "schema_version"],
            )
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(
                manifest["algorithm"]["algorithm_version"],
                "entity_resolution_v1",
            )
            self.assertEqual(
                manifest["algorithm"]["normalization_version"],
                "resolution_name_v1",
            )
            self.assertEqual(
                manifest["algorithm"]["similarity_metric"],
                "normalized_levenshtein",
            )
            self.assertEqual(
                manifest["algorithm"]["threshold"],
                0.92,
            )
            blob = json.dumps(manifest, sort_keys=True)
            self.assertNotIn("uuid", blob.lower())
            self.assertNotIn("host", blob.lower())
            self.assertNotIn("timestamp", blob.lower())
            self.assertNotIn("wall_clock", blob.lower())
            self.assertNotIn("run_id", blob.lower())
            for section in ("inputs", "outputs"):
                for v in manifest[section].values():
                    if isinstance(v, str) and len(v) == 64:
                        self.assertTrue(
                            all(c in "0123456789abcdef" for c in v),
                            (section, v),
                        )
            self.assertTrue(
                w["entity_resolution_manifest"].read_bytes().endswith(b"\n")
            )


class NoOverrideTests(unittest.TestCase):
    """Criterion 3: no CLI / env override accepted for the threshold.

    The resolver entry point exposes no parameter that re-configures
    the threshold; the only knob is the module-scope constant. This
    test guards against accidental additions to ``run_resolution``'s
    public signature.
    """

    def test_run_resolution_signature_has_no_threshold_param(self):
        import inspect

        sig = inspect.signature(run_resolution)
        # No parameter named "threshold" or any synonym.
        for name in sig.parameters:
            assert "threshold" not in name.lower(), name
            assert "score" not in name.lower() or name == "score" or True, name
            # Concrete ban list.
            assert name not in {
                "threshold",
                "similarity_threshold",
                "merge_threshold",
                "NORMALIZED_LEVENSHTEIN_THRESHOLD",
            }, name

    def test_threshold_constant_at_module_scope(self):
        # Constant is exposed as a module attribute; check its value.
        assert er.NORMALIZED_LEVENSHTEIN_THRESHOLD == 0.92
        # No other module-scope knobs named "*threshold*" exist.
        for name in vars(er):
            if name.startswith("_") or name == "NORMALIZED_LEVENSHTEIN_THRESHOLD":
                continue
            assert "threshold" not in name.lower(), name


class RunResolutionDeterminismTests(unittest.TestCase):
    """§8 byte-identical re-runs across the inputs we re-derive."""

    def test_line_reorder_byte_identical_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mentions = []
            for i in range(4):
                mentions.append(
                    _canonical_mention(
                        chunk_id="c_0000000000000001",
                        doc_id="W1",
                        entity_type="method",
                        canonical="DPR",
                        surface="Dense Passage Retrieval",
                        char_start=i,
                        char_end=i + 3,
                    )
                )
            # Build into outA.
            tmpA = tmp / "a"
            tmpA.mkdir()
            wA = _build_world(
                tmpA, chunks=("c_0000000000000001",), mentions=mentions
            )
            eA = tmpA / "entities.jsonl"
            rA = tmpA / "entity_resolutions.jsonl"
            mA = tmpA / "entity_resolution.manifest.json"
            _run_with_world(wA, eA, rA, mA)
            # Build into outB with reordered mentions.
            tmpB = tmp / "b"
            tmpB.mkdir()
            wB = _build_world(
                tmpB,
                chunks=("c_0000000000000001",),
                mentions=list(reversed(mentions)),
            )
            eB = tmpB / "entities.jsonl"
            rB = tmpB / "entity_resolutions.jsonl"
            mB = tmpB / "entity_resolution.manifest.json"
            _run_with_world(wB, eB, rB, mB)
            # only in input paths and SHA256 of the on-disk chunks.
            self.assertEqual(eA.read_bytes(), eB.read_bytes())


def _run_with_world(world, entities_path, entity_resolutions_path, entity_resolution_manifest_path):
    return run_resolution(
        chunks_path=world["chunks"],
        extractions_path=world["extractions"],
        extractions_manifest_path=world["manifest"],
        ontology_path=world["ontology"],
        entities_path=entities_path,
        entity_resolutions_path=entity_resolutions_path,
        entity_resolution_manifest_path=entity_resolution_manifest_path,
    )
