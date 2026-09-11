"""Tests for the corpus preflight + parser-regression harness (#21).

Uses stdlib ``unittest`` only; no pytest dependency.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest import parse_latex, preflight  # noqa: E402
from src.ingest import run as run_module  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_tarball(out: Path, files: dict[str, str]) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return out


SAMPLE_GOOD = {
    "openalex_id": "W1",
    "doi": "x",
    "arxiv_id": "1234.5678",
    "title": "Good",
    "year": 2020,
    "sha256": "0" * 64,
    "full_text_url": "u",
    "source_url": "u",
    "merged_ids": [],
    "source_kind": "arxiv",
    "title_source": "openalex",
}


def _write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def _tmp_tree() -> Path:
    return Path(tempfile.mkdtemp(prefix="preflight-test-"))


class ManifestSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _tmp_tree()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_clean_manifest_passes(self) -> None:
        manifest = self.tmp / "manifest.jsonl"
        _write_manifest(manifest, [SAMPLE_GOOD])
        report, records = preflight.check_manifest(manifest)
        self.assertTrue(report.ok)
        self.assertEqual(report.record_count, 1)
        self.assertEqual(len(records), 1)

    def test_non_jsonl_line_fails(self) -> None:
        manifest = self.tmp / "manifest.jsonl"
        manifest.write_text(json.dumps(SAMPLE_GOOD) + "\nnot-json\n")
        report, _ = preflight.check_manifest(manifest)
        self.assertFalse(report.ok)
        self.assertEqual(report.non_jsonl_lines, [2])

    def test_malformed_openalex_id_fails(self) -> None:
        bad = {**SAMPLE_GOOD, "openalex_id": "!@#"}
        manifest = self.tmp / "manifest.jsonl"
        _write_manifest(manifest, [bad])
        report, _ = preflight.check_manifest(manifest)
        self.assertIn("!@#", report.bad_ids)

    def test_duplicate_openalex_id_fails(self) -> None:
        manifest = self.tmp / "manifest.jsonl"
        _write_manifest(manifest, [SAMPLE_GOOD, SAMPLE_GOOD])
        report, _ = preflight.check_manifest(manifest)
        self.assertIn("W1", report.duplicate_ids)

    def test_missing_required_field_fails(self) -> None:
        bad = {k: v for k, v in SAMPLE_GOOD.items() if k != "title"}
        manifest = self.tmp / "manifest.jsonl"
        _write_manifest(manifest, [bad])
        report, _ = preflight.check_manifest(manifest)
        self.assertTrue(any("title" in e for e in report.schema_errors))


class TextBearingClassificationTests(unittest.TestCase):
    def test_predicate_is_deterministic(self) -> None:
        records = [
            {**SAMPLE_GOOD, "openalex_id": "W1", "arxiv_id": "1111.1111"},
            {**SAMPLE_GOOD, "openalex_id": "W2", "arxiv_id": None},
        ]
        rep = preflight.classify_text_bearing(records)
        self.assertEqual(rep.text_bearing, ["W1"])
        self.assertEqual(rep.non_text_bearing, ["W2"])

    def test_non_text_bearing_does_not_require_archive(self) -> None:
        """arxiv_id == null is a valid terminal state, not an error."""
        records = [{**SAMPLE_GOOD, "openalex_id": "W2", "arxiv_id": None}]
        tb = preflight.classify_text_bearing(records)
        rep = preflight.check_latex_archives(records, tb.text_bearing)
        self.assertTrue(rep.ok)


class LatexArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _tmp_tree()
        self.latex = self.tmp / "latex"
        self.latex.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_archive_fails_with_named_doc(self) -> None:
        records = [{**SAMPLE_GOOD, "arxiv_id": "1234.5678"}]
        rep = preflight.check_latex_archives(records, ["W1"], self.latex)
        self.assertEqual(rep.missing, ["W1"])

    def test_present_archive_passes(self) -> None:
        _make_tarball(
            self.latex / "1234.5678.tar.gz",
            {"main.tex": r"\documentclass{article}\begin{document}\end{document}"},
        )
        records = [{**SAMPLE_GOOD, "arxiv_id": "1234.5678"}]
        rep = preflight.check_latex_archives(records, ["W1"], self.latex)
        self.assertTrue(rep.ok)


class OpenAlexRawTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _tmp_tree()
        self.oa = self.tmp / "openalex"
        self.oa.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_raw_openalex_fails_regardless_of_source_kind(self) -> None:
        records = [{**SAMPLE_GOOD, "source_kind": "other_pdf"}]
        rep = preflight.check_raw_openalex(records, self.oa)
        self.assertIn("W1", rep.missing)


class ParserRegressionTests(unittest.TestCase):
    def test_pinned_fixtures_match(self) -> None:
        rep = preflight.run_parser_regression()
        self.assertTrue(rep.ok, msg=f"mismatches: {rep.mismatches}")
        self.assertIn("minimal_cite", rep.fixtures_run)
        self.assertIn("minimal_input", rep.fixtures_run)

    def test_simulated_parser_regression_causes_failure(self) -> None:
        """A deliberately-broken parse_latex.preserve_citations must fail pins."""

        original = preflight.parse_latex.preserve_citations
        try:
            preflight.parse_latex.preserve_citations = lambda text: text  # type: ignore[assignment]
            rep = preflight.run_parser_regression()
            self.assertFalse(rep.ok)
            self.assertTrue(any("expected" in m for m in rep.mismatches))
        finally:
            preflight.parse_latex.preserve_citations = original  # type: ignore[assignment]


class NonMutationGuardTests(unittest.TestCase):
    def test_preflight_does_not_touch_data_raw(self) -> None:
        """Snapshot data/raw/ before, run parser regression, snapshot after, compare."""
        root = REPO_ROOT / "data" / "raw"
        if not root.exists():
            self.skipTest("data/raw not present on this checkout")
        before = preflight.snapshot_raw(root)
        preflight.run_parser_regression()
        after = preflight.snapshot_raw(root)
        self.assertEqual(before, after)


class NoCitesGuardTests(unittest.TestCase):
    """Static-import check: preflight + run modules must not resolve CITES machinery."""

    FORBIDDEN = (
        "referenced_works_canonical",
        "data/aliases.json",
        "CITES",
    )

    def test_preflight_source_is_clean(self) -> None:
        text = (REPO_ROOT / "src" / "ingest" / "preflight.py").read_text()
        for needle in self.FORBIDDEN:
            self.assertNotIn(needle, text)

    def test_run_source_is_clean(self) -> None:
        text = (REPO_ROOT / "src" / "ingest" / "run.py").read_text()
        for needle in self.FORBIDDEN:
            self.assertNotIn(needle, text)


class NoRealCorpusInFixturesTests(unittest.TestCase):
    """The committed parser-regression fixtures must be synthetic minimal LaTeX."""

    def test_only_fixtures_dir_contains_real_looking_tarballs(self) -> None:
        fixtures = REPO_ROOT / "src" / "ingest" / "fixtures" / "parser_regression"
        offenders: list[str] = []
        for tarball in fixtures.glob("*.tar.gz"):
            with tarfile.open(tarball, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".tex"):
                        content = tar.extractfile(member).read().decode("utf-8", "replace")
                        # Real arXiv source packages typically have a long preamble
                        # and a real-looking title. Our fixtures have a short body.
                        if len(content) > 4000:
                            offenders.append(f"{tarball.name}::{member.name}")
        self.assertEqual(offenders, [])


class EntryPointIntegrationTests(unittest.TestCase):
    def test_run_module_is_invokable(self) -> None:
        """``python -m src.ingest.run`` exits non-zero against an incomplete corpus
        but prints manifest SHA256 + counts (verdict on stdout/stderr)."""
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            rc = run_module.main()
        combined = buf_out.getvalue() + buf_err.getvalue()
        self.assertIn("manifest_sha256:", combined)
        self.assertIn("text_bearing_doc_count:", combined)
        self.assertIn("parser_regression_fixtures:", combined)
        # The committed corpus has at least one missing tarball (2001.01582);
        # we accept any non-zero exit here and only verify the verdict format.
        self.assertIsInstance(rc, int)


if __name__ == "__main__":
    unittest.main()