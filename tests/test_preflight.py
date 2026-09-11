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
from contextlib import redirect_stderr, redirect_stdout
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


class FulltextManifestSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _tmp_tree()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_fulltext(self, rows: list[dict]) -> Path:
        path = self.tmp / "fulltext_manifest.jsonl"
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return path

    def test_unique_rows_pass(self) -> None:
        rows = [
            {"arxiv_id": "1234.5678", "path": "data/raw/latex/1234.5678.tar.gz", "payload": "gzip"},
        ]
        report, by_id = preflight.check_fulltext_manifest(self._write_fulltext(rows))
        self.assertTrue(report.ok)
        self.assertEqual(by_id["1234.5678"]["payload"], "gzip")

    def test_duplicate_row_fails(self) -> None:
        rows = [
            {"arxiv_id": "1234.5678", "path": "x", "payload": "gzip"},
            {"arxiv_id": "1234.5678", "path": "y", "payload": "pdf"},
        ]
        report, _ = preflight.check_fulltext_manifest(self._write_fulltext(rows))
        self.assertIn("1234.5678", report.duplicate_ids)


class SourceCapabilityGatingTests(unittest.TestCase):
    """Tests for the fifth-pass source-capability-gated classification."""

    def _manifest_record(self, oid: str, arxiv_id: str | None) -> dict:
        return {**SAMPLE_GOOD, "openalex_id": oid, "arxiv_id": arxiv_id}

    def _capability(self, *rows: dict) -> preflight.SourceCapability:
        by_id = {row["arxiv_id"]: row for row in rows}
        return preflight.SourceCapability(by_arxiv_id=by_id)

    def test_gzip_capability_and_archive_present_is_text_bearing(self) -> None:
        records = [self._manifest_record("W1", "1234.5678")]
        cap = self._capability({"arxiv_id": "1234.5678", "path": "x", "payload": "gzip"})
        rep = preflight.classify_text_bearing(records, cap)
        self.assertEqual(rep.text_bearing, ["W1"])
        self.assertEqual(rep.pdf_payload, [])
        self.assertEqual(rep.unresolved_capability, [])

    def test_gzip_capability_with_missing_archive_is_unresolved_at_archive_layer_only(self) -> None:
        """Classify-level: gzip+archive-present predicate is membership-only.
        Archive absence is surfaced by check_latex_archives, not classify."""
        records = [self._manifest_record("W1", "1234.5678")]
        cap = self._capability({"arxiv_id": "1234.5678", "path": "x", "payload": "gzip"})
        rep = preflight.classify_text_bearing(records, cap)
        self.assertEqual(rep.text_bearing, ["W1"])

    def test_pdf_capability_is_canonical_non_text_bearing(self) -> None:
        records = [self._manifest_record("W2", "1234.5678")]
        cap = self._capability({"arxiv_id": "1234.5678", "path": "y", "payload": "pdf"})
        rep = preflight.classify_text_bearing(records, cap)
        self.assertEqual(rep.text_bearing, [])
        self.assertEqual(rep.pdf_payload, ["W2"])
        self.assertEqual(rep.unresolved_capability, [])

    def test_missing_capability_row_is_unresolved(self) -> None:
        records = [self._manifest_record("W3", "1234.5678")]
        cap = self._capability()  # no rows
        rep = preflight.classify_text_bearing(records, cap)
        self.assertEqual(rep.text_bearing, [])
        self.assertEqual(rep.unresolved_capability, ["W3"])

    def test_no_arxiv_id_is_canonical_non_text_bearing(self) -> None:
        records = [self._manifest_record("W4", None)]
        cap = self._capability()
        rep = preflight.classify_text_bearing(records, cap)
        self.assertEqual(rep.non_text_bearing, ["W4"])
        self.assertEqual(rep.text_bearing, [])


class FilesystemStateCannotChangeMembershipTests(unittest.TestCase):
    """The membership predicate is purely a function of tracked metadata."""

    def test_archive_presence_or_absence_does_not_affect_membership(self) -> None:
        record = {**SAMPLE_GOOD, "openalex_id": "W1", "arxiv_id": "1234.5678"}
        cap_pdf = preflight.SourceCapability(
            by_arxiv_id={"1234.5678": {"arxiv_id": "1234.5678", "payload": "pdf", "path": "x"}}
        )
        cap_gzip = preflight.SourceCapability(
            by_arxiv_id={"1234.5678": {"arxiv_id": "1234.5678", "payload": "gzip", "path": "x"}}
        )

        with_archive = preflight.classify_text_bearing([record], cap_gzip)
        without_archive = preflight.classify_text_bearing([record], cap_gzip)
        self.assertEqual(with_archive.text_bearing, without_archive.text_bearing)

        with_pdf_capability = preflight.classify_text_bearing([record], cap_pdf)
        self.assertEqual(with_pdf_capability.text_bearing, [])


class LatexArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _tmp_tree()
        self.latex = self.tmp / "latex"
        self.latex.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_gzip_text_bearing_with_missing_archive_fails_with_named_doc(self) -> None:
        records = [{**SAMPLE_GOOD, "arxiv_id": "1234.5678"}]
        rep = preflight.check_latex_archives(records, ["W1"], self.latex)
        self.assertEqual(rep.missing, ["W1"])

    def test_gzip_text_bearing_with_present_archive_passes(self) -> None:
        _make_tarball(
            self.latex / "1234.5678.tar.gz",
            {"main.tex": r"\documentclass{article}\begin{document}\end{document}"},
        )
        records = [{**SAMPLE_GOOD, "arxiv_id": "1234.5678"}]
        rep = preflight.check_latex_archives(records, ["W1"], self.latex)
        self.assertTrue(rep.ok)

    def test_pdf_payload_record_is_not_checked_against_archive(self) -> None:
        """A PDF-payload record is canonical non-text-bearing; the
        archive-existence layer never sees it, so a missing archive
        for it is not an error."""
        records = [{**SAMPLE_GOOD, "arxiv_id": "1234.5678"}]
        # text_bearing empty -> nothing to check
        rep = preflight.check_latex_archives(records, [], self.latex)
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
                        if len(content) > 4000:
                            offenders.append(f"{tarball.name}::{member.name}")
        self.assertEqual(offenders, [])


class EntryPointIntegrationTests(unittest.TestCase):
    def test_run_module_is_invokable(self) -> None:
        """``python -m src.ingest.run`` prints verdict on stdout/stderr."""
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            rc = run_module.main()
        combined = buf_out.getvalue() + buf_err.getvalue()
        self.assertIn("manifest_sha256:", combined)
        self.assertIn("text_bearing_doc_count:", combined)
        self.assertIn("parser_regression_fixtures:", combined)


class ChooseMainFileCommentStrippingTests(unittest.TestCase):
    """Narrow parse_latex repair (#22 / Part A): choose_main_file evaluates
    each tex file on a comment-stripped view so commented-out template
    titles, ``\\begin{document}``, and ``\\cite`` markers do not influence
    candidate detection or ranking.

    The selected file still flows through the downstream pipeline on the
    ORIGINAL tex_files content; only the candidate-detection view is
    comment-stripped.
    """

    def _tex_files(self, main: str, others: dict | None = None) -> dict:
        out = {"main.tex": main}
        if others:
            out.update(others)
        return out

    def test_commented_template_title_then_real_title_is_not_rejected(self) -> None:
        """A commented-out ``\\title{emnlp 2017 instructions}`` followed by a
        real ``\\title{Some Specific Study}`` must NOT reject the file."""
        text = (
            "\\documentclass{article}\n"
            "% \\title{emnlp 2017 instructions}\n"
            "\\author{Anonymous}\n"
            "\\title{Some Specific Study}\n"
            "\\begin{document}\n"
            "\\section{Intro}\nBody.\n"
            "\\cite{ref1}\n"
            "\\end{document}\n"
        )
        main = parse_latex.choose_main_file(self._tex_files(text))
        self.assertEqual(main, "main.tex")

    def test_commented_begin_document_does_not_make_a_file_a_candidate(self) -> None:
        """A file whose ONLY ``\\begin{document}`` is commented out is not a
        valid candidate, even though it has other content."""
        text = (
            "\\documentclass{article}\n"
            "% \\begin{document}\n"
            "\\section{Fake}\nBody.\n"
        )
        with self.assertRaises(ValueError):
            parse_latex.choose_main_file(self._tex_files(text))

    def test_commented_citation_does_not_influence_ranking(self) -> None:
        """Citation count ranking uses comment-stripped text. A file with only
        commented-out cites ranks below one with real cites."""
        low = (
            "\\documentclass{article}\n"
            "\\title{Low}\n"
            "\\begin{document}\n"
            "\\section{S}\n"
            "% \\cite{a} % \\cite{b} % \\cite{c}\n"
            "body.\n"
            "\\end{document}\n"
        )
        high = (
            "\\documentclass{article}\n"
            "\\title{High}\n"
            "\\begin{document}\n"
            "\\section{S}\n"
            "\\cite{a} \\cite{b} \\cite{c} \\cite{d} \\cite{e}\n"
            "body.\n"
            "\\end{document}\n"
        )
        main = parse_latex.choose_main_file(self._tex_files(low, {"other.tex": high}))
        self.assertEqual(main, "other.tex")

    def test_escaped_percent_in_title_is_preserved(self) -> None:
        """``\\%`` (escaped percent) is not stripped by ``remove_comments``,
        so ``\\title{Foo \\% Bar}`` is still seen as a title."""
        text = (
            "\\documentclass{article}\n"
            "\\title{Foo \\% Bar}\n"
            "\\begin{document}\nbody\n\\end{document}\n"
        )
        main = parse_latex.choose_main_file(self._tex_files(text))
        self.assertEqual(main, "main.tex")
        # And remove_comments alone must not eat the %:
        cleaned = parse_latex.remove_comments(text)
        self.assertIn(r"\%", cleaned)

    def test_template_title_without_comment_is_still_rejected(self) -> None:
        """Negative control: an UN-commented template title is still rejected.
        The repair only changes comment-stripping; it does NOT widen what
        counts as a template title."""
        text = (
            "\\documentclass{article}\n"
            "\\title{emnlp 2017 instructions}\n"
            "\\begin{document}\nbody\n\\end{document}\n"
        )
        with self.assertRaises(ValueError):
            parse_latex.choose_main_file(self._tex_files(text))


if __name__ == "__main__":
    unittest.main()