"""Corpus preflight + parser-regression harness (impl-spec §4).

Source-capability-gated text-bearing membership (fifth pass):
membership depends on the tracked ``data/fulltext_manifest.jsonl`` row
having ``payload == "gzip"``, not on local filesystem state.

Pure checks. No side effects under ``data/`` or ``data/derived/``.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from src.ingest import parse_latex

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest.jsonl"
FULLTEXT_MANIFEST = ROOT / "data" / "fulltext_manifest.jsonl"
RAW_OPENALEX = ROOT / "data" / "raw" / "openalex"
RAW_LATEX = ROOT / "data" / "raw" / "latex"
FIXTURES = ROOT / "src" / "ingest" / "fixtures" / "parser_regression"
PINS_FILE = FIXTURES / "pins.json"

OPENALEX_ID_RE = re.compile(r"^W[0-9]+$")

REQUIRED_FIELDS = (
    "openalex_id",
    "doi",
    "arxiv_id",
    "source_kind",
    "title",
    "year",
    "sha256",
    "full_text_url",
    "source_url",
    "merged_ids",
    "title_source",
)

REQUIRED_FULLTEXT_FIELDS = ("arxiv_id", "path", "payload")


@dataclass
class ManifestReport:
    sha256: str
    record_count: int
    openalex_ids: list[str] = field(default_factory=list)
    bad_ids: list[str] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)
    non_jsonl_lines: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.bad_ids
            or self.duplicate_ids
            or self.schema_errors
            or self.non_jsonl_lines
        )


@dataclass
class FulltextReport:
    duplicate_ids: list[str] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)
    non_jsonl_lines: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.duplicate_ids or self.schema_errors or self.non_jsonl_lines)


@dataclass
class SourceCapability:
    by_arxiv_id: dict[str, dict]
    missing_manifest_arxiv_ids: list[str] = field(default_factory=list)


@dataclass
class RawOpenAlexReport:
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing


@dataclass
class TextBearingReport:
    text_bearing: list[str] = field(default_factory=list)
    pdf_payload: list[str] = field(default_factory=list)
    non_text_bearing: list[str] = field(default_factory=list)
    unresolved_capability: list[str] = field(default_factory=list)


@dataclass
class LatexArchiveReport:
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing


@dataclass
class RegressionReport:
    fixtures_run: list[str] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _iter_jsonl(path: Path):
    with path.open("rb") as fh:
        for lineno, raw in enumerate(fh, start=1):
            yield lineno, raw


def check_manifest(path: Path = MANIFEST) -> tuple[ManifestReport, list[dict]]:
    raw = path.read_bytes()
    report = ManifestReport(sha256=sha256_bytes(raw), record_count=0)
    records: list[dict] = []
    seen: dict[str, int] = {}

    for lineno, line in _iter_jsonl(path):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        try:
            record = json.loads(line_stripped)
        except json.JSONDecodeError:
            report.non_jsonl_lines.append(lineno)
            continue

        records.append(record)
        oid = record.get("openalex_id")
        if not isinstance(oid, str) or not OPENALEX_ID_RE.match(oid):
            report.bad_ids.append(str(oid))
        if oid in seen:
            report.duplicate_ids.append(oid)
        seen[oid] = lineno
        report.openalex_ids.append(oid)

        for field_name in REQUIRED_FIELDS:
            if field_name not in record:
                report.schema_errors.append(f"line {lineno}: missing {field_name}")

    report.record_count = len(records)
    return report, records


def check_fulltext_manifest(path: Path = FULLTEXT_MANIFEST) -> tuple[FulltextReport, dict[str, dict]]:
    report = FulltextReport()
    by_id: dict[str, dict] = {}
    for lineno, line in _iter_jsonl(path):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        try:
            row = json.loads(line_stripped)
        except json.JSONDecodeError:
            report.non_jsonl_lines.append(lineno)
            continue
        arxiv_id = row.get("arxiv_id")
        if arxiv_id in by_id:
            report.duplicate_ids.append(arxiv_id)
        for field_name in REQUIRED_FULLTEXT_FIELDS:
            if field_name not in row:
                report.schema_errors.append(f"line {lineno}: missing {field_name}")
        by_id[arxiv_id] = row
    return report, by_id


def load_source_capability(path: Path = FULLTEXT_MANIFEST) -> tuple[FulltextReport, SourceCapability]:
    ft_report, by_id = check_fulltext_manifest(path)
    return ft_report, SourceCapability(by_arxiv_id=by_id)


def check_raw_openalex(records: list[dict], raw_dir: Path = RAW_OPENALEX) -> RawOpenAlexReport:
    report = RawOpenAlexReport()
    for record in records:
        oid = record["openalex_id"]
        if not (raw_dir / f"{oid}.json").exists():
            report.missing.append(oid)
    return report


def classify_text_bearing(
    records: list[dict],
    capability: SourceCapability,
) -> TextBearingReport:
    """Source-capability-gated classification (impl-spec fifth pass).

    - ``arxiv_id is not None`` and tracked ``payload == "gzip"`` -> v1 text-bearing.
    - ``arxiv_id is not None`` and tracked ``payload == "pdf"`` -> valid canonical
      non-text-bearing in v1 (NOT an error).
    - ``arxiv_id is not None`` and no tracked row -> unresolved capability
      (downstream caller fails the preflight).
    - ``arxiv_id is None`` -> canonical non-text-bearing.
    """
    report = TextBearingReport()
    for record in records:
        oid = record["openalex_id"]
        arxiv_id = record.get("arxiv_id")
        if arxiv_id is None:
            report.non_text_bearing.append(oid)
            continue
        row = capability.by_arxiv_id.get(arxiv_id)
        if row is None:
            report.unresolved_capability.append(oid)
            continue
        payload = row.get("payload")
        if payload == "gzip":
            report.text_bearing.append(oid)
        elif payload == "pdf":
            report.pdf_payload.append(oid)
        else:
            report.unresolved_capability.append(oid)
    report.text_bearing.sort()
    report.pdf_payload.sort()
    report.non_text_bearing.sort()
    report.unresolved_capability.sort()
    return report


def check_latex_archives(
    records: list[dict],
    text_bearing: list[str],
    latex_dir: Path = RAW_LATEX,
) -> LatexArchiveReport:
    by_id = {r["openalex_id"]: r for r in records}
    report = LatexArchiveReport()
    for oid in text_bearing:
        arxiv_id = by_id[oid]["arxiv_id"]
        path = latex_dir / f"{arxiv_id}.tar.gz"
        if not path.exists() or not path.is_file():
            report.missing.append(oid)
            continue
        try:
            with tarfile.open(path, "r:gz"):
                pass
        except (tarfile.TarError, OSError):
            report.missing.append(oid)
    return report


def snapshot_raw(root: Path = ROOT / "data" / "raw") -> dict[str, tuple[float, str]]:
    snap: dict[str, tuple[float, str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            snap[str(path.relative_to(root))] = (
                path.stat().st_mtime,
                sha256_file(path),
            )
    return snap


def run_parser_regression(
    fixtures_dir: Path = FIXTURES,
    pins_file: Path = PINS_FILE,
) -> RegressionReport:
    """Run the existing parse_latex pipeline against synthetic fixtures.

    Pins the cleaned text SHA256 for each fixture. Does NOT mutate the
    fixtures directory or any file under ``data/``.
    """
    report = RegressionReport()
    pins = json.loads(pins_file.read_text()) if pins_file.exists() else {}
    for tarball in sorted(fixtures_dir.glob("*.tar.gz")):
        fid = tarball.name.removesuffix(".tar.gz")
        tex = parse_latex.read_tex_files(tarball)
        main = parse_latex.choose_main_file(tex)
        expanded, _warnings = parse_latex.resolve_includes(tex, main)
        cleaned = parse_latex.remove_comments(expanded)
        text = parse_latex.preserve_citations(cleaned)
        observed = sha256_bytes(text.encode())

        expected = pins.get(fid, {}).get("sha256")
        if expected is None:
            report.mismatches.append(f"{fid}: no pinned sha256")
        elif observed != expected:
            report.mismatches.append(f"{fid}: expected {expected}, got {observed}")

        if "[[CITE:" not in text and (pins.get(fid, {}).get("cite_count", 0) or 0) == 0:
            pass  # template fixture without citations is fine
        elif "[[CITE:" not in text:
            report.mismatches.append(f"{fid}: missing [[CITE:...]] marker")

        report.fixtures_run.append(fid)
    return report