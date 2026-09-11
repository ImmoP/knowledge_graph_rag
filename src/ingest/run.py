"""Run the corpus preflight + parser-regression harness (impl-spec §4).

    python -m src.ingest.run

Writes verdict to stdout/stderr only. Does NOT modify ``data/manifest.jsonl``
or write any sidecar under ``data/`` / ``data/derived/``.
"""

from __future__ import annotations

import sys

from src.ingest import preflight


def main() -> int:
    manifest_report, records = preflight.check_manifest()
    print(f"manifest_sha256: {manifest_report.sha256}")
    print(f"manifest_records: {manifest_report.record_count}")

    failures: list[str] = []

    if not manifest_report.ok:
        failures.append("manifest schema/integrity failed")
        if manifest_report.non_jsonl_lines:
            print(f"  non_jsonl_lines: {manifest_report.non_jsonl_lines}", file=sys.stderr)
        if manifest_report.bad_ids:
            print(f"  bad_openalex_ids: {manifest_report.bad_ids}", file=sys.stderr)
        if manifest_report.duplicate_ids:
            print(f"  duplicate_openalex_ids: {manifest_report.duplicate_ids}", file=sys.stderr)
        if manifest_report.schema_errors:
            print(f"  schema_errors: {manifest_report.schema_errors}", file=sys.stderr)

    ft_report, capability = preflight.load_source_capability()
    print(f"fulltext_manifest_rows: {len(capability.by_arxiv_id)}")
    if not ft_report.ok:
        failures.append("fulltext manifest integrity failed")
        if ft_report.non_jsonl_lines:
            print(f"  fulltext_non_jsonl_lines: {ft_report.non_jsonl_lines}", file=sys.stderr)
        if ft_report.duplicate_ids:
            print(f"  fulltext_duplicate_arxiv_ids: {ft_report.duplicate_ids}", file=sys.stderr)
        if ft_report.schema_errors:
            print(f"  fulltext_schema_errors: {ft_report.schema_errors}", file=sys.stderr)

    oa_report = preflight.check_raw_openalex(records)
    print(f"raw_openalex_missing: {len(oa_report.missing)}")
    if not oa_report.ok:
        failures.append("openalex raw artefacts missing")
        print(f"  missing: {oa_report.missing}", file=sys.stderr)

    tb_report = preflight.classify_text_bearing(records, capability)
    print(f"text_bearing_doc_count: {len(tb_report.text_bearing)}")
    print(f"pdf_payload_doc_count: {len(tb_report.pdf_payload)}")
    print(f"non_text_bearing_doc_count: {len(tb_report.non_text_bearing)}")
    print(f"unresolved_source_capability_count: {len(tb_report.unresolved_capability)}")
    if tb_report.unresolved_capability:
        failures.append("unresolved source capability")
        print(f"  unresolved_doc_ids: {tb_report.unresolved_capability}", file=sys.stderr)

    latex_report = preflight.check_latex_archives(records, tb_report.text_bearing)
    print(f"text_bearing_with_missing_archive: {len(latex_report.missing)}")
    if not latex_report.ok:
        failures.append("latex archive input-integrity failure")
        print(f"  missing_doc_ids: {latex_report.missing}", file=sys.stderr)

    regression = preflight.run_parser_regression()
    print(f"parser_regression_fixtures: {len(regression.fixtures_run)}")
    if not regression.ok:
        failures.append("parser regression failure")
        print(f"  mismatches: {regression.mismatches}", file=sys.stderr)

    if failures:
        print(f"PREFLIGHT FAIL: {failures}", file=sys.stderr)
        return 1
    print("PREFLIGHT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())