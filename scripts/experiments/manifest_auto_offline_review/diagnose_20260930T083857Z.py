#!/usr/bin/env python3
"""Offline diagnosis for the manifest-auto 10-chunk run ``20260930T083857Z-c69654``.

Reads only on-disk artifacts:
- the run directory under ~/migration-incoming/manifest-auto-autopilot-live/
- the frozen benchmark chunks
- the existing experimental whitespace repair (``src/extract/experimental_quote_repair``)

No network calls. No new model calls. No mutation of live data.

Outputs:
- a structured JSON diagnosis at the configured output root
- a stderr summary of the seven rejects and five empty chunks
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import re
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.extract.align import (  # noqa: E402
    RejectReason,
    _enumerate_occurrences,
)
from src.extract.experimental_quote_repair import repair_quote  # noqa: E402


RUN_DIR = Path(
    os.environ.get(
        "MANIFEST_AUTO_RUN_DIR",
        "/home/immo/migration-incoming/manifest-auto-autopilot-live/20260930T083857Z-c69654",
    )
)
CHUNKS_JSONL = REPO_ROOT / "src" / "benchmark" / "dataset" / "benchmark_chunks.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "experiments" / "manifest-auto-review"


@dataclasses.dataclass
class ChunkView:
    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int
    chunk_index: int
    section_path: list[str]



def _utc_compact() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def _load_chunks() -> dict[str, ChunkView]:
    out: dict[str, ChunkView] = {}
    for line in CHUNKS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["chunk_id"]] = ChunkView(
            chunk_id=rec["chunk_id"],
            doc_id=rec["doc_id"],
            text=rec["text"],
            char_start=rec["char_start"],
            char_end=rec["char_end"],
            chunk_index=rec.get("chunk_index", -1),
            section_path=rec.get("section_path") or [],
        )
    return out




def diagnose() -> dict:
    chunks = _load_chunks()
    run_summary = json.loads((RUN_DIR / "run_summary.json").read_text(encoding="utf-8"))
    extractions = _read_jsonl(RUN_DIR / "extractions.jsonl")
    rejects = _read_jsonl(RUN_DIR / "extractions.rejects.jsonl")

    # Index extractions by chunk and kind
    by_chunk: dict[str, dict[str, list[dict]]] = {}
    chunk_statuses: dict[str, dict] = {}
    for rec in extractions:
        cid = rec.get("chunk_id")
        kind = rec.get("kind")
        if not cid:
            continue
        if kind == "chunk_status":
            chunk_statuses[cid] = rec
        else:
            by_chunk.setdefault(cid, {}).setdefault(kind, []).append(rec)

    rejects_by_chunk: dict[str, list[dict]] = {}
    for rec in rejects:
        rejects_by_chunk.setdefault(rec["chunk_id"], []).append(rec)

    target_chunks = run_summary["chunks"]
    # Sort chunks into technical buckets
    bucket_complete_with_records: list[str] = []
    bucket_complete_empty: list[str] = []
    bucket_incomplete_partial: list[str] = []
    for cid in target_chunks:
        status = chunk_statuses.get(cid, {})
        mentions = by_chunk.get(cid, {}).get("mention", [])
        assertions = by_chunk.get(cid, {}).get("assertion", [])
        if status.get("extraction_complete") is True:
            if mentions or assertions:
                bucket_complete_with_records.append(cid)
            else:
                bucket_complete_empty.append(cid)
        elif mentions or assertions:
            bucket_incomplete_partial.append(cid)

    # ---- 1. Per-chunk summary ----
    chunk_summaries: list[dict] = []
    for cid in target_chunks:
        view = chunks.get(cid)
        status = chunk_statuses.get(cid, {})
        mentions = by_chunk.get(cid, {}).get("mention", [])
        assertions = by_chunk.get(cid, {}).get("assertion", [])
        rec_rejects = rejects_by_chunk.get(cid, [])
        # Bucket membership is based on accepted records only. Rejects are
        # reported separately and do not turn an empty chunk into a
        # with-records chunk.
        if status.get("extraction_complete") is True:
            bucket = "complete_with_records" if mentions or assertions else "complete_empty"
        else:
            bucket = "incomplete_partial" if mentions or assertions else "incomplete_empty"
        chunk_summaries.append({
            "chunk_id": cid,
            "doc_id": view.doc_id if view else None,
            "section_path": view.section_path if view else [],
            "extraction_complete": status.get("extraction_complete"),
            "entity_attempts": status.get("phases", {}).get("entity", {}).get("attempt_count"),
            "entity_mentions": len(mentions),
            "entity_reject_count": sum(1 for r in rec_rejects if r.get("phase") == "entity"),
            "relation_attempts": status.get("phases", {}).get("relation", {}).get("attempt_count"),
            "relation_assertions": len(assertions),
            "relation_reject_count": sum(1 for r in rec_rejects if r.get("phase") == "relation"),
            "reject_reasons": sorted({
                r.get("reason") for r in rec_rejects if r.get("reason")
            }),
            "bucket": bucket,
        })

    # ---- 2. Reject deep-dive ----
    reject_details: list[dict] = []
    for rec in rejects:
        cid = rec["chunk_id"]
        view = chunks.get(cid)
        phase = rec.get("phase")
        reason = rec.get("aligner_reason") or rec.get("reason")
        offending = rec.get("offending_record", {}) or {}
        offending_value = offending.get("offending_value")

        entry: dict = {
            "chunk_id": cid,
            "phase": phase,
            "reason": reason,
            "offending_record": offending,
            "section_path": view.section_path if view else [],
            "chunk_text_excerpt": (
                textwrap.shorten(view.text.replace("\n", " "), width=320, placeholder=" ...")
                if view else None
            ),
        }

        if reason == RejectReason.INVALID_OCCURRENCE_INDEX.value:
            # Diagnose: candidate's literal is missing from chunk?
            # Determine which candidate triggered: walk extractions.
            if phase == "entity":
                # We need the literal that produced this reject. The
                # runner stores the record id at offending.candidate_id
                # but in this corpus it is None. Identify candidates by
                # indexing against the raw record list and matching the
                # offending_value occurrence_index field.
                offending_index = offending.get("offending_value")
                # Find entities whose occurrence_index equals this value.
                # However, in this run, all invalid occurrence_index
                # rejects come from the cached raw responses; the entity
                # list in extractions.jsonl is filtered.
                # We must inspect ALL candidates by reconstructing from
                # the cached responses; those are not available offline.
                # Diagnose purely from the available record shapes and
                # the chunk text.
                if view is not None:
                    # Re-run the aligner against the EMPTY mention list to
                    # see how many surface forms are missing/valid; we
                    # cannot recover the original candidates here.
                    entry["diagnostic"] = {
                        "note": (
                            "Entity-phase INVALID_OCCURRENCE_INDEX; raw "
                            "model response is not persisted in this run, "
                            "so the offending literal is recoverable only "
                            "by re-running the cached response through the "
                            "aligner. Offending value (occurrence_index "
                            "field) is "
                            f"{offending_index!r}; the model emitted an "
                            "index that does not address a valid occurrence "
                            "of its surface_form in chunk_text."
                        ),
                    }
            elif phase == "relation":
                entry["diagnostic"] = {
                    "note": (
                        "Relation-phase INVALID_OCCURRENCE_INDEX; raw "
                        "model response is not persisted in this run, so "
                        "the offending evidence_quote is recoverable only "
                        "by re-running the cached response."
                    ),
                }
        elif reason == RejectReason.ENDPOINT_NOT_CONTAINED.value:
            ofv = offending_value or {}
            quote = ofv.get("evidence_quote")
            subj = ofv.get("subject_ref")
            obj = ofv.get("object_ref")
            diagnosis: dict = {"evidence_quote": quote, "subject_ref": subj, "object_ref": obj}
            if view is not None and quote:
                starts = _enumerate_occurrences(view.text, quote)
                diagnosis["exact_quote_present"] = bool(starts)
                diagnosis["exact_quote_span"] = (
                    [starts[0], starts[0] + len(quote)] if starts else None
                )
                # Accepted mention records already carry canonical evidence spans.
                # Use those persisted spans directly: occurrence_index is not
                # retained in extractions.jsonl, so re-aligning repeated literals
                # could incorrectly lose an otherwise accepted mention.
                mentions = by_chunk.get(cid, {}).get("mention", [])
                aligned_by_id = {
                    m.get("mention_id"): (
                        m.get("evidence", {}).get("char_start"),
                        m.get("evidence", {}).get("char_end"),
                    )
                    for m in mentions
                    if isinstance(m.get("mention_id"), str)
                    and isinstance(m.get("evidence"), dict)
                    and isinstance(m.get("evidence", {}).get("char_start"), int)
                    and isinstance(m.get("evidence", {}).get("char_end"), int)
                }
                # The rejected record refers to mention ids that we can
                # look up in aligned_by_id to see whether their spans
                # lie inside the evidence quote.
                rel_start, rel_end = (
                    (starts[0], starts[0] + len(quote)) if starts else (None, None)
                )
                diagnosis["relation_span"] = (
                    [rel_start, rel_end] if rel_start is not None else None
                )
                for endpoint_label, endpoint in (("subject_ref", subj), ("object_ref", obj)):
                    if not isinstance(endpoint, dict):
                        diagnosis[f"{endpoint_label}_resolved_span"] = None
                        continue
                    if endpoint.get("kind") == "mention":
                        span = aligned_by_id.get(endpoint.get("id"))
                        diagnosis[f"{endpoint_label}_resolved_span"] = span
                        if rel_start is not None and span is not None:
                            contained = rel_start <= span[0] and span[1] <= rel_end
                            diagnosis[f"{endpoint_label}_contained_in_evidence"] = contained
                        else:
                            diagnosis[f"{endpoint_label}_contained_in_evidence"] = False
                    else:
                        diagnosis[f"{endpoint_label}_resolved_span"] = "document"
                        diagnosis[f"{endpoint_label}_contained_in_evidence"] = True
            entry["diagnostic"] = diagnosis
        elif reason == RejectReason.ZERO_OCCURRENCES.value:
            literal = offending.get("offending_value") if isinstance(offending, dict) else None
            entry["offending_literal"] = literal
            diagnosis = {"literal": literal}
            if view is not None and isinstance(literal, str):
                # Identify a likely original span via the experimental
                # whitespace repair (no mutation of source; we only
                # report the verdict and the candidate span).
                repair = repair_quote(view.text, literal)
                diagnosis["repair_status"] = repair.status
                diagnosis["repaired_quote"] = repair.quote
                diagnosis["repaired_span"] = (
                    [repair.char_start, repair.char_end]
                    if repair.char_start is not None else None
                )
                # Compare raw lengths of literal vs. potential original span
                # and dump a char-level diff window for visual inspection.
                if repair.quote is not None and repair.char_start is not None:
                    start = max(0, repair.char_start - 30)
                    end = min(len(view.text), repair.char_end + 30)
                    original_window = view.text[start:end]
                    # Find longest contiguous literal-prefix match in the
                    # repair span vs. literal to characterise the diff.
                    def _common_prefix_len(a: str, b: str) -> int:
                        i = 0
                        while i < min(len(a), len(b)) and a[i] == b[i]:
                            i += 1
                        return i
                    def _common_suffix_len(a: str, b: str) -> int:
                        i = 0
                        while (
                            i < min(len(a), len(b))
                            and a[-1 - i] == b[-1 - i]
                        ):
                            i += 1
                        return i
                    pfx = _common_prefix_len(repair.quote, literal)
                    sfx = _common_suffix_len(repair.quote, literal)
                    diagnosis["diff_window"] = original_window
                    diagnosis["diff_common_prefix"] = pfx
                    diagnosis["diff_common_suffix"] = sfx
                    diagnosis["diff_diverging_original"] = repair.quote[pfx:len(repair.quote)-sfx if sfx else len(repair.quote)]
                    diagnosis["diff_diverging_literal"] = literal[pfx:len(literal)-sfx if sfx else len(literal)]
                    # Tag whether the divergence is purely whitespace
                    # (line breaks, tabs, multiple spaces).
                    diverging_original = diagnosis["diff_diverging_original"]
                    diverging_literal = diagnosis["diff_diverging_literal"]
                    diagnosis["divergence_is_pure_whitespace"] = bool(
                        diverging_original and diverging_literal and (
                            re.fullmatch(r"[ \t\r\n]+", diverging_original)
                            or re.fullmatch(r"[ \t\r\n]+", diverging_literal)
                        )
                    )
            entry["diagnostic"] = diagnosis
        else:
            entry["diagnostic"] = {"note": "Unhandled reason"}

        reject_details.append(entry)

    # ---- 3. Empty-chunk excerpts ----
    empty_excerpts: list[dict] = []
    for cid in bucket_complete_empty:
        view = chunks.get(cid)
        if view is None:
            continue
        snippet = view.text[:600]
        # Heuristic look for likely entity/relation surface forms: title-case
        # phrases, acronyms, dataset/method names. Strictly indicative.
        candidate_surfaces = _heuristic_candidates(snippet)
        empty_excerpts.append({
            "chunk_id": cid,
            "doc_id": view.doc_id,
            "section_path": view.section_path,
            "char_length": len(view.text),
            "excerpt_head": snippet,
            "heuristic_surface_candidates": candidate_surfaces,
            "assessment": (
                "Manuell heuristisch, nicht wissenschaftlich bestätigt: "
                "Chunk ist technisch vollständig (kein Reject, keine "
                "akzeptierte Mention/Assertion); das Modell hat im "
                "Entity-Schritt keinen Datensatz/Methode/Task emittiert "
                "und im Relations-Schritt keine Relation. Inhaltlich "
                "könnten extrahierbare Erwähnungen vorhanden sein — eine "
                "belastbare Aussage erfordert eine verblindete "
                "Fachbewertung oder ein Gold Set."
            ),
        })

    # ---- 4. Aggregate counters ----
    counter = {
        "n_chunks_total": len(target_chunks),
        "n_chunks_complete_with_records": len(bucket_complete_with_records),
        "n_chunks_complete_empty": len(bucket_complete_empty),
        "n_chunks_incomplete_partial": len(bucket_incomplete_partial),
        "n_rejects_total": len(rejects),
        "rejects_by_phase": {},
        "rejects_by_phase_and_reason": {},
    }
    for r in rejects:
        phase = r.get("phase", "?")
        reason = r.get("aligner_reason") or r.get("reason") or "?"
        counter["rejects_by_phase"][phase] = counter["rejects_by_phase"].get(phase, 0) + 1
        key = f"{phase}/{reason}"
        counter["rejects_by_phase_and_reason"][key] = (
            counter["rejects_by_phase_and_reason"].get(key, 0) + 1
        )

    # ---- 5. Identified-vs-hypothesis split ----
    confirmed_causes: list[dict] = []
    hypotheses: list[dict] = []
    for entry in reject_details:
        cause: dict = {
            "chunk_id": entry["chunk_id"],
            "phase": entry["phase"],
            "reason": entry["reason"],
        }
        if entry["reason"] == RejectReason.ENDPOINT_NOT_CONTAINED.value:
            diag = entry.get("diagnostic", {})
            confirmed_causes.append({
                **cause,
                "kind": "endpoint_outside_evidence",
                "offending_endpoint": _pick_outside_endpoint(diag),
                "evidence_quote_span": diag.get("relation_span"),
                "evidence_quote_present": diag.get("exact_quote_present"),
            })
        elif entry["reason"] == RejectReason.ZERO_OCCURRENCES.value:
            diag = entry.get("diagnostic", {})
            confirmed_causes.append({
                **cause,
                "kind": "quote_not_in_chunk",
                "literal": diag.get("literal"),
                "repair_status": diag.get("repair_status"),
                "repaired_span": diag.get("repaired_span"),
                "divergence_is_pure_whitespace": diag.get("divergence_is_pure_whitespace"),
            })
        else:
            # INVALID_OCCURRENCE_INDEX: cannot pinpoint literal without
            # cached raw response; mark as hypothesis with structured notes.
            hypotheses.append({
                **cause,
                "kind": "occurrence_index_invalid",
                "note": (
                    "Belegbar: das Modell hat einen occurrence_index "
                    "emittiert, der nicht in das Enumeratorergebnis des "
                    "betroffenen surface_form passt. Hypothese: ohne "
                    "gespeicherte Rohantwort ist nicht entscheidbar, ob "
                    "falsche Nummerierung (z. B. 1-basiert statt 0-basiert) "
                    "oder mehrfach vorkommender Text die Ursache ist. Die "
                    "Rohantworten sind im Run-Output leer "
                    "(raw_responses/ ist nicht befüllt worden); eine "
                    "Anfrage an die Rohantwort ist außerhalb dieses "
                    "Offline-Auftrags."
                ),
            })

    # ---- 6. Minimal fix proposals ----
    fixes: list[dict] = []
    # ZERO_OCCURRENCES: experimental repair could accept (offline verdict
    # added by the caller of this script).
    for entry in reject_details:
        if entry["reason"] != RejectReason.ZERO_OCCURRENCES.value:
            continue
        diag = entry.get("diagnostic", {})
        status = diag.get("repair_status")
        if status == "repaired" and diag.get("divergence_is_pure_whitespace"):
            fixes.append({
                "chunk_id": entry["chunk_id"],
                "phase": entry["phase"],
                "type": "experimental_quote_repair",
                "minimal_change": (
                    "Opt-in Whitespace-Reparatur über "
                    "src/extract/experimental_quote_repair.repair_quote "
                    "anwenden. Rekonstruktionsspan in der Diagnose "
                    "vermerkt; danach greift die unveränderte V2-"
                    "Validierung. Wissenschaftliche Bewertung bleibt "
                    "offen."
                ),
                "repaired_span": diag.get("repaired_span"),
            })
    # ENDPOINT_NOT_CONTAINED: minimal fix = drop offending assertion
    # (not adopted here; just noted).
    for entry in reject_details:
        if entry["reason"] != RejectReason.ENDPOINT_NOT_CONTAINED.value:
            continue
        fixes.append({
            "chunk_id": entry["chunk_id"],
            "phase": entry["phase"],
            "type": "endpoint_relax_or_drop",
            "minimal_change": (
                "Entweder den gewählten Beleg so erweitern, dass das "
                "Endpoint vollständig enthalten ist, oder die Relation "
                "verwerfen. Wissenschaftliche Tragfähigkeit muss "
                "extern bewertet werden."
            ),
        })
    # INVALID_OCCURRENCE_INDEX: minimal fix = re-run with cached raw
    # response and corrected indexing policy.
    for entry in reject_details:
        if entry["reason"] != RejectReason.INVALID_OCCURRENCE_INDEX.value:
            continue
        fixes.append({
            "chunk_id": entry["chunk_id"],
            "phase": entry["phase"],
            "type": "occurrence_index_semantics",
            "minimal_change": (
                "Belege aus dem Run sichern (Rohantworten) und offline "
                "mit dem unveränderten V2-Aligner gegenprüfen. Erst "
                "danach entscheiden, ob die Indizierung semantisch "
                "korrigiert (z. B. 0- vs. 1-basiert) oder der Modell-"
                "Output gefiltert werden muss. Außerhalb dieses "
                "Offline-Auftrags."
            ),
        })

    return {
        "diagnosis_version": "manifest-auto-offline/v1",
        "generated_utc": _utc_compact(),
        "run_dir": str(RUN_DIR),
        "counters": counter,
        "chunk_summaries": chunk_summaries,
        "reject_details": reject_details,
        "empty_chunk_excerpts": empty_excerpts,
        "confirmed_causes": confirmed_causes,
        "hypotheses": hypotheses,
        "minimal_fixes": fixes,
    }


def _pick_outside_endpoint(diag: dict) -> str:
    for label in ("subject_ref", "object_ref"):
        contained = diag.get(f"{label}_contained_in_evidence")
        if contained is False:
            return label
    return "unknown"


def _heuristic_candidates(text: str) -> list[str]:
    """Heuristically surface likely entity candidates from a text excerpt.

    NOT scientifically validated: this is purely a navigation aid for
    the reviewer. Patterns considered: ALL-CAPS acronyms, title-case
    multi-word tokens, and the common surface forms that appear in the
    prompt's worked example (e.g., ``NQ``, ``TriviaQA``, ``DPR``).
    """
    found: list[str] = []
    seen: set[str] = set()
    # Acronyms of length >=2, surrounded by word boundaries
    for m in re.finditer(r"\b[A-Z]{2,}\b", text):
        tok = m.group(0)
        if tok not in seen:
            seen.add(tok)
            found.append(tok)
    # CamelCase / TitleCase multiword tokens of length >=3
    for m in re.finditer(r"\b[A-Z][a-zA-Z]+(?:[ -][A-Z][a-zA-Z]+)+\b", text):
        tok = m.group(0)
        if tok not in seen:
            seen.add(tok)
            found.append(tok)
    return found[:20]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print a human-readable summary to stdout in addition to writing JSON.",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = diagnose()
    out_path = args.output_dir / f"offline_diagnosis_{_utc_compact()}.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"diagnosis: {out_path}")
    if args.print:
        print_summary(payload)
    return 0


def print_summary(payload: dict) -> None:
    print("\n=== Counter ===")
    print(json.dumps(payload["counters"], indent=2))
    print("\n=== Rejects ===")
    for entry in payload["reject_details"]:
        diag = entry.get("diagnostic", {})
        print(
            f"- {entry['chunk_id']} [{entry['phase']}/{entry['reason']}]"
        )
        if entry["reason"] == "ENDPOINT_NOT_CONTAINED":
            for k in (
                "exact_quote_present",
                "relation_span",
                "subject_ref_resolved_span",
                "subject_ref_contained_in_evidence",
                "object_ref_resolved_span",
                "object_ref_contained_in_evidence",
            ):
                print(f"    {k}: {diag.get(k)}")
        elif entry["reason"] == "ZERO_OCCURRENCES":
            for k in (
                "repair_status",
                "repaired_span",
                "diff_common_prefix",
                "diff_common_suffix",
                "divergence_is_pure_whitespace",
            ):
                print(f"    {k}: {diag.get(k)}")
            if "diff_window" in diag:
                print("    diff_window:")
                print(textwrap.indent(diag["diff_window"], "      "))
        else:
            print(f"    note: {diag.get('note', '-')}")
    print("\n=== Empty chunks ===")
    for ec in payload["empty_chunk_excerpts"]:
        print(f"- {ec['chunk_id']} [{ec['doc_id']}] {ec['section_path']}")
        print(f"    excerpt_head:\n{textwrap.indent(ec['excerpt_head'], '      ')}")
        print(f"    heuristic_candidates: {ec['heuristic_surface_candidates']}")
        print(f"    assessment: {ec['assessment']}")
    print("\n=== Confirmed causes ===")
    for c in payload["confirmed_causes"]:
        print(json.dumps(c, ensure_ascii=False))
    print("\n=== Hypotheses ===")
    for h in payload["hypotheses"]:
        print(json.dumps(h, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
