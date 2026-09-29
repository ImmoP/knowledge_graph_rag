"""Programmatic verification of every documented character span in
the manifest-auto review table.

Each span in the review markdown is derived from the frozen chunk
text in ``src/benchmark/dataset/benchmark_chunks.jsonl``. This test
re-derives the spans from the verbatim text and asserts the offsets
match what the review documents. If the chunk text ever drifts (the
dataset SHA-256 changes), the test fails before any human reviewer
re-reads stale offsets.

Qwen-Quote-Repair comparison: the prior experiment reports the full
metric surface ``Answer Coverage at Top-20/100`` at ``[53, 82)``.
The earlier review draft reported ``Answer Coverage`` at
``[42, 57)``, which is wrong; this test pins both the correct
``Answer Coverage`` span (``[53, 68)``) and the full-metric span
(``[53, 82)``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_JSONL = (
    REPO_ROOT / "src" / "benchmark" / "dataset" / "benchmark_chunks.jsonl"
)


def _load_chunk(chunk_id: str) -> str:
    for line in CHUNKS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["chunk_id"] == chunk_id:
            return rec["text"]
    raise KeyError(chunk_id)


def _locate(text: str, needle: str) -> tuple[int, int]:
    """Return ``(char_start, char_end)`` for ``needle`` in ``text``.

    Raises ``AssertionError`` if the needle is missing or occurs more
    than once (because every documented span must be unambiguous).
    """
    occurrences = []
    pos = 0
    while True:
        idx = text.find(needle, pos)
        if idx == -1:
            break
        occurrences.append(idx)
        pos = idx + 1
    assert occurrences, f"needle {needle!r} not found in chunk text"
    assert len(occurrences) == 1, (
        f"needle {needle!r} occurs {len(occurrences)} times; expected exactly once"
    )
    start = occurrences[0]
    return start, start + len(needle)


# ---------- chunk 2: ANCE / Answer Coverage -------------------------


def test_chunk2_caption_span_starts_at_25_and_covers_trivial_qa() -> None:
    text = _load_chunk("c_001452fd142ce075")
    caption = (
        "\\caption{Retrieval results (Answer Coverage at Top-20/100) on \n"
        "         Natural Questions (NQ) and Trivial QA (TQA) in the setting from ~"
    )
    start, end = _locate(text, caption)
    assert (start, end) == (25, 162)
    assert text[25:162] == caption


def test_chunk2_answer_coverage_verbatim_span_is_53_68() -> None:
    """The earlier review markdown reported ``[42, 57)``, which is
    an off-by-five error: at offset 42 the chunk contains ``sults
    (`` not ``Answer Coverage``. The verbatim token starts at 53.
    """
    text = _load_chunk("c_001452fd142ce075")
    start, end = _locate(text, "Answer Coverage")
    assert (start, end) == (53, 68)
    assert text[42:57] != "Answer Coverage"  # regression guard


def test_chunk2_full_metric_span_matches_qwen_quote_repair_53_82() -> None:
    """Qwen-Quote-Repair reports ``Answer Coverage at Top-20/100``
    at ``[53, 82)``. The verbatim token must agree with that span
    so cross-experiment comparison stays coherent.
    """
    text = _load_chunk("c_001452fd142ce075")
    full_metric = "Answer Coverage at Top-20/100"
    start, end = _locate(text, full_metric)
    assert (start, end) == (53, 82)


def test_chunk2_natural_questions_span_is_97_119() -> None:
    text = _load_chunk("c_001452fd142ce075")
    start, end = _locate(text, "Natural Questions (NQ)")
    assert (start, end) == (97, 119)


def test_chunk2_trivial_qa_span_is_124_140() -> None:
    text = _load_chunk("c_001452fd142ce075")
    start, end = _locate(text, "Trivial QA (TQA)")
    assert (start, end) == (124, 140)


def test_chunk2_dr_mention_span_is_706_708() -> None:
    text = _load_chunk("c_001452fd142ce075")
    start, end = _locate(text, "DR")
    assert (start, end) == (706, 708)


def test_chunk2_ance_mention_span_is_718_722() -> None:
    text = _load_chunk("c_001452fd142ce075")
    start, end = _locate(text, "ANCE")
    assert (start, end) == (718, 722)


def test_chunk2_ance_appears_in_training_change_caption_not_table_caption() -> None:
    """The review corrected the ANCE source: the token appears in
    the *second* tab caption (training change), not in the first
    (table of answer coverage). This test pins the textual anchor.
    """
    text = _load_chunk("c_001452fd142ce075")
    start, _ = _locate(text, "ANCE")
    # The ANCE token must be inside the training-change caption
    # sentence, not inside the answer-coverage caption.
    training_caption_marker = "production DR model to ANCE"
    tc_start, tc_end = _locate(text, training_caption_marker)
    assert tc_start <= start < tc_end
    # The answer-coverage caption (which contains NQ/TQA) must end
    # before ANCE appears.
    answer_caption_end = 162  # the span from test_chunk2_caption_span...
    assert start > answer_caption_end


# ---------- chunk 1: DC-BERT / ColBERT ------------------------------


def test_chunk1_dc_bert_span_is_178_185() -> None:
    text = _load_chunk("c_0001921855a8156b")
    start, end = _locate(text, "DC-BERT")
    assert (start, end) == (178, 185)


def test_chunk1_colbert_span_is_254_261() -> None:
    text = _load_chunk("c_0001921855a8156b")
    start, end = _locate(text, "ColBERT")
    assert (start, end) == (254, 261)


def test_chunk1_architecture_phrase_contains_dc_bert() -> None:
    """The OMP rationale for DC-BERT mentions the caption phrase
    'shows the architecture of DC-BERT'. Pin its start/end so a
    future review can't accidentally misattribute the architecture
    to a different sub-figure.
    """
    text = _load_chunk("c_0001921855a8156b")
    phrase = "shows the architecture of DC-BERT"
    start, end = _locate(text, phrase)
    # The phrase ends exactly at the DC-BERT span end.
    assert end == 185
    assert text[start:end] == phrase


# ---------- review-table truth table --------------------------------


@pytest.mark.parametrize(
    "chunk_id, needle, span",
    [
        ("c_001452fd142ce075", "Answer Coverage", (53, 68)),
        ("c_001452fd142ce075", "Answer Coverage at Top-20/100", (53, 82)),
        ("c_001452fd142ce075", "Natural Questions (NQ)", (97, 119)),
        ("c_001452fd142ce075", "Trivial QA (TQA)", (124, 140)),
        ("c_001452fd142ce075", "DR", (706, 708)),
        ("c_001452fd142ce075", "ANCE", (718, 722)),
        ("c_001452fd142ce075", "production DR model to ANCE", (695, 722)),
        ("c_001452fd142ce075", "The gains are from changing the training of a production DR model to ANCE", (649, 722)),
        ("c_001452fd142ce075",
         "\\caption{Retrieval results (Answer Coverage at Top-20/100) on \n"
         "         Natural Questions (NQ) and Trivial QA (TQA) in the setting from ~",
         (25, 162)),
        ("c_0001921855a8156b", "DC-BERT", (178, 185)),
        ("c_0001921855a8156b", "ColBERT", (254, 261)),
        ("c_0001921855a8156b", "shows the architecture of DC-BERT", (152, 185)),
    ],
)
def test_every_documented_span_matches_chunk_text(
    chunk_id: str, needle: str, span: tuple[int, int]
) -> None:
    """Single source of truth for the review table. Any new span
    must be added to this table so the next reviewer can trust the
    offsets without re-reading the chunk by hand.
    """
    text = _load_chunk(chunk_id)
    start, end = _locate(text, needle)
    assert (start, end) == span
    # Sanity: the slice is exactly the verbatim needle.
    assert text[start:end] == needle
