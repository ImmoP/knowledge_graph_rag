"""Focused acceptance tests for the deterministic exact-span v2 aligner (#49).

Covers the 20 acceptance items in #49 plus a small set of edge cases
(adjacent / overlapping potential matches, missing mention endpoints
in relation containment).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extract.align import (  # noqa: E402
    Reject,
    RejectReason,
    _enumerate_occurrences,
    align_mentions,
    align_relations,
)


CHUNK_ID = "chunk-001"
CHUNK_TEXT_REL = (
    "The Cat sat on the mat. The Cat purred. "
    "A Cat named Whiskers appeared."
)


def _mention(
    *,
    cid: str = "m1",
    surface: str = "brown fox",
    entity_type: str = "animal",
    canonical: str = "brown fox",
    confidence: str = "stated",
    occurrence_index=None,
):
    return {
        "id": cid,
        "entity_type": entity_type,
        "surface_form": surface,
        "canonical_form": canonical,
        "confidence": confidence,
        "occurrence_index": occurrence_index,
    }


def _relation(
    *,
    cid: str = "r1",
    evidence_quote: str = "brown fox jumps over",
    relation_type: str = "PREDICATE",
    subject_ref: dict | None = None,
    object_ref: dict | None = None,
    confidence: str = "stated",
    occurrence_index=None,
):
    return {
        "id": cid,
        "relation_type": relation_type,
        "subject_ref": subject_ref or {"kind": "document", "id": "doc-1"},
        "object_ref": object_ref or {"kind": "document", "id": "doc-2"},
        "evidence_quote": evidence_quote,
        "confidence": confidence,
        "occurrence_index": occurrence_index,
    }


class TestEnumerationInvariants(unittest.TestCase):
    """Acceptance 16: exact, left-to-right, non-overlapping enumeration."""

    CHUNK_TEXT = "The quick brown fox jumps over the lazy dog. The quick brown fox is fast."

    def _assert_starts(self, chunk_text: str, literal: str, expected: list[int]) -> None:
        starts = _enumerate_occurrences(chunk_text, literal)
        self.assertEqual(starts, expected)
        # Every returned start yields the exact literal at that offset.
        for s in starts:
            self.assertEqual(chunk_text[s:s + len(literal)], literal)
        # Strictly monotonic; previous end <= next start (non-overlapping).
        for prev, nxt in zip(starts, starts[1:]):
            self.assertLess(prev, nxt)
            self.assertLessEqual(prev + len(literal), nxt)

    def test_ordinary_repeated_literal(self):
        # Two non-contiguous matches in CHUNK_TEXT.
        literal = "quick brown fox"
        self._assert_starts(self.CHUNK_TEXT, literal, [4, 49])

    def test_adjacent_non_overlapping_occurrences(self):
        # "abab" with literal "ab" yields [0, 2], not [0, 2, 2...].
        self._assert_starts("abab", "ab", [0, 2])

    def test_potentially_overlapping_returns_only_leftmost(self):
        # "aaaa" with literal "aa": overlapping substrings exist at 0, 1, 2,
        # but only non-overlapping starts [0, 2] must be returned.
        self._assert_starts("aaaa", "aa", [0, 2])

    def test_exact_start_offsets_for_repeated_word(self):
        # "the" appears 3 times at known offsets in this fixture.
        self._assert_starts(
            "the cat and the dog and the bird", "the", [0, 12, 24]
        )

    def test_empty_and_absent_literals(self):
        self.assertEqual(_enumerate_occurrences("anything", ""), [])
        self.assertEqual(_enumerate_occurrences("hello world", "xyz"), [])

    def test_single_occurrence(self):
        self._assert_starts("alpha beta gamma", "beta", [6])

    def test_literal_at_end_and_start(self):
        self._assert_starts("xxabxxab", "ab", [2, 6])


class TestProviderDependency(unittest.TestCase):
    """Acceptance 17: module imports only stdlib + project types."""

    def test_no_provider_dependency(self):
        import inspect

        from src.extract import align

        src = inspect.getsource(align)
        self.assertNotIn("from src.providers", src)
        self.assertNotIn("import src.providers", src)


# ---------- 1-3: unique-literal acceptance ----------


class TestUniqueLiteralAcceptance(unittest.TestCase):
    TEXT = "alpha beta gamma delta epsilon"

    def test_unique_literal_no_index_accepts(self):
        cands = [_mention(surface="gamma")]
        accepted, rejects = align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
        self.assertEqual(rejects, [])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].char_start, self.TEXT.find("gamma"))
        self.assertEqual(accepted[0].char_end, self.TEXT.find("gamma") + len("gamma"))

    def test_unique_literal_index_zero_accepts(self):
        cands = [_mention(surface="gamma", occurrence_index=0)]
        accepted, rejects = align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
        self.assertEqual(rejects, [])
        self.assertEqual(len(accepted), 1)

    def test_unique_literal_invalid_nonzero_index_rejects(self):
        cands = [_mention(surface="gamma", occurrence_index=1)]
        accepted, rejects = align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejects), 1)
        self.assertEqual(rejects[0].reason, RejectReason.INVALID_OCCURRENCE_INDEX)
        self.assertEqual(rejects[0].offending_value, 1)


# ---------- 4-8: repeated-literal rejection rules ----------


class TestRepeatedLiteralRules(unittest.TestCase):
    TEXT = "the cat and the dog and the bird"

    def test_repeated_literal_missing_index_rejects(self):
        cands = [_mention(surface="the")]
        accepted, rejects = align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejects), 1)
        self.assertEqual(rejects[0].reason, RejectReason.INVALID_OCCURRENCE_INDEX)
        self.assertIsNone(rejects[0].offending_value)

    def test_repeated_literal_valid_index_zero_accepts(self):
        cands = [_mention(cid="m1", surface="the", occurrence_index=0)]
        accepted, rejects = align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
        self.assertEqual(rejects, [])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].char_start, 0)
        self.assertEqual(accepted[0].char_end, 3)

    def test_repeated_literal_valid_index_one_accepts(self):
        cands = [_mention(surface="the", occurrence_index=1)]
        accepted, rejects = align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
        self.assertEqual(rejects, [])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].char_start, 12)
        self.assertEqual(accepted[0].char_end, 15)

    def test_repeated_literal_negative_index_rejects(self):
        cands = [_mention(surface="the", occurrence_index=-1)]
        accepted, rejects = align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
        self.assertEqual(accepted, [])
        self.assertEqual(rejects[0].reason, RejectReason.INVALID_OCCURRENCE_INDEX)
        self.assertEqual(rejects[0].offending_value, -1)

    def test_repeated_literal_out_of_range_rejects(self):
        cands = [_mention(surface="the", occurrence_index=3)]
        accepted, rejects = align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
        self.assertEqual(accepted, [])
        self.assertEqual(rejects[0].reason, RejectReason.INVALID_OCCURRENCE_INDEX)
        self.assertEqual(rejects[0].offending_value, 3)

    def test_non_int_indices_rejected(self):
        for bad in ["0", 1.5, None]:
            with self.subTest(index=bad):
                cands = [_mention(surface="the", occurrence_index=bad)]
                accepted, rejects = align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
                self.assertEqual(accepted, [])
                self.assertEqual(len(rejects), 1)
                self.assertEqual(rejects[0].reason, RejectReason.INVALID_OCCURRENCE_INDEX)
                self.assertEqual(rejects[0].offending_value, bad)

    def test_bool_index_rejected_not_silently_accepted(self):
        for bad in [True, False]:
            with self.subTest(index=bad):
                cands = [_mention(surface="the", occurrence_index=bad)]
                accepted, rejects = align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
                self.assertEqual(accepted, [])
                self.assertEqual(len(rejects), 1)
                self.assertEqual(rejects[0].reason, RejectReason.INVALID_OCCURRENCE_INDEX)
                self.assertEqual(rejects[0].offending_value, bad)


# ---------- 9-11: literal-shape rejection rules ----------


class TestLiteralShapeRejections(unittest.TestCase):
    TEXT = "hello world this is a string"

    def test_absent_literal_rejects_zero_occurrences(self):
        cands = [_mention(surface="not-in-text")]
        accepted, rejects = align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejects), 1)
        self.assertEqual(rejects[0].reason, RejectReason.ZERO_OCCURRENCES)
        self.assertEqual(rejects[0].offending_value, "not-in-text")

    def test_empty_literal_rejects_empty_surface(self):
        cands = [_mention(surface="")]
        accepted, rejects = align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejects), 1)
        self.assertEqual(rejects[0].reason, RejectReason.EMPTY_SURFACE)
        self.assertEqual(rejects[0].offending_value, "")

    def test_non_string_literal_rejects_non_string_surface(self):
        for bad in [None, 123, ["hi"], {"x": 1}]:
            with self.subTest(value=bad):
                cands = [_mention(surface=bad)]
                accepted, rejects = align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
                self.assertEqual(accepted, [])
                self.assertEqual(len(rejects), 1)
                self.assertEqual(rejects[0].reason, RejectReason.NON_STRING_SURFACE)
                self.assertEqual(rejects[0].offending_value, bad)


# ---------- 12-15: relation alignment rules ----------


class TestRelationAlignment(unittest.TestCase):
    TEXT = CHUNK_TEXT_REL

    def _aligned_cat_span(self, mention_id: str = "m-cat"):
        accepted, _ = align_mentions(
            self.TEXT,
            [_mention(cid=mention_id, surface="Cat", occurrence_index=0)],
            chunk_id=CHUNK_ID,
        )
        self.assertEqual(len(accepted), 1)
        return (accepted[0].char_start, accepted[0].char_end)

    def test_unique_evidence_quote_with_contained_mention_endpoints_accepts(self):
        m_start, m_end = self._aligned_cat_span()
        aligned = {"m-cat": (m_start, m_end)}
        cands = [
            _relation(
                evidence_quote="The Cat sat",
                subject_ref={"kind": "mention", "id": "m-cat"},
                object_ref={"kind": "mention", "id": "m-cat"},
            )
        ]
        accepted, rejects = align_relations(self.TEXT, cands, aligned, chunk_id=CHUNK_ID)
        self.assertEqual(rejects, [])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].char_start, self.TEXT.find("The Cat sat"))
        self.assertEqual(accepted[0].char_end, self.TEXT.find("The Cat sat") + len("The Cat sat"))

    def test_evidence_quote_missing_endpoint_span_rejects(self):
        aligned = {"m-cat": (self.TEXT.find("Cat"), self.TEXT.find("Cat") + 3)}
        cands = [
            _relation(
                evidence_quote="purred",
                subject_ref={"kind": "mention", "id": "m-cat"},
                object_ref={"kind": "mention", "id": "m-cat"},
            )
        ]
        accepted, rejects = align_relations(self.TEXT, cands, aligned, chunk_id=CHUNK_ID)
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejects), 1)
        self.assertEqual(rejects[0].reason, RejectReason.ENDPOINT_NOT_CONTAINED)

    def test_document_endpoints_impose_no_containment_check(self):
        cands = [
            _relation(
                evidence_quote="The Cat sat on the mat",
                subject_ref={"kind": "document", "id": "doc-1"},
                object_ref={"kind": "document", "id": "doc-2"},
            )
        ]
        accepted, rejects = align_relations(self.TEXT, cands, {}, chunk_id=CHUNK_ID)
        self.assertEqual(rejects, [])
        self.assertEqual(len(accepted), 1)

    def test_repeated_evidence_quote_without_index_rejects(self):
        cands = [
            _relation(
                evidence_quote="The Cat",
                subject_ref={"kind": "document", "id": "doc-1"},
                object_ref={"kind": "document", "id": "doc-2"},
            )
        ]
        accepted, rejects = align_relations(self.TEXT, cands, {}, chunk_id=CHUNK_ID)
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejects), 1)
        self.assertEqual(rejects[0].reason, RejectReason.INVALID_OCCURRENCE_INDEX)

    def test_repeated_evidence_quote_with_valid_index_accepts(self):
        cands = [
            _relation(
                evidence_quote="The Cat",
                occurrence_index=1,
                subject_ref={"kind": "document", "id": "doc-1"},
                object_ref={"kind": "document", "id": "doc-2"},
            )
        ]
        accepted, rejects = align_relations(self.TEXT, cands, {}, chunk_id=CHUNK_ID)
        self.assertEqual(rejects, [])
        self.assertEqual(len(accepted), 1)
        first = self.TEXT.find("The Cat")
        expected = self.TEXT.find("The Cat", first + 1)
        self.assertEqual(accepted[0].char_start, expected)

    def test_relation_empty_evidence_quote_rejects(self):
        cands = [_relation(evidence_quote="")]
        accepted, rejects = align_relations(self.TEXT, cands, {}, chunk_id=CHUNK_ID)
        self.assertEqual(accepted, [])
        self.assertEqual(rejects[0].reason, RejectReason.EMPTY_SURFACE)

    def test_relation_non_string_evidence_quote_rejects(self):
        cands = [_relation(evidence_quote=42)]
        accepted, rejects = align_relations(self.TEXT, cands, {}, chunk_id=CHUNK_ID)
        self.assertEqual(accepted, [])
        self.assertEqual(rejects[0].reason, RejectReason.NON_STRING_SURFACE)

    def test_relation_absent_evidence_quote_rejects(self):
        cands = [_relation(evidence_quote="definitely not in text")]
        accepted, rejects = align_relations(self.TEXT, cands, {}, chunk_id=CHUNK_ID)
        self.assertEqual(accepted, [])
        self.assertEqual(rejects[0].reason, RejectReason.ZERO_OCCURRENCES)

    def test_relation_endpoint_with_unknown_mention_id_rejects(self):
        cands = [
            _relation(
                evidence_quote="The Cat sat",
                subject_ref={"kind": "mention", "id": "missing-id"},
                object_ref={"kind": "mention", "id": "m-cat"},
            )
        ]
        accepted, rejects = align_relations(self.TEXT, cands, {}, chunk_id=CHUNK_ID)
        self.assertEqual(accepted, [])
        self.assertEqual(rejects[0].reason, RejectReason.ENDPOINT_NOT_CONTAINED)


# ---------- 18: structured rejects ----------


class TestStructuredRejects(unittest.TestCase):
    TEXT = "alpha beta gamma"

    def test_reject_carries_chunk_id_candidate_id_reason_value(self):
        cands = [_mention(cid="mention-7", surface="not-in-text")]
        _, rejects = align_mentions(self.TEXT, cands, chunk_id="chunk-xyz")
        self.assertEqual(len(rejects), 1)
        r = rejects[0]
        self.assertEqual(r.chunk_id, "chunk-xyz")
        self.assertEqual(r.candidate_id, "mention-7")
        self.assertEqual(r.reason, RejectReason.ZERO_OCCURRENCES)
        self.assertEqual(r.offending_value, "not-in-text")
        rec = r.to_record()
        self.assertEqual(rec["chunk_id"], "chunk-xyz")
        self.assertEqual(rec["candidate_id"], "mention-7")
        self.assertEqual(rec["reason"], "ZERO_OCCURRENCES")
        self.assertEqual(rec["offending_value"], "not-in-text")

    def test_reject_candidate_id_none_when_absent(self):
        cands = [_mention(cid=None, surface="nope")]
        _, rejects = align_mentions(self.TEXT, cands, chunk_id="c")
        self.assertIsNone(rejects[0].candidate_id)


# ---------- 19: inputs are not mutated ----------


class TestNoInputMutation(unittest.TestCase):
    TEXT = "alpha beta gamma delta"

    def test_mention_candidate_not_mutated(self):
        cands = [_mention(surface="beta", canonical="beta", confidence="stated")]
        snapshot = {k: (list(v) if isinstance(v, list) else v) for k, v in cands[0].items()}
        align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
        self.assertEqual(cands[0], snapshot)

    def test_rejected_candidate_not_mutated(self):
        cands = [_mention(cid="r1", surface="missing", occurrence_index=5)]
        snapshot = {k: (list(v) if isinstance(v, list) else v) for k, v in cands[0].items()}
        align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
        self.assertEqual(cands[0], snapshot)

    def test_relation_candidate_not_mutated(self):
        cands = [
            _relation(
                evidence_quote="beta gamma",
                subject_ref={"kind": "document", "id": "d"},
                object_ref={"kind": "document", "id": "d"},
            )
        ]
        subject_snapshot = dict(cands[0]["subject_ref"])
        object_snapshot = dict(cands[0]["object_ref"])
        align_relations(self.TEXT, cands, {}, chunk_id=CHUNK_ID)
        self.assertEqual(cands[0]["subject_ref"], subject_snapshot)
        self.assertEqual(cands[0]["object_ref"], object_snapshot)

    def test_chunk_text_not_mutated(self):
        snapshot = self.TEXT
        align_mentions(self.TEXT, [_mention(surface="beta")], chunk_id=CHUNK_ID)
        self.assertEqual(self.TEXT, snapshot)


# ---------- 20: exact Python-string offset invariant ----------


class TestOffsetInvariant(unittest.TestCase):
    TEXT = "the rain in Spain stays mainly in the plain"

    def test_mention_slice_matches_literal(self):
        cands = [_mention(surface="Spain")]
        accepted, _ = align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
        m = accepted[0]
        self.assertEqual(self.TEXT[m.char_start:m.char_end], "Spain")

    def test_repeated_mention_slice_matches_literal(self):
        # "in" appears three times.
        cands = [_mention(surface="in", occurrence_index=2)]
        accepted, _ = align_mentions(self.TEXT, cands, chunk_id=CHUNK_ID)
        m = accepted[0]
        self.assertEqual(self.TEXT[m.char_start:m.char_end], "in")

    def test_relation_slice_matches_evidence_quote(self):
        aligned = {}
        cands = [
            _relation(
                evidence_quote="mainly in the plain",
                subject_ref={"kind": "document", "id": "d"},
                object_ref={"kind": "document", "id": "d"},
            )
        ]
        accepted, _ = align_relations(self.TEXT, cands, aligned, chunk_id=CHUNK_ID)
        a = accepted[0]
        self.assertEqual(self.TEXT[a.char_start:a.char_end], "mainly in the plain")


class TestNoStringLenientMatch(unittest.TestCase):
    """#49: exact Python-string equality; no whitespace / case / Unicode normalization."""

    def test_case_difference_is_not_a_match(self):
        cands = [_mention(surface="hello")]
        accepted, rejects = align_mentions("Hello world", cands, chunk_id=CHUNK_ID)
        self.assertEqual(accepted, [])
        self.assertEqual(rejects[0].reason, RejectReason.ZERO_OCCURRENCES)

    def test_whitespace_difference_is_not_a_match(self):
        cands = [_mention(surface="hello world")]
        accepted, rejects = align_mentions("hello  world", cands, chunk_id=CHUNK_ID)
        self.assertEqual(accepted, [])
        self.assertEqual(rejects[0].reason, RejectReason.ZERO_OCCURRENCES)

    def test_unicode_normalization_not_applied(self):
        # "café" with composed é vs NFC-normalized; we do not normalize.
        composed = "café"  # U+00E9 single codepoint
        decomposed = "cafe\u0301"  # 'e' + U+0301 combining acute
        cands = [_mention(surface=composed)]
        accepted, rejects = align_mentions(decomposed, cands, chunk_id=CHUNK_ID)
        self.assertEqual(accepted, [])
        self.assertEqual(rejects[0].reason, RejectReason.ZERO_OCCURRENCES)


class TestAlignedAssertionEquality(unittest.TestCase):
    """Regression: subject_ref / object_ref MUST participate in equality.

    Two aligned assertions differing only in endpoints must not compare
    equal. Endpoints are dict fields, which are unhashable, so they
    must opt out of hashing (hash=False) but not out of equality
    (compare=True).
    """

    def _make(self, *, subject_ref: dict, object_ref: dict):
        # Construct directly to isolate the equality / hash semantics
        # under test from the mention-alignment pipeline.
        from src.extract.align import AlignedAssertion  # local import

        return AlignedAssertion(
            chunk_id=CHUNK_ID,
            candidate_id="r1",
            relation_type="PREDICATE",
            subject_ref=subject_ref,
            object_ref=object_ref,
            evidence_quote="some quote",
            confidence="stated",
            char_start=0,
            char_end=11,
        )

    def test_different_subject_ref_unequal(self):
        a = self._make(
            subject_ref={"kind": "mention", "id": "m1"},
            object_ref={"kind": "mention", "id": "m2"},
        )
        b = self._make(
            subject_ref={"kind": "mention", "id": "m2"},
            object_ref={"kind": "mention", "id": "m1"},
        )
        self.assertNotEqual(a.subject_ref, b.subject_ref)
        self.assertNotEqual(a, b)

    def test_different_object_ref_unequal(self):
        a = self._make(
            subject_ref={"kind": "document", "id": "d1"},
            object_ref={"kind": "document", "id": "d1"},
        )
        b = self._make(
            subject_ref={"kind": "document", "id": "d1"},
            object_ref={"kind": "document", "id": "d2"},
        )
        self.assertNotEqual(a.object_ref, b.object_ref)
        self.assertNotEqual(a, b)

    def test_identical_assertions_equal(self):
        a = self._make(
            subject_ref={"kind": "document", "id": "d1"},
            object_ref={"kind": "document", "id": "d1"},
        )
        b = self._make(
            subject_ref={"kind": "document", "id": "d1"},
            object_ref={"kind": "document", "id": "d1"},
        )
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
