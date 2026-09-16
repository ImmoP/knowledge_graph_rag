"""Phase-one/Phase-two/Phase-three tests for ``src.grounded_gen``."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.providers.protocol import (  # noqa: E402
    LLMProvider,
    ProviderTransportError,
    Usage,
)




from src.grounded_gen import (  # noqa: E402

    FORMAT_REPAIR_SUFFIX,
    Claim,
    ClaimSupportVerdict,
    CitationOccurrence,
    EvidenceBlock,
    EvidenceIntegrityError,
    GenerationParseError,
    GenerationRequest,
    GenerationResult,
    GenerationTraceEvent,
    GenerationTraceSink,
    GroundedGenerator,
    InMemoryTraceSink,
    PROMPT_FILENAME,
    SYSTEM_PROMPT,
    _ProtocolError,
    evidence_fingerprint,
    lexical_support,
    lexical_tokens,
    parse_and_validate_answer,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _record(chunk_id: str, doc_id: str, text: str) -> dict:
    return {"chunk_id": chunk_id, "doc_id": doc_id, "text": text}


def _write_chunks(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _independent_fingerprint(evidence: tuple[EvidenceBlock, ...]) -> str:
    payload = [
        {
            "chunk_id": block.chunk_id,
            "doc_id": block.doc_id,
            "source_text_sha256": hashlib.sha256(
                block.text.encode("utf-8")
            ).hexdigest(),
        }
        for block in evidence
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ForbiddenProvider:
    name = "forbidden"
    model = "forbidden"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("Phase 1 must not call the provider")


class GroundedGeneratorFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"
        self.first = EvidenceBlock("c_0000000000000001", "W1", "first canonical text")
        self.second = EvidenceBlock("c_0000000000000002", "W2", "second canonical text")
        _write_chunks(
            self.chunks_path,
            [
                _record(self.first.chunk_id, self.first.doc_id, self.first.text),
                _record(self.second.chunk_id, self.second.doc_id, self.second.text),
            ],
        )
        self.provider = ForbiddenProvider()
        self.generator = GroundedGenerator(
            self.provider, canonical_chunks_path=self.chunks_path
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def request(self, evidence: tuple[EvidenceBlock, ...]) -> GenerationRequest:
        return GenerationRequest(query="question", evidence=evidence)

    def test_valid_evidence_is_accepted_and_rendered_in_caller_order(self) -> None:
        request = self.request((self.second, self.first))
        self.generator.preflight(request)
        rendered = self.generator.render_user_prompt(request)
        self.assertLess(rendered.index(self.second.chunk_id), rendered.index(self.first.chunk_id))
        self.assertIn("Prompt template filename: prompts/grounded_answer_v1.txt", rendered)
        self.assertIn("Prompt template content SHA256: ", rendered)
        self.assertIn(self.first.text, rendered)
        self.assertIn(self.second.text, rendered)

    def test_non_string_query_is_type_error_before_artifact_access(self) -> None:
        with self.assertRaises(TypeError):
            self.generator.generate(GenerationRequest(query=1, evidence=()))  # type: ignore[arg-type]
        self.assertEqual(self.provider.calls, 0)

    def test_malformed_chunk_id_is_rejected(self) -> None:
        bad = EvidenceBlock("not-a-chunk", "W1", self.first.text)
        with self.assertRaises(EvidenceIntegrityError):
            self.generator.generate(self.request((bad,)))
        self.assertEqual(self.provider.calls, 0)

    def test_malformed_doc_id_is_rejected(self) -> None:
        bad = EvidenceBlock(self.first.chunk_id, "bad", self.first.text)
        with self.assertRaises(EvidenceIntegrityError):
            self.generator.generate(self.request((bad,)))
        self.assertEqual(self.provider.calls, 0)

    def test_duplicate_request_chunk_id_is_rejected(self) -> None:
        with self.assertRaises(EvidenceIntegrityError):
            self.generator.generate(self.request((self.first, self.first)))
        self.assertEqual(self.provider.calls, 0)

    def test_unknown_chunk_id_is_rejected(self) -> None:
        bad = EvidenceBlock("c_ffffffffffffffff", "W9", "unknown")
        with self.assertRaises(EvidenceIntegrityError):
            self.generator.generate(self.request((bad,)))
        self.assertEqual(self.provider.calls, 0)

    def test_wrong_doc_id_is_rejected(self) -> None:
        bad = EvidenceBlock(self.first.chunk_id, "W9", self.first.text)
        with self.assertRaises(EvidenceIntegrityError):
            self.generator.generate(self.request((bad,)))
        self.assertEqual(self.provider.calls, 0)

    def test_stale_text_is_rejected(self) -> None:
        bad = EvidenceBlock(self.first.chunk_id, self.first.doc_id, "stale text")
        with self.assertRaises(EvidenceIntegrityError):
            self.generator.generate(self.request((bad,)))
        self.assertEqual(self.provider.calls, 0)

    def test_malformed_canonical_jsonl_fails_closed(self) -> None:
        self.chunks_path.write_text("{broken\n", encoding="utf-8")
        with self.assertRaises(EvidenceIntegrityError):
            self.generator.generate(self.request((self.first,)))
        self.assertEqual(self.provider.calls, 0)

    def test_unreadable_canonical_bytes_fail_closed(self) -> None:
        self.chunks_path.write_bytes(b"\xff")
        with self.assertRaises(EvidenceIntegrityError):
            self.generator.generate(self.request((self.first,)))
        self.assertEqual(self.provider.calls, 0)

    def test_duplicate_canonical_chunk_id_fails_closed(self) -> None:
        _write_chunks(
            self.chunks_path,
            [
                _record(self.first.chunk_id, self.first.doc_id, self.first.text),
                _record(self.first.chunk_id, self.first.doc_id, self.first.text),
            ],
        )
        with self.assertRaises(EvidenceIntegrityError):
            self.generator.generate(self.request((self.first,)))
        self.assertEqual(self.provider.calls, 0)

    def test_missing_canonical_artifact_fails_closed(self) -> None:
        self.chunks_path.unlink()
        with self.assertRaises(EvidenceIntegrityError):
            self.generator.generate(self.request((self.first,)))
        self.assertEqual(self.provider.calls, 0)

    def test_empty_evidence_abstains_without_provider_call(self) -> None:
        result = self.generator.generate(self.request(()))
        self.assertEqual(result.answer_text, "ABSTAIN")
        self.assertTrue(result.abstained)
        self.assertEqual(result.citations, ())
        self.assertEqual(result.claims, ())
        self.assertEqual(result.claim_support, ())
        self.assertEqual(result.usage_calls, ())
        self.assertEqual(result.evidence_fingerprint, _independent_fingerprint(()))
        self.assertEqual(self.provider.calls, 0)

    def test_empty_evidence_abstains_when_prompt_artifact_is_unreadable(self) -> None:
        with patch("src.grounded_gen.Path.read_text", side_effect=OSError):
            result = self.generator.generate(self.request(()))

        self.assertEqual(result.answer_text, "ABSTAIN")
        self.assertTrue(result.abstained)
        self.assertEqual(result.citations, ())
        self.assertEqual(result.claims, ())
        self.assertEqual(result.claim_support, ())
        self.assertEqual(result.usage_calls, ())
        self.assertEqual(result.evidence_fingerprint, _independent_fingerprint(()))
        self.assertEqual(self.provider.calls, 0)

    def test_nonempty_evidence_uses_acceptance_provider_when_provided(self) -> None:
        # Phase 3 wires the orchestrator to ``LLMProvider.generate``; the
        # ``ForbiddenProvider`` is no longer the right surface for this
        # assertion. The full provider-call matrix lives in
        # ``PhaseThreeGenerationTests``; here we just verify the wiring
        # accepts a real provider and does not raise ``NotImplementedError``.
        from tests.test_grounded_gen import _AcceptingProvider
        provider = _AcceptingProvider()
        fixture_generator = GroundedGenerator(provider, canonical_chunks_path=self.chunks_path)
        result = fixture_generator.generate(self.request((self.first,)))
        self.assertEqual(provider.calls, 1)
        self.assertTrue(result.abstained)
        self.assertEqual(result.usage_calls, (provider.usage,))


class PureAnswerValidationTests(unittest.TestCase):
    supporting = EvidenceBlock("c_0000000000000001", "W1", "Dense passage retrieval supports open domain questions.")
    non_supporting = EvidenceBlock("c_0000000000000002", "W2", "This unrelated astronomy evidence has planets.")

    def parse(self, answer: str):
        return parse_and_validate_answer(answer, (self.supporting, self.non_supporting))

    def test_status_matrix_spans_attachment_and_verdicts(self) -> None:
        answer = (
            "Dense passage retrieval supports open domain questions [[c_0000000000000001]]. "
            "A different unsupported assertion [[c_0000000000000002]]. "
            "Outside claim [[c_0000000000000003]]. "
            "Uncited statement."
        )
        abstained, citations, claims, verdicts = self.parse(answer)
        self.assertFalse(abstained)
        self.assertEqual([(claim.claim_id, claim.text, claim.char_start, claim.char_end, claim.attached_chunk_ids) for claim in claims], [
            (0, "Dense passage retrieval supports open domain questions .", 0, 78, ("c_0000000000000001",)),
            (1, "A different unsupported assertion .", 79, 136, ("c_0000000000000002",)),
            (2, "Outside claim .", 137, 174, ("c_0000000000000003",)),
            (3, "Uncited statement.", 175, 193, ()),
        ])
        self.assertEqual([(citation.chunk_id, citation.claim_id, citation.char_start, citation.char_end, citation.status) for citation in citations], [
            ("c_0000000000000001", 0, 55, 77, "IN_EVIDENCE_SUPPORTING"),
            ("c_0000000000000002", 1, 113, 135, "IN_EVIDENCE_NON_SUPPORTING"),
            ("c_0000000000000003", 2, 151, 173, "OUT_OF_EVIDENCE"),
        ])
        self.assertEqual([(item.claim_id, item.verdict, item.supporting_chunk_ids, item.non_supporting_chunk_ids, item.out_of_evidence_chunk_ids) for item in verdicts], [
            (0, "SUPPORTED", ("c_0000000000000001",), (), ()),
            (1, "CITED_UNSUPPORTED", (), ("c_0000000000000002",), ()),
            (2, "CITED_UNSUPPORTED", (), (), ("c_0000000000000003",)),
            (3, "UNCITED", (), (), ()),
        ])

    def test_segmentation_uses_only_pinned_separators_and_original_spans(self) -> None:
        answer = "  One. Two? Three! Four\n\nFive\nSix.  "
        _, _, claims, _ = self.parse(answer)
        self.assertEqual([(claim.text, claim.char_start, claim.char_end) for claim in claims], [
            ("One.", 2, 6), ("Two?", 7, 11), ("Three!", 12, 18), ("Four", 19, 23), ("Five\nSix.", 25, 34),
        ])

    def test_citations_before_punctuation_and_adjacency_are_preserved(self) -> None:
        _, citations, claims, verdicts = self.parse("Fact [[c_0000000000000001]][[c_0000000000000001]].")
        self.assertEqual(len(citations), 2)
        self.assertEqual(claims[0].attached_chunk_ids, ("c_0000000000000001",))
        self.assertEqual(verdicts[0].non_supporting_chunk_ids, ("c_0000000000000001",))

    def test_same_chunk_in_separate_claims_is_independent(self) -> None:
        _, citations, claims, verdicts = self.parse("Dense passage retrieval [[c_0000000000000001]]. Dense passage retrieval [[c_0000000000000001]].")
        self.assertEqual(len(citations), 2)
        self.assertEqual([claim.attached_chunk_ids for claim in claims], [("c_0000000000000001",), ("c_0000000000000001",)])
        self.assertEqual([verdict.verdict for verdict in verdicts], ["SUPPORTED", "SUPPORTED"])

    def test_malformed_double_bracket_constructs_fail_protocol(self) -> None:
        for token in ("[[CITE:c_aabbccddeeff0011]]", "[[c_AABBCCDDEEFF0011]]", "[[ c_aabbccddeeff0011 ]]", "[[c_aabbccddeeff0011, c_1122334455667788]]", "[[c_aabbccddeeff0011:p3]]", "[[c_short]]", "[[c_aabbccddeeff0011", "[ [c_aabbccddeeff0011] ]"):
            with self.subTest(token=token), self.assertRaises(_ProtocolError):
                self.parse(f"Claim {token}.")
        self.parse("ordinary [brackets] remain prose")
        _, citations, _, _ = self.parse("Valid [[c_0000000000000003]].")
        self.assertEqual(citations[0].status, "OUT_OF_EVIDENCE")

    def test_exact_abstain_only(self) -> None:
        self.assertEqual(self.parse("ABSTAIN"), (True, (), (), ()))
        for answer in ("ABSTAIN ", " ABSTAIN", "ABSTAIN.", "Answer: ABSTAIN", "abstain"):
            abstained, _, claims, _ = self.parse(answer)
            self.assertFalse(abstained)
            self.assertTrue(claims)

    def test_empty_normalized_claim_is_uncited_or_cited_unsupported(self) -> None:
        _, citations, claims, verdicts = self.parse("[[c_0000000000000001]]")
        self.assertFalse(claims[0].scorable)
        self.assertEqual(citations[0].status, "IN_EVIDENCE_NON_SUPPORTING")
        self.assertEqual(verdicts[0].verdict, "CITED_UNSUPPORTED")
        _, _, claims, verdicts = self.parse("!!!")
        self.assertFalse(claims[0].scorable)
        self.assertEqual(verdicts[0].verdict, "UNCITED")

    def test_lexical_normalization_and_pinned_ngram_containment(self) -> None:
        self.assertEqual(lexical_tokens("GPT-4 Recall@10 512 tokens 2020 a_b ＤＰＲ café"), ("gpt", "4", "recall", "10", "512", "tokens", "2020", "a", "b", "dpr", "café"))
        for claim in ("DPR", "open domain", "dense passage retrieval", "dense passage retrieval model"):
            self.assertTrue(lexical_support(claim, "A dense passage retrieval model for open domain DPR."))
        self.assertTrue(lexical_support("one two three four five", "zero two three four five six"))
        self.assertFalse(lexical_support("dense passage retrieval model", "dense passage unrelated retrieval model"))
        self.assertFalse(lexical_support("dense passage retrieval model", "dense retrieval passage model"))


class PublicContractTests(unittest.TestCase):
    def test_canonical_dataclasses_have_exact_fields_and_are_frozen(self) -> None:
        expected = {
            EvidenceBlock: ["chunk_id", "doc_id", "text"],
            GenerationRequest: ["query", "evidence", "max_tokens", "temperature"],
            CitationOccurrence: ["chunk_id", "claim_id", "char_start", "char_end", "status"],
            Claim: ["claim_id", "text", "char_start", "char_end", "scorable", "attached_chunk_ids"],
            ClaimSupportVerdict: ["claim_id", "verdict", "supporting_chunk_ids", "non_supporting_chunk_ids", "out_of_evidence_chunk_ids"],
            GenerationResult: ["answer_text", "abstained", "citations", "claims", "claim_support", "usage_calls", "evidence_fingerprint"],
        }
        for cls, field_names in expected.items():
            self.assertEqual([field.name for field in dataclasses.fields(cls)], field_names)
            self.assertTrue(cls.__dataclass_params__.frozen)
        self.assertNotIn("parsed_ok", GenerationResult.__dataclass_fields__)
        self.assertNotIn("parse_error", GenerationResult.__dataclass_fields__)
        self.assertNotIn("cache_hit", GenerationResult.__dataclass_fields__)

    def test_fingerprint_is_independent_canonical_construction(self) -> None:
        first = EvidenceBlock("c_0000000000000001", "W1", "alpha")
        second = EvidenceBlock("c_0000000000000002", "W2", "beta")
        evidence = (first, second)
        self.assertEqual(evidence_fingerprint(evidence), _independent_fingerprint(evidence))
        self.assertEqual(evidence_fingerprint(evidence), evidence_fingerprint(evidence))
        self.assertNotEqual(evidence_fingerprint(evidence), evidence_fingerprint((second, first)))
        self.assertNotEqual(evidence_fingerprint(evidence), evidence_fingerprint((EvidenceBlock(first.chunk_id, first.doc_id, "changed"), second)))
        self.assertNotEqual(evidence_fingerprint(evidence), evidence_fingerprint((EvidenceBlock(first.chunk_id, "W3", first.text), second)))
        self.assertNotEqual(evidence_fingerprint(evidence), evidence_fingerprint((EvidenceBlock("c_0000000000000003", first.doc_id, first.text), second)))
        self.assertEqual(evidence_fingerprint(()), "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945")

    def test_prompt_contains_pinned_requirements(self) -> None:
        prompt = (REPO_ROOT / "prompts" / "grounded_answer_v1.txt").read_text(encoding="utf-8").lower()
        for requirement in (
            "supplied evidence",
            "outside knowledge",
            "every factual claim",
            "before the sentence-ending punctuation",
            "abstain and nothing else",
            "do not include headings",
            "do not include a references section",
            "do not include meta-commentary",
        ):
            self.assertIn(requirement, prompt)


    def test_static_guards_keep_generation_isolated(self) -> None:
        source = (REPO_ROOT / "src" / "grounded_gen.py").read_text(encoding="utf-8")
        for forbidden in (
            "src.retrieval.vector",
            "src.retrieval.graph",
            "src.retrieval.hybrid",
            ".generate_json(",
            "eval/questions.jsonl",
            "expected_answer",
            "hop_bucket",
            "sentence_transformers",
            "neo4j",
            "psycopg",
        ):
            self.assertNotIn(forbidden, source)

    def test_static_orchestration_guards(self) -> None:
        source = (REPO_ROOT / "src" / "grounded_gen.py").read_text(encoding="utf-8")
        for forbidden in (
            ".generate_json(",
            "from src.retrieval",
            "import src.retrieval",
            "from src.providers.cache",
            "questions.jsonl",
            "expected_answer",
            "hop_bucket",
            "huggingface",
            "llm_cache",
            "BudgetExceeded",
        ):
            self.assertNotIn(forbidden, source)

    def test_format_repair_suffix_is_pinned(self) -> None:
        self.assertTrue(FORMAT_REPAIR_SUFFIX.startswith("\n\n"))
        self.assertTrue(FORMAT_REPAIR_SUFFIX.endswith("\n"))
        for required in (
            "[[c_<16 lowercase hex",
            "sentence-ending",
            "ABSTAIN alone",
            "headings",
            "References",
            "meta-commentary",
        ):
            self.assertIn(required, FORMAT_REPAIR_SUFFIX)



class _AcceptingProvider:
    """Minimal scripted provider that returns ``ABSTAIN`` once."""

    name = "accepting"
    model = "accepting-v1"

    def __init__(self) -> None:
        self.calls = 0
        self.usage = Usage(input_tokens=0, output_tokens=1, cost_usd=0.0)

    def generate(self, system, user, *, max_tokens=1024, temperature=0.0):
        self.calls += 1
        return ("ABSTAIN", self.usage)

    def generate_json(self, *args, **kwargs):
        raise AssertionError("generate_json must not be called")


class _ScriptedProvider:
    """Scripted LLMProvider that returns pre-recorded responses in order."""

    name = "scripted"
    model = "scripted-v1"

    def __init__(self, responses):
        if not responses:
            raise ValueError("at least one response is required")
        self._responses = list(responses)
        self.calls = []

    def generate(self, system, user, *, max_tokens=1024, temperature=0.0):
        self.calls.append({
            "system": system,
            "user": user,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        if not self._responses:
            raise AssertionError("scripted provider exhausted")
        return self._responses.pop(0)

    def generate_json(self, *args, **kwargs):
        raise AssertionError("generate_json must not be called")


class _ExplodingAfterScriptedProvider:
    """Returns the first scripted response, then raises on the second call."""

    name = "scripted-then-explode"
    model = "scripted-then-explode-v1"

    def __init__(self, *, first_text, first_usage, exc):
        self._first_text = first_text
        self._first_usage = first_usage
        self._exc = exc
        self.calls = 0

    def generate(self, system, user, *, max_tokens=1024, temperature=0.0):
        self.calls += 1
        if self.calls == 1:
            return (self._first_text, self._first_usage)
        raise self._exc

    def generate_json(self, *args, **kwargs):
        raise AssertionError("generate_json must not be called")


class _AlwaysExplodingProvider:
    name = "explode"
    model = "explode-v1"

    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        raise self._exc

    def generate_json(self, *args, **kwargs):
        raise AssertionError("generate_json must not be called")


class _ProviderError(ProviderTransportError):


    pass


class PhaseThreeGenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"
        self.supporting = EvidenceBlock(
            "c_0000000000000001", "W1",
            "Dense passage retrieval supports open domain question answering.",
        )
        self.non_supporting = EvidenceBlock(
            "c_0000000000000002", "W2",
            "An unrelated astronomy passage about planetary orbits.",
        )
        _write_chunks(self.chunks_path, [
            _record(self.supporting.chunk_id, self.supporting.doc_id, self.supporting.text),
            _record(self.non_supporting.chunk_id, self.non_supporting.doc_id, self.non_supporting.text),
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def _generator(self, provider, sink=None):
        kwargs = {"canonical_chunks_path": self.chunks_path}
        if sink is not None:
            kwargs["trace_sink"] = sink
        return GroundedGenerator(provider, **kwargs)

    def test_one_call_success_preserves_provider_text_and_records_trace(self):
        answer_text = (
            "Dense passage retrieval supports open domain questions "
            "[[c_0000000000000001]]. An uncited trailing sentence."
        )
        usage = Usage(input_tokens=11, output_tokens=22, cost_usd=0.0)
        provider = _ScriptedProvider([(answer_text, usage)])
        sink = InMemoryTraceSink()
        generator = self._generator(provider, sink=sink)
        result = generator.generate(GenerationRequest(
            query="Q?", evidence=(self.supporting, self.non_supporting),
        ))
        self.assertEqual(len(provider.calls), 1)
        call = provider.calls[0]
        self.assertEqual(call["system"], SYSTEM_PROMPT)
        self.assertIn(self.supporting.text, call["user"])
        self.assertIn(self.supporting.chunk_id, call["user"])
        self.assertIn(PROMPT_FILENAME, call["user"])
        self.assertEqual(call["max_tokens"], 1024)
        self.assertEqual(call["temperature"], 0.0)
        self.assertEqual(result.answer_text, answer_text)
        self.assertEqual(result.usage_calls, (usage,))
        self.assertFalse(result.abstained)
        self.assertEqual(result.citations[0].chunk_id, self.supporting.chunk_id)
        self.assertEqual(result.citations[0].status, "IN_EVIDENCE_SUPPORTING")
        self.assertEqual([verdict.verdict for verdict in result.claim_support], ["SUPPORTED", "UNCITED"])
        self.assertEqual(result.evidence_fingerprint, _independent_fingerprint((self.supporting, self.non_supporting)))
        self.assertEqual(len(sink.events), 1)
        event = sink.events[0]
        self.assertEqual(event.outcome, "success")
        self.assertEqual(event.provider_call_count, 1)
        self.assertEqual(event.usage_calls, (usage,))
        self.assertFalse(event.repair_attempted)
        self.assertEqual(event.evidence_fingerprint, result.evidence_fingerprint)
        self.assertEqual(event.prompt_filename, PROMPT_FILENAME)
        self.assertEqual(len(event.prompt_content_sha256), 64)

    def test_model_abstain_records_usage_and_emits_abstained_event(self):
        usage = Usage(input_tokens=7, output_tokens=1, cost_usd=0.0)
        provider = _ScriptedProvider([("ABSTAIN", usage)])
        sink = InMemoryTraceSink()
        generator = self._generator(provider, sink=sink)
        result = generator.generate(GenerationRequest(query="Q?", evidence=(self.supporting,)))
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(result.answer_text, "ABSTAIN")
        self.assertTrue(result.abstained)
        self.assertEqual(result.citations, ())
        self.assertEqual(result.claims, ())
        self.assertEqual(result.claim_support, ())
        self.assertEqual(result.usage_calls, (usage,))
        self.assertEqual(sink.events[-1].outcome, "abstained")
        self.assertEqual(sink.events[-1].provider_call_count, 1)
        self.assertEqual(sink.events[-1].usage_calls, (usage,))

    def test_retry_appends_pinned_suffix_and_keeps_system_prompt(self):
        malformed = "Claim [[c_AABBCCDDEEFF0011]]."
        valid = "Dense passage retrieval supports open domain questions [[c_0000000000000001]]."
        first_usage = Usage(input_tokens=5, output_tokens=5, cost_usd=0.0)
        second_usage = Usage(input_tokens=8, output_tokens=8, cost_usd=0.0)
        provider = _ScriptedProvider([(malformed, first_usage), (valid, second_usage)])
        sink = InMemoryTraceSink()
        generator = self._generator(provider, sink=sink)
        rendered = generator.render_user_prompt(GenerationRequest(query="Q?", evidence=(self.supporting,)))
        result = generator.generate(GenerationRequest(query="Q?", evidence=(self.supporting,)))
        self.assertEqual(len(provider.calls), 2)
        first, second = provider.calls
        self.assertEqual(first["system"], SYSTEM_PROMPT)
        self.assertEqual(second["system"], SYSTEM_PROMPT)
        self.assertEqual(first["user"], rendered)
        self.assertEqual(second["user"], rendered + FORMAT_REPAIR_SUFFIX)
        self.assertEqual(first["max_tokens"], second["max_tokens"])
        self.assertEqual(first["temperature"], second["temperature"])
        self.assertEqual(result.usage_calls, (first_usage, second_usage))
        self.assertEqual(result.answer_text, valid)
        self.assertEqual(result.citations[0].chunk_id, self.supporting.chunk_id)
        self.assertEqual(result.citations[0].status, "IN_EVIDENCE_SUPPORTING")
        self.assertEqual(sink.events[-1].outcome, "success")
        self.assertTrue(sink.events[-1].repair_attempted)
        self.assertEqual(sink.events[-1].provider_call_count, 2)
        self.assertEqual(sink.events[-1].usage_calls, (first_usage, second_usage))

    def test_persistent_parse_failure_records_trace_before_raise(self):
        malformed = "Claim [[c_AABBCCDDEEFF0011]]."
        first_usage = Usage(input_tokens=2, output_tokens=2, cost_usd=0.0)
        second_usage = Usage(input_tokens=3, output_tokens=3, cost_usd=0.0)
        provider = _ScriptedProvider([(malformed, first_usage), (malformed, second_usage)])
        sink = InMemoryTraceSink()
        generator = self._generator(provider, sink=sink)
        with self.assertRaises(GenerationParseError) as raised:
            generator.generate(GenerationRequest(query="Q?", evidence=(self.supporting,)))
        self.assertEqual(len(provider.calls), 2)
        error = raised.exception
        self.assertEqual(error.usage_calls, (first_usage, second_usage))
        self.assertEqual(error.evidence_fingerprint, _independent_fingerprint((self.supporting,)))
        self.assertEqual(len(sink.events), 1)
        event = sink.events[0]
        self.assertEqual(event.outcome, "parse_error")
        self.assertEqual(event.provider_call_count, 2)
        self.assertEqual(event.usage_calls, (first_usage, second_usage))
        self.assertTrue(event.repair_attempted)
        self.assertEqual(event.evidence_fingerprint, error.evidence_fingerprint)
        self.assertEqual(event.prompt_filename, PROMPT_FILENAME)

    def test_no_retry_outcome_matrix_uses_one_call(self):
        cases = {
            "out_of_evidence": (
                "An out-of-evidence reference [[c_0000000000000003]].",
                Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
                "CITED_UNSUPPORTED",
                "success",
            ),
            "in_evidence_non_supporting": (
                "An unsupported claim about astronomy [[c_0000000000000002]].",
                Usage(input_tokens=2, output_tokens=2, cost_usd=0.0),
                "CITED_UNSUPPORTED",
                "success",
            ),
            "cited_unsupported": (
                "An unsupported claim about astronomy [[c_0000000000000002]].",
                Usage(input_tokens=3, output_tokens=3, cost_usd=0.0),
                "CITED_UNSUPPORTED",
                "success",
            ),
            "uncited": (
                "An uncited claim about the moon.",
                Usage(input_tokens=4, output_tokens=4, cost_usd=0.0),
                "UNCITED",
                "success",
            ),
            "abstain": (
                "ABSTAIN",
                Usage(input_tokens=5, output_tokens=5, cost_usd=0.0),
                None,
                "abstained",
            ),
        }
        for label, (answer_text, usage, expected_verdict, outcome) in cases.items():
            with self.subTest(label=label):
                provider = _ScriptedProvider([(answer_text, usage)])
                sink = InMemoryTraceSink()
                generator = self._generator(provider, sink=sink)
                result = generator.generate(GenerationRequest(
                    query="Q?", evidence=(self.supporting, self.non_supporting),
                ))
                self.assertEqual(len(provider.calls), 1)
                self.assertEqual(result.usage_calls, (usage,))
                self.assertEqual(len(sink.events), 1)
                self.assertEqual(sink.events[0].provider_call_count, 1)
                self.assertEqual(sink.events[0].outcome, outcome)
                if label == "abstain":
                    self.assertTrue(result.abstained)
                else:
                    self.assertEqual(result.claim_support[0].verdict, expected_verdict)

    def test_provider_error_propagates_without_retry_or_trace(self):
        sentinel = _ProviderError(transient=False, status=503)
        provider = _AlwaysExplodingProvider(sentinel)
        sink = InMemoryTraceSink()
        generator = self._generator(provider, sink=sink)
        with self.assertRaises(_ProviderError) as raised:
            generator.generate(GenerationRequest(query="Q?", evidence=(self.supporting,)))
        self.assertIs(raised.exception, sentinel)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(sink.events, [])

    def test_provider_error_after_parse_failure_preserves_usage1_in_trace(self):
        malformed = "Claim [[c_AABBCCDDEEFF0011]]."
        first_usage = Usage(input_tokens=11, output_tokens=22, cost_usd=0.33)
        sentinel = _ProviderError(transient=True, status=502)
        provider = _ExplodingAfterScriptedProvider(
            first_text=malformed, first_usage=first_usage, exc=sentinel,
        )
        sink = InMemoryTraceSink()
        generator = self._generator(provider, sink=sink)
        with self.assertRaises(_ProviderError) as raised:
            generator.generate(GenerationRequest(query="Q?", evidence=(self.supporting,)))
        self.assertIs(raised.exception, sentinel)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(len(sink.events), 1)
        event = sink.events[0]
        self.assertEqual(event.outcome, "provider_error")
        self.assertTrue(event.repair_attempted)
        self.assertEqual(event.usage_calls, (first_usage,))
        self.assertEqual(event.provider_call_count, 2)
        self.assertEqual(event.evidence_fingerprint, _independent_fingerprint((self.supporting,)))
        self.assertEqual(event.prompt_filename, PROMPT_FILENAME)
        self.assertEqual(len(event.prompt_content_sha256), 64)

    def test_budget_exceeded_after_parse_failure_preserves_usage1_in_trace(self):
        from src.providers.budget import BudgetExceeded
        malformed = "Claim [[c_AABBCCDDEEFF0011]]."
        first_usage = Usage(input_tokens=11, output_tokens=22, cost_usd=0.33)
        sentinel = BudgetExceeded(cumulative=11.0, cap=10.0, unit="USD")
        provider = _ExplodingAfterScriptedProvider(
            first_text=malformed, first_usage=first_usage, exc=sentinel,
        )
        sink = InMemoryTraceSink()
        generator = self._generator(provider, sink=sink)
        with self.assertRaises(BudgetExceeded) as raised:
            generator.generate(GenerationRequest(query="Q?", evidence=(self.supporting,)))
        self.assertIs(raised.exception, sentinel)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(len(sink.events), 1)
        event = sink.events[0]
        self.assertEqual(event.outcome, "provider_error")
        self.assertTrue(event.repair_attempted)
        self.assertEqual(event.usage_calls, (first_usage,))
        self.assertEqual(event.provider_call_count, 2)
        self.assertEqual(event.evidence_fingerprint, _independent_fingerprint((self.supporting,)))
        self.assertEqual(event.prompt_filename, PROMPT_FILENAME)
        self.assertEqual(len(event.prompt_content_sha256), 64)

    def test_trace_sink_failure_propagates_without_swallowing(self):
        class _RaisingSink(GenerationTraceSink):
            def record(self, event):
                raise RuntimeError("sink boom")

        answer_text = "Dense passage retrieval [[c_0000000000000001]]."
        usage = Usage(input_tokens=4, output_tokens=4, cost_usd=0.0)
        provider = _ScriptedProvider([(answer_text, usage)])
        sink = _RaisingSink()
        generator = self._generator(provider, sink=sink)
        with self.assertRaises(RuntimeError) as raised:
            generator.generate(GenerationRequest(query="Q?", evidence=(self.supporting,)))
        self.assertEqual(str(raised.exception), "sink boom")

    def test_zero_evidence_records_zero_evidence_trace_event(self):
        sink = InMemoryTraceSink()
        generator = self._generator(_ScriptedProvider([("ABSTAIN", Usage(0, 0, 0.0))]), sink=sink)
        result = generator.generate(GenerationRequest(query="Q?", evidence=()))
        self.assertTrue(result.abstained)
        self.assertEqual(result.answer_text, "ABSTAIN")
        self.assertEqual(result.usage_calls, ())
        self.assertEqual(len(sink.events), 1)
        self.assertEqual(sink.events[0].outcome, "zero_evidence")
        self.assertEqual(sink.events[0].provider_call_count, 0)
        self.assertEqual(sink.events[0].usage_calls, ())

    def test_deterministic_equality_between_two_runs(self):
        answer_text = "Dense passage retrieval [[c_0000000000000001]]."
        usage = Usage(input_tokens=4, output_tokens=4, cost_usd=0.0)
        result_one = self._generator(_ScriptedProvider([(answer_text, usage)])).generate(
            GenerationRequest(query="Q?", evidence=(self.supporting,)))
        result_two = self._generator(_ScriptedProvider([(answer_text, usage)])).generate(
            GenerationRequest(query="Q?", evidence=(self.supporting,)))
        self.assertEqual(result_one, result_two)

    def test_null_trace_sink_is_default_and_records_nothing(self):
        generator = self._generator(_ScriptedProvider([("ABSTAIN", Usage(0, 0, 0.0))]))
        from src.grounded_gen import NullTraceSink
        self.assertIsInstance(generator._trace_sink, NullTraceSink)
        generator.generate(GenerationRequest(query="Q?", evidence=()))
        generator.generate(GenerationRequest(query="Q?", evidence=(self.supporting,)))

    def test_in_memory_sink_preserves_event_order(self):
        sink = InMemoryTraceSink()
        self.assertEqual(sink.events, [])
        event = GenerationTraceEvent(
            prompt_filename=PROMPT_FILENAME,
            prompt_content_sha256="0" * 64,
            evidence_fingerprint="abc",
            usage_calls=(Usage(1, 1, 0.0),),
            provider_call_count=1,
            outcome="success",
            repair_attempted=False,
        )
        sink.record(event)
        self.assertEqual(sink.events, [event])

if __name__ == "__main__":
    unittest.main()
