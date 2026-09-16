"""Grounded-generation contracts, evidence preflight, parsing, and orchestration.

Phase 3 wires the accepted Phase-1 preflight/prompt renderer and the
Phase-2 pure parser to ``LLMProvider.generate`` with an exact-one
parse/protocol retry and a minimal injectable trace seam.
"""


from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.providers.protocol import LLMProvider, Usage


CHUNK_ID_RE = re.compile(r"^c_[0-9a-f]{16}$")
DOC_ID_RE = re.compile(r"^W[0-9]+$")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_PATH = REPO_ROOT / "data" / "derived" / "chunks.jsonl"
PROMPT_PATH = REPO_ROOT / "prompts" / "grounded_answer_v1.txt"
PROMPT_FILENAME = "prompts/grounded_answer_v1.txt"
SYSTEM_PROMPT = "Answer only from the supplied evidence."
CITATION_TOKEN_RE = re.compile(r"\[\[(c_[0-9a-f]{16})\]\]")
BRACKET_CONSTRUCT_RE = re.compile(r"\[\[.*?(?:\]\]|$)|\[\s+\[.*?\]\s+\]", re.DOTALL)
CLAIM_SEPARATOR_RE = re.compile(r"\. |\? |! |\n\n")


@dataclass(frozen=True)
class EvidenceBlock:
    chunk_id: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class GenerationRequest:
    query: str
    evidence: tuple[EvidenceBlock, ...]
    max_tokens: int = 1024
    temperature: float = 0.0


@dataclass(frozen=True)
class CitationOccurrence:
    chunk_id: str
    claim_id: int
    char_start: int
    char_end: int
    status: Literal[
        "IN_EVIDENCE_SUPPORTING",
        "IN_EVIDENCE_NON_SUPPORTING",
        "OUT_OF_EVIDENCE",
    ]


@dataclass(frozen=True)
class Claim:
    claim_id: int
    text: str
    char_start: int
    char_end: int
    scorable: bool
    attached_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class ClaimSupportVerdict:
    claim_id: int
    verdict: Literal["SUPPORTED", "CITED_UNSUPPORTED", "UNCITED"]
    supporting_chunk_ids: tuple[str, ...]
    non_supporting_chunk_ids: tuple[str, ...]
    out_of_evidence_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class GenerationResult:
    answer_text: str
    abstained: bool
    citations: tuple[CitationOccurrence, ...]
    claims: tuple[Claim, ...]
    claim_support: tuple[ClaimSupportVerdict, ...]
    usage_calls: tuple[Usage, ...]
    evidence_fingerprint: str


class EvidenceIntegrityError(RuntimeError):
    """The caller evidence does not match the current canonical artifact."""

OUTCOME_SUCCESS = "success"
OUTCOME_ABSTAINED = "abstained"
OUTCOME_PARSE_ERROR = "parse_error"
OUTCOME_ZERO_EVIDENCE = "zero_evidence"
OUTCOME_PROVIDER_ERROR = "provider_error"


class GenerationTraceEvent:
    """Deterministic generation-trace payload consumed by #32/#33.

    The generator emits one of these on every non-preflight-rejected call.
    The trace seam is intentionally minimal: this dataclass does NOT add
    benchmark / gold / retrieval / credential fields.
    """

    __slots__ = (
        "prompt_filename",
        "prompt_content_sha256",
        "evidence_fingerprint",
        "usage_calls",
        "provider_call_count",
        "outcome",
        "repair_attempted",
    )

    def __init__(
        self,
        *,
        prompt_filename: str,
        prompt_content_sha256: str,
        evidence_fingerprint: str,
        usage_calls: tuple[Usage, ...],
        provider_call_count: int,
        outcome: str,
        repair_attempted: bool,
    ) -> None:
        self.prompt_filename = prompt_filename
        self.prompt_content_sha256 = prompt_content_sha256
        self.evidence_fingerprint = evidence_fingerprint
        self.usage_calls = usage_calls
        self.provider_call_count = provider_call_count
        self.outcome = outcome
        self.repair_attempted = repair_attempted


class GenerationTraceSink:
    """Injectable application seam for the deterministic generation trace.

    Application code may subclass this and pass it to
    ``GroundedGenerator``; tests use the in-memory ``InMemoryTraceSink``.
    A sink failure MUST NOT alter generation semantics: exceptions raised
    by ``record`` propagate to the caller, because suppressing them would
    hide audit-log bugs behind silent dropouts.
    """

    def record(self, event: GenerationTraceEvent) -> None:
        raise NotImplementedError


class InMemoryTraceSink(GenerationTraceSink):
    """Minimal in-memory sink for tests."""

    def __init__(self) -> None:
        self.events: list[GenerationTraceEvent] = []

    def record(self, event: GenerationTraceEvent) -> None:
        self.events.append(event)


class NullTraceSink(GenerationTraceSink):
    """Default sink that records nothing."""

    def record(self, event: GenerationTraceEvent) -> None:
        return None


FORMAT_REPAIR_SUFFIX = (
    "\n\nRe-answer while obeying the supplied output protocol exactly:"
    " use only canonical [[c_<16 lowercase hex>]] citation tokens;"
    " place each citation token immediately before its sentence-ending"
    " punctuation; if the evidence is insufficient, output exactly"
    " ABSTAIN alone on its own line; do not include headings,"
    " a References section, or any meta-commentary.\n"
)


class GenerationParseError(RuntimeError):
    """Raised when the parse/protocol retry also fails.

    Carries the deterministic audit context that ``#32``/``#33`` need:
    the ``usage_calls`` (one ``Usage`` per actual provider call, in call
    order) and the ``evidence_fingerprint``. The exception shape is not
    part of the canonical ``GenerationResult`` contract; only the
    generator's audit trace and error path inspect it.
    """

    def __init__(
        self,
        message: str,
        *,
        usage_calls: tuple[Usage, ...] = (),
        evidence_fingerprint: str = "",
    ) -> None:
        super().__init__(message)
        self.usage_calls = usage_calls
        self.evidence_fingerprint = evidence_fingerprint


class _ProtocolError(RuntimeError):
    """A pure answer violates the citation/claim wire protocol."""


def lexical_tokens(text: str) -> tuple[str, ...]:
    """NFKC/casefold Unicode-alphanumeric tokens; underscores are separators."""
    normalized = unicodedata.normalize("NFKC", CITATION_TOKEN_RE.sub("", text)).casefold()
    tokens: list[str] = []
    token: list[str] = []
    for character in normalized:
        if character.isalnum():
            token.append(character)
        elif token:
            tokens.append("".join(token))
            token.clear()
    if token:
        tokens.append("".join(token))
    return tuple(tokens)


def lexical_support(claim_text: str, evidence_text: str) -> bool:
    """Return whether one pinned claim n-gram occurs in the evidence tokens."""
    claim_tokens = lexical_tokens(claim_text)
    if not claim_tokens:
        return False
    evidence_tokens = lexical_tokens(evidence_text)
    ngram_size = min(4, len(claim_tokens))
    evidence_windows = {
        evidence_tokens[index:index + ngram_size]
        for index in range(len(evidence_tokens) - ngram_size + 1)
    }
    return any(
        claim_tokens[index:index + ngram_size] in evidence_windows
        for index in range(len(claim_tokens) - ngram_size + 1)
    )


def _claim_spans(answer_text: str) -> list[tuple[int, int]]:
    """Split only on the four pinned separators, retaining terminal punctuation."""
    spans: list[tuple[int, int]] = []
    start = 0
    for separator in CLAIM_SEPARATOR_RE.finditer(answer_text):
        end = separator.start() + (1 if separator.group()[0] in ".?!" else 0)
        left = start
        right = end
        while left < right and answer_text[left].isspace():
            left += 1
        while right > left and answer_text[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append((left, right))
        start = separator.end()
    left = start
    right = len(answer_text)
    while left < right and answer_text[left].isspace():
        left += 1
    while right > left and answer_text[right - 1].isspace():
        right -= 1
    if left < right:
        spans.append((left, right))
    return spans


def parse_and_validate_answer(
    answer_text: str, evidence: tuple[EvidenceBlock, ...]
) -> tuple[
    bool,
    tuple[CitationOccurrence, ...],
    tuple[Claim, ...],
    tuple[ClaimSupportVerdict, ...],
]:
    """Pure deterministic answer parser; protocol errors remain retryable."""
    if answer_text == "ABSTAIN":
        return True, (), (), ()
    if not isinstance(answer_text, str):
        raise TypeError("answer_text must be a str")

    valid_matches = list(CITATION_TOKEN_RE.finditer(answer_text))
    valid_ranges = {(match.start(), match.end()) for match in valid_matches}
    for construct in BRACKET_CONSTRUCT_RE.finditer(answer_text):
        if construct.span() not in valid_ranges:
            raise _ProtocolError("malformed citation bracket construct")

    spans = _claim_spans(answer_text)
    if not spans:
        raise _ProtocolError("answer has no segmentable claims")

    match_claim_ids: list[int] = []
    for match in valid_matches:
        claim_id = next(
            (index for index, (start, end) in enumerate(spans) if start <= match.start() < end),
            None,
        )
        if claim_id is None:
            raise _ProtocolError("citation does not attach to a claim")
        match_claim_ids.append(claim_id)

    cited_ids_by_claim: list[list[str]] = [[] for _ in spans]
    for match, claim_id in zip(valid_matches, match_claim_ids):
        chunk_id = match.group(1)
        if chunk_id not in cited_ids_by_claim[claim_id]:
            cited_ids_by_claim[claim_id].append(chunk_id)
    claims = tuple(
        Claim(
            claim_id=index,
            text=CITATION_TOKEN_RE.sub("", answer_text[start:end]).strip(),
            char_start=start,
            char_end=end,
            scorable=bool(lexical_tokens(answer_text[start:end])),
            attached_chunk_ids=tuple(cited_ids_by_claim[index]),
        )
        for index, (start, end) in enumerate(spans)
    )
    evidence_by_id = {block.chunk_id: block for block in evidence}
    citations: list[CitationOccurrence] = []
    statuses_by_claim: list[dict[str, str]] = [{} for _ in claims]
    for match, claim_id in zip(valid_matches, match_claim_ids):
        chunk_id = match.group(1)
        block = evidence_by_id.get(chunk_id)
        if block is None:
            status = "OUT_OF_EVIDENCE"
        elif lexical_support(claims[claim_id].text, block.text):
            status = "IN_EVIDENCE_SUPPORTING"
        else:
            status = "IN_EVIDENCE_NON_SUPPORTING"
        citations.append(CitationOccurrence(chunk_id, claim_id, match.start(), match.end(), status))
        statuses_by_claim[claim_id].setdefault(chunk_id, status)

    verdicts: list[ClaimSupportVerdict] = []
    for claim, statuses in zip(claims, statuses_by_claim):
        supporting = tuple(chunk_id for chunk_id in claim.attached_chunk_ids if statuses[chunk_id] == "IN_EVIDENCE_SUPPORTING")
        non_supporting = tuple(chunk_id for chunk_id in claim.attached_chunk_ids if statuses[chunk_id] == "IN_EVIDENCE_NON_SUPPORTING")
        out_of_evidence = tuple(chunk_id for chunk_id in claim.attached_chunk_ids if statuses[chunk_id] == "OUT_OF_EVIDENCE")
        verdict = "SUPPORTED" if supporting else "CITED_UNSUPPORTED" if claim.attached_chunk_ids else "UNCITED"
        verdicts.append(ClaimSupportVerdict(claim.claim_id, verdict, supporting, non_supporting, out_of_evidence))
    return False, tuple(citations), claims, tuple(verdicts)

def evidence_fingerprint(evidence: tuple[EvidenceBlock, ...]) -> str:
    """Return the pinned SHA256 over the ordered evidence identity array."""
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
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_canonical_chunks(path: Path) -> dict[str, tuple[str, str]]:
    """Load the identity fields needed to validate caller evidence.

    This intentionally does not reuse the vector loader: generation only needs
    canonical identity, not embedding-token validation.
    """
    try:
        with path.open("r", encoding="utf-8") as artifact:
            lines = list(artifact)
    except (OSError, UnicodeError) as exc:
        raise EvidenceIntegrityError(
            f"canonical chunks artifact is unreadable: {path}"
        ) from exc

    canonical: dict[str, tuple[str, str]] = {}
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            raise EvidenceIntegrityError(
                f"canonical chunks line {line_no} is blank"
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceIntegrityError(
                f"canonical chunks line {line_no} is malformed JSON"
            ) from exc
        if not isinstance(record, dict):
            raise EvidenceIntegrityError(
                f"canonical chunks line {line_no} is not an object"
            )
        try:
            chunk_id = record["chunk_id"]
            doc_id = record["doc_id"]
            text = record["text"]
        except KeyError as exc:
            raise EvidenceIntegrityError(
                f"canonical chunks line {line_no} lacks {exc.args[0]!r}"
            ) from exc
        if not isinstance(chunk_id, str) or not CHUNK_ID_RE.fullmatch(chunk_id):
            raise EvidenceIntegrityError(
                f"canonical chunks line {line_no} has malformed chunk_id"
            )
        if not isinstance(doc_id, str) or not DOC_ID_RE.fullmatch(doc_id):
            raise EvidenceIntegrityError(
                f"canonical chunks line {line_no} has malformed doc_id"
            )
        if not isinstance(text, str):
            raise EvidenceIntegrityError(
                f"canonical chunks line {line_no} has non-string text"
            )
        if chunk_id in canonical:
            raise EvidenceIntegrityError(
                f"canonical chunks has duplicate chunk_id {chunk_id!r}"
            )
        canonical[chunk_id] = (doc_id, text)
    return canonical


def _render_evidence(evidence: tuple[EvidenceBlock, ...]) -> str:
    """Render each caller block verbatim in its supplied order."""
    return "\n\n".join(
        f"[Evidence block {index}]\n"
        f"chunk_id: {block.chunk_id}\n"
        f"doc_id: {block.doc_id}\n"
        f"text:\n{block.text}"
        for index, block in enumerate(evidence)
    )


class GroundedGenerator:
    """Canonical evidence gate, prompt renderer, and provider orchestration.

    Phase 1 owns preflight + prompt rendering. Phase 2 owns the pure
    parser. Phase 3 wires the two to ``LLMProvider.generate`` with one
    parse/protocol retry and a deterministic trace seam for #32/#33.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        canonical_chunks_path: Path = DEFAULT_CHUNKS_PATH,
        trace_sink: GenerationTraceSink | None = None,
    ) -> None:
        self._provider = provider
        self._canonical_chunks_path = Path(canonical_chunks_path)
        self._trace_sink: GenerationTraceSink = trace_sink or NullTraceSink()

    def preflight(self, request: GenerationRequest) -> None:
        """Reject malformed or stale evidence before any generation action."""
        if not isinstance(request.query, str):
            raise TypeError("GenerationRequest.query must be a str")

        canonical = _load_canonical_chunks(self._canonical_chunks_path)
        seen_chunk_ids: set[str] = set()
        for block in request.evidence:
            if not isinstance(block.chunk_id, str) or not CHUNK_ID_RE.fullmatch(
                block.chunk_id
            ):
                raise EvidenceIntegrityError("malformed evidence chunk_id")
            if not isinstance(block.doc_id, str) or not DOC_ID_RE.fullmatch(block.doc_id):
                raise EvidenceIntegrityError("malformed evidence doc_id")
            if block.chunk_id in seen_chunk_ids:
                raise EvidenceIntegrityError(
                    f"duplicate request chunk_id {block.chunk_id!r}"
                )
            seen_chunk_ids.add(block.chunk_id)
            if not isinstance(block.text, str):
                raise EvidenceIntegrityError("evidence text must be a str")
            if canonical.get(block.chunk_id) != (block.doc_id, block.text):
                raise EvidenceIntegrityError(
                    f"evidence does not match canonical chunk {block.chunk_id!r}"
                )

    def render_user_prompt(self, request: GenerationRequest) -> str:
        """Render the versioned template and caller evidence deterministically."""
        try:
            template = PROMPT_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise EvidenceIntegrityError(f"prompt artifact is unreadable: {PROMPT_PATH}") from exc
        content_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
        rendered_template = template.format(
            query=request.query,
            evidence=_render_evidence(request.evidence),
        )
        return (
            f"Prompt template filename: {PROMPT_FILENAME}\n"
            f"Prompt template content SHA256: {content_sha256}\n\n"
            f"{rendered_template}"
        )

    def _prompt_content_sha256(self) -> str:
        """Return the SHA256 of the pinned prompt template, computed at call time."""
        try:
            template = PROMPT_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise EvidenceIntegrityError(f"prompt artifact is unreadable: {PROMPT_PATH}") from exc
        return hashlib.sha256(template.encode("utf-8")).hexdigest()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run preflight, call ``LLMProvider.generate``, parse, and optionally retry.

        Behaviour:
          * Zero evidence → deterministic ``ABSTAIN``, no provider call,
            ``usage_calls == ()``, optional zero-evidence trace event.
          * Non-empty evidence → exactly one provider call; on parse/protocol
            failure, exactly one retry with the pinned ``FORMAT_REPAIR_SUFFIX``
            appended to the user prompt. Provider/budget/transport errors
            propagate unchanged.
        """
        self.preflight(request)
        fingerprint = evidence_fingerprint(request.evidence)
        prompt_sha = self._prompt_content_sha256()
        user_prompt = self.render_user_prompt(request)

        if not request.evidence:
            self._record_trace(
                GenerationTraceEvent(
                    prompt_filename=PROMPT_FILENAME,
                    prompt_content_sha256=prompt_sha,
                    evidence_fingerprint=fingerprint,
                    usage_calls=(),
                    provider_call_count=0,
                    outcome=OUTCOME_ZERO_EVIDENCE,
                    repair_attempted=False,
                )
            )
            return GenerationResult(
                answer_text="ABSTAIN",
                abstained=True,
                citations=(),
                claims=(),
                claim_support=(),
                usage_calls=(),
                evidence_fingerprint=fingerprint,
            )

        first_text, first_usage = self._provider.generate(
            SYSTEM_PROMPT,
            user_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        usage_calls: list[Usage] = [first_usage]
        answer_text = first_text
        repair_attempted = False
        try:
            result = self._build_result(answer_text, request.evidence, tuple(usage_calls), fingerprint)
        except _ProtocolError:
            repair_attempted = True
            try:
                second_text, second_usage = self._provider.generate(
                    SYSTEM_PROMPT,
                    user_prompt + FORMAT_REPAIR_SUFFIX,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
            except Exception:
                self._record_trace(
                    GenerationTraceEvent(
                        prompt_filename=PROMPT_FILENAME,
                        prompt_content_sha256=prompt_sha,
                        evidence_fingerprint=fingerprint,
                        usage_calls=tuple(usage_calls),
                        provider_call_count=2,
                        outcome=OUTCOME_PROVIDER_ERROR,
                        repair_attempted=repair_attempted,
                    )
                )
                raise
            usage_calls.append(second_usage)
            answer_text = second_text
            try:
                result = self._build_result(
                    answer_text, request.evidence, tuple(usage_calls), fingerprint
                )
            except _ProtocolError as exc:
                self._record_trace(
                    GenerationTraceEvent(
                        prompt_filename=PROMPT_FILENAME,
                        prompt_content_sha256=prompt_sha,
                        evidence_fingerprint=fingerprint,
                        usage_calls=tuple(usage_calls),
                        provider_call_count=len(usage_calls),
                        outcome=OUTCOME_PARSE_ERROR,
                        repair_attempted=repair_attempted,
                    )
                )
                raise GenerationParseError(
                    f"parse/protocol failure after retry: {exc}",
                    usage_calls=tuple(usage_calls),
                    evidence_fingerprint=fingerprint,
                )

        if result.abstained:
            outcome = OUTCOME_ABSTAINED
        else:
            outcome = OUTCOME_SUCCESS
        self._record_trace(
            GenerationTraceEvent(
                prompt_filename=PROMPT_FILENAME,
                prompt_content_sha256=prompt_sha,
                evidence_fingerprint=fingerprint,
                usage_calls=result.usage_calls,
                provider_call_count=len(result.usage_calls),
                outcome=outcome,
                repair_attempted=repair_attempted,
            )
        )
        return result

    def _build_result(
        self,
        answer_text: str,
        evidence: tuple[EvidenceBlock, ...],
        usage_calls: tuple[Usage, ...],
        fingerprint: str,
    ) -> GenerationResult:
        """Parse a provider response into a ``GenerationResult``.

        Only the Phase-2 pure ``_ProtocolError`` family signals a parse
        failure; model outcomes such as ``OUT_OF_EVIDENCE`` or
        ``CITED_UNSUPPORTED`` flow through as valid results.
        """
        abstained, citations, claims, verdicts = parse_and_validate_answer(
            answer_text, evidence
        )
        return GenerationResult(
            answer_text=answer_text,
            abstained=abstained,
            citations=citations,
            claims=claims,
            claim_support=verdicts,
            usage_calls=usage_calls,
            evidence_fingerprint=fingerprint,
        )

    def _record_trace(self, event: GenerationTraceEvent) -> None:
        """Forward a trace event to the configured sink without suppression."""
        self._trace_sink.record(event)



def _smoke_main() -> None:
    """Deterministic offline smoke: run the real generator over a tiny fixture."""
    import json
    import os
    import tempfile

    class _ScriptedProvider(LLMProvider):
        name = "smoke"
        model = "smoke-v1"

        def __init__(self, text: str, usage: Usage) -> None:
            self._text = text
            self._usage = usage

        def generate(
            self,
            system: str,
            user: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.0,
        ) -> tuple[str, Usage]:
            return self._text, self._usage

        def generate_json(
            self,
            system: str,
            user: str,
            schema: dict,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.0,
        ) -> tuple[dict, Usage]:
            raise AssertionError("generate_json is not part of the generator surface")

    supporting = EvidenceBlock(
        "c_0000000000000001",
        "W1",
        "Dense passage retrieval supports open domain question answering.",
    )
    non_supporting = EvidenceBlock(
        "c_0000000000000002",
        "W2",
        "An unrelated astronomy passage about planetary orbits and seasons.",
    )
    answer_text = (
        "Dense passage retrieval supports open domain questions "
        "[[c_0000000000000001]]. "
        "An unrelated astronomy statement [[c_0000000000000002]]. "
        "An out-of-evidence reference [[c_0000000000000003]]. "
        "An uncited trailing remark."
    )
    with tempfile.TemporaryDirectory() as tmp:
        chunks_path = Path(tmp) / "chunks.jsonl"
        chunks_path.write_text(
            "\n".join(
                json.dumps(
                    {"chunk_id": block.chunk_id, "doc_id": block.doc_id, "text": block.text},
                    ensure_ascii=False,
                )
                for block in (supporting, non_supporting)
            )
            + "\n",
            encoding="utf-8",
        )
        provider = _ScriptedProvider(answer_text, Usage(input_tokens=12, output_tokens=34, cost_usd=0.0))
        generator = GroundedGenerator(provider, canonical_chunks_path=chunks_path)
        result = generator.generate(
            GenerationRequest(query="What does DPR do?", evidence=(supporting, non_supporting))
        )
        payload = {
            "answer_text": result.answer_text,
            "abstained": result.abstained,
            "citations": [
                {
                    "chunk_id": citation.chunk_id,
                    "claim_id": citation.claim_id,
                    "char_start": citation.char_start,
                    "char_end": citation.char_end,
                    "status": citation.status,
                }
                for citation in result.citations
            ],
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "text": claim.text,
                    "char_start": claim.char_start,
                    "char_end": claim.char_end,
                    "scorable": claim.scorable,
                    "attached_chunk_ids": list(claim.attached_chunk_ids),
                }
                for claim in result.claims
            ],
            "claim_support": [
                {
                    "claim_id": verdict.claim_id,
                    "verdict": verdict.verdict,
                    "supporting_chunk_ids": list(verdict.supporting_chunk_ids),
                    "non_supporting_chunk_ids": list(verdict.non_supporting_chunk_ids),
                    "out_of_evidence_chunk_ids": list(verdict.out_of_evidence_chunk_ids),
                }
                for verdict in result.claim_support
            ],
            "usage_calls": [
                {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cost_usd": usage.cost_usd,
                }
                for usage in result.usage_calls
            ],
            "evidence_fingerprint": result.evidence_fingerprint,
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    _ = os


if __name__ == "__main__":
    _smoke_main()
