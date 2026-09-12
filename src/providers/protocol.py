"""Provider protocols for #19.

Resolution of ``docs/specs/provider-abstractions.md`` §1.

Two ``Protocol`` interfaces and a frozen ``Usage`` dataclass. Application
code consumes these; it never imports a provider SDK. Adapters are
identified structurally by method shape — the spec explicitly rejected
ABC inheritance so adapters need not import this module to satisfy the
contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Usage:
    """Per-call accounting returned by every adapter method.

    ``cost_usd`` is ``None`` when the adapter / provider cannot report
    cost; the budget layer treats ``None`` as "USD cap not enforced for
    this call", not as a zero-priced evidence point. Token counts use
    provider-native precision; adapters that cannot report either
    ``input_tokens`` or ``output_tokens`` set the field to ``None`` and
    rely on the chunk-record ``len(text) // 4`` fallback only where
    explicitly permitted by the calling spec.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Natural-language + JSON-structured LLM surface (spec §1)."""

    name: str
    model: str

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[str, Usage]: ...

    def generate_json(
        self,
        system: str,
        user: str,
        schema: dict,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[dict, Usage]: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Batch embedding surface (spec §1)."""

    name: str
    model: str
    dim: int

    def embed(self, texts: list[str]) -> tuple[list[list[float]], Usage]: ...


class ProviderTransportError(Exception):
    """Adapter-reported transport-layer or HTTP failure (spec §6.6, repair #23).

    Adapters raise this for any transport-layer or HTTP failure. ``transient``
    is the caller's retry-eligibility flag (extraction uses extraction-contract
    §12.1 bounded retry on ``transient=True`` only). ``status`` is the HTTP
    status code if known, else ``None``. ``cause`` preserves the underlying
    SDK exception for logging.

    ``LLMValidationError`` (transport-success, output-invalid) and
    ``BudgetExceeded`` (cap-exceeded) are separate exception types and
    propagate independently — adapters MUST NOT convert them to
    :class:`ProviderTransportError`.

    Adapters MUST wrap only known transport/status failures from their
    outbound HTTP boundary; they MUST NOT use a broad ``except Exception``.
    """

    def __init__(
        self,
        *,
        transient: bool,
        status: int | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.transient = transient
        self.status = status
        self.cause = cause
        detail = f"transient={transient}"
        if status is not None:
            detail += f" status={status}"
        super().__init__(detail)
        if cause is not None:
            self.__cause__ = cause


class LLMValidationError(Exception):
    """Raised when ``generate_json`` cannot produce schema-valid output.

    Carries the model's actual text output, the schema it was supposed
    to satisfy, and the underlying validation error so the calling
    pipeline can decide whether to log-and-skip or fail the run. Per
    spec §2 the adapter has already retried exactly once before
    raising this.
    """

    def __init__(self, model_output: str, schema: dict, validation_error: Exception):
        super().__init__(
            f"LLM output failed schema validation after one retry: {validation_error}"
        )
        self.model_output = model_output
        self.schema = schema
        self.validation_error = validation_error