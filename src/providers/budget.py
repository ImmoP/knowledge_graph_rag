"""Budget cap wrapper around an ``LLMProvider`` (#19).

Resolution of ``docs/specs/provider-abstractions.md`` §6: a thin
wrapper that decorates the active LLM adapter, accumulates ``Usage``
after every call, and raises :class:`BudgetExceeded` when the
configured ``LLM_BUDGET_USD`` or ``LLM_BUDGET_TOKENS`` cap is crossed.

Cap semantics (spec §6):

* USD cap is enforced iff a value is configured **and** the provider
  returned a non-``None`` ``cost_usd`` for the call. ``cost_usd=None``
  is not treated as zero-priced evidence; it skips the USD check for
  that call.
* Token cap is enforced iff a value is configured; it uses
  ``input_tokens + output_tokens`` where both are available, treating
  ``None`` as "no contribution".
* Comparison is inclusive: ``cumulative >= cap`` raises.
* State is per-process and never cached (spec §5.4).

Embedding calls are not wrapped — spec §6 budgets LLM cost/tokens
only. Embedding cost is outside the cap surface for #19.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.providers.protocol import LLMProvider, Usage


class BudgetExceeded(Exception):
    """Raised when cumulative usage crosses a configured cap.

    ``unit`` is ``"USD"`` or ``"tokens"``; ``cap`` is the configured
    value that was crossed; ``cumulative`` is the running total at the
    point of raise. The calling script catches this, logs the partial
    run, and exits non-zero (spec §6.4).
    """

    def __init__(self, *, cumulative: float, cap: float, unit: str) -> None:
        super().__init__(
            f"BudgetExceeded: cumulative {unit}={cumulative} reached cap {cap}"
        )
        self.cumulative = cumulative
        self.cap = cap
        self.unit = unit


@dataclass
class Budget:
    """Per-process budget state, separate from the wrapper so tests
    can drive accounting deterministically.

    ``usd_cap`` and ``tokens_cap`` of ``None`` mean "no cap" for that
    unit. The wrapper raises :class:`BudgetExceeded` exactly once the
    inclusive ``>=`` comparison crosses a configured cap; see spec §6.3.
    """

    usd_cap: Optional[float] = None
    tokens_cap: Optional[int] = None
    cumulative_usd: float = 0.0
    cumulative_tokens: int = 0

    def record(self, usage: Usage) -> None:
        """Add ``usage`` to the cumulative counters and check caps.

        Spec §6.2 says "the wrapper adds to its cumulative counters"
        after every call, so we accumulate even when no cap is
        configured — that way :class:`Budget` carries a true post-hoc
        ledger of LLM spend/tokens for callers that want it.

        Cap order matches spec §6.3: USD is checked first when a USD
        cap is configured **and** this call reported a non-``None``
        ``cost_usd``; tokens are checked whenever a token cap is
        configured (using whatever token fields the call reported).
        """
        if usage.cost_usd is not None:
            self.cumulative_usd += float(usage.cost_usd)
            if self.usd_cap is not None and self.cumulative_usd >= self.usd_cap:
                raise BudgetExceeded(
                    cumulative=self.cumulative_usd,
                    cap=self.usd_cap,
                    unit="USD",
                )
        added = _token_count(usage)
        if added:
            self.cumulative_tokens += added
        if self.tokens_cap is not None and self.cumulative_tokens >= self.tokens_cap:
            raise BudgetExceeded(
                cumulative=self.cumulative_tokens,
                cap=self.tokens_cap,
                unit="tokens",
            )


def _token_count(usage: Usage) -> int:
    """Sum ``input_tokens`` + ``output_tokens`` treating ``None`` as 0.

    Spec §6.1 token counts are reported as integers. A provider that
    does not expose precise counts leaves both ``None`` and contributes
    nothing to the cumulative total — the calling spec's estimator is
    the fallback only when explicitly permitted (spec §6.1 final
    sentence). The provider-layer wrapper does not estimate.
    """
    n = 0
    if usage.input_tokens is not None:
        n += usage.input_tokens
    if usage.output_tokens is not None:
        n += usage.output_tokens
    return n


class BudgetedLLM:
    """Wrap an ``LLMProvider`` with cumulative budget enforcement.

    Delegates ``name`` and ``model`` through to the inner adapter (so
    cache-key construction sees the real provider identity) and
    delegates every call to it before accumulating ``Usage`` and
    checking caps.

    Embedding adapters live outside this surface — spec §6 budgets LLM
    calls only.
    """

    name: str
    model: str

    def __init__(self, inner: LLMProvider, budget: Budget) -> None:
        self._inner = inner
        self._budget = budget
        self.name = inner.name
        self.model = inner.model

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[str, Usage]:
        text, usage = self._inner.generate(
            system,
            user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._budget.record(usage)
        return text, usage

    def generate_json(
        self,
        system: str,
        user: str,
        schema: dict,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[dict, Usage]:
        parsed, usage = self._inner.generate_json(
            system,
            user,
            schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._budget.record(usage)
        return parsed, usage