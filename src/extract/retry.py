"""Bounded transport-retry policy for extraction (#23).

Implements ``docs/specs/extraction-contract.md`` §12 verbatim.

* Only :class:`src.providers.protocol.ProviderTransportError` with
  ``transient=True`` qualifies for retry. The transient flag is the
  adapter's contract per provider-abstractions §6.6.
* Schedule: attempt 1 immediately, then ``sleep(2.0)`` before attempt
  2, then ``sleep(4.0)`` before attempt 3. No ``sleep(0)``, no
  sleep after attempt 3, exactly 3 attempts total.
* On a third transient failure the final :class:`ProviderTransportError`
  is re-raised unchanged — there is no ``RetryExhausted`` wrapper.
* Non-transient :class:`ProviderTransportError`,
  :class:`LLMValidationError`, :class:`BudgetExceeded`, ontology and
  evidence failures, and arbitrary exceptions propagate immediately.
* The retry helper is provider-independent. It does not import
  ``httpx``, ``requests``, ``openai``, ``anthropic``, or any SDK; the
  classifier inspects only the typed ``transient`` attribute.

The ``sleep`` callable is injectable so tests can assert the exact
schedule without actually waiting.
"""

from __future__ import annotations

from typing import Callable

from src.providers.protocol import ProviderTransportError


# Spec §12.1 — schedule is pinned: 2 s before attempt 2, 4 s before
# attempt 3. Attempt 1 has no pre-sleep (``sleep(0)`` is forbidden).
BACKOFF_SCHEDULE_S: tuple[float, ...] = (2.0, 4.0)
MAX_ATTEMPTS = len(BACKOFF_SCHEDULE_S) + 1  # 3 (attempt 1 + two retries)


def is_transient(exc: BaseException) -> bool:
    """Return ``True`` iff ``exc`` is a transient transport failure.

    Per provider-abstractions §6.6 and extraction-contract §12.1 the
    only retryable transport failure is a
    :class:`ProviderTransportError` whose ``transient`` flag is
    ``True``. There are no string-matching, regex, or status-code
    inspection rules: the adapter decides transient-vs-not when it
    raises.
    """
    return isinstance(exc, ProviderTransportError) and exc.transient is True


def with_transport_retry(
    op: Callable[[], object],
    *,
    sleep: Callable[[float], None],
) -> object:
    """Run ``op()`` with bounded transport retry per spec §12.1.

    * Maximum 3 attempts total.
    * No sleep before attempt 1.
    * Sleep 2 s before attempt 2; sleep 4 s before attempt 3.
    * Sleep callable is injectable; tests pass a recorder.
    * Only :class:`ProviderTransportError` with ``transient=True``
      is retried.
    * The final transient :class:`ProviderTransportError` is re-raised
      unchanged on attempt 3.
    * Every other exception type propagates immediately.
    """
    # Attempt 1 — no pre-sleep.
    try:
        return op()
    except ProviderTransportError as exc:
        if not is_transient(exc):
            raise
    # Attempt 2 — sleep 2 s first.
    sleep(BACKOFF_SCHEDULE_S[0])
    try:
        return op()
    except ProviderTransportError as exc:
        if not is_transient(exc):
            raise
    # Attempt 3 — sleep 4 s first.
    sleep(BACKOFF_SCHEDULE_S[1])
    return op()
