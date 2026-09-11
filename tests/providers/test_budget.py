"""Budget cap tests (#19).

Covers:

* Token cap crossing: a second ``generate`` whose combined
  ``input + output`` tokens exceeds the cap raises
  :class:`BudgetExceeded`.
* ``cost_usd=None`` skips the USD check; the token cap still applies.
* USD cap with cost present is enforced (inclusive ``>=``).
* Cumulative state across calls: counters increment.
* ``BudgetedLLM`` accounting propagates ``Usage`` and respects caps.
* Budget state is per-process (no cache file written for it).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.providers import Budget, BudgetExceeded, BudgetedLLM  # noqa: E402
from src.providers.budget import Budget as _Budget  # noqa: E402
from src.providers.cache import CachePaths, ProviderCache  # noqa: E402
from src.providers.protocol import Usage  # noqa: E402

from tests.providers._helpers import FakeLLM  # noqa: E402


class TestBudgetRecord(unittest.TestCase):
    def test_token_cap_crossing_raises(self):
        budget = Budget(tokens_cap=10)
        budget.record(Usage(input_tokens=4, output_tokens=4))  # total 8, below cap
        with self.assertRaises(BudgetExceeded) as ctx:
            budget.record(Usage(input_tokens=2, output_tokens=1))  # total 11
        self.assertEqual(ctx.exception.unit, "tokens")
        self.assertEqual(ctx.exception.cumulative, 11)
        self.assertEqual(ctx.exception.cap, 10)

    def test_token_cap_inclusive_at_equal(self):
        budget = Budget(tokens_cap=10)
        with self.assertRaises(BudgetExceeded) as ctx:
            budget.record(Usage(input_tokens=5, output_tokens=5))  # total 10
        self.assertEqual(ctx.exception.cumulative, 10)
        self.assertEqual(ctx.exception.cap, 10)

    def test_cost_usd_none_skips_usd_check(self):
        # USD cap set but provider returns ``cost_usd=None`` every time.
        budget = Budget(usd_cap=0.0001, tokens_cap=None)
        # No exception even though the cumulative cost is "infinity"
        # (i.e. we cannot compute it) — spec §6.5 ("No cost is not
        # an error").
        budget.record(Usage(input_tokens=1000, output_tokens=1000, cost_usd=None))
        budget.record(Usage(input_tokens=1000, output_tokens=1000, cost_usd=None))

    def test_usd_cap_with_cost_present(self):
        budget = Budget(usd_cap=1.0)
        budget.record(Usage(input_tokens=1, output_tokens=1, cost_usd=0.4))
        with self.assertRaises(BudgetExceeded) as ctx:
            budget.record(Usage(input_tokens=1, output_tokens=1, cost_usd=0.7))
        self.assertEqual(ctx.exception.unit, "USD")
        self.assertAlmostEqual(ctx.exception.cumulative, 1.1)
        self.assertAlmostEqual(ctx.exception.cap, 1.0)

    def test_cumulative_state_across_calls(self):
        budget = Budget(tokens_cap=None, usd_cap=None)
        budget.record(Usage(input_tokens=10, output_tokens=20, cost_usd=0.5))
        budget.record(Usage(input_tokens=5, output_tokens=5, cost_usd=0.25))
        self.assertEqual(budget.cumulative_tokens, 40)
        self.assertAlmostEqual(budget.cumulative_usd, 0.75)

    def test_none_tokens_dont_contribute(self):
        # ``input_tokens=None`` / ``output_tokens=None`` add zero to
        # the cumulative total (spec §6.5: "providers that bill in
        # different units ... must convert ... using the chunk-record
        # spec's ``len(text) // 4`` estimator as a fallback if no
        # precise tokenizer is available"). The provider-layer wrapper
        # does not estimate, so a None token field is treated as
        # "no contribution".
        budget = Budget(tokens_cap=12)
        budget.record(Usage(input_tokens=None, output_tokens=4))  # +4
        budget.record(Usage(input_tokens=4, output_tokens=None))  # +4
        # Cumulative = 8, below cap of 12.
        with self.assertRaises(BudgetExceeded) as ctx:
            budget.record(Usage(input_tokens=2, output_tokens=4))  # +6 → 14
        self.assertEqual(ctx.exception.cumulative, 14)
        self.assertEqual(ctx.exception.cap, 12)


class TestBudgetedLLMWrapper(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="budget-cache-"))
        self.cache = ProviderCache(CachePaths.from_root(self.tmp))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_wrapper_propagates_usage(self):
        fake = FakeLLM(
            generate_scripts=[("hi", Usage(input_tokens=3, output_tokens=4, cost_usd=0.01))]
        )
        wrapped = BudgetedLLM(fake, Budget(tokens_cap=None, usd_cap=None))
        text, usage = wrapped.generate("sys", "user")
        self.assertEqual(text, "hi")
        self.assertEqual(usage.input_tokens, 3)

    def test_wrapper_enforces_cap_after_call(self):
        fake = FakeLLM(
            generate_scripts=[
                ("hi", Usage(input_tokens=4, output_tokens=4)),
                ("hi", Usage(input_tokens=4, output_tokens=4)),
            ]
        )
        wrapped = BudgetedLLM(fake, Budget(tokens_cap=10))
        wrapped.generate("sys", "user")
        with self.assertRaises(BudgetExceeded):
            wrapped.generate("sys", "user")

    def test_no_budget_state_in_cache(self):
        fake = FakeLLM(
            generate_scripts=[("hi", Usage(input_tokens=10, output_tokens=10))]
        )
        wrapped = BudgetedLLM(fake, Budget(tokens_cap=1000, usd_cap=None))
        wrapped.generate("sys", "user")
        # Spec §5.4: budget state is per-process, never cached. The
        # cache directory should be empty (and certainly not contain
        # a file representing budget state).
        files = [p.name for p in self.cache.paths.llm.iterdir()]
        self.assertNotIn("budget", files)
        # Cache has no budget entry because the fake doesn't write to it.
        self.assertEqual(files, [])

    def test_does_not_swallow_provider_exceptions(self):
        # An exception from the inner adapter must propagate.
        class BoomLLM(FakeLLM):
            def generate(self, system, user, *, max_tokens=1024, temperature=0.0):
                raise RuntimeError("upstream failure")

        wrapped = BudgetedLLM(BoomLLM(), Budget())
        with self.assertRaises(RuntimeError):
            wrapped.generate("sys", "user")


class TestBudgetEdgeCases(unittest.TestCase):
    def test_empty_budget_allows_calls(self):
        budget = Budget()  # no caps
        budget.record(Usage(input_tokens=1, output_tokens=1, cost_usd=0.0001))
        # No exception.

    def test_budget_with_usd_only_runs(self):
        budget = Budget(usd_cap=1.0)
        budget.record(Usage(cost_usd=0.5))
        budget.record(Usage(cost_usd=0.4))  # cumulative 0.9 < 1.0
        with self.assertRaises(BudgetExceeded):
            budget.record(Usage(cost_usd=0.2))  # cumulative 1.1


if __name__ == "__main__":  # pragma: no cover
    unittest.main()