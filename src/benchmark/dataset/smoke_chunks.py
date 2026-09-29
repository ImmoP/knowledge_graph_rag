"""Pinned smoke chunks for the initial limited live comparison.

The smoke comparison runs at most four manifest calls and exactly
two local Qwen3 calls. The two chunks below are the only ones that
participate in this initial comparison; both are known to the gold
sampler (``data/gold_set/sample.v2.jsonl``) and both have a stored
Qwen3 baseline response in the prior pilot cache.
"""

# Pinned order matters: the smoke runner charges calls in this
# order and records a per-chunk entry into the report.
SMOKE_CHUNK_IDS: tuple[str, ...] = (
    "c_0001921855a8156b",
    "c_001452fd142ce075",
)

# Hard cap on live gateway calls for the entire smoke comparison.
# Two chunks × two phases (entity + relation) = 4 calls when nothing
# is cached. The runner aborts on the fifth attempt.
SMOKE_MAX_LIVE_CALLS: int = 4

# Hard cap on local Qwen3 calls for the smoke comparison (default
# runs everything offline against the frozen baseline).
SMOKE_MAX_LOCAL_CALLS: int = 2