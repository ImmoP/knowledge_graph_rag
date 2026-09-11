"""#19 smoke entry point.

Run with ``python -m tests.providers.smoke`` after configuring the
provider env vars. Exercises the configured real adapters against
the three contract §5.1 acceptance bullets:

  * ``generate("reply with exactly: pong", "ping")`` returns text
    containing ``pong``.
  * ``generate_json(...)`` against a tiny required-key schema returns a
    dict with the required key.
  * ``embed(["hello", "world"])`` returns two non-zero vectors of
    dimension 384.

If the configured LLM or embedding dependencies are unavailable in the
local environment (Ollama not running, sentence-transformers not
installed, model weights not downloaded), the smoke prints
``SMOKE: NOT RUN (<reason>)`` and exits non-zero. Per the #19 ticket:
do NOT fake a PASS — report NOT RUN with the exact reason while all
mocked/unit acceptance tests remain runnable.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.providers import make_embedding, make_llm, validate_provider_config  # noqa: E402
from src.providers import registry as registry_module  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]


def _abort(reason: str, code: int = 1) -> None:
    print(f"SMOKE: NOT RUN ({reason})")
    sys.exit(code)


def main() -> int:
    env = None  # make_llm / make_embedding read os.environ by default
    # Reuse the validator to surface config errors as NOT RUN rather
    # than a Python exception.
    report = validate_provider_config(
        env,
        llm_registry_keys=list(registry_module.LLM_REGISTRY),
        embedding_registry_keys=list(registry_module.EMBEDDING_REGISTRY),
        llm_adapter_for_provider=lambda k: registry_module.LLM_REGISTRY[k],
        embedding_adapter_for_provider=lambda k: registry_module.EMBEDDING_REGISTRY[k],
    )
    if not report.ok:
        reasons = "; ".join(v.message for v in report.violations)
        _abort(f"invalid provider config: {reasons}")

    # --- LLM ------------------------------------------------------
    try:
        llm = make_llm(env, with_budget=False)
    except Exception as e:
        _abort(f"LLM adapter construction failed: {e}")

    try:
        text, _ = llm.generate("reply with exactly: pong", "ping")
    except Exception as e:
        _abort(f"LLM generate failed: {type(e).__name__}: {e}")
    if "pong" not in text:
        print(f"SMOKE: FAIL — generate did not contain 'pong' (got {text!r})")
        return 2
    print(f"SMOKE: generate OK — {text!r}")

    schema = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "integer"}}}
    try:
        parsed, _ = llm.generate_json(
            "reply with valid JSON matching the schema", '{"answer": <int>}', schema
        )
    except Exception as e:
        _abort(f"LLM generate_json failed: {type(e).__name__}: {e}")
    if not isinstance(parsed, dict) or "answer" not in parsed:
        print(f"SMOKE: FAIL — generate_json missing 'answer' key (got {parsed!r})")
        return 2
    print(f"SMOKE: generate_json OK — {parsed!r}")

    # --- Embedding ------------------------------------------------
    try:
        embed = make_embedding(env)
    except Exception as e:
        _abort(f"embedding adapter construction failed: {e}")
    try:
        vectors, _ = embed.embed(["hello", "world"])
    except Exception as e:
        _abort(f"embedding failed: {type(e).__name__}: {e}")
    if len(vectors) != 2:
        print(f"SMOKE: FAIL — expected 2 vectors, got {len(vectors)}")
        return 2
    for v in vectors:
        if len(v) != embed.dim:
            print(f"SMOKE: FAIL — vector length {len(v)} != dim {embed.dim}")
            return 2
        if not any(x != 0.0 for x in v):
            print(f"SMOKE: FAIL — zero vector: {v!r}")
            return 2
    print(f"SMOKE: embed OK — dim={embed.dim} count={len(vectors)}")
    print("SMOKE: PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        # Last-ditch: print NOT RUN rather than traceback.
        traceback.print_exc()
        _abort("unhandled exception during smoke")