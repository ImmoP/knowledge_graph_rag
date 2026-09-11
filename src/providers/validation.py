"""Provider configuration validator (#19).

Resolution of ``docs/specs/provider-abstractions.md`` §3.4.

Reads the same env vars as the registries and checks:

  * required provider/model env vars are present;
  * the selected provider is registered;
  * the configured model's model is a non-empty string;
  * the active adapter's required secrets are present (when the
    adapter declares any);
  * optional numeric settings parse cleanly.

Does **not** perform network reachability (spec §3.4 final sentence)
and does **not** fail on unrelated-provider secrets being missing or
present (spec §3.3: "Mismatched keys are not an error at module load").
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class Violation:
    rule: str
    message: str


@dataclass(frozen=True)
class ProviderConfigReport:
    env: dict
    violations: list[Violation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


# Names of env vars this validator inspects. Centralised so the
# registry's make_* helpers and the validator cannot drift.
ENV_LLM_PROVIDER = "LLM_PROVIDER"
ENV_LLM_MODEL = "LLM_MODEL"
ENV_EMBEDDING_PROVIDER = "EMBEDDING_PROVIDER"
ENV_EMBEDDING_MODEL = "EMBEDDING_MODEL"

ENV_LLM_BUDGET_USD = "LLM_BUDGET_USD"
ENV_LLM_BUDGET_TOKENS = "LLM_BUDGET_TOKENS"
ENV_LLM_CACHE_DIR = "LLM_CACHE_DIR"
ENV_LLM_PROVIDER_TIMEOUT_S = "LLM_PROVIDER_TIMEOUT_S"
ENV_EMBEDDING_BATCH_SIZE = "EMBEDDING_BATCH_SIZE"

ALL_OPTIONAL_NUMERIC_ENVS = (
    ENV_LLM_BUDGET_USD,
    ENV_LLM_BUDGET_TOKENS,
    ENV_LLM_PROVIDER_TIMEOUT_S,
    ENV_EMBEDDING_BATCH_SIZE,
)


# Adapter side-channel: each adapter class may expose ``REQUIRED_SECRETS``
# (a tuple of env var names the adapter needs at invocation time) and
# ``provider_key`` (the registry key it answers to). Adapters declare
# these directly; the validator discovers them via the registry.
AdapterSecretProbe = Callable[[type], tuple[str, ...]]


def _adapter_secrets(adapter_cls) -> tuple[str, ...]:
    """Return the env vars the adapter declares as required.

    Adapters opt in by setting ``REQUIRED_SECRETS = (...)`` on the
    class. Adapters that declare no required secrets (e.g. Ollama,
    which has no API key) implicitly require nothing.
    """
    declared = getattr(adapter_cls, "REQUIRED_SECRETS", None)
    if not declared:
        return ()
    if not isinstance(declared, tuple) or not all(isinstance(s, str) for s in declared):
        # Misdeclared; we treat it as "no secrets" rather than guessing,
        # so a typo in REQUIRED_SECRETS doesn't silently allow the
        # adapter to run with missing credentials.
        return ()
    return tuple(declared)


def _check_required(
    env: dict,
    *,
    llm_registry_keys: Iterable[str],
    embedding_registry_keys: Iterable[str],
) -> tuple[list[Violation], list[str]]:
    """Validate the four required provider env vars (spec §3.1)."""
    violations: list[Violation] = []
    warnings: list[str] = []

    llm_provider = env.get(ENV_LLM_PROVIDER, "")
    if not llm_provider:
        violations.append(
            Violation(
                "llm_provider",
                f"{ENV_LLM_PROVIDER} is required",
            )
        )
    elif llm_provider not in llm_registry_keys:
        violations.append(
            Violation(
                "llm_provider",
                f"{ENV_LLM_PROVIDER}={llm_provider!r} is not a registered adapter "
                f"(known: {sorted(llm_registry_keys)})",
            )
        )

    embedding_provider = env.get(ENV_EMBEDDING_PROVIDER, "")
    if not embedding_provider:
        violations.append(
            Violation(
                "embedding_provider",
                f"{ENV_EMBEDDING_PROVIDER} is required",
            )
        )
    elif embedding_provider not in embedding_registry_keys:
        violations.append(
            Violation(
                "embedding_provider",
                f"{ENV_EMBEDDING_PROVIDER}={embedding_provider!r} is not a registered adapter "
                f"(known: {sorted(embedding_registry_keys)})",
            )
        )

    llm_model = env.get(ENV_LLM_MODEL, "")
    if not llm_model or not llm_model.strip():
        violations.append(
            Violation(
                "llm_model",
                f"{ENV_LLM_MODEL} must be a non-empty string",
            )
        )

    embedding_model = env.get(ENV_EMBEDDING_MODEL, "")
    if not embedding_model or not embedding_model.strip():
        violations.append(
            Violation(
                "embedding_model",
                f"{ENV_EMBEDDING_MODEL} must be a non-empty string",
            )
        )

    return violations, warnings


def _check_optional_numeric(env: dict) -> list[Violation]:
    """Optional float/int settings must parse cleanly when present."""
    violations: list[Violation] = []
    for name in ALL_OPTIONAL_NUMERIC_ENVS:
        raw = env.get(name)
        if raw is None or raw == "":
            continue
        try:
            value = float(raw)
        except ValueError:
            violations.append(
                Violation(
                    "optional_numeric",
                    f"{name}={raw!r} is not a valid number",
                )
            )
            continue
        # Token/batch-size fields must be integer-valued; reject 3.5.
        if name in (ENV_LLM_BUDGET_TOKENS, ENV_EMBEDDING_BATCH_SIZE) and not value.is_integer():
            violations.append(
                Violation(
                    "optional_numeric",
                    f"{name}={raw!r} must be an integer",
                )
            )
        # USD cap and timeout reject negatives.
        if value < 0:
            violations.append(
                Violation(
                    "optional_numeric",
                    f"{name}={raw!r} must be non-negative",
                )
            )
    return violations


def _check_secrets(
    env: dict,
    *,
    llm_provider: str,
    llm_adapter_for_provider: Callable[[str], type],
    embedding_provider: str,
    embedding_adapter_for_provider: Callable[[str], type],
) -> list[Violation]:
    """Ensure the active adapter's declared secrets are present."""
    violations: list[Violation] = []
    if llm_provider:
        try:
            adapter_cls = llm_adapter_for_provider(llm_provider)
        except KeyError:
            adapter_cls = None
        if adapter_cls is not None:
            for secret in _adapter_secrets(adapter_cls):
                if not env.get(secret):
                    violations.append(
                        Violation(
                            "missing_secret",
                                f"{secret} is required by LLM adapter {llm_provider!r}",
                            )
                    )
    if embedding_provider:
        try:
            adapter_cls = embedding_adapter_for_provider(embedding_provider)
        except KeyError:
            adapter_cls = None
        if adapter_cls is not None:
            for secret in _adapter_secrets(adapter_cls):
                if not env.get(secret):
                    violations.append(
                        Violation(
                            "missing_secret",
                                f"{secret} is required by embedding adapter {embedding_provider!r}",
                        )
                    )
    return violations


def validate_provider_config(
    env: dict | None = None,
    *,
    llm_registry_keys: Iterable[str],
    embedding_registry_keys: Iterable[str],
    llm_adapter_for_provider: Callable[[str], type],
    embedding_adapter_for_provider: Callable[[str], type],
) -> ProviderConfigReport:
    """Run the spec §3.4 validation matrix against ``env`` (default: ``os.environ``).

    ``env`` is injectable so tests can drive the validator without
    mutating process state. The function does not reach out to
    the network; it is also safe to call before any model is loaded.
    """
    src = env if env is not None else dict(os.environ)
    violations: list[Violation] = []
    warnings: list[str] = []

    required_violations, required_warnings = _check_required(
        src,
        llm_registry_keys=llm_registry_keys,
        embedding_registry_keys=embedding_registry_keys,
    )
    violations.extend(required_violations)
    warnings.extend(required_warnings)

    violations.extend(_check_optional_numeric(src))
    violations.extend(
        _check_secrets(
            src,
            llm_provider=src.get(ENV_LLM_PROVIDER, ""),
            llm_adapter_for_provider=llm_adapter_for_provider,
            embedding_provider=src.get(ENV_EMBEDDING_PROVIDER, ""),
            embedding_adapter_for_provider=embedding_adapter_for_provider,
        )
    )

    return ProviderConfigReport(env=dict(src), violations=violations, warnings=warnings)


def default_cache_dir() -> Path:
    """Resolve ``LLM_CACHE_DIR`` against the repo root.

    Spec §3.2 default is ``data/derived/llm_cache/``. We anchor against
    the repo root (``src/providers`` → ``<repo>``) so the cache lives
    under the gitignored derived tree whether the entry point is run
    from the repo root or a subdirectory.
    """
    raw = os.environ.get(ENV_LLM_CACHE_DIR)
    if raw:
        return Path(raw)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "derived" / "llm_cache"