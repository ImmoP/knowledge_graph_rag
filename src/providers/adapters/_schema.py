"""JSON-Schema validation for :meth:`LLMProvider.generate_json` (#19).

Thin wrapper around the standard ``jsonschema`` package. The provider
contract is "validate the parsed model output against the caller-
supplied schema"; #19 and downstream tickets (#23 in particular) need
full JSON-Schema support, so we do not maintain a custom subset. The
``jsonschema`` package itself is the only added third-party schema
dependency; the provider-facing surface here is intentionally small.

Public surface:

* :class:`SchemaValidationError` — wraps every ``jsonschema``
  validation failure (and only those) so the retry flow in the
  Ollama adapter can catch it without leaking ``jsonschema``
  specifics.
* :func:`validate` — validate a parsed value against a schema; raise
  :class:`SchemaValidationError` on failure, return ``None`` on success.

Schemas used by callers should follow JSON-Schema Draft 2020-12 as
supported by ``jsonschema`` 4.x. Behaviour:

* Every standard Draft 2020-12 keyword (``type``, ``required``,
  ``properties``, ``additionalProperties``, ``items``, ``enum``,
  ``minLength`` / ``maxLength`` / ``pattern``, ``minimum`` /
  ``maximum`` / ``exclusiveMinimum`` / ``exclusiveMaximum``,
  ``multipleOf``, ``minItems`` / ``maxItems`` / ``uniqueItems``,
  ``minProperties`` / ``maxProperties``, ``prefixItems``,
  ``allOf`` / ``anyOf`` / ``oneOf`` / ``not``, ``const``, etc.)
  is enforced.
* The ``format`` keyword is not enforced by default (jsonschema
  requires an explicit ``FormatChecker`` for format validation; this
  is a known jsonschema caveat, not a custom choice).
* Unknown keywords are treated as annotations and ignored — this is
  the standard JSON Schema behaviour. Callers should stick to
  recognised Draft 2020-12 keywords.
* Malformed schemas raise :class:`SchemaValidationError` before any
  value validation runs (so a bad schema never produces a silent
  zero-retry pass).
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError as _JSONSchemaValidationError


class SchemaValidationError(Exception):
    """Raised when a parsed model output does not satisfy a schema.

    Used internally by :mod:`src.providers.adapters.ollama` to drive
    the spec §2 retry-once semantics. The wrapped ``jsonschema``
    error is available as :attr:`original` for callers that want to
    inspect ``path`` / ``message`` / ``validator`` directly.
    """

    def __init__(self, message: str, original: _JSONSchemaValidationError | SchemaError) -> None:
        super().__init__(message)
        self.original = original


def validate(value: Any, schema: Any) -> None:
    """Validate ``value`` against ``schema``; raise :class:`SchemaValidationError` on failure.

    Schema errors (the schema itself is malformed) raise
    :class:`SchemaValidationError` too — we do not want a malformed
    schema to look like a successful first attempt followed by a
    silent zero-retry validation pass. The retry path checks both the
    schema error and the value error the same way.
    """
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as e:
        raise SchemaValidationError(f"invalid schema: {e.message}", e) from e
    errors = list(Draft202012Validator(schema).iter_errors(value))
    if errors:
        # First error is the highest-priority; the full list is on the
        # original if the caller wants it.
        first = errors[0]
        raise SchemaValidationError(first.message, first) from first
