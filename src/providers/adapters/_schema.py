"""Minimal JSON-Schema subset validator (#19).

The spec describes schemas as "JSON-Schema-flavoured dict" (provider-
abstractions.md §2). #19 does not add a third-party JSON Schema
dependency: instead it implements the subset of the spec that the
smoke test exercises (``type: object``, ``required``) plus the
``type`` constraints callers (e.g. #23's extraction pipeline) will
rely on most. The validator is deliberately not a full JSON-Schema
implementation; the docstring below documents the supported surface.

Supported keywords:

* ``type`` — one of ``"object"``, ``"array"``, ``"string"``,
  ``"number"``, ``"integer"``, ``"boolean"``, ``"null"``. Applied at
  the root and inside ``properties`` / ``items``.
* ``required`` — list of property names that must be present in an
  object value (only meaningful when combined with ``type: object``).
* ``properties`` — per-key schema applied when the key is present.
  Extra keys are allowed by default (JSON Schema's default).
* ``items`` — schema applied to every element of an array.
* ``enum`` — list of allowed scalar values.

Not supported (will raise :class:`SchemaValidationError` on encounter
so the caller knows immediately rather than silently over-permitting):

* ``$ref``, ``allOf``, ``anyOf``, ``oneOf``, ``not`` — composition.
* ``pattern``, ``format``, ``minLength``, ``maxLength`` — string
  constraints.
* ``minimum``, ``maximum``, ``exclusiveMinimum`` / ``exclusiveMaximum``
  — numeric bounds.
* ``additionalProperties`` with a schema (boolean ``false`` is treated
  as "no extras allowed"; we keep that one affordance because it is
  cheap and matches a common contract).

If a future ticket (#23 likely) needs the unsupported surface, the
ticket should add ``jsonschema`` as a dependency and replace this
module. For #19 the goal is a zero-new-dependency validator that
catches the same failures the smoke test and the retry semantics
exercise.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


_SCALAR_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


class SchemaValidationError(Exception):
    """Raised when a value does not satisfy a schema.

    Used internally by :mod:`src.providers.adapters.ollama` to drive
    the spec §2 retry-once semantics. ``path`` is a JSON-pointer-ish
    list identifying the failing sub-value, ``schema_path`` does the
    same for the schema; ``message`` is human-readable.
    """

    def __init__(self, path: list, schema_path: list, message: str) -> None:
        super().__init__(f"{'/'.join(map(str, schema_path))}: {message} (at {path})")
        self.path = path
        self.schema_path = schema_path
        self.message = message


def _check_type(value: Any, expected: str, path: list, schema_path: list) -> None:
    """Check that ``value`` matches the JSON-Schema ``type`` ``expected``."""
    # ``bool`` is a subclass of ``int`` in Python; JSON Schema keeps them
    # distinct, so we explicitly exclude booleans from integer/number.
    if expected == "array":
        if not isinstance(value, list):
            raise SchemaValidationError(path, schema_path, f"expected array, got {type(value).__name__}")
        return
    if expected == "object":
        if not isinstance(value, Mapping):
            raise SchemaValidationError(path, schema_path, f"expected object, got {type(value).__name__}")
        return
    py_type = _SCALAR_TYPES.get(expected)
    if py_type is None:
        raise SchemaValidationError(
            path, schema_path, f"unsupported type {expected!r}"
        )
    if expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaValidationError(
                path, schema_path, f"expected integer, got {type(value).__name__}"
            )
        return
    if expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaValidationError(
                path, schema_path, f"expected number, got {type(value).__name__}"
            )
        return
    if not isinstance(value, py_type):
        raise SchemaValidationError(
            path, schema_path, f"expected {expected}, got {type(value).__name__}"
        )


def _check_required(
    value: Mapping, required: Sequence, path: list, schema_path: list
) -> None:
    for key in required:
        if key not in value:
            raise SchemaValidationError(
                path + [key],
                schema_path + ["required"],
                f"missing required key {key!r}",
            )


def _check_enum(value: Any, allowed: Sequence, path: list, schema_path: list) -> None:
    if value not in allowed:
        raise SchemaValidationError(
            path,
            schema_path + ["enum"],
            f"value {value!r} not in enum {list(allowed)!r}",
        )


def _validate(value: Any, schema: Any, path: list, schema_path: list) -> None:
    """Recursively validate ``value`` against ``schema``."""
    if not isinstance(schema, Mapping):
        raise SchemaValidationError(
            path, schema_path, "schema must be a mapping"
        )
    expected_type = schema.get("type")
    if expected_type is not None:
        if isinstance(expected_type, str):
            _check_type(value, expected_type, path, schema_path + ["type"])
        elif isinstance(expected_type, Sequence) and not isinstance(expected_type, (str, bytes)):
            # JSON Schema allows ``type`` to be a list of types.
            for t in expected_type:
                try:
                    _check_type(value, t, path, schema_path + ["type"])
                    break
                except SchemaValidationError:
                    continue
            else:
                raise SchemaValidationError(
                    path,
                    schema_path + ["type"],
                    f"value did not match any of {list(expected_type)!r}",
                )
        else:
            raise SchemaValidationError(
                path, schema_path, f"unsupported type spec {expected_type!r}"
            )
    if "enum" in schema:
        _check_enum(value, schema["enum"], path, schema_path)
    if expected_type == "object" or "properties" in schema or "required" in schema:
        if not isinstance(value, Mapping):
            # ``properties`` / ``required`` only meaningful for objects.
            return
        if "required" in schema:
            req = schema["required"]
            if isinstance(req, Sequence) and not isinstance(req, (str, bytes)):
                _check_required(value, req, path, schema_path + ["required"])
            else:
                raise SchemaValidationError(
                    path,
                    schema_path + ["required"],
                    "required must be a list",
                )
        properties = schema.get("properties")
        additional = schema.get("additionalProperties")
        if isinstance(properties, Mapping):
            for key, sub_schema in properties.items():
                if key in value:
                    _validate(value[key], sub_schema, path + [key], schema_path + ["properties", str(key)])
        if additional is False and isinstance(properties, Mapping):
            extra = set(value.keys()) - set(properties.keys())
            if extra:
                raise SchemaValidationError(
                    path,
                    schema_path + ["additionalProperties"],
                    f"unexpected keys {sorted(extra)!r}",
                )
    if expected_type == "array" or "items" in schema:
        if not isinstance(value, list):
            return
        items = schema.get("items")
        if isinstance(items, Mapping):
            for idx, item in enumerate(value):
                _validate(item, items, path + [idx], schema_path + ["items"])


def validate(value: Any, schema: Any) -> None:
    """Validate ``value`` against ``schema``; raise :class:`SchemaValidationError` on failure."""
    _validate(value, schema, [], [])