"""Provider-SDK boundary guard (#19).

Acceptance gate §4 / spec §9.5: a ``grep`` over ``src/`` for
``import (anthropic|openai|ollama|sentence_transformers)`` must match
only files under ``src/providers/``. No business logic learns that a
particular SDK exists.

We implement the guard in pure Python (no ``subprocess`` / ``grep``)
so the test stays in-process and stable on Windows. The scan covers
the public surface: ``import`` statements plus ``__import__`` builtins.
"""

from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PROVIDERS_ROOT = SRC_ROOT / "providers"

# Modules considered "provider SDKs" for the boundary check. The spec
# (provider-abstractions.md §9.5) lists anthropic / openai / ollama;
# sentence_transformers belongs to the same family (provider SDK) and
# lives in ``src/providers/`` for the same reason.
_SDK_MODULES = {"anthropic", "openai", "ollama", "sentence_transformers"}

# Regex matching the imports the spec §9.5 gate checks.
_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)")
_DUNDER_IMPORT_RE = re.compile(r'__import__\(\s*[\'"]([A-Za-z_][\w.]*)[\'"]')


def _iter_python_files(root: Path):
    for p in root.rglob("*.py"):
        yield p


def _violations():
    out = []
    for py in _iter_python_files(SRC_ROOT):
        rel = py.relative_to(SRC_ROOT)
        # Skip the providers package itself: that's where SDK imports
        # are allowed.
        if rel.parts and rel.parts[0] == "providers":
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in _SDK_MODULES:
                        out.append((rel, node.lineno, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom):
                top = (node.module or "").split(".")[0]
                if top in _SDK_MODULES:
                    out.append((rel, node.lineno, f"from {node.module} import ..."))
    return out


class TestProviderSDKBoundary(unittest.TestCase):
    """No provider-SDK imports outside ``src/providers/``."""

    def test_no_sdk_imports_outside_providers(self):
        violations = _violations()
        if violations:
            formatted = "\n".join(
                f"{rel}:{lineno}: {snippet}" for rel, lineno, snippet in violations
            )
            self.fail(f"Provider-SDK imports outside src/providers/:\n{formatted}")

    def test_providers_package_can_import_sdk(self):
        # Sanity: at least the adapter modules are allowed to import
        # their SDK. We check the file actually exists.
        from src.providers.adapters import ollama as ollama_mod

        self.assertTrue(Path(ollama_mod.__file__).exists())

    def test_no_dunder_import_of_sdk(self):
        # Same check for ``__import__('ollama')`` patterns.
        for py in _iter_python_files(SRC_ROOT):
            rel = py.relative_to(SRC_ROOT)
            if rel.parts and rel.parts[0] == "providers":
                continue
            text = py.read_text(encoding="utf-8")
            for m in _DUNDER_IMPORT_RE.finditer(text):
                top = m.group(1).split(".")[0]
                if top in _SDK_MODULES:
                    self.fail(f"{rel}: __import__ of {top!r}")


class TestSchemaValidatorEdges(unittest.TestCase):
    """The minimal in-house schema validator handles the cases the
    smoke / retry semantics exercise, plus the common ``type`` /
    ``properties`` / ``additionalProperties`` patterns callers will
    rely on."""

    def _validate(self, value, schema):
        from src.providers.adapters._schema import validate

        validate(value, schema)

    def test_additional_properties_false_rejects_extras(self):
        from src.providers.adapters._schema import SchemaValidationError

        schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}, "additionalProperties": False}
        with self.assertRaises(SchemaValidationError):
            self._validate({"a": 1, "extra": "x"}, schema)

    def test_bool_is_not_integer(self):
        from src.providers.adapters._schema import SchemaValidationError

        schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}
        with self.assertRaises(SchemaValidationError):
            self._validate({"a": True}, schema)

    def test_number_accepts_int(self):
        schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "number"}}}
        self._validate({"a": 1}, schema)

    def test_array_items_validated(self):
        from src.providers.adapters._schema import SchemaValidationError

        with self.assertRaises(SchemaValidationError):
            self._validate([1, "two", 3], {"type": "array", "items": {"type": "integer"}})

    def test_enum(self):
        from src.providers.adapters._schema import SchemaValidationError

        with self.assertRaises(SchemaValidationError):
            self._validate("d", {"type": "string", "enum": ["a", "b", "c"]})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()