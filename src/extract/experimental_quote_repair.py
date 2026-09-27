"""Experimental, opt-in reconstruction of whitespace-only quote changes.

Not part of the canonical extraction-contract v2.
No model calls, fuzzy matching, or semantic correction.
"""

import re
from dataclasses import dataclass

REPAIR_VERSION = "experimental-whitespace-v1"
_LAYOUT = r"[ \t\r\n]+"
_PARAGRAPH = r"(?:\r?\n[ \t]*){2,}"


@dataclass(frozen=True)
class RepairResult:
    status: str
    quote: str | None = None
    char_start: int | None = None
    char_end: int | None = None


def repair_quote(source: str, quote: str) -> RepairResult:
    """Restore only a unique, paragraph-safe original text span."""

    if not isinstance(source, str) or not isinstance(quote, str):
        return RepairResult("invalid")

    if not quote.strip():
        return RepairResult("invalid")

    # Exact quotes belong to the existing v2 aligner.
    # Do not silently choose an occurrence if repeated.
    if quote in source:
        return RepairResult("exact", quote)

    if re.search(_PARAGRAPH, quote):
        return RepairResult("paragraph_boundary")

    parts = re.split(f"({_LAYOUT})", quote)
    pattern = "".join(
        _LAYOUT if re.fullmatch(_LAYOUT, part)
        else re.escape(part)
        for part in parts if part
    )

    matches = list(re.finditer(pattern, source))

    if not matches:
        return RepairResult("not_found")

    if len(matches) != 1:
        return RepairResult("ambiguous")

    match = matches[0]
    original = match.group(0)

    if re.search(_PARAGRAPH, original):
        return RepairResult("paragraph_boundary")

    return RepairResult(
        "repaired",
        original,
        match.start(),
        match.end(),
    )
