"""Tests for the optional experimental quote reconstruction."""

import pytest

from src.extract.experimental_quote_repair import (
    REPAIR_VERSION,
    repair_quote,
)


@pytest.mark.parametrize(
    "source,quote,expected,status",
    [
        (
            "DPR\n         evaluates NQ",
            "DPR evaluates NQ",
            "DPR\n         evaluates NQ",
            "repaired",
        ),
        (
            "DPR evaluates NQ",
            "DPR evaluates NQ",
            "DPR evaluates NQ",
            "exact",
        ),
        (
            "DPR\r\n\t evaluates NQ",
            "DPR evaluates NQ",
            "DPR\r\n\t evaluates NQ",
            "repaired",
        ),
        (
            "DPR\n evaluates NQ. DPR\n evaluates NQ.",
            "DPR evaluates NQ",
            None,
            "ambiguous",
        ),
        (
            "DPR evaluates NQ",
            "DPR tests NQ",
            None,
            "not_found",
        ),
        (
            "DPR evaluates NQ",
            "dpr evaluates NQ",
            None,
            "not_found",
        ),
        (
            "DPR evaluates NQ.",
            "DPR evaluates NQ!",
            None,
            "not_found",
        ),
        (
            "DPR\n\nsupports NQ",
            "DPR supports NQ",
            None,
            "paragraph_boundary",
        ),
        (
            "DPR supports NQ",
            "DPR\n\nsupports NQ",
            None,
            "paragraph_boundary",
        ),
        (
            "DPR evaluates NQ",
            "",
            None,
            "invalid",
        ),
    ],
)
def test_quote_reconstruction(source, quote, expected, status):
    result = repair_quote(source, quote)

    assert result.status == status
    assert result.quote == expected

    if status == "repaired":
        assert (
            source[result.char_start:result.char_end]
            == expected
        )


def test_source_is_never_modified():
    source = "DPR\n    evaluates NQ"
    original = source

    repair_quote(source, "DPR evaluates NQ")

    assert source == original


def test_experiment_is_versioned():
    assert REPAIR_VERSION == "experimental-whitespace-v1"
