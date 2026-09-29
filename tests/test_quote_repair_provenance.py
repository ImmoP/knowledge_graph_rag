"""Provenance, source-evidence and assertion-mapping tests for the quote-repair-v1 review package.

These tests verify:
  1. The versioned provenance manifest itself is internally consistent and the SHA-256
     values for tracked artefacts match the bytes on disk.
  2. Each of the three review entries is mapped to the correct assertion_id and chunk
     in the manifest, and each claim points to a verbatim substring of the local original
     LaTeX caption (`Tables/joint.tex` inside `data/raw/latex/2007.00808.tar.gz`).
  3. The local source-evidence inventory matches the manifest's claims about which
     artefacts are present versus missing.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPO_ROOT
    / "docs"
    / "experiments"
    / "quote-repair-v1-review"
    / "provenance"
    / "manifest.v1.json"
)
REVIEW_DIR = REPO_ROOT / "docs" / "experiments" / "quote-repair-v1-review"
SOURCE_TARBALL = REPO_ROOT / "data" / "raw" / "latex" / "2007.00808.tar.gz"
ASSERTION_ID_RE = re.compile(r"^a_[0-9a-f]{16}$")


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _read_original_joint_caption() -> bytes:
    assert SOURCE_TARBALL.is_file(), "Original LaTeX tarball is missing"
    with tarfile.open(SOURCE_TARBALL, mode="r:gz") as tar:
        member = tar.getmember("Tables/joint.tex")
        handle = tar.extractfile(member)
        assert handle is not None
        return handle.read()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return _load_manifest()


@pytest.fixture(scope="module")
def review_entries() -> dict:
    out: dict = {}
    for entry in REVIEW_DIR.glob("review-a_*.md"):
        text = entry.read_text(encoding="utf-8")
        first_line = text.splitlines()[0]
        # Format: `# Review-Protokoll `a_<hex>```
        assert first_line.startswith("# Review-Protokoll `a_"), entry
        tail = first_line.removeprefix("# Review-Protokoll `").rstrip("`").strip()
        assert ASSERTION_ID_RE.fullmatch(tail), entry
        out[tail] = {
            "path": entry,
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    return out


@pytest.fixture(scope="module")
def original_caption_bytes() -> bytes:
    return _read_original_joint_caption()


def test_manifest_is_present_and_valid_json(manifest: dict) -> None:
    assert manifest["manifest_version"] == "1.0"
    assert manifest["experiment"] == "quote-repair-v1"


@pytest.mark.parametrize(
    "rel_path",
    [
        "src/extract/experimental_quote_repair.py",
        "scripts/experiments/compare_quote_repair_v1.py",
        "ontology.yaml",
        "docs/experiments/quote-repair-v1.md",
        "docs/experiments/quote-repair-v1-review/README.md",
    ],
)
def test_manifest_declares_existing_in_repo_artefacts(rel_path: str) -> None:
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"Manifest references missing in-repo artefact {rel_path}"


def test_manifest_internal_sha256_match(manifest: dict) -> None:
    repo_tracked = {
        "src/extract/experimental_quote_repair.py": manifest["implementation"]["repair_module"]["sha256"],
        "scripts/experiments/compare_quote_repair_v1.py": manifest["implementation"]["runner"]["sha256"],
        "ontology.yaml": manifest["inputs"]["ontology"]["sha256"],
        "docs/experiments/quote-repair-v1.md": manifest["review_packet"]["experiment_documentation"]["sha256"],
        "docs/experiments/quote-repair-v1-review/README.md": manifest["review_packet"]["readme"]["sha256"],
    }
    for rel_path, expected in repo_tracked.items():
        actual = _sha256_of(REPO_ROOT / rel_path)
        assert actual == expected, f"{rel_path} hash drift: manifest={expected} disk={actual}"


def test_manifest_review_entries_match_disk_sha256(
    manifest: dict, review_entries: dict
) -> None:
    declared = {
        entry["assertion_id"]: entry["sha256"]
        for entry in manifest["review_packet"]["review_entries"]
    }
    assert set(declared) == set(review_entries), (
        f"manifest vs disk mismatch: {set(declared) ^ set(review_entries)}"
    )
    for aid, info in review_entries.items():
        assert declared[aid] == info["sha256"], f"{aid} hash drift"


def test_assertion_records_match_review_files(
    manifest: dict, review_entries: dict
) -> None:
    by_id = {a["assertion_id"]: a for a in manifest["assertions"]}
    assert set(by_id) == set(review_entries), (
        f"manifest assertions vs review entries: {set(by_id) ^ set(review_entries)}"
    )
    for aid, info in review_entries.items():
        record = by_id[aid]
        # Each review file must point to a tracked assertion
        assert Path(record["review_path"]).name == info["path"].name, (
            f"{aid} review_path={record['review_path']} != {info['path'].name}"
        )
        # Human review label must remain unset
        assert record["human_review_label"] == "NOT ASSIGNED", aid
        # Provenance ties the assertion to the correct document and chunk
        assert record["subject"] == {"kind": "document", "id": "W3038572442"}, aid


def test_review_entries_reference_correct_object_endpoints(
    manifest: dict,
) -> None:
    expected_objects = {
        "a_44999b3871c3f58f": ("metric", "Answer Coverage at Top-20/100", 53, 82),
        "a_0a775699fcc8a5ce": ("dataset", "Natural Questions (NQ)", 97, 119),
        "a_2814c002a3ebd912": ("dataset", "Trivial QA (TQA)", 124, 140),
    }
    by_id = {a["assertion_id"]: a for a in manifest["assertions"]}
    for aid, (etype, surface, cs, ce) in expected_objects.items():
        obj = by_id[aid]["object"]
        assert obj["entity_type"] == etype
        assert obj["surface_form"] == surface
        assert obj["char_start"] == cs
        assert obj["char_end"] == ce


def test_review_entries_use_shared_evidence_span(manifest: dict) -> None:
    by_id = {a["assertion_id"]: a for a in manifest["assertions"]}
    starts = {a["char_start_in_chunk"] for a in by_id.values()}
    ends = {a["char_end_in_chunk"] for a in by_id.values()}
    assert starts == {34}, "all three assertions must use chunk-relative start 34"
    assert ends == {190}, "all three assertions must use chunk-relative end 190"


def test_original_caption_contains_evidence_substring(
    original_caption_bytes: bytes,
) -> None:
    text = original_caption_bytes.decode("utf-8")
    expected_substring = (
        "Retrieval results (Answer Coverage at Top-20/100) on \n"
        "         Natural Questions (NQ) and Trivial QA (TQA) in the setting from "
        "~\\citet{karpukhin2020dense}."
    )
    assert expected_substring in text, (
        "Tables/joint.tex no longer contains the canonical caption substring"
    )


def test_evidence_quote_is_a_substring_of_chunk_text() -> None:
    chunk_record = json.loads(
        (REPO_ROOT.parent / "v2-multichunk-01" / "three_chunks.jsonl")
        .read_text()
        .splitlines()[0]
    )
    chunk_text = chunk_record["text"]
    evidence = (
        "Retrieval results (Answer Coverage at Top-20/100) on \n"
        "         Natural Questions (NQ) and Trivial QA (TQA) in the setting from "
        "~[[CITE:karpukhin2020dense]]."
    )
    assert chunk_text[34:190] == evidence, (
        "chunk text does not contain the expected restored evidence span"
    )


def test_manifest_lists_local_source_files(manifest: dict) -> None:
    files = manifest["local_original_source"]["files_of_interest"]
    with tarfile.open(SOURCE_TARBALL, mode="r:gz") as tar:
        members = {m.name: tar.extractfile(m) for m in tar.getmembers() if m.isfile()}
        for info in files:
            handle = members[info["relative_path"]]
            assert handle is not None, info["relative_path"]
            digest = hashlib.sha256(handle.read()).hexdigest()
            assert digest == info["sha256"], info["relative_path"]


def test_manifest_documents_missing_artifacts(manifest: dict) -> None:
    missing_ids = {entry["artifact"] for entry in manifest["missing_or_incomplete_artifacts"]}
    assert "Gold-set / ann-B human annotation for any of the three review entries" in missing_ids


def test_manifest_does_not_assign_review_labels(manifest: dict) -> None:
    for a in manifest["assertions"]:
        assert a["human_review_label"] == "NOT ASSIGNED"
        assert a["baseline_status"] == "ZERO_OCCURRENCES reject"
        assert "post_repair_status" in a


def test_cache_keys_match_stored_filenames(manifest: dict) -> None:
    cache_root = (REPO_ROOT.parent / "v2-pilot-03" / "cache" / "llm").resolve()
    for entry in manifest["cache_records"]["entries"]:
        rel = entry["stored_relative_path"]
        expected_key = entry["reproduced_cache_key"]
        path = cache_root / Path(rel).name
        assert path.is_file(), f"Missing cache file {path}"
        # The stored filename must equal the reproduced cache key.
        assert path.stem == expected_key, (
            f"{rel} filename {path.stem} does not match reproduced key {expected_key}"
        )