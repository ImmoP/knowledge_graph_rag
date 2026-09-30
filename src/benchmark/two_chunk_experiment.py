"""Frozen selection contract for the controlled Manifest two-chunk experiment."""

from __future__ import annotations

from pathlib import Path

from src.benchmark.dataset.manifest import CHUNKS_PATH, load_frozen_chunks
from src.benchmark.dataset.smoke_chunks import SMOKE_CHUNK_IDS

EXPERIMENT_CHUNK_IDS = ("c_ca129edd34052284", "c_8a84177f479aea1f")
DPR_CHUNK_ID = "c_8a84177f479aea1f"
DATASET_SHA256 = "38182c6cdbb9f86abc6ce20ece7a3d9e1cb69ccac2576e2d6274e727c2f84985"


def load_recording_selection(
    dataset: Path = CHUNKS_PATH, chunk_ids: tuple[str, ...] | None = None
) -> tuple[list[dict], dict]:
    """Verify the frozen dataset and select smoke defaults or both problem chunks.

    Explicit selection permits either order, but no duplicates, subsets or other
    IDs. Alternate paths must contain the same frozen dataset bytes, not a new
    dataset claiming the same IDs.
    """
    if chunk_ids is None:
        requested = tuple(SMOKE_CHUNK_IDS)
        source = "smoke_default"
    else:
        requested = tuple(chunk_ids)
        if len(requested) != 2 or set(requested) != set(EXPERIMENT_CHUNK_IDS):
            raise ValueError("explicit selection must contain each experiment chunk exactly once")
        source = "explicit_chunk_id_flag"
    chunks, manifest = load_frozen_chunks(chunks_path=Path(dataset))
    if manifest.chunks_sha256 != DATASET_SHA256:
        raise ValueError("experiment dataset SHA-256 differs from the pinned digest")
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    if any(chunk_id not in by_id for chunk_id in requested):
        raise ValueError("frozen dataset is missing a selected chunk")
    return [by_id[chunk_id] for chunk_id in requested], {
        "source": source,
        "dataset_sha256": DATASET_SHA256,
        "chunk_ids": list(requested),
    }


def validate_selection(chunks: list[dict], selection: dict) -> None:
    """Bind manifest selection, original dataset identity and exact frozen rows."""
    if not isinstance(selection, dict) or set(selection) != {"source", "dataset_sha256", "chunk_ids"}:
        raise ValueError("invalid recording selection")
    if selection["dataset_sha256"] != DATASET_SHA256:
        raise ValueError("recording dataset SHA-256 differs from the pinned digest")
    ids = selection["chunk_ids"]
    if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
        raise ValueError("recording selection IDs must be a list of strings")
    if selection["source"] == "smoke_default":
        requested = None
    elif selection["source"] == "explicit_chunk_id_flag":
        requested = tuple(ids)
    else:
        raise ValueError("unknown recording selection source")
    expected_chunks, expected_selection = load_recording_selection(chunk_ids=requested)
    if selection != expected_selection or chunks != expected_chunks:
        raise ValueError("recorded chunks do not match the frozen selection and order")


def verify_experiment_recording(manifest: dict) -> None:
    """Reject smoke, unrelated or unverified recordings before experiment reports."""
    selection = manifest.get("selection")
    validate_selection(manifest["_chunks"], selection)
    if selection["source"] != "explicit_chunk_id_flag":
        raise ValueError("comparison requires the two problem chunks, not the smoke default")


def recording_exit_code(manifest: dict) -> int:
    """Classify retained recordings without conflating partial results with success."""
    if (manifest.get("terminal") or {}).get("status") == 429:
        return 5
    if manifest["status"] == "budget_aborted":
        return 3
    return 0 if manifest["status"] == "complete" else 4
