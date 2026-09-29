"""Frozen dataset manifest accessor.

The committed dataset files live under ``src/benchmark/dataset/``:

* ``benchmark_chunks.jsonl`` — the selected chunks.
* ``benchmark_chunks.sha256`` — the SHA-256 of the JSONL bytes.
* ``benchmark_dataset_manifest.json`` — the selection manifest.

Every benchmark run loads these paths through this module so the
loaders cannot drift between runs. Any drift raises immediately.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent

CHUNKS_FILENAME = "benchmark_chunks.jsonl"
SHA256_FILENAME = "benchmark_chunks.sha256"
MANIFEST_FILENAME = "benchmark_dataset_manifest.json"

CHUNKS_PATH = DATASET_DIR / CHUNKS_FILENAME
SHA256_PATH = DATASET_DIR / SHA256_FILENAME
MANIFEST_PATH = DATASET_DIR / MANIFEST_FILENAME


@dataclass(frozen=True)
class DatasetManifest:
    algorithm_version: str
    sampler_seed: int
    config: dict
    selected_chunk_ids: tuple[str, ...]
    documents_selected: tuple[str, ...]
    counts_by_tag: dict[str, int]
    counts_by_length_tercile: dict[str, int]
    chunks_path: str
    chunks_sha256: str

    @classmethod
    def load(cls, *, manifest_path: Path = MANIFEST_PATH) -> "DatasetManifest":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls(
            algorithm_version=payload["algorithm_version"],
            sampler_seed=int(payload["sampler_seed"]),
            config=dict(payload["config"]),
            selected_chunk_ids=tuple(payload["selected_chunk_ids"]),
            documents_selected=tuple(payload["documents_selected"]),
            counts_by_tag=dict(payload["counts_by_tag"]),
            counts_by_length_tercile=dict(payload["counts_by_length_tercile"]),
            chunks_path=str(payload["chunks_path"]),
            chunks_sha256=str(payload["chunks_sha256"]),
        )


def verify_dataset_files(
    *,
    chunks_path: Path = CHUNKS_PATH,
    sha256_path: Path = SHA256_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> DatasetManifest:
    """Verify the committed dataset files match their declared SHA.

    Raises ``FileNotFoundError`` if any file is missing and
    ``RuntimeError`` if the SHA-256 of the JSONL body does not match
    the value the manifest advertises.
    """
    for label, path in (
        ("chunks", chunks_path),
        ("sha256", sha256_path),
        ("manifest", manifest_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"dataset {label} file missing: {path}")

    declared_sha = sha256_path.read_text(encoding="utf-8").strip().split()[0]
    actual_sha = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    if actual_sha != declared_sha:
        raise RuntimeError(
            f"dataset sha256 drift: declared {declared_sha} != actual {actual_sha}"
        )

    manifest = DatasetManifest.load(manifest_path=manifest_path)
    if manifest.chunks_sha256 != declared_sha:
        raise RuntimeError(
            "manifest.chunks_sha256 disagrees with the on-disk sha256 file"
        )
    return manifest


def load_frozen_chunks(
    *,
    chunks_path: Path = CHUNKS_PATH,
    sha256_path: Path = SHA256_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[list[dict], DatasetManifest]:
    """Load and verify the committed chunk set. Returns
    ``(chunks, manifest)``."""
    manifest = verify_dataset_files(
        chunks_path=chunks_path,
        sha256_path=sha256_path,
        manifest_path=manifest_path,
    )
    out: list[dict] = []
    seen: set[str] = set()
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = rec["chunk_id"]
        if cid not in manifest.selected_chunk_ids:
            raise RuntimeError(f"chunk {cid!r} not declared in manifest")
        if cid in seen:
            raise RuntimeError(f"chunk {cid!r} appears twice in {chunks_path}")
        seen.add(cid)
        out.append(rec)
    if seen != set(manifest.selected_chunk_ids):
        raise RuntimeError("on-disk chunks do not match manifest set")
    return out, manifest