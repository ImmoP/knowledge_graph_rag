"""Immutable Manifest Auto recordings and fail-closed offline replay.

This experimental module deliberately drives :func:`run_extraction_v2` without
changing its production validation behavior.  A recording contains every input
needed by that function, byte-for-byte canonical outputs, and the HTTP attempt
stream captured by :mod:`src.benchmark.manifest_recording`.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from src.benchmark.manifest_recording import (
    MissingRawResponse,
    RecordingClient,
    RecordingError,
    ReplayClient,
    assert_safe_bytes,
    external_directory,
    private_write,
    sha256_bytes,
)
from src.extract import prompts as prompts_module
from src.extract import schema as schema_module
from src.extract.pipeline_v2 import RunResultV2, run_extraction_v2
from src.providers.adapters.manifest_openai import ManifestAutoBenchmarkLLM


RECORDING_FORMAT = "manifest-auto-recording-v1"
RECORDING_FILENAME = "recording.json"
EXTRACTIONS_FILENAME = "extractions.jsonl"
REJECTS_FILENAME = "extractions.rejects.jsonl"
REPLAY_REPORT_FILENAME = "replay_report.json"
EMPTY_REVIEW_FILENAME = "empty_chunk_review.json"
_MAX_RECORDING_CHUNKS = 2

_INPUT_FILES = {
    "chunks": "inputs/chunks.jsonl",
    "ontology": "inputs/ontology.yaml",
    "entity_prompt": "inputs/prompts/extract_entities_v2.txt",
    "relation_prompt": "inputs/prompts/extract_relations_v2.txt",
    "entity_schema": "inputs/schemas/entity.v2.json",
    "relation_schema": "inputs/schemas/relation.v2.json",
}
_IMPLEMENTATION_PATHS = (
    "requirements.txt",
    "src/benchmark/empty_chunk_review.py",
    "src/benchmark/manifest_comparison.py",
    "src/benchmark/manifest_recording.py",
    "src/benchmark/manifest_replay.py",
    "src/extract/align.py",
    "src/extract/experimental_quote_repair.py",
    "src/extract/ids.py",
    "src/extract/pipeline_v2.py",
    "src/extract/prompts.py",
    "src/extract/retry.py",
    "src/extract/schema.py",
    "src/extract/validation.py",
    "src/ontology/validate.py",
    "src/providers/adapters/_schema.py",
    "src/providers/adapters/manifest_openai.py",
    "src/providers/budget.py",
    "src/providers/cache.py",
    "src/providers/protocol.py",
)
_REPO_ROOT = Path(__file__).resolve().parents[2]


class RecordingManifestError(RecordingError):
    """The immutable recording manifest or one of its snapshots is invalid."""


class ReplayMismatch(RecordingError):
    """Offline execution did not reproduce the recorded canonical artifacts."""

class _ReplayedTerminal(RecordingError):
    """A fully consumed replay reproduced a non-V2 terminal exception."""

    def __init__(self, original: BaseException, client: ReplayClient, provider: Any) -> None:
        super().__init__("recorded terminal outcome reproduced")
        self.original = original
        self.client = client
        self.provider = provider


@contextmanager
def _private_umask():
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _jsonl_bytes(records: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for record in records
    )


def _file_entry(relative_path: str, content: bytes) -> dict[str, Any]:
    return {
        "path": relative_path,
        "sha256": sha256_bytes(content),
        "size": len(content),
    }


def _read_jsonl_bytes(content: bytes, *, label: str) -> list[dict]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecordingManifestError(f"invalid {label} snapshot") from exc
    rows: list[dict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecordingManifestError(f"invalid {label} snapshot") from exc
        if not isinstance(row, dict):
            raise RecordingManifestError(f"invalid {label}: non-object row")
        rows.append(row)
    return rows


def _ensure_fresh_external_directory(path: Path) -> Path:
    candidate = Path(path)
    if candidate.exists() or candidate.is_symlink():
        external_directory(candidate, create=False)
        raise RecordingManifestError("output directory already exists")
    return external_directory(candidate, create=True)


def _implementation_identity() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for relative in _IMPLEMENTATION_PATHS:
        path = _REPO_ROOT / relative
        raw = path.read_bytes()
        files[relative] = {"sha256": sha256_bytes(raw), "size": len(raw)}
    dependencies = {
        distribution: importlib.metadata.version(distribution)
        for distribution in (
            "attrs", "httpcore", "httpx", "jsonschema",
            "jsonschema-specifications", "PyYAML", "referencing", "rpds-py",
        )
    }
    return {
        "files": files,
        "dependencies": dependencies,
        "python": {
            "implementation": platform.python_implementation(),
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
        },
        "platform": platform.platform(),
        "baseline_commit": "790e472",
        "recording_format": RECORDING_FORMAT,
    }


def _verify_implementation_identity(identity: Any) -> None:
    if not isinstance(identity, dict) or identity.get("recording_format") != RECORDING_FORMAT:
        raise RecordingManifestError("unsupported recording implementation identity")
    python_id = identity.get("python")
    if not isinstance(python_id, dict):
        raise RecordingManifestError("missing Python runtime identity")
    current_python = {
        "implementation": platform.python_implementation(),
        "major": sys.version_info.major,
        "minor": sys.version_info.minor,
        "micro": sys.version_info.micro,
    }
    if python_id != current_python:
        raise RecordingManifestError("Python runtime identity drift")
    current_dependencies = {
        distribution: importlib.metadata.version(distribution)
        for distribution in (
            "attrs", "httpcore", "httpx", "jsonschema",
            "jsonschema-specifications", "PyYAML", "referencing", "rpds-py",
        )
    }
    if identity.get("dependencies") != current_dependencies:
        raise RecordingManifestError("dependency identity drift")
    expected_files = identity.get("files")
    if not isinstance(expected_files, dict) or set(expected_files) != set(_IMPLEMENTATION_PATHS):
        raise RecordingManifestError("implementation file identity is incomplete")
    for relative in _IMPLEMENTATION_PATHS:
        entry = expected_files[relative]
        if not isinstance(entry, dict):
            raise RecordingManifestError("invalid implementation file identity")
        raw = (_REPO_ROOT / relative).read_bytes()
        if entry != {"sha256": sha256_bytes(raw), "size": len(raw)}:
            raise RecordingManifestError(f"implementation drift: {relative}")


def _safe_terminal(exc: BaseException) -> dict[str, Any]:
    value: dict[str, Any] = {"type": type(exc).__name__}
    # These fields are numeric/boolean adapter classifications, never remote text.
    status = getattr(exc, "status", None)
    transient = getattr(exc, "transient", None)
    if type(status) is int:
        value["status"] = status
    if type(transient) is bool:
        value["transient"] = transient
    return value


def _check_no_api_key(content_by_name: dict[str, bytes], api_key: str) -> None:
    secret = api_key.encode("utf-8")
    if not secret:
        raise RecordingManifestError("Manifest API key is required")
    escaped = json.dumps(api_key, ensure_ascii=True)[1:-1].encode("ascii")
    for content in content_by_name.values():
        assert_safe_bytes(content, secrets=(secret, escaped))


def _snapshot_inputs(
    *, directory: Path, chunks: list[dict], ontology_path: Path, prompts_dir: Path, api_key: str
) -> dict[str, dict[str, Any]]:
    if not chunks or len(chunks) > _MAX_RECORDING_CHUNKS:
        raise RecordingManifestError("recording requires one or two chunks")
    if any(not isinstance(chunk, dict) for chunk in chunks):
        raise RecordingManifestError("chunks must be JSON objects")

    chunk_bytes = _jsonl_bytes(chunks)
    ontology_bytes = Path(ontology_path).read_bytes()
    prompt_artifacts = prompts_module.verify_prompt_artifacts_v2(Path(prompts_dir))
    entity_prompt = prompt_artifacts[prompts_module.ENTITY_PROMPT_FILE_V2]["bytes"]
    relation_prompt = prompt_artifacts[prompts_module.RELATION_PROMPT_FILE_V2]["bytes"]

    # Generate and freeze the exact schemas that the unchanged pipeline derives.
    ontology = schema_module.load_ontology(Path(ontology_path))
    entity_schema = _json_bytes(schema_module.entity_pass_v2_schema(ontology))
    relation_schema = _json_bytes(schema_module.relation_pass_v2_schema(ontology))
    snapshots = {
        "chunks": chunk_bytes,
        "ontology": ontology_bytes,
        "entity_prompt": entity_prompt,
        "relation_prompt": relation_prompt,
        "entity_schema": entity_schema,
        "relation_schema": relation_schema,
    }
    _check_no_api_key(snapshots, api_key)

    entries: dict[str, dict[str, Any]] = {}
    for name, content in snapshots.items():
        relative = _INPUT_FILES[name]
        private_write(directory / relative, content)
        entries[name] = _file_entry(relative, content)
    return entries


def _read_verified_file(directory: Path, entry: Any, *, label: str) -> bytes:
    if not isinstance(entry, dict):
        raise RecordingManifestError(f"missing {label} file identity")
    relative = entry.get("path")
    expected_hash = entry.get("sha256")
    expected_size = entry.get("size")
    if not isinstance(relative, str) or not isinstance(expected_hash, str) or type(expected_size) is not int:
        raise RecordingManifestError(f"invalid {label} file identity")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RecordingManifestError(f"unsafe {label} path")
    path = directory / relative_path
    if path.is_symlink() or path.resolve(strict=False) != path.absolute():
        raise RecordingManifestError(f"unsafe {label} path")
    if not path.is_file():
        raise RecordingManifestError(f"missing {label}")
    raw = path.read_bytes()
    if len(raw) != expected_size or sha256_bytes(raw) != expected_hash:
        raise RecordingManifestError(f"{label} hash mismatch")
    return raw


def _write_manifest(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    private_write(directory / RECORDING_FILENAME, _json_bytes(manifest))
    return manifest


def record_run(
    *,
    output_dir: Path,
    chunks: list[dict],
    ontology_path: Path,
    prompts_dir: Path,
    max_live_calls: int = 4,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    api_key: str | None = None,
    client: Any = None,
) -> dict:
    """Perform a bounded live run while durably recording every HTTP attempt."""
    if type(max_live_calls) is not int or not 1 <= max_live_calls <= 4:
        raise RecordingManifestError("max_live_calls must be between 1 and 4")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise RecordingManifestError("max_tokens must be a positive integer")
    if not isinstance(temperature, (int, float)) or not math.isfinite(float(temperature)):
        raise RecordingManifestError("temperature must be finite")
    key = api_key if api_key is not None else os.environ.get("MANIFEST_API_KEY", "")
    if not key:
        raise RecordingManifestError("Manifest API key is required")

    implementation = _implementation_identity()
    directory = _ensure_fresh_external_directory(Path(output_dir))
    files = _snapshot_inputs(
        directory=directory,
        chunks=chunks,
        ontology_path=Path(ontology_path),
        prompts_dir=Path(prompts_dir),
        api_key=key,
    )
    recording_client: RecordingClient | None = None
    result: RunResultV2 | None = None
    terminal: dict[str, Any] | None = None
    terminal_exc: BaseException | None = None
    try:
        recording_client = RecordingClient(directory, api_key=key, client=client)
        provider = ManifestAutoBenchmarkLLM(
            api_key=key,
            client=recording_client,
            max_live_calls=max_live_calls,
        )
        with _private_umask():
            result = run_extraction_v2(
                provider=provider,
                output_jsonl=directory / EXTRACTIONS_FILENAME,
                output_rejects=directory / REJECTS_FILENAME,
                chunks_path=directory / _INPUT_FILES["chunks"],
                ontology_path=directory / _INPUT_FILES["ontology"],
                prompts_dir=directory / "inputs/prompts",
                max_tokens=max_tokens,
                temperature=temperature,
            )
    except BaseException as exc:
        terminal = _safe_terminal(exc)
        terminal_exc = exc
    finally:
        if recording_client is not None:
            recording_client.close()

    for label, filename in (("extractions", EXTRACTIONS_FILENAME), ("rejects", REJECTS_FILENAME)):
        artifact = directory / filename
        if artifact.is_file():
            raw = artifact.read_bytes()
            files[label] = _file_entry(filename, raw)

    implementation_drift = _implementation_identity() != implementation
    adapter_terminal = provider.last_terminal_transport_error if "provider" in locals() else None
    if implementation_drift:
        terminal = {"type": "ImplementationDrift"}
        status = "terminal_error"
    elif adapter_terminal is not None:
        terminal = _safe_terminal(adapter_terminal)
        status = "terminal_error"
    elif result is not None:
        status = "budget_aborted" if result.budget_aborted else "complete"
        if result.budget_aborted:
            terminal = {"type": "CallBudgetExceeded", "cap": max_live_calls}
    elif isinstance(terminal_exc, (KeyboardInterrupt, SystemExit)):
        status = "interrupted"
    else:
        status = "terminal_error"
    attempts_path = directory / "attempts.jsonl"
    attempts_entry = (
        _file_entry("attempts.jsonl", attempts_path.read_bytes())
        if attempts_path.is_file() else None
    )
    manifest: dict[str, Any] = {
        "format": RECORDING_FORMAT,
        "status": status,
        "inputs": {name: files[name] for name in _INPUT_FILES},
        "artifacts": {name: files[name] for name in ("extractions", "rejects") if name in files},
        "raw_responses": {
            "path": "raw_responses",
            "attempt_count": recording_client.attempt_count if recording_client else 0,
            "attempts_index": attempts_entry,
        },
        "settings": {
            "requested_model": "auto",
            "max_live_calls": max_live_calls,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chunk_count": len(chunks),
            "chunk_ids": [chunk.get("chunk_id") for chunk in chunks],
            "schema_version": schema_module.SCHEMA_VERSION_V2,
            "ontology_version": schema_module.load_ontology(
                directory / _INPUT_FILES["ontology"]
            ).get("version"),
            "entity_prompt_filename": prompts_module.ENTITY_PROMPT_FILE_V2,
            "relation_prompt_filename": prompts_module.RELATION_PROMPT_FILE_V2,
        },
        "implementation": implementation,
        "result": asdict(result) if result is not None else None,
        "terminal": terminal,
    }
    _write_manifest(directory, manifest)
    if isinstance(terminal_exc, (KeyboardInterrupt, SystemExit)):
        raise terminal_exc
    return manifest


def load_recording(recording_dir: Path) -> dict:
    """Load and fully verify a recording without constructing a network client."""
    directory = external_directory(Path(recording_dir), create=False)
    manifest_path = directory / RECORDING_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RecordingManifestError("recording manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordingManifestError("recording manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != RECORDING_FORMAT:
        raise RecordingManifestError("unsupported recording format")
    if manifest.get("status") not in {"complete", "budget_aborted", "terminal_error", "interrupted"}:
        raise RecordingManifestError("invalid recording terminal status")
    if manifest.get("status") == "interrupted":
        raise MissingRawResponse("recording was interrupted and is not replayable")
    settings = manifest.get("settings")
    if not isinstance(settings, dict) or settings.get("requested_model") != "auto":
        raise RecordingManifestError("invalid recording settings")
    if type(settings.get("max_live_calls")) is not int or not 1 <= settings["max_live_calls"] <= 4:
        raise RecordingManifestError("invalid recorded call cap")
    if type(settings.get("max_tokens")) is not int or not isinstance(settings.get("temperature"), (int, float)):
        raise RecordingManifestError("invalid recorded generation settings")
    if settings.get("schema_version") != schema_module.SCHEMA_VERSION_V2:
        raise RecordingManifestError("recorded V2 schema contract mismatch")
    if type(settings.get("ontology_version")) is not int:
        raise RecordingManifestError("invalid ontology version identity")
    if settings.get("entity_prompt_filename") != prompts_module.ENTITY_PROMPT_FILE_V2:
        raise RecordingManifestError("entity prompt contract mismatch")
    if settings.get("relation_prompt_filename") != prompts_module.RELATION_PROMPT_FILE_V2:
        raise RecordingManifestError("relation prompt contract mismatch")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(_INPUT_FILES):
        raise RecordingManifestError("recording inputs are incomplete")
    verified_inputs = {
        name: _read_verified_file(directory, inputs[name], label=name) for name in _INPUT_FILES
    }
    chunks = _read_jsonl_bytes(verified_inputs["chunks"], label="chunks")
    if len(chunks) != settings.get("chunk_count") or [c.get("chunk_id") for c in chunks] != settings.get("chunk_ids"):
        raise RecordingManifestError("recorded chunk identity mismatch")

    # Check frozen generated schemas against the frozen ontology, not caller files.
    ontology = schema_module.load_ontology(directory / inputs["ontology"]["path"])
    if ontology.get("version") != settings["ontology_version"]:
        raise RecordingManifestError("recorded ontology version mismatch")
    expected_entity = _json_bytes(schema_module.entity_pass_v2_schema(ontology))
    expected_relation = _json_bytes(schema_module.relation_pass_v2_schema(ontology))
    if verified_inputs["entity_schema"] != expected_entity or verified_inputs["relation_schema"] != expected_relation:
        raise RecordingManifestError("recorded schema snapshot drift")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RecordingManifestError("invalid artifact identities")
    verified_artifacts = {
        name: _read_verified_file(directory, entry, label=name) for name, entry in artifacts.items()
    }
    if set(verified_artifacts) - {"extractions", "rejects"}:
        raise RecordingManifestError("unknown canonical artifact")
    replayable_outputs = manifest.get("status") in {"complete", "budget_aborted"}
    if replayable_outputs and set(verified_artifacts) != {"extractions", "rejects"}:
        raise RecordingManifestError("canonical artifacts are incomplete")
    if manifest.get("status") == "terminal_error" and verified_artifacts:
        if set(verified_artifacts) != {"extractions", "rejects"}:
            raise RecordingManifestError("partial canonical artifact set")
    raw_entry = manifest.get("raw_responses")
    if not isinstance(raw_entry, dict) or raw_entry.get("path") != "raw_responses":
        raise RecordingManifestError("invalid raw response location")
    attempts_bytes = _read_verified_file(
        directory, raw_entry.get("attempts_index"), label="attempts index"
    )
    if not attempts_bytes or type(raw_entry.get("attempt_count")) is not int:
        raise MissingRawResponse("raw attempt index is incomplete")
    if raw_entry["attempt_count"] < 1:
        raise MissingRawResponse("recording contains no HTTP attempts")

    _verify_implementation_identity(manifest.get("implementation"))
    manifest["_directory"] = directory
    manifest["_verified_inputs"] = verified_inputs
    manifest["_verified_artifacts"] = verified_artifacts
    manifest["_manifest_sha256"] = sha256_bytes(manifest_path.read_bytes())
    manifest["_chunks"] = chunks
    return manifest


def run_replay_pipeline(
    recording_dir: Path,
    output_dir: Path,
    *,
    provider_wrapper: Callable[[Any], Any] | None = None,
) -> tuple[RunResultV2 | None, ReplayClient, Any]:
    """Run the frozen V2 inputs through the offline HTTP replay client.

    ``provider_wrapper`` is intentionally narrow: experiments may transform the
    adapter's parsed value, but HTTP request generation and parsing still happen
    in the unchanged Manifest Auto adapter.
    """
    manifest = load_recording(recording_dir)
    source = manifest["_directory"]
    original_output = Path(output_dir)
    target_candidate = original_output.expanduser().absolute().resolve(strict=False)
    if source == target_candidate or source in target_candidate.parents or target_candidate in source.parents:
        raise RecordingManifestError("replay output overlaps recording source")
    target = _ensure_fresh_external_directory(original_output)

    raw_entry = manifest.get("raw_responses")
    if not isinstance(raw_entry, dict) or raw_entry.get("path") != "raw_responses":
        raise RecordingManifestError("invalid raw response location")
    replay_client = ReplayClient(source)
    if replay_client.attempt_count != raw_entry["attempt_count"]:
        raise MissingRawResponse("attempt count disagrees with recording manifest")
    settings = manifest["settings"]
    provider = ManifestAutoBenchmarkLLM(
        api_key="offline-replay",
        client=replay_client,
        max_live_calls=settings["max_live_calls"],
    )
    active_provider = provider_wrapper(provider) if provider_wrapper is not None else provider
    if active_provider is None:
        raise RecordingManifestError("provider wrapper returned no provider")

    result: RunResultV2 | None = None
    caught: BaseException | None = None
    try:
        with _private_umask():
            result = run_extraction_v2(
                provider=active_provider,
                output_jsonl=target / EXTRACTIONS_FILENAME,
                output_rejects=target / REJECTS_FILENAME,
                chunks_path=source / manifest["inputs"]["chunks"]["path"],
                ontology_path=source / manifest["inputs"]["ontology"]["path"],
                prompts_dir=source / "inputs/prompts",
                max_tokens=settings["max_tokens"],
                temperature=float(settings["temperature"]),
            )
    except BaseException as exc:
        caught = exc
    try:
        replay_client.assert_consumed()
    except BaseException:
        replay_client.close()
        raise
    replay_client.close()
    if caught is not None:
        raise _ReplayedTerminal(caught, replay_client, provider)
    return result, replay_client, active_provider


def _exact_comparison(manifest: dict, output_dir: Path) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    expected = manifest["_verified_artifacts"]
    for label, filename in (("extractions", EXTRACTIONS_FILENAME), ("rejects", REJECTS_FILENAME)):
        output_path = output_dir / filename
        actual = output_path.read_bytes() if output_path.is_file() else None
        wanted = expected.get(label)
        match = actual is not None and wanted is not None and actual == wanted
        comparisons[label] = {
            "match": match,
            "expected_sha256": sha256_bytes(wanted) if wanted is not None else None,
            "actual_sha256": sha256_bytes(actual) if actual is not None else None,
        }
    return {"match": all(v["match"] for v in comparisons.values()), "files": comparisons}


def _load_rows(path: Path) -> list[dict]:
    return _read_jsonl_bytes(path.read_bytes(), label=path.name)


def _write_empty_review(manifest: dict, output_dir: Path, exact: dict[str, Any]) -> None:
    from src.benchmark.empty_chunk_review import build_empty_chunk_review

    review = build_empty_chunk_review(
        chunks=manifest["_chunks"],
        records=_load_rows(output_dir / EXTRACTIONS_FILENAME),
        rejects=_load_rows(output_dir / REJECTS_FILENAME),
        provenance={
            "recording_manifest_sha256": manifest["_manifest_sha256"],
            "canonical_comparison": exact,
        },
    )
    private_write(output_dir / EMPTY_REVIEW_FILENAME, _json_bytes(review))


def replay_run(
    *, recording_dir: Path, output_dir: Path, whitespace_comparison: bool = False
) -> dict:
    """Replay a verified recording offline and require byte-identical artifacts."""
    manifest = load_recording(recording_dir)
    output = Path(output_dir)
    result: RunResultV2 | None = None
    terminal: dict[str, Any] | None = None
    replay_error: BaseException | None = None
    try:
        result, client, replay_provider = run_replay_pipeline(recording_dir, output)
        attempt_count = client.attempt_count
    except MissingRawResponse:
        # Corrupt/incomplete raw data is never a reproduced historical error.
        raise
    except _ReplayedTerminal as exc:
        replay_error = exc.original
        replay_provider = exc.provider
        terminal = _safe_terminal(exc.original)
        attempt_count = exc.client.attempt_count
    except BaseException as exc:
        replay_error = exc
        terminal = _safe_terminal(exc)
        attempt_count = None

    recorded_terminal = manifest.get("terminal")
    if replay_error is not None:
        if manifest.get("status") != "terminal_error" or terminal != recorded_terminal:
            raise replay_error
        if manifest["_verified_artifacts"]:
            raise ReplayMismatch("terminal recording unexpectedly has canonical artifacts")
        if attempt_count != manifest["raw_responses"]["attempt_count"]:
            raise ReplayMismatch("terminal replay attempt count differs")
        report = {
            "format": RECORDING_FORMAT,
            "status": "terminal_error_reproduced",
            "recording_manifest_sha256": manifest["_manifest_sha256"],
            "attempt_count": attempt_count,
            "terminal": terminal,
            "exact": {"match": True, "files": {}},
        }
        private_write(output / REPLAY_REPORT_FILENAME, _json_bytes(report))
        return report

    adapter_terminal = getattr(replay_provider, "last_terminal_transport_error", None)
    if adapter_terminal is not None:
        if manifest.get("status") != "terminal_error" or _safe_terminal(adapter_terminal) != recorded_terminal:
            raise ReplayMismatch("replay produced an unrecorded terminal error")
    exact = _exact_comparison(manifest, output)
    if not exact["match"]:
        raise ReplayMismatch("canonical extraction or rejects bytes differ")
    expected_result = manifest.get("result")
    actual_result = asdict(result) if result is not None else None
    # run_id embeds wall-clock time; it is audit metadata, not canonical output.
    comparable_actual = dict(actual_result or {})
    comparable_expected = dict(expected_result or {})
    comparable_actual.pop("run_id", None)
    comparable_expected.pop("run_id", None)
    if comparable_actual != comparable_expected:
        raise ReplayMismatch("replayed run result differs")
    expected_attempts = manifest["raw_responses"].get("attempt_count")
    if attempt_count != expected_attempts:
        raise ReplayMismatch("replayed HTTP attempt count differs")

    report: dict[str, Any] = {
        "format": RECORDING_FORMAT,
        "status": "matched",
        "recording_manifest_sha256": manifest["_manifest_sha256"],
        "attempt_count": attempt_count,
        "exact": exact,
        "result": actual_result,
    }
    private_write(output / REPLAY_REPORT_FILENAME, _json_bytes(report))
    _write_empty_review(manifest, output, exact)
    if whitespace_comparison:
        from src.benchmark.manifest_comparison import compare_whitespace

        report["whitespace_comparison"] = compare_whitespace(
            recording_dir=Path(recording_dir), output_dir=output
        )
        private_write(output / REPLAY_REPORT_FILENAME, _json_bytes(report))
    return report
