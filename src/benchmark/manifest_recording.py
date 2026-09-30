"""Private raw HTTP recording and offline replay for Manifest benchmarks."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import httpx

from src.providers.cache import current_extraction_cache_scope


class RecordingError(RuntimeError):
    """A recording or replay artifact is unsafe or invalid."""


class MissingRawResponse(RecordingError):
    """A replay response is missing, incomplete, or fails integrity checks."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


_CREDENTIAL_KEYS = frozenset(
    {"authorization", "proxy-authorization", "api_key", "api-key", "x-api-key"}
)
_HEADER_CREDENTIAL_RE = re.compile(
    rb"(?:authorization|proxy-authorization|x-api-key|api-key|api_key)\s*:\s*\S+",
    re.IGNORECASE,
)
_SECRET_SHAPE_RE = re.compile(
    rb"(?:sk-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._-]{8,}|ghp_[A-Za-z0-9]{20,})",
    re.IGNORECASE,
)


def _scan_decoded(value: Any, secrets: tuple[bytes, ...], *, depth: int = 0) -> None:
    if depth > 4:
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            key_bytes = key_text.encode("utf-8")
            if key_text.strip().lower() in _CREDENTIAL_KEYS:
                raise RecordingError("credential field cannot be recorded")
            for secret in secrets:
                if secret and secret in key_bytes:
                    raise RecordingError("configured credential cannot be recorded")
            _scan_decoded(nested, secrets, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _scan_decoded(nested, secrets, depth=depth + 1)
        return
    if not isinstance(value, str):
        return
    encoded = value.encode("utf-8")
    for secret in secrets:
        if secret and secret in encoded:
            raise RecordingError("configured credential cannot be recorded")
    if _HEADER_CREDENTIAL_RE.search(encoded) or _SECRET_SHAPE_RE.search(encoded):
        raise RecordingError("credential material cannot be recorded")
    unicode_decoded = re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda match: chr(int(match.group(1), 16)),
        value,
    )
    if unicode_decoded != value:
        _scan_decoded(unicode_decoded, secrets, depth=depth + 1)
    if depth < 4 and value.lstrip().startswith(("{", "[", '"')):
        try:
            nested = json.loads(value)
        except json.JSONDecodeError:
            return
        if nested != value:
            _scan_decoded(nested, secrets, depth=depth + 1)


def assert_safe_bytes(content: bytes, *, secrets: tuple[bytes, ...] = ()) -> None:
    """Reject recognized credential material without mutating ``content``.

    ``secrets`` should include exact UTF-8 credentials known to the caller.
    JSON input is decoded recursively, which catches credentials hidden behind
    JSON Unicode escapes as well as nested credential-bearing fields. Arbitrary
    unknown secrets cannot be proven absent; strict recognition also rejects
    common Authorization/API-key field names and token shapes.
    """
    for secret in secrets:
        if secret and secret in content:
            raise RecordingError("configured credential cannot be recorded")
    if _HEADER_CREDENTIAL_RE.search(content) or _SECRET_SHAPE_RE.search(content):
        raise RecordingError("credential material cannot be recorded")
    try:
        decoded = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return
    _scan_decoded(decoded, secrets)


def _repo_root() -> Path:
    return Path(__file__).resolve(strict=True).parents[2]


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _git_ancestor(path: Path) -> Path | None:
    for ancestor in (path, *path.parents):
        marker = ancestor / ".git"
        if marker.is_dir() or marker.is_file():
            return ancestor
    return None


def _require_private_mode(path: Path) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise RecordingError(f"directory is not private: {path} ({oct(mode)})")


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise RecordingError(f"symlink path component is forbidden: {current}")


def external_directory(path: Path, *, create: bool = False) -> Path:
    """Validate or create a directory outside every Git worktree."""
    original = Path(path).expanduser()
    if not original.is_absolute():
        original = Path.cwd() / original
    _reject_symlink_components(original)
    resolved = original.resolve(strict=False)
    if _inside(resolved, _repo_root()) or _git_ancestor(resolved) is not None:
        raise RecordingError(f"Git-repository-contained directory is forbidden: {resolved}")
    if resolved.exists():
        if not resolved.is_dir():
            raise RecordingError(f"not a directory: {resolved}")
        return resolved
    if not create:
        raise RecordingError(f"directory does not exist: {resolved}")
    missing: list[Path] = []
    cursor = resolved
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if not cursor.is_dir():
        raise RecordingError(f"ancestor is not a directory: {cursor}")
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
        os.chmod(directory, 0o700)
    return resolved


def private_write(path: Path, content: bytes) -> None:
    """Atomically write an external regular file with mode 0600."""
    if not isinstance(content, bytes):
        raise TypeError("private_write content must be bytes")
    destination = Path(path)
    parent = external_directory(destination.parent, create=True)
    destination = parent / destination.name
    if destination.is_symlink():
        raise RecordingError(f"symlink file is forbidden: {destination}")
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        if destination.is_symlink():
            raise RecordingError(f"symlink file is forbidden: {destination}")
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@dataclasses.dataclass(frozen=True)
class AttemptRecord:
    attempt: int
    complete: bool
    scope: dict[str, str] | None
    request_sha256: str
    request_bytes: int
    response_sha256: str | None
    response_bytes: int | None
    status: int | None
    requested_model: str | None
    reported_model: str | None
    finish_reason: str | None
    abort_reason: str | None
    error_type: str | None
    transient: bool | None
    started_at: str
    duration_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_ATTEMPTS = "attempts.jsonl"
_RAW = "raw_responses"
_REQUESTS = "raw_requests"
_UNKNOWN_MODEL = "<unknown>"
_TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_NETWORK_TYPES: dict[str, type[httpx.HTTPError]] = {
    "ConnectError": httpx.ConnectError,
    "ConnectTimeout": httpx.ConnectTimeout,
    "ReadTimeout": httpx.ReadTimeout,
    "WriteTimeout": httpx.WriteTimeout,
    "PoolTimeout": httpx.PoolTimeout,
    "ReadError": httpx.ReadError,
    "WriteError": httpx.WriteError,
    "RemoteProtocolError": httpx.RemoteProtocolError,
    "LocalProtocolError": httpx.LocalProtocolError,
    "ProxyError": httpx.ProxyError,
    "UnsupportedProtocol": httpx.UnsupportedProtocol,
    "TimeoutException": httpx.TimeoutException,
    "NetworkError": httpx.NetworkError,
    "ProtocolError": httpx.ProtocolError,
}


_HEX_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def _valid_optional_text(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _validate_record(record: AttemptRecord, expected: int) -> None:
    if type(record.attempt) is not int or record.attempt != expected or record.attempt < 1:
        raise MissingRawResponse(f"attempt id/order mismatch at line {expected}")
    if type(record.complete) is not bool:
        raise MissingRawResponse(f"invalid completion flag for attempt {expected}")
    if record.scope is not None and (
        not isinstance(record.scope, dict)
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in record.scope.items())
    ):
        raise MissingRawResponse(f"invalid scope for attempt {expected}")
    if not isinstance(record.request_sha256, str) or not _HEX_DIGEST_RE.fullmatch(record.request_sha256):
        raise MissingRawResponse(f"invalid request hash for attempt {expected}")
    if type(record.request_bytes) is not int or record.request_bytes < 0:
        raise MissingRawResponse(f"invalid request length for attempt {expected}")
    if record.response_sha256 is not None and (
        not isinstance(record.response_sha256, str)
        or not _HEX_DIGEST_RE.fullmatch(record.response_sha256)
    ):
        raise MissingRawResponse(f"invalid response hash for attempt {expected}")
    if record.response_bytes is not None and (
        type(record.response_bytes) is not int or record.response_bytes < 0
    ):
        raise MissingRawResponse(f"invalid response length for attempt {expected}")
    if record.status is not None and (
        type(record.status) is not int or not 100 <= record.status <= 599
    ):
        raise MissingRawResponse(f"invalid HTTP status for attempt {expected}")
    if any(
        not _valid_optional_text(value)
        for value in (
            record.requested_model, record.reported_model, record.finish_reason,
            record.abort_reason, record.error_type,
        )
    ):
        raise MissingRawResponse(f"invalid text metadata for attempt {expected}")
    if record.transient is not None and type(record.transient) is not bool:
        raise MissingRawResponse(f"invalid transient flag for attempt {expected}")
    if (
        not isinstance(record.started_at, str)
        or not record.started_at
        or type(record.duration_seconds) not in (int, float)
        or not math.isfinite(record.duration_seconds)
        or record.duration_seconds < 0
    ):
        raise MissingRawResponse(f"invalid timing metadata for attempt {expected}")


def _scope() -> dict[str, str] | None:
    active = current_extraction_cache_scope()
    if active is None:
        return None
    return {str(key): str(active[key]) for key in sorted(active)}


def _safe_response_metadata(
    content: bytes, secrets: tuple[bytes, ...]
) -> tuple[str | None, str | None, str | None]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, None, None
    if not isinstance(value, Mapping):
        return None, None, None
    model = value.get("model")
    reported = model if isinstance(model, str) and model else _UNKNOWN_MODEL
    finish: str | None = None
    choices = value.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        candidate = choices[0].get("finish_reason") or choices[0].get("stop_reason")
        if isinstance(candidate, str):
            finish = candidate
    abort: str | None = None
    error = value.get("error")
    if isinstance(error, Mapping):
        candidate = error.get("type") or error.get("code")
        if isinstance(candidate, (str, int)):
            abort = str(candidate)
    assert_safe_bytes(
        json.dumps([reported, finish, abort], ensure_ascii=False).encode("utf-8"),
        secrets=secrets,
    )
    return reported, finish, abort


class RecordingClient:
    """HTTP-compatible client that stores exact request/response bytes.

    When ``client`` is omitted, the owned client uses an HTTP transport with
    ``retries=0``. Redirects are always disabled. Injected custom transports
    are trusted test-only boundaries because arbitrary hidden transport
    retries cannot be introspected.
    """

    def __init__(
        self,
        directory: Path,
        *,
        api_key: str,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise RecordingError("api_key is required for credential scanning")
        self.directory = external_directory(directory, create=True)
        _require_private_mode(self.directory)
        self._requests = external_directory(self.directory / _REQUESTS, create=True)
        self._raw = external_directory(self.directory / _RAW, create=True)
        self._index = self.directory / _ATTEMPTS
        if self._index.exists() or any(self._requests.iterdir()) or any(self._raw.iterdir()):
            raise RecordingError("recording directory must be fresh")
        self._owned = client is None
        self._client = client or httpx.Client(
            transport=httpx.HTTPTransport(retries=0),
            timeout=60.0,
            follow_redirects=False,
        )
        self._secrets = (api_key.encode("utf-8"),)
        self._records: list[AttemptRecord] = []
        self._closed = False

    @property
    def attempt_count(self) -> int:
        return len(self._records)

    def _persist(self) -> None:
        content = b"".join(
            json.dumps(
                record.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            + b"\n"
            for record in self._records
        )
        private_write(self._index, content)

    def _replace_last(self, record: AttemptRecord) -> None:
        if not self._records or self._records[-1].attempt != record.attempt:
            raise RecordingError("attempt finalization is out of sequence")
        self._records[-1] = record
        self._persist()

    def post(
        self,
        url: str,
        json: Any = None,
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        if self._closed:
            raise RecordingError("RecordingClient is closed")
        request = self._client.build_request(
            "POST", url, json=json, headers=dict(headers or {}), **kwargs
        )
        request_body = request.content
        assert_safe_bytes(request_body, secrets=self._secrets)
        scope = _scope()
        if scope is not None:
            assert_safe_bytes(json_dumps_bytes(scope), secrets=self._secrets)
        requested = None
        if isinstance(json, Mapping) and isinstance(json.get("model"), str):
            requested = json["model"]
        attempt = len(self._records) + 1
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        intent = AttemptRecord(
            attempt=attempt,
            complete=False,
            scope=scope,
            request_sha256=sha256_bytes(request_body),
            request_bytes=len(request_body),
            response_sha256=None,
            response_bytes=None,
            status=None,
            requested_model=requested,
            reported_model=None,
            finish_reason=None,
            abort_reason="incomplete",
            error_type=None,
            transient=None,
            started_at=started,
            duration_seconds=0.0,
        )
        # Persist exact request bytes and a durable incomplete index entry
        # before sending. A crash at any later point is detectable offline.
        private_write(self._requests / f"{attempt:06d}.bin", request_body)
        self._records.append(intent)
        self._persist()
        began = time.perf_counter()
        try:
            response = self._client.send(request, follow_redirects=False)
        except httpx.HTTPError as exc:
            duration = time.perf_counter() - began
            name = type(exc).__name__
            try:
                assert_safe_bytes(str(exc).encode("utf-8"), secrets=self._secrets)
            except RecordingError:
                self._replace_last(
                    dataclasses.replace(
                        intent,
                        abort_reason="credential_material_unavailable",
                        duration_seconds=duration,
                    )
                )
                raise
            if name not in _NETWORK_TYPES:
                self._replace_last(
                    dataclasses.replace(
                        intent,
                        abort_reason="unknown_transport_error",
                        duration_seconds=duration,
                    )
                )
                raise RecordingError("unsupported transport error type") from exc
            self._replace_last(
                dataclasses.replace(
                    intent,
                    complete=True,
                    abort_reason=name,
                    error_type=name,
                    transient=True,
                    duration_seconds=duration,
                )
            )
            raise
        duration = time.perf_counter() - began
        content = bytes(response.content)
        try:
            assert_safe_bytes(content, secrets=self._secrets)
            reported, finish, abort = _safe_response_metadata(content, self._secrets)
        except RecordingError:
            self._replace_last(
                dataclasses.replace(
                    intent,
                    abort_reason="credential_material_unavailable",
                    duration_seconds=duration,
                )
            )
            # Per spec: API keys and Authorization material must never be
            # persisted. The secret-bearing response body is refused and
            # discarded; only the durable incomplete attempt remains.
            raise
        response_hash = sha256_bytes(content)
        private_write(self._raw / f"{attempt:06d}.bin", content)
        status = int(response.status_code)
        self._replace_last(
            dataclasses.replace(
                intent,
                complete=True,
                response_sha256=response_hash,
                response_bytes=len(content),
                status=status,
                reported_model=reported,
                finish_reason=finish,
                abort_reason=abort,
                transient=status in _TRANSIENT_STATUSES if status >= 400 else None,
                duration_seconds=duration,
            )
        )
        return response

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owned:
            self._client.close()


def json_dumps_bytes(value: Any) -> bytes:
    """Small internal canonical JSON helper for metadata only."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


class ReplayClient:
    """Network-incapable client that validates and replays recorded attempts."""

    def __init__(self, directory: Path) -> None:
        self.directory = external_directory(directory)
        self._requests = self.directory / _REQUESTS
        self._raw = self.directory / _RAW
        self._index = self.directory / _ATTEMPTS
        if self._index.is_symlink() or not self._index.is_file():
            raise MissingRawResponse("attempt index is missing or symlinked")
        if self._requests.is_symlink() or not self._requests.is_dir():
            raise MissingRawResponse("raw request directory is missing or symlinked")
        if self._raw.is_symlink() or not self._raw.is_dir():
            raise MissingRawResponse("raw response directory is missing or symlinked")
        try:
            _require_private_mode(self.directory)
            _require_private_mode(self._requests)
            _require_private_mode(self._raw)
        except RecordingError as exc:
            raise MissingRawResponse(str(exc)) from exc
        self._records: list[AttemptRecord] = []
        self._bodies: dict[int, bytes] = {}
        self._cursor = 0
        self._load()

    @property
    def attempt_count(self) -> int:
        return len(self._records)

    @staticmethod
    def _private_file(path: Path) -> None:
        mode = stat.S_IMODE(os.stat(path, follow_symlinks=False).st_mode)
        if mode & 0o077:
            raise MissingRawResponse(f"unsafe artifact permissions: {path}")

    def _load(self) -> None:
        self._private_file(self._index)
        try:
            lines = self._index.read_bytes().splitlines()
        except OSError as exc:
            raise MissingRawResponse("cannot read attempt index") from exc
        if not lines:
            raise MissingRawResponse("attempt index is empty")
        for expected, line in enumerate(lines, 1):
            try:
                value = json.loads(line.decode("utf-8"))
                if not isinstance(value, dict):
                    raise TypeError
                record = AttemptRecord(**value)
                _validate_record(record, expected)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                raise MissingRawResponse(f"invalid attempt index line {expected}") from exc
            raw_request = self._requests / f"{expected:06d}.bin"
            if raw_request.is_symlink() or not raw_request.is_file():
                raise MissingRawResponse(f"raw request {expected} is missing or symlinked")
            self._private_file(raw_request)
            request_body = raw_request.read_bytes()
            if sha256_bytes(request_body) != record.request_sha256:
                raise MissingRawResponse(f"raw request {expected} hash mismatch")
            if len(request_body) != record.request_bytes:
                raise MissingRawResponse(f"raw request {expected} length mismatch")
            if not record.complete:
                raise MissingRawResponse(f"attempt {expected} is incomplete")
            if record.error_type is not None:
                if record.error_type not in _NETWORK_TYPES or record.response_sha256 is not None:
                    raise MissingRawResponse(f"invalid transport attempt {expected}")
                self._bodies[expected] = b""
            else:
                if record.response_sha256 is None or record.status is None:
                    raise MissingRawResponse(f"attempt {expected} has no raw response")
                raw = self._raw / f"{expected:06d}.bin"
                if raw.is_symlink() or not raw.is_file():
                    raise MissingRawResponse(f"raw response {expected} is missing or symlinked")
                self._private_file(raw)
                body = raw.read_bytes()
                if sha256_bytes(body) != record.response_sha256:
                    raise MissingRawResponse(f"raw response {expected} hash mismatch")
                if record.response_bytes != len(body):
                    raise MissingRawResponse(f"raw response {expected} length mismatch")
                self._bodies[expected] = body
            self._records.append(record)
        request_names = {f"{record.attempt:06d}.bin" for record in self._records}
        if {path.name for path in self._requests.iterdir()} != request_names:
            raise MissingRawResponse("raw request set does not match attempt index")
        response_names = {
            f"{record.attempt:06d}.bin"
            for record in self._records
            if record.response_sha256 is not None
        }
        if {path.name for path in self._raw.iterdir()} != response_names:
            raise MissingRawResponse("raw response set does not match attempt index")

    def post(
        self,
        url: str,
        json: Any = None,
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        if self._cursor == len(self._records):
            raise MissingRawResponse("replay has no remaining attempt")
        record = self._records[self._cursor]
        request_body = httpx.Request("POST", "http://offline.invalid/", json=json).content
        if sha256_bytes(request_body) != record.request_sha256:
            raise MissingRawResponse(f"request body mismatch at attempt {record.attempt}")
        if _scope() != record.scope:
            raise MissingRawResponse(f"extraction scope mismatch at attempt {record.attempt}")
        self._cursor += 1
        offline_request = httpx.Request("POST", "http://offline.invalid/")
        if record.error_type is not None:
            error_class = _NETWORK_TYPES[record.error_type]
            raise error_class("offline replay", request=offline_request)
        return httpx.Response(
            record.status or 200,
            content=self._bodies[record.attempt],
            request=offline_request,
        )

    def assert_consumed(self) -> None:
        if self._cursor != len(self._records):
            raise MissingRawResponse(
                f"{len(self._records) - self._cursor} replay attempt(s) remain"
            )

    def close(self) -> None:
        return None


__all__ = [
    "AttemptRecord",
    "MissingRawResponse",
    "RecordingClient",
    "RecordingError",
    "ReplayClient",
    "assert_safe_bytes",
    "external_directory",
    "private_write",
    "sha256_bytes",
]
