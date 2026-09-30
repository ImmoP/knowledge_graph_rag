"""Behavior tests for private Manifest raw recording and offline replay."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import httpx
import pytest

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
from src.extract.retry import with_transport_retry
from src.providers.adapters.manifest_openai import ManifestAutoBenchmarkLLM
from src.providers.cache import extraction_cache_scope


API_KEY = "recording-test-key-not-a-live-secret"
URL = "http://manifest.test/v1/chat/completions"
SCOPE = {
    "chunk_id": "c_test",
    "source_text_sha256": "1" * 64,
    "ontology_version": "test",
    "ontology_sha256": "2" * 64,
    "prompt_template_filename": "extract_entities.txt",
    "prompt_template_content_sha256": "3" * 64,
}


def _response(content: bytes, status: int = 200) -> httpx.Response:
    return httpx.Response(status, content=content)


def _success_body(content: str = '{"mentions":[]}') -> bytes:
    return json.dumps(
        {
            "model": "provider/model",
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _recording(tmp_path: Path, handler) -> tuple[Path, RecordingClient, httpx.Client]:
    root = tmp_path / "recording"
    underlying = httpx.Client(transport=httpx.MockTransport(handler))
    recorder = RecordingClient(root, api_key=API_KEY, client=underlying)
    return root, recorder, underlying


def _lines(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()]


def test_records_and_replays_exact_request_and_response_bytes(tmp_path: Path) -> None:
    raw = b' {"model":"provider/model","choices":[]} \n\xff'
    sent: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        return _response(raw)

    root, recorder, underlying = _recording(tmp_path, handler)
    payload = {"model": "auto", "messages": [{"content": "Grüße 東京"}]}
    response = recorder.post(URL, json=payload, headers={"Authorization": f"Bearer {API_KEY}"})
    recorder.close()
    underlying.close()

    assert response.content == raw
    assert (root / "raw_requests" / "000001.bin").read_bytes() == sent[0]
    assert (root / "raw_responses" / "000001.bin").read_bytes() == raw
    line = _lines(root)[0]
    assert line["request_sha256"] == sha256_bytes(sent[0])
    assert line["response_sha256"] == sha256_bytes(raw)
    all_bytes = b"".join(path.read_bytes() for path in root.rglob("*") if path.is_file())
    assert b"Authorization" not in all_bytes
    assert API_KEY.encode() not in all_bytes

    replay = ReplayClient(root)
    replayed = replay.post("http://real-url-is-ignored.invalid", json=payload)
    assert replayed.content == raw
    assert str(replayed.request.url).startswith("http://offline.invalid/")
    replay.assert_consumed()
    replay.close()


def test_multiple_attempts_remain_ordered_and_replay_immutable(tmp_path: Path) -> None:
    bodies = [_success_body("one"), _success_body("two")]
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        result = _response(bodies[calls], status=(503 if calls == 0 else 200))
        calls += 1
        return result

    root, recorder, underlying = _recording(tmp_path, handler)
    first = {"model": "auto", "messages": [{"content": "first"}]}
    second = {"model": "auto", "messages": [{"content": "second"}]}
    recorder.post(URL, json=first)
    recorder.post(URL, json=second)
    assert [line["attempt"] for line in _lines(root)] == [1, 2]
    assert recorder.attempt_count == 2
    underlying.close()

    replay = ReplayClient(root)
    assert replay.post(URL, json=first).status_code == 503
    with pytest.raises(MissingRawResponse, match="request body mismatch"):
        replay.post(URL, json={"model": "auto", "messages": [{"content": "wrong"}]})


def test_extraction_scope_must_match_exactly(tmp_path: Path) -> None:
    root, recorder, underlying = _recording(
        tmp_path, lambda request: _response(_success_body())
    )
    payload = {"model": "auto", "messages": []}
    with extraction_cache_scope(**SCOPE):
        recorder.post(URL, json=payload)
    underlying.close()

    replay = ReplayClient(root)
    with pytest.raises(MissingRawResponse, match="scope mismatch"):
        replay.post(URL, json=payload)

    replay = ReplayClient(root)
    with extraction_cache_scope(**SCOPE):
        replay.post(URL, json=payload)
    replay.assert_consumed()


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "auto", "api_key": "anything"},
        {"model": "auto", "nested": {"Authorization": "Bearer anything"}},
        {"model": "auto", "text": "Authorization: Bearer anything"},
        {"model": "auto", "text": API_KEY},
        {"model": "auto", "text": "recording-test-key-not-a-live-\\u0073ecret"},
    ],
)
def test_secret_request_is_refused_before_network_or_artifacts(
    tmp_path: Path, payload: dict
) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return _response(_success_body())

    root, recorder, underlying = _recording(tmp_path, handler)
    with pytest.raises(RecordingError):
        recorder.post(URL, json=payload)
    assert called is False
    assert not list((root / "raw_requests").iterdir())
    assert not list((root / "raw_responses").iterdir())
    assert not (root / "attempts.jsonl").exists()
    underlying.close()


@pytest.mark.parametrize(
    "body",
    [
        b'{"authorization":"Bearer leaked"}',
        json.dumps({"message": API_KEY}).encode(),
        b'{"message":"Authorization: Bearer leaked"}',
    ],
)
def test_secret_response_aborts_without_storing_secret_bytes(
    tmp_path: Path, body: bytes
) -> None:
    root, recorder, underlying = _recording(
        tmp_path, lambda request: _response(body)
    )
    with pytest.raises(RecordingError):
        recorder.post(URL, json={"model": "auto"})
    assert not list((root / "raw_responses").iterdir())
    persisted = b"".join(path.read_bytes() for path in root.rglob("*") if path.is_file())
    assert body not in persisted
    assert _lines(root)[0]["complete"] is False
    with pytest.raises(MissingRawResponse, match="incomplete"):
        ReplayClient(root)
    underlying.close()


def test_secret_transport_exception_is_unavailable_and_never_persisted(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"Bearer {API_KEY}", request=request)

    root, recorder, underlying = _recording(tmp_path, handler)
    with pytest.raises(RecordingError):
        recorder.post(URL, json={"model": "auto"})
    all_bytes = b"".join(path.read_bytes() for path in root.rglob("*") if path.is_file())
    assert API_KEY.encode() not in all_bytes
    assert b"Bearer" not in all_bytes
    with pytest.raises(MissingRawResponse, match="incomplete"):
        ReplayClient(root)
    underlying.close()


def test_network_exception_is_recorded_and_replayed_as_same_httpx_type(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("upstream timed out", request=request)

    root, recorder, underlying = _recording(tmp_path, handler)
    with pytest.raises(httpx.ReadTimeout):
        recorder.post(URL, json={"model": "auto"})
    assert _lines(root)[0]["error_type"] == "ReadTimeout"
    underlying.close()

    replay = ReplayClient(root)
    with pytest.raises(httpx.ReadTimeout) as excinfo:
        replay.post(URL, json={"model": "auto"})
    assert "upstream" not in str(excinfo.value)
    replay.assert_consumed()


def test_unknown_transport_exception_leaves_incomplete_intent(tmp_path: Path) -> None:
    class Unexpected(RuntimeError):
        pass

    def handler(request: httpx.Request) -> httpx.Response:
        raise Unexpected("do not persist this")

    root, recorder, underlying = _recording(tmp_path, handler)
    with pytest.raises(Unexpected):
        recorder.post(URL, json={"model": "auto"})
    assert b"do not persist this" not in (root / "attempts.jsonl").read_bytes()
    with pytest.raises(MissingRawResponse, match="incomplete"):
        ReplayClient(root)
    underlying.close()


def test_transient_http_and_network_attempts_drive_adapter_retry(tmp_path: Path) -> None:
    scripted: list[object] = [
        _response(b'{"error":{"type":"rate_limit"}}', 429),
        httpx.ConnectTimeout("temporary", request=httpx.Request("POST", URL)),
        _response(_success_body()),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        item = scripted.pop(0)
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, httpx.Response)
        return item

    root, recorder, underlying = _recording(tmp_path, handler)
    adapter = ManifestAutoBenchmarkLLM(
        api_key=API_KEY,
        client=recorder,
        max_live_calls=3,
    )
    sleeps: list[float] = []
    result = with_transport_retry(
        lambda: adapter.generate_json(
            "system",
            "user",
            {"type": "object", "properties": {"mentions": {"type": "array"}}},
        ),
        sleep=sleeps.append,
    )
    assert result[0] == {"mentions": []}
    assert sleeps == [2.0, 4.0]
    assert recorder.attempt_count == 3
    assert [line["status"] for line in _lines(root)] == [429, None, 200]
    underlying.close()


def test_permissions_and_existing_source_mode(tmp_path: Path) -> None:
    existing = tmp_path / "source"
    existing.mkdir(mode=0o755)
    os.chmod(existing, 0o755)
    assert external_directory(existing) == existing.resolve()
    assert stat.S_IMODE(existing.stat().st_mode) == 0o755

    created = external_directory(tmp_path / "new" / "nested", create=True)
    assert stat.S_IMODE((tmp_path / "new").stat().st_mode) == 0o700
    assert stat.S_IMODE(created.stat().st_mode) == 0o700
    target = created / "private.bin"
    private_write(target, b"private")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_in_repo_and_symlink_paths_are_rejected(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    with pytest.raises(RecordingError, match="repository-contained"):
        external_directory(repo / "recordings", create=True)

    external_target = tmp_path / "external"
    external_target.mkdir()
    symlink = tmp_path / "alias"
    symlink.symlink_to(external_target, target_is_directory=True)
    with pytest.raises(RecordingError, match="symlink"):
        external_directory(symlink)

    repo_alias = tmp_path / "repo-alias"
    repo_alias.symlink_to(repo, target_is_directory=True)
    with pytest.raises(RecordingError, match="symlink"):
        external_directory(repo_alias / "recordings", create=True)


def test_assert_safe_bytes_finds_unicode_escaped_and_nested_credentials() -> None:
    secret = "pässword".encode("utf-8")
    escaped = json.dumps({"value": "pässword"}, ensure_ascii=True).encode("ascii")
    with pytest.raises(RecordingError):
        assert_safe_bytes(escaped, secrets=(secret,))
    with pytest.raises(RecordingError):
        assert_safe_bytes(b'{"nested":{"x-api-key":"value"}}')


def test_replay_rejects_missing_tampered_duplicate_order_and_incomplete(tmp_path: Path) -> None:
    root, recorder, underlying = _recording(
        tmp_path, lambda request: _response(_success_body())
    )
    payload = {"model": "auto"}
    recorder.post(URL, json=payload)
    underlying.close()

    response = root / "raw_responses" / "000001.bin"
    original = response.read_bytes()
    response.write_bytes(original + b"tamper")
    os.chmod(response, 0o600)
    with pytest.raises(MissingRawResponse, match="hash mismatch"):
        ReplayClient(root)
    private_write(response, original)

    request = root / "raw_requests" / "000001.bin"
    original_request = request.read_bytes()
    request.unlink()
    with pytest.raises(MissingRawResponse, match="raw request"):
        ReplayClient(root)
    private_write(request, original_request)

    line = _lines(root)[0]
    index = root / "attempts.jsonl"
    private_write(index, (json.dumps(line) + "\n" + json.dumps(line) + "\n").encode())
    with pytest.raises(MissingRawResponse, match="attempt id/order"):
        ReplayClient(root)

    incomplete = dict(line, complete=False)
    private_write(index, (json.dumps(incomplete) + "\n").encode())
    with pytest.raises(MissingRawResponse, match="incomplete"):
        ReplayClient(root)


def test_replay_rejects_orphan_blob_permissions_and_unconsumed(tmp_path: Path) -> None:
    root, recorder, underlying = _recording(
        tmp_path, lambda request: _response(_success_body())
    )
    payload = {"model": "auto"}
    recorder.post(URL, json=payload)
    underlying.close()

    replay = ReplayClient(root)
    with pytest.raises(MissingRawResponse, match="remain"):
        replay.assert_consumed()

    orphan = root / "raw_responses" / "999999.bin"
    private_write(orphan, b"orphan")
    with pytest.raises(MissingRawResponse, match="set does not match"):
        ReplayClient(root)
    orphan.unlink()

    response = root / "raw_responses" / "000001.bin"
    os.chmod(response, 0o644)
    with pytest.raises(MissingRawResponse, match="unsafe artifact permissions"):
        ReplayClient(root)
