"""Contract tests for the sidecar's OpenAI-compatible surface.

These run on any platform against `FakeEngine`; they pin the wire shape the
cachemax proxy's openai/mlxlm path depends on. The real mlx-lm engine is
exercised by `test_real_mlx.py`, which skips unless mlx-lm can run here.
"""

from fastapi.testclient import TestClient

from cache_max.engine import FakeEngine
from cache_max.server import build_app


def client() -> TestClient:
    return TestClient(build_app(FakeEngine(reply="hi there")))


def test_health():
    r = client().get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_completion_has_openai_shape():
    r = client().post(
        "/v1/chat/completions",
        json={
            "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hi there"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["prompt_tokens"] >= 1
    assert body["usage"]["completion_tokens"] >= 1
    assert (
        body["usage"]["total_tokens"]
        == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )


def test_response_carries_no_cache_truth_field():
    # mlx-lm exposes no cache truth; the proxy records no_cache_truth. The
    # sidecar must NOT fabricate a cached_tokens field.
    body = client().post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    ).json()
    assert "prompt_tokens_details" not in body["usage"]
    assert "cached_tokens" not in body["usage"]


def test_max_tokens_is_honoured_as_an_input():
    # The field is accepted (the proxy forwards OpenAI-shaped requests verbatim).
    r = client().post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
    )
    assert r.status_code == 200


def test_engine_sees_the_messages():
    # The prompt-token count is derived from message content, so a longer
    # message must produce a larger reported prompt size.
    short = client().post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "a"}]},
    ).json()
    long = client().post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "a" * 400}]},
    ).json()
    assert long["usage"]["prompt_tokens"] > short["usage"]["prompt_tokens"]
