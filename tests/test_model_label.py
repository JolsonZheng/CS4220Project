"""Offline tests for the optional model labeler (fake HTTP session, no network, no key)."""

import json

import pandas as pd
import pytest

from steam_reviewer.model_label import ChatCompletion, ModelAPIError, label_with_model, run_summary


class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.responses.pop(0)


def _reply(content, prompt_tokens=100, completion_tokens=20):
    return FakeResponse(200, {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    })


REVIEWS = pd.DataFrame({
    "recommendation_id": ["1", "2"],
    "review_text": ["great maps", "servers are dead"],
})

GOOD = json.dumps([
    {"id": "1", "sentiment": "positive", "praise": ["gameplay"], "complaints": [], "summary": "maps"},
    {"id": "2", "sentiment": "negative", "praise": [], "complaints": ["multiplayer"], "summary": "dead"},
])


def test_chat_completion_request_shape_and_auth():
    session = FakeSession([_reply("hi")])
    complete = ChatCompletion(model="m", base_url="http://local/v1/", api_key="k", session=session)
    assert complete("prompt") == "hi"
    call = session.calls[0]
    assert call["url"] == "http://local/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer k"
    assert call["json"]["model"] == "m" and call["json"]["temperature"] == 0.0
    assert complete.log[0]["prompt_tokens"] == 100


def test_no_auth_header_without_key():
    session = FakeSession([_reply("hi")])
    ChatCompletion(session=session)("p")
    assert "Authorization" not in session.calls[0]["headers"]


def test_retries_on_429_then_succeeds():
    waits = []
    session = FakeSession([FakeResponse(429), FakeResponse(503), _reply("ok")])
    complete = ChatCompletion(session=session, sleep=waits.append, backoff_s=1.0)
    assert complete("p") == "ok"
    assert waits == [1.0, 2.0]
    assert [e["status"] for e in complete.log] == [429, 503, 200]


def test_gives_up_and_raises():
    session = FakeSession([FakeResponse(429)] * 3)
    complete = ChatCompletion(session=session, sleep=lambda _: None, max_retries=2)
    with pytest.raises(ModelAPIError):
        complete("p")


def test_client_error_raises_immediately():
    session = FakeSession([FakeResponse(401, {"error": "bad key"})])
    with pytest.raises(ModelAPIError):
        ChatCompletion(session=session)("p")


def test_label_with_model_and_summary():
    session = FakeSession([_reply(GOOD)])
    complete = ChatCompletion(model="m", session=session)
    labels = label_with_model(REVIEWS, complete, "m", batch_size=10)
    assert list(labels["sentiment"]) == ["positive", "negative"]
    assert set(labels["model"]) == {"m"}
    summary = run_summary(labels, complete.log, "m")
    assert summary["n_label_errors"] == 0
    assert summary["status_counts"] == {"200": 1}
    assert summary["prompt_tokens"] == 100 and summary["completion_tokens"] == 20


def test_api_failure_becomes_counted_label_error():
    session = FakeSession([FakeResponse(500)] * 8)
    complete = ChatCompletion(model="m", session=session, sleep=lambda _: None, max_retries=1)
    labels = label_with_model(REVIEWS, complete, "m", batch_size=10)
    assert labels["sentiment"].isna().all()
    assert labels["label_error"].str.startswith("api:").all()
    assert run_summary(labels, complete.log, "m")["status_counts"] == {"500": 4}
