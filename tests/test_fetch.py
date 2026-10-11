import json

import pytest

from steam_reviewer.fetch import SteamAPIError, build_params, fetch_page, iter_pages, save_pages


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class FakeSession:
    """Returns queued responses in order and records the params of every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params)))
        return self.responses.pop(0)


def _page(ids, cursor):
    return {"success": 1, "cursor": cursor, "reviews": [{"recommendationid": i} for i in ids]}


def test_build_params_bounds():
    assert build_params()["num_per_page"] == 100
    with pytest.raises(ValueError):
        build_params(num_per_page=101)


def test_retries_on_429_then_succeeds():
    sleeps = []
    s = FakeSession([FakeResponse(429), FakeResponse(200, _page(["1"], "c1"))])
    page = fetch_page(440, session=s, sleep=sleeps.append, backoff_s=1.0)
    assert page["reviews"][0]["recommendationid"] == "1"
    assert sleeps == [1.0]
    assert s.calls[0][0].endswith("/appreviews/440")


def test_gives_up_after_retries():
    s = FakeSession([FakeResponse(503)] * 3)
    with pytest.raises(SteamAPIError):
        fetch_page(440, session=s, sleep=lambda _: None, max_retries=2)


def test_success_flag_checked():
    s = FakeSession([FakeResponse(200, {"success": 2})])
    with pytest.raises(SteamAPIError):
        fetch_page(440, session=s)


def test_iter_pages_follows_cursor_and_stops_on_repeat():
    s = FakeSession([
        FakeResponse(200, _page(["1", "2"], "c1")),
        FakeResponse(200, _page(["3"], "c2")),
        FakeResponse(200, _page(["4"], "c2")),  # cursor repeats -> stop after this page
    ])
    pages = list(iter_pages(440, max_reviews=100, session=s, sleep=lambda _: None))
    assert len(pages) == 3
    assert [c[1]["cursor"] for c in s.calls] == ["*", "c1", "c2"]


def test_iter_pages_respects_max_reviews():
    s = FakeSession([FakeResponse(200, _page(["1", "2"], "c1"))])
    pages = list(iter_pages(440, max_reviews=2, session=s, sleep=lambda _: None))
    assert len(pages) == 1 and len(s.calls) == 1


def test_save_pages(tmp_path):
    paths = save_pages(440, iter([_page(["1"], "c1")]), raw_dir=tmp_path)
    assert paths[0].name == "page_000.json"
    assert json.loads(paths[0].read_text())["reviews"][0]["recommendationid"] == "1"
