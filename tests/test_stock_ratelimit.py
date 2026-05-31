"""Rate-limit handling for the stock APIs (Pexels / Pixabay 429 management)."""

import time
from unittest import mock

import pytest

import core.stock_apis as s


@pytest.fixture(autouse=True)
def _reset_gates():
    s._PEXELS_GATE.until = 0.0
    s._PIXABAY_GATE.until = 0.0
    yield
    s._PEXELS_GATE.until = 0.0
    s._PIXABAY_GATE.until = 0.0


class _Resp:
    def __init__(self, status, headers=None, body=None):
        self.status_code = status
        self.ok = status < 400
        self.headers = headers or {}
        self._body = body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if not self.ok:
            raise Exception(f"{self.status_code} error")


def test_429_trips_gate_and_blocks_siblings():
    """One 429 trips the breaker; further queries skip without hitting network."""
    reset = str(int(time.time()) + 3600)
    calls = {"n": 0}

    def fake_get(url, headers=None, params=None, timeout=None):
        calls["n"] += 1
        return _Resp(429, {"X-Ratelimit-Reset": reset})

    with mock.patch("core.stock_apis.requests.get", fake_get), \
         mock.patch("core.stock_apis.time.sleep", lambda *_: None):
        errs = []
        assert s.search_pexels("q1", "key", 3, errors=errs) == []
        for i in range(9):
            assert s.search_pexels(f"q{i}", "key", 3, errors=errs) == []

    assert calls["n"] == 1                  # only the first query actually called out
    assert len(errs) == 10                  # every query reported something
    assert len(set(errs)) == 1              # ...but all identical → de-dupes to one line
    assert "rate limit reached" in errs[0]
    assert s._PEXELS_GATE.blocked_for() > 3000


def test_remaining_zero_trips_gate_proactively():
    """A good response with X-Ratelimit-Remaining: 0 trips the gate so the
    *next* query is skipped before it can 429."""
    reset = str(int(time.time()) + 1800)
    good = _Resp(200, {"X-Ratelimit-Remaining": "0", "X-Ratelimit-Reset": reset},
                 {"videos": []})

    with mock.patch("core.stock_apis.requests.get", lambda *a, **k: good):
        errs = []
        s.search_pexels("first", "key", 3, errors=errs)  # succeeds, but quota now 0
    assert s._PEXELS_GATE.blocked_for() > 1000
    assert errs == []  # the successful call itself reported no error


def test_successful_response_does_not_trip_gate():
    good = _Resp(200, {"X-Ratelimit-Remaining": "150"},
                 {"videos": []})
    with mock.patch("core.stock_apis.requests.get", lambda *a, **k: good):
        errs = []
        s.search_pexels("ok", "key", 3, errors=errs)
    assert s._PEXELS_GATE.blocked_for() == 0
    assert errs == []


def test_parse_reset_prefers_retry_after_then_epoch():
    assert s._parse_reset_seconds({"Retry-After": "30"}) == pytest.approx(30, abs=1)
    epoch = str(int(time.time()) + 120)
    assert s._parse_reset_seconds({"X-Ratelimit-Reset": epoch}) == pytest.approx(120, abs=2)
    assert s._parse_reset_seconds({}) is None


def test_pixabay_has_independent_gate():
    """Tripping Pexels must not block Pixabay (separate quotas)."""
    s._PEXELS_GATE.trip_for(3600)
    assert s._PEXELS_GATE.blocked_for() > 0
    assert s._PIXABAY_GATE.blocked_for() == 0
