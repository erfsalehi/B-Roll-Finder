"""DeepSeek as the preferred (paid) provider, with Groq/OpenRouter fallback."""

import json
import pytest
import core.keywords as kw


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Start every test from a known, key-free environment.
    for var in ("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_2", "DEEPSEEK_MODEL",
                "DEEPSEEK_MODEL_FAST", "DEEPSEEK_MODEL_SMART", "DEEPSEEK_REASONING",
                "DEEPSEEK_MAX_TOKENS", "GROQ_API_KEY", "GROQ_API_KEY_2"):
        monkeypatch.delenv(var, raising=False)


class _FakeResp:
    def __init__(self, content):
        self._content = content
    def raise_for_status(self):
        pass
    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_deepseek_keys_reads_env(monkeypatch):
    assert kw._deepseek_keys() == []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")
    assert kw._deepseek_keys() == ["sk-abc"]


def test_deepseek_request_uses_endpoint_model_and_json_mode(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json
        return _FakeResp('{"shots": []}')

    monkeypatch.setattr(kw.requests, "post", _fake_post)
    out = kw._call_deepseek_json("sys", "user")
    assert out == {"shots": []}
    # Routed through OpenRouter's OpenAI-compatible endpoint.
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-abc"
    # Default tier is "fast" → flash, reasoning off.
    assert captured["payload"]["model"] == "deepseek/deepseek-v4-flash"
    assert captured["payload"]["reasoning"] == {"enabled": False}
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_deepseek_fast_tier_is_flash_no_reasoning(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")
    captured = {}
    monkeypatch.setattr(kw.requests, "post",
                        lambda url, headers=None, json=None, timeout=None: captured.update(payload=json) or _FakeResp("{}"))
    kw._call_deepseek_json("sys", "user", tier="fast")
    assert captured["payload"]["model"] == "deepseek/deepseek-v4-flash"
    assert captured["payload"]["reasoning"] == {"enabled": False}


def test_deepseek_smart_tier_is_pro_with_reasoning(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")
    captured = {}
    monkeypatch.setattr(kw.requests, "post",
                        lambda url, headers=None, json=None, timeout=None: captured.update(payload=json) or _FakeResp("{}"))
    kw._call_deepseek_json("sys", "user", tier="smart")
    assert captured["payload"]["model"] == "deepseek/deepseek-v4-pro"
    assert captured["payload"]["reasoning"] == {"enabled": True}


def test_deepseek_tier_model_overrides(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")
    monkeypatch.setenv("DEEPSEEK_MODEL_FAST", "deepseek/custom-flash")
    monkeypatch.setenv("DEEPSEEK_MODEL_SMART", "deepseek/custom-pro")
    captured = {}
    monkeypatch.setattr(kw.requests, "post",
                        lambda url, headers=None, json=None, timeout=None: captured.update(payload=json) or _FakeResp("{}"))
    kw._call_deepseek_json("sys", "user", tier="fast")
    assert captured["payload"]["model"] == "deepseek/custom-flash"
    kw._call_deepseek_json("sys", "user", tier="smart")
    assert captured["payload"]["model"] == "deepseek/custom-pro"


def test_deepseek_reasoning_override_forces_off_on_smart(monkeypatch):
    # DEEPSEEK_REASONING=off overrides the smart tier's reasoning default.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")
    monkeypatch.setenv("DEEPSEEK_REASONING", "off")
    captured = {}
    monkeypatch.setattr(kw.requests, "post",
                        lambda url, headers=None, json=None, timeout=None: captured.update(payload=json) or _FakeResp("{}"))
    kw._call_deepseek_json("sys", "user", tier="smart")
    assert captured["payload"]["reasoning"] == {"enabled": False}


def test_deepseek_raises_max_tokens_floor(monkeypatch):
    # Thinking mode counts CoT against max_tokens, so a small caller cap is
    # floored up to leave room for both reasoning and the answer.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")
    captured = {}
    monkeypatch.setattr(kw.requests, "post",
                        lambda url, headers=None, json=None, timeout=None: captured.update(payload=json) or _FakeResp("{}"))
    kw._call_deepseek_json("sys", "user", max_tokens=3000)
    assert captured["payload"]["max_tokens"] >= 8000


def test_deepseek_max_tokens_floor_env_override(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")
    monkeypatch.setenv("DEEPSEEK_MAX_TOKENS", "16000")
    captured = {}
    monkeypatch.setattr(kw.requests, "post",
                        lambda url, headers=None, json=None, timeout=None: captured.update(payload=json) or _FakeResp("{}"))
    kw._call_deepseek_json("sys", "user", max_tokens=3000)
    assert captured["payload"]["max_tokens"] == 16000


def test_deepseek_keeps_larger_caller_max_tokens(monkeypatch):
    # A caller asking for more than the floor is respected as-is.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")
    captured = {}
    monkeypatch.setattr(kw.requests, "post",
                        lambda url, headers=None, json=None, timeout=None: captured.update(payload=json) or _FakeResp("{}"))
    kw._call_deepseek_json("sys", "user", max_tokens=12000)
    assert captured["payload"]["max_tokens"] == 12000


def test_deepseek_model_override(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    captured = {}
    monkeypatch.setattr(kw.requests, "post",
                        lambda url, headers=None, json=None, timeout=None: captured.update(payload=json) or _FakeResp("{}"))
    kw._call_deepseek_json("sys", "user")
    assert captured["payload"]["model"] == "deepseek-v4-flash"


def test_str_request_omits_json_mode(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")
    captured = {}
    monkeypatch.setattr(kw.requests, "post",
                        lambda url, headers=None, json=None, timeout=None: captured.update(payload=json) or _FakeResp("  hello  "))
    out = kw._call_deepseek_str("sys", "user")
    assert out == "hello"
    assert "response_format" not in captured["payload"]


def test_deepseek_retries_transient_chunked_error(monkeypatch):
    import time as _t
    from requests.exceptions import ChunkedEncodingError
    monkeypatch.setattr(_t, "sleep", lambda s: None)   # don't actually wait
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")

    calls = {"n": 0}

    def _flaky_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ChunkedEncodingError("Response ended prematurely")
        return _FakeResp('{"ok": 1}')

    monkeypatch.setattr(kw.requests, "post", _flaky_post)
    out = kw._call_deepseek_json("sys", "user")
    assert out == {"ok": 1}
    assert calls["n"] == 2   # retried the premature-end, then succeeded


def test_deepseek_retries_empty_content_then_succeeds(monkeypatch):
    # DeepSeek's "200 with empty content" quirk → json.loads("") would be the
    # "Expecting value: line 1 column 1" error. We retry instead.
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda s: None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")

    calls = {"n": 0}

    def _post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        return _FakeResp("" if calls["n"] == 1 else '{"ok": 1}')

    monkeypatch.setattr(kw.requests, "post", _post)
    out = kw._call_deepseek_json("sys", "user")
    assert out == {"ok": 1}
    assert calls["n"] == 2   # empty first response was retried


def test_deepseek_persistent_empty_content_raises(monkeypatch):
    # If every attempt is empty, give up (so the caller falls back) — and never
    # hand "" to json.loads.
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda s: None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")
    monkeypatch.setattr(kw.requests, "post",
                        lambda *a, **k: _FakeResp(""))
    with pytest.raises(Exception):
        kw._call_deepseek_json("sys", "user")


def test_deepseek_does_not_retry_auth_error(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")
    calls = {"n": 0}

    class _Resp401:
        status_code = 401
        def raise_for_status(self):
            err = kw.HTTPError("401 Unauthorized")
            err.response = self
            raise err

    def _post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        return _Resp401()

    monkeypatch.setattr(kw.requests, "post", _post)
    with pytest.raises(kw.HTTPError):
        kw._call_deepseek_json("sys", "user")
    assert calls["n"] == 1   # auth errors are not retried


def test_llm_json_prefers_deepseek_when_key_set(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setattr(kw, "_call_deepseek_json", lambda *a, **k: {"who": "deepseek"})
    monkeypatch.setattr(kw, "_call_groq_json",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Groq must not be called")))
    assert kw._call_llm_json(None, "sys", "user") == {"who": "deepseek"}


def test_llm_json_falls_back_to_groq_when_deepseek_fails(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abc")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setattr(kw, "_call_deepseek_json",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("402")))
    monkeypatch.setattr(kw, "_call_groq_json", lambda *a, **k: {"who": "groq"})
    assert kw._call_llm_json(None, "sys", "user") == {"who": "groq"}


def test_llm_json_skips_deepseek_when_no_key(monkeypatch):
    # No DeepSeek key → _call_deepseek_json must never be invoked.
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setattr(kw, "_call_deepseek_json",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("DeepSeek must not be called")))
    monkeypatch.setattr(kw, "_call_groq_json", lambda *a, **k: {"who": "groq"})
    assert kw._call_llm_json(None, "sys", "user") == {"who": "groq"}
