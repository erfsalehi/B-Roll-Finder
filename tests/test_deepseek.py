"""DeepSeek as the preferred (paid) provider, with Groq/OpenRouter fallback."""

import json
import pytest
import core.keywords as kw


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Start every test from a known, key-free environment.
    for var in ("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_2", "DEEPSEEK_MODEL",
                "GROQ_API_KEY", "GROQ_API_KEY_2"):
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
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-abc"
    assert captured["payload"]["model"] == "deepseek-v4-pro"  # default
    assert captured["payload"]["response_format"] == {"type": "json_object"}


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
