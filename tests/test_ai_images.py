"""AI image generation (core.ai_images). The OpenRouter Image API is faked at the one
transport seam (_post); what matters is the contract — paid key first, a dead key is
skipped, a model that fails hands over to the next, spend is capped per video, and
an image that already exists is never paid for twice."""

import base64
import io

import pytest
from PIL import Image

import core.ai_images as ai


def _png_b64(size=(64, 48)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, (30, 90, 160)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _ok(cost=0.0147):
    return 200, {"data": [{"b64_json": _png_b64(), "media_type": "image/png"}],
                 "usage": {"cost": cost}}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_2", "OPENROUTER_API_KEY",
              "OPENROUTER_API_KEY_2", "AI_IMAGE_MODELS", "ENABLE_AI_IMAGES",
              "AI_IMAGES_MAX_PER_VIDEO", "AI_IMAGES_MAX_USD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "paid-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "free-key")
    monkeypatch.setattr(ai.time, "sleep", lambda s: None)
    ai.reset_run()
    yield
    ai.reset_run()


def _calls(monkeypatch, responder):
    log = []

    def post(url, headers, body, timeout):
        log.append({"key": headers["Authorization"].split()[-1], "model": body["model"],
                    "prompt": body["prompt"], "aspect": body["aspect_ratio"]})
        return responder(log[-1], len(log))

    monkeypatch.setattr(ai, "_post", post)
    return log


def test_generates_with_the_first_model_and_the_paid_key_first(monkeypatch, tmp_path):
    log = _calls(monkeypatch, lambda c, n: _ok())
    out = tmp_path / "a.jpg"
    r = ai.generate("a gloved hand wiping a wheel", str(out))
    assert r["model"] == "microsoft/mai-image-2.6-flash" and r["cost"] == pytest.approx(0.0147)
    assert out.exists() and Image.open(out).format == "JPEG"
    assert log[0]["key"] == "paid-key" and log[0]["aspect"] == "4:3"
    assert "no text" in log[0]["prompt"] and log[0]["prompt"].startswith("a gloved hand")
    assert ai.run_stats() == {"images": 1, "usd": 0.0147,
                              "by_model": {"microsoft/mai-image-2.6-flash": 1}, "capped": False}


def test_a_key_without_credits_is_skipped_and_remembered(monkeypatch, tmp_path):
    log = _calls(monkeypatch, lambda c, n: (402, {"error": {"message": "Insufficient credits"}})
                 if c["key"] == "paid-key" else _ok())
    r = ai.generate("x", str(tmp_path / "a.jpg"))
    assert r and [c["key"] for c in log] == ["paid-key", "free-key"]
    log.clear()
    ai.generate("y", str(tmp_path / "b.jpg"))
    assert [c["key"] for c in log] == ["free-key"]       # the dead key is not retried this run


def test_a_failing_model_hands_over_to_the_next(monkeypatch, tmp_path):
    def responder(c, n):
        if c["model"] == "microsoft/mai-image-2.6-flash":
            return 400, {"error": {"message": "prompt rejected"}}
        return _ok(0.007)
    log = _calls(monkeypatch, responder)
    r = ai.generate("x", str(tmp_path / "a.jpg"))
    assert r["model"] == "recraft/recraft-v4.1-flash"
    assert [c["model"] for c in log][:2] == ["microsoft/mai-image-2.6-flash",
                                              "recraft/recraft-v4.1-flash"]


def test_a_hanging_model_is_abandoned(monkeypatch, tmp_path):
    def responder(c, n):
        if n == 1:
            raise TimeoutError("read timed out")
        return _ok()
    _calls(monkeypatch, responder)
    r = ai.generate("x", str(tmp_path / "a.jpg"))
    assert r and r["model"] == "recraft/recraft-v4.1-flash"


def test_rate_limit_is_retried_once(monkeypatch, tmp_path):
    log = _calls(monkeypatch, lambda c, n: (429, None) if n == 1 else _ok())
    r = ai.generate("x", str(tmp_path / "a.jpg"))
    assert r["model"] == "microsoft/mai-image-2.6-flash" and len(log) == 2


def test_everything_failing_reports_why(monkeypatch, tmp_path):
    _calls(monkeypatch, lambda c, n: (500, None))
    errors = []
    assert ai.generate("x", str(tmp_path / "a.jpg"), errors=errors) is None
    assert errors and "generation failed" in errors[0]


def test_budget_caps_images_and_says_so_once(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_IMAGES_MAX_PER_VIDEO", "2")
    log = _calls(monkeypatch, lambda c, n: _ok())
    errors = []
    got = [ai.generate("x", str(tmp_path / f"{i}.jpg"), errors=errors) for i in range(4)]
    assert [bool(g) for g in got] == [True, True, False, False]
    assert len(log) == 2
    assert sum("budget reached" in e for e in errors) == 1
    assert not ai.budget_left()


def test_dollar_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_IMAGES_MAX_USD", "0.02")
    _calls(monkeypatch, lambda c, n: _ok(0.015))
    assert ai.generate("a", str(tmp_path / "a.jpg"))
    assert ai.generate("b", str(tmp_path / "b.jpg"))      # 0.015 spent < 0.02: still allowed
    assert ai.generate("c", str(tmp_path / "c.jpg")) is None   # 0.030 >= 0.02


def test_existing_file_is_reused_without_a_call(monkeypatch, tmp_path):
    out = tmp_path / "a.jpg"
    Image.new("RGB", (40, 30)).save(out, "JPEG")
    log = _calls(monkeypatch, lambda c, n: _ok())
    r = ai.generate("x", str(out))
    assert r["cached"] and r["cost"] == 0.0 and r["width"] == 40 and not log


def test_switched_off_or_keyless(monkeypatch, tmp_path):
    log = _calls(monkeypatch, lambda c, n: _ok())
    monkeypatch.setenv("ENABLE_AI_IMAGES", "false")
    assert not ai.enabled() and ai.generate("x", str(tmp_path / "a.jpg")) is None
    monkeypatch.delenv("ENABLE_AI_IMAGES")
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert not ai.enabled() and ai.generate("x", str(tmp_path / "b.jpg")) is None
    assert not log


def test_models_can_be_overridden_and_prompts_get_a_style():
    assert ai.models()[0] == "microsoft/mai-image-2.6-flash"
    assert "illustration" in ai.build_prompt("a generator", "illustration")
    assert ai.build_prompt("q" * 2000, "photo").count("q") == 700
    import os
    os.environ["AI_IMAGE_MODELS"] = "a/b, c/d"
    try:
        assert ai.models() == ["a/b", "c/d"]
    finally:
        del os.environ["AI_IMAGE_MODELS"]
