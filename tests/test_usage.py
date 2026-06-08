"""Per-job API usage + cost accounting."""

from core import usage
from bot import telegram_bot as tb


def test_summary_prices_tokens_and_whisper(monkeypatch):
    monkeypatch.delenv("API_PRICING_JSON", raising=False)
    monkeypatch.setenv("WHISPER_USD_PER_HOUR", "0.111")
    usage.reset()
    # 1M in + 1M out on deepseek-chat → 0.27 + 1.10 = 1.37
    usage.record_llm("deepseek", "deepseek-chat", 1_000_000, 1_000_000)
    usage.record_transcription("groq", "whisper-large-v3", 3600)  # 1 hour → 0.111

    s = usage.summary()
    assert s["priced"] is True
    assert s["total_tokens"] == 2_000_000
    assert abs(s["total_usd"] - (1.37 + 0.111)) < 1e-6
    assert s["by_provider"]["deepseek"]["prompt_tokens"] == 1_000_000
    assert abs(s["by_provider"]["groq"]["audio_seconds"] - 3600) < 1e-6


def test_openrouter_free_model_is_zero_cost(monkeypatch):
    monkeypatch.delenv("API_PRICING_JSON", raising=False)
    monkeypatch.setenv("WHISPER_USD_PER_HOUR", "0")
    usage.reset()
    # The free model name also contains 'llama-3.3-70b' — must still price to $0.
    usage.record_llm("openrouter", "meta-llama/llama-3.3-70b-instruct:free",
                     500_000, 500_000)
    s = usage.summary()
    assert s["total_usd"] == 0.0
    assert s["priced"] is False           # nothing paid → tokens-only display
    assert s["total_tokens"] == 1_000_000


def test_pricing_override_via_env(monkeypatch):
    monkeypatch.setenv("API_PRICING_JSON", '{"llama-3.3-70b": [10.0, 20.0]}')
    monkeypatch.setenv("WHISPER_USD_PER_HOUR", "0")
    usage.reset()
    usage.record_llm("groq", "llama-3.3-70b-versatile", 1_000_000, 0)  # → $10
    s = usage.summary()
    assert abs(s["total_usd"] - 10.0) < 1e-6
    assert s["priced"] is True


def test_format_cost_line_priced_and_unpriced():
    priced = {"cost": {"priced": True, "total_usd": 0.0321, "total_tokens": 48230,
                       "by_provider": {"deepseek": {"prompt_tokens": 40000,
                                                    "completion_tokens": 8230,
                                                    "audio_seconds": 0.0}}}}
    line = tb.format_cost_line(priced)
    assert line.startswith("💸 API ~$0.0321")
    assert "48.2k tokens" in line and "deepseek 48.2k" in line

    unpriced = {"cost": {"priced": False, "total_usd": 0.0, "total_tokens": 1000,
                         "by_provider": {"openrouter": {"prompt_tokens": 600,
                                                        "completion_tokens": 400,
                                                        "audio_seconds": 0.0}}}}
    line2 = tb.format_cost_line(unpriced)
    assert line2.startswith("🔢 API:") and "1k tokens" in line2

    assert tb.format_cost_line({}) is None        # nothing tracked → no line
