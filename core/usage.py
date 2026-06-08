"""Per-job API usage + cost accounting.

Every LLM call funnels through ``core.keywords`` and transcription through
``core.transcription``; those modules report token counts (and audio seconds for
Whisper) here. ``run_pipeline_headless`` calls :func:`reset` at the start of a
job and :func:`summary` at the end, so each video gets a usage/cost breakdown.

Process-global (lock-guarded) rather than thread-local: ranking fans LLM calls
out across worker threads (``director_rank``), so a thread-local store would miss
them. The bot runs ONE job at a time (the ``_BUSY`` guard), so a single global
keyed to "the current job" is safe; :func:`reset` clears it between jobs.

Dollar figures are ESTIMATES from a configurable price table. OpenRouter ':free'
models always price to $0. The defaults use paid list prices for Groq/DeepSeek,
so if you're on a FREE Groq tier the estimate will overcount — set those models
to 0 (or your real rate) via ``API_PRICING_JSON`` / ``WHISPER_USD_PER_HOUR``.
"""

import json
import os
import threading

_lock = threading.Lock()
_records: list = []

# USD per 1,000,000 tokens, as (input, output). Matched to a model name by the
# longest substring key. Rough public list prices — override per your plan via
# API_PRICING_JSON, a JSON object like {"deepseek-chat": [0.27, 1.10], ...}.
_DEFAULT_PRICING = {
    ":free": (0.0, 0.0),
    "llama-3.3-70b": (0.59, 0.79),
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-chat": (0.27, 1.10),
    "deepseek": (0.27, 1.10),
}

# USD per hour of audio for Whisper transcription (Groq whisper-large-v3).
_DEFAULT_WHISPER_USD_PER_HOUR = 0.111


def reset() -> None:
    with _lock:
        _records.clear()


def record_llm(provider: str, model: str, prompt_tokens=0, completion_tokens=0) -> None:
    """Record one LLM call's token usage. Best-effort; never raises."""
    try:
        with _lock:
            _records.append({
                "kind": "llm", "provider": provider, "model": model or "",
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
            })
    except Exception:
        pass


def record_transcription(provider: str, model: str, audio_seconds=0.0) -> None:
    """Record one transcription call's audio length. Best-effort; never raises."""
    try:
        with _lock:
            _records.append({
                "kind": "asr", "provider": provider, "model": model or "",
                "audio_seconds": float(audio_seconds or 0.0),
            })
    except Exception:
        pass


def _pricing() -> dict:
    table = dict(_DEFAULT_PRICING)
    try:
        override = json.loads(os.getenv("API_PRICING_JSON", "") or "{}")
        for k, v in override.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                table[str(k)] = (float(v[0]), float(v[1]))
    except Exception:
        pass
    return table


def _whisper_rate() -> float:
    try:
        return float(os.getenv("WHISPER_USD_PER_HOUR", _DEFAULT_WHISPER_USD_PER_HOUR))
    except (TypeError, ValueError):
        return _DEFAULT_WHISPER_USD_PER_HOUR


def _match_price(model: str, table: dict):
    """Price for a model name, by longest-substring match; None if unpriced.

    ``:free`` (OpenRouter free models) always wins — a free model name also
    contains its base model (e.g. 'llama-3.3-70b-instruct:free'), so without this
    it'd be billed at the paid base rate."""
    m = (model or "").lower()
    if ":free" in m:
        return (0.0, 0.0)
    best = None
    for key, rate in table.items():
        if key == ":free":
            continue
        if key.lower() in m and (best is None or len(key) > len(best[0])):
            best = (key, rate)
    return best[1] if best else None


def summary() -> dict:
    """Aggregate the current job's usage into a cost breakdown::

        {total_usd, priced, total_tokens, by_provider: {provider: {...}}}

    ``priced`` is True when at least one recorded item had a non-zero configured
    price (so the dollar figure is meaningful rather than an all-free $0)."""
    with _lock:
        records = list(_records)
    table = _pricing()
    whisper = _whisper_rate()
    by_provider: dict = {}
    total_usd = 0.0
    priced = False
    for r in records:
        p = by_provider.setdefault(r["provider"], {
            "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "audio_seconds": 0.0, "usd": 0.0,
        })
        p["calls"] += 1
        if r["kind"] == "llm":
            p["prompt_tokens"] += r["prompt_tokens"]
            p["completion_tokens"] += r["completion_tokens"]
            rate = _match_price(r["model"], table)
            if rate is not None:
                c = (r["prompt_tokens"] / 1e6 * rate[0]
                     + r["completion_tokens"] / 1e6 * rate[1])
                p["usd"] += c
                total_usd += c
                if rate != (0.0, 0.0):
                    priced = True
        else:
            p["audio_seconds"] += r["audio_seconds"]
            if whisper:
                c = r["audio_seconds"] / 3600.0 * whisper
                p["usd"] += c
                total_usd += c
                priced = True
    for p in by_provider.values():
        p["usd"] = round(p["usd"], 4)
    total_tokens = sum(p["prompt_tokens"] + p["completion_tokens"]
                       for p in by_provider.values())
    return {
        "total_usd": round(total_usd, 4),
        "priced": priced,
        "total_tokens": total_tokens,
        "by_provider": by_provider,
    }
