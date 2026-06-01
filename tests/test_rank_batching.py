"""Batched ranking: fewer LLM requests by judging several shots per call.

Covers the pure apply/validation helper and the batch dispatch (with the LLM
call mocked, so these run offline).
"""

import threading
import pytest
import core.director_rank as dr
from core.director_rank import _apply_ranked_to_shot, _rank_one_batch, rank_shot_candidates


@pytest.fixture(autouse=True)
def _no_jitter(monkeypatch):
    """Keep the suite fast — the real jitter sleep is exercised separately."""
    monkeypatch.setenv("RANK_JITTER_MAX", "0")


def _shot(slot_id, urls):
    return {
        "slot_id": slot_id, "priority": "medium", "text": f"shot {slot_id}",
        "shot_intent": "x", "video_results": [{"url": u} for u in urls],
        "selected_results": [],
    }


# ── _apply_ranked_to_shot (pure, no LLM) ──────────────────────────────────────

def test_apply_reorders_and_sets_reason():
    shot = _shot(1, ["a", "b", "c"])
    _apply_ranked_to_shot(shot, [
        {"index": 2, "reason": "best"}, {"index": 0}, {"index": 1, "irrelevant": True},
    ])
    assert [c["url"] for c in shot["video_results"]] == ["c", "a", "b"]
    assert shot["rank_reason"] == "best"
    assert shot["video_results"][2].get("irrelevant") is True
    assert "irrelevant" not in shot["video_results"][0]


def test_apply_drops_malformed_and_appends_omitted():
    shot = _shot(1, ["a", "b", "c"])
    # index 9 out of range, duplicate 0, non-int — all dropped; index 1 omitted.
    _apply_ranked_to_shot(shot, [
        {"index": 0}, {"index": 9}, {"index": 0}, {"index": "x"},
    ])
    urls = [c["url"] for c in shot["video_results"]]
    assert urls[0] == "a"               # honored pick leads
    assert set(urls) == {"a", "b", "c"}  # omitted candidates still present


def test_apply_empty_ranked_is_safe():
    shot = _shot(1, ["a"])
    _apply_ranked_to_shot(shot, [])
    assert shot["rank_reason"] == ""
    assert [c["url"] for c in shot["video_results"]] == ["a"]


# ── _rank_one_batch (LLM mocked) ──────────────────────────────────────────────

def _patch_llm(monkeypatch, response):
    monkeypatch.setattr(dr, "_call_llm_json", lambda *a, **k: response)


def test_batch_applies_per_shot_keyed_by_id(monkeypatch):
    s1, s2 = _shot(11, ["a", "b"]), _shot(12, ["c", "d"])
    _patch_llm(monkeypatch, {"shots": [
        {"shot_id": 11, "ranked": [{"index": 1, "reason": "r11"}, {"index": 0}]},
        {"shot_id": 12, "ranked": [{"index": 0, "reason": "r12"}, {"index": 1}]},
    ]})
    _rank_one_batch([s1, s2], "sys", client=None, errors=[], errors_lock=threading.Lock())
    assert [c["url"] for c in s1["video_results"]] == ["b", "a"]
    assert s1["rank_reason"] == "r11"
    assert [c["url"] for c in s2["video_results"]] == ["c", "d"]
    assert s2["rank_reason"] == "r12"
    assert "rank_error" not in s1 and "rank_error" not in s2


def test_batch_missing_shot_left_unranked_not_failed(monkeypatch):
    s1, s2 = _shot(11, ["a", "b"]), _shot(12, ["c", "d"])
    _patch_llm(monkeypatch, {"shots": [
        {"shot_id": 11, "ranked": [{"index": 1, "reason": "r11"}]},
    ]})
    _rank_one_batch([s1, s2], "sys", client=None, errors=[], errors_lock=threading.Lock())
    # s2 absent from response: original order kept, no hard error.
    assert [c["url"] for c in s2["video_results"]] == ["c", "d"]
    assert "rank_error" not in s2
    assert s2["rank_reason"] == ""


def test_batch_llm_exception_marks_all_shots(monkeypatch):
    s1, s2 = _shot(11, ["a"]), _shot(12, ["c"])
    def _boom(*a, **k):
        raise RuntimeError("429")
    monkeypatch.setattr(dr, "_call_llm_json", _boom)
    errors = []
    _rank_one_batch([s1, s2], "sys", client=None, errors=errors, errors_lock=threading.Lock())
    assert s1["rank_error"] == "429" and s2["rank_error"] == "429"
    assert errors  # surfaced for the UI


# ── rank_shot_candidates end-to-end (LLM mocked, batching observable) ──────────

def test_rank_groups_into_batches(monkeypatch):
    monkeypatch.setenv("RANK_BATCH_SIZE", "2")
    calls = []

    def _fake_llm(client, system_prompt, user_msg, **k):
        calls.append(user_msg)
        # Echo a trivial ranking for whatever shot_ids appear in this call.
        import re
        ids = [int(x) for x in re.findall(r"shot_id=(\d+)", user_msg)]
        return {"shots": [{"shot_id": i, "ranked": [{"index": 0, "reason": "ok"}]} for i in ids]}

    monkeypatch.setattr(dr, "_call_llm_json", _fake_llm)
    shots = [_shot(i, ["a", "b"]) for i in range(1, 6)]  # 5 shots
    rank_shot_candidates(shots, api_key="k", max_workers=1)
    # 5 shots / batch_size 2 → 3 LLM calls instead of 5.
    assert len(calls) == 3
    assert all(s["rank_reason"] == "ok" for s in shots)


# ── jitter + worker knobs ─────────────────────────────────────────────────────

def test_jitter_disabled_does_not_sleep(monkeypatch):
    monkeypatch.setenv("RANK_JITTER_MAX", "0")
    slept = []
    monkeypatch.setattr(dr.time, "sleep", lambda s: slept.append(s))
    dr._rank_jitter()
    assert slept == []


def test_jitter_enabled_sleeps_within_bounds(monkeypatch):
    monkeypatch.setenv("RANK_JITTER_MIN", "0.5")
    monkeypatch.setenv("RANK_JITTER_MAX", "1.5")
    slept = []
    monkeypatch.setattr(dr.time, "sleep", lambda s: slept.append(s))
    dr._rank_jitter()
    assert len(slept) == 1 and 0.5 <= slept[0] <= 1.5


def test_rank_max_workers_env_override(monkeypatch):
    monkeypatch.setenv("RANK_BATCH_SIZE", "1")
    monkeypatch.setenv("RANK_MAX_WORKERS", "2")
    captured = {}
    real_pool = dr.concurrent.futures.ThreadPoolExecutor

    def _spy_pool(max_workers=None, **k):
        captured["workers"] = max_workers
        return real_pool(max_workers=max_workers, **k)

    monkeypatch.setattr(dr.concurrent.futures, "ThreadPoolExecutor", _spy_pool)
    monkeypatch.setattr(dr, "_call_llm_json",
                        lambda *a, **k: {"shots": [{"shot_id": 1, "ranked": [{"index": 0, "reason": "ok"}]}]})
    # 5 shots, batch size 1 → 5 batches; env caps workers at 2.
    shots = [_shot(i, ["a"]) for i in range(1, 6)]
    rank_shot_candidates(shots, api_key="k", max_workers=4)
    assert captured["workers"] == 2
