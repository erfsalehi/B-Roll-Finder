"""Boundary validation gate: no clip that fails the output limits reaches the XML."""

import core.director
import core.director_youtube
import core.director_search
import core.director_rank
from core.pipeline import (
    validate_timeline, enforce_timeline, _clip_violations, PipelineState,
)


# A clean, passing clip: horizontal 1080p, short enough, has a URL.
GOOD = {"url": "http://x/a.mp4", "source": "pexels", "width": 1920, "height": 1080,
        "duration": 12.0}


def test_good_clip_has_no_violations():
    assert _clip_violations(GOOD) == []


def test_known_bad_dimensions_and_duration_flagged():
    vertical = {**GOOD, "width": 1080, "height": 1920}
    assert "vertical/portrait" in _clip_violations(vertical)

    low_res = {**GOOD, "height": 480}
    assert any("min" in r for r in _clip_violations(low_res))

    long_yt = {"url": "u", "source": "youtube", "duration": 5000}
    assert any("cap" in r for r in _clip_violations(long_yt))

    no_url = {"source": "pexels", "height": 1080, "width": 1920}
    assert "no download URL" in _clip_violations(no_url)


def test_unknown_metadata_is_tolerated():
    # Missing dimensions / duration must NOT be treated as a violation (mirrors
    # the candidate-filter policy — many direct stock MP4s carry no metadata).
    bare = {"url": "http://x/a.mp4", "source": "pexels"}
    assert _clip_violations(bare) == []


def test_validate_timeline_reports_per_clip():
    shots = [
        {"slot_id": 1, "priority": "high", "selected_results": [GOOD]},
        {"slot_id": 2, "priority": "high",
         "selected_results": [{**GOOD, "height": 360}]},   # too small
        {"slot_id": 3, "priority": "none",
         "selected_results": [{**GOOD, "height": 360}]},   # skipped → ignored
    ]
    rep = validate_timeline(shots)
    assert rep["ok"] is False
    assert rep["checked"] == 2                       # priority none not checked
    assert [f["slot_id"] for f in rep["failures"]] == [2]


def _stub_refine(monkeypatch, *, fix=True):
    """Stub the heavy refine deps. When ``fix`` is True the re-pick yields a
    clean clip; when False it yields another bad clip (forces the drop path)."""
    monkeypatch.setattr(core.director, "regenerate_shot_queries", lambda *a, **k: None)
    monkeypatch.setattr(core.director_youtube, "seed_youtube_keywords", lambda shots, **k: shots)
    monkeypatch.setattr(core.director_search, "fetch_with_retries", lambda shots, **k: None)
    monkeypatch.setattr(core.director_rank, "rank_shot_candidates", lambda shots, **k: None)
    monkeypatch.setattr(core.director_rank, "prioritize_youtube", lambda shots, **k: None)
    new_clip = GOOD if fix else {**GOOD, "height": 360}

    def _select(shots, **k):
        for s in shots:
            if not s.get("selected_results"):
                s["selected_results"] = [dict(new_clip)]
    monkeypatch.setattr(core.director_rank, "auto_select_top_candidates", _select)
    monkeypatch.setenv("AUTO_USE_LIBRARY", "false")
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)


def test_enforce_recovers_failing_shot(monkeypatch):
    _stub_refine(monkeypatch, fix=True)
    shots = [
        {"slot_id": 1, "priority": "high", "selected_results": [dict(GOOD)]},
        {"slot_id": 2, "priority": "high",
         "selected_results": [{**GOOD, "height": 360}]},
    ]
    rep = enforce_timeline(shots, groq_key="k", video_topic="t", errors=[])
    assert rep["ok"] is True
    assert rep["rounds"] == 1
    assert rep["dropped"] == 0
    assert shots[0]["selected_results"] == [GOOD]    # passing shot untouched
    assert validate_timeline(shots)["ok"] is True


def test_enforce_drops_when_unrecoverable(monkeypatch):
    _stub_refine(monkeypatch, fix=False)             # re-pick keeps failing
    shots = [
        {"slot_id": 1, "priority": "high", "selected_results": [dict(GOOD)]},
        {"slot_id": 2, "priority": "high",
         "selected_results": [{**GOOD, "height": 360}]},
    ]
    errors = []
    rep = enforce_timeline(shots, groq_key="k", video_topic="t", errors=errors,
                           max_rounds=2)
    assert rep["ok"] is True                          # ok because the bad clip was dropped
    assert rep["rounds"] == 2
    assert rep["dropped"] >= 1
    assert "selected_results" not in shots[1]         # emptied, never written to XML
    assert any("dropped clip" in e for e in errors)


def test_pipeline_state_to_result():
    st = PipelineState(project_name="p", topic="t",
                       shots=[{"selected_results": [GOOD, GOOD]},
                              {"selected_results": []}])
    st.attempts["fill"] = 1
    r = st.to_result()
    assert r["n_shots"] == 2
    assert r["n_selected"] == 1
    assert r["n_clips"] == 2
    assert r["attempts"]["fill"] == 1
    assert "validation" in r
