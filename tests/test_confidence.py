"""Footage confidence (core.confidence): the evidence a shot's picks carry, added up
into one score with reasons. The weights are starting guesses, so these tests pin
the *ordering* the score must respect rather than exact numbers."""

import core.confidence as conf


def _clip(url, source="youtube", **extra):
    return dict({"url": url, "source": source, "title": f"clip {url}"}, **extra)


def _shot(selected, ranked=None, slot_id=1, **extra):
    """A shot whose ranker order is ``ranked`` (default: the selected clips first)."""
    shot = {"slot_id": slot_id, "text": "Rinse the engine bay with a low pressure stream.",
            "priority": "medium", "selected_results": selected,
            "video_results": ranked if ranked is not None else list(selected)}
    shot.update(extra)
    return shot


def test_top_ranked_youtube_clip_is_high_confidence():
    c = _clip("a")
    r = conf.shot_confidence(_shot([c]))
    assert r["tier"] == "high" and r["score"] >= 0.65


def test_deep_ranked_pick_with_a_qa_flag_is_low():
    pool = [_clip(str(i)) for i in range(8)]
    shot = _shot([pool[6]], ranked=pool)
    clean = conf.shot_confidence(shot)
    flagged = conf.shot_confidence(shot, [{"slot_id": 1, "severity": "high"}])
    assert flagged["score"] < clean["score"]
    assert flagged["tier"] == "low"
    assert any("QA flagged" in r for r in flagged["reasons"])
    assert any("ranked #7 of 8" in r for r in clean["reasons"])


def test_rank_position_orders_the_score():
    pool = [_clip(str(i)) for i in range(6)]
    scores = [conf.shot_confidence(_shot([pool[i]], ranked=pool))["score"] for i in (0, 2, 5)]
    assert scores[0] > scores[1] > scores[2]


def test_irrelevant_candidates_do_not_count_toward_the_rank():
    pool = [_clip("bad1", irrelevant=True), _clip("bad2", irrelevant=True), _clip("good")]
    r = conf.shot_confidence(_shot([pool[2]], ranked=pool))
    assert r["tier"] == "high"          # it is the ranker's first RELEVANT candidate


def test_verified_library_clip_beats_stock():
    lib = _clip("seg", source="library", library_segment_id=7, segment_trust="verified")
    stock = _clip("p", source="pexels")
    assert conf.shot_confidence(_shot([lib]))["score"] > conf.shot_confidence(_shot([stock]))["score"]
    assert conf.shot_confidence(_shot([lib]))["score"] >= 0.9


def test_one_terrible_clip_drags_the_shot_down():
    good, bad = _clip("g"), _clip("b", sb_checked=True, sb_verdict="bad")
    both = conf.shot_confidence(_shot([good, bad], ranked=[good, bad]))["score"]
    alone = conf.shot_confidence(_shot([good]))["score"]
    mean_only = (conf.clip_fit(_shot([good, bad], ranked=[good, bad]), good)[0]
                 + conf.clip_fit(_shot([good, bad], ranked=[good, bad]), bad)[0]) / 2
    assert both < alone
    assert both < mean_only            # the worst clip counts extra, not just in the average


def test_topic_fallback_is_capped_low():
    c = _clip("p", source="pexels")
    r = conf.shot_confidence(_shot([c], _topic_fallback=True))
    assert r["score"] <= 0.30 and r["tier"] == "low"
    assert any("topic fallback" in x for x in r["reasons"])


def test_unscored_shots():
    assert conf.shot_confidence(_shot([], slot_id=2))["tier"] == "none"
    assert conf.shot_confidence(_shot([], slot_id=2))["score"] == 0.0
    assert conf.shot_confidence(_shot([_clip("a")], priority="none"))["tier"] == "skip"
    assert conf.shot_confidence(_shot([_clip("a")], is_extra=True))["score"] is None
    failed = _clip("a", _dl_failed=True)
    assert conf.shot_confidence(_shot([failed]))["tier"] == "none"


def test_repair_rounds_and_graphic_shots_cost_confidence():
    base = conf.shot_confidence(_shot([_clip("a")]))["score"]
    repaired = conf.shot_confidence(_shot([_clip("a")], tried_queries=["q1", "q2", "q3", "q4", "q5"]))
    graphic = conf.shot_confidence(_shot([_clip("a")], shot_type="graphic"))
    assert repaired["score"] < base and any("rewritten" in r for r in repaired["reasons"])
    assert graphic["score"] < base and any("graphic" in r for r in graphic["reasons"])


def test_named_subject_no_clip_mentions_is_penalised():
    shot = _shot([_clip("a", title="Foam on a car")], line_subjects=["Simple Green degreaser"])
    r = conf.shot_confidence(shot)
    assert any("names something" in x for x in r["reasons"])
    ok = _shot([_clip("a", title="Simple Green degreaser demo")], line_subjects=["Simple Green degreaser"])
    assert not any("names something" in x for x in conf.shot_confidence(ok)["reasons"])


def test_named_things_falls_back_to_capitalised_words_and_numbers():
    shot = {"text": "Use a cleaner such as Simple Green or Gunk on the 2012 Camry."}
    got = conf.named_things(shot)
    assert "Simple" in got and "Gunk" in got and "Camry" in got and "2012" in got
    assert "Use" not in got            # sentence-initial capital is not a name


def test_editor_record_moves_the_fit():
    loved = _clip("a", edit_record="editor record: this clip kept 3/3")
    hated = _clip("b", edit_record="editor record: this clip kept 0/4")
    plain = _clip("c")
    f = lambda c: conf.clip_fit(_shot([c]), c)[0]
    assert f(loved) > f(plain) > f(hated)


def test_threshold_is_tunable_and_tiers_follow_it(monkeypatch):
    shot = _shot([_clip("a")])
    score = conf.shot_confidence(shot)["score"]
    monkeypatch.setenv("CARD_CONF_THRESHOLD", str(score + 0.01))
    assert conf.shot_confidence(shot)["tier"] == "low"
    monkeypatch.setenv("CARD_CONF_THRESHOLD", "0.1")
    assert conf.shot_confidence(shot)["tier"] in ("medium", "high")


def test_annotate_confidence_tallies_and_stores():
    shots = [_shot([_clip("a")], slot_id=1), _shot([], slot_id=2),
             _shot([_clip("b")], slot_id=3, priority="none")]
    tally = conf.annotate_confidence(shots)
    assert tally["high"] == 1 and tally["none"] == 1 and tally["skip"] == 1
    assert shots[0]["confidence"]["tier"] == "high"
    assert shots[1]["confidence"]["score"] == 0.0
