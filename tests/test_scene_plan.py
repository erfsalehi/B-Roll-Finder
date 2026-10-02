"""The scene plan (core.scene_plan): which moments get a card and what it says.
The LLM is faked at core.keywords._call_llm_json; the contract under test is the
user's rule — a card on every list STEP and on LOW-CONFIDENCE footage, the right
half never empty, a number only for a real step, a figure only if the narration
states it, non-overlapping windows, and re-planning that does not redo finished work."""

import pytest

import core.keywords as kw
import core.scene_plan as sp


def _clip(url, source="youtube"):
    return {"url": url, "source": source, "title": f"clip {url}"}


def _shot(slot, text, start, end, *, bad=False, **extra):
    """A timeline shot. ``bad`` gives it a deep-ranked pick that also needed several query
    rewrites and that the director called a graphic — one weak signal alone stays
    'medium' by design, so low confidence takes corroborating evidence."""
    pool = [_clip(f"{slot}-{i}") for i in range(8)]
    pick = pool[6] if bad else pool[0]
    s = {"slot_id": slot, "text": text, "timestamp": start, "end_timestamp": end,
         "duration_needed_sec": end - start, "priority": "medium", "shot_type": "medium",
         "shot_intent": text[:40], "selected_results": [pick], "video_results": pool}
    if bad:
        s.update(tried_queries=["a", "b", "c", "d", "e"], shot_type="graphic")
    s.update(extra)
    return s


class FakeLLM:
    """Answers the two prompts this module sends."""

    def __init__(self, steps=None, cards=None, fail_steps=False):
        self.steps, self.cards, self.fail_steps = steps or [], cards or {}, fail_steps
        self.design_calls = []

    def __call__(self, client, system, user, **kw):
        if "LIST STRUCTURE" in system:
            if self.fail_steps:
                raise RuntimeError("llm down")
            return {"items": self.steps}
        slots = [int(x) for x in __import__("re").findall(r"slot_id (\d+)", user)]
        self.design_calls.append(slots)
        out = []
        for sid in slots:
            c = {"slot_id": sid, "label": f"label {sid}", "right": {"kind": "image"},
                 "image_query": f"query {sid}", "image_prompt": f"prompt {sid}",
                 "subject_kind": "generic", "image_style": "photo"}
            c.update(self.cards.get(sid, {}))
            out.append(c)
        return {"cards": out}


@pytest.fixture
def llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(kw, "_call_llm_json", fake)
    monkeypatch.delenv("MOTION_CARDS_MODE", raising=False)
    monkeypatch.delenv("ENABLE_MOTION_CARDS", raising=False)
    monkeypatch.delenv("CARD_MAX_FRACTION", raising=False)
    monkeypatch.delenv("CARD_MIN_SEC", raising=False)
    sp._STEP_CACHE.clear()
    return fake


def _video():
    return [
        _shot(1, "Most owners ignore the engine bay.", 0.0, 4.0),
        _shot(2, "Method 1. Low pressure foam and a soft brush.", 4.0, 12.0),
        _shot(3, "Spray, wait two minutes, then brush.", 12.0, 17.0),
        _shot(4, "Dirt and highway dust build up.", 17.0, 22.0, bad=True),
        _shot(5, "Method 2 is a waterless spray and wipe.", 22.0, 30.0),
        _shot(6, "Wipe with a clean towel.", 30.0, 36.0),
        _shot(7, "That is it.", 36.0, 40.0),
        _shot(8, "Thanks for watching.", 40.0, 44.0),
    ]


# ── steps ────────────────────────────────────────────────────────────────────

def test_regex_steps_numbers_each_kind_of_list_on_its_own():
    shots = [_shot(1, "Method 1. Foam.", 0, 5), _shot(2, "Method 2 — Waterless.", 5, 10),
             _shot(3, "Safety step one is verifying the engine is cool.", 10, 15),
             _shot(4, "Safety Step 2 requires addressing the battery.", 15, 20),
             _shot(5, "Looking back at step two we saw it.", 20, 25),
             _shot(6, "Number three on our list.", 25, 30)]
    got = [(x["slot_id"], x["kind"], x["number"]) for x in sp.regex_steps(shots)]
    assert got == [(1, "METHOD", 1), (2, "METHOD", 2), (3, "STEP", 1), (4, "STEP", 2), (6, "STEP", 3)]


def test_detect_steps_uses_the_llm_and_validates_it(llm):
    llm.steps = [{"slot_id": 2, "number": 1, "kind": "method", "label": "Foam and brush"},
                 {"slot_id": 5, "number": "2", "kind": "METHOD", "label": "Waterless wipe"},
                 {"slot_id": 99, "number": 3, "kind": "METHOD", "label": "ghost"},      # no such shot
                 {"slot_id": 6, "number": "many", "kind": "METHOD", "label": "bad number"}]
    got = sp.detect_steps(_video(), None)
    assert [(x["slot_id"], x["number"], x["kind"]) for x in got] == [(2, 1, "METHOD"), (5, 2, "METHOD")]
    assert got[0]["label"] == "Foam and brush"


def test_detect_steps_falls_back_to_wording_when_the_llm_fails(llm):
    llm.fail_steps = True
    errors = []
    got = sp.detect_steps(_video(), None, errors)
    assert [x["slot_id"] for x in got] == [2, 5]
    assert errors and "step detection failed" in errors[0]


def test_talking_head_and_extra_shots_are_never_steps(llm):
    shots = _video()
    shots[1]["priority"] = "none"
    llm.steps = [{"slot_id": 2, "number": 1, "kind": "METHOD"}, {"slot_id": 5, "number": 2, "kind": "METHOD"}]
    assert [x["slot_id"] for x in sp.detect_steps(shots, None)] == [5]


# ── the plan ─────────────────────────────────────────────────────────────────

def test_plan_puts_cards_on_steps_and_low_confidence_footage(llm):
    llm.steps = [{"slot_id": 2, "number": 1, "kind": "METHOD", "label": "Foam and brush"},
                 {"slot_id": 5, "number": 2, "kind": "METHOD", "label": "Waterless wipe"}]
    diag = {}
    beats = sp.plan_scene(_video(), "engine bay cleaning", None, diag=diag)
    by_slot = {b["slot_ids"][0]: b for b in beats}
    assert set(by_slot) == {2, 4, 5}
    assert by_slot[2]["trigger"] == "step" and by_slot[4]["trigger"] == "low_confidence"
    # step card: first 4 s of the shot; low-confidence card: the whole shot
    assert (by_slot[2]["start"], by_slot[2]["end"]) == (4.0, 8.0)
    assert (by_slot[4]["start"], by_slot[4]["end"]) == (17.0, 21.85)   # stops 0.15 s before the next card
    assert by_slot[2]["right"] == {"kind": "number", "text": "1", "sub": "METHOD"} or \
        by_slot[2]["right"]["kind"] in ("number", "image")
    assert diag["steps"] == 2 and diag["confidence"]["low"] >= 1
    assert all(b["template"] == "split_card" and b["enabled"] for b in beats)


def test_right_panel_is_never_empty_and_follows_the_rules(llm):
    llm.steps = [{"slot_id": 2, "number": 1, "kind": "METHOD", "label": "Foam"},
                 {"slot_id": 5, "number": 2, "kind": "METHOD", "label": "Wipe"}]
    shots = _video()
    shots[3]["text"] = "Dust adds 47% more wear over time."
    llm.cards = {
        2: {"right": {"kind": "number"}},
        5: {"right": {"kind": "image"}},                                  # a step may choose an image
        4: {"right": {"kind": "stat", "text": "47%", "sub": "more wear"}},
    }
    by = {b["slot_ids"][0]: b for b in sp.plan_scene(shots, "t", None)}
    assert by[2]["right"] == {"kind": "number", "text": "1", "sub": "METHOD"}
    assert by[5]["right"] == {"kind": "image"}
    assert by[4]["right"] == {"kind": "stat", "text": "47%", "sub": "MORE WEAR"}


def test_invented_figures_and_numbers_on_non_steps_become_images(llm):
    shots = _video()
    llm.cards = {4: {"right": {"kind": "stat", "text": "90%", "sub": "x"}}}      # 90 is not in the line
    by = {b["slot_ids"][0]: b for b in sp.plan_scene(shots, "t", None)}
    assert by[4]["right"] == {"kind": "image"}
    llm.cards = {4: {"right": {"kind": "number"}}}                                # not a step
    by = {b["slot_ids"][0]: b for b in sp.plan_scene(shots, "t", None)}
    assert by[4]["right"] == {"kind": "image"}
    llm.cards = {4: {"right": {}}}                                                # garbage
    assert {b["slot_ids"][0]: b for b in sp.plan_scene(shots, "t", None)}[4]["right"] == {"kind": "image"}


def test_labels_are_cleaned(llm):
    llm.cards = {4: {"label": "  the very, long & rambling title of this card!! "}}
    b = {x["slot_ids"][0]: x for x in sp.plan_scene(_video(), "t", None)}[4]
    assert b["label"] == b["label"].upper() and len(b["label"]) <= 24 and not b["label"].endswith("!")


def test_a_label_never_ends_on_a_dangling_word():
    # the live run cut "PROTECT ELECTRICAL AND ELECTRONIC …" off mid-phrase
    assert sp._clean_label("Protect electrical and electronic components") == "PROTECT ELECTRICAL"
    assert sp._clean_label("Low-pressure foam and soft brush") == "LOW-PRESSURE FOAM"
    assert sp._clean_label("Personal protective equipment") == "PERSONAL PROTECTIVE"    # 24-char cut, no stop-word
    assert sp._clean_label("Waterless spray and wipe") == "WATERLESS SPRAY AND WIPE"    # fits whole: untouched
    assert sp._clean_label("Use respirator if") == "USE RESPIRATOR"
    assert sp._clean_label("Save $300") == "SAVE $300" and sp._clean_label("Under 90°C!") == "UNDER 90°C"
    assert sp._clean_label("Save $2,000 on AC") == "SAVE $2,000 ON AC"                   # the bot's output had "$2 000"
    assert sp._clean_label("Rated 4.5 stars, wow.") == "RATED 4.5 STARS WOW"
    assert sp._clean_label("The") == "THE"                                              # a lone word is kept
    assert sp._clean_label("", "FALLBACK") == "FALLBACK"


def test_a_named_product_the_designer_called_generic_is_never_generated(llm):
    shots = _video()
    shots[3]["text"] = "I cleaned it with Simple Green or Gunk degreaser."
    llm.cards = {4: {"image_query": "Simple Green Gunk degreaser bottle", "subject_kind": "generic"}}
    b = {x["slot_ids"][0]: x for x in sp.plan_scene(shots, "t", None)}[4]
    assert b["subject_kind"] == "identifiable"
    assert any("real picture only" in r for r in b["reasons"])


def test_a_bare_figure_or_unrelated_query_does_not_force_identifiable(llm):
    shots = _video()
    shots[3]["text"] = "The shop wants $300 for it, around 2012 prices."
    llm.cards = {4: {"image_query": "300 dollar bills 2012", "subject_kind": "generic"}}
    assert {x["slot_ids"][0]: x for x in sp.plan_scene(shots, "t", None)}[4]["subject_kind"] == "generic"
    shots[3]["text"] = "I cleaned it with Simple Green."
    llm.cards = {4: {"image_query": "foam sprayer on engine", "subject_kind": "generic"}}   # names nothing
    assert {x["slot_ids"][0]: x for x in sp.plan_scene(shots, "t", None)}[4]["subject_kind"] == "generic"


def test_a_low_confidence_card_is_held_for_three_seconds(llm, monkeypatch):
    shots = [_shot(1, "A short bad shot.", 0.0, 2.4, bad=True), _shot(2, "Then more narration.", 2.4, 12.0),
             _shot(3, "And more.", 12.0, 20.0)]
    b = {x["slot_ids"][0]: x for x in sp.plan_scene(shots, "t", None)}[1]
    assert b["end"] - b["start"] >= 3.0 - 1e-6                  # was 2.4: all reveal, no read time
    monkeypatch.setenv("CARD_MIN_SEC", "2.4")
    b = {x["slot_ids"][0]: x for x in sp.plan_scene(shots, "t", None)}[1]
    assert abs((b["end"] - b["start"]) - 2.4) < 1e-6


def test_step_detection_is_remembered_per_narration(llm, monkeypatch):
    llm.steps = [{"slot_id": 2, "number": 1, "kind": "METHOD", "label": "Foam and brush"}]
    calls = []

    def counting(client, system, user, **kw):
        if "LIST STRUCTURE" in system:
            calls.append(1)
        return llm(client, system, user, **kw)
    monkeypatch.setattr(kw, "_call_llm_json", counting)
    first = sp.detect_steps(_video(), None)
    llm.steps = [{"slot_id": 5, "number": 2, "kind": "METHOD", "label": "drifted"}]
    again = sp.detect_steps(_video(), None)
    assert len(calls) == 1 and again == first                     # no second call, no label drift
    edited = _video()
    edited[0]["text"] = "A different script."
    sp.detect_steps(edited, None)
    assert len(calls) == 2                                        # another narration is asked afresh


def test_a_failed_step_call_is_not_remembered(llm):
    llm.fail_steps = True
    assert [x["slot_id"] for x in sp.detect_steps(_video(), None, [])] == [2, 5]      # wording fallback
    llm.fail_steps = False
    llm.steps = [{"slot_id": 2, "number": 1, "kind": "METHOD", "label": "Foam"}]
    assert [x["slot_id"] for x in sp.detect_steps(_video(), None)] == [2]             # asked again, now works


def test_modes(llm, monkeypatch):
    llm.steps = [{"slot_id": 2, "number": 1, "kind": "METHOD"}]
    monkeypatch.setenv("MOTION_CARDS_MODE", "steps")
    assert {b["trigger"] for b in sp.plan_scene(_video(), "t", None)} == {"step"}
    monkeypatch.setenv("MOTION_CARDS_MODE", "off")
    assert sp.plan_scene(_video(), "t", None) == []
    monkeypatch.delenv("MOTION_CARDS_MODE")
    monkeypatch.setenv("ENABLE_MOTION_CARDS", "false")
    assert sp.plan_scene(_video(), "t", None) == []


def test_low_confidence_cards_are_capped(llm, monkeypatch):
    shots = [_shot(i, f"Line {i} about something.", i * 5.0, i * 5.0 + 5.0, bad=True) for i in range(1, 11)]
    monkeypatch.setenv("CARD_MAX_FRACTION", "0.3")
    diag = {}
    beats = sp.plan_scene(shots, "t", None, diag=diag)
    assert len(beats) == 3 and diag["low_confidence"] == 10 and diag["low_capped"] == 7


def test_shots_with_no_footage_get_a_card_too(llm):
    shots = _video()
    shots[5]["selected_results"] = []
    b = {x["slot_ids"][0]: x for x in sp.plan_scene(shots, "t", None)}[6]
    assert b["trigger"] == "empty" and b["confidence"]["footage"] == 0.0


def test_talking_head_shots_never_get_a_card(llm):
    shots = _video()
    shots[3]["priority"] = "none"
    assert 4 not in {b["slot_ids"][0] for b in sp.plan_scene(shots, "t", None)}


def test_a_step_cuts_a_colliding_low_confidence_card_short(llm):
    llm.steps = [{"slot_id": 2, "number": 1, "kind": "STEP"}]
    shots = [_shot(1, "Intro line.", 0, 3, bad=True),
             _shot(2, "Step one is to rinse.", 3.0, 9.0)]
    beats = sp.plan_scene(shots, "t", None)
    first, second = beats
    assert first["slot_ids"] == [1] and second["slot_ids"] == [2]
    assert first["end"] <= second["start"] - 0.15 + 1e-6       # no overlap on the track


def test_replanning_keeps_finished_work_and_studio_edits(llm):
    shots = _video()
    first = sp.plan_scene(shots, "t", None)
    for b in first:
        b["label"] = "CUSTOM"
        b["assets"] = {"left": {"kind": "google", "path": "x.jpg"}}
    manual = {"id": "bx", "template": "split_card", "trigger": "step", "slot_ids": [7], "start": 36.0,
              "end": 40.0, "enabled": True, "label": "MINE", "right": {"kind": "image"},
              "edited": True, "assets": {}, "confidence": {}, "reasons": []}
    llm.design_calls.clear()
    again = sp.plan_scene(shots, "t", None, existing=first + [manual])
    assert llm.design_calls == []                                    # nothing redesigned
    kept = {b["slot_ids"][0]: b for b in again}
    assert all(kept[s]["label"] == "CUSTOM" and kept[s]["assets"]["left"]["path"] == "x.jpg"
               for s in set(kept) - {7})
    assert kept[7]["label"] == "MINE"
    # a changed trigger is redesigned
    shots[3]["selected_results"] = [shots[3]["video_results"][0]]       # slot 4 is fine now
    again = sp.plan_scene(shots, "t", None, existing=first)
    assert 4 not in {b["slot_ids"][0] for b in again}


def test_design_failure_still_yields_usable_cards(llm, monkeypatch):
    def boom(client, system, user, **kw):
        if "LIST STRUCTURE" in system:
            return {"items": []}
        raise RuntimeError("design down")
    monkeypatch.setattr(kw, "_call_llm_json", boom)
    errors = []
    beats = sp.plan_scene(_video(), "t", None, errors=errors)
    assert beats and all(b["label"] and b["right"]["kind"] for b in beats)
    assert any("design call failed" in e for e in errors)


def test_resolve_overlaps_drops_what_cannot_fit():
    mk = lambda i, a, b, t="low_confidence": {"id": i, "trigger": t, "start": a, "end": b}
    got = sp.resolve_overlaps([mk("a", 0, 5), mk("b", 4, 6), mk("c", 5.5, 9)])
    assert [x["id"] for x in got] == ["a", "c"]             # b would have <2.4s left → dropped
    assert got[1]["start"] >= got[0]["end"]
    got = sp.resolve_overlaps([mk("a", 0, 6), mk("s", 3, 7, "step")])
    assert got[0]["id"] == "a" and got[0]["end"] <= 3 - 0.15 + 1e-6 and got[1]["id"] == "s"
    got = sp.resolve_overlaps([mk("a", 0, 3), mk("s", 1, 5, "step")])
    assert [x["id"] for x in got] == ["s"]                    # too little of "a" survives


def test_summary_counts():
    beats = [{"trigger": "step", "enabled": True}, {"trigger": "low_confidence", "enabled": True,
             "assets": {"left": {"kind": "ai"}}}, {"trigger": "empty", "enabled": True},
             {"trigger": "step", "enabled": False}]
    assert sp.summary(beats) == {"total": 3, "steps": 1, "low": 1, "empty": 1, "ai": 1}
