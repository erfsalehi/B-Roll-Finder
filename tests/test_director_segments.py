"""Director output: shots named by segment numbers (no echoed script/times) and
YouTube queries written by the director instead of copied from stock queries."""

import core.director as d
from core.director_youtube import seed_youtube_keywords


def _segments(n=6):
    return [{"start": float(i * 3), "end": float(i * 3 + 2.5), "text": f"line {i + 1}"}
            for i in range(n)]


def _by_index(*ranges, **extra):
    return {"shots": [{"segments": list(r), "search_queries": ["q"], **extra} for r in ranges]}


# ── _resolve_block_shot_spans ────────────────────────────────────────────────

def test_spans_convert_to_zero_based_and_tile_the_block():
    block = _segments(6)
    assert d._resolve_block_shot_spans(block, _by_index([1, 2], [3, 3], [4, 6])["shots"]) \
        == [[0, 1], [2, 2], [3, 5]]


def test_spans_absorb_gaps_so_no_narration_is_left_uncovered():
    block = _segments(8)
    shots = _by_index([2, 3], [5, 5], [7, 7])["shots"]
    # segment 1 → first shot, 4 → the shot before it, 6 → the shot before it, 8 → last shot
    assert d._resolve_block_shot_spans(block, shots) == [[0, 3], [4, 5], [6, 7]]


def test_spans_reject_malformed_answers():
    block = _segments(4)
    bad = [
        [{"script_chunk": "x", "start": 0, "end": 1}],            # legacy shape
        [{"segments": "1-2"}],                                    # not a pair
        [{"segments": [1]}],
        [{"segments": ["a", "b"]}],
        [{"segments": [0, 2]}],                                   # 1-based: 0 is out of range
        [{"segments": [3, 5]}],                                   # past the block
        [{"segments": [3, 2]}],                                   # reversed
        _by_index([1, 3], [3, 4])["shots"],                       # overlap
        _by_index([3, 4], [1, 2])["shots"],                       # out of order
        _by_index([1, 2])["shots"] + [{"script_chunk": "x"}],     # mixed
    ]
    for shots in bad:
        assert d._resolve_block_shot_spans(block, shots) is None, shots
    assert d._resolve_block_shot_spans(block, []) is None
    assert d._resolve_block_shot_spans([], _by_index([1, 1])["shots"]) is None


def test_spans_accept_numeric_strings_and_floats():
    block = _segments(4)
    shots = [{"segments": ["1", "2"]}, {"segments": [3.0, 4.0]}]
    assert d._resolve_block_shot_spans(block, shots) == [[0, 1], [2, 3]]


# ── generate_shot_list_from_transcription ────────────────────────────────────

def _run(monkeypatch, response, segments=None):
    seen = {}

    def _fake_llm(client, system_prompt, user_msg, temperature=0.4, max_tokens=3000):
        seen["system"], seen["user"] = system_prompt, user_msg
        return response
    monkeypatch.setattr(d, "_call_llm_json", _fake_llm)
    shots = d.generate_shot_list_from_transcription(segments or _segments(6), api_key="k")
    return shots, seen


def test_shots_take_text_and_times_from_the_segments(monkeypatch):
    shots, seen = _run(monkeypatch, {"shots": [
        {"segments": [1, 2], "search_queries": ["a"], "youtube_queries": ["bmw e90 review"]},
        {"segments": [3, 6], "search_queries": ["b"], "priority": "high"},
    ]})
    assert [s["text"] for s in shots] == ["line 1 line 2", "line 3 line 4 line 5 line 6"]
    assert (shots[0]["timestamp"], shots[0]["end_timestamp"]) == (0.0, 5.5)
    assert (shots[1]["timestamp"], shots[1]["end_timestamp"]) == (6.0, 17.5)
    assert shots[0]["duration_needed_sec"] == 5.5
    assert [s["slot_id"] for s in shots] == [1, 2]
    # The model is shown numbered segments and told to answer with the numbers.
    assert "#1 [0.00 - 2.50]: line 1" in seen["user"] and "#6 [" in seen["user"]
    assert '"segments": [first, last]' in seen["system"]


def test_director_youtube_queries_become_the_shots_youtube_keywords(monkeypatch):
    shots, _ = _run(monkeypatch, {"shots": [
        {"segments": [1, 3], "search_queries": ["mechanic hands wrench engine"],
         "youtube_queries": ["  BMW E90 335i  exhaust sound ", "bmw e90 335i exhaust sound",
                             "E90 turbo replacement", "third one is dropped"]},
        {"segments": [4, 6], "search_queries": ["stock chart"]},
    ]})
    assert shots[0]["youtube_keywords"] == ["BMW E90 335i exhaust sound",
                                            "E90 turbo replacement"]
    # No youtube_queries given → key left out so seeding derives it from stock queries.
    assert "youtube_keywords" not in shots[1]
    seed_youtube_keywords(shots)
    assert shots[0]["youtube_keywords"][0] == "BMW E90 335i exhaust sound"
    assert shots[1]["youtube_keywords"] == ["stock chart"]


def test_talking_head_shot_gets_no_youtube_keywords(monkeypatch):
    shots, _ = _run(monkeypatch, {"shots": [
        {"segments": [1, 6], "priority": "none", "search_queries": [],
         "youtube_queries": ["should be ignored"]},
    ]})
    assert "youtube_keywords" not in shots[0]


def test_legacy_start_end_answer_still_works(monkeypatch):
    shots, _ = _run(monkeypatch, {"shots": [
        {"script_chunk": "line 1 line 2", "start": 0.0, "end": 5.5, "search_queries": ["a"]},
        {"script_chunk": "the rest", "start": 6.0, "end": 17.5, "search_queries": ["b"]},
    ]})
    assert [s["text"] for s in shots] == ["line 1 line 2", "the rest"]
    assert (shots[1]["timestamp"], shots[1]["end_timestamp"]) == (6.0, 17.5)


def test_bad_segment_numbers_fall_back_to_the_echoed_times(monkeypatch):
    shots, _ = _run(monkeypatch, {"shots": [
        {"segments": [1, 99], "script_chunk": "line 1 line 2", "start": 0.0, "end": 5.5,
         "search_queries": ["a"]},
        {"segments": [2, 3], "script_chunk": "rest", "start": 6.0, "end": 17.5,
         "search_queries": ["b"]},
    ]})
    assert [s["text"] for s in shots] == ["line 1 line 2", "rest"]
    assert shots[1]["timestamp"] == 6.0


def test_one_shot_can_cover_the_whole_block(monkeypatch):
    shots, _ = _run(monkeypatch, {"shots": [{"segments": [1, 6], "search_queries": ["a"]}]})
    assert len(shots) == 1
    assert shots[0]["timestamp"] == 0.0 and shots[0]["end_timestamp"] == 17.5


# ── regenerate_shot_queries ──────────────────────────────────────────────────

def _regen(monkeypatch, reply, **shot_extra):
    monkeypatch.setattr(d, "_call_llm_json", lambda *a, **k: reply)
    monkeypatch.setattr(d, "Groq", lambda api_key: None)
    monkeypatch.setattr(d, "_render_director_system_prompt", lambda t, v, c: "sys")
    shot = {"slot_id": 1, "text": "gearbox whine", "shot_type": "medium",
            "shot_intent": "gearbox", "priority": "medium",
            "search_queries": ["old"], **shot_extra}
    d.regenerate_shot_queries([shot], {1}, api_key="k")
    return shot


def test_regen_replaces_youtube_keywords_with_the_new_youtube_queries(monkeypatch):
    shot = _regen(monkeypatch, {"shots": [{"search_queries": ["new stock"],
                                           "youtube_queries": ["new yt one", "new yt two"]}]},
                  youtube_keywords=["old yt"])
    assert shot["search_queries"] == ["new stock"]
    assert shot["youtube_keywords"] == ["new yt one", "new yt two"]


def test_regen_without_youtube_queries_leaves_keywords_to_the_caller(monkeypatch):
    shot = _regen(monkeypatch, {"shots": [{"search_queries": ["new stock"]}]})
    assert "youtube_keywords" not in shot


# ── cross-block context (ranked lists) ───────────────────────────────────────

ROADMAP = {"video_global_subject": "cars that fail early", "segments": [
    {"subject": "#1 BMW X6", "start_time": 0.0, "end_time": 8.9},
    {"subject": "#2 Mercedes-Benz S500", "start_time": 9.0, "end_time": 24.0},
]}


def _run_blocks(monkeypatch, segments, replies, roadmap=None, block_size="3"):
    monkeypatch.setenv("DIRECTOR_BLOCK_SIZE", block_size)
    seen, calls = [], iter(replies)

    def _fake_llm(client, system_prompt, user_msg, temperature=0.4, max_tokens=3000):
        seen.append(user_msg)
        return next(calls)
    monkeypatch.setattr(d, "_call_llm_json", _fake_llm)
    shots = d.generate_shot_list_from_transcription(
        segments, api_key="k", video_topic="cars that fail early", segment_roadmap=roadmap)
    return shots, seen


def _car_segments():
    texts = ["Number one, the BMW X6.", "Its turbo fails early.", "Owners report rattling.",
             "Number two, the Mercedes S500.", "Its air suspension fails.", "The compressor dies first.",
             "Repairs are costly.", "Now a look at the numbers.", "Thanks for watching."]
    return [{"start": float(i * 3), "end": float(i * 3 + 2.5), "text": t}
            for i, t in enumerate(texts)]


def test_next_block_sees_the_previous_blocks_last_shots(monkeypatch):
    replies = [
        {"shots": [{"segments": [1, 3], "search_queries": ["turbo engine"],
                    "youtube_queries": ["bmw x6 turbo failure"]}]},
        {"shots": [{"segments": [1, 3], "search_queries": ["a"]}]},
        {"shots": [{"segments": [1, 3], "search_queries": ["b"]}]},
    ]
    _, seen = _run_blocks(monkeypatch, _car_segments(), replies)
    assert "PREVIOUS SHOTS" not in seen[0]
    assert "PREVIOUS SHOTS" in seen[1] and "bmw x6 turbo failure" in seen[1]
    assert "Owners report rattling." in seen[1]
    # The numbered segments still follow, restarting at #1.
    assert seen[1].index("PREVIOUS SHOTS") < seen[1].index("#1 [9.00 - 11.50]")


def test_previous_shots_skip_error_fallbacks_and_cap_at_two():
    shots = [{"text": f"shot {i}", "search_queries": [f"q{i}"]} for i in range(4)]
    shots.append({"text": "broken", "shot_intent": "Error Fallback", "search_queries": []})
    ctx = d._previous_shots_context(shots)
    assert "shot 3" in ctx and "shot 2" in ctx and "shot 1" not in ctx and "broken" not in ctx
    assert d._previous_shots_context([]) == ""


def test_lines_carry_their_own_subject_on_a_ranked_list(monkeypatch):
    replies = [{"shots": [{"segments": [1, 3], "search_queries": ["x"]}]}] * 3
    _, seen = _run_blocks(monkeypatch, _car_segments(), replies, roadmap=ROADMAP)
    # Block 2 starts at 9.0s: its first line is the S500 intro, not the X6.
    assert "#1 [9.00 - 11.50] (subject: #2 Mercedes-Benz S500): Number two" in seen[1]
    assert "(subject: #1 BMW X6): Number one" in seen[0]


def test_single_subject_roadmap_adds_no_line_tags():
    one = {"video_global_subject": "g", "segments": [
        {"subject": "cars", "start_time": 0.0, "end_time": 30.0}]}
    assert d._segment_subject_tags(one, _segments(3)) == []
    assert d._segment_subject_tags({}, _segments(3)) == []
    assert d._segment_subject_tags(ROADMAP, _segments(3)) == ["#1 BMW X6"] * 3
