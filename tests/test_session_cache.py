from core.session_cache import load_session_cache, save_session_cache


def test_session_cache_round_trips_known_keys(tmp_path):
    cache_path = tmp_path / "session_state.json"
    state = {
        "slots": [{"timestamp": 1}],
        "director_shots": [{"slot_id": 7}],
        "script_text": "hello",
        "audio_duration": 3.5,
        "global_themes": [{"theme": "shop"}],
        "transcription_segments": [{"start": 0, "end": 1, "text": "hello"}],
        "transcription_chunks": [{"start": 0, "end": 1}],
        "active_chunk_idx": 2,
        "transient_ui_key": "not persisted",
    }

    save_session_cache(str(cache_path), state)
    loaded = {}
    load_session_cache(str(cache_path), loaded)

    assert loaded["slots"] == [{"timestamp": 1}]
    assert loaded["director_shots"] == [{"slot_id": 7}]
    assert loaded["script_text"] == "hello"
    assert "transient_ui_key" not in loaded
