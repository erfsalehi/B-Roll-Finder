import json
import os
from typing import MutableMapping


SESSION_KEYS = (
    "slots",
    "director_shots",
    "script_text",
    "audio_duration",
    "global_themes",
    "transcription_segments",
    "transcription_chunks",
    "active_chunk_idx",
)


DEFAULTS = {
    "slots": [],
    "director_shots": [],
    "script_text": "",
    "audio_duration": 0.0,
    "global_themes": [],
    "transcription_segments": [],
    "transcription_chunks": [],
    "active_chunk_idx": 0,
}


def load_session_cache(path: str, state: MutableMapping) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in SESSION_KEYS:
        state[key] = data.get(key, DEFAULTS[key])


def save_session_cache(path: str, state: MutableMapping) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {key: state.get(key, DEFAULTS[key]) for key in SESSION_KEYS}
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)
