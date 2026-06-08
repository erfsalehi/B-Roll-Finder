"""Per-chat bot settings — what the inline /settings menu controls.

Settings persist to ``.cache/bot_settings.json`` (keyed by chat id) so they
survive bot restarts. Most map onto the pipeline's existing ``AUTO_*`` env vars;
``apply_env`` turns a chat's settings into the env overrides a job runs under
(jobs are serialised, so process-wide env is safe). A few keys (review_gate,
auto_refine, quality) are consumed by the bot/pipeline directly, not via env.
"""

import json
import os
import threading
from contextlib import contextmanager

_SETTINGS_PATH = os.path.join(".cache", "bot_settings.json")
_lock = threading.Lock()


# ── Option schema ─────────────────────────────────────────────────────────────
# Each option is rendered as one row in the inline menu. ``type`` drives the
# widget: "bool" toggles on tap; "choice" cycles through ``choices`` on tap.
# ``env`` (when present) is the pipeline env var the value maps to.
OPTIONS = [
    {"key": "use_pexels",  "label": "Pexels",        "type": "bool",   "env": "AUTO_USE_PEXELS"},
    {"key": "pexels_num",  "label": "Pexels / query","type": "choice", "env": "AUTO_PEXELS_NUM",  "choices": [1, 2, 3, 4, 5, 6, 8, 10]},
    {"key": "use_youtube", "label": "YouTube",       "type": "bool",   "env": "AUTO_USE_YOUTUBE"},
    {"key": "youtube_num", "label": "YouTube / query","type": "choice","env": "AUTO_YOUTUBE_NUM", "choices": [1, 2, 3, 4, 5, 6, 8, 10]},
    {"key": "use_library", "label": "Clip Library",  "type": "bool",   "env": "AUTO_USE_LIBRARY"},
    {"key": "library_num", "label": "Library / shot","type": "choice", "env": "AUTO_LIBRARY_NUM",  "choices": [1, 2, 3, 5, 8, 10]},
    {"key": "min_height",  "label": "Min height",    "type": "choice", "env": "AUTO_MIN_HEIGHT",   "choices": [0, 480, 720, 1080], "fmt": lambda v: "off" if not v else f"{v}p"},
    {"key": "quality",     "label": "Download quality","type": "choice","choices": [480, 720, 1080], "fmt": lambda v: f"{v}p"},
    {"key": "qa",          "label": "QA review (5.5)","type": "bool",   "env": "ENABLE_QA_REVIEW"},
    {"key": "auto_refine", "label": "Auto-refine flagged", "type": "bool"},
    {"key": "auto_fill",   "label": "Auto-fill empty shots", "type": "bool"},
    {"key": "review_gate", "label": "Pause to review", "type": "bool"},
    {"key": "overlays",    "label": "Text overlays",  "type": "bool",   "env": "ENABLE_TEXT_OVERLAYS"},
]

_OPT_BY_KEY = {o["key"]: o for o in OPTIONS}

DEFAULTS = {
    "use_pexels": True,  "pexels_num": 3,
    "use_youtube": True, "youtube_num": 4,
    "use_library": True, "library_num": 5,
    "min_height": 720,
    "quality": 1080,
    "qa": True,
    "auto_refine": True,
    "auto_fill": True,
    "review_gate": True,
    "overlays": False,
}


# ── persistence ────────────────────────────────────────────────────────────────

def _load_all() -> dict:
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_all(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_SETTINGS_PATH), exist_ok=True)
        tmp = _SETTINGS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, _SETTINGS_PATH)
    except Exception as e:
        print(f"[bot.settings] save failed: {e}")


def get_settings(chat_id) -> dict:
    """Return a chat's settings, merged over defaults (defaults fill any gaps)."""
    with _lock:
        stored = _load_all().get(str(chat_id), {})
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in stored.items() if k in DEFAULTS})
    return merged


def set_value(chat_id, key: str, value) -> dict:
    """Persist one setting and return the chat's full merged settings."""
    if key not in DEFAULTS:
        raise KeyError(key)
    with _lock:
        data = _load_all()
        chat = data.setdefault(str(chat_id), {})
        chat[key] = value
        _save_all(data)
    return get_settings(chat_id)


def reset(chat_id) -> dict:
    """Clear a chat's overrides so it falls back to DEFAULTS."""
    with _lock:
        data = _load_all()
        if str(chat_id) in data:
            del data[str(chat_id)]
            _save_all(data)
    return get_settings(chat_id)


def toggle(chat_id, key: str) -> dict:
    """Advance one option: flip a bool, or cycle a choice to the next value."""
    opt = _OPT_BY_KEY.get(key)
    if not opt:
        raise KeyError(key)
    cur = get_settings(chat_id).get(key)
    if opt["type"] == "bool":
        return set_value(chat_id, key, not bool(cur))
    choices = opt["choices"]
    try:
        nxt = choices[(choices.index(cur) + 1) % len(choices)]
    except ValueError:
        nxt = choices[0]
    return set_value(chat_id, key, nxt)


# ── display + env mapping ────────────────────────────────────────────────────

def display_value(opt: dict, value) -> str:
    """Human-readable value for the menu (✅/⬜ for bools, fmt() for choices)."""
    if opt["type"] == "bool":
        return "✅ on" if value else "⬜ off"
    fmt = opt.get("fmt")
    return fmt(value) if fmt else str(value)


def env_overrides(settings: dict) -> dict:
    """Map settings → the pipeline env vars they control (str values)."""
    out = {}
    for opt in OPTIONS:
        env = opt.get("env")
        if not env:
            continue
        v = settings.get(opt["key"])
        out[env] = ("true" if v else "false") if opt["type"] == "bool" else str(v)
    return out


@contextmanager
def apply_env(settings: dict):
    """Temporarily apply a chat's settings to os.environ for the duration of a
    job, restoring the prior environment afterwards. Safe because jobs run one
    at a time."""
    overrides = env_overrides(settings)
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
