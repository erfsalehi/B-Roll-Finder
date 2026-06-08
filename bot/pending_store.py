"""Durable backing for the bot's review-gate state.

The Telegram bot holds projects paused at the review gate in an in-memory
``_PENDING`` dict (keyed by chat_id). That dict is lost if the process dies or
``/forcestop`` restarts the bot, stranding the user — they'd have to re-run the
whole pipeline. This module mirrors ``_PENDING`` to a JSON file under ``.cache``
so a fresh process can pick the paused projects back up and still answer
``/download`` / ``/refine``.

Everything stored in ``_PENDING`` (project name, the selected ``shots``, the QA
report, settings, the result dict) is plain JSON-friendly data, so a whole-dict
snapshot round-trips cleanly. Saving is best-effort: a persistence failure must
never take the bot down, so callers ignore exceptions.
"""

import json
import os

# Default location — alongside the other cached bot state (cookies, audio).
DEFAULT_PATH = os.path.join(".cache", "bot_pending.json")


def save_pending(pending: dict, path: str = DEFAULT_PATH) -> None:
    """Atomically write the whole ``_PENDING`` dict to ``path``. chat_id keys
    are coerced to strings for JSON; ``load_pending`` turns them back into ints.
    ``default=str`` is a safety net so a stray non-serializable value degrades to
    a string instead of raising mid-save."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {str(k): v for k, v in (pending or {}).items()}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, default=str)
    os.replace(tmp, path)


def load_pending(path: str = DEFAULT_PATH) -> dict:
    """Read the persisted review-gate state back into a ``{chat_id: entry}`` dict
    with integer chat_id keys. Returns ``{}`` if nothing was saved or the file is
    unreadable (corrupt snapshot must not block startup)."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    out = {}
    for k, v in (data or {}).items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            out[k] = v
    return out
