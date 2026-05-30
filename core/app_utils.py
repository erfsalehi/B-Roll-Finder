import json
import os
import socket
import subprocess
import sys
import threading
from datetime import date, datetime


# ── yt-dlp auto-update ───────────────────────────────────────────────────────
# YouTube changes its player/format pipeline constantly; a yt-dlp that's a few
# weeks stale starts failing downloads ("Requested format is not available",
# signature errors, etc.). We upgrade yt-dlp via pip at most once per day so
# the app stays current without hammering PyPI on every launch.
_YT_DLP_UPDATE_MARKER = os.path.join(".cache", "yt_dlp_update.json")


def _read_update_marker() -> dict:
    try:
        with open(_YT_DLP_UPDATE_MARKER, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_update_marker(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_YT_DLP_UPDATE_MARKER), exist_ok=True)
        with open(_YT_DLP_UPDATE_MARKER, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _installed_yt_dlp_version() -> str:
    """Version pip currently has on disk (may differ from the imported module
    after an in-process upgrade)."""
    try:
        from importlib.metadata import version
        return version("yt-dlp")
    except Exception:
        try:
            import yt_dlp
            return getattr(yt_dlp.version, "__version__", "") or ""
        except Exception:
            return ""


def update_yt_dlp(force: bool = False, timeout: int = 180) -> dict:
    """Upgrade yt-dlp via pip, at most once per calendar day.

    Returns a dict: {ran, old, new, changed, error}. ``changed`` is True when a
    newer version was actually installed. The new version only takes effect
    after the app restarts, since yt_dlp is already imported into this process.
    """
    today = date.today().isoformat()
    marker = _read_update_marker()
    old = _installed_yt_dlp_version()

    if not force and marker.get("last_check") == today:
        return {"ran": False, "old": old, "new": old, "changed": False, "error": ""}

    result = {"ran": True, "old": old, "new": old, "changed": False, "error": ""}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade",
             "--disable-pip-version-check", "yt-dlp"],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "pip failed").strip().splitlines()
            result["error"] = (tail[-1] if tail else "pip failed")[:200]
    except Exception as e:
        result["error"] = str(e)[:200]

    result["new"] = _installed_yt_dlp_version()
    result["changed"] = bool(result["new"] and result["new"] != old)
    # Record the check even on failure so a broken network doesn't make us
    # retry the slow pip call on every single rerun.
    _write_update_marker({
        "last_check":  today,
        "version":     result["new"],
        "changed":     result["changed"],
        "error":       result["error"],
        "checked_at":  datetime.utcnow().isoformat(),
    })
    return result


def update_yt_dlp_background(force: bool = False) -> None:
    """Fire-and-forget daily yt-dlp upgrade in a daemon thread, so app startup
    is never blocked on a pip/network round-trip."""
    threading.Thread(
        target=update_yt_dlp,
        kwargs={"force": force},
        daemon=True,
        name="YtDlpAutoUpdate",
    ).start()


def check_network() -> bool:
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except Exception:
        return False


def format_speed(bps) -> str:
    if not bps:
        return ""
    if bps >= 1_048_576:
        return f"{bps / 1_048_576:.1f} MB/s"
    return f"{bps / 1024:.0f} KB/s"


def format_eta(seconds) -> str:
    if seconds is None:
        return ""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60}s"
