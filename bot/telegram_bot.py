"""Telegram bot → headless B-Roll pipeline.

Run this on an always-on machine (e.g. the office laptop). Send it a voice
message or an audio file and it runs the full Director pipeline headless —
transcribe → … → auto-select → QA review → download → FCPXML — then replies
with a summary and attaches the Premiere XML. By the time you reach the office
the clips are downloaded and the project is ready to edit.

Setup (.env):
    TELEGRAM_BOT_TOKEN=123456:ABC...        # from @BotFather
    TELEGRAM_ALLOWED_USERS=11111111,2222    # your numeric Telegram user id(s)
    (plus the usual GROQ_API_KEY / PEXELS_API_KEY / … the pipeline needs)

Run:
    python -m bot.telegram_bot

Security: only messages from TELEGRAM_ALLOWED_USERS are processed. An empty
allowlist denies everyone (fail-closed). Big video files stay on the laptop;
only the small FCPXML is sent back over Telegram.
"""

import os
import re
import json
import time
import shutil
import threading
from collections import Counter

# Cap native thread pools (torch/OMP/MKL) before anything imports them, so the
# CPU-only server doesn't oversubscribe cores. See core/runtime.py.
from core.runtime import configure_runtime
configure_runtime()

import requests

from bot import settings as bot_settings
from bot import fileserver
from bot import logsetup
from bot import healthserver
from bot import pending_store

_API = "https://api.telegram.org/bot{token}/{method}"
_FILE_API = "https://api.telegram.org/file/bot{token}/{path}"
_AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".oga", ".aac", ".flac")

# Telegram bot upload cap for sendDocument (~50 MB). Bigger bundles go by link.
_TG_UPLOAD_LIMIT = 50 * 1024 * 1024

# Tracks the single in-flight job. The pipeline runs in a background thread so
# the polling loop stays responsive (can answer /status and /cancel mid-job).
# ``cancel`` is a threading.Event the running pipeline checks at each stage.
_BUSY = {"active": False, "project": None, "cancel": None}

# Projects paused at the review gate, keyed by chat_id: holds the selected shots
# + QA so /download, /refine, /details can act on them between messages. Mirrored
# to disk (see pending_store) so a restart / forcestop recovers them.
_PENDING: dict = {}


def _persist_pending() -> None:
    """Snapshot ``_PENDING`` to disk after every mutation. Best-effort — a
    persistence failure must never break the bot, so we swallow exceptions."""
    try:
        pending_store.save_pending(_PENDING)
    except Exception as e:
        print(f"[bot] couldn't persist review-gate state: {e}")

# File server port (set in main() when BOT_FILE_SERVER is enabled), used to build
# download links.
_FILESERVER = {"port": None}

# Last completed project name, so /zip knows what to bundle by default.
_LAST = {"project": None}

# Audio waiting on the user's "clear disk?" answer before the job starts.
# chat_id → {"file_id", "name"}.
_PENDING_START: dict = {}

# Chats that ran /overlay — their NEXT upload is offered as text-overlay-only.
_OVERLAY_NEXT: set = set()


# ── config ────────────────────────────────────────────────────────────────────

def _token() -> str:
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


def _proxies():
    """Explicit proxy for all bot HTTP calls, from BOT_PROXY (e.g. when Telegram
    is only reachable via a local VPN proxy like http://127.0.0.1:10809). When
    unset, requests falls back to the HTTP(S)_PROXY env / direct connection."""
    p = os.getenv("BOT_PROXY", "").strip()
    return {"http": p, "https": p} if p else None


def allowed_user_ids() -> set:
    raw = os.getenv("TELEGRAM_ALLOWED_USERS", "")
    out = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part.lstrip("-").isdigit():
            out.add(int(part))
    return out


def is_allowed(user_id, allowed: set) -> bool:
    """Fail-closed: an empty allowlist denies everyone."""
    return bool(allowed) and user_id in allowed


# ── message parsing (pure) ─────────────────────────────────────────────────────

def extract_audio(message: dict):
    """Return ``(file_id, suggested_name)`` for a voice / audio / audio-document
    message, or ``(None, None)`` if the message carries no audio."""
    if not isinstance(message, dict):
        return None, None
    if message.get("voice"):
        v = message["voice"]
        return v["file_id"], f"voice_{v.get('file_unique_id', 'msg')}.ogg"
    if message.get("audio"):
        a = message["audio"]
        return a["file_id"], a.get("file_name") or f"audio_{a.get('file_unique_id', 'msg')}.mp3"
    doc = message.get("document")
    if doc:
        mime = (doc.get("mime_type") or "").lower()
        name = (doc.get("file_name") or "").lower()
        if mime.startswith("audio") or name.endswith(_AUDIO_EXTS):
            return doc["file_id"], doc.get("file_name") or "audio.mp3"
    return None, None


# Telegram's Bot API only lets a bot DOWNLOAD files up to 20 MB via getFile.
# A self-hosted local Bot API server (TELEGRAM_API_BASE) lifts this to ~2 GB.
_TG_DOWNLOAD_LIMIT = 20 * 1024 * 1024


def _audio_size(message: dict) -> int:
    """Best-effort byte size of the audio in a message (0 if unknown)."""
    if not isinstance(message, dict):
        return 0
    for key in ("voice", "audio", "document"):
        obj = message.get(key)
        if obj and obj.get("file_size"):
            try:
                return int(obj["file_size"])
            except (TypeError, ValueError):
                return 0
    return 0


def _using_local_bot_api() -> bool:
    """True when pointed at a self-hosted Bot API server (no 20 MB download cap)."""
    base = (os.getenv("TELEGRAM_API_BASE") or "").strip()
    return bool(base) and "api.telegram.org" not in base


def too_big_message(size: int) -> str:
    return (
        f"❌ That file is {_human_size(size)} — Telegram's Bot API only lets bots "
        "download files up to 20 MB, so I can't fetch it.\n\n"
        "Fix it one of these ways:\n"
        "• Re-export the voiceover smaller — mono 64 kbps MP3 is plenty for speech "
        "(a ~30-min VO ≈ 14 MB):\n"
        "   ffmpeg -i input.wav -ac 1 -b:a 64k output.mp3\n"
        "• Or split it into parts under 20 MB and send them one at a time.\n"
        "• Or self-host a local Telegram Bot API server and set TELEGRAM_API_BASE "
        "(lifts the limit to ~2 GB)."
    )


def extract_cookies_doc(message: dict):
    """Return ``(file_id, name)`` for a YouTube cookies upload — a ``.txt``
    document whose name contains 'cookie', or any ``.txt`` sent with a /cookies
    (or 'cookie') caption — else ``(None, None)``."""
    if not isinstance(message, dict):
        return None, None
    doc = message.get("document")
    if not doc:
        return None, None
    name = (doc.get("file_name") or "").lower()
    caption = (message.get("caption") or "").lower().strip()
    if not name.endswith(".txt"):
        return None, None
    if "cookie" in name or "cookie" in caption or caption.startswith("/cookies"):
        return doc["file_id"], doc.get("file_name") or "cookies.txt"
    return None, None


def extract_library_doc(message: dict):
    """Return ``(file_id, name)`` for a Clip Library upload — a ``.db`` document,
    a file named like ``clip_library``, or any document sent with a ``/library``
    caption — else ``(None, None)``."""
    if not isinstance(message, dict):
        return None, None
    doc = message.get("document")
    if not doc:
        return None, None
    name = (doc.get("file_name") or "").lower()
    caption = (message.get("caption") or "").lower().strip()
    if name.endswith(".db") or "clip_library" in name or caption.startswith("/library"):
        return doc["file_id"], doc.get("file_name") or "clip_library.db"
    return None, None


def extract_xml_doc(message: dict):
    """Return ``(file_id, name)`` for an exported FCP7 XML upload — a ``.xml``
    document, or any document sent with an ``/import`` / ``/learn`` caption — so
    the bot can learn the editor's preferred trims from it. Else ``(None, None)``."""
    if not isinstance(message, dict):
        return None, None
    doc = message.get("document")
    if not doc:
        return None, None
    name = (doc.get("file_name") or "").lower()
    caption = (message.get("caption") or "").lower().strip()
    if name.endswith(".xml") or caption.startswith("/import") or caption.startswith("/learn"):
        return doc["file_id"], doc.get("file_name") or "timeline.xml"
    return None, None


def project_name_from(suggested_name: str, fallback: str = "") -> str:
    base = os.path.splitext(os.path.basename(suggested_name or ""))[0].strip()
    return (base or fallback or "voice")[:50]


# ── Telegram API (thin wrappers over requests) ─────────────────────────────────

def _call(method: str, _timeout: int = 60, **params) -> dict:
    resp = requests.post(_API.format(token=_token(), method=method), data=params,
                         timeout=_timeout, proxies=_proxies())
    if not resp.ok:
        # Surface Telegram's own reason (e.g. "file is too big", "Conflict:
        # terminated by other getUpdates") instead of a bare "400 Bad Request",
        # so /logs and user-facing errors actually say what went wrong.
        desc = ""
        try:
            desc = (resp.json() or {}).get("description") or ""
        except Exception:
            pass
        if desc:
            raise requests.HTTPError(f"{resp.status_code} {desc}", response=resp)
        resp.raise_for_status()
    return resp.json()


def send_message(chat_id, text: str, reply_markup: dict = None) -> dict:
    params = {"chat_id": chat_id, "text": text}
    if reply_markup is not None:
        params["reply_markup"] = json.dumps(reply_markup)
    try:
        return _call("sendMessage", **params).get("result", {})
    except Exception as e:
        print(f"[bot] sendMessage failed: {e}")
        return {}


def edit_message(chat_id, message_id, text: str, reply_markup: dict = None) -> None:
    if not message_id:
        return
    params = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if reply_markup is not None:
        params["reply_markup"] = json.dumps(reply_markup)
    try:
        _call("editMessageText", **params)
    except Exception:
        pass  # editing is best-effort (rate limits / identical text)


def answer_callback(callback_query_id: str, text: str = "") -> None:
    try:
        _call("answerCallbackQuery", callback_query_id=callback_query_id, text=text)
    except Exception:
        pass


def download_telegram_file(file_id: str, dest_path: str) -> str:
    path = _call("getFile", file_id=file_id)["result"]["file_path"]
    url = _FILE_API.format(token=_token(), path=path)
    with requests.get(url, stream=True, timeout=180, proxies=_proxies()) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    return dest_path


def send_document(chat_id, path: str, caption: str = "") -> None:
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                _API.format(token=_token(), method="sendDocument"),
                data={"chat_id": chat_id, "caption": caption[:1024]},
                files={"document": f}, timeout=180, proxies=_proxies(),
            )
        resp.raise_for_status()
    except Exception as e:
        send_message(chat_id, f"(Couldn't attach the XML: {e})")


# ── settings menu (inline keyboard) ─────────────────────────────────────────────

def build_settings_keyboard(chat_id) -> dict:
    """Inline keyboard: one tappable button per option (tap to toggle/cycle),
    plus Reset and Close. callback_data is ``set:<key>`` / ``settings:<action>``."""
    s = bot_settings.get_settings(chat_id)
    rows = []
    for opt in bot_settings.OPTIONS:
        val = bot_settings.display_value(opt, s.get(opt["key"]))
        rows.append([{"text": f"{opt['label']}: {val}", "callback_data": f"set:{opt['key']}"}])
    rows.append([
        {"text": "↺ Reset", "callback_data": "settings:reset"},
        {"text": "✖ Close", "callback_data": "settings:close"},
    ])
    return {"inline_keyboard": rows}


def _settings_text() -> str:
    return ("⚙️ Settings — tap a row to change it.\n"
            "Counts are per query/shot. Changes are saved and used for your next job.")


def send_settings(chat_id) -> None:
    send_message(chat_id, _settings_text(), reply_markup=build_settings_keyboard(chat_id))


def handle_settings_callback(cb: dict) -> None:
    """React to a tap on the /settings keyboard."""
    data = cb.get("data") or ""
    msg = cb.get("message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    message_id = msg.get("message_id")
    cb_id = cb.get("id")

    # Answer to the upload prompt: "start:<mode>:<disk>" where mode is
    # full|overlay and disk is clear|keep (#6 disk-clear + the Full vs
    # text-overlay-only mode choice).
    if data.startswith("start:"):
        try:
            _, mode, disk = data.split(":")
        except ValueError:
            answer_callback(cb_id, "Bad option")
            return
        pend = _PENDING_START.pop(chat_id, None)
        if not pend:
            edit_message(chat_id, message_id, "↪️ That start prompt expired — resend the audio file.")
            answer_callback(cb_id, "Expired")
            return
        if _BUSY.get("active") or _PENDING.get(chat_id):
            _PENDING_START[chat_id] = pend  # restore so they can retry later
            edit_message(chat_id, message_id, "⏳ A job is already running or awaiting review.")
            answer_callback(cb_id, "Busy")
            return
        note = ""
        if disk == "clear":
            n, freed = _clear_all_projects()
            note = f"🧹 Cleared {n} project(s), freed {_human_size(freed)}. "
        label = "Text-overlay only" if mode == "overlay" else "Full process"
        edit_message(chat_id, message_id, f"{note}▶️ {label} — starting…")
        answer_callback(cb_id, "Starting")
        if mode == "overlay":
            _start_job(handle_overlay_only, chat_id, pend["file_id"], pend["name"])
        else:
            _start_job(handle_audio, chat_id, pend["file_id"], pend["name"])
        return

    if data == "settings:close":
        edit_message(chat_id, message_id, "⚙️ Settings saved.")
        answer_callback(cb_id, "Saved")
        return
    if data == "settings:reset":
        bot_settings.reset(chat_id)
        edit_message(chat_id, message_id, _settings_text(),
                     reply_markup=build_settings_keyboard(chat_id))
        answer_callback(cb_id, "Reset to defaults")
        return
    if data.startswith("set:"):
        key = data[4:]
        try:
            s = bot_settings.toggle(chat_id, key)
            opt = next(o for o in bot_settings.OPTIONS if o["key"] == key)
            edit_message(chat_id, message_id, _settings_text(),
                         reply_markup=build_settings_keyboard(chat_id))
            answer_callback(cb_id, f"{opt['label']}: {bot_settings.display_value(opt, s[key])}")
        except Exception:
            answer_callback(cb_id, "Couldn't update that.")


# ── command predicates ───────────────────────────────────────────────────────

def is_settings_command(text: str) -> bool:
    return _command(text) in ("/settings", "/options", "/config")


def is_overlay_command(text: str) -> bool:
    return _command(text) in ("/overlay", "/overlays", "/textonly")


def is_download_command(text: str) -> bool:
    return _command(text) in ("/download", "/go", "/approve", "/accept")


def is_refine_command(text: str) -> bool:
    return _command(text) in ("/refine", "/improve")


def is_redo_command(text: str) -> bool:
    return _command(text) in ("/redo", "/fill", "/rescue")


def is_details_command(text: str) -> bool:
    return _command(text) in ("/details", "/detail", "/shots")


def is_help_command(text: str) -> bool:
    return _command(text) in ("/help", "/start", "/commands")


def is_logs_command(text: str) -> bool:
    return _command(text) in ("/logs", "/log")


def is_cookies_command(text: str) -> bool:
    return _command(text) in ("/cookies", "/cookie")


def is_cleanup_command(text: str) -> bool:
    return _command(text) in ("/cleanup", "/clean", "/disk")


def is_test_command(text: str) -> bool:
    return _command(text) in ("/test", "/selftest", "/preflight", "/check")


def is_proxies_command(text: str) -> bool:
    return _command(text) in ("/proxies", "/proxy")


# ── reporting (per-shot + QA + errors) ───────────────────────────────────────

def _shot_clip_info(shots: list) -> list:
    """Per selected shot: ``(slot_id, priority, n_clips, sources)``."""
    out = []
    for s in shots:
        sel = s.get("selected_results") or []
        if not sel:
            continue
        srcs = []
        for r in sel:
            src = (r.get("source") or "?").lower()
            if src not in srcs:
                srcs.append(src)
        out.append((s.get("slot_id"), s.get("priority", "—"), len(sel), ",".join(srcs)))
    return out


def format_qa_block(qa: dict, limit: int = 8) -> list:
    """QA verdict + flagged issues, each pinned to its shot number."""
    lines = []
    verdict = (qa or {}).get("overall") or "—"
    issues = (qa or {}).get("issues") or []
    head = f"QA: {verdict}"
    if qa.get("refined"):
        head += f"  (auto-refined {qa['refined']} shot[s])"
    lines.append(head)
    for it in issues[:limit]:
        sev = it.get("severity", "med")
        sug = it.get("suggestion", "")
        line = f"  ⚠️ #{it.get('slot_id')} ({sev}) {it.get('problem', '')}"
        if sug:
            line += f" → {sug}"
        lines.append(line[:300])
    if len(issues) > limit:
        lines.append(f"  …and {len(issues) - limit} more")
    return lines


def _err_signature(msg: str) -> str:
    """Collapse a specific error to its type: mask quoted queries/paths and
    numbers so 3,000 'search failed for <query>' lines become one signature."""
    s = re.sub(r"'[^']*'", "'…'", str(msg))
    s = re.sub(r"\d+", "#", s)
    return s.strip()[:180]


def format_errors_block(errors: list, limit: int = 6) -> list:
    """Group errors by type so a systemic failure shows as one line with a
    count, not thousands of near-identical messages."""
    if not errors:
        return []
    counts = Counter(_err_signature(e) for e in errors)
    lines = [f"❗ {len(errors)} issue(s) during processing ({len(counts)} type[s]):"]
    for sig, n in counts.most_common(limit):
        lines.append(f"  • {sig}" + (f"  (×{n})" if n > 1 else ""))
    if len(counts) > limit:
        lines.append(f"  …and {len(counts) - limit} more type(s)")
    return lines


def format_review(proj: str, result: dict) -> str:
    """Pre-download review: counts, per-shot clip spread, QA flags, errors, and
    the action prompt."""
    shots = result.get("shots") or []
    lines = [
        f"📋 {proj} — ready to review",
        f"Shots: {result.get('n_shots', 0)}  ·  with clips: {result.get('n_selected', 0)}"
        f"  ·  clips: {result.get('n_clips', 0)}",
    ]
    empties = [s.get("slot_id") for s in shots
               if s.get("priority") != "none" and not s.get("selected_results")]
    if empties:
        lines.append(f"⚪ No clip for shot(s): {', '.join(str(e) for e in empties[:20])}")
    lines += format_qa_block(result.get("qa") or {})
    lines += format_errors_block(result.get("errors") or [])
    lines.append("")
    lines.append("Reply:  /download  ·  /refine  ·  /details  ·  /cancel")
    return "\n".join(lines)


def format_details(shots: list) -> str:
    info = _shot_clip_info(shots)
    if not info:
        return "No shots have clips selected yet."
    lines = ["🎞 Per-shot selection:"]
    for slot, prio, n, srcs in info[:60]:
        lines.append(f"  #{slot} [{prio}] {n} clip(s) · {srcs}")
    return "\n".join(lines)


# ── project delivery (zip → Telegram attach + download link) ─────────────────

def deliver_project(chat_id, project: str) -> None:
    """Zip the finished project, attach it to Telegram when it fits under the
    upload cap, and always provide a signed download link (when the file server
    is up) plus the scp path."""
    from core.output import zip_project
    # Bundling a big project (hundreds of clips, multi-GB) takes a while and is
    # otherwise silent — show a live "zipping N/total" line on one edited message.
    zip_status = send_message(chat_id, f"📦 Zipping '{project}'…") or {}
    zip_msg_id = zip_status.get("message_id")
    _zip_state = {"last": 0.0}

    def _zip_progress(done, total):
        now = time.time()
        if done < total and now - _zip_state["last"] < 3.0:
            return  # throttle so we don't trip Telegram's edit rate limit
        _zip_state["last"] = now
        edit_message(chat_id, zip_msg_id, f"📦 Zipping '{project}'… {done}/{total} file(s)")

    try:
        res = zip_project(project, progress=_zip_progress)
    except FileNotFoundError:
        edit_message(chat_id, zip_msg_id, f"(No files to bundle for '{project}'.)")
        return
    except Exception as e:
        edit_message(chat_id, zip_msg_id, f"(Couldn't zip '{project}': {e})")
        return

    abs_path = os.path.abspath(res["path"])
    size = res["size_bytes"]
    lines = [f"📦 {project} — {res['files']} file(s), {_human_size(size)}"]

    port = _FILESERVER.get("port")
    if port:
        # Prefer a TLS domain (BOT_PUBLIC_URL → https://host/d/...) when set, so
        # links work through Traefik/Coolify; else fall back to raw host:port.
        base = fileserver.public_base_url()
        link = (fileserver.build_link(abs_path, base=base) if base
                else fileserver.build_link(abs_path, fileserver.public_host(), port))
        if link:
            lines.append(f"🔗 {link}\n(link expires in 24h)")
    lines.append(f"scp USER@SERVER:'{abs_path}' .")
    send_message(chat_id, "\n".join(lines))

    if size <= _TG_UPLOAD_LIMIT and os.path.exists(abs_path):
        send_document(chat_id, abs_path, caption=f"{project} — clips + XML")


# ── job ────────────────────────────────────────────────────────────────────────

def _fmt_tokens(n: int) -> str:
    """Compact token count: 1234 → '1.2k', 48230 → '48k', 950 → '950'."""
    n = int(n or 0)
    if n >= 1000:
        return f"{n / 1000:.1f}k".replace(".0k", "k")
    return str(n)


def format_cost_line(result: dict):
    """One-line API usage/cost estimate for the summary, or None if nothing was
    tracked. Shows a $ estimate when priced, else token counts only."""
    c = result.get("cost") or {}
    by = c.get("by_provider") or {}
    if not by:
        return None
    parts = []
    for prov, p in sorted(by.items()):
        tok = (p.get("prompt_tokens", 0) or 0) + (p.get("completion_tokens", 0) or 0)
        if tok:
            parts.append(f"{prov} {_fmt_tokens(tok)}")
        elif p.get("audio_seconds"):
            parts.append(f"{prov} {p['audio_seconds']:.0f}s")
    detail = ", ".join(parts)
    total_tok = c.get("total_tokens", 0)
    if c.get("priced"):
        return (f"💸 API ~${c.get('total_usd', 0):.4f} (est.)"
                f"  ·  {_fmt_tokens(total_tok)} tokens  ·  {detail}")
    return f"🔢 API: {_fmt_tokens(total_tok)} tokens  ·  {detail}"


def format_summary(proj: str, result: dict) -> str:
    dl = result.get("download") or {}
    qa = result.get("qa") or {}
    n_issues = len(qa.get("issues") or [])
    lines = [
        f"✅ Done — {proj}",
        f"Shots: {result.get('n_shots', 0)}  ·  with clips: {result.get('n_selected', 0)}  ·  clips: {result.get('n_clips', 0)}",
    ]
    if result.get("download") is not None:
        extra = ""
        if dl.get("repaired"):
            extra += f", {dl['repaired']} re-picked"
        if dl.get("dropped"):
            extra += f", {dl['dropped']} dropped"
        lines.append(f"Downloaded: {dl.get('ok', 0)} ok, {dl.get('failed', 0)} failed, "
                     f"{dl.get('skipped', 0)} cached{extra}")
    verdict = qa.get("overall") or "—"
    lines.append(f"QA: {verdict}" + (f"  ⚠️ {n_issues} flag(s)" if n_issues else ""))
    cost_line = format_cost_line(result)
    if cost_line:
        lines.append(cost_line)
    if dl.get("dir"):
        lines.append(f"📁 {dl['dir']}")
    return "\n".join(lines)


def _progress_logger(chat_id, proj, msg_id):
    """App-style running step log: appends each stage (✅ done, ⏳ current) to a
    single edited message. A repeated call with the same ``step`` is treated as
    a sub-progress update (e.g. 'Fetching candidates · 60%') and rewrites the
    current line in place rather than appending a new one. In-place updates are
    throttled so frequent ticks don't trip Telegram's edit rate limit."""
    steps: list = []
    state = {"step": None, "last_edit": 0.0}

    def _cb(step, total, label):
        new_stage = step != state["step"]
        if new_stage or not steps:
            steps.append(label)
            state["step"] = step
        else:
            steps[-1] = label   # same stage → live sub-progress, replace line

        # Always render a new stage (and a final 100%); throttle the rest so a
        # busy phase doesn't spam editMessage and hit HTTP 429.
        now = time.time()
        if not new_stage and "100%" not in label and now - state["last_edit"] < 3.0:
            return
        state["last_edit"] = now

        lines = [f"🎬 {proj}  ({step}/{total})"]
        for i, lab in enumerate(steps):
            lines.append(f"{'⏳' if i == len(steps) - 1 else '✅'} {lab}")
        edit_message(chat_id, msg_id, "\n".join(lines[-18:]))   # cap under TG limits

    return _cb


def _should_cancel():
    cancel = _BUSY.get("cancel")
    return cancel.is_set if cancel is not None else None


def send_shots_srt(chat_id, proj: str, shots: list) -> None:
    """Generate and send the shot-number SRT (drop it on the audio to see
    'Shot N' over each shot's run). Written next to the project's XML."""
    try:
        from core.output import generate_shots_srt, clip_base_dir, _safe_for_fs
        srt = generate_shots_srt(shots)
        proj_dir = os.path.dirname(clip_base_dir(proj))   # downloads/<proj>/
        os.makedirs(proj_dir, exist_ok=True)
        srt_path = os.path.join(proj_dir, f"{_safe_for_fs(proj, 50)}.shots.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt)
        send_document(chat_id, srt_path, caption=f"{proj} — shot-number SRT")
    except Exception as e:
        send_message(chat_id, f"(Couldn't build the shot SRT: {e})")


def _deliver_completed(chat_id, proj, result) -> None:
    """Final reporting for a fully-processed (downloaded) project: summary, the
    Premiere XML, the shot-number SRT, then the zipped bundle (Telegram attach +
    download link)."""
    send_message(chat_id, format_summary(proj, result))
    xml_path = result.get("xml_path")
    if xml_path and os.path.exists(xml_path):
        send_document(chat_id, xml_path, caption=f"{proj} — Premiere XML")
    send_shots_srt(chat_id, proj, result.get("shots") or [])
    # If any shot ended with no footage, surface the source-link manifest up front
    # so the user can re-download those by hand (it's also bundled in the zip).
    shots = result.get("shots") or []
    empty = [s for s in shots if s.get("priority") != "none"
             and not s.get("skipped") and not s.get("selected_results")]
    links_path = _links_file_path(proj)
    if empty and os.path.exists(links_path):
        send_document(chat_id, links_path,
                      caption=f"⚠️ {len(empty)} shot(s) have no footage — links to re-download them")
    deliver_project(chat_id, proj)


def handle_audio(chat_id, file_id: str, suggested_name: str) -> dict:
    """Download the audio and run the pipeline under the chat's settings. With
    the review gate on, stops after selection + QA and waits for /download;
    otherwise downloads and delivers end-to-end."""
    from core.pipeline import run_pipeline_headless, PipelineCancelled

    proj = project_name_from(suggested_name, time.strftime("video_%Y%m%d_%H%M%S"))
    ext = os.path.splitext(suggested_name or "")[1].lower() or ".mp3"
    audio_path = os.path.join(".cache", f"bot_{proj}{ext}")

    settings = bot_settings.get_settings(chat_id)
    review_gate = bool(settings.get("review_gate"))

    status = send_message(chat_id, f"🎬 Received '{proj}'. Downloading…")
    msg_id = status.get("message_id")
    try:
        download_telegram_file(file_id, audio_path)
    except Exception as e:
        send_message(chat_id, f"❌ Couldn't fetch the audio: {e}")
        return {}

    progress = _progress_logger(chat_id, proj, msg_id)
    try:
        with bot_settings.apply_env(settings):
            result = run_pipeline_headless(
                audio_path, project_name=proj, download=not review_gate,
                progress_callback=progress, should_cancel=_should_cancel(),
                run_qa=bool(settings.get("qa")),
                auto_refine=bool(settings.get("auto_refine")),
                auto_fill=bool(settings.get("auto_fill")),
                quality=str(settings.get("quality", 1080)),
            )
    except PipelineCancelled:
        send_message(chat_id, f"⏹ Cancelled '{proj}'.")
        return {}
    except Exception as e:
        send_message(chat_id, f"❌ Pipeline failed for '{proj}': {e}")
        return {}

    edit_message(chat_id, msg_id, f"🎬 {proj} — ✅ stages complete")
    _LAST["project"] = proj

    if review_gate:
        _PENDING[chat_id] = {
            "project": proj, "shots": result.get("shots") or [],
            "qa": result.get("qa") or {}, "topic": result.get("topic", ""),
            "errors": result.get("errors") or [], "result": result, "settings": settings,
        }
        _persist_pending()
        send_message(chat_id, format_review(proj, result))
        send_shots_srt(chat_id, proj, result.get("shots") or [])
    else:
        _deliver_completed(chat_id, proj, result)
    return result


def handle_overlay_only(chat_id, file_id: str, suggested_name: str) -> dict:
    """Text-overlay-only path: transcribe → render animated overlays (Remotion,
    alpha ProRes + baked SFX) → deliver the clips + an overlays-only FCPXML.
    Skips all footage fetch/download, so it's fast and small."""
    from core.pipeline import run_overlays_only, PipelineCancelled

    proj = project_name_from(suggested_name, time.strftime("overlay_%Y%m%d_%H%M%S"))
    ext = os.path.splitext(suggested_name or "")[1].lower() or ".mp3"
    audio_path = os.path.join(".cache", f"bot_{proj}{ext}")

    status = send_message(chat_id, f"🅰️ Overlays-only for '{proj}'. Downloading audio…")
    msg_id = status.get("message_id")
    try:
        download_telegram_file(file_id, audio_path)
    except Exception as e:
        send_message(chat_id, f"❌ Couldn't fetch the audio: {e}")
        return {}

    progress = _progress_logger(chat_id, proj, msg_id)
    try:
        with bot_settings.apply_env(bot_settings.get_settings(chat_id)):
            result = run_overlays_only(audio_path, project_name=proj,
                                       progress_callback=progress,
                                       should_cancel=_should_cancel())
    except PipelineCancelled:
        send_message(chat_id, f"⏹ Cancelled '{proj}'.")
        return {}
    except Exception as e:
        send_message(chat_id, f"❌ Overlay pass failed for '{proj}': {e}")
        return {}

    n = result.get("n_overlays", 0)
    edit_message(chat_id, msg_id, f"🅰️ {proj} — ✅ {n} overlay clip(s) rendered")
    _LAST["project"] = proj
    if n == 0:
        send_message(chat_id, "No overlay-worthy moments were found (or the overlay "
                              "renderer isn't installed on the server yet).")
        return result
    xml_path = result.get("xml_path")
    if xml_path and os.path.exists(xml_path):
        send_document(chat_id, xml_path, caption=f"{proj} — overlays FCPXML")
    deliver_project(chat_id, proj)
    return result


def is_overlay_text_command(text: str) -> bool:
    return _command(text) in ("/overlaytext", "/textoverlay", "/overlaytxt")


def _parse_overlay_text_args(text: str):
    """Pull ``(duration_sec, content)`` from a /overlaytext command. The first
    token is read as the duration when numeric ('/overlaytext 4 CAR BATTERY' →
    (4.0, 'CAR BATTERY')); otherwise duration defaults to 4s and the whole body is
    the text. Returns ``(None, "")`` when no text was given."""
    parts = text.split(maxsplit=1)
    body = parts[1].strip() if len(parts) > 1 else ""
    if not body:
        return None, ""
    head = body.split(maxsplit=1)
    try:
        dur = float(head[0])
    except ValueError:
        return 4.0, body                      # no leading number → all text
    content = head[1].strip() if len(head) > 1 else ""
    if content:
        return dur, content
    return 4.0, head[0]                        # only a bare number → use it as text


def handle_overlay_text(chat_id, text: str) -> None:
    """Render ONE animated text overlay from a user-supplied text + duration and
    send back the transparent ProRes clip (drop it on your timeline anywhere).
    Stateless — no audio/project needed."""
    from core.overlays_remotion import render_one_overlay, remotion_available
    dur, content = _parse_overlay_text_args(text)
    if not content:
        send_message(chat_id,
                     "Usage: /overlaytext <seconds> <text>\n"
                     "e.g. /overlaytext 4 CHECK YOUR CAR BATTERY\n"
                     "(seconds optional — defaults to 4. Text is shown in CAPS; "
                     "a figure like \"200AH\" animates as a stat.)")
        return
    if not remotion_available():
        send_message(chat_id, "The overlay renderer isn't installed on this server yet.")
        return
    status = send_message(chat_id, f"🅰️ Rendering overlay ({dur:g}s): \"{content[:60]}\"…")
    msg_id = (status or {}).get("message_id")
    out_dir = os.path.join(".cache", "manual_overlays")
    try:
        ov = render_one_overlay(content, dur, out_dir)
    except Exception as e:
        send_message(chat_id, f"❌ Overlay render failed: {e}")
        return
    if not ov or not os.path.exists(ov.get("filepath", "")):
        send_message(chat_id, "❌ Couldn't render that overlay (see logs).")
        return
    if msg_id:
        edit_message(chat_id, msg_id, f"🅰️ Overlay ready ({dur:g}s).")
    send_document(chat_id, ov["filepath"],
                  caption=f"Overlay · {dur:g}s · transparent ProRes 4444 (.mov) — "
                          "drop on your top video track.")


def _run_download(chat_id) -> None:
    """Review gate → /download: fetch the approved selection and deliver."""
    pend = _PENDING.get(chat_id)
    if not pend:
        send_message(chat_id, "Nothing is waiting for /download. Send a voice file first.")
        return
    from core.pipeline import finalize_project, PipelineCancelled
    proj, settings, result = pend["project"], pend["settings"], pend["result"]

    status = send_message(chat_id, f"⬇️ Downloading clips for '{proj}'…")
    msg_id = status.get("message_id")

    def _p(done, total):
        edit_message(chat_id, msg_id, f"⬇️ {proj}: downloaded {done}/{total} clip(s)…")

    def _status(label):
        # Surface the pre-download proxy-validation phase (otherwise a silent,
        # possibly minute-long wait on a free proxy list).
        edit_message(chat_id, msg_id, f"⬇️ {proj}: {label}")

    try:
        with bot_settings.apply_env(settings):
            fin = finalize_project(pend["shots"], proj, quality=str(settings.get("quality", 1080)),
                                   should_cancel=_should_cancel(), progress=_p,
                                   overlays=result.get("overlays"),
                                   sfx_list=result.get("sfx_list"),
                                   video_topic=pend.get("topic", ""),
                                   errors=pend.get("errors"), status=_status)
    except PipelineCancelled:
        send_message(chat_id, f"⏹ Cancelled '{proj}'.")
        return
    except Exception as e:
        send_message(chat_id, f"❌ Download failed for '{proj}': {e}")
        return
    result["download"] = fin["download"]
    result["xml_path"] = fin["xml_path"]
    _PENDING.pop(chat_id, None)
    _persist_pending()
    _deliver_completed(chat_id, proj, result)


def _parse_slots(text: str) -> set:
    """Pull shot numbers out of a '/refine 4 9' or '/refine 4,9' command."""
    import re
    return {int(n) for n in re.findall(r"\d+", text or "")}


def _run_refine(chat_id, only_slots=None) -> None:
    """Review gate → /refine: re-pick flagged shots (or the specific shot numbers
    the user named), re-review, re-show."""
    pend = _PENDING.get(chat_id)
    if not pend:
        send_message(chat_id, "Nothing to /refine. Send a voice file first.")
        return
    from core.pipeline import refine_flagged_shots, write_fcpxml
    from core.director_rank import review_timeline
    proj, settings, qa, shots = pend["project"], pend["settings"], pend["qa"], pend["shots"]
    if not only_slots and not qa.get("issues"):
        send_message(chat_id, "No QA-flagged shots to refine. Name shots to redo, "
                              "e.g. /refine 4 9 — or /download to proceed.")
        return

    target_desc = (f"shot(s) {', '.join(map(str, sorted(only_slots)))}" if only_slots
                   else f"{len(qa['issues'])} flagged shot(s)")
    status = send_message(chat_id, f"🛠 Refining {target_desc} for '{proj}'…")
    msg_id = status.get("message_id")
    key = os.getenv("GROQ_API_KEY")
    try:
        with bot_settings.apply_env(settings):
            n = refine_flagged_shots(shots, qa, groq_key=key, video_topic=pend["topic"],
                                     errors=pend["errors"], only_slots=only_slots)
            new_qa = review_timeline(shots, api_key=key, video_topic=pend["topic"]) if n else qa
            write_fcpxml(shots, proj)
    except Exception as e:
        send_message(chat_id, f"❌ Refine failed: {e}")
        return

    result = pend["result"]
    result["n_selected"] = sum(1 for s in shots if s.get("selected_results"))
    result["n_clips"] = sum(len(s.get("selected_results") or []) for s in shots)
    if n:
        new_qa["refined"] = n
    result["qa"] = new_qa
    pend["qa"] = new_qa
    _persist_pending()   # selection changed — re-snapshot so a restart keeps it
    edit_message(chat_id, msg_id, f"🛠 {proj} — refined {n} shot(s).")
    send_message(chat_id, format_review(proj, result))


def _run_redo(chat_id) -> None:
    """Review gate → /redo: aggressively get a clip onto every shot that has
    none, with YouTube emphasized."""
    pend = _PENDING.get(chat_id)
    if not pend:
        send_message(chat_id, "Nothing to /redo. Send a voice file first.")
        return
    from core.pipeline import fill_empty_shots, write_fcpxml
    from core.director_rank import review_timeline
    proj, settings, shots = pend["project"], pend["settings"], pend["shots"]

    empties = [s for s in shots
               if s.get("priority") != "none" and not s.get("selected_results")]
    if not empties:
        send_message(chat_id, "✅ Every shot already has a clip. /download to proceed.")
        return

    status = send_message(chat_id, f"♻️ Re-fetching {len(empties)} empty shot(s) for '{proj}' "
                                   "(YouTube-first)…")
    msg_id = status.get("message_id")

    def _p(label):
        edit_message(chat_id, msg_id, f"♻️ {proj}: {label}")

    # Force YouTube on and pull more YouTube candidates while filling gaps.
    redo_settings = dict(settings, use_youtube=True,
                         youtube_num=max(8, int(settings.get("youtube_num", 4))))
    key = os.getenv("GROQ_API_KEY")
    try:
        with bot_settings.apply_env(redo_settings):
            n = fill_empty_shots(shots, groq_key=key, video_topic=pend["topic"],
                                 errors=pend["errors"], progress=_p, passes=3)
            new_qa = review_timeline(shots, api_key=key, video_topic=pend["topic"]) if n else pend["qa"]
            write_fcpxml(shots, proj)
    except Exception as e:
        send_message(chat_id, f"❌ Redo failed: {e}")
        return

    result = pend["result"]
    result["n_selected"] = sum(1 for s in shots if s.get("selected_results"))
    result["n_clips"] = sum(len(s.get("selected_results") or []) for s in shots)
    result["qa"] = new_qa
    pend["qa"] = new_qa
    _persist_pending()   # selection changed — re-snapshot so a restart keeps it
    still = sum(1 for s in shots
                if s.get("priority") != "none" and not s.get("selected_results"))
    edit_message(chat_id, msg_id, f"♻️ {proj} — filled {n} shot(s); {still} still empty.")
    send_message(chat_id, format_review(proj, result))


# ── log export + command registration ────────────────────────────────────────

_COOKIES_HELP = (
    "🍪 To give me YouTube cookies, send the cookies.txt file as a document "
    "(Netscape format — export with a 'Get cookies.txt' browser extension while "
    "logged into YouTube). I'll save it and use it right away — no redeploy.\n"
    "Tip: name it with 'cookies' or add the caption /cookies."
)


def handle_cookies_upload(chat_id, file_id: str, name: str) -> None:
    """Save an uploaded cookies.txt to the persistent .cache and start using it."""
    from core.youtube import (uploaded_cookie_path, reset_cookies_state,
                              _sanitized_cookie_file)
    dest = uploaded_cookie_path()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        download_telegram_file(file_id, dest)
    except Exception as e:
        send_message(chat_id, f"❌ Couldn't save cookies: {e}")
        return
    reset_cookies_state()   # re-enable cookies for this session
    # Validate by sanitizing (strips BOM) + loading.
    try:
        clean = _sanitized_cookie_file(dest)
        from yt_dlp.cookies import YoutubeDLCookieJar
        jar = YoutubeDLCookieJar(clean)
        jar.load()
        n = len(jar)
        if n == 0:
            send_message(chat_id, "⚠️ Saved, but 0 cookies parsed — the file is "
                                  "empty or not a real Netscape cookies.txt. Re-export "
                                  "with a 'Get cookies.txt' extension and resend.")
        else:
            send_message(chat_id, f"✅ Cookies saved & loaded ({n} cookies). "
                                  "YouTube will use them now. /status to confirm.")
    except Exception as e:
        send_message(chat_id, f"⚠️ Saved, but this doesn't look like a valid "
                              f"Netscape cookies.txt: {str(e)[:120]}\n"
                              "Re-export with a 'Get cookies.txt' extension.")


def handle_library_upload(chat_id, file_id: str, name: str) -> None:
    """Merge an uploaded clip_library.db into the server's Clip Library so the
    bot reuses previously-collected footage as candidates."""
    import tempfile
    from core.clip_library import merge_library_db
    tmp = os.path.join(tempfile.gettempdir(), "uploaded_clip_library.db")
    try:
        download_telegram_file(file_id, tmp)
    except Exception as e:
        send_message(chat_id, f"❌ Couldn't download the library DB: {e}")
        return
    res = merge_library_db(tmp, http_only=True)
    try:
        os.remove(tmp)
    except OSError:
        pass
    if res.get("error"):
        send_message(chat_id, f"❌ {res['error']}")
        return
    send_message(
        chat_id,
        f"✅ Clip Library merged ({res['total_src']} clips in file):\n"
        f"• added: {res['added']}\n"
        f"• already had: {res['skipped_dupe']}\n"
        f"• skipped local-only (not re-downloadable): {res['skipped_local']}\n"
        "Imported clips are re-downloaded from their source URL when selected.")


def handle_xml_upload(chat_id, file_id: str, name: str) -> None:
    """Learn the editor's preferred trims from an exported (and Premiere-edited)
    FCP7 XML: each clip's in/out is recorded against the Clip Library so future
    projects reuse your cut points. Writes to the persistent DB; idempotent."""
    import tempfile
    from core.xml_reimport import ingest_reimported_xml
    tmp = os.path.join(tempfile.gettempdir(), f"reimport_{int(time.time())}.xml")
    try:
        download_telegram_file(file_id, tmp)
    except Exception as e:
        send_message(chat_id, f"❌ Couldn't download the XML: {e}")
        return
    try:
        s = ingest_reimported_xml(tmp)
    except Exception as e:
        send_message(chat_id, f"❌ Couldn't parse '{name}': {e}")
        return
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    lines = [
        f"✅ Learned trims from '{name}':",
        f"• clips parsed: {s['parsed']} ({s['video']} video)",
        f"• matched to library: {s['matched']}  ·  new rows: {s['created']}",
        f"• preferred trims recorded: {s['recorded']}",
    ]
    if s.get("skipped_non_video"):
        lines.append(f"• skipped audio/SFX: {s['skipped_non_video']}")
    if s.get("unmatched"):
        n = len(s["unmatched"])
        lines.append(f"• unresolved: {n} (e.g. {', '.join(s['unmatched'][:3])})")
    lines.append("These cut points will be reused when the same footage is picked again.")
    send_message(chat_id, "\n".join(lines))


def handle_forcestop(chat_id) -> None:
    """Hard stop: signal cancel, then exit the process so a wedged worker (stuck
    in a blocking network call /cancel can't interrupt) is killed for sure. Under
    Docker --restart / systemd Restart=always the bot comes back idle in seconds;
    downloaded files and uploaded cookies persist on the mounted volumes."""
    cancel = _BUSY.get("cancel")
    if cancel is not None:
        cancel.set()
    proj = _BUSY.get("project") or "—"
    send_message(chat_id, f"🛑 Force-stopping '{proj}' and restarting the bot. "
                          "Back in a few seconds (downloads & cookies are kept).")
    # Delay the exit briefly so the message actually flushes to Telegram.
    def _boom():
        time.sleep(1.0)
        os._exit(0)
    threading.Thread(target=_boom, daemon=True).start()


def handle_logs(chat_id) -> None:
    """Send the captured bot log as a .txt document."""
    dest = os.path.join(".cache", f"broll-bot-{time.strftime('%Y%m%d_%H%M%S')}.log")
    path = logsetup.snapshot_logs(dest)
    if not path:
        send_message(chat_id, "No logs captured yet.")
        return
    send_document(chat_id, path, caption="B-Roll bot logs")
    try:
        os.remove(path)
    except OSError:
        pass


def _format_proxy_stats() -> str:
    from core import proxy_pool
    if not proxy_pool.pool_active():
        return ("No dynamic proxy list configured. Set YT_DLP_PROXY_URL (a free "
                "proxy list) or YT_DLP_PROXY (your own proxies) to use proxies.")
    s = proxy_pool.stats()
    sample = ", ".join(_mask_proxy(p) for p in proxy_pool.working_snapshot()[:6])
    if s["working"]:
        lines = [f"✅ Proxy pool ready — {s['working']} working · {s['dead']} dead "
                 f"this session · {s['raw']} in list"]
        if sample:
            lines.append(f"working: {sample}")
        lines.append("Downloads will use these. /proxies refresh re-tests the list.")
    else:
        lines = [f"⚠️ No proxies validated yet — {s['dead']} tested-dead · "
                 f"{s['raw']} fetched from the list",
                 "The pool builds itself when a project starts (the 'Finding "
                 "working proxies' step). /proxies refresh tests it right now."]
    return "\n".join(lines)


def _mask_proxy(url: str) -> str:
    import re
    return re.sub(r"//[^@/]+@", "//***@", url or "")


def handle_proxies(chat_id, text: str) -> None:
    """Show the validated proxy pool, or (``/proxies refresh``) re-research the
    list to (re)find working proxies."""
    from core import proxy_pool
    if not proxy_pool.pool_active():
        send_message(chat_id, _format_proxy_stats())
        return
    if "refresh" in (text or "").lower() or "research" in (text or "").lower():
        status = send_message(chat_id, "🔎 Researching the proxy list for working "
                                       "proxies — searching until it finds enough. "
                                       "/cancel to stop.")
        msg_id = status.get("message_id")

        last = {"t": 0.0}

        def _p(label):
            # Throttle edits so a fast scan doesn't trip Telegram's rate limit.
            now = time.time()
            if now - last["t"] < 2.0:
                return
            last["t"] = now
            try:
                edit_message(chat_id, msg_id, f"🔎 {label} · /cancel to stop")
            except Exception:
                pass

        n = proxy_pool.refresh(progress=_p, should_cancel=_should_cancel())
        verdict = (f"✅ Proxy research done — {n} working proxy(ies)." if n
                   else "⚠️ Stopped — no working proxies found yet. Try again later "
                        "(/proxies refresh); free lists vary a lot by the hour.")
        edit_message(chat_id, msg_id, verdict)
        send_message(chat_id, _format_proxy_stats())
        return
    send_message(chat_id, _format_proxy_stats())


def format_selftest(report: dict) -> str:
    """Render the preflight report: a headline verdict then one line per check."""
    ok = report.get("ok")
    lines = ["✅ Preflight passed — safe to send a long voice file." if ok
             else "❌ Preflight found problems — look at these BEFORE a real run:"]
    for r in report.get("results", []):
        if r["ok"] is None:
            icon = "➖"                       # skipped (e.g. no key for that source)
        elif r["ok"]:
            icon = "✅"
        else:
            icon = "❌" if r["critical"] else "⚠️"   # ⚠️ = non-blocking
        secs = f"  ({r['secs']}s)" if r.get("secs") else ""
        lines.append(f"{icon} {r['name']}: {r['detail']}{secs}")
    if not ok:
        lines.append("")
        lines.append("Tip: most YouTube download failures are cookies/anti-bot — "
                     "see /cookies. Re-run /test after fixing.")
    return "\n".join(lines)


def handle_selftest(chat_id, text: str) -> None:
    """Run the live preflight self-test (LLM, transcription, Pexels + yt-dlp
    search and real downloads) so problems surface before a long voiceover is
    processed. ``/test quick`` skips the actual clip downloads."""
    from core.selftest import run_self_test

    quick = "quick" in (text or "").lower()
    head = ("🧪 Running preflight (quick — no downloads)…" if quick
            else "🧪 Running preflight — LLM, transcription, Pexels + yt-dlp search "
                 "and a couple of real test downloads. ~30–60s…")
    status = send_message(chat_id, head)
    msg_id = status.get("message_id")

    def _p(label):
        try:
            edit_message(chat_id, msg_id, f"{head}\n⏳ checking: {label}")
        except Exception:
            pass

    try:
        report = run_self_test(do_downloads=not quick, quality="360", progress=_p)
    except Exception as e:
        send_message(chat_id, f"❌ Preflight itself errored: {e}")
        return
    edit_message(chat_id, msg_id, "🧪 Preflight complete.")
    send_message(chat_id, format_selftest(report))


# (command, description) pairs registered with Telegram so typing '/' shows a menu.
_BOT_COMMANDS = [
    ("settings", "Sources, counts, quality, QA, review gate"),
    ("status", "Is the bot online and ready?"),
    ("test", "Preflight: test yt-dlp/Pexels/LLM before a real run"),
    ("proxies", "Show the working proxy pool (/proxies refresh to re-research)"),
    ("details", "Per-shot clip breakdown (pending project)"),
    ("download", "Fetch clips for the reviewed project"),
    ("refine", "Re-pick QA-flagged shots (or /refine 4 9)"),
    ("redo", "Re-fetch shots with no clip (YouTube-first)"),
    ("cancel", "Stop the running job / discard pending"),
    ("forcestop", "Hard stop + restart the bot"),
    ("zip", "Bundle a finished project (link + attach)"),
    ("overlaytext", "Render one overlay clip: /overlaytext 4 YOUR TEXT"),
    ("links", "Source links per shot (re-download empty shots)"),
    ("cleanup", "Show disk usage / delete a project"),
    ("logs", "Export the bot logs as a file"),
    ("cookies", "How to upload YouTube cookies.txt"),
    ("help", "Show the command list"),
]


def register_commands() -> None:
    """Tell Telegram our command list so clients show a '/' suggestion menu."""
    cmds = [{"command": c, "description": d} for c, d in _BOT_COMMANDS]
    try:
        _call("setMyCommands", commands=json.dumps(cmds))
    except Exception as e:
        print(f"[bot] setMyCommands failed: {e}")


# ── health / status check ──────────────────────────────────────────────────────

# Lightweight reachability probes — a general-connectivity endpoint plus the API
# hosts the pipeline actually needs (useful behind a TUN VPN, where the tunnel
# may be up for some hosts and not others).
_PROBE_URLS = [
    ("internet", "https://www.google.com/generate_204"),
    ("Groq", "https://api.groq.com"),
    ("OpenRouter", "https://openrouter.ai"),
]


def _probe_internet(timeout: int = 8):
    """Return ``(ok, detail)``. ``ok`` is True if any host responds at all (any
    HTTP status counts — only connection/timeout errors are failures)."""
    reachable, failed = [], []
    for name, url in _PROBE_URLS:
        try:
            requests.head(url, timeout=timeout, allow_redirects=True, proxies=_proxies())
            reachable.append(name)
        except Exception:
            failed.append(name)
    detail = "reachable: " + (", ".join(reachable) or "none")
    if failed:
        detail += "  ·  unreachable: " + ", ".join(failed)
    return bool(reachable), detail


def check_health(timeout: int = 8) -> list:
    """Return a list of ``(name, ok, detail)`` health checks for the host:
    required keys, search sources, internet/API reachability, ffmpeg, and that
    the pipeline imports cleanly."""
    checks = []

    has_groq = bool(os.getenv("GROQ_API_KEY"))
    checks.append(("Groq key", has_groq, "set" if has_groq else "MISSING — required"))

    srcs = [n for n, k in (("Pexels", "PEXELS_API_KEY"), ("Pixabay", "PIXABAY_API_KEY"),
                           ("YouTube", "YOUTUBE_API_KEY")) if os.getenv(k)]
    checks.append(("Search sources", bool(srcs), ", ".join(srcs) or "NONE enabled"))

    # Pexels key pool — confirms extra keys for rate-limit rotation are loaded.
    try:
        from core.stock_apis import pexels_key_pool
        n_pex = len(pexels_key_pool())
        if n_pex:
            checks.append(("Pexels keys", True,
                           f"{n_pex} key(s)" + (" (rotation on)" if n_pex > 1 else "")))
    except Exception:
        pass

    # YouTube cookie source — the #1 thing that silently breaks YouTube on a server.
    try:
        from core.youtube import cookie_mode
        ck_ok, ck_detail = cookie_mode()
        checks.append(("YouTube cookies", ck_ok, ck_detail))
    except Exception as e:
        checks.append(("YouTube cookies", False, f"unknown ({e})"))

    ds = bool(os.getenv("DEEPSEEK_API_KEY"))
    checks.append(("DeepSeek (OpenRouter)", ds, "set" if ds else "not set (free tier)"))

    net_ok, net_detail = _probe_internet(timeout)
    checks.append(("Internet", net_ok, net_detail))

    ff = shutil.which("ffmpeg") is not None
    checks.append(("ffmpeg", ff, "found" if ff else "MISSING — needed for downloads"))

    try:
        import core.pipeline  # noqa: F401
        checks.append(("Pipeline", True, "ready"))
    except Exception as e:
        checks.append(("Pipeline", False, f"import error: {e}"))

    return checks


def _fmt_duration(secs: float) -> str:
    secs = int(max(0, secs))
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def format_health(checks: list, busy: dict = None, pending: list = None) -> str:
    lines = ["🟢 Bot online — laptop is up and polling."]
    for name, ok, detail in checks:
        lines.append(f"{'✅' if ok else '❌'} {name}: {detail}")
    if busy and busy.get("active"):
        started = busy.get("started")
        dur = f" for {_fmt_duration(time.time() - started)}" if started else ""
        lines.append(f"⏳ Busy — processing '{busy.get('project', 'a job')}'{dur}.")
        lines.append("🛑 Ongoing job — /cancel to stop it gracefully, or /forcestop to force.")
    elif pending:
        names = ", ".join(str(p) for p in pending)
        lines.append(f"📋 Awaiting your review: {names}")
        lines.append("➡️ Reply /download, /refine, /redo — or /cancel to discard.")
    else:
        lines.append("💤 Idle — ready for a voice file.")
    return "\n".join(lines)


def _command(text: str) -> str:
    """Lowercased leading command token (`/status@Bot` → `/status`), or ''."""
    if not text:
        return ""
    return text.strip().split()[0].split("@")[0].lower()


def is_status_command(text: str) -> bool:
    return _command(text) in ("/status", "/health", "/ping")


def is_cancel_command(text: str) -> bool:
    return _command(text) in ("/cancel", "/stop", "/abort")


def is_forcestop_command(text: str) -> bool:
    return _command(text) in ("/forcestop", "/kill", "/forcecancel", "/fstop")


def is_zip_command(text: str) -> bool:
    return _command(text) in ("/zip", "/package", "/bundle")


def is_links_command(text: str) -> bool:
    return _command(text) in ("/links", "/sources")


def _links_file_path(name: str) -> str:
    from core.output import _safe_for_fs
    return os.path.join("downloads", _safe_for_fs(name, 50), "download_links.txt")


def handle_links(chat_id, text: str) -> None:
    """Send the per-shot 'title — source link' manifest so the user can manually
    re-download any shot that ended with no footage. Reads the file written next
    to the project XML; if it isn't on disk yet (e.g. a project still awaiting
    /download), builds it on the fly from the pending project's shots.
    ``/links <name>`` targets a specific project; bare /links uses the last one."""
    from core.output import build_download_links_txt
    parts = text.strip().split(maxsplit=1)
    name = parts[1].strip() if len(parts) > 1 else ""
    pend = _PENDING.get(chat_id)
    if not name:
        name = (pend or {}).get("project") or _LAST.get("project") or ""
    if not name:
        send_message(chat_id, "No project yet. Send a voice file, or /links <project-name>.")
        return

    path = _links_file_path(name)
    if not os.path.exists(path):
        # Not downloaded yet — build from the pending project's in-memory shots.
        shots = pend.get("shots") if (pend and pend.get("project") == name) else None
        if not shots:
            send_message(chat_id, f"No links file for '{name}' yet — it's written when the "
                                  "project's XML is built. Run /download or re-send the audio.")
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(build_download_links_txt(shots, name))
        except Exception as e:
            send_message(chat_id, f"(Couldn't build links for '{name}': {e})")
            return
    send_document(chat_id, path, caption=f"{name} — source links (title + link per shot)")


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024


def handle_zip(chat_id, text: str) -> None:
    """Bundle a project's clips + XML and deliver it (Telegram attach when small,
    plus a signed download link and the scp path). ``/zip <name>`` targets a
    specific project; bare /zip uses the last one."""
    parts = text.strip().split(maxsplit=1)
    name = parts[1].strip() if len(parts) > 1 else (_LAST.get("project") or "")
    if not name:
        send_message(chat_id, "No recent project to zip. Send a voice file first, "
                              "or use /zip <project-name>.")
        return
    send_message(chat_id, f"📦 Zipping '{name}'…")
    deliver_project(chat_id, name)


def _project_disk_usage() -> list:
    """List downloaded projects as ``(name, bytes)`` sorted largest-first."""
    root = os.path.abspath("downloads")
    out = []
    if not os.path.isdir(root):
        return out
    for name in os.listdir(root):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        total = 0
        for r, _dirs, files in os.walk(d):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(r, fn))
                except OSError:
                    pass
        out.append((name, total))
    return sorted(out, key=lambda x: x[1], reverse=True)


def _library_kept_note() -> str:
    """One-liner reassuring that cleanup frees video files only — the Clip Library
    DB (semantic reuse + learned trims) lives outside downloads/ and is kept."""
    try:
        from core.clip_library import get_library_stats
        n = get_library_stats().get("total", 0)
        return f"📚 Clip Library kept ({n} clip(s)) — only video files were removed."
    except Exception:
        return "📚 Clip Library kept — only video files were removed."


def handle_cleanup(chat_id, text: str) -> None:
    """Disk management. ``/cleanup`` lists projects + sizes (plus the shared
    overlay render cache); ``/cleanup <name>`` deletes that project's folder (and
    its .zip); ``/cleanup all`` clears every downloaded project AND the overlay
    cache; ``/cleanup overlays`` clears just the overlay render cache (the
    cross-project ProRes cache that project deletes don't touch). The Clip Library
    DB lives outside downloads/ and is always preserved. Frees space on the
    always-on server."""
    import shutil as _shutil
    parts = text.strip().split(maxsplit=1)
    arg = parts[1].strip() if len(parts) > 1 else ""
    root = os.path.abspath("downloads")

    try:
        from core.overlays_remotion import overlay_cache_size, clear_overlay_cache
    except Exception:
        overlay_cache_size = lambda: 0
        clear_overlay_cache = lambda: (0, 0)

    if not arg:
        usage = _project_disk_usage()
        ov_bytes = overlay_cache_size()
        if not usage and not ov_bytes:
            send_message(chat_id, "No downloaded projects on the server.")
            return
        total = sum(b for _, b in usage)
        lines = [f"💾 Downloads: {_human_size(total)} across {len(usage)} project(s)"]
        for name, b in usage[:25]:
            lines.append(f"  • {name} — {_human_size(b)}")
        if ov_bytes:
            lines.append(f"🅰️ Overlay render cache — {_human_size(ov_bytes)} "
                         "(shared across projects; clear with /cleanup overlays)")
        lines.append("\nDelete one: /cleanup <name>   ·   delete all: /cleanup all")
        lines.append(_library_kept_note())
        send_message(chat_id, "\n".join(lines))
        return

    if arg.lower() == "overlays":
        n, freed = clear_overlay_cache()
        send_message(chat_id, f"🗑 Cleared overlay render cache "
                              f"({n} clip(s), {_human_size(freed)} freed).")
        return

    if arg.lower() == "all":
        usage = _project_disk_usage()
        for name, _ in usage:
            _shutil.rmtree(os.path.join(root, name), ignore_errors=True)
            zp = os.path.join(root, f"{name}.zip")
            if os.path.exists(zp):
                try:
                    os.remove(zp)
                except OSError:
                    pass
        ov_n, ov_freed = clear_overlay_cache()
        msg = f"🗑 Cleared {len(usage)} project(s)"
        if ov_n:
            msg += f" + overlay cache ({_human_size(ov_freed)})"
        send_message(chat_id, msg + ". " + _library_kept_note())
        return

    # Delete a single named project (guard against path traversal).
    from core.output import _safe_for_fs
    safe = _safe_for_fs(arg, 50)
    proj_dir = os.path.join(root, safe)
    if not os.path.isdir(proj_dir):
        send_message(chat_id, f"❌ No project named '{arg}'. Use /cleanup to list them.")
        return
    _shutil.rmtree(proj_dir, ignore_errors=True)
    zp = os.path.join(root, f"{safe}.zip")
    if os.path.exists(zp):
        try:
            os.remove(zp)
        except OSError:
            pass
    if _LAST.get("project") == arg:
        _LAST["project"] = None
    send_message(chat_id, f"🗑 Deleted '{arg}'. " + _library_kept_note())


_HELP = (
    "🎬 Send a voice message or audio file and I'll build the B-roll project.\n"
    "/settings — choose sources, counts, quality, QA, review gate, text overlays\n"
    "/overlay — next voice file → animated text-overlays only (no footage)\n"
    "/overlaytext <secs> <text> — render ONE overlay clip for exact text + duration\n"
    "/status — am I online & ready\n"
    "/test — preflight: live-test LLM, transcription, Pexels + yt-dlp search and "
    "real downloads before a long run (/test quick = no downloads)\n"
    "/proxies — show the validated proxy pool (/proxies refresh to re-research the list)\n"
    "/details — per-shot breakdown of the project awaiting review\n"
    "/download — fetch clips for the reviewed project\n"
    "/refine [shots] — re-pick QA-flagged shots (or named ones, e.g. /refine 4 9)\n"
    "/redo — re-fetch shots with no clip (YouTube-first)\n"
    "/cancel — stop the running job (graceful; or discard a pending one)\n"
    "/forcestop — hard stop + restart the bot (when /cancel won't catch)\n"
    "/zip [name] — bundle a finished project (link + attach)\n"
    "/links [name] — per-shot source links (title + link) to manually re-download empty shots\n"
    "/cleanup [name|all|overlays] — list/delete projects or clear the overlay cache (free disk)\n"
    "/logs — export the bot logs as a file\n"
    "/cookies — how to give me YouTube cookies (or just send cookies.txt)\n"
    "📥 Send an exported .xml (after editing in Premiere) to teach me your preferred "
    "trims — I'll reuse those cut points when the same footage comes up again."
)


def _help_text() -> str:
    return _HELP


def _job_thread(fn, chat_id, *args) -> None:
    """Background-thread wrapper: run one unit of work and always clear busy."""
    try:
        fn(chat_id, *args)
    except Exception as e:
        send_message(chat_id, f"❌ Unexpected error: {e}")
    finally:
        _BUSY.update(active=False, project=None, cancel=None, started=None)


def build_start_keyboard(overlay_only: bool = False) -> dict:
    """Pre-job prompt: pick the mode (Full vs text-overlay-only) AND whether to
    clear the disk first. callback_data = start:<mode>:<disk>. When
    ``overlay_only`` is set (the /overlay command) only the overlay rows show.

    One button per row so the labels stay fully readable on narrow screens."""
    rows = []
    if not overlay_only:
        rows.append([{"text": "🎬 Full · clear disk", "callback_data": "start:full:clear"}])
        rows.append([{"text": "🎬 Full · keep",       "callback_data": "start:full:keep"}])
    rows.append([{"text": "🅰️ Overlays only · clear", "callback_data": "start:overlay:clear"}])
    rows.append([{"text": "🅰️ Overlays only · keep",  "callback_data": "start:overlay:keep"}])
    return {"inline_keyboard": rows}


def _clear_all_projects() -> tuple:
    """Delete every downloaded project folder and every top-level .zip under
    downloads/. Keeps .cache (Clip Library DB, cookies). Returns
    ``(project_count, bytes_freed)``."""
    import shutil as _shutil
    root = os.path.abspath("downloads")
    usage = _project_disk_usage()
    freed = sum(b for _, b in usage)
    for name, _b in usage:
        _shutil.rmtree(os.path.join(root, name), ignore_errors=True)
    try:
        for fn in os.listdir(root):
            if fn.lower().endswith(".zip"):
                fp = os.path.join(root, fn)
                try:
                    freed += os.path.getsize(fp)
                except OSError:
                    pass
                try:
                    os.remove(fp)
                except OSError:
                    pass
    except OSError:
        pass
    return len(usage), freed


def _start_job(fn, chat_id, file_id: str, name: str) -> None:
    """Claim the busy slot and run ``fn`` (handle_audio or handle_overlay_only)
    for an uploaded audio file, in a background thread."""
    _BUSY.update(active=True,
                 project=project_name_from(name, time.strftime("video_%Y%m%d_%H%M%S")),
                 cancel=threading.Event(), started=time.time())
    threading.Thread(target=_job_thread, args=(fn, chat_id, file_id, name),
                     daemon=True).start()


# ── polling loop ────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        from dotenv import load_dotenv
        # Load .env from the project root (parent of this bot/ folder) so it works
        # regardless of the current working directory the bot is launched from.
        _root_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        load_dotenv(_root_env if os.path.exists(_root_env) else ".env")
    except Exception:
        pass

    # Only strip proxy env vars when explicitly asked (TUN-mode users with stale
    # proxies). By default we keep them — for many users Telegram is reachable
    # only through their VPN's proxy, and clearing it would break the bot.
    if os.getenv("BROLL_BYPASS_HTTP_PROXY", "").strip().lower() in ("1", "true", "yes"):
        for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(_v, None)

    # Route the pipeline's traffic (Pexels/Groq/etc.) through a proxy too, so a
    # job triggered on a censored network can still reach the stock/AI APIs.
    # Prefer APP_PROXY; fall back to BOT_PROXY so one setting can cover both.
    _app_proxy = os.getenv("APP_PROXY", "").strip() or os.getenv("BOT_PROXY", "").strip()
    if _app_proxy:
        for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[_v] = _app_proxy

    # Point at a self-hosted Bot API server when TELEGRAM_API_BASE is set, which
    # lifts the 20 MB file-download cap to ~2 GB (for long/large voiceovers).
    # Resolved here — after .env is loaded — and applied to both API endpoints.
    _api_base = (os.getenv("TELEGRAM_API_BASE") or "").strip().rstrip("/")
    if _api_base:
        global _API, _FILE_API
        _API = _api_base + "/bot{token}/{method}"
        _FILE_API = _api_base + "/file/bot{token}/{path}"
        print(f"[bot] using Bot API server: {_api_base} (20 MB download cap lifted)")

    # Capture stdout/stderr to .cache/bot.log so /logs can export them.
    logsetup.install_file_logging()

    # Always-on /health endpoint so orchestrators (Coolify/Docker) can verify the
    # bot is alive and swap containers cleanly — preventing the double-poller 409.
    healthserver.start_health_server()

    if not _token():
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env (get one from @BotFather).")

    # Single-poller guard. Telegram allows only ONE getUpdates poller per token;
    # a second machine polling the same token causes a perpetual 409 Conflict and
    # lets a stale instance steal jobs (e.g. re-introducing old cookie errors).
    # Polling is therefore opt-in: only the host with BOT_POLLING_ENABLED=1 runs
    # the loop. Other machines can be cloned/updated and run the Streamlit UI or
    # one-off tasks without ever colliding with the designated bot host.
    if os.getenv("BOT_POLLING_ENABLED", "").strip().lower() not in ("1", "true", "yes", "on"):
        raise SystemExit(
            "Bot polling is disabled on this host (BOT_POLLING_ENABLED is not set).\n"
            "Set BOT_POLLING_ENABLED=1 in the environment of the ONE machine that "
            "should run the bot (e.g. the Coolify server). This guard prevents the "
            "Telegram 409 Conflict caused by two machines polling the same token."
        )

    # Register the command list so Telegram shows a '/' suggestion menu.
    register_commands()

    # Optional HTTP file server for project-zip download links (BOT_FILE_SERVER=1).
    if os.getenv("BOT_FILE_SERVER", "").strip().lower() in ("1", "true", "yes", "on"):
        _FILESERVER["port"] = fileserver.start_server()

    # Recover any projects that were paused at the review gate when the bot last
    # stopped, so /download and /refine still work after a restart / forcestop.
    restored = pending_store.load_pending()
    if restored:
        _PENDING.update(restored)
        print(f"[bot] restored {len(restored)} project(s) awaiting review.")
        for cid, pend in restored.items():
            try:
                send_message(cid, f"♻️ Restored '{pend.get('project', '?')}' — it's still "
                                  "awaiting your review. /download to proceed, /refine to fix, "
                                  "or /cancel to discard.")
            except Exception:
                pass

    allowed = allowed_user_ids()
    if not allowed:
        print("WARNING: TELEGRAM_ALLOWED_USERS is empty — every message will be ignored "
              "(fail-closed). Add your numeric Telegram user id to start accepting jobs.")
    print(f"B-Roll bot polling… (authorized users: {sorted(allowed) or 'none'})")

    offset = None
    fails = 0
    last_err = None          # signature of the previous poll error
    repeat = 0               # how many times in a row it's repeated (suppressed)
    while True:
        try:
            resp = _call("getUpdates", _timeout=60, offset=offset, timeout=50)
            fails = 0
            if repeat:
                print(f"[bot] (previous poll error repeated ×{repeat}) — recovered")
                last_err, repeat = None, 0
        except Exception as e:
            fails += 1
            # Collapse a storm of identical errors (e.g. the 409 Conflict from a
            # second poller) into one line + a periodic ×N tally instead of one
            # line per second, which used to flood /logs.
            sig = _err_signature(str(e))
            if sig == last_err:
                repeat += 1
                if repeat % 30 == 0:
                    print(f"[bot] poll error still repeating ×{repeat}: {e}")
            else:
                if repeat:
                    print(f"[bot] (previous poll error repeated ×{repeat})")
                print(f"[bot] poll error: {e}")
                last_err, repeat = sig, 1
            if fails == 3:
                # A 409 Conflict isn't a network problem — it means a SECOND
                # poller is hitting the same bot token (usually the old container
                # still running for a few seconds during a redeploy; if it
                # persists, two hosts have BOT_POLLING_ENABLED=1). Don't send the
                # VPN/proxy advice for it.
                if "409" in str(e) or "conflict" in str(e).lower():
                    print("[bot] 409 Conflict: another poller is using this bot token. "
                          "Harmless during a redeploy (old container overlapping the new "
                          "one) and clears on its own. If it never recovers, make sure only "
                          "ONE instance has BOT_POLLING_ENABLED=1.")
                else:
                    print("[bot] Repeated connection failures reaching api.telegram.org. "
                          "If Telegram needs your VPN proxy, set BOT_PROXY in .env "
                          "(e.g. BOT_PROXY=http://127.0.0.1:10809). Make sure the VPN is up.")
            time.sleep(min(5 + fails, 30))
            continue

        for upd in resp.get("result", []):
            offset = upd["update_id"] + 1

            # Inline-keyboard taps from the /settings menu.
            cb = upd.get("callback_query")
            if cb:
                if is_allowed((cb.get("from") or {}).get("id"), allowed):
                    handle_settings_callback(cb)
                continue

            msg = upd.get("message") or upd.get("channel_post")
            if not msg:
                continue
            user_id = (msg.get("from") or {}).get("id")
            chat_id = (msg.get("chat") or {}).get("id")
            if not is_allowed(user_id, allowed):
                continue
            text = msg.get("text") or ""

            if is_help_command(text):
                send_message(chat_id, _help_text())
                continue

            if is_status_command(text):
                send_message(chat_id, "🔎 Checking…")
                send_message(chat_id, format_health(
                    check_health(), _BUSY,
                    pending=[v.get("project") for v in _PENDING.values()]))
                continue

            if is_settings_command(text):
                send_settings(chat_id)
                continue

            if is_overlay_text_command(text):
                if _BUSY.get("active"):
                    send_message(chat_id, f"⏳ Busy with '{_BUSY.get('project')}'. "
                                          "Try /overlaytext once it's idle.")
                    continue
                _BUSY.update(active=True, project="overlay text",
                             cancel=threading.Event(), started=time.time())
                threading.Thread(target=_job_thread,
                                 args=(handle_overlay_text, chat_id, text),
                                 daemon=True).start()
                continue

            if is_overlay_command(text):
                _OVERLAY_NEXT.add(chat_id)
                send_message(chat_id, "🅰️ Next voice file → text-overlay only "
                                      "(animated overlays + SFX, no footage). Send it now.")
                continue

            if is_cancel_command(text):
                cancel = _BUSY.get("cancel")
                if _BUSY.get("active") and cancel is not None:
                    cancel.set()
                    send_message(chat_id, f"⏹ Cancelling '{_BUSY.get('project')}' — it stops at "
                                          "the next stage/shot. If it won't stop, use /forcestop.")
                elif _PENDING.get(chat_id):
                    proj = _PENDING.pop(chat_id)["project"]
                    _persist_pending()
                    send_message(chat_id, f"⏹ Discarded pending project '{proj}'.")
                else:
                    send_message(chat_id, "Nothing is running right now.")
                continue

            if is_forcestop_command(text):
                handle_forcestop(chat_id)
                continue

            if is_details_command(text):
                pend = _PENDING.get(chat_id)
                send_message(chat_id, format_details(pend["shots"]) if pend
                             else "No project is waiting for review.")
                continue

            if is_zip_command(text):
                handle_zip(chat_id, text)
                continue

            if is_links_command(text):
                handle_links(chat_id, text)
                continue

            if is_cleanup_command(text):
                handle_cleanup(chat_id, text)
                continue

            if is_logs_command(text):
                handle_logs(chat_id)
                continue

            if is_test_command(text):
                if _BUSY.get("active"):
                    send_message(chat_id, f"⏳ Busy with '{_BUSY.get('project')}'. "
                                          "Run /test once it's idle.")
                    continue
                _BUSY.update(active=True, project="preflight self-test",
                             cancel=threading.Event(), started=time.time())
                threading.Thread(target=_job_thread,
                                 args=(handle_selftest, chat_id, text),
                                 daemon=True).start()
                continue

            if is_proxies_command(text):
                from core import proxy_pool as _pp
                # Research (slow) when the user asks for a refresh OR the pool is
                # still empty (so a bare /proxies actually tests + reports instead
                # of showing a useless "0 working"). A populated view is instant.
                _empty = _pp.pool_active() and _pp.stats().get("working", 0) == 0
                if _pp.pool_active() and ("refresh" in text.lower()
                                          or "research" in text.lower() or _empty):
                    if _BUSY.get("active"):
                        send_message(chat_id, f"⏳ Busy with '{_BUSY.get('project')}'. "
                                              "Check proxies once it's idle.")
                        continue
                    _BUSY.update(active=True, project="proxy research",
                                 cancel=threading.Event(), started=time.time())
                    threading.Thread(target=_job_thread,
                                     args=(handle_proxies, chat_id, "refresh"),
                                     daemon=True).start()
                else:
                    handle_proxies(chat_id, text)
                continue

            if is_cookies_command(text):
                send_message(chat_id, _COOKIES_HELP)
                continue

            # A cookies.txt document upload — save & use it (before audio routing).
            cfid, cname = extract_cookies_doc(msg)
            if cfid:
                handle_cookies_upload(chat_id, cfid, cname)
                continue

            # A clip_library.db upload — merge it into the server's library.
            lfid, lname = extract_library_doc(msg)
            if lfid:
                handle_library_upload(chat_id, lfid, lname)
                continue

            # An exported FCP7 XML upload — learn the editor's preferred trims.
            xfid, xname = extract_xml_doc(msg)
            if xfid:
                handle_xml_upload(chat_id, xfid, xname)
                continue

            # Review-gate actions: need a pending project and a free worker.
            if is_download_command(text) or is_refine_command(text) or is_redo_command(text):
                if _BUSY.get("active"):
                    send_message(chat_id, f"⏳ Busy with '{_BUSY.get('project')}'. One moment…")
                    continue
                if not _PENDING.get(chat_id):
                    send_message(chat_id, "Nothing to act on — send a voice file first.")
                    continue
                _BUSY.update(active=True, project=_PENDING[chat_id]["project"],
                             cancel=threading.Event(), started=time.time())
                if is_download_command(text):
                    threading.Thread(target=_job_thread, args=(_run_download, chat_id),
                                     daemon=True).start()
                elif is_redo_command(text):
                    threading.Thread(target=_job_thread, args=(_run_redo, chat_id),
                                     daemon=True).start()
                else:
                    slots = _parse_slots(text)
                    threading.Thread(target=_job_thread,
                                     args=(_run_refine, chat_id, slots or None),
                                     daemon=True).start()
                continue

            file_id, name = extract_audio(msg)
            if file_id:
                # Telegram won't let a bot download a file over 20 MB — catch it
                # here with a clear fix instead of an opaque getFile 400 later.
                size = _audio_size(msg)
                if size > _TG_DOWNLOAD_LIMIT and not _using_local_bot_api():
                    send_message(chat_id, too_big_message(size))
                    continue
                if _BUSY.get("active"):
                    send_message(chat_id, f"⏳ Still processing '{_BUSY.get('project')}'. "
                                          "Send /cancel to stop it, or wait and resend.")
                    continue
                if _PENDING.get(chat_id):
                    send_message(chat_id, f"📋 '{_PENDING[chat_id]['project']}' is awaiting review. "
                                          "Reply /download, /refine, or /cancel first.")
                    continue
                # Ask the mode (full vs text-overlay-only) and whether to clear
                # disk first. The job starts from the button handler once the user
                # answers. /overlay pre-restricts the prompt to overlay-only.
                overlay_only = chat_id in _OVERLAY_NEXT
                _OVERLAY_NEXT.discard(chat_id)
                _PENDING_START[chat_id] = {"file_id": file_id, "name": name}
                send_message(
                    chat_id,
                    f"🎬 Ready: '{project_name_from(name, 'voice')}'.\n"
                    "Pick a mode — and whether to clear old projects first "
                    "(frees disk; keeps Clip Library & cookies):",
                    reply_markup=build_start_keyboard(overlay_only=overlay_only),
                )
            elif text:
                send_message(chat_id, _help_text())


if __name__ == "__main__":
    main()
