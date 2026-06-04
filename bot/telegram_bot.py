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
import json
import time
import shutil
import threading

# Cap native thread pools (torch/OMP/MKL) before anything imports them, so the
# CPU-only server doesn't oversubscribe cores. See core/runtime.py.
from core.runtime import configure_runtime
configure_runtime()

import requests

from bot import settings as bot_settings
from bot import fileserver

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
# + QA so /download, /refine, /details can act on them between messages.
_PENDING: dict = {}

# File server port (set in main() when BOT_FILE_SERVER is enabled), used to build
# download links.
_FILESERVER = {"port": None}

# Last completed project name, so /zip knows what to bundle by default.
_LAST = {"project": None}


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


def project_name_from(suggested_name: str, fallback: str = "") -> str:
    base = os.path.splitext(os.path.basename(suggested_name or ""))[0].strip()
    return (base or fallback or "voice")[:50]


# ── Telegram API (thin wrappers over requests) ─────────────────────────────────

def _call(method: str, _timeout: int = 60, **params) -> dict:
    resp = requests.post(_API.format(token=_token(), method=method), data=params,
                         timeout=_timeout, proxies=_proxies())
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


def is_download_command(text: str) -> bool:
    return _command(text) in ("/download", "/go", "/approve", "/accept")


def is_refine_command(text: str) -> bool:
    return _command(text) in ("/refine", "/improve", "/redo")


def is_details_command(text: str) -> bool:
    return _command(text) in ("/details", "/detail", "/shots")


def is_help_command(text: str) -> bool:
    return _command(text) in ("/help", "/start", "/commands")


def is_cleanup_command(text: str) -> bool:
    return _command(text) in ("/cleanup", "/clean", "/disk")


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


def format_errors_block(errors: list, limit: int = 6) -> list:
    if not errors:
        return []
    lines = [f"❗ {len(errors)} issue(s) during processing:"]
    for e in errors[:limit]:
        lines.append(f"  • {str(e)[:200]}")
    if len(errors) > limit:
        lines.append(f"  …and {len(errors) - limit} more")
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
    try:
        res = zip_project(project)
    except FileNotFoundError:
        send_message(chat_id, f"(No files to bundle for '{project}'.)")
        return
    except Exception as e:
        send_message(chat_id, f"(Couldn't zip '{project}': {e})")
        return

    abs_path = os.path.abspath(res["path"])
    size = res["size_bytes"]
    lines = [f"📦 {project} — {res['files']} file(s), {_human_size(size)}"]

    port = _FILESERVER.get("port")
    if port:
        link = fileserver.build_link(abs_path, fileserver.public_host(), port)
        if link:
            lines.append(f"🔗 {link}\n(link expires in 24h)")
    lines.append(f"scp USER@SERVER:'{abs_path}' .")
    send_message(chat_id, "\n".join(lines))

    if size <= _TG_UPLOAD_LIMIT and os.path.exists(abs_path):
        send_document(chat_id, abs_path, caption=f"{project} — clips + XML")


# ── job ────────────────────────────────────────────────────────────────────────

def format_summary(proj: str, result: dict) -> str:
    dl = result.get("download") or {}
    qa = result.get("qa") or {}
    n_issues = len(qa.get("issues") or [])
    lines = [
        f"✅ Done — {proj}",
        f"Shots: {result.get('n_shots', 0)}  ·  with clips: {result.get('n_selected', 0)}  ·  clips: {result.get('n_clips', 0)}",
    ]
    if result.get("download") is not None:
        lines.append(f"Downloaded: {dl.get('ok', 0)} ok, {dl.get('failed', 0)} failed, {dl.get('skipped', 0)} cached")
    verdict = qa.get("overall") or "—"
    lines.append(f"QA: {verdict}" + (f"  ⚠️ {n_issues} flag(s)" if n_issues else ""))
    if dl.get("dir"):
        lines.append(f"📁 {dl['dir']}")
    return "\n".join(lines)


def _progress_logger(chat_id, proj, msg_id):
    """App-style running step log: appends each stage (✅ done, ⏳ current) to a
    single edited message."""
    steps: list = []

    def _cb(step, total, label):
        steps.append(label)
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
        send_message(chat_id, format_review(proj, result))
        send_shots_srt(chat_id, proj, result.get("shots") or [])
    else:
        _deliver_completed(chat_id, proj, result)
    return result


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

    try:
        with bot_settings.apply_env(settings):
            fin = finalize_project(pend["shots"], proj, quality=str(settings.get("quality", 1080)),
                                   should_cancel=_should_cancel(), progress=_p)
    except PipelineCancelled:
        send_message(chat_id, f"⏹ Cancelled '{proj}'.")
        return
    except Exception as e:
        send_message(chat_id, f"❌ Download failed for '{proj}': {e}")
        return
    result["download"] = fin["download"]
    result["xml_path"] = fin["xml_path"]
    _PENDING.pop(chat_id, None)
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
    edit_message(chat_id, msg_id, f"🛠 {proj} — refined {n} shot(s).")
    send_message(chat_id, format_review(proj, result))


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


def format_health(checks: list, busy: dict = None) -> str:
    lines = ["🟢 Bot online — laptop is up and polling."]
    for name, ok, detail in checks:
        lines.append(f"{'✅' if ok else '❌'} {name}: {detail}")
    if busy and busy.get("active"):
        lines.append(f"⏳ Busy — processing '{busy.get('project', 'a job')}'. Try again when it's done.")
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


def is_zip_command(text: str) -> bool:
    return _command(text) in ("/zip", "/package", "/bundle")


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


def handle_cleanup(chat_id, text: str) -> None:
    """Disk management. ``/cleanup`` lists projects + sizes; ``/cleanup <name>``
    deletes that project's folder (and its .zip); ``/cleanup all`` clears every
    downloaded project. Frees space on the always-on server."""
    import shutil as _shutil
    parts = text.strip().split(maxsplit=1)
    arg = parts[1].strip() if len(parts) > 1 else ""
    root = os.path.abspath("downloads")

    if not arg:
        usage = _project_disk_usage()
        if not usage:
            send_message(chat_id, "No downloaded projects on the server.")
            return
        total = sum(b for _, b in usage)
        lines = [f"💾 Downloads: {_human_size(total)} across {len(usage)} project(s)"]
        for name, b in usage[:25]:
            lines.append(f"  • {name} — {_human_size(b)}")
        lines.append("\nDelete one: /cleanup <name>   ·   delete all: /cleanup all")
        send_message(chat_id, "\n".join(lines))
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
        send_message(chat_id, f"🗑 Cleared {len(usage)} project(s).")
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
    send_message(chat_id, f"🗑 Deleted '{arg}'.")


_HELP = (
    "🎬 Send a voice message or audio file and I'll build the B-roll project.\n"
    "/settings — choose sources, counts, quality, QA, review gate\n"
    "/status — am I online & ready\n"
    "/details — per-shot breakdown of the project awaiting review\n"
    "/download — fetch clips for the reviewed project\n"
    "/refine [shots] — re-pick QA-flagged shots (or named ones, e.g. /refine 4 9)\n"
    "/cancel — stop the running job (or discard a pending one)\n"
    "/zip [name] — bundle a finished project (link + attach)\n"
    "/cleanup [name|all] — list or delete downloaded projects (free disk)"
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
        _BUSY.update(active=False, project=None, cancel=None)


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

    if not _token():
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env (get one from @BotFather).")

    # Optional HTTP file server for project-zip download links (BOT_FILE_SERVER=1).
    if os.getenv("BOT_FILE_SERVER", "").strip().lower() in ("1", "true", "yes", "on"):
        _FILESERVER["port"] = fileserver.start_server()

    allowed = allowed_user_ids()
    if not allowed:
        print("WARNING: TELEGRAM_ALLOWED_USERS is empty — every message will be ignored "
              "(fail-closed). Add your numeric Telegram user id to start accepting jobs.")
    print(f"B-Roll bot polling… (authorized users: {sorted(allowed) or 'none'})")

    offset = None
    fails = 0
    while True:
        try:
            resp = _call("getUpdates", _timeout=60, offset=offset, timeout=50)
            fails = 0
        except Exception as e:
            fails += 1
            print(f"[bot] poll error: {e}")
            if fails == 3:
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
                send_message(chat_id, format_health(check_health(), _BUSY))
                continue

            if is_settings_command(text):
                send_settings(chat_id)
                continue

            if is_cancel_command(text):
                cancel = _BUSY.get("cancel")
                if _BUSY.get("active") and cancel is not None:
                    cancel.set()
                    send_message(chat_id, f"⏹ Cancelling '{_BUSY.get('project')}' "
                                          "after the current stage…")
                elif _PENDING.get(chat_id):
                    proj = _PENDING.pop(chat_id)["project"]
                    send_message(chat_id, f"⏹ Discarded pending project '{proj}'.")
                else:
                    send_message(chat_id, "Nothing is running right now.")
                continue

            if is_details_command(text):
                pend = _PENDING.get(chat_id)
                send_message(chat_id, format_details(pend["shots"]) if pend
                             else "No project is waiting for review.")
                continue

            if is_zip_command(text):
                handle_zip(chat_id, text)
                continue

            if is_cleanup_command(text):
                handle_cleanup(chat_id, text)
                continue

            # Review-gate actions: need a pending project and a free worker.
            if is_download_command(text) or is_refine_command(text):
                if _BUSY.get("active"):
                    send_message(chat_id, f"⏳ Busy with '{_BUSY.get('project')}'. One moment…")
                    continue
                if not _PENDING.get(chat_id):
                    send_message(chat_id, "Nothing to act on — send a voice file first.")
                    continue
                _BUSY.update(active=True, project=_PENDING[chat_id]["project"],
                             cancel=threading.Event())
                if is_download_command(text):
                    threading.Thread(target=_job_thread, args=(_run_download, chat_id),
                                     daemon=True).start()
                else:
                    slots = _parse_slots(text)
                    threading.Thread(target=_job_thread,
                                     args=(_run_refine, chat_id, slots or None),
                                     daemon=True).start()
                continue

            file_id, name = extract_audio(msg)
            if file_id:
                if _BUSY.get("active"):
                    send_message(chat_id, f"⏳ Still processing '{_BUSY.get('project')}'. "
                                          "Send /cancel to stop it, or wait and resend.")
                    continue
                if _PENDING.get(chat_id):
                    send_message(chat_id, f"📋 '{_PENDING[chat_id]['project']}' is awaiting review. "
                                          "Reply /download, /refine, or /cancel first.")
                    continue
                # Claim the slot in THIS thread (avoids a race with a second file),
                # then run the job in the background so the loop keeps polling and
                # can answer /status and /cancel mid-job.
                _BUSY.update(active=True,
                             project=project_name_from(name, time.strftime("video_%Y%m%d_%H%M%S")),
                             cancel=threading.Event())
                threading.Thread(target=_job_thread, args=(handle_audio, chat_id, file_id, name),
                                 daemon=True).start()
            elif text:
                send_message(chat_id, _help_text())


if __name__ == "__main__":
    main()
