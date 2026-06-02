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
import time
import shutil

import requests

_API = "https://api.telegram.org/bot{token}/{method}"
_FILE_API = "https://api.telegram.org/file/bot{token}/{path}"
_AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".oga", ".aac", ".flac")

# Whether a pipeline job is currently running (so /status can report it and a
# second audio file can be politely deferred instead of overloading the host).
_BUSY = {"active": False, "project": None}


# ── config ────────────────────────────────────────────────────────────────────

def _token() -> str:
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


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
    resp = requests.post(_API.format(token=_token(), method=method), data=params, timeout=_timeout)
    resp.raise_for_status()
    return resp.json()


def send_message(chat_id, text: str) -> dict:
    try:
        return _call("sendMessage", chat_id=chat_id, text=text).get("result", {})
    except Exception as e:
        print(f"[bot] sendMessage failed: {e}")
        return {}


def edit_message(chat_id, message_id, text: str) -> None:
    if not message_id:
        return
    try:
        _call("editMessageText", chat_id=chat_id, message_id=message_id, text=text)
    except Exception:
        pass  # editing is best-effort (rate limits / identical text)


def download_telegram_file(file_id: str, dest_path: str) -> str:
    path = _call("getFile", file_id=file_id)["result"]["file_path"]
    url = _FILE_API.format(token=_token(), path=path)
    with requests.get(url, stream=True, timeout=180) as r:
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
                files={"document": f}, timeout=180,
            )
        resp.raise_for_status()
    except Exception as e:
        send_message(chat_id, f"(Couldn't attach the XML: {e})")


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


def handle_audio(chat_id, file_id: str, suggested_name: str) -> dict:
    """Download the audio and run the headless pipeline, reporting progress back
    to the chat. Returns the pipeline result (or ``{}`` on early failure)."""
    from core.pipeline import run_pipeline_headless

    proj = project_name_from(suggested_name, time.strftime("video_%Y%m%d_%H%M%S"))
    ext = os.path.splitext(suggested_name or "")[1].lower() or ".mp3"
    audio_path = os.path.join(".cache", f"bot_{proj}{ext}")

    status = send_message(chat_id, f"🎬 Received '{proj}'. Downloading…")
    msg_id = status.get("message_id")

    try:
        download_telegram_file(file_id, audio_path)
    except Exception as e:
        send_message(chat_id, f"❌ Couldn't fetch the audio: {e}")
        return {}

    def _progress(step, total, label):
        edit_message(chat_id, msg_id, f"⚙️ [{step}/{total}] {label}…  ·  {proj}")

    _BUSY.update(active=True, project=proj)
    try:
        result = run_pipeline_headless(audio_path, project_name=proj, download=True,
                                       progress_callback=_progress)
    except Exception as e:
        send_message(chat_id, f"❌ Pipeline failed for '{proj}': {e}")
        return {}
    finally:
        _BUSY.update(active=False, project=None)

    send_message(chat_id, format_summary(proj, result))
    xml_path = result.get("xml_path")
    if xml_path and os.path.exists(xml_path):
        send_document(chat_id, xml_path, caption=f"{proj} — Premiere XML")
    return result


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
            requests.head(url, timeout=timeout, allow_redirects=True)
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


def is_status_command(text: str) -> bool:
    if not text:
        return False
    cmd = text.strip().split()[0].split("@")[0].lower()
    return cmd in ("/status", "/health", "/ping")


# ── polling loop ────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(".env")
    except Exception:
        pass

    if not _token():
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env (get one from @BotFather).")
    allowed = allowed_user_ids()
    if not allowed:
        print("WARNING: TELEGRAM_ALLOWED_USERS is empty — every message will be ignored "
              "(fail-closed). Add your numeric Telegram user id to start accepting jobs.")
    print(f"B-Roll bot polling… (authorized users: {sorted(allowed) or 'none'})")

    offset = None
    while True:
        try:
            resp = _call("getUpdates", _timeout=60, offset=offset, timeout=50)
        except Exception as e:
            print(f"[bot] poll error: {e}")
            time.sleep(5)
            continue

        for upd in resp.get("result", []):
            offset = upd["update_id"] + 1
            msg = upd.get("message") or upd.get("channel_post")
            if not msg:
                continue
            user_id = (msg.get("from") or {}).get("id")
            chat_id = (msg.get("chat") or {}).get("id")
            if not is_allowed(user_id, allowed):
                continue
            text = msg.get("text") or ""
            if is_status_command(text):
                send_message(chat_id, "🔎 Checking…")
                send_message(chat_id, format_health(check_health(), _BUSY))
                continue
            file_id, name = extract_audio(msg)
            if file_id:
                if _BUSY.get("active"):
                    send_message(chat_id, f"⏳ Still processing '{_BUSY.get('project')}'. "
                                          "I'll be free once it finishes — send it again then.")
                    continue
                try:
                    handle_audio(chat_id, file_id, name)
                except Exception as e:
                    _BUSY.update(active=False, project=None)
                    send_message(chat_id, f"❌ Unexpected error: {e}")
            elif text:
                send_message(chat_id, "Send me a voice message or an audio file and I'll build "
                                      "the B-roll project. Send /status to check I'm online and ready.")


if __name__ == "__main__":
    main()
