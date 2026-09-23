"""
Segment library — the channel's own growing stock of verified footage.

Every cut an editor actually used (read from their returned Premiere XML) and
every good part a reviewer marks becomes a **segment**: a stored, trimmed piece
of a source video with a description of what's on screen. Reviewers then check
each segment on the rating page, and the picker searches this library first for
every new narration line.

What a segment knows
    source         where it came from (page URL, channel), in/out on the source
    file           a trimmed copy on the server, so it never depends on the
                   source still being online and places instantly
    description    what's on screen, independent of any script line
    subject        the exact thing shown ("2019 Honda Civic", "Bosch oil filter")
    identifiable   can a viewer tell it's that exact subject?  This is what
                   decides where it may be used:
                     identifiable  → only for lines about that subject
                     not identifiable → fine as general footage (generic_use)
    generic_use    what it can stand in for ("oil draining from a car engine")
    trust          suggested · used · verified · avoid  (see TRUST)
    lines          the narration lines it served, and whether it fit them

So an oil change on a Camry with the badge in shot is used for Camry lines
only; the same action with no badge or model visible is general oil-change
footage for any car.

Files live under ``.cache/library/segments`` (the persistent volume); override
with ``SEGMENT_LIBRARY_DIR``. Records are in the project DB.
"""

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from datetime import datetime

import numpy as np

from core import project_store

_ROOT = os.path.join(os.path.dirname(__file__), "..", ".cache", "library", "segments")

# trust levels, weakest to strongest; 'avoid' is never offered.
TRUST = ("suggested", "used", "verified", "avoid")
_TRUST_BOOST = {"suggested": 0.0, "used": 0.03, "verified": 0.08}

SHOT_TYPES = [("close_up", "Close-up"), ("detail", "Detail / macro"), ("medium", "Medium"),
              ("wide", "Wide"), ("aerial", "Aerial / drone"), ("pov", "POV / in-car"),
              ("screen", "Screen / graphic")]
PROBLEMS = [("text_logo", "Text, logo or watermark"), ("person", "Presenter / face"),
            ("shaky", "Shaky"), ("low_res", "Low quality"), ("vertical", "Vertical / cropped"),
            ("dark", "Too dark")]
_SHOT_IDS = {s for s, _ in SHOT_TYPES}
_PROBLEM_IDS = {p for p, _ in PROBLEMS}

# Cosine floor for a library match (MiniLM). Measured: right matches score
# ~0.54-0.61, unrelated footage < 0.2 — the ranker makes the final call.
_MIN_SCORE = 0.45
_STRONG_SCORE = 0.62      # verified + this strong → promoted to the top after ranking
_MIN_SECONDS = 1.0

# Repeat control, counted in delivered videos (the run in progress and failed
# runs don't count). Each is env-tunable under the name in brackets.
_REUSE_GAP = 3                 # [LIBRARY_REUSE_GAP] skip a segment used in the last N videos
_REUSE_GAP_IDENTIFIABLE = 10   # [LIBRARY_REUSE_GAP_IDENTIFIABLE] same for a recognisable
                               # subject (that exact car, that shop), which viewers spot sooner
_USE_WINDOW = 20               # [LIBRARY_USE_WINDOW] uses within the last N videos…
_USE_PENALTY = 0.03            # [LIBRARY_USE_PENALTY] …each lower the match score by this,
                               # so a few favourites can't win every video
_SHARE_WINDOW = 10             # /library reports the library share over the last N videos
_DELIVERED = ("delivered", "learned")

_WORKER = {"thread": None, "kick": threading.Event(), "lock": threading.Lock()}


def enabled() -> bool:
    return os.getenv("SEGMENT_LIBRARY", "1").strip().lower() not in ("0", "false", "no", "off")


def _env_num(name: str, default, cast=int):
    try:
        return cast(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def lib_dir() -> str:
    d = os.getenv("SEGMENT_LIBRARY_DIR", "").strip() or os.path.abspath(_ROOT)
    os.makedirs(d, exist_ok=True)
    return d


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_db() -> None:
    project_store.init_db()
    with project_store._conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS segments (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                page_url        TEXT    NOT NULL,
                url             TEXT    DEFAULT '',
                source          TEXT    DEFAULT '',
                channel         TEXT    DEFAULT '',
                title           TEXT    DEFAULT '',
                src_in          REAL    NOT NULL,
                src_out         REAL    NOT NULL,
                src_file        TEXT    DEFAULT '',     -- where to cut from, if known
                src_zip         TEXT    DEFAULT '',     -- or a zip + member
                src_member      TEXT    DEFAULT '',
                file_path       TEXT    DEFAULT '',
                file_status     TEXT    DEFAULT 'pending',  -- pending|ready|failed
                file_error      TEXT    DEFAULT '',
                frames          TEXT    DEFAULT '[]',
                description     TEXT    DEFAULT '',
                subject         TEXT    DEFAULT '',
                identifiable    INTEGER,                -- NULL unknown · 1 · 0
                generic_use     TEXT    DEFAULT '',
                shot_type       TEXT    DEFAULT '',
                problems        TEXT    DEFAULT '[]',
                draft_source    TEXT    DEFAULT '',     -- vision | text | reviewer
                trust           TEXT    DEFAULT 'used',
                good_votes      INTEGER DEFAULT 0,
                bad_votes       INTEGER DEFAULT 0,
                n_reviews       INTEGER DEFAULT 0,
                times_used      INTEGER DEFAULT 0,
                times_kept      INTEGER DEFAULT 0,
                times_dropped   INTEGER DEFAULT 0,
                embedding       BLOB,
                origin_project  INTEGER,
                created_at      TEXT    DEFAULT '',
                updated_at      TEXT    DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_segments_page ON segments(page_url);
            CREATE TABLE IF NOT EXISTS segment_lines (
                segment_id  INTEGER NOT NULL,
                project_id  INTEGER,
                slot_id     TEXT    DEFAULT '',
                line_text   TEXT    DEFAULT '',
                fit         TEXT    DEFAULT 'served',   -- served | yes | partly | no
                created_at  TEXT    DEFAULT '',
                UNIQUE (segment_id, project_id, slot_id)
            );
            CREATE TABLE IF NOT EXISTS segment_reviews (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id   INTEGER NOT NULL,
                rater_id     INTEGER NOT NULL,
                fits_line    TEXT    DEFAULT '',
                usable       TEXT    NOT NULL,        -- yes | no
                description  TEXT    DEFAULT '',
                subject      TEXT    DEFAULT '',
                identifiable INTEGER,
                generic_use  TEXT    DEFAULT '',
                shot_type    TEXT    DEFAULT '',
                problems     TEXT    DEFAULT '[]',
                note         TEXT    DEFAULT '',
                created_at   TEXT    DEFAULT '',
                UNIQUE (segment_id, rater_id)
            );
            CREATE TABLE IF NOT EXISTS segment_uses (
                segment_id  INTEGER NOT NULL,
                project_id  INTEGER NOT NULL,
                created_at  TEXT    DEFAULT '',
                UNIQUE (segment_id, project_id)
            );
        """)
        # Stills share the table: kind 'image', no in/out, origin_page = the web
        # page the image came from (for checking reuse rights).
        for col in ("kind TEXT DEFAULT 'clip'", "origin_page TEXT DEFAULT ''"):
            try:
                c.execute(f"ALTER TABLE segments ADD COLUMN {col}")
            except Exception:
                pass


def _row(r) -> dict:
    d = dict(r)
    for k in ("frames", "problems"):
        if k in d:
            try:
                d[k] = json.loads(d[k] or "[]")
            except ValueError:
                d[k] = []
    return d


def get_segment(segment_id: int) -> dict | None:
    init_db()
    with project_store._conn() as c:
        r = c.execute("SELECT * FROM segments WHERE id=?", (segment_id,)).fetchone()
    return _row(r) if r else None


# ── adding segments ─────────────────────────────────────────────────────────

def _overlap(a0, a1, b0, b1) -> float:
    inter = min(a1, b1) - max(a0, b0)
    return inter / max(1e-6, min(a1 - a0, b1 - b0)) if inter > 0 else 0.0


def add_segment(page_url: str, src_in: float, src_out: float, *, url: str = "",
                source: str = "", channel: str = "", title: str = "", src_file: str = "",
                src_zip: str = "", src_member: str = "", trust: str = "used",
                project_id=None, slot_id="", line_text: str = "") -> tuple:
    """Add a segment, or merge into an existing one of the same source that
    covers mostly the same part. Returns ``(segment_id, created)``. A merge keeps
    the stronger trust level and records the extra line it served."""
    init_db()
    src_in, src_out = max(0.0, float(src_in)), float(src_out)
    if not page_url or src_out - src_in < _MIN_SECONDS:
        raise ValueError("segment too short or has no source")
    now = _now()
    with project_store._conn() as c:
        existing = [dict(r) for r in c.execute(
            "SELECT id, src_in, src_out, trust FROM segments WHERE page_url=?", (page_url,))]
        match = next((e for e in existing
                      if _overlap(src_in, src_out, e["src_in"], e["src_out"]) >= 0.5), None)
        if match:
            sid, created = match["id"], False
            if TRUST.index(trust) > TRUST.index(match["trust"]) and match["trust"] != "avoid":
                c.execute("UPDATE segments SET trust=?, updated_at=? WHERE id=?",
                          (trust, now, sid))
        else:
            cur = c.execute(
                """INSERT INTO segments (page_url, url, source, channel, title, src_in, src_out,
                                         src_file, src_zip, src_member, trust, origin_project,
                                         created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (page_url, url or page_url, source or "", channel or "", (title or "")[:300],
                 src_in, src_out, src_file or "", src_zip or "", src_member or "", trust,
                 project_id, now, now))
            sid, created = cur.lastrowid, True
        if line_text:
            c.execute("""INSERT OR IGNORE INTO segment_lines
                         (segment_id, project_id, slot_id, line_text, created_at)
                         VALUES (?, ?, ?, ?, ?)""",
                      (sid, project_id, str(slot_id or ""), line_text, now))
    return sid, created


def add_image(image_url: str, *, page: str = "", title: str = "", src_file: str = "",
              src_zip: str = "", src_member: str = "", trust: str = "used",
              project_id=None, slot_id="", line_text: str = "") -> tuple:
    """Add a still image to the library (or record another line it served).
    Identity is the image URL. Returns ``(segment_id, created)``."""
    init_db()
    if not image_url:
        raise ValueError("image has no URL")
    now = _now()
    with project_store._conn() as c:
        row = c.execute("SELECT id, trust FROM segments WHERE kind='image' AND page_url=?",
                        (image_url,)).fetchone()
        if row:
            sid, created = row["id"], False
            if TRUST.index(trust) > TRUST.index(row["trust"]) and row["trust"] != "avoid":
                c.execute("UPDATE segments SET trust=?, updated_at=? WHERE id=?", (trust, now, sid))
        else:
            cur = c.execute(
                """INSERT INTO segments (kind, page_url, url, source, title, origin_page, src_in,
                                         src_out, src_file, src_zip, src_member, trust,
                                         origin_project, created_at, updated_at)
                   VALUES ('image', ?, ?, 'google_image', ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, ?)""",
                (image_url, image_url, (title or "")[:300], page or "", src_file or "",
                 src_zip or "", src_member or "", trust, project_id, now, now))
            sid, created = cur.lastrowid, True
        if line_text:
            c.execute("""INSERT OR IGNORE INTO segment_lines
                         (segment_id, project_id, slot_id, line_text, created_at)
                         VALUES (?, ?, ?, ?, ?)""",
                      (sid, project_id, str(slot_id or ""), line_text, now))
    return sid, created


def _find_delivered(project: dict, filename: str) -> tuple:
    """``(local_path, zip_path, member)`` for any delivered file of a project —
    searches the project folder, then its zip (clips live in director/, images
    in images/shots/shot_NN/)."""
    from core.output import _safe_for_fs
    safe = _safe_for_fs(project.get("project_name") or project.get("title") or "", 50)
    root = os.path.join(os.path.abspath("downloads"), safe)
    for dirpath, _dirs, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename), "", ""
    zp = os.path.join(os.path.abspath("downloads"), f"{safe}.zip")
    if os.path.isfile(zp):
        try:
            with zipfile.ZipFile(zp) as z:
                member = next((n for n in z.namelist() if n.endswith("/" + filename)), "")
            if member:
                return "", zp, member
        except zipfile.BadZipFile:
            pass
    return "", "", ""


def _source_locations(project: dict, filename: str) -> tuple:
    """Where a delivered clip's file may still be: the project folder, or its zip."""
    from core.output import _safe_for_fs, clip_base_dir
    name = project.get("project_name") or project.get("title") or ""
    path = os.path.join(clip_base_dir(name), filename)
    safe = _safe_for_fs(name, 50)
    zp = os.path.join(os.path.abspath("downloads"), f"{safe}.zip")
    return (path if os.path.isfile(path) else ""), \
           (zp if os.path.isfile(zp) else ""), f"{safe}/director/{filename}"


def ingest_from_import(project_id: int) -> int:
    """Turn every cut and every still image the editor used in a learned project
    into a library entry (trust 'used'). Items that were themselves library
    entries update that entry's record instead. Returns how many are new."""
    init_db()
    project = project_store.get_project(project_id) or {}
    with project_store._conn() as c:
        rows = [dict(r) for r in c.execute(
            """SELECT u.in_sec, u.out_sec, u.used_slot_id, u.verdict, a.id AS asset_id,
                      a.kind, a.slot_id, a.page_url, a.url, a.source, a.channel, a.title,
                      a.filename, a.segment_id
                 FROM asset_usage u JOIN project_assets a ON a.id = u.asset_id
                WHERE u.project_id=? AND a.kind IN ('clip', 'image')
                  AND u.verdict IN ('used_here','used_elsewhere')""", (project_id,))]
        lines = {r["slot_id"]: r["text"] for r in c.execute(
            "SELECT slot_id, text FROM project_shots WHERE project_id=?", (project_id,))}
        used_segments = {r[0]: r[1] for r in c.execute(
            """SELECT segment_id, MAX(verdict IN ('used_here','used_elsewhere'))
                 FROM project_assets WHERE project_id=? AND segment_id IS NOT NULL
                GROUP BY segment_id""", (project_id,))}
    # Library segments the editor kept or dropped in this edit.
    with project_store._conn() as c:
        for sid, kept in used_segments.items():
            col = "times_kept" if kept else "times_dropped"
            c.execute(f"UPDATE segments SET {col} = {col} + 1, updated_at=? WHERE id=?",
                      (_now(), sid))
    added = 0
    for r in rows:
        if r["segment_id"]:
            continue
        slot = r["used_slot_id"] or r["slot_id"]
        if r["kind"] == "image":
            local, zp, member = _find_delivered(project, r["filename"])
            try:
                _sid, created = add_image(
                    r["url"], page=r["page_url"], title=r["title"], src_file=local,
                    src_zip=zp, src_member=member, trust="used", project_id=project_id,
                    slot_id=slot, line_text=lines.get(slot, ""))
                added += created
            except ValueError:
                pass
            continue
        if r["in_sec"] is None or r["out_sec"] is None:
            continue
        local, zp, member = _source_locations(project, r["filename"])
        try:
            _sid, created = add_segment(
                r["page_url"] or r["url"], r["in_sec"], r["out_sec"], url=r["url"],
                source=r["source"], channel=r["channel"], title=r["title"], src_file=local,
                src_zip="" if local else zp, src_member="" if local else member,
                trust="used", project_id=project_id, slot_id=slot,
                line_text=lines.get(slot, ""))
            added += created
        except ValueError:
            continue
    if added:
        kick()
    return added


def ingest_from_ratings() -> int:
    """Good parts reviewers marked on the rating page (usable yes/trim) become
    'suggested' segments. Their source is downloaded by the worker."""
    init_db()
    from core import ratings
    added = 0
    for l in ratings.item_labels():
        if l["yes"] <= l["no"]:
            continue
        with project_store._conn() as c:
            line = c.execute("SELECT text FROM project_shots WHERE project_id=? AND slot_id=?",
                             (l["project_id"], l["slot_id"])).fetchone()
        for segs in l["segments"]:
            for a, b in segs:
                try:
                    _sid, created = add_segment(
                        l["page_url"], a, b, url=l["url"], source=l["source"],
                        channel=l["channel"], title=l["title"], trust="suggested",
                        project_id=l["project_id"], slot_id=l["slot_id"],
                        line_text=line["text"] if line else "")
                    added += created
                except ValueError:
                    continue
    if added:
        kick()
    return added


# ── the worker: cut, look at, describe ─────────────────────────────────────

def _ffmpeg(args: list, timeout: int = 300) -> None:
    from core.ffmpeg_utils import _FFMPEG_THREADS
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-threads",
           str(_FFMPEG_THREADS)] + args
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError((res.stderr or "ffmpeg failed").strip()[-300:])


def _fetch_source(seg: dict, workdir: str) -> str:
    """A local file to cut from: the delivered clip, the project zip, or a fresh
    download of the source."""
    if seg["src_file"] and os.path.isfile(seg["src_file"]):
        return seg["src_file"]
    if seg["src_zip"] and os.path.isfile(seg["src_zip"]):
        with zipfile.ZipFile(seg["src_zip"]) as z:
            if seg["src_member"] in z.namelist():
                out = os.path.join(workdir, "src" + os.path.splitext(seg["src_member"])[1])
                with z.open(seg["src_member"]) as fin, open(out, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
                return out
    out = os.path.join(workdir, "src.mp4")
    state: dict = {}
    if (seg["source"] or "").lower() == "youtube":
        from core.youtube import download_video
        download_video(seg["url"] or seg["page_url"], out, "1080", state, no_audio=True)
    else:
        from core.direct_downloader import download_direct_video
        download_direct_video(seg["url"] or seg["page_url"], out, state)
    if not os.path.isfile(out) or not os.path.getsize(out):
        raise RuntimeError(state.get("error_msg") or "couldn't download the source")
    return out


def cut_segment(seg: dict) -> tuple:
    """Cut the segment to its own file (1080p max, no audio) and grab three
    frames. Returns ``(file_path, [frame paths])``."""
    d = lib_dir()
    dest = os.path.join(d, f"seg_{seg['id']}.mp4")
    with tempfile.TemporaryDirectory() as work:
        src = _fetch_source(seg, work)
        dur = seg["src_out"] - seg["src_in"]
        _ffmpeg(["-ss", f"{seg['src_in']:.3f}", "-i", src, "-t", f"{dur:.3f}", "-an",
                 "-vf", "scale='min(1920,iw)':-2", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dest])
    frames = []
    for i, frac in enumerate((0.15, 0.5, 0.85), 1):
        fp = os.path.join(d, f"seg_{seg['id']}_{i}.jpg")
        _ffmpeg(["-ss", f"{dur * frac:.3f}", "-i", dest, "-frames:v", "1",
                 "-vf", "scale=640:-2", "-q:v", "4", fp], timeout=60)
        frames.append(fp)
    return dest, frames


def prepare_image(seg: dict) -> tuple:
    """Store a still: copy the delivered image (or fetch it again) into the
    library, plus a JPEG preview the vision model and the page can use.
    Returns ``(file_path, [preview path])``."""
    d = lib_dir()
    with tempfile.TemporaryDirectory() as work:
        src = ""
        if seg["src_file"] and os.path.isfile(seg["src_file"]):
            src = seg["src_file"]
        elif seg["src_zip"] and os.path.isfile(seg["src_zip"]):
            with zipfile.ZipFile(seg["src_zip"]) as z:
                if seg["src_member"] in z.namelist():
                    src = os.path.join(work, "src" + os.path.splitext(seg["src_member"])[1])
                    with z.open(seg["src_member"]) as fin, open(src, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
        if not src:
            from core.pipeline import _download_image
            src = _download_image(seg["url"] or seg["page_url"], work, 1, stem="src")
        if not src or not os.path.isfile(src):
            raise RuntimeError("couldn't get the image (not on the server and the link failed)")
        ext = os.path.splitext(src)[1].lower() or ".jpg"
        dest = os.path.join(d, f"seg_{seg['id']}{ext}")
        shutil.copyfile(src, dest)
    preview = os.path.join(d, f"seg_{seg['id']}_1.jpg")
    _ffmpeg(["-i", dest, "-frames:v", "1", "-vf", "scale='min(1280,iw)':-2", "-q:v", "3",
             preview], timeout=60)
    return dest, [preview]


_DRAFT_PROMPT = (
    "You catalogue B-roll footage for a video channel. You see frames from ONE short clip — "
    "or ONE still image (photo or graphic) — and the narration lines it was used for. "
    "Describe what is SHOWN, not the narration.\n"
    "Return STRICT JSON with:\n"
    '  "description": 1-2 factual sentences of what is on screen (subject, action, framing, '
    "setting). No opinions.\n"
    '  "subject": the most specific identity of the main subject you can actually SEE '
    '(brand, model, year, part name), or "" if nothing specific is visible.\n'
    '  "identifiable": true only if a viewer could tell that exact subject from the frames '
    "(badge, logo, distinctive design, readable text); false if it could pass for any similar thing.\n"
    '  "generic_use": what this could stand in for as general footage, e.g. "oil draining from a '
    'car engine".\n'
    '  "shot_type": one of close_up, detail, medium, wide, aerial, pov, screen.\n'
    '  "problems": list from text_logo, person, shaky, low_res, vertical, dark.\n'
    "Do not guess a model you cannot see. If the narration names one but the frames don't "
    "show it, leave subject generic and identifiable false."
)


GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def gemini_keys() -> list:
    """Gemini keys for the backup drafter, in rotation order."""
    keys = []
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GOOGLE_API_KEY"):
        k = os.getenv(name, "").strip()
        if k and k not in keys:
            keys.append(k)
    return keys


def gemini_model() -> str:
    # VERIFY_MODEL is the old name, from the removed Gemini visual-verify stage.
    return (os.getenv("GEMINI_MODEL", "").strip() or os.getenv("VERIFY_MODEL", "").strip()
            or "gemini-2.5-flash")


def _gemini_text(payload: dict) -> str:
    """First non-empty text part of a generateContent response ('' when none)."""
    for cand in payload.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return ""


def _draft_vision(frames: list, lines: list, topic: str) -> dict:
    from core.keywords import _loads_llm_json
    import requests
    keys = gemini_keys()
    if not keys:
        raise ValueError("no Gemini key")
    parts = [{"text": _DRAFT_PROMPT}]
    for fp in frames:
        with open(fp, "rb") as f:
            parts.append({"inline_data": {"mime_type": "image/jpeg",
                                          "data": base64.b64encode(f.read()).decode()}})
    parts.append({"text": f"Video topic: {topic or 'unknown'}\nUsed for these lines:\n"
                          + "\n".join(f"- {l}" for l in lines[:3])})
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}}
    last = None
    for key in keys:
        try:
            r = requests.post(f"{GEMINI_BASE}/{gemini_model()}:generateContent", json=body,
                              headers={"x-goog-api-key": key}, timeout=(10, 90))
            r.raise_for_status()
            return _loads_llm_json(_gemini_text(r.json()))
        except Exception as e:
            last = e
    raise last or RuntimeError("Gemini request failed")


VISION_FALLBACK_MODEL = "z-ai/glm-5.3-flash"


def vision_fallback_model() -> str:
    return os.getenv("SEGMENT_VISION_MODEL", "").strip() or VISION_FALLBACK_MODEL


def _openrouter_keys() -> list:
    return [k for k in (os.getenv("OPENROUTER_API_KEY", "").strip(),
                        os.getenv("OPENROUTER_API_KEY_2", "").strip()) if k]


def _draft_openrouter(frames: list, lines: list, topic: str) -> dict:
    """Primary vision drafter: the prompt and frames through OpenRouter
    (default z-ai/glm-5.3-flash — live-tested: accurate, doesn't invent
    subjects from the narration, ~$0.0003 a clip). Retries OpenRouter's
    transient 429 'couldn't verify credits in time'."""
    import requests
    from core.keywords import _loads_llm_json
    keys = _openrouter_keys()
    if not keys:
        raise ValueError("no OpenRouter key")
    content = [{"type": "text", "text": _DRAFT_PROMPT}]
    for fp in frames:
        with open(fp, "rb") as f:
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()}})
    content.append({"type": "text", "text": f"Video topic: {topic or 'unknown'}\nUsed for these lines:\n"
                                            + "\n".join(f"- {l}" for l in lines[:3])})
    body = {"model": vision_fallback_model(), "temperature": 0.2,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"}}
    last = None
    for key in keys:
        for attempt in range(3):
            try:
                r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                                  headers={"Authorization": f"Bearer {key}"}, timeout=(10, 120))
                if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(min(30, float(r.headers.get("Retry-After") or 10)))
                    continue
                r.raise_for_status()
                j = r.json()
                try:
                    from core.keywords import _record_api_usage
                    _record_api_usage("openrouter", body["model"], j.get("usage"))
                except Exception:
                    pass
                return _loads_llm_json(j["choices"][0]["message"]["content"])
            except Exception as e:
                last = e
                break
    raise last or RuntimeError("OpenRouter request failed")


# Vision drafters in order of preference: OpenRouter GLM (cheap, tested),
# then Gemini as the backup.
_VISION_DRAFTERS = (("openrouter", lambda *a: _draft_openrouter(*a)),
                    ("gemini", lambda *a: _draft_vision(*a)))


def check_vision(frame_path: str) -> list:
    """Send one real frame through every configured vision drafter.
    ``[(provider, ok|None, detail)]`` — None when that provider has no key."""
    configured = {"gemini": bool(gemini_keys()), "openrouter": bool(_openrouter_keys())}
    out = []
    for name, fn in _VISION_DRAFTERS:
        if not configured[name]:
            out.append((name, None, "no key"))
            continue
        t0 = time.time()
        try:
            d = _clean_draft(fn([frame_path], ["test line"], "test"))
            ok = bool(d["description"])
            model = gemini_model() if name == "gemini" else vision_fallback_model()
            out.append((name, ok, f"{model}, {time.time() - t0:.0f}s"
                                  + ("" if ok else " — empty description")))
        except Exception as e:
            out.append((name, False, f"{type(e).__name__}: {str(e)[:150]}"))
    return out


def _draft_text(seg: dict, lines: list, topic: str) -> dict:
    """No vision key: a cautious draft from the title and lines alone."""
    from groq import Groq
    from core.keywords import _call_llm_json
    user = (f"Clip title: {seg['title']}\nSource: {seg['source']}\nVideo topic: {topic}\n"
            "Used for:\n" + "\n".join(f"- {l}" for l in lines[:3]) +
            "\nYou cannot see the clip: keep subject generic unless the title names it, and "
            "set identifiable false unless the title makes the exact subject certain.")
    return _call_llm_json(Groq(api_key=os.getenv("GROQ_API_KEY") or "unused"),
                          _DRAFT_PROMPT, user, temperature=0.2, max_tokens=500)


def _clean_draft(d: dict) -> dict:
    shot = str(d.get("shot_type") or "").strip()
    return {
        "description": str(d.get("description") or "").strip()[:600],
        "subject": str(d.get("subject") or "").strip()[:120],
        "identifiable": 1 if d.get("identifiable") is True else 0 if d.get("identifiable") is False else None,
        "generic_use": str(d.get("generic_use") or "").strip()[:200],
        "shot_type": shot if shot in _SHOT_IDS else "",
        "problems": [p for p in (d.get("problems") or []) if p in _PROBLEM_IDS],
    }


def _embed_text(seg: dict) -> str:
    return " | ".join(x for x in (seg.get("description"), seg.get("generic_use"),
                                  seg.get("subject")) if x)


def _embed(text: str):
    from core.clip_library import _embed as emb
    return np.asarray(emb(text), dtype=np.float32)


def _lines_of(c, segment_id: int) -> list:
    return [r[0] for r in c.execute(
        "SELECT line_text FROM segment_lines WHERE segment_id=? AND line_text != ''",
        (segment_id,))]


def process_one(seg: dict) -> None:
    """Cut (or, for a still, copy), draft a description if nobody has written
    one, and embed."""
    path, frames = (prepare_image if seg.get("kind") == "image" else cut_segment)(seg)
    with project_store._conn() as c:
        lines = _lines_of(c, seg["id"])
        topic = (c.execute("SELECT topic FROM projects WHERE id=?",
                           (seg["origin_project"],)).fetchone() or {"topic": ""})["topic"]
    fields = {}
    if not seg["description"]:
        # OpenRouter GLM, then Gemini (both look at the frames), then a
        # text-only guess from the title as the last resort.
        attempts = [(name, "vision", fn, (frames, lines, topic)) for name, fn in _VISION_DRAFTERS]
        attempts.append(("text", "text", _draft_text, (seg, lines, topic)))
        for name, source, draft, args in attempts:
            try:
                fields = dict(_clean_draft(draft(*args)), draft_source=source)
                if fields["description"]:
                    break
            except Exception as e:
                print(f"[segments] {name} draft failed for #{seg['id']}: {e}")
            fields = {}
    merged = dict(seg, **fields)
    emb = None
    try:
        emb = _embed(_embed_text(merged)).tobytes() if _embed_text(merged) else None
    except Exception as e:
        print(f"[segments] embedding failed for #{seg['id']}: {e}")
    sets = {"file_path": path, "file_status": "ready", "file_error": "",
            "frames": json.dumps(frames), "embedding": emb, "updated_at": _now()}
    for k in ("description", "subject", "identifiable", "generic_use", "shot_type",
              "draft_source"):
        if k in fields:
            sets[k] = fields[k]
    if "problems" in fields:
        sets["problems"] = json.dumps(fields["problems"])
    with project_store._conn() as c:
        c.execute(f"UPDATE segments SET {', '.join(k + '=?' for k in sets)} WHERE id=?",
                  (*sets.values(), seg["id"]))


def process_pending(limit: int = None) -> int:
    """Work through segments waiting for a file. One at a time — cutting and
    downloading are heavy and the bot's own jobs come first."""
    init_db()
    done = 0
    while limit is None or done < limit:
        with project_store._conn() as c:
            r = c.execute("""SELECT * FROM segments WHERE file_status='pending'
                              ORDER BY (src_file != '' OR src_zip != '') DESC, id
                              LIMIT 1""").fetchone()
        if not r:
            break
        seg = _row(r)
        try:
            process_one(seg)
        except Exception as e:
            with project_store._conn() as c:
                c.execute("UPDATE segments SET file_status='failed', file_error=?, updated_at=? "
                          "WHERE id=?", (str(e)[:300], _now(), seg["id"]))
        done += 1
    return done


def kick() -> None:
    _WORKER["kick"].set()


def start_worker(is_busy=None) -> None:
    """Background thread that processes pending segments whenever it's kicked
    (or every 10 minutes), yielding while ``is_busy()`` reports a bot job."""
    if not enabled() or (_WORKER["thread"] and _WORKER["thread"].is_alive()):
        return

    def _loop():
        while True:
            _WORKER["kick"].wait(timeout=600)
            _WORKER["kick"].clear()
            try:
                while is_busy and is_busy():
                    time.sleep(30)
                ingest_from_ratings()
                _WORKER["kick"].clear()
                n = process_pending()
                if n:
                    print(f"[segments] processed {n} segment(s)")
            except Exception as e:
                print(f"[segments] worker error: {e}")

    t = threading.Thread(target=_loop, daemon=True, name="SegmentLibrary")
    _WORKER["thread"] = t
    t.start()
    kick()


# ── reviewer step ───────────────────────────────────────────────────────────

def next_review_task(rater_id: int, skip: list = None) -> dict | None:
    """The next ready segment this reviewer hasn't reviewed, fewest reviews and
    most-used first."""
    init_db()
    skip = {int(s) for s in (skip or [])}
    with project_store._conn() as c:
        rows = c.execute(
            """SELECT s.* FROM segments s
                WHERE s.file_status='ready' AND s.trust != 'avoid'
                  AND NOT EXISTS (SELECT 1 FROM segment_reviews r
                                   WHERE r.segment_id=s.id AND r.rater_id=?)
                ORDER BY s.n_reviews ASC, (s.trust='used') DESC, s.times_used DESC, s.id
                LIMIT 50""", (rater_id,)).fetchall()
        seg = next((_row(r) for r in rows if r["id"] not in skip), None)
        if not seg:
            return None
        lines = [dict(r) for r in c.execute(
            "SELECT project_id, slot_id, line_text, fit FROM segment_lines WHERE segment_id=?",
            (seg["id"],))]
        topic = c.execute("SELECT topic FROM projects WHERE id=?",
                          (seg["origin_project"],)).fetchone()
    return {
        "mode": "library", "segment_id": seg["id"], "title": seg["title"],
        "kind": seg.get("kind") or "clip", "origin_page": seg.get("origin_page") or "",
        "media_ext": os.path.splitext(seg["file_path"] or "")[1].lstrip(".") or "mp4",
        "source": seg["source"], "channel": seg["channel"], "page_url": seg["page_url"],
        "src_in": seg["src_in"], "src_out": seg["src_out"], "topic": topic[0] if topic else "",
        "trust": seg["trust"], "lines": [l["line_text"] for l in lines][:5],
        "draft": {k: seg[k] for k in ("description", "subject", "identifiable", "generic_use",
                                      "shot_type", "problems")},
        "draft_source": seg["draft_source"],
        "shot_types": [{"id": i, "label": l} for i, l in SHOT_TYPES],
        "problem_list": [{"id": i, "label": l} for i, l in PROBLEMS],
    }


_VAGUE = {"good", "bad", "ok", "okay", "nice", "fine", "great", "relevant", "irrelevant",
          "clip", "video", "footage", "shot", "yes", "no"}


def description_issues(text: str) -> list:
    """Why a description isn't useful yet (empty list = fine). Shared with the
    page so reviewers see the same checks before saving."""
    words = re.findall(r"[^\W_]+", (text or "").lower())
    issues = []
    if len(words) < 6:
        issues.append("Too short — say what's on screen in a full sentence.")
    elif len([w for w in words if w not in _VAGUE]) < 4:
        issues.append("Too vague — name the subject and what's happening.")
    return issues


def save_review(rater_id: int, segment_id: int, *, usable: str, fits_line: str = "",
                description: str = "", subject: str = "", identifiable=None,
                generic_use: str = "", shot_type: str = "", problems=None,
                note: str = "") -> dict:
    """Store a reviewer's check of a segment and update the segment: its
    description fields take the latest review, and its trust follows the votes
    (usable by most reviewers → verified; not usable by most → avoid)."""
    init_db()
    if usable not in ("yes", "no"):
        raise ValueError("say whether the clip is usable")
    if fits_line not in ("", "yes", "partly", "no"):
        raise ValueError("bad fits_line")
    ident = None if identifiable in (None, "") else (1 if identifiable in (True, 1, "1", "yes") else 0)
    if usable == "yes":
        issues = description_issues(description)
        if issues:
            raise ValueError(issues[0])
        if ident is None:
            raise ValueError("say whether a viewer can tell exactly what it is")
        if ident == 1 and not subject.strip():
            raise ValueError("name the exact subject a viewer can see")
        if ident == 0 and not generic_use.strip():
            raise ValueError("say what it works for as general footage")
    shot_type = shot_type if shot_type in _SHOT_IDS else ""
    problems = [p for p in (problems or []) if p in _PROBLEM_IDS]
    now = _now()
    with project_store._conn() as c:
        if not c.execute("SELECT 1 FROM segments WHERE id=?", (segment_id,)).fetchone():
            raise ValueError(f"unknown segment {segment_id}")
        c.execute(
            """INSERT INTO segment_reviews (segment_id, rater_id, fits_line, usable, description,
                   subject, identifiable, generic_use, shot_type, problems, note, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(segment_id, rater_id) DO UPDATE SET fits_line=excluded.fits_line,
                 usable=excluded.usable, description=excluded.description,
                 subject=excluded.subject, identifiable=excluded.identifiable,
                 generic_use=excluded.generic_use, shot_type=excluded.shot_type,
                 problems=excluded.problems, note=excluded.note, created_at=excluded.created_at""",
            (segment_id, rater_id, fits_line, usable, description.strip()[:600],
             subject.strip()[:120], ident, generic_use.strip()[:200], shot_type,
             json.dumps(problems), (note or "").strip()[:1000], now))
        votes = c.execute("""SELECT SUM(usable='yes'), SUM(usable='no'), COUNT(*)
                               FROM segment_reviews WHERE segment_id=?""",
                          (segment_id,)).fetchone()
        good, bad, n = votes[0] or 0, votes[1] or 0, votes[2] or 0
        trust = "avoid" if bad > good else "verified" if good else None
        sets = {"good_votes": good, "bad_votes": bad, "n_reviews": n, "updated_at": now}
        if trust:
            sets["trust"] = trust
        if usable == "yes":
            sets.update(description=description.strip()[:600], subject=subject.strip()[:120],
                        identifiable=ident, generic_use=generic_use.strip()[:200],
                        shot_type=shot_type, problems=json.dumps(problems),
                        draft_source="reviewer")
        if fits_line:
            c.execute("UPDATE segment_lines SET fit=? WHERE segment_id=?", (fits_line, segment_id))
        c.execute(f"UPDATE segments SET {', '.join(k + '=?' for k in sets)} WHERE id=?",
                  (*sets.values(), segment_id))
    seg = get_segment(segment_id)
    if usable == "yes":
        try:
            with project_store._conn() as c:
                c.execute("UPDATE segments SET embedding=? WHERE id=?",
                          (_embed(_embed_text(seg)).tobytes(), segment_id))
        except Exception as e:
            print(f"[segments] re-embed failed for #{segment_id}: {e}")
    try:
        from core import ratings
        ratings.log_impact([rater_id], "segment_reviewed", f"segment:{segment_id}",
                           seg.get("description") or seg.get("title") or "", once=True)
    except Exception:
        pass
    return seg


# ── picking from the library ────────────────────────────────────────────────

_TAG_PROMPT = (
    "For each narration line of a video, list the SPECIFIC subjects it names or clearly "
    "refers to (brand, model, year, product, part, place, person — use the video topic to "
    "resolve 'it'/'the car'), and the general action/topic in 2-5 words.\n"
    'Return STRICT JSON: {"lines": [{"slot_id": <id>, "subjects": ["..."], "topic": "..."}]}'
)


def tag_lines(shots: list, video_topic: str = "") -> None:
    """Set ``shot['line_subjects']`` / ``shot['line_topic']`` with one batched
    LLM call per 40 lines. Leaves them unset when the call fails."""
    todo = [s for s in shots if "line_subjects" not in s and (s.get("text") or "").strip()]
    if not todo:
        return
    from groq import Groq
    from core.keywords import _call_llm_json
    client = Groq(api_key=os.getenv("GROQ_API_KEY") or "unused")
    for i in range(0, len(todo), 40):
        batch = todo[i:i + 40]
        user = f"Video topic: {video_topic or 'unknown'}\n" + "\n".join(
            f'slot_id {s.get("slot_id")}: "{s.get("text", "").strip()[:300]}"' for s in batch)
        try:
            data = _call_llm_json(client, _TAG_PROMPT, user, temperature=0.1, max_tokens=3000)
        except Exception as e:
            print(f"[segments] line tagging failed: {e}")
            continue
        by = {str(x.get("slot_id")): x for x in data.get("lines") or [] if isinstance(x, dict)}
        for s in batch:
            x = by.get(str(s.get("slot_id")))
            if x:
                s["line_subjects"] = [str(v).strip() for v in x.get("subjects") or [] if str(v).strip()]
                s["line_topic"] = str(x.get("topic") or "").strip()


_STOP = {"the", "a", "an", "of", "and", "for", "with", "car", "cars", "new", "old"}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[^\W_]+", (text or "").lower()) if len(w) > 1 and w not in _STOP}


def subject_matches(seg_subject: str, line_subjects: list, line_text: str = "") -> bool:
    """Does an identifiable segment's subject fit this line? Every distinctive
    word of the segment's subject must appear in one named subject (or, with no
    tags, in the line itself): 'Toyota Camry' fits 'the 2020 Toyota Camry', not
    'Honda Civic'."""
    need = _tokens(seg_subject) - {str(y) for y in range(1950, 2100)}
    if not need:
        return True
    pools = [_tokens(s) for s in line_subjects or []] or [_tokens(line_text)]
    return any(need <= pool for pool in pools)


def _identifiable(seg: dict) -> bool:
    return seg["identifiable"] == 1 and bool(seg["subject"])


def _delivered_ids(c, n: int) -> list:
    """The last ``n`` delivered videos' project ids, newest first."""
    q = ",".join("?" * len(_DELIVERED))
    return [r[0] for r in c.execute(
        f"SELECT id FROM projects WHERE status IN ({q}) ORDER BY id DESC LIMIT ?",
        (*_DELIVERED, n))]


def _use_history(c, n: int) -> dict:
    """``{segment_id: [videos_ago, …]}`` over the last ``n`` delivered videos
    (0 = the latest)."""
    ago = {pid: i for i, pid in enumerate(_delivered_ids(c, n))}
    if not ago:
        return {}
    out = {}
    q = ",".join("?" * len(ago))
    for sid, pid in c.execute(
            f"SELECT segment_id, project_id FROM segment_uses WHERE project_id IN ({q})",
            list(ago)):
        out.setdefault(sid, []).append(ago[pid])
    return out


def find_matches(shot: dict, segs: list, embed=None, top_k: int = 3) -> list:
    """``[(score, segment)]`` for one shot, best first. Identifiable segments
    must match the line's subject; generic ones compete on meaning alone.
    Each recent use (``recent_uses``, set by :func:`_usable`) costs score."""
    text = (shot.get("text") or "").strip()
    if not text or not segs:
        return []
    embed = embed or _embed
    query = text + (f" | {shot['line_topic']}" if shot.get("line_topic") else "")
    q = embed(query)
    subjects = shot.get("line_subjects")
    need = max(_MIN_SECONDS, min(float(shot.get("duration_needed_sec") or 0) / 2, 4.0))
    use_penalty = _env_num("LIBRARY_USE_PENALTY", _USE_PENALTY, float)
    out = []
    for s in segs:
        if s.get("kind") != "image" and s["src_out"] - s["src_in"] < need:
            continue
        if _identifiable(s) and not subject_matches(s["subject"], subjects or [], text):
            continue
        v = np.frombuffer(s["embedding"], dtype=np.float32)
        if v.shape != q.shape:
            continue
        score = float(np.dot(q, v)) + _TRUST_BOOST.get(s["trust"], 0.0)
        score += 0.02 * min(s["times_kept"], 5) - 0.03 * min(s["times_dropped"], 5)
        score -= use_penalty * s.get("recent_uses", 0)
        if _identifiable(s):
            score += 0.04          # the exact subject beats a stand-in
        if score >= _MIN_SCORE:
            out.append((score, s))
    out.sort(key=lambda t: -t[0])
    return out[:top_k]


def _candidate(seg: dict, score: float) -> dict:
    trust_word = {"verified": "verified by reviewers", "used": "used by an editor",
                  "suggested": "suggested by a reviewer"}.get(seg["trust"], seg["trust"])
    what = seg["subject"] if seg["identifiable"] == 1 else (seg["generic_use"] or seg["subject"])
    return {
        "url": seg["page_url"], "page_url": seg["page_url"], "source": seg["source"] or "youtube",
        "title": f"[Library] {(seg['description'] or seg['title'])[:110]}",
        "description": f"library clip ({trust_word}) — {what}",
        "channel": seg["channel"], "matched_query": "library",
        "duration": round(seg["src_out"] - seg["src_in"], 2),
        "width": 1920, "height": 1080, "is_short": False,
        "library_segment_id": seg["id"], "segment_path": seg["file_path"],
        "segment_trust": seg["trust"], "library_score": round(score, 3),
        "edit_record": (f"library clip {trust_word}; kept {seg['times_kept']}×, "
                        f"dropped {seg['times_dropped']}×"),
    }


def _usable(kind: str) -> list:
    """Ready, not-avoided library entries of ``kind`` with a file on disk,
    minus the ones resting after a recent use (``LIBRARY_REUSE_GAP``, longer
    for identifiable subjects). Each carries ``recent_uses`` for the score
    penalty in :func:`find_matches`."""
    init_db()
    gap = _env_num("LIBRARY_REUSE_GAP", _REUSE_GAP)
    gap_identifiable = _env_num("LIBRARY_REUSE_GAP_IDENTIFIABLE", _REUSE_GAP_IDENTIFIABLE)
    window = _env_num("LIBRARY_USE_WINDOW", _USE_WINDOW)
    with project_store._conn() as c:
        segs = [_row(r) for r in c.execute(
            """SELECT * FROM segments WHERE file_status='ready' AND trust != 'avoid'
                  AND embedding IS NOT NULL AND COALESCE(kind, 'clip') = ?""", (kind,))]
        history = _use_history(c, max(gap, gap_identifiable, window))
    out = []
    for s in segs:
        if not os.path.isfile(s["file_path"] or ""):
            continue
        ago = history.get(s["id"], [])
        if ago and min(ago) < (gap_identifiable if _identifiable(s) else gap):
            continue
        s["recent_uses"] = sum(1 for a in ago if a < window)
        out.append(s)
    return out


def inject_candidates(shots: list, video_topic: str = "", errors: list = None) -> int:
    """Add the best library clips to each shot's candidates (front of the
    list). Returns how many were added."""
    if not enabled():
        return 0
    segs = _usable("clip")
    if not segs:
        return 0
    targets = [s for s in shots if not s.get("is_extra") and s.get("priority") != "none"]
    try:
        tag_lines(targets, video_topic)
    except Exception as e:
        if errors is not None:
            errors.append(f"library line tags: {e}")
    added = 0
    for s in targets:
        try:
            hits = find_matches(s, segs)
        except Exception as e:
            if errors is not None:
                errors.append(f"library match (shot {s.get('slot_id')}): {e}")
            continue
        if not hits:
            continue
        have = {c.get("library_segment_id") for c in s.get("video_results") or []}
        new = [_candidate(seg, score) for score, seg in hits if seg["id"] not in have]
        # A library segment replaces the untrimmed copy of the same source video.
        pages = {c["page_url"] for c in new}
        rest = [c for c in s.get("video_results") or []
                if c.get("library_segment_id") or project_store.asset_key(c) not in pages]
        s["video_results"] = new + rest
        added += len(new)
    return added


def add_library_images(shots: list, project_name: str, video_topic: str = "",
                       per_shot: int = 2, errors: list = None) -> int:
    """Put matching library stills in front of each shot's images: copied into
    the project's ``images/shots/shot_NN/`` like the Google ones, named
    ``NN-L1-library-….ext`` so the editor can tell them apart. Same subject
    rule as clips. Returns how many were added."""
    if not enabled() or per_shot <= 0:
        return 0
    imgs = _usable("image")
    if not imgs:
        return 0
    from core.output import _safe_for_fs
    from core.shot_images import _shot_dir_name, _slug
    targets = [s for s in shots if not s.get("is_extra") and s.get("priority") != "none"]
    try:
        tag_lines(targets, video_topic)
    except Exception as e:
        if errors is not None:
            errors.append(f"library line tags: {e}")
    base = os.path.join(os.path.abspath("downloads"), _safe_for_fs(project_name, 50),
                        "images", "shots")
    added = 0
    taken = set()                      # a still goes to one shot per video
    for s in targets:
        try:
            hits = find_matches(s, [i for i in imgs if i["id"] not in taken], top_k=per_shot)
        except Exception as e:
            if errors is not None:
                errors.append(f"library images (shot {s.get('slot_id')}): {e}")
            continue
        have = {i.get("url") for i in s.get("images") or []}
        new = []
        for k, (score, seg) in enumerate(hits, 1):
            if seg["url"] in have:
                continue
            d = os.path.join(base, _shot_dir_name(s.get("slot_id")))
            os.makedirs(d, exist_ok=True)
            ext = os.path.splitext(seg["file_path"])[1] or ".jpg"
            label = _slug(seg["subject"] or seg["generic_use"] or seg["description"], 30)
            dest = os.path.join(d, f"{_shot_dir_name(s.get('slot_id'))[5:]}-L{k}-library-{label}{ext}")
            try:
                shutil.copyfile(seg["file_path"], dest)
            except OSError as e:
                if errors is not None:
                    errors.append(f"library image copy: {e}")
                continue
            new.append({"url": seg["url"], "local_path": dest,
                        "title": f"[Library] {(seg['description'] or seg['title'])[:110]}",
                        "page": seg.get("origin_page") or "", "query": "library",
                        "library_segment_id": seg["id"], "segment_trust": seg["trust"],
                        "library_score": round(score, 3)})
            taken.add(seg["id"])
        if new:
            s["images"] = new + list(s.get("images") or [])
            added += len(new)
    return added


def promote_strong(shots: list) -> int:
    """After ranking: a verified segment with a strong match that the judge
    didn't reject goes to the front, so selection takes it first."""
    moved = 0
    for s in shots:
        cands = s.get("video_results") or []
        strong = [c for c in cands if c.get("segment_trust") == "verified"
                  and (c.get("library_score") or 0) >= _STRONG_SCORE and not c.get("irrelevant")]
        if strong and cands[0] is not strong[0]:
            s["video_results"] = strong + [c for c in cands if c not in strong]
            moved += 1
    return moved


def mark_used(project_id: int, shots: list) -> int:
    """At delivery: remember which segments went into this project (for repeat
    avoidance), and credit the reviewers who verified them."""
    init_db()
    used = {}
    for s in shots or []:
        for c in list(s.get("selected_results") or []) + list(s.get("images") or []):
            if c.get("library_segment_id") and not c.get("_dl_failed"):
                used[c["library_segment_id"]] = c
    if not used:
        return 0
    now = _now()
    with project_store._conn() as c:
        for sid in used:
            cur = c.execute("INSERT OR IGNORE INTO segment_uses (segment_id, project_id, created_at) "
                            "VALUES (?, ?, ?)", (sid, project_id, now))
            if cur.rowcount:
                c.execute("UPDATE segments SET times_used = times_used + 1 WHERE id=?", (sid,))
        reviewers = {}
        for sid in used:
            reviewers[sid] = [r[0] for r in c.execute(
                "SELECT rater_id FROM segment_reviews WHERE segment_id=? AND usable='yes'", (sid,))]
    try:
        from core import ratings
        for sid, cand in used.items():
            ratings.log_impact(reviewers[sid], "library_used", f"{project_id}:segment:{sid}",
                               cand.get("title", ""), project_id, once=True)
    except Exception:
        pass
    return len(used)


def stats() -> dict:
    init_db()
    with project_store._conn() as c:
        by_trust = {r[0]: r[1] for r in c.execute(
            "SELECT trust, COUNT(*) FROM segments WHERE file_status='ready' GROUP BY trust")}
        by_kind = {r[0]: r[1] for r in c.execute(
            """SELECT COALESCE(kind, 'clip'), COUNT(*) FROM segments
                WHERE file_status='ready' AND trust != 'avoid' GROUP BY 1""")}
        by_status = {r[0]: r[1] for r in c.execute(
            "SELECT file_status, COUNT(*) FROM segments GROUP BY file_status")}
        subjects = [(r[0], r[1]) for r in c.execute(
            """SELECT subject, COUNT(*) FROM segments WHERE subject != '' AND trust != 'avoid'
                GROUP BY lower(subject) ORDER BY COUNT(*) DESC LIMIT 8""")]
        seconds = c.execute("""SELECT COALESCE(SUM(src_out - src_in), 0) FROM segments
                                WHERE file_status='ready' AND trust != 'avoid'""").fetchone()[0]
        reuse = _reuse_stats(c)
    return {"by_trust": by_trust, "by_status": by_status, "by_kind": by_kind,
            "subjects": subjects, "seconds": seconds, "reuse": reuse}


def _reuse_stats(c, n: int = _SHARE_WINDOW) -> dict:
    """How much of the last ``n`` delivered videos came from the library, and
    the clip repeated most often across them — the early warning that videos
    are starting to look alike."""
    pids = _delivered_ids(c, n)
    if not pids:
        return {"videos": 0, "clips": 0, "library_clips": 0, "top": None}
    q = ",".join("?" * len(pids))
    clips, lib = c.execute(
        f"""SELECT COUNT(*), COUNT(segment_id) FROM project_assets
             WHERE kind='clip' AND project_id IN ({q})""", pids).fetchone()
    top = c.execute(
        f"""SELECT s.description, s.subject, s.title, COUNT(*) FROM segment_uses u
              JOIN segments s ON s.id = u.segment_id
             WHERE COALESCE(s.kind, 'clip') = 'clip' AND u.project_id IN ({q})
             GROUP BY u.segment_id ORDER BY 4 DESC, u.segment_id LIMIT 1""", pids).fetchone()
    return {"videos": len(pids), "clips": clips, "library_clips": lib,
            "top": ((top[0] or top[1] or top[2] or "a library clip")[:80], top[3]) if top else None}
