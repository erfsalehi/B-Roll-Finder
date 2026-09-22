"""
Human ratings — reviewers teaching the picker faster than finished edits can.

For every delivered project we queue each shot's candidates for rating: the
clips the bot picked, a couple of runner-ups it passed over, and one it judged
irrelevant (so reviewers catch good clips it missed as well as bad picks). A
reviewer sees the narration line and the clips (without being told which ones
the bot picked), and for each clip gives:

    usable   yes | trim (usable with a trim) | no
    fit      1–5
    reasons  quick chips (wrong subject, talking head, …)
    note     free text: why
    segments the parts of the clip that are good to use, [[in, out], …] seconds

and can suggest a better clip of their own (URL + good segments + note).

:mod:`core.edit_feedback` turns these into ranking hints, rater segments become
in-points, suggestions are offered as candidates for similar lines, and the
written reasons are distilled into short "house rules" for the ranking prompt.

Lives in the project DB (``.cache/projects.db``) next to the project store.
"""

import json
import os
import re
import threading
import zlib
from collections import Counter
from datetime import datetime

from core import project_store

USABLE = ("yes", "trim", "no")

# (id, label) — quick reasons a reviewer can tick.
REASONS = [
    ("great_match", "Great match"),
    ("wrong_subject", "Wrong subject"),
    ("wrong_domain", "Right thing, wrong kind"),
    ("other_line", "Good clip, wrong line"),
    ("generic_ok", "Wrong model/brand — OK as general footage"),
    ("talking_head", "Talking head / presenter"),
    ("text_logo", "Text, logo or watermark"),
    ("low_quality", "Low quality / blurry"),
    ("vertical", "Vertical / cropped"),
    ("too_generic", "Too generic"),
]
_REASON_IDS = {r for r, _ in REASONS}

# Labels a reviewer can give an editor-XML asset. The first three mirror
# project_store's automatic verdicts; the last two only a human can tell apart
# from "unused": the clip was fine but not needed, or it was genuinely bad.
LABEL_VERDICTS = ("used_here", "used_elsewhere", "unused", "neutral", "bad")

# What counts as a reviewer helping the picker (see log_impact).
IMPACT_KINDS = {
    "suggestion_used": "Your suggested clip was picked for a video",
    "helped_pick": "A clip you rated usable was picked",
    "in_point_used": "Your marked good part set a clip's in-point",
    "blocked_bad_clip": "A clip you rated not usable was pushed down",
    "note_in_rules": "Your note went into the house rules",
    "label_corrected": "You corrected an editor-XML label",
    "label_confirmed": "You confirmed an editor-XML label",
    "segment_reviewed": "You described a library clip",
    "library_used": "A library clip you checked was placed in a video",
}

# Kinds that record a reviewer's own work rather than a change to picks.
WORK_KINDS = {"label_corrected", "label_confirmed", "segment_reviewed"}

RUNNER_UPS = 2          # non-picked but relevant candidates queued per shot
_MAX_SEGMENTS = 10
_RULES_EVERY = 20       # new written notes before house rules are re-distilled

_YT_ID = re.compile(r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:.*&)?v=|embed/|shorts/|live/))"
                    r"([A-Za-z0-9_-]{11})")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_db() -> None:
    project_store.init_db()
    with project_store._conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS rating_items (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id      INTEGER NOT NULL,
                slot_id         TEXT    NOT NULL,
                role            TEXT    NOT NULL,   -- pick | runner_up | rejected | suggested
                url             TEXT    DEFAULT '',
                page_url        TEXT    NOT NULL,
                source          TEXT    DEFAULT '',
                title           TEXT    DEFAULT '',
                channel         TEXT    DEFAULT '',
                matched_query   TEXT    DEFAULT '',
                verified_in_sec REAL,
                suggested_by    INTEGER,
                created_at      TEXT    DEFAULT '',
                UNIQUE (project_id, slot_id, page_url)
            );
            CREATE TABLE IF NOT EXISTS ratings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id     INTEGER NOT NULL,
                rater_id    INTEGER NOT NULL,
                usable      TEXT    NOT NULL,
                fit         INTEGER,
                reasons     TEXT    DEFAULT '[]',
                note        TEXT    DEFAULT '',
                segments    TEXT    DEFAULT '[]',
                created_at  TEXT    DEFAULT '',
                UNIQUE (item_id, rater_id)
            );
            CREATE INDEX IF NOT EXISTS idx_ratings_item ON ratings(item_id);
            CREATE TABLE IF NOT EXISTS raters (
                rater_id    INTEGER PRIMARY KEY,
                name        TEXT    DEFAULT '',
                updated_at  TEXT    DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS house_rules (
                id          INTEGER PRIMARY KEY CHECK (id = 1),
                rules       TEXT    DEFAULT '[]',
                notes_seen  INTEGER DEFAULT 0,
                updated_at  TEXT    DEFAULT ''
            );
            -- A reviewer checking an automatic label from an editor's XML.
            CREATE TABLE IF NOT EXISTS label_reviews (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id      INTEGER NOT NULL,
                rater_id      INTEGER NOT NULL,
                action        TEXT    NOT NULL,        -- confirm | correct
                verdict       TEXT    NOT NULL,        -- see LABEL_VERDICTS
                used_slot_id  TEXT    DEFAULT '',
                note          TEXT    DEFAULT '',
                created_at    TEXT    DEFAULT '',
                UNIQUE (asset_id, rater_id)
            );
            -- Moments a reviewer's work changed what the picker did.
            CREATE TABLE IF NOT EXISTS impact_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                rater_id    INTEGER NOT NULL,
                kind        TEXT    NOT NULL,          -- see IMPACT_KINDS
                clip_key    TEXT    DEFAULT '',
                title       TEXT    DEFAULT '',
                project_id  INTEGER,
                day         TEXT    DEFAULT '',
                created_at  TEXT    DEFAULT '',
                UNIQUE (rater_id, kind, clip_key, day)
            );
            CREATE INDEX IF NOT EXISTS idx_impact_rater ON impact_events(rater_id);
        """)


# ── helpers ─────────────────────────────────────────────────────────────────

def youtube_id(url: str) -> str:
    m = _YT_ID.search(url or "")
    return m.group(1) if m else ""


def normalize_suggestion_url(url: str) -> tuple[str, str]:
    """``(canonical_url, source)`` for a URL a reviewer pasted. YouTube links of
    any shape become ``https://www.youtube.com/watch?v=<id>``."""
    url = (url or "").strip()
    vid = youtube_id(url)
    if vid:
        return f"https://www.youtube.com/watch?v={vid}", "youtube"
    host = re.sub(r"^www\.", "", re.sub(r"^https?://", "", url.lower()).split("/")[0])
    if host.endswith("pexels.com"):
        return url, "pexels"
    return url, "web"


def clean_segments(raw) -> list:
    """Validated ``[[in, out], …]`` (seconds, in < out, max 10), sorted."""
    out = []
    for seg in raw or []:
        try:
            a, b = float(seg[0]), float(seg[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < b and b - a <= 3600:
            out.append([round(a, 2), round(b, 2)])
    return sorted(out)[:_MAX_SEGMENTS]


def _item_row(project_id, slot, role, c: dict) -> tuple:
    return (project_id, slot, role, c.get("url") or "", project_store.asset_key(c),
            c.get("source") or "", (c.get("title") or "")[:300],
            project_store.creator_of(c), c.get("matched_query") or "",
            c.get("verified_in_sec"), _now())


# ── queueing ────────────────────────────────────────────────────────────────

def record_candidates(project_id: int, shots: list, runner_ups: int = RUNNER_UPS) -> int:
    """Queue each shot's picks, top ``runner_ups`` relevant non-picks and its best
    irrelevant candidate for rating. Idempotent (existing items and their
    ratings are kept). Returns how many new items were queued."""
    init_db()
    rows = []
    for idx, shot in enumerate(shots or []):
        if shot.get("is_extra") or shot.get("priority") == "none":
            continue
        slot = str(shot.get("slot_id", idx))
        picks = [c for c in shot.get("selected_results") or [] if not c.get("_dl_failed")]
        picked = {project_store.asset_key(c) for c in picks}
        rest = [c for c in shot.get("video_results") or []
                if project_store.asset_key(c) and project_store.asset_key(c) not in picked]
        rows += [_item_row(project_id, slot, "pick", c) for c in picks
                 if project_store.asset_key(c)]
        rows += [_item_row(project_id, slot, "runner_up", c)
                 for c in [c for c in rest if not c.get("irrelevant")][:runner_ups]]
        rows += [_item_row(project_id, slot, "rejected", c)
                 for c in [c for c in rest if c.get("irrelevant")][:1]]
    if not rows:
        return 0
    with project_store._conn() as c:
        before = c.total_changes
        c.executemany(
            """INSERT OR IGNORE INTO rating_items
               (project_id, slot_id, role, url, page_url, source, title, channel,
                matched_query, verified_in_sec, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows)
        return c.total_changes - before


def backfill_queue() -> int:
    """Queue projects that are in the DB but not in the review queue yet — e.g.
    ones delivered before reviewing existed. Uses the saved selection snapshot
    when there is one (picks + runner-ups + rejected), else the delivered clips.
    Returns how many clips were queued."""
    init_db()
    with project_store._conn() as c:
        todo = [r["id"] for r in c.execute(
            """SELECT p.id FROM projects p
                WHERE NOT EXISTS (SELECT 1 FROM rating_items i WHERE i.project_id = p.id)""")]
    added = 0
    for pid in todo:
        snap = project_store.load_snapshot(pid)
        if snap and snap.get("shots"):
            with project_store._conn() as c:
                has_shots = c.execute("SELECT 1 FROM project_shots WHERE project_id=?",
                                      (pid,)).fetchone()
            if not has_shots:
                project_store.record_shots(pid, snap["shots"])
            added += record_candidates(pid, snap["shots"])
            continue
        with project_store._conn() as c:
            assets = [dict(r) for r in c.execute(
                "SELECT * FROM project_assets WHERE project_id=? AND kind='clip'", (pid,))]
        rows = [(pid, a["slot_id"], "pick", a["url"], a["page_url"] or a["url"], a["source"],
                 a["title"], a["channel"], a["matched_query"], None, _now())
                for a in assets if a["page_url"] or a["url"]]
        if rows:
            with project_store._conn() as c:
                before = c.total_changes
                c.executemany(
                    """INSERT OR IGNORE INTO rating_items
                       (project_id, slot_id, role, url, page_url, source, title, channel,
                        matched_query, verified_in_sec, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows)
                added += c.total_changes - before
    return added


def upsert_rater(rater_id: int, name: str = "") -> None:
    init_db()
    with project_store._conn() as c:
        c.execute("""INSERT INTO raters (rater_id, name, updated_at) VALUES (?, ?, ?)
                     ON CONFLICT(rater_id) DO UPDATE SET name=excluded.name,
                                                         updated_at=excluded.updated_at""",
                  (rater_id, name or "", _now()))


def _item_public(r: dict, mine: dict = None) -> dict:
    """What the page gets for one clip. The bot's role (pick / runner-up /
    rejected) is deliberately left out so it can't bias the reviewer."""
    return {
        "id": r["id"], "url": r["url"], "page_url": r["page_url"],
        "source": r["source"], "title": r["title"], "channel": r["channel"],
        "youtube_id": youtube_id(r["page_url"] or r["url"]),
        "start": r["verified_in_sec"] or 0,
        "suggested": r["role"] == "suggested",
        "mine": mine,
    }


def next_task(rater_id: int, skip: list = None, project_id: int = None) -> dict | None:
    """The next shot for ``rater_id``: one they haven't fully rated, preferring
    shots with the fewest reviewers so far, newest projects first. ``skip`` is a
    list of ``(project_id, slot_id)`` the reviewer passed on this session;
    ``project_id`` limits it to one project (walked in shot order)."""
    init_db()
    skip = {(int(p), str(s)) for p, s in (skip or [])}
    with project_store._conn() as c:
        groups = c.execute(
            """SELECT i.project_id, i.slot_id,
                      COUNT(DISTINCT r.rater_id) AS n_raters,
                      SUM(CASE WHEN mine.id IS NULL THEN 1 ELSE 0 END) AS todo,
                      COALESCE(s.shot_index, 0) AS shot_index
                 FROM rating_items i
                 LEFT JOIN ratings r ON r.item_id = i.id
                 LEFT JOIN ratings mine ON mine.item_id = i.id AND mine.rater_id = ?
                 LEFT JOIN project_shots s ON s.project_id = i.project_id
                                          AND s.slot_id = i.slot_id
                WHERE (? IS NULL OR i.project_id = ?)
                GROUP BY i.project_id, i.slot_id
               HAVING todo > 0
                ORDER BY CASE WHEN ? IS NULL THEN n_raters ELSE 0 END ASC,
                         i.project_id DESC, shot_index ASC
                LIMIT 200""", (rater_id, project_id, project_id, project_id)).fetchall()
        pick = next((g for g in groups if (g["project_id"], g["slot_id"]) not in skip), None)
        if not pick:
            return None
        pid, slot = pick["project_id"], pick["slot_id"]
        proj = c.execute("SELECT title, topic FROM projects WHERE id=?", (pid,)).fetchone()
        shot = c.execute("SELECT text, start_sec, end_sec FROM project_shots "
                         "WHERE project_id=? AND slot_id=?", (pid, slot)).fetchone()
        items = [dict(r) for r in c.execute(
            "SELECT * FROM rating_items WHERE project_id=? AND slot_id=? ORDER BY id",
            (pid, slot))]
        mine = {r["item_id"]: dict(r) for r in c.execute(
            f"""SELECT * FROM ratings WHERE rater_id=? AND item_id IN
                ({','.join('?' * len(items))})""", (rater_id, *[i['id'] for i in items]))}
    # Stable shuffle per shot so picks aren't always on top.
    items.sort(key=lambda i: zlib.crc32(f"{pid}|{slot}|{i['page_url']}".encode()))
    return {
        "project_id": pid, "slot_id": slot,
        "project": proj["title"] if proj else "", "topic": proj["topic"] if proj else "",
        "text": shot["text"] if shot else "",
        "start": shot["start_sec"] if shot else 0, "end": shot["end_sec"] if shot else 0,
        "items": [_item_public(i, _rating_public(mine.get(i["id"]))) for i in items],
    }


def _rating_public(r: dict | None) -> dict | None:
    if not r:
        return None
    return {"usable": r["usable"], "fit": r["fit"], "reasons": json.loads(r["reasons"] or "[]"),
            "note": r["note"], "segments": json.loads(r["segments"] or "[]")}


# ── saving ──────────────────────────────────────────────────────────────────

def save_rating(rater_id: int, item_id: int, usable: str, fit=None, reasons=None,
                note: str = "", segments=None) -> None:
    """Upsert one reviewer's rating of one clip. Raises ValueError on bad input."""
    init_db()
    if usable not in USABLE:
        raise ValueError(f"usable must be one of {USABLE}")
    if fit is not None:
        fit = int(fit)
        if not 1 <= fit <= 5:
            raise ValueError("fit must be 1–5")
    reasons = [r for r in (reasons or []) if r in _REASON_IDS]
    with project_store._conn() as c:
        if not c.execute("SELECT 1 FROM rating_items WHERE id=?", (item_id,)).fetchone():
            raise ValueError(f"unknown item {item_id}")
        c.execute(
            """INSERT INTO ratings (item_id, rater_id, usable, fit, reasons, note, segments,
                                    created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(item_id, rater_id) DO UPDATE SET
                 usable=excluded.usable, fit=excluded.fit, reasons=excluded.reasons,
                 note=excluded.note, segments=excluded.segments,
                 created_at=excluded.created_at""",
            (item_id, rater_id, usable, fit, json.dumps(reasons), (note or "").strip()[:1000],
             json.dumps(clean_segments(segments)), _now()))


def add_suggestion(rater_id: int, project_id: int, slot_id, url: str, note: str = "",
                   segments=None, title: str = "") -> int:
    """A reviewer's own clip for a shot: queued as a 'suggested' item (so other
    reviewers can rate it too) and rated yes/5 by its suggester. Returns the
    item id. Raises ValueError on a bad URL or unknown shot."""
    init_db()
    canon, source = normalize_suggestion_url(url)
    if not re.match(r"^https?://", canon):
        raise ValueError("paste a full http(s) link")
    slot = str(slot_id)
    with project_store._conn() as c:
        if not c.execute("SELECT 1 FROM rating_items WHERE project_id=? AND slot_id=?",
                         (project_id, slot)).fetchone():
            raise ValueError("unknown shot")
        c.execute(
            """INSERT OR IGNORE INTO rating_items
               (project_id, slot_id, role, url, page_url, source, title, suggested_by,
                created_at)
               VALUES (?, ?, 'suggested', ?, ?, ?, ?, ?, ?)""",
            (project_id, slot, canon, canon, source,
             (title or fetch_title(canon) or canon)[:300], rater_id, _now()))
        item_id = c.execute("SELECT id FROM rating_items WHERE project_id=? AND slot_id=? "
                            "AND page_url=?", (project_id, slot, canon)).fetchone()[0]
    save_rating(rater_id, item_id, "yes", 5, ["great_match"], note, segments)
    return item_id


def fetch_title(url: str) -> str:
    """Video title via YouTube oEmbed (no API key); '' for anything else or on error."""
    if not youtube_id(url):
        return ""
    try:
        import requests
        r = requests.get("https://www.youtube.com/oembed",
                         params={"url": url, "format": "json"}, timeout=6)
        return (r.json() or {}).get("title", "") if r.ok else ""
    except Exception:
        return ""


# ── reading back (for edit_feedback / the bot) ──────────────────────────────

def item_labels() -> list:
    """One row per rated clip, aggregated over reviewers: its line, identity,
    yes/no counts, reviewer notes and good segments."""
    init_db()
    with project_store._conn() as c:
        items = [dict(r) for r in c.execute(
            """SELECT i.* FROM rating_items i
                WHERE EXISTS (SELECT 1 FROM ratings r WHERE r.item_id = i.id)""")]
        ratings = [dict(r) for r in c.execute("SELECT * FROM ratings")]
    by_item: dict = {}
    for r in ratings:
        by_item.setdefault(r["item_id"], []).append(r)
    out = []
    for it in items:
        rs = by_item.get(it["id"], [])
        yes = sum(1 for r in rs if r["usable"] in ("yes", "trim"))
        # "Good clip, wrong line" / "wrong model, OK as general footage" judge the
        # line, not the clip — they mustn't count against the clip itself.
        other_line = sum(1 for r in rs if {"other_line", "generic_ok"}
                         & set(json.loads(r["reasons"] or "[]")))
        no = sum(1 for r in rs if r["usable"] == "no")
        segs = [json.loads(r["segments"] or "[]") for r in rs if r["usable"] in ("yes", "trim")]
        out.append(dict(it, yes=yes, no=no, other_line=other_line,
                        notes=[r["note"] for r in rs if r["note"]],
                        segments=[s for s in segs if s],
                        yes_raters=[r["rater_id"] for r in rs if r["usable"] in ("yes", "trim")],
                        no_raters=[r["rater_id"] for r in rs if r["usable"] == "no"],
                        seg_raters=[r["rater_id"] for r in rs if r["usable"] in ("yes", "trim")
                                    and json.loads(r["segments"] or "[]")]))
    return out


def rater_stats() -> list:
    """Per reviewer: clips rated, notes written, agreement with other reviewers
    and with the real editor (on clips the editor's XML labelled)."""
    init_db()
    with project_store._conn() as c:
        rows = [dict(r) for r in c.execute(
            """SELECT r.rater_id, r.item_id, r.usable, r.note, i.project_id, i.slot_id,
                      i.page_url, COALESCE(rt.name, '') AS name
                 FROM ratings r JOIN rating_items i ON i.id = r.item_id
                 LEFT JOIN raters rt ON rt.rater_id = r.rater_id""")]
        verdicts = {(r["project_id"], r["page_url"]): r["verdict"] for r in c.execute(
            """SELECT a.project_id, a.page_url, a.verdict FROM project_assets a
                 JOIN projects p ON p.id = a.project_id
                WHERE p.status = 'learned' AND a.kind = 'clip' AND a.page_url != ''""")}
    good = lambda u: u in ("yes", "trim")
    by_item: dict = {}
    for r in rows:
        by_item.setdefault(r["item_id"], []).append(r)
    stats: dict = {}
    for r in rows:
        s = stats.setdefault(r["rater_id"], {"rater_id": r["rater_id"], "name": r["name"],
                                             "rated": 0, "notes": 0, "peer_agree": 0,
                                             "peer_shared": 0, "editor_agree": 0,
                                             "editor_shared": 0})
        s["rated"] += 1
        s["notes"] += bool(r["note"])
        for o in by_item[r["item_id"]]:
            if o["rater_id"] != r["rater_id"]:
                s["peer_shared"] += 1
                s["peer_agree"] += good(o["usable"]) == good(r["usable"])
        v = verdicts.get((r["project_id"], r["page_url"]))
        if v in ("used_here", "used_elsewhere", "unused"):
            s["editor_shared"] += 1
            s["editor_agree"] += good(r["usable"]) == (v != "unused")

    # Label checks and picks changed (reviewers who only check labels count too).
    with project_store._conn() as c:
        labels = {r[0]: r[1] for r in c.execute(
            "SELECT rater_id, COUNT(*) FROM label_reviews GROUP BY rater_id")}
        changed = {r[0]: r[1] for r in c.execute(
            f"""SELECT rater_id, COUNT(*) FROM impact_events
                WHERE kind NOT IN ({','.join('?' * len(WORK_KINDS))}) GROUP BY rater_id""",
            sorted(WORK_KINDS))}
        names = {r[0]: r[1] for r in c.execute("SELECT rater_id, name FROM raters")}
    for rid in set(labels) | set(changed):
        stats.setdefault(rid, {"rater_id": rid, "name": names.get(rid, ""), "rated": 0,
                               "notes": 0, "peer_agree": 0, "peer_shared": 0,
                               "editor_agree": 0, "editor_shared": 0})
    for rid, s in stats.items():
        s["labels"] = labels.get(rid, 0)
        s["changed"] = changed.get(rid, 0)
    return sorted(stats.values(), key=lambda s: -(s["rated"] + s["labels"]))


# ── checking editor-XML labels ──────────────────────────────────────────────

def next_label_task(rater_id: int, skip: list = None, project_id: int = None) -> dict | None:
    """The next shot of a learned project whose automatic labels (from the
    editor's XML) this reviewer hasn't checked yet, fewest reviews first
    (or, with ``project_id``, that project's shots in order)."""
    init_db()
    skip = {(int(p), str(s)) for p, s in (skip or [])}
    with project_store._conn() as c:
        groups = c.execute(
            """SELECT a.project_id, a.slot_id,
                      COUNT(DISTINCT lr.rater_id) AS n_reviews,
                      SUM(CASE WHEN mine.id IS NULL THEN 1 ELSE 0 END) AS todo,
                      COALESCE(s.shot_index, 0) AS shot_index
                 FROM project_assets a
                 JOIN projects p ON p.id = a.project_id AND p.status = 'learned'
                 LEFT JOIN label_reviews lr ON lr.asset_id = a.id
                 LEFT JOIN label_reviews mine ON mine.asset_id = a.id AND mine.rater_id = ?
                 LEFT JOIN project_shots s ON s.project_id = a.project_id
                                          AND s.slot_id = a.slot_id
                WHERE a.verdict != '' AND (? IS NULL OR a.project_id = ?)
                GROUP BY a.project_id, a.slot_id
               HAVING todo > 0
                ORDER BY CASE WHEN ? IS NULL THEN n_reviews ELSE 0 END ASC,
                         a.project_id DESC, shot_index ASC
                LIMIT 200""", (rater_id, project_id, project_id, project_id)).fetchall()
        pick = next((g for g in groups if (g["project_id"], g["slot_id"]) not in skip), None)
        if not pick:
            return None
        pid, slot = pick["project_id"], pick["slot_id"]
        proj = c.execute("SELECT title, topic FROM projects WHERE id=?", (pid,)).fetchone()
        lines = [dict(r) for r in c.execute(
            """SELECT slot_id, text, start_sec, end_sec FROM project_shots
                WHERE project_id=? AND is_extra=0 ORDER BY shot_index""", (pid,))]
        assets = [dict(r) for r in c.execute(
            "SELECT * FROM project_assets WHERE project_id=? AND slot_id=? ORDER BY kind, position",
            (pid, slot))]
        ids = [a["id"] for a in assets]
        qs = ",".join("?" * len(ids))
        usage = {}
        for u in c.execute(f"""SELECT * FROM asset_usage WHERE asset_id IN ({qs})
                               ORDER BY timeline_start_sec""", ids):
            usage.setdefault(u["asset_id"], dict(u))
        mine = {r["asset_id"]: dict(r) for r in c.execute(
            f"SELECT * FROM label_reviews WHERE rater_id=? AND asset_id IN ({qs})",
            (rater_id, *ids))}
    text_of = {l["slot_id"]: l["text"] for l in lines}
    shot = next((l for l in lines if l["slot_id"] == slot), {})
    items = []
    for a in assets:
        u = usage.get(a["id"]) or {}
        m = mine.get(a["id"])
        items.append({
            "asset_id": a["id"], "kind": a["kind"], "url": a["url"],
            "page_url": a["page_url"], "source": a["source"], "title": a["title"] or a["filename"],
            "channel": a["channel"], "youtube_id": youtube_id(a["page_url"] or a["url"]),
            "verdict": a["verdict"], "used_slot_id": a["used_slot_id"],
            "used_line": text_of.get(a["used_slot_id"], "") if a["verdict"] == "used_elsewhere" else "",
            "in_sec": u.get("in_sec"), "out_sec": u.get("out_sec"),
            "start": u.get("in_sec") or a["exported_in_sec"] or 0,
            "mine": {"action": m["action"], "verdict": m["verdict"],
                     "used_slot_id": m["used_slot_id"], "note": m["note"]} if m else None,
        })
    return {"mode": "labels", "project_id": pid, "slot_id": slot,
            "project": proj["title"] if proj else "", "topic": proj["topic"] if proj else "",
            "text": shot.get("text", ""), "start": shot.get("start_sec", 0),
            "end": shot.get("end_sec", 0),
            "lines": [{"slot_id": l["slot_id"], "text": l["text"][:90]} for l in lines],
            "items": items}


def save_label_review(rater_id: int, asset_id: int, action: str, verdict: str = None,
                      used_slot_id: str = "", note: str = "") -> None:
    """Confirm an automatic label, or correct it. A confirm stores the automatic
    verdict itself, so a label's reviews are a plain majority vote."""
    init_db()
    if action not in ("confirm", "correct"):
        raise ValueError("action must be confirm or correct")
    with project_store._conn() as c:
        a = c.execute("SELECT verdict, used_slot_id, page_url, title, filename, project_id "
                      "FROM project_assets WHERE id=?", (asset_id,)).fetchone()
        if not a or not a["verdict"]:
            raise ValueError(f"unknown or unlabelled asset {asset_id}")
        if action == "confirm":
            verdict, used_slot_id = a["verdict"], a["used_slot_id"]
        elif verdict not in LABEL_VERDICTS:
            raise ValueError(f"verdict must be one of {LABEL_VERDICTS}")
        if verdict != "used_elsewhere":
            used_slot_id = ""
        elif not used_slot_id:
            raise ValueError("say which line it belongs to")
        c.execute(
            """INSERT INTO label_reviews (asset_id, rater_id, action, verdict, used_slot_id,
                                          note, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(asset_id, rater_id) DO UPDATE SET action=excluded.action,
                 verdict=excluded.verdict, used_slot_id=excluded.used_slot_id,
                 note=excluded.note, created_at=excluded.created_at""",
            (asset_id, rater_id, action, verdict, str(used_slot_id or ""),
             (note or "").strip()[:1000], _now()))
    log_impact([rater_id], "label_confirmed" if action == "confirm" else "label_corrected",
               f"asset:{asset_id}", a["title"] or a["filename"], a["project_id"], once=True)


def reviewed_labels() -> dict:
    """``{asset_id: (verdict, used_slot_id)}`` — the reviewers' majority label
    for every editor-XML asset they checked (ties keep the automatic label)."""
    init_db()
    with project_store._conn() as c:
        rows = [dict(r) for r in c.execute(
            """SELECT lr.asset_id, lr.verdict, lr.used_slot_id, a.verdict AS auto,
                      a.used_slot_id AS auto_slot
                 FROM label_reviews lr JOIN project_assets a ON a.id = lr.asset_id""")]
    votes: dict = {}
    auto: dict = {}
    for r in rows:
        votes.setdefault(r["asset_id"], []).append((r["verdict"], r["used_slot_id"]))
        auto[r["asset_id"]] = (r["auto"], r["auto_slot"])
    out = {}
    for aid, vs in votes.items():
        (best, n), *rest = Counter(vs).most_common()
        out[aid] = auto[aid] if rest and rest[0][1] == n else best
    return out


# ── contribution log ────────────────────────────────────────────────────────

def log_impact(rater_ids, kind: str, clip_key: str = "", title: str = "",
               project_id=None, once: bool = False) -> None:
    """Credit reviewers for a moment their work changed what the picker did.
    Deduped per reviewer, kind and clip per day (``once``: ever), so a clip
    re-ranked on every run doesn't inflate anyone's count. Best-effort."""
    rater_ids = sorted({int(r) for r in rater_ids or [] if r is not None})
    if not rater_ids or kind not in IMPACT_KINDS:
        return
    day = "" if once else datetime.now().strftime("%Y-%m-%d")
    try:
        init_db()
        with project_store._conn() as c:
            c.executemany(
                """INSERT OR IGNORE INTO impact_events
                   (rater_id, kind, clip_key, title, project_id, day, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(r, kind, clip_key or "", (title or "")[:200], project_id, day, _now())
                 for r in rater_ids])
    except Exception as e:
        print(f"[ratings] couldn't log impact: {e}")


def contribution_summary(rater_id: int, recent: int = 25) -> dict:
    """What a reviewer has done and what it changed, for the 'My impact' tab
    and the bot."""
    init_db()
    with project_store._conn() as c:
        work = c.execute(
            """SELECT COUNT(*) AS rated,
                      SUM(CASE WHEN note != '' THEN 1 ELSE 0 END) AS notes,
                      SUM(CASE WHEN segments != '[]' THEN 1 ELSE 0 END) AS segments
                 FROM ratings WHERE rater_id=?""", (rater_id,)).fetchone()
        suggested = c.execute("SELECT COUNT(*) FROM rating_items WHERE suggested_by=?",
                              (rater_id,)).fetchone()[0]
        labels = c.execute("SELECT COUNT(*) FROM label_reviews WHERE rater_id=?",
                           (rater_id,)).fetchone()[0]
        impact = {k: 0 for k in IMPACT_KINDS}
        for r in c.execute("SELECT kind, COUNT(*) AS n FROM impact_events WHERE rater_id=? "
                           "GROUP BY kind", (rater_id,)):
            impact[r["kind"]] = r["n"]
        log = [dict(r) for r in c.execute(
            """SELECT kind, title, created_at FROM impact_events WHERE rater_id=?
                ORDER BY id DESC LIMIT ?""", (rater_id, recent))]
    mine = next((s for s in rater_stats() if s["rater_id"] == rater_id), None) or {}
    pct = lambda a, n: round(100 * a / n) if n else None
    return {
        "rated": work["rated"] or 0, "notes": work["notes"] or 0,
        "segments": work["segments"] or 0, "suggested": suggested, "labels_checked": labels,
        "impact": impact,
        "impact_labels": IMPACT_KINDS,
        "peer_agreement": pct(mine.get("peer_agree", 0), mine.get("peer_shared", 0)),
        "editor_agreement": pct(mine.get("editor_agree", 0), mine.get("editor_shared", 0)),
        "log": [dict(e, label=IMPACT_KINDS.get(e["kind"], e["kind"])) for e in log],
    }


def project_progress(rater_id: int, limit: int = 100) -> list:
    """Projects that have something to review, newest first, with how far this
    reviewer (and everyone) has got — for the page's project picker."""
    init_db()
    with project_store._conn() as c:
        rows = c.execute(
            """SELECT p.id, p.title, p.created_at, p.status,
                      (SELECT COUNT(*) FROM rating_items i WHERE i.project_id = p.id) AS clips,
                      (SELECT COUNT(*) FROM rating_items i JOIN ratings r ON r.item_id = i.id
                        WHERE i.project_id = p.id AND r.rater_id = ?) AS clips_mine,
                      (SELECT COUNT(DISTINCT i.id) FROM rating_items i JOIN ratings r
                          ON r.item_id = i.id WHERE i.project_id = p.id) AS clips_any,
                      (SELECT COUNT(*) FROM project_assets a WHERE a.project_id = p.id
                          AND p.status = 'learned' AND a.verdict != '') AS labels,
                      (SELECT COUNT(*) FROM project_assets a JOIN label_reviews lr
                          ON lr.asset_id = a.id WHERE a.project_id = p.id
                          AND lr.rater_id = ?) AS labels_mine
                 FROM projects p ORDER BY p.id DESC LIMIT ?""",
            (rater_id, rater_id, limit)).fetchall()
    return [dict(r) for r in rows if r["clips"] or r["labels"]]


def queue_size() -> dict:
    init_db()
    with project_store._conn() as c:
        items = c.execute("SELECT COUNT(*) FROM rating_items").fetchone()[0]
        rated = c.execute("SELECT COUNT(DISTINCT item_id) FROM ratings").fetchone()[0]
        labels = c.execute(
            """SELECT COUNT(*) FROM project_assets a JOIN projects p ON p.id = a.project_id
                WHERE p.status = 'learned' AND a.verdict != ''""").fetchone()[0]
        checked = c.execute("SELECT COUNT(DISTINCT asset_id) FROM label_reviews").fetchone()[0]
    return {"items": items, "rated": rated, "labels": labels, "labels_checked": checked}


# ── house rules ─────────────────────────────────────────────────────────────

_RULES_PROMPT = (
    "You maintain the house rules a B-roll picker follows. Below are human "
    "reviewers' notes on clips they judged for narrated videos (usable or not, and "
    "why), each with the narration line. Distil them into at most 12 short, concrete, "
    "general rules the picker should follow for future videos (e.g. 'Never use "
    "news-anchor or presenter footage for car-repair topics'). Merge duplicates, "
    "drop one-off opinions that don't generalise, keep the reviewers' wording where "
    "it's already a rule. Return STRICT JSON: {\"rules\": [\"...\", ...]}"
)

_RULES_LOCK = threading.Lock()


def get_house_rules() -> list:
    try:
        init_db()
        with project_store._conn() as c:
            row = c.execute("SELECT rules FROM house_rules WHERE id=1").fetchone()
        return json.loads(row["rules"]) if row else []
    except Exception:
        return []


def _noted_ratings(limit: int = 300) -> list:
    with project_store._conn() as c:
        return [dict(r) for r in c.execute(
            """SELECT r.id, r.rater_id, r.usable, r.reasons, r.note, i.title, i.source,
                      COALESCE(s.text, '') AS line
                 FROM ratings r JOIN rating_items i ON i.id = r.item_id
                 LEFT JOIN project_shots s ON s.project_id = i.project_id
                                          AND s.slot_id = i.slot_id
                WHERE r.note != '' ORDER BY r.id DESC LIMIT ?""", (limit,))]


def distill_house_rules(force: bool = False, api_key: str = None) -> list | None:
    """Re-write the house rules from reviewers' notes when enough new notes
    have come in (or ``force``). Returns the new rules, or None when skipped."""
    init_db()
    with _RULES_LOCK:
        with project_store._conn() as c:
            n_notes = c.execute("SELECT COUNT(*) FROM ratings WHERE note != ''").fetchone()[0]
            row = c.execute("SELECT notes_seen FROM house_rules WHERE id=1").fetchone()
        seen = row["notes_seen"] if row else 0
        if not n_notes or (not force and n_notes - seen < _RULES_EVERY):
            return None
        lines = []
        noted = _noted_ratings()
        for r in noted:
            verdict = {"yes": "USABLE", "trim": "USABLE WITH TRIM", "no": "NOT USABLE"}[r["usable"]]
            lines.append(f'- line "{r["line"][:120]}" | [{(r["source"] or "?").upper()}] '
                         f'{r["title"][:80]} | {verdict} | {r["note"][:300]}')
        from groq import Groq
        from core.keywords import _call_llm_json
        key = api_key or os.getenv("GROQ_API_KEY") or "unused"
        data = _call_llm_json(Groq(api_key=key), _RULES_PROMPT, "\n".join(lines),
                              temperature=0.2, max_tokens=1500, tier="smart")
        rules = [str(x).strip() for x in (data.get("rules") or []) if str(x).strip()][:12]
        with project_store._conn() as c:
            c.execute("""INSERT INTO house_rules (id, rules, notes_seen, updated_at)
                         VALUES (1, ?, ?, ?)
                         ON CONFLICT(id) DO UPDATE SET rules=excluded.rules,
                              notes_seen=excluded.notes_seen, updated_at=excluded.updated_at""",
                      (json.dumps(rules), n_notes, _now()))
        for r in noted:
            log_impact([r["rater_id"]], "note_in_rules", f"rating:{r['id']}", r["title"],
                       once=True)
        return rules


def maybe_distill_async() -> None:
    """Kick off a background re-distil if enough new notes arrived."""
    def _run():
        try:
            distill_house_rules()
        except Exception as e:
            print(f"[ratings] house rules distil failed: {e}")
    threading.Thread(target=_run, daemon=True, name="HouseRules").start()
