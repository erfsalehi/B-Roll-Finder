"""
Project store — every bot project, what it picked, and what the editor kept.

The learning loop needs ground truth: for each delivered project we remember
who ran it, every clip and per-shot image we handed the editor (by file name
and source URL), and where each one sat on the exported timeline. When the
editor later sends back their finished Premiere XML and says which project it
belongs to, :func:`analyze_import` compares the two and labels every asset:

    used_here       used, on (or overlapping) the shot we picked it for
    used_elsewhere  used, but the editor moved it to a different shot
    unused          delivered but never placed on the final timeline
    external        a clip/image on the final timeline we never supplied

plus the source in/out the editor actually used. Those labels are what later
ranking / trim learning reads.

SQLite at ``.cache/projects.db`` (the persistent volume on the server);
override with ``PROJECTS_DB``.
"""

import json
import os
import sqlite3
from collections import Counter
from datetime import datetime

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", ".cache", "projects.db")

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mxf")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff")

# An image is "on its shot" if it lands within this many seconds of the shot's
# range — editors often let a still start a beat early or late.
_IMAGE_SLACK_SEC = 1.0


def _db_path() -> str:
    return os.getenv("PROJECTS_DB", "").strip() or _DB_PATH


def _conn() -> sqlite3.Connection:
    path = _db_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    return c


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_db() -> None:
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                title         TEXT    NOT NULL,
                project_name  TEXT    DEFAULT '',
                operator_id   INTEGER,
                operator_name TEXT    DEFAULT '',
                chat_id       INTEGER,
                topic         TEXT    DEFAULT '',
                status        TEXT    DEFAULT 'running',
                ignore_names  TEXT    DEFAULT '[]',
                created_at    TEXT    DEFAULT '',
                updated_at    TEXT    DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS project_shots (
                project_id  INTEGER NOT NULL,
                slot_id     TEXT    NOT NULL,
                shot_index  INTEGER DEFAULT 0,
                start_sec   REAL    DEFAULT 0,
                end_sec     REAL    DEFAULT 0,
                text        TEXT    DEFAULT '',
                queries     TEXT    DEFAULT '[]',
                is_extra    INTEGER DEFAULT 0,
                PRIMARY KEY (project_id, slot_id)
            );
            CREATE TABLE IF NOT EXISTS project_assets (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id          INTEGER NOT NULL,
                slot_id             TEXT    DEFAULT '',
                kind                TEXT    NOT NULL,          -- clip | image
                position            INTEGER DEFAULT 0,
                filename            TEXT    NOT NULL,
                url                 TEXT    DEFAULT '',
                source              TEXT    DEFAULT '',
                title               TEXT    DEFAULT '',
                matched_query       TEXT    DEFAULT '',
                page_url            TEXT    DEFAULT '',
                exported_start_sec  REAL,
                exported_end_sec    REAL,
                exported_in_sec     REAL,
                exported_out_sec    REAL,
                verdict             TEXT    DEFAULT '',        -- latest import's label
                used_slot_id        TEXT    DEFAULT '',
                learned_at          TEXT    DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_assets_project ON project_assets(project_id);
            CREATE INDEX IF NOT EXISTS idx_assets_filename ON project_assets(filename);
            CREATE TABLE IF NOT EXISTS xml_imports (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id   INTEGER NOT NULL,
                imported_by  INTEGER,
                filename     TEXT    DEFAULT '',
                shift_sec    REAL    DEFAULT 0,
                summary      TEXT    DEFAULT '{}',
                imported_at  TEXT    DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS asset_usage (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                import_id          INTEGER NOT NULL,
                project_id         INTEGER NOT NULL,
                asset_id           INTEGER,                    -- NULL for external
                kind               TEXT    DEFAULT '',
                name               TEXT    DEFAULT '',
                verdict            TEXT    NOT NULL,
                used_slot_id       TEXT    DEFAULT '',
                timeline_start_sec REAL,
                timeline_end_sec   REAL,
                in_sec             REAL,
                out_sec            REAL
            );
            CREATE INDEX IF NOT EXISTS idx_usage_project ON asset_usage(project_id);
        """)
        # Columns added after the first release; ALTER is a no-op error once present.
        for table, col in (("project_assets", "channel TEXT DEFAULT ''"),
                           ("project_assets", "in_rule TEXT DEFAULT ''"),
                           ("projects", "snapshot TEXT"),
                           ("project_assets", "segment_id INTEGER"),
                           ("project_shots", "embedding BLOB")):
            try:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass


def creator_of(c: dict) -> str:
    """The YouTube channel / Pexels author of a candidate, or ''. YouTube API
    results carry ``channel``; otherwise it's packed into the description as
    'by <name> — …' (YouTube) or 'By <name>' (Pexels)."""
    if c.get("channel"):
        return str(c["channel"]).strip()
    desc = (c.get("description") or "").strip()
    if desc[:3].lower() == "by ":
        return desc[3:].split(" — ")[0].strip()
    return ""


def asset_key(c: dict) -> str:
    """Stable identity of a clip across projects: the video page (a Pexels file
    link changes with the chosen quality; the page doesn't)."""
    return (c.get("page_url") or c.get("url") or "").strip()


# ── projects ────────────────────────────────────────────────────────────────

def create_project(title: str, project_name: str = "", operator_id=None,
                   operator_name: str = "", chat_id=None) -> int:
    init_db()
    now = _now()
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO projects (title, project_name, operator_id, operator_name,
                                     chat_id, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'running', ?, ?)""",
            (title or project_name or "untitled", project_name or "", operator_id,
             operator_name or "", chat_id, now, now))
        return cur.lastrowid


def set_status(project_id: int, status: str, topic: str = None) -> None:
    init_db()
    with _conn() as c:
        if topic is not None:
            c.execute("UPDATE projects SET status=?, topic=?, updated_at=? WHERE id=?",
                      (status, topic, _now(), project_id))
        else:
            c.execute("UPDATE projects SET status=?, updated_at=? WHERE id=?",
                      (status, _now(), project_id))


def get_project(project_id: int) -> dict | None:
    init_db()
    with _conn() as c:
        row = c.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    return dict(row) if row else None


def list_projects(limit: int = 10) -> list:
    """Most recent projects first, each with asset counts."""
    init_db()
    with _conn() as c:
        rows = c.execute(
            """SELECT p.id, p.title, p.project_name, p.operator_id, p.operator_name,
                      p.chat_id, p.topic, p.status, p.created_at, p.updated_at,
                      (SELECT COUNT(*) FROM project_assets a
                        WHERE a.project_id=p.id AND a.kind='clip')  AS n_clips,
                      (SELECT COUNT(*) FROM project_assets a
                        WHERE a.project_id=p.id AND a.kind='image') AS n_images,
                      (SELECT COUNT(*) FROM xml_imports x
                        WHERE x.project_id=p.id)                     AS n_imports
                 FROM projects p ORDER BY p.id DESC LIMIT ?""", (limit,)).fetchall()
    return [dict(r) for r in rows]


def project_names() -> set:
    """Every folder name a recorded project has used."""
    init_db()
    with _conn() as c:
        return {r[0] for r in c.execute("SELECT DISTINCT project_name FROM projects")
                if r[0]}


def find_projects(query: str = "", limit: int = 8) -> list:
    """Projects matching ``query`` (title / folder name contains it, or ``#id``),
    newest first; the most recent ones when ``query`` is empty. Rows carry a
    ``has_snapshot`` flag — whether the selection can be rebuilt from the DB."""
    init_db()
    q = (query or "").strip()
    sql = """SELECT id, title, project_name, operator_name, status, created_at,
                    snapshot IS NOT NULL AS has_snapshot FROM projects"""
    args: list = []
    if q.lstrip("#").isdigit():
        sql += " WHERE id = ?"
        args.append(int(q.lstrip("#")))
    elif q:
        sql += " WHERE title LIKE ? OR project_name LIKE ?"
        args += [f"%{q}%", f"%{q}%"]
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with _conn() as c:
        return [dict(r) for r in c.execute(sql, args)]


# Result fields worth keeping to rebuild a project later (/download of an old
# project): the full selection plus what the XML and summary need.
_SNAPSHOT_KEYS = ("shots", "overlays", "sfx_list", "topic", "qa", "n_shots",
                  "n_selected", "n_clips")


def save_snapshot(project_id: int, result: dict, quality=None) -> None:
    """Store the project's selection (shots with their candidates, overlays…)
    so it can be re-downloaded or reviewed long after the bot's in-memory state
    is gone."""
    init_db()
    data = {k: result.get(k) for k in _SNAPSHOT_KEYS if result.get(k) is not None}
    if quality is not None:
        data["quality"] = str(quality)
    with _conn() as c:
        c.execute("UPDATE projects SET snapshot=?, updated_at=? WHERE id=?",
                  (json.dumps(data, default=str), _now(), project_id))


def load_snapshot(project_id: int) -> dict | None:
    init_db()
    with _conn() as c:
        row = c.execute("SELECT snapshot FROM projects WHERE id=?", (project_id,)).fetchone()
    if not row or not row["snapshot"]:
        return None
    try:
        return json.loads(row["snapshot"])
    except ValueError:
        return None


def _write_shots(c, project_id: int, shots: list) -> None:
    c.execute("DELETE FROM project_shots WHERE project_id=?", (project_id,))
    for idx, shot in enumerate(shots or []):
        c.execute(
            """INSERT OR REPLACE INTO project_shots
               (project_id, slot_id, shot_index, start_sec, end_sec, text, queries, is_extra)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (project_id, _slot(shot.get("slot_id", idx)), idx, float(shot.get("timestamp") or 0),
             float(shot.get("end_timestamp") or shot.get("timestamp") or 0),
             shot.get("text") or "", json.dumps(shot.get("search_queries") or []),
             1 if shot.get("is_extra") else 0))


def record_shots(project_id: int, shots: list) -> None:
    """Store the shot list (narration lines + timing) on its own — used at the
    review gate, before anything is downloaded, so reviewers can start."""
    init_db()
    with _conn() as c:
        _write_shots(c, project_id, shots)


# ── recording what was delivered ────────────────────────────────────────────

def _slot(v) -> str:
    return "" if v is None else str(v)


def _exported_placements(xml_path: str) -> tuple[dict, set]:
    """``({basename_lower: first clipitem}, {every basename in the XML})`` from the
    XML we exported, so we know where each asset sat and in-point we suggested."""
    if not xml_path or not os.path.exists(xml_path):
        return {}, set()
    from core.xml_reimport import parse_fcpxml
    try:
        items = parse_fcpxml(xml_path)
    except Exception:
        return {}, set()
    first, names = {}, set()
    for it in items:
        key = _item_key(it)
        if not key:
            continue
        names.add(key)
        first.setdefault(key, it)
    return first, names


def _item_key(it: dict) -> str:
    return os.path.basename(it.get("local_path") or it.get("name") or "").lower()


def _timeline_secs(it: dict):
    """Timeline (start, end) seconds of a parsed clipitem, or (None, None).
    Premiere writes -1 for an edge that touches a transition; recover it from the
    other edge plus the source length."""
    fps = it.get("fps") or 0
    if fps <= 0:
        return None, None
    s, e = it.get("start_frame"), it.get("end_frame")
    length = (it.get("out_frame") or 0) - (it.get("in_frame") or 0)
    s = s if s is not None and s >= 0 else None
    e = e if e is not None and e >= 0 else None
    if s is None and e is not None and length > 0:
        s = e - length
    if e is None and s is not None and length > 0:
        e = s + length
    if s is None or e is None:
        return None, None
    return s / fps, e / fps


def record_delivery(project_id: int, shots: list, xml_path: str = None,
                    topic: str = None) -> dict:
    """Snapshot what the editor received: shots, every downloaded clip and every
    per-shot image (file name + URL), and each clip's exported placement.
    Replaces any earlier snapshot for this project (a /refine then /download
    re-delivers). Returns ``{"clips": n, "images": n}``."""
    init_db()
    placed, xml_names = _exported_placements(xml_path)
    n_clips = n_images = 0
    asset_names = set()
    with _conn() as c:
        _write_shots(c, project_id, shots)
        c.execute("DELETE FROM project_assets WHERE project_id=?", (project_id,))
        for idx, shot in enumerate(shots or []):
            slot = _slot(shot.get("slot_id", idx))

            for pos, res in enumerate(shot.get("selected_results") or []):
                path = res.get("local_path") or ""
                if not path or res.get("_dl_failed"):
                    continue            # never reached the editor
                fname = os.path.basename(path)
                it = placed.get(fname.lower()) or {}
                ts, te = _timeline_secs(it) if it else (None, None)
                c.execute(
                    """INSERT INTO project_assets
                       (project_id, slot_id, kind, position, filename, url, source, title,
                        matched_query, page_url, channel, in_rule, segment_id,
                        exported_start_sec, exported_end_sec, exported_in_sec, exported_out_sec)
                       VALUES (?, ?, 'clip', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (project_id, slot, pos, fname, res.get("url") or "",
                     res.get("source") or "", res.get("title") or "",
                     res.get("matched_query") or "", asset_key(res), creator_of(res),
                     res.get("in_rule") or "", res.get("library_segment_id"),
                     ts, te, it.get("in_seconds"), it.get("out_seconds")))
                asset_names.add(fname.lower())
                n_clips += 1

            for pos, img in enumerate(shot.get("images") or []):
                path = img.get("local_path") or ""
                if not path:
                    continue
                fname = os.path.basename(path)
                c.execute(
                    """INSERT INTO project_assets
                       (project_id, slot_id, kind, position, filename, url, source,
                        title, page_url, segment_id)
                       VALUES (?, ?, 'image', ?, ?, ?, ?, ?, ?, ?)""",
                    (project_id, slot, pos, fname, img.get("url") or "",
                     "library" if img.get("library_segment_id") else "google",
                     img.get("title") or "", img.get("page") or "",
                     img.get("library_segment_id")))
                asset_names.add(fname.lower())
                n_images += 1

        # Things we exported that aren't clips/images (overlays, SFX) must not
        # show up as "external" footage when the editor's XML comes back.
        ignore = sorted(xml_names - asset_names)
        c.execute("UPDATE projects SET ignore_names=?, status='delivered', updated_at=?"
                  + (", topic=?" if topic is not None else "") + " WHERE id=?",
                  (json.dumps(ignore), _now(), *([topic] if topic is not None else []),
                   project_id))
    return {"clips": n_clips, "images": n_images}


# ── matching an uploaded XML to a project ───────────────────────────────────

def match_counts(clipitems: list) -> dict:
    """``{project_id: number of distinct asset files found in the XML}``, so the
    bot can suggest which project an uploaded XML belongs to."""
    init_db()
    names = sorted({_item_key(it) for it in clipitems} - {""})
    if not names:
        return {}
    out: dict = {}
    with _conn() as c:
        for i in range(0, len(names), 500):          # SQLite variable limit
            chunk = names[i:i + 500]
            qs = ",".join("?" * len(chunk))
            for r in c.execute(
                    f"""SELECT project_id, COUNT(DISTINCT lower(filename)) AS n
                          FROM project_assets WHERE lower(filename) IN ({qs})
                         GROUP BY project_id""", chunk):
                out[r["project_id"]] = out.get(r["project_id"], 0) + r["n"]
    return out


# ── analysis ────────────────────────────────────────────────────────────────

def _estimate_shift(deltas: list) -> float:
    """The editor may have moved the whole edit (an intro, a cold open). If most
    matched clips moved by the same amount, that's a global shift, not a
    per-clip decision — return it so it doesn't read as "used elsewhere"."""
    if len(deltas) < 3:
        return 0.0
    buckets = Counter(round(d * 2) / 2 for d in deltas)
    bucket, count = buckets.most_common(1)[0]
    if bucket == 0 or count < max(3, 0.4 * len(deltas)):
        return 0.0
    near = sorted(d for d in deltas if abs(d - bucket) <= 0.5)
    return near[len(near) // 2]


def _shot_ranges(shots: list) -> list:
    """``[(slot_id, start, end)]`` for main-timeline shots, each running until the
    next shot starts (the export stretches clips over the pauses between)."""
    main = sorted((s for s in shots if not s["is_extra"]), key=lambda s: s["start_sec"])
    out = []
    for i, s in enumerate(main):
        end = main[i + 1]["start_sec"] if i + 1 < len(main) else max(s["end_sec"], s["start_sec"])
        out.append((s["slot_id"], s["start_sec"], end))
    return out


def _slot_at(ranges: list, t: float) -> str:
    if not ranges:
        return ""
    hit = ranges[0][0]
    for slot, start, _end in ranges:
        if start <= t:
            hit = slot
        else:
            break
    return hit


def analyze_import(project_id: int, clipitems: list, imported_by=None,
                   filename: str = "") -> dict:
    """Compare the editor's final timeline (parsed clipitems) with what we
    delivered for ``project_id`` and store a label for every asset. Re-importing
    replaces the previous labels. Returns a summary dict for the bot to show."""
    init_db()
    with _conn() as c:
        proj = c.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if not proj:
            raise ValueError(f"project {project_id} not found")
        shots = [dict(r) for r in c.execute(
            "SELECT * FROM project_shots WHERE project_id=?", (project_id,))]
        assets = [dict(r) for r in c.execute(
            "SELECT * FROM project_assets WHERE project_id=?", (project_id,))]
    ignore = set(json.loads(proj["ignore_names"] or "[]"))
    by_name = {a["filename"].lower(): a for a in assets}
    ranges = _shot_ranges(shots)
    own_range = {s: (a, b) for s, a, b in ranges}

    # Only picture: skip voiceover/SFX/music and anything we exported that isn't
    # footage (overlays).
    usages, external = [], []
    for it in clipitems:
        key = _item_key(it)
        ext = os.path.splitext(key)[1]
        if not key or key in ignore or (ext and ext not in VIDEO_EXTS + IMAGE_EXTS):
            continue
        ts, te = _timeline_secs(it)
        (usages if key in by_name else external).append((key, it, ts, te))

    deltas = [ts - by_name[k]["exported_start_sec"] for k, _it, ts, _te in usages
              if ts is not None and by_name[k]["exported_start_sec"] is not None]
    shift = _estimate_shift(deltas)

    rows = []            # (asset_id, kind, name, verdict, used_slot, ts, te, in, out)
    best: dict = {}      # asset_id → (verdict, used_slot); used_here beats elsewhere
    in_moves = []
    for key, it, ts, te in usages:
        a = by_name[key]
        verdict, used_slot = "used_here", a["slot_id"]
        if ts is not None:
            ts, te = ts - shift, te - shift
            mid = (ts + te) / 2
            es, ee = a["exported_start_sec"], a["exported_end_sec"]
            overlaps = es is not None and ee is not None and min(te, ee) - max(ts, es) > 0
            lo, hi = own_range.get(a["slot_id"], (None, None))
            slack = _IMAGE_SLACK_SEC if a["kind"] == "image" else 0.0
            on_own = lo is not None and lo - slack <= mid <= hi + slack
            if not (overlaps or on_own):
                verdict, used_slot = "used_elsewhere", _slot_at(ranges, mid)
        if a["kind"] == "clip" and a["exported_in_sec"] is not None:
            in_moves.append(it["in_seconds"] - a["exported_in_sec"])
        rows.append((a["id"], a["kind"], a["filename"], verdict, used_slot,
                     ts, te, it.get("in_seconds"), it.get("out_seconds")))
        prev = best.get(a["id"])
        if prev is None or (prev[0] != "used_here" and verdict == "used_here"):
            best[a["id"]] = (verdict, used_slot)

    for key, it, ts, te in external:
        slot = _slot_at(ranges, (ts - shift + te - shift) / 2) if ts is not None else ""
        kind = "image" if os.path.splitext(key)[1] in IMAGE_EXTS else "clip"
        rows.append((None, kind, os.path.basename(it.get("local_path") or it.get("name") or key),
                     "external", slot,
                     None if ts is None else ts - shift, None if te is None else te - shift,
                     it.get("in_seconds"), it.get("out_seconds")))

    for a in assets:
        if a["id"] not in best:
            rows.append((a["id"], a["kind"], a["filename"], "unused", "",
                         None, None, None, None))

    # Summary.
    counts = {k: Counter() for k in ("clip", "image")}
    for a in assets:
        counts[a["kind"]][best.get(a["id"], ("unused",))[0]] += 1
    ext_counts = Counter(r[1] for r in rows if r[3] == "external")
    covered = {r[4] for r in rows if r[3] != "unused" and r[0] is not None}
    uncovered = [s for s, _a, _b in ranges if s not in covered]
    moves = sorted(in_moves)
    summary = {
        "project_id": project_id,
        "title": proj["title"],
        "clips": dict(counts["clip"]),
        "images": dict(counts["image"]),
        "external_clips": ext_counts.get("clip", 0),
        "external_images": ext_counts.get("image", 0),
        "shots_without_our_footage": uncovered,
        "shift_sec": round(shift, 2),
        "median_in_move_sec": round(moves[len(moves) // 2], 2) if moves else None,
    }

    now = _now()
    with _conn() as c:
        c.execute("DELETE FROM asset_usage WHERE project_id=?", (project_id,))
        c.execute("""UPDATE project_assets SET verdict='unused', used_slot_id='', learned_at=?
                      WHERE project_id=?""", (now, project_id))
        cur = c.execute(
            """INSERT INTO xml_imports (project_id, imported_by, filename, shift_sec,
                                        summary, imported_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (project_id, imported_by, filename, shift, json.dumps(summary), now))
        import_id = cur.lastrowid
        c.executemany(
            """INSERT INTO asset_usage (import_id, project_id, asset_id, kind, name, verdict,
                                        used_slot_id, timeline_start_sec, timeline_end_sec,
                                        in_sec, out_sec)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(import_id, project_id, *r) for r in rows])
        c.executemany(
            "UPDATE project_assets SET verdict=?, used_slot_id=? WHERE id=?",
            [(v, s, aid) for aid, (v, s) in best.items()])
        c.execute("UPDATE projects SET status='learned', updated_at=? WHERE id=?",
                  (now, project_id))
    summary["import_id"] = import_id
    return summary
