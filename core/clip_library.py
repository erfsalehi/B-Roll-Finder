"""
Clip Library — SQLite + sentence-transformers RAG store.

Stores every confirmed-downloaded clip with a semantic embedding of the
shot description.  At query time, cosine similarity search retrieves the
most relevant past clips so editors can reuse footage without re-fetching.
"""

import os
import json
import sqlite3
import numpy as np
from datetime import datetime

_DB_PATH = os.path.join(os.path.dirname(__file__), '..', '.cache', 'clip_library.db')


# ── DB bootstrap ────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS clips (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                project         TEXT    DEFAULT '',
                shot_description TEXT   DEFAULT '',
                slot_index      INTEGER DEFAULT 0,
                keywords        TEXT    DEFAULT '[]',
                search_query    TEXT    DEFAULT '',
                source          TEXT    DEFAULT '',
                clip_url        TEXT    NOT NULL,
                clip_title      TEXT    DEFAULT '',
                duration        REAL    DEFAULT 0,
                thumbnail_url   TEXT    DEFAULT '',
                local_path      TEXT    DEFAULT '',
                embedding       BLOB,
                usage_count     INTEGER DEFAULT 1,
                last_used_at    TEXT    DEFAULT '',
                created_at      TEXT    DEFAULT ''
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_clips_url ON clips(clip_url);
            CREATE INDEX IF NOT EXISTS idx_clips_source  ON clips(source);
            CREATE INDEX IF NOT EXISTS idx_clips_project ON clips(project);

            CREATE TABLE IF NOT EXISTS clip_preferred_trims (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_id          INTEGER NOT NULL,
                shot_description TEXT    NOT NULL DEFAULT '',
                in_seconds       REAL    NOT NULL,
                out_seconds      REAL    NOT NULL,
                source_xml_path  TEXT    DEFAULT '',
                confirmed_at     TEXT    DEFAULT '',
                FOREIGN KEY(clip_id) REFERENCES clips(id) ON DELETE CASCADE
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_trims_clip_shot
                ON clip_preferred_trims(clip_id, shot_description);
        """)


# ── Embedding ────────────────────────────────────────────────────────────────

def _embed(text: str) -> np.ndarray:
    """
    Returns a normalised float32 embedding vector (384-dim) for text.
    Lazy-loads all-MiniLM-L6-v2 on first call (~80 MB download once).
    """
    # Force the PyTorch backend. With Keras 3 installed, transformers tries
    # the TensorFlow backend and dies with "Keras 3 ... not yet supported;
    # install tf-keras" — which previously bubbled up through store_clip and
    # left the whole Clip Library silently empty. sentence-transformers only
    # needs torch, so opt out of TF/Flax before the import resolves a backend.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for the Clip Library.\n"
            "Run:  pip install sentence-transformers"
        )
    if not hasattr(_embed, '_model'):
        _embed._model = SentenceTransformer('all-MiniLM-L6-v2')
    vec = _embed._model.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)


# ── Write ────────────────────────────────────────────────────────────────────

def store_clip(
    shot_description: str,
    clip_data: dict,
    project: str = "default",
    slot_index: int = 0,
    keywords: list = None,
    search_query: str = "",
) -> bool:
    """
    Store or increment usage_count for a confirmed-downloaded clip.
    Returns True on success.
    """
    try:
        init_db()
        clip_url = clip_data.get("url") or clip_data.get("webpage_url", "")
        if not clip_url:
            return False

        # Embed for semantic search, but never let an embedding failure lose
        # the clip: a null-embedding row is still recorded (and still usable
        # for preferred-trim matching by path/url) — it just won't surface in
        # semantic search until re-embedded.
        try:
            emb_bytes = _embed(shot_description).tobytes()
        except Exception as e:
            print(f"[ClipLibrary] embedding unavailable, storing without it: {e}")
            emb_bytes = None
        now = datetime.utcnow().isoformat()

        with _conn() as c:
            existing = c.execute(
                "SELECT id, usage_count FROM clips WHERE clip_url = ?", (clip_url,)
            ).fetchone()
            if existing:
                c.execute(
                    "UPDATE clips SET usage_count=?, last_used_at=?, project=? WHERE id=?",
                    (existing["usage_count"] + 1, now, project, existing["id"]),
                )
            else:
                c.execute(
                    """INSERT INTO clips
                         (project, shot_description, slot_index, keywords, search_query,
                          source, clip_url, clip_title, duration, thumbnail_url,
                          local_path, embedding, usage_count, last_used_at, created_at)
                       VALUES (?,?,?,?,?, ?,?,?,?,?, ?,?,1,?,?)""",
                    (
                        project,
                        shot_description,
                        slot_index,
                        json.dumps(keywords or []),
                        search_query,
                        clip_data.get("source", "unknown"),
                        clip_url,
                        clip_data.get("title", ""),
                        float(clip_data.get("duration") or 0),
                        clip_data.get("thumbnail", ""),
                        clip_data.get("local_path", ""),
                        emb_bytes,
                        now,
                        now,
                    ),
                )
        return True
    except Exception as e:
        print(f"[ClipLibrary] store_clip error: {e}")
        return False


# ── Read / Search ────────────────────────────────────────────────────────────

def search_library(
    query: str,
    top_k: int = 10,
    min_score: float = 0.55,
) -> list:
    """
    Semantic search over stored clips.
    Returns a list of clip dicts sorted by a combined score of
    cosine similarity + a small usage-count boost.
    """
    try:
        init_db()
        with _conn() as c:
            rows = c.execute(
                """SELECT id, project, shot_description, source, clip_url, clip_title,
                          duration, thumbnail_url, local_path, usage_count, embedding
                   FROM clips"""
            ).fetchall()

        if not rows:
            return []

        q_emb = _embed(query)

        results = []
        for row in rows:
            raw = row["embedding"]
            if not raw:
                continue
            stored = np.frombuffer(raw, dtype=np.float32)
            if stored.shape != q_emb.shape:
                continue
            sim = float(np.dot(q_emb, stored))
            if sim < min_score:
                continue
            usage = row["usage_count"] or 1
            # Small logarithmic boost for frequently-used clips (max +0.10)
            score = sim + 0.10 * min(np.log1p(usage) / np.log1p(20), 1.0)
            results.append({
                "id":              row["id"],
                "title":           row["clip_title"],
                "url":             row["clip_url"],
                "source":          "library",
                "original_source": row["source"],
                "duration":        row["duration"],
                "thumbnail":       row["thumbnail_url"],
                "local_path":      row["local_path"],
                "usage_count":     usage,
                "shot_description": row["shot_description"],
                "similarity":      round(sim, 3),
                "_score":          round(score, 3),
                "from_library":    True,
            })

        results.sort(key=lambda x: x["_score"], reverse=True)
        return results[:top_k]
    except Exception as e:
        print(f"[ClipLibrary] search_library error: {e}")
        return []


# ── Preferred trims (learned from re-imported Premiere edits) ───────────────

def record_trim(
    clip_id: int,
    shot_description: str,
    in_seconds: float,
    out_seconds: float,
    source_xml_path: str = "",
) -> bool:
    """
    Upsert the preferred in/out trim for a (clip, shot_description) pair.
    Most-recent edit wins — the table is keyed on (clip_id, shot_description).
    """
    if out_seconds <= in_seconds:
        return False
    try:
        init_db()
        now = datetime.utcnow().isoformat()
        with _conn() as c:
            c.execute(
                """INSERT INTO clip_preferred_trims
                     (clip_id, shot_description, in_seconds, out_seconds,
                      source_xml_path, confirmed_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(clip_id, shot_description) DO UPDATE SET
                     in_seconds      = excluded.in_seconds,
                     out_seconds     = excluded.out_seconds,
                     source_xml_path = excluded.source_xml_path,
                     confirmed_at    = excluded.confirmed_at""",
                (clip_id, shot_description, float(in_seconds), float(out_seconds),
                 source_xml_path, now),
            )
        return True
    except Exception as e:
        print(f"[ClipLibrary] record_trim error: {e}")
        return False


def get_preferred_trim(clip_id: int, shot_description: str) -> dict | None:
    """
    Returns {'in_seconds', 'out_seconds', 'confirmed_at'} for an exact
    (clip_id, shot_description) match, or None if no learned trim exists.
    """
    try:
        init_db()
        with _conn() as c:
            row = c.execute(
                """SELECT in_seconds, out_seconds, confirmed_at
                   FROM clip_preferred_trims
                   WHERE clip_id = ? AND shot_description = ?""",
                (clip_id, shot_description),
            ).fetchone()
        if not row:
            return None
        return {
            "in_seconds":   row["in_seconds"],
            "out_seconds":  row["out_seconds"],
            "confirmed_at": row["confirmed_at"],
        }
    except Exception as e:
        print(f"[ClipLibrary] get_preferred_trim error: {e}")
        return None


def find_clip_by_path_or_url(local_path: str = "", clip_url: str = "",
                             filename: str = "") -> dict | None:
    """
    Resolves a re-imported clipitem back to a library row.

    Resolution order, most-to-least robust:
      1. exact ``local_path`` match,
      2. ``clip_url`` match,
      3. basename match — the export writes absolute paths while the library
         stores whatever ``output_path`` the downloader used (often relative),
         so an exact path compare misses; the filename is unique per project,
         so a trailing-component ``LIKE`` recovers it.

    Returns ``{'id', 'clip_url', 'local_path', 'clip_title',
    'shot_description'}`` or ``None``.
    """
    _cols = "id, clip_url, local_path, clip_title, shot_description"
    try:
        init_db()
        with _conn() as c:
            row = None
            if local_path:
                row = c.execute(
                    f"SELECT {_cols} FROM clips WHERE local_path = ?",
                    (local_path,),
                ).fetchone()
            if not row and clip_url:
                row = c.execute(
                    f"SELECT {_cols} FROM clips WHERE clip_url = ?",
                    (clip_url,),
                ).fetchone()
            if not row:
                base = filename or os.path.basename(local_path or "")
                if base:
                    # Escape LIKE wildcards in the filename, then match any
                    # stored path ending in that basename (after a separator).
                    safe = base.replace("\\", "/").split("/")[-1]
                    safe = safe.replace("%", r"\%").replace("_", r"\_")
                    row = c.execute(
                        f"SELECT {_cols} FROM clips WHERE local_path LIKE ? ESCAPE '\\'",
                        (f"%{safe}",),
                    ).fetchone()
        return dict(row) if row else None
    except Exception as e:
        print(f"[ClipLibrary] find_clip_by_path_or_url error: {e}")
        return None


def ensure_clip(
    local_path: str,
    shot_description: str = "",
    clip_url: str = "",
    source: str = "reimport",
    clip_title: str = "",
) -> int | None:
    """
    Find an existing library clip by path / url / basename, or create a
    minimal row for it. Returns the clip id, or None on error.

    Used by the XML re-import path so trims can be learned for clips that
    were never saved through the normal download flow (e.g. footage
    downloaded before the library existed, or when the embedding model was
    unavailable). The created row carries no embedding — it won't appear in
    semantic search, but it is fully usable for preferred-trim matching by
    path/basename, which is how the exporter looks it up.
    """
    try:
        existing = find_clip_by_path_or_url(
            local_path=local_path, clip_url=clip_url,
            filename=os.path.basename(local_path or ""),
        )
        if existing:
            return existing["id"]

        # Synthesize a stable, unique key. clip_url is UNIQUE NOT NULL, so fall
        # back to the local path when no real source URL is known.
        url_key = clip_url or local_path
        if not url_key:
            return None
        now = datetime.utcnow().isoformat()
        with _conn() as c:
            cur = c.execute(
                """INSERT OR IGNORE INTO clips
                     (project, shot_description, source, clip_url, clip_title,
                      local_path, embedding, usage_count, last_used_at, created_at)
                   VALUES (?,?,?,?,?,?,?,1,?,?)""",
                ("", shot_description, source, url_key, clip_title,
                 local_path, None, now, now),
            )
            if cur.lastrowid:
                return cur.lastrowid
            row = c.execute(
                "SELECT id FROM clips WHERE clip_url = ?", (url_key,)
            ).fetchone()
            return row["id"] if row else None
    except Exception as e:
        print(f"[ClipLibrary] ensure_clip error: {e}")
        return None


# ── Stats ────────────────────────────────────────────────────────────────────

def get_library_stats() -> dict:
    """Summary stats for the sidebar panel."""
    try:
        init_db()
        with _conn() as c:
            total = c.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
            by_src = {
                r["source"]: r["cnt"]
                for r in c.execute(
                    "SELECT source, COUNT(*) AS cnt FROM clips GROUP BY source"
                ).fetchall()
            }
            top = [
                dict(r)
                for r in c.execute(
                    """SELECT clip_title, clip_url, source, usage_count
                       FROM clips ORDER BY usage_count DESC LIMIT 5"""
                ).fetchall()
            ]
        return {"total": total, "by_source": by_src, "top_clips": top}
    except Exception as e:
        print(f"[ClipLibrary] get_library_stats error: {e}")
        return {"total": 0, "by_source": {}, "top_clips": []}
