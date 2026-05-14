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
        """)


# ── Embedding ────────────────────────────────────────────────────────────────

def _embed(text: str) -> np.ndarray:
    """
    Returns a normalised float32 embedding vector (384-dim) for text.
    Lazy-loads all-MiniLM-L6-v2 on first call (~80 MB download once).
    """
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

        emb_bytes = _embed(shot_description).tobytes()
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
