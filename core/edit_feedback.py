"""
Edit feedback → picking.

Reads what editors did with past deliveries (labels written by
:func:`core.project_store.analyze_import`) and turns it into signals the picker
uses:

* **Similar-line history** — for each shot, the most similar narration lines
  from past edits and which clips the editor kept or dropped there. Rendered
  into the ranking prompt so the LLM judge learns the editor's taste.
* **Track records** — how often an exact clip, and a YouTube channel / Pexels
  author, survived the edit. Shown on each candidate line; a clip dropped
  repeatedly and never kept is pushed to the bottom after ranking.
* **In-point habit** — where into the source the editor starts a clip, per
  source, measured only on clips whose in-point we didn't derive from the
  footage itself. Used as the default in-point for clips nothing else
  positioned.
* **Image sites** — pages whose images get used float up; sites never used
  sink.

Human reviewers (:mod:`core.ratings`) feed the same signals: their yes/no per
clip joins the similar-line history (with their written reason) and gets its
own track record, the good segments they mark become that clip's in-point, and
clips they suggest are offered as candidates on similar lines.

Everything no-ops (returns empty/None) until at least one project has been
learned or rated. ``EDIT_LEARNING=0`` switches it all off.
"""

import os
import re
import threading
from collections import defaultdict
from urllib.parse import urlparse

import numpy as np

from core import project_store

KEPT = ("used_here", "used_elsewhere")

# How similar a past narration line must be to count (cosine on MiniLM
# embeddings; Jaccard on words when the embedder isn't available).
_MIN_COSINE = 0.55
_MIN_JACCARD = 0.3

_LOCK = threading.Lock()
_CACHE: dict = {"stamp": None, "history": None}


def enabled() -> bool:
    return os.getenv("EDIT_LEARNING", "1").strip().lower() not in ("0", "false", "no", "off")


# ── loading ─────────────────────────────────────────────────────────────────

class History:
    def __init__(self):
        self.lines: list = []            # {text, emb, kept, dropped, suggested: [asset]}
        self.clip_record = defaultdict(lambda: [0, 0])     # asset_key → [kept, delivered]
        self.creator_record = defaultdict(lambda: [0, 0])  # "source:creator" → [kept, delivered]
        self.site_record = defaultdict(lambda: [0, 0])     # image domain → [kept, delivered]
        self.in_starts = defaultdict(list)                  # source → [editor in-point]
        self.rater_record = defaultdict(lambda: [0, 0])     # asset_key → [usable, rated]
        self.rated_starts = defaultdict(list)               # asset_key → [good segment start]
        # Who to credit when a signal changes a pick (asset_key → rater ids).
        self.credit_yes = defaultdict(set)
        self.credit_no = defaultdict(set)
        self.credit_seg = defaultdict(set)
        self.suggester = defaultdict(set)
        self.n_projects = 0

    def empty(self) -> bool:
        return self.n_projects == 0


def _stamp():
    path = project_store._db_path()
    try:
        st = os.stat(path)
        return (path, st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def load_history() -> History | None:
    """The learned history, cached until the project DB changes. ``None`` when
    disabled or nothing has been learned yet."""
    if not enabled():
        return None
    stamp = _stamp()
    if stamp is None:
        return None
    with _LOCK:
        if _CACHE["stamp"] == stamp:
            return _CACHE["history"]
        try:
            h = _build_history()
        except Exception as e:
            print(f"[edit_feedback] couldn't load history: {e}")
            h = None
        h = None if (h is None or h.empty()) else h
        _CACHE.update(stamp=_stamp(), history=h)   # re-stat: embedding backfill writes
        return h


def _site(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _creator_key(source: str, creator: str) -> str:
    return f"{(source or '').lower()}:{creator.strip().lower()}" if creator else ""


def _build_history() -> History:
    from core import ratings
    ratings.init_db()
    h = History()
    labels = ratings.item_labels()
    with project_store._conn() as c:
        pids = [r[0] for r in c.execute("SELECT id FROM projects WHERE status='learned'")]
        all_pids = sorted(set(pids) | {l["project_id"] for l in labels})
        if not all_pids:
            return h
        h.n_projects = len(all_pids)
        qa = ",".join("?" * len(all_pids))
        shots = [dict(r) for r in c.execute(
            f"""SELECT project_id, slot_id, text, embedding FROM project_shots
                 WHERE project_id IN ({qa}) AND is_extra=0""", all_pids)]
        # Editor verdicts only exist for learned projects.
        qs = ",".join("?" * len(pids)) or "NULL"
        assets = [dict(r) for r in c.execute(
            f"SELECT * FROM project_assets WHERE project_id IN ({qs})", pids)]
        usages = [dict(r) for r in c.execute(
            f"""SELECT u.asset_id, u.in_sec, a.exported_in_sec, a.in_rule, a.source
                  FROM asset_usage u JOIN project_assets a ON a.id=u.asset_id
                 WHERE u.project_id IN ({qs}) AND a.kind='clip'
                   AND u.verdict IN ('used_here','used_elsewhere')""", pids)]

    by_line = {(s["project_id"], s["slot_id"]): {"text": s["text"] or "", "kept": [],
                                                 "dropped": [], "suggested": [],
                                                 "emb": s["embedding"]}
               for s in shots}
    # Reviewers may have confirmed or corrected the automatic labels.
    overrides = ratings.reviewed_labels()
    for a in assets:
        if a["id"] in overrides:
            verdict, slot = overrides[a["id"]]
            a = dict(a, verdict=verdict, used_slot_id=slot or a["used_slot_id"])
        if a["verdict"] == "neutral":
            continue          # fine clip, just not needed: no signal either way
        kept = a["verdict"] in KEPT
        if a["kind"] == "image":
            site = _site(a["page_url"] or a["url"])
            if site:
                rec = h.site_record[site]
                rec[0] += kept
                rec[1] += 1
            continue
        key = a["page_url"] or a["url"]
        if key:
            rec = h.clip_record[key]
            rec[0] += kept
            rec[1] += 1
        ck = _creator_key(a["source"], a["channel"] or "")
        if ck:
            rec = h.creator_record[ck]
            rec[0] += kept
            rec[1] += 1
        # Where it belongs: kept on its own line, or on the line it moved to.
        own = by_line.get((a["project_id"], a["slot_id"]))
        if a["verdict"] == "used_here" and own:
            own["kept"].append(a)
        elif a["verdict"] == "used_elsewhere":
            dest = by_line.get((a["project_id"], a["used_slot_id"]))
            if dest:
                dest["kept"].append(a)
            if own:
                own["dropped"].append(a)
        elif own:
            own["dropped"].append(a)

    # Only clips we started by default (or by this same habit) say something
    # about the habit; a trim or a verified in-point is about that footage.
    # Older deliveries have no rule recorded: count them if we started at 0.
    for u in usages:
        if u["in_sec"] is None:
            continue
        rule = u["in_rule"] or ""
        if rule in ("default", "habit") or (not rule and (u["exported_in_sec"] or 0) < 0.05):
            h.in_starts[(u["source"] or "").lower()].append(u["in_sec"])

    _add_ratings(h, labels, by_line)

    lines = [dict(v, key=k) for k, v in by_line.items()
             if v["text"].strip() and (v["kept"] or v["dropped"] or v["suggested"])]
    _attach_embeddings(lines)
    h.lines = lines
    return h


def _add_ratings(h: History, labels: list, by_line: dict) -> None:
    """Fold reviewer ratings (aggregated per clip) into the history. A clip the
    reviewers mostly marked usable counts as kept on its line, mostly not usable
    as dropped. 'Good clip, wrong line' is a verdict on the line, not the clip,
    so it doesn't count against the clip's own record."""
    for l in labels:
        line = by_line.get((l["project_id"], l["slot_id"]))
        entry = dict(l, note=(l["notes"] or [""])[0])
        if line is not None:
            if l["yes"] > l["no"]:
                (line["suggested"] if l["role"] == "suggested" else line["kept"]).append(entry)
            elif l["no"] > l["yes"]:
                line["dropped"].append(entry)
        key = l["page_url"]
        if key:
            rec = h.rater_record[key]
            rec[0] += l["yes"]
            rec[1] += l["yes"] + max(0, l["no"] - l["other_line"])
            h.rated_starts[key] += [segs[0][0] for segs in l["segments"] if segs]
            h.credit_yes[key].update(l.get("yes_raters") or [])
            h.credit_no[key].update(l.get("no_raters") or [])
            h.credit_seg[key].update(l.get("seg_raters") or [])
            if l["role"] == "suggested" and l.get("suggested_by"):
                h.suggester[key].add(l["suggested_by"])


# ── similarity ──────────────────────────────────────────────────────────────

def _embedder():
    try:
        from core.clip_library import _embed
        return _embed
    except Exception:
        return None


def _attach_embeddings(lines: list) -> None:
    """Give every history line a vector, computing and caching missing ones in
    the DB. Leaves ``emb`` None when no embedder is available."""
    embed = _embedder()
    missing = []
    for ln in lines:
        blob = ln.get("emb")
        ln["emb"] = np.frombuffer(blob, dtype=np.float32) if blob else None
        if ln["emb"] is None:
            missing.append(ln)
    if not missing or embed is None:
        return
    try:
        with project_store._conn() as c:
            for ln in missing:
                v = np.asarray(embed(ln["text"]), dtype=np.float32)
                ln["emb"] = v
                c.execute("UPDATE project_shots SET embedding=? WHERE project_id=? AND slot_id=?",
                          (v.tobytes(), ln["key"][0], ln["key"][1]))
    except Exception as e:
        print(f"[edit_feedback] embedding unavailable, using word overlap: {e}")
        for ln in missing:
            ln["emb"] = None


_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _words(text: str) -> set:
    return {w for w in _WORD.findall((text or "").lower()) if len(w) > 2}


def similar_lines(history: History, text: str, k: int = 2) -> list:
    """Up to ``k`` past lines most similar to ``text``, best first."""
    if not history or not history.lines or not (text or "").strip():
        return []
    embed = _embedder()
    q = None
    if embed is not None and any(ln["emb"] is not None for ln in history.lines):
        try:
            q = np.asarray(embed(text), dtype=np.float32)
        except Exception:
            q = None
    scored = []
    qw = _words(text)
    for ln in history.lines:
        if q is not None and ln["emb"] is not None and ln["emb"].shape == q.shape:
            score, floor = float(np.dot(q, ln["emb"])), _MIN_COSINE
        else:
            lw = _words(ln["text"])
            score = len(qw & lw) / len(qw | lw) if qw and lw else 0.0
            floor = _MIN_JACCARD
        if score >= floor:
            scored.append((score, ln))
    scored.sort(key=lambda t: -t[0])
    return [ln for _s, ln in scored[:k]]


# ── ranking hooks ───────────────────────────────────────────────────────────

def _asset_label(a: dict) -> str:
    title = (a.get("title") or a.get("filename") or "?")[:70]
    q = f' (query "{a["matched_query"]}")' if a.get("matched_query") else ""
    why = f' — reviewer: "{a["note"][:100]}"' if a.get("note") else ""
    return f"[{(a.get('source') or '?').upper()}] {title}{q}{why}"


def history_note(history: History, text: str) -> str:
    """The 'EDITOR HISTORY' block for one shot, or '' when nothing similar."""
    out = []
    for ln in similar_lines(history, text):
        kept = "; ".join(_asset_label(a) for a in ln["kept"][:3]) or "none"
        dropped = "; ".join(_asset_label(a) for a in ln["dropped"][:3]) or "none"
        line = f'- "{ln["text"][:140]}" → kept: {kept} | dropped: {dropped}'
        if ln["suggested"]:
            line += " | reviewer suggested: " + "; ".join(
                _asset_label(a) for a in ln["suggested"][:2])
        out.append(line)
    if not out:
        return ""
    return ("EDITOR HISTORY (what the editor and reviewers chose on similar lines "
            "in past videos):\n" + "\n".join(out))


def track_record(history: History, cand: dict) -> str:
    """'editor record: …' for a candidate with history, else ''."""
    parts = []
    rec = history.clip_record.get(project_store.asset_key(cand))
    if rec and rec[1]:
        parts.append(f"this clip kept {rec[0]}/{rec[1]}")
    ck = _creator_key(cand.get("source"), project_store.creator_of(cand))
    rec = history.creator_record.get(ck) if ck else None
    if rec and rec[1] >= 2:
        parts.append(f"{project_store.creator_of(cand)[:40]} kept {rec[0]}/{rec[1]}")
    rec = history.rater_record.get(project_store.asset_key(cand))
    if rec and rec[1]:
        parts.append(f"reviewers: usable {rec[0]}/{rec[1]}")
    return "editor record: " + ", ".join(parts) if parts else ""


def suggested_candidates(history: History, text: str) -> list:
    """YouTube clips reviewers suggested on lines similar to ``text``, as fresh
    candidate dicts (downloadable, in-point at the first good segment)."""
    out, seen = [], set()
    for ln in similar_lines(history, text, k=3):
        for a in ln["suggested"]:
            if a.get("source") != "youtube" or a["page_url"] in seen:
                continue
            seen.add(a["page_url"])
            first = next((segs[0][0] for segs in a.get("segments") or [] if segs), None)
            out.append({
                "url": a["page_url"], "page_url": a["page_url"], "source": "youtube",
                "title": a.get("title") or a["page_url"],
                "description": "suggested by a human reviewer for a similar line"
                               + (f': "{a["note"][:150]}"' if a.get("note") else ""),
                "matched_query": "reviewer suggestion",
                "duration": None, "is_short": False, "width": None, "height": None,
                "verified_in_sec": first, "reviewer_suggested": True,
            })
    return out


def annotate_for_ranking(shots: list, history: History = None) -> None:
    """Set ``edit_history`` on shots and ``edit_record`` on candidates for the
    ranking prompt. Candidate dicts can be shared across shots (the query cache),
    but the record depends only on the candidate, so writing it is safe."""
    history = history or load_history()
    if not history:
        return
    for s in shots:
        note = history_note(history, s.get("text") or "")
        if note:
            s["edit_history"] = note
        else:
            s.pop("edit_history", None)
        have = {project_store.asset_key(c) for c in s.get("video_results") or []}
        extra = [c for c in suggested_candidates(history, s.get("text") or "")
                 if c["page_url"] not in have]
        if extra:
            s["video_results"] = list(s.get("video_results") or []) + extra
        for c in s.get("video_results") or []:
            rec = track_record(history, c)
            if rec:
                c["edit_record"] = rec


def _rejected(history: History, cand: dict) -> bool:
    """Dropped in at least two edits and never kept, or judged not usable by
    at least two reviewers and usable by none."""
    key = project_store.asset_key(cand)
    for rec in (history.clip_record.get(key), history.rater_record.get(key)):
        if rec and rec[0] == 0 and rec[1] >= 2:
            return True
    return False


def demote_rejected(shots: list, history: History = None) -> int:
    """After ranking: move clips the editor keeps rejecting to the end of each
    shot's list (order otherwise kept). Returns how many were moved."""
    history = history or load_history()
    if not history:
        return 0
    moved = 0
    for s in shots:
        cands = s.get("video_results") or []
        bad = [c for c in cands if _rejected(history, c)]
        if bad and len(bad) < len(cands):
            s["video_results"] = [c for c in cands if not _rejected(history, c)] + bad
            moved += len(bad)
            for c in bad:
                key = project_store.asset_key(c)
                _credit(history.credit_no.get(key), "blocked_bad_clip", key, c.get("title"))
    return moved


def _credit(rater_ids, kind, key, title="", project_id=None, once=False) -> None:
    if not rater_ids:
        return
    try:
        from core import ratings
        ratings.log_impact(rater_ids, kind, key, title or "", project_id, once=once)
    except Exception as e:
        print(f"[edit_feedback] couldn't credit reviewers: {e}")


def credit_delivery(project_id: int, shots: list, history: History = None) -> None:
    """At delivery, credit the reviewers whose work shaped the final picks: a
    suggestion that got picked, a rated good part that set the in-point, a clip
    they rated usable that got picked."""
    history = history or load_history()
    if not history:
        return
    for s in shots or []:
        for c in s.get("selected_results") or []:
            if c.get("_dl_failed"):
                continue
            key = project_store.asset_key(c)
            if not key:
                continue
            ref = f"{project_id}:{key}"
            if c.get("reviewer_suggested"):
                _credit(history.suggester.get(key), "suggestion_used", ref, c.get("title"),
                        project_id, once=True)
            if c.get("in_rule") == "rated":
                _credit(history.credit_seg.get(key), "in_point_used", ref, c.get("title"),
                        project_id, once=True)
            _credit(history.credit_yes.get(key), "helped_pick", ref, c.get("title"),
                    project_id, once=True)


# ── in-points ───────────────────────────────────────────────────────────────

def learned_in_offset(source: str, history: History = None, min_samples: int = 5):
    """Median seconds into a ``source`` clip the editor starts it, when that's a
    consistent habit (≥ ``min_samples`` clips, > 0.5s). Else None."""
    history = history or load_history()
    if not history:
        return None
    starts = sorted(history.in_starts.get((source or "").lower(), []))
    if len(starts) < min_samples:
        return None
    med = starts[len(starts) // 2]
    return med if med > 0.5 else None


def rated_in_point(cand: dict, history: History = None):
    """Where reviewers said the good part of this exact clip starts (median of
    their first marked segment), or None."""
    history = history or load_history()
    if not history:
        return None
    starts = sorted(history.rated_starts.get(project_store.asset_key(cand), []))
    return starts[len(starts) // 2] if starts else None


# ── images ──────────────────────────────────────────────────────────────────

def order_image_candidates(cands: list, history: History = None) -> list:
    """Reorder Google image results: sites whose images the editor used (≥2
    used, ≥50%) first; sites delivered ≥3 times and never used last; the rest
    keep Google's order."""
    history = history or load_history()
    if not history or not history.site_record:
        return cands

    def _rank(c):
        rec = history.site_record.get(_site(c.get("context") or c.get("url")))
        if not rec:
            return 1
        kept, n = rec
        if kept >= 2 and kept / n >= 0.5:
            return 0
        if n >= 3 and kept == 0:
            return 2
        return 1
    return sorted(cands, key=_rank)
