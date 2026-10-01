"""Footage confidence: how sure are we that the clips picked for a shot fit its
narration?

The pipeline already throws off a lot of evidence about every pick — where the
ranker put it, whether an editor vouched for it, what the QA reviewer said, how
many times the shot's queries had to be rewritten — but nothing adds it up. This
module does, so the motion-card stage (and later the review studio) can ask one
question: "should this moment get a designed graphic instead of footage we're not
sure about?"

The score is evidence, not a probability. Every weight below is a starting guess;
each score is stored with the reasons behind it (``shot["confidence"]``) so it can
be checked against what reviewers and editors actually decide and tuned.

    clip fit   source/trust of the clip, its rank position, storyboard verdict,
               the editor's record with it
    shot score 0.7 x mean(clip fits) + 0.3 x worst clip fit, minus shot penalties:
               QA flags, repair rounds, topic-fallback picks, a named subject no clip
               mentions, shots the director marked graphic/abstract

Tiers: ``high`` >= 0.65, ``low`` < ``CARD_CONF_THRESHOLD`` (0.45), else ``medium``;
``none`` (no clip at all) and ``skip`` (talking head / extra) are not scored.
"""

import os
import re

_HIGH = 0.65
_LOW = 0.45

# Starting point for a clip, by where it came from.
_BASE = {
    "library_verified": 0.92, "library_used": 0.85, "library_suggested": 0.75,
    "library_other": 0.70,
    "youtube": 0.60, "pexels": 0.55, "pixabay": 0.55, "library": 0.55, "other": 0.50,
}
# Adjustment by the clip's position among the ranker's non-irrelevant candidates
# (the ranker returns an order, not a number; the per-shot quota often reaches deep).
_RANK_ADJ = (0.15, 0.08, 0.02, -0.04, -0.04)
_RANK_DEEP = -0.10
_STORYBOARD = {"good": 0.08, "weak": -0.15, "bad": -0.30}
_QA = {"high": 0.35, "medium": 0.20, "low": 0.08}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def low_threshold() -> float:
    """Below this a shot counts as low confidence (``CARD_CONF_THRESHOLD``)."""
    return _env_float("CARD_CONF_THRESHOLD", _LOW)


def high_threshold() -> float:
    return _env_float("CARD_CONF_HIGH", _HIGH)


def tier_of(score) -> str:
    if score is None:
        return "none"
    if score < low_threshold():
        return "low"
    return "high" if score >= high_threshold() else "medium"


# ── one clip ─────────────────────────────────────────────────────────────────

def _source_key(clip: dict) -> str:
    if clip.get("library_segment_id"):
        trust = (clip.get("segment_trust") or "").lower()
        return f"library_{trust}" if f"library_{trust}" in _BASE else "library_other"
    src = (clip.get("source") or clip.get("original_source") or "").lower()
    return src if src in _BASE else "other"


def _rank_position(shot: dict, clip: dict):
    """(index among the shot's non-irrelevant ranked candidates, how many there
    are). The index is None when the clip isn't one of them (a manual pick)."""
    pool = [c for c in shot.get("video_results") or [] if not c.get("irrelevant")]
    key = clip.get("url") or clip.get("page_url")
    for i, c in enumerate(pool):
        if c is clip or (key and key in (c.get("url"), c.get("page_url"))):
            return i, len(pool)
    return None, len(pool)


def clip_fit(shot: dict, clip: dict) -> tuple:
    """``(fit 0..1, [reasons])`` for one selected clip."""
    key = _source_key(clip)
    fit = _BASE[key]
    reasons = []
    if key.startswith("library_"):
        reasons.append(f"{key.replace('_', ' ')} clip")

    pos, n = _rank_position(shot, clip)
    if pos is not None:
        adj = _RANK_ADJ[pos] if pos < len(_RANK_ADJ) else _RANK_DEEP
        fit += adj
        if adj <= -0.04:
            reasons.append(f"ranked #{pos + 1} of {n}")

    verdict = (clip.get("sb_verdict") or "").lower() if clip.get("sb_checked") else ""
    if verdict in _STORYBOARD:
        fit += _STORYBOARD[verdict]
        if _STORYBOARD[verdict] < 0:
            reasons.append(f"footage check: {verdict}")

    m = re.search(r"this clip kept (\d+)/(\d+)", clip.get("edit_record") or "")
    if m and int(m.group(2)) >= 2:
        kept = int(m.group(1)) / int(m.group(2))
        fit += 0.10 if kept >= 0.75 else (-0.10 if kept <= 0.25 else 0.0)
        if kept <= 0.25:
            reasons.append(f"editors dropped it {m.group(2)} times")
    return max(0.0, min(1.0, fit)), reasons


# ── the named-subject check ──────────────────────────────────────────────────

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = {"the", "a", "an", "of", "and", "for", "with", "car", "cars", "new", "old",
         "engine", "bay", "method", "step"}


def _tokens(text: str) -> set:
    return {w for w in _WORD.findall((text or "").lower()) if len(w) > 2 and w not in _STOP}


def named_things(shot: dict) -> list:
    """Specific things the narration names: the segment library's tagged
    ``line_subjects`` when present, else capitalised words that don't start a
    sentence and number-bearing tokens (``2012``, ``V6``, ``$300``)."""
    tagged = [str(s).strip() for s in shot.get("line_subjects") or [] if str(s).strip()]
    if tagged:
        return tagged
    text = (shot.get("text") or "").strip()
    out = []
    for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][\w'-]{2,})", text):
        w = m.group(1)
        if w.lower() not in _STOP:
            out.append(w)
    out += re.findall(r"\b\d[\w.,%$]*|\b[A-Z]\d+\b", text)
    return list(dict.fromkeys(out))[:6]


def _subject_gap(shot: dict, clips: list) -> bool:
    """The narration names something specific and no picked clip mentions any of
    it (a Pexels 'foam on car' clip under 'such as Simple Green or Gunk')."""
    things = named_things(shot)
    if not things:
        return False
    need = set()
    for t in things:
        need |= _tokens(t)
    if not need:
        return False
    for c in clips:
        have = _tokens(" ".join(str(c.get(k) or "") for k in
                                ("title", "clip_title", "description", "matched_query")))
        if need & have:
            return False
    return True


# ── one shot ─────────────────────────────────────────────────────────────────

def _qa_penalty(slot_id, qa_issues) -> tuple:
    sevs = [str(i.get("severity", "medium")).lower() for i in qa_issues or []
            if i.get("slot_id") == slot_id]
    if not sevs:
        return 0.0, ""
    worst = max(sevs, key=lambda s: _QA.get(s, 0.2))
    pen = min(0.40, _QA.get(worst, 0.2) + 0.05 * (len(sevs) - 1))
    return pen, f"QA flagged ({worst}{'' if len(sevs) == 1 else f', {len(sevs)} notes'})"


def shot_confidence(shot: dict, qa_issues: list = None) -> dict:
    """``{"score", "tier", "reasons"}`` for one shot. ``score`` is None for shots
    that aren't scored (talking head, extras) and 0.0 when no clip is placed."""
    if shot.get("is_extra") or shot.get("priority") == "none" or shot.get("skipped"):
        return {"score": None, "tier": "skip", "reasons": []}
    clips = [c for c in shot.get("selected_results") or [] if c.get("url") and not c.get("_dl_failed")]
    if not clips:
        return {"score": 0.0, "tier": "none", "reasons": ["no footage placed"]}

    fits, reasons = [], []
    for c in clips:
        f, why = clip_fit(shot, c)
        fits.append(f)
        reasons += why
    score = 0.7 * (sum(fits) / len(fits)) + 0.3 * min(fits)

    pen, why = _qa_penalty(shot.get("slot_id"), qa_issues)
    if pen:
        score -= pen
        reasons.append(f"{why} -{pen:.2f}")
    tried = shot.get("tried_queries") or []
    if tried:
        rounds = max(1, len(tried) // 4)
        p = min(0.18, 0.06 * rounds)
        score -= p
        reasons.append(f"queries rewritten ~{rounds}x -{p:.2f}")
    if _subject_gap(shot, clips):
        score -= 0.15
        reasons.append("narration names something no clip mentions -0.15")
    stype = (shot.get("shot_type") or "").lower()
    if stype == "graphic":
        score -= 0.12
        reasons.append("director asked for a graphic -0.12")
    elif stype == "abstract":
        score -= 0.04
        reasons.append("abstract line -0.04")
    if shot.get("_topic_fallback"):
        score = min(score, 0.30)
        reasons.append("generic topic fallback clip (capped at 0.30)")

    score = round(max(0.0, min(1.0, score)), 2)
    return {"score": score, "tier": tier_of(score), "reasons": list(dict.fromkeys(reasons))}


def annotate_confidence(shots: list, qa: dict = None) -> dict:
    """Score every shot and store it on ``shot["confidence"]``. Returns the tier
    tally ``{"high": n, "medium": n, "low": n, "none": n, "skip": n}``."""
    issues = (qa or {}).get("issues") or []
    tally = {"high": 0, "medium": 0, "low": 0, "none": 0, "skip": 0}
    for s in shots:
        c = shot_confidence(s, issues)
        s["confidence"] = c
        tally[c["tier"]] += 1
    return tally
