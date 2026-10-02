"""The scene plan: which moments of the video get a designed graphic, and what it says.

The plan is a list of *beats*, plain JSON so it survives the review gate's pending
store, rides in the project snapshot, and can be edited later (the review studio):

    {"id": "b005", "template": "split_card", "trigger": "step" | "low_confidence" | "empty",
     "slot_ids": [5], "start": 12.4, "end": 16.4, "enabled": true,
     "label": "CLEAN EXTERIOR", "step": {"number": 1, "kind": "STEP"},
     "right": {"kind": "number"|"stat"|"image", "text": "1", "sub": "STEP"},
     "image_query": "...", "image_prompt": "...", "image_style": "photo",
     "subject_kind": "generic" | "identifiable",
     "confidence": {"footage": 0.31, "tier": "low"}, "reasons": [...],
     "assets": {"left": {...}, "right": {...}}}        # filled by core.motion_cards

Two things put a card on a moment (the user's rule, 2026-10-01):

  * it is a STEP of a list ("Method 2…", "Safety step one…") — a structural graphic
    that sits on the first seconds of the step, over whatever footage is there;
  * the footage there is LOW CONFIDENCE (core.confidence) — the card covers the
    shot, and the footage stays underneath on the layer below.

This module only decides and words things (one LLM call to find steps, one batched
call to design the cards); finding pictures and rendering live in core.motion_cards.
"""

import concurrent.futures
import hashlib
import math
import os
import re

from core import confidence

try:
    from groq import Groq
except Exception:  # pragma: no cover - import guard
    Groq = None

TEMPLATES = {
    # Remotion composition, and the shortest/longest it can be on screen. The reveal
    # needs ~1.1 s and the caption another beat, so anything under ~2.4 s is skipped.
    # A card over weak footage (not a step) is held at least ``CARD_MIN_SEC`` (3 s): a 2.4 s
    # card is all reveal and no read time.
    "split_card": {"composition": "ImageCard", "min_sec": 2.4, "max_sec": 12.0},
}

_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
             "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
_KINDS = ("method", "step", "tip", "rule", "reason", "mistake", "way", "hack", "secret",
          "lesson", "trick", "option", "habit", "tactic")


# ── settings ─────────────────────────────────────────────────────────────────

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def mode() -> str:
    """``off`` | ``steps`` (list steps only) | ``auto`` (steps + low confidence)."""
    legacy = os.getenv("ENABLE_MOTION_CARDS", "").strip().lower()
    if legacy in ("0", "false", "no", "off"):
        return "off"
    m = os.getenv("MOTION_CARDS_MODE", "").strip().lower()
    return m if m in ("off", "steps", "auto") else "auto"


def enabled() -> bool:
    return mode() != "off"


# ── step detection ───────────────────────────────────────────────────────────

def _num(tok: str):
    tok = (tok or "").strip().lower().rstrip(".:,")
    if tok.isdigit():
        n = int(tok)
        return n if 1 <= n <= 30 else None
    return _NUM_WORDS.get(tok) or _ORDINALS.get(tok)


_STEP_RE = re.compile(
    r"\b(" + "|".join(_KINDS) + r")\s*(?:number\s*|no\.?\s*|#\s*)?(\d{1,2}|"
    + "|".join(_NUM_WORDS) + r")\b", re.I)
_NUMBER_RE = re.compile(r"\bnumber\s+(\d{1,2}|" + "|".join(_NUM_WORDS) + r")\b", re.I)
_ORDINAL_RE = re.compile(r"^\s*(" + "|".join(_ORDINALS) + r")\b\s*[,:.\-—]", re.I)


def _timeline_shots(shots: list) -> list:
    return [s for s in shots if not s.get("is_extra") and s.get("priority") != "none"
            and not s.get("skipped") and (s.get("text") or "").strip()]


def regex_steps(shots: list) -> list:
    """Steps found by wording alone — the fallback when the LLM is unavailable.
    ``[{slot_id, number, kind}]`` in timeline order. Each kind of list (methods,
    safety steps…) is numbered on its own and its numbers must rise, so a stray
    "step two" later in a sentence is ignored."""
    out, last = [], {}
    for s in _timeline_shots(shots):
        head = (s.get("text") or "").strip()[:160]
        m = _STEP_RE.search(head)
        n, kind = (_num(m.group(2)), m.group(1).upper()) if m else (None, "STEP")
        if n is None:
            m = _NUMBER_RE.search(head)
            n = _num(m.group(1)) if m else None
        if n is None:
            m = _ORDINAL_RE.match(head)
            n = _num(m.group(1)) if m else None
        if n is not None and n > last.get(kind, 0):
            out.append({"slot_id": s.get("slot_id"), "number": n, "kind": kind})
            last[kind] = n
    return out


_STEPS_PROMPT = """You find the LIST STRUCTURE in a narrated explainer video. You get the narration, one numbered line per shot as "slot_id: text".

Find the shots where a NEW item of an ordered list is first announced — "Method 2 — Waterless Spray and Wipe", "Safety step one is verifying the engine is cool", "Number three: …", "Tip 5". Rules:
- Numbers may be implicit: if the video announces "four methods" and the third is introduced without saying "three", it is still item 3. Count the items in order.
- When the video announces a count ("five safety steps", "four methods"), expect that many items. If your first pass found fewer, read the lines again for the missing ones: an item can start in the middle of a line, or be introduced as "the physical workspace demands its own precautions" with no number or the word "step".
- Only the shot where the item STARTS. Not later shots about the same item, not passing mentions ("looking back at the four options", "stick around for five safety steps").
- Two different lists in one video (methods, then safety steps) are numbered separately; give each its "kind" (METHOD, STEP, TIP, RULE, REASON, MISTAKE …).
- "label": the item's name in 2-4 words, as it would appear on screen ("Waterless spray and wipe", "Check the engine is cool"). No numbers.
- If the video has no list, return an empty items array.

Output ONLY valid JSON: {"items": [{"slot_id": <id>, "number": <int>, "kind": "METHOD", "label": "..."}]}"""


# What the LLM said about a given narration, so every /refine and /download re-plan of the
# same script reuses the same steps and labels (they would otherwise drift) without
# another call. Only successful answers are kept; the narration text is the key.
_STEP_CACHE: dict = {}
_STEP_CACHE_MAX = 16


def detect_steps(shots: list, api_key: str = None, errors: list = None) -> list:
    """List steps in timeline order: ``[{slot_id, number, kind, label}]``. One fast-
    tier LLM call over the whole narration (remembered per narration); the regex pass
    is the fallback when the call fails and the fallback for any step the LLM names
    without a number."""
    if errors is None:
        errors = []
    tl = _timeline_shots(shots)
    found = []
    if tl:
        from core.keywords import _call_llm_json
        client = Groq(api_key=api_key) if (api_key and Groq) else None
        lines = "\n".join(f'{s.get("slot_id")}: {(s.get("text") or "").strip()[:240]}'
                          for s in tl[:300])
        ckey = hashlib.sha1(lines.encode("utf-8")).hexdigest()
        if ckey in _STEP_CACHE:
            found = [dict(x) for x in _STEP_CACHE[ckey]]
        else:
            try:
                data = _call_llm_json(client, _STEPS_PROMPT, lines, temperature=0.1,
                                      max_tokens=2500, tier="fast", allow_fallback=True)
                valid = {s.get("slot_id") for s in tl}
                for it in (data.get("items") or []) if isinstance(data, dict) else []:
                    n = _num(str(it.get("number")))
                    sid = it.get("slot_id")
                    if n is None or sid not in valid:
                        continue
                    found.append({"slot_id": sid, "number": n,
                                  "kind": str(it.get("kind") or "STEP").strip().upper()[:12] or "STEP",
                                  "label": str(it.get("label") or "").strip()})
                if isinstance(data, dict):
                    if len(_STEP_CACHE) >= _STEP_CACHE_MAX:
                        _STEP_CACHE.pop(next(iter(_STEP_CACHE)))
                    _STEP_CACHE[ckey] = [dict(x) for x in found]
            except Exception as e:
                errors.append(f"cards: step detection failed ({type(e).__name__}); using wording rules")
    if not found:
        found = regex_steps(shots)
    order = {s.get("slot_id"): i for i, s in enumerate(tl)}
    found.sort(key=lambda x: order.get(x["slot_id"], 1 << 30))
    seen, out = set(), []
    for it in found:                                # one step per shot
        if it["slot_id"] not in seen:
            seen.add(it["slot_id"])
            out.append(it)
    return out


# ── designing the cards ──────────────────────────────────────────────────────

_DESIGN_PROMPT = """You design split-screen "image cards" for a narrated explainer video. A card shows ONE photo with a short caption on the left half of the screen and, on the right half, a big number or a second picture. You get several moments; design a card for each.

For every moment return:
- "label": the idea in 2-3 words, at most 22 characters, a COMPLETE phrase that stands alone on screen — never cut off mid-thought and never ending in "and", "of", "the" or "to": "Clean exterior", "Dry-ice blasting", "Hot engine risk". For a list step it is the step's NAME, not "step 1". No trailing punctuation.
- "right": what fills the right half. It is NEVER empty. Choose one:
    {"kind": "number"}                                       only for a numbered list step (we fill in the number)
    {"kind": "stat", "text": "90°C", "sub": "UNDER HOOD"}    only when the narration itself states that figure — copy it exactly; "sub" is 1-2 words
    {"kind": "image"}                                        a second picture that adds something: the contrasting or related view
  For a list step prefer "number" unless a second picture clearly adds more. Anything else: "stat" when the line states a figure, otherwise "image".
- "image_query": a 3-7 word Google Images query for the main picture — the concrete thing (brand/model/part if the line names one).
- "image_prompt": one photographic sentence for an image generator describing a GENERIC scene of the same thing: no brand names, no real people, no text or logos.
- "subject_kind": "identifiable" when the moment is about a specific named product, brand, model, person or place whose real picture matters (a generated picture would be wrong); otherwise "generic".
- "image_style": "photo" normally; "illustration" for abstract ideas no photo can show (costs, risks, systems, trade-offs).

Output ONLY valid JSON: {"cards": [{"slot_id": <id>, "label": "...", "right": {...}, "image_query": "...", "image_prompt": "...", "subject_kind": "generic", "image_style": "photo"}]}
Include every slot_id you were given."""

_BATCH = 24


# A caption must not end on one of these ("PROTECT ELECTRICAL AND").
_DANGLING = {"AND", "OR", "OF", "THE", "A", "AN", "TO", "IN", "ON", "FOR", "WITH", "YOUR",
             "FROM", "BY", "AT", "AS", "IS", "ARE", "VS", "THAN", "THEN", "BUT", "IF", "WHEN",
             "WHILE", "SO", "INTO", "THAT", "WHICH", "BECAUSE"}


def _clean_label(text: str, fallback: str = "") -> str:
    t = re.sub(r"[^\w&%$°'+/\-,. ]+", " ", str(text or ""), flags=re.UNICODE)
    t = re.sub(r"(?<!\d)[,.]|[,.](?!\d)", " ", t)          # keep only the ones inside a number ($2,000, 4.5)
    t = " ".join(t.split()).upper().strip(" -")
    if len(t) > 24:
        cut = t[:24].rsplit(" ", 1)[0]
        t = cut if len(cut) >= 6 else t[:24].strip()
    words = t.split()
    while len(words) > 1 and words[-1] in _DANGLING:
        words.pop()
    return " ".join(words) or fallback


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def names_in_picture(shot: dict, beat: dict) -> list:
    """Words the narration names (a brand, model, place — never a bare figure) that the
    card's image query or caption also uses. A hit means the picture is about that
    specific thing, whatever the designer called it."""
    wanted = confidence._tokens(f'{beat.get("image_query", "")} {beat.get("label", "")}')
    hits = []
    for thing in confidence.named_things(shot):
        for tok in confidence._tokens(thing):
            if tok in wanted and re.search(r"[^\W\d_]", tok) and tok not in hits:
                hits.append(tok)
    return hits


def _clean_right(raw, beat: dict, narration: str) -> dict:
    """The right panel after the rules: a number only for a step, a figure only if the
    narration states it, otherwise a second image. Never empty."""
    raw = raw if isinstance(raw, dict) else {}
    kind = str(raw.get("kind") or "").lower()
    step = beat.get("step")
    if kind == "number" and step:
        return {"kind": "number", "text": str(step["number"]), "sub": step.get("kind", "STEP")}
    if kind == "stat":
        text = " ".join(str(raw.get("text") or "").split())[:14]
        if text and _digits(text) and _digits(text) in _digits(narration):
            return {"kind": "stat", "text": text, "sub": _clean_label(raw.get("sub"), "")[:18]}
    if step and kind != "image":
        return {"kind": "number", "text": str(step["number"]), "sub": step.get("kind", "STEP")}
    return {"kind": "image"}


def _design_batch(beats: list, shots_by_id: dict, topic: str, api_key: str, errors: list) -> None:
    from core.keywords import _call_llm_json
    client = Groq(api_key=api_key) if (api_key and Groq) else None
    lines = [f"VIDEO TOPIC: {topic or 'unknown'}", ""]
    for b in beats:
        s = shots_by_id.get(b["slot_ids"][0], {})
        why = (f'LIST STEP {b["step"]["number"]} ({b["step"]["kind"]})'
               + (f' — working name "{b["step"]["name"]}"' if b["step"].get("name") else "")
               if b.get("step") else "footage here is weak")
        picked = "; ".join((c.get("title") or c.get("clip_title") or "")[:60]
                           for c in (s.get("selected_results") or [])[:2]) or "—"
        lines.append(f'slot_id {b["slot_ids"][0]} [{why}] "{(s.get("text") or "").strip()[:300]}"'
                     f' | intent: {(s.get("shot_intent") or "")[:100]} | footage there now: {picked}')
    try:
        data = _call_llm_json(client, _DESIGN_PROMPT, "\n".join(lines), temperature=0.4,
                              max_tokens=3500, tier="fast", allow_fallback=True)
    except Exception as e:
        errors.append(f"cards: design call failed ({type(e).__name__}: {str(e)[:100]})")
        data = {}
    by = {}
    for c in (data.get("cards") or []) if isinstance(data, dict) else []:
        if isinstance(c, dict):
            by[str(c.get("slot_id"))] = c
    for b in beats:
        s = shots_by_id.get(b["slot_ids"][0], {})
        c = by.get(str(b["slot_ids"][0]), {})
        narration = s.get("text", "")
        name = (b.get("step") or {}).get("name", "")
        b["label"] = _clean_label(c.get("label"), "") or _clean_label(name, "") \
            or _clean_label(" ".join((s.get("shot_intent") or narration).split()[:3]), "KEY POINT")
        b["right"] = _clean_right(c.get("right"), b, narration)
        b["image_query"] = " ".join(str(c.get("image_query") or "").split())[:80] \
            or s.get("image_query") or b["label"].lower()
        b["image_prompt"] = " ".join(str(c.get("image_prompt") or "").split())[:300] \
            or f"a photograph of {b['label'].lower()}"
        b["subject_kind"] = "identifiable" if str(c.get("subject_kind")).lower() == "identifiable" \
            else "generic"
        named = names_in_picture(s, b)
        if b["subject_kind"] == "generic" and named:
            # The model called a named product "generic" ("Simple Green Gunk degreaser
            # bottle"): never generate that one — a made-up bottle is the wrong bottle.
            b["subject_kind"] = "identifiable"
            b["reasons"] = list(b.get("reasons") or []) + [
                f"names {', '.join(named[:3])} — real picture only"]
        b["image_style"] = "illustration" if str(c.get("image_style")).lower() == "illustration" \
            else "photo"
        if not c:
            b["reasons"] = list(b.get("reasons") or []) + ["design call returned nothing for this card"]


# ── windows ──────────────────────────────────────────────────────────────────

def _span(shot: dict) -> tuple:
    start = float(shot.get("timestamp") or 0.0)
    end = shot.get("end_timestamp")
    if end is None:
        end = start + float(shot.get("duration_needed_sec") or 0.0)
    return start, max(start, float(end))


def _window(trigger: str, shot: dict, video_end: float) -> tuple:
    start, end = _span(shot)
    lo = TEMPLATES["split_card"]["min_sec"]
    if trigger == "step":
        e = start + _env_float("CARD_STEP_SEC", 4.0)         # sits on the first seconds only
    else:
        e = min(end, start + _env_float("CARD_MAX_SEC", 10.0))
        hold = max(lo, _env_float("CARD_MIN_SEC", 3.0))
        if e - start < hold:
            e = start + hold                                   # let it run into the next shot
    return round(start, 2), round(min(max(e, start + lo), video_end), 2)


_PRIORITY = {"step": 0, "empty": 1, "low_confidence": 2}


def resolve_overlaps(beats: list, gap: float = 0.15) -> list:
    """Time-ordered, non-overlapping beats (the XML validator rejects overlap on a
    track). A step outranks an empty-shot card, which outranks a low-confidence one:
    the lower-ranked beat is trimmed to make room and dropped when too little is left.
    Equal rank: the earlier beat keeps its place and the later one starts after it."""
    lo = TEMPLATES["split_card"]["min_sec"]
    rank = lambda x: _PRIORITY.get(x["trigger"], 3)
    kept = []
    for b in sorted(beats, key=lambda x: (x["start"], rank(x))):
        placed = True
        while kept and b["start"] < kept[-1]["end"] + gap:
            prev = kept[-1]
            if rank(b) < rank(prev):
                if b["start"] - gap - prev["start"] >= lo:
                    prev["end"] = round(b["start"] - gap, 2)      # the older card is cut short
                    break
                kept.pop()                                         # nothing worth keeping of it
                continue
            new_start = round(prev["end"] + gap, 2)
            if b["end"] - new_start >= lo:
                b["start"] = new_start
            else:
                placed = False
            break
        if placed:
            kept.append(b)
    return kept


# ── the plan ─────────────────────────────────────────────────────────────────

def _beat_id(slot_id, used: set) -> str:
    base = f"b{int(slot_id):03d}" if str(slot_id).lstrip("-").isdigit() else f"b-{slot_id}"
    bid, n = base, 2
    while bid in used:
        bid, n = f"{base}-{n}", n + 1
    used.add(bid)
    return bid


def plan_scene(shots: list, topic: str = "", api_key: str = None, qa: dict = None,
               existing: list = None, errors: list = None, diag: dict = None) -> list:
    """Decide the beats for ``shots`` and word them. Returns the beats (sorted,
    non-overlapping). Scores footage first (``shot["confidence"]``), finds the list
    steps, takes the lowest-confidence shots up to ``CARD_MAX_FRACTION`` (default 25%
    of the timeline), has the designer word them, and gives each a window.

    ``existing`` is the previous plan: a beat whose slot and trigger are unchanged keeps
    its wording and pictures (a /refine must not re-spend on the cards it didn't touch),
    and a beat the studio edited (``edited``) always stays. Shots that ended with no
    footage at all are the least confident of every shot, so they get a card too."""
    if errors is None:
        errors = []
    if diag is None:
        diag = {}
    if not enabled():
        return []
    tally = confidence.annotate_confidence(shots, qa)
    diag["confidence"] = tally
    tl = _timeline_shots(shots)
    if not tl:
        return []
    by_id = {s.get("slot_id"): s for s in tl}
    video_end = max(_span(s)[1] for s in tl)

    targets = {}          # slot_id -> (trigger, step-info or None)
    steps = detect_steps(shots, api_key, errors)
    for it in steps:
        targets[it["slot_id"]] = ("step", {"number": it["number"], "kind": it["kind"],
                                           "name": it.get("label", "")})
    diag["steps"] = len(steps)
    if mode() == "auto":
        lows = [s for s in tl if s["slot_id"] not in targets
                and (s.get("confidence") or {}).get("tier") in ("low", "none")]
        lows.sort(key=lambda s: s["confidence"]["score"])
        cap = max(1, math.ceil(_env_float("CARD_MAX_FRACTION", 0.25) * len(tl)))
        diag["low_confidence"] = len(lows)
        for s in lows[:cap]:
            targets[s["slot_id"]] = ("empty" if s["confidence"]["tier"] == "none"
                                     else "low_confidence", None)
        diag["low_capped"] = max(0, len(lows) - cap)

    old = {}
    for b in existing or []:
        for sid in b.get("slot_ids") or []:
            old[sid] = b
    used_ids, beats, to_design = set(), [], []
    for slot, (trigger, step) in targets.items():
        shot = by_id[slot]
        start, end = _window(trigger, shot, video_end)
        prev = old.get(slot)
        reuse = prev and prev.get("trigger") == trigger and prev.get("label") \
            and ((prev.get("step") or {}).get("number") == (step or {}).get("number"))
        conf = shot.get("confidence") or {}
        b = dict(prev) if reuse else {
            "template": "split_card", "trigger": trigger, "slot_ids": [slot], "enabled": True,
            "step": step, "assets": {}, "file": None}
        b.update(start=start, end=end, trigger=trigger,
                 confidence={"footage": conf.get("score"), "tier": conf.get("tier"),
                             **({"image": b["confidence"].get("image")} if reuse and
                                b.get("confidence", {}).get("image") is not None else {})},
                 reasons=list(conf.get("reasons") or [])
                 if trigger != "step" else [f"{step['kind'].lower()} {step['number']} of the list"])
        b["id"] = _beat_id(slot, used_ids)
        beats.append(b)
        if not reuse:
            to_design.append(b)
    for prev in existing or []:                       # studio edits survive a re-plan
        if prev.get("edited") and prev.get("slot_ids") and not any(
                b["slot_ids"] == prev["slot_ids"] for b in beats):
            prev = dict(prev)
            prev["id"] = _beat_id(prev["slot_ids"][0], used_ids)
            beats.append(prev)

    batches = [to_design[i:i + _BATCH] for i in range(0, len(to_design), _BATCH)]
    if batches:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(batches))) as ex:
            list(ex.map(lambda bt: _design_batch(bt, by_id, topic, api_key, errors), batches))
    beats = resolve_overlaps(beats)
    diag["planned"] = len(beats)
    return beats


def summary(beats: list) -> dict:
    """Counts for the review message."""
    live = [b for b in beats if b.get("enabled", True)]
    return {"total": len(live),
            "steps": sum(1 for b in live if b.get("trigger") == "step"),
            "low": sum(1 for b in live if b.get("trigger") == "low_confidence"),
            "empty": sum(1 for b in live if b.get("trigger") == "empty"),
            "ai": sum(1 for b in live if ((b.get("assets") or {}).get("left") or {}).get("kind") == "ai")}
