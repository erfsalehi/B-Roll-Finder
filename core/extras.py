"""Extra contextual B-roll clips.

After the narration timeline is built, we mine the script for the concrete
entities it names AND the video's overall concept, then fetch a handful of
generic, on-theme YouTube clips for each — spare footage the editor can swap in
for any unwanted pick.

Rules (per the product spec):
  * BRAND   (e.g. Toyota)     -> the company, its logo, its manufacturing line
  * MODEL   (e.g. Camry)      -> POV driving, test drive, review/introduction
  * PART    (e.g. brakes)     -> "<part> explained" / how it works
  * PRODUCT (e.g. Sea Foam)   -> "<product> review" / close up
  * THEME   (e.g. fuel video) -> the subject's iconic imagery, as ready-made
                                 visual queries ("gas station forecourt",
                                 "pumping fuel into car", "fuel gauge dashboard")

The theme queries capture what the video is ABOUT even when it names no specific
entities, so a concept-driven video still gets relevant extras. Each keyword
pulls 2-3 YouTube videos (same filters as the main timeline: HD, landscape, no
Shorts). They're appended AFTER the last narration shot as extra "shots" named
``Extra - <keyword>``. YouTube-only by design.
"""

import os

from core.keywords import _call_llm_json

try:
    from groq import Groq
except Exception:  # pragma: no cover - import guard
    Groq = None


_ENTITY_PROMPT = (
    "You read a voiceover script and extract what's needed to source extra stock "
    "B-roll. Return STRICT JSON only:\n"
    '{"brands": [...], "models": [...], "parts": [...], "products": [...], '
    '"themes": [...]}\n'
    "- brands: vehicle manufacturers/companies named (e.g. Toyota, Honda, BMW).\n"
    "- models: specific vehicle models named (e.g. Camry, Corolla, Model 3). Do NOT "
    "include the brand word alone here.\n"
    "- parts: specific car parts/components named (e.g. brake pads, transmission, "
    "alternator, timing belt).\n"
    "- products: named consumer products the script reviews or recommends — motor "
    "oils, additives, supplements, fuel/injector cleaners, engine treatments, "
    "fluids, sprays, tools, and the like (e.g. Sea Foam, Liqui Moly, motor oil "
    "additive, fuel injector cleaner, engine flush). This is the right bucket for a "
    "product round-up / 'best X' video — put each reviewed item here.\n"
    "- themes: 4-8 short, concrete, VISUAL stock-footage search phrases that capture "
    "the video's OVERALL subject and its iconic imagery — generic evergreen B-roll "
    "that fits anywhere, NOT tied to one named entity. Each 2-4 words, filmable, "
    "clearly in the subject's domain. For a video about FUEL: \"gas station "
    "forecourt\", \"pumping fuel into car\", \"fuel gauge dashboard\", \"petrol pump "
    "nozzle\", \"fuel tanker highway\". For COFFEE: \"barista pouring espresso\", "
    "\"coffee beans roasting\", \"cafe interior\". Always provide themes, even when "
    "no specific brands/models/parts/products are named.\n"
    "brands/models/parts/products: only things ACTUALLY named, canonical Title-Case, "
    "deduplicated. themes: lowercase search phrases. Empty entity arrays are fine. "
    "No prose, JSON only."
)


def _theme_max() -> int:
    try:
        return max(0, int(os.getenv("EXTRA_THEME_MAX", "8") or 8))
    except ValueError:
        return 8


def extract_extra_entities(script_text: str, api_key: str) -> dict:
    """Pull ``{brands, models, parts, products, themes}`` from the script — the
    named automotive entities plus a few concept/theme B-roll search phrases for
    the video's overall subject (so even an entity-less video still gets on-theme
    extras). Returns empty lists on any failure."""
    out = {"brands": [], "models": [], "parts": [], "products": [], "themes": []}
    if not api_key or not (script_text or "").strip():
        return out
    client = Groq(api_key=api_key) if Groq else None
    try:
        data = _call_llm_json(client, _ENTITY_PROMPT,
                              f"SCRIPT:\n{script_text[:12000]}", tier="smart")
    except Exception as e:
        print(f"[extras] entity extraction failed: {e}")
        return out
    for k in out:
        seen = set()
        for v in (data.get(k) or []):
            v = str(v).strip()
            key = v.lower()
            if v and key not in seen:
                seen.add(key)
                out[k].append(v)
    out["themes"] = out["themes"][:_theme_max()]   # bound the concept budget
    return out


def build_extra_keywords(entities: dict, max_keywords: int = None) -> list:
    """Turn extracted entities into ``[{"keyword", "kind"}]`` search terms per the
    brand/model/part/product rules, plus the concept ``themes`` (used verbatim).
    Capped at ``max_keywords`` (env EXTRA_MAX_KEYWORDS, default 18) so a script
    naming dozens of things can't explode the fetch.

    Keywords are flattened *round-robin* across entities — every entity's FIRST
    keyword comes before any entity's second, so a product round-up naming 13
    items still covers all 13 under the cap instead of spending the whole budget
    on the first few."""
    if max_keywords is None:
        try:
            max_keywords = int(os.getenv("EXTRA_MAX_KEYWORDS", "18") or 18)
        except ValueError:
            max_keywords = 18

    # One ordered keyword group per named entity (best/most-distinctive first).
    groups = []
    for b in (entities.get("brands") or []):
        groups.append([
            {"keyword": f"{b} cars company", "kind": "brand"},
            {"keyword": f"{b} logo", "kind": "brand"},
            {"keyword": f"{b} factory manufacturing line", "kind": "brand"},
        ])
    for m in (entities.get("models") or []):
        groups.append([
            {"keyword": f"{m} pov driving", "kind": "model"},
            {"keyword": f"{m} test drive", "kind": "model"},
            {"keyword": f"{m} review introduction", "kind": "model"},
        ])
    for p in (entities.get("parts") or []):
        groups.append([
            {"keyword": f"{p} explained", "kind": "part"},
            {"keyword": f"how {p} works", "kind": "part"},
        ])
    for pr in (entities.get("products") or []):
        groups.append([
            {"keyword": f"{pr} review", "kind": "product"},
            {"keyword": f"{pr} close up", "kind": "product"},
        ])
    # Concept/theme queries are already full visual search phrases (e.g. "gas
    # station forecourt") — one standalone keyword each, so they interleave into
    # the round-robin "firsts" pass and are guaranteed footage even when the video
    # names no specific entities.
    for t in (entities.get("themes") or []):
        groups.append([{"keyword": t, "kind": "theme"}])

    # Round-robin flatten (column-major across groups): all firsts, then all
    # seconds, then all thirds — so the cap spreads across entities, not depth.
    kws = []
    if groups:
        width = max(len(g) for g in groups)
        for i in range(width):
            for g in groups:
                if i < len(g):
                    kws.append(g[i])

    # De-dup (case-insensitive) preserving order, then cap.
    seen, deduped = set(), []
    for k in kws:
        key = k["keyword"].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(k)
    return deduped[:max(0, max_keywords)]


def _extra_per_keyword() -> int:
    try:
        return max(1, int(os.getenv("EXTRA_PER_KEYWORD", "2") or 2))
    except ValueError:
        return 2


def fetch_extra_shots(script_text: str, api_key: str, errors: list = None,
                      start_slot_id: int = 1, start_sec: float = 0.0,
                      clip_sec: float = None, min_height: int = 720) -> list:
    """Build the extra "shots" to append after the timeline.

    Extracts entities → keywords → fetches 2-3 HD landscape YouTube clips per
    keyword (same filters as the main pipeline) → returns a list of shot dicts,
    one per clip, placed back-to-back starting at ``start_sec``. Each is tagged
    ``is_extra`` and ``extra_label='Extra - <keyword>'``. Returns ``[]`` when
    extras are disabled, nothing is named, or no clips qualify."""
    from core.director_search import search_youtube_classic, filter_youtube_sd_candidates
    from core.pipeline import drop_shorts, drop_vertical, drop_long_videos

    if errors is None:
        errors = []
    if clip_sec is None:
        try:
            clip_sec = float(os.getenv("EXTRA_CLIP_SEC", "6") or 6)
        except ValueError:
            clip_sec = 6.0

    entities = extract_extra_entities(script_text, api_key)
    keywords = build_extra_keywords(entities)
    if not keywords:
        return []

    per_kw = _extra_per_keyword()
    yt_api_key = os.getenv("YOUTUBE_API_KEY", "")

    shots = []
    slot = start_slot_id
    cursor = float(start_sec)
    for kw in keywords:
        term = kw["keyword"]
        # Fetch a buffer, filter, then keep the top per_kw.
        probe = {"slot_id": f"EX-{slot}", "priority": "low",
                 "search_queries": [term], "youtube_keywords": [term],
                 "video_results": search_youtube_classic(term, num_results=per_kw * 2,
                                                          errors=errors)}
        if not probe["video_results"]:
            continue
        drop_shorts([probe])
        drop_long_videos([probe])
        drop_vertical([probe])
        if yt_api_key:
            try:
                filter_youtube_sd_candidates([probe], api_key=yt_api_key)
            except Exception as e:
                errors.append(f"extras hd_filter '{term}': {e}")

        picks = (probe.get("video_results") or [])[:per_kw]
        for c in picks:
            if not c.get("url"):
                continue
            c = dict(c)
            c["matched_query"] = term
            shots.append({
                "slot_id": slot,
                "timestamp": round(cursor, 3),
                "end_timestamp": round(cursor + clip_sec, 3),
                "priority": "low",
                "is_extra": True,
                "extra_keyword": term,
                "extra_kind": kw["kind"],
                "extra_label": f"Extra - {term}",
                "search_queries": [term],
                "shot_intent": f"extra B-roll: {term}",
                "selected_results": [c],
                "auto_selected": True,
            })
            slot += 1
            cursor += clip_sec
    return shots
