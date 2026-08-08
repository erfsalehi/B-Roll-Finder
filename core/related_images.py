"""Related still images (Google) → Clip Library.

A companion to :mod:`core.extras`. Where extras mine the script for named
entities and pull *video* B-roll, this mines the same script for the things a
car/product video talks about and pulls *still images* from Google — brands,
car models, products, car parts & systems, and general concept/idea imagery —
so the editor has high-resolution stills to drop in.

Like extras, these are **library-only**: each image is downloaded and stored in
the searchable Clip Library (source ``google_image``). They never go on a
timeline and are never auto-injected into a video's candidate pool (stills would
break a b-roll edit); the editor finds them by searching the library.

Images come from the official Google Programmable Search (Custom Search JSON)
API — set ``GOOGLE_CSE_API_KEY`` and ``GOOGLE_CSE_CX`` (a Programmable Search
Engine configured to search the whole web with Image search enabled). Only
images large enough for a 1080p edit are kept.
"""

import os

import requests

from core.extras import extract_extra_entities


# Google Custom Search JSON API — image search endpoint.
_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

# Minimum pixel dimensions to be usable in a 1080p edit. 1280×720 is the floor
# (upscales acceptably to fill a 1080p frame or sits fine as an inset); we prefer
# ≥1920×1080 via the imgSize bias below but don't hard-require it, or too few
# images survive.
_MIN_WIDTH = 1280
_MIN_HEIGHT = 720


def cse_configured() -> bool:
    """True when both Google Custom Search credentials are present."""
    return bool(os.getenv("GOOGLE_CSE_API_KEY", "").strip()
                and os.getenv("GOOGLE_CSE_CX", "").strip())


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default)) or default))
    except ValueError:
        return default


def build_image_queries(entities: dict, max_queries: int = None) -> list:
    """Turn extracted entities into ``[{"query", "kind"}]`` image searches.

    Mirrors :func:`core.extras.build_extra_keywords` but with phrasing tuned for
    *still* imagery (logos, exteriors, packaging, part diagrams) across the
    categories the user asked for: brands, car models, products, car parts &
    systems, and general ideas (themes). Flattened round-robin so a script naming
    many things still gets one image query for each before any second query, then
    capped at ``max_queries`` (env ``RELATED_IMAGE_MAX_QUERIES``, default 24 — a
    higher budget than extras, since the user wants more images)."""
    if max_queries is None:
        max_queries = _int_env("RELATED_IMAGE_MAX_QUERIES", 24)

    groups = []
    for b in (entities.get("brands") or []):
        groups.append([
            {"query": f"{b} logo", "kind": "brand"},
            {"query": f"{b} car emblem", "kind": "brand"},
        ])
    for m in (entities.get("models") or []):
        groups.append([
            {"query": f"{m} car", "kind": "model"},
            {"query": f"{m} exterior", "kind": "model"},
            {"query": f"{m} interior dashboard", "kind": "model"},
        ])
    for p in (entities.get("parts") or []):
        groups.append([
            {"query": f"{p} car part", "kind": "part"},
            {"query": f"{p} diagram", "kind": "part"},
        ])
    for pr in (entities.get("products") or []):
        groups.append([
            {"query": f"{pr} product", "kind": "product"},
            {"query": f"{pr} packaging", "kind": "product"},
        ])
    # General ideas / concepts — the theme phrases are already visual; use them
    # verbatim so an idea-driven video (no named entities) still gets imagery.
    for t in (entities.get("themes") or []):
        groups.append([{"query": t, "kind": "idea"}])

    # Round-robin flatten (all firsts, then all seconds, …) so the cap spreads
    # across subjects rather than exhausting on the first few.
    queries = []
    if groups:
        width = max(len(g) for g in groups)
        for i in range(width):
            for g in groups:
                if i < len(g):
                    queries.append(g[i])

    seen, deduped = set(), []
    for q in queries:
        key = q["query"].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(q)
    return deduped[:max(0, max_queries)]


def google_image_search(query: str, num: int = 5, errors: list = None,
                        min_width: int = _MIN_WIDTH,
                        min_height: int = _MIN_HEIGHT) -> list:
    """Search Google Images via the Custom Search JSON API for ``query``.

    Returns up to ``num`` image dicts ``{url, source, title, thumbnail, width,
    height, context}`` large enough for a 1080p edit (≥ ``min_width`` ×
    ``min_height``). Returns ``[]`` (and appends to ``errors``) when the
    credentials are missing or the request fails."""
    if errors is None:
        errors = []
    api_key = os.getenv("GOOGLE_CSE_API_KEY", "").strip()
    cx = os.getenv("GOOGLE_CSE_CX", "").strip()
    if not (api_key and cx):
        errors.append("google images: GOOGLE_CSE_API_KEY / GOOGLE_CSE_CX not set")
        return []

    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "searchType": "image",
        # Bias toward big images so results clear the 1080p bar; the returned
        # width/height are still checked below.
        "imgSize": os.getenv("RELATED_IMAGE_SIZE", "xlarge").strip() or "xlarge",
        "num": min(10, max(1, num * 2)),   # over-fetch, then filter by resolution
        "safe": "off",
    }
    try:
        r = requests.get(_CSE_ENDPOINT, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        errors.append(f"google images '{query}': {e}")
        return []

    out = []
    for item in (data.get("items") or []):
        link = (item.get("link") or "").strip()
        if not link:
            continue
        img = item.get("image") or {}
        try:
            w = int(img.get("width") or 0)
            h = int(img.get("height") or 0)
        except (TypeError, ValueError):
            w = h = 0
        if w and h and (w < min_width or h < min_height):
            continue
        out.append({
            "url": link,
            "source": "google_image",
            "title": (item.get("title") or query).strip(),
            "thumbnail": (img.get("thumbnailLink") or "").strip(),
            "width": w,
            "height": h,
            "context": (img.get("contextLink") or "").strip(),
        })
        if len(out) >= num:
            break
    return out


def fetch_related_images(script_text: str, api_key: str, errors: list = None,
                         per_query: int = None) -> list:
    """Mine the script for subjects and fetch related still images for each.

    Extracts the same entities as extras (brands / models / parts / products /
    themes) → builds image queries → searches Google Images → returns a flat,
    de-duplicated list of image dicts, each tagged with its ``query`` and
    ``kind``. Returns ``[]`` when nothing is named / no credentials / no images
    qualify."""
    if errors is None:
        errors = []
    if not cse_configured():
        errors.append("related images: Google Custom Search not configured "
                      "(set GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX)")
        return []
    if per_query is None:
        per_query = _int_env("RELATED_IMAGE_PER_QUERY", 4)

    entities = extract_extra_entities(script_text, api_key)
    queries = build_image_queries(entities)
    if not queries:
        return []

    images = []
    seen_urls = set()
    for q in queries:
        term = q["query"]
        for img in google_image_search(term, num=per_query, errors=errors):
            if img["url"] in seen_urls:
                continue
            seen_urls.add(img["url"])
            img = dict(img)
            img["query"] = term
            img["kind"] = q["kind"]
            images.append(img)
    return images
