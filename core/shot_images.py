"""Per-shot Google images.

For every timeline shot, find a few real Google Images results (not stock) that
show what the narration is talking about, download them next to the project,
and attach them to the shot — so the editor has a still ready for each moment.

Why a dedicated query: a shot's ``search_queries`` are written for *video*
search ("slow motion close up of engine bay, 4k") and read badly as a Google
Images search. One cheap batched LLM call rewrites each shot into a short,
concrete image query that names the actual subject ("2024 Toyota Camry engine
bay"). If that call fails, the query falls back to the shot's first video query
with the camera/footage words stripped.

Output layout (inside the project folder, so it rides along in the zip)::

    downloads/<project>/images/shots/shot_03/03-1-toyota-camry-engine.jpg
    downloads/<project>/images/shots/sources.txt   ← page each image came from

Images are still NOT placed on the timeline. Google Images results are not
licensed for reuse; ``sources.txt`` exists so the editor can check each one.

Costs one Serper credit per distinct image query (≤ one per shot). Gate:
``ENABLE_SHOT_IMAGES``; count: ``SHOT_IMAGES_PER_SHOT`` (default 3).
"""

import concurrent.futures
import os
import re
import threading

from core.related_images import (backend_tripped, google_image_search,
                                  serper_configured)


_BATCH = 40   # shots per query-writing LLM call

_QUERY_PROMPT = (
    "You write Google Images search queries for a video editor. Each item is ONE "
    "shot of a narrated video: its narration line plus the video-search queries "
    "already written for it. For EACH shot write ONE Google Images query (2-7 "
    "words) that finds a real photo of the concrete subject being talked about.\n"
    "Rules:\n"
    "- Name the specific thing: brand, model + year, product, part, place, person, "
    "event, document — whatever the narration actually refers to. Use the video "
    "topic to disambiguate (\"the engine\" in a Camry video → \"Toyota Camry engine\").\n"
    "- No video/camera words: footage, b-roll, clip, video, 4k, slow motion, "
    "cinematic, aerial shot, close up, drone, stock.\n"
    "- If the line is abstract, pick the most concrete subject that illustrates it.\n"
    "- Plain search phrase, no quotes or operators.\n"
    'Return STRICT JSON: {"queries": [{"slot_id": <id>, "query": "<text>"}]}'
)

# Words that make sense for video search but pollute an image search.
_VIDEO_WORDS = re.compile(
    r"\b(4k|hd|1080p|footage|b-?roll|clip|clips|video|videos|stock|cinematic|"
    r"slow[- ]?motion|timelapse|time[- ]lapse|drone|aerial|shot|shots|close[- ]?up|"
    r"pov|pan|panning|tracking|montage)\b", re.I)


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default)) or default))
    except ValueError:
        return default


def per_shot_count() -> int:
    return _int_env("SHOT_IMAGES_PER_SHOT", 3)


def _fallback_query(shot: dict, video_topic: str = "") -> str:
    for q in (shot.get("search_queries") or []):
        cleaned = " ".join(_VIDEO_WORDS.sub(" ", str(q)).split())
        if cleaned:
            return cleaned
    return (video_topic or "").strip()


def build_shot_image_queries(shots: list, api_key: str = None,
                             video_topic: str = "", errors: list = None) -> dict:
    """Return ``{slot_id: image_query}`` for ``shots``. One fast-tier LLM call
    per :data:`_BATCH` shots; any shot the LLM skips (or a failed batch) gets
    :func:`_fallback_query`."""
    from core.keywords import _call_llm_json
    try:
        from groq import Groq
    except Exception:
        Groq = None
    if errors is None:
        errors = []

    out: dict = {}
    client = Groq(api_key=api_key) if (api_key and Groq) else None
    for i in range(0, len(shots), _BATCH):
        batch = shots[i:i + _BATCH]
        lines = [f"VIDEO TOPIC: {video_topic or 'unknown'}", ""]
        for s in batch:
            vq = " | ".join(str(q) for q in (s.get("search_queries") or [])[:3])
            lines.append(f'slot_id {s.get("slot_id")}: "{(s.get("text") or "").strip()[:300]}"'
                         f"  [video queries: {vq or '—'}]")
        try:
            data = _call_llm_json(client, _QUERY_PROMPT, "\n".join(lines),
                                  temperature=0.3, max_tokens=3000, tier="fast",
                                  allow_fallback=True)
            for item in (data.get("queries") or []):
                q = " ".join(str(item.get("query") or "").split())
                if q:
                    out[str(item.get("slot_id"))] = q
        except Exception as e:
            errors.append(f"shot images: query writing failed ({e}); using video queries")

    result = {}
    for s in shots:
        q = out.get(str(s.get("slot_id"))) or _fallback_query(s, video_topic)
        if q:
            result[s.get("slot_id")] = q
    return result


def _slug(text: str, max_len: int = 40) -> str:
    cleaned = "".join(c if c.isalnum() else " " for c in (text or ""))
    return "-".join(cleaned.split()).lower()[:max_len].strip("-") or "image"


def _shot_dir_name(slot_id) -> str:
    try:
        return f"shot_{int(slot_id):02d}"
    except (TypeError, ValueError):
        return f"shot_{_slug(str(slot_id), 20)}"


def fetch_shot_images(shots: list, project_name: str, api_key: str = None,
                      video_topic: str = "", errors: list = None,
                      per_shot: int = None, should_cancel=None,
                      progress=None) -> int:
    """Find, download and attach ``per_shot`` Google images to every timeline
    shot. Sets ``shot["images"] = [{url, local_path, title, page, width, height,
    query}]`` and returns the number of images saved. No-op (returns 0) when no
    image backend is configured.

    The same image URL is never given to two shots, and each shot keeps a few
    spare results so an image host that refuses the download doesn't leave the
    shot short."""
    from core.pipeline import _download_image, _safe_for_fs

    if errors is None:
        errors = []
    if per_shot is None:
        per_shot = per_shot_count()
    if per_shot <= 0:
        return 0
    # Serper only. Google Custom Search's free tier is 100 queries a day, and a
    # 10-minute video has 100+ shots, so the fallback just produced a wall of
    # 429s. Related images (a few dozen queries) still accept it.
    if not serper_configured():
        errors.append("shot images: skipped — SERPER_API_KEY isn't set in this "
                      "bot's environment (per-shot images don't use Google Custom Search)")
        return 0

    targets = [s for s in shots if not s.get("is_extra")
               and s.get("priority") != "none" and (s.get("text") or s.get("search_queries"))]
    if not targets:
        return 0

    queries = build_shot_image_queries(targets, api_key, video_topic, errors)

    # One search per distinct query (shots often repeat a subject).
    results: dict = {}
    uniq = sorted({q for q in queries.values()})

    def _search(q):
        if should_cancel and should_cancel():
            return q, []
        return q, google_image_search(q, num=10, errors=errors)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for q, imgs in ex.map(_search, uniq):
            results[q] = imgs

    # The backend refused (out of credits / bad key): the one error it logged
    # says why, so don't add a "no images" line for every shot on top of it.
    if backend_tripped("serper"):
        return 0

    # Claim candidates shot by shot (in timeline order) so no URL repeats.
    claimed: set = set()
    plan = []   # (shot, query, [candidates])
    for s in targets:
        q = queries.get(s.get("slot_id"))
        if not q:
            continue
        cands = [c for c in results.get(q, []) if c["url"] not in claimed]
        cands = cands[:per_shot + 3]          # a few spares for failed downloads
        claimed.update(c["url"] for c in cands)
        if cands:
            plan.append((s, q, cands))
        else:
            errors.append(f"shot {s.get('slot_id')}: no Google images for '{q}'")

    base = os.path.join(os.path.abspath("downloads"), _safe_for_fs(project_name, 50),
                        "images", "shots")
    lock = threading.Lock()
    done = {"n": 0}

    def _download_for(item):
        s, q, cands = item
        if should_cancel and should_cancel():
            return s, q, []
        out_dir = os.path.join(base, _shot_dir_name(s.get("slot_id")))
        got = []
        for c in cands:
            if len(got) >= per_shot:
                break
            stem = f"{_shot_dir_name(s.get('slot_id'))[5:]}-{len(got) + 1}-{_slug(q, 30)}"
            path = _download_image(c["url"], out_dir, len(got) + 1, stem=stem)
            if path:
                got.append({"url": c["url"], "local_path": path,
                            "title": c.get("title", ""), "page": c.get("context", ""),
                            "width": c.get("width", 0), "height": c.get("height", 0),
                            "query": q})
        with lock:
            done["n"] += 1
            n = done["n"]
        if progress:
            try:
                progress(n, len(plan))
            except Exception:
                pass
        return s, q, got

    saved = 0
    source_lines = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for s, q, got in ex.map(_download_for, plan):
            if not got:
                errors.append(f"shot {s.get('slot_id')}: every image download failed for '{q}'")
                continue
            s["images"] = got
            s["image_query"] = q
            saved += len(got)
            for g in got:
                source_lines.append(f"{os.path.basename(g['local_path'])}\t{g['page'] or g['url']}")

    if source_lines:
        try:
            os.makedirs(base, exist_ok=True)
            with open(os.path.join(base, "sources.txt"), "w", encoding="utf-8") as f:
                f.write("# file\tpage the image was found on (check licensing before publishing)\n")
                f.write("\n".join(sorted(source_lines)) + "\n")
        except OSError as e:
            errors.append(f"shot images: sources.txt: {e}")
    return saved
