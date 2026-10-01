"""Pictures for motion cards: real photos first, generated only when none fits.

For one card the order is

  1. the Google images the pipeline already pulled for the shot (stage 8f) — or a
     fresh Serper search with the card's own image query when there are none;
  2. a vision model judges each candidate against the moment ("does this show what
     the narration is about, and is it clean?") — watermarks, captioned graphics,
     collages and screenshots are rejected outright;
  3. the best usable one wins; if none fits and the subject is generic, one image
     is generated (core.ai_images). A named brand/model/person is NEVER generated —
     a picture of the wrong Camry is worse than no card;
  4. every picture is cropped to the card's shape around where the detail is.

Each picture keeps its provenance (kind google/library/ai/detail, source page,
model, prompt, fit) so the editor — and the review studio — can see where it came
from. The vision call is the same provider chain as the storyboard check.
"""

import io
import os

import numpy as np
from PIL import Image, ImageFilter

# The card as the Remotion template draws it (see remotion/src/ImageCard.tsx).
CARD_W, CARD_H = 663, 480
RATIO = CARD_W / CARD_H
OUT_W = 1000                       # stored width; the card shows it at 663
MIN_SOURCE_W = 480                 # a source narrower than this would look soft
_HARD_ISSUES = {"watermark", "text_overlay", "collage", "ad_layout", "screenshot",
                "low_quality", "wrong_subject"}
_SOFT_ISSUES = {"logo_heavy": 0.15, "people_faces": 0.10}


def min_fit() -> float:
    try:
        return float(os.getenv("CARD_IMAGE_MIN_FIT", "").strip() or 0.6)
    except ValueError:
        return 0.6


# ── cropping ─────────────────────────────────────────────────────────────────

def _energy(gray: np.ndarray) -> np.ndarray:
    """Edge energy (where the detail is) — gradient magnitude on a small copy."""
    gx = np.abs(np.diff(gray, axis=1))[:-1, :]
    gy = np.abs(np.diff(gray, axis=0))[:, :-1]
    return gx + gy


def _best_start(profile: np.ndarray, win: int, bias_up: bool = False) -> int:
    """Start of the ``win``-long window with the most energy; ties and near-ties go
    to the middle (or the top for tall photos, where heads and subjects are)."""
    n = len(profile)
    win = max(1, min(win, n))
    if win >= n:
        return 0
    csum = np.concatenate([[0.0], np.cumsum(profile)])
    sums = csum[win:] - csum[:-win]
    idx = np.arange(len(sums))
    mid = (len(sums) - 1) / 2.0
    weight = 1.0 - 0.25 * np.abs(idx - mid) / max(mid, 1.0)
    if bias_up:
        weight = weight * (1.0 - 0.15 * idx / max(len(sums) - 1, 1))
    return int(np.argmax(sums * weight))


def crop_box(im: Image.Image, ratio: float = RATIO) -> tuple:
    """The ``ratio``-shaped window of ``im`` with the most going on in it."""
    w, h = im.size
    if abs(w / h - ratio) < 0.02:
        return 0, 0, w, h
    scale = 256.0 / max(w, h)
    small = im.convert("L").resize((max(8, int(w * scale)), max(8, int(h * scale))))
    e = _energy(np.asarray(small, dtype=float))
    if w / h > ratio:                       # too wide: keep the height, slide sideways
        win_full = int(round(h * ratio))
        start = _best_start(e.sum(axis=0), max(1, int(round(win_full * scale))))
        x0 = min(int(round(start / scale)), w - win_full)
        return x0, 0, x0 + win_full, h
    win_full = int(round(w / ratio))        # too tall: keep the width, slide down
    start = _best_start(e.sum(axis=1), max(1, int(round(win_full * scale))), bias_up=True)
    y0 = min(int(round(start / scale)), h - win_full)
    return 0, y0, w, y0 + win_full


def _fit_with_blur(im: Image.Image, ratio: float, width: int) -> Image.Image:
    """For a very wide or very tall picture a crop would cut the subject away:
    show it whole over a blurred, darkened copy of itself."""
    height = int(round(width / ratio))
    cover = max(width / im.width, height / im.height)
    bg = im.resize((max(width, int(im.width * cover) + 1), max(height, int(im.height * cover) + 1)))
    left, top = (bg.width - width) // 2, (bg.height - height) // 2
    bg = bg.crop((left, top, left + width, top + height)).filter(ImageFilter.GaussianBlur(26))
    bg = Image.eval(bg, lambda v: int(v * 0.55))
    fit = min(width / im.width, height / im.height)
    fg = im.resize((max(1, int(im.width * fit)), max(1, int(im.height * fit))))
    bg.paste(fg, ((width - fg.width) // 2, (height - fg.height) // 2))
    return bg


def prepare(src_path: str, out_path: str, ratio: float = RATIO) -> dict | None:
    """Write ``src_path`` as a card-shaped JPEG (``OUT_W`` wide). None when the file
    can't be read or is too small to look sharp at card size."""
    try:
        with Image.open(src_path) as im0:
            im = im0.convert("RGB")
    except Exception:
        return None
    if im.width < MIN_SOURCE_W or im.height < MIN_SOURCE_W / 2:
        return None
    r = im.width / im.height
    if r > ratio * 1.7 or r < ratio / 1.7:                      # would lose too much
        out = _fit_with_blur(im, ratio, min(OUT_W, max(im.width, 700)))
        mode = "fit"
    else:
        x0, y0, x1, y1 = crop_box(im, ratio)
        out = im.crop((x0, y0, x1, y1))
        mode = "crop"
        if out.width > OUT_W:
            out = out.resize((OUT_W, int(round(OUT_W / ratio))), Image.LANCZOS)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out.save(out_path, "JPEG", quality=90)
    return {"path": out_path, "width": out.width, "height": out.height, "mode": mode}


def detail(prepared_path: str, out_path: str, zoom: float = 0.6) -> dict | None:
    """A close-up of the busiest part of an already-prepared card picture, for the
    right panel when there is no second photo."""
    try:
        with Image.open(prepared_path) as im0:
            im = im0.convert("RGB")
    except Exception:
        return None
    cw, ch = int(im.width * zoom), int(im.height * zoom)
    if cw < 560:                                                 # would be too soft
        return None
    e = _energy(np.asarray(im.convert("L").resize((max(8, im.width // 4), max(8, im.height // 4))),
                           dtype=float))
    sx = _best_start(e.sum(axis=0), max(1, cw // 4))
    sy = _best_start(e.sum(axis=1), max(1, ch // 4))
    x0, y0 = min(sx * 4, im.width - cw), min(sy * 4, im.height - ch)
    out = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((OUT_W, int(round(OUT_W / RATIO))), Image.LANCZOS)
    out.save(out_path, "JPEG", quality=90)
    return {"path": out_path, "width": out.width, "height": out.height, "mode": "detail"}


# ── judging candidates ───────────────────────────────────────────────────────

_RATE_SYSTEM = """You help a video editor choose ONE photo for a single moment of a narrated video. You get numbered candidate images (numbered in the order given) and what the moment is about.

For every image give:
- "fit": 0.0-1.0 — how well the picture shows what the moment is about. 1.0 = exactly the subject, clean and ready to use; 0.6 = acceptable, clearly related; below 0.4 = wrong subject or generic filler.
- "issues": any that apply from: "watermark" (stock-agency or website watermark), "text_overlay" (captions, titles, badges, "2 Pack"/"SALE" banners or infographic callouts ADDED on or around the picture — text that is part of the object itself, such as a product's own label or a street sign, is NOT an overlay), "collage" (several photos, a grid, a before/after composite, or the same product repeated many times as in a multi-pack listing), "ad_layout" (a retail or marketing graphic: colored header bars, badges, big empty areas or a product pushed to one side — a plain product photo on white is fine), "screenshot" (web page or app UI), "logo_heavy", "low_quality" (blurry, tiny, pixelated), "wrong_subject", "people_faces" (a person's face is the main subject).
Be strict: a related-but-different object is "wrong_subject" (a phone battery is not a car battery). A specific brand/model/product only counts as a match when it is clearly the one asked for. Judge only what is visible; never identify real people.

Output ONLY valid JSON:
{"images": [{"id": 1, "fit": 0.8, "issues": [], "note": "<=10 words on what it shows"}], "best": <id or null>}
Include every image id you were given."""


def _vision(system: str, user: str, jpegs: list):
    """The prompt and thumbnails through the storyboard check's provider chain.
    None when no vision provider is configured or the call fails."""
    try:
        from core import storyboard
        if not storyboard.vision_configured():
            return None
        return storyboard._vision_json(system, user, jpegs)
    except Exception as e:
        print(f"[card_images] vision call failed: {type(e).__name__}: {str(e)[:120]}")
        return None


def _thumb(path: str, side: int = 512):
    try:
        with Image.open(path) as im0:
            im = im0.convert("RGB")
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        return buf.getvalue()
    except Exception:
        return None


def rate(candidates: list, ctx: dict) -> list:
    """Score ``candidates`` (dicts with ``local_path``) against the moment ``ctx``
    (``topic, narration, label, query, subject_kind``). Returns one dict per
    candidate, same order: ``{fit, issues, note, usable, rated}``. ``rated`` is
    False when no vision verdict was available — the caller decides how far to
    trust such an image."""
    out = [{"fit": None, "issues": [], "note": "", "usable": False, "rated": False}
           for _ in candidates]
    thumbs = [_thumb(c.get("local_path", "")) for c in candidates]
    live = [i for i, t in enumerate(thumbs) if t]
    for i in range(len(candidates)):
        if i not in live:
            out[i]["issues"] = ["unreadable"]
    if not live:
        return out
    user = "\n".join([
        f"VIDEO TOPIC: {ctx.get('topic') or 'unknown'}",
        f"NARRATION AT THIS MOMENT: \"{(ctx.get('narration') or '').strip()[:300]}\"",
        f"CARD CAPTION: {ctx.get('label') or ''}",
        f"PICTURE WANTED: {ctx.get('query') or ctx.get('label') or ''}",
        f"SUBJECT KIND: {ctx.get('subject_kind') or 'generic'}"
        + (" (a specific named thing: it must be clearly the right one)"
           if ctx.get("subject_kind") == "identifiable" else ""),
        f"IMAGES: {len(live)} attached, numbered 1-{len(live)} in order.",
    ])
    data = _vision(_RATE_SYSTEM, user, [thumbs[i] for i in live])
    rows = (data or {}).get("images") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return out
    by_id = {}
    for r in rows:
        try:
            by_id[int(r.get("id"))] = r
        except (TypeError, ValueError, AttributeError):
            continue
    floor = min_fit()
    for n, i in enumerate(live, 1):
        r = by_id.get(n)
        if not r:
            continue
        try:
            fit = max(0.0, min(1.0, float(r.get("fit"))))
        except (TypeError, ValueError):
            continue
        issues = [str(x).lower() for x in (r.get("issues") or [])]
        for iss in issues:
            fit -= _SOFT_ISSUES.get(iss, 0.0)
        fit = round(max(0.0, fit), 2)
        out[i] = {"fit": fit, "issues": issues, "note": str(r.get("note") or "")[:80],
                  "rated": True,
                  "usable": fit >= floor and not (set(issues) & _HARD_ISSUES)}
    return out


# ── the orchestration ────────────────────────────────────────────────────────

def _asset(kind: str, path: str, **extra) -> dict:
    return dict({"kind": kind, "path": path}, **extra)


def _from_candidate(c: dict, verdict: dict, prepared: dict) -> dict:
    lib = bool(c.get("library_segment_id"))
    return _asset("library" if lib else "google", prepared["path"],
                  source_path=c.get("local_path", ""), url=c.get("url", ""),
                  page=c.get("page", ""), title=c.get("title", ""),
                  fit=verdict.get("fit"), rated=verdict.get("rated", False),
                  note=verdict.get("note", ""))


def _search_more(query: str, project_dir: str, tag: str, errors: list, num: int = 4) -> list:
    """Fresh Google images for the card's own query (one Serper credit)."""
    from core.related_images import google_image_search, serper_configured
    if not query or not serper_configured():
        return []
    from core.pipeline import _download_image
    out = []
    try:
        found = google_image_search(query, num=num + 2, errors=errors)
    except Exception as e:
        errors.append(f"card images: search '{query}' failed ({type(e).__name__})")
        return []
    folder = os.path.join(project_dir, "images", "cards")
    for k, c in enumerate(found):
        if len(out) >= num:
            break
        path = _download_image(c["url"], folder, k + 1, stem=f"{tag}-search-{k + 1}")
        if path:
            out.append({"url": c["url"], "local_path": path, "title": c.get("title", ""),
                        "page": c.get("context", ""), "width": c.get("width", 0),
                        "height": c.get("height", 0), "query": query})
    return out


def source_images(beat: dict, shot: dict, project_dir: str, topic: str = "",
                  errors: list = None, allow_search: bool = True) -> dict:
    """Pick (or generate) the picture(s) for one card beat. Returns
    ``{"left": asset|None, "right": asset|None, "ai": bool, "fit": float|None}``;
    ``right`` is filled only when the beat's right panel is an image. Never raises."""
    if errors is None:
        errors = []
    from core import ai_images
    folder = os.path.join(project_dir, "images", "cards")
    os.makedirs(folder, exist_ok=True)
    tag = beat["id"]
    want_right = (beat.get("right") or {}).get("kind") == "image"
    ctx = {"topic": topic, "narration": (shot or {}).get("text", ""),
           "label": beat.get("label", ""), "query": beat.get("image_query", ""),
           "subject_kind": beat.get("subject_kind", "generic")}

    cands = [c for c in (shot or {}).get("images") or [] if os.path.exists(c.get("local_path", ""))]
    pool, verdicts = [], []

    def _consider(new):
        rated = rate(new, ctx) if new else []
        pool.extend(new)
        verdicts.extend(rated)

    _consider(cands[:6])
    best = [i for i, v in enumerate(verdicts) if v["usable"]]
    # Nothing usable from the shot's own images: try the card's own query once.
    if not best and allow_search and beat.get("image_query") and \
            beat.get("image_query") != (shot or {}).get("image_query"):
        _consider(_search_more(beat["image_query"], project_dir, tag, errors))
        best = [i for i, v in enumerate(verdicts) if v["usable"]]
    # Without a vision verdict nothing is "usable" — accept the top search result
    # (unrated) rather than spend a generation on every card.
    if not best and pool and not any(v["rated"] for v in verdicts):
        best = [i for i, v in enumerate(verdicts) if "unreadable" not in v["issues"]][:2]

    def _ranked(idx):
        return sorted(idx, key=lambda i: (-(verdicts[i]["fit"] if verdicts[i]["fit"] is not None else 0.55),
                                          i))

    left = right = None
    for i in _ranked(best):
        prepared = prepare(pool[i]["local_path"], os.path.join(folder, f"{tag}-left.jpg"))
        if prepared:
            left = _from_candidate(pool[i], verdicts[i], prepared)
            best = [j for j in best if j != i]
            break
    if left is None and beat.get("subject_kind", "generic") != "identifiable" \
            and ai_images.enabled():
        gen = ai_images.generate(beat.get("image_prompt") or beat.get("label", ""),
                                 os.path.join(folder, f"{tag}-ai.jpg"),
                                 style=beat.get("image_style", "photo"), errors=errors)
        if gen:
            prepared = prepare(gen["path"], os.path.join(folder, f"{tag}-left.jpg"))
            if prepared:
                left = _asset("ai", prepared["path"], source_path=gen["path"], model=gen["model"],
                              prompt=beat.get("image_prompt") or beat.get("label", ""),
                              cost=gen["cost"], fit=0.7, rated=False,
                              style=beat.get("image_style", "photo"))
    if left and want_right:
        for i in _ranked(best):
            prepared = prepare(pool[i]["local_path"], os.path.join(folder, f"{tag}-right.jpg"))
            if prepared:
                right = _from_candidate(pool[i], verdicts[i], prepared)
                break
        if right is None:
            d = detail(left["path"], os.path.join(folder, f"{tag}-right.jpg"))
            if d:
                right = _asset("detail", d["path"], source_path=left["path"], fit=left.get("fit"))
    return {"left": left, "right": right, "ai": bool(left and left["kind"] == "ai"),
            "fit": left.get("fit") if left else None}
