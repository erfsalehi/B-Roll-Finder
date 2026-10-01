"""AI image generation through OpenRouter's Image API (``POST /api/v1/images``).

Used by the motion cards when no real photo fits a moment. Chosen by a bake-off on
2026-10-01 (two photographic test prompts, 4:3, priced from OpenRouter's own
endpoint listing):

    microsoft/mai-image-2.6-flash        ~$0.015  most photographic, best framing   <- first
    recraft/recraft-v4.1-flash            $0.007  cheapest, tighter crops           <- fallback
    black-forest-labs/flux.2-klein-4b     ~$0.015 good                              <- fallback

Slower or dearer models (MAI 2.6 $0.029, Gemini 2.5 Flash Image $0.039, FLUX.2 Pro
$0.045) looked no better. A few calls hang for minutes, hence the short timeout and
the fallback order. Override the order with ``AI_IMAGE_MODELS`` (comma list).

Keys: image generation needs CREDITS. The DEEPSEEK_API_KEY slot is the paid
OpenRouter account; the two OPENROUTER_API_KEY entries are free-tier and answer
"insufficient credits", so the paid slots are tried first.

Spend is capped per video (``AI_IMAGES_MAX_PER_VIDEO``, default 6, and
``AI_IMAGES_MAX_USD``, default 0.30). Switch the whole thing off with
``ENABLE_AI_IMAGES=false``.
"""

import base64
import io
import os
import threading
import time

import requests

ENDPOINT = "https://openrouter.ai/api/v1/images"
MODELS_ENDPOINT = "https://openrouter.ai/api/v1/images/models"
CREDITS_ENDPOINT = "https://openrouter.ai/api/v1/credits"
DEFAULT_MODELS = ("microsoft/mai-image-2.6-flash",
                  "recraft/recraft-v4.1-flash",
                  "black-forest-labs/flux.2-klein-4b")

# Appended to every prompt: the cards sit next to real photos, so keep generations
# plain, text-free and logo-free (generated lettering and logos are always wrong).
STYLES = {
    "photo": ", photorealistic photograph, natural lighting, sharp focus, "
             "no text, no watermark, no logos",
    "illustration": ", clean hand-drawn ink and colored-pencil illustration on a plain "
                    "white background, simple line art with soft color accents, "
                    "no text, no logos",
}

_lock = threading.Lock()
_run = {"images": 0, "usd": 0.0, "by_model": {}, "dead_keys": set(), "capped": False}


# ── configuration ────────────────────────────────────────────────────────────

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def keys() -> list:
    """Candidate keys, paid account first, de-duplicated."""
    out = []
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_2",
                 "OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"):
        v = os.getenv(name, "").strip()
        if v and v not in out:
            out.append(v)
    return out


def models() -> list:
    raw = os.getenv("AI_IMAGE_MODELS", "").strip()
    chosen = [m.strip() for m in raw.split(",") if m.strip()] if raw else []
    return chosen or list(DEFAULT_MODELS)


def enabled() -> bool:
    v = os.getenv("ENABLE_AI_IMAGES", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return bool(keys())


def max_images() -> int:
    return max(0, int(_env_float("AI_IMAGES_MAX_PER_VIDEO", 6)))


def max_usd() -> float:
    return max(0.0, _env_float("AI_IMAGES_MAX_USD", 0.30))


def reset_run() -> None:
    """Start a job's spend tally from zero (the pipeline calls this per run)."""
    with _lock:
        _run.update(images=0, usd=0.0, by_model={}, capped=False)
        _run["dead_keys"] = set()


def run_stats() -> dict:
    with _lock:
        return {"images": _run["images"], "usd": round(_run["usd"], 4),
                "by_model": dict(_run["by_model"]), "capped": _run["capped"]}


def budget_left() -> bool:
    with _lock:
        return _run["images"] < max_images() and _run["usd"] < max_usd()


# ── transport ────────────────────────────────────────────────────────────────

def _post(url: str, headers: dict, body: dict, timeout: float):
    """``(status, json-or-None)``. A single seam so tests (and a flaky-VPN dev box,
    where curl works and requests doesn't) can swap the transport."""
    r = requests.post(url, json=body, headers=headers, timeout=timeout)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, None


def _save_jpeg(b64: str, out_path: str) -> tuple:
    from PIL import Image
    raw = base64.b64decode(b64)
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    im.save(out_path, "JPEG", quality=92)
    return im.size


def build_prompt(prompt: str, style: str = "photo") -> str:
    base = " ".join((prompt or "").split()).rstrip(" .,")[:700]
    return base + STYLES.get(style, STYLES["photo"])


# ── generation ───────────────────────────────────────────────────────────────

def _attempt(model: str, key: str, full_prompt: str, aspect: str, out_path: str,
             timeout: float) -> tuple:
    """One model on one key. Returns ``("ok", {width, height, cost})``,
    ``("key", reason)`` when this key is unusable (no credits / rejected), or
    ``("model", reason)`` when this model should be skipped (hung, refused the
    prompt, returned junk)."""
    body = {"model": model, "prompt": full_prompt, "aspect_ratio": aspect, "n": 1}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(2):
        try:
            status, data = _post(ENDPOINT, headers, body, timeout)
        except Exception as e:                      # timeout / connection error
            return "model", f"{model}: {type(e).__name__}"
        if status == 429 and attempt == 0:
            time.sleep(2.0)
            continue
        if status in (401, 402, 403):
            return "key", f"{model}: HTTP {status}"
        if status != 200 or not isinstance(data, dict) or not data.get("data"):
            return "model", f"{model}: HTTP {status}"   # moderation / bad params
        try:
            w, h = _save_jpeg(data["data"][0]["b64_json"], out_path)
        except Exception as e:
            return "model", f"{model}: unreadable image ({type(e).__name__})"
        cost = float((data.get("usage") or {}).get("cost") or 0.0)
        return "ok", {"width": w, "height": h, "cost": cost}
    return "model", f"{model}: HTTP 429"


def generate(prompt: str, out_path: str, *, aspect: str = "4:3", style: str = "photo",
             errors: list = None) -> dict | None:
    """Generate one image into ``out_path`` (a ``.jpg``). Returns
    ``{path, model, cost, width, height, prompt, cached}`` or None when disabled,
    over budget, or every model/key failed (the reason goes into ``errors``).

    An existing non-empty ``out_path`` is returned as is (cost 0), so a re-plan of
    the same moment never pays twice."""
    if errors is None:
        errors = []
    if out_path and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            from PIL import Image
            with Image.open(out_path) as im:
                w, h = im.size
        except Exception:
            w = h = 0
        return {"path": out_path, "model": "", "cost": 0.0, "width": w, "height": h,
                "prompt": prompt, "cached": True}
    if not enabled():
        return None
    if not budget_left():
        with _lock:
            first = not _run["capped"]
            _run["capped"] = True
        if first:
            errors.append(f"ai images: budget reached ({max_images()} images / ${max_usd():.2f}) "
                          "— remaining cards use real photos only")
        return None

    full = build_prompt(prompt, style)
    timeout = _env_float("AI_IMAGE_TIMEOUT", 100.0)
    last = ""
    for model in models():
        for key in keys():
            with _lock:
                if key in _run["dead_keys"]:
                    continue
            kind, info = _attempt(model, key, full, aspect, out_path, timeout)
            if kind == "ok":
                with _lock:
                    _run["images"] += 1
                    _run["usd"] += info["cost"]
                    _run["by_model"][model] = _run["by_model"].get(model, 0) + 1
                try:
                    from core import usage
                    usage.record_llm("openrouter-images", model, cost=info["cost"])
                except Exception:
                    pass
                return {"path": out_path, "model": model, "cost": info["cost"],
                        "width": info["width"], "height": info["height"],
                        "prompt": prompt, "cached": False}
            last = info
            if kind == "key":
                with _lock:
                    _run["dead_keys"].add(key)      # no credits / rejected: next key
                continue
            break                                   # next model
    errors.append(f"ai images: generation failed ({last or 'no usable key'})")
    return None


# ── health ───────────────────────────────────────────────────────────────────

def health() -> tuple:
    """``(ok, message)`` for the /test self-check: a key with credits is present and
    the first configured model is still listed. No image is generated (that costs)."""
    ks = keys()
    if not ks:
        return False, "no OpenRouter key set"
    credit = None
    for k in ks:
        try:
            r = requests.get(CREDITS_ENDPOINT, headers={"Authorization": f"Bearer {k}"}, timeout=15)
            d = (r.json() or {}).get("data") or {}
            left = float(d.get("total_credits") or 0) - float(d.get("total_usage") or 0)
            if left > 0.05:
                credit = left
                break
        except Exception:
            continue
    if credit is None:
        return False, "no key with credits (the OPENROUTER_API_KEY entries are free-tier)"
    try:
        listed = {m.get("id") for m in (requests.get(MODELS_ENDPOINT, timeout=15).json() or {}).get("data", [])}
    except Exception:
        listed = set()
    first = models()[0]
    if listed and first not in listed:
        return False, f"model {first} is no longer listed — set AI_IMAGE_MODELS"
    return True, f"${credit:.2f} credit left, model {first}"
