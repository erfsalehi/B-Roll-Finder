"""Free, deterministic picture-geometry check for downloaded clips: black bars.

A clip that sits inside a black frame (a 4:3 upload pillarboxed inside 16:9, a 2.39:1 film
letterboxed, a vertical Short, a picture inset in the middle of a black canvas) looks like a
mistake on the timeline. Seen in the first real output: three of 32 shots (a Dodge in a
driveway with two-thirds of the frame black, two shots with black sides).

Two things shrink the picture, and both are measured here without any model call:

  * the file's own shape against the 1920x1080 sequence it will be placed in
    (Premiere fits a 4:3 or portrait file with black bars);
  * black bars baked into the picture. A bar is a border row/column that is dark AND flat
    (a real dark scene has noise or texture), black in every sampled frame, so a night shot
    or a fade is not mistaken for one.

``problem(path)`` answers "why can't this clip be used" ('' / None when it can). Everything
fails open: no ffmpeg, an unreadable file or an inconclusive (blank) sample says nothing.
Switch off with ``ENABLE_BAR_CHECK=false``; the smallest acceptable share of the frame the
picture may fill is ``CLIP_MIN_PICTURE`` (default 0.80).
"""

import os
import subprocess

import numpy as np

SEQ_W, SEQ_H = 1920, 1080          # the sequence the XML builds
_GW, _GH = 320, 180                # analysis size
_BAR_MEAN = 28                     # a bar row is darker than this ...
_BAR_STD = 4                       # ... and flatter than this (limited-range black is ~16)
_TINY = 2                          # border rows/cols ignored (encoder edges), in analysis px


def enabled() -> bool:
    return os.getenv("ENABLE_BAR_CHECK", "true").strip().lower() not in ("0", "false", "no", "off")


def min_picture() -> float:
    try:
        return float(os.getenv("CLIP_MIN_PICTURE", "").strip() or 0.80)
    except ValueError:
        return 0.80


def _gray_frame(path: str, t: float):
    cmd = ["ffmpeg", "-v", "error", "-threads", "1", "-ss", f"{max(0.0, t):.2f}", "-i", path,
           "-frames:v", "1", "-vf", f"scale={_GW}:{_GH},format=gray", "-f", "rawvideo", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or len(r.stdout) < _GW * _GH:
        return None
    return np.frombuffer(r.stdout[:_GW * _GH], dtype=np.uint8).reshape(_GH, _GW)


def _flat_dark(g: np.ndarray) -> np.ndarray:
    """Per row: dark and flat enough to be a bar."""
    return (g.mean(axis=1) < _BAR_MEAN) & (g.std(axis=1) < _BAR_STD)


def _run(flags) -> int:
    n = 0
    for f in flags:
        if not f:
            break
        n += 1
    return n


def bars(frames: list):
    """``(top, bottom, left, right)`` as fractions of the frame, or None when the sample is
    inconclusive (blank / faded frames)."""
    if not frames:
        return None
    rows = np.logical_and.reduce([_flat_dark(g) for g in frames])
    cols = np.logical_and.reduce([_flat_dark(g.T) for g in frames])
    t, b = _run(rows), _run(rows[::-1])
    l, r = _run(cols), _run(cols[::-1])
    if t + b >= 0.9 * _GH or l + r >= 0.9 * _GW:
        return None                                  # all dark: a fade, not a frame with bars
    t, b = (t if t > _TINY else 0), (b if b > _TINY else 0)
    l, r = (l if l > _TINY else 0), (r if r > _TINY else 0)
    return t / _GH, b / _GH, l / _GW, r / _GW


def native_fraction(width: int, height: int) -> float:
    """Share of the 1920x1080 frame a ``width`` x ``height`` file covers when fitted."""
    if width <= 0 or height <= 0:
        return 1.0
    s = min(SEQ_W / width, SEQ_H / height)
    return min(1.0, (width * s) * (height * s) / (SEQ_W * SEQ_H))


def analyse(path: str) -> dict | None:
    """``{width, height, native, inner, picture, bars}`` or None when it can't be measured."""
    if not path or not os.path.exists(path):
        return None
    from core.ffmpeg_utils import get_video_metadata
    meta = get_video_metadata(path)
    w, h, dur = int(meta.get("width") or 0), int(meta.get("height") or 0), float(meta.get("duration") or 0)
    if w <= 0 or h <= 0 or dur <= 0 or dur >= 3600.0:     # 3600 = ffprobe's "unknown" default
        return None
    marks = [0.5] if dur < 3 else [0.2, 0.45, 0.7]
    frames = [g for g in (_gray_frame(path, min(dur * m, max(0.0, dur - 0.5))) for m in marks)
              if g is not None]
    found = bars(frames)
    if found is None:
        return None
    t, b, l, r = found
    inner = max(0.0, (1 - l - r) * (1 - t - b))
    native = native_fraction(w, h)
    return {"width": w, "height": h, "native": round(native, 3), "inner": round(inner, 3),
            "picture": round(native * inner, 3),
            "bars": {"top": round(t, 3), "bottom": round(b, 3), "left": round(l, 3), "right": round(r, 3)}}


def problem(path: str) -> str | None:
    """Why the clip's picture is too small for the frame, or None when it is fine or unknown."""
    if not enabled():
        return None
    try:
        a = analyse(path)
    except Exception:
        return None
    if not a or a["picture"] >= min_picture():
        return None
    why = "portrait" if a["height"] > a["width"] else "black bars"
    return (f"{why}: the picture fills only {a['picture'] * 100:.0f}% of the frame "
            f"({a['width']}x{a['height']} file)")
