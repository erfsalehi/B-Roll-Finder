"""Reviewer rating page, served by the bot's file server under ``/rate``.

Reviewers open a personal link from ``/rate`` in Telegram:

    <base>/rate?r=<telegram user id>&t=<HMAC token>

The token is ``sign_token("rate:<id>", 0)`` from :mod:`bot.fileserver`, so links
don't expire but can't be forged for another id. Every API call re-checks that
the id is still on ``TELEGRAM_RATERS`` (or ``TELEGRAM_ALLOWED_USERS``), so
removing someone from the list revokes their link.

Routes:
    GET  /rate                    the page (tabs: Rate clips · Check editor labels · My impact)
    GET  /rate/api/meta           reason chips + queue counts
    GET  /rate/api/projects       projects with something to review + progress
    GET  /rate/api/next           next shot to rate (?skip=pid:slot,…&project=<id>)
    POST /rate/api/submit         {project_id, slot_id, ratings: [...], suggestions: [...]}
    GET  /rate/api/labels/next    next shot of editor-XML labels to check
    POST /rate/api/labels/submit  {reviews: [{asset_id, action, verdict, used_slot_id, note}]}
    GET  /rate/api/impact         the reviewer's contribution log
    GET  /rate/api/library/next   next segment-library clip to describe (?skip=id,id)
    POST /rate/api/library/submit {segment_id, usable, fits_line, description, subject,
                                   identifiable, generic_use, shot_type, problems, note}
    GET  /rate/media/segment/<id>.mp4   the stored segment (auth in the query)
"""

import json
import os
import re
import urllib.parse

from bot import fileserver

_MAX_BODY = 256 * 1024


def _ids(var: str) -> set:
    out = set()
    for part in os.getenv(var, "").replace(";", ",").split(","):
        part = part.strip()
        if part.lstrip("-").isdigit():
            out.add(int(part))
    return out


def rater_ids() -> set:
    """Everyone who may rate: dedicated reviewers plus the bot's operators."""
    return _ids("TELEGRAM_RATERS") | _ids("TELEGRAM_ALLOWED_USERS")


def rate_token(rater_id: int) -> str:
    return fileserver.sign_token(f"rate:{int(rater_id)}", fileserver.NEVER)


def rate_link(rater_id: int, base: str) -> str:
    q = urllib.parse.urlencode({"r": int(rater_id), "t": rate_token(rater_id)})
    return f"{base.rstrip('/')}/rate?{q}"


def _auth(qs: dict):
    """The rater id if the request carries a valid token for a current rater."""
    try:
        rid = int((qs.get("r") or [""])[0])
    except ValueError:
        return None
    token = (qs.get("t") or [""])[0]
    if not fileserver.verify_token(f"rate:{rid}", fileserver.NEVER, token):
        return None
    return rid if rid in rater_ids() else None


def _send(handler, code: int, body, ctype: str = "application/json") -> None:
    data = body if isinstance(body, bytes) else (
        json.dumps(body).encode("utf-8") if ctype == "application/json" else body.encode("utf-8"))
    handler.send_response(code)
    handler.send_header("Content-Type", f"{ctype}; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(data)


def _send_file(handler, path: str, ctype: str) -> None:
    """Serve a file with Range support (video seeking needs it)."""
    size = os.path.getsize(path)
    start, end, partial = 0, size - 1, False
    rng = handler.headers.get("Range") or ""
    if rng.startswith("bytes="):
        try:
            a, _, b = rng[6:].split(",")[0].partition("-")
            if a:
                start, end = int(a), (int(b) if b else size - 1)
            else:
                start = max(0, size - int(b))
            end = min(end, size - 1)
            partial = start <= end
        except ValueError:
            start, end = 0, size - 1
    if start > end:
        handler.send_response(416)
        handler.send_header("Content-Range", f"bytes */{size}")
        handler.end_headers()
        return
    handler.send_response(206 if partial else 200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(end - start + 1))
    handler.send_header("Cache-Control", "private, max-age=3600")
    if partial:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.end_headers()
    if handler.command == "HEAD":
        return
    with open(path, "rb") as f:
        f.seek(start)
        left = end - start + 1
        while left > 0:
            chunk = f.read(min(1 << 16, left))
            if not chunk:
                break
            try:
                handler.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                break
            left -= len(chunk)


def handle(handler) -> bool:
    """Serve a /rate request on ``handler`` (a BaseHTTPRequestHandler). Returns
    False when the path isn't ours."""
    parsed = urllib.parse.urlparse(handler.path)
    path = parsed.path.rstrip("/")
    if path != "/rate" and not path.startswith("/rate/"):
        return False
    qs = urllib.parse.parse_qs(parsed.query)

    if path == "/rate" and handler.command in ("GET", "HEAD"):
        _send(handler, 200, PAGE_HTML, "text/html")
        return True

    rid = _auth(qs)
    if rid is None:
        _send(handler, 403, {"error": "This link isn't valid (or you're no longer a reviewer). "
                                      "Ask the bot for a new one with /rate."})
        return True

    from core import ratings, segment_library
    m = re.fullmatch(r"/rate/media/segment/(\d+)\.mp4", path)
    if m and handler.command in ("GET", "HEAD"):
        seg = segment_library.get_segment(int(m.group(1)))
        if not seg or not os.path.isfile(seg.get("file_path") or ""):
            _send(handler, 404, {"error": "not found"})
        else:
            _send_file(handler, seg["file_path"], "video/mp4")
        return True
    try:
        if path == "/rate/api/library/next" and handler.command == "GET":
            skip = [int(x) for x in (qs.get("skip") or [""])[0].split(",") if x.isdigit()]
            _send(handler, 200, segment_library.next_review_task(rid, skip=skip) or {"done": True})
            return True

        if path == "/rate/api/meta" and handler.command == "GET":
            mine = next((s for s in ratings.rater_stats() if s["rater_id"] == rid), None)
            _send(handler, 200, {"reasons": [{"id": i, "label": l} for i, l in ratings.REASONS],
                                 "rated": mine["rated"] if mine else 0,
                                 "queue": ratings.queue_size(),
                                 "library": segment_library.stats()})
            return True

        if path in ("/rate/api/next", "/rate/api/labels/next") and handler.command == "GET":
            skip = []
            for tok in (qs.get("skip") or [""])[0].split(","):
                p, _, s = tok.partition(":")
                if p.isdigit() and s:
                    skip.append((int(p), s))
            proj = (qs.get("project") or [""])[0]
            fn = ratings.next_label_task if "labels" in path else ratings.next_task
            _send(handler, 200, fn(rid, skip=skip, project_id=int(proj) if proj.isdigit()
                                   else None) or {"done": True})
            return True

        if path == "/rate/api/projects" and handler.command == "GET":
            _send(handler, 200, {"projects": ratings.project_progress(rid)})
            return True

        if path == "/rate/api/impact" and handler.command == "GET":
            _send(handler, 200, ratings.contribution_summary(rid))
            return True

        if path in ("/rate/api/submit", "/rate/api/labels/submit",
                    "/rate/api/library/submit") and handler.command == "POST":
            n = int(handler.headers.get("Content-Length") or 0)
            if n <= 0 or n > _MAX_BODY:
                _send(handler, 413, {"error": "request too large"})
                return True
            body = json.loads(handler.rfile.read(n).decode("utf-8"))
            saved = 0
            if "library" in path:
                segment_library.save_review(
                    rid, int(body["segment_id"]), usable=body.get("usable"),
                    fits_line=body.get("fits_line") or "", description=body.get("description", ""),
                    subject=body.get("subject", ""), identifiable=body.get("identifiable"),
                    generic_use=body.get("generic_use", ""), shot_type=body.get("shot_type", ""),
                    problems=body.get("problems"), note=body.get("note", ""))
                _send(handler, 200, {"ok": True, "saved": 1})
                return True
            if "labels" in path:
                for r in body.get("reviews") or []:
                    ratings.save_label_review(rid, int(r["asset_id"]), r.get("action"),
                                              r.get("verdict"), r.get("used_slot_id") or "",
                                              r.get("note", ""))
                    saved += 1
                _send(handler, 200, {"ok": True, "saved": saved})
                return True
            for r in body.get("ratings") or []:
                ratings.save_rating(rid, int(r["item_id"]), r.get("usable"), r.get("fit"),
                                    r.get("reasons"), r.get("note", ""), r.get("segments"))
                saved += 1
            for s in body.get("suggestions") or []:
                ratings.add_suggestion(rid, int(body["project_id"]), body["slot_id"],
                                       s.get("url", ""), s.get("note", ""), s.get("segments"))
                saved += 1
            ratings.maybe_distill_async()
            _send(handler, 200, {"ok": True, "saved": saved})
            return True
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        _send(handler, 400, {"error": str(e)})
        return True
    except Exception as e:
        print(f"[rating_page] {path}: {e}")
        _send(handler, 500, {"error": "server error"})
        return True

    _send(handler, 404, {"error": "not found"})
    return True


PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>B-roll Review</title>
<style>
:root {
  --bg: #f6f5f2; --surface: #ffffff; --ink: #1d1d1b; --muted: #6b6a66;
  --line: #e2e0da; --accent: #2f5bd3; --accent-ink: #ffffff;
  --yes: #1f7a4d; --trim: #9a6a00; --no: #b3261e; --chip: #efede8;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #151514; --surface: #1f1f1d; --ink: #ecebe7; --muted: #a09f99;
    --line: #34332f; --accent: #7d9cff; --accent-ink: #10131c;
    --yes: #5cc491; --trim: #e0b44c; --no: #f07a70; --chip: #2a2a27;
  }
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body { margin: 0; background: var(--bg); color: var(--ink);
       font: 15px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--bg);
         border-bottom: 1px solid var(--line); }
.bar { max-width: 880px; margin: 0 auto; padding: 10px 16px 0; display: flex;
       align-items: center; gap: 12px; flex-wrap: wrap; }
.bar h1 { font-size: 16px; margin: 0; flex: 1; }
.stat { color: var(--muted); font-size: 13px; }
.tabs { max-width: 880px; margin: 0 auto; padding: 0 16px; display: flex; gap: 4px;
        overflow-x: auto; }
.tab { border: 0; border-bottom: 2px solid transparent; border-radius: 0; background: none;
       color: var(--muted); padding: 10px 10px; white-space: nowrap; }
.tab[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--accent); font-weight: 600; }
main { max-width: 880px; margin: 0 auto; padding: 16px; }
.guide { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
         margin-bottom: 16px; }
.guide summary { cursor: pointer; padding: 12px 14px; font-weight: 600; }
.guide .g { padding: 0 14px 12px; }
.guide ol, .guide ul { margin: 6px 0 10px; padding-left: 20px; }
.guide li { margin: 4px 0; }
.ex { font-size: 13px; border-radius: 8px; padding: 8px 10px; margin: 6px 0; }
.ex.good { background: color-mix(in srgb, var(--yes) 14%, transparent); }
.ex.bad { background: color-mix(in srgb, var(--no) 14%, transparent); }
.hint { font-size: 13px; color: var(--trim); }
.draft { font-size: 12px; color: var(--muted); }
.field { display: grid; gap: 4px; }
.field > span { font-size: 13px; color: var(--muted); }
.picker { display: flex; gap: 8px; align-items: center; margin-bottom: 16px; }
.picker label { color: var(--muted); font-size: 13px; white-space: nowrap; }
.picker select { flex: 1; min-width: 0; }
.shot { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
        padding: 16px; margin-bottom: 16px; }
.eyebrow { color: var(--muted); font-size: 13px; margin-bottom: 6px; }
.line { font-size: 18px; margin: 0; }
.clip { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
        margin-bottom: 16px; overflow: hidden; }
.clip.missing { border-color: var(--no); box-shadow: 0 0 0 1px var(--no); }
.player { position: relative; aspect-ratio: 16 / 9; background: #000; }
.player > iframe, .player > video, .player > div, .player > img { position: absolute; inset: 0;
        width: 100%; height: 100%; border: 0; }
.player > img { object-fit: contain; }
.noplay { display: flex; align-items: center; justify-content: center; color: #bbb;
          padding: 16px; text-align: center; }
.body { padding: 12px 14px 14px; display: grid; gap: 10px; align-content: start; }
.title { font-weight: 600; overflow-wrap: anywhere; }
.title small { font-weight: 400; color: var(--muted); }
.title a { color: var(--accent); font-weight: 400; font-size: 13px; margin-left: 6px; }
.row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.label { color: var(--muted); font-size: 13px; min-width: 64px; }
button { font: inherit; cursor: pointer; border-radius: 8px; border: 1px solid var(--line);
         background: var(--surface); color: var(--ink); padding: 6px 12px; min-height: 36px; }
button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px; }
.seg button[aria-pressed="true"].yes { background: var(--yes); border-color: var(--yes); color: #fff; }
.seg button[aria-pressed="true"].trim { background: var(--trim); border-color: var(--trim); color: #fff; }
.seg button[aria-pressed="true"].no { background: var(--no); border-color: var(--no); color: #fff; }
.fit button { width: 40px; padding: 6px 0; }
.fit button[aria-pressed="true"], .pick button[aria-pressed="true"] {
  background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }
.chip { background: var(--chip); border-color: transparent; font-size: 13px;
        min-height: 30px; padding: 4px 10px; border-radius: 999px; }
.chip[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); }
textarea, input[type=text], input[type=url], select { width: 100%; font: inherit; color: var(--ink);
  background: var(--bg); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
textarea { min-height: 44px; resize: vertical; }
.segs { display: flex; flex-wrap: wrap; gap: 6px; }
.segpill { background: var(--chip); border-radius: 999px; padding: 3px 4px 3px 10px;
           font-size: 13px; display: inline-flex; align-items: center; gap: 4px; }
.segpill button { min-height: 24px; padding: 0 8px; border: 0; background: transparent; }
.pending { color: var(--trim); font-size: 13px; }
.badge { display: inline-block; font-size: 13px; padding: 2px 10px; border-radius: 999px;
         background: var(--chip); }
.badge.used_here { color: var(--yes); } .badge.used_elsewhere { color: var(--trim); }
.badge.unused { color: var(--no); }
.actions { position: sticky; bottom: 0; background: var(--bg); border-top: 1px solid var(--line);
           padding: 10px 16px; }
.actions .inner { max-width: 880px; margin: 0 auto; display: flex; gap: 8px;
                  justify-content: flex-end; align-items: center; }
.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-ink);
           font-weight: 600; }
.msg { flex: 1; color: var(--muted); font-size: 13px; }
.msg.err { color: var(--no); }
h2 { font-size: 15px; margin: 24px 0 8px; }
.empty { text-align: center; padding: 60px 16px; color: var(--muted); }
.sugg-list .clip { margin-bottom: 8px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px; }
.tile b { display: block; font-size: 26px; font-variant-numeric: tabular-nums; }
.tile span { color: var(--muted); font-size: 13px; }
.list { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }
.list > div { display: flex; gap: 12px; padding: 10px 14px; border-top: 1px solid var(--line); }
.list > div:first-child { border-top: 0; }
.list .n { font-variant-numeric: tabular-nums; font-weight: 600; min-width: 36px; text-align: right; }
.list .when { color: var(--muted); font-size: 13px; white-space: nowrap; margin-left: auto; }
@media (min-width: 900px) {
  .bar, .tabs, main, .actions .inner { max-width: 1120px; }
  .clip { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr);
          align-items: start; }
  .clip .player { align-self: stretch; aspect-ratio: auto; min-height: 280px; }
}
</style>
</head>
<body>
<header>
  <div class="bar"><h1>B-roll review</h1><span class="stat" id="stat"></span></div>
  <nav class="tabs" role="tablist">
    <button class="tab" role="tab" data-mode="rate" aria-selected="true">Rate clips</button>
    <button class="tab" role="tab" data-mode="labels" aria-selected="false">Check editor labels</button>
    <button class="tab" role="tab" data-mode="library" aria-selected="false">Library clips</button>
    <button class="tab" role="tab" data-mode="impact" aria-selected="false">My impact</button>
  </nav>
</header>
<main id="main"><div class="empty">Loading…</div></main>
<div class="actions" id="actions" hidden><div class="inner">
  <span class="msg" id="msg"></span>
  <button id="skip" type="button">Skip shot</button>
  <button id="save" type="button" class="primary">Save &amp; next</button>
</div></div>
<script>
const Q = new URLSearchParams(location.search);
const AUTH = "r=" + encodeURIComponent(Q.get("r") || "") + "&t=" + encodeURIComponent(Q.get("t") || "");
const skipped = { rate: [], labels: [], library: [] };
let mode = "rate", REASONS = [], task = null, cards = [], suggestions = [], players = {}, ytReady = null;
let project = "", PROJECTS = [];

function api(path, opts) {
  const sep = path.includes("?") ? "&" : "?";
  return fetch("/rate/api/" + path + sep + AUTH, opts).then(async r => {
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
    return j;
  });
}
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "on") for (const [ev, fn] of Object.entries(v)) e.addEventListener(ev, fn);
    else if (v !== null && v !== undefined && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined) e.append(k.nodeType ? k : String(k));
  return e;
}
const fmt = s => { s = Math.max(0, Math.round(s)); return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); };
function parseTime(s) {
  s = (s || "").trim(); if (!s) return NaN;
  return s.split(":").reduce((a, p) => a * 60 + parseFloat(p), 0);
}
function ytId(url) {
  const m = /(?:youtu\.be\/|youtube\.com\/(?:watch\?(?:.*&)?v=|embed\/|shorts\/|live\/))([A-Za-z0-9_-]{11})/.exec(url || "");
  return m ? m[1] : "";
}
function loadYT() {
  if (ytReady) return ytReady;
  ytReady = new Promise(res => {
    window.onYouTubeIframeAPIReady = res;
    document.head.append(el("script", { src: "https://www.youtube.com/iframe_api" }));
  });
  return ytReady;
}
function setMsg(t, err) { const m = document.getElementById("msg"); m.textContent = t; m.className = "msg" + (err ? " err" : ""); }

// A player (or image) + a way to read its current time.
function makePlayer(key, item) {
  const box = el("div", { class: "player" });
  if (item.kind === "image") {
    box.append(el("img", { src: item.url, alt: item.title || "Image", loading: "lazy", referrerpolicy: "no-referrer" }));
    return { box, time: () => NaN };
  }
  const vid = item.youtube_id || ytId(item.url || item.page_url);
  if (vid) {
    const slot = el("div", { id: "yt-" + key });
    box.append(slot);
    loadYT().then(() => {
      players[key] = new YT.Player(slot.id, { videoId: vid,
        playerVars: { start: Math.floor(item.start || 0), rel: 0, modestbranding: 1, playsinline: 1 } });
    });
    return { box, time: () => (players[key] && players[key].getCurrentTime ? players[key].getCurrentTime() : 0) };
  }
  if (/^https?:\/\//.test(item.url || "") && !/pexels\.com\/video\//.test(item.url)) {
    const v = el("video", { src: item.url + (item.start ? "#t=" + item.start : ""), controls: true,
                            preload: "metadata", playsinline: true });
    box.append(v);
    return { box, time: () => v.currentTime || 0 };
  }
  box.append(el("div", { class: "noplay" }, "No preview — open the source link to watch."));
  return { box, time: () => NaN };
}

function segmentEditor(state, getTime) {
  const list = el("div", { class: "segs" });
  const pending = el("span", { class: "pending" });
  const manual = el("input", { type: "text", placeholder: "or type 0:12-0:18", "aria-label": "Add good part as start-end", style: "max-width:170px" });
  let markIn = null;
  function draw() {
    list.replaceChildren(...state.segments.map((s, i) => el("span", { class: "segpill" },
      fmt(s[0]) + "–" + fmt(s[1]),
      el("button", { type: "button", "aria-label": "Remove good part", on: { click: () => { state.segments.splice(i, 1); draw(); } } }, "×"))));
    pending.textContent = markIn === null ? "" : "in at " + fmt(markIn) + " — press Mark out";
  }
  function add(a, b) {
    if (isNaN(a) || isNaN(b) || b <= a) return false;
    state.segments.push([a, b]); state.segments.sort((x, y) => x[0] - y[0]); draw(); return true;
  }
  const inBtn = el("button", { type: "button", on: { click: () => { const t = getTime(); if (!isNaN(t)) { markIn = t; draw(); } } } }, "Mark in");
  const outBtn = el("button", { type: "button", on: { click: () => { const t = getTime(); if (markIn !== null && add(markIn, t)) { markIn = null; draw(); } } } }, "Mark out");
  manual.addEventListener("keydown", e => {
    if (e.key !== "Enter") return;
    const [a, b] = manual.value.split("-").map(parseTime);
    if (add(a, b)) manual.value = "";
  });
  draw();
  return el("div", {}, el("div", { class: "row" }, el("span", { class: "label" }, "Good parts"), inBtn, outBtn, manual, pending), list);
}

function toggleGroup(values, current, onPick, cls) {
  const wrap = el("div", { class: "row " + (cls || "") });
  const btns = values.map(([val, label, c]) => el("button", { type: "button", class: c || "", "aria-pressed": String(current() === val),
    on: { click: () => { onPick(current() === val ? null : val); btns.forEach((x, i) => x.setAttribute("aria-pressed", String(current() === values[i][0]))); } } }, label));
  wrap.append(...btns);
  return wrap;
}

function titleRow(item, extra) {
  const src = (item.source || "?").toUpperCase();
  return el("div", { class: "title" }, item.title || "Untitled", " ",
    el("small", {}, "· " + src + (item.channel ? " · " + item.channel : "") + (extra || "")),
    item.page_url ? el("a", { href: item.page_url, target: "_blank", rel: "noopener" }, "source ↗") : null);
}

function shotHeader() {
  return el("div", { class: "shot" },
    el("div", { class: "eyebrow" }, (task.project || "Project") + (task.topic ? " · " + task.topic : "") + " · " + fmt(task.start) + "–" + fmt(task.end)),
    el("p", { class: "line" }, "“" + (task.text || "") + "”"));
}

// ── Rate clips ──────────────────────────────────────────────────────────────
function rateCard(item, idx) {
  const mine = item.mine || {};
  const st = { item_id: item.id, usable: mine.usable || null, fit: mine.fit || null,
               reasons: new Set(mine.reasons || []), note: mine.note || "", segments: (mine.segments || []).slice() };
  const pl = makePlayer("c" + item.id, item);
  const card = el("section", { class: "clip", "aria-label": "Clip " + (idx + 1) }, pl.box,
    el("div", { class: "body" },
      titleRow(item, item.suggested ? " · suggested by a reviewer" : ""),
      el("div", { class: "row" }, el("span", { class: "label" }, "Usable?"),
        toggleGroup([["yes", "Yes", "yes"], ["trim", "With a trim", "trim"], ["no", "No", "no"]],
                    () => st.usable, v => { st.usable = v; card.classList.remove("missing"); }, "seg")),
      el("div", { class: "row" }, el("span", { class: "label" }, "Fit"),
        toggleGroup([1, 2, 3, 4, 5].map(n => [n, String(n)]), () => st.fit, v => st.fit = v, "fit")),
      el("div", { class: "row" }, el("span", { class: "label" }, "Why"),
        ...REASONS.map(r => el("button", { type: "button", class: "chip", "aria-pressed": String(st.reasons.has(r.id)),
          on: { click: e => { st.reasons.has(r.id) ? st.reasons.delete(r.id) : st.reasons.add(r.id);
                               e.currentTarget.setAttribute("aria-pressed", String(st.reasons.has(r.id))); } } }, r.label))),
      (() => { const nh = noteHint(() => st.note);
        return el("div", { class: "field" }, el("textarea", { placeholder: "What do you see, and why does it fit or not? e.g. “Toyota badge visible — Camry, but the line is about a Civic.”",
          "aria-label": "Note", on: { input: e => { st.note = e.target.value; nh.upd(); } } }, st.note), nh.h); })(),
      segmentEditor(st, pl.time)));
  st.card = card;
  st.ok = () => !!st.usable;
  cards.push(st);
  return card;
}

function suggestBox() {
  const list = el("div", { class: "sugg-list" });
  const url = el("input", { type: "url", placeholder: "Paste a YouTube (or other) link that fits this line better", "aria-label": "Suggested clip link" });
  const area = el("div");
  function preview() {
    const u = url.value.trim(); if (!/^https?:\/\//.test(u)) return;
    const pl = makePlayer("s" + Date.now(), { url: u, page_url: u, start: 0 });
    const cur = { url: u, note: "", segments: [] };
    area.replaceChildren(el("div", { class: "clip" }, pl.box, el("div", { class: "body" },
      el("div", { class: "title" }, u), segmentEditor(cur, pl.time),
      el("textarea", { placeholder: "Why this one? (optional)", "aria-label": "Suggestion note", on: { input: e => cur.note = e.target.value } }),
      el("div", { class: "row" }, el("button", { type: "button", class: "primary", on: { click: () => {
        suggestions.push(cur);
        list.append(el("div", { class: "stat" }, "✓ Suggested: " + cur.url + (cur.segments.length ? " (" + cur.segments.length + " good part" + (cur.segments.length > 1 ? "s" : "") + ")" : "")));
        area.replaceChildren(); url.value = "";
      } } }, "Add suggestion")))));
  }
  url.addEventListener("change", preview);
  url.addEventListener("keydown", e => { if (e.key === "Enter") preview(); });
  return el("div", {}, el("h2", {}, "Know a better clip for this line?"),
    el("div", { class: "row" }, url, el("button", { type: "button", on: { click: preview } }, "Preview")), area, list);
}

// ── Check editor labels ─────────────────────────────────────────────────────
const AUTO = { used_here: "Editor used it on this line", used_elsewhere: "Editor moved it to another line",
               unused: "Editor didn't use it" };
const FIX = [["used_here", "Right for this line"], ["used_elsewhere", "Right for another line"],
             ["neutral", "Fine, just not needed"], ["bad", "Not usable"]];

function labelCard(item, idx) {
  const mine = item.mine || {};
  const st = { asset_id: item.asset_id, action: mine.action || null, verdict: mine.action === "correct" ? mine.verdict : null,
               used_slot_id: mine.used_slot_id || "", note: mine.note || "" };
  const pl = makePlayer("l" + item.asset_id, item);
  const used = item.in_sec != null && item.kind !== "image" ? "Editor used " + fmt(item.in_sec) + "–" + fmt(item.out_sec) + " of the source" : "";
  const where = item.verdict === "used_elsewhere" && item.used_line ? el("div", { class: "stat" }, "Moved to: “" + item.used_line + "”") : null;
  const lineSel = el("select", { "aria-label": "Which line it belongs to", on: { change: e => st.used_slot_id = e.target.value } },
    el("option", { value: "" }, "Pick the line…"),
    ...task.lines.filter(l => l.slot_id !== task.slot_id).map(l => el("option", { value: l.slot_id, selected: l.slot_id === st.used_slot_id }, l.text)));
  const fixBox = el("div", { class: "body", style: "padding:0", hidden: st.action !== "correct" },
    toggleGroup(FIX, () => st.verdict, v => { st.verdict = v; lineSel.hidden = v !== "used_elsewhere"; }, "pick"),
    lineSel,
    el("textarea", { placeholder: "What's wrong with the label? (optional)", "aria-label": "Note", on: { input: e => st.note = e.target.value } }, st.note));
  lineSel.hidden = st.verdict !== "used_elsewhere";
  const card = el("section", { class: "clip", "aria-label": (item.kind === "image" ? "Image " : "Clip ") + (idx + 1) }, pl.box,
    el("div", { class: "body" },
      titleRow(item, item.kind === "image" ? " · image" : ""),
      el("div", {}, el("span", { class: "badge " + item.verdict }, AUTO[item.verdict] || item.verdict)),
      where, used ? el("div", { class: "stat" }, used) : null,
      el("div", { class: "row" }, el("span", { class: "label" }, "Label is"),
        toggleGroup([["confirm", "Right"], ["correct", "Wrong"]], () => st.action,
          v => { st.action = v; fixBox.hidden = v !== "correct"; card.classList.remove("missing"); }, "pick")),
      fixBox));
  st.card = card;
  st.ok = () => st.action === "confirm" || (st.action === "correct" && st.verdict && (st.verdict !== "used_elsewhere" || st.used_slot_id));
  cards.push(st);
  return card;
}

// ── My impact ───────────────────────────────────────────────────────────────
async function renderImpact() {
  const main = document.getElementById("main");
  document.getElementById("actions").hidden = true;
  main.replaceChildren(el("div", { class: "empty" }, "Loading…"));
  try {
    const s = await api("impact");
    const tile = (n, label) => el("div", { class: "tile" }, el("b", {}, n == null ? "–" : n), el("span", {}, label));
    const pct = v => v == null ? "–" : v + "%";
    const kinds = Object.entries(s.impact_labels);
    main.replaceChildren(
      el("h2", { style: "margin-top:0" }, "Your work"),
      el("div", { class: "tiles" }, tile(s.rated, "clips rated"), tile(s.notes, "notes written"),
        tile(s.segments, "good parts marked"), tile(s.suggested, "clips suggested"),
        tile(s.labels_checked, "editor labels checked")),
      el("h2", {}, "What it changed"),
      el("div", { class: "list" }, ...kinds.map(([k, label]) =>
        el("div", {}, el("span", { class: "n" }, s.impact[k] || 0), el("span", {}, label)))),
      el("h2", {}, "How often you agree"),
      el("div", { class: "tiles" }, tile(pct(s.editor_agreement), "with the real editor"),
        tile(pct(s.peer_agreement), "with other reviewers")),
      el("h2", {}, "Recent"),
      s.log.length ? el("div", { class: "list" }, ...s.log.map(e => el("div", {},
        el("span", {}, e.label + (e.title ? " — " + e.title : "")), el("span", { class: "when" }, e.created_at.replace("T", " ").slice(0, 16)))))
        : el("div", { class: "stat" }, "Nothing yet — your reviews start counting as soon as the bot uses them."));
  } catch (e) { main.replaceChildren(el("div", { class: "empty" }, e.message)); }
}

// ── review guides ───────────────────────────────────────────────────────────
const GUIDES = {
  rate: {
    title: "How to rate well (read once)",
    body: () => [
      el("p", {}, "Judge each clip against this exact line — would it look right on screen while these words are spoken?"),
      el("ol", {},
        el("li", {}, el("b", {}, "Yes"), " = shows what the line is about. ", el("b", {}, "With a trim"), " = only part of it works — mark that part. ", el("b", {}, "No"), " = wrong for this line."),
        el("li", {}, "In the note, say ", el("b", {}, "what you see"), " and ", el("b", {}, "why it fits or not"), ", in one sentence. Name the exact thing: brand, model, part."),
        el("li", {}, "Right action but wrong car/brand? Tick ", el("b", {}, "“Wrong model/brand — OK as general footage”"), ". The clip isn't bad — it's just not for this line."),
        el("li", {}, "Good parts: start when the subject is clearly on screen; stop before a cut, on-screen text or a face.")),
      el("div", { class: "ex good" }, "✅ “Toyota badge visible at 0:03 — it's a Camry, the line is about a Civic. Fine as general oil-change footage.”"),
      el("div", { class: "ex bad" }, "❌ “not relevant” · “good” · “nice clip” — the picker can't learn anything from these.")]
  },
  labels: {
    title: "How to check editor labels",
    body: () => [
      el("p", {}, "The bot read the editor's finished timeline. Check each label:"),
      el("ul", {},
        el("li", {}, el("b", {}, "Right"), " when it's correct."),
        el("li", {}, el("b", {}, "Fine, just not needed"), " — a good clip the editor skipped because they had enough. It won't count against the clip."),
        el("li", {}, el("b", {}, "Not usable"), " — the clip is genuinely bad (wrong subject, logo, low quality)."),
        el("li", {}, el("b", {}, "Right for another line"), " — pick the line where it actually belongs."))]
  },
  library: {
    title: "How to describe a library clip (important — read this)",
    body: () => [
      el("p", {}, "This clip will be reused in future videos. Describe what is ", el("b", {}, "on screen"), " — not what the narration said. A good description is what lets the app put the right clip in the right spot later."),
      el("ol", {},
        el("li", {}, el("b", {}, "What's on screen: "), "subject + action + framing, in one or two plain sentences."),
        el("li", {}, el("b", {}, "Exact subject: "), "the most specific thing you can ", el("i", {}, "see"), " — “Toyota Camry (2018–2022)”, “K&N oil filter”. If the narration names a model you can't see, don't write it."),
        el("li", {}, el("b", {}, "Can a viewer tell exactly what it is? "), "Yes only if a badge, logo, readable text or unmistakable design shows it. ",
          el("b", {}, "Yes"), " → used only for that subject. ", el("b", {}, "No"), " → used as general footage."),
        el("li", {}, el("b", {}, "General use: "), "what it could stand in for in any video — “oil draining from a car engine”, “mechanic working under a car”."),
        el("li", {}, el("b", {}, "Usable? "), "Say No only if it's unusable ", el("i", {}, "anywhere"), " (watermark, face, shaky, blurry). Being wrong for one line is not a reason.")),
      el("div", { class: "ex good" }, "✅ Description: “Close-up of a hand unscrewing the oil drain plug under a silver sedan; dark oil pours into a black pan.” · Subject: “Toyota Camry” · Viewer can tell: No (no badge in shot) · General use: “draining engine oil from a car”."),
      el("div", { class: "ex good" }, "✅ Same action, but the Toyota badge fills the first second → Viewer can tell: Yes · Subject: “Toyota Camry”. It will only be used for Camry lines."),
      el("div", { class: "ex bad" }, "❌ “oil change clip” · “good footage for the engine part” · copying the narration.")]
  },
};
function guide(key) {
  const g = GUIDES[key]; if (!g) return null;
  let open = true;
  try { open = localStorage.getItem("guide-seen-" + key) !== "1"; } catch (e) {}
  const d = el("details", { class: "guide", open: open }, el("summary", {}, g.title), el("div", { class: "g" }, ...g.body()));
  d.addEventListener("toggle", () => { if (!d.open) try { localStorage.setItem("guide-seen-" + key, "1"); } catch (e) {} });
  return d;
}
const VAGUE = new Set(["good","bad","ok","okay","nice","fine","great","relevant","irrelevant","clip","video","footage","shot","yes","no"]);
function textIssues(t) {
  const w = (t || "").toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
  if (w.length < 6) return "Too short — say what's on screen in a full sentence.";
  if (w.filter(x => !VAGUE.has(x)).length < 4) return "Too vague — name the subject and what's happening.";
  return "";
}
function noteHint(getText) {
  const h = el("div", { class: "hint", "aria-live": "polite" });
  const upd = () => { const t = getText(); h.textContent = t && t.trim().length && textIssues(t) ? "Tip: " + textIssues(t) + " Example: “Toyota badge visible — Camry, not the Civic in this line.”" : ""; };
  return { h, upd };
}

// ── Library clips ───────────────────────────────────────────────────────────
function libraryCard(t) {
  const d = t.draft || {};
  const st = { segment_id: t.segment_id, usable: null, fits_line: null, description: d.description || "",
               subject: d.subject || "", identifiable: d.identifiable === 1 ? "1" : d.identifiable === 0 ? "0" : null,
               generic_use: d.generic_use || "", shot_type: d.shot_type || null, problems: new Set(d.problems || []), note: "" };
  const box = el("div", { class: "player" }, el("video", { src: "/rate/media/segment/" + t.segment_id + ".mp4?" + AUTH,
    controls: true, preload: "metadata", playsinline: true, loop: true }));
  const descHint = el("div", { class: "hint", "aria-live": "polite" });
  const checkDesc = () => { descHint.textContent = st.usable === "yes" ? textIssues(st.description) : ""; };
  const subj = el("input", { type: "text", value: st.subject, placeholder: "e.g. Toyota Camry (2018–2022) — only what you can see", on: { input: e => st.subject = e.target.value } });
  const gen = el("input", { type: "text", value: st.generic_use, placeholder: "e.g. draining engine oil from a car", on: { input: e => st.generic_use = e.target.value } });
  const draftNote = t.draft_source === "vision" || t.draft_source === "text"
    ? el("div", { class: "draft" }, "Pre-filled by AI" + (t.draft_source === "text" ? " from the title only (it couldn't see the clip)" : "") + " — check every field and correct it.") : null;
  const card = el("section", { class: "clip", "aria-label": "Library clip" }, box,
    el("div", { class: "body" },
      el("div", { class: "title" }, t.title || "Untitled", " ", el("small", {}, "· " + (t.source || "?").toUpperCase() + (t.channel ? " · " + t.channel : "") + " · " + fmt(t.src_in) + "–" + fmt(t.src_out) + " of the source"),
        t.page_url ? el("a", { href: t.page_url, target: "_blank", rel: "noopener" }, "source ↗") : null),
      t.lines.length ? el("div", { class: "stat" }, "Used for: " + t.lines.map(l => "“" + l + "”").join(" · ")) : null,
      t.lines.length ? el("div", { class: "row" }, el("span", { class: "label" }, "Right for that line?"),
        toggleGroup([["yes", "Yes"], ["partly", "Partly"], ["no", "No"]], () => st.fits_line, v => st.fits_line = v, "pick")) : null,
      el("div", { class: "row" }, el("span", { class: "label" }, "Usable?"),
        toggleGroup([["yes", "Yes, keep it", "yes"], ["no", "No, unusable anywhere", "no"]], () => st.usable,
          v => { st.usable = v; card.classList.remove("missing"); checkDesc(); }, "seg")),
      draftNote,
      el("label", { class: "field" }, el("span", {}, "What's on screen"),
        el("textarea", { placeholder: "Subject + action + framing. e.g. Close-up of a hand unscrewing the oil drain plug under a silver sedan; oil pours into a pan.",
                         on: { input: e => { st.description = e.target.value; checkDesc(); } } }, st.description), descHint),
      el("label", { class: "field" }, el("span", {}, "Exact subject you can see"), subj),
      el("div", { class: "row" }, el("span", { class: "label" }, "Can a viewer tell exactly what it is?"),
        toggleGroup([["1", "Yes — only for this subject"], ["0", "No — general footage"]], () => st.identifiable, v => st.identifiable = v, "pick")),
      el("label", { class: "field" }, el("span", {}, "Works as general footage for"), gen),
      el("div", { class: "row" }, el("span", { class: "label" }, "Shot"),
        toggleGroup((t.shot_types || []).map(x => [x.id, x.label]), () => st.shot_type, v => st.shot_type = v, "pick")),
      el("div", { class: "row" }, el("span", { class: "label" }, "Problems"),
        ...(t.problem_list || []).map(x => el("button", { type: "button", class: "chip", "aria-pressed": String(st.problems.has(x.id)),
          on: { click: e => { st.problems.has(x.id) ? st.problems.delete(x.id) : st.problems.add(x.id); e.currentTarget.setAttribute("aria-pressed", String(st.problems.has(x.id))); } } }, x.label))),
      el("textarea", { placeholder: "Anything else the next editor should know? (optional)", "aria-label": "Note", on: { input: e => st.note = e.target.value } })));
  st.card = card;
  st.ok = () => {
    if (!st.usable) return false;
    if (st.usable === "no") return true;
    return !textIssues(st.description) && st.identifiable !== null
      && (st.identifiable !== "1" || st.subject.trim()) && (st.identifiable !== "0" || st.generic_use.trim());
  };
  st.why = () => !st.usable ? "Say whether it's usable." : textIssues(st.description) || (st.identifiable === null ? "Say whether a viewer can tell exactly what it is."
    : st.identifiable === "1" && !st.subject.trim() ? "Name the exact subject." : "Say what it works for as general footage.");
  cards.push(st);
  return card;
}

// ── project picker ──────────────────────────────────────────────────────────
async function loadProjects() {
  try { PROJECTS = (await api("projects")).projects || []; } catch (e) { PROJECTS = []; }
}
function picker() {
  const labels = mode === "labels";
  const list = PROJECTS.filter(p => labels ? p.labels : p.clips);
  const sel = el("select", { id: "project", "aria-label": "Project", on: { change: e => {
    project = e.target.value; skipped.rate = []; skipped.labels = []; next(); } } },
    el("option", { value: "" }, labels ? "All finished edits — fewest checks first" : "All projects — fewest reviews first"),
    ...list.map(p => el("option", { value: String(p.id), selected: String(p.id) === project },
      p.title + " · " + (p.created_at || "").slice(0, 10) + " · " +
      (labels ? p.labels_mine + "/" + p.labels + " checked by you" : p.clips_mine + "/" + p.clips + " rated by you"))));
  return el("div", { class: "picker" }, el("label", { for: "project" }, "Project"), sel);
}

// ── flow ────────────────────────────────────────────────────────────────────
function render() {
  const main = document.getElementById("main");
  cards = []; suggestions = []; players = {};
  setMsg("");
  if (mode === "library") {
    if (!task || task.done) {
      main.replaceChildren(guide("library"), el("div", { class: "empty" }, "No library clips waiting. They appear after an editor sends back a finished XML."));
      document.getElementById("actions").hidden = true;
      return;
    }
    main.replaceChildren(guide("library"), el("h2", {}, "Describe this clip for the library"), libraryCard(task));
    document.getElementById("actions").hidden = false;
    setMsg(""); window.scrollTo(0, 0);
    return;
  }
  if (!task || task.done) {
    const msg = project ? "You've finished this project — pick another above."
      : mode === "labels" ? "No editor labels to check right now. They appear after an editor sends back a finished XML."
      : "All caught up — nothing left to review right now. Thank you!";
    main.replaceChildren(guide(mode), picker(), el("div", { class: "empty" }, msg));
    document.getElementById("actions").hidden = true;
    return;
  }
  if (mode === "labels") {
    main.replaceChildren(guide("labels"), picker(), shotHeader(), el("h2", {}, "Is each automatic label right?"), ...task.items.map(labelCard));
  } else {
    main.replaceChildren(guide("rate"), picker(), shotHeader(), el("h2", {}, "Rate each clip for this line"), ...task.items.map(rateCard), suggestBox());
  }
  document.getElementById("actions").hidden = false;
  setMsg("");
  window.scrollTo(0, 0);
}

async function refreshStat() {
  const meta = await api("meta"); REASONS = meta.reasons;
  const q = meta.queue;
  const lib = meta.library && meta.library.by_trust || {};
  document.getElementById("stat").textContent = q.rated + "/" + q.items + " clips reviewed · " + q.labels_checked + "/" + q.labels + " editor labels checked · "
    + (lib.verified || 0) + " library clips verified";
}
async function next() {
  try {
    const path = mode === "library" ? "library/next?skip=" + encodeURIComponent(skipped.library.join(","))
      : (mode === "labels" ? "labels/next" : "next") + "?skip=" + encodeURIComponent(skipped[mode].join(","))
      + (project ? "&project=" + encodeURIComponent(project) : "");
    task = await api(path);
    render();
  } catch (e) { document.getElementById("main").replaceChildren(el("div", { class: "empty" }, e.message)); }
}
function setMode(m) {
  if (m !== mode) project = "";
  mode = m;
  document.querySelectorAll(".tab").forEach(t => t.setAttribute("aria-selected", String(t.dataset.mode === m)));
  if (m === "impact") renderImpact(); else next();
}
document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => setMode(t.dataset.mode)));
document.getElementById("skip").addEventListener("click", () => {
  if (task && !task.done) skipped[mode].push(mode === "library" ? String(task.segment_id) : task.project_id + ":" + task.slot_id); next();
});
document.getElementById("save").addEventListener("click", async () => {
  const missing = cards.filter(c => !c.ok());
  missing.forEach(c => c.card.classList.add("missing"));
  if (missing.length) {
    setMsg(mode === "library" ? missing[0].why() : mode === "labels" ? "Mark every label Right or Wrong (and say what's right)." : "Mark every clip Yes / With a trim / No first.", true);
    missing[0].card.scrollIntoView({ behavior: "smooth", block: "center" }); return;
  }
  const btn = document.getElementById("save"); btn.disabled = true; setMsg("Saving…");
  try {
    if (mode === "library") {
      const c = cards[0];
      await api("library/submit", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
        segment_id: c.segment_id, usable: c.usable, fits_line: c.fits_line, description: c.description, subject: c.subject,
        identifiable: c.identifiable, generic_use: c.generic_use, shot_type: c.shot_type, problems: [...c.problems], note: c.note }) });
    } else if (mode === "labels") {
      await api("labels/submit", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
        reviews: cards.map(c => ({ asset_id: c.asset_id, action: c.action, verdict: c.verdict, used_slot_id: c.used_slot_id, note: c.note })) }) });
    } else {
      await api("submit", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
        project_id: task.project_id, slot_id: task.slot_id,
        ratings: cards.map(c => ({ item_id: c.item_id, usable: c.usable, fit: c.fit, reasons: [...c.reasons], note: c.note, segments: c.segments })),
        suggestions: suggestions.map(s => ({ url: s.url, note: s.note, segments: s.segments })) }) });
    }
    await refreshStat(); await loadProjects(); await next();
  } catch (e) { setMsg(e.message, true); }
  finally { btn.disabled = false; }
});
refreshStat().then(loadProjects).then(next).catch(e => document.getElementById("main").replaceChildren(el("div", { class: "empty" }, e.message)));
</script>
</body>
</html>
"""
