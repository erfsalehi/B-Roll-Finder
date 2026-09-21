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
    GET  /rate/api/next           next shot to rate (?skip=pid:slot,pid:slot)
    POST /rate/api/submit         {project_id, slot_id, ratings: [...], suggestions: [...]}
    GET  /rate/api/labels/next    next shot of editor-XML labels to check
    POST /rate/api/labels/submit  {reviews: [{asset_id, action, verdict, used_slot_id, note}]}
    GET  /rate/api/impact         the reviewer's contribution log
"""

import json
import os
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

    from core import ratings
    try:
        if path == "/rate/api/meta" and handler.command == "GET":
            mine = next((s for s in ratings.rater_stats() if s["rater_id"] == rid), None)
            _send(handler, 200, {"reasons": [{"id": i, "label": l} for i, l in ratings.REASONS],
                                 "rated": mine["rated"] if mine else 0,
                                 "queue": ratings.queue_size()})
            return True

        if path in ("/rate/api/next", "/rate/api/labels/next") and handler.command == "GET":
            skip = []
            for tok in (qs.get("skip") or [""])[0].split(","):
                p, _, s = tok.partition(":")
                if p.isdigit() and s:
                    skip.append((int(p), s))
            fn = ratings.next_label_task if "labels" in path else ratings.next_task
            _send(handler, 200, fn(rid, skip=skip) or {"done": True})
            return True

        if path == "/rate/api/impact" and handler.command == "GET":
            _send(handler, 200, ratings.contribution_summary(rid))
            return True

        if path in ("/rate/api/submit", "/rate/api/labels/submit") and handler.command == "POST":
            n = int(handler.headers.get("Content-Length") or 0)
            if n <= 0 or n > _MAX_BODY:
                _send(handler, 413, {"error": "request too large"})
                return True
            body = json.loads(handler.rfile.read(n).decode("utf-8"))
            saved = 0
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
const skipped = { rate: [], labels: [] };
let mode = "rate", REASONS = [], task = null, cards = [], suggestions = [], players = {}, ytReady = null;

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
      el("textarea", { placeholder: "Anything else? Why it works or doesn't…", "aria-label": "Note",
                       on: { input: e => st.note = e.target.value } }, st.note),
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

// ── flow ────────────────────────────────────────────────────────────────────
function render() {
  const main = document.getElementById("main");
  cards = []; suggestions = []; players = {};
  if (!task || task.done) {
    main.replaceChildren(el("div", { class: "empty" }, mode === "labels"
      ? "No editor labels to check right now. They appear after an editor sends back a finished XML."
      : "All caught up — nothing left to review right now. Thank you!"));
    document.getElementById("actions").hidden = true;
    return;
  }
  if (mode === "labels") {
    main.replaceChildren(shotHeader(), el("h2", {}, "Is each automatic label right?"), ...task.items.map(labelCard));
  } else {
    main.replaceChildren(shotHeader(), el("h2", {}, "Rate each clip for this line"), ...task.items.map(rateCard), suggestBox());
  }
  document.getElementById("actions").hidden = false;
  setMsg("");
  window.scrollTo(0, 0);
}

async function refreshStat() {
  const meta = await api("meta"); REASONS = meta.reasons;
  const q = meta.queue;
  document.getElementById("stat").textContent = q.rated + "/" + q.items + " clips reviewed · " + q.labels_checked + "/" + q.labels + " editor labels checked";
}
async function next() {
  try {
    const path = (mode === "labels" ? "labels/next" : "next") + "?skip=" + encodeURIComponent(skipped[mode].join(","));
    task = await api(path);
    render();
  } catch (e) { document.getElementById("main").replaceChildren(el("div", { class: "empty" }, e.message)); }
}
function setMode(m) {
  mode = m;
  document.querySelectorAll(".tab").forEach(t => t.setAttribute("aria-selected", String(t.dataset.mode === m)));
  if (m === "impact") renderImpact(); else next();
}
document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => setMode(t.dataset.mode)));
document.getElementById("skip").addEventListener("click", () => {
  if (task && !task.done) skipped[mode].push(task.project_id + ":" + task.slot_id); next();
});
document.getElementById("save").addEventListener("click", async () => {
  const missing = cards.filter(c => !c.ok());
  missing.forEach(c => c.card.classList.add("missing"));
  if (missing.length) {
    setMsg(mode === "labels" ? "Mark every label Right or Wrong (and say what's right)." : "Mark every clip Yes / With a trim / No first.", true);
    missing[0].card.scrollIntoView({ behavior: "smooth", block: "center" }); return;
  }
  const btn = document.getElementById("save"); btn.disabled = true; setMsg("Saving…");
  try {
    if (mode === "labels") {
      await api("labels/submit", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
        reviews: cards.map(c => ({ asset_id: c.asset_id, action: c.action, verdict: c.verdict, used_slot_id: c.used_slot_id, note: c.note })) }) });
    } else {
      await api("submit", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
        project_id: task.project_id, slot_id: task.slot_id,
        ratings: cards.map(c => ({ item_id: c.item_id, usable: c.usable, fit: c.fit, reasons: [...c.reasons], note: c.note, segments: c.segments })),
        suggestions: suggestions.map(s => ({ url: s.url, note: s.note, segments: s.segments })) }) });
    }
    await refreshStat(); await next();
  } catch (e) { setMsg(e.message, true); }
  finally { btn.disabled = false; }
});
refreshStat().then(next).catch(e => document.getElementById("main").replaceChildren(el("div", { class: "empty" }, e.message)));
</script>
</body>
</html>
"""
