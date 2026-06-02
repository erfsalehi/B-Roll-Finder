# B-Roll Finder

**From script to Premiere-ready sequence — automatically.**

B-Roll Finder takes a voiceover script and audio file, transcribes it, breaks it into shots, searches YouTube / Pexels / Pixabay for the right footage, lets you review and pick clips, generates on-screen text overlays with sound effects, and exports a complete FCP7 XML that drops straight into Premiere Pro.

---

## The problem it solves

Editors working on video-heavy content (tutorials, documentaries, product explainers) spend hours manually searching stock sites for B-roll, downloading clips, naming files, and building text overlay graphics. For recurring topics — like automotive content — the same clips get found and re-downloaded across projects, wasting time and API quota.

B-Roll Finder automates the search-and-select loop entirely, and builds a **local clip library** that gets smarter with every project: clips your team has confirmed and downloaded are indexed semantically so future searches surface them first.

It can run fully hands-off (**context-aware keywords + auto-selection** turn a transcript into a finished, download-ready sequence with no manual picking), runs on **free LLMs by default** with an optional **paid DeepSeek tier** for higher quality, and lets editors on **separate machines pool their libraries** by exporting and merging.

---

## Workflow

```
Upload script + audio
        ↓
  Transcribe (Whisper via Groq)
        ↓
  Generate shot list  ←  AI director assigns intent, type, queries per shot
                         (optional context-aware pre-pass anchors queries to
                          the macro-subject in play at each timestamp)
        ↓
  Fetch candidates    ←  Pexels · Pixabay · YouTube · 📚 Clip Library
        ↓
  LLM ranking         ←  Judge filters irrelevant clips, sorts by fit
                         (optional auto-select binds the top clip per shot)
        ↓
  Human review        ←  Pick clips in a gallery UI — or just review/override
                          the auto-selected picks
        ↓
  Download            ←  Parallel, deduped, cross-session cached
        ↓
  AI text overlays    ←  On-screen captions, SFX, animations
        ↓
  Export              ←  Premiere Pro XML  ·  shot_list.json  ·  .srt
```

---

## Fully Automatic Mode

Upload your voiceover, click **🚀 Run everything automatically**, and the app drives the entire pipeline with default settings — transcribe → topic → shot list → fetch → drop SD YouTube → rank → auto-select → **Step 5.5 QA review** — then stops so you can review and download. A progress bar + live status panel show each stage. Tick **"Also download automatically"** to have it kick off the download too once the review is done. Everything below still works step-by-step if you'd rather drive it manually.

### Telegram bot (run jobs from your phone)

The same pipeline runs **headless** (`core/pipeline.py`), so you can trigger it remotely. Run `python -m bot.telegram_bot` on an always-on machine (e.g. the office laptop), then **send the bot a voice message or audio file**: it runs the full pipeline — including downloading the clips and writing the Premiere XML into the project folder — and replies with a summary + the FCPXML attached, posting per-stage progress as it goes. By the time you reach the office, the project is downloaded and ready to edit. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ALLOWED_USERS` (fail-closed allowlist) in `.env`.

Send **`/status`** anytime to confirm the laptop is up and ready before starting — it reports internet/API reachability, ffmpeg, the configured keys, pipeline health, and whether a job is already running.

---

## What each step does

**Step 1 — Transcribe & chunk**  
Whisper transcribes the voiceover via Groq. The transcript is split into ~2-minute chunks at sentence boundaries so you can work one section at a time within API rate limits.

**Step 2 — Generate shot list**  
An LLM reads each chunk and produces a structured shot list: timestamp range, visual intent, shot type (wide / close-up / aerial / abstract), 2–3 search queries per shot, and priority. Talking-head moments are marked `none` and skipped automatically. The table is editable before you fetch.

*Optional — Context-aware keywords (`🧭`):* for structured videos (listicles, ranked countdowns, multi-product reviews) a fast pre-pass first maps the whole transcript into subject segments (e.g. *"BMW M3 E90" → 75s–165s*). The director then anchors each shot's queries to the subject active at that timestamp — so "the transmission is jerky" searches for *that car's* gearbox instead of a generic one. Toggle it on in Step 2; it degrades silently to the flat path if the pre-pass fails.

**Step 3 — Fetch candidates**  
All queries fan out in parallel across your chosen sources. The **Clip Library** source runs a semantic search over your own previously-downloaded footage first — no API calls, no re-downloading.

**Step 4 — Rank**  
An LLM-as-Judge reads each candidate's title, description, dimensions, and the query that found it, then ranks by relevance to the shot's narration and intent. Off-topic clips are hidden from the review grid automatically. Candidates are judged in **batches** (multiple shots per LLM call) with **jittered pacing** between calls, so large projects stay under provider rate limits instead of triggering 429 storms.

*Optional — Drop SD YouTube (`🎥`):* a one-click filter checks each YouTube candidate's HD/SD via the Data API — batched (≈1 quota unit per 50 clips) and capped — and removes confirmed-SD clips so only HD footage reaches ranking and selection. Stock clips (already HD) are skipped.

*Optional — Auto-selection (`🤖`):* when enabled, ranking also **binds the best clips to each shot automatically**, so every shot becomes download-ready with no manual pass. It picks **multiple clips scaled to the shot's length** (≈ one per 5s, min 2) — so a 30s shot gets ~6 clips spread across it, and even a short shot gets a couple of alternatives — while a deterministic variety guard avoids repeating the same clip back-to-back. You choose the **starting shot number** here (bounded to your shot count) — e.g. hand-pick an intro yourself and auto-select from shot 4 onward. A **Re-apply** button changes the start without re-ranking; manual picks are always preserved.

**Step 5 — Review & select**  
A paginated gallery per shot. Clip Library results appear with a purple border and a similarity % + usage count. Select clips, skip shots, or refetch individual shots with new queries. With auto-selection on, this step becomes **optional review/override** — auto-picks are flagged with a `🤖` badge, and picking a different clip clears the badge.

**Step 5.5 — Final QA review (optional)**  
An AI "executive producer" reads your whole *selected* timeline in one pass (the smart/reasoning tier) and flags thematic breaks, repeated visuals, pacing problems, or clips that don't match the narration — each pinned to a shot number with a concrete fix. Returns an empty list when the timeline reads as coherent. Run it after selecting, before downloading; jump back to Step 5 to swap any flagged shot.

**Step 6 — AI text overlays**  
Extract highlights from your script automatically (money amounts, statistics, headings, key concepts). Generate transparent 1920×1080 PNGs with:
- Per-category colors · font family selector · font size · text opacity
- Background rounded-rectangle box · 8-direction text outline or drop shadow  
- Emoji category prefix (📌 💰 📊 💡) · auto font scaling for long headings
- Animation per overlay: Fade In/Out · Slide Up · Slide In Left · **Random**
- Per-category animation suppression
- Freesound SFX download matched to each overlay moment

**Step 7 — Export**  
FCP7 XML with two video tracks (B-roll + overlay PNGs) and an audio track (SFX). Animations are written as keyframe effects so they play back immediately in Premiere. Also exports `shot_list.json`, `shot_list.txt`, and a `.srt` transcription.

---

## Clip Library

Every clip your team downloads and confirms gets saved to a local SQLite database (`.cache/clip_library.db`) with a semantic embedding (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim). On the next project, the Clip Library source runs a cosine similarity search against that database before hitting any external API — results are ranked by relevance and weighted by how often a clip has been used before.

Over time the library becomes the **first and fastest source** for recurring content categories (car engines, tools, road shots, etc.) — no quota, no network latency, no re-downloading.

### What's stored

**Per downloaded clip** (`clips` table):

| Field | What it is |
|-------|------------|
| `clip_url` | Original source URL (Pexels/Pixabay/YouTube) — the dedup key, and what enables re-download |
| `shot_description` | Narration/intent text for the shot — this is what gets embedded |
| `embedding` | 384-dim vector of the description (powers semantic search) |
| `clip_title`, `source` | Title and which provider it came from |
| `keywords`, `search_query` | The queries that found the clip |
| `project`, `slot_index` | Which project/shot it was used in |
| `duration`, `thumbnail_url`, `local_path` | Clip length, thumbnail, and where the file landed on this machine |
| `usage_count`, `last_used_at`, `created_at` | Reuse stats |

The actual video file is **not** stored in the database — it lives in `downloads/`, referenced by `local_path` (and re-fetchable from `clip_url`).

**Learned trims from XML re-import** (`clip_preferred_trims` table): when you re-import a Premiere/FCP7 XML, the app learns how you actually cut each clip. It records the trim `in_seconds` / `out_seconds`, the `shot_description` it applied to, the source XML path, and `confirmed_at` — so the next time that clip is reused, your proven trim is suggested instead of the full length. Re-import also creates minimal `clips` rows for any footage not already in the library (no embedding until you run **Re-embed**, so those stay invisible to search but fully usable for trims).

### Sharing across machines

Editors on separate machines can pool footage without a server:

- **⬇️ Export my library** writes a small portable JSON bundle (clip metadata + embeddings + learned trims).
- **🔀 Merge a teammate's export** pulls their clips into yours, deduped by `clip_url` — new clips are searchable immediately (embeddings travel in the bundle), known clips get their embedding backfilled and usage count raised, and trims merge newest-wins. Re-merging is idempotent.

Only the small metadata bundle is shared — **the actual videos never travel**; when a teammate reuses a clip, it re-downloads from its stored source URL. Both buttons live in the sidebar **Library Health** panel, which also flags clips missing embeddings and offers a one-click **Re-embed**.

---

## Quick start

**Prerequisites:** Python 3.10+ · FFmpeg on PATH

```bash
# Windows
run.bat

# macOS / Linux
chmod +x run.sh && ./run.sh
```

Opens at `http://localhost:8501`. The launcher creates a venv and installs dependencies automatically.

**Manual install:**
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## API keys

Enter keys in the in-app Setup expander or copy `.env.example` → `.env`.

| Key | Required | Powers |
|-----|----------|--------|
| `GROQ_API_KEY` | **yes** | Whisper transcription, shot-list LLM, ranking LLM, overlay extraction |
| `PEXELS_API_KEY` | optional | Pexels stock footage |
| `PIXABAY_API_KEY` | optional | Pixabay stock footage |
| `YOUTUBE_API_KEY` | optional | YouTube Data API v3 (100 quota units/search) |
| `DEEPSEEK_API_KEY` | optional | **An OpenRouter key — routes the paid `deepseek-v4-pro` model; preferred for all AI steps when set** |
| `OPENROUTER_API_KEY` | optional | Automatic free fallback when Groq hits rate limits |
| `FREESOUND_API_KEY` | optional | SFX download for text overlays |

At least one of Pexels, Pixabay, or YouTube must be enabled. Groq is free to start at [console.groq.com](https://console.groq.com/) and stays required even with DeepSeek set, because it also does the Whisper audio transcription.

### LLM provider priority

All text-LLM steps (shot list, ranking, keywords, topic, the context segmenter) route through one dispatcher with automatic fallback:

```
DeepSeek (if DEEPSEEK_API_KEY set)  →  Groq (cycles GROQ_API_KEY / _2)  →  OpenRouter
```

The DeepSeek tier is reached **through OpenRouter** — put an OpenRouter key in `DEEPSEEK_API_KEY`. It leads when available and transparently falls back to the free providers on any error.

DeepSeek runs as **two model tiers, switched per call with the same key**, matching each task to the right model:

| Tier | Model | Reasoning | Used for |
|------|-------|-----------|----------|
| **fast** | `deepseek/deepseek-v4-flash` | off | High-volume loop calls — shot slicing, ranking, keywords. Fast & cheap; no chain-of-thought to starve the JSON answer. |
| **smart** | `deepseek/deepseek-v4-pro` | on | Once-per-video global passes — video topic, global themes, the structural pre-pass. Single calls that benefit from synthesis, so no rate-limit risk. |

Override slugs with `DEEPSEEK_MODEL_FAST` / `DEEPSEEK_MODEL_SMART`.

### Advanced configuration (`.env`)

Optional toggles and tuning knobs, all off/default unless set:

| Variable | Default | Effect |
|----------|---------|--------|
| `ENABLE_CONTEXT_AWARE_KEYWORDS` | `false` | Run the subject-segmentation pre-pass (Step 2 `🧭`) |
| `ENABLE_AUTO_SELECTION` | `false` | Auto-bind the top-ranked clip per shot (Step 4 `🤖`) |
| `AUTO_SELECT_LOOKBACK` | `3` | Auto-select variety guard: don't reuse the same clip within this many recent shots (picks the next alternative instead). `0` disables |
| `AUTO_SELECT_SECONDS_PER_CLIP` | `5` | Auto-select binds ~1 clip per this many seconds of shot duration (a 30s shot → ~6 clips, spread across it) |
| `AUTO_SELECT_MIN_CLIPS` | `2` | Minimum clips auto-bound per shot — even a short shot gets alternatives for the editor |
| `AUTO_SELECT_MAX_CLIPS` | `8` | Cap on clips auto-bound per shot |
| `DIRECTOR_BLOCK_SIZE` | `20` | Transcription segments per shot-list LLM call. Raise it (e.g. `40`) on long transcripts to cut the number of calls and ease rate limits |
| `DEEPSEEK_MODEL_FAST` | `deepseek/deepseek-v4-flash` | Model for the fast tier (loop calls) |
| `DEEPSEEK_MODEL_SMART` | `deepseek/deepseek-v4-pro` | Model for the smart tier (global passes, reasoning on) |
| `DEEPSEEK_REASONING` | _(unset)_ | Emergency override of the per-tier reasoning default (`on`/`off`); leave unset to use each tier's default |
| `DEEPSEEK_MAX_TOKENS` | `8000` | Min token budget per DeepSeek call (matters mainly for the smart tier, where reasoning counts against it) |
| `RANK_BATCH_SIZE` | `6` | Shots judged per ranking LLM call |
| `RANK_MAX_WORKERS` | `3` | Concurrent ranking calls (lower under rate pressure) |
| `RANK_JITTER_MIN` / `RANK_JITTER_MAX` | `0.5` / `1.5` | Random delay (s) before each ranking call; set `MAX=0` to disable |

The two `ENABLE_*` toggles are also surfaced as checkboxes in the app's Step 2.

---

## Project layout

```
B-Roll Finder/
├── app.py                    # Streamlit UI — all 7 steps
├── core/
│   ├── transcription.py      # Groq Whisper wrapper
│   ├── timing.py             # Audio duration, sentence-aware chunking
│   ├── director.py           # Shot list generation + context-aware segmenter pre-pass
│   ├── director_search.py    # Parallel candidate fetch
│   ├── director_rank.py      # LLM-as-Judge ranking (batched) + auto-selection
│   ├── stock_apis.py         # Pexels / Pixabay / YouTube clients
│   ├── captions.py           # Text overlay PNG generation
│   ├── clip_library.py       # Local RAG clip database (SQLite + embeddings) + export/merge
│   ├── sfx.py                # Freesound SFX search & download
│   ├── download_manager.py   # Parallel downloads, retries, dedup, cache
│   ├── output.py             # FCP7 XML, shot list, SRT export
│   ├── xml_reimport.py       # Premiere/FCP7 XML re-import → learned trims
│   └── keywords.py           # LLM dispatcher (DeepSeek→Groq→OpenRouter) + classic keywords
├── prompts/                  # LLM system prompts (director, ranking, segmenter, …)
├── tests/                    # pytest suite
├── .env.example
├── requirements.txt
└── run.bat / run.sh
```
