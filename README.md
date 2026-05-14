# B-Roll Finder

**From script to Premiere-ready sequence — automatically.**

B-Roll Finder takes a voiceover script and audio file, transcribes it, breaks it into shots, searches YouTube / Pexels / Pixabay for the right footage, lets you review and pick clips, generates on-screen text overlays with sound effects, and exports a complete FCP7 XML that drops straight into Premiere Pro.

---

## The problem it solves

Editors working on video-heavy content (tutorials, documentaries, product explainers) spend hours manually searching stock sites for B-roll, downloading clips, naming files, and building text overlay graphics. For recurring topics — like automotive content — the same clips get found and re-downloaded across projects, wasting time and API quota.

B-Roll Finder automates the search-and-select loop entirely, and builds a **local clip library** that gets smarter with every project: clips your team has confirmed and downloaded are indexed semantically so future searches surface them first.

---

## Workflow

```
Upload script + audio
        ↓
  Transcribe (Whisper via Groq)
        ↓
  Generate shot list  ←  AI director assigns intent, type, queries per shot
        ↓
  Fetch candidates    ←  Pexels · Pixabay · YouTube · 📚 Clip Library
        ↓
  LLM ranking         ←  Judge filters irrelevant clips, sorts by fit
        ↓
  Human review        ←  Pick clips shot-by-shot in a gallery UI
        ↓
  Download            ←  Parallel, deduped, cross-session cached
        ↓
  AI text overlays    ←  On-screen captions, SFX, animations
        ↓
  Export              ←  Premiere Pro XML  ·  shot_list.json  ·  .srt
```

---

## What each step does

**Step 1 — Transcribe & chunk**  
Whisper transcribes the voiceover via Groq. The transcript is split into ~2-minute chunks at sentence boundaries so you can work one section at a time within API rate limits.

**Step 2 — Generate shot list**  
An LLM reads each chunk and produces a structured shot list: timestamp range, visual intent, shot type (wide / close-up / aerial / abstract), 2–3 search queries per shot, and priority. Talking-head moments are marked `none` and skipped automatically. The table is editable before you fetch.

**Step 3 — Fetch candidates**  
All queries fan out in parallel across your chosen sources. The **Clip Library** source runs a semantic search over your own previously-downloaded footage first — no API calls, no re-downloading.

**Step 4 — Rank**  
An LLM-as-Judge reads each candidate's title, description, dimensions, and the query that found it, then ranks by relevance to the shot's narration and intent. Off-topic clips are hidden from the review grid automatically.

**Step 5 — Review & select**  
A paginated gallery per shot. Clip Library results appear with a purple border and a similarity % + usage count. Select clips, skip shots, or refetch individual shots with new queries.

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

Every clip your team downloads and confirms gets saved to a local SQLite database with a semantic embedding (`sentence-transformers/all-MiniLM-L6-v2`). On the next project, the Clip Library source runs a cosine similarity search against that database before hitting any external API — results are ranked by relevance and weighted by how often a clip has been used before.

Over time the library becomes the **first and fastest source** for recurring content categories (car engines, tools, road shots, etc.) — no quota, no network latency, no re-downloading.

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
| `OPENROUTER_API_KEY` | optional | Automatic fallback when Groq hits rate limits |
| `FREESOUND_API_KEY` | optional | SFX download for text overlays |

At least one of Pexels, Pixabay, or YouTube must be enabled. Groq is free to start at [console.groq.com](https://console.groq.com/).

---

## Project layout

```
B-Roll Finder/
├── app.py                    # Streamlit UI — all 7 steps
├── core/
│   ├── transcription.py      # Groq Whisper wrapper
│   ├── timing.py             # Audio duration, sentence-aware chunking
│   ├── director.py           # Shot list generation (LLM)
│   ├── director_search.py    # Parallel candidate fetch
│   ├── director_rank.py      # LLM-as-Judge ranking
│   ├── stock_apis.py         # Pexels / Pixabay / YouTube clients
│   ├── captions.py           # Text overlay PNG generation
│   ├── clip_library.py       # Local RAG clip database (SQLite + embeddings)
│   ├── sfx.py                # Freesound SFX search & download
│   ├── download_manager.py   # Parallel downloads, retries, dedup, cache
│   ├── output.py             # FCP7 XML, shot list, SRT export
│   └── keywords.py           # Classic mode keyword generation
├── prompts/                  # LLM system prompts
├── .env.example
├── requirements.txt
└── run.bat / run.sh
```
