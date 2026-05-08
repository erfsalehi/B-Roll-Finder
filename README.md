# B-Roll Finder

An AI-driven Streamlit app that turns a voiceover into a downloadable shot list of relevant B-roll footage. Upload your audio, the app transcribes it, splits the script into manageable chunks, drafts a director's shot list per chunk, fetches candidates from Pexels / Pixabay / YouTube, has an LLM judge each one for relevance, and lets you tick the clips you want and download them in bulk.

Two modes are available:

- **Director** (default, recommended) — the full audio-in / clips-out pipeline described below.
- **Classic Finder** — an older keyword-based path, kept for compatibility.

---

## Quick start

### Prerequisites

- **Python 3.10+** (3.8 may work; 3.10+ is what's tested).
- **FFmpeg** on PATH ([ffmpeg.org/download.html](https://ffmpeg.org/download.html)). Used to compress audio above 25 MB before sending it to Whisper, and for optional video normalization.

### One-shot launcher

`run.bat` (Windows) and `run.sh` (macOS/Linux) create a venv, install dependencies, clear stale proxy env vars, and start Streamlit:

```bash
# Windows
run.bat

# macOS / Linux
./run.sh
```

### Manual install

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>.

---

## API keys

Copy `.env.example` to `.env` and fill in the keys you have. All keys can also be entered in the in-app Setup expander — the same `.env` is loaded either way.

| Key                         | Required? | What it powers                                                                  |
|-----------------------------|-----------|---------------------------------------------------------------------------------|
| `GROQ_API_KEY`              | **yes**   | Whisper transcription, the director LLM, the judge LLM                          |
| `PEXELS_API_KEY`            | optional  | Pexels stock-footage search                                                     |
| `PIXABAY_API_KEY`           | optional  | Pixabay stock-footage search                                                    |
| `YOUTUBE_API_KEY`           | optional  | YouTube Data API v3 search (quota: 10,000 units/day, 100 per call)              |
| `OPENROUTER_API_KEY`        | optional  | Automatic fallback when Groq returns rate-limit errors                          |
| `BROLL_BYPASS_HTTP_PROXY`   | optional  | Set to `1` if you run a TUN-mode VPN (V2RayN, Hiddify) and hit `WinError 10061` |

At least one search source (Pexels, Pixabay, or YouTube) must be enabled.

Where to get keys:

- **Groq** — [Groq Cloud Console](https://console.groq.com/) → API Keys.
- **Pexels** — [pexels.com/api](https://www.pexels.com/api/) → Get Started.
- **Pixabay** — [pixabay.com/api/docs](https://pixabay.com/api/docs/) — your key shows in the Parameters section once you're logged in.
- **YouTube Data API v3** — [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → enable YouTube Data API v3 → create an API key.
- **OpenRouter** — [openrouter.ai/keys](https://openrouter.ai/keys). The fallback model is the free `meta-llama/llama-3.3-70b-instruct:free`.

---

## Director mode workflow

Director walks you from audio file to a folder of B-roll clips in seven steps. State persists across page refreshes via `.cache/session_state.json`.

### Step 1 — Upload & Transcribe

Upload the voiceover (`.mp3`, `.wav`, or `.m4a`) and click **Transcribe & chunk**. Whisper transcribes it via Groq, then `chunk_segments_by_duration` splits the transcript into ~2-minute chunks at sentence boundaries (with a 3-minute hard force-cut as a safety net for monologues without punctuation).

You then pick **one chunk to work on at a time**. Each chunk shows its time range, word count, and a preview snippet; an expander lets you read the full chunk text to verify it's a meaningful section. The remaining steps act on the active chunk's segments only — keeps each batch comfortably within Groq's TPM rate limits and lets you iterate one section at a time.

### Step 2 — Generate Shot List

Enter the **video topic** (e.g. "car mechanics and engine repair") and optional style hints, then click **Generate Shot List**.

The director LLM reads the active chunk's transcribed segments and returns a JSON shot list. Each shot has a slot ID, timestamp range, the script chunk it covers, an intent ("show the precise mechanical action"), a shot type (wide / medium / close-up / extreme close-up / aerial / abstract / graphic), 2–3 search queries that explore different angles (subject + action + setting), and a priority (high / medium / low / none — talking-head shots get `none` and skip B-roll fetch entirely).

The shot list renders as an editable table: you can tweak the intent, change the priority, or rewrite the queries inline before fetching candidates.

### Step 3 — Fetch Candidates

Tick which sources to use (Pexels, Pixabay, YouTube) and choose how many results per query. Click **Fetch Candidates**.

For each shot, all queries fan out to Pexels and Pixabay in parallel via a `ThreadPoolExecutor`. YouTube fires once per shot using only the first query — its 100-quota-units-per-search cost would otherwise burn the daily 10k allowance fast. Results are deduplicated within each shot by URL.

### Step 4 — Rank & Filter Candidates

The LLM-as-Judge stage. See [the section below](#llm-as-judge) for the design.

### Step 5 — Review & Select

For each shot, a table shows every relevant candidate: thumbnail, title, source, dimensions, duration, description, the query that found it, and an "Open ↗" link to the source page. Tick checkboxes for the clips you want. **Select all** / **Clear** buttons handle whole shots at once.

The "Also" column flags candidates already picked for other shots — useful editorial info (you might want a different clip to avoid visual repetition across cuts) even though Step 6 hardlinks duplicates anyway.

Prev / Save & Next / Skip / Finish buttons appear above and below the table. Narration and search queries are recapped under the table so you don't have to scroll back up.

### Step 6 — Download Selected

Pick Quality (1080p / 720p / 480p / Best / Worst), Max Size in MB, and the concurrent-worker count. Click **Download**.

The download manager handles several edge cases worth calling out:

- **Cross-shot deduplication.** Same clip selected for multiple shots = one download + hardlinks to the other paths (same inode, zero extra disk).
- **Cross-session cache.** A small JSON registry at `.cache/downloads_registry.json` remembers URLs you've already downloaded; cache hits skip the network entirely. Stale entries (file deleted) prune lazily on lookup.
- **Per-task retry with new settings.** Change Quality from 1080p to 720p, click ↻ Retry on a failed task — the retry runs at the new setting.
- **Three-attempts cap.** Prevents infinite loops on permanently-broken URLs; button greys out as `Max retries`.
- **Error categorization.** Cryptic blobs like `HTTPSConnectionPool(host='pixabay.com'...)NameResolutionError` become `DNS resolution failed (check your VPN / DNS settings)`. Full original error available in a "Full error" expander.
- **History preservation.** Completed/failed tasks from prior batches stay visible when you click Download again.
- **Pause / Resume / Cancel** per task, plus Cancel All.
- **Direct downloads** (Pexels/Pixabay) use HTTP `Range` requests for resume; **YouTube** downloads go through `yt-dlp`.

### Step 7 — Export

Three buttons: **shot_list.json** (full structured data), **shot_list.txt** (human-readable), **markers.fcpxml** (Final Cut Pro / Premiere Pro chapter markers). All include shot IDs, timestamps, intents, and the URLs you selected.

---

## LLM-as-Judge

Step 4 — the ranking stage — is what makes the search results usable. Stock libraries surface clips by tag overlap, which means a query for *"mechanic engine repair"* can return woodworking close-ups (both involve "tool"). Without an explicit relevance check, every shot's review devolves into a 12-card-grid scavenger hunt.

### What the judge sees

For each shot, the judge receives:

- The narration text for that moment.
- The shot intent (what visual it should convey).
- The overall video topic and any user style notes.
- A list of candidate videos, formatted as:

  ```
  INDEX. [SOURCE] TITLE — DESCRIPTION | WIDTHxHEIGHT (orientation) | query: "QUERY USED"
  ```

The candidate enrichment is intentional. Stock-library responses are sparse:

- **Pexels** doesn't return descriptive titles. We extract the page-URL slug (`https://www.pexels.com/video/woman-walking-in-the-park-12345/` → `Woman Walking In The Park`) so the judge gets real semantic content instead of `Pexels Video 12345`.
- **Pixabay** returns the uploader's tag list verbatim — used directly as the title.
- **YouTube** returns the video title plus a 200-char description excerpt, prefixed with the channel name (`by AutoFix Garage — How to torque cylinder head bolts step by step…`).

Each candidate also carries the `matched_query` that surfaced it — useful prior signal but not decisive.

### What the judge produces

A `ranked` array, best to worst, with:

- `index` — the 0-based candidate index (validated; duplicates / out-of-range / non-int values are dropped with a counter).
- `reason` — a one-sentence justification for the top pick.
- `irrelevant: true` — flag for off-topic clips, which then get hidden in Step 5.

A horizontal-orientation tie-breaker prefers wide clips when relevance is similar (vertical clips would need cropping or letterboxing in 16:9).

### Quality safeguards

- **Temperature 0.1.** Judging is deterministic; re-running the rank stage on the same input produces consistent rankings.
- **Index validation.** Duplicates, negative indices, out-of-range indices, and non-int indices from a noisy LLM are rejected; the safety-net loop ensures every candidate appears exactly once in the output.
- **Per-shot parallelism.** `ThreadPoolExecutor` with 4 workers — a 60-shot script ranks ~4× faster than serial.
- **Failure surfacing.** Per-shot failures set `shot["rank_error"]` and append to an `errors` list the UI displays in an expander. Partial failures (some shots ranked, some failed) read `Ranked X/Y shots` rather than a generic success.
- **OpenRouter fallback.** If Groq hits its rate limit, the judge automatically retries through OpenRouter using the same JSON-mode contract.

### What the judge does NOT do (yet)

- **YouTube pixel dimensions.** The Data API doesn't expose width/height for normal user videos (`contentDetails.dimension` is just `2d`/`3d`, thumbnails are normalized to 16:9). YouTube clips show `unknown` orientation; the prompt explicitly tells the judge not to penalize unknown. Shorts are detected via the `#shorts` / `#youtubeshorts` hashtag in the title or description and synthesized to 1080×1920 so the horizontal preference correctly demotes them.
- **Duration awareness.** The judge doesn't pre-screen for clip length vs. the shot's `duration_needed_sec`. The editor handles trimming downstream.
- **Cross-shot reasoning.** Each shot is judged independently; the judge can't see what other shots have already picked.

---

## Troubleshooting

### Rate limits

Groq's TPM quota can bite on long videos. The Step-1 chunking is the primary defense — work one ~2-minute chunk at a time. If a chunk still fails, set `OPENROUTER_API_KEY` for automatic fallback.

### `WinError 10061` / proxy connection refused

Your shell has stale `HTTP_PROXY` / `HTTPS_PROXY` env vars pointing at a proxy listener that isn't running. If you use a TUN-mode VPN (V2RayN, Hiddify), the OS handles routing at the network layer and these vars should be empty. Set `BROLL_BYPASS_HTTP_PROXY=1` in `.env` to strip them at startup.

### `getaddrinfo failed` / DNS resolution failed

Your VPN's TUN mode isn't routing DNS. In V2RayN: set DNS to `1.1.1.1` or `8.8.8.8` and enable DNS hijacking. Or set Windows DNS to the same in Network Settings → Wi-Fi/Ethernet → Edit DNS server assignment.

### YouTube quota exhausted

Default Data API v3 quota is 10,000 units/day; each search costs 100 — so ~100 shots/day at the Director's one-search-per-shot rate. Request a quota increase in Google Cloud Console (free, ~1 day approval).

### Selections sometimes don't register in Step 5

Should be fixed as of recent commits — the candidate dataframe is now built once per shot visit instead of on every render, which avoids the Streamlit `data_editor` state-mismatch race that was eating clicks. If you still see flicker, restart Streamlit so Python module state is fresh.

---

## Project layout

```
B-Roll Finder/
├── app.py                       # Streamlit UI (Director + Classic modes)
├── core/
│   ├── transcription.py         # Groq Whisper wrapper
│   ├── timing.py                # Audio duration, sentence-aware chunking
│   ├── director.py              # Shot list generation (LLM)
│   ├── director_search.py       # Parallel candidate fetch
│   ├── director_rank.py         # LLM-as-Judge ranking
│   ├── stock_apis.py            # Pexels / Pixabay / YouTube clients
│   ├── youtube.py               # yt-dlp downloader (for YouTube clips)
│   ├── direct_downloader.py     # HTTP downloader (for Pexels/Pixabay)
│   ├── download_manager.py      # Orchestrates downloads, retries, dedup
│   ├── download_cache.py        # Cross-session URL → file registry
│   ├── ffmpeg_utils.py          # Audio compression, video normalization
│   ├── output.py                # Export: txt / json / FCPXML
│   └── keywords.py              # Classic-mode keyword generation
├── prompts/
│   ├── director.txt             # The director LLM prompt
│   ├── director_rank.txt        # The LLM-as-Judge prompt
│   ├── visual_keywords.txt      # Classic mode
│   ├── visual_keywords_json.txt
│   └── global_themes.txt
├── .env.example
├── requirements.txt
└── run.bat / run.sh
```

---

## Security notes

- **Don't commit `.env`.** It's in `.gitignore` along with `*Api.txt` and `*api.txt`. If you've already exposed a key, rotate it immediately in the provider's console.
- **Cached state** lives in `.cache/` (gitignored): session state, transcription chunks, the download registry.
- **Downloads** go to `downloads/director/` (gitignored).
