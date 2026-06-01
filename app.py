import streamlit as st
import os
import json
import zipfile
import io
import re
import pandas as pd
from dotenv import load_dotenv, set_key
from core.timing import get_audio_duration, parse_script_to_slots, calculate_wps
from core.keywords import generate_keywords_for_slots, generate_keywords_with_ai_chunking
from core.youtube import fetch_youtube_results
from core.stock_apis import search_pexels, search_pixabay
from core.output import (
    generate_keywords_txt, generate_youtube_txt, generate_srt,
    generate_transcription_srt, generate_shots_srt, generate_failed_downloads_txt,
    generate_fcpxml, generate_overlays_fcpxml, generate_shot_list_txt,
    filter_overlays_for_shots, _safe_for_fs
)
from core.captions import (
    extract_highlights, create_text_overlay, get_available_fonts,
    render_overlay_preview,
)
from core.clip_library import store_clip, search_library, get_library_stats, get_recent_trims
from core.sfx import search_freesound, download_sfx
from core.download_manager import DownloadManager, MAX_RETRIES, link_or_copy
from core import download_cache
from core.app_utils import check_network, format_speed, format_eta, update_yt_dlp_background
from core.session_cache import load_session_cache, save_session_cache
import time
import math
import threading

# --- Config & Initialization ---
st.set_page_config(page_title="B-Roll Finder", layout="wide")
ENV_FILE = ".env"
CACHE_FILE = ".cache/session_state.json"
if not os.path.exists(".cache"):
    os.makedirs(".cache")
load_dotenv(ENV_FILE)


def _env_flag(name: str) -> bool:
    """Truthy reading of a boolean .env toggle (matches the idiom used below)."""
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _set_env_flag(name: str, value: bool) -> None:
    """Persist a boolean toggle to .env and reflect it in the live process."""
    val = "true" if value else "false"
    set_key(ENV_FILE, name, val)
    os.environ[name] = val
# When the user runs a TUN-mode VPN (e.g. V2RayN, Hiddify), the OS
# captures all traffic at the network layer, so app-level HTTP_PROXY /
# HTTPS_PROXY env vars become stale — they point at an HTTP proxy
# listener that may not be running, and every request fails with
# WinError 10061 ("connection refused"). Set BROLL_BYPASS_HTTP_PROXY=1
# in .env to strip these vars on startup. Default off so users whose
# proxy *is* running keep working.
if os.getenv("BROLL_BYPASS_HTTP_PROXY", "").strip().lower() in ("1", "true", "yes"):
    for _var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                 "ALL_PROXY", "all_proxy"):
        os.environ.pop(_var, None)


# Keep yt-dlp fresh — YouTube breaks stale versions and downloads start
# failing. @st.cache_resource makes this fire exactly once per server process
# (i.e. each time the app is launched), and the helper itself is rate-limited
# to one pip upgrade per day. It runs in a background thread so startup isn't
# blocked; a newly-installed version takes effect on the next app restart.
@st.cache_resource(show_spinner=False)
def _kickoff_yt_dlp_update():
    if os.getenv("BROLL_SKIP_YT_DLP_UPDATE", "").strip().lower() not in ("1", "true", "yes"):
        update_yt_dlp_background()
    return True


_kickoff_yt_dlp_update()
# Initialize Session State
if "slots" not in st.session_state:
    st.session_state.slots = []
if "script_text" not in st.session_state:
    st.session_state.script_text = ""
if "audio_duration" not in st.session_state:
    st.session_state.audio_duration = 0.0
if "transcription_segments" not in st.session_state:
    st.session_state.transcription_segments = []
if "transcription_chunks" not in st.session_state:
    st.session_state.transcription_chunks = []
if "text_overlays" not in st.session_state:
    st.session_state.text_overlays = []
if "overlay_settings" not in st.session_state:
    st.session_state.overlay_settings = {
        "color": "#FFFFFF",
        "shadow": "#000000",
        "size": 120,
        "placement": "Bottom",
        "animation": "Fade In/Out",
        "no_anim_categories": [],
        "font_family": "Arial Bold",
        "effect_type": "Shadow",
        "text_opacity": 255,
        "bg_box": False,
        "bg_box_color": "#000000",
        "bg_box_opacity": 160,
        "emoji_prefix": False,
        # Fine-tune offsets applied on top of the Top/Middle/Bottom anchor.
        # Both are in 1080p pixels; positive y_offset moves text DOWN, positive
        # x_offset moves text RIGHT.
        "y_offset": 0,
        "x_offset": 0,
        "category_colors": {
            "headings/titles": "#FFFFFF",
            "money/pricing":   "#FFD700",
            "statistics":      "#00BFFF",
            "core concepts":   "#FFFF00",
        },
    }
    st.session_state.transcription_chunks = []
if "active_chunk_indices" not in st.session_state:
    st.session_state.active_chunk_indices = [0]
if "global_themes" not in st.session_state:
    st.session_state.global_themes = []
if "dm" not in st.session_state:
    st.session_state.dm = DownloadManager()
if "is_fetching" not in st.session_state:
    st.session_state.is_fetching = False
# Chunked background fetch state — see Step 3 dispatcher below.
if "chunk_fetch_status" not in st.session_state:
    st.session_state.chunk_fetch_status = {}      # {chunk_idx: 'pending'|'fetching'|'done'|'error'}
if "chunk_fetch_errors" not in st.session_state:
    st.session_state.chunk_fetch_errors = {}      # {chunk_idx: [error_str, ...]}
if "chunk_fetch_slot_ids" not in st.session_state:
    st.session_state.chunk_fetch_slot_ids = []    # [[slot_id, ...], ...] parallel to chunk indices
if "chunk_fetch_thread" not in st.session_state:
    st.session_state.chunk_fetch_thread = None
if "fetch_chunk_size" not in st.session_state:
    st.session_state.fetch_chunk_size = 6
if "d_video_topic" not in st.session_state:
    st.session_state.d_video_topic = ""
if "d_style" not in st.session_state:
    st.session_state.d_style = ""
# Attempt to load from cache

def load_cache():
    try:
        load_session_cache(CACHE_FILE, st.session_state)
    except Exception as e:
        st.error(f"Error loading cache: {e}")

def save_cache():
    try:
        save_session_cache(CACHE_FILE, st.session_state)
    except Exception as e:
        st.error(f"Error saving cache: {e}")

def toggle_pick(url, cand_dict, shot_dict):
    """Toggle a candidate in the shot's selected_results."""
    current_urls = {r.get("url") for r in shot_dict.get("selected_results", [])}
    if url in current_urls:
        shot_dict["selected_results"] = [r for r in shot_dict["selected_results"] if r.get("url") != url]
    else:
        if "selected_results" not in shot_dict:
            shot_dict["selected_results"] = []
        shot_dict["selected_results"].append(cand_dict)
    # A manual pick supersedes any automatic one — clear the auto badge.
    shot_dict.pop("auto_selected", None)
    save_cache()


# ── Chunked fetch live-status renderers ──────────────────────────────────────
# `_render_chunk_status_polling` is a Streamlit fragment that polls the
# background dispatcher's status dict every 2s and triggers a full app rerun
# whenever a chunk transitions to done/error. That rerun is what makes Step 5
# pick up the newly-fetched shots.

def _render_chunk_status_body():
    status_dict = st.session_state.get("chunk_fetch_status") or {}
    if not status_dict:
        return
    n_total = len(status_dict)
    n_done = sum(1 for v in status_dict.values() if v == "done")
    n_err = sum(1 for v in status_dict.values() if v == "error")
    n_active = sum(1 for v in status_dict.values() if v == "fetching")
    n_pending = sum(1 for v in status_dict.values() if v == "pending")
    completed = n_done + n_err

    parts = [f"**Fetch progress:** {completed}/{n_total} chunks"]
    if n_active:  parts.append(f"🔄 {n_active} fetching")
    if n_pending: parts.append(f"⏳ {n_pending} pending")
    if n_err:     parts.append(f"❌ {n_err} errored")
    if completed == n_total and n_err == 0:
        parts.append("✅ done")
    st.markdown(" · ".join(parts))

    icons = {"pending": "⏳", "fetching": "🔄", "done": "✅", "error": "❌"}
    slot_groups = st.session_state.get("chunk_fetch_slot_ids", [])
    n_cols = min(max(n_total, 1), 6)
    cols = st.columns(n_cols)
    for i in range(n_total):
        v = status_dict.get(i, "pending")
        ico = icons.get(v, "?")
        sids = slot_groups[i] if i < len(slot_groups) else []
        label = f"shots {sids[0]}–{sids[-1]}" if sids else ""
        with cols[i % n_cols]:
            st.markdown(
                f"{ico} **Chunk {i+1}**<br><small>{label}</small>",
                unsafe_allow_html=True,
            )

    err_dict = st.session_state.get("chunk_fetch_errors", {})
    if err_dict:
        with st.expander(f"⚠️ Fetch errors in {len(err_dict)} chunk(s)"):
            for cidx in sorted(err_dict):
                st.write(f"**Chunk {cidx+1}:**")
                for e in (err_dict[cidx] or [])[:5]:
                    st.write(f"  • {e}")


@st.fragment(run_every=2.0)
def _render_chunk_status_polling():
    with st.container(border=True):
        _render_chunk_status_body()
    status_dict = st.session_state.get("chunk_fetch_status") or {}
    n_total = len(status_dict)
    completed = sum(1 for v in status_dict.values() if v in ("done", "error"))
    still_running = completed < n_total
    st.session_state.is_fetching = still_running
    # Trigger a full app rerun when chunk completion changes, OR when the
    # final chunk lands (so the page switches from polling-fragment to the
    # static "Fetch complete" display and polling stops).
    last_seen = st.session_state.get("_last_chunk_completed_count", -1)
    if completed != last_seen or not still_running:
        st.session_state._last_chunk_completed_count = completed
        # The background dispatcher mutates the shot dicts in place but never
        # touches the on-disk cache. Persist once everything's done so the
        # fetched candidates survive a browser refresh.
        if not still_running:
            save_cache()
        st.rerun(scope="app")


def _render_chunk_status_final():
    with st.container(border=True):
        _render_chunk_status_body()
    st.session_state.is_fetching = False


# ── Quality-selectbox label formatter ────────────────────────────────────────
# The Quality dropdowns in the downloader (Classic Step 6 / Global / Director
# Step 6) all share the same option list. The raw values like "1080p" are
# ambiguous — they read as "exactly 1080p" but actually mean "best stream
# capped at 1080p". This format_func makes that explicit in the UI without
# changing the underlying option strings, so the backend (download_video)
# and any cached selections in session state keep working unchanged.
_QUALITY_LABELS = {
    "Best":  "Best available (no cap — may be 4K)",
    "1080p": "Best up to 1080p",
    "720p":  "Best up to 720p",
    "480p":  "Best up to 480p",
    "Worst": "Worst (smallest, fastest)",
}


def _quality_label(opt: str) -> str:
    return _QUALITY_LABELS.get(opt, opt)

def toggle_global_pick(url):
    if 'picked_global_urls' not in st.session_state:
        st.session_state.picked_global_urls = set()
    if url in st.session_state.picked_global_urls:
        st.session_state.picked_global_urls.remove(url)
    else:
        st.session_state.picked_global_urls.add(url)
    save_cache()

def perform_chunk_regeneration(chunk_idx):
    """Regenerates the shot list for a specific chunk, preserving selected videos."""
    if not os.getenv("GROQ_API_KEY"):
        st.error("Set Groq API Key.")
        return
    # 1. Collect preserved selections
    preserved_results = []
    seen_urls = set()
    for s in st.session_state.director_shots:
        if s.get("chunk_id") == chunk_idx:
            for res in s.get("selected_results", []):
                if res.get("url") not in seen_urls:
                    preserved_results.append(res)
                    seen_urls.add(res.get("url"))
    
    # 2. Generate new shots
    from core.director import generate_shot_list_from_transcription
    chunk = st.session_state.transcription_chunks[chunk_idx]
    _roadmap = st.session_state.get("segment_roadmap") if _env_flag("ENABLE_CONTEXT_AWARE_KEYWORDS") else None
    new_shots = generate_shot_list_from_transcription(
        chunk.get("segments", []),
        os.getenv("GROQ_API_KEY"),
        custom_instructions=st.session_state.get("d_style", ""),
        video_topic=st.session_state.get("d_video_topic", ""),
        chunk_id=chunk_idx,
        segment_roadmap=_roadmap or None,
    )
    
    # 3. Add preserved results to new shots
    for ns in new_shots:
        ns["video_results"] = preserved_results + ns.get("video_results", [])
        ns["selected_results"] = [r for r in preserved_results]
    
    # 4. Splice back into global list
    # If chunk_id is missing, we treat it as 0 for backwards compatibility.
    def get_cid(s):
        cid = s.get("chunk_id")
        return 0 if cid is None else cid
    other_shots = [s for s in st.session_state.director_shots if get_cid(s) != chunk_idx]
    combined = other_shots + new_shots
    combined.sort(key=lambda x: (x.get("chunk_id", 0), x.get("timestamp", 0)))
    
    # 5. Re-index slot_ids
    for idx, s in enumerate(combined):
        s["slot_id"] = idx + 1
        
    st.session_state.director_shots = combined
    save_cache()

def ensure_shots_have_chunk_ids():
    """Repairs shots that are missing a chunk_id by inferring it from timestamps."""
    if not st.session_state.get("director_shots") or not st.session_state.get("transcription_chunks"):
        return
    
    chunks = st.session_state.transcription_chunks
    changed = False
    for shot in st.session_state.director_shots:
        if shot.get("chunk_id") is None:
            ts = float(shot.get("timestamp", 0))
            found = False
            for i, chunk in enumerate(chunks):
                if not chunk.get("segments"):
                    continue
                # Use float to ensure safe comparison
                c_end = float(chunk["segments"][-1].get("end", 99999))
                # Add a small buffer (e.g. 5 seconds) to catch edge cases
                if ts <= (c_end + 5.0):
                    shot["chunk_id"] = i
                    changed = True
                    found = True
                    break
            if not found:
                # If timestamp is outside all chunks, default to the last chunk
                shot["chunk_id"] = max(0, len(chunks) - 1)
                changed = True
    if changed:
        save_cache()
# Load the on-disk session cache exactly ONCE per session, not on every
# rerun. The old guard (`not st.session_state.slots`) was always true in
# Director mode — `slots` is the Classic-mode keyword list and stays empty —
# so the cache was re-read on every 2s poll-rerun. That clobbered the
# director_shots dicts the background fetch thread mutates in place (it
# replaced them with stale, empty-result copies from disk), which is why
# Step 3 results never reached Step 5. session_state already persists across
# reruns in memory, so a single startup load is all we need.
if "_cache_loaded" not in st.session_state:
    st.session_state._cache_loaded = True
    if os.path.exists(CACHE_FILE):
        load_cache()

# --- UI Components ---
st.sidebar.title("App Mode")
app_mode = st.sidebar.radio(
    "Select Mode",
    ["Director", "Classic Finder"],
    help=(
        "**Director** is the standard workflow. "
        "**Classic Finder** is the legacy keyword-based path."
    ),
)


def render_library_health():
    """Sidebar panel: at-a-glance Clip Library health, shown in every mode.

    Surfaces the failure mode that silently emptied the library before — clips
    saved without an embedding (because the embedding model was unavailable)
    are invisible to semantic search, so we flag them and offer an on-demand
    model test rather than auto-loading the ~80 MB model on every render.
    """
    stats = get_library_stats()
    total = stats.get("total", 0)
    no_emb = stats.get("without_embedding", 0)
    trims = stats.get("trims", 0)

    if total == 0:
        icon, label = "⚪", "Empty"
    elif no_emb == 0:
        icon, label = "🟢", "Healthy"
    elif no_emb >= total:
        icon, label = "🔴", "Embeddings missing"
    else:
        icon, label = "🟡", "Partially indexed"

    with st.sidebar.expander(f"{icon} Library Health — {label}", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Clips", total)
        c2.metric("Searchable", stats.get("with_embedding", 0))
        c3.metric("Trims", trims)

        if total and no_emb:
            st.warning(
                f"{no_emb} clip(s) have no embedding — they won't appear in "
                "semantic search (but still work for downloads and learned "
                "trims). New downloads embed automatically now."
            )
            if st.button(f"Re-embed {no_emb} clip(s)", key="lib_health_reembed",
                         help="Backfills embeddings so these clips become semantically searchable. "
                              "Loads the model once (~60s first time)."):
                _pbar = st.progress(0.0)
                with st.spinner("Embedding clips…"):
                    try:
                        from core.clip_library import reembed_missing_clips
                        _res = reembed_missing_clips(progress_callback=lambda f: _pbar.progress(min(1.0, f)))
                    except Exception as e:
                        _res = {"error": str(e)}
                if _res.get("error"):
                    st.error(f"❌ Re-embed failed: {_res['error']}")
                else:
                    msg = f"✅ Re-embedded {_res['updated']} clip(s)."
                    if _res.get("skipped"):
                        msg += f" Skipped {_res['skipped']} with no text."
                    st.success(msg)
                    st.rerun()
        elif total and no_emb == 0:
            st.caption("All clips are embedded and semantically searchable.")

        if st.button("Test embedding model", key="lib_health_test",
                     help="Loads the model once (~60s first time) and embeds a test phrase."):
            with st.spinner("Loading embedding model…"):
                try:
                    from core.clip_library import _embed
                    vec = _embed("library health check")
                    st.success(f"✅ Embedding model works (dim={len(vec)}). "
                               "New downloads will be searchable.")
                except Exception as e:
                    st.error(
                        "❌ Embedding model unavailable — clips save without "
                        f"semantic search. Detail: {e}"
                    )


render_library_health()


def render_classic_mode():
    st.title("🎬 B-Roll Finder")
    # ── Setup: API keys (status pill in header, auto-collapses when ready) ──
    # Auto-import keys from any *Api.txt files in the project root.
    for txt_file, env_key in [
        ("groq api.txt",      "GROQ_API_KEY"),
        ("groq api 2.txt",    "GROQ_API_KEY_2"),
        ("Pexels Api.txt",    "PEXELS_API_KEY"),
        ("Pixabay Api.txt",   "PIXABAY_API_KEY"),
        ("YT Api.txt",        "YOUTUBE_API_KEY"),
        ("Openrouter Api.txt","OPENROUTER_API_KEY"),
        ("Freesound Api.txt", "FREESOUND_API_KEY"),
    ]:
        if not os.getenv(env_key) and os.path.exists(txt_file):
            try:
                with open(txt_file, 'r') as f:
                    key_val = f.read().strip()
                    if key_val:
                        set_key(ENV_FILE, env_key, key_val)
            except Exception:
                pass
    load_dotenv(ENV_FILE, override=True)
    _key_status = {
        "Groq 1":     bool(os.getenv("GROQ_API_KEY")),
        "Groq 2":     bool(os.getenv("GROQ_API_KEY_2")),
        "Pexels":     bool(os.getenv("PEXELS_API_KEY")),
        "Pixabay":    bool(os.getenv("PIXABAY_API_KEY")),
        "YouTube":    bool(os.getenv("YOUTUBE_API_KEY")),
        "OpenRouter": bool(os.getenv("OPENROUTER_API_KEY")),
        "Freesound":  bool(os.getenv("FREESOUND_API_KEY")),
    }
    _pill = " · ".join(f"{'✅' if v else '○'} {k}" for k, v in _key_status.items())
    with st.expander(f"⚙️  Setup — API keys  ·  {_pill}",
                     expanded=not (_key_status["Groq 1"] or _key_status["Groq 2"])):
        st.caption(
            "Groq is required (script analysis). Pexels/Pixabay/YouTube are search "
            "sources — enable at least one. OpenRouter is an automatic fallback "
            "when Groq hits its rate limit."
        )
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            groq_input    = st.text_input("Groq API Key 1 (required)",
                                          value=os.getenv("GROQ_API_KEY", ""),
                                          type="password")
            groq_input_2  = st.text_input("Groq API Key 2 (fallback)",
                                          value=os.getenv("GROQ_API_KEY_2", ""),
                                          type="password",
                                          help="Secondary Groq key to use when the first one hits rate limits.")
            pexels_input  = st.text_input("Pexels API Key",
                                          value=os.getenv("PEXELS_API_KEY", ""),
                                          type="password")
        with col_k2:
            pixabay_input = st.text_input("Pixabay API Key",
                                          value=os.getenv("PIXABAY_API_KEY", ""),
                                          type="password")
        with col_k2:
            youtube_input    = st.text_input("YouTube Data API Key",
                                             value=os.getenv("YOUTUBE_API_KEY", ""),
                                             type="password",
                                             help="Optional. 10,000 quota units/day; each search costs 100.")
            openrouter_input = st.text_input("OpenRouter API Key",
                                             value=os.getenv("OPENROUTER_API_KEY", ""),
                                             type="password",
                                             help="Optional fallback when Groq returns rate-limit errors.")
            freesound_input  = st.text_input("Freesound API Key",
                                             value=os.getenv("FREESOUND_API_KEY", ""),
                                             type="password",
                                             help="Required for automatic sound effects downloads.")
            st.markdown("&nbsp;")
        _browser_options = ["None", "chrome", "firefox", "edge", "brave", "safari", "opera", "chromium"]
        _current_browser = os.getenv("YT_COOKIE_BROWSER", "None")
        _browser_idx = _browser_options.index(_current_browser) if _current_browser in _browser_options else 0
        cookie_browser = st.selectbox(
            "YouTube Cookie Browser",
            options=_browser_options,
            index=_browser_idx,
            help=(
                "Fixes 'Sign in to confirm you're not a bot' errors. "
                "Select the browser where you are logged into YouTube — yt-dlp will "
                "borrow its cookies for authentication. The browser must be installed "
                "and you must be logged into YouTube in it."
            ),
        )
        if st.button("Save API Keys", type="primary"):
            if groq_input:
                if not os.path.exists(ENV_FILE):
                    open(ENV_FILE, 'w').close()
                set_key(ENV_FILE, "GROQ_API_KEY", groq_input)
                if groq_input_2:      set_key(ENV_FILE, "GROQ_API_KEY_2",      groq_input_2)
                if openrouter_input: set_key(ENV_FILE, "OPENROUTER_API_KEY", openrouter_input)
                if pexels_input:     set_key(ENV_FILE, "PEXELS_API_KEY",     pexels_input)
                if pixabay_input:    set_key(ENV_FILE, "PIXABAY_API_KEY",    pixabay_input)
                if youtube_input:    set_key(ENV_FILE, "YOUTUBE_API_KEY",    youtube_input)
                if freesound_input:  set_key(ENV_FILE, "FREESOUND_API_KEY",  freesound_input)
                set_key(ENV_FILE, "YT_COOKIE_BROWSER", cookie_browser)
                load_dotenv(ENV_FILE, override=True)
                st.success("API keys saved to .env. Refresh the page to re-evaluate the status pill.")
            else:
                st.warning("Groq API key is required.")

    # ── Clip Library sidebar panel ───────────────────────────────────────────
    with st.sidebar.expander("📚 Clip Library", expanded=False):
        _stats = get_library_stats()
        st.metric("Total clips stored", _stats["total"])
        if _stats["by_source"]:
            for _src, _cnt in _stats["by_source"].items():
                st.caption(f"  {_src}: {_cnt}")
        if _stats["top_clips"]:
            st.markdown("**Most-used clips**")
            for _clip in _stats["top_clips"]:
                st.markdown(
                    f"- [{_clip['clip_title'][:40] or 'untitled'}]({_clip['clip_url']}) "
                    f"· {_clip['source']} · used {_clip['usage_count']}×"
                )
        if _stats["total"] == 0:
            st.info("No clips yet. Select and download clips — they'll be saved here automatically.")

    # Step 1: Upload
    st.header("Step 1: Upload Files")
    col1, col2 = st.columns(2)
    with col1:
        script_file = st.file_uploader("Upload Script (.txt)", type=["txt"])
    with col2:
        audio_file = st.file_uploader("Upload Voiceover (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a", "mp4"])
    if script_file and audio_file:
        # Read files
        script_content = script_file.read().decode("utf-8")
        # Save audio temporarily to read duration
        audio_path = os.path.join(".cache", "temp_audio" + os.path.splitext(audio_file.name)[1])
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())
        duration = get_audio_duration(audio_path)
        word_count = len(script_content.split())
        st.info(f"Detected Audio Duration: {duration:.2f} seconds | Script Word Count: {word_count}")
        
        st.audio(audio_path)
        with st.expander("View Script / Transcription"):
            if st.session_state.transcription_segments:
                st.write("**AI Transcription (Timestamped):**")
                for seg in st.session_state.transcription_segments:
                    start_m = int(seg['start'] // 60)
                    start_s = int(seg['start'] % 60)
                    st.write(f"**[{start_m:02d}:{start_s:02d}]** {seg['text']}")
            else:
                st.text_area("Full Script", value=script_content, height=200, disabled=True)
        st.session_state.script_text = script_content
        st.session_state.audio_duration = duration
    # Step 2: AI Transcription (Optional for Precise Timing)
    st.header("Step 2: AI Transcription (Optional)")
    if audio_file and st.button("Transcribe Audio for Precise Timing"):
        if not os.getenv("GROQ_API_KEY"):
            st.error("Please set your Groq API key in the Setup section above.")
        else:
            from core.transcription import transcribe_audio
            with st.spinner("Transcribing with Whisper..."):
                try:
                    # Save audio temporarily if not already saved
                    audio_ext = os.path.splitext(audio_file.name)[1]
                    audio_path = os.path.join(".cache", f"audio_to_transcribe{audio_ext}")
                    with open(audio_path, "wb") as f:
                        f.write(audio_file.getvalue())
                        
                    segments = transcribe_audio(audio_path, os.getenv("GROQ_API_KEY"))
                    st.session_state.transcription_segments = segments
                    
                    # Also update script_text with transcription if it's better
                    full_text = " ".join([s['text'] for s in segments])
                    st.session_state.script_text = full_text
                    st.success("Transcription complete! Precise timestamps will now be used.")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
    if st.session_state.transcription_segments:
        st.info(f"✨ Using {len(st.session_state.transcription_segments)} transcription segments for precise timing.")
    # Step 3: Settings
    st.header("Step 3: Settings")
    
    with st.expander("Range Selection", expanded=False):
        use_portion = st.checkbox("Process Specific Portion", value=False)
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            start_time_s = st.number_input("Start Time (seconds)", value=0.0, step=1.0)
        with col_p2:
            end_time_s = st.number_input("End Time (seconds)", value=min(st.session_state.audio_duration, 300.0) if st.session_state.audio_duration > 0 else 60.0, step=1.0)
    chunking_method = st.radio("Script Chunking Method", ["AI Meaningful Chunks (Recommended)", "Strict Fixed Intervals (Legacy)"])
    col1, col2, col3, col4 = st.columns(4)
    if chunking_method == "Strict Fixed Intervals (Legacy)":
        with col1:
            intro_dur = st.number_input("Intro Duration (s)", value=30.0, step=1.0)
        with col2:
            intro_int = st.number_input("Intro Keyword Interval (s)", value=1.0, step=0.5)
        with col3:
            body_int = st.number_input("Body Keyword Interval (s)", value=2.0, step=0.5)
    with col4:
        num_alt = st.number_input("Keyword Alternatives", value=3, step=1, min_value=1, max_value=5)
    # Step 4: Generate Keywords
    st.header("Step 4: Generate Keywords")
    custom_instructions = st.text_area("Custom Style / Instructions (Optional)", placeholder="e.g. Make it look like a 90s VHS tape, or POV branded style")
    if st.button("Generate Keywords"):
        if not os.getenv("GROQ_API_KEY"):
            st.error("Please set your Groq API key in the Setup section above.")
        elif not st.session_state.script_text or st.session_state.audio_duration <= 0:
            st.error("Please upload valid script and audio files first.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            def update_progress(p):
                progress_bar.progress(p)
                status_text.text(f"Generating keywords: {int(p * 100)}% complete")
            try:
                # Prepare script and duration based on portion
                target_script = st.session_state.script_text
                target_duration = st.session_state.audio_duration
                start_offset = 0.0
                
                full_wps = calculate_wps(st.session_state.script_text, st.session_state.audio_duration)
                
                if use_portion:
                    from core.timing import slice_script_by_time
                    target_script = slice_script_by_time(st.session_state.script_text, st.session_state.audio_duration, start_time_s, end_time_s)
                    target_duration = end_time_s - start_time_s
                    start_offset = start_time_s
                    st.info(f"Processing portion: {start_time_s}s to {end_time_s}s ({target_duration:.2f}s)")
                if st.session_state.transcription_segments and not use_portion:
                    # Use precise transcription segments
                    from core.keywords import generate_keywords_from_transcription
                    st.session_state.slots = generate_keywords_from_transcription(
                        st.session_state.transcription_segments,
                        api_key=os.getenv("GROQ_API_KEY"),
                        num_alternatives=num_alt,
                        progress_callback=update_progress,
                        custom_instructions=custom_instructions
                    )
                elif chunking_method == "AI Meaningful Chunks (Recommended)":
                    st.session_state.slots = generate_keywords_with_ai_chunking(
                        target_script,
                        wps=full_wps,
                        api_key=os.getenv("GROQ_API_KEY"),
                        num_alternatives=num_alt,
                        progress_callback=update_progress,
                        custom_instructions=custom_instructions,
                        start_offset=start_offset
                    )
                else:
                    # 1. Parse slots mathematically
                    slots = parse_script_to_slots(
                        target_script, 
                        target_duration, 
                        intro_duration=intro_dur, 
                        intro_interval=intro_int, 
                        body_interval=body_int,
                        start_offset=start_offset
                    )
                    # 2. Call Groq
                    st.session_state.slots = generate_keywords_for_slots(
                        slots, 
                        api_key=os.getenv("GROQ_API_KEY"), 
                        num_alternatives=num_alt, 
                        progress_callback=update_progress,
                        custom_instructions=custom_instructions
                    )
                save_cache()
                st.success("Keywords generated successfully!")
            except Exception as e:
                st.error(f"Error generating keywords: {e}")
    if st.session_state.slots:
        st.subheader("Edit Keywords")
        # Display an editable table of keywords
        # Flatten data for dataframe
        table_data = []
        for i, slot in enumerate(st.session_state.slots):
            row = {
                "Time": f"{slot['timestamp']}s",
                "Script Segment": slot['text'],
                "Primary Keyword": slot.get('keywords', [''])[0] if slot.get('keywords') else "",
                "Alternatives": " | ".join(slot.get('keywords', [])[1:])
            }
            table_data.append(row)
        edited_df = st.data_editor(table_data, num_rows="dynamic", width="stretch")
        # Sync edits back to session state
        if st.button("Save Edited Keywords"):
            for i, row in enumerate(edited_df):
                if i < len(st.session_state.slots):
                    primary = row["Primary Keyword"]
                    alts = [a.strip() for a in row["Alternatives"].split("|") if a.strip()]
                    st.session_state.slots[i]['keywords'] = [primary] + alts
            save_cache()
            st.success("Keywords updated!")
    # Step 5: Fetch Links
    st.header("Step 5: Fetch Video Links (Optional)")
    with st.expander("🛠️ Advanced Search Filters", expanded=False):
        c_res_map = {"None": 0, "720p": 720, "1080p": 1080, "2K": 1440, "4K": 2160}
        c_min_res_label = st.selectbox("Minimum Resolution Filter", ["None", "720p", "1080p", "2K", "4K"], 
                                     index=0, key="c_min_res_sel")
        c_min_h = c_res_map.get(c_min_res_label, 0)
    st.write("Select the number of videos per timestamp for each source:")
    col_src1, col_src2, col_src3 = st.columns(3)
    with col_src1:
        st.write("**YouTube**")
        yt_shorts = st.number_input("YT Shorts", value=0, min_value=0, max_value=5)
        yt_longs = st.number_input("YT Longs", value=3, min_value=0, max_value=5)
    with col_src2:
        st.write("**Pexels**")
        pexels_count = st.number_input("Pexels Videos", value=1, min_value=0, max_value=5)
    with col_src3:
        st.write("**Pixabay**")
        pixabay_count = st.number_input("Pixabay Videos", value=1, min_value=0, max_value=5)
    fetch_btn = st.button("Fetch Links", disabled=st.session_state.is_fetching)
    if fetch_btn:
        if not st.session_state.slots:
            st.error("Please generate keywords first.")
        elif st.session_state.is_fetching:
            st.warning("A fetch is already in progress. Please wait.")
        else:
            if not check_network():
                st.error("No network connection detected. Please check your internet and try again.")
            else:
                st.session_state.is_fetching = True
                fetch_errors = []
                try:
                    progress_bar_yt = st.progress(0)
                    status_text_yt = st.empty()
                    # First YouTube
                    if yt_shorts > 0 or yt_longs > 0:
                        status_text_yt.text("Searching YouTube...")
                        st.session_state.slots = fetch_youtube_results(
                            st.session_state.slots,
                            num_shorts=yt_shorts,
                            num_longs=yt_longs,
                            progress_callback=lambda p: progress_bar_yt.progress(p * 0.3),
                            errors=fetch_errors,
                            min_height=c_min_h
                        )
                    # Then Pexels & Pixabay
                    total_slots = len(st.session_state.slots)
                    pexels_failures = 0
                    pixabay_failures = 0
                    _CIRCUIT_LIMIT = 3
                    for idx, slot in enumerate(st.session_state.slots):
                        keywords = slot.get('keywords', [])
                        primary_kw = keywords[0] if keywords else None
                        slot['video_results'] = []
                        if 'youtube_results' in slot:
                            slot['video_results'].extend(slot['youtube_results'])
                            del slot['youtube_results']
                        if primary_kw:
                            if pexels_count > 0 and os.getenv("PEXELS_API_KEY") and pexels_failures < _CIRCUIT_LIMIT:
                                prev_err_count = len(fetch_errors)
                                pexels_res = search_pexels(primary_kw, os.getenv("PEXELS_API_KEY"), pexels_count, errors=fetch_errors, min_height=c_min_h)
                                if len(fetch_errors) > prev_err_count:
                                    pexels_failures += 1
                                    if pexels_failures >= _CIRCUIT_LIMIT:
                                        fetch_errors.append("Pexels: 3 consecutive failures —  skipping remaining Pexels searches.")
                                else:
                                    pexels_failures = 0
                                slot['video_results'].extend(pexels_res)
                            if pixabay_count > 0 and os.getenv("PIXABAY_API_KEY") and pixabay_failures < _CIRCUIT_LIMIT:
                                prev_err_count = len(fetch_errors)
                                pixabay_res = search_pixabay(primary_kw, os.getenv("PIXABAY_API_KEY"), pixabay_count, errors=fetch_errors, min_height=c_min_h)
                                if len(fetch_errors) > prev_err_count:
                                    pixabay_failures += 1
                                    if pixabay_failures >= _CIRCUIT_LIMIT:
                                        fetch_errors.append("Pixabay: 3 consecutive failures —  skipping remaining Pixabay searches.")
                                else:
                                    pixabay_failures = 0
                                slot['video_results'].extend(pixabay_res)
                        status_text_yt.text(f"Fetching stock video links... ({idx + 1}/{total_slots})")
                        progress_bar_yt.progress(0.3 + (0.7 * (idx + 1) / total_slots))
                    progress_bar_yt.progress(1.0)
                    status_text_yt.text("Done.")
                    save_cache()
                    total_results = sum(len(s.get('video_results', [])) for s in st.session_state.slots)
                    if total_results == 0 and fetch_errors:
                        st.error(f"Fetch completed but 0 results were found. All {len(fetch_errors)} searches failed.")
                    elif total_results > 0:
                        st.success(f"Fetched {total_results} video candidates across {total_slots} slots.")
                    if fetch_errors:
                        # Deduplicate by error type (strip the per-keyword prefix for grouping)
                        unique_errors = list(dict.fromkeys(fetch_errors))
                        with st.expander(f"⚠️  {len(unique_errors)} search error(s) —  click to see details"):
                            for err in unique_errors[:20]:
                                st.write(f"• {err}")
                            if len(unique_errors) > 20:
                                st.write(f"... and {len(unique_errors) - 20} more.")
                        ssl_keywords = ("ssl", "certificate", "eof occurred", "handshake", "tlsv1")
                        if any(kw in e.lower() for e in unique_errors for kw in ssl_keywords):
                            st.warning("YouTube SSL errors detected. Try running `yt-dlp -U` in your terminal to update yt-dlp, then restart the app.")
                finally:
                    st.session_state.is_fetching = False
    # Step 5.5: Review Candidates
    has_cands = any(len(s.get('video_results', [])) > 0 for s in st.session_state.slots)
    if has_cands:
        st.header("Step 5.5: Review Candidates")
        
        c_q_filter = st.selectbox(
            "Quality Filter (Stock Sources)",
            options=["All", "720p and above", "1080p and above"],
            index=0,
            key="classic_q_filter",
            help="Filter displayed candidates by minimum resolution. (YouTube candidates are always shown)."
        )
        with st.expander("🖼️  View Search Results Gallery", expanded=True):
            st.caption("Review the footage found for each keyword. Titles and thumbnails are clickable.")
            
            # Show in a grid or table? Let's do a table first for compactness.
            all_rows = []
            for i, slot in enumerate(st.session_state.slots):
                primary_kw = slot.get('keywords', ['Unknown'])[0]
                for res in slot.get('video_results', []):
                    # Apply quality filter
                    h = res.get('height') or 0
                    if c_q_filter == "720p and above" and h < 720 and res.get('source') != 'youtube':
                        continue
                    if c_q_filter == "1080p and above" and h < 1080 and res.get('source') != 'youtube':
                        continue
                    
                    title = res.get('title') or "Video"
                    url = res.get('page_url') or res.get('url') or ""
                    title_safe = str(title).replace(" ", "_").replace("'", "").replace('"', '')
                    title_link = f"{url}#title={title_safe}" if url else ""
                    
                    all_rows.append({
                        "Slot": i + 1,
                        "Keyword": primary_kw,
                        "Preview": res.get("thumbnail") or "",
                        "Title": title_link,
                        "Source": (res.get("source") or "?").upper(),
                        "Duration": f"{res.get('duration')}s" if res.get('duration') else "— ",
                    })
            
            if all_rows:
                df_review = pd.DataFrame(all_rows)
                st.data_editor(
                    df_review,
                    column_config={
                        "Slot": st.column_config.NumberColumn("#", width="small"),
                        "Keyword": st.column_config.TextColumn("Keyword", width="medium"),
                        "Preview": st.column_config.ImageColumn("Preview", width="medium"),
                        "Title": st.column_config.LinkColumn("Title", display_text=r"#title=(.*)", width="large"),
                        "Source": st.column_config.TextColumn("Source", width="small"),
                    },
                    hide_index=True,
                    width="stretch",
                    disabled=True,
                    key="classic_review_table"
                )
                
                # Optional Gallery for Classic mode (only first 12 results to avoid lag)
                st.subheader("🖼️  Quick Preview Gallery")
                g_cols = st.columns(4)
                for idx, row in df_review.head(12).iterrows():
                    with g_cols[idx % 4]:
                        thumb = row["Preview"]
                        url = row["Title"].split("#title=")[0] if "#title=" in row["Title"] else ""
                        if thumb:
                            st.markdown(
                                f'<a href="{url}" target="_blank">'
                                f'<img src="{thumb}" style="width:100%; border-radius:5px; aspect-ratio:16/9; object-fit:cover;">'
                                f'</a>',
                                unsafe_allow_html=True
                            )
                        st.caption(f"Slot {row['Slot']}: {row['Keyword'][:20]}...")
    # Step 6: Download Videos (Local)
    st.header("Step 6: Download Videos")
    col_vid1, col_vid2, col_vid3, col_vid4, col_vid5 = st.columns(5)
    with col_vid1:
        video_quality = st.selectbox(
            "Video Quality",
            ["1080p", "720p", "480p", "Best", "Worst"],
            index=0,
            format_func=_quality_label,
            help="Each option downloads the BEST stream available within that cap. '1080p' = best stream up to 1080p (not exactly 1080p).",
        )
    with col_vid2:
        strict_quality = st.checkbox("Strict Quality", value=False)
    with col_vid3:
        normalize_res = st.checkbox("Normalize (1080p)", value=True)
    with col_vid4:
        max_size_mb = st.number_input("Max Size (MB)", value=100, min_value=1)
    with col_vid5:
        max_workers = st.slider("Max Concurrent", 1, 10, 3)
    c1, c2 = st.columns(2)
    if c1.button("Start Downloading Videos"):
        if not st.session_state.slots:
            st.error("Please generate keywords and fetch video links first.")
        elif not check_network():
            st.error("No network connection detected. Please check your internet and try again.")
        else:
            # Resize the pool in place (preserves download history) rather
            # than re-instantiating the manager, which would wipe it.
            st.session_state.dm.set_max_workers(max_workers)
            st.session_state.dm.clear_and_reset()
            added_count = 0
            for i, slot in enumerate(st.session_state.slots):
                results = slot.get('video_results', slot.get('youtube_results', []))
                primary_kw = slot.get('keywords', ['Unknown'])[0]
                safe_kw = "".join([c if c.isalnum() or c in " -_" else "_" for c in primary_kw]).strip()
                for j, res in enumerate(results):
                    url = res.get('url')
                    if not url:
                        continue
                    source = res.get('source', 'youtube')
                    filename = f"{i+1}-{j+1}-{source}-{safe_kw[:20]}.mp4"
                    output_path = os.path.join("downloads", filename)
                    dm_source = 'direct' if source in ['pexels', 'pixabay'] else 'youtube'
                    try:
                        task_id = st.session_state.dm.add_download(
                            url,
                            output_path,
                            video_quality,
                            source=dm_source,
                            max_size_mb=max_size_mb,
                            strict_quality=strict_quality,
                            normalize=normalize_res
                        )
                        st.session_state.dm.start_download(task_id)
                        added_count += 1
                    except Exception as e:
                        st.error(f"Error adding {filename}: {e}")
            if added_count > 0:
                st.success(f"Added {added_count} downloads to the queue! View progress below.")
            else:
                st.warning("No video links found to download. Please run **Fetch Links** (Step 5) first to search for footage.")
    if c2.button("Cancel All Downloads"):
        st.session_state.dm.cancel_all()
    # Render Download Tasks (Dashboard View)
    tasks = st.session_state.dm.get_all_tasks()
    if tasks:
        stats = st.session_state.dm.get_stats()
        total = stats['total']
        completed = stats['completed']
        # Summary Bar
        st.progress(completed / total if total > 0 else 0.0)
        st.write(f"**Overall Progress:** {completed} / {total} Completed | Active: {stats['downloading']} | Queued: {stats['queued']} | Failed: {stats['error']}")
        # Failed Tasks Expander
        failed_tasks = st.session_state.dm.get_failed_tasks()
        if failed_tasks:
            with st.expander(f"⚠️  {len(failed_tasks)} Failed Downloads"):
                if st.button("Retry All Failed"):
                    st.session_state.dm.retry_all_failed()
                    st.rerun()
                for ft in failed_tasks:
                    st.write(f"❌ {os.path.basename(ft['output_path'])} - {ft.get('error_msg', 'Unknown Error')}")
        # Active Tasks
        active_tasks = st.session_state.dm.get_active_tasks()
        if active_tasks:
            st.subheader("Currently Downloading")
            for t in active_tasks:
                is_processing = t['status'] == 'processing'
                speed_str = format_speed(t.get('speed'))
                eta_str = format_eta(t.get('eta'))
                meta = " | ".join(filter(None, [speed_str, f"ETA {eta_str}" if eta_str else ""]))
                label = "Normalizing…" if is_processing else f"{t['status'].title()} ({t['progress']*100:.1f}%){(' —  ' + meta) if meta else ''}"
                st.write(f"**{os.path.basename(t['output_path'])}** —  {label}")
                st.progress(t['progress'])
                b1, b2, _ = st.columns(3)
                if not is_processing:
                    if t['status'] == 'downloading':
                        if b1.button("Pause", key=f"p_{t['id']}"):
                            st.session_state.dm.pause_download(t['id'])
                            st.rerun()
                    elif t['status'] == 'paused':
                        if b1.button("Resume", key=f"r_{t['id']}"):
                            st.session_state.dm.resume_download(t['id'])
                            st.rerun()
                if b2.button("Cancel", key=f"c_{t['id']}"):
                    st.session_state.dm.cancel_download(t['id'])
                    st.rerun()
            time.sleep(1)
            st.rerun()
        elif stats['queued'] > 0:
            # If tasks are queued but none are active yet, we should still loop
            time.sleep(1)
            st.rerun()
    # Step 7: Download Outputs
    st.header("Step 7: Download Output Files")
    if st.session_state.slots:
        col1, col2, col3, col4 = st.columns(4)
        keywords_txt = generate_keywords_txt(st.session_state.slots)
        yt_txt = generate_youtube_txt(st.session_state.slots)
        srt_txt = generate_srt(st.session_state.slots)
        
        trans_srt = ""
        if st.session_state.transcription_segments:
            trans_srt = generate_transcription_srt(st.session_state.transcription_segments)
            
        failed_tasks = st.session_state.dm.get_failed_tasks()
        failed_txt = ""
        if failed_tasks:
            failed_txt = generate_failed_downloads_txt(failed_tasks)
        with col1:
            st.download_button("Download keywords.txt", data=keywords_txt, file_name="keywords.txt", mime="text/plain")
            if trans_srt:
                st.download_button("Download transcription.srt", data=trans_srt, file_name="transcription.srt", mime="text/plain")
        with col2:
            st.download_button("Download youtube_results.txt", data=yt_txt, file_name="youtube_results.txt", mime="text/plain")
            if failed_txt:
                st.download_button("Download failed_downloads.txt", data=failed_txt, file_name="failed_downloads.txt", mime="text/plain")
        with col3:
            st.download_button("Download timing.srt", data=srt_txt, file_name="timing.srt", mime="text/plain")
        with col4:
            # Create ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("keywords.txt", keywords_txt)
                zip_file.writestr("youtube_results.txt", yt_txt)
                zip_file.writestr("timing.srt", srt_txt)
                if trans_srt:
                    zip_file.writestr("transcription.srt", trans_srt)
                if failed_txt:
                    zip_file.writestr("failed_downloads.txt", failed_txt)
            st.download_button(
                label="Download All (.zip)",
                data=zip_buffer.getvalue(),
                file_name="broll_finder_output.zip",
                mime="application/zip"
            )
    # Step 8: Global B-Roll Library (Advanced)
    st.header("Step 8: Global B-Roll Library (Optional)")
    st.write("Generate overarching visual themes for the entire script and gather general-purpose footage.")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        num_themes = st.number_input("Number of Themes", value=5, min_value=1, max_value=10, key="g_num_themes")
    with col_g2:
        if st.button("Analyze Global Themes"):
            if not st.session_state.script_text:
                st.error("Please upload a script first.")
            elif not os.getenv("GROQ_API_KEY"):
                st.error("Please set your Groq API key in the Setup section above.")
            else:
                from core.keywords import generate_global_themes
                with st.spinner("Analyzing themes..."):
                    try:
                        st.session_state.global_themes = generate_global_themes(st.session_state.script_text, os.getenv("GROQ_API_KEY"), num_themes=num_themes)
                        if st.session_state.global_themes:
                            st.success(f"Generated {len(st.session_state.global_themes)} global themes!")
                        else:
                            st.warning("No themes returned. Try again or check your Groq API key.")
                    except Exception as e:
                        st.error(f"Error analyzing themes: {e}")
    if st.session_state.global_themes:
        st.subheader("Global Themes & Keywords")
        
        # Flatten for table
        g_table = []
        for theme in st.session_state.global_themes:
            g_table.append({
                "Theme": theme.get('name', ''),
                "Keywords": ", ".join(theme.get('keywords', []))
            })
        
        edited_g_df = st.data_editor(g_table, num_rows="dynamic", width="stretch", key="g_editor_v2")
        
        st.subheader("Global Fetch Settings")
        col_gsrc1, col_gsrc2, col_gsrc3 = st.columns(3)
        with col_gsrc1:
            st.write("**YouTube**")
            g_yt_shorts = st.number_input("YT Shorts", value=0, min_value=0, max_value=5, key="g_yt_s")
            g_yt_longs = st.number_input("YT Longs", value=2, min_value=0, max_value=5, key="g_yt_l")
        with col_gsrc2:
            st.write("**Stock**")
            g_pexels = st.number_input("Pexels", value=2, min_value=0, max_value=5, key="g_pex")
            g_pixabay = st.number_input("Pixabay", value=2, min_value=0, max_value=5, key="g_pix")
        with col_gsrc3:
            st.write("**Quality**")
            g_vq = st.selectbox(
                "Quality",
                ["1080p", "720p", "480p", "Best", "Worst"],
                index=0,
                key="g_vq",
                format_func=_quality_label,
                help="Each option downloads the BEST stream available within that cap. '1080p' = best stream up to 1080p (not exactly 1080p).",
            )
            g_strict = st.checkbox("Strict Quality", value=False, key="g_strict")
            g_max_size = st.number_input("Max Size (MB)", value=100, min_value=1, key="g_size")
        c_gf1, c_gf2 = st.columns(2)
        
        if c_gf1.button("Fetch Global Video Links", disabled=st.session_state.is_fetching):
            if st.session_state.is_fetching:
                st.warning("A fetch is already in progress. Please wait.")
            elif not check_network():
                st.error("No network connection detected. Please check your internet and try again.")
            else:
                st.session_state.is_fetching = True
                fetch_errors_g = []
                try:
                    new_themes = []
                    for row in edited_g_df:
                        kws = [k.strip() for k in row["Keywords"].split(",") if k.strip()]
                        new_themes.append({"name": row["Theme"], "keywords": kws, "video_results": []})
                    st.session_state.global_themes = new_themes
                    pbar_g = st.progress(0)
                    status_g = st.empty()
                    from core.youtube import search_youtube_single
                    total_g = len(st.session_state.global_themes)
                    found_total = 0
                    pexels_failures = 0
                    pixabay_failures = 0
                    _CIRCUIT_LIMIT = 3
                    for idx, theme in enumerate(st.session_state.global_themes):
                        status_g.text(f"Fetching links for Theme: {theme['name']}...")
                        primary_kw = theme['keywords'][0] if theme.get('keywords') else None
                        if not primary_kw:
                            pbar_g.progress((idx + 1) / total_g)
                            continue
                        results = []
                        if g_yt_shorts > 0 or g_yt_longs > 0:
                            yt_res = search_youtube_single(primary_kw, num_shorts=g_yt_shorts, num_longs=g_yt_longs, errors=fetch_errors_g)
                            for r in yt_res:
                                r['source'] = 'youtube'
                            results.extend(yt_res)
                        if g_pexels > 0 and os.getenv("PEXELS_API_KEY") and pexels_failures < _CIRCUIT_LIMIT:
                            prev = len(fetch_errors_g)
                            pex_res = search_pexels(primary_kw, os.getenv("PEXELS_API_KEY"), g_pexels, errors=fetch_errors_g)
                            if len(fetch_errors_g) > prev:
                                pexels_failures += 1
                                if pexels_failures >= _CIRCUIT_LIMIT:
                                    fetch_errors_g.append("Pexels: 3 consecutive failures —  skipping remaining Pexels searches.")
                            else:
                                pexels_failures = 0
                            results.extend(pex_res)
                        if g_pixabay > 0 and os.getenv("PIXABAY_API_KEY") and pixabay_failures < _CIRCUIT_LIMIT:
                            prev = len(fetch_errors_g)
                            pix_res = search_pixabay(primary_kw, os.getenv("PIXABAY_API_KEY"), g_pixabay, errors=fetch_errors_g)
                            if len(fetch_errors_g) > prev:
                                pixabay_failures += 1
                                if pixabay_failures >= _CIRCUIT_LIMIT:
                                    fetch_errors_g.append("Pixabay: 3 consecutive failures —  skipping remaining Pixabay searches.")
                            else:
                                pixabay_failures = 0
                            results.extend(pix_res)
                        theme['video_results'] = results
                        found_total += len(results)
                        pbar_g.progress((idx + 1) / total_g)
                    status_g.text("Done.")
                    if found_total > 0:
                        st.success(f"Fetched {found_total} global video links across {total_g} themes.")
                    else:
                        st.warning("No video links found. Try different keywords or check your API keys.")
                    if fetch_errors_g:
                        unique_g_errors = list(dict.fromkeys(fetch_errors_g))
                        with st.expander(f"⚠️  {len(unique_g_errors)} search error(s)"):
                            for err in unique_g_errors[:20]:
                                st.write(f"• {err}")
                        ssl_keywords = ("ssl", "certificate", "eof occurred", "handshake", "tlsv1")
                        if any(kw in e.lower() for e in unique_g_errors for kw in ssl_keywords):
                            st.warning("YouTube SSL errors detected. Try running `yt-dlp -U` in your terminal to update yt-dlp, then restart the app.")
                    save_cache()
                finally:
                    st.session_state.is_fetching = False
        # Global Review Section
        has_g_results = any(len(t.get('video_results', [])) > 0 for t in st.session_state.global_themes)
        if has_g_results:
            st.subheader("🌍  Global Footage Review")
            
            # Global Gallery Section
            st.subheader("🖼️  Global Gallery")
            
            g_q_filter = st.selectbox(
                "Filter Global by Quality",
                options=["All", "720p and above", "1080p and above"],
                index=0,
                key="global_q_filter_gal_fix"
            )
            if 'picked_global_urls' not in st.session_state:
                st.session_state.picked_global_urls = set()
            g_all_cands = []
            for theme in st.session_state.global_themes:
                for res in theme.get('video_results', []):
                    # Apply quality filter
                    h = res.get('height') or 0
                    if g_q_filter == "720p and above" and h < 720 and res.get('source') != 'youtube':
                        continue
                    if g_q_filter == "1080p and above" and h < 1080 and res.get('source') != 'youtube':
                        continue
                    g_all_cands.append((theme, res))
            st.caption(f"**{len(st.session_state.picked_global_urls)} selected** · Showing {len(g_all_cands)} results.")
            gn_cols = 3
            for i_g in range(0, len(g_all_cands), gn_cols):
                gcols = st.columns(gn_cols)
                for j_g in range(gn_cols):
                    if i_g + j_g < len(g_all_cands):
                        theme, res = g_all_cands[i_g + j_g]
                        thumb = res.get('thumbnail')
                        url = res.get('page_url') or res.get('url')
                        source = res.get('source', '').upper()
                        w, h = res.get('width'), res.get('height')
                        dur = res.get('duration')
                        avail_res = res.get("available_resolutions", [])
                        if avail_res:
                            high_res = [r for r in [2160, 1440, 1080, 720] if r in avail_res]
                            res_str = f"{w}x{h}" if w and h else f"{max(avail_res)}p"
                            if high_res:
                                res_str += f" ({', '.join([('4K' if r==2160 else '2K' if r==1440 else str(r)+'p') for r in high_res])})"
                        else:
                            res_str = f"{w}x{h}" if w and h else "Resolution Unknown"
                        dur_str = f"{dur}s" if dur else ""
                        with gcols[j_g]:
                            with st.container(border=True):
                                if thumb:
                                    st.markdown(
                                        f'<div style="position: relative;">' 
                                        f'<img src="{thumb}" style="width:100%; border-radius:4px; aspect-ratio:16/9; object-fit:cover; border:1px solid #444;">' 
                                        f'<div style="position: absolute; top: 5px; right: 5px;">' 
                                        f'<a href="{url}" target="_blank" style="text-decoration: none; background: rgba(0,0,0,0.7); padding: 2px 8px; border-radius: 4px; color: white; font-size: 12px; font-weight: bold;">📺 WATCH</a>' 
                                        f'</div>' 
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                                is_g_picked = res.get('url') in st.session_state.picked_global_urls
                                g_btn_label = "✅ SELECTED" if is_g_picked else "⬜ PICK CLIP"
                                if st.button(g_btn_label, key=f"gbtn_{hash(res.get('url'))}", width="stretch", type="primary" if is_g_picked else "secondary"):
                                    toggle_global_pick(res.get('url'))
                                    st.rerun()
                                st.markdown(f"**{source}** · {res_str} · {dur_str}")
                                st.caption(f"{theme['name']}: {res.get('title', 'Video')[:40]}...")
        if c_gf2.button("Start Downloading Global Footage"):
            # Update state from editor just in case keywords were changed, but DON'T clear results
            # if we have them. This is tricky. Let's just use what's in session state.
            
            added_count = 0
            has_any_results = any(len(t.get('video_results', [])) > 0 for t in st.session_state.global_themes)
            
            if not has_any_results:
                st.error("No links have been fetched yet. Please click 'Fetch Global Video Links' first.")
            else:
                # Deduplicate and filter by st.session_state.picked_global_urls
                for theme in st.session_state.global_themes:
                    results = theme.get('video_results', [])
                    if not results: continue
                    
                    for j, res in enumerate(results):
                        url = res.get('url')
                        if not url or url not in st.session_state.picked_global_urls:
                            continue
                        
                        source = res.get('source', 'stock')
                        safe_theme = "".join([c if c.isalnum() or c in " -_" else "_" for c in theme['name']]).strip()
                        # Shorten name for file safety
                        filename = f"Global-{safe_theme[:15]}-{source}-{j+1}.mp4"
                        output_path = os.path.join("downloads", "global", filename)
                        
                        dm_source = 'direct' if source in ['pexels', 'pixabay'] else 'youtube'
                        try:
                            task_id = st.session_state.dm.add_download(
                                url,
                                output_path,
                                g_vq,
                                source=dm_source,
                                max_size_mb=g_max_size,
                                strict_quality=g_strict,
                                normalize=True
                            )
                            st.session_state.dm.start_download(task_id)
                            added_count += 1
                        except Exception as e:
                            st.error(f"Error adding {filename}: {e}")
                
                if added_count > 0:
                    st.success(f"Added {added_count} global downloads to the queue!")
                else:
                    st.warning("No footage was added. All links might have been filtered by quality/size settings.")
if app_mode == "Classic Finder":
    render_classic_mode()
elif app_mode in ["Director", "Smart Mode"]:
    ensure_shots_have_chunk_ids()
    from core.director_search import fetch_director_footage, clear_query_cache
    from core.director_rank import rank_shot_candidates
    from core.director_youtube import generate_youtube_keywords_for_shots, seed_youtube_keywords
    from core.output import generate_fcpxml, generate_shot_list_txt, _safe_for_fs
    st.title(f"🎬 B-Roll {app_mode}")
    
    if app_mode == "Smart Mode":
        from core.indexer import VideoIndexer
        from core.smart_search import SmartSearch
        indexer = VideoIndexer()
        searcher = SmartSearch()
        
        st.info("💡 **The AI searches for visual meaning, not just keywords.**")
        with st.expander("📦 AI Visual Search Manager (Feed the AI)", expanded=False):
            st.markdown("### Build Your Foundation")
            st.caption("Point the AI to your local footage or high-quality YouTube compilations to build your cinematic brain.")
            
            tab_yt, tab_local = st.tabs(["Bulk YouTube", "Local Folder"])
            
            with tab_yt:
                lib_urls = st.text_area("Paste YouTube URLs (one per line)", placeholder="https://youtube.com/watch?v=...\nhttps://youtube.com/watch?v=...", height=100, key="lib_urls_bulk")
                if st.button("🚀 Start Bulk Ingestion", width="stretch"):
                    urls = [u.strip() for u in lib_urls.split("\n") if u.strip()]
                    if urls:
                        with st.status(f"Ingesting {len(urls)} videos...") as status:
                            for i, url in enumerate(urls):
                                def _prog(p, msg): status.update(label=f"[{i+1}/{len(urls)}] {msg}", state="running")
                                count = indexer.download_and_index_youtube(url, progress_cb=_prog)
                                st.write(f"✅ Indexed {count} scenes from video {i+1}")
                            status.update(label=f"Done! Bulk ingestion complete.", state="complete")
                            st.rerun()
                    else:
                        st.warning("Paste some URLs first.")
            
            with tab_local:
                local_dir = st.text_input("Local Folder Path", placeholder="C:/Users/Name/Videos/MyStock", key="lib_local_dir")
                if st.button("ðŸâ€   Index Local Folder", width="stretch"):
                    if os.path.isdir(local_dir):
                        video_files = [f for f in os.listdir(local_dir) if f.lower().endswith(('.mp4', '.mov', '.mkv', '.avi'))]
                        if video_files:
                            with st.status(f"Indexing {len(video_files)} local videos...") as status:
                                for i, vf in enumerate(video_files):
                                    v_path = os.path.join(local_dir, vf)
                                    def _prog(p, msg): status.update(label=f"[{i+1}/{len(video_files)}] {msg}", state="running")
                                    count = indexer.index_video(v_path, progress_cb=_prog)
                                    st.write(f"✅ Indexed {count} scenes from {vf}")
                                status.update(label="Local indexing complete.", state="complete")
                                st.rerun()
                        else:
                            st.error("No video files found in that folder.")
                    else:
                        st.error("Invalid folder path.")
            
            st.divider()
            stats = searcher.get_library_stats()
            st.caption(f"Library Status: **{stats['unique_videos']}** videos | **{stats['total_segments']}** indexed scenes with Scene-Level Understanding.")
            if st.button("🗑️👀 Clear Library", type="secondary"):
                indexer.db.clear()
                st.success("Library cleared.")
                st.rerun()
    # ── Setup: API keys (collapsible, auto-collapses once Groq key is present) ──
    _key_status = {
        "Groq 1":     bool(os.getenv("GROQ_API_KEY")),
        "Groq 2":     bool(os.getenv("GROQ_API_KEY_2")),
        "Pexels":     bool(os.getenv("PEXELS_API_KEY")),
        "Pixabay":    bool(os.getenv("PIXABAY_API_KEY")),
        "YouTube":    bool(os.getenv("YOUTUBE_API_KEY")),
        "OpenRouter": bool(os.getenv("OPENROUTER_API_KEY")),
        "Freesound":  bool(os.getenv("FREESOUND_API_KEY")),
    }
    _pill = " · ".join(f"{'✅' if v else '○'} {k}" for k, v in _key_status.items())
    with st.expander(f"⚙️  Setup —  API keys  ·  {_pill}",
                     expanded=not (_key_status["Groq 1"] or _key_status["Groq 2"])):
        st.caption(
            "Groq is required (script analysis & ranking). Pexels/Pixabay/YouTube are "
            "search sources —  enable at least one. OpenRouter is an automatic fallback "
            "when Groq hits its rate limit."
        )
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            groq_input    = st.text_input("Groq API Key 1 (required)",
                                          value=os.getenv("GROQ_API_KEY", ""),
                                          type="password", key="d_groq")
            groq_input_2  = st.text_input("Groq API Key 2 (fallback)",
                                          value=os.getenv("GROQ_API_KEY_2", ""),
                                          type="password", key="d_groq_2",
                                          help="Secondary Groq key to use when the first one hits rate limits.")
            pexels_input  = st.text_input("Pexels API Key",
                                          value=os.getenv("PEXELS_API_KEY", ""),
                                          type="password", key="d_pex")
            pixabay_input = st.text_input("Pixabay API Key",
                                          value=os.getenv("PIXABAY_API_KEY", ""),
                                          type="password", key="d_pix")
        with col_k2:
            youtube_input    = st.text_input("YouTube Data API Key",
                                             value=os.getenv("YOUTUBE_API_KEY", ""),
                                             type="password", key="d_yt",
                                             help="Optional. 10,000 quota units/day; each search costs 100.")
            openrouter_input = st.text_input("OpenRouter API Key",
                                             value=os.getenv("OPENROUTER_API_KEY", ""),
                                             type="password", key="d_or",
                                             help="Optional fallback when Groq returns rate-limit errors.")
            openrouter_2_input = st.text_input("OpenRouter Key 2 (Fallback)",
                                               value=os.getenv("OPENROUTER_API_KEY_2", ""),
                                               type="password", key="d_or_2",
                                               help="Second fallback key if the first OpenRouter key is also limited.")
            freesound_input  = st.text_input("Freesound API Key",
                                             value=os.getenv("FREESOUND_API_KEY", ""),
                                             type="password", key="d_free",
                                             help="Required for automatic sound effects downloads.")
            st.markdown("&nbsp;")  # vertical spacer to align rows
        _browser_options_d = ["None", "chrome", "firefox", "edge", "brave", "safari", "opera", "chromium"]
        _current_browser_d = os.getenv("YT_COOKIE_BROWSER", "None")
        _browser_idx_d = _browser_options_d.index(_current_browser_d) if _current_browser_d in _browser_options_d else 0
        cookie_browser_d = st.selectbox(
            "YouTube Cookie Browser",
            options=_browser_options_d,
            index=_browser_idx_d,
            key="d_cookie_browser",
            help=(
                "Fixes 'Sign in to confirm you're not a bot' errors. "
                "Select the browser where you are logged into YouTube — yt-dlp will "
                "borrow its cookies for authentication. The browser must be installed "
                "and you must be logged into YouTube in it."
            ),
        )
        if st.button("Save API Keys", key="d_save_keys", type="primary"):
            if groq_input:
                set_key(ENV_FILE, "GROQ_API_KEY", groq_input)
                if groq_input_2:      set_key(ENV_FILE, "GROQ_API_KEY_2",      groq_input_2)
                if pexels_input:     set_key(ENV_FILE, "PEXELS_API_KEY",     pexels_input)
                if pixabay_input:    set_key(ENV_FILE, "PIXABAY_API_KEY",    pixabay_input)
                if youtube_input:    set_key(ENV_FILE, "YOUTUBE_API_KEY",    youtube_input)
                if openrouter_input: set_key(ENV_FILE, "OPENROUTER_API_KEY", openrouter_input)
                if openrouter_2_input: set_key(ENV_FILE, "OPENROUTER_API_KEY_2", openrouter_2_input)
                if freesound_input:  set_key(ENV_FILE, "FREESOUND_API_KEY",  freesound_input)
                set_key(ENV_FILE, "YT_COOKIE_BROWSER", cookie_browser_d)
                load_dotenv(ENV_FILE, override=True)
                st.success("API keys saved to .env. Refresh the page to re-evaluate the status pill.")
            else:
                st.warning("Groq API key is required.")
    st.header("Step 1: Upload & Transcribe")
    with st.container(border=True):
        audio_file = st.file_uploader(
            "Voiceover audio (.mp3 / .wav / .m4a)",
            type=["mp3", "wav", "m4a"],
            key="d_audio",
            help="Upload your voiceover. The AI will transcribe it and generate a shot list for the whole video in one go.",
        )
        if audio_file:
            # ── Project Isolation & Auto-Reset ───────────────────────────
            # Track the audio filename separately from the project name so
            # the user can rename the project without nuking shots/transcription.
            # Auto-reset only fires when a *different* audio file is uploaded.
            current_audio = audio_file.name
            audio_changed = bool(
                st.session_state.get("source_audio_name")
                and st.session_state.source_audio_name != current_audio
            )
            if audio_changed:
                st.session_state.director_shots = []
                st.session_state.transcription_segments = []
                st.session_state.transcription_chunks = []
                st.session_state.active_chunk_indices = [0]
                st.session_state.script_text = ""
                st.session_state.d_review_idx = 0
            st.session_state.source_audio_name = current_audio
            # Seed project_name from the audio filename on first load or
            # whenever the audio file changes. Set BEFORE the text_input so
            # Streamlit picks it up as the initial value for the key="project_name"
            # widget; after the widget renders, only the user controls the value.
            if audio_changed or not st.session_state.get("project_name"):
                st.session_state.project_name = os.path.splitext(current_audio)[0]

            audio_path = os.path.join(
                ".cache", "temp_audio_director" + os.path.splitext(audio_file.name)[1]
            )
            with open(audio_path, "wb") as f:
                f.write(audio_file.read())
            st.session_state.audio_duration = get_audio_duration(audio_path)

            # ── Project Name (editable) ──────────────────────────────────
            proj_a, proj_b = st.columns([3, 2])
            with proj_a:
                st.text_input(
                    "Project name",
                    key="project_name",
                    help="Used to organize downloaded media, exports, and Clip Library entries. "
                         "Type whatever you like — it gets sanitized to a folder-safe slug for disk paths.",
                )
            with proj_b:
                folder_slug = _safe_for_fs(st.session_state.get("project_name", ""), 50) or "default"
                st.caption(f"📁 `downloads/director/{folder_slug}/`")
                if folder_slug == "default":
                    st.caption("⚠️ Empty project name — files will go to the shared `default/` folder.")

            top_a, top_b, top_c = st.columns([2, 2, 3])
            with top_a:
                st.metric("Duration", f"{st.session_state.audio_duration / 60:.1f} min")
            with top_b:
                seg_count = len(st.session_state.get("transcription_segments", []))
                st.metric("Segments", str(seg_count) if seg_count else "—")
            with top_c:
                st.audio(audio_path)
            cta_a, cta_b = st.columns([3, 1])
            with cta_a:
                if st.button("🎙️ Transcribe", key="d_transcribe", type="primary",
                             width="stretch"):
                    if not os.getenv("GROQ_API_KEY"):
                        st.error("Set Groq API Key in Setup above.")
                    else:
                        from core.transcription import transcribe_audio
                        with st.spinner("Transcribing with Whisper…"):
                            try:
                                segments = transcribe_audio(audio_path, os.getenv("GROQ_API_KEY"))
                                st.session_state.transcription_segments = segments
                                st.session_state.script_text = " ".join(
                                    s["text"].strip() for s in segments
                                ).strip()
                                # Store as a single chunk containing all segments
                                st.session_state.transcription_chunks = [{
                                    "start":    segments[0]["start"] if segments else 0.0,
                                    "end":      segments[-1]["end"]  if segments else 0.0,
                                    "text":     st.session_state.script_text,
                                    "segments": segments,
                                }]
                                st.session_state.active_chunk_indices = [0]
                                save_cache()
                                st.success(
                                    f"Transcribed —  {len(segments)} segment(s). "
                                    "Continue to Step 2 to generate the shot list."
                                )
                                st.rerun()
                            except Exception as e:
                                st.error(f"Transcription failed: {e}")
            with cta_b:
                if st.session_state.get("transcription_segments"):
                    if st.button("Reset", key="d_reset_transcription",
                                 width="stretch",
                                 help="Discard the current transcription."):
                        st.session_state.transcription_segments = []
                        st.session_state.transcription_chunks   = []
                        st.session_state.active_chunk_indices   = [0]
                        st.session_state.script_text            = ""
                        save_cache()
                        st.rerun()
        else:
            st.caption("Upload your voiceover to begin.")
    # Show a brief stats line when transcription is ready (no chunk picker needed)
    chunks = st.session_state.get("transcription_chunks", [])
    if chunks:
        segs = st.session_state.get("transcription_segments", [])
        dur  = chunks[0]["end"] - chunks[0]["start"]
        wds  = len(chunks[0]["text"].split())
        wps  = wds / dur if dur > 0 else 0
        s1, s2, s3 = st.columns(3)
        with s1: st.metric("Length", f"{dur/60:.1f} min")
        with s2: st.metric("Words",  f"{wds:,}")
        with s3: st.metric("Speaking rate", f"{wps:.2f} wps")
        with st.expander("📄  View full transcription", expanded=False):
            for seg in segs:
                m = int(seg["start"] // 60); s = int(seg["start"] % 60)
                st.write(f"**[{m:02d}:{s:02d}]** {seg['text']}")
    st.header("Step 2: Generate Shot List")
    # Video topic —  single source of truth shared with Step 4 (ranking).
    # Editing it here updates Step 4, and vice versa, because Streamlit
    # binds widgets with the same `key` to one session_state slot.
    col_topic_in, col_topic_btn = st.columns([5, 1])
    # IMPORTANT: We check the button BEFORE rendering the text_input widget
    # so we can update its session_state key without triggering the 
    # "cannot be modified after instantiation" error.
    with col_topic_btn:
        st.markdown("<div style='padding-top: 1.7rem;'></div>", unsafe_allow_html=True)
        if st.button("✨ Suggest", key="d_suggest_topic", width="stretch", help="AI analyzes your script to suggest a topic description."):
            if not st.session_state.get("script_text"):
                st.error("Upload audio and transcribe first.")
            elif not os.getenv("GROQ_API_KEY"):
                st.error("Set Groq API key in Setup at the top.")
            else:
                from core.keywords import generate_video_topic
                with st.spinner("Analyzing..."):
                    try:
                        suggested = generate_video_topic(st.session_state.script_text, os.getenv("GROQ_API_KEY"))
                        if suggested:
                            st.session_state.d_video_topic = suggested
                            st.rerun()
                        else:
                            st.warning("Could not generate a suggestion. Try again.")
                    except Exception as e:
                        st.error(f"Error: {e}")
    with col_topic_in:
        d_video_topic = st.text_input(
            "What is this video about? (sharpens both query generation and ranking)",
            placeholder="e.g. car mechanics and engine repair",
            key="d_video_topic",
            on_change=save_cache,
            help="One sentence describing the topic. Used to disambiguate shot queries (so \"tool\" in a car video means \"wrench\", not \"saw\") and to filter off-topic candidates during ranking."
        )
    custom_instructions = st.text_area("Style Hints (Optional)", placeholder="e.g. cinematic, slow motion, no talking heads", key="d_style", on_change=save_cache)

    # Pipeline mode toggles (persist to .env so os.getenv sees them everywhere).
    col_ctx, col_auto = st.columns(2)
    with col_ctx:
        _enable_ctx = st.checkbox(
            "🧭 Context-aware keywords",
            value=_env_flag("ENABLE_CONTEXT_AWARE_KEYWORDS"),
            key="d_ctx_aware",
            help="Pre-scan the whole transcript into subject segments, then anchor each "
                 "shot's queries to the subject in play at that moment (e.g. keep \"BMW M3\" "
                 "in mind while the script discusses its transmission). Adds one fast LLM "
                 "pass before the shot list is built.",
        )
        if _enable_ctx != _env_flag("ENABLE_CONTEXT_AWARE_KEYWORDS"):
            _set_env_flag("ENABLE_CONTEXT_AWARE_KEYWORDS", _enable_ctx)
    with col_auto:
        _enable_auto = st.checkbox(
            "🤖 Auto-select best clip",
            value=_env_flag("ENABLE_AUTO_SELECTION"),
            key="d_auto_select",
            help="After ranking (Step 4), automatically bind the top-ranked candidate to "
                 "each shot instead of leaving every pick for manual review. You can still "
                 "override any auto-pick in Step 5.",
        )
        if _enable_auto != _env_flag("ENABLE_AUTO_SELECTION"):
            _set_env_flag("ENABLE_AUTO_SELECTION", _enable_auto)

    if "director_shots" not in st.session_state:
        st.session_state.director_shots = []
    if st.button("Generate Shot List", type="primary"):
        if not os.getenv("GROQ_API_KEY"):
            st.error("Set Groq API Key in Setup at the top.")
        elif not st.session_state.get("transcription_segments"):
            st.error("Upload audio and click *Transcribe* in Step 1 first.")
        else:
            try:
                from core.director import generate_shot_list_from_transcription
                segments = st.session_state.transcription_segments
                pbar = st.progress(0)
                status = st.empty()

                # Phase-1 context-aware pre-pass: map the transcript into subject
                # segments so the Director can anchor localized queries to the
                # macro-subject. Degrades silently to the flat path on failure.
                roadmap = {}
                if _env_flag("ENABLE_CONTEXT_AWARE_KEYWORDS"):
                    status.text("Mapping video structure (context-aware pre-pass)…")
                    from core.director import segment_script_structure
                    roadmap = segment_script_structure(
                        segments, os.getenv("GROQ_API_KEY"), video_topic=d_video_topic
                    )
                    st.session_state.segment_roadmap = roadmap
                    if roadmap.get("segments"):
                        st.caption(
                            f"🧭 Detected {len(roadmap['segments'])} subject segment(s)"
                            + (f" — overall: **{roadmap.get('video_global_subject')}**"
                               if roadmap.get("video_global_subject") else "")
                        )

                status.text("Finalizing shot list...")
                shots = generate_shot_list_from_transcription(
                    segments,
                    os.getenv("GROQ_API_KEY"),
                    custom_instructions=custom_instructions,
                    video_topic=d_video_topic,
                    chunk_id=0,
                    segment_roadmap=roadmap or None,
                    progress_callback=lambda p: pbar.progress(p),
                )
                # Sequential slot IDs, no chunk prefix
                for idx, shot in enumerate(shots):
                    shot["slot_id"] = idx + 1
                    shot.pop("chunk_id", None)
                
                if shots:
                    st.session_state.director_shots = shots
                    pbar.progress(1.0)
                    status.empty()
                    st.success(f"Shot list generated — {len(shots)} shot(s).")
                    save_cache()
                else:
                    status.empty()
                    st.warning("The AI returned no shots for this transcription. Check your Groq API key or try again.")
            except Exception as e:
                st.error(f"Error generating shot list: {e}")

    if st.session_state.director_shots:
        st.subheader("Shot List")
        st.caption(
            "Edit **Intent**, **Priority**, or **Queries** inline. After editing queries, "
            "re-click *Fetch Candidates* in Step 3 to refresh results. Time, candidate count, "
            "and pick count are read-only."
        )
        _fallback_shots = [
            s for s in st.session_state.director_shots if s.get("queries_fallback")
        ]
        if _fallback_shots:
            _ids = ", ".join(str(s.get("slot_id")) for s in _fallback_shots[:15])
            st.info(
                f"ℹ️ {len(_fallback_shots)} shot(s) had no queries from the AI and were "
                f"auto-filled from their intent/topic (shots {_ids}). They'll still fetch, "
                "but **Regenerate Queries** in Step 3 usually produces better visual searches."
            )
        # Build rows from current state.
        rows = []
        for shot in st.session_state.director_shots:
            sel = shot.get("selected_results", [])
            # Highlight missing selections with a warning emoji if priority isn't 'none'
            if sel:
                pick_status = f"🤖 {len(sel)}" if shot.get("auto_selected") else f"✅ {len(sel)}"
            else:
                pick_status = "⏭ skip" if shot.get("skipped") else "—"
            if not sel and shot.get("priority") != "none" and not shot.get("skipped"):
                pick_status = "🔖 FLAGGED" if shot.get("flagged") else "⚠️ MISSING"
            rows.append({
                "#":          shot.get("slot_id", 0),
                "Time":       f"{shot.get('timestamp_start_str')} – {shot.get('timestamp_end_str')}",
                "Intent":     shot.get("shot_intent", ""),
                "Priority":   shot.get("priority", "medium"),
                "Queries":    " | ".join(shot.get("search_queries", [])),
                "YT Keywords": " | ".join(shot.get("youtube_keywords", [])),
                "Cand":       len(shot.get("video_results", [])),
                "Picked":     pick_status,
            })
        df_shots = pd.DataFrame(rows)
        edited_shots = st.data_editor(
            df_shots,
            column_config={
                "#":        st.column_config.NumberColumn("#", width="small", help="Slot ID"),
                "Time":     st.column_config.TextColumn("Time", width="small"),
                "Intent":   st.column_config.TextColumn("Intent", width="medium",
                              help="Short phrase describing what this moment must convey visually."),
                "Priority": st.column_config.SelectboxColumn("Priority",
                              options=["high", "medium", "low", "none"],
                              required=True, width="small",
                              help="'none' means talking-head — no B-roll is fetched for this shot."),
                "Queries":  st.column_config.TextColumn("Queries (Stock)", width="medium",
                              help="Used for Pexels/Pixabay. Separate with ' | '."),
                "YT Keywords": st.column_config.TextColumn("Keywords (YouTube)", width="medium",
                              help="Used for YouTube search. Separate with ' | '."),
                "Cand":     st.column_config.NumberColumn("Cand", width="small",
                              help="Number of candidates fetched for this shot."),
                "Picked":   st.column_config.TextColumn("Picked", width="small",
                              help="Shows if you've selected footage for this shot yet."),
            },
            disabled=["#", "Time", "Cand", "Picked"],
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            key="d_shotlist_editor",
        )
        # Sync edits back to director_shots only when something actually changed,
        # so we don't churn save_cache on every rerun.
        changed = False
        for i, shot in enumerate(st.session_state.director_shots):
            if i >= len(edited_shots):
                continue
            row = edited_shots.iloc[i]
            new_intent  = str(row["Intent"] or "").strip()
            new_pri     = str(row["Priority"] or "medium").strip().lower()
            if new_pri not in ("high", "medium", "low", "none"):
                new_pri = "medium"
            new_queries_str = str(row["Queries"] or "")
            new_queries    = [q.strip() for q in new_queries_str.split("|") if q.strip()]
            # 'none' shots keep search_queries empty regardless of what was typed.
            if new_pri == "none":
                new_queries = []
            if shot.get("shot_intent", "") != new_intent:
                shot["shot_intent"] = new_intent
                changed = True
            if shot.get("priority", "medium") != new_pri:
                shot["priority"] = new_pri
                changed = True
            if shot.get("search_queries", []) != new_queries:
                shot["search_queries"] = new_queries
                changed = True
            
            # Sync YouTube Keywords
            new_yt_str = str(row.get("YT Keywords", ""))
            new_yt     = [q.strip() for q in new_yt_str.split("|") if q.strip()]
            if shot.get("youtube_keywords", []) != new_yt:
                shot["youtube_keywords"] = new_yt
                changed = True
        if changed:
            save_cache()

        st.caption("💡 **YouTube Keyword Tools:** Use these to bulk-fill the YouTube Keywords column in the table above.")
        yk1, yk2, yk3 = st.columns([2, 2, 6])
        with yk1:
            if st.button("✨ Seed from Queries", key="d_seed_yt_keywords", width="stretch"):
                st.session_state.director_shots = seed_youtube_keywords(
                    st.session_state.director_shots,
                    max_keywords=2,
                )
                save_cache()
                st.rerun()
        with yk2:
            if st.button("🤖 Generate with AI", key="d_gen_yt_keywords", width="stretch"):
                if not os.getenv("GROQ_API_KEY"):
                    st.error("Groq API key required.")
                else:
                    pbar_yk = st.progress(0)
                    try:
                        from core.director_youtube import generate_youtube_keywords_for_shots
                        st.session_state.director_shots = generate_youtube_keywords_for_shots(
                            st.session_state.director_shots,
                            api_key=os.getenv("GROQ_API_KEY"),
                            video_topic=st.session_state.get("d_video_topic", ""),
                            custom_instructions=st.session_state.get("d_style", ""),
                            max_keywords=2,
                            progress_callback=lambda p: pbar_yk.progress(p),
                        )
                        pbar_yk.progress(1.0)
                        save_cache()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        with yk3:
            st.caption("Keywords are used only for YouTube search. Pexels/Pixabay use Stock Queries.")
    # Step 2.5: optional YouTube-specific keywords for Classic-style search.
    # Step 3 — Fetch Candidates
    if st.session_state.director_shots:
        st.header("Step 3: Fetch Candidates")
        
        with st.expander("🛠️ Advanced Search Filters", expanded=True):
            res_map = {"None": 0, "720p": 720, "1080p": 1080, "2K": 1440, "4K": 2160}
            min_res_label = st.selectbox("Minimum Resolution Filter", ["None", "720p", "1080p", "2K", "4K"], 
                                         index=0, key="d_min_res_sel", 
                                         help="Videos below this resolution will be skipped and replaced with next best matches.")
            min_h = res_map.get(min_res_label, 0)

        col_s2a, col_s2b, col_s2c, col_s2d, col_s2e = st.columns(5)
        with col_s2a:
            use_pexels  = st.checkbox("Pexels",  value=bool(os.getenv("PEXELS_API_KEY")),  key="d_pex_cb", on_change=save_cache)
            pex_num = st.number_input("Results/query", value=3, min_value=1, max_value=10, key="d_pex_nr", on_change=save_cache) if use_pexels else 0
        with col_s2b:
            use_pixabay = st.checkbox("Pixabay", value=bool(os.getenv("PIXABAY_API_KEY")), key="d_pix_cb", on_change=save_cache)
            pix_num = st.number_input("Results/query", value=3, min_value=1, max_value=10, key="d_pix_nr", on_change=save_cache) if use_pixabay else 0
        with col_s2c:
            use_youtube_api = st.checkbox(
                "YouTube API",
                value=bool(os.getenv("YOUTUBE_API_KEY")),
                key="d_yt_api_cb",
                on_change=save_cache,
                help=(
                    "Adds YouTube as a search source. Each YT search costs "
                    "100 quota units (default daily quota: 10,000), so to stay "
                    "within budget the director only runs ONE YouTube search "
                    "per shot — using the first query. Pexels and Pixabay still "
                    "run all queries. Selected YT clips download via yt-dlp."
                ),
            )
            yt_api_num = st.number_input("Results/query", value=3, min_value=1, max_value=10, key="d_ytapi_nr", on_change=save_cache) if use_youtube_api else 0
        with col_s2d:
            use_youtube_search = st.checkbox(
                "YouTube Search",
                value=True,
                key="d_yt_search_cb",
                on_change=save_cache,
                help=(
                    "Uses Classic Finder-style yt-dlp search and the Step 2.5 "
                    "YouTube keywords. No YouTube Data API quota used."
                ),
            )
            yt_search_num = st.number_input("Results/query", value=3, min_value=1, max_value=10, key="d_yts_nr", on_change=save_cache) if use_youtube_search else 0
        with col_s2e:
            use_smart = st.checkbox("Smart Library", value=False, key="d_smart_cb", disabled=True, help="Smart Library is currently disabled.")
            smart_num = st.number_input("Results/query", value=5, min_value=1, max_value=20, key="d_smart_nr") if use_smart else 0
        col_s2f, = st.columns(1)
        with col_s2f:
            _lib_total = get_library_stats().get("total", 0)
            use_library = st.checkbox(
                f"📚 Clip Library ({_lib_total} clips)",
                value=_lib_total > 0,
                key="d_lib_cb",
                help="Search your saved clip database first. Clips appear at the top of results with a 📚 badge.",
            )
            lib_num = st.number_input("Top results", value=5, min_value=1, max_value=20, key="d_lib_nr") if use_library else 0
        if use_youtube_search:
            yt_calls = sum(
                len(s.get("youtube_keywords") or s.get("search_queries", [])[:1])
                for s in st.session_state.director_shots
                if s.get("priority") != "none"
            )
            st.caption(
                f"YouTube Search enabled - ~{yt_calls} yt-dlp search(es), no YouTube Data API quota used."
            )
        if use_youtube_api and os.getenv("YOUTUBE_API_KEY"):
            yt_calls = sum(
                1 for s in st.session_state.director_shots
                if s.get("priority") != "none" and s.get("search_queries")
            )
            est_units = yt_calls * 100
            st.caption(
                f"📺 YouTube enabled — ~{yt_calls} search call(s), "
                f"~{est_units:,} quota unit(s) (daily quota is 10,000)."
            )

        # ── Pre-fetch quota estimate for the stock APIs ───────────────────────
        # Each (source, query) is one request against an hourly quota
        # (~200/hr free for Pexels). Warn up-front and offer a one-click cap so
        # a big project doesn't trip the limiter mid-run.
        from core.director_search import estimate_stock_requests
        _PEXELS_HOURLY = 200
        pexels_max_q = None
        pixabay_max_q = None
        if use_pexels or use_pixabay:
            _est = estimate_stock_requests(st.session_state.director_shots)
            _pex_est = _est["pexels"] if use_pexels else 0
            _pix_est = _est["pixabay"] if use_pixabay else 0
            _bits = []
            if use_pexels:  _bits.append(f"Pexels ~{_pex_est}")
            if use_pixabay: _bits.append(f"Pixabay ~{_pix_est}")
            st.caption(f"📊 Estimated stock requests: {' · '.join(_bits)} "
                       f"(across {_est['shots']} shots).")
            if use_pexels and _pex_est > _PEXELS_HOURLY:
                _cap = max(1, _PEXELS_HOURLY // max(1, _est["shots"]))
                _capped_est = estimate_stock_requests(
                    st.session_state.director_shots, pexels_max_queries=_cap
                )["pexels"]
                st.warning(
                    f"⚠️ ~{_pex_est} Pexels requests exceeds the free tier's "
                    f"~{_PEXELS_HOURLY}/hour — the run will likely hit the rate limit "
                    "partway through."
                )
                if st.checkbox(
                    f"Cap Pexels to first {_cap} quer{'y' if _cap == 1 else 'ies'}/shot "
                    f"(~{_capped_est} requests, fits the limit)",
                    value=True, key="d_pexels_cap",
                ):
                    pexels_max_q = _cap

        col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1])
        with col_btn1:
            fetch_clicked = st.button("Fetch", disabled=st.session_state.is_fetching, key="d_fetch", width="stretch")
        with col_btn2:
            retry_clicked = st.button("Retry Empty", disabled=st.session_state.is_fetching, key="d_retry", width="stretch", help="Only searches for shots that currently have 0 candidates. Preserves existing successful searches.")
        with col_btn3:
            _new_chunk_size = st.number_input(
                "Chunk size",
                min_value=2, max_value=20,
                value=int(st.session_state.get("fetch_chunk_size", 6)),
                key="d_chunk_size",
                help="Shots are fetched in chunks of this many. Smaller = first chunk lands sooner so Step 5 review can begin earlier; larger = fewer chunk transitions, slightly less overhead.",
            )
            if int(_new_chunk_size) != st.session_state.fetch_chunk_size:
                st.session_state.fetch_chunk_size = int(_new_chunk_size)
                save_cache()

        if fetch_clicked or retry_clicked:
            if not check_network():
                st.error("No network connection detected.")
            elif not (use_pexels and os.getenv("PEXELS_API_KEY")) and \
                 not (use_pixabay and os.getenv("PIXABAY_API_KEY")) and \
                 not use_youtube_search and \
                 not (use_youtube_api and os.getenv("YOUTUBE_API_KEY")):
                st.error("No search sources enabled. Add Pexels/Pixabay keys or enable YouTube Search.")
            elif use_youtube_api and not os.getenv("YOUTUBE_API_KEY"):
                st.error("YouTube API requires a YouTube API key. Untick YouTube API or add the key in Setup.")
            elif retry_clicked:
                # ── Synchronous retry: only re-fetches shots with 0 results.
                # Workload is unpredictable but typically small, so we keep
                # this path blocking — a single spinner is clearer than
                # spinning up chunked dispatch for what's often <5 shots.
                st.session_state.is_fetching = True
                d_fetch_errors = []
                try:
                    pbar2 = st.progress(0)
                    status2 = st.empty()
                    status2.text("Re-fetching empty shots…")
                    from core.director_search import fetch_director_footage
                    fetch_director_footage(
                        st.session_state.director_shots,
                        use_pexels=use_pexels, use_pixabay=use_pixabay,
                        use_youtube=use_youtube_search,
                        pexels_num_results=pex_num, pixabay_num_results=pix_num,
                        youtube_api_num_results=yt_api_num,
                        youtube_search_num_results=yt_search_num,
                        use_youtube_api=use_youtube_api,
                        use_youtube_search=use_youtube_search,
                        progress_callback=lambda p: pbar2.progress(p * 0.9),
                        errors=d_fetch_errors,
                        retry_only=True,
                        min_height=min_h,
                        pexels_max_queries=pexels_max_q,
                        pixabay_max_queries=pixabay_max_q,
                    )
                    if use_library and lib_num > 0:
                        status2.text("📚 Searching Clip Library…")
                        for shot in st.session_state.director_shots:
                            if shot.get("priority") == "none":
                                continue
                            # Retry path: only inject for shots that had work this round
                            if not shot.get("video_results"):
                                continue
                            q = shot.get("shot_intent", "") or " ".join(shot.get("search_queries", [])[:2])
                            if not q:
                                continue
                            lib_hits = search_library(q, top_k=lib_num)
                            if lib_hits:
                                existing_urls = {r.get("url") for r in shot.get("video_results", [])}
                                new_hits = [h for h in lib_hits if h.get("url") not in existing_urls]
                                shot["video_results"] = new_hits + shot["video_results"]
                    pbar2.progress(1.0)
                    status2.text("Done.")
                    if d_fetch_errors:
                        unique_fe = list(dict.fromkeys(d_fetch_errors))
                        with st.expander(f"⚠️ {len(unique_fe)} fetch error(s)"):
                            for e in unique_fe[:20]:
                                st.write(f"• {e}")
                    save_cache()
                finally:
                    st.session_state.is_fetching = False
            else:
                # ── Chunked background fetch ──────────────────────────────
                from core.director_search import (
                    group_shots_into_fetch_chunks,
                    dispatch_chunked_fetch,
                    clear_query_cache,
                )
                clear_query_cache()
                _chunk_size = max(2, int(st.session_state.fetch_chunk_size))
                _chunks = group_shots_into_fetch_chunks(
                    st.session_state.director_shots, _chunk_size
                )
                if not _chunks:
                    st.error("No shots have search queries to fetch.")
                else:
                    st.session_state.chunk_fetch_status = {
                        i: "pending" for i in range(len(_chunks))
                    }
                    st.session_state.chunk_fetch_errors = {}
                    st.session_state.chunk_fetch_slot_ids = [
                        [s["slot_id"] for s in c] for c in _chunks
                    ]
                    st.session_state._last_chunk_completed_count = -1
                    # Clear stale video_results for everything we're about to fetch
                    # so the UI doesn't show old candidates next to "pending" badges.
                    for _shot in st.session_state.director_shots:
                        if _shot.get("priority") == "none":
                            continue
                        if not (_shot.get("search_queries") or _shot.get("youtube_keywords")):
                            continue
                        _shot["video_results"] = []
                    _fetch_kwargs = dict(
                        use_pexels=use_pexels, use_pixabay=use_pixabay,
                        use_youtube=use_youtube_search,
                        pexels_num_results=pex_num, pixabay_num_results=pix_num,
                        youtube_api_num_results=yt_api_num,
                        youtube_search_num_results=yt_search_num,
                        use_youtube_api=use_youtube_api,
                        use_youtube_search=use_youtube_search,
                        retry_only=False,
                        min_height=min_h,
                        pexels_max_queries=pexels_max_q,
                        pixabay_max_queries=pixabay_max_q,
                    )
                    _lib_inject = None
                    if use_library and lib_num > 0:
                        _lib_num = int(lib_num)
                        def _lib_inject(chunk_shots, _k=_lib_num):
                            for shot in chunk_shots:
                                q = shot.get("shot_intent", "") or " ".join(shot.get("search_queries", [])[:2])
                                if not q:
                                    continue
                                lib_hits = search_library(q, top_k=_k)
                                if lib_hits:
                                    existing_urls = {r.get("url") for r in shot.get("video_results", [])}
                                    new_hits = [h for h in lib_hits if h.get("url") not in existing_urls]
                                    shot.setdefault("video_results", [])
                                    shot["video_results"] = new_hits + shot["video_results"]
                    try:
                        from streamlit.runtime.scriptrunner import (
                            add_script_run_ctx, get_script_run_ctx,
                        )
                    except ImportError:
                        add_script_run_ctx = lambda t, c: None
                        get_script_run_ctx = lambda: None
                    _st_ctx = get_script_run_ctx()
                    _worker = threading.Thread(
                        target=dispatch_chunked_fetch,
                        kwargs=dict(
                            shots=st.session_state.director_shots,
                            fetch_chunk_size=_chunk_size,
                            fetch_kwargs=_fetch_kwargs,
                            status_dict=st.session_state.chunk_fetch_status,
                            errors_dict=st.session_state.chunk_fetch_errors,
                            clip_library_inject=_lib_inject,
                        ),
                        daemon=True,
                        name="ChunkedFetchDispatcher",
                    )
                    if _st_ctx:
                        add_script_run_ctx(_worker, _st_ctx)
                    _worker.start()
                    st.session_state.chunk_fetch_thread = _worker
                    st.session_state.is_fetching = True
                    save_cache()
                    st.rerun()

        # Live chunk-fetch status display — polls every 2s while any chunk is
        # in flight, switches to a static summary once everything's done.
        _status_dict = st.session_state.get("chunk_fetch_status") or {}
        if _status_dict:
            _still_running = any(
                v in ("pending", "fetching") for v in _status_dict.values()
            )
            if _still_running:
                _render_chunk_status_polling()
            else:
                _render_chunk_status_final()
    # Regenerate Failed — shown after any fetch when some shots still have 0 candidates
    empty_shots = [
        s for s in st.session_state.get("director_shots", [])
        if len(s.get("video_results", [])) == 0
        and s.get("priority") != "none"
        and s.get("search_queries")
    ]

    # ── Retry failed shots (same queries) — rate-limit aware ──────────────────
    # When a stock source is throttled, the empty shots failed on the quota,
    # not the query. Let the user re-fetch ONLY those shots once the window
    # resets, keeping everything already found.
    if empty_shots and not st.session_state.get("is_fetching"):
        from core.stock_apis import get_rate_limit_status
        _rl = get_rate_limit_status()
        _rl_blocked = {k: v for k, v in _rl.items() if v["blocked"]}
        st.divider()
        with st.container():
            st.subheader(f"🔁 Retry {len(empty_shots)} failed shot(s)")
            if _rl_blocked:
                _names = " · ".join(
                    f"{k.title()} resets ~{v['reset_at']}" for k, v in _rl_blocked.items()
                )
                _mins = max(int(v["seconds"] // 60) + 1 for v in _rl_blocked.values())
                st.caption(
                    f"⏳ Stock API rate-limited ({_names}). These shots hit the hourly "
                    "quota, not a bad query — retry the same searches once it resets. "
                    "Everything already found is kept."
                )
                _retry_label = f"🔁 Retry available in ~{_mins} min"
            else:
                st.caption(
                    "Re-fetch only the shots with 0 candidates, using the same queries "
                    "and your current source settings. Existing results are preserved."
                )
                _retry_label = f"🔁 Retry these {len(empty_shots)} shot(s) now"
            if st.button(_retry_label, key="d_retry_failed_same",
                         disabled=bool(_rl_blocked)):
                from core.director_search import (
                    fetch_director_footage as _fdf, estimate_stock_requests as _esr,
                )
                _cap = None
                if st.session_state.get("d_pexels_cap"):
                    _e = _esr(st.session_state.director_shots)
                    if _e["shots"]:
                        _cap = max(1, 200 // _e["shots"])
                _minh = {"None": 0, "720p": 720, "1080p": 1080, "2K": 1440, "4K": 2160}.get(
                    st.session_state.get("d_min_res_sel", "None"), 0
                )
                _rerrs = []
                with st.spinner(f"Retrying {len(empty_shots)} shot(s)…"):
                    _fdf(
                        st.session_state.director_shots,
                        use_pexels=st.session_state.get("d_pex_cb", False),
                        use_pixabay=st.session_state.get("d_pix_cb", False),
                        use_youtube=st.session_state.get("d_yt_search_cb", True),
                        pexels_num_results=int(st.session_state.get("d_pex_nr", 3)),
                        pixabay_num_results=int(st.session_state.get("d_pix_nr", 3)),
                        youtube_api_num_results=int(st.session_state.get("d_ytapi_nr", 3)),
                        youtube_search_num_results=int(st.session_state.get("d_yts_nr", 3)),
                        use_youtube_api=st.session_state.get("d_yt_api_cb", False),
                        use_youtube_search=st.session_state.get("d_yt_search_cb", True),
                        retry_only=True, min_height=_minh, errors=_rerrs,
                        pexels_max_queries=_cap,
                    )
                _still = sum(
                    1 for s in st.session_state.director_shots
                    if not s.get("video_results") and s.get("priority") != "none"
                    and s.get("search_queries")
                )
                save_cache()
                _found = len(empty_shots) - _still
                if _found > 0:
                    st.success(f"Found candidates for {_found} shot(s). {_still} still empty.")
                else:
                    st.warning(f"No new candidates — {_still} shot(s) still empty.")
                if _rerrs:
                    _uniq = list(dict.fromkeys(_rerrs))
                    with st.expander(f"⚠️ {len(_uniq)} message(s)"):
                        for e in _uniq[:20]:
                            st.write(f"• {e}")
                st.rerun()

    if empty_shots and not st.session_state.get("is_fetching"):
        st.divider()
        with st.container():
            st.subheader(f"⚠️ {len(empty_shots)} shot(s) returned no candidates")
            st.caption(
                "The AI will re-read surrounding script context to rewrite the queries for these "
                "shots, then re-fetch automatically. Previously tried queries are shown to the "
                "model so it produces something different."
            )
            if st.button("🔄 Regenerate Queries & Re-fetch", key="d_regen_failed", type="primary"):
                if not os.getenv("GROQ_API_KEY"):
                    st.error("Groq API key required.")
                else:
                    failed_ids = {s["slot_id"] for s in empty_shots}
                    regen_pbar   = st.progress(0)
                    regen_status = st.empty()
                    regen_status.text(f"Regenerating queries for {len(failed_ids)} shot(s)…")
                    try:
                        from core.director import regenerate_shot_queries
                        from core.director_search import clear_query_cache
                        st.session_state.director_shots = regenerate_shot_queries(
                            st.session_state.director_shots,
                            slot_ids=failed_ids,
                            api_key=os.getenv("GROQ_API_KEY"),
                            video_topic=st.session_state.get("d_video_topic", ""),
                            custom_instructions=st.session_state.get("d_style", ""),
                            progress_callback=lambda p: regen_pbar.progress(p * 0.45),
                        )
                        regen_status.text("Re-fetching candidates for regenerated shots…")
                        clear_query_cache()
                        d_regen_errors = []
                        updated = fetch_director_footage(
                            st.session_state.director_shots,
                            use_pexels=st.session_state.get("d_pex_cb", True),
                            use_pixabay=st.session_state.get("d_pix_cb", True),
                            use_youtube_search=st.session_state.get("d_yt_search_cb", True),
                            use_youtube_api=st.session_state.get("d_yt_api_cb", False),
                            pexels_num_results=int(st.session_state.get("d_pex_nr", 3)),
                            pixabay_num_results=int(st.session_state.get("d_pix_nr", 3)),
                            youtube_api_num_results=int(st.session_state.get("d_ytapi_nr", 3)),
                            youtube_search_num_results=int(st.session_state.get("d_yts_nr", 3)),
                            min_height={"None": 0, "720p": 720, "1080p": 1080, "2K": 1440, "4K": 2160}.get(
                                st.session_state.get("d_min_res_sel", "None"), 0
                            ),
                            retry_only=True,
                            errors=d_regen_errors,
                            progress_callback=lambda p: regen_pbar.progress(0.45 + p * 0.55),
                        )
                        st.session_state.director_shots = updated
                        regen_pbar.progress(1.0)
                        newly_found = sum(
                            len(s.get("video_results", []))
                            for s in updated if s.get("slot_id") in failed_ids
                        )
                        regen_status.text("Done.")
                        st.success(f"Found {newly_found} new candidates across {len(failed_ids)} shots.")
                        if d_regen_errors:
                            with st.expander(f"⚠️ {len(d_regen_errors)} error(s)"):
                                for e in d_regen_errors[:10]:
                                    st.write(f"• {e}")
                        save_cache()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Regeneration error: {e}")

    # Step 4 — LLM Ranking
    # Ranker now also flags single-candidate shots as irrelevant when needed,
    # so we offer it whenever any shot has at least one candidate.
    has_candidates = any(len(s.get("video_results", [])) >= 1 for s in st.session_state.get("director_shots", []))
    if has_candidates:
        st.header("Step 4: Rank & Filter Candidates")
        st.caption("AI ranks candidates by visual relevance, flags off-topic clips, and prefers horizontal videos.")
        # Read-only display of the topic set at Step 2 (the input lives there
        # to also benefit query generation; we cannot have a second text_input
        # with the same key).
        d_video_topic = st.session_state.get("d_video_topic", "")
        if d_video_topic:
            st.caption(f"📌 Using video topic: **{d_video_topic}** (edit in Step 2 above to change)")
        else:
            st.warning("⚠️ No video topic set. Set one in Step 2 above for the ranker to filter off-topic clips effectively.")
        if st.button("Rank Candidates with AI", key="d_rank"):
            if not os.getenv("GROQ_API_KEY"):
                st.error("Groq API key required for ranking.")
            else:
                pbar3 = st.progress(0)
                status3 = st.empty()
                status3.text("Ranking…")
                d_rank_errors = []
                try:
                    st.session_state.director_shots = rank_shot_candidates(
                        st.session_state.director_shots,
                        api_key=os.getenv("GROQ_API_KEY"),
                        custom_instructions=custom_instructions,
                        video_topic=d_video_topic,
                        progress_callback=lambda p: pbar3.progress(p),
                        errors=d_rank_errors,
                    )
                    pbar3.progress(1.0)
                    status3.text("Done.")

                    # Phase-3 auto-selection: bind the top-ranked clip per shot
                    # when enabled. Runs after ranking so it acts on the sorted
                    # candidates; never overwrites a manual pick.
                    auto_n = 0
                    if _env_flag("ENABLE_AUTO_SELECTION"):
                        from core.director_rank import auto_select_top_candidates
                        auto_select_top_candidates(st.session_state.director_shots)
                        auto_n = sum(1 for s in st.session_state.director_shots
                                     if s.get("auto_selected") and s.get("selected_results"))

                    failed = sum(1 for s in st.session_state.director_shots if s.get("rank_error"))
                    total  = sum(1 for s in st.session_state.director_shots
                                 if s.get("video_results") and s.get("priority") != "none")
                    if failed == 0:
                        st.success("Candidates ranked! Irrelevant clips are hidden in the review.")
                    elif failed < total:
                        st.warning(f"Ranked {total - failed}/{total} shots. {failed} shot(s) failed — see details below and check those shots in the review.")
                    else:
                        st.error("Ranking failed for all shots. Showing original order. Check API key and rate limits.")
                    if auto_n:
                        st.info(f"🤖 Auto-selected the top clip for {auto_n} shot(s). Review or override any pick in Step 5.")
                    if d_rank_errors:
                        unique_re = list(dict.fromkeys(d_rank_errors))
                        with st.expander(f"⚠️ {len(unique_re)} ranking error(s)"):
                            for e in unique_re[:20]:
                                st.write(f"• {e}")
                    save_cache()
                except Exception as e:
                    st.error(f"Ranking error: {e}")
    # Step 5 — Editor Review (paginated)
    review_shots = [s for s in st.session_state.get("director_shots", [])
                    if s.get("video_results") and s.get("priority") != "none"]
    # Banner about chunks still fetching — shows when a background fetch is
    # in flight so the user knows more shots will appear as chunks complete.
    _status_dict_for_step5 = st.session_state.get("chunk_fetch_status") or {}
    _pending_chunks = [i for i, v in _status_dict_for_step5.items()
                       if v in ("pending", "fetching")]
    if _pending_chunks:
        _pending_slots = []
        _groups = st.session_state.get("chunk_fetch_slot_ids", [])
        for ci in _pending_chunks:
            if ci < len(_groups):
                _pending_slots.extend(_groups[ci])
        if review_shots:
            st.info(
                f"🔄 {len(_pending_chunks)} chunk(s) still fetching "
                f"({len(_pending_slots)} more shot(s) coming). "
                "You can start reviewing what's ready below — remaining shots will "
                "appear automatically as their chunks finish."
            )
        else:
            st.info(
                f"🔄 Fetching first chunk… {len(_pending_chunks)} chunk(s) in flight. "
                "Step 5 will populate as soon as the first chunk lands."
            )
    if review_shots:
        st.header("Step 5: Review & Select")
        st.caption(
            "Tick the clips you want for each shot. Selections persist as you navigate — "
            "nothing downloads until you click **Download** in Step 6. "
            "Picking the same clip across multiple shots downloads it once."
        )
        # Build a global URL → list of slot_ids map across ALL shots so we
        # can flag candidates that are already picked elsewhere — useful
        # signal for editorial decisions even though Step 6 dedupes the
        # downloads automatically.
        global_pick_map = {}
        for s in review_shots:
            sid = s.get("slot_id", "?")
            for r in s.get("selected_results", []):
                u = r.get("url")
                if u:
                    global_pick_map.setdefault(u, []).append(sid)
        if "d_review_idx" not in st.session_state:
            st.session_state.d_review_idx = 0
        st.session_state.d_review_idx = max(
            0, min(st.session_state.d_review_idx, len(review_shots) - 1)
        )
        idx = st.session_state.d_review_idx
        shot = review_shots[idx]
        # --- Variables used for display ---
        n_selected = sum(1 for s in review_shots if s.get("selected_results"))
        n_skipped  = sum(1 for s in review_shots if s.get("skipped"))
        n_flagged  = sum(1 for s in review_shots if s.get("flagged") and not s.get("selected_results") and not s.get("skipped"))
        n_pending  = len(review_shots) - n_selected - n_skipped - n_flagged
        slot_id  = shot.get("slot_id", "?")
        ts       = f"{shot.get('timestamp_start_str')} – {shot.get('timestamp_end_str')}"
        reason   = shot.get("rank_reason", "")
        rank_err = shot.get("rank_error", "")
        sel_urls = {r.get("url") for r in shot.get("selected_results", [])}
        skipped  = shot.get("skipped", False)
        candidates = [c for c in shot["video_results"] if not c.get("irrelevant")]
        if not candidates:
            st.warning("No relevant candidates found for this shot.")
        else:
            st.markdown("---")
            st.subheader("🖼️ Selection Gallery")
            # ── Shot context (TOP) ────────────────────────────────────────────
            with st.container(border=True):
                pri = shot.get("priority", "medium")
                pri_badge = {"high": "🔴 HIGH", "medium": "🟡 MED", "low": "⚪ LOW"}.get(pri, pri)
                st.markdown(
                    f"**Shot {slot_id}** &nbsp;·&nbsp; 🕒 {ts} &nbsp;·&nbsp; "
                    f"🎬 {shot.get('shot_intent', '—')} &nbsp;·&nbsp; {pri_badge}"
                )
                st.markdown(f"💬 _{shot.get('text', '')}_")
                top_queries = shot.get("search_queries", [])
                if top_queries:
                    st.markdown(" &nbsp;·&nbsp; ".join(f"`{q}`" for q in top_queries))
                if reason:
                    st.caption(f"🤖 {reason}")
            # ── Navigation Strip ──────────────────────────────────────────────
            # Build jump options with per-shot status icons so users can see
            # at a glance which shots still need attention.
            jump_options = []
            for _ji, _js in enumerate(review_shots):
                _sel = _js.get("selected_results")
                if _sel:                    _icon = "✅"
                elif _js.get("skipped"):    _icon = "⏭"
                elif _js.get("flagged"):    _icon = "🔖"
                else:                       _icon = "⏳"
                _tag = " [H]" if _js.get("priority") == "high" else ""
                jump_options.append(
                    f"Shot {_ji+1} {_icon}{_tag} — {(_js.get('shot_intent') or '')[:30]}"
                )
            next_unpicked_idx = next(
                (j for j in range(idx + 1, len(review_shots))
                 if not review_shots[j].get("selected_results")
                 and not review_shots[j].get("skipped")
                 and not review_shots[j].get("flagged")),
                None,
            )

            # on_change callback for the jump dropdowns. Streamlit fires
            # on_change BEFORE the script reruns, so by the time we read
            # d_review_idx below, it already reflects the user's pick. This
            # lets the dropdown both *show* the current shot (via the sync
            # writes below) AND *change* it (via this callback) without the
            # two clobbering each other.
            def _on_jump_change(state_key):
                sel = st.session_state.get(state_key)
                if sel in jump_options:
                    new_i = jump_options.index(sel)
                    if new_i != st.session_state.d_review_idx:
                        save_cache()
                        st.session_state.d_review_idx = new_i

            nav1, nav2, nav3, nav4, nav5 = st.columns([1.2, 2.5, 2.5, 1.8, 1.2])
            with nav1:
                if st.button("◀ Prev", key="d_prev", disabled=idx == 0, width="stretch"):
                    save_cache()
                    st.session_state.d_review_idx -= 1
                    st.rerun()
            with nav2:
                # Force the widget's stored value to match the current shot
                # BEFORE the widget renders. on_change already aligned
                # d_review_idx with the user's pick (if any) before the script
                # started, so this write is a no-op in the user-pick case.
                # When Prev/Next changed d_review_idx, this write is what
                # actually moves the dropdown to the new shot.
                st.session_state["d_jump_top"] = jump_options[idx]
                st.selectbox(
                    "Jump (top)", options=jump_options,
                    label_visibility="collapsed", key="d_jump_top",
                    on_change=_on_jump_change, args=("d_jump_top",),
                )
            with nav3:
                st.markdown(
                    f"<div style='text-align:center; padding-top:0.4em; font-size:13px;'>"
                    f"✅ {n_selected} &nbsp;·&nbsp; ⏭ {n_skipped} "
                    f"&nbsp;·&nbsp; 🔖 {n_flagged} &nbsp;·&nbsp; ⏳ {n_pending}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with nav4:
                if st.button(
                    "⏭ Next Unpicked",
                    key="d_next_unpicked",
                    disabled=next_unpicked_idx is None,
                    width="stretch",
                    help="Jump to the next shot with no selection yet.",
                ):
                    save_cache()
                    st.session_state.d_review_idx = next_unpicked_idx
                    st.rerun()
            with nav5:
                if st.button("Next ▶", key="d_next", disabled=idx == len(review_shots) - 1, width="stretch"):
                    save_cache()
                    st.session_state.d_review_idx += 1
                    st.rerun()
            st.progress((idx + 1) / len(review_shots))
            # ── Skip to here ──────────────────────────────────────────────────
            # Jump-and-skip: mark every earlier un-picked shot as skipped so the
            # editor can start selecting from this one. Shots already selected
            # are left untouched so no work is lost.
            _earlier_pending = [
                s for s in review_shots[:idx]
                if not s.get("selected_results") and not s.get("skipped")
            ]
            if _earlier_pending:
                sk1, sk2 = st.columns([1.5, 3])
                with sk1:
                    if st.button(
                        f"⏭ Skip to here (Shot {idx + 1})",
                        key=f"skip_till_{slot_id}",
                        width="stretch",
                        help=(
                            f"Mark the {len(_earlier_pending)} earlier un-picked shot(s) "
                            f"as skipped so you can start selecting from Shot {idx + 1}. "
                            "Shots you've already picked are kept."
                        ),
                    ):
                        for _s in _earlier_pending:
                            _s["skipped"] = True
                            _s["selected_results"] = []
                            _s["flagged"] = False
                        save_cache()
                        st.rerun()
                with sk2:
                    st.caption(
                        f"Skips the {len(_earlier_pending)} un-picked shot(s) before this "
                        "one (already-picked shots are kept)."
                    )
            # ── Bulk Actions & Quality Filter ─────────────────────────────────
            ba1, ba2, ba3 = st.columns([1, 1, 2])
            with ba1:
                q_filter_key = "director_global_q_filter"
                temp_q_filter = st.session_state.get(q_filter_key, "All")
                temp_filtered = [c for c in candidates]
                if temp_q_filter == "720p and above":
                    temp_filtered = [c for c in temp_filtered if (c.get("height") or 0) >= 720 or c.get("source") == "youtube"]
                elif temp_q_filter == "1080p and above":
                    temp_filtered = [c for c in temp_filtered if (c.get("height") or 0) >= 1080 or c.get("source") == "youtube"]
                if st.button("☑ Select All", key=f"sel_all_{slot_id}", width="stretch"):
                    shot["selected_results"] = list(temp_filtered)
                    shot["skipped"] = False
                    save_cache()
                    st.rerun()
            with ba2:
                if st.button("☒ Clear All", key=f"sel_none_{slot_id}", width="stretch"):
                    shot["selected_results"] = []
                    save_cache()
                    st.rerun()
            with ba3:
                q_filter = st.selectbox(
                    "Filter by Quality",
                    options=["All", "720p and above", "1080p and above"],
                    index=0,
                    key="director_global_q_filter",
                    label_visibility="collapsed",
                    help="Filter candidates by minimum resolution."
                )
            # ── Auto-advance & Pick Top ───────────────────────────────────────
            aa1, aa2 = st.columns([2, 3])
            with aa1:
                st.checkbox("⚡ Auto-advance after picking", key="d_auto_advance", value=False)
            is_ranked = any(s.get("rank_reason") for s in review_shots)
            with aa2:
                if is_ranked and candidates and idx < len(review_shots) - 1:
                    if st.button("🥇 Pick Top & Next", key=f"pick_top_{slot_id}", width="stretch",
                                 help="Pick the AI's top-ranked clip and advance to the next shot."):
                        top_cand = candidates[0]
                        if top_cand.get("url") not in sel_urls:
                            toggle_pick(top_cand["url"], top_cand, shot)
                        st.session_state.d_review_idx = idx + 1
                        st.rerun()
            # CSS for cards
            st.markdown("""<style>.gallery-card { border-radius: 8px; }</style>""", unsafe_allow_html=True)
            # Apply quality filter
            filtered_cands = [c for c in candidates]
            if q_filter == "720p and above":
                filtered_cands = [c for c in filtered_cands if (c.get("height") or 0) >= 720 or c.get("source") == "youtube"]
            elif q_filter == "1080p and above":
                filtered_cands = [c for c in filtered_cands if (c.get("height") or 0) >= 1080 or c.get("source") == "youtube"]
            n_picked_in_shot = sum(1 for c in filtered_cands if c.get("url") in sel_urls)
            st.caption(f"**{n_picked_in_shot} selected** · Showing {len(filtered_cands)} of {len(candidates)} candidates.")
            # ── Bulk HD/SD check for YouTube clips in this shot ───────────────
            # ── Auto HD/SD check ──────────────────────────────────────────────
            # Runs automatically on every page load for any YouTube clip that
            # hasn't been checked yet. Mutates candidates in-place before the
            # gallery renders, so labels appear on the cards immediately.
            # New clips added by a refetch are also handled here since they
            # won't have definition_checked=True.
            _yt_api_key = os.getenv("YOUTUBE_API_KEY")
            _yt_unchecked = [
                c for c in filtered_cands
                if c.get("source") == "youtube" and not c.get("definition_checked")
            ]
            if _yt_unchecked:
                if _yt_api_key:
                    from core.stock_apis import fetch_youtube_definitions_batch
                    with st.spinner(f"Checking HD/SD for {len(_yt_unchecked)} YouTube clip(s)…"):
                        _defs = fetch_youtube_definitions_batch(
                            [c["url"] for c in _yt_unchecked], _yt_api_key,
                        )
                    for _c in _yt_unchecked:
                        _c["definition"] = _defs.get(_c["url"], "unknown")
                        _c["definition_checked"] = True
                    save_cache()
                else:
                    st.caption("⚙️ Set `YOUTUBE_API_KEY` in Setup to enable automatic HD/SD check")
            # Gallery Grid
            n_cols = 3
            for i_g in range(0, len(filtered_cands), n_cols):
                cols = st.columns(n_cols)
                for j_g in range(n_cols):
                    if i_g + j_g < len(filtered_cands):
                        cand = filtered_cands[i_g + j_g]
                        thumb = cand.get("thumbnail")
                        url = cand.get("page_url") or cand.get("url")
                        title = cand.get("title", "Video")
                        source_val = cand.get("source", "").upper()
                        if source_val == "SMART_LIBRARY":
                            source_display = "🤖 AI VISUAL SEARCH"
                        elif source_val == "LIBRARY":
                            _orig = cand.get("original_source", "").upper()
                            _sim  = cand.get("similarity", 0)
                            _uses = cand.get("usage_count", 1)
                            source_display = f"📚 LIBRARY ({_orig}) · {_sim*100:.0f}% match · used {_uses}×"
                        else:
                            source_display = f"🌐 {source_val}"
                        score = cand.get("score")
                        if score and source_val != "LIBRARY":
                            source_display += f" · {score*100:.0f}% match"
                        w, h = cand.get("width"), cand.get("height")
                        dur = cand.get("duration")
                        avail_res = cand.get("available_resolutions", [])
                        if avail_res:
                            high_res = [r for r in [2160, 1440, 1080, 720] if r in avail_res]
                            res_str = f"{w}x{h}" if w and h else f"{max(avail_res)}p"
                            if high_res:
                                res_str += f" ({', '.join([('4K' if r==2160 else '2K' if r==1440 else str(r)+'p') for r in high_res])})"
                        else:
                            res_str = f"{w}x{h}" if w and h else "Resolution Unknown"
                        dur_str = f"{int(dur)}s" if dur else ""
                        with cols[j_g]:
                            cand_url = cand.get("url")
                            is_picked = cand_url in sel_urls
                            other_slots = [sid for sid in global_pick_map.get(cand_url, []) if str(sid) != str(slot_id)]
                            st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
                            with st.container(border=True):
                                if thumb:
                                    selected_badge = '<div style="position:absolute;top:6px;left:6px;z-index:10;background:#00cc44;color:white;border-radius:50%;width:26px;height:26px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:14px;border:2px solid white;">✓</div>' if is_picked else ""
                                    used_badge = f'<div style="position:absolute;bottom:6px;left:6px;z-index:10;background:#ffcc00;color:black;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:bold;border:1px solid #999;">USED IN {", ".join(map(str, other_slots))}</div>' if other_slots else ""
                                    _is_lib = cand.get("from_library")
                                    if is_picked:
                                        border_style = "border: 3px solid #00cc44; border-radius:6px;"
                                    elif _is_lib:
                                        border_style = "border: 2px solid #a78bfa; border-radius:6px;"
                                    else:
                                        border_style = ""
                                    st.markdown(f'<div style="position:relative;{border_style}"><img src="{thumb}" style="width:100%;border-radius:4px;aspect-ratio:16/9;object-fit:cover;display:block;">{selected_badge}{used_badge}</div>', unsafe_allow_html=True)
                                st.markdown(f"**{title}**")
                                st.caption(f"{source_display} · {dur_str} · {res_str}")
                                mq = cand.get("matched_query", "")
                                if mq:
                                    st.caption(f"🔍 _{mq}_")
                                extra_info = cand.get("tags") if (source_val not in ("YOUTUBE", "SMART_LIBRARY") and cand.get("tags")) else cand.get("description")
                                if extra_info:
                                    with st.expander("📄 Details", expanded=False): st.write(extra_info)
                                if cand.get("smart_reason"): st.caption(f"🤖 _{cand.get('smart_reason')}_")
                                if source_val == "SMART_LIBRARY":
                                    if st.button("🎬 Preview", key=f"prev_{slot_id}_{i_g+j_g}"): st.video(cand.get("segment_path"))
                                else:
                                    st.markdown(f'<a href="{url}" target="_blank" style="text-decoration:none;font-size:12px;">📺 Watch on source ↗</a>', unsafe_allow_html=True)
                                    # ── Inline preview ────────────────────────
                                    # st.popover lazy-loads its content — the
                                    # video iframe / direct stream only loads
                                    # when the user clicks the trigger, so all
                                    # 30+ cards don't preload players. For
                                    # YouTube URLs Streamlit auto-embeds the
                                    # official player; for Pexels/Pixabay the
                                    # `url` is a direct MP4 which the HTML5
                                    # player streams natively.
                                    _preview_src = cand.get("preview_url") or cand.get("url")
                                    if _preview_src:
                                        with st.popover("▶️ Preview here", width="stretch"):
                                            try:
                                                st.video(_preview_src)
                                                st.caption(f"Source: {source_display} · {res_str}")
                                            except Exception as _e:
                                                st.warning(f"Cannot preview inline: {_e}")
                                                st.caption("Use the 'Watch on source' link above.")
                                # ── Inspect HD/SD via YouTube Data API ──
                                # Switched from yt-dlp-based exact-resolution
                                # probing to YouTube Data API's
                                # videos.list?part=contentDetails. The Data API
                                # returns 'hd' (>=720p available) or 'sd'
                                # authoritatively — no JS-runtime / SABR /
                                # storyboard guesswork. Costs 1 quota unit per
                                # lookup (default budget: 10,000/day).
                                if cand.get("source") == "youtube":
                                    if not cand.get("definition_checked"):
                                        yt_api_key = os.getenv("YOUTUBE_API_KEY")
                                        if not yt_api_key:
                                            st.caption("⚙️ Set `YOUTUBE_API_KEY` in Setup to enable HD check")
                                        elif st.button(
                                            "🔍 Check HD/SD",
                                            key=f"check_def_{slot_id}_{hash(cand_url)}",
                                            width="stretch",
                                            help="Look up this video's HD/SD status via the YouTube Data API. Costs 1 quota unit (default budget: 10,000/day).",
                                        ):
                                            with st.spinner("Checking…"):
                                                from core.stock_apis import fetch_youtube_definition
                                                definition = fetch_youtube_definition(cand["url"], yt_api_key)
                                            cand["definition"] = definition
                                            cand["definition_checked"] = True
                                            save_cache()
                                            st.rerun(scope="app")
                                    else:
                                        defn = (cand.get("definition") or "unknown").lower()
                                        if defn == "hd":
                                            st.caption("✅ **HD** (720p+)")
                                        elif defn == "sd":
                                            st.caption("⚠️ **SD** (under 720p)")
                                        else:
                                            st.caption("❓ Definition unknown (API lookup failed)")
                                btn_label = "✅ SELECTED" if is_picked else "⬜ PICK CLIP"
                                if st.button(btn_label, key=f"galpick_{slot_id}_{hash(cand.get('url'))}", width="stretch", type="primary" if is_picked else "secondary"):
                                    was_selected = cand_url in sel_urls
                                    toggle_pick(cand.get("url"), cand, shot)
                                    if not was_selected and st.session_state.get("d_auto_advance") and idx < len(review_shots) - 1:
                                        st.session_state.d_review_idx = idx + 1
                                    st.rerun()
                                if mq and source_val.lower() in ("pexels", "pixabay", "youtube"):
                                    if st.button("➕ More like this", key=f"mlt_{slot_id}_{hash(cand_url)}", width="stretch"):
                                        with st.spinner("Fetching more…"):
                                            from core.director_search import fetch_more_like_this
                                            more = fetch_more_like_this(shot, mq, source_val.lower())
                                        if more:
                                            shot["video_results"] = shot.get("video_results", []) + more
                                            save_cache()
                                            st.rerun()
                                        else:
                                            st.toast("No new results found for this query.")
                            st.markdown('</div>', unsafe_allow_html=True)
            # 2. Footer Navigation Bar
            fa1, fa2, fa3, fa4, fa5, fa6 = st.columns([1.2, 1.2, 1.2, 2.2, 2.5, 1.2])
            with fa1:
                if st.button("◀ Prev", key=f"d_prev_bot_{slot_id}", disabled=idx == 0, width="stretch"):
                    save_cache(); st.session_state.d_review_idx -= 1; st.rerun()
            with fa2:
                skip_label = "↺ Unskip" if skipped else "⏭ Skip"
                if st.button(skip_label, key=f"skip_{slot_id}", width="stretch"):
                    shot["skipped"] = not skipped; shot["selected_results"] = []; save_cache(); st.rerun()
            with fa3:
                flag_label = "🔖 Unflag" if shot.get("flagged") else "🔖 Flag"
                if st.button(flag_label, key=f"flag_{slot_id}", width="stretch",
                             help="Mark this shot to revisit later without skipping it."):
                    shot["flagged"] = not shot.get("flagged", False)
                    if shot["flagged"]:
                        shot["skipped"] = False
                    save_cache(); st.rerun()
            with fa4:
                # Same sync-then-on_change pattern as the top dropdown.
                st.session_state["d_jump_bot"] = jump_options[idx]
                st.selectbox(
                    "Jump (bottom)", options=jump_options,
                    label_visibility="collapsed", key="d_jump_bot",
                    on_change=_on_jump_change, args=("d_jump_bot",),
                )
            with fa5:
                if idx < len(review_shots) - 1:
                    if st.button("Save & Next ▶", key=f"d_save_next_{slot_id}", type="primary", width="stretch"):
                        save_cache()
                        st.session_state.d_review_idx += 1
                        st.rerun()
                else:
                    if st.button("✅ Finish Review", key=f"d_finish_{slot_id}", type="primary", width="stretch"):
                        save_cache(); st.success("Review complete! Scroll down to Step 6 to start downloads.")
            with fa6:
                if st.button("Next ▶", key=f"d_next_bot_{slot_id}", disabled=idx == len(review_shots) - 1, width="stretch"):
                    save_cache()
                    st.session_state.d_review_idx += 1
                    st.rerun()
            # 3. Shot Details / Recap (At the bottom)
            with st.container(border=True):
                st.markdown(f"**Shot {slot_id}** &nbsp;·&nbsp; 🕒 {ts} &nbsp;·&nbsp; 🎬 {shot.get('shot_intent', '—')}")
                st.markdown(f"💬 _{shot.get('text', '')}_")
                queries = shot.get("search_queries", [])
                if queries:
                    q_md = " &nbsp;·&nbsp; ".join(f"`{q}`" for q in queries)
                    st.markdown(f"🔍 **Searched:** {q_md}")
                if reason: st.markdown(f"🤖 _{reason}_")
            if rank_err: st.warning(f"⚠️ AI ranking failed for this shot ({rank_err}). Showing unranked candidates.")
            if skipped: st.info("⏭ This shot is marked as skipped. Selecting any clip will unskip it.")
            if shot.get("flagged") and not shot.get("selected_results"): st.info("🔖 This shot is flagged for review. Pick a clip or click Unflag to clear.")
            if shot.get("auto_selected") and shot.get("selected_results"): st.info("🤖 Auto-selected the top-ranked clip. Pick a different clip to override.")
            # 4. Tweak & Refetch
            st.markdown("---")
            with st.expander("🔄 Tweak & Refetch THIS Shot", expanded=False):
                st.caption("Edit the keywords and fetch new candidates for this shot only.")
                curr_queries = " | ".join(shot.get("search_queries", []))
                new_queries_str = st.text_input("Stock Queries", value=curr_queries, key=f"refetch_q_{slot_id}")
                curr_yt = " | ".join(shot.get("youtube_keywords", []))
                new_yt_str = st.text_input("YouTube Keywords", value=curr_yt, key=f"refetch_yt_{slot_id}")
                if st.button("Refetch Candidates", key=f"refetch_btn_{slot_id}", type="primary", width="stretch"):
                    shot["search_queries"] = [q.strip() for q in new_queries_str.split("|") if q.strip()]
                    shot["youtube_keywords"] = [q.strip() for q in new_yt_str.split("|") if q.strip()]
                    save_cache()
                    with st.spinner("Fetching..."):
                        from core.director_search import fetch_director_footage
                        updated_subset = fetch_director_footage([shot], use_pexels=st.session_state.get("d_pex_cb"), use_pixabay=st.session_state.get("d_pix_cb"), use_youtube=st.session_state.get("d_yt_search_cb"), pexels_num_results=st.session_state.get("d_pex_nr", 3), pixabay_num_results=st.session_state.get("d_pix_nr", 3), youtube_api_num_results=st.session_state.get("d_ytapi_nr", 3), youtube_search_num_results=st.session_state.get("d_yts_nr", 3), use_youtube_api=st.session_state.get("d_yt_api_cb"), use_youtube_search=st.session_state.get("d_yt_search_cb"), progress_callback=None, errors=[], retry_only=False)
                        if updated_subset:
                            for s in st.session_state.director_shots:
                                if s["slot_id"] == slot_id:
                                    s.update(updated_subset[0])
                                    break
                            save_cache()
                            st.success("Refetched!")
                            st.rerun()
    # --- Step 6 --- Download ---
    selected_shots = [s for s in st.session_state.get("director_shots", []) if s.get("selected_results")]
    if selected_shots:
        st.header("Step 6: Download Selected")
        st.caption(
            "Settings below apply to new downloads and to retries. Change "
            "Quality or Max Size, then click Retry on a failed item to re-attempt "
            "with the new settings. Clips downloaded in earlier sessions are reused "
            "automatically — no re-download."
        )
        # Persistent download cache info — collapsed by default, but shows
        # the user we're remembering past downloads and gives them a way
        # to invalidate the registry without touching the actual files.
        _cache_stats = download_cache.stats()
        if _cache_stats["count"]:
            with st.expander(
                f"📄¦ Download cache — {_cache_stats['count']} clip(s), "
                f"{_cache_stats['size_bytes'] / (1024 * 1024):.1f} MB"
                + (f"  ·  {_cache_stats['stale']} stale" if _cache_stats['stale'] else "")
            ):
                st.caption(
                    "URLs you've already downloaded across all sessions are remembered. "
                    "When you pick the same clip again, it's hardlinked from the cached "
                    "file instead of being re-downloaded. Stale entries (file deleted "
                    "from disk) are pruned automatically on lookup. "
                    "Clearing the registry doesn't delete any files — it just forgets the "
                    "URLââ€ 'path mapping."
                )
                if st.button("Clear download cache", key="d_cache_clear"):
                    download_cache.clear()
                    st.success("Cache cleared. The actual files in `downloads/` are untouched.")
                    st.rerun()
        # Live-bound settings — read on each render so retry/new-download both see them.
        col_dv1, col_dv2, col_dv3, col_dv4 = st.columns(4)
        with col_dv1:
            d_quality  = st.selectbox(
                "Quality",
                ["1080p", "720p", "480p", "Best", "Worst"],
                key="d_vq",
                format_func=_quality_label,
                help="Each option downloads the BEST stream available within that cap. '1080p' = best stream up to 1080p (not exactly 1080p).",
            )
        with col_dv2:
            d_max_size = st.number_input("Max Size (MB)", value=200, min_value=1, key="d_maxsize")
        with col_dv3:
            d_workers  = st.slider("Concurrent", 1, 10, 3, key="d_workers")
        with col_dv4:
            d_no_audio = st.checkbox(
                "Video only (no audio)",
                value=True,
                key="d_no_audio",
                help="Download video stream only — skips audio mixing and reduces file size. Ideal for B-roll clips.",
            )
        total_selected_files = sum(len(s.get("selected_results", [])) for s in selected_shots)
        col_dl1, col_dl2 = st.columns([2, 1])
        with col_dl1:
            start_dl = st.button(f"📥 Download {total_selected_files} selected videos",
                                key="d_dl_start", type="primary", width="stretch")
        with col_dl2:
            if st.button("📄 Export Manifest", key="d_manifest", width="stretch", help="Generate a text file listing all shots, grouped by chunk."):
                # _safe_for_fs is imported from core.output at the top of this file.
                proj_folder = _safe_for_fs(st.session_state.get("project_name", "default"), 50)
                manifest = [
                    "B-ROLL FINDER - SHOT MANIFEST",
                    "=============================",
                    f"Project: {st.session_state.get('project_name', 'Unnamed')}",
                    f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    ""
                ]
                all_shots = st.session_state.get("director_shots", [])
                sorted_shots = sorted(all_shots, key=lambda x: x.get("slot_id", 0))
                for shot in sorted_shots:
                    slot_id = shot.get("slot_id", "?")
                    start   = shot.get("timestamp_start_str", "00:00")
                    end     = shot.get("timestamp_end_str", "00:00")
                    intent  = shot.get("shot_intent", "No intent")
                    manifest.append(f"  Shot {slot_id}: [{start} - {end}] - {intent}")
                    sel = shot.get("selected_results", [])
                    if sel:
                        seen_filenames = set()
                        for footage_num, res in enumerate(sel, start=1):
                            kw       = _safe_for_fs(res.get("matched_query", ""), 30) or "clip"
                            base     = f"{slot_id}-{footage_num}-{kw}"
                            filename = f"{base}.mp4"
                            n = 1
                            while filename in seen_filenames:
                                n += 1
                                filename = f"{base}-{n}.mp4"
                            seen_filenames.add(filename)
                            manifest.append(f"    -> File: {filename}")
                    else:
                        manifest.append("    (No clip selected)")
                    manifest.append("")
                out_txt = "\n".join(manifest)
                out_path = os.path.join("downloads", "director", proj_folder, "shot_manifest.txt")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(out_txt)
                st.success(f"Manifest exported! Saved to: `{out_path}`")
        if start_dl:
            if not check_network():
                st.error("No network connection detected.")
            else:
                # Resize the pool in place (preserves download history) rather
                # than re-instantiating the manager, which would wipe it.
                st.session_state.dm.set_max_workers(d_workers)
                st.session_state.dm.clear_and_reset()
                # ── Group selected clips by URL across all shots ─────────
                # _safe_for_fs is imported from core.output at the top of this file.
                proj_folder = _safe_for_fs(st.session_state.get("project_name", "default"), 50)
                url_groups = {}      # url -> {"primary": (path, dm_source), "extras": [...], "shots": [...]}
                seen_filenames = set()
                for shot in selected_shots:
                    slot_id = shot.get("slot_id", "X")
                    footage_num = 0
                    for res in shot.get("selected_results", []):
                        url = res.get("url")
                        if not url:
                            continue
                        footage_num += 1
                        source  = res.get("source", "stock")
                        keyword = _safe_for_fs(res.get("matched_query", ""), 30) or "clip"
                        base    = f"{slot_id}-{footage_num}-{keyword}"
                        filename = f"{base}.mp4"
                        n = 1
                        while filename in seen_filenames:
                            n += 1
                            filename = f"{base}-{n}.mp4"
                        seen_filenames.add(filename)
                        output_path = os.path.join("downloads", "director", proj_folder, filename)
                        dm_source   = "direct" if source in ("pexels", "pixabay") else "youtube"
                        if url not in url_groups:
                            url_groups[url] = {
                                "primary": (output_path, dm_source),
                                "extras":  [],
                                "shots":   [slot_id],
                            }
                        else:
                            url_groups[url]["extras"].append(output_path)
                            url_groups[url]["shots"].append(slot_id)
                duplicate_count = sum(len(g["extras"]) for g in url_groups.values())
                added = 0
                cached_hits = 0
                for url, group in url_groups.items():
                    primary_path, primary_source = group["primary"]
                    extras = group["extras"]
                    # ── Cross-session cache lookup ──────────────────
                    # If we've previously downloaded this URL (any earlier
                    # session) and the file is still on disk, skip the
                    # network entirely and just hardlink/copy from the
                    # cached file to all expected per-shot paths.
                    cached = download_cache.lookup_path(url)
                    if cached:
                        if cached == primary_path:
                            # Already exactly where we want it. Just link any extras.
                            for ext in extras:
                                link_or_copy(cached, ext)
                            cached_hits += 1
                            continue
                        else:
                            # Cached under a different filename, copy/link it over.
                            ok_primary = link_or_copy(cached, primary_path)
                            for ext in extras:
                                link_or_copy(cached, ext)
                            if ok_primary:
                                cached_hits += 1
                                continue
                        # If we couldn't materialize from cache, fall
                        # through to a fresh download below.
                    # ── smart_proxy: timestamp-precise section download ──────────
                    # Look up the candidate metadata so we have timestamps & source
                    res_meta = None
                    for shot in selected_shots:
                        for res in shot.get("selected_results", []):
                            if res.get("url") == url and res.get("source") == "smart_proxy":
                                res_meta = res
                                break
                        if res_meta:
                            break
                    if res_meta:
                        try:
                            from core.indexer import VideoIndexer
                            _vi = VideoIndexer()
                            quality_val = d_quality.replace("p", "") if d_quality not in ("Best", "Worst") else "1080"
                            _vi.download_section(
                                url=url,
                                start_time=res_meta.get("timestamp_start", 0),
                                end_time=res_meta.get("timestamp_end", 30),
                                quality=quality_val,
                                output_path=primary_path,
                            )
                            # Auto-index into permanent Smart Library
                            if os.path.exists(primary_path):
                                _vi.embed_and_index(
                                    video_path=primary_path,
                                    video_url=url,
                                    video_title=res_meta.get("title"),
                                    timestamp_start=res_meta.get("timestamp_start"),
                                )
                            for ext in extras:
                                link_or_copy(primary_path, ext)
                            added += 1
                        except Exception as e:
                            st.error(f"Smart section download failed for Shot {group['shots'][0]}: {e}")
                        continue  # skip the regular dm.add_download below
                    try:
                        task_id = st.session_state.dm.add_download(
                            url, primary_path, d_quality,
                            source=primary_source, max_size_mb=d_max_size, normalize=False,
                            no_audio=d_no_audio,
                            extra_paths=extras or None,
                        )
                        st.session_state.dm.start_download(task_id)
                        added += 1
                    except Exception as e:
                        st.error(f"Shot {group['shots'][0]}: {e}")
                # Build a single, concise summary message covering all paths.
                parts = []
                if added:
                    parts.append(f"queued **{added}** new download(s)")
                if cached_hits:
                    parts.append(f"reused **{cached_hits}** clip(s) from cache")
                if duplicate_count:
                    parts.append(f"saved **{duplicate_count}** duplicate transfer(s) via hardlink")
                if added or cached_hits:
                    st.success("Done — " + "; ".join(parts) + ".")
                else:
                    st.warning("Nothing was queued — check URLs.")
        # ── Dashboard ────────────────────────────────────────────────────
        tasks = st.session_state.dm.get_all_tasks()
        if tasks:
            stats     = st.session_state.dm.get_stats()
            total_t   = stats["total"]
            completed = stats["completed"]
            st.progress(completed / total_t if total_t else 0.0)
            stat_col, cancel_col = st.columns([5, 1])
            with stat_col:
                st.markdown(
                    f"**{completed}/{total_t}** done · "
                    f"â¬â€¡ {stats['downloading']} active · "
                    f"â¸ {stats.get('paused', 0)} paused · "
                    f"â³ {stats['queued']} queued · "
                    f"âÅ’ {stats['error']} failed · "
                    f"â­ {stats['cancelled']} cancelled"
                )
            with cancel_col:
                if stats["downloading"] + stats["queued"] + stats.get("paused", 0) > 0:
                    if st.button("âÅ“– Cancel All", key="d_cancel", width="stretch"):
                        st.session_state.dm.cancel_all()
                        st.rerun()
            # ── Active tasks ─────────────────────────────────────────────
            active = st.session_state.dm.get_active_tasks()
            if active:
                st.subheader("In progress")
                for t in active:
                    is_proc   = t["status"] == "processing"
                    speed_str = format_speed(t.get("speed"))
                    eta_str   = format_eta(t.get("eta"))
                    meta      = " · ".join(filter(None, [speed_str, f"ETA {eta_str}" if eta_str else ""]))
                    label     = "Normalizing…" if is_proc else (
                        f"{t['status'].title()} ({t['progress']*100:.1f}%)"
                        + (f" — {meta}" if meta else "")
                    )
                    extras_n = len(t.get("extra_paths") or [])
                    extras_lbl = f" · ââ€ — +{extras_n} mirror{'s' if extras_n > 1 else ''}" if extras_n else ""
                    st.markdown(f"**{os.path.basename(t['output_path'])}**{extras_lbl} — {label}")
                    st.progress(t["progress"])
                    b1, b2, _b3 = st.columns([1, 1, 5])
                    if not is_proc:
                        if t["status"] == "downloading":
                            if b1.button("Pause", key=f"pd_{t['id']}", width="stretch"):
                                st.session_state.dm.pause_download(t["id"]); st.rerun()
                        elif t["status"] == "paused":
                            if b1.button("Resume", key=f"rd_{t['id']}", width="stretch"):
                                st.session_state.dm.resume_download(t["id"]); st.rerun()
                    if b2.button("Cancel", key=f"cd_{t['id']}", width="stretch"):
                        st.session_state.dm.cancel_download(t["id"]); st.rerun()
            # ── Failed tasks (URL + per-task retry with current settings) ──
            failed = st.session_state.dm.get_failed_tasks()
            if failed:
                st.subheader(f"âÅ’ Failed ({len(failed)})")
                retryable = [f for f in failed if st.session_state.dm.can_retry(f["id"])]
                if retryable:
                    if st.button(
                        f"ââ€ » Retry / resume all ({len(retryable)}) with current settings",
                        key="d_retry_all", width="stretch",
                        help=(
                            "Re-queues each failed task. yt-dlp resumes from any "
                            "existing .part file via Range requests, so progress "
                            "is preserved — only the bytes that didn't arrive "
                            "get re-downloaded."
                        ),
                    ):
                        st.session_state.dm.retry_all_failed(overrides={
                            "quality": d_quality, "max_size_mb": d_max_size,
                            "no_audio": d_no_audio,
                        })
                        st.rerun()
                for ft in failed:
                    with st.container(border=True):
                        top1, top2 = st.columns([5, 1])
                        with top1:
                            extras_n = len(ft.get("extra_paths") or [])
                            extras_lbl = f" · ââ€ — +{extras_n} mirror{'s' if extras_n > 1 else ''}" if extras_n else ""
                            st.markdown(f"**{os.path.basename(ft['output_path'])}**{extras_lbl}")
                            st.caption(
                                f"âš  {ft.get('error_summary') or ft.get('error_msg') or 'Unknown error'} · "
                                f"attempt {ft.get('attempts', 0)}"
                            )
                        with top2:
                             if st.button(
                                "ââ€ » Retry",
                                key=f"retry_{ft['id']}",
                                width="stretch",
                                help="Retry with the current Quality / Max Size settings.",
                            ):
                                st.session_state.dm.retry_failed(ft["id"], overrides={
                                    "quality": d_quality, "max_size_mb": d_max_size,
                                    "no_audio": d_no_audio,
                                })
                                st.rerun()
                        # URL row — let the user copy or open the source page
                        url_col, link_col = st.columns([5, 1])
                        with url_col:
                            st.code(ft.get("url", ""), language=None)
                        with link_col:
                            if ft.get("url"):
                                st.link_button("Open ââ€ —", ft["url"], width="stretch")
                        # Full error in a collapsed expander for power users
                        if ft.get("error_msg") and ft["error_msg"] != ft.get("error_summary"):
                            with st.expander("Full error"):
                                st.code(ft["error_msg"], language=None)
            # ── History (preserved across batches) ─────────────────────────
            history = [t for t in st.session_state.dm.get_history()
                       if t["status"] in ("completed", "cancelled")
                       and t["id"] not in {x["id"] for x in tasks}]  # exclude current batch
            if history:
                with st.expander(f"📄Å“ History — {len(history)} task(s) from earlier batches"):
                    h_completed = sum(1 for h in history if h["status"] == "completed")
                    h_cancelled = sum(1 for h in history if h["status"] == "cancelled")
                    st.caption(f"✅ {h_completed} completed · â­ {h_cancelled} cancelled")
                    if st.button("Clear history", key="d_clear_history"):
                        st.session_state.dm.clear_history(); st.rerun()
                    for h in history[-20:]:  # cap display to avoid blowing up the page
                        icon = "✅" if h["status"] == "completed" else "â­"
                        st.write(f"{icon} {os.path.basename(h['output_path'])}")
            # ── Auto-Save to Clip Library on Completion ────────────────────
            if completed == total_t and total_t > 0:
                _batch_key = tuple(sorted(t["id"] for t in tasks if t["status"] == "completed"))
                _saved_keys = st.session_state.get("d_library_saved_batches", set())
                if _batch_key and _batch_key not in _saved_keys:
                    _done_urls = {t["url"]: t for t in tasks if t["status"] == "completed"}
                    _lib_saved = 0
                    _lib_attempts = 0
                    for _shot in st.session_state.director_shots:
                        for _res in _shot.get("selected_results", []):
                            _url = _res.get("url", "")
                            if _url in _done_urls:
                                _task = _done_urls[_url]
                                _lib_attempts += 1
                                if store_clip(
                                    shot_description=_shot.get("shot_intent", _shot.get("text", "")),
                                    clip_data={**_res, "local_path": _task.get("output_path", "")},
                                    project=st.session_state.get("project_name", "default"),
                                    slot_index=_shot.get("slot_id", 0),
                                    keywords=_shot.get("search_queries", []),
                                    search_query=_res.get("matched_query", ""),
                                ):
                                    _lib_saved += 1
                    if _lib_saved:
                        st.success(f"💾 {_lib_saved} clip(s) saved to your Clip Library.")
                    elif _lib_attempts:
                        # Completed downloads but nothing persisted — the exact
                        # silent failure that left the library empty before.
                        st.warning(
                            f"⚠️ {_lib_attempts} clip(s) downloaded but none were saved "
                            "to your Clip Library. Open **Library Health** in the sidebar "
                            "and click *Test embedding model* to see why."
                        )
                    if "d_library_saved_batches" not in st.session_state:
                        st.session_state.d_library_saved_batches = set()
                    st.session_state.d_library_saved_batches.add(_batch_key)

            # ── Auto-Index on Completion ───────────────────────────────────
            if completed == total_t and total_t > 0:
                current_batch_ids = [t["id"] for t in tasks if t["status"] == "completed"]
                if current_batch_ids:
                    indexed_batch_ids = st.session_state.get("d_indexed_batches", set())
                    batch_key = tuple(sorted(current_batch_ids))
                    
                    # Auto-indexing disabled for now
                    # if batch_key not in indexed_batch_ids:
                    #     with st.status("ðŸ§  Auto-indexing new footage for AI Visual Search...") as status:
                    #         from core.indexer import VideoIndexer
                    #         indexer = VideoIndexer()
                    #         count = 0
                    #         for t in tasks:
                    #             if t["status"] == "completed" and os.path.exists(t["output_path"]):
                    #                 status.update(label=f"Indexing {os.path.basename(t['output_path'])}...", state="running")
                    #                 indexer.index_video(t["output_path"], video_url=t.get("url"))
                    #                 count += 1
                    #         status.update(label=f"Done! Auto-indexed {count} scenes into your library.", state="complete")
                    #     
                    #     if "d_indexed_batches" not in st.session_state:
                    #          st.session_state.d_indexed_batches = set()
                    #     st.session_state.d_indexed_batches.add(batch_key)
                    #     save_cache()
            # Auto-refresh while anything is still moving (downloading, queued,
            # paused, or post-download processing).
            in_motion = (stats["downloading"] + stats["queued"]
                         + stats.get("paused", 0) + stats.get("processing", 0))
            if in_motion > 0:
                time.sleep(1); st.rerun()

    # ── Step 6 — AI Text Overlays ───────────────────────────────────────────
    if st.session_state.director_shots:
        st.header("Step 6: AI Text Overlays (On-Screen Captions)")
        with st.expander("✨ Generate High-Impact Overlays", expanded=False):
            st.write("Extract money, stats, and key headings from your script automatically.")
            if st.button("Extract Highlights with AI"):
                if not st.session_state.script_text:
                    st.error("Please upload a script first.")
                else:
                    with st.spinner("Analyzing script for highlights..."):
                        # Hand the LLM the Whisper segments when we have them —
                        # those carry real-audio timestamps that include
                        # silences, so the highlight start/end times line up
                        # with the voice instead of being estimated from text.
                        _segs_for_hl = st.session_state.get("transcription_segments") or None
                        st.session_state.text_overlays = extract_highlights(
                            st.session_state.script_text,
                            os.getenv("GROQ_API_KEY"),
                            segments=_segs_for_hl,
                        )
                        if _segs_for_hl:
                            st.success(f"Extracted {len(st.session_state.text_overlays)} highlights (timed against the real audio).")
                        else:
                            st.success(f"Extracted {len(st.session_state.text_overlays)} highlights!")
            
            if st.session_state.text_overlays:
                st.subheader("Fine-tune Overlays & SFX")
                # Editable data table.
                #
                # Cell-jump fix: we used to eagerly write the editor's
                # returned DataFrame back into st.session_state.text_overlays
                # on every rerun. The round-trip through
                # `DataFrame.to_dict('records')` quietly coerces types
                # (None -> NaN, int -> np.int64), so the `!=` equality guard
                # treated every rerun as a "change" and overwrote the input
                # backing the data_editor. That mutated the widget's input
                # mid-edit, Streamlit's cached delta got stomped, and the
                # cell visibly reverted the moment the user pressed Enter.
                #
                # The data_editor's own widget state (keyed by "ov_editor")
                # already persists edits across reruns, and the Generate
                # PNGs handler below reads `edited_df.to_dict('records')`
                # directly and writes it to session_state at that point —
                # which is the right place to commit. So we just render the
                # editor and stop touching session_state until commit time.
                #
                # Ensure each overlay row has a 'size' field so the
                # data_editor renders the per-row size column without "None"
                # gaps in the UI. None means "use global size"; a number
                # overrides for that row.
                for _ov_row in st.session_state.text_overlays:
                    _ov_row.setdefault("size", None)
                edited_df = st.data_editor(
                    st.session_state.text_overlays,
                    num_rows="dynamic",
                    width="stretch",
                    key="ov_editor",
                    column_config={
                        "size": st.column_config.NumberColumn(
                            "Size (px)",
                            help=(
                                "Per-row font size override (in 1080p pixels). "
                                "Leave blank to use the global Font Size set below."
                            ),
                            min_value=20,
                            max_value=500,
                            step=2,
                        ),
                    },
                )

                def _resolved_overlays_from_editor():
                    """Return overlays with the data_editor's deltas applied.

                    Streamlit 1.57's data_editor, when fed a list-of-dicts
                    AND given a `key`, returns the original input list
                    unchanged — the user's edits live ONLY in
                    st.session_state["ov_editor"] as deltas
                    (edited_rows / added_rows / deleted_rows). That's why
                    editing `highlight_text` in the table never made it
                    into the rendered PNG: `edited_df` looked like the
                    input. The widget state IS authoritative, so we apply
                    it onto a fresh copy of the input here.
                    """
                    base = [dict(o) for o in (st.session_state.text_overlays or [])]
                    state = st.session_state.get("ov_editor") or {}

                    # edited_rows: {row_idx: {col_name: new_value}}.
                    # Streamlit stores row_idx as int but be defensive.
                    for row_key, col_changes in (state.get("edited_rows") or {}).items():
                        try:
                            i = int(row_key)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= i < len(base) and isinstance(col_changes, dict):
                            base[i].update(col_changes)

                    # added_rows: list of dicts the user inserted.
                    for row in (state.get("added_rows") or []):
                        if isinstance(row, dict):
                            new_row = dict(row)
                            new_row.setdefault("size", None)
                            base.append(new_row)

                    # deleted_rows: list of row indices the user removed.
                    deleted_idx = []
                    for r in (state.get("deleted_rows") or []):
                        try:
                            deleted_idx.append(int(r))
                        except (TypeError, ValueError):
                            continue
                    for i in sorted(set(deleted_idx), reverse=True):
                        if 0 <= i < len(base):
                            base.pop(i)

                    return base

                st.divider()
                st.subheader("Visual Settings")
                _ov = st.session_state.overlay_settings

                # ── Row 1: Typography & Layout ──────────────────────────────────
                c_v1, c_v2, c_v3 = st.columns(3)
                with c_v1:
                    _avail_fonts = get_available_fonts()
                    _font_names = list(_avail_fonts.keys()) or ["Arial Bold"]
                    _fi = _font_names.index(_ov["font_family"]) if _ov["font_family"] in _font_names else 0
                    _ov["font_family"] = st.selectbox("Font Family", _font_names, index=_fi)
                    _ov["size"] = st.slider(
                        "Font Size (global default, px @ 1080p)",
                        20, 500, _ov["size"],
                        help="Used for any row that doesn't set its own Size column above.",
                    )
                with c_v2:
                    _ov["color"] = st.color_picker("Text Color", _ov["color"])
                    _ov["text_opacity"] = st.slider("Text Opacity", 0, 255, _ov.get("text_opacity", 255))
                with c_v3:
                    _ov["placement"] = st.selectbox("Visual Placement", ["Top", "Middle", "Bottom"],
                                                    index=["Top","Middle","Bottom"].index(_ov.get("placement","Bottom")))
                    _ov["effect_type"] = st.radio("Text Effect", ["Shadow", "Outline"], horizontal=True,
                                                   index=0 if _ov.get("effect_type","Shadow") == "Shadow" else 1)
                    _ov["shadow"] = st.color_picker("Effect Color", _ov["shadow"])

                # ── Position fine-tune ──────────────────────────────────────────
                # Offsets stack on top of the Top/Middle/Bottom anchor above —
                # picking "Bottom" with Y offset +50 nudges the text 50 px lower
                # than the default 850. The live preview below uses the same
                # math the PNG generator does, so what you see is final output.
                c_p1, c_p2, c_p3 = st.columns([2, 2, 1])
                with c_p1:
                    _ov["y_offset"] = st.slider(
                        "Y offset (px, +down)", -400, 400,
                        int(_ov.get("y_offset", 0)), 5,
                        help="Vertical nudge from the placement anchor, in 1080p pixels.",
                    )
                with c_p2:
                    _ov["x_offset"] = st.slider(
                        "X offset (px, +right)", -800, 800,
                        int(_ov.get("x_offset", 0)), 5,
                        help="Horizontal shift from centre, in 1080p pixels.",
                    )
                with c_p3:
                    if st.button("Reset offsets", key="ov_reset_offsets"):
                        _ov["y_offset"] = 0
                        _ov["x_offset"] = 0
                        st.rerun()

                # ── Row 2: Background Box & Animation ───────────────────────────
                c_a1, c_a2, c_a3 = st.columns(3)
                with c_a1:
                    _ov["bg_box"] = st.checkbox("Background Box", value=_ov.get("bg_box", False))
                    if _ov["bg_box"]:
                        _ov["bg_box_color"]   = st.color_picker("Box Color", _ov.get("bg_box_color", "#000000"))
                        _ov["bg_box_opacity"] = st.slider("Box Opacity", 0, 255, _ov.get("bg_box_opacity", 160))
                with c_a2:
                    _ov["animation"] = st.selectbox("Animation Style",
                                                     ["None", "Fade In/Out", "Slide Up", "Slide In Left", "Random"],
                                                     index=["None","Fade In/Out","Slide Up","Slide In Left","Random"].index(
                                                         _ov.get("animation","Fade In/Out")))
                    _ov["no_anim_categories"] = st.multiselect(
                        "No Animation for Categories",
                        ["headings/titles", "money/pricing", "statistics", "core concepts"],
                        default=_ov.get("no_anim_categories", []),
                        help="These categories always render as static (no animation).",
                    )
                with c_a3:
                    _ov["emoji_prefix"] = st.checkbox("Emoji Category Prefix", value=_ov.get("emoji_prefix", False),
                                                       help="Prepends 📌 headings  💰 money  📊 stats  💡 concepts")

                # ── Per-Category Colors ─────────────────────────────────────────
                with st.expander("Per-Category Colors"):
                    _cat_cols = st.columns(4)
                    _cat_list = ["headings/titles", "money/pricing", "statistics", "core concepts"]
                    _cat_defaults = {"headings/titles":"#FFFFFF","money/pricing":"#FFD700",
                                     "statistics":"#00BFFF","core concepts":"#FFFF00"}
                    _ov.setdefault("category_colors", _cat_defaults)
                    for ci, cat in enumerate(_cat_list):
                        with _cat_cols[ci]:
                            _ov["category_colors"][cat] = st.color_picker(
                                cat.title(), _ov["category_colors"].get(cat, _cat_defaults[cat]),
                                key=f"catcol_{cat}"
                            )

                # ── Live 1080p preview ──────────────────────────────────────────
                # Lets the user judge size/contrast against a real frame before
                # committing to a full PNG batch — pulls a still from one of
                # their downloaded clips so it looks like the actual edit.
                with st.expander("🖼️ Live 1080p Preview", expanded=True):
                    p_name_prev = st.session_state.get("project_name", "default")
                    proj_folder_prev = _safe_for_fs(p_name_prev, 50)
                    proj_dl_dir = os.path.abspath(
                        os.path.join("downloads", "director", proj_folder_prev)
                    )

                    # Collect downloaded video files from this project (recursively
                    # — clips may live in chunk subfolders).
                    _video_exts = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
                    clip_paths = []
                    if os.path.isdir(proj_dl_dir):
                        for root, _, files in os.walk(proj_dl_dir):
                            # Skip overlay and sfx output folders
                            if os.path.basename(root) in ("overlays", "sfx"):
                                continue
                            for f in files:
                                if f.lower().endswith(_video_exts):
                                    clip_paths.append(os.path.join(root, f))
                    clip_paths.sort()

                    pc1, pc2 = st.columns([2, 3])
                    with pc1:
                        if clip_paths:
                            _clip_labels = ["(neutral gradient)"] + [
                                os.path.relpath(p, proj_dl_dir) for p in clip_paths
                            ]
                            _last_choice = st.session_state.get("ov_preview_clip", _clip_labels[0])
                            if _last_choice not in _clip_labels:
                                _last_choice = _clip_labels[0]
                            picked = st.selectbox(
                                "Background clip",
                                _clip_labels,
                                index=_clip_labels.index(_last_choice),
                                key="ov_preview_clip",
                                help="Pick a downloaded clip; a frame is grabbed at the timestamp below.",
                            )
                            picked_path = None if picked == _clip_labels[0] else clip_paths[_clip_labels.index(picked) - 1]
                        else:
                            st.info("No downloaded clips found in this project — preview will use a neutral gradient.")
                            picked_path = None

                        frame_ts = st.slider(
                            "Frame timestamp (sec)", 0.0, 30.0,
                            float(st.session_state.get("ov_preview_ts", 1.0)),
                            0.5, key="ov_preview_ts",
                        )

                        # Source the sample text from the overlay rows so the
                        # preview reflects real content. Heading rows use the
                        # full caption; others use highlight_text. Pull from
                        # the helper so in-progress edits in the table show
                        # up in the preview immediately — without waiting
                        # for the Generate-PNGs click to commit.
                        _sample_src = _resolved_overlays_from_editor()
                        sample_options = []
                        for _i, _ov_row in enumerate(_sample_src):
                            _cat = str(_ov_row.get("category", "")).lower()
                            _is_hdg = "heading" in _cat
                            _txt = str(_ov_row.get("original_caption", "") if _is_hdg else _ov_row.get("highlight_text", "")).strip()
                            if _txt:
                                sample_options.append((f"#{_i+1}: {_txt[:60]}", _txt, _ov_row.get("size"), _cat))
                        sample_options.append(("(custom text)", "$10,000 SAVED", None, ""))

                        _labels = [o[0] for o in sample_options]
                        _last_label = st.session_state.get("ov_preview_sample_label", _labels[0])
                        if _last_label not in _labels:
                            _last_label = _labels[0]
                        chosen_label = st.selectbox(
                            "Sample highlight", _labels,
                            index=_labels.index(_last_label),
                            key="ov_preview_sample_label",
                        )
                        chosen = sample_options[_labels.index(chosen_label)]
                        if chosen[0] == "(custom text)":
                            preview_text = st.text_input("Custom text", chosen[1], key="ov_preview_custom_text")
                            preview_row_size = None
                            preview_cat = ""
                        else:
                            preview_text = chosen[1]
                            preview_row_size = chosen[2]
                            preview_cat = chosen[3]

                    with pc2:
                        # Resolve effective size: per-row override wins, then global.
                        try:
                            eff_size = int(preview_row_size) if preview_row_size not in (None, "", 0) else _ov["size"]
                        except (TypeError, ValueError):
                            eff_size = _ov["size"]

                        # Per-category color: same logic as PNG generator.
                        eff_color = _ov["color"]
                        _cat_colors_pv = _ov.get("category_colors", {})
                        for _ck, _cv in _cat_colors_pv.items():
                            if preview_cat and (_ck in preview_cat or preview_cat in _ck):
                                eff_color = _cv
                                break

                        placement_map_pv = {"Top": 120, "Middle": 480, "Bottom": 850}
                        y_anchor_pv = placement_map_pv.get(_ov.get("placement", "Bottom"), 850)
                        y_off_pv = int(_ov.get("y_offset", 0))
                        x_off_pv = int(_ov.get("x_offset", 0))
                        # Clamp the effective Y so the text centre never lands
                        # outside the 1080p canvas — keeps the preview readable
                        # even when the user drags the slider to an extreme.
                        y_pv = max(40, min(1040, y_anchor_pv + y_off_pv))

                        _avail_fonts_pv = get_available_fonts()
                        _font_path_pv = _avail_fonts_pv.get(_ov.get("font_family", ""), None)
                        _is_hdg_pv = "heading" in preview_cat

                        try:
                            preview_img = render_overlay_preview(
                                text=preview_text or " ",
                                video_path=picked_path,
                                video_timestamp_sec=frame_ts,
                                font_path=_font_path_pv,
                                font_size=eff_size,
                                color=eff_color,
                                shadow_color=_ov["shadow"],
                                y_position=y_pv,
                                x_offset=x_off_pv,
                                bg_color=_ov.get("bg_box_color") if _ov.get("bg_box") else None,
                                bg_opacity=_ov.get("bg_box_opacity", 160),
                                outline=_ov.get("effect_type", "Shadow") == "Outline",
                                auto_scale=_is_hdg_pv,
                                text_opacity=_ov.get("text_opacity", 255),
                            )
                            st.image(
                                preview_img,
                                caption=(
                                    f"1080p preview · {eff_size}px · "
                                    f"{_ov.get('placement','Bottom')} (Y={y_pv}, X offset={x_off_pv})"
                                ),
                                width="stretch",
                            )
                        except Exception as _e:
                            st.error(f"Preview render failed: {_e}")

                if st.button("Generate & Preview Overlays", type="primary"):
                    p_name = st.session_state.get("project_name", "default")
                    proj_folder = _safe_for_fs(p_name, 50)
                    ov_dir = os.path.abspath(os.path.join("downloads", "director", proj_folder, "overlays"))
                    os.makedirs(ov_dir, exist_ok=True)
                    
                    def ts_to_sec(ts):
                        if isinstance(ts, (int, float)): return float(ts)
                        if not ts: return 0.0
                        # HH:MM:SS,mmm or HH:MM:SS
                        parts = re.split('[:.,]', str(ts))
                        if len(parts) >= 3:
                            try:
                                h, m, s = map(int, parts[:3])
                                ms = int(parts[3]) if len(parts) > 3 else 0
                                return h * 3600 + m * 60 + s + ms / 1000.0
                            except (ValueError, IndexError):
                                pass
                        try:
                            return float(ts)
                        except:
                            return 0.0

                    with st.spinner("Generating transparent PNGs..."):
                        # Apply the data_editor's widget-state deltas onto
                        # a fresh copy of the original overlays. Trusting
                        # `edited_df` here silently drops every cell edit
                        # in Streamlit 1.57 (see helper comment above).
                        final_ovs = _resolved_overlays_from_editor()
                        # Map placement to Y coordinate for Pillow, then apply
                        # the Y offset slider on top so the generated PNGs land
                        # at exactly the position the user dialed in on the
                        # preview. Clamp to the canvas just like the preview.
                        placement_map = {"Top": 120, "Middle": 480, "Bottom": 850}
                        _placement = st.session_state.overlay_settings["placement"]
                        _y_anchor = placement_map.get(_placement, 850)
                        _y_off = int(st.session_state.overlay_settings.get("y_offset", 0))
                        _x_off = int(st.session_state.overlay_settings.get("x_offset", 0))
                        target_y = max(40, min(1040, _y_anchor + _y_off))

                        import random as _random
                        _ov_s        = st.session_state.overlay_settings
                        _anim_pool   = ["Fade In/Out", "Slide Up", "Slide In Left"]
                        _global_anim = _ov_s["animation"]
                        _no_anim_cats = [c.lower() for c in _ov_s.get("no_anim_categories", [])]
                        _avail_fonts  = get_available_fonts()
                        _sel_font_path = _avail_fonts.get(_ov_s.get("font_family", ""), None)
                        _cat_colors   = _ov_s.get("category_colors", {})
                        _emoji_map    = {
                            "headings/titles": "📌",
                            "money/pricing":   "💰",
                            "statistics":      "📊",
                            "core concepts":   "💡",
                        }
                        _use_emoji    = _ov_s.get("emoji_prefix", False)
                        _use_outline  = _ov_s.get("effect_type", "Shadow") == "Outline"
                        _use_bg       = _ov_s.get("bg_box", False)
                        _bg_col       = _ov_s.get("bg_box_color", "#000000") if _use_bg else None
                        _bg_opa       = _ov_s.get("bg_box_opacity", 160)
                        _txt_opa      = _ov_s.get("text_opacity", 255)

                        for idx, ov in enumerate(final_ovs):
                            fname     = os.path.join(ov_dir, f"overlay_{idx+1}.png")
                            cat_lower = str(ov.get("category", "")).lower()
                            is_heading = "heading" in cat_lower

                            # Text: headings use full original_caption; others use highlight_text
                            overlay_text = str(ov.get("original_caption", "") if is_heading else ov.get("highlight_text", ""))

                            # Emoji prefix
                            if _use_emoji:
                                for ek, ev in _emoji_map.items():
                                    if ek in cat_lower or cat_lower in ek:
                                        overlay_text = f"{ev} {overlay_text}"
                                        break

                            # Per-category color (fall back to global color)
                            ov_color = _ov_s["color"]
                            for ck, cv in _cat_colors.items():
                                if ck in cat_lower or cat_lower in ck:
                                    ov_color = cv
                                    break

                            # Per-row size override → blank cell falls back to global.
                            _row_size_raw = ov.get("size")
                            try:
                                _row_size = int(_row_size_raw) if _row_size_raw not in (None, "", 0) else None
                            except (TypeError, ValueError):
                                _row_size = None
                            row_font_size = _row_size if _row_size else _ov_s["size"]

                            create_text_overlay(
                                overlay_text,
                                fname,
                                font_path=_sel_font_path,
                                font_size=row_font_size,
                                color=ov_color,
                                shadow_color=_ov_s["shadow"],
                                y_position=target_y,
                                x_offset=_x_off,
                                bg_color=_bg_col,
                                bg_opacity=_bg_opa,
                                outline=_use_outline,
                                auto_scale=is_heading,
                                text_opacity=_txt_opa,
                            )
                            ov["filepath"] = fname
                            # Animation: suppression wins, then random, then global
                            cat_suppressed = any(no_cat in cat_lower or cat_lower in no_cat for no_cat in _no_anim_cats)
                            if cat_suppressed:
                                ov["animation"] = "None"
                            elif _global_anim == "Random":
                                ov["animation"] = _random.choice(_anim_pool)
                            else:
                                ov["animation"] = _global_anim
                            ov["start_sec"] = ts_to_sec(ov.get("start_time", 0))
                            raw_end = ov.get("end_time")
                            ov["end_sec"] = ts_to_sec(raw_end) if raw_end else (ov["start_sec"] + 3)
                        st.session_state.text_overlays = final_ovs
                        st.success(f"Generated {len(st.session_state.text_overlays)} PNG overlays in {ov_dir}")

                    # Preview generated overlays
                    all_preview = [ov for ov in st.session_state.text_overlays if ov.get("filepath") and os.path.exists(ov["filepath"])]
                    if all_preview:
                        st.subheader("Preview")
                        _all_cats = sorted({str(ov.get("category","")).lower() for ov in all_preview if ov.get("category")})
                        _filter_cats = st.multiselect("Filter by category", _all_cats, default=_all_cats, key="prev_cat_filter")
                        preview_ovs = [ov for ov in all_preview if str(ov.get("category","")).lower() in _filter_cats]
                        if preview_ovs:
                            cols = st.columns(min(4, len(preview_ovs)))
                            for i, ov in enumerate(preview_ovs):
                                with cols[i % len(cols)]:
                                    is_hdg = "heading" in str(ov.get("category", "")).lower()
                                    preview_cap = ov.get("original_caption", "") if is_hdg else ov.get("highlight_text", "")
                                    st.image(ov["filepath"], caption=preview_cap, width="stretch")
                                    if ov.get("sfx_path") and os.path.exists(ov["sfx_path"]):
                                        st.audio(ov["sfx_path"])

                    # --- SFX Download Logic ---
                    freesound_api = os.getenv("FREESOUND_API_KEY")
                    if freesound_api:
                        with st.spinner("Downloading Sound Effects from Freesound..."):
                            sfx_dir = os.path.abspath(os.path.join("downloads", "director", proj_folder, "sfx"))
                            os.makedirs(sfx_dir, exist_ok=True)
                            
                            downloaded_count = 0
                            for ov in st.session_state.text_overlays:
                                sfx_query = ov.get("suggested_sfx")
                                if sfx_query and sfx_query.lower() != "none":
                                    sfx_data = search_freesound(sfx_query, freesound_api)
                                    if sfx_data:
                                        sfx_ext = sfx_data.get("type", "mp3")
                                        sfx_fname = os.path.join(sfx_dir, f"sfx_{ov.get('highlight_text', 'sound')}.{sfx_ext}")
                                        if download_sfx(sfx_data, sfx_fname, freesound_api):
                                            ov["sfx_path"] = sfx_fname
                                            downloaded_count += 1
                            if downloaded_count > 0:
                                st.success(f"Downloaded {downloaded_count} sound effects to {sfx_dir}")
                    else:
                        st.info("💡 Tip: Add FREESOUND_API_KEY to your .env to automatically download sound effects!")
    # ── Step 7 — Export ──────────────────────────────────────────────────────
    if st.session_state.director_shots:
        st.header("Step 7: Export")
        
        col_s1, col_s2 = st.columns([1, 3])
        with col_s1:
            num_parts = st.number_input("Split XML into Parts", value=1, min_value=1, max_value=10, help="Large projects can be split to save RAM in Premiere Pro.")
        
        shot_list_json = json.dumps(st.session_state.director_shots, indent=2)
        shot_list_txt  = generate_shot_list_txt(st.session_state.director_shots)
        p_name = st.session_state.get("project_name", "default")
        proj_folder = _safe_for_fs(p_name, 50)

        # ── Generate XML Parts ──
        xml_parts = []
        total_shots = len(st.session_state.director_shots)
        chunk_size = math.ceil(total_shots / num_parts)
        
        for i in range(num_parts):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            shot_chunk = st.session_state.director_shots[start_idx:end_idx]
            if not shot_chunk:
                break
            
            # Filter overlays & SFX for this specific chunk
            chunk_overlays = filter_overlays_for_shots(shot_chunk, st.session_state.text_overlays)
            
            chunk_sfx = []
            if "text_overlays" in st.session_state:
                # We reuse the filtered overlays to find valid SFX paths
                chunk_sfx = [ov for ov in chunk_overlays if ov.get("sfx_path")]
                # Map back to the structure generate_fcpxml expects
                chunk_sfx = [{"filepath": s["sfx_path"], "start_sec": s["start_sec"]} for s in chunk_sfx]

            part_display_name = p_name if num_parts == 1 else f"{p_name}_Part_{i+1}"
            # When the timeline is split, re-zero each chunk so its first shot
            # lives at frame 0 of its own sequence. Otherwise chunk 2's clips
            # keep their absolute frame positions (shot 29 ≈ 29 s in) and the
            # imported sequence has a giant empty gap at the head — which is
            # what showed shot 29's footage landing right after shot 23's.
            chunk_offset = float(shot_chunk[0].get("timestamp", 0.0)) if num_parts > 1 else 0.0
            xml_content = generate_fcpxml(
                shot_chunk,
                project_name=part_display_name,
                overlays=chunk_overlays,
                sfx_list=chunk_sfx,
                time_offset=chunk_offset,
            )
            
            fname = "b_roll_sequence.xml" if num_parts == 1 else f"b_roll_sequence_part_{i+1}.xml"
            xml_parts.append({
                "filename": fname,
                "content": xml_content,
                "offset_sec": chunk_offset,
                "first_slot": shot_chunk[0].get("slot_id"),
                "last_slot": shot_chunk[-1].get("slot_id"),
            })
            
            # Auto-save each part to disk
            xml_path = os.path.join("downloads", "director", proj_folder, fname)
            try:
                os.makedirs(os.path.dirname(xml_path), exist_ok=True)
                with open(xml_path, "w", encoding="utf-8") as f:
                    f.write(xml_content)
            except:
                pass

        trans_srt = ""
        if st.session_state.transcription_segments:
            trans_srt = generate_transcription_srt(st.session_state.transcription_segments)

        # Standalone overlays-only XML. Lets the user re-import just the text
        # overlays into an existing Premiere sequence after editing — no need
        # to re-import the whole b-roll timeline. Same absolute timecode as
        # the main export, so the clipitems can be pasted onto V2 directly.
        overlays_xml = ""
        if st.session_state.text_overlays:
            try:
                overlays_xml = generate_overlays_fcpxml(
                    st.session_state.text_overlays,
                    project_name=p_name,
                    time_offset=0.0,
                )
                ov_xml_path = os.path.join("downloads", "director", proj_folder, "b_roll_overlays.xml")
                os.makedirs(os.path.dirname(ov_xml_path), exist_ok=True)
                with open(ov_xml_path, "w", encoding="utf-8") as f:
                    f.write(overlays_xml)
            except Exception:
                overlays_xml = ""

        # Mirror SRT showing "Shot N" cues on the same timeline as the audio,
        # so the user can drag it onto their voiceover and visually verify
        # each shot lands on the right line (silences included). Auto-saved
        # next to the XML for convenience.
        shots_srt = generate_shots_srt(st.session_state.director_shots)
        try:
            shots_srt_path = os.path.join("downloads", "director", proj_folder, "shots.srt")
            os.makedirs(os.path.dirname(shots_srt_path), exist_ok=True)
            with open(shots_srt_path, "w", encoding="utf-8") as f:
                f.write(shots_srt)
        except Exception:
            pass

        failed_tasks = st.session_state.dm.get_failed_tasks()
        failed_txt = ""
        if failed_tasks:
            failed_txt = generate_failed_downloads_txt(failed_tasks)
            
        c1, c2, c3     = st.columns(3)
        with c1:
            st.download_button("shot_list.json", data=shot_list_json, file_name="shot_list.json", mime="application/json")
            if trans_srt:
                st.download_button("transcription.srt", data=trans_srt, file_name="transcription.srt", mime="text/plain")
            if shots_srt:
                st.download_button(
                    "shots.srt",
                    data=shots_srt,
                    file_name="shots.srt",
                    mime="text/plain",
                    help="Drop onto your voiceover in any player to verify each shot lands on the right line — silences are folded into the previous shot's cue.",
                )
        with c2: 
            st.download_button("shot_list.txt",  data=shot_list_txt,  file_name="shot_list.txt",  mime="text/plain")
            if failed_txt:
                st.download_button("failed_downloads.txt", data=failed_txt, file_name="failed_downloads.txt", mime="text/plain")
        with c3:
            if num_parts == 1:
                st.download_button(
                    "B-Roll Sequence (.xml)",
                    data=xml_parts[0]["content"],
                    file_name="b_roll_sequence.xml",
                    mime="text/xml",
                    help="Import this file into Premiere Pro to build your sequence."
                )
            if overlays_xml:
                st.download_button(
                    "Overlays Only (.xml)",
                    data=overlays_xml,
                    file_name="b_roll_overlays.xml",
                    mime="text/xml",
                    help=(
                        "Standalone XML containing ONLY the text-overlay PNG track. "
                        "Use it to re-import just the overlays after editing — "
                        "no need to re-import the full b-roll sequence. Same "
                        "timecode as the main export, so paste onto V2 of your "
                        "existing sequence."
                    ),
                )
            else:
                st.write("**XML Sequences (Split):**")
                st.caption(
                    "Each part is a self-contained sequence starting at frame 0. "
                    "Drop each part at the listed audio offset on your master timeline "
                    "to line back up with the voiceover."
                )
                for p in xml_parts:
                    off = p.get("offset_sec", 0.0) or 0.0
                    h = int(off) // 3600
                    m = (int(off) % 3600) // 60
                    s = off - h * 3600 - m * 60
                    off_str = f"{h:02d}:{m:02d}:{s:06.3f}"
                    st.download_button(
                        f"Download {p['filename']}  (shots {p.get('first_slot')}–{p.get('last_slot')} • starts at {off_str})",
                        data=p["content"],
                        file_name=p["filename"],
                        mime="text/xml",
                        help=f"Drag this sequence's content to {off_str} on your master timeline to align with the voiceover.",
                    )

        # ── Learn preferred trims from a re-imported Premiere edit ─────────────
        # Closes the loop: export → trim in Premiere → re-import here. The app
        # records where you cut each clip and starts future exports of the same
        # clip at that in-point (see _preferred_in_frame in core/output.py).
        st.divider()
        _trims_learned = get_library_stats().get("trims", 0)
        _reimport_label = (
            f"🎯 Teach the Clip Library your trims  ·  ✅ {_trims_learned} learned"
            if _trims_learned else
            "🎯 Teach the Clip Library your trims (re-import an edited XML)"
        )
        with st.expander(_reimport_label):
            st.caption(
                "After you fine-tune in/out points in Premiere, export the sequence "
                "back to FCP7 XML (File ▸ Export ▸ Final Cut Pro XML) and upload it "
                "here. B-Roll Finder learns where you trimmed each clip and will "
                "start future exports of that clip at the same in-point. Only clips "
                "you downloaded through B-Roll Finder (already in your Clip Library) "
                "can be matched."
            )

            # Persistent state — trims live in the DB, so show what's already
            # learned after a restart and let the user redo / clear it.
            if _trims_learned:
                _recent = get_recent_trims(limit=5)
                st.info(
                    f"📌 **{_trims_learned} preferred trim(s) already learned** "
                    "(persisted from earlier imports). Re-uploading an edited XML "
                    "**updates** matching clips; clearing starts fresh. Exports already "
                    "use these in-points."
                )
                if _recent:
                    with st.expander("Recently learned trims"):
                        for _t in _recent:
                            _title = (_t.get("clip_title") or "clip")[:48]
                            _when = (_t.get("confirmed_at") or "")[:19].replace("T", " ")
                            st.write(
                                f"• **{_title}** — in {float(_t['in_seconds']):.2f}s → "
                                f"out {float(_t['out_seconds']):.2f}s"
                                + (f"  ·  _{_when}_" if _when else "")
                            )
                _cc1, _cc2 = st.columns([3, 1])
                with _cc2:
                    if st.button("🗑️ Clear all", key="d_reimport_clear",
                                 help="Delete all learned trims so you can redo from scratch. "
                                      "Your clips and embeddings are untouched."):
                        from core.clip_library import clear_trims
                        _n = clear_trims()
                        st.success(f"Cleared {_n} learned trim(s).")
                        st.rerun()

            reimport_file = st.file_uploader(
                "Upload edited sequence (.xml)", type=["xml"], key="d_reimport_xml"
            )
            create_missing = st.checkbox(
                "Create library entries for clips not already saved",
                value=True,
                key="d_reimport_create",
                help=(
                    "On: learn trims for every video clip in the XML, adding any "
                    "that aren't in your Clip Library yet (audio/SFX are ignored). "
                    "Off: only learn trims for clips already downloaded through "
                    "B-Roll Finder."
                ),
            )
            if reimport_file is not None and st.button(
                "Learn trims from this edit", key="d_reimport_btn", type="primary"
            ):
                try:
                    from core.xml_reimport import ingest_reimported_xml
                    summary = ingest_reimported_xml(
                        reimport_file.getvalue(), create_missing=create_missing
                    )
                    if summary["recorded"]:
                        bits = []
                        if summary["matched"]:
                            bits.append(f"{summary['matched']} already in library")
                        if summary["created"]:
                            bits.append(f"{summary['created']} newly added")
                        detail = f" ({', '.join(bits)})" if bits else ""
                        st.success(
                            f"✅ Learned {summary['recorded']} trim(s) from "
                            f"{summary['video']} video clip(s){detail}. "
                            "Future exports of these clips will start at your in-points."
                        )
                    elif summary["video"] == 0:
                        st.warning(
                            "No video clips found in that XML "
                            f"({summary['skipped_non_video']} audio/SFX clip(s) skipped)."
                        )
                    else:
                        st.warning(
                            f"Found {summary['video']} video clip(s) but learned no trims. "
                            "Enable the checkbox above to add clips that aren't in your "
                            "library yet."
                        )
                    if summary["skipped_non_video"]:
                        st.caption(
                            f"Skipped {summary['skipped_non_video']} non-video clip(s) "
                            "(voiceover / sound effects)."
                        )
                    if summary["unmatched"]:
                        with st.expander(f"⚠️ {len(summary['unmatched'])} unresolved clip(s)"):
                            for nm in summary["unmatched"]:
                                st.write(f"• {nm}")
                except Exception as e:
                    st.error(f"Could not process that XML: {e}")
