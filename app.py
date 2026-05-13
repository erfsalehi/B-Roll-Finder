import streamlit as st
import os
import json
import zipfile
import io
import pandas as pd
from dotenv import load_dotenv, set_key
from core.timing import get_audio_duration, parse_script_to_slots, calculate_wps
from core.keywords import generate_keywords_for_slots, generate_keywords_with_ai_chunking
from core.youtube import fetch_youtube_results
from core.stock_apis import search_pexels, search_pixabay
from core.output import (
    generate_keywords_txt, generate_youtube_txt, generate_srt,
    generate_transcription_srt, generate_failed_downloads_txt,
    generate_fcpxml, generate_shot_list_txt, filter_overlays_for_shots, _safe_for_fs
)
from core.captions import extract_highlights, create_text_overlay
from core.download_manager import DownloadManager, MAX_RETRIES, link_or_copy
from core import download_cache
from core.app_utils import check_network, format_speed, format_eta
from core.session_cache import load_session_cache, save_session_cache
import time
import math

# --- Config & Initialization ---
st.set_page_config(page_title="B-Roll Finder", layout="wide")
ENV_FILE = ".env"
CACHE_FILE = ".cache/session_state.json"
if not os.path.exists(".cache"):
    os.makedirs(".cache")
load_dotenv(ENV_FILE)
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
        "y": 800,
        "animation": "Fade In/Out"
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
    save_cache()

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
    new_shots = generate_shot_list_from_transcription(
        chunk.get("segments", []),
        os.getenv("GROQ_API_KEY"),
        custom_instructions=st.session_state.get("d_style", ""),
        video_topic=st.session_state.get("d_video_topic", ""),
        chunk_id=chunk_idx
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
if not st.session_state.slots and os.path.exists(CACHE_FILE):
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
            st.markdown("&nbsp;")
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
                load_dotenv(ENV_FILE, override=True)
                st.success("API keys saved to .env. Refresh the page to re-evaluate the status pill.")
            else:
                st.warning("Groq API key is required.")


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
                    use_container_width=True,
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
        video_quality = st.selectbox("Video Quality", ["1080p", "720p", "480p", "Best", "Worst"], index=0)
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
            if st.session_state.dm.max_workers != max_workers:
                st.session_state.dm = DownloadManager(max_workers=max_workers)
            
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
            g_vq = st.selectbox("Quality", ["1080p", "720p", "480p", "Best", "Worst"], index=0, key="g_vq")
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
                                if st.button(g_btn_label, key=f"gbtn_{hash(res.get('url'))}", use_container_width=True, type="primary" if is_g_picked else "secondary"):
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
    from core.director_search import fetch_director_footage
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
                if st.button("🚀 Start Bulk Ingestion", use_container_width=True):
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
                if st.button("ðŸâ€   Index Local Folder", use_container_width=True):
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
            st.markdown("&nbsp;")  # vertical spacer to align rows
        if st.button("Save API Keys", key="d_save_keys", type="primary"):
            if groq_input:
                set_key(ENV_FILE, "GROQ_API_KEY", groq_input)
                if groq_input_2:      set_key(ENV_FILE, "GROQ_API_KEY_2",      groq_input_2)
                if pexels_input:     set_key(ENV_FILE, "PEXELS_API_KEY",     pexels_input)
                if pixabay_input:    set_key(ENV_FILE, "PIXABAY_API_KEY",    pixabay_input)
                if youtube_input:    set_key(ENV_FILE, "YOUTUBE_API_KEY",    youtube_input)
                if openrouter_input: set_key(ENV_FILE, "OPENROUTER_API_KEY", openrouter_input)
                if openrouter_2_input: set_key(ENV_FILE, "OPENROUTER_API_KEY_2", openrouter_2_input)
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
            current_project = audio_file.name
            if st.session_state.get("project_name") and st.session_state.project_name != current_project:
                st.session_state.director_shots = []
                st.session_state.transcription_segments = []
                st.session_state.transcription_chunks = []
                st.session_state.active_chunk_indices = [0]
                st.session_state.script_text = ""
                st.session_state.d_review_idx = 0
            st.session_state.project_name = current_project
            audio_path = os.path.join(
                ".cache", "temp_audio_director" + os.path.splitext(audio_file.name)[1]
            )
            with open(audio_path, "wb") as f:
                f.write(audio_file.read())
            st.session_state.audio_duration = get_audio_duration(audio_path)
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
                             use_container_width=True):
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
                                 use_container_width=True,
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
        if st.button("✨ Suggest", key="d_suggest_topic", use_container_width=True, help="AI analyzes your script to suggest a topic description."):
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
                status.text("Finalizing shot list...")
                shots = generate_shot_list_from_transcription(
                    segments,
                    os.getenv("GROQ_API_KEY"),
                    custom_instructions=custom_instructions,
                    video_topic=d_video_topic,
                    chunk_id=0,
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
        # Build rows from current state.
        rows = []
        for shot in st.session_state.director_shots:
            sel = shot.get("selected_results", [])
            # Highlight missing selections with a warning emoji if priority isn't 'none'
            pick_status = f"✅ {len(sel)}" if sel else ("⏭ skip" if shot.get("skipped") else "—")
            if not sel and shot.get("priority") != "none" and not shot.get("skipped"):
                pick_status = "⚠️ MISSING"
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
            use_container_width=True,
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
            if st.button("✨ Seed from Queries", key="d_seed_yt_keywords", use_container_width=True):
                st.session_state.director_shots = seed_youtube_keywords(
                    st.session_state.director_shots,
                    max_keywords=2,
                )
                save_cache()
                st.rerun()
        with yk2:
            if st.button("🤖 Generate with AI", key="d_gen_yt_keywords", use_container_width=True):
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
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            fetch_clicked = st.button("Fetch", disabled=st.session_state.is_fetching, key="d_fetch", use_container_width=True)
        with col_btn2:
            retry_clicked = st.button("Retry Empty", disabled=st.session_state.is_fetching, key="d_retry", use_container_width=True, help="Only searches for shots that currently have 0 candidates. Preserves existing successful searches.")
        chunk_to_fetch = None
        if fetch_clicked or retry_clicked or chunk_to_fetch is not None:
            if not check_network():
                st.error("No network connection detected.")
            elif not (use_pexels and os.getenv("PEXELS_API_KEY")) and \
                 not (use_pixabay and os.getenv("PIXABAY_API_KEY")) and \
                 not use_youtube_search and \
                 not (use_youtube_api and os.getenv("YOUTUBE_API_KEY")):
                st.error("No search sources enabled. Add Pexels/Pixabay keys or enable YouTube Search.")
            elif use_youtube_api and not os.getenv("YOUTUBE_API_KEY"):
                st.error("YouTube API requires a YouTube API key. Untick YouTube API or add the key in Setup.")
            else:
                st.session_state.is_fetching = True
                d_fetch_errors = []
                try:
                    pbar2 = st.progress(0)
                    status2 = st.empty()
                    
                    target_shots = st.session_state.director_shots
                    if chunk_to_fetch is not None:
                        target_shots = [s for s in st.session_state.director_shots if s.get("chunk_id") == chunk_to_fetch]
                        status2.info(f"Fetching candidates for Chunk {chunk_to_fetch+1}…")
                    else:
                        status2.text("Fetching candidates…")
                    
                    from core.director_search import fetch_director_footage
                    updated_subset = fetch_director_footage(
                        target_shots,
                        use_pexels=use_pexels,
                        use_pixabay=use_pixabay,
                        use_youtube=use_youtube_search,
                        pexels_num_results=pex_num,
                        pixabay_num_results=pix_num,
                        youtube_api_num_results=yt_api_num,
                        youtube_search_num_results=yt_search_num,
                        use_youtube_api=use_youtube_api,
                        use_youtube_search=use_youtube_search,
                        progress_callback=lambda p: pbar2.progress(p * 0.9),
                        errors=d_fetch_errors,
                        retry_only=retry_clicked,
                        min_height=min_h
                    )
                    # Smart Mode: Low-Res Proxy Fetch (hijacks YouTube search)
                    if use_smart and app_mode == "Smart Mode":
                        status2.text("🧠 Smart Mode: Fetching low-res proxies & analyzing scenes…")
                        from core.proxy_fetcher import ProxyFetcher
                        pf = ProxyFetcher()
                        for i, shot in enumerate(updated_subset):
                            if retry_clicked and shot.get("video_results"):
                                continue
                            shot_p_start = 0.9 + (0.1 * i / max(len(updated_subset), 1))
                            def _proxy_cb(p, msg, _i=i):
                                pbar2.progress(0.9 + (0.1 * (_i + p) / max(len(updated_subset), 1)))
                                status2.text(f"[Shot {_i+1}/{len(updated_subset)}] {msg}")
                            try:
                                proxy_cands = pf.fetch_for_shot(shot, max_videos=2, top_k_per_video=2, progress_cb=_proxy_cb)
                                if proxy_cands:
                                    # Generate Match Reasons in one batched Groq call
                                    from core.smart_search import SmartSearch
                                    ss_temp = SmartSearch()
                                    query = shot.get("shot_intent", "")
                                    reasons = ss_temp.generate_match_reasons_batched(query, proxy_cands, os.getenv("GROQ_API_KEY"))
                                    for cand, reason in zip(proxy_cands, reasons):
                                        cand["smart_reason"] = reason
                                    if "video_results" not in shot:
                                        shot["video_results"] = []
                                    shot["video_results"].extend(proxy_cands)
                            except Exception as e:
                                print(f"[ProxyFetch] Shot {i} failed: {e}")
                    elif use_smart:
                        # Classic Smart Library search (non-YouTube sources)
                        status2.text("Searching local library semantically…")
                        from core.smart_search import SmartSearch
                        ss = SmartSearch()
                        for i, shot in enumerate(updated_subset):
                            query = shot.get("shot_intent", "")
                            if query and (not retry_clicked or not shot.get("video_results")):
                                smart_hits = ss.search(query, k=int(smart_num))
                                if smart_hits:
                                    reasons = ss.generate_match_reasons_batched(query, smart_hits, os.getenv("GROQ_API_KEY"))
                                    for hit, reason in zip(smart_hits, reasons):
                                        cand = {
                                            "title": hit.get("video_title"),
                                            "url": hit.get("video_url") or hit.get("video_path"),
                                            "source": "smart_library",
                                            "thumbnail": None,
                                            "duration": hit.get("duration"),
                                            "matched_query": query,
                                            "score": hit.get("score"),
                                            "segment_path": hit.get("segment_path"),
                                            "smart_reason": reason
                                        }
                                        if "video_results" not in shot:
                                            shot["video_results"] = []
                                        shot["video_results"].append(cand)
                            pbar2.progress(0.9 + (0.1 * (i+1)/len(updated_subset)))
                    if chunk_to_fetch is not None:
                        update_map = {s["slot_id"]: s for s in updated_subset}
                        for s in st.session_state.director_shots:
                            if s["slot_id"] in update_map:
                                s.update(update_map[s["slot_id"]])
                    else:
                        st.session_state.director_shots = updated_subset
                        
                    pbar2.progress(1.0)
                    total_found = sum(len(s.get("video_results", [])) for s in st.session_state.director_shots)
                    status2.text("Done.")
                    if total_found > 0:
                        st.success(f"Found {total_found} candidates across {len(st.session_state.director_shots)} shots.")
                    else:
                        st.warning("No candidates found. Check your API keys or try different style hints.")
                    if d_fetch_errors:
                        unique_fe = list(dict.fromkeys(d_fetch_errors))
                        with st.expander(f"⚠️ {len(unique_fe)} fetch error(s)"):
                            for e in unique_fe[:20]:
                                st.write(f"• {e}")
                    save_cache()
                finally:
                    st.session_state.is_fetching = False
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
                    failed = sum(1 for s in st.session_state.director_shots if s.get("rank_error"))
                    total  = sum(1 for s in st.session_state.director_shots
                                 if s.get("video_results") and s.get("priority") != "none")
                    if failed == 0:
                        st.success("Candidates ranked! Irrelevant clips are hidden in the review.")
                    elif failed < total:
                        st.warning(f"Ranked {total - failed}/{total} shots. {failed} shot(s) failed — see details below and check those shots in the review.")
                    else:
                        st.error("Ranking failed for all shots. Showing original order. Check API key and rate limits.")
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
        n_pending  = len(review_shots) - n_selected - n_skipped
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
            # 1. Navigation Strip (Now at the top of the gallery)
            nav1, nav2, nav3, nav4 = st.columns([1.2, 2, 4, 1.2])
            with nav1:
                if st.button("◀ Prev", key="d_prev", disabled=idx == 0, use_container_width=True):
                    save_cache()
                    st.session_state.d_review_idx -= 1
                    st.rerun()
            with nav2:
                options = [f"Shot {i+1} / {len(review_shots)}" for i in range(len(review_shots))]
                st.session_state["d_jump_top"] = options[idx]
                selected = st.selectbox("Jump", options=options, label_visibility="collapsed", key="d_jump_top")
                new_idx_top = int(selected.split(" ")[1]) - 1
                if new_idx_top != idx:
                    save_cache()
                    st.session_state.d_review_idx = new_idx_top
                    st.rerun()
            with nav3:
                st.markdown(
                    f"<div style='text-align:center; padding-top:0.4em; font-size:14px;'>"
                    f"✅ {n_selected} selected"
                    f" &nbsp;·&nbsp; ⏭ {n_skipped} skipped"
                    f" &nbsp;·&nbsp; ⏳ {n_pending} pending"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with nav4:
                if st.button("Next ▶", key="d_next", disabled=idx == len(review_shots) - 1, use_container_width=True):
                    save_cache()
                    st.session_state.d_review_idx += 1
                    st.rerun()
            st.progress((idx + 1) / len(review_shots))
            # Bulk Actions & Quality Filter
            ba1, ba2, ba3 = st.columns([1, 1, 2])
            with ba1:
                q_filter_key = "director_global_q_filter"
                temp_q_filter = st.session_state.get(q_filter_key, "All")
                temp_filtered = [c for c in candidates]
                if temp_q_filter == "720p and above":
                    temp_filtered = [c for c in temp_filtered if (c.get("height") or 0) >= 720 or c.get("source") == "youtube"]
                elif temp_q_filter == "1080p and above":
                    temp_filtered = [c for c in temp_filtered if (c.get("height") or 0) >= 1080 or c.get("source") == "youtube"]
                
                if st.button("☑ Select All", key=f"sel_all_{slot_id}", use_container_width=True):
                    shot["selected_results"] = list(temp_filtered)
                    shot["skipped"] = False
                    save_cache()
                    st.rerun()
            with ba2:
                if st.button("☒ Clear All", key=f"sel_none_{slot_id}", use_container_width=True):
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
                        source_display = "🤖 AI VISUAL SEARCH" if source_val == "SMART_LIBRARY" else f"🌐 {source_val}"
                        score = cand.get("score")
                        if score: source_display += f" · {score*100:.0f}% match"
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
                                    border_style = "border: 3px solid #00cc44; border-radius:6px;" if is_picked else ""
                                    st.markdown(f'<div style="position:relative;{border_style}"><img src="{thumb}" style="width:100%;border-radius:4px;aspect-ratio:16/9;object-fit:cover;display:block;">{selected_badge}{used_badge}</div>', unsafe_allow_html=True)
                                st.markdown(f"**{title}**")
                                st.caption(f"{source_display} · {dur_str} · {res_str}")
                                extra_info = cand.get("tags") if (source_val not in ("YOUTUBE", "SMART_LIBRARY") and cand.get("tags")) else cand.get("description")
                                if extra_info:
                                    with st.expander("📄 Details", expanded=False): st.write(extra_info)
                                if cand.get("smart_reason"): st.caption(f"🤖 _{cand.get('smart_reason')}_")
                                if source_val == "SMART_LIBRARY":
                                    if st.button("🎬 Preview", key=f"prev_{slot_id}_{i_g+j_g}"): st.video(cand.get("segment_path"))
                                else:
                                    st.markdown(f'<a href="{url}" target="_blank" style="text-decoration:none;font-size:12px;">📺 Watch</a>', unsafe_allow_html=True)
                                btn_label = "✅ SELECTED" if is_picked else "⬜ PICK CLIP"
                                if st.button(btn_label, key=f"galpick_{slot_id}_{hash(cand.get('url'))}", use_container_width=True, type="primary" if is_picked else "secondary"):
                                    toggle_pick(cand.get("url"), cand, shot)
                                    st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)
            # 2. Footer Navigation Bar (Immediately below gallery)
            fa1, fa2, fa3, fa4, fa5 = st.columns([1.2, 1.2, 2.5, 2.5, 1.2])
            with fa1:
                if st.button("◀ Prev", key=f"d_prev_bot_{slot_id}", disabled=idx == 0, use_container_width=True):
                    save_cache(); st.session_state.d_review_idx -= 1; st.rerun()
            with fa2:
                skip_label = "↺ Unskip" if skipped else "⏭ Skip"
                if st.button(skip_label, key=f"skip_{slot_id}", use_container_width=True):
                    shot["skipped"] = not skipped; shot["selected_results"] = []; save_cache(); st.rerun()
            with fa3:
                options = [f"Shot {i+1} / {len(review_shots)}" for i in range(len(review_shots))]
                st.session_state[f"d_jump_bot_{slot_id}"] = options[idx]
                selected = st.selectbox("Jump", options=options, label_visibility="collapsed", key=f"d_jump_bot_{slot_id}")
                new_idx_bot = int(selected.split(" ")[1]) - 1
                if new_idx_bot != idx: save_cache(); st.session_state.d_review_idx = new_idx_bot; st.rerun()
            with fa4:
                if idx < len(review_shots) - 1:
                    if st.button("Save & Next ▶", key=f"d_save_next_{slot_id}", type="primary", use_container_width=True):
                        save_cache(); st.session_state.d_review_idx += 1; st.rerun()
                else:
                    if st.button("✅ Finish Review", key=f"d_finish_{slot_id}", type="primary", use_container_width=True):
                        save_cache(); st.success("Review complete! Scroll down to Step 6 to start downloads.")
            with fa5:
                if st.button("Next ▶", key=f"d_next_bot_{slot_id}", disabled=idx == len(review_shots) - 1, use_container_width=True):
                    save_cache(); st.session_state.d_review_idx += 1; st.rerun()
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
            if skipped: st.warning("⏭ This shot is marked as skipped. Selecting any clip will unskip it.")
            # 4. Tweak & Refetch
            st.markdown("---")
            with st.expander("🔄 Tweak & Refetch THIS Shot", expanded=False):
                st.caption("Edit the keywords and fetch new candidates for this shot only.")
                curr_queries = " | ".join(shot.get("search_queries", []))
                new_queries_str = st.text_input("Stock Queries", value=curr_queries, key=f"refetch_q_{slot_id}")
                curr_yt = " | ".join(shot.get("youtube_keywords", []))
                new_yt_str = st.text_input("YouTube Keywords", value=curr_yt, key=f"refetch_yt_{slot_id}")
                if st.button("Refetch Candidates", key=f"refetch_btn_{slot_id}", type="primary", use_container_width=True):
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
        col_dv1, col_dv2, col_dv3 = st.columns(3)
        with col_dv1:
            d_quality  = st.selectbox("Quality", ["1080p", "720p", "480p", "Best", "Worst"], key="d_vq")
        with col_dv2:
            d_max_size = st.number_input("Max Size (MB)", value=200, min_value=1, key="d_maxsize")
        with col_dv3:
            d_workers  = st.slider("Concurrent", 1, 10, 3, key="d_workers")
        total_selected_files = sum(len(s.get("selected_results", [])) for s in selected_shots)
        col_dl1, col_dl2 = st.columns([2, 1])
        with col_dl1:
            start_dl = st.button(f"📥 Download {total_selected_files} selected videos",
                                key="d_dl_start", type="primary", use_container_width=True)
        with col_dl2:
            if st.button("📄 Export Manifest", key="d_manifest", use_container_width=True, help="Generate a text file listing all shots, grouped by chunk."):
                def _safe_for_fs(text: str, max_len: int = 30) -> str:
                    if not text: return ""
                    cleaned = "".join(c if (c.isalnum() or c in " -_") else " " for c in text)
                    cleaned = "-".join(cleaned.split()).lower()
                    return cleaned[:max_len].strip("-") or ""
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
                if st.session_state.dm.max_workers != d_workers:
                    st.session_state.dm = DownloadManager(max_workers=d_workers)
                st.session_state.dm.clear_and_reset()
                # ── Group selected clips by URL across all shots ─────────
                def _safe_for_fs(text: str, max_len: int = 30) -> str:
                    if not text:
                        return ""
                    cleaned = "".join(c if (c.isalnum() or c in " -_") else " "
                                      for c in text)
                    cleaned = "-".join(cleaned.split()).lower()
                    return cleaned[:max_len].strip("-") or ""
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
                    if st.button("âÅ“– Cancel All", key="d_cancel", use_container_width=True):
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
                            if b1.button("Pause", key=f"pd_{t['id']}", use_container_width=True):
                                st.session_state.dm.pause_download(t["id"]); st.rerun()
                        elif t["status"] == "paused":
                            if b1.button("Resume", key=f"rd_{t['id']}", use_container_width=True):
                                st.session_state.dm.resume_download(t["id"]); st.rerun()
                    if b2.button("Cancel", key=f"cd_{t['id']}", use_container_width=True):
                        st.session_state.dm.cancel_download(t["id"]); st.rerun()
            # ── Failed tasks (URL + per-task retry with current settings) ──
            failed = st.session_state.dm.get_failed_tasks()
            if failed:
                st.subheader(f"âÅ’ Failed ({len(failed)})")
                retryable = [f for f in failed if st.session_state.dm.can_retry(f["id"])]
                if retryable:
                    if st.button(
                        f"ââ€ » Retry all ({len(retryable)}) with current settings",
                        key="d_retry_all", use_container_width=True,
                    ):
                        st.session_state.dm.retry_all_failed(overrides={
                            "quality": d_quality, "max_size_mb": d_max_size,
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
                                use_container_width=True,
                                help="Retry with the current Quality / Max Size settings.",
                            ):
                                st.session_state.dm.retry_failed(ft["id"], overrides={
                                    "quality": d_quality, "max_size_mb": d_max_size,
                                })
                                st.rerun()
                        # URL row — let the user copy or open the source page
                        url_col, link_col = st.columns([5, 1])
                        with url_col:
                            st.code(ft.get("url", ""), language=None)
                        with link_col:
                            if ft.get("url"):
                                st.link_button("Open ââ€ —", ft["url"], use_container_width=True)
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
                        st.session_state.text_overlays = extract_highlights(
                            st.session_state.script_text, 
                            os.getenv("GROQ_API_KEY")
                        )
                        st.success(f"Extracted {len(st.session_state.text_overlays)} highlights!")
            
            if st.session_state.text_overlays:
                # Data Editor for fine-tuning
                df_ov = pd.DataFrame(st.session_state.text_overlays)
                edited_df = st.data_editor(df_ov, num_rows="dynamic", use_container_width=True, key="ov_editor")
                st.session_state.text_overlays = edited_df.to_dict('records')
                
                st.divider()
                st.subheader("Visual Settings")
                c_v1, c_v2, c_v3 = st.columns(3)
                with c_v1:
                    st.session_state.overlay_settings["color"] = st.color_picker("Text Color", st.session_state.overlay_settings["color"])
                    st.session_state.overlay_settings["size"] = st.slider("Font Size", 50, 250, st.session_state.overlay_settings["size"])
                with c_v2:
                    st.session_state.overlay_settings["shadow"] = st.color_picker("Shadow Color", st.session_state.overlay_settings["shadow"])
                    st.session_state.overlay_settings["y"] = st.slider("Vertical Position (0-1080)", 0, 1080, st.session_state.overlay_settings["y"])
                with c_v3:
                    st.session_state.overlay_settings["animation"] = st.selectbox("Animation Style", ["None", "Fade In/Out"], index=1)
                    
                if st.button("Generate & Preview Overlays", type="primary"):
                    p_name = st.session_state.get("project_name", "default")
                    proj_folder = _safe_for_fs(p_name, 50)
                    ov_dir = os.path.abspath(os.path.join("downloads", "director", proj_folder, "overlays"))
                    os.makedirs(ov_dir, exist_ok=True)
                    
                    def ts_to_sec(ts):
                        if isinstance(ts, (int, float)): return float(ts)
                        # HH:MM:SS,mmm or HH:MM:SS
                        parts = re.split('[:.,]', str(ts))
                        if len(parts) >= 3:
                            h, m, s = map(int, parts[:3])
                            ms = int(parts[3]) if len(parts) > 3 else 0
                            return h * 3600 + m * 60 + s + ms / 1000.0
                        return float(ts or 0)

                    with st.spinner("Generating transparent PNGs..."):
                        for idx, ov in enumerate(st.session_state.text_overlays):
                            fname = os.path.join(ov_dir, f"overlay_{idx+1}.png")
                            create_text_overlay(
                                str(ov.get("highlight_text", "")),
                                fname,
                                font_size=st.session_state.overlay_settings["size"],
                                color=st.session_state.overlay_settings["color"],
                                shadow_color=st.session_state.overlay_settings["shadow"],
                                y_position=st.session_state.overlay_settings["y"]
                            )
                            ov["filepath"] = fname
                            ov["animation"] = st.session_state.overlay_settings["animation"]
                            ov["start_sec"] = ts_to_sec(ov.get("start_time", 0))
                            ov["end_sec"] = ts_to_sec(ov.get("end_time", ov.get("start_time", 0) + 3))
                        st.success(f"Generated {len(st.session_state.text_overlays)} PNG overlays in {ov_dir}")
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
            
            # Filter overlays for this specific chunk
            chunk_overlays = filter_overlays_for_shots(shot_chunk, st.session_state.text_overlays)
            
            part_display_name = p_name if num_parts == 1 else f"{p_name}_Part_{i+1}"
            xml_content = generate_fcpxml(shot_chunk, project_name=part_display_name, overlays=chunk_overlays)
            
            fname = "b_roll_sequence.xml" if num_parts == 1 else f"b_roll_sequence_part_{i+1}.xml"
            xml_parts.append({"filename": fname, "content": xml_content})
            
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
            
        failed_tasks = st.session_state.dm.get_failed_tasks()
        failed_txt = ""
        if failed_tasks:
            failed_txt = generate_failed_downloads_txt(failed_tasks)
            
        c1, c2, c3     = st.columns(3)
        with c1: 
            st.download_button("shot_list.json", data=shot_list_json, file_name="shot_list.json", mime="application/json")
            if trans_srt:
                st.download_button("transcription.srt", data=trans_srt, file_name="transcription.srt", mime="text/plain")
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
            else:
                st.write("**XML Sequences (Split):**")
                for p in xml_parts:
                    st.download_button(
                        f"Download {p['filename']}", 
                        data=p["content"], 
                        file_name=p["filename"], 
                        mime="text/xml"
                    )
