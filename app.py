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
    generate_transcription_srt, generate_failed_downloads_txt
)
from core.download_manager import DownloadManager, MAX_RETRIES, link_or_copy
from core import download_cache
from core.app_utils import check_network, format_speed, format_eta
from core.session_cache import load_session_cache, save_session_cache
import time

# --- Config & Initialization ---
st.set_page_config(page_title="B-Roll Finder", layout="wide")

ENV_FILE = ".env"
CACHE_FILE = ".cache/session_state.json"

if not os.path.exists(".cache"):
    os.makedirs(".cache")

load_dotenv(ENV_FILE)

# When the user runs a TUN-mode VPN (e.g. V2RayN, Hiddify), the OS
# captures all traffic at the network layer, so app-level HTTP_PROXY /
# HTTPS_PROXY env vars become stale — they point at an HTTP proxy
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
    other_shots = [s for s in st.session_state.director_shots if s.get("chunk_id") != chunk_idx]
    combined = other_shots + new_shots
    combined.sort(key=lambda x: (x.get("chunk_id", 0), x.get("timestamp", 0)))
    
    # 5. Re-index slot_ids
    for idx, s in enumerate(combined):
        s["slot_id"] = idx + 1
        
    st.session_state.director_shots = combined
    save_cache()
if not st.session_state.slots and os.path.exists(CACHE_FILE):
    load_cache()

# --- UI Components ---
st.sidebar.title("App Mode")
app_mode = st.sidebar.radio(
    "Select Mode",
    ["Director", "Classic Finder"],
    help=(
        "**Director** is the recommended workflow — upload a voiceover, "
        "auto-transcribe, generate shot lists, fetch + rank candidates, "
        "review with checkboxes, download. **Classic Finder** is the older "
        "keyword-based path; kept for compatibility."
    ),
)

def render_classic_mode():
    st.title("🎬 B-Roll Finder")

    # ── Setup: API keys (status pill in header, auto-collapses when ready) ──
    # Auto-import keys from any *Api.txt files in the project root.
    for txt_file, env_key in [
        ("groq api.txt",      "GROQ_API_KEY"),
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
        "Groq":       bool(os.getenv("GROQ_API_KEY")),
        "Pexels":     bool(os.getenv("PEXELS_API_KEY")),
        "Pixabay":    bool(os.getenv("PIXABAY_API_KEY")),
        "YouTube":    bool(os.getenv("YOUTUBE_API_KEY")),
        "OpenRouter": bool(os.getenv("OPENROUTER_API_KEY")),
    }
    _pill = " · ".join(f"{'✅' if v else '○'} {k}" for k, v in _key_status.items())
    with st.expander(f"⚙️ Setup — API keys  ·  {_pill}",
                     expanded=not _key_status["Groq"]):
        st.caption(
            "Groq is required (script analysis). Pexels/Pixabay/YouTube are search "
            "sources — enable at least one. OpenRouter is an automatic fallback "
            "when Groq hits its rate limit."
        )
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            groq_input    = st.text_input("Groq API Key (required)",
                                          value=os.getenv("GROQ_API_KEY", ""),
                                          type="password")
            pexels_input  = st.text_input("Pexels API Key",
                                          value=os.getenv("PEXELS_API_KEY", ""),
                                          type="password")
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
                            errors=fetch_errors
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
                                pexels_res = search_pexels(primary_kw, os.getenv("PEXELS_API_KEY"), pexels_count, errors=fetch_errors)
                                if len(fetch_errors) > prev_err_count:
                                    pexels_failures += 1
                                    if pexels_failures >= _CIRCUIT_LIMIT:
                                        fetch_errors.append("Pexels: 3 consecutive failures — skipping remaining Pexels searches.")
                                else:
                                    pexels_failures = 0
                                slot['video_results'].extend(pexels_res)

                            if pixabay_count > 0 and os.getenv("PIXABAY_API_KEY") and pixabay_failures < _CIRCUIT_LIMIT:
                                prev_err_count = len(fetch_errors)
                                pixabay_res = search_pixabay(primary_kw, os.getenv("PIXABAY_API_KEY"), pixabay_count, errors=fetch_errors)
                                if len(fetch_errors) > prev_err_count:
                                    pixabay_failures += 1
                                    if pixabay_failures >= _CIRCUIT_LIMIT:
                                        fetch_errors.append("Pixabay: 3 consecutive failures — skipping remaining Pixabay searches.")
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
                        with st.expander(f"⚠️ {len(unique_errors)} search error(s) — click to see details"):
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
        with st.expander("🔍 View Search Results Gallery", expanded=True):
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
                        "Duration": f"{res.get('duration')}s" if res.get('duration') else "—",
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
                st.subheader("🖼️ Quick Preview Gallery")
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
            with st.expander(f"⚠️ {len(failed_tasks)} Failed Downloads"):
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
                label = "Normalizing…" if is_processing else f"{t['status'].title()} ({t['progress']*100:.1f}%){(' — ' + meta) if meta else ''}"
                st.write(f"**{os.path.basename(t['output_path'])}** — {label}")
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
                                    fetch_errors_g.append("Pexels: 3 consecutive failures — skipping remaining Pexels searches.")
                            else:
                                pexels_failures = 0
                            results.extend(pex_res)

                        if g_pixabay > 0 and os.getenv("PIXABAY_API_KEY") and pixabay_failures < _CIRCUIT_LIMIT:
                            prev = len(fetch_errors_g)
                            pix_res = search_pixabay(primary_kw, os.getenv("PIXABAY_API_KEY"), g_pixabay, errors=fetch_errors_g)
                            if len(fetch_errors_g) > prev:
                                pixabay_failures += 1
                                if pixabay_failures >= _CIRCUIT_LIMIT:
                                    fetch_errors_g.append("Pixabay: 3 consecutive failures — skipping remaining Pixabay searches.")
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
                        with st.expander(f"⚠️ {len(unique_g_errors)} search error(s)"):
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
            st.subheader("🌐 Global Footage Review")
            
            # Global Gallery Section
            st.subheader("🖼️ Global Gallery")
            
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
elif app_mode == "Director":
    from core.director_search import fetch_director_footage
    from core.director_rank import rank_shot_candidates
    from core.director_youtube import generate_youtube_keywords_for_shots, seed_youtube_keywords
    from core.output import generate_fcpxml, generate_shot_list_txt
    st.title("🎬 B-Roll Director")

    # ── Setup: API keys (collapsible, auto-collapses once Groq key is present) ──
    _key_status = {
        "Groq":       bool(os.getenv("GROQ_API_KEY")),
        "Pexels":     bool(os.getenv("PEXELS_API_KEY")),
        "Pixabay":    bool(os.getenv("PIXABAY_API_KEY")),
        "YouTube":    bool(os.getenv("YOUTUBE_API_KEY")),
        "OpenRouter": bool(os.getenv("OPENROUTER_API_KEY")),
    }
    _pill = " · ".join(f"{'✅' if v else '○'} {k}" for k, v in _key_status.items())
    with st.expander(f"⚙️ Setup — API keys  ·  {_pill}",
                     expanded=not _key_status["Groq"]):
        st.caption(
            "Groq is required (script analysis & ranking). Pexels/Pixabay/YouTube are "
            "search sources — enable at least one. OpenRouter is an automatic fallback "
            "when Groq hits its rate limit."
        )
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            groq_input    = st.text_input("Groq API Key (required)",
                                          value=os.getenv("GROQ_API_KEY", ""),
                                          type="password", key="d_groq")
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
            help=(
                "We'll transcribe this with Groq Whisper, then split it into "
                "~2-minute chunks at sentence boundaries. You'll then pick "
                "which chunk to work on — keeps each batch within Groq's "
                "rate limits and lets you iterate one section at a time."
            ),
        )

        if audio_file:
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
                st.metric("Chunks", str(len(st.session_state.get("transcription_chunks", []))) or "—")
            with top_c:
                st.audio(audio_path)

            cta_a, cta_b = st.columns([3, 1])
            with cta_a:
                if st.button("🎙 Transcribe & chunk", key="d_transcribe", type="primary",
                             use_container_width=True):
                    if not os.getenv("GROQ_API_KEY"):
                        st.error("Set Groq API Key in Setup above.")
                    else:
                        from core.transcription import transcribe_audio
                        from core.timing import chunk_segments_by_duration
                        with st.spinner("Transcribing with Whisper…"):
                            try:
                                segments = transcribe_audio(audio_path, os.getenv("GROQ_API_KEY"))
                                st.session_state.transcription_segments = segments
                                # Reconstruct script_text from segments so the rest
                                # of the pipeline (WPS calc, shot generation) works.
                                st.session_state.script_text = " ".join(
                                    s["text"].strip() for s in segments
                                ).strip()
                                chunks = chunk_segments_by_duration(
                                    segments, target_duration=120.0, max_duration=180.0
                                )
                                st.session_state.transcription_chunks = chunks
                                st.session_state.active_chunk_indices = [0]
                                save_cache()
                                st.success(
                                    f"Transcribed and split into {len(chunks)} chunk(s). "
                                    f"Pick one below and continue to Step 2."
                                )
                                st.rerun()
                            except Exception as e:
                                st.error(f"Transcription failed: {e}")
            with cta_b:
                if st.session_state.get("transcription_chunks"):
                    if st.button("Reset", key="d_reset_transcription",
                                 use_container_width=True,
                                 help="Discard the current transcription and chunks."):
                        st.session_state.transcription_segments = []
                        st.session_state.transcription_chunks   = []
                        st.session_state.active_chunk_indices   = [0]
                        st.session_state.script_text            = ""
                        save_cache()
                        st.rerun()
        else:
            st.caption("Upload your voiceover to begin.")

    # ── Chunk picker ─────────────────────────────────────────────────────
    chunks = st.session_state.get("transcription_chunks", [])
    if chunks:
        st.subheader("Pick a chunk to work on")
        st.caption(
            "Each chunk is ~2 minutes, ending on a sentence boundary. "
            "The shot list, candidate fetch, ranking, and download in the "
            "later steps all act on the **active chunk only** — let you "
            "iterate one section at a time within Groq's rate limits."
        )

        def _format_chunk(i: int) -> str:
            c = chunks[i]
            ms_s = f"{int(c['start']//60)}:{int(c['start']%60):02d}"
            ms_e = f"{int(c['end']//60)}:{int(c['end']%60):02d}"
            wc   = len(c['text'].split())
            preview = c['text'][:60].replace("\n", " ")
            if len(c['text']) > 60:
                preview += "…"
            return f"Chunk {i+1} · {ms_s}–{ms_e} · {wc} words · \"{preview}\""

        # multiselect stores the indices directly via the format_func pattern
        active_indices = st.multiselect(
            "Active chunks",
            options=list(range(len(chunks))),
            format_func=_format_chunk,
            default=st.session_state.get("active_chunk_indices", [0] if chunks else []),
            key="active_chunk_indices",
            on_change=save_cache,
        )

        if not active_indices:
            st.warning("Please select at least one chunk to proceed.")
            st.stop()

        # Compact stats for the active chunks
        c_dur = sum(chunks[i]['end'] - chunks[i]['start'] for i in active_indices)
        c_words = sum(len(chunks[i]['text'].split()) for i in active_indices)
        c_wps = c_words / c_dur if c_dur > 0 else 0
        s1, s2, s3 = st.columns(3)
        with s1: st.metric("Length", f"{c_dur/60:.1f} min")
        with s2: st.metric("Words", f"{c_words:,}")
        with s3: st.metric("Speaking rate", f"{c_wps:.2f} wps")

        idx_str = ", ".join(str(i+1) for i in sorted(active_indices))
        with st.expander(f"📝 Read text for Chunk(s) {idx_str}", expanded=False):
            if st.session_state.transcription_segments:
                st.write(f"**AI Transcription for Chunk(s) {idx_str}:**")
                for i in sorted(active_indices):
                    chunk_start = chunks[i]['start']
                    chunk_end = chunks[i]['end']
                    for seg in st.session_state.transcription_segments:
                        if seg['start'] >= chunk_start and seg['end'] <= chunk_end:
                            start_m = int(seg['start'] // 60)
                            start_s = int(seg['start'] % 60)
                            st.write(f"**[{start_m:02d}:{start_s:02d}]** {seg['text']}")
            else:
                combined_text = "\n\n".join(chunks[i]["text"] for i in sorted(active_indices))
                st.text_area(
                    "Chunk text", value=combined_text, height=200,
                    disabled=True, key=f"chunk_text_{hash(tuple(active_indices))}",
                    label_visibility="collapsed",
                )

    st.header("Step 2: Generate Shot List")

    # Video topic — single source of truth shared with Step 4 (ranking).
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
        elif not st.session_state.get("transcription_chunks"):
            st.error("Upload audio and click *Transcribe & chunk* in Step 1 first.")
        else:
            chunks = st.session_state.transcription_chunks
            indices = st.session_state.get("active_chunk_indices", [])
            if not indices:
                st.error("Select at least one chunk in Step 1.")
            else:
                try:
                    chunks = st.session_state.transcription_chunks
                    indices = sorted(st.session_state.get("active_chunk_indices", []))
                    
                    pbar = st.progress(0)
                    status = st.empty()
                    status.text("Generating shot list...")
                    
                    all_new_shots = []
                    total_to_gen = len(indices)
                    
                    from core.director import generate_shot_list_from_transcription
                    
                    for i, chunk_idx in enumerate(indices):
                        status.info(f"Generating shots for Chunk {chunk_idx+1}...")
                        chunk = chunks[chunk_idx]
                        chunk_segments = chunk.get("segments", [])
                        
                        if not chunk_segments:
                            continue
                            
                        shot_list = generate_shot_list_from_transcription(
                            chunk_segments,
                            os.getenv("GROQ_API_KEY"),
                            custom_instructions=custom_instructions,
                            video_topic=d_video_topic,
                            chunk_id=chunk_idx
                        )
                        all_new_shots.extend(shot_list)
                        pbar.progress((i + 1) / total_to_gen)
                    
                    # Re-index all shots to be sequential
                    for idx, shot in enumerate(all_new_shots):
                        shot["slot_id"] = idx + 1
                    
                    st.session_state.director_shots = all_new_shots
                    pbar.progress(1.0)
                    status.empty()
                    st.success(
                        f"Shot list generated — {len(st.session_state.director_shots)} shot(s) across {len(indices)} chunk(s)."
                    )
                    save_cache()
                except Exception as e:
                    st.error(f"Error generating shot list: {e}")

    # ── Generated shot list (editable) ──────────────────────────────────────
    if st.session_state.director_shots:
        st.subheader("Step 2.2: Selective Regeneration")
        st.caption("If you're unhappy with a specific chunk, redo it here. Selections from the old version are preserved.")
        
        active_indices = sorted(st.session_state.get("active_chunk_indices", []))
        if active_indices:
            # Group into rows of 4 for space
            for r_idx in range(0, len(active_indices), 4):
                chunk_cols = st.columns(4)
                for c_idx, chunk_idx in enumerate(active_indices[r_idx:r_idx+4]):
                    with chunk_cols[c_idx]:
                        if st.button(f"♻️ Redo Chunk {chunk_idx+1}", key=f"redo_chunk_{chunk_idx}", use_container_width=True):
                            with st.spinner(f"Redoing Chunk {chunk_idx+1}..."):
                                perform_chunk_regeneration(chunk_idx)
                                st.success(f"Chunk {chunk_idx+1} regenerated. Check Step 5 to review.")
                                st.rerun()

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
            rows.append({
                "#":          shot.get("slot_id", 0),
                "Time":       f"{shot.get('timestamp_start_str')} – {shot.get('timestamp_end_str')}",
                "Intent":     shot.get("shot_intent", ""),
                "Priority":   shot.get("priority", "medium"),
                "Queries":    " | ".join(shot.get("search_queries", [])),
                "Cand":       len(shot.get("video_results", [])),
                "Picked":     f"✅ {len(sel)}" if sel else ("⏭ skip" if shot.get("skipped") else "—"),
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
                "Queries":  st.column_config.TextColumn("Queries (use ' | ' to separate)", width="large",
                              help="Used by Step 3 to fetch candidates. Edit and re-fetch to update results."),
                "Cand":     st.column_config.NumberColumn("Cand", width="small",
                              help="Number of candidates fetched for this shot."),
                "Picked":   st.column_config.TextColumn("Picked", width="small",
                              help="How many candidates you've ticked in Step 5."),
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
        if changed:
            save_cache()

    # Step 2.5: optional YouTube-specific keywords for Classic-style search.
    if st.session_state.director_shots:
        st.header("Step 2.5: YouTube Keywords")
        st.caption(
            "Optional. These keywords are used only by Director's Classic-style YouTube search. "
            "Pexels and Pixabay still use the stock-footage queries in the shot list above."
        )

        yt_kw_count = st.number_input(
            "YouTube keywords per shot",
            value=2,
            min_value=1,
            max_value=4,
            key="d_yt_kw_count",
        )
        yk1, yk2, yk3 = st.columns([2, 2, 6])
        with yk1:
            if st.button("Seed from Queries", key="d_seed_yt_keywords", use_container_width=True):
                st.session_state.director_shots = seed_youtube_keywords(
                    st.session_state.director_shots,
                    max_keywords=int(yt_kw_count),
                )
                save_cache()
                st.rerun()
        with yk2:
            if st.button("Generate with AI", key="d_gen_yt_keywords", use_container_width=True):
                if not os.getenv("GROQ_API_KEY"):
                    st.error("Groq API key required to generate YouTube keywords.")
                else:
                    pbar_yk = st.progress(0)
                    try:
                        st.session_state.director_shots = generate_youtube_keywords_for_shots(
                            st.session_state.director_shots,
                            api_key=os.getenv("GROQ_API_KEY"),
                            video_topic=st.session_state.get("d_video_topic", ""),
                            custom_instructions=custom_instructions,
                            max_keywords=int(yt_kw_count),
                            progress_callback=lambda p: pbar_yk.progress(p),
                        )
                        pbar_yk.progress(1.0)
                        save_cache()
                        st.success("YouTube keywords generated. Edit any row before fetching.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not generate YouTube keywords: {e}")
        with yk3:
            st.caption("Use plain YouTube search language, separated with ` | `.")

        yt_rows = []
        for shot in st.session_state.director_shots:
            yt_rows.append({
                "#": shot.get("slot_id", 0),
                "Time": f"{shot.get('timestamp_start_str')} - {shot.get('timestamp_end_str')}",
                "Intent": shot.get("shot_intent", ""),
                "Priority": shot.get("priority", "medium"),
                "YouTube Keywords": " | ".join(shot.get("youtube_keywords", [])),
            })
        yt_df = pd.DataFrame(yt_rows)
        edited_yt = st.data_editor(
            yt_df,
            column_config={
                "#": st.column_config.NumberColumn("#", width="small"),
                "Time": st.column_config.TextColumn("Time", width="small"),
                "Intent": st.column_config.TextColumn("Intent", width="medium"),
                "Priority": st.column_config.TextColumn("Priority", width="small"),
                "YouTube Keywords": st.column_config.TextColumn(
                    "YouTube Keywords (use ' | ' to separate)",
                    width="large",
                ),
            },
            disabled=["#", "Time", "Intent", "Priority"],
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key="d_youtube_keywords_editor",
        )
        yt_changed = False
        for i, shot in enumerate(st.session_state.director_shots):
            if i >= len(edited_yt):
                continue
            if shot.get("priority") == "none":
                new_keywords = []
            else:
                raw = str(edited_yt.iloc[i]["YouTube Keywords"] or "")
                new_keywords = [kw.strip() for kw in raw.split("|") if kw.strip()]
            if shot.get("youtube_keywords", []) != new_keywords:
                shot["youtube_keywords"] = new_keywords
                yt_changed = True
        if yt_changed:
            save_cache()

    # ── Step 3 — Fetch Candidates ────────────────────────────────────────────
    if st.session_state.director_shots:
        st.header("Step 3: Fetch Candidates")

        col_s2a, col_s2b, col_s2c, col_s2d = st.columns(4)
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
            fetch_clicked = st.button("Fetch Candidates", disabled=st.session_state.is_fetching, key="d_fetch", use_container_width=True)
        with col_btn2:
            retry_clicked = st.button("Retry Failed/Empty", disabled=st.session_state.is_fetching, key="d_retry", use_container_width=True, help="Only searches for shots that currently have 0 candidates. Preserves existing successful searches.")

        st.subheader("Step 3.2: Selective Fetching")
        st.caption("Fetch candidates for a specific chunk only. Useful after tweaking queries above.")
        active_indices = sorted(st.session_state.get("active_chunk_indices", []))
        chunk_to_fetch = None
        if active_indices:
            for r_idx in range(0, len(active_indices), 4):
                f_cols = st.columns(4)
                for c_idx, chunk_idx in enumerate(active_indices[r_idx:r_idx+4]):
                    with f_cols[c_idx]:
                        if st.button(f"🔎 Fetch Chunk {chunk_idx+1}", key=f"fetch_chunk_{chunk_idx}", use_container_width=True, disabled=st.session_state.is_fetching):
                            chunk_to_fetch = chunk_idx

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
                        progress_callback=lambda p: pbar2.progress(p),
                        errors=d_fetch_errors,
                        retry_only=retry_clicked,
                    )
                    
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

    # ── Step 4 — LLM Ranking ─────────────────────────────────────────────────
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

    # ── Step 5 — Editor Review (paginated) ──────────────────────────────────
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
        # Clamp BOTH ends. Stale session state from a prior batch (or
        # an interrupted rerun via browser back) can leave d_review_idx
        # negative — that fed st.progress a negative ratio and crashed.
        st.session_state.d_review_idx = max(
            0, min(st.session_state.d_review_idx, len(review_shots) - 1)
        )
        idx = st.session_state.d_review_idx
        shot = review_shots[idx]

        # Compact navigation strip: Prev / position / Next on one row.
        n_selected = sum(1 for s in review_shots if s.get("selected_results"))
        n_skipped  = sum(1 for s in review_shots if s.get("skipped"))
        n_pending  = len(review_shots) - n_selected - n_skipped

        nav1, nav2, nav3 = st.columns([1, 4, 1])
        with nav1:
            if st.button("◀ Prev", key="d_prev", disabled=idx == 0, use_container_width=True):
                save_cache()
                st.session_state.d_review_idx -= 1
                st.rerun()
        with nav2:
            st.markdown(
                f"<div style='text-align:center; padding-top:0.4em;'>"
                f"<strong>Shot {idx + 1} of {len(review_shots)}</strong>"
                f" &nbsp;·&nbsp; ✅ {n_selected} selected"
                f" &nbsp;·&nbsp; ⏭ {n_skipped} skipped"
                f" &nbsp;·&nbsp; ⏳ {n_pending} pending"
                f"</div>",
                unsafe_allow_html=True,
            )
        with nav3:
            if st.button("Next ▶", key="d_next", disabled=idx == len(review_shots) - 1, use_container_width=True):
                save_cache()
                st.session_state.d_review_idx += 1
                st.rerun()
        st.progress((idx + 1) / len(review_shots))

        # Shot details — gather data first.
        slot_id  = shot.get("slot_id", "?")
        ts       = f"{shot.get('timestamp_start_str')} – {shot.get('timestamp_end_str')}"
        reason   = shot.get("rank_reason", "")
        rank_err = shot.get("rank_error", "")
        sel_urls = {r.get("url") for r in shot.get("selected_results", [])}
        skipped  = shot.get("skipped", False)

        with st.container(border=True):
            st.markdown(
                f"**Shot {slot_id}** &nbsp;·&nbsp; ⏱ {ts} &nbsp;·&nbsp; "
                f"🎬 {shot.get('shot_intent', '—')}"
            )
            st.markdown(f"💬 _{shot.get('text', '')}_")
            if reason:
                st.markdown(f"🤖 _{reason}_")

        if rank_err:
            st.warning(f"⚠️ AI ranking failed for this shot ({rank_err}). Showing unranked candidates — review carefully.")
        if skipped:
            st.warning("⏭ This shot is marked as skipped. Selecting any clip will unskip it.")

        candidates = [c for c in shot["video_results"] if not c.get("irrelevant")]

        if not candidates:
            st.warning("No relevant candidates found for this shot.")
        else:
            # ── Gallery Selection UI ───────────────────────────────────────
            st.markdown("---")
            st.subheader("🖼️ Selection Gallery")
            
            # Context at the top: Script & Keywords
            with st.container(border=True):
                st.markdown(f"💬 **Script:** _{shot.get('text', '')}_")
                queries = shot.get("search_queries", [])
                if queries:
                    q_md = " &nbsp;·&nbsp; ".join(f"`{q}`" for q in queries)
                    st.markdown(f"🔎 **Keywords:** {q_md}")

            # CSS for clickable images
            st.markdown("""
                <style>
                /* Container for the whole gallery card */
                .gallery-card {
                    position: relative !important;
                }
                
                /* Target the status button at the bottom and expand its clickable area */
                .gallery-card [data-testid="stButton"] button {
                    position: relative;
                }
                
                .gallery-card [data-testid="stButton"] button::after {
                    content: "";
                    position: absolute;
                    top: -450px; /* Pull up to cover image and metadata */
                    left: -20px;
                    right: -20px;
                    bottom: -20px;
                    z-index: 1;
                    cursor: pointer;
                }

                .watch-btn-overlay {
                    z-index: 10 !important;
                    position: relative;
                }
                
                /* Selection indicators */
                .selection-border-active {
                    border: 3px solid #00ff00 !important;
                    border-radius: 6px;
                }
                </style>
            """, unsafe_allow_html=True)

            # Bulk Actions & Quality Filter
            ba1, ba2, ba3, ba4 = st.columns([1, 1, 1.5, 1.5])
            with ba1:
                # Use global quality filter key
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
                if st.button("☐ Clear All", key=f"sel_none_{slot_id}", use_container_width=True):
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

            # Apply quality filter (now q_filter is defined)
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
                        source = cand.get("source", "").upper()
                        w, h = cand.get("width"), cand.get("height")
                        dur = cand.get("duration")
                        res_str = f"{w}x{h}" if w and h else "Resolution Unknown"
                        dur_str = f"{dur}s" if dur else ""
                        
                        with cols[j_g]:
                            cand_url = cand.get("url")
                            is_picked = cand_url in sel_urls
                            
                            # Check if selected in OTHER shots
                            other_slots = [sid for sid in global_pick_map.get(cand_url, []) if str(sid) != str(slot_id)]
                            others_tag = ""
                            if other_slots:
                                others_tag = (
                                    f'<div style="position: absolute; bottom: 5px; left: 5px; z-index: 10; '
                                    f'background: #ffcc00; color: black; padding: 2px 6px; border-radius: 4px; '
                                    f'font-size: 10px; font-weight: bold; border: 1px solid black;">'
                                    f'USED IN {", ".join(map(str, other_slots))}'
                                    f'</div>'
                                )

                            # Use a unique class for the card to target with CSS
                            st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
                            with st.container(border=True):
                                if thumb:
                                    # 1. The Image with overlays
                                    selection_overlay = f'<div style="position: absolute; top: 5px; left: 5px; z-index: 10; background: #00ff00; color: black; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-weight: bold; border: 2px solid black;">✓</div>' if is_picked else ""
                                    
                                    st.markdown(
                                        f'<div style="position: relative; border: {"3px solid #00ff00" if is_picked else "1px solid #444"}; border-radius: 6px; overflow: hidden;">' 
                                        f'{selection_overlay}'
                                        f'{others_tag}'
                                        f'<img src="{thumb}" style="width:100%; aspect-ratio:16/9; object-fit:cover;">' 
                                        f'<div class="watch-btn-overlay" style="position: absolute; top: 5px; right: 5px; z-index: 10;">' 
                                        f'<a href="{url}" target="_blank" style="text-decoration: none; background: rgba(0,0,0,0.7); padding: 2px 8px; border-radius: 4px; color: white; font-size: 12px; font-weight: bold;">📺 WATCH</a>' 
                                        f'</div>' 
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.info("No preview")

                                st.markdown(f"**{source}** · {res_str} · {dur_str}")
                                st.caption(title[:60] + ("..." if len(title) > 60 else ""))
                                
                                # 3. The Status Button (Also serves as the full-card click trigger)
                                btn_label = "✅ SELECTED" if is_picked else "⬜ PICK CLIP"
                                if st.button(btn_label, key=f"galpick_{slot_id}_{hash(cand.get('url'))}", use_container_width=True, type="primary" if is_picked else "secondary"):
                                    toggle_pick(cand.get("url"), cand, shot)
                                    st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)
        # ── Shot recap below the table ───────────────────────────────────
        # Repeat the shot's narration and the search queries that produced
        # this candidate set, so the editor doesn't need to scroll back up
        # after reviewing the table.
        with st.container(border=True):
            st.markdown(
                f"**Shot {slot_id}** &nbsp;·&nbsp; ⏱ {ts} &nbsp;·&nbsp; "
                f"🎬 {shot.get('shot_intent', '—')}"
            )
            st.markdown(f"💬 _{shot.get('text', '')}_")
            queries = shot.get("search_queries", [])
            if queries:
                q_md = " &nbsp;·&nbsp; ".join(f"`{q}`" for q in queries)
                st.markdown(f"🔎 **Searched:** {q_md}")

        # ── Footer action bar ────────────────────────────────────────────
        # Prev | Skip | Save & Next (primary) | Next — gives the editor
        # quick navigation without scrolling back up to the top nav.
        # ── Redo Chunk Action ────────────────────────────────────────────
        st.markdown("---")
        chunk_id = shot.get("chunk_id", 0)
        with st.expander(f"♻️ Not happy with Chunk {chunk_id+1}? Tweak & Redo", expanded=False):
            st.caption("You can edit the search queries for all shots in this chunk here, then regenerate the suggestions.")
            
            # Find all shots in this chunk
            chunk_shots = [s for s in st.session_state.director_shots if s.get("chunk_id") == chunk_id]
            
            # Build a small editor for their queries
            c_rows = []
            for cs in chunk_shots:
                c_rows.append({
                    "Shot #": cs.get("slot_id"),
                    "Queries": " | ".join(cs.get("search_queries", [])),
                })
            c_df = pd.DataFrame(c_rows)
            edited_c = st.data_editor(
                c_df,
                column_config={
                    "Shot #": st.column_config.NumberColumn("Shot #", width="small", disabled=True),
                    "Queries": st.column_config.TextColumn("Queries (use ' | ' to separate)", width="large"),
                },
                hide_index=True,
                use_container_width=True,
                key=f"redo_editor_{chunk_id}"
            )
            
            if st.button(f"Regenerate Chunk {chunk_id+1}", key=f"redo_btn_s5_{chunk_id}", type="primary", use_container_width=True):
                # Apply edited queries back to session state first
                for i_c, cs in enumerate(chunk_shots):
                    new_q_raw = str(edited_c.iloc[i_c]["Queries"] or "")
                    new_q = [q.strip() for q in new_q_raw.split("|") if q.strip()]
                    cs["search_queries"] = new_q
                
                with st.spinner(f"Regenerating Chunk {chunk_id+1}..."):
                    perform_chunk_regeneration(chunk_id)
                    st.success(f"Chunk {chunk_id+1} regenerated.")
                    st.rerun()

        st.divider()
        fa1, fa2, fa3, fa4 = st.columns([1, 1, 2, 1])
        with fa1:
            if st.button(
                "◀ Prev", key="d_prev_bot",
                disabled=idx == 0, use_container_width=True,
            ):
                save_cache()
                st.session_state.d_review_idx -= 1
                st.rerun()
        with fa2:
            skip_label = "↩ Unskip" if skipped else "⏭ Skip"
            if st.button(skip_label, key=f"skip_{slot_id}", use_container_width=True):
                shot["skipped"] = not skipped
                shot["selected_results"] = []
                save_cache()
                st.rerun()
        with fa3:
            if idx < len(review_shots) - 1:
                if st.button(
                    "Save & Next ▶", key="d_save_next",
                    type="primary", use_container_width=True,
                ):
                    save_cache()
                    st.session_state.d_review_idx += 1
                    st.rerun()
            else:
                if st.button(
                    "✅ Finish Review", key="d_finish",
                    type="primary", use_container_width=True,
                ):
                    save_cache()
                    st.success("Review complete! Scroll down to Step 6 to start downloads.")
        with fa4:
            if st.button(
                "Next ▶", key="d_next_bot",
                disabled=idx == len(review_shots) - 1, use_container_width=True,
            ):
                save_cache()
                st.session_state.d_review_idx += 1
                st.rerun()

    # ── Step 6 — Download ────────────────────────────────────────────────────
    selected_shots = [s for s in st.session_state.get("director_shots", []) if s.get("selected_results")]
    if selected_shots:
        st.header("Step 6: Download Selected")
        st.caption(
            "Settings below apply to **new** downloads and to **retries**. Change "
            "Quality or Max Size, then click *Retry* on a failed item to re-attempt "
            "with the new settings. Clips downloaded in earlier sessions are reused "
            "automatically — no re-download."
        )

        # Persistent download cache info — collapsed by default, but shows
        # the user we're remembering past downloads and gives them a way
        # to invalidate the registry without touching the actual files.
        _cache_stats = download_cache.stats()
        if _cache_stats["count"]:
            with st.expander(
                f"📦 Download cache — {_cache_stats['count']} clip(s), "
                f"{_cache_stats['size_bytes'] / (1024 * 1024):.1f} MB"
                + (f"  ·  {_cache_stats['stale']} stale" if _cache_stats['stale'] else "")
            ):
                st.caption(
                    "URLs you've already downloaded across all sessions are remembered. "
                    "When you pick the same clip again, it's hardlinked from the cached "
                    "file instead of being re-downloaded. Stale entries (file deleted "
                    "from disk) are pruned automatically on lookup. "
                    "Clearing the registry doesn't delete any files — it just forgets the "
                    "URL→path mapping."
                )
                if st.button("Clear download cache", key="d_cache_clear"):
                    download_cache.clear()
                    st.success("Cache cleared. The actual files in `downloads/` are untouched.")
                    st.rerun()

        # Live-bound settings — read on each render so retry/new-download both see them.
        col_dv1, col_dv2, col_dv3 = st.columns(3)
        with col_dv1:
            d_quality  = st.selectbox("Quality", ["1080p", "720p", "480p", "Best", "Worst"], key="d_vq")
        with col_dv2:
            d_max_size = st.number_input("Max Size (MB)", value=200, min_value=1, key="d_maxsize")
        with col_dv3:
            d_workers  = st.slider("Concurrent", 1, 10, 3, key="d_workers")

        total_selected_files = sum(len(s.get("selected_results", [])) for s in selected_shots)

        if st.button(f"⬇ Download {total_selected_files} selected videos",
                     key="d_dl_start", type="primary", use_container_width=True):
            if not check_network():
                st.error("No network connection detected.")
            else:
                if st.session_state.dm.max_workers != d_workers:
                    st.session_state.dm = DownloadManager(max_workers=d_workers)
                st.session_state.dm.clear_and_reset()

                # ── Group selected clips by URL across all shots ─────────
                # When the same clip is picked for multiple shots we want to
                # download it once and hardlink it into the other per-shot
                # filenames. Without this grouping we'd transfer the same
                # MP4 multiple times through the user's network.
                #
                # Filename format: {chunk}-{shot}-{source}-{keyword}.mp4
                # where chunk is the active chunk index (1-based), shot is
                # the slot_id within that chunk, source is pexels/pixabay/
                # youtube, and keyword is the matched search query that
                # surfaced this clip (sanitized for the filesystem). If a
                # second pick in the same shot collides on keyword (e.g.
                # two Pexels results from one query), we suffix -2, -3 etc.
                # only the duplicates.
                def _safe_for_fs(text: str, max_len: int = 30) -> str:
                    if not text:
                        return ""
                    cleaned = "".join(c if (c.isalnum() or c in " -_") else " "
                                      for c in text)
                    cleaned = "-".join(cleaned.split()).lower()
                    return cleaned[:max_len].strip("-") or ""

                indices = st.session_state.get("active_chunk_indices", [0])
                chunk_str = "_".join(str(i+1) for i in sorted(indices))

                url_groups = {}      # url -> {"primary": (path, dm_source), "extras": [...], "shots": [...]}
                seen_filenames = set()  # collision detector for this batch
                for shot in selected_shots:
                    slot_id = shot.get("slot_id", "X")
                    for res in shot.get("selected_results", []):
                        url = res.get("url")
                        if not url:
                            continue
                        source  = res.get("source", "stock")
                        keyword = _safe_for_fs(res.get("matched_query", ""), 30) or "clip"
                        base    = f"{chunk_str}-{slot_id}-{source}-{keyword}"
                        filename = f"{base}.mp4"
                        n = 1
                        while filename in seen_filenames:
                            n += 1
                            filename = f"{base}-{n}.mp4"
                        seen_filenames.add(filename)

                        output_path = os.path.join("downloads", "director", filename)
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
                    if cached and cached != primary_path:
                        ok_primary = link_or_copy(cached, primary_path)
                        for ext in extras:
                            link_or_copy(cached, ext)
                        if ok_primary:
                            cached_hits += 1
                            continue
                        # If we couldn't materialize from cache, fall
                        # through to a fresh download below.

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
                    st.success("Done — " + "; ".join(parts) + ".")
                else:
                    st.warning("Nothing was queued — check URLs.")

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
                    f"⬇ {stats['downloading']} active · "
                    f"⏸ {stats.get('paused', 0)} paused · "
                    f"⏳ {stats['queued']} queued · "
                    f"❌ {stats['error']} failed · "
                    f"⏭ {stats['cancelled']} cancelled"
                )
            with cancel_col:
                if stats["downloading"] + stats["queued"] + stats.get("paused", 0) > 0:
                    if st.button("✖ Cancel All", key="d_cancel", use_container_width=True):
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
                        + (f" — {meta}" if meta else "")
                    )
                    extras_n = len(t.get("extra_paths") or [])
                    extras_lbl = f" · ↗ +{extras_n} mirror{'s' if extras_n > 1 else ''}" if extras_n else ""
                    st.markdown(f"**{os.path.basename(t['output_path'])}**{extras_lbl} — {label}")
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
                st.subheader(f"❌ Failed ({len(failed)})")
                retryable = [f for f in failed if st.session_state.dm.can_retry(f["id"])]
                if retryable:
                    if st.button(
                        f"↻ Retry all ({len(retryable)}) with current settings",
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
                            extras_lbl = f" · ↗ +{extras_n} mirror{'s' if extras_n > 1 else ''}" if extras_n else ""
                            st.markdown(f"**{os.path.basename(ft['output_path'])}**{extras_lbl}")
                            st.caption(
                                f"⚠ {ft.get('error_summary') or ft.get('error_msg') or 'Unknown error'} · "
                                f"attempt {ft.get('attempts', 0)}"
                            )
                        with top2:
                             if st.button(
                                "↻ Retry",
                                key=f"retry_{ft['id']}",
                                use_container_width=True,
                                help="Retry with the current Quality / Max Size settings.",
                            ):
                                st.session_state.dm.retry_failed(ft["id"], overrides={
                                    "quality": d_quality, "max_size_mb": d_max_size,
                                })
                                st.rerun()
                        # URL row — let the user copy or open the source page
                        url_col, link_col = st.columns([5, 1])
                        with url_col:
                            st.code(ft.get("url", ""), language=None)
                        with link_col:
                            if ft.get("url"):
                                st.link_button("Open ↗", ft["url"], use_container_width=True)
                        # Full error in a collapsed expander for power users
                        if ft.get("error_msg") and ft["error_msg"] != ft.get("error_summary"):
                            with st.expander("Full error"):
                                st.code(ft["error_msg"], language=None)

            # ── History (preserved across batches) ─────────────────────────
            history = [t for t in st.session_state.dm.get_history()
                       if t["status"] in ("completed", "cancelled")
                       and t["id"] not in {x["id"] for x in tasks}]  # exclude current batch
            if history:
                with st.expander(f"📜 History — {len(history)} task(s) from earlier batches"):
                    h_completed = sum(1 for h in history if h["status"] == "completed")
                    h_cancelled = sum(1 for h in history if h["status"] == "cancelled")
                    st.caption(f"✅ {h_completed} completed · ⏭ {h_cancelled} cancelled")
                    if st.button("Clear history", key="d_clear_history"):
                        st.session_state.dm.clear_history(); st.rerun()
                    for h in history[-20:]:  # cap display to avoid blowing up the page
                        icon = "✅" if h["status"] == "completed" else "⏭"
                        st.write(f"{icon} {os.path.basename(h['output_path'])}")

            # Auto-refresh while anything is still moving (downloading, queued,
            # paused, or post-download processing).
            in_motion = (stats["downloading"] + stats["queued"]
                         + stats.get("paused", 0) + stats.get("processing", 0))
            if in_motion > 0:
                time.sleep(1); st.rerun()

    # ── Step 7 — Export ──────────────────────────────────────────────────────
    if st.session_state.director_shots:
        st.header("Step 7: Export")
        shot_list_json = json.dumps(st.session_state.director_shots, indent=2)
        shot_list_txt  = generate_shot_list_txt(st.session_state.director_shots)
        fcpxml         = generate_fcpxml(st.session_state.director_shots)
        
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
            st.download_button("markers.fcpxml", data=fcpxml,         file_name="markers.fcpxml", mime="text/xml")
