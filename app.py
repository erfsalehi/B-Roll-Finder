import streamlit as st
import os
import json
import zipfile
import io
import socket
from dotenv import load_dotenv, set_key

from core.timing import get_audio_duration, parse_script_to_slots, calculate_wps
from core.keywords import generate_keywords_for_slots, generate_keywords_with_ai_chunking
from core.youtube import fetch_youtube_results
from core.stock_apis import search_pexels, search_pixabay
from core.output import generate_keywords_txt, generate_youtube_txt, generate_srt
from core.download_manager import DownloadManager
import time

# --- Config & Initialization ---
st.set_page_config(page_title="B-Roll Finder", layout="wide")

ENV_FILE = ".env"
CACHE_FILE = ".cache/session_state.json"

if not os.path.exists(".cache"):
    os.makedirs(".cache")

load_dotenv(ENV_FILE)

def _check_network() -> bool:
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except Exception:
        return False

def _format_speed(bps) -> str:
    if not bps:
        return ""
    if bps >= 1_048_576:
        return f"{bps / 1_048_576:.1f} MB/s"
    return f"{bps / 1024:.0f} KB/s"

def _format_eta(seconds) -> str:
    if seconds is None:
        return ""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60}s"

# Initialize Session State
if "slots" not in st.session_state:
    st.session_state.slots = []
if "script_text" not in st.session_state:
    st.session_state.script_text = ""
if "audio_duration" not in st.session_state:
    st.session_state.audio_duration = 0.0
if "transcription_segments" not in st.session_state:
    st.session_state.transcription_segments = []
if "global_themes" not in st.session_state:
    st.session_state.global_themes = []
if "dm" not in st.session_state:
    st.session_state.dm = DownloadManager()
if "is_fetching" not in st.session_state:
    st.session_state.is_fetching = False

# Attempt to load from cache
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
                st.session_state.slots = data.get("slots", [])
                st.session_state.director_shots = data.get("director_shots", [])
                st.session_state.script_text = data.get("script_text", "")
                st.session_state.audio_duration = data.get("audio_duration", 0.0)
                st.session_state.global_themes = data.get("global_themes", [])
        except Exception as e:
            st.error(f"Error loading cache: {e}")

def save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({
                "slots": st.session_state.slots,
                "director_shots": st.session_state.get("director_shots", []),
                "script_text": st.session_state.script_text,
                "audio_duration": st.session_state.audio_duration,
                "global_themes": st.session_state.get("global_themes", [])
            }, f)
    except Exception as e:
        st.error(f"Error saving cache: {e}")

if not st.session_state.slots and os.path.exists(CACHE_FILE):
    load_cache()

# --- UI Components ---
st.sidebar.title("App Mode")
app_mode = st.sidebar.radio("Select Mode", ["Classic Finder", "Director (v0.2)"])

def render_classic_mode():
    st.title("🎬 B-Roll Finder")

    # Step 1: Setup
    with st.expander("Step 1: Setup (API Keys)", expanded=not bool(os.getenv("GROQ_API_KEY"))):

        # Auto-load keys from txt files if available and not in env
        for txt_file, env_key in [("groq api.txt", "GROQ_API_KEY"), ("Pexels Api.txt", "PEXELS_API_KEY"), ("Pixabay Api.txt", "PIXABAY_API_KEY")]:
            if not os.getenv(env_key) and os.path.exists(txt_file):
                try:
                    with open(txt_file, 'r') as f:
                        key_val = f.read().strip()
                        if key_val:
                            set_key(ENV_FILE, env_key, key_val)
                except Exception as e:
                    pass
        load_dotenv(ENV_FILE, override=True)

        current_groq = os.getenv("GROQ_API_KEY", "")
        current_pexels = os.getenv("PEXELS_API_KEY", "")
        current_pixabay = os.getenv("PIXABAY_API_KEY", "")

        groq_input = st.text_input("Groq API Key (Required)", value=current_groq, type="password")
        pexels_input = st.text_input("Pexels API Key (Optional)", value=current_pexels, type="password")
        pixabay_input = st.text_input("Pixabay API Key (Optional)", value=current_pixabay, type="password")

        if st.button("Save API Keys"):
            if groq_input:
                if not os.path.exists(ENV_FILE):
                    open(ENV_FILE, 'w').close()
                set_key(ENV_FILE, "GROQ_API_KEY", groq_input)
                if pexels_input: set_key(ENV_FILE, "PEXELS_API_KEY", pexels_input)
                if pixabay_input: set_key(ENV_FILE, "PIXABAY_API_KEY", pixabay_input)
                load_dotenv(ENV_FILE, override=True)
                st.success("API Keys saved to .env!")
            else:
                st.warning("Please enter a valid Groq API key.")

    # Step 2: Upload
    st.header("Step 2: Upload Files")
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
        with st.expander("View Uploaded Script"):
            st.text_area("Full Script", value=script_content, height=200, disabled=True)

        st.session_state.script_text = script_content
        st.session_state.audio_duration = duration

    # Step 2.5: AI Transcription (Optional for Precise Timing)
    st.header("Step 2.5: AI Transcription (Optional)")
    if audio_file and st.button("Transcribe Audio for Precise Timing"):
        if not os.getenv("GROQ_API_KEY"):
            st.error("Please set Groq API key in Step 1.")
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
            st.error("Please set your Groq API Key in Step 1.")
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
            if not _check_network():
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

    # Step 6: Download Videos (Local)
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
        elif not _check_network():
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
                speed_str = _format_speed(t.get('speed'))
                eta_str = _format_eta(t.get('eta'))
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

        with col1:
            st.download_button("Download keywords.txt", data=keywords_txt, file_name="keywords.txt", mime="text/plain")
        with col2:
            st.download_button("Download youtube_results.txt", data=yt_txt, file_name="youtube_results.txt", mime="text/plain")
        with col3:
            st.download_button("Download timing.srt", data=srt_txt, file_name="timing.srt", mime="text/plain")

        with col4:
            # Create ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("keywords.txt", keywords_txt)
                zip_file.writestr("youtube_results.txt", yt_txt)
                zip_file.writestr("timing.srt", srt_txt)

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
                st.error("Please set your Groq API Key in Step 1.")
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
            elif not _check_network():
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

        if c_gf2.button("Start Downloading Global Footage"):
            # Update state from editor just in case keywords were changed, but DON'T clear results
            # if we have them. This is tricky. Let's just use what's in session state.
            
            added_count = 0
            has_any_results = any(len(t.get('video_results', [])) > 0 for t in st.session_state.global_themes)
            
            if not has_any_results:
                st.error("No links have been fetched yet. Please click 'Fetch Global Video Links' first.")
            else:
                for theme in st.session_state.global_themes:
                    results = theme.get('video_results', [])
                    if not results: continue
                    
                    for j, res in enumerate(results):
                        url = res.get('url')
                        if not url: continue
                        
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
elif app_mode == "Director (v0.2)":
    from core.director import generate_shot_list
    from core.director_search import fetch_director_footage
    from core.director_rank import rank_shot_candidates
    from core.output import generate_fcpxml, generate_shot_list_txt
    st.title("🎬 B-Roll Director (v0.2)")
    
    with st.expander("Step 1: Setup (API Keys)", expanded=not bool(os.getenv("GROQ_API_KEY"))):
        # We can reuse the keys from .env
        groq_input = st.text_input("Groq API Key (Required)", value=os.getenv("GROQ_API_KEY", ""), type="password", key="d_groq")
        pexels_input = st.text_input("Pexels API Key (Optional)", value=os.getenv("PEXELS_API_KEY", ""), type="password", key="d_pex")
        pixabay_input = st.text_input("Pixabay API Key (Optional)", value=os.getenv("PIXABAY_API_KEY", ""), type="password", key="d_pix")
        if st.button("Save API Keys", key="d_save_keys"):
            if groq_input:
                set_key(ENV_FILE, "GROQ_API_KEY", groq_input)
                if pexels_input: set_key(ENV_FILE, "PEXELS_API_KEY", pexels_input)
                if pixabay_input: set_key(ENV_FILE, "PIXABAY_API_KEY", pixabay_input)
                load_dotenv(ENV_FILE, override=True)
                st.success("API Keys saved to .env!")
    
    st.header("Step 2: Upload Files")
    col1, col2 = st.columns(2)
    with col1:
        script_file = st.file_uploader("Upload Script (.txt)", type=["txt"], key="d_script")
    with col2:
        audio_file = st.file_uploader("Upload Voiceover (to extract duration)", type=["mp3", "wav", "m4a"], key="d_audio")
        
    if script_file and audio_file:
        script_content = script_file.read().decode("utf-8")
        
        audio_path = os.path.join(".cache", "temp_audio_director" + os.path.splitext(audio_file.name)[1])
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())
            
        st.session_state.script_text = script_content
        st.session_state.audio_duration = get_audio_duration(audio_path)
        
        st.info(f"Script words: {len(st.session_state.script_text.split())} | Audio Duration: {st.session_state.audio_duration:.2f}s")
        st.audio(audio_path)
        with st.expander("View Uploaded Script"):
            st.text_area("Full Script", value=st.session_state.script_text, height=200, disabled=True, key="d_full_script")
        
    st.header("Step 3: Director's Vision (Stage 1)")
    
    with st.expander("Range Selection", expanded=False):
        d_use_portion = st.checkbox("Process Specific Portion", value=False, key="d_use_p")
        col_dp1, col_dp2 = st.columns(2)
        with col_dp1:
            d_start_time = st.number_input("Start Time (seconds)", value=0.0, step=1.0, key="d_start")
        with col_dp2:
            d_end_time = st.number_input("End Time (seconds)", value=min(st.session_state.audio_duration, 300.0) if st.session_state.audio_duration > 0 else 60.0, step=1.0, key="d_end")

    custom_instructions = st.text_area("Style Hints (Optional)", placeholder="e.g. cinematic, slow motion, no talking heads", key="d_style")
    
    if "director_shots" not in st.session_state:
        st.session_state.director_shots = []
        
    if st.button("Generate Shot List"):
        if not os.getenv("GROQ_API_KEY"):
            st.error("Please set Groq API key.")
        elif not st.session_state.script_text or st.session_state.audio_duration <= 0:
            st.error("Please upload script and audio.")
        else:
            wps = calculate_wps(st.session_state.script_text, st.session_state.audio_duration)
            pbar = st.progress(0)
            status = st.empty()
            try:
                target_script = st.session_state.script_text
                start_offset = 0.0
                if d_use_portion:
                    from core.timing import slice_script_by_time
                    target_script = slice_script_by_time(st.session_state.script_text, st.session_state.audio_duration, d_start_time, d_end_time)
                    start_offset = d_start_time
                    status.info(f"Processing portion: {d_start_time}s to {d_end_time}s")

                if st.session_state.transcription_segments and not d_use_portion:
                    from core.director import generate_shot_list_from_transcription
                    st.session_state.director_shots = generate_shot_list_from_transcription(
                        st.session_state.transcription_segments,
                        os.getenv("GROQ_API_KEY"),
                        progress_callback=lambda p: pbar.progress(p),
                        custom_instructions=custom_instructions
                    )
                else:
                    st.session_state.director_shots = generate_shot_list(
                        target_script,
                        wps,
                        os.getenv("GROQ_API_KEY"),
                        progress_callback=lambda p: pbar.progress(p),
                        custom_instructions=custom_instructions,
                        start_offset=start_offset
                    )
                pbar.progress(1.0)
                st.success("Shot list generated successfully!")
            except Exception as e:
                st.error(f"Error generating shot list: {e}")
                
    # ── Session state for director ──────────────────────────────────────────
    if "director_shots" not in st.session_state:
        st.session_state.director_shots = []

    # ── Stage 1 shot list table ──────────────────────────────────────────────
    if st.session_state.director_shots:
        st.subheader("Shot List")
        table_data = []
        for shot in st.session_state.director_shots:
            sel = shot.get("selected_results", [])
            table_data.append({
                "Time":       f"{shot.get('timestamp_start_str')} – {shot.get('timestamp_end_str')}",
                "Intent":     shot.get("shot_intent", ""),
                "Priority":   shot.get("priority", ""),
                "Queries":    " | ".join(shot.get("search_queries", [])),
                "Candidates": len(shot.get("video_results", [])),
                "Selected":   f"✅ {len(sel)}" if sel else ("⏭ Skipped" if shot.get("skipped") else "—"),
            })
        st.dataframe(table_data, use_container_width=True)

    # ── Stage 2 — Fetch Candidates ───────────────────────────────────────────
    if st.session_state.director_shots:
        st.header("Stage 2: Fetch Candidates")

        col_s2a, col_s2b, col_s2c = st.columns(3)
        with col_s2a:
            use_pexels  = st.checkbox("Pexels",  value=bool(os.getenv("PEXELS_API_KEY")),  key="d_pex_cb")
        with col_s2b:
            use_pixabay = st.checkbox("Pixabay", value=bool(os.getenv("PIXABAY_API_KEY")), key="d_pix_cb")
        with col_s2c:
            d_num_results = st.number_input("Results per query", value=3, min_value=1, max_value=10, key="d_nr")

        if st.button("Fetch Candidates", disabled=st.session_state.is_fetching, key="d_fetch"):
            if not _check_network():
                st.error("No network connection detected.")
            elif not (use_pexels and os.getenv("PEXELS_API_KEY")) and \
                 not (use_pixabay and os.getenv("PIXABAY_API_KEY")):
                st.error("No stock API keys configured. Add Pexels and/or Pixabay keys in Step 1.")
            else:
                st.session_state.is_fetching = True
                d_fetch_errors = []
                try:
                    pbar2 = st.progress(0)
                    status2 = st.empty()
                    status2.text("Fetching candidates…")
                    st.session_state.director_shots = fetch_director_footage(
                        st.session_state.director_shots,
                        use_pexels=use_pexels,
                        use_pixabay=use_pixabay,
                        num_results=d_num_results,
                        progress_callback=lambda p: pbar2.progress(p),
                        errors=d_fetch_errors,
                    )
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

    # ── Stage 3 — LLM Ranking ────────────────────────────────────────────────
    has_candidates = any(len(s.get("video_results", [])) > 1 for s in st.session_state.get("director_shots", []))
    if has_candidates:
        st.header("Stage 3: Rank Candidates")
        st.caption("AI reorders candidates by visual relevance and flags off-topic results.")
        d_video_topic = st.text_input(
            "What is this video about? (helps reject off-topic candidates)",
            placeholder="e.g. car mechanics and engine repair",
            key="d_video_topic"
        )
        if st.button("Rank Candidates with AI", key="d_rank"):
            if not os.getenv("GROQ_API_KEY"):
                st.error("Groq API key required for ranking.")
            else:
                pbar3 = st.progress(0)
                status3 = st.empty()
                status3.text("Ranking…")
                try:
                    st.session_state.director_shots = rank_shot_candidates(
                        st.session_state.director_shots,
                        api_key=os.getenv("GROQ_API_KEY"),
                        custom_instructions=custom_instructions,
                        video_topic=d_video_topic,
                        progress_callback=lambda p: pbar3.progress(p),
                    )
                    pbar3.progress(1.0)
                    status3.text("Done.")
                    st.success("Candidates ranked! Irrelevant clips are hidden in the review.")
                    save_cache()
                except Exception as e:
                    st.error(f"Ranking error: {e}")

    # ── Stage 4 — Editor Review (paginated) ─────────────────────────────────
    review_shots = [s for s in st.session_state.get("director_shots", [])
                    if s.get("video_results") and s.get("priority") != "none"]
    if review_shots:
        st.header("Stage 4: Review & Select")

        if "d_review_idx" not in st.session_state:
            st.session_state.d_review_idx = 0
        st.session_state.d_review_idx = min(st.session_state.d_review_idx, len(review_shots) - 1)
        idx = st.session_state.d_review_idx
        shot = review_shots[idx]

        # Progress / navigation bar
        n_selected = sum(1 for s in review_shots if s.get("selected_results"))
        n_skipped  = sum(1 for s in review_shots if s.get("skipped"))
        n_pending  = len(review_shots) - n_selected - n_skipped
        st.caption(f"Shot {idx + 1} of {len(review_shots)}  ·  ✅ {n_selected} selected · ⏭ {n_skipped} skipped · ⏳ {n_pending} pending")
        st.progress((idx + 1) / len(review_shots))

        nav1, nav2, nav3 = st.columns([1, 6, 1])
        with nav1:
            if st.button("◀ Prev", key="d_prev", disabled=idx == 0):
                st.session_state.d_review_idx -= 1
                st.rerun()
        with nav3:
            if st.button("Next ▶", key="d_next", disabled=idx == len(review_shots) - 1):
                st.session_state.d_review_idx += 1
                st.rerun()

        # Shot details
        slot_id  = shot.get("slot_id", "?")
        ts       = f"{shot.get('timestamp_start_str')} – {shot.get('timestamp_end_str')}"
        reason   = shot.get("rank_reason", "")
        sel_urls = {r.get("url") for r in shot.get("selected_results", [])}
        skipped  = shot.get("skipped", False)

        st.markdown(f"**Shot {slot_id}** · {ts}")
        st.markdown(f"**Intent:** {shot.get('shot_intent', '')}")
        st.markdown(f"**Narration:** _{shot.get('text', '')}_")
        if reason:
            st.info(f"🤖 {reason}")
        if skipped:
            st.warning("⏭ This shot is marked as skipped.")

        # Candidate cards — filter out irrelevant ones if ranked
        candidates = [c for c in shot["video_results"] if not c.get("irrelevant")]
        hidden     = len(shot["video_results"]) - len(candidates)
        if hidden:
            st.caption(f"🚫 {hidden} off-topic candidate(s) hidden by AI ranking.")
        if not candidates:
            st.warning("No relevant candidates found for this shot.")
        else:
            cols = st.columns(min(len(candidates), 3))
            for j, res in enumerate(candidates):
                with cols[j % 3]:
                    thumb = res.get("thumbnail", "")
                    if thumb:
                        st.image(thumb, use_container_width=True)
                    src = res.get("source", "?").upper()
                    w   = res.get("width", "?")
                    h   = res.get("height", "?")
                    dur = res.get("duration", "?")
                    st.markdown(f"**{res.get('title', '?')}**")
                    st.caption(f"{src} · {w}×{h} · {dur}s")
                    if res.get("description"):
                        st.caption(res["description"])

                    is_sel = res.get("url") in sel_urls
                    label  = "✅ Deselect" if is_sel else "Select"
                    if st.button(label, key=f"sel_{slot_id}_{j}", use_container_width=True):
                        current = shot.get("selected_results", [])
                        if is_sel:
                            shot["selected_results"] = [r for r in current if r.get("url") != res.get("url")]
                        else:
                            shot["selected_results"] = current + [res]
                        shot["skipped"] = False
                        # Stay on current shot — no rerun jump

        # Skip / next controls
        sk1, sk2 = st.columns(2)
        with sk1:
            skip_label = "↩ Unskip" if skipped else "⏭ Skip this shot"
            if st.button(skip_label, key=f"skip_{slot_id}"):
                shot["skipped"] = not skipped
                shot["selected_results"] = []
        with sk2:
            if idx < len(review_shots) - 1:
                if st.button("Save & Next ▶", key="d_save_next"):
                    save_cache()
                    st.session_state.d_review_idx += 1
                    st.rerun()
            else:
                if st.button("✅ Finish Review", key="d_finish"):
                    save_cache()
                    st.success("Review complete! Scroll down to download.")

    # ── Stage 5 — Download ───────────────────────────────────────────────────
    selected_shots = [s for s in st.session_state.get("director_shots", []) if s.get("selected_results")]
    if selected_shots:
        st.header("Stage 5: Download Selected")

        col_dv1, col_dv2, col_dv3 = st.columns(3)
        with col_dv1:
            d_quality  = st.selectbox("Quality", ["1080p", "720p", "480p", "Best", "Worst"], key="d_vq")
        with col_dv2:
            d_max_size = st.number_input("Max Size (MB)", value=200, min_value=1, key="d_maxsize")
        with col_dv3:
            d_workers  = st.slider("Concurrent", 1, 10, 3, key="d_workers")

        total_selected_files = sum(len(s.get("selected_results", [])) for s in selected_shots)
        c_dl1, c_dl2 = st.columns(2)
        if c_dl1.button(f"Download {total_selected_files} Selected Videos", key="d_dl_start"):
            if not _check_network():
                st.error("No network connection detected.")
            else:
                if st.session_state.dm.max_workers != d_workers:
                    st.session_state.dm = DownloadManager(max_workers=d_workers)
                st.session_state.dm.clear_and_reset()

                added = 0
                for shot in selected_shots:
                    slot_id  = shot.get("slot_id", "X")
                    safe_int = "".join(c if c.isalnum() or c in " -_" else "_"
                                       for c in shot.get("shot_intent", "shot")).strip()
                    for k, res in enumerate(shot.get("selected_results", [])):
                        url = res.get("url")
                        if not url:
                            continue
                        source      = res.get("source", "stock")
                        filename    = f"Shot{slot_id}-{source}-{safe_int[:25]}-{k+1}.mp4"
                        output_path = os.path.join("downloads", "director", filename)
                        dm_source   = "direct" if source in ("pexels", "pixabay") else "youtube"
                        try:
                            task_id = st.session_state.dm.add_download(
                                url, output_path, d_quality,
                                source=dm_source, max_size_mb=d_max_size, normalize=False,
                            )
                            st.session_state.dm.start_download(task_id)
                            added += 1
                        except Exception as e:
                            st.error(f"Shot {slot_id}: {e}")

                if added:
                    st.success(f"Queued {added} downloads.")
                else:
                    st.warning("Nothing was queued — check URLs.")

        if c_dl2.button("Cancel All", key="d_cancel"):
            st.session_state.dm.cancel_all()

        # Download dashboard
        tasks = st.session_state.dm.get_all_tasks()
        if tasks:
            stats     = st.session_state.dm.get_stats()
            total_t   = stats["total"]
            completed = stats["completed"]
            st.progress(completed / total_t if total_t else 0.0)
            st.write(f"**{completed}/{total_t}** done · Active: {stats['downloading']} · Queued: {stats['queued']} · Failed: {stats['error']}")

            failed = st.session_state.dm.get_failed_tasks()
            if failed:
                with st.expander(f"⚠️ {len(failed)} failed"):
                    if st.button("Retry All Failed", key="d_retry"):
                        st.session_state.dm.retry_all_failed(); st.rerun()
                    for ft in failed:
                        st.write(f"❌ {os.path.basename(ft['output_path'])} — {ft.get('error_msg','?')}")

            active = st.session_state.dm.get_active_tasks()
            if active:
                st.subheader("Downloading")
                for t in active:
                    is_proc    = t["status"] == "processing"
                    speed_str  = _format_speed(t.get("speed"))
                    eta_str    = _format_eta(t.get("eta"))
                    meta       = " | ".join(filter(None, [speed_str, f"ETA {eta_str}" if eta_str else ""]))
                    label      = "Normalizing…" if is_proc else f"{t['status'].title()} ({t['progress']*100:.1f}%){(' — ' + meta) if meta else ''}"
                    st.write(f"**{os.path.basename(t['output_path'])}** — {label}")
                    st.progress(t["progress"])
                    b1, b2, _ = st.columns(3)
                    if not is_proc:
                        if t["status"] == "downloading":
                            if b1.button("Pause",  key=f"pd_{t['id']}"): st.session_state.dm.pause_download(t["id"]);  st.rerun()
                        elif t["status"] == "paused":
                            if b1.button("Resume", key=f"rd_{t['id']}"): st.session_state.dm.resume_download(t["id"]); st.rerun()
                    if b2.button("Cancel", key=f"cd_{t['id']}"): st.session_state.dm.cancel_download(t["id"]); st.rerun()
                time.sleep(1); st.rerun()
            elif stats["queued"] > 0:
                time.sleep(1); st.rerun()

    # ── Export ───────────────────────────────────────────────────────────────
    if st.session_state.director_shots:
        st.header("Export")
        shot_list_json = json.dumps(st.session_state.director_shots, indent=2)
        shot_list_txt  = generate_shot_list_txt(st.session_state.director_shots)
        fcpxml         = generate_fcpxml(st.session_state.director_shots)
        c1, c2, c3     = st.columns(3)
        with c1: st.download_button("shot_list.json", data=shot_list_json, file_name="shot_list.json", mime="application/json")
        with c2: st.download_button("shot_list.txt",  data=shot_list_txt,  file_name="shot_list.txt",  mime="text/plain")
        with c3: st.download_button("markers.fcpxml", data=fcpxml,         file_name="markers.fcpxml", mime="text/xml")
