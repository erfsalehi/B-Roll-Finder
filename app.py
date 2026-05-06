import streamlit as st
import os
import json
import zipfile
import io
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

# Initialize Session State
if "slots" not in st.session_state:
    st.session_state.slots = []
if "script_text" not in st.session_state:
    st.session_state.script_text = ""
if "audio_duration" not in st.session_state:
    st.session_state.audio_duration = 0.0
if "dm" not in st.session_state:
    st.session_state.dm = DownloadManager()

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
        except Exception as e:
            st.error(f"Error loading cache: {e}")

def save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({
                "slots": st.session_state.slots,
                "director_shots": st.session_state.get("director_shots", []),
                "script_text": st.session_state.script_text,
                "audio_duration": st.session_state.audio_duration
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

        st.session_state.script_text = script_content
        st.session_state.audio_duration = duration

    # Step 3: Settings
    st.header("Step 3: Settings")
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
                if chunking_method == "AI Meaningful Chunks (Recommended)":
                    wps = calculate_wps(st.session_state.script_text, st.session_state.audio_duration)
                    st.session_state.slots = generate_keywords_with_ai_chunking(
                        st.session_state.script_text,
                        wps=wps,
                        api_key=os.getenv("GROQ_API_KEY"),
                        num_alternatives=num_alt,
                        progress_callback=update_progress,
                        custom_instructions=custom_instructions
                    )
                else:
                    # 1. Parse slots mathematically
                    slots = parse_script_to_slots(
                        st.session_state.script_text, 
                        st.session_state.audio_duration, 
                        intro_duration=intro_dur, 
                        intro_interval=intro_int, 
                        body_interval=body_int
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

        edited_df = st.data_editor(table_data, num_rows="dynamic", use_container_width=True)

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

    if st.button("Fetch Links"):
        if not st.session_state.slots:
            st.error("Please generate keywords first.")
        else:
            progress_bar_yt = st.progress(0)
            status_text_yt = st.empty()

            # First YouTube
            if yt_shorts > 0 or yt_longs > 0:
                status_text_yt.text("Searching YouTube...")
                st.session_state.slots = fetch_youtube_results(
                    st.session_state.slots, 
                    num_shorts=yt_shorts, 
                    num_longs=yt_longs, 
                    progress_callback=lambda p: progress_bar_yt.progress(p * 0.3)
                )

            # Then Pexels & Pixabay
            total_slots = len(st.session_state.slots)
            for idx, slot in enumerate(st.session_state.slots):
                keywords = slot.get('keywords', [])
                primary_kw = keywords[0] if keywords else None

                # CLEAR existing video results so we respect exactly what the user currently asked for
                slot['video_results'] = []

                # Migrate fresh youtube_results from the above fetch
                if 'youtube_results' in slot:
                    slot['video_results'].extend(slot['youtube_results'])
                    del slot['youtube_results']

                if primary_kw:
                    if pexels_count > 0 and os.getenv("PEXELS_API_KEY"):
                        pexels_res = search_pexels(primary_kw, os.getenv("PEXELS_API_KEY"), pexels_count)
                        slot['video_results'].extend(pexels_res)

                    if pixabay_count > 0 and os.getenv("PIXABAY_API_KEY"):
                        pixabay_res = search_pixabay(primary_kw, os.getenv("PIXABAY_API_KEY"), pixabay_count)
                        slot['video_results'].extend(pixabay_res)

                progress_bar_yt.progress(0.3 + (0.7 * (idx + 1) / total_slots))

            progress_bar_yt.progress(1.0)
            status_text_yt.text("All video links fetched!")
            save_cache()
            st.success("Video results fetched!")

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
        else:
            if st.session_state.dm.max_workers != max_workers:
                st.session_state.dm = DownloadManager(max_workers=max_workers)

            # Safely cancel all in-flight tasks and reset the executor
            st.session_state.dm.clear_and_reset()

            for i, slot in enumerate(st.session_state.slots):
                results = slot.get('video_results', slot.get('youtube_results', []))
                primary_kw = slot.get('keywords', ['Unknown'])[0]
                safe_kw = "".join([c if c.isalnum() or c in " -_" else "_" for c in primary_kw]).strip()

                # In Classic Mode, we might want to download all or just the top one. 
                # Current behavior seems to download all candidates.
                for j, res in enumerate(results):
                    source = res.get('source', 'youtube')
                    filename = f"{i+1}-{j+1}-{source}-{safe_kw}.mp4"
                    output_path = os.path.join("downloads", filename)

                    dm_source = 'direct' if source in ['pexels', 'pixabay'] else 'youtube'
                    task_id = st.session_state.dm.add_download(
                        res['url'], 
                        output_path, 
                        video_quality, 
                        source=dm_source,
                        max_size_mb=max_size_mb,
                        strict_quality=strict_quality,
                        normalize=normalize_res
                    )
                    st.session_state.dm.start_download(task_id)

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
                st.write(f"**{os.path.basename(t['output_path'])}** - {t['status'].title()} ({t['progress']*100:.1f}%)")
                st.progress(t['progress'])

                b1, b2, b3 = st.columns(3)
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

if app_mode == "Classic Finder":
    render_classic_mode()
elif app_mode == "Director (v0.2)":
    from core.director import generate_shot_list
    from core.director_search import fetch_director_footage
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
        
    st.header("Step 3: Director's Vision (Stage 1)")
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
                st.session_state.director_shots = generate_shot_list(
                    st.session_state.script_text,
                    wps,
                    os.getenv("GROQ_API_KEY"),
                    progress_callback=lambda p: pbar.progress(p),
                    custom_instructions=custom_instructions
                )
                pbar.progress(1.0)
                st.success("Shot list generated successfully!")
            except Exception as e:
                st.error(f"Error generating shot list: {e}")
                
    if st.session_state.director_shots:
        st.subheader("Generated Shot List")
        # Display as a table
        table_data = []
        for shot in st.session_state.director_shots:
            table_data.append({
                "Time": f"{shot.get('timestamp_start_str')} - {shot.get('timestamp_end_str')}",
                "Intent": shot.get("shot_intent"),
                "Type": shot.get("shot_type"),
                "Priority": shot.get("priority"),
                "Queries": ", ".join(shot.get("search_queries", [])),
                "Candidate Specs": f"{shot['video_results'][0].get('width', '?')}x{shot['video_results'][0].get('height', '?')} | {round(shot['video_results'][0].get('file_size', 0)/(1024*1024), 1) if shot['video_results'][0].get('file_size') else '?'}MB" if shot.get('video_results') else "No candidates"
            })
        st.dataframe(table_data, use_container_width=True)
        
        st.header("Step 4: Fetch Footage (Stage 2)")
        col_src1, col_src2, col_src3 = st.columns(3)
        with col_src1:
            use_youtube = st.checkbox("Include YouTube (yt-dlp)", value=True, key="d_yt")
        with col_src2:
            use_pexels = st.checkbox("Include Pexels", value=True, key="d_pex_cb")
        with col_src3:
            use_pixabay = st.checkbox("Include Pixabay", value=True, key="d_pix_cb")
            
        if st.button("Find Footage"):
            pbar2 = st.progress(0)
            st.session_state.director_shots = fetch_director_footage(
                st.session_state.director_shots,
                use_youtube=use_youtube,
                use_pexels=use_pexels,
                use_pixabay=use_pixabay,
                progress_callback=lambda p: pbar2.progress(p)
            )
            pbar2.progress(1.0)
            save_cache()
            st.success("Footage candidates found!")
            
        if st.button("Apply Filters to Candidates"):
            filtered_count = 0
            for shot in st.session_state.director_shots:
                results = shot.get('video_results', [])
                if not results:
                    continue
                
                res_map = {'1080p': 1080, '720p': 720, '480p': 480}
                min_h = res_map.get(video_quality, 0)
                
                new_results = []
                for res in results:
                    # Filter by resolution if strict
                    if strict_quality:
                        if res.get('height') and res.get('height') < min_h:
                            continue
                    
                    # Filter by size if known
                    if max_size_mb:
                        if res.get('file_size') and res.get('file_size') / (1024*1024) > max_size_mb:
                            continue
                            
                    new_results.append(res)
                
                if len(new_results) != len(results):
                    filtered_count += (len(results) - len(new_results))
                    shot['video_results'] = new_results
            
            st.success(f"Filtered out {filtered_count} candidates that didn't meet criteria.")
            save_cache()
            st.rerun()
            
        # --- Download Manager UI for Director Mode ---
        col_vid1, col_vid2, col_vid3, col_vid4, col_vid5 = st.columns(5)
        with col_vid1:
            video_quality = st.selectbox("Video Quality", ["1080p", "720p", "480p", "Best", "Worst"], index=0, key="d_vq")
        with col_vid2:
            strict_quality = st.checkbox("Strict Quality", value=False, key="d_strict")
        with col_vid3:
            normalize_res = st.checkbox("Normalize (1080p)", value=True, key="d_norm")
        with col_vid4:
            max_size_mb = st.number_input("Max Size (MB)", value=100, min_value=1, key="d_max_size")
        with col_vid5:
            max_workers = st.slider("Max Concurrent", 1, 10, 3, key="d_mw")
            
        c_dl1, c_dl2 = st.columns(2)
        if c_dl1.button("Start Downloading Top Candidates"):
            if st.session_state.dm.max_workers != max_workers:
                st.session_state.dm = DownloadManager(max_workers=max_workers)
            
            st.session_state.dm.clear_and_reset()
            
            added_count = 0
            for shot in st.session_state.director_shots:
                results = shot.get('video_results', [])
                if not results:
                    continue
                
                # Logic to find the first candidate that fits size/resolution
                # Since size check is async in DM, we'll just add the top candidate
                # but we'll improve this to at least filter by resolution locally if possible
                
                res = results[0]
                # If strict quality is on, we can filter locally for stock APIs
                if strict_quality and res.get('source') in ['pexels', 'pixabay']:
                    # Simple resolution check: 1080p -> height >= 1080
                    res_map = {'1080p': 1080, '720p': 720, '480p': 480}
                    min_h = res_map.get(video_quality, 0)
                    if res.get('height') and res.get('height') < min_h:
                        # Try to find another candidate in results
                        found = False
                        for alt_res in results[1:]:
                            if alt_res.get('height') and alt_res.get('height') >= min_h:
                                res = alt_res
                                found = True
                                break
                        if not found:
                            continue # Skip this shot if no candidates meet resolution

                url = res.get('url')
                if not url:
                    continue
                    
                source = res.get('source', 'unknown')
                slot_id = shot.get('slot_id', 'X')
                
                # Sanitize intent for filename
                intent_raw = shot.get('shot_intent', 'shot')
                safe_intent = "".join([c if c.isalnum() or c in " -_" else "_" for c in intent_raw]).strip()
                
                filename = f"Shot{slot_id}-{source}-{safe_intent[:30]}.mp4"
                output_path = os.path.join("downloads", "director", filename)
                
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
                    st.error(f"Failed to add download for Shot {slot_id}: {e}")
            
            if added_count > 0:
                st.success(f"Added {added_count} downloads to the queue!")
            else:
                st.warning("No downloadable videos found in the top candidates.")

        if c_dl2.button("Cancel All Downloads", key="d_cancel"):
            st.session_state.dm.cancel_all()
            
        # Render Download Tasks (Dashboard View)
        tasks = st.session_state.dm.get_all_tasks()
        if tasks:
            stats = st.session_state.dm.get_stats()
            total = stats['total']
            completed = stats['completed']
            
            st.progress(completed / total if total > 0 else 0.0)
            st.write(f"**Overall Progress:** {completed} / {total} Completed | Active: {stats['downloading']} | Queued: {stats['queued']} | Failed: {stats['error']}")
            
            failed_tasks = st.session_state.dm.get_failed_tasks()
            if failed_tasks:
                with st.expander(f"⚠️ {len(failed_tasks)} Failed Downloads"):
                    if st.button("Retry All Failed", key="d_retry"):
                        st.session_state.dm.retry_all_failed()
                        st.rerun()
                    for ft in failed_tasks:
                        st.write(f"❌ {os.path.basename(ft['output_path'])} - {ft.get('error_msg', 'Unknown Error')}")
                        
            active_tasks = st.session_state.dm.get_active_tasks()
            if active_tasks:
                st.subheader("Currently Downloading")
                for t in active_tasks:
                    st.write(f"**{os.path.basename(t['output_path'])}** - {t['status'].title()} ({t['progress']*100:.1f}%)")
                    st.progress(t['progress'])
                    
                    b1, b2, b3 = st.columns(3)
                    if t['status'] == 'downloading':
                        if b1.button("Pause", key=f"pd_{t['id']}"):
                            st.session_state.dm.pause_download(t['id'])
                            st.rerun()
                    elif t['status'] == 'paused':
                        if b1.button("Resume", key=f"rd_{t['id']}"):
                            st.session_state.dm.resume_download(t['id'])
                            st.rerun()
                            
                    if b2.button("Cancel", key=f"cd_{t['id']}"):
                        st.session_state.dm.cancel_download(t['id'])
                        st.rerun()
                        
                import time
                time.sleep(1)
                st.rerun()
            elif stats['queued'] > 0:
                import time
                time.sleep(1)
                st.rerun()
            
        st.header("Step 5: Output & Download")
        c1, c2, c3 = st.columns(3)
        
        shot_list_txt = generate_shot_list_txt(st.session_state.director_shots)
        shot_list_json = json.dumps(st.session_state.director_shots, indent=2)
        fcpxml = generate_fcpxml(st.session_state.director_shots)
        srt_txt = generate_srt(st.session_state.director_shots) # reuse standard srt
        
        with c1:
            st.download_button("Download shot_list.json", data=shot_list_json, file_name="shot_list.json", mime="application/json")
        with c2:
            st.download_button("Download shot_list.txt", data=shot_list_txt, file_name="shot_list.txt", mime="text/plain")
        with c3:
            st.download_button("Download markers.fcpxml", data=fcpxml, file_name="markers.fcpxml", mime="text/xml")
            
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("shot_list.json", shot_list_json)
            zip_file.writestr("shot_list.txt", shot_list_txt)
            zip_file.writestr("markers.fcpxml", fcpxml)
            zip_file.writestr("timing.srt", srt_txt)
            
        st.download_button(
            label="Download All (.zip)",
            data=zip_buffer.getvalue(),
            file_name="director_outputs.zip",
            mime="application/zip"
        )
