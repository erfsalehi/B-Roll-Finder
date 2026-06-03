"""Headless end-to-end Director pipeline — no Streamlit.

Runs the same stages the Streamlit "Fully Automatic Mode" runs, but as plain
Python so it can be driven by automation (e.g. the Telegram bot) on an
always-on machine. Returns a structured result and, optionally, downloads the
selected clips and writes the Premiere FCPXML into the project folder.

    from core.pipeline import run_pipeline_headless
    result = run_pipeline_headless("voice.mp3", project_name="my_video")
"""

import os

from core.output import clip_base_dir, clip_filename, generate_fcpxml, _safe_for_fs


class PipelineCancelled(Exception):
    """Raised when a caller's ``should_cancel()`` asks the run to stop."""


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _flag_default(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _check_cancel(should_cancel) -> None:
    if should_cancel and should_cancel():
        raise PipelineCancelled("Cancelled by user.")


def download_selected_clips(shots: list, project_name: str, quality: str = "1080",
                            progress=None, should_cancel=None) -> dict:
    """Download every selected clip into the project's clip folder, using the
    exact filenames :func:`generate_fcpxml` will reference.

    Routes by source: YouTube clips via yt-dlp (``core.youtube.download_video``),
    everything else (direct stock MP4s) via ``download_direct_video``. Returns
    ``{ok, failed, skipped, dir, errors}``. ``progress(done, total)`` is called
    as clips complete.
    """
    from core.direct_downloader import download_direct_video
    from core.youtube import download_video

    base_dir = clip_base_dir(project_name)
    os.makedirs(base_dir, exist_ok=True)

    jobs = []  # (shot, res, filename)
    seen = set()
    for shot in shots:
        if shot.get("priority") == "none":
            continue
        for idx, res in enumerate(shot.get("selected_results") or []):
            if not res.get("url"):
                continue
            fn = clip_filename(shot.get("slot_id", "X"), idx + 1,
                               res.get("matched_query", ""), seen)
            jobs.append((shot, res, fn))

    ok = failed = skipped = 0
    errors = []
    total = len(jobs)
    for i, (shot, res, fn) in enumerate(jobs, start=1):
        _check_cancel(should_cancel)
        out_path = os.path.join(base_dir, fn)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            skipped += 1
            if progress:
                progress(i, total)
            continue
        task_state: dict = {}
        try:
            if (res.get("source") or "").lower() == "youtube":
                download_video(res["url"], out_path, quality, task_state, no_audio=True)
            else:
                download_direct_video(res["url"], out_path, task_state)
            if task_state.get("status") == "completed" or (
                os.path.exists(out_path) and os.path.getsize(out_path) > 0
            ):
                ok += 1
            else:
                failed += 1
                errors.append(f"{fn}: {task_state.get('error_msg', 'unknown error')}")
        except Exception as e:
            failed += 1
            errors.append(f"{fn}: {e}")
        if progress:
            progress(i, total)

    return {"ok": ok, "failed": failed, "skipped": skipped, "dir": base_dir, "errors": errors}


def run_pipeline_headless(audio_path: str, groq_key: str = None, project_name: str = "auto",
                          download: bool = True, context_aware: bool = None,
                          progress_callback=None, should_cancel=None, run_qa: bool = None) -> dict:
    """Drive transcribe → topic → (context pre-pass) → shot list → fetch → HD
    filter → rank → auto-select → QA review → [download] → FCPXML, headless.

    ``progress_callback(step, total, label)`` is invoked at each stage. Returns a
    dict with ``project_name, topic, shots, n_shots, n_selected, n_clips, qa,
    xml_path, download, errors``. Raises only on the hard failures that make
    continuing pointless (no transcription / no shots / no key).
    """
    from core.transcription import transcribe_audio
    from core.director import generate_shot_list_from_transcription, segment_script_structure
    from core.keywords import generate_video_topic
    from core.director_search import (fetch_with_retries, filter_youtube_sd_candidates,
                                      auto_fetch_plan)
    from core.director_youtube import seed_youtube_keywords
    from core.director_rank import rank_shot_candidates, auto_select_top_candidates, review_timeline

    key = groq_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is required.")
    if context_aware is None:
        context_aware = _flag("ENABLE_CONTEXT_AWARE_KEYWORDS")
    if run_qa is None:
        # Default on; set ENABLE_QA_REVIEW=false to skip the Step 5.5 review.
        run_qa = os.getenv("ENABLE_QA_REVIEW", "true").strip().lower() in ("1", "true", "yes", "on")

    total = 11 if download else 10
    errors = []

    def _p(step, label):
        _check_cancel(should_cancel)   # cooperative cancel at every stage boundary
        if progress_callback:
            try:
                progress_callback(step, total, label)
            except Exception:
                pass

    # 1 — Transcribe
    _p(1, "Transcribing voiceover")
    segments = transcribe_audio(audio_path, key)
    if not segments:
        raise RuntimeError("Transcription produced no segments.")
    script_text = " ".join(s["text"].strip() for s in segments).strip()

    # 2 — Topic
    _p(2, "Analyzing topic")
    topic = ""
    try:
        topic = generate_video_topic(script_text, key) or ""
    except Exception as e:
        errors.append(f"topic: {e}")

    # 3 — Optional context-aware pre-pass
    roadmap = {}
    if context_aware:
        _p(3, "Mapping video structure")
        try:
            roadmap = segment_script_structure(segments, key, video_topic=topic)
        except Exception as e:
            errors.append(f"segmenter: {e}")

    # 4 — Shot list
    _p(4, "Generating shot list")
    shots = generate_shot_list_from_transcription(
        segments, key, video_topic=topic, segment_roadmap=roadmap or None,
    )
    for i, s in enumerate(shots):
        s["slot_id"] = i + 1
        s.pop("chunk_id", None)
    if not shots:
        raise RuntimeError("No shots were generated.")

    # 5 — Fetch
    _p(5, "Fetching candidates")
    try:
        seed_youtube_keywords(shots)
    except Exception:
        pass
    fetch_with_retries(shots, errors=errors)  # retries empty shots on flaky links

    # Clip Library: add previously-downloaded footage as candidates (free).
    if _flag_default("AUTO_USE_LIBRARY", True):
        try:
            from core.clip_library import get_library_stats, inject_library_candidates
            if get_library_stats().get("total", 0):
                _p(5, "Searching Clip Library")
                inject_library_candidates(shots, top_k=int(os.getenv("AUTO_LIBRARY_NUM", "5") or 5))
        except Exception as e:
            errors.append(f"clip_library: {e}")

    # 6 — HD filter
    if os.getenv("YOUTUBE_API_KEY"):
        _p(6, "Dropping SD YouTube clips")
        try:
            filter_youtube_sd_candidates(shots, api_key=os.getenv("YOUTUBE_API_KEY"))
        except Exception as e:
            errors.append(f"hd_filter: {e}")

    # 7 — Rank
    _p(7, "Ranking candidates")
    try:
        rank_shot_candidates(shots, api_key=key, video_topic=topic)
    except Exception as e:
        errors.append(f"rank: {e}")

    # 8 — Auto-select
    _p(8, "Auto-selecting clips")
    auto_select_top_candidates(shots)

    # 9 — QA review (optional)
    if run_qa:
        _p(9, "Final QA review")
        qa = review_timeline(shots, api_key=key, video_topic=topic)
    else:
        qa = {"overall": "QA review skipped.", "issues": []}

    result = {
        "project_name": project_name,
        "topic": topic,
        "shots": shots,
        "n_shots": len(shots),
        "n_selected": sum(1 for s in shots if s.get("selected_results")),
        "n_clips": sum(len(s.get("selected_results") or []) for s in shots),
        "qa": qa,
        "errors": errors,
        "download": None,
        "xml_path": None,
    }

    # 10 — Download (optional)
    if download:
        _p(10, "Downloading clips")
        result["download"] = download_selected_clips(shots, project_name, should_cancel=should_cancel)

    # Final — FCPXML
    _p(total, "Writing Premiere XML")
    xml = generate_fcpxml(shots, project_name=project_name)
    proj = _safe_for_fs(project_name, 50)
    xml_path = os.path.join(os.path.abspath("downloads"), proj, f"{proj}.xml")
    os.makedirs(os.path.dirname(xml_path), exist_ok=True)
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml)
    result["xml_path"] = xml_path
    return result
