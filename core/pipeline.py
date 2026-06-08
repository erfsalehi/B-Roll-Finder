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


# Hard ceiling on download resolution — we never pull above 1080p.
MAX_DOWNLOAD_HEIGHT = 1080


def cap_quality(quality) -> str:
    """Clamp a requested quality to the 1080p ceiling. Returns a digit string
    yt-dlp / the direct downloader understand (e.g. '1080')."""
    digits = "".join(ch for ch in str(quality) if ch.isdigit())
    h = int(digits) if digits else MAX_DOWNLOAD_HEIGHT
    return str(min(h, MAX_DOWNLOAD_HEIGHT))


def _is_short(c: dict) -> bool:
    """A candidate is a YouTube Short if flagged, URL-marked, or <=60s."""
    if c.get("is_short"):
        return True
    url = (c.get("url") or c.get("page_url") or "").lower()
    if "/shorts/" in url:
        return True
    dur = c.get("duration")
    try:
        # Only treat a known-short duration as a Short for YouTube sources; stock
        # clips are legitimately short and must not be dropped.
        if dur is not None and float(dur) <= 60 and \
                (c.get("source") or "").lower() == "youtube":
            return True
    except (TypeError, ValueError):
        pass
    return False


def drop_shorts(shots: list) -> int:
    """Remove YouTube Shorts from every shot's candidate list (in place).
    Returns how many candidates were dropped."""
    removed = 0
    for s in shots:
        cands = s.get("video_results") or []
        kept = [c for c in cands if not _is_short(c)]
        removed += len(cands) - len(kept)
        s["video_results"] = kept
    return removed


# Candidates longer than this (e.g. full films / 24-7 livestreams) are never
# good B-roll and waste the download cap — drop them before ranking. YouTube gets
# a tighter cap than stock (stock clips are short anyway), since long YouTube
# videos are usually full uploads rather than clean B-roll.
MAX_CLIP_SECONDS = float(os.getenv("MAX_CLIP_SECONDS", "10000") or 10000)
YT_MAX_CLIP_SECONDS = float(os.getenv("YT_MAX_CLIP_SECONDS", "1000") or 1000)


def drop_long_videos(shots: list, max_seconds: float = MAX_CLIP_SECONDS,
                     yt_max_seconds: float = YT_MAX_CLIP_SECONDS) -> int:
    """Remove candidates whose duration exceeds the limit for their source
    (YouTube → ``yt_max_seconds``, everything else → ``max_seconds``), in place.
    Candidates with an unknown duration are kept (we can't judge them)."""
    removed = 0
    for s in shots:
        cands = s.get("video_results") or []
        kept = []
        for c in cands:
            try:
                dur = float(c.get("duration") or 0)
            except (TypeError, ValueError):
                dur = 0.0
            is_yt = (c.get("source") or c.get("original_source") or "").lower() == "youtube"
            limit = yt_max_seconds if is_yt else max_seconds
            if dur and dur > limit:
                removed += 1
            else:
                kept.append(c)
        s["video_results"] = kept
    return removed


def drop_vertical(shots: list) -> int:
    """Keep only horizontal (landscape) candidates — drop portrait clips whose
    height exceeds width (in place). Candidates with unknown dimensions are kept
    (we can't judge them)."""
    removed = 0
    for s in shots:
        cands = s.get("video_results") or []
        kept = []
        for c in cands:
            try:
                w = int(c.get("width") or 0)
                h = int(c.get("height") or 0)
            except (TypeError, ValueError):
                w = h = 0
            if w and h and h > w:   # portrait/vertical
                removed += 1
            else:
                kept.append(c)
        s["video_results"] = kept
    return removed


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
                # Record the downloaded clip in the Clip Library so the server
                # accumulates reusable, semantically-searchable footage over time
                # (the bot pipeline previously only read the library, never wrote
                # to it). Dedupes by URL; best-effort so it can't fail a download.
                try:
                    from core.clip_library import store_clip
                    store_clip(
                        (shot.get("shot_intent") or shot.get("shot_description")
                         or res.get("matched_query") or "").strip(),
                        {**res, "local_path": out_path},
                        project=project_name,
                        search_query=res.get("matched_query", ""),
                    )
                except Exception as e:
                    errors.append(f"clip_library store ({fn}): {e}")
            else:
                failed += 1
                errors.append(f"{fn}: {task_state.get('error_msg', 'unknown error')}")
        except Exception as e:
            failed += 1
            errors.append(f"{fn}: {e}")
        if progress:
            progress(i, total)

    return {"ok": ok, "failed": failed, "skipped": skipped, "dir": base_dir, "errors": errors}


def repair_empty_shots(shots: list, groq_key: str = None, video_topic: str = "",
                       errors: list = None, progress=None) -> int:
    """Final repair pass before download: rescue shots that ended up with NO
    selected clip — whether from a failed fetch (connection reset / DNS) or a
    failed shot-list block (the LLM JSON got truncated, leaving a shot with no
    queries). For those shots it regenerates missing queries, re-fetches (with
    its own retry passes), re-injects the Clip Library, re-ranks, and re-runs
    auto-select. Idempotent and safe to call repeatedly. Returns how many shots
    were recovered (now have a selection).
    """
    from core.director import ensure_shot_queries
    from core.director_youtube import seed_youtube_keywords
    from core.director_search import fetch_with_retries
    from core.director_rank import rank_shot_candidates, auto_select_top_candidates

    key = groq_key or os.getenv("GROQ_API_KEY")
    targets = [s for s in shots if s.get("priority") != "none" and not s.get("selected_results")]
    if not targets:
        return 0
    still_empty_before = len(targets)
    if progress:
        progress(f"Repairing {still_empty_before} empty shot(s)…")

    # Shots from a truncated director block have no queries — synthesize them.
    ensure_shot_queries(targets, video_topic)
    try:
        seed_youtube_keywords(targets)
    except Exception:
        pass

    if errors is None:
        errors = []
    fetch_with_retries(targets, errors=errors)   # re-fetch just the empties

    try:
        from core.clip_library import get_library_stats, inject_library_candidates
        if _flag_default("AUTO_USE_LIBRARY", True) and get_library_stats().get("total", 0):
            inject_library_candidates(targets, top_k=int(os.getenv("AUTO_LIBRARY_NUM", "5") or 5))
    except Exception:
        pass

    try:
        rank_shot_candidates(targets, api_key=key, video_topic=video_topic)
    except Exception as e:
        errors.append(f"repair rank: {e}")

    auto_select_top_candidates(shots)   # full list → only fills the still-empty ones
    return still_empty_before - sum(1 for s in targets if not s.get("selected_results"))


def fill_empty_shots(shots: list, groq_key: str = None, video_topic: str = "",
                     errors: list = None, progress=None, passes: int = 2) -> int:
    """Aggressively get a clip onto EVERY shot that has none.

    Runs :func:`repair_empty_shots` up to ``passes`` times (regenerate queries →
    re-fetch → re-rank → re-select), and after each pass drops Shorts, purges any
    Short that slipped into a selection, biases YouTube to the front, and
    re-selects so a Short-free YouTube clip is preferred. Returns the number of
    shots newly filled across all passes.
    """
    from core.director_rank import auto_select_top_candidates, prioritize_youtube

    key = groq_key or os.getenv("GROQ_API_KEY")
    if errors is None:
        errors = []
    filled = 0
    for p in range(max(1, passes)):
        empties = [s for s in shots
                   if s.get("priority") != "none" and not s.get("selected_results")]
        if not empties:
            break
        if progress:
            progress(f"Fill pass {p + 1}: {len(empties)} empty shot(s)…")
        before = len(empties)
        repair_empty_shots(shots, groq_key=key, video_topic=video_topic,
                           errors=errors, progress=progress)
        drop_shorts(shots)
        # A Short may already sit in a selection (repair selects before we drop) —
        # purge it so the slot re-fills with a non-Short clip.
        for s in shots:
            sel = s.get("selected_results") or []
            cleaned = [c for c in sel if not _is_short(c)]
            if len(cleaned) != len(sel):
                if cleaned:
                    s["selected_results"] = cleaned
                else:
                    s.pop("selected_results", None)
                    s.pop("auto_selected", None)
        prioritize_youtube(shots)
        auto_select_top_candidates(shots)
        still = sum(1 for s in shots
                    if s.get("priority") != "none" and not s.get("selected_results"))
        filled += before - still
        if still == 0 or before == still:   # done, or no further progress
            break
    return filled


def refine_flagged_shots(shots: list, qa: dict, groq_key: str = None, video_topic: str = "",
                         errors: list = None, progress=None,
                         severities=("high", "medium"), only_slots=None) -> int:
    """QA-driven re-pick: for each shot the Step 5.5 review flagged, regenerate
    better search queries from the reviewer's suggestion, re-fetch, re-rank, and
    re-select — so the timeline self-corrects the problems the QA pass found.

    By default touches shots whose issue ``severity`` is in ``severities``. Pass
    ``only_slots`` (an iterable of slot_ids) to refine exactly those shots instead
    — even ones the QA pass didn't flag (used by the bot's ``/refine 4 9``).
    Returns how many targeted shots ended up with a (re)selection.
    """
    from core.director import regenerate_shot_queries
    from core.director_youtube import seed_youtube_keywords
    from core.director_search import fetch_with_retries, filter_youtube_sd_candidates
    from core.director_rank import rank_shot_candidates, auto_select_top_candidates, prioritize_youtube

    key = groq_key or os.getenv("GROQ_API_KEY")
    by_slot: dict = {}
    for it in (qa.get("issues") or []):
        if it.get("slot_id") is None:
            continue
        if only_slots is not None:
            if it["slot_id"] in only_slots:
                by_slot.setdefault(it["slot_id"], []).append(it)
        elif it.get("severity") in severities:
            by_slot.setdefault(it["slot_id"], []).append(it)
    if only_slots is not None:
        # Include explicitly-named slots even if QA didn't flag them.
        for sid in only_slots:
            by_slot.setdefault(sid, [])
    if not by_slot:
        return 0
    if errors is None:
        errors = []

    targets = [s for s in shots if s.get("slot_id") in by_slot and s.get("priority") != "none"]
    if not targets:
        return 0
    if progress:
        progress(f"Refining {len(targets)} flagged shot(s)…")

    # Regenerate queries per flagged shot, guided by that shot's QA feedback, then
    # clear its old candidates/selection so it gets a fresh fetch + pick.
    for s in targets:
        sid = s.get("slot_id")
        notes = "; ".join(
            f"{i.get('problem', '')} → {i.get('suggestion', '')}".strip(" →")
            for i in by_slot[sid]
        ) or "Find a more relevant, higher-quality clip than the current pick."
        try:
            regenerate_shot_queries(
                shots, {sid}, api_key=key, video_topic=video_topic,
                custom_instructions=f"QA feedback to fix for this shot: {notes}",
            )
        except Exception as e:
            errors.append(f"refine regen slot {sid}: {e}")
        s["video_results"] = []
        s.pop("selected_results", None)
        s.pop("auto_selected", None)

    try:
        seed_youtube_keywords(targets)
    except Exception:
        pass
    fetch_with_retries(targets, errors=errors)

    if os.getenv("YOUTUBE_API_KEY"):
        try:
            filter_youtube_sd_candidates(targets, api_key=os.getenv("YOUTUBE_API_KEY"))
        except Exception as e:
            errors.append(f"refine hd_filter: {e}")

    drop_shorts(targets)   # Shorts never allowed

    if _flag_default("AUTO_USE_LIBRARY", True):
        try:
            from core.clip_library import get_library_stats, inject_library_candidates
            if get_library_stats().get("total", 0):
                inject_library_candidates(targets, top_k=int(os.getenv("AUTO_LIBRARY_NUM", "5") or 5))
        except Exception as e:
            errors.append(f"refine clip_library: {e}")

    try:
        rank_shot_candidates(targets, api_key=key, video_topic=video_topic)
    except Exception as e:
        errors.append(f"refine rank: {e}")

    prioritize_youtube(targets)         # keep the ~70% YouTube bias on refresh
    auto_select_top_candidates(shots)   # fills the now-empty refreshed shots
    return sum(1 for s in targets if s.get("selected_results"))


def write_fcpxml(shots: list, project_name: str) -> str:
    """Render the Premiere FCPXML for ``shots`` and write it to the project dir.
    Returns the absolute XML path."""
    xml = generate_fcpxml(shots, project_name=project_name)
    proj = _safe_for_fs(project_name, 50)
    xml_path = os.path.join(os.path.abspath("downloads"), proj, f"{proj}.xml")
    os.makedirs(os.path.dirname(xml_path), exist_ok=True)
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml)
    return xml_path


def finalize_project(shots: list, project_name: str, quality: str = "1080",
                     should_cancel=None, progress=None) -> dict:
    """Download the selected clips and (re)write the FCPXML — the back half of the
    pipeline, split out so the bot's review gate can run it after the user
    approves (or after a /refine). Returns ``{download, xml_path}``."""
    dl = download_selected_clips(shots, project_name, quality=cap_quality(quality),
                                 progress=progress, should_cancel=should_cancel)
    return {"download": dl, "xml_path": write_fcpxml(shots, project_name)}


def run_pipeline_headless(audio_path: str, groq_key: str = None, project_name: str = "auto",
                          download: bool = True, context_aware: bool = None,
                          progress_callback=None, should_cancel=None, run_qa: bool = None,
                          auto_refine: bool = None, quality: str = "1080",
                          auto_fill: bool = None) -> dict:
    """Drive transcribe → topic → (context pre-pass) → shot list → fetch → HD
    filter → rank → auto-select → QA review → [auto-refine] → [download] → FCPXML.

    ``progress_callback(step, total, label)`` is invoked at each stage. Returns a
    dict with ``project_name, topic, shots, n_shots, n_selected, n_clips, qa,
    xml_path, download, errors``. Raises only on the hard failures that make
    continuing pointless (no transcription / no shots / no key).

    ``auto_refine`` (default env ``AUTO_REFINE``, off) re-picks QA-flagged shots
    before download. ``quality`` is the download resolution passed to yt-dlp.
    """
    from core.transcription import transcribe_audio
    from core.director import generate_shot_list_from_transcription, segment_script_structure
    from core.keywords import generate_video_topic
    from core.director_search import (fetch_with_retries, filter_youtube_sd_candidates,
                                      auto_fetch_plan)
    from core.director_youtube import seed_youtube_keywords
    from core.director_rank import (rank_shot_candidates, auto_select_top_candidates,
                                     review_timeline, prioritize_youtube)

    key = groq_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is required.")
    if context_aware is None:
        context_aware = _flag("ENABLE_CONTEXT_AWARE_KEYWORDS")
    if run_qa is None:
        # Default on; set ENABLE_QA_REVIEW=false to skip the Step 5.5 review.
        run_qa = os.getenv("ENABLE_QA_REVIEW", "true").strip().lower() in ("1", "true", "yes", "on")
    if auto_refine is None:
        auto_refine = _flag_default("AUTO_REFINE", False)
    if auto_fill is None:
        auto_fill = _flag_default("AUTO_FILL", True)

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

    def _fetch_progress(frac):
        # Re-emit step 5 with a live percentage so the bot/UI can update the
        # "Fetching candidates" line in place instead of leaving it static for
        # the whole (often longest) phase.
        _p(5, f"Fetching candidates · {int(max(0.0, min(1.0, frac)) * 100)}%")

    fetch_with_retries(shots, errors=errors, should_cancel=should_cancel,
                       progress_callback=_fetch_progress)  # cancellable

    # Clip Library: add previously-downloaded footage as candidates (free).
    if _flag_default("AUTO_USE_LIBRARY", True):
        try:
            from core.clip_library import get_library_stats, inject_library_candidates
            if get_library_stats().get("total", 0):
                _p(5, "Searching Clip Library")
                inject_library_candidates(shots, top_k=int(os.getenv("AUTO_LIBRARY_NUM", "5") or 5))
        except Exception as e:
            errors.append(f"clip_library: {e}")

    # 6 — HD filter (YouTube Data API is used ONLY here — to drop SD clips; the
    # YouTube *search* itself is done with yt-dlp + keywords, no Data API quota).
    if os.getenv("YOUTUBE_API_KEY"):
        _p(6, "Dropping SD YouTube clips")
        try:
            filter_youtube_sd_candidates(shots, api_key=os.getenv("YOUTUBE_API_KEY"))
        except Exception as e:
            errors.append(f"hd_filter: {e}")

    # 6b — Candidate filters: no Shorts, no >max-duration clips, landscape only.
    drop_shorts(shots)
    drop_long_videos(shots)
    drop_vertical(shots)

    # 7 — Rank
    _p(7, "Ranking candidates")
    try:
        rank_shot_candidates(shots, api_key=key, video_topic=topic)
    except Exception as e:
        errors.append(f"rank: {e}")

    # 7b — Bias selection toward YouTube (~70% of shots lead with a YouTube clip).
    prioritize_youtube(shots)

    # 8 — Auto-select
    _p(8, "Auto-selecting clips")
    auto_select_top_candidates(shots)

    # 8b — Rescue any shots still empty (failed fetch / bad block). auto_fill
    # (default on) runs the aggressive multi-pass fill so nearly every shot ends
    # with a clip; otherwise a single lighter repair pass.
    if auto_fill:
        _p(8, "Filling empty shots")
        fill_empty_shots(shots, groq_key=key, video_topic=topic, errors=errors,
                         progress=lambda lbl: _p(8, lbl))
    else:
        _p(8, "Repairing empty shots")
        repair_empty_shots(shots, groq_key=key, video_topic=topic, errors=errors)

    # 9 — QA review (optional)
    if run_qa:
        _p(9, "Final QA review")
        qa = review_timeline(shots, api_key=key, video_topic=topic)
        # 9b — auto-refine flagged shots, then re-review so the report reflects
        # the fixes. Off by default; the bot turns it on per-job.
        if auto_refine and qa.get("issues"):
            _p(9, f"Refining {len(qa['issues'])} flagged shot(s)")
            n_ref = refine_flagged_shots(shots, qa, groq_key=key, video_topic=topic,
                                         errors=errors)
            if n_ref:
                _p(9, "Re-reviewing refined timeline")
                qa = review_timeline(shots, api_key=key, video_topic=topic)
                qa["refined"] = n_ref
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

    # 10 — Download (optional) — capped at 1080p.
    if download:
        _p(10, "Downloading clips")
        result["download"] = download_selected_clips(shots, project_name,
                                                     quality=cap_quality(quality),
                                                     should_cancel=should_cancel)

    # Final — FCPXML
    _p(total, "Writing Premiere XML")
    result["xml_path"] = write_fcpxml(shots, project_name)
    return result
