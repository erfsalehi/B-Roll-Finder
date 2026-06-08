import os
from pathlib import Path
from xml.sax.saxutils import escape
from core.ffmpeg_utils import get_video_metadata


def _xml_attr(value) -> str:
    return escape(str(value), {'"': "&quot;"})


def _get_premiere_safe_pathurl(filepath: str) -> str:
    r"""
    Formats paths to prevent Premiere's Windows \\C:\ network drive bug.
    Forces the file://localhost/C:/... format.
    """
    abs_path = os.path.abspath(filepath)
    # Convert backslashes to forward slashes for XML URIs
    forward_slash_path = abs_path.replace('\\', '/')
    
    # Ensure it starts with a slash before the drive letter
    if not forward_slash_path.startswith('/'):
        forward_slash_path = '/' + forward_slash_path
        
    return f"file://localhost{forward_slash_path}"


def sec_to_frames(seconds: float, fps: float = 23.976) -> int:
    """Converts seconds to exact frame counts for Premiere Pro."""
    return int(round(seconds * fps))


def _get_media_duration(path: str, fallback_duration: float = 3600.0) -> float:
    """Helper to get duration using our ffprobe utility."""
    meta = get_video_metadata(path)
    # If get_video_metadata returned the default 3600 but we have a better fallback, use it.
    if meta['duration'] == 3600.0 and fallback_duration != 3600.0:
        return fallback_duration
    return meta['duration']


def format_time(seconds) -> str:
    # Shots now carry float timestamps; floor to whole seconds for display.
    total = int(float(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"[{h:02d}:{m:02d}:{s:02d}]"

def format_srt_time(seconds: float) -> str:
    millis = int((seconds % 1) * 1000)
    total_seconds = int(seconds)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"

def generate_keywords_txt(slots: list) -> str:
    lines = []
    for slot in slots:
        ts_str = format_time(slot['timestamp'])
        script_text = slot.get('text', '')
        lines.append(f'{ts_str} Script: "{script_text}"')
        
        keywords = slot.get('keywords', [])
        for i, kw in enumerate(keywords, 1):
            lines.append(f"  {i}. {kw}")
        lines.append("") # Empty line separator
    return "\n".join(lines)

def generate_youtube_txt(slots: list) -> str:
    lines = []
    for slot in slots:
        results = slot.get('video_results', slot.get('youtube_results', []))
        if not results:
            continue
            
        ts_str = format_time(slot['timestamp'])
        primary_kw = slot.get('keywords', [''])[0]
        lines.append(f"{ts_str} {primary_kw}")
        
        for res in results:
            source = res.get('source', 'youtube').upper()
            lines.append(f"[{source}] {res['title']} | {res['url']}")
        lines.append("") # Empty line separator
    return "\n".join(lines)

def generate_srt(slots: list) -> str:
    lines = []
    for i, slot in enumerate(slots, 1):
        start_ts = format_srt_time(slot['timestamp'])
        
        # Close gaps: Use next slot's start time if available
        if i < len(slots):
            end_val = slots[i]['timestamp']
        else:
            end_val = slot.get('end_timestamp', slot['timestamp'] + 1)
            
        end_ts = format_srt_time(end_val)
        
        primary_kw = "No keyword"
        keywords = slot.get('keywords', [])
        if keywords:
            primary_kw = keywords[0]
            
        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(primary_kw)
        lines.append("") # Empty line separator
    return "\n".join(lines)

def generate_transcription_srt(segments: list) -> str:
    """Generates an SRT from Whisper segments, closing gaps between segments."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start_ts = format_srt_time(seg['start'])
        
        # Close gaps: Use next segment's start time if available
        if i < len(segments):
            end_val = segments[i]['start']
        else:
            end_val = seg['end']
            
        end_ts = format_srt_time(end_val)
        text = seg['text'].strip()
        
        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("") # Empty line separator
    return "\n".join(lines)

def generate_shots_srt(shots: list) -> str:
    """SRT mirror of the shot list, for sanity-checking timing against audio.

    Each cue spans from the shot's own start to the *next* shot's start, so
    any silence after a shot's voice ends is rolled into that shot's cue.
    The final shot ends at its own ``end_timestamp``. Drop this SRT onto the
    audio in any player and you should see "Shot N" appear and stay on
    screen for the entire run of that shot — including the trailing pause —
    until "Shot N+1" replaces it the moment the next line of voice begins.
    """
    lines = []
    sorted_shots = sorted(shots, key=lambda s: float(s.get('timestamp', 0)))
    for i, shot in enumerate(sorted_shots, 1):
        start_val = float(shot.get('timestamp', 0))

        if i < len(sorted_shots):
            end_val = float(sorted_shots[i].get('timestamp', start_val))
        else:
            end_val = float(shot.get('end_timestamp', start_val + 1.0))

        # Guard against zero/negative spans so the cue is always visible in a
        # player; one second is enough to read the label.
        if end_val <= start_val:
            end_val = start_val + 1.0

        slot_id = shot.get('slot_id', i)
        lines.append(str(i))
        lines.append(f"{format_srt_time(start_val)} --> {format_srt_time(end_val)}")
        lines.append(f"Shot {slot_id}")
        lines.append("")

    return "\n".join(lines)


def generate_failed_downloads_txt(failed_tasks: list) -> str:
    """Generates a text file listing failed downloads."""
    lines = []
    for task in failed_tasks:
        # Format: title (1-1 youtube - keyword) + link
        filename = os.path.basename(task['output_path'])
        url = task.get('url', 'No URL')
        lines.append(f"{filename} | {url}")
    return "\n".join(lines)

import json

def generate_shot_list_txt(shots: list) -> str:
    lines = []
    for shot in shots:
        start = shot.get('timestamp_start_str', '00:00:00')
        end = shot.get('timestamp_end_str', '00:00:00')
        priority = shot.get('priority', 'medium').upper()
        shot_type = shot.get('shot_type', 'medium').upper()
        
        lines.append(f"[{start} - {end}]  PRIORITY: {priority}  SHOT: {shot_type}")
        lines.append(f"Script:  {shot.get('text', '')}")
        lines.append(f"Intent:  {shot.get('shot_intent', '')}")
        lines.append(f"Queries: {' | '.join(shot.get('search_queries', []))}")
        
        results = shot.get('video_results', [])
        if results:
            lines.append("Candidates:")
            for idx, res in enumerate(results):
                source = res.get('source', 'unknown').capitalize()
                lines.append(f"  {idx+1}. [{source}] {res.get('title', 'Video')} - {res.get('url', '')}")
        lines.append("")
    return "\n".join(lines)

def filter_overlays_for_shots(shots: list, overlays: list) -> list:
    """Returns only those overlays that fall within the time range of the given shots."""
    if not shots or not overlays:
        return []
    
    # Sort shots by timestamp to be safe
    sorted_shots = sorted(shots, key=lambda x: float(x.get('timestamp', 0)))
    start_time = float(sorted_shots[0].get('timestamp', 0))
    
    # For the end time, we look at the last shot's end point
    last_shot = sorted_shots[-1]
    last_start = float(last_shot.get('timestamp', 0))
    # Approximate end if not explicit
    end_time = float(last_shot.get('end_timestamp', last_start + 5.0))
    
    filtered = []
    for ov in overlays:
        ov_start = float(ov.get('start_sec', 0))
        # If the overlay starts within this shot block, include it
        if start_time <= ov_start <= end_time:
            filtered.append(ov)
    return filtered

def _safe_for_fs(text: str, max_len: int = 30) -> str:
    if not text:
        return ""
    cleaned = "".join(c if (c.isalnum() or c in " -_") else " "
                      for c in text)
    cleaned = "-".join(cleaned.split()).lower()
    return cleaned[:max_len].strip("-") or ""

def _preferred_in_frame(clip_url: str, filename: str, duration_frames: int,
                        media_dur_frames: int, fps: float) -> int:
    """
    Source in-point (frames) for a clip, honouring any trim learned from a
    re-imported Premiere edit. Returns 0 when there's no learned trim, when
    the trim wouldn't leave room for the timeline slot, or on any error — so
    the export never breaks because of this lookup.
    """
    if not clip_url and not filename:
        return 0
    try:
        from core import clip_library
        row = clip_library.find_clip_by_path_or_url(clip_url=clip_url, filename=filename)
        if not row:
            return 0
        trim = clip_library.get_preferred_trim(row["id"], row.get("shot_description", "") or "")
        if not trim:
            return 0
        in_frame = sec_to_frames(float(trim["in_seconds"]), fps)
        # Keep the slot length: only honour the trim if in + slot fits the media.
        if in_frame <= 0 or in_frame + duration_frames > media_dur_frames:
            return 0
        return in_frame
    except Exception:
        return 0


def clip_base_dir(project_name: str) -> str:
    """Absolute folder where a project's downloaded clips live — and where the
    FCPXML expects to find them. Shared by the exporter and the headless
    downloader so their paths always agree."""
    return os.path.abspath(os.path.join("downloads", _safe_for_fs(project_name, 50), "director"))


def clip_filename(slot_id, footage_num: int, matched_query: str, seen_filenames: set) -> str:
    """Deterministic, deduplicated clip filename — the single source of truth so
    the downloader writes to exactly the path generate_fcpxml references."""
    keyword = _safe_for_fs(matched_query or "", 30) or "clip"
    base = f"{slot_id}-{footage_num}-{keyword}"
    filename = f"{base}.mp4"
    n = 1
    while filename in seen_filenames:
        n += 1
        filename = f"{base}-{n}.mp4"
    seen_filenames.add(filename)
    return filename


def zip_project(project_name: str, out_path: str = None) -> dict:
    """Bundle a project's downloaded clips + FCPXML into one .zip for transfer
    (e.g. download from the server to your editing machine).

    Packages everything under ``downloads/<project>/`` preserving the folder
    layout, so unzipping recreates ``<project>/director/*.mp4`` + the XML. Returns
    ``{path, size_bytes, files}``. Raises FileNotFoundError if the project folder
    doesn't exist yet.
    """
    import zipfile
    proj = _safe_for_fs(project_name, 50)
    proj_dir = os.path.join("downloads", proj)
    if not os.path.isdir(proj_dir):
        raise FileNotFoundError(f"No downloaded project at {proj_dir}")

    out_path = out_path or os.path.join("downloads", f"{proj}.zip")
    files = 0
    # ZIP_DEFLATED barely shrinks already-compressed mp4s but keeps the bundle
    # to a single portable file; allowZip64 handles multi-GB projects.
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for root, _dirs, names in os.walk(proj_dir):
            for name in names:
                fp = os.path.join(root, name)
                if os.path.abspath(fp) == os.path.abspath(out_path):
                    continue  # never zip the zip itself
                z.write(fp, os.path.relpath(fp, "downloads"))
                files += 1
    return {"path": out_path, "size_bytes": os.path.getsize(out_path), "files": files}


def generate_fcpxml(shots: list, project_name: str = "default", overlays: list = None,
                    sfx_list: list = None, time_offset: float = 0.0) -> str:
    """
    Generates a bulletproof Legacy FCP 7 XML (<xmeml>) for Premiere Pro.
    Uses exact frame math and implicit gaps to guarantee compatibility.

    ``time_offset`` (seconds) is subtracted from every shot / overlay / SFX
    start so a chunked export can begin at frame 0 instead of inheriting an
    empty gap from the original timeline. Pass ``shots[0]['timestamp']`` when
    splitting into multiple XML parts; pass 0 for a single-file export.
    """
    proj_folder = _safe_for_fs(project_name, 50)
    # We use absolute paths for the 'pathurl' attribute so Premiere can find them instantly.
    base_dir = clip_base_dir(project_name)
    
    fps_exact = 23.976
    timebase = 24  # Standard timebase for 23.98 in FCP7 XML

    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<!DOCTYPE xmeml>')
    xml.append('<xmeml version="4">')
    xml.append('  <project>')
    xml.append(f'    <name>{_xml_attr(project_name)}</name>')
    xml.append('    <children>')
    
    # ── Sequence Setup ──
    xml.append('      <sequence id="b-roll-seq">')
    xml.append('        <name>B-Roll Sequence</name>')
    xml.append('        <rate>')
    xml.append(f'          <timebase>{timebase}</timebase>')
    xml.append('          <ntsc>TRUE</ntsc>')
    xml.append('        </rate>')
    # Remember this index so the sequence-level <duration>/<timecode> can be
    # injected right before <media> once we know the timeline's total length.
    seq_media_idx = len(xml)
    xml.append('        <media>')
    xml.append('          <video>')
    xml.append('            <format>')
    xml.append('              <samplecharacteristics>')
    # A <rate> inside the format is what fixes the sequence's editing timebase;
    # without it Premiere can't build the timeline (clips land in the bin only).
    xml.append('                <rate>')
    xml.append(f'                  <timebase>{timebase}</timebase>')
    xml.append('                  <ntsc>TRUE</ntsc>')
    xml.append('                </rate>')
    xml.append('                <width>1920</width>')
    xml.append('                <height>1080</height>')
    xml.append('                <anamorphic>FALSE</anamorphic>')
    xml.append('                <pixelaspectratio>square</pixelaspectratio>')
    xml.append('                <fielddominance>none</fielddominance>')
    xml.append('              </samplecharacteristics>')
    xml.append('            </format>')
    xml.append('            <track>')

    asset_map = {}
    next_asset_id = 1
    seen_filenames = set()
    defined_files = set() # To track which files have already been injected
    max_end_frame = 0      # longest clip end across all tracks → sequence duration

    # ── Timeline Math & Placement ──
    for i, shot in enumerate(shots):
        start_sec = float(shot.get('timestamp', 0)) - time_offset

        # Voice end is where the shot's audio actually finishes. Sub-clip
        # cuts inside the shot divide *this* range evenly so visible
        # transitions land on the voice instead of drifting through silences.
        voice_end = float(shot.get('end_timestamp', start_sec + time_offset + 5)) - time_offset

        # next_start: where the next shot's voice begins. Used to clamp a
        # slightly-overlapping LLM answer and — more importantly — to extend
        # the very last sub-clip across the trailing silence so the timeline
        # has no gap before the next shot starts.
        if i < len(shots) - 1:
            next_start = float(shots[i+1].get('timestamp', voice_end + time_offset)) - time_offset
        else:
            next_start = voice_end
        if next_start < voice_end:
            voice_end = next_start

        voice_duration_sec = voice_end - start_sec
        if voice_duration_sec <= 0:
            voice_duration_sec = 1.0

        sel = shot.get("selected_results", [])
        if not sel:
            continue

        # Distribute the voice range evenly across selected candidates.
        num_clips = len(sel)
        clip_duration_sec = voice_duration_sec / num_clips

        for res_idx, res in enumerate(sel):
            url = res.get("url")
            if not url:
                continue

            # Placement for this specific clip within the shot. Anchor BOTH
            # start_frame and end_frame to absolute seconds (not start+duration)
            # so adjacent sub-clips touch exactly on the same frame and the
            # shot's total length matches the voice down to one frame.
            clip_start_sec = start_sec + (res_idx * clip_duration_sec)
            is_last_clip = (res_idx == num_clips - 1)
            if is_last_clip:
                # Stretch the last pick of this shot across the silence
                # before the next shot begins, so the b-roll covers the
                # pause instead of leaving an empty timeline slot. For the
                # very last shot of the project next_start == voice_end,
                # so this just ends on the voice — no extension.
                clip_end_sec = next_start
            else:
                clip_end_sec = clip_start_sec + clip_duration_sec

            start_frame     = sec_to_frames(clip_start_sec, fps_exact)
            end_frame       = sec_to_frames(clip_end_sec,   fps_exact)
            duration_frames = max(1, end_frame - start_frame)
            max_end_frame   = max(max_end_frame, end_frame)

            # Shared filename logic so the headless downloader and this exporter
            # always agree on the path.
            slot_id = shot.get("slot_id", "X")
            footage_num = res_idx + 1
            filename = clip_filename(slot_id, footage_num, res.get("matched_query", ""), seen_filenames)

            filepath = os.path.join(base_dir, filename)
            file_uri = _get_premiere_safe_pathurl(filepath)
            
            if filename not in asset_map:
                # Media duration
                media_dur_sec = _get_media_duration(filepath, fallback_duration=clip_duration_sec + 10.0)
                media_dur_frames = sec_to_frames(media_dur_sec, fps_exact)
                
                asset_map[filename] = {
                    "id": f"file-{next_asset_id}",
                    "filename": filename,
                    "uri": file_uri,
                    "media_dur_frames": media_dur_frames
                }
                next_asset_id += 1

            asset_info = asset_map[filename]
            clip_id = f"clip-{i}-{res_idx}"

            # ── Insert Clip on Timeline ──
            xml.append(f'              <clipitem id="{clip_id}">')
            xml.append(f'                <name>{_xml_attr(asset_info["filename"])}</name>')
            xml.append(f'                <duration>{asset_info["media_dur_frames"]}</duration>')
            xml.append('                <rate>')
            xml.append(f'                  <timebase>{timebase}</timebase>')
            xml.append('                  <ntsc>TRUE</ntsc>')
            xml.append('                </rate>')
            
            # Placement on timeline
            xml.append(f'                <start>{start_frame}</start>')
            xml.append(f'                <end>{end_frame}</end>')
            
            # Cut points on the source file. Default: start at frame 0. If
            # this clip has a preferred trim learned from a re-imported
            # Premiere edit, start at that in-point instead — keeping the same
            # timeline-slot length so placement/layout is unchanged. Fully
            # defensive: any miss or error falls back to in=0.
            in_frame_src = _preferred_in_frame(
                res.get("url", ""), filename,
                duration_frames, asset_info["media_dur_frames"], fps_exact,
            )
            xml.append(f'                <in>{in_frame_src}</in>')
            xml.append(f'                <out>{in_frame_src + duration_frames}</out>')
            
            # File Definition Block
            if asset_info["id"] not in defined_files:
                xml.append(f'                <file id="{asset_info["id"]}">')
                xml.append(f'                  <name>{_xml_attr(asset_info["filename"])}</name>')
                xml.append(f'                  <pathurl>{_xml_attr(asset_info["uri"])}</pathurl>')
                xml.append('                  <rate>')
                xml.append(f'                    <timebase>{timebase}</timebase>')
                xml.append('                    <ntsc>TRUE</ntsc>')
                xml.append('                  </rate>')
                xml.append(f'                  <duration>{asset_info["media_dur_frames"]}</duration>')
                xml.append('                  <media>')
                xml.append('                    <video>')
                xml.append(f'                      <duration>{asset_info["media_dur_frames"]}</duration>')
                xml.append('                    </video>')
                xml.append('                  </media>')
                xml.append('                </file>')
                defined_files.add(asset_info["id"])
            else:
                xml.append(f'                <file id="{asset_info["id"]}"/>')

            # Add Marker
            intent = _xml_attr(shot.get('shot_intent', ''))
            if intent:
                xml.append('                <marker>')
                xml.append('                  <name>Shot Intent</name>')
                xml.append(f'                  <comment>{intent}</comment>')
                xml.append('                  <in>0</in>')
                xml.append('                  <out>1</out>')
                xml.append('                </marker>')

            xml.append('              </clipitem>')

    # ── Close XML ──
    # ── Close Track 1 ──
    xml.append('            </track>')

    # ── Track 2: Overlays ──
    if overlays:
        # Premiere needs every video track to be explicitly enabled/unlocked,
        # otherwise the overlay track imports as a phantom and the clips don't
        # show on the timeline.
        xml.append('            <track>')
        xml.append('              <enabled>TRUE</enabled>')
        xml.append('              <locked>FALSE</locked>')
        for idx, ov in enumerate(overlays):
            ov_path = ov.get("filepath")
            if not ov_path or not os.path.exists(ov_path):
                continue

            ov_filename = os.path.basename(ov_path)
            ov_uri = _get_premiere_safe_pathurl(ov_path)

            s_sec = float(ov.get("start_sec", 0)) - time_offset
            e_sec = float(ov.get("end_sec", s_sec + time_offset + 3)) - time_offset
            if e_sec <= s_sec:
                e_sec = s_sec + 3
            # Skip overlays whose audio time falls outside this chunk — the
            # caller is expected to pre-filter, but a stray entry shouldn't
            # land at a negative timeline position.
            if s_sec < 0:
                continue

            # Anchor both endpoints to absolute seconds so the on-screen text
            # appears/disappears exactly with the voice — separately rounding
            # start and duration could push the overlay off by a frame.
            s_frame = sec_to_frames(s_sec, fps_exact)
            e_frame = sec_to_frames(e_sec, fps_exact)
            d_frame = max(1, e_frame - s_frame)

            # An animated overlay is a real video file (Remotion alpha ProRes);
            # a classic overlay is a still PNG. A still can claim an "infinite"
            # media duration, but a movie must declare its actual length and the
            # timeline segment can't run past it.
            is_video = bool(ov.get("is_video")) or ov_path.lower().endswith(
                (".mov", ".mp4", ".webm", ".mkv"))
            if is_video:
                clip_dur_sec = _get_media_duration(ov_path, fallback_duration=(e_sec - s_sec))
                media_dur_frames = max(1, sec_to_frames(clip_dur_sec, fps_exact))
                d_frame = min(d_frame, media_dur_frames)
                e_frame = s_frame + d_frame
            else:
                # PNGs are usually 1 frame or infinite, but FCP7 likes a duration.
                media_dur_frames = sec_to_frames(3600.0, fps_exact)  # 1 hour fallback

            max_end_frame = max(max_end_frame, e_frame)
            file_id = f"file-ov-{idx}"
            clip_id = f"clip-ov-{idx}"

            # FCP7-XML still images require <samplecharacteristics>
            # (width/height/pixelaspectratio/fielddominance) inside the file's
            # <media><video>. Without those Premiere silently drops PNG
            # stills on import — the track is empty even though the XML
            # looks valid. <alphatype>straight</alphatype> on the clipitem
            # is what makes Premiere honor the PNG's transparency rather
            # than rendering the alpha as black. <masterclipid> + <enabled>
            # are the standard pairing Premiere expects for any clipitem.
            xml.append(f'              <clipitem id="{clip_id}">')
            xml.append(f'                <name>{_xml_attr(ov_filename)}</name>')
            xml.append('                <enabled>TRUE</enabled>')
            xml.append(f'                <duration>{media_dur_frames}</duration>')
            xml.append('                <rate>')
            xml.append(f'                  <timebase>{timebase}</timebase>')
            xml.append('                  <ntsc>TRUE</ntsc>')
            xml.append('                </rate>')
            xml.append(f'                <start>{s_frame}</start>')
            xml.append(f'                <end>{e_frame}</end>')
            xml.append('                <in>0</in>')
            xml.append(f'                <out>{d_frame}</out>')
            xml.append(f'                <masterclipid>{file_id}</masterclipid>')
            xml.append('                <alphatype>straight</alphatype>')

            xml.append(f'                <file id="{file_id}">')
            xml.append(f'                  <name>{_xml_attr(ov_filename)}</name>')
            xml.append(f'                  <pathurl>{_xml_attr(ov_uri)}</pathurl>')
            xml.append('                  <rate>')
            xml.append(f'                    <timebase>{timebase}</timebase>')
            xml.append('                    <ntsc>TRUE</ntsc>')
            xml.append('                  </rate>')
            xml.append(f'                  <duration>{media_dur_frames}</duration>')
            xml.append('                  <media>')
            xml.append('                    <video>')
            xml.append(f'                      <duration>{media_dur_frames}</duration>')
            xml.append('                      <samplecharacteristics>')
            xml.append('                        <width>1920</width>')
            xml.append('                        <height>1080</height>')
            xml.append('                        <anamorphic>FALSE</anamorphic>')
            xml.append('                        <pixelaspectratio>square</pixelaspectratio>')
            xml.append('                        <fielddominance>none</fielddominance>')
            xml.append('                      </samplecharacteristics>')
            xml.append('                    </video>')
            xml.append('                  </media>')
            xml.append('                </file>')
            
            # ── Animations ──
            anim = ov.get("animation")
            if anim == "Fade In/Out":
                xml.append('                <filter>')
                xml.append('                  <effect>')
                xml.append('                    <name>Opacity</name>')
                xml.append('                    <effectid>opacity</effectid>')
                xml.append('                    <parameter>')
                xml.append('                      <parameterid>opacity</parameterid>')
                xml.append('                      <name>Opacity</name>')
                xml.append('                      <valuemin>0</valuemin>')
                xml.append('                      <valuemax>100</valuemax>')
                xml.append('                      <value>100</value>')
                fade_frames = sec_to_frames(0.5, fps_exact)
                xml.append('                      <keyframe><when>0</when><value>0</value></keyframe>')
                xml.append(f'                      <keyframe><when>{fade_frames}</when><value>100</value></keyframe>')
                xml.append(f'                      <keyframe><when>{d_frame - fade_frames}</when><value>100</value></keyframe>')
                xml.append(f'                      <keyframe><when>{d_frame}</when><value>0</value></keyframe>')
                xml.append('                    </parameter>')
                xml.append('                  </effect>')
                xml.append('                </filter>')

            elif anim in ["Slide Up", "Slide In Left"]:
                # Basic Motion filter
                xml.append('                <filter>')
                xml.append('                  <effect>')
                xml.append('                    <name>Basic Motion</name>')
                xml.append('                    <effectid>basic</effectid>')
                xml.append('                    <parameter>')
                xml.append('                      <parameterid>center</parameterid>')
                xml.append('                      <name>Center</name>')
                
                # Default is center (0,0)
                xml.append('                      <value><horiz>0</horiz><vert>0</vert></value>')
                
                anim_frames = sec_to_frames(0.6, fps_exact)
                if anim == "Slide Up":
                    # Start below (vert 100), slide to 0, stay, slide down
                    xml.append('                      <keyframe>')
                    xml.append('                        <when>0</when>')
                    xml.append('                        <value><horiz>0</horiz><vert>100</vert></value>')
                    xml.append('                      </keyframe>')
                    xml.append('                      <keyframe>')
                    xml.append(f'                        <when>{anim_frames}</when>')
                    xml.append('                        <value><horiz>0</horiz><vert>0</vert></value>')
                    xml.append('                      </keyframe>')
                    xml.append('                      <keyframe>')
                    xml.append(f'                        <when>{d_frame - anim_frames}</when>')
                    xml.append('                        <value><horiz>0</horiz><vert>0</vert></value>')
                    xml.append('                      </keyframe>')
                    xml.append('                      <keyframe>')
                    xml.append(f'                        <when>{d_frame}</when>')
                    xml.append('                        <value><horiz>0</horiz><vert>100</vert></value>')
                    xml.append('                      </keyframe>')
                
                elif anim == "Slide In Left":
                    # Start left (horiz -100), slide to 0, stay, slide right
                    xml.append('                      <keyframe>')
                    xml.append('                        <when>0</when>')
                    xml.append('                        <value><horiz>-100</horiz><vert>0</vert></value>')
                    xml.append('                      </keyframe>')
                    xml.append('                      <keyframe>')
                    xml.append(f'                        <when>{anim_frames}</when>')
                    xml.append('                        <value><horiz>0</horiz><vert>0</vert></value>')
                    xml.append('                      </keyframe>')
                    xml.append('                      <keyframe>')
                    xml.append(f'                        <when>{d_frame - anim_frames}</when>')
                    xml.append('                        <value><horiz>0</horiz><vert>0</vert></value>')
                    xml.append('                      </keyframe>')
                    xml.append('                      <keyframe>')
                    xml.append(f'                        <when>{d_frame}</when>')
                    xml.append('                        <value><horiz>100</horiz><vert>0</vert></value>')
                    xml.append('                      </keyframe>')

                xml.append('                    </parameter>')
                xml.append('                  </effect>')
                xml.append('                </filter>')
                
            xml.append('              </clipitem>')
        xml.append('            </track>')
    xml.append('          </video>')

    # ── AUDIO TRACKS ──
    xml.append('          <audio>')
    # Track 1: Placeholder for Narration/Direct Audio (Empty for now)
    xml.append('            <track/>')
    
    # Track 2: SFX
    if sfx_list:
        xml.append('            <track>')
        for idx, sfx in enumerate(sfx_list):
            sfx_path = sfx.get("filepath")
            if not sfx_path or not os.path.exists(sfx_path):
                continue
                
            sfx_filename = os.path.basename(sfx_path)
            sfx_uri = _get_premiere_safe_pathurl(sfx_path)
            
            s_sec = float(sfx.get("start_sec", 0)) - time_offset
            if s_sec < 0:
                continue

            # Get actual duration of the SFX file
            sfx_dur_sec = _get_media_duration(sfx_path, fallback_duration=2.0)

            s_frame = sec_to_frames(s_sec, fps_exact)
            d_frame = sec_to_frames(sfx_dur_sec, fps_exact)
            e_frame = s_frame + d_frame
            max_end_frame = max(max_end_frame, e_frame)

            file_id = f"file-sfx-{idx}"
            clip_id = f"clip-sfx-{idx}"
            
            xml.append(f'              <clipitem id="{clip_id}">')
            xml.append(f'                <name>{_xml_attr(sfx_filename)}</name>')
            xml.append(f'                <duration>{d_frame}</duration>')
            xml.append('                <rate>')
            xml.append(f'                  <timebase>{timebase}</timebase>')
            xml.append('                  <ntsc>TRUE</ntsc>')
            xml.append('                </rate>')
            xml.append(f'                <start>{s_frame}</start>')
            xml.append(f'                <end>{e_frame}</end>')
            xml.append('                <in>0</in>')
            xml.append(f'                <out>{d_frame}</out>')
            
            xml.append(f'                <file id="{file_id}">')
            xml.append(f'                  <name>{_xml_attr(sfx_filename)}</name>')
            xml.append(f'                  <pathurl>{_xml_attr(sfx_uri)}</pathurl>')
            xml.append('                  <rate>')
            xml.append(f'                    <timebase>{timebase}</timebase>')
            xml.append('                    <ntsc>TRUE</ntsc>')
            xml.append('                  </rate>')
            xml.append(f'                  <duration>{d_frame}</duration>')
            xml.append('                  <media>')
            xml.append('                    <audio>')
            xml.append(f'                      <duration>{d_frame}</duration>')
            xml.append('                      <samplecharacteristics>')
            xml.append('                        <samplerate>48000</samplerate>')
            xml.append('                        <depth>16</depth>')
            xml.append('                      </samplecharacteristics>')
            xml.append('                    </audio>')
            xml.append('                  </media>')
            xml.append('                </file>')
            
            # Simple panning (center)
            xml.append('                <sourcetrack>')
            xml.append('                  <trackindex>1</trackindex>')
            xml.append('                </sourcetrack>')
            
            xml.append('              </clipitem>')
        xml.append('            </track>')
    else:
        # Placeholder A2
        xml.append('            <track/>')
        
    xml.append('          </audio>')
    xml.append('        </media>')
    xml.append('      </sequence>')
    xml.append('    </children>')
    xml.append('  </project>')
    xml.append('</xmeml>')

    # Inject the sequence-level <duration> + <timecode> right before <media>.
    # Premiere's FCP7 importer needs a fully-formed sequence (a duration, a start
    # timecode, and a format rate) to construct the timeline; without them it
    # parses the <file> defs into the project bin but never places the clips —
    # the "clips import but the timeline stays empty" symptom.
    seq_frames = max(1, max_end_frame)
    xml[seq_media_idx:seq_media_idx] = [
        f'        <duration>{seq_frames}</duration>',
        '        <timecode>',
        '          <rate>',
        f'            <timebase>{timebase}</timebase>',
        '            <ntsc>TRUE</ntsc>',
        '          </rate>',
        '          <string>00:00:00:00</string>',
        '          <frame>0</frame>',
        '          <displayformat>NDF</displayformat>',
        '        </timecode>',
    ]

    return "\n".join(xml)


def generate_overlays_fcpxml(overlays: list, project_name: str = "default",
                             time_offset: float = 0.0) -> str:
    """Standalone FCP7 XML containing ONLY the text-overlay PNG track.

    Use case: the user wants to tweak overlay text (or fonts/positions) and
    re-import just the overlays into an existing Premiere sequence, without
    re-importing the full b-roll timeline. Importing this XML produces a
    sequence whose V1 holds every overlay clipitem at the same absolute
    timecode as the main export — copy/paste that track onto V2 of the
    main sequence and the overlays land exactly where they belong.

    Frame math, alpha handling, and sample-characteristics mirror the
    overlay track inside ``generate_fcpxml`` so Premiere accepts the PNGs.
    """
    fps_exact = 23.976
    timebase = 24

    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<!DOCTYPE xmeml>')
    xml.append('<xmeml version="4">')
    xml.append('  <project>')
    xml.append(f'    <name>{_xml_attr(project_name)} (Overlays)</name>')
    xml.append('    <children>')
    xml.append('      <sequence id="b-roll-overlays-seq">')
    xml.append(f'        <name>{_xml_attr(project_name)} — Overlays</name>')
    xml.append('        <rate>')
    xml.append(f'          <timebase>{timebase}</timebase>')
    xml.append('          <ntsc>TRUE</ntsc>')
    xml.append('        </rate>')
    seq_media_idx = len(xml)   # where to inject the sequence <duration>/<timecode>
    xml.append('        <media>')
    xml.append('          <video>')
    xml.append('            <format>')
    xml.append('              <samplecharacteristics>')
    xml.append('                <rate>')
    xml.append(f'                  <timebase>{timebase}</timebase>')
    xml.append('                  <ntsc>TRUE</ntsc>')
    xml.append('                </rate>')
    xml.append('                <width>1920</width>')
    xml.append('                <height>1080</height>')
    xml.append('                <anamorphic>FALSE</anamorphic>')
    xml.append('                <pixelaspectratio>square</pixelaspectratio>')
    xml.append('                <fielddominance>none</fielddominance>')
    xml.append('              </samplecharacteristics>')
    xml.append('            </format>')
    xml.append('            <track>')
    xml.append('              <enabled>TRUE</enabled>')
    xml.append('              <locked>FALSE</locked>')

    emitted = 0
    max_end_frame = 0
    for idx, ov in enumerate(overlays or []):
        ov_path = ov.get("filepath")
        if not ov_path or not os.path.exists(ov_path):
            continue

        ov_filename = os.path.basename(ov_path)
        ov_uri = _get_premiere_safe_pathurl(ov_path)

        s_sec = float(ov.get("start_sec", 0)) - time_offset
        e_sec = float(ov.get("end_sec", s_sec + time_offset + 3)) - time_offset
        if e_sec <= s_sec:
            e_sec = s_sec + 3
        if s_sec < 0:
            continue

        s_frame = sec_to_frames(s_sec, fps_exact)
        e_frame = sec_to_frames(e_sec, fps_exact)
        d_frame = max(1, e_frame - s_frame)

        # Animated overlays are real video files (alpha ProRes); stills are PNGs.
        is_video = bool(ov.get("is_video")) or ov_path.lower().endswith(
            (".mov", ".mp4", ".webm", ".mkv"))
        if is_video:
            clip_dur_sec = _get_media_duration(ov_path, fallback_duration=(e_sec - s_sec))
            media_dur_frames = max(1, sec_to_frames(clip_dur_sec, fps_exact))
            d_frame = min(d_frame, media_dur_frames)
            e_frame = s_frame + d_frame
        else:
            media_dur_frames = sec_to_frames(3600.0, fps_exact)

        max_end_frame = max(max_end_frame, e_frame)
        file_id = f"file-ov-{idx}"
        clip_id = f"clip-ov-{idx}"

        xml.append(f'              <clipitem id="{clip_id}">')
        xml.append(f'                <name>{_xml_attr(ov_filename)}</name>')
        xml.append('                <enabled>TRUE</enabled>')
        xml.append(f'                <duration>{media_dur_frames}</duration>')
        xml.append('                <rate>')
        xml.append(f'                  <timebase>{timebase}</timebase>')
        xml.append('                  <ntsc>TRUE</ntsc>')
        xml.append('                </rate>')
        xml.append(f'                <start>{s_frame}</start>')
        xml.append(f'                <end>{e_frame}</end>')
        xml.append('                <in>0</in>')
        xml.append(f'                <out>{d_frame}</out>')
        xml.append(f'                <masterclipid>{file_id}</masterclipid>')
        xml.append('                <alphatype>straight</alphatype>')

        xml.append(f'                <file id="{file_id}">')
        xml.append(f'                  <name>{_xml_attr(ov_filename)}</name>')
        xml.append(f'                  <pathurl>{_xml_attr(ov_uri)}</pathurl>')
        xml.append('                  <rate>')
        xml.append(f'                    <timebase>{timebase}</timebase>')
        xml.append('                    <ntsc>TRUE</ntsc>')
        xml.append('                  </rate>')
        xml.append(f'                  <duration>{media_dur_frames}</duration>')
        xml.append('                  <media>')
        xml.append('                    <video>')
        xml.append(f'                      <duration>{media_dur_frames}</duration>')
        xml.append('                      <samplecharacteristics>')
        xml.append('                        <width>1920</width>')
        xml.append('                        <height>1080</height>')
        xml.append('                        <anamorphic>FALSE</anamorphic>')
        xml.append('                        <pixelaspectratio>square</pixelaspectratio>')
        xml.append('                        <fielddominance>none</fielddominance>')
        xml.append('                      </samplecharacteristics>')
        xml.append('                    </video>')
        xml.append('                  </media>')
        xml.append('                </file>')

        # Animations — same keyframe patterns the main exporter uses, so an
        # imported overlay carries its fade/slide motion across.
        anim = ov.get("animation")
        if anim == "Fade In/Out":
            fade_frames = sec_to_frames(0.5, fps_exact)
            xml.append('                <filter>')
            xml.append('                  <effect>')
            xml.append('                    <name>Opacity</name>')
            xml.append('                    <effectid>opacity</effectid>')
            xml.append('                    <parameter>')
            xml.append('                      <parameterid>opacity</parameterid>')
            xml.append('                      <name>Opacity</name>')
            xml.append('                      <valuemin>0</valuemin>')
            xml.append('                      <valuemax>100</valuemax>')
            xml.append('                      <value>100</value>')
            xml.append('                      <keyframe><when>0</when><value>0</value></keyframe>')
            xml.append(f'                      <keyframe><when>{fade_frames}</when><value>100</value></keyframe>')
            xml.append(f'                      <keyframe><when>{d_frame - fade_frames}</when><value>100</value></keyframe>')
            xml.append(f'                      <keyframe><when>{d_frame}</when><value>0</value></keyframe>')
            xml.append('                    </parameter>')
            xml.append('                  </effect>')
            xml.append('                </filter>')
        elif anim in ("Slide Up", "Slide In Left"):
            anim_frames = sec_to_frames(0.6, fps_exact)
            xml.append('                <filter>')
            xml.append('                  <effect>')
            xml.append('                    <name>Basic Motion</name>')
            xml.append('                    <effectid>basic</effectid>')
            xml.append('                    <parameter>')
            xml.append('                      <parameterid>center</parameterid>')
            xml.append('                      <name>Center</name>')
            xml.append('                      <value><horiz>0</horiz><vert>0</vert></value>')
            if anim == "Slide Up":
                kfs = [(0, 0, 100), (anim_frames, 0, 0),
                       (d_frame - anim_frames, 0, 0), (d_frame, 0, 100)]
            else:  # Slide In Left
                kfs = [(0, -100, 0), (anim_frames, 0, 0),
                       (d_frame - anim_frames, 0, 0), (d_frame, 100, 0)]
            for when, h, v in kfs:
                xml.append('                      <keyframe>')
                xml.append(f'                        <when>{when}</when>')
                xml.append(f'                        <value><horiz>{h}</horiz><vert>{v}</vert></value>')
                xml.append('                      </keyframe>')
            xml.append('                    </parameter>')
            xml.append('                  </effect>')
            xml.append('                </filter>')

        xml.append('              </clipitem>')
        emitted += 1

    xml.append('            </track>')
    xml.append('          </video>')
    # Empty audio block keeps the sequence valid; no SFX in this export.
    xml.append('          <audio>')
    xml.append('            <track/>')
    xml.append('          </audio>')
    xml.append('        </media>')
    xml.append('      </sequence>')
    xml.append('    </children>')
    xml.append('  </project>')
    xml.append('</xmeml>')

    # Same sequence-level fix as generate_fcpxml: a fully-formed sequence
    # (duration + timecode) so Premiere builds the overlay timeline instead of
    # dropping the clips into the bin only.
    seq_frames = max(1, max_end_frame)
    xml[seq_media_idx:seq_media_idx] = [
        f'        <duration>{seq_frames}</duration>',
        '        <timecode>',
        '          <rate>',
        f'            <timebase>{timebase}</timebase>',
        '            <ntsc>TRUE</ntsc>',
        '          </rate>',
        '          <string>00:00:00:00</string>',
        '          <frame>0</frame>',
        '          <displayformat>NDF</displayformat>',
        '        </timecode>',
    ]

    return "\n".join(xml)
