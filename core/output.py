import os
from pathlib import Path
from xml.sax.saxutils import escape
from core.ffmpeg_utils import get_video_metadata


def _xml_attr(value) -> str:
    return escape(str(value), {'"': "&quot;"})


def _get_premiere_safe_pathurl(filepath: str) -> str:
    """
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


def format_time(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
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

def _safe_for_fs(text: str, max_len: int = 30) -> str:
    if not text:
        return ""
    cleaned = "".join(c if (c.isalnum() or c in " -_") else " "
                      for c in text)
    cleaned = "-".join(cleaned.split()).lower()
    return cleaned[:max_len].strip("-") or ""

def generate_fcpxml(shots: list, project_name: str = "default", overlays: list = None) -> str:
    """
    Generates a bulletproof Legacy FCP 7 XML (<xmeml>) for Premiere Pro.
    Uses exact frame math and implicit gaps to guarantee compatibility.
    """
    proj_folder = _safe_for_fs(project_name, 50)
    # We use absolute paths for the 'pathurl' attribute so Premiere can find them instantly.
    base_dir = os.path.abspath(os.path.join("downloads", proj_folder, "director"))
    
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
    xml.append('        <media>')
    xml.append('          <video>')
    xml.append('            <format>')
    xml.append('              <samplecharacteristics>')
    xml.append('                <width>1920</width>')
    xml.append('                <height>1080</height>')
    xml.append('              </samplecharacteristics>')
    xml.append('            </format>')
    xml.append('            <track>')

    asset_map = {}
    next_asset_id = 1
    seen_filenames = set()
    defined_files = set() # To track which files have already been injected

    # ── Timeline Math & Placement ──
    for i, shot in enumerate(shots):
        start_sec = float(shot.get('timestamp', 0))
        
        # Calculate overall shot duration
        if i < len(shots) - 1:
            end_sec = float(shots[i+1].get('timestamp', start_sec + 5))
        else:
            end_sec = float(shot.get('end_timestamp', start_sec + 5))
            
        total_duration_sec = end_sec - start_sec
        if total_duration_sec <= 0:
            total_duration_sec = 1.0

        sel = shot.get("selected_results", [])
        if not sel:
            continue
            
        # Distribute the total duration among all selected candidates for this shot
        num_clips = len(sel)
        clip_duration_sec = total_duration_sec / num_clips
        
        for res_idx, res in enumerate(sel):
            url = res.get("url")
            if not url:
                continue
                
            # Placement for this specific clip within the shot
            clip_start_sec = start_sec + (res_idx * clip_duration_sec)
            clip_end_sec = clip_start_sec + clip_duration_sec
            
            start_frame = sec_to_frames(clip_start_sec, fps_exact)
            duration_frames = sec_to_frames(clip_duration_sec, fps_exact)
            end_frame = start_frame + duration_frames

            # File pathing logic - follow app.py naming exactly
            slot_id = shot.get("slot_id", "X")
            keyword = _safe_for_fs(res.get("matched_query", ""), 30) or "clip"
            footage_num = res_idx + 1
            filename = f"{slot_id}-{footage_num}-{keyword}.mp4"
            
            # Deduplicate filename if necessary
            n = 1
            temp_filename = filename
            while temp_filename in seen_filenames:
                n += 1
                temp_filename = f"{slot_id}-{footage_num}-{keyword}-{n}.mp4"
            filename = temp_filename
            seen_filenames.add(filename)
            
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
            
            # Cut points on the source file
            xml.append('                <in>0</in>')
            xml.append(f'                <out>{duration_frames}</out>')
            
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
        xml.append('            <track>')
        for idx, ov in enumerate(overlays):
            ov_path = ov.get("filepath")
            if not ov_path or not os.path.exists(ov_path):
                continue
            
            ov_filename = os.path.basename(ov_path)
            ov_uri = _get_premiere_safe_pathurl(ov_path)
            
            s_sec = float(ov.get("start_sec", 0))
            e_sec = float(ov.get("end_sec", s_sec + 3))
            d_sec = e_sec - s_sec
            
            s_frame = sec_to_frames(s_sec, fps_exact)
            d_frame = sec_to_frames(d_sec, fps_exact)
            e_frame = s_frame + d_frame
            
            # Media duration (PNGs are usually 1 frame or infinite, but FCP7 likes a duration)
            media_dur_frames = sec_to_frames(3600.0, fps_exact) # 1 hour fallback for stills
            
            file_id = f"file-ov-{idx}"
            clip_id = f"clip-ov-{idx}"
            
            xml.append(f'              <clipitem id="{clip_id}">')
            xml.append(f'                <name>{_xml_attr(ov_filename)}</name>')
            xml.append(f'                <duration>{media_dur_frames}</duration>')
            xml.append('                <rate>')
            xml.append(f'                  <timebase>{timebase}</timebase>')
            xml.append('                  <ntsc>TRUE</ntsc>')
            xml.append('                </rate>')
            xml.append(f'                <start>{s_frame}</start>')
            xml.append(f'                <end>{e_frame}</end>')
            xml.append('                <in>0</in>')
            xml.append(f'                <out>{d_frame}</out>')
            
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
            xml.append('                    </video>')
            xml.append('                  </media>')
            xml.append('                </file>')
            
            # Simple Fade In/Out if requested
            if ov.get("animation") == "Fade In/Out":
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
                # Keyframes for fade
                fade_frames = sec_to_frames(0.5, fps_exact)
                xml.append('                      <keyframe>')
                xml.append('                        <when>0</when>')
                xml.append('                        <value>0</value>')
                xml.append('                      </keyframe>')
                xml.append('                      <keyframe>')
                xml.append(f'                        <when>{fade_frames}</when>')
                xml.append('                        <value>100</value>')
                xml.append('                      </keyframe>')
                xml.append('                      <keyframe>')
                xml.append(f'                        <when>{d_frame - fade_frames}</when>')
                xml.append('                        <value>100</value>')
                xml.append('                      </keyframe>')
                xml.append('                      <keyframe>')
                xml.append(f'                        <when>{d_frame}</when>')
                xml.append('                        <value>0</value>')
                xml.append('                      </keyframe>')
                xml.append('                    </parameter>')
                xml.append('                  </effect>')
                xml.append('                </filter>')
                
            xml.append('              </clipitem>')
        xml.append('            </track>')
    xml.append('          </video>')
    xml.append('        </media>')
    xml.append('      </sequence>')
    xml.append('    </children>')
    xml.append('  </project>')
    xml.append('</xmeml>')
    
    return "\n".join(xml)
