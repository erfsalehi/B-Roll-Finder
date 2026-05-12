import os
from xml.sax.saxutils import escape


def _xml_attr(value) -> str:
    return escape(str(value), {'"': "&quot;"})


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

def generate_fcpxml(shots: list, project_name: str = "default") -> str:
    """
    Generates a Final Cut Pro XML (v1.9) that Premiere Pro can import.
    Creates a sequence with all selected video clips placed on the timeline
    at their respective timestamps, with gaps closed to match the SRT.
    """
    proj_folder = _safe_for_fs(project_name, 50)
    # We use absolute paths for the 'src' attribute so Premiere can find them instantly.
    base_dir = os.path.abspath(os.path.join("downloads", "director", proj_folder))
    
    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<!DOCTYPE fcpxml>')
    xml.append('<fcpxml version="1.9">')
    
    # ── Resources ────────────────────────────────────────────────────────────
    xml.append('  <resources>')
    xml.append('    <format id="r1" name="FFVideoFormat1080p2398" frameDuration="1001/24000s" width="1920" height="1080"/>')
    
    asset_map = {} # url -> asset_id
    next_asset_id = 1
    
    # Pre-scan for unique assets (selected clips)
    seen_filenames = set()
    for shot in shots:
        sel = shot.get("selected_results", [])
        if not sel:
            continue
        
        # We only assign the FIRST selected clip to the timeline for this shot
        res = sel[0]
        url = res.get("url")
        if not url:
            continue
            
        if url not in asset_map:
            slot_id = shot.get("slot_id", "X")
            keyword = _safe_for_fs(res.get("matched_query", ""), 30) or "clip"
            # Filename logic must match app.py EXACTLY: {slot_id}-1-{keyword}.mp4
            # (since we only use the 1st selected clip for the timeline)
            base = f"{slot_id}-1-{keyword}"
            filename = f"{base}.mp4"
            
            # Collision handling (rare for 1-1 naming but for safety)
            n = 1
            while filename in seen_filenames:
                n += 1
                filename = f"{base}-{n}.mp4"
            seen_filenames.add(filename)
            
            asset_id = f"a{next_asset_id}"
            asset_map[url] = {
                "id": asset_id,
                "filename": filename,
                "path": os.path.join(base_dir, filename)
            }
            
            file_url = "file://" + _xml_attr(asset_map[url]["path"].replace("\\", "/"))
            xml.append(f'    <asset id="{asset_id}" name="{_xml_attr(filename)}" src="{file_url}" start="0s" duration="3600s" hasVideo="1" hasAudio="1"/>')
            next_asset_id += 1
            
    xml.append('  </resources>')
    
    # ── Library & Event ──────────────────────────────────────────────────────
    xml.append('  <library>')
    xml.append(f'    <event name="B-Roll Director - {_xml_attr(project_name)}">')
    xml.append('      <project name="B-Roll Sequence">')
    # Defaulting to 23.98fps (1001/24000) as it's standard for cinematic edits
    xml.append('        <sequence format="r1" tcStart="0s" tcFormat="NDF" duration="3600s">')
    xml.append('          <spine>')
    
    # ── Timeline ─────────────────────────────────────────────────────────────
    current_timeline_pos = 0.0
    
    for i, shot in enumerate(shots):
        start_sec = float(shot.get('timestamp', 0))
        
        # Calculate duration by looking ahead (matching SRT logic)
        if i < len(shots) - 1:
            end_val = float(shots[i+1].get('timestamp', start_sec + 5))
        else:
            end_val = float(shot.get('end_timestamp', start_sec + 5))
        
        duration = end_val - start_sec
        if duration <= 0:
            duration = 1.0
            
        # If there's a gap between current_timeline_pos and start_sec, add a gap resource
        if start_sec > current_timeline_pos:
            gap_dur = start_sec - current_timeline_pos
            xml.append(f'            <gap name="Gap" offset="{current_timeline_pos}s" duration="{gap_dur}s" start="3600s"/>')
        
        sel = shot.get("selected_results", [])
        asset_info = None
        if sel and sel[0].get("url") in asset_map:
            asset_info = asset_map[sel[0]["url"]]
            
        if asset_info:
            # Place the clip
            clip_name = _xml_attr(asset_info["filename"])
            # Start at 0s of the clip, or if it's a smart_proxy, we might want to offset.
            # But usually the downloader trims it. For now, assume 0s start.
            xml.append(f'            <asset-clip name="{clip_name}" offset="{start_sec}s" ref="{asset_info["id"]}" duration="{duration}s" start="0s">')
            # Add a marker with the shot intent inside the clip for convenience
            intent = _xml_attr(shot.get('shot_intent', ''))
            xml.append(f'              <marker start="0s" duration="1s" value="Shot {shot.get("slot_id")}: {intent}"/>')
            xml.append('            </asset-clip>')
        else:
            # Placeholder gap if no clip selected
            xml.append(f'            <gap name="MISSING: Shot {shot.get("slot_id")}" offset="{start_sec}s" duration="{duration}s" start="3600s"/>')
            
        current_timeline_pos = start_sec + duration
        
    xml.append('          </spine>')
    xml.append('        </sequence>')
    xml.append('      </project>')
    xml.append('    </event>')
    xml.append('  </library>')
    xml.append('</fcpxml>')
    
    return "\n".join(xml)
