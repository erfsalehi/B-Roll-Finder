"""The per-shot source-link manifest used to manually re-download empty shots."""

from core.output import build_download_links_txt


def _shots():
    return [
        {"slot_id": 1, "shot_intent": "engine close-up", "priority": "normal",
         "selected_results": [
             {"title": "V8 Engine", "source": "youtube", "page_url": "https://youtu.be/aaa"},
             {"title": "Pexels engine", "source": "pexels", "url": "https://pexels/x"},
         ]},
        {"slot_id": 2, "shot_intent": "transmission gears", "priority": "normal",
         "selected_results": [],  # empty shot — failed/dropped downloads
         "video_results": [
             {"title": "Gearbox teardown", "source": "youtube", "page_url": "https://youtu.be/bbb"},
             {"title": "Gearbox teardown", "source": "youtube", "page_url": "https://youtu.be/bbb"},  # dup
             {"title": "CVT explained", "source": "youtube", "url": "https://youtu.be/ccc"},
         ]},
        {"slot_id": 3, "shot_intent": "intentionally skipped", "priority": "none",
         "selected_results": []},
    ]


def test_manifest_lists_empty_shot_candidates_for_redownload():
    txt = build_download_links_txt(_shots(), "demo")
    # Empty shot's candidate links are offered for manual re-download.
    assert "SHOTS WITH NO FOOTAGE" in txt
    assert "https://youtu.be/bbb" in txt
    assert "https://youtu.be/ccc" in txt
    # Title and link share a line.
    assert "Gearbox teardown - https://youtu.be/bbb" in txt


def test_manifest_dedupes_and_counts_and_skips_priority_none():
    txt = build_download_links_txt(_shots(), "demo")
    # Duplicate candidate link appears once.
    assert txt.count("https://youtu.be/bbb") == 1
    # priority=none shot is excluded from the active count and the listing.
    assert "2 shot(s): 1 with footage, 1 with NONE" in txt
    assert "intentionally skipped" not in txt


def test_manifest_lists_used_sources_per_shot():
    txt = build_download_links_txt(_shots(), "demo")
    assert "ALL SHOTS" in txt
    assert "V8 Engine - https://youtu.be/aaa" in txt
    assert "Pexels engine - https://pexels/x" in txt


def test_manifest_handles_no_shots():
    txt = build_download_links_txt([], "empty")
    assert "0 shot(s)" in txt
    assert txt.endswith("\n")
