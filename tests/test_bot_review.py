"""Bot review-gate formatting, command predicates, menu, and delivery."""

import bot.telegram_bot as tb


# ── command predicates ────────────────────────────────────────────────────────

def test_new_command_predicates():
    assert tb.is_settings_command("/settings") and tb.is_settings_command("/config")
    assert tb.is_download_command("/download") and tb.is_download_command("/go@Bot")
    assert tb.is_refine_command("/refine") and tb.is_refine_command("/improve")
    assert tb.is_details_command("/details") and tb.is_details_command("/shots")
    assert tb.is_help_command("/help") and tb.is_help_command("/start")
    assert not tb.is_download_command("/status")


# ── review summary ──────────────────────────────────────────────────────────

def test_format_review_shows_counts_qa_errors_and_actions():
    result = {
        "n_shots": 12, "n_selected": 11, "n_clips": 34,
        "shots": [
            {"slot_id": 4, "priority": "high", "selected_results": [{"source": "pexels"}]},
            {"slot_id": 7, "priority": "medium", "selected_results": []},  # empty
        ],
        "qa": {"overall": "Mostly solid.", "issues": [
            {"slot_id": 4, "severity": "high", "problem": "repetitive with #3",
             "suggestion": "vary the subject"},
        ]},
        "errors": ["pexels: timeout on shot 9"],
    }
    out = tb.format_review("myvid", result)
    assert "Shots: 12" in out and "clips: 34" in out
    assert "#4 (high) repetitive with #3" in out and "vary the subject" in out
    assert "No clip for shot(s): 7" in out
    # errors are grouped by signature (numbers masked to #)
    assert "pexels: timeout on shot #" in out
    assert "/download" in out and "/refine" in out


def test_format_errors_groups_systemic_failures():
    # 3000 near-identical YouTube cookie errors should collapse to ONE type line.
    errors = [f"YouTube search failed for 'query {i}': could not find firefox cookies database"
              for i in range(3000)]
    out = "\n".join(tb.format_errors_block(errors))
    assert "3000 issue(s)" in out and "1 type[s]" in out
    assert "(×3000)" in out


def test_format_details_lists_selected_shots():
    shots = [
        {"slot_id": 1, "priority": "high", "selected_results": [{"source": "pexels"}, {"source": "youtube"}]},
        {"slot_id": 2, "priority": "low", "selected_results": []},
    ]
    out = tb.format_details(shots)
    assert "#1 [high] 2 clip(s)" in out and "pexels,youtube" in out
    assert "#2" not in out                       # no selection → not listed


# ── settings keyboard ─────────────────────────────────────────────────────────

def test_build_settings_keyboard_has_row_per_option_plus_actions(monkeypatch):
    kb = tb.build_settings_keyboard(123)
    rows = kb["inline_keyboard"]
    # one row per option + a final actions row
    assert len(rows) == len(tb.bot_settings.OPTIONS) + 1
    cbs = [btn["callback_data"] for row in rows for btn in row]
    assert "set:pexels_num" in cbs and "set:qa" in cbs
    assert "settings:reset" in cbs and "settings:close" in cbs


# ── delivery (zip → link + attach) ────────────────────────────────────────────

def test_deliver_project_links_and_attaches_small(monkeypatch):
    sent, docs = [], []
    monkeypatch.setattr(tb, "send_message", lambda chat, text: sent.append(text))
    monkeypatch.setattr(tb, "send_document", lambda chat, path, caption="": docs.append(path))
    monkeypatch.setitem(tb._FILESERVER, "port", 8770)
    # A real (empty) file must exist for the attach path to fire.
    import os
    os.makedirs("downloads", exist_ok=True)
    zip_path = os.path.join("downloads", "small_proj.zip")
    open(zip_path, "wb").close()
    try:
        import core.output
        monkeypatch.setattr(core.output, "zip_project",
                            lambda name, progress=None: {"path": f"downloads/{name}.zip",
                                          "size_bytes": 10 * 1024 * 1024, "files": 5})
        tb.deliver_project(99, "small_proj")
    finally:
        os.remove(zip_path)
    assert any("small_proj" in m and "10.0MB" in m for m in sent)
    assert any("/d/small_proj.zip?" in m for m in sent)   # signed link present
    assert docs and docs[0].endswith("small_proj.zip")     # attached (under 50MB)


def test_deliver_project_large_skips_attach(monkeypatch):
    sent, docs = [], []
    monkeypatch.setattr(tb, "send_message", lambda chat, text: sent.append(text))
    monkeypatch.setattr(tb, "send_document", lambda chat, path, caption="": docs.append(path))
    monkeypatch.setitem(tb._FILESERVER, "port", 8770)
    import core.output
    monkeypatch.setattr(core.output, "zip_project",
                        lambda name, progress=None: {"path": f"downloads/{name}.zip",
                                      "size_bytes": 200 * 1024 * 1024, "files": 40})
    tb.deliver_project(99, "big_proj")
    assert any("/d/big_proj.zip?" in m for m in sent)      # link still provided
    assert docs == []                                       # too big to attach


# ── chunked delivery (zip → download → /next) ─────────────────────────────────

def _stub_telegram(monkeypatch, sent, docs):
    monkeypatch.setattr(tb, "send_message",
                        lambda chat, text: sent.append(text) or {"message_id": 1})
    monkeypatch.setattr(tb, "edit_message", lambda chat, mid, text: None)
    monkeypatch.setattr(tb, "send_document",
                        lambda chat, path, caption="": docs.append(path))


def test_chunked_delivery_walks_pieces_and_purges(tmp_path, monkeypatch):
    import os
    monkeypatch.chdir(tmp_path)
    sent, docs = [], []
    _stub_telegram(monkeypatch, sent, docs)
    monkeypatch.setitem(tb._FILESERVER, "port", 8770)
    monkeypatch.setattr(tb.bot_settings, "get_settings",
                        lambda chat: {"chunked_download": True, "chunk_size_mb": 1,
                                      "purge_after_zip": True})
    # Project ~ 6 clips of 400KB → several 1MB chunks.
    proj = tmp_path / "downloads" / "proj"
    (proj / "director").mkdir(parents=True)
    (proj / "proj.xml").write_text("<xmeml/>")
    for i in range(1, 7):
        (proj / "director" / f"{i}-1-clip.mp4").write_bytes(b"x" * (400 * 1024))

    tb.deliver_project(99, "proj")            # chunked path (setting on)
    assert 99 in tb._CHUNKED
    n = len(tb._CHUNKED[99]["manifest"]["chunks"])
    assert n >= 2                              # genuinely split
    # First chunk delivered, its source purged, a part-zip on disk.
    assert any("Chunk 1/" in m for m in sent)
    assert os.path.isfile(tb._CHUNKED[99]["last_zip"])

    # Walk to the end via /next; each step removes the previous part-zip.
    prev = tb._CHUNKED[99]["last_zip"]
    for _ in range(n):
        tb.handle_next(99)
        assert not os.path.exists(prev) or 99 not in tb._CHUNKED
        if 99 in tb._CHUNKED:
            prev = tb._CHUNKED[99]["last_zip"]

    assert 99 not in tb._CHUNKED               # session finished + cleared
    assert any("All" in m and "delivered" in m for m in sent)
    # All source clips were reclaimed as we went.
    assert not any(f.endswith(".mp4") for f in os.listdir(proj / "director")) \
        if (proj / "director").exists() else True


def test_zip_chunked_command_forces_pieces(tmp_path, monkeypatch):
    import os
    monkeypatch.chdir(tmp_path)
    sent, docs = [], []
    _stub_telegram(monkeypatch, sent, docs)
    # Setting is OFF, but "/zip chunked" forces it.
    monkeypatch.setattr(tb.bot_settings, "get_settings",
                        lambda chat: {"chunked_download": False, "chunk_size_mb": 1,
                                      "purge_after_zip": False})
    proj = tmp_path / "downloads" / "myproj"
    (proj / "director").mkdir(parents=True)
    (proj / "myproj.xml").write_text("<xmeml/>")
    (proj / "director" / "1-1-clip.mp4").write_bytes(b"x" * (100 * 1024))

    tb.handle_zip(99, "/zip chunked myproj")
    assert 99 in tb._CHUNKED                    # chunked session started
    tb._CHUNKED.pop(99, None)                   # don't leak state across tests


def test_next_without_session_is_graceful(monkeypatch):
    sent = []
    monkeypatch.setattr(tb, "send_message", lambda chat, text: sent.append(text))
    tb._CHUNKED.pop(123, None)
    tb.handle_next(123)
    assert sent and "Nothing to continue" in sent[0]
