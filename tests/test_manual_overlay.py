"""Manual single-overlay feature: /overlaytext <secs> <text> -> one rendered clip."""

import bot.telegram_bot as tb
import core.overlays_remotion as ovr


# ── command + arg parsing ─────────────────────────────────────────────────────

def test_is_overlay_text_command():
    assert tb.is_overlay_text_command("/overlaytext 4 hi")
    assert tb.is_overlay_text_command("/textoverlay hello")
    assert not tb.is_overlay_text_command("/overlay")          # distinct command
    assert not tb.is_overlay_text_command("/status")


def test_parse_overlay_text_args():
    assert tb._parse_overlay_text_args("/overlaytext 4 CAR BATTERY") == (4.0, "CAR BATTERY")
    assert tb._parse_overlay_text_args("/overlaytext 2.5 200AH") == (2.5, "200AH")
    # no leading number → default duration, whole body is text
    assert tb._parse_overlay_text_args("/overlaytext CHECK BATTERY") == (4.0, "CHECK BATTERY")
    # only a bare number → treat it as the text
    assert tb._parse_overlay_text_args("/overlaytext 2026") == (4.0, "2026")
    # nothing → usage
    assert tb._parse_overlay_text_args("/overlaytext") == (None, "")


# ── handler ──────────────────────────────────────────────────────────────────

def test_handle_overlay_text_renders_and_sends(monkeypatch, tmp_path):
    sent, docs = [], []
    monkeypatch.setattr(tb, "send_message", lambda c, t: {"message_id": 1} or sent.append(t))
    monkeypatch.setattr(tb, "edit_message", lambda c, m, t: None)
    monkeypatch.setattr(tb, "send_document", lambda c, p, caption="": docs.append(p))
    import core.overlays_remotion as o
    monkeypatch.setattr(o, "remotion_available", lambda: True)
    clip = tmp_path / "ov.mov"
    clip.write_bytes(b"x")
    monkeypatch.setattr(o, "render_one_overlay",
                        lambda text, dur, out_dir: {"filepath": str(clip)})
    tb.handle_overlay_text(7, "/overlaytext 3 CAR BATTERY")
    assert docs == [str(clip)]


def test_handle_overlay_text_usage_when_empty(monkeypatch):
    sent = []
    monkeypatch.setattr(tb, "send_message", lambda c, t: sent.append(t))
    tb.handle_overlay_text(7, "/overlaytext")
    assert any("Usage:" in m for m in sent)


def test_handle_overlay_text_when_renderer_missing(monkeypatch):
    sent = []
    monkeypatch.setattr(tb, "send_message", lambda c, t: sent.append(t))
    import core.overlays_remotion as o
    monkeypatch.setattr(o, "remotion_available", lambda: False)
    tb.handle_overlay_text(7, "/overlaytext 3 HELLO")
    assert any("isn't installed" in m for m in sent)


# ── auto-classification of the single overlay ─────────────────────────────────

def test_render_one_overlay_classifies_stat_vs_title(monkeypatch):
    captured = {}

    def fake_render(highlights, out_dir, fps=30, style=None, **kw):
        captured["h"] = highlights[0]
        return [{**highlights[0], "filepath": "x.mov", "start_sec": 0.0, "end_sec": 1.0}]

    monkeypatch.setattr(ovr, "render_overlay_clips", fake_render)

    ovr.render_one_overlay("200AH", 3, "out")
    assert captured["h"]["type"] == "stat" and captured["h"]["anim"] == "stat_pop"

    ovr.render_one_overlay("CHECK YOUR CAR BATTERY", 3, "out")
    assert captured["h"]["type"] == "title" and captured["h"]["anim"] == "title_card"

    # duration floor + carried through
    ovr.render_one_overlay("HELLO", 0.01, "out")
    assert captured["h"]["end"] == 0.4


# ── style presets ─────────────────────────────────────────────────────────────

def test_resolve_style_default_and_pick(monkeypatch):
    monkeypatch.delenv("OVERLAY_STYLE", raising=False)
    monkeypatch.delenv("OVERLAY_TEXT_COLOR", raising=False)
    monkeypatch.delenv("OVERLAY_ACCENT_COLOR", raising=False)
    assert ovr._resolve_style()["name"] == "bold_yellow"

    monkeypatch.setenv("OVERLAY_STYLE", "neon")
    s = ovr._resolve_style()
    assert s["name"] == "neon" and s["glow"] is True

    # Unknown name falls back to the default preset.
    monkeypatch.setenv("OVERLAY_STYLE", "does_not_exist")
    assert ovr._resolve_style()["name"] == "bold_yellow"


def test_resolve_style_color_override(monkeypatch):
    monkeypatch.setenv("OVERLAY_STYLE", "boxed_news")
    monkeypatch.setenv("OVERLAY_TEXT_COLOR", "#123456")
    s = ovr._resolve_style()
    assert s["name"] == "boxed_news" and s["color"] == "#123456" and s["box"] == "solid"


def test_props_carry_style_tokens(monkeypatch):
    monkeypatch.setenv("OVERLAY_STYLE", "clean_white")
    h = {"text": "HELLO", "type": "title", "anim": "title_card",
         "sfx": "none", "start": 0.0, "end": 2.0}
    props = ovr._props_for(h, 30, ovr._resolve_style())
    assert props["color"] == "#FFFFFF"
    assert props["style"]["name"] == "clean_white"
    assert props["style"]["upper"] is False and props["style"]["box"] == "none"
    # All Overlay.tsx style tokens are present.
    for k in ("weight", "upper", "box", "stroke", "glow", "radius", "name"):
        assert k in props["style"]
