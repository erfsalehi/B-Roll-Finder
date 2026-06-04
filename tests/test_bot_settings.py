"""Per-chat bot settings: persistence, toggling, env mapping."""

import os
import importlib


def _fresh(tmp_path, monkeypatch):
    from bot import settings as s
    importlib.reload(s)
    monkeypatch.setattr(s, "_SETTINGS_PATH", str(tmp_path / "bot_settings.json"))
    return s


def test_defaults_when_unset(tmp_path, monkeypatch):
    s = _fresh(tmp_path, monkeypatch)
    cfg = s.get_settings(42)
    assert cfg == s.DEFAULTS
    assert cfg["pexels_num"] == 3 and cfg["qa"] is True


def test_set_value_persists_and_merges(tmp_path, monkeypatch):
    s = _fresh(tmp_path, monkeypatch)
    s.set_value(7, "pexels_num", 8)
    # A second logical load still sees it, with other keys defaulted.
    cfg = s.get_settings(7)
    assert cfg["pexels_num"] == 8 and cfg["youtube_num"] == s.DEFAULTS["youtube_num"]


def test_toggle_bool_and_cycle_choice(tmp_path, monkeypatch):
    s = _fresh(tmp_path, monkeypatch)
    assert s.toggle(1, "qa")["qa"] is False
    assert s.toggle(1, "qa")["qa"] is True
    # choice cycles to the next value and wraps around
    first = s.get_settings(1)["pexels_num"]
    seen = {first}
    for _ in range(len(s._OPT_BY_KEY["pexels_num"]["choices"])):
        seen.add(s.toggle(1, "pexels_num")["pexels_num"])
    assert seen == set(s._OPT_BY_KEY["pexels_num"]["choices"])


def test_reset_clears_overrides(tmp_path, monkeypatch):
    s = _fresh(tmp_path, monkeypatch)
    s.set_value(5, "quality", 2160)
    assert s.get_settings(5)["quality"] == 2160
    assert s.reset(5)["quality"] == s.DEFAULTS["quality"]


def test_env_overrides_types(tmp_path, monkeypatch):
    s = _fresh(tmp_path, monkeypatch)
    cfg = dict(s.DEFAULTS, use_pexels=False, pexels_num=10, qa=True)
    env = s.env_overrides(cfg)
    assert env["AUTO_USE_PEXELS"] == "false"
    assert env["AUTO_PEXELS_NUM"] == "10"
    assert env["ENABLE_QA_REVIEW"] == "true"
    # quality/review_gate/auto_refine are NOT env-mapped (consumed directly)
    assert "quality" not in env and all("REVIEW_GATE" not in k for k in env)


def test_apply_env_sets_and_restores(tmp_path, monkeypatch):
    s = _fresh(tmp_path, monkeypatch)
    monkeypatch.setenv("AUTO_PEXELS_NUM", "999")
    monkeypatch.delenv("AUTO_USE_YOUTUBE", raising=False)
    cfg = dict(s.DEFAULTS, pexels_num=2, use_youtube=False)
    with s.apply_env(cfg):
        assert os.environ["AUTO_PEXELS_NUM"] == "2"
        assert os.environ["AUTO_USE_YOUTUBE"] == "false"
    # prior value restored; previously-unset var removed again
    assert os.environ["AUTO_PEXELS_NUM"] == "999"
    assert "AUTO_USE_YOUTUBE" not in os.environ


def test_display_value(tmp_path, monkeypatch):
    s = _fresh(tmp_path, monkeypatch)
    qa_opt = s._OPT_BY_KEY["qa"]
    mh_opt = s._OPT_BY_KEY["min_height"]
    assert "on" in s.display_value(qa_opt, True) and "off" in s.display_value(qa_opt, False)
    assert s.display_value(mh_opt, 0) == "off" and s.display_value(mh_opt, 720) == "720p"
