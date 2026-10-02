import pytest


@pytest.fixture(autouse=True)
def _reset_module_caches():
    """Module-level run state that would otherwise leak between tests: the
    extras entity cache and the image-search circuit breaker."""
    import core.extras as extras
    import core.related_images as ri
    extras._ENTITY_CACHE.clear()
    ri.reset_image_backends()
    yield
    extras._ENTITY_CACHE.clear()
    ri.reset_image_backends()


@pytest.fixture(autouse=True)
def _accept_fake_clip_files(monkeypatch):
    """Download tests write placeholder bytes, not real videos; skip the ffprobe
    playability check for them (tests of the check itself call it directly)."""
    import core.pipeline as pl
    monkeypatch.setattr(pl, "_verify_clip_file", lambda path: (True, ""))
    monkeypatch.setenv("ENABLE_BAR_CHECK", "false")      # placeholder bytes have no picture to measure


@pytest.fixture(autouse=True)
def _storyboard_off_and_isolated(monkeypatch, tmp_path):
    """The optional storyboard check is off unless a test switches it on, never
    touches the real .cache/storyboards, and starts each test with a clean tally."""
    import core.storyboard as sb
    monkeypatch.delenv("ENABLE_STORYBOARD_CHECK", raising=False)
    monkeypatch.setenv("STORYBOARD_CACHE_DIR", str(tmp_path / "storyboards"))
    sb.reset_run_stats()
    yield
    sb.reset_run_stats()


@pytest.fixture(autouse=True)
def _isolated_project_store(monkeypatch, tmp_path):
    """Never let a test write the real .cache/projects.db (or reuse another
    test's learned edit history)."""
    import core.edit_feedback as ef
    monkeypatch.setenv("PROJECTS_DB", str(tmp_path / "projects.db"))
    ef._CACHE.update(stamp=None, history=None)
