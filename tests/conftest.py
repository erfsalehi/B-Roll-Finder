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
def _isolated_project_store(monkeypatch, tmp_path):
    """Never let a test write the real .cache/projects.db."""
    monkeypatch.setenv("PROJECTS_DB", str(tmp_path / "projects.db"))
