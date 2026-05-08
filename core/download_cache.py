"""Cross-session registry of URL → already-downloaded file path.

The director downloads many MB of stock footage per script. Re-running
the app on the same script (or even a different one that picks an
overlapping clip) shouldn't fetch the same MP4 twice. This registry
remembers what we've already pulled, and Step 6 consults it before
queueing a download.

The registry is a tiny JSON file at ``.cache/downloads_registry.json``.
Stale entries (file no longer on disk) are pruned lazily on lookup.

Reads go through an in-memory cache that's invalidated whenever the
on-disk file's mtime changes — so external writes (e.g. a parallel
Streamlit reload, a manual edit) are picked up on the next call. Our
own writes update the in-memory cache atomically. Concurrent operations
within the process are serialized via a module-level lock.
"""

import json
import os
import threading
import time
from typing import Optional

REGISTRY_FILE = os.path.join(".cache", "downloads_registry.json")
_lock = threading.Lock()

# In-memory copy of the registry. ``None`` means "not loaded yet".
# We track the file's mtime alongside so we can detect external writes
# and re-read on demand.
_cache: Optional[dict] = None
_cache_mtime: Optional[float] = None


def _read_from_disk() -> dict:
    if not os.path.exists(REGISTRY_FILE):
        return {}
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _current_mtime() -> Optional[float]:
    try:
        return os.path.getmtime(REGISTRY_FILE)
    except OSError:
        return None


def _get_registry() -> dict:
    """Return the in-memory registry, re-reading from disk only when
    the file's mtime no longer matches what we last saw."""
    global _cache, _cache_mtime
    current = _current_mtime()
    if _cache is None or current != _cache_mtime:
        _cache = _read_from_disk()
        _cache_mtime = current
    return _cache


def _save(registry: dict) -> None:
    """Persist atomically (write to .tmp, rename) and refresh the
    in-memory cache so the next read doesn't bounce off the mtime
    check we just triggered."""
    global _cache, _cache_mtime
    try:
        os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)
        tmp_path = REGISTRY_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        os.replace(tmp_path, REGISTRY_FILE)
        _cache = registry
        _cache_mtime = _current_mtime()
    except Exception as e:
        print(f"download_cache: could not persist registry: {e}")


def lookup_path(url: str) -> Optional[str]:
    """Return the on-disk path for ``url`` if we have it cached AND the
    file is still there. Stale entries are pruned automatically."""
    if not url:
        return None
    with _lock:
        registry = _get_registry()
        entry = registry.get(url)
        if not entry:
            return None
        path = entry.get("path") if isinstance(entry, dict) else None
        if path and os.path.exists(path):
            return path
        # Stale — file is gone. Work on a copy so iteration on the live
        # cache elsewhere can't see a half-mutated dict, then persist.
        new_registry = dict(registry)
        new_registry.pop(url, None)
        _save(new_registry)
        return None


def register(url: str, path: str) -> None:
    """Record that ``url`` has been successfully downloaded to ``path``."""
    if not url or not path or not os.path.exists(path):
        return
    with _lock:
        new_registry = dict(_get_registry())
        try:
            size = os.path.getsize(path)
        except OSError:
            size = None
        new_registry[url] = {
            "path":          path,
            "size":          size,
            "downloaded_at": time.time(),
        }
        _save(new_registry)


def forget(url: str) -> None:
    """Drop a single URL from the registry (e.g. after the file is deleted)."""
    if not url:
        return
    with _lock:
        registry = _get_registry()
        if url not in registry:
            return
        new_registry = dict(registry)
        new_registry.pop(url)
        _save(new_registry)


def stats() -> dict:
    """Return summary stats: count, total size, plus how many entries
    point at files that are now missing.

    Note: ``size_bytes`` reflects only entries whose file still exists
    on disk; stale entries are counted but not summed.
    """
    with _lock:
        registry = dict(_get_registry())  # copy so we can iterate post-lock
    count = 0
    total = 0
    stale = 0
    for url, entry in registry.items():
        if not isinstance(entry, dict):
            continue
        count += 1
        path = entry.get("path", "")
        if path and os.path.exists(path):
            total += entry.get("size") or 0
        else:
            stale += 1
    return {"count": count, "size_bytes": total, "stale": stale}


def clear() -> None:
    """Forget everything. Doesn't touch the actual files on disk."""
    with _lock:
        _save({})


def _reset_for_tests() -> None:
    """Clear the in-memory cache (test-only — never call in app code)."""
    global _cache, _cache_mtime
    _cache = None
    _cache_mtime = None
