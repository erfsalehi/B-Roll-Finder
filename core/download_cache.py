"""Cross-session registry of URL → already-downloaded file path.

The director downloads many MB of stock footage per script. Re-running
the app on the same script (or even a different one that picks an
overlapping clip) shouldn't fetch the same MP4 twice. This registry
remembers what we've already pulled, and Step 6 consults it before
queueing a download.

The registry is a tiny JSON file at ``.cache/downloads_registry.json``.
Stale entries (file no longer on disk) are pruned lazily on lookup.
Concurrent writes are serialized with a module-level lock — ample for
the single-process Streamlit app.
"""

import json
import os
import threading
import time
from typing import Optional

REGISTRY_FILE = os.path.join(".cache", "downloads_registry.json")
_lock = threading.Lock()


def _load() -> dict:
    if not os.path.exists(REGISTRY_FILE):
        return {}
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(registry: dict) -> None:
    try:
        os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)
        # Write to a temp file then rename — avoids leaving a half-written
        # registry on disk if the process is killed mid-write.
        tmp_path = REGISTRY_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        os.replace(tmp_path, REGISTRY_FILE)
    except Exception as e:
        print(f"download_cache: could not persist registry: {e}")


def lookup_path(url: str) -> Optional[str]:
    """Return the on-disk path for ``url`` if we have it cached AND the
    file is still there. Stale entries are pruned automatically."""
    if not url:
        return None
    with _lock:
        registry = _load()
        entry = registry.get(url)
        if not entry:
            return None
        path = entry.get("path") if isinstance(entry, dict) else None
        if path and os.path.exists(path):
            return path
        # Stale — file is gone. Drop the entry.
        registry.pop(url, None)
        _save(registry)
        return None


def register(url: str, path: str) -> None:
    """Record that ``url`` has been successfully downloaded to ``path``."""
    if not url or not path or not os.path.exists(path):
        return
    with _lock:
        registry = _load()
        try:
            size = os.path.getsize(path)
        except OSError:
            size = None
        registry[url] = {
            "path":          path,
            "size":          size,
            "downloaded_at": time.time(),
        }
        _save(registry)


def forget(url: str) -> None:
    """Drop a single URL from the registry (e.g. after the file is deleted)."""
    if not url:
        return
    with _lock:
        registry = _load()
        if url in registry:
            registry.pop(url)
            _save(registry)


def stats() -> dict:
    """Return summary stats: count, total size, plus how many entries
    point at files that are now missing."""
    with _lock:
        registry = _load()
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
