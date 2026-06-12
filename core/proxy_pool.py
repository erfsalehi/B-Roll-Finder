"""Validated YouTube proxy pool.

A free proxy list (``YT_DLP_PROXY_URL``) is mostly dead — round-robining 2000 raw
proxies makes downloads crawl, since most attempts time out before hitting a live
one. This keeps a small pool of proxies *actually validated against YouTube* (with
cookies, matching how downloads run), so the pipeline only ever uses known-good
IPs. It researches the raw list on demand to (re)fill the pool, and replenishes
in the background as proxies die.

Flow:
* ``ensure_working()`` — called before a run; blocks until ``PROXY_POOL_SIZE``
  working proxies are found (or the raw list is exhausted).
* ``get_proxy()`` — round-robins the validated pool (lazily researches if empty).
* ``mark_dead()`` — drops a proxy that failed mid-download and kicks off a
  background top-up so the pool stays full.

Env knobs: ``PROXY_POOL_SIZE`` (5), ``PROXY_VALIDATE_WORKERS`` (25),
``PROXY_VALIDATE_TIMEOUT`` (10s), ``PROXY_RESEARCH_MAX`` (200 checks/research),
``PROXY_TEST_URL`` (validation video). Pool mode is active whenever
``YT_DLP_PROXY_URL`` is set.
"""

import concurrent.futures
import os
import random
import threading
import time

_lock = threading.Lock()           # guards _working / _dead / _rr
_research_lock = threading.Lock()  # serializes research() so workers don't dogpile
_working: list = []                # validated, good proxies (round-robin order)
_dead: set = set()                 # proxies that failed validation/download this session
_last_validated: dict = {}         # proxy -> epoch seconds
_rr = 0                            # round-robin cursor over _working


def _cfg_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def pool_active() -> bool:
    """True when a dynamic proxy list is configured — the only case where the
    validated pool is worth the trouble (a short static YT_DLP_PROXY list is used
    directly)."""
    return bool((os.getenv("YT_DLP_PROXY_URL") or "").strip())


def target_size() -> int:
    return max(1, _cfg_int("PROXY_POOL_SIZE", 5))


def _default_test_url() -> str:
    # yt-dlp's classic, long-lived public test video — a stable validation target.
    return (os.getenv("PROXY_TEST_URL") or "").strip() or \
        "https://www.youtube.com/watch?v=BaW_jenozKc"


def _use_cookies() -> bool:
    return os.getenv("YT_DOWNLOAD_NO_COOKIES", "").strip().lower() not in \
        ("1", "true", "yes", "on")


def _raw_proxies() -> list:
    """The full configured pool (inline + fetched list), de-duped. Network-free
    except for the cached list fetch inside youtube._dynamic_youtube_proxies."""
    from core.youtube import _static_youtube_proxies, _dynamic_youtube_proxies
    seen, out = set(), []
    for p in _static_youtube_proxies() + _dynamic_youtube_proxies():
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def research(need: int = None, test_url: str = None, progress=None) -> int:
    """Probe raw proxies (concurrently, with cookies when downloads use them)
    until the pool holds ``need`` working ones — or ``PROXY_RESEARCH_MAX`` checks
    are spent / the list is exhausted. Serialized so parallel callers don't all
    research at once. Returns how many new working proxies were added."""
    if not pool_active():
        return 0
    need = need or target_size()
    test_url = test_url or _default_test_url()

    with _research_lock:
        with _lock:
            if len(_working) >= need:
                return 0
            known = set(_working) | set(_dead)
        candidates = [p for p in _raw_proxies() if p not in known]
        random.shuffle(candidates)
        candidates = candidates[:_cfg_int("PROXY_RESEARCH_MAX", 200)]
        if not candidates:
            return 0

        from core.youtube import probe_proxy
        workers = max(1, _cfg_int("PROXY_VALIDATE_WORKERS", 25))
        timeout = _cfg_int("PROXY_VALIDATE_TIMEOUT", 10)
        use_cookies = _use_cookies()
        added = checked = 0
        it = iter(candidates)
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=workers,
                                                   thread_name_prefix="proxyval")
        pending: dict = {}

        def _submit_more():
            while len(pending) < workers:
                try:
                    p = next(it)
                except StopIteration:
                    return
                pending[ex.submit(probe_proxy, test_url, p, timeout, use_cookies)] = p

        try:
            _submit_more()
            while pending:
                done, _ = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    p = pending.pop(f)
                    checked += 1
                    try:
                        ok = bool(f.result()[0])
                    except Exception:
                        ok = False
                    with _lock:
                        if ok:
                            if p not in _working:
                                _working.append(p)
                                _last_validated[p] = time.time()
                                added += 1
                        else:
                            _dead.add(p)
                        have = len(_working)
                    if progress:
                        try:
                            progress(f"Testing proxies… {checked} checked, {have} working")
                        except Exception:
                            pass
                if have >= need:
                    break
                _submit_more()
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        return added


def ensure_working(min_count: int = None, test_url: str = None, progress=None) -> list:
    """Guarantee at least ``min_count`` (default PROXY_POOL_SIZE) validated
    proxies before a run touches YouTube. Returns the working snapshot."""
    if not pool_active():
        return []
    min_count = min_count or target_size()
    with _lock:
        have = len(_working)
    if have < min_count:
        research(min_count, test_url, progress)
    return working_snapshot()


def get_proxy() -> str:
    """Round-robin one validated proxy; lazily research if the pool is empty.
    Returns '' when pool mode is off or nothing validated."""
    if not pool_active():
        return ""
    with _lock:
        empty = not _working
    if empty:
        research()
    global _rr
    with _lock:
        if not _working:
            return ""
        p = _working[_rr % len(_working)]
        _rr += 1
        return p


def mark_dead(proxy: str) -> None:
    """Drop a proxy that failed during a real download and, if the pool fell below
    target, top it up in the background (don't block the current download)."""
    if not proxy:
        return
    with _lock:
        if proxy in _working:
            _working.remove(proxy)
        _dead.add(proxy)
        low = len(_working) < target_size()
    if low and pool_active():
        threading.Thread(target=lambda: research(), daemon=True,
                         name="ProxyResearch").start()


def refresh(progress=None) -> int:
    """Forget everything (working + dead) and re-research from scratch — for a
    manual /proxies refresh. Returns the new working count."""
    with _lock:
        _working.clear()
        _dead.clear()
        _last_validated.clear()
    research(progress=progress)
    return len(working_snapshot())


def working_snapshot() -> list:
    with _lock:
        return list(_working)


def stats() -> dict:
    raw = len(_raw_proxies()) if pool_active() else 0
    with _lock:
        return {"working": len(_working), "dead": len(_dead), "raw": raw,
                "active": pool_active()}


def _reset() -> None:
    """Test helper: clear all state."""
    global _rr
    with _lock:
        _working.clear()
        _dead.clear()
        _last_validated.clear()
        _rr = 0
