"""Capture the bot's stdout/stderr to a file so it can be exported via /logs.

The codebase logs with plain ``print()`` (→ stdout, which on the server goes to
journald/`docker logs`). To let the operator pull logs from inside Telegram, we
tee stdout+stderr into ``.cache/bot.log``. The file is size-capped: when it
exceeds ``max_bytes`` it rotates once to ``bot.log.1`` and starts fresh, so it
never grows without bound. ``snapshot_logs`` stitches the rotated + current file
into a single text file for sending.

Each bot launch starts a FRESH log: the previous run's ``bot.log`` is archived
to ``bot.log.prev`` (kept for forensics but NOT exported by ``/logs``) and the
stale within-session ``bot.log.1`` is cleared, so ``/logs`` always reflects the
current run instead of accumulating every past session's noise.
"""

import logging
import os
import sys
import threading
import time

LOG_PATH = os.path.join(".cache", "bot.log")
_PREV_PATH = LOG_PATH + ".prev"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file


def silence_noisy_loggers() -> None:
    """Quiet third-party log spam that's meaningless in the bot/headless context.

    Streamlit emits a WARNING ('missing ScriptRunContext!') every time one of its
    functions runs in a worker thread outside a Streamlit script run — which the
    bot's background job threads trip constantly. It's pure noise here and was a
    big chunk of what cluttered /logs, so pin those loggers to ERROR."""
    for name in (
        "streamlit",
        "streamlit.runtime.scriptrunner_utils.script_run_context",
        "streamlit.runtime.scriptrunner.script_run_context",
        "streamlit.runtime.state.session_state_proxy",
    ):
        try:
            logging.getLogger(name).setLevel(logging.ERROR)
        except Exception:
            pass


class _Tee:
    """Write-through to the real stream *and* the log file."""
    def __init__(self, stream, logger):
        self._stream = stream
        self._logger = logger

    def write(self, data):
        try:
            self._stream.write(data)
        except Exception:
            pass
        self._logger._write(data)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


class _FileLogger:
    def __init__(self, path, max_bytes):
        self.path = path
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8", errors="replace")
        self._writes = 0

    def _write(self, data):
        with self._lock:
            try:
                self._fh.write(data)
                self._fh.flush()
                self._writes += 1
                if self._writes % 200 == 0 and self._fh.tell() > self.max_bytes:
                    self._rotate()
            except Exception:
                pass

    def _rotate(self):
        try:
            self._fh.close()
            prev = self.path + ".1"
            if os.path.exists(prev):
                os.remove(prev)
            os.replace(self.path, prev)
        except Exception:
            pass
        self._fh = open(self.path, "a", encoding="utf-8", errors="replace")


_logger = None


def _start_fresh_session(path: str) -> None:
    """Archive the previous run's log and clear the stale rotation so this
    session starts clean. Best-effort — never raises."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    except Exception:
        pass
    # Clear the previous session's mid-run rotation so it can't leak into /logs.
    try:
        if os.path.exists(path + ".1"):
            os.remove(path + ".1")
    except Exception:
        pass
    # Archive the previous session's log to .prev (overwrites the older archive).
    try:
        if os.path.exists(path):
            if os.path.exists(_PREV_PATH):
                os.remove(_PREV_PATH)
            os.replace(path, _PREV_PATH)
    except Exception:
        pass


def install_file_logging(path: str = LOG_PATH, max_bytes: int = _MAX_BYTES) -> str:
    """Tee stdout/stderr into ``path``, starting a fresh per-session log.
    Idempotent. Returns the log path."""
    global _logger
    if _logger is not None:
        return _logger.path
    silence_noisy_loggers()
    _start_fresh_session(path)
    _logger = _FileLogger(path, max_bytes)
    sys.stdout = _Tee(sys.__stdout__, _logger)
    sys.stderr = _Tee(sys.__stderr__, _logger)
    banner = (f"===== B-Roll bot session started "
              f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    try:
        _logger._write(banner)
    except Exception:
        pass
    return path


def snapshot_logs(dest: str, max_bytes: int = _MAX_BYTES) -> str | None:
    """Write the rotated + current log (tail, up to ``max_bytes``) to ``dest``.
    Returns ``dest`` if anything was written, else None."""
    if _logger is not None:
        try:
            _logger._fh.flush()
        except Exception:
            pass
    parts = []
    for p in (LOG_PATH + ".1", LOG_PATH):
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    parts.append(f.read())
            except Exception:
                pass
    if not parts:
        return None
    blob = b"".join(parts)[-max_bytes:]
    try:
        with open(dest, "wb") as f:
            f.write(blob)
        return dest
    except Exception:
        return None
