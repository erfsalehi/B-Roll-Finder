"""Process-wide resource limits for CPU-only servers.

The Telegram bot and the Streamlit UI both run on a shared-vCPU box with no
GPU (e.g. the Hetzner deployment). Left to their defaults, the native BLAS /
OpenMP pools behind numpy and onnxruntime (fastembed's Clip Library embedder)
size themselves to *all* logical cores, so embedding work can grab every vCPU
while concurrent yt-dlp downloads and ffmpeg encodes also run, and the box
thrashes instead of doing more work.

``configure_runtime()`` caps those native thread pools to a sane fraction of the
cores. Call it once, as early as possible in each entry point (before numpy /
onnxruntime import), so the ``*_NUM_THREADS`` env vars are read at import time.
"""

import os

_configured = False


def default_threads() -> int:
    """Half the logical cores, clamped to [1, 4].

    Half leaves room for the download/ffmpeg work that runs alongside embedding;
    the cap of 4 stops a big box from spinning up more BLAS threads than the
    small MiniLM embedder can actually use.
    """
    try:
        n = os.cpu_count() or 4
    except Exception:
        n = 4
    return max(1, min(4, n // 2))


def configure_runtime() -> int:
    """Pin native thread pools. Idempotent; returns the chosen thread count.

    ``BROLL_TORCH_THREADS`` overrides the default if set to a positive integer.
    (Name kept for backwards-compat with existing .env files; it now governs the
    OpenMP/BLAS pools used by numpy and onnxruntime.)
    """
    global _configured

    override = os.getenv("BROLL_TORCH_THREADS", "").strip()
    threads = int(override) if override.isdigit() and int(override) > 0 else default_threads()

    # These must be set before numpy / onnxruntime load their native libraries —
    # hence the early call in each entry point. setdefault so an explicit env
    # from the operator always wins.
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))

    _configured = True
    return threads
