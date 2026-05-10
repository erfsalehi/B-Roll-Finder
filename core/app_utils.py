import socket


def check_network() -> bool:
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except Exception:
        return False


def format_speed(bps) -> str:
    if not bps:
        return ""
    if bps >= 1_048_576:
        return f"{bps / 1_048_576:.1f} MB/s"
    return f"{bps / 1024:.0f} KB/s"


def format_eta(seconds) -> str:
    if seconds is None:
        return ""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60}s"
