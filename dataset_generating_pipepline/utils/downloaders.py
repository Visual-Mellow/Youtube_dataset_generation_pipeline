from core.constants import *

def download_error_text(exc: BaseException) -> str:
    """One CSV-sized reason. The batch log uses this same sentence."""
    raw = str(exc).strip()
    line = raw.splitlines()[-1] if raw else exc.__class__.__name__
    while True:
        if line.lower().startswith("video is not downloadable:"):
            line = line.split(":", 1)[1].strip()
            continue
        if line.startswith("ERROR:"):
            line = line.removeprefix("ERROR:").strip()
            continue
        break
    if len(line) > 300:
        line = line[:300]
    return f"video is not downloadable: {line}"
