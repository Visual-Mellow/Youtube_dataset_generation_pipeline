import logging

from core.constants import LOG


def setup_logger(log_level: str) -> logging.Logger:
    """Configure root logging and return the shared youtube_links logger."""
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    LOG.setLevel(level)
    return LOG
