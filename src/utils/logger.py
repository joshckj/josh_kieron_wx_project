"""One-line logging setup so every module logs in the same format."""
import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger. Safe to call more than once."""
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
        force=True,  # replace any handlers set earlier (e.g. by a notebook)
    )
