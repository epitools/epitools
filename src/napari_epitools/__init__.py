from __future__ import annotations

import logging

__all__: list[str] = []
__version__ = "0.0.1"

# get the logger instance
logger = logging.getLogger(__name__)

formatter = logging.Formatter(
    "%(levelname)s [%(asctime)s] epitools: %(message)s",
    datefmt="%Y/%m/%d %I:%M:%S %p",
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.setLevel("INFO")
logger.propagate = False
