import time
import json
import logging
import pathlib

from types import SimpleNamespace
from decorator import decorator
from typing import Any, Callable, Optional

from dirs import LOG_DIR
from .typehint import P, R

# basic logging config
LOG_FORMAT = "%(levelname)s: %(asctime)s: %(name)s: %(message)s"
logging.basicConfig(
    filename=LOG_DIR / "general.log", level=logging.INFO, format=LOG_FORMAT
)


@decorator
def loggit(
    fn: Callable[P, R], logger: Optional[logging.Logger] = None, *args, **kwds
) -> R:

    logfn = logging.info if logger is None else logger.info
    result = fn(*args, **kwds)
    logfn(repr(result))
    return result


@decorator
def timeit(fn: Callable[P, R], *args, **kwds) -> tuple[R, float]:

    start, result, end = time.time(), fn(*args, **kwds), time.time()
    return result, end - start


def read_json(
    json_name: str | pathlib.Path, as_namespace: bool = False
) -> dict[Any, Any] | SimpleNamespace:

    with open(json_name) as json_file:
        result = json.load(json_file)
    return result if not as_namespace else SimpleNamespace(**result)


def create_logger(
    name: str, logfile: str, level: int | str = logging.DEBUG
) -> logging.Logger:

    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(LOG_DIR / logfile)
    formatter = logging.Formatter(LOG_FORMAT)

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(level)

    return logger
