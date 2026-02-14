"""Environment setup, output suppression, and logging for voiss.

setup_environment() must be called before importing MLX or transformers
to suppress noisy warnings and progress bars in a real-time CLI context.
"""

import contextlib
import io
import logging
import os
import warnings
from collections.abc import Generator

LOGGER = logging.getLogger("speech")


def setup_environment() -> None:
    """Configure warning filters and env vars before library imports."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


@contextlib.contextmanager
def suppress_output() -> Generator[None, None, None]:
    """Hide noisy library prints/warnings during background inference.

    Redirects fd-level stderr and Python-level stdout/stderr to devnull
    so that library code writing directly to file descriptors is silenced.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    stderr_fd = os.dup(2)
    try:
        os.dup2(devnull, 2)
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            yield
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull)
