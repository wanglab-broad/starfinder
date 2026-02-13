"""Logging utilities for FOV processing steps."""

import logging
from functools import wraps
from time import perf_counter

logger = logging.getLogger("starfinder")


def log_step(func):
    """Decorator to log FOV processing steps with timing."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        step_name = func.__name__
        logger.info(f"[{self.fov_id}] Starting {step_name}")
        start = perf_counter()
        try:
            result = func(self, *args, **kwargs)
            elapsed = perf_counter() - start
            logger.info(f"[{self.fov_id}] Completed {step_name} in {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.error(f"[{self.fov_id}] Failed {step_name}: {e}")
            raise

    return wrapper
