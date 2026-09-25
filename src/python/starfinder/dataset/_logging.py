"""Logging utilities for FOV processing steps."""

import logging
from functools import wraps
from time import perf_counter

logger = logging.getLogger("starfinder")


def _step_name(name, kwargs):
    name = name.lstrip("_")
    return f"{name}:{kwargs['stage']}" if "stage" in kwargs else name


def _step_round(kwargs):
    rounds = kwargs.get("rounds")
    if rounds is not None and len(rounds) == 1:
        return rounds[0]
    return kwargs.get("round_name")


def _log_step(func):
    """Decorator to log FOV processing steps with timing.

    Parameters
    ----------
    func : callable
        FOV instance method; self must expose fov_id.

    Returns
    -------
    callable
        Wrapper preserving signature/metadata, return value and exceptions.
        Logs start, completion with elapsed seconds, or failure, to ``starfinder``.
        While FOV.run has checkpoints active, also appends the step name (with
        any ``stage`` keyword), round (a single-element ``rounds`` or
        ``round_name`` keyword), seconds and status to the run record, and
        rewrites run.json after each completed step.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        step_name = func.__name__
        record = getattr(self, "_run_record", None)
        logger.info(f"[{self.fov_id}] Starting {step_name}")
        start = perf_counter()
        try:
            result = func(self, *args, **kwargs)
        except BaseException as e:
            elapsed = perf_counter() - start
            logger.error(f"[{self.fov_id}] Failed {step_name}: {e}")
            if record is not None:
                name, round_name = _step_name(step_name, kwargs), _step_round(kwargs)
                record.add_step(name, round_name, elapsed, "failed" if isinstance(e, Exception) else "interrupted")
                record.fail_step(name, round_name)
            raise
        elapsed = perf_counter() - start
        logger.info(f"[{self.fov_id}] Completed {step_name} in {elapsed:.2f}s")
        if record is not None:
            record.add_step(_step_name(step_name, kwargs), _step_round(kwargs), elapsed, "succeeded")
            record.write()
        return result

    return wrapper
