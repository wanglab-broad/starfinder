"""Logging utilities for FOV processing steps."""

import logging
from contextlib import nullcontext
import inspect
from functools import wraps
from time import perf_counter

logger = logging.getLogger("starfinder")


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
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        recorder = kwargs.get('provenance') or getattr(self, '_provenance', None)
        if func.__name__ == 'run' and recorder is not None:
            from starfinder.provenance import RunRecorder
            if not isinstance(recorder, RunRecorder):
                raise TypeError('provenance must be RunRecorder')
            bound = inspect.signature(func).bind(self, *args, **kwargs)
            bound.apply_defaults()
            with recorder._bind(self, bound.arguments['config'], bound.arguments['execution']):
                return observed(self, args, kwargs, recorder)
        return observed(self, args, kwargs, recorder)

    def observed(self, args, kwargs, recorder):
        step_name = func.__name__
        logger.info(f"[{self.fov_id}] Starting {step_name}")
        start = perf_counter()
        try:
            context = nullcontext()
            if recorder is not None and step_name != 'run':
                bound = inspect.signature(func).bind(self, *args, **kwargs)
                bound.apply_defaults()
                parameters = {k: v for k, v in bound.arguments.items()
                              if k not in ('self', 'provenance', 'reference')}
                rounds = parameters.get('rounds')
                round_name = parameters.get('round_name')
                if rounds is not None and len(rounds) == 1:
                    round_name = rounds[0]
                if step_name == 'find_spots':
                    round_name = self.rounds.reference_round
                context = recorder._operation(step_name, parameters, round_name=round_name)
            with context as diagnostics:
                result = func(self, *args, **kwargs)
                if diagnostics is not None:
                    diagnostics['state'] = recorder._encode(recorder._snapshot(self))
            elapsed = perf_counter() - start
            logger.info(f"[{self.fov_id}] Completed {step_name} in {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.error(f"[{self.fov_id}] Failed {step_name}: {e}")
            raise

    return wrapper
