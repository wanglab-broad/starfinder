"""Spot-finding errors and the spot-finding warning."""


class SpotFindingBackendUnavailableError(ImportError):
    """An optional dependency of a spot-finding method (its extra) is not installed."""


class SpotFindingError(ValueError):
    """A detection refuses its input: a ``noise``-mode channel with signal whose noise MAD is 0 (W-789)."""


class SpotFindingWarning(UserWarning):
    """A detection ran but something about its input deserves attention (for example a noise estimated above the
    channel's minimum)."""


class MissingModelError(FileNotFoundError):
    """The weights folder of a known model, or a file it lists, is missing; fetch it explicitly."""


class ModelHashMismatchError(ValueError):
    """A weights file's SHA-256 differs from the known-weights table."""
