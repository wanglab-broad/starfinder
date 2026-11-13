"""Spot-finding errors and the spot-finding warning."""


class SpotFindingBackendUnavailableError(ImportError):
    """An optional dependency of a spot-finding method (its extra) is not installed."""


class SpotFindingError(ValueError):
    """A detection refuses its input: a ``noise``-mode floating-point channel with signal whose noise MAD is 0 (W-789;
    an integer channel has its MAD floored at one grey level instead, W-848)."""


class SpotFindingWarning(UserWarning):
    """A detection ran but something about its input deserves attention (for example a noise estimated above the
    channel's minimum, or a noise MAD that the one-grey-level floor of an integer image replaced)."""


class MissingModelError(FileNotFoundError):
    """The weights folder of a known model, or a file it lists, is missing; fetch it explicitly."""


class ModelHashMismatchError(ValueError):
    """A weights file's SHA-256 differs from the known-weights table."""
