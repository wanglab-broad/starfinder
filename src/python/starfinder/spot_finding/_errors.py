"""Spot-finding errors and the spot-finding warning."""


class SpotFindingBackendUnavailableError(ImportError):
    """An optional dependency of a spot-finding method (its extra) is not installed."""


class SpotFindingWarning(UserWarning):
    """A detection ran but something about its input deserves attention (for example a noise MAD of 0)."""


class MissingWeightsError(FileNotFoundError):
    """The weights folder of a known model, or a file it lists, is missing; fetch it explicitly."""


class WeightsHashMismatchError(ValueError):
    """A weights file's SHA-256 differs from the known-weights table."""
