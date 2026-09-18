"""Registration failures, without automatic algorithm substitution."""


class InvalidRegistrationConfigError(ValueError):
    """Unsupported or invalid registration configuration."""


class RegistrationEstimationError(RuntimeError):
    """The selected estimator could not produce a valid transform."""


class InsufficientLandmarksError(RegistrationEstimationError):
    """Too few eligible landmarks or correspondences."""


class RegistrationBackendUnavailableError(ImportError):
    """An explicitly selected optional backend is unavailable."""


class UnsupportedTransformOperationError(NotImplementedError):
    """Transform conversion, inversion or application is unsupported."""
