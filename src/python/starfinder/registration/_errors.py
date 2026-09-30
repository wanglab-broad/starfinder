"""Registration failures, without automatic algorithm substitution."""


class InvalidRegistrationConfigError(ValueError):
    """Unsupported or invalid registration configuration."""


class RegistrationEstimationError(RuntimeError):
    """The selected estimator could not produce a valid transform."""


class InsufficientLandmarksError(RegistrationEstimationError):
    """Too few eligible landmarks or correspondences."""


class RegistrationRejectedError(RegistrationEstimationError):
    """A step failed a configured routine QC criterion (RegistrationQcConfig).

    The message names the criterion, the value and the bound; recovery may
    allow it like any other estimation error.
    """


class RegistrationBackendUnavailableError(ImportError):
    """An explicitly selected optional backend is unavailable."""


class UnsupportedTransformOperationError(NotImplementedError):
    """Transform conversion, inversion or application is unsupported."""
