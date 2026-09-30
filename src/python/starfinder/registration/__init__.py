"""Typed transform estimation and application in voxel-index coordinates."""

from ._api import apply_transform, estimate_transform
from ._config import (
    CpdConfig,
    DemonsConfig,
    RegistrationQcConfig,
    TpsConfig,
    TranslationConfig,
    WarpConfig,
)
from ._errors import (
    InsufficientLandmarksError,
    InvalidRegistrationConfigError,
    RegistrationBackendUnavailableError,
    RegistrationEstimationError,
    RegistrationRejectedError,
    UnsupportedTransformOperationError,
)
from ._methods import REGISTRATION_METHODS as _REGISTRATION_METHODS
from ._methods import RegistrationSpec
from ._types import (
    DenseDisplacementTransform,
    RegistrationDiagnostics,
    RegistrationResult,
    TranslationTransform,
)

#: The registration method registry, mapping each exact frozen config type to its
#: RegistrationSpec. estimate_transform, RegistrationStep, RecoveryConfig, the workflow
#: adapter, the checkpoint reader and the benchmark adapter derive their method sets from it.
REGISTRATION_METHODS = _REGISTRATION_METHODS

__all__ = [
    "estimate_transform",
    "apply_transform",
    "TranslationConfig",
    "DemonsConfig",
    "TpsConfig",
    "CpdConfig",
    "WarpConfig",
    "RegistrationQcConfig",
    "REGISTRATION_METHODS",
    "RegistrationSpec",
    "TranslationTransform",
    "DenseDisplacementTransform",
    "RegistrationResult",
    "RegistrationDiagnostics",
    "InvalidRegistrationConfigError",
    "RegistrationEstimationError",
    "InsufficientLandmarksError",
    "RegistrationRejectedError",
    "RegistrationBackendUnavailableError",
    "UnsupportedTransformOperationError",
]
