"""Typed transform estimation and application in voxel-index coordinates."""

from ._api import apply_transform, estimate_transform
from ._config import (
    AffineConfig,
    BSplineConfig,
    CpdConfig,
    DemonsConfig,
    RigidConfig,
    RegistrationQcConfig,
    RegistrationSignalConfig,
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
from ._chain import TransformChain
from ._methods import REGISTRATION_METHODS as _REGISTRATION_METHODS
from ._methods import RegistrationSpec
from ._types import (
    AffineTransform,
    BSplineTransform,
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
    "RigidConfig",
    "AffineConfig",
    "BSplineConfig",
    "DemonsConfig",
    "TpsConfig",
    "CpdConfig",
    "WarpConfig",
    "RegistrationQcConfig",
    "RegistrationSignalConfig",
    "REGISTRATION_METHODS",
    "RegistrationSpec",
    "TranslationTransform",
    "AffineTransform",
    "BSplineTransform",
    "DenseDisplacementTransform",
    "TransformChain",
    "RegistrationResult",
    "RegistrationDiagnostics",
    "InvalidRegistrationConfigError",
    "RegistrationEstimationError",
    "InsufficientLandmarksError",
    "RegistrationRejectedError",
    "RegistrationBackendUnavailableError",
    "UnsupportedTransformOperationError",
]
