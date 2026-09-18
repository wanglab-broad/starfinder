starfinder.registration
=======================

Estimate once, then apply the returned transform with its application config.
Translations store correction vectors; dense fields are pull fields. See
:doc:`contracts` and :doc:`backends` for geometry, errors and supported dimensions.

.. currentmodule:: starfinder.registration

.. autosummary::
   :toctree: generated

   CpdConfig
   DemonsConfig
   DenseDisplacementTransform
   InsufficientLandmarksError
   InvalidRegistrationConfigError
   RegistrationBackendUnavailableError
   RegistrationDiagnostics
   RegistrationEstimationError
   RegistrationResult
   TpsConfig
   TranslationConfig
   TranslationTransform
   UnsupportedTransformOperationError
   WarpConfig
   apply_transform
   estimate_transform
