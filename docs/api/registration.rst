starfinder.registration
=======================

Estimate once, then apply the returned transform with its application config.
Translations store correction vectors; dense fields are pull fields. See
:doc:`contracts` and :doc:`backends` for geometry, errors and supported dimensions.

:py:data:`~starfinder.registration.REGISTRATION_METHODS` maps each exact config type to
its :py:class:`~starfinder.registration.RegistrationSpec`: the method name (the config's
``method`` value), step kind, dimensions, minimum shape, transform kind, estimation space
and optional dependencies. :py:func:`~starfinder.registration.estimate_transform`, the
dataset configs, the workflow adapter, the checkpoint reader and the benchmark adapter
derive their accepted methods from it; see :doc:`../method-registry` and
:doc:`../registration-contract`. :py:class:`~starfinder.registration.RegistrationQcConfig`
and :py:class:`~starfinder.registration.RegistrationRejectedError` are the routine QC
criteria and their rejection error; nothing is rejected by default.

.. currentmodule:: starfinder.registration

.. autosummary::
   :toctree: generated

   apply_transform
   CpdConfig
   DemonsConfig
   DenseDisplacementTransform
   estimate_transform
   InsufficientLandmarksError
   InvalidRegistrationConfigError
   REGISTRATION_METHODS
   RegistrationBackendUnavailableError
   RegistrationDiagnostics
   RegistrationEstimationError
   RegistrationQcConfig
   RegistrationRejectedError
   RegistrationResult
   RegistrationSpec
   TpsConfig
   TranslationConfig
   TranslationTransform
   UnsupportedTransformOperationError
   WarpConfig
