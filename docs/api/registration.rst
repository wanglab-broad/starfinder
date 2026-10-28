starfinder.registration
=======================

Transform estimation and image resampling. The
:doc:`registration guide <../registration-contract>` shows an estimate-and-apply example;
the :doc:`registration reference <../registration-algorithms>` lists methods and options.

.. currentmodule:: starfinder.registration

Estimation, application and results
-----------------------------------

.. autosummary::
   :toctree: generated

   apply_transform
   estimate_transform
   RegistrationDiagnostics
   RegistrationResult
   WarpConfig

Method configurations and registry
----------------------------------

.. autosummary::
   :toctree: generated

   AffineConfig
   BSplineConfig
   CpdConfig
   DemonsConfig
   REGISTRATION_METHODS
   RegistrationSpec
   RigidConfig
   TpsConfig
   TranslationConfig

Transforms
----------

.. autosummary::
   :toctree: generated

   AffineTransform
   BSplineTransform
   DenseDisplacementTransform
   TransformChain
   TranslationTransform

Recipes and quality control
---------------------------

.. autosummary::
   :toctree: generated

   ExternalReference
   RecoveryConfig
   RegistrationQcConfig
   RegistrationRecipe
   RegistrationSignalConfig
   RegistrationStep

Errors
------

.. autosummary::
   :toctree: generated

   InsufficientLandmarksError
   InvalidRegistrationConfigError
   RegistrationBackendUnavailableError
   RegistrationEstimationError
   RegistrationRejectedError
   UnsupportedTransformOperationError
