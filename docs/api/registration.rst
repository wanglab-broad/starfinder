starfinder.registration
=======================

Estimate once, then apply the returned transform with its application config.
Every transform is a pull map from reference to moving coordinates: translations
store the detected displacement vector; affine, B-spline and dense transforms
their matrices, control grids and fields. See :doc:`contracts` and :doc:`backends` for geometry, errors and
supported dimensions.

:py:class:`~starfinder.registration.RigidConfig`,
:py:class:`~starfinder.registration.AffineConfig` and
:py:class:`~starfinder.registration.BSplineConfig` are estimated in physical space with
elastix (optional extra ``registration-elastix``, imported only when one of them runs)
and return an :py:class:`~starfinder.registration.AffineTransform` (index-space matrix
with the physical parameters beside it) or a
:py:class:`~starfinder.registration.BSplineTransform` (physical control grid, evaluated
to a dense field with SimpleITK). Rigid, affine, B-spline and demons estimate a Z=1
input as genuine 2D and embed the result with no Z motion; see
:doc:`../registration-algorithms`.

:py:data:`~starfinder.registration.REGISTRATION_METHODS` maps each exact config type to
its :py:class:`~starfinder.registration.RegistrationSpec`: the method name (the config's
``method`` value), step kind, dimensions, minimum shape, transform kind, estimation space
and optional dependencies. :py:func:`~starfinder.registration.estimate_transform`, the
dataset configs, the workflow adapter, the checkpoint reader and the benchmark adapter
derive their accepted methods from it; see :doc:`../method-registry` and
:doc:`../registration-contract`. :py:class:`~starfinder.registration.RegistrationQcConfig`
and :py:class:`~starfinder.registration.RegistrationRejectedError` are the routine QC
criteria and their rejection error; nothing is rejected by default.

A :py:class:`~starfinder.registration.TransformChain` composes the step transforms of
one moving round into one pull map, ``T1(T2(...Tn(p)))``; its ``pull_field()`` is the
composite float64 displacement, and ``apply_transform`` resamples an image once through
it. :py:class:`~starfinder.registration.RegistrationSignalConfig` builds the float64 ZYX
registration signal of a round (channel maximum by default, sum, or one channel). Both
belong to the registration recipe of :py:class:`~starfinder.registration.RegistrationRecipe`;
see :doc:`../coordination`.

The recipe types are part of this package: a
:py:class:`~starfinder.registration.RegistrationRecipe` holds ordered
:py:class:`~starfinder.registration.RegistrationStep` entries, each with an optional
:py:class:`~starfinder.registration.RecoveryConfig`, and an
:py:class:`~starfinder.registration.ExternalReference` is a reference signal that is not a
round of the dataset. ``FOV.register``, ``FOV.register_rounds`` and
``PipelineConfig.registration`` take them; see :doc:`../registration-contract`.

.. currentmodule:: starfinder.registration

.. autosummary::
   :toctree: generated

   AffineConfig
   AffineTransform
   apply_transform
   BSplineConfig
   BSplineTransform
   CpdConfig
   DemonsConfig
   DenseDisplacementTransform
   estimate_transform
   ExternalReference
   InsufficientLandmarksError
   InvalidRegistrationConfigError
   RecoveryConfig
   REGISTRATION_METHODS
   RegistrationBackendUnavailableError
   RegistrationDiagnostics
   RegistrationEstimationError
   RegistrationQcConfig
   RegistrationRecipe
   RegistrationRejectedError
   RegistrationResult
   RegistrationSignalConfig
   RegistrationSpec
   RegistrationStep
   RigidConfig
   TpsConfig
   TransformChain
   TranslationConfig
   TranslationTransform
   UnsupportedTransformOperationError
   WarpConfig
