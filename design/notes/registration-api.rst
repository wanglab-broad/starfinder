starfinder.registration
=======================

Estimate once, then apply the returned transform with its application config.
Every transform is a pull map from reference to moving coordinates: translations
store the detected displacement vector; affine, B-spline and dense transforms
their matrices, control grids and fields. See `Array and coordinate contracts <../../docs/api/contracts.md>`_ for coordinate conventions, `Dependencies and backends <../../docs/api/backends.md>`_ for
dependencies and `Registration reference <../../docs/registration-algorithms.md>`_ for method dimensions. Error
details are in the `registration specification
<https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/registration.md#failure-and-recovery-records>`_.

:py:class:`~starfinder.registration.RigidConfig`,
:py:class:`~starfinder.registration.AffineConfig` and
:py:class:`~starfinder.registration.BSplineConfig` are estimated in physical space with
elastix (optional extra ``registration-elastix``, imported only when one of them runs)
and return an :py:class:`~starfinder.registration.AffineTransform` (index-space matrix
with the physical parameters beside it) or a
:py:class:`~starfinder.registration.BSplineTransform` (physical control grid, evaluated
to a dense field with SimpleITK). Rigid, affine, B-spline and demons estimate a Z=1
input as genuine 2D and embed the result with no Z motion; see
`Registration reference <../../docs/registration-algorithms.md>`_.

:py:data:`~starfinder.registration.REGISTRATION_METHODS` maps each exact config type to
its :py:class:`~starfinder.registration.RegistrationSpec`: the method name (the config's
``method`` value), step kind, dimensions, minimum shape, transform kind, estimation space
and optional dependencies. :py:func:`~starfinder.registration.estimate_transform`, the
dataset configs, the workflow adapter, the checkpoint reader and the benchmark adapter
derive their accepted methods from it; see `Method registry: adding a method <../../docs/method-registry.md>`_ and
`Registration reference <../../docs/registration-algorithms.md>`_. :py:class:`~starfinder.registration.RegistrationQcConfig`
and :py:class:`~starfinder.registration.RegistrationRejectedError` are the routine QC
criteria and their rejection error; nothing is rejected by default.

A :py:class:`~starfinder.registration.TransformChain` composes the step transforms of
one moving round into one pull map, ``T1(T2(...Tn(p)))``; its ``pull_field()`` is the
composite float64 displacement, and ``apply_transform`` resamples an image once through
it. :py:class:`~starfinder.registration.RegistrationSignalConfig` builds the float64 ZYX
registration signal of a round (channel maximum by default, sum, or one channel). Both
belong to the registration recipe of :py:class:`~starfinder.registration.RegistrationRecipe`;
see `Dataset and FOV coordination <../../docs/coordination.md>`_.

The recipe types are part of this package: a
:py:class:`~starfinder.registration.RegistrationRecipe` holds ordered
:py:class:`~starfinder.registration.RegistrationStep` entries, each with an optional
:py:class:`~starfinder.registration.RecoveryConfig`, and an
:py:class:`~starfinder.registration.ExternalReference` is a reference signal that is not a
round of the dataset. ``FOV.register`` (on both of its routes) and
``PipelineConfig.registration`` take them; see `Registration <../../docs/registration-contract.md>`_.

