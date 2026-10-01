starfinder.spot_finding
==============================

Detection returns a typed SpotFindingResult with zero-based voxel coordinates,
explicit geometry and namespace. See :doc:`contracts` for distinct policies.

:py:data:`~starfinder.spot_finding.SPOT_FINDING_METHODS` maps each exact config type to
its :py:class:`~starfinder.spot_finding.SpotFindingSpec`: the method name (the config's
``method`` value), whether the pipeline accepts it, its dimensions, minimum shape,
output columns, weights and optional dependencies. :py:func:`~starfinder.spot_finding.find_spots`,
:py:class:`~starfinder.spot_finding.SpotFindingResult`, the dataset configs,
``FOV.find_spots``, the workflow adapter and the checkpoint reader derive their accepted
methods from it, so a subclass of a config is not accepted; see :doc:`../method-registry`
and :doc:`../spot-finding-contract`. A missing optional dependency raises
:py:class:`~starfinder.spot_finding.SpotFindingBackendUnavailableError`.

A :py:class:`~starfinder.spot_finding.SpotFindingPlan` runs one method on every channel
and replaces the whole config of the channels named by its
:py:class:`~starfinder.spot_finding.ChannelOverride` entries;
``diagnostics["effective_settings"]`` records every channel's config. Local maxima
records the noise median, MAD, zero fraction and threshold of each channel in
``diagnostics["noise"]`` and emits a :py:class:`~starfinder.spot_finding.SpotFindingWarning`
when a channel's MAD is 0 or more than half of its voxels are zero; the threshold does not
change. ``diagnostics["execution"]`` records the device (only ``"cpu"``) and the thread
settings in effect.

:py:class:`~starfinder.spot_finding.LocalMaximaConfig` has the opt-in W-218 within-channel
merge ``merge_radius_zyx`` (default ``None``, the legacy result): of the maxima of one channel
within the given ZYX ellipsoid only the brightest is kept (ties broken by z, y, x), and
``diagnostics["merged"]`` records the number removed per channel. Channels are never merged
with each other. :py:class:`~starfinder.spot_finding.StarfishLogConfig` selects
``starfish_log``, a native reimplementation of starfish ``BlobDetector`` (``blob_log``,
``is_volume=True``, no reference image) that reproduces starfish's tables exactly; it scales
integer images by their dtype maximum, adds the ``radius`` column and records its scale-space
memory estimate in ``diagnostics["geometry"]``. See :doc:`../spot-finding-algorithms`.

:py:data:`~starfinder.spot_finding.KNOWN_WEIGHTS` lists the pretrained weights Starfinder
can fetch and verify, with their SHA-256 values; it sets no default model.
:py:func:`~starfinder.spot_finding.fetch_weights` (``starfinder weights fetch <method>
<model>``) is the only code that downloads; ``starfinder weights list`` and ``starfinder
weights verify`` show and re-hash the local copies. The cache is ``STARFINDER_WEIGHTS_DIR``,
else ``$XDG_CACHE_HOME/starfinder/weights`` (``~/.cache/starfinder/weights``).
:py:func:`~starfinder.spot_finding.resolve_weights` returns a verified model folder or
raises :py:class:`~starfinder.spot_finding.MissingWeightsError` or
:py:class:`~starfinder.spot_finding.WeightsHashMismatchError`.

.. currentmodule:: starfinder.spot_finding

.. autosummary::
   :toctree: generated

   ChannelOverride
   fetch_weights
   find_spots
   KNOWN_WEIGHTS
   KnownWeights
   LocalMaximaConfig
   MissingWeightsError
   NoiseLandmarkConfig
   PercentileCentroidConfig
   resolve_weights
   SPOT_FINDING_METHODS
   SpotFindingBackendUnavailableError
   SpotFindingPlan
   SpotFindingResult
   SpotFindingSpec
   SpotFindingWarning
   StarfishLogConfig
   WeightsFile
   WeightsHashMismatchError
