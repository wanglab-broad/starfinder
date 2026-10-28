starfinder.spot_finding
=======================

Spot detection, detection plans and result inspection. The
:doc:`spot-finding guide <../spot-finding-contract>` shows the workflow; the
:doc:`spot-finding reference <../spot-finding-algorithms>` lists methods, options and diagnostics.

.. currentmodule:: starfinder.spot_finding

Detection and plans
-------------------

.. autosummary::
   :toctree: generated

   ChannelOverride
   find_spots
   plot_detections
   SpotFindingPlan
   SpotFindingResult

Method configurations and registry
----------------------------------

.. autosummary::
   :toctree: generated

   LocalMaximaConfig
   NoiseLandmarkConfig
   PercentileCentroidConfig
   PiscisConfig
   SPOT_FINDING_METHODS
   SpotFindingSpec
   SpotiflowConfig
   StarfishLogConfig

Models
------

.. autosummary::
   :toctree: generated

   fetch_model
   KNOWN_MODELS
   KnownModel
   ModelFile
   resolve_model

Errors and warnings
-------------------

.. autosummary::
   :toctree: generated

   MissingModelError
   ModelHashMismatchError
   SpotFindingBackendUnavailableError
   SpotFindingWarning
