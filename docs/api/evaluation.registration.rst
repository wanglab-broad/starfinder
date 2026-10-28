starfinder.evaluation
=====================

Metrics for supplied images, points, masks and read tables. The
:doc:`benchmark guide <../benchmark>` shows how to evaluate saved results; the
:doc:`benchmark reference <../benchmark-recipes>` lists cases and metric outputs.

Result record
-------------

.. currentmodule:: starfinder.evaluation

.. autosummary::
   :toctree: generated

   EvaluationResult

Readout metrics
---------------

.. currentmodule:: starfinder.evaluation.readout

.. autosummary::
   :toctree: generated

   evaluate_decoding
   evaluate_deduplication
   ranking_quality

Point matching
--------------

.. currentmodule:: starfinder.evaluation.matching

.. autosummary::
   :toctree: generated

   match_points

Registration metrics
--------------------

.. currentmodule:: starfinder.evaluation.registration

.. autosummary::
   :toctree: generated

   evaluate_displacement_field
   evaluate_landmark_alignment
   evaluate_mask_overlap
   evaluate_registration
   evaluate_translation
   normalized_cross_correlation
   registration_qc
   structural_similarity

Spot metrics
------------

.. currentmodule:: starfinder.evaluation.spot_finding

.. autosummary::
   :toctree: generated

   classify_detections
   evaluate_spots
   localization_errors

Examples
--------

.. dropdown:: Compare a detected spot with a supplied truth point

   .. code-block:: python

       import numpy as np
       from starfinder.image import ImageMetadata
       from starfinder.evaluation.spot_finding import evaluate_spots

       metadata = ImageMetadata("example/reference")
       result = evaluate_spots(
           np.array([[0., 1., 1.25]]), np.array([[0., 1., 1.]]),
           policy="greedy", threshold=0.5, boundary="exclusive", units="voxel",
           reference_metadata=metadata, observed_metadata=metadata,
       )
       assert result.values["mean_distance"] == 0.25
       assert result.counts["matched"] == 1
