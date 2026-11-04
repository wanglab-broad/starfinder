starfinder.evaluation
=====================

Metrics for supplied images, points, masks and read tables. The
:doc:`experiments guide <../benchmark/experiments>` shows how to evaluate saved results; the
:doc:`commands reference <../benchmark/commands>` lists the evaluation's records and metric outputs.

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

   correct_fraction
   cutoff_curve
   evaluate_decoding
   evaluate_deduplication
   ranking_quality
   read_status_summary

Point matching
--------------

.. currentmodule:: starfinder.evaluation.matching

.. autosummary::
   :toctree: generated

   evaluate_molecules
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
   shift_difference
   structural_similarity

Spot metrics
------------

.. currentmodule:: starfinder.evaluation.spot_finding

.. autosummary::
   :toctree: generated

   auprc
   classify_detections
   evaluate_spots
   f1_score
   localization_errors
   scored_detection_auprc

Preprocessing metrics
---------------------

``direct_metrics`` retains the development evaluation's background stages and
output keys. Detected-spot spread uses observed color sequences; call
``fov_variation`` separately for each metric and each supplied FOV population.

.. currentmodule:: starfinder.evaluation.preprocessing

.. autosummary::
   :toctree: generated

   detected_spot_intensity_spread
   direct_metrics
   fov_variation

Segmentation metrics
--------------------

Instance F1 uses inclusive IoU thresholds from 0.5 to 0.9 with descending-IoU
greedy matching and label-ID ties. Split counts include both qualifying owners
at an exact half tie; merge counts count predicted labels, with distinct merged
truth instances reported separately.

.. currentmodule:: starfinder.evaluation.segmentation

.. autosummary::
   :toctree: generated

   instance_f1
   label_size_summary
   split_merge_counts

Stitching metrics
-----------------

Coordinates supplied to these metrics are already in a common physical frame
and unit. Origin errors remove the translation of a stated anchor; overlap
populations use explicit half-open physical bounds.

.. currentmodule:: starfinder.evaluation.stitching

.. autosummary::
   :toctree: generated

   gauge_aligned_origin_error
   overlap_molecules

Resource predictions
--------------------

.. currentmodule:: starfinder.evaluation.resources

.. autosummary::
   :toctree: generated

   resource_prediction_errors

Workflow output comparison
--------------------------

The comparison of two workflow runs' extracted values, moved from
``scripts/workflow_golden.py``, which records the workflow goldens and calls
it. Identities (``exact``, ``same_environment``) never take a tolerance;
``float64`` and ``float32`` values compare within the tolerances of
:doc:`../contributing` ("Expected values and tolerances").

.. currentmodule:: starfinder.evaluation.workflow

.. autosummary::
   :toctree: generated

   array_differences
   COMPARISON_KINDS
   record_differences
   table_differences
   TOLERANCES
   value_differences
   values_match

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
