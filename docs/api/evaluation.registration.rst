starfinder.evaluation
========================

Pure metrics accept supplied images, Boolean masks, points, truth and metadata.
They never run detection/registration or write files. Benchmark adapters own
loading, detection preparation, aggregation and reporting.

Every metric returns ``EvaluationResult`` with ``values``, ``units``, ``counts``,
``status``, ``reasons``, effective ``config`` and per-item ``details``. Undefined
values are ``None`` (JSON null), never zero, infinity or a passing check. Status
is ``ok``, ``undefined``, ``missing`` or ``failed``. Translation records absent
and failed rounds explicitly. Exclude the reference with ``eligible_rounds``;
partial comparisons never pass. ``tolerance=None`` requests errors without a gate.

Coordinate policies
-------------------

Supply finite zero-based ZYX arrays, or already converted physical coordinates.
Both populations require identical ``ImageMetadata``. Physical units require
complete calibration and must match ``spatial_unit``. Evaluation never converts
frames, rescales coordinates or inverts shift signs. Subpixel shifts remain
floats. Eligibility masks define candidate populations and recall/precision
denominators. Match indices refer to original rows; decoding tables must retain
that row order.

``match_points`` requires a policy, threshold and units. ``greedy`` sorts all
admissible pairs by distance/index. ``nearest_candidate`` proposes one nearest
point per reference and removes duplicate assignments, preserving registration's
KDTree policy and tie ordering. Neither is optimal bipartite matching. The
inclusive/exclusive boundary is recorded. Historical spot-truth comparisons
used exclusive distance 5; registration quality used inclusive distance 2.
These are retained caller choices, not new scientific thresholds.

Image policies
--------------

SSIM requires positive ``data_range`` and an explicit ``volume``, ``mip``,
``slice`` (with Z index), or ``plane`` policy. The window is recorded. Domains
smaller than three along any measured axis yield undefined SSIM. NCC measures
all supplied elements; constant images yield undefined NCC. Empty mask unions
and empty matching denominators remain undefined.

Combined registration evaluation checks image geometry. Supplied masks and
landmarks represent the caller's measurement domain. Benchmark adapters retain
volume NCC, MIP/volume SSIM, percentile 99.5 detections, and percentile 99 MIP /
99.5 volume masks, recording those settings outside evaluation.

Migration example
-----------------

Previously ``benchmark.compare_spots(table, ground_truth, fov_id)`` selected
truth and settings implicitly. Select them at the caller now::

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

``compare_shifts`` becomes ``evaluate_translation`` over labeled displacement
maps; ``compare_genes`` becomes ``evaluate_decoding`` over tables and supplied
matches. Missing truth labels are excluded from conditional accuracy denominators;
missing predicted labels are incorrect calls. Missing columns are undefined.
Replaced benchmark exports have no aliases. Reports remain in benchmark.

Intentional corrections: exact centered NCC without epsilon bias, undefined
zero denominators and missing shifts, and float-preserving shift errors. Matching
policies and historical thresholds remain unchanged. These checks do not qualify
scientific truth/calibration or define an E01 protocol. MATLAB is excluded.

.. currentmodule:: starfinder.evaluation

.. autosummary::
   :toctree: generated

   EvaluationResult

.. currentmodule:: starfinder.evaluation.registration

.. autosummary::
   :toctree: generated

   evaluate_landmark_alignment
   evaluate_mask_overlap
   evaluate_registration
   evaluate_translation
   normalized_cross_correlation
   structural_similarity

.. currentmodule:: starfinder.evaluation.matching

.. autosummary::
   :toctree: generated

   match_points

.. currentmodule:: starfinder.evaluation.spot_finding

.. autosummary::
   :toctree: generated

   evaluate_spots

.. currentmodule:: starfinder.evaluation.barcode

.. autosummary::
   :toctree: generated

   evaluate_decoding
