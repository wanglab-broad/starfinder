starfinder.registration
=======================

Global shifts are detected displacement in voxel units; negate them for :func:`starfinder.registration.apply_shift`. Dense fields instead map output coordinates to input coordinates. See :doc:`contracts` and :doc:`backends`.

.. currentmodule:: starfinder.registration

.. autosummary::
   :toctree: generated

   phase_correlate
   apply_shift
   register_volume
   phase_correlate_skimage
   demons_register
   apply_deformation
   register_volume_local
   matlab_compatible_config
   tps_register
   register_volume_tps
   cpd_register
   register_volume_cpd
   sanitize_displacement_field
   normalized_cross_correlation
   structural_similarity
   spot_colocalization
   spot_matching_accuracy
   registration_quality_report
   print_quality_report
