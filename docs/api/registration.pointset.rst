starfinder.registration.pointset
================================

Lower-level point-set helpers, available by explicit submodule import. Their fields use backward sampling: ``output[p] = moving[p + field[p]]``. See :doc:`registration` for the high-level TPS/CPD entry points.

.. currentmodule:: starfinder.registration.pointset

.. autosummary::
   :toctree: generated

   detect_and_match_spots
   subsample_control_points
   tps_displacement_field
   apply_tps_deformation
   cpd_affine
   cpd_nonrigid
   cpd_displacement_field
