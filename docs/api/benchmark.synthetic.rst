starfinder.benchmark.synthetic
==============================

Coordinate-first synthetic generators and configurations. Arrays use ``(Z, Y, X[, C])`` and spots use zero-based ``(z, y, x)`` voxel positions. Preset values below come from current source, not historical benchmark tables. Use explicit output paths and seeds; large presets are not documentation examples.

.. currentmodule:: starfinder.benchmark.synthetic

.. autosummary::
   :toctree: generated

   generate_codebook
   encode_barcode_to_colors
   scale_deformation_config
   create_deformation_field
   apply_shift_to_spots
   apply_deformation_to_spots
   SyntheticConfig
   get_preset_config
   create_test_image_stack
   create_test_volume
   generate_synthetic_dataset
   generate_registration_benchmark

.. autodata:: starfinder.benchmark.synthetic.TEST_CODEBOOK

.. autodata:: starfinder.benchmark.synthetic.DEFORMATION_CONFIGS

.. py:data:: SpotTuple

   Alias of ``tuple[int, int, int, int, float]``: ``(z, y, x, intensity,
   sigma)``. Coordinate-transform helpers can return fractional coordinates;
   rendering consumes these as floating point despite the integer annotation.
