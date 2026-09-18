Public API coverage inventory
=============================

Coverage boundary
-----------------

Every name in the root and seven subpackage ``__all__`` lists is covered below.
The synthetic module, utility function, and non-underscored definitions in the
point-set, pyramid and legacy registration-benchmark modules are also included.
Class pages include public methods, properties and dataclass fields. Signatures
show literal defaults; ``<factory>`` means a fresh collection/value per instance.

This is a source-based API inventory, not a scientific validation claim or a
promise that every parameter combination is supported.

Root aliases
------------

``import starfinder`` re-exports the following names. Use the corresponding
subpackage page for each object's canonical reference; aliases refer to the same
objects and are not duplicate implementations.

* ``starfinder.STARMapDataset`` → :doc:`dataset`
* ``starfinder.ImageMetadata`` → :doc:`image`
* ``starfinder.FOV`` → :doc:`dataset`
* ``starfinder.load_volume`` → :doc:`io`
* ``starfinder.load_round`` → :doc:`io`
* ``starfinder.save_volume`` → :doc:`io`
* ``starfinder.registration`` → :doc:`registration` (module)
* ``starfinder.phase_correlate`` → :doc:`registration`
* ``starfinder.apply_shift`` → :doc:`registration`
* ``starfinder.register_volume`` → :doc:`registration`
* ``starfinder.phase_correlate_skimage`` → :doc:`registration`
* ``starfinder.spot_finding`` → :doc:`spot_finding` (module)
* ``starfinder.find_spots`` → :doc:`spot_finding`
* ``starfinder.barcode`` → :doc:`barcode` (module)
* ``starfinder.extract_from_location`` → :doc:`barcode`
* ``starfinder.load_codebook`` → :doc:`barcode`
* ``starfinder.filter_reads`` → :doc:`barcode`
* ``starfinder.preprocessing`` → :doc:`preprocessing` (module)
* ``starfinder.normalize_intensity`` → :doc:`preprocessing`
* ``starfinder.match_histogram`` → :doc:`preprocessing`
* ``starfinder.reconstruct_background`` → :doc:`preprocessing`
* ``starfinder.filter_tophat`` → :doc:`preprocessing`
* ``starfinder.project_image`` → :doc:`preprocessing`
* ``starfinder.__version__``: package version string (``0.1.0``).

Canonical coverage
------------------

Synthetic names on :doc:`benchmark.synthetic` are also re-exported by
``starfinder.benchmark``. The barcode codebook-aware helpers are also available
from ``starfinder.barcode.codebook_aware``; its ``__all__`` is a subset of the
barcode table. Other implementation-module import aliases point to the same
objects documented here.

.. list-table:: Export inventory
   :header-rows: 1
   :widths: 70 30

   * - Export / explicit submodule interface
     - Reference page
   * - ``starfinder.image.ImageMetadata``
     - :doc:`image`
   * - ``starfinder.io.ImageConversionConfig``
     - :doc:`io`
   * - ``starfinder.io.ImageLoadConfig``
     - :doc:`io`
   * - ``starfinder.io.ImageLoadResult``
     - :doc:`io`
   * - ``starfinder.io.convert_image``
     - :doc:`io`
   * - ``starfinder.preprocessing.HistogramMatchingConfig``
     - :doc:`preprocessing`
   * - ``starfinder.preprocessing.MinMaxNormalizationConfig``
     - :doc:`preprocessing`
   * - ``starfinder.preprocessing.ProjectionConfig``
     - :doc:`preprocessing`
   * - ``starfinder.preprocessing.ReconstructionConfig``
     - :doc:`preprocessing`
   * - ``starfinder.preprocessing.TophatConfig``
     - :doc:`preprocessing`
   * - ``starfinder.io.load_volume``
     - :doc:`io`
   * - ``starfinder.io.load_round``
     - :doc:`io`
   * - ``starfinder.io.save_volume``
     - :doc:`io`
   * - ``starfinder.preprocessing.normalize_intensity``
     - :doc:`preprocessing`
   * - ``starfinder.preprocessing.match_histogram``
     - :doc:`preprocessing`
   * - ``starfinder.preprocessing.reconstruct_background``
     - :doc:`preprocessing`
   * - ``starfinder.preprocessing.filter_tophat``
     - :doc:`preprocessing`
   * - ``starfinder.registration.phase_correlate``
     - :doc:`registration`
   * - ``starfinder.registration.apply_shift``
     - :doc:`registration`
   * - ``starfinder.registration.register_volume``
     - :doc:`registration`
   * - ``starfinder.registration.phase_correlate_skimage``
     - :doc:`registration`
   * - ``starfinder.registration.demons_register``
     - :doc:`registration`
   * - ``starfinder.registration.apply_deformation``
     - :doc:`registration`
   * - ``starfinder.registration.register_volume_local``
     - :doc:`registration`
   * - ``starfinder.registration.matlab_compatible_config``
     - :doc:`registration`
   * - ``starfinder.registration.tps_register``
     - :doc:`registration`
   * - ``starfinder.registration.register_volume_tps``
     - :doc:`registration`
   * - ``starfinder.registration.cpd_register``
     - :doc:`registration`
   * - ``starfinder.registration.register_volume_cpd``
     - :doc:`registration`
   * - ``starfinder.registration.sanitize_displacement_field``
     - :doc:`registration`
   * - ``starfinder.registration.normalized_cross_correlation``
     - :doc:`registration`
   * - ``starfinder.registration.structural_similarity``
     - :doc:`registration`
   * - ``starfinder.registration.spot_colocalization``
     - :doc:`registration`
   * - ``starfinder.registration.spot_matching_accuracy``
     - :doc:`registration`
   * - ``starfinder.registration.registration_quality_report``
     - :doc:`registration`
   * - ``starfinder.registration.print_quality_report``
     - :doc:`registration`
   * - ``starfinder.spot_finding.LocalMaximaConfig``
     - :doc:`spot_finding`
   * - ``starfinder.spot_finding.NoiseLandmarkConfig``
     - :doc:`spot_finding`
   * - ``starfinder.spot_finding.PercentileCentroidConfig``
     - :doc:`spot_finding`
   * - ``starfinder.spot_finding.SpotFindingResult``
     - :doc:`spot_finding`
   * - ``starfinder.spot_finding.find_spots``
     - :doc:`spot_finding`
   * - ``starfinder.barcode.BASE_PAIR_TO_COLOR``
     - :doc:`barcode`
   * - ``starfinder.barcode.COLOR_TO_BASE_PAIRS``
     - :doc:`barcode`
   * - ``starfinder.barcode.COLOR_TO_CHANNEL``
     - :doc:`barcode`
   * - ``starfinder.barcode.build_one_error_index``
     - :doc:`barcode`
   * - ``starfinder.barcode.candidate_sequences``
     - :doc:`barcode`
   * - ``starfinder.barcode.channel_probabilities``
     - :doc:`barcode`
   * - ``starfinder.barcode.decode_color_seq``
     - :doc:`barcode`
   * - ``starfinder.barcode.decode_codebook_aware``
     - :doc:`barcode`
   * - ``starfinder.barcode.encode_bases``
     - :doc:`barcode`
   * - ``starfinder.barcode.extract_from_location``
     - :doc:`barcode`
   * - ``starfinder.barcode.extract_intensity_tensor``
     - :doc:`barcode`
   * - ``starfinder.barcode.filter_reads``
     - :doc:`barcode`
   * - ``starfinder.barcode.load_codebook``
     - :doc:`barcode`
   * - ``starfinder.barcode.score_candidates``
     - :doc:`barcode`
   * - ``starfinder.barcode.wta_color_sequences``
     - :doc:`barcode`
   * - ``starfinder.dataset.STARMapDataset``
     - :doc:`dataset`
   * - ``starfinder.dataset.FOV``
     - :doc:`dataset`
   * - ``starfinder.dataset.FOVPaths``
     - :doc:`dataset`
   * - ``starfinder.dataset.LayerState``
     - :doc:`dataset`
   * - ``starfinder.dataset.Codebook``
     - :doc:`dataset`
   * - ``starfinder.dataset.CropWindow``
     - :doc:`dataset`
   * - ``starfinder.dataset.SubtileConfig``
     - :doc:`dataset`
   * - ``starfinder.dataset.Shift3D``
     - :doc:`dataset`
   * - ``starfinder.dataset.ImageArray``
     - :doc:`dataset`
   * - ``starfinder.dataset.ChannelOrder``
     - :doc:`dataset`
   * - ``starfinder.dataset.log_step``
     - :doc:`dataset`
   * - ``starfinder.benchmark.BenchmarkResult``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.BenchmarkSuite``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.benchmark``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.measure``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.run_comparison``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.RegistrationBenchmarkRunner``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.RegistrationResult``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.BenchmarkPair``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.PRESET_ORDER``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.timeout_handler``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.DEFAULT_BENCHMARK_DIR``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.BENCHMARK_TASK``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.SIZE_PRESETS``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.SPOT_COUNTS``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.SHIFT_RANGES``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.get_size_preset``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.generate_inspection_image``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.generate_overview_grid``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.extract_real_benchmark_data``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.REAL_DATASETS``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.evaluate_registration``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.evaluate_directory``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.evaluate_single``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.generate_inspection``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.compare_shifts``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.compare_spots``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.compare_genes``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.e2e_summary``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.print_table``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.save_csv``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.save_json``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.synthetic.generate_codebook``
     - :doc:`benchmark.synthetic`
   * - ``starfinder.benchmark.synthetic.encode_barcode_to_colors``
     - :doc:`benchmark.synthetic`
   * - ``starfinder.benchmark.synthetic.scale_deformation_config``
     - :doc:`benchmark.synthetic`
   * - ``starfinder.benchmark.synthetic.create_deformation_field``
     - :doc:`benchmark.synthetic`
   * - ``starfinder.benchmark.synthetic.apply_shift_to_spots``
     - :doc:`benchmark.synthetic`
   * - ``starfinder.benchmark.synthetic.apply_deformation_to_spots``
     - :doc:`benchmark.synthetic`
   * - ``starfinder.benchmark.synthetic.SyntheticConfig``
     - :doc:`benchmark.synthetic`
   * - ``starfinder.benchmark.synthetic.get_preset_config``
     - :doc:`benchmark.synthetic`
   * - ``starfinder.benchmark.synthetic.create_test_image_stack``
     - :doc:`benchmark.synthetic`
   * - ``starfinder.benchmark.synthetic.create_test_volume``
     - :doc:`benchmark.synthetic`
   * - ``starfinder.benchmark.synthetic.generate_synthetic_dataset``
     - :doc:`benchmark.synthetic`
   * - ``starfinder.benchmark.synthetic.generate_registration_benchmark``
     - :doc:`benchmark.synthetic`
   * - ``starfinder.benchmark.synthetic.TEST_CODEBOOK``
     - :doc:`benchmark.synthetic`
   * - ``starfinder.benchmark.synthetic.DEFORMATION_CONFIGS``
     - :doc:`benchmark.synthetic`
   * - ``starfinder.preprocessing.project_image``
     - :doc:`preprocessing`
   * - ``starfinder.registration.pointset.detect_and_match_spots``
     - :doc:`registration.pointset`
   * - ``starfinder.registration.pointset.subsample_control_points``
     - :doc:`registration.pointset`
   * - ``starfinder.registration.pointset.tps_displacement_field``
     - :doc:`registration.pointset`
   * - ``starfinder.registration.pointset.apply_tps_deformation``
     - :doc:`registration.pointset`
   * - ``starfinder.registration.pointset.cpd_affine``
     - :doc:`registration.pointset`
   * - ``starfinder.registration.pointset.cpd_nonrigid``
     - :doc:`registration.pointset`
   * - ``starfinder.registration.pointset.cpd_displacement_field``
     - :doc:`registration.pointset`
   * - ``starfinder.registration.pyramid.butterworth_3d``
     - :doc:`registration.pyramid`
   * - ``starfinder.registration.pyramid.antialias_resize``
     - :doc:`registration.pyramid`
   * - ``starfinder.registration.pyramid.pad_for_pyramiding``
     - :doc:`registration.pyramid`
   * - ``starfinder.registration.pyramid.crop_padding``
     - :doc:`registration.pyramid`
   * - ``starfinder.registration.benchmark.benchmark_registration``
     - :doc:`registration.benchmark`
   * - ``starfinder.registration.benchmark.run_benchmark``
     - :doc:`registration.benchmark`
   * - ``starfinder.registration.benchmark.print_benchmark_table``
     - :doc:`registration.benchmark`

Additional type alias
---------------------

``starfinder.benchmark.synthetic.SpotTuple`` is documented on
:doc:`benchmark.synthetic`; it is not re-exported by the benchmark package.

Internal and deprecated interfaces
----------------------------------

* ``starfinder.registration.benchmark.run_benchmark`` is explicitly deprecated
  in source; use ``benchmark_registration`` in that module, or the current
  benchmark runner. No other exported interface carries a deprecation marker.
* Underscored functions/modules, including implementation routines in
  ``registration._skimage_backend``, are internal. Its deliberately re-exported
  ``phase_correlate_skimage`` function is public and covered above.
* ``starfinder.benchmark.__main__.main`` implements the ``starfinder-generate``
  CLI; it is not a Python call interface. Its parser is available through
  ``uv run python -m starfinder.benchmark --help``.
* Implementation-only constants ``SPOT_COLUMNS``, ``OUTPUT_COLUMNS`` and
  ``DEFAULT_BENCHMARK_DATA_DIR`` and
  ``benchmark.evaluate.DEFAULT_BENCHMARK_TASK_DIR``, ``SYNTHETIC_PRESETS``,
  and its separate ``REAL_DATASETS`` lookup define output schemas/default paths; their
  behavior is described on the corresponding function pages. ``T`` in
  ``benchmark.core`` is an internal typing variable. Imported dependencies
  (``np``, ``pd``, ``Path``), loggers and nested closures are not STARfinder APIs.

When changing exports, update this inventory and the relevant autosummary list.
Generated object stubs and HTML are build products; do not commit them.
