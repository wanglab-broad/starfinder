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
* ``starfinder.estimate_transform`` → :doc:`registration`
* ``starfinder.apply_transform`` → :doc:`registration`
* ``starfinder.registration`` → :doc:`registration` (module)
* ``starfinder.spot_finding`` → :doc:`spot_finding` (module)
* ``starfinder.find_spots`` → :doc:`spot_finding`
* ``starfinder.barcode`` → :doc:`barcode` (module)
* ``starfinder.extract_intensities`` → :doc:`barcode`
* ``starfinder.decode_barcodes`` → :doc:`barcode`
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
``starfinder.benchmark``. Barcode mechanics and lookup constants are private; use structured decoding
diagnostics for per-round probabilities and candidate scores.

.. list-table:: Export inventory
   :header-rows: 1
   :widths: 70 30

   * - Export / explicit submodule interface
     - Reference page
   * - ``starfinder.image.InvalidImageError``
     - :doc:`image`
   * - ``starfinder.image.IncompatibleGeometryError``
     - :doc:`image`
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
   * - ``starfinder.barcode.BarcodeDecodingResult``
     - :doc:`barcode`
   * - ``starfinder.barcode.Codebook``
     - :doc:`barcode`
   * - ``starfinder.barcode.CodebookAwareDecoderConfig``
     - :doc:`barcode`
   * - ``starfinder.barcode.EncodingConfig``
     - :doc:`barcode`
   * - ``starfinder.barcode.IntensityExtractionResult``
     - :doc:`barcode`
   * - ``starfinder.barcode.InvalidIntensityError``
     - :doc:`barcode`
   * - ``starfinder.barcode.NeighborhoodSumConfig``
     - :doc:`barcode`
   * - ``starfinder.barcode.ReadFilterConfig``
     - :doc:`barcode`
   * - ``starfinder.barcode.ReadFilteringResult``
     - :doc:`barcode`
   * - ``starfinder.barcode.WtaDecoderConfig``
     - :doc:`barcode`
   * - ``starfinder.barcode.decode_barcodes``
     - :doc:`barcode`
   * - ``starfinder.barcode.decode_color_sequence``
     - :doc:`barcode`
   * - ``starfinder.barcode.encode_bases``
     - :doc:`barcode`
   * - ``starfinder.barcode.extract_intensities``
     - :doc:`barcode`
   * - ``starfinder.barcode.filter_reads``
     - :doc:`barcode`
   * - ``starfinder.barcode.load_codebook``
     - :doc:`barcode`
   * - ``starfinder.dataset.STARMapDataset``
     - :doc:`dataset`
   * - ``starfinder.dataset.FOV``
     - :doc:`dataset`
   * - ``starfinder.dataset.FOVPaths``
     - :doc:`dataset`
   * - ``starfinder.dataset.LayerState``
     - :doc:`dataset`
   * - ``starfinder.dataset.CropWindow``
     - :doc:`dataset`
   * - ``starfinder.dataset.SubtileConfig``
     - :doc:`dataset`
   * - ``starfinder.dataset.ImageArray``
     - :doc:`dataset`
   * - ``starfinder.dataset.ChannelOrder``
     - :doc:`dataset`
   * - ``starfinder.dataset.log_step``
     - :doc:`dataset`
   * - ``starfinder.registration.estimate_transform``
     - :doc:`registration`
   * - ``starfinder.registration.apply_transform``
     - :doc:`registration`
   * - ``starfinder.registration.TranslationConfig``
     - :doc:`registration`
   * - ``starfinder.registration.DemonsConfig``
     - :doc:`registration`
   * - ``starfinder.registration.TpsConfig``
     - :doc:`registration`
   * - ``starfinder.registration.CpdConfig``
     - :doc:`registration`
   * - ``starfinder.registration.WarpConfig``
     - :doc:`registration`
   * - ``starfinder.registration.TranslationTransform``
     - :doc:`registration`
   * - ``starfinder.registration.DenseDisplacementTransform``
     - :doc:`registration`
   * - ``starfinder.registration.RegistrationResult``
     - :doc:`registration`
   * - ``starfinder.registration.RegistrationDiagnostics``
     - :doc:`registration`
   * - ``starfinder.registration.InvalidRegistrationConfigError``
     - :doc:`registration`
   * - ``starfinder.registration.RegistrationEstimationError``
     - :doc:`registration`
   * - ``starfinder.registration.InsufficientLandmarksError``
     - :doc:`registration`
   * - ``starfinder.registration.RegistrationBackendUnavailableError``
     - :doc:`registration`
   * - ``starfinder.registration.UnsupportedTransformOperationError``
     - :doc:`registration`
   * - ``starfinder.benchmark.BenchmarkResult``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.BenchmarkSuite``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.benchmark``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.measure``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.run_benchmark``
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

Additional type alias
---------------------

``starfinder.benchmark.synthetic.SpotTuple`` is documented on
:doc:`benchmark.synthetic`; it is not re-exported by the benchmark package.

Internal and deprecated interfaces
----------------------------------

* Registration numerical modules are private. Removed public modules and
  wrappers have no aliases; use ``estimate_transform`` and ``apply_transform``.
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
