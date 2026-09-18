Public API coverage inventory
=============================

Coverage boundary
-----------------

Supported root and subpackage ``__all__`` exports are covered below, including
the independent synthetic package. Removed names have no compatibility aliases.
Class pages include public methods, properties and dataclass fields. Signatures
show literal defaults; ``<factory>`` means a fresh collection/value per instance.

This is a source-based API inventory, not a scientific validation claim or a
promise that every parameter combination is supported.

Root aliases
------------

``import starfinder`` re-exports the following names. Use the corresponding
subpackage page for each object's canonical reference; aliases refer to the same
objects and are not duplicate implementations.

* ``starfinder.Dataset`` → :doc:`dataset`
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

Synthetic names belong to :doc:`synthetic` exclusively. Barcode mechanics and lookup constants are private; use structured decoding
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
   * - ``starfinder.dataset.Dataset``
     - :doc:`dataset`
   * - ``starfinder.dataset.FOV``
     - :doc:`dataset`
   * - ``starfinder.dataset.RoundState``
     - :doc:`dataset`
   * - ``starfinder.dataset.CropWindow``
     - :doc:`dataset`
   * - ``starfinder.dataset.SubtileConfig``
     - :doc:`dataset`
   * - ``starfinder.dataset.PipelineConfig``
     - :doc:`dataset`
   * - ``starfinder.dataset.ExecutionConfig``
     - :doc:`dataset`
   * - ``starfinder.dataset.RegistrationStep``
     - :doc:`dataset`
   * - ``starfinder.dataset.RecoveryConfig``
     - :doc:`dataset`
   * - ``starfinder.dataset.WorkflowConfig``
     - :doc:`dataset`
   * - ``starfinder.dataset.from_workflow_config``
     - :doc:`dataset`
   * - ``starfinder.io.export_spots``
     - :doc:`io`
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
   * - ``starfinder.evaluation.registration.evaluate_registration``
     - :doc:`evaluation.registration`
   * - ``starfinder.evaluation.registration.evaluate_translation``
     - :doc:`evaluation.registration`
   * - ``starfinder.evaluation.spot_finding.evaluate_spots``
     - :doc:`evaluation.registration`
   * - ``starfinder.evaluation.barcode.evaluate_decoding``
     - :doc:`evaluation.registration`
   * - ``starfinder.synthetic.SyntheticConfig``
     - :doc:`synthetic`
   * - ``starfinder.synthetic.SyntheticDataset``
     - :doc:`synthetic`
   * - ``starfinder.synthetic.generate_codebook``
     - :doc:`synthetic`
   * - ``starfinder.synthetic.generate_dataset``
     - :doc:`synthetic`
   * - ``starfinder.synthetic.generate_displacement_field``
     - :doc:`synthetic`
   * - ``starfinder.synthetic.generate_registration_pairs``
     - :doc:`synthetic`
   * - ``starfinder.synthetic.generate_volume``
     - :doc:`synthetic`
   * - ``starfinder.synthetic.get_preset_config``
     - :doc:`synthetic`
   * - ``starfinder.synthetic.render_spots``
     - :doc:`synthetic`
   * - ``starfinder.preprocessing.project_image``
     - :doc:`preprocessing`

   * - ``starfinder.benchmark.BenchmarkCase``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.BenchmarkTrialResult``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.run_benchmark``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.evaluate_benchmark``
     - :doc:`benchmark`
   * - ``starfinder.benchmark.report_benchmark``
     - :doc:`benchmark`

Internal interfaces
-------------------

* Registration numerical modules are private. Removed public modules and
  wrappers have no aliases; use ``estimate_transform`` and ``apply_transform``.
* ``starfinder.__main__.main`` implements the ``starfinder`` CLI;
  ``uv run starfinder --help`` lists the supported commands.
* Benchmark storage, adapters, measurement and rendering helpers are private.
  Imported dependencies, loggers and nested closures are not STARfinder APIs.

When changing exports, update this inventory and the relevant autosummary list.
Generated object stubs and HTML are build products; do not commit them.
