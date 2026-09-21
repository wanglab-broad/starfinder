Public API coverage inventory
=============================

Supported names below match the explicit ``__all__`` of the installed checkout.
Replaced Python interfaces have no compatibility aliases. Private implementation
modules, imported dependencies and CLI implementation helpers are excluded.
Class pages include public methods, properties and dataclass fields; signatures
show literal defaults (``<factory>`` means a fresh per-instance value).
This inventory is software coverage, not scientific qualification.

Root convenience exports
------------------------

Root exports refer to the same objects as their owning namespaces; modules and
``__version__`` are listed as metadata rather than callable APIs.

* ``starfinder.__version__`` (module or version metadata)
* ``starfinder.apply_transform`` → :py:obj:`starfinder.registration.apply_transform`
* ``starfinder.barcode`` (module or version metadata)
* ``starfinder.Dataset`` → :py:obj:`starfinder.dataset.Dataset`
* ``starfinder.decode_barcodes`` → :py:obj:`starfinder.barcode.decode_barcodes`
* ``starfinder.estimate_transform`` → :py:obj:`starfinder.registration.estimate_transform`
* ``starfinder.extract_intensities`` → :py:obj:`starfinder.barcode.extract_intensities`
* ``starfinder.filter_reads`` → :py:obj:`starfinder.barcode.filter_reads`
* ``starfinder.filter_tophat`` → :py:obj:`starfinder.preprocessing.filter_tophat`
* ``starfinder.find_spots`` → :py:obj:`starfinder.spot_finding.find_spots`
* ``starfinder.FOV`` → :py:obj:`starfinder.dataset.FOV`
* ``starfinder.ImageMetadata`` → :py:obj:`starfinder.image.ImageMetadata`
* ``starfinder.load_codebook`` → :py:obj:`starfinder.barcode.load_codebook`
* ``starfinder.load_round`` → :py:obj:`starfinder.io.load_round`
* ``starfinder.load_volume`` → :py:obj:`starfinder.io.load_volume`
* ``starfinder.match_histogram`` → :py:obj:`starfinder.preprocessing.match_histogram`
* ``starfinder.normalize_intensity`` → :py:obj:`starfinder.preprocessing.normalize_intensity`
* ``starfinder.preprocessing`` (module or version metadata)
* ``starfinder.project_image`` → :py:obj:`starfinder.preprocessing.project_image`
* ``starfinder.reconstruct_background`` → :py:obj:`starfinder.preprocessing.reconstruct_background`
* ``starfinder.registration`` (module or version metadata)
* ``starfinder.save_volume`` → :py:obj:`starfinder.io.save_volume`
* ``starfinder.spot_finding`` (module or version metadata)

Owning namespaces (alphabetical)
--------------------------------

starfinder.barcode
~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.barcode.BarcodeDecodingResult`
* :py:obj:`starfinder.barcode.Codebook`
* :py:obj:`starfinder.barcode.CodebookAwareDecoderConfig`
* :py:obj:`starfinder.barcode.decode_barcodes`
* :py:obj:`starfinder.barcode.decode_color_sequence`
* :py:obj:`starfinder.barcode.encode_bases`
* :py:obj:`starfinder.barcode.EncodingConfig`
* :py:obj:`starfinder.barcode.extract_intensities`
* :py:obj:`starfinder.barcode.filter_reads`
* :py:obj:`starfinder.barcode.IntensityExtractionResult`
* :py:obj:`starfinder.barcode.InvalidIntensityError`
* :py:obj:`starfinder.barcode.load_codebook`
* :py:obj:`starfinder.barcode.NeighborhoodSumConfig`
* :py:obj:`starfinder.barcode.ReadFilterConfig`
* :py:obj:`starfinder.barcode.ReadFilteringResult`
* :py:obj:`starfinder.barcode.WtaDecoderConfig`

starfinder.benchmark
~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.benchmark.BenchmarkCase`
* :py:obj:`starfinder.benchmark.BenchmarkTrialResult`
* :py:obj:`starfinder.benchmark.evaluate_benchmark`
* :py:obj:`starfinder.benchmark.report_benchmark`
* :py:obj:`starfinder.benchmark.run_benchmark`

starfinder.dataset
~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.dataset.CropWindow`
* :py:obj:`starfinder.dataset.Dataset`
* :py:obj:`starfinder.dataset.ExecutionConfig`
* :py:obj:`starfinder.dataset.FOV`
* :py:obj:`starfinder.dataset.from_workflow_config`
* :py:obj:`starfinder.dataset.PipelineConfig`
* :py:obj:`starfinder.dataset.RecoveryConfig`
* :py:obj:`starfinder.dataset.RegistrationStep`
* :py:obj:`starfinder.dataset.RoundState`
* :py:obj:`starfinder.dataset.SubtileConfig`
* :py:obj:`starfinder.dataset.WorkflowConfig`

starfinder.evaluation
~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.evaluation.EvaluationResult`

starfinder.evaluation.barcode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.evaluation.barcode.evaluate_decoding`

starfinder.evaluation.matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.evaluation.matching.match_points`

starfinder.evaluation.registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.evaluation.registration.evaluate_landmark_alignment`
* :py:obj:`starfinder.evaluation.registration.evaluate_mask_overlap`
* :py:obj:`starfinder.evaluation.registration.evaluate_registration`
* :py:obj:`starfinder.evaluation.registration.evaluate_translation`
* :py:obj:`starfinder.evaluation.registration.normalized_cross_correlation`
* :py:obj:`starfinder.evaluation.registration.structural_similarity`

starfinder.evaluation.spot_finding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.evaluation.spot_finding.evaluate_spots`

starfinder.image
~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.image.ImageMetadata`
* :py:obj:`starfinder.image.IncompatibleGeometryError`
* :py:obj:`starfinder.image.InvalidImageError`

starfinder.io
~~~~~~~~~~~~~

* :py:obj:`starfinder.io.convert_image`
* :py:obj:`starfinder.io.CandidateCheckpoint`
* :py:obj:`starfinder.io.CandidateSaveResult`
* :py:obj:`starfinder.io.load_candidate_checkpoint`
* :py:obj:`starfinder.io.save_candidate_checkpoint`
* :py:obj:`starfinder.io.export_spots`
* :py:obj:`starfinder.io.ImageCheckpoint`
* :py:obj:`starfinder.io.ImageConversionConfig`
* :py:obj:`starfinder.io.ImageLayer`
* :py:obj:`starfinder.io.ImageLoadConfig`
* :py:obj:`starfinder.io.ImageLoadResult`
* :py:obj:`starfinder.io.ImageProcessingState`
* :py:obj:`starfinder.io.load_image_checkpoint`
* :py:obj:`starfinder.io.load_round`
* :py:obj:`starfinder.io.load_volume`
* :py:obj:`starfinder.io.save_image_checkpoint`
* :py:obj:`starfinder.io.save_volume`

starfinder.preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.preprocessing.filter_tophat`
* :py:obj:`starfinder.preprocessing.HistogramMatchingConfig`
* :py:obj:`starfinder.preprocessing.match_histogram`
* :py:obj:`starfinder.preprocessing.MinMaxNormalizationConfig`
* :py:obj:`starfinder.preprocessing.normalize_intensity`
* :py:obj:`starfinder.preprocessing.project_image`
* :py:obj:`starfinder.preprocessing.ProjectionConfig`
* :py:obj:`starfinder.preprocessing.reconstruct_background`
* :py:obj:`starfinder.preprocessing.ReconstructionConfig`
* :py:obj:`starfinder.preprocessing.TophatConfig`

starfinder.provenance
~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.provenance.read_run`
* :py:obj:`starfinder.provenance.RunRecorder`

starfinder.registration
~~~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.registration.apply_transform`
* :py:obj:`starfinder.registration.CpdConfig`
* :py:obj:`starfinder.registration.DemonsConfig`
* :py:obj:`starfinder.registration.DenseDisplacementTransform`
* :py:obj:`starfinder.registration.estimate_transform`
* :py:obj:`starfinder.registration.InsufficientLandmarksError`
* :py:obj:`starfinder.registration.InvalidRegistrationConfigError`
* :py:obj:`starfinder.registration.RegistrationBackendUnavailableError`
* :py:obj:`starfinder.registration.RegistrationDiagnostics`
* :py:obj:`starfinder.registration.RegistrationEstimationError`
* :py:obj:`starfinder.registration.RegistrationResult`
* :py:obj:`starfinder.registration.TpsConfig`
* :py:obj:`starfinder.registration.TranslationConfig`
* :py:obj:`starfinder.registration.TranslationTransform`
* :py:obj:`starfinder.registration.UnsupportedTransformOperationError`
* :py:obj:`starfinder.registration.WarpConfig`

starfinder.spot_finding
~~~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.spot_finding.find_spots`
* :py:obj:`starfinder.spot_finding.LocalMaximaConfig`
* :py:obj:`starfinder.spot_finding.NoiseLandmarkConfig`
* :py:obj:`starfinder.spot_finding.PercentileCentroidConfig`
* :py:obj:`starfinder.spot_finding.SpotFindingResult`

starfinder.synthetic
~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.synthetic.formed_scene_preset`
* :py:obj:`starfinder.synthetic.FormedScene`
* :py:obj:`starfinder.synthetic.FormedSceneConfig`
* :py:obj:`starfinder.synthetic.generate_codebook`
* :py:obj:`starfinder.synthetic.generate_dataset`
* :py:obj:`starfinder.synthetic.generate_displacement_field`
* :py:obj:`starfinder.synthetic.generate_formed_scene`
* :py:obj:`starfinder.synthetic.generate_registration_pairs`
* :py:obj:`starfinder.synthetic.generate_volume`
* :py:obj:`starfinder.synthetic.get_preset_config`
* :py:obj:`starfinder.synthetic.render_spots`
* :py:obj:`starfinder.synthetic.ScalarDistribution`
* :py:obj:`starfinder.synthetic.SyntheticConfig`
* :py:obj:`starfinder.synthetic.SyntheticDataset`

Update this inventory and the owning autosummary list when exports change.
Generated object stubs and HTML are disposable build products, never authored API.
