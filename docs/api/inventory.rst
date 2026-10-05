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

starfinder.assignment
~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.assignment.assign_molecules`
* :py:obj:`starfinder.assignment.ASSIGNMENT_STATUSES`
* :py:obj:`starfinder.assignment.AssignmentConfig`
* :py:obj:`starfinder.assignment.AssignmentResult`
* :py:obj:`starfinder.assignment.CELL_CORRESPONDENCE`
* :py:obj:`starfinder.assignment.CELL_STATUSES`
* :py:obj:`starfinder.assignment.COMPARTMENT_STATES`
* :py:obj:`starfinder.assignment.CorrespondenceConfig`
* :py:obj:`starfinder.assignment.match_nuclei`
* :py:obj:`starfinder.assignment.molecule_table`
* :py:obj:`starfinder.assignment.molecule_table_from_csv`
* :py:obj:`starfinder.assignment.MoleculeTable`
* :py:obj:`starfinder.assignment.NUCLEUS_STATUSES`
* :py:obj:`starfinder.assignment.plot_assignment`
* :py:obj:`starfinder.assignment.sample_labels`
* :py:obj:`starfinder.assignment.summarize_assignment`

starfinder.barcode
~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.barcode.assign_direct`
* :py:obj:`starfinder.barcode.BarcodeDecodingResult`
* :py:obj:`starfinder.barcode.BarcodeLayout`
* :py:obj:`starfinder.barcode.Codebook`
* :py:obj:`starfinder.barcode.CodebookAwareDecoderConfig`
* :py:obj:`starfinder.barcode.decode_barcodes`
* :py:obj:`starfinder.barcode.decode_color_sequence`
* :py:obj:`starfinder.barcode.DECODING_METHODS`
* :py:obj:`starfinder.barcode.DecodingSpec`
* :py:obj:`starfinder.barcode.deduplicate_reads`
* :py:obj:`starfinder.barcode.DeduplicationConfig`
* :py:obj:`starfinder.barcode.DirectAssignmentConfig`
* :py:obj:`starfinder.barcode.DirectPanel`
* :py:obj:`starfinder.barcode.encode_bases`
* :py:obj:`starfinder.barcode.EncodingConfig`
* :py:obj:`starfinder.barcode.ENCODINGS`
* :py:obj:`starfinder.barcode.EncodingSpec`
* :py:obj:`starfinder.barcode.explain_read`
* :py:obj:`starfinder.barcode.extract_intensities`
* :py:obj:`starfinder.barcode.filter_reads`
* :py:obj:`starfinder.barcode.inspect_read`
* :py:obj:`starfinder.barcode.IntensityExtractionResult`
* :py:obj:`starfinder.barcode.InvalidIntensityError`
* :py:obj:`starfinder.barcode.load_codebook`
* :py:obj:`starfinder.barcode.load_direct_panel`
* :py:obj:`starfinder.barcode.LocalBackgroundConfig`
* :py:obj:`starfinder.barcode.NeighborhoodSumConfig`
* :py:obj:`starfinder.barcode.OneBaseEncodingConfig`
* :py:obj:`starfinder.barcode.plot_read`
* :py:obj:`starfinder.barcode.ReadDeduplicationResult`
* :py:obj:`starfinder.barcode.ReadFilterConfig`
* :py:obj:`starfinder.barcode.ReadFilteringResult`
* :py:obj:`starfinder.barcode.ReadScoreConfig`
* :py:obj:`starfinder.barcode.ReadScoringResult`
* :py:obj:`starfinder.barcode.score_reads`
* :py:obj:`starfinder.barcode.Segment`
* :py:obj:`starfinder.barcode.summarize_reads`
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

* :py:obj:`starfinder.dataset.ChannelInfo`
* :py:obj:`starfinder.dataset.CheckpointConfig`
* :py:obj:`starfinder.dataset.CropWindow`
* :py:obj:`starfinder.dataset.Dataset`
* :py:obj:`starfinder.dataset.ExecutionConfig`
* :py:obj:`starfinder.dataset.ExternalReference`
* :py:obj:`starfinder.dataset.FOV`
* :py:obj:`starfinder.dataset.from_workflow_config`
* :py:obj:`starfinder.dataset.PipelineConfig`
* :py:obj:`starfinder.dataset.RecoveryConfig`
* :py:obj:`starfinder.dataset.RegistrationRecipe`
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
* :py:obj:`starfinder.evaluation.barcode.evaluate_deduplication`
* :py:obj:`starfinder.evaluation.barcode.ranking_quality`

starfinder.evaluation.matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.evaluation.matching.match_points`

starfinder.evaluation.registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.evaluation.registration.evaluate_displacement_field`
* :py:obj:`starfinder.evaluation.registration.evaluate_landmark_alignment`
* :py:obj:`starfinder.evaluation.registration.evaluate_mask_overlap`
* :py:obj:`starfinder.evaluation.registration.evaluate_registration`
* :py:obj:`starfinder.evaluation.registration.evaluate_translation`
* :py:obj:`starfinder.evaluation.registration.normalized_cross_correlation`
* :py:obj:`starfinder.evaluation.registration.registration_qc`
* :py:obj:`starfinder.evaluation.registration.structural_similarity`

starfinder.evaluation.spot_finding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.evaluation.spot_finding.classify_detections`
* :py:obj:`starfinder.evaluation.spot_finding.evaluate_spots`
* :py:obj:`starfinder.evaluation.spot_finding.localization_errors`

starfinder.image
~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.image.ImageMetadata`
* :py:obj:`starfinder.image.IncompatibleGeometryError`
* :py:obj:`starfinder.image.InvalidImageError`

starfinder.io
~~~~~~~~~~~~~

* :py:obj:`starfinder.io.convert_image`
* :py:obj:`starfinder.io.export_spots`
* :py:obj:`starfinder.io.ImageConversionConfig`
* :py:obj:`starfinder.io.ImageLoadConfig`
* :py:obj:`starfinder.io.ImageLoadResult`
* :py:obj:`starfinder.io.load_round`
* :py:obj:`starfinder.io.load_volume`
* :py:obj:`starfinder.io.load_volume_zyxc`
* :py:obj:`starfinder.io.read_checkpoint`
* :py:obj:`starfinder.io.save_volume`

starfinder.preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.preprocessing.Background3DConfig`
* :py:obj:`starfinder.preprocessing.filter_tophat`
* :py:obj:`starfinder.preprocessing.histogram_percentile`
* :py:obj:`starfinder.preprocessing.HistogramMatchingConfig`
* :py:obj:`starfinder.preprocessing.HistogramSummary`
* :py:obj:`starfinder.preprocessing.match_histogram`
* :py:obj:`starfinder.preprocessing.merge_histograms`
* :py:obj:`starfinder.preprocessing.MinMaxNormalizationConfig`
* :py:obj:`starfinder.preprocessing.normalize_intensity`
* :py:obj:`starfinder.preprocessing.normalize_percentile`
* :py:obj:`starfinder.preprocessing.PercentileNormalizationConfig`
* :py:obj:`starfinder.preprocessing.PREPROCESSING_METHODS`
* :py:obj:`starfinder.preprocessing.PreprocessingRecipe`
* :py:obj:`starfinder.preprocessing.PreprocessingSpec`
* :py:obj:`starfinder.preprocessing.PreprocessingStep`
* :py:obj:`starfinder.preprocessing.project_image`
* :py:obj:`starfinder.preprocessing.ProjectionConfig`
* :py:obj:`starfinder.preprocessing.read_histograms`
* :py:obj:`starfinder.preprocessing.read_supplied_statistics`
* :py:obj:`starfinder.preprocessing.reconstruct_background`
* :py:obj:`starfinder.preprocessing.ReconstructionConfig`
* :py:obj:`starfinder.preprocessing.run_step`
* :py:obj:`starfinder.preprocessing.scalar_background_histograms`
* :py:obj:`starfinder.preprocessing.ScalarBackgroundConfig`
* :py:obj:`starfinder.preprocessing.step_config_type`
* :py:obj:`starfinder.preprocessing.step_spec`
* :py:obj:`starfinder.preprocessing.StepContext`
* :py:obj:`starfinder.preprocessing.StepResult`
* :py:obj:`starfinder.preprocessing.subtract_background_3d`
* :py:obj:`starfinder.preprocessing.subtract_scalar_background`
* :py:obj:`starfinder.preprocessing.summarize_histograms`
* :py:obj:`starfinder.preprocessing.summary_stage`
* :py:obj:`starfinder.preprocessing.supplied_section`
* :py:obj:`starfinder.preprocessing.supplied_statistics`
* :py:obj:`starfinder.preprocessing.TophatConfig`
* :py:obj:`starfinder.preprocessing.write_histograms`
* :py:obj:`starfinder.preprocessing.write_supplied_statistics`

starfinder.registration
~~~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.registration.AffineConfig`
* :py:obj:`starfinder.registration.AffineTransform`
* :py:obj:`starfinder.registration.apply_transform`
* :py:obj:`starfinder.registration.BSplineConfig`
* :py:obj:`starfinder.registration.BSplineTransform`
* :py:obj:`starfinder.registration.CpdConfig`
* :py:obj:`starfinder.registration.DemonsConfig`
* :py:obj:`starfinder.registration.DenseDisplacementTransform`
* :py:obj:`starfinder.registration.estimate_transform`
* :py:obj:`starfinder.registration.InsufficientLandmarksError`
* :py:obj:`starfinder.registration.InvalidRegistrationConfigError`
* :py:obj:`starfinder.registration.REGISTRATION_METHODS`
* :py:obj:`starfinder.registration.RegistrationBackendUnavailableError`
* :py:obj:`starfinder.registration.RegistrationDiagnostics`
* :py:obj:`starfinder.registration.RegistrationEstimationError`
* :py:obj:`starfinder.registration.RegistrationQcConfig`
* :py:obj:`starfinder.registration.RegistrationRejectedError`
* :py:obj:`starfinder.registration.RegistrationResult`
* :py:obj:`starfinder.registration.RegistrationSignalConfig`
* :py:obj:`starfinder.registration.RegistrationSpec`
* :py:obj:`starfinder.registration.RigidConfig`
* :py:obj:`starfinder.registration.TpsConfig`
* :py:obj:`starfinder.registration.TransformChain`
* :py:obj:`starfinder.registration.TranslationConfig`
* :py:obj:`starfinder.registration.TranslationTransform`
* :py:obj:`starfinder.registration.UnsupportedTransformOperationError`
* :py:obj:`starfinder.registration.WarpConfig`

starfinder.segmentation
~~~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.segmentation.CellposeConfig`
* :py:obj:`starfinder.segmentation.composite_nuclei_amplicon`
* :py:obj:`starfinder.segmentation.CompositeConfig`
* :py:obj:`starfinder.segmentation.enhance_with_flamingo`
* :py:obj:`starfinder.segmentation.expand_labels`
* :py:obj:`starfinder.segmentation.ExpandLabelsConfig`
* :py:obj:`starfinder.segmentation.extend_labels_through_z`
* :py:obj:`starfinder.segmentation.FlamingoEnhancementConfig`
* :py:obj:`starfinder.segmentation.import_labels`
* :py:obj:`starfinder.segmentation.InputChannel`
* :py:obj:`starfinder.segmentation.KNOWN_MODELS`
* :py:obj:`starfinder.segmentation.KnownModel`
* :py:obj:`starfinder.segmentation.LabelImportConfig`
* :py:obj:`starfinder.segmentation.labels_to_grid`
* :py:obj:`starfinder.segmentation.MethodContext`
* :py:obj:`starfinder.segmentation.MissingModelError`
* :py:obj:`starfinder.segmentation.ModelFile`
* :py:obj:`starfinder.segmentation.ModelHashMismatchError`
* :py:obj:`starfinder.segmentation.normalize_percentiles`
* :py:obj:`starfinder.segmentation.reference_grid_from_file`
* :py:obj:`starfinder.segmentation.ReferenceGrid`
* :py:obj:`starfinder.segmentation.rescale_input`
* :py:obj:`starfinder.segmentation.resolve_model`
* :py:obj:`starfinder.segmentation.SeededWatershedConfig`
* :py:obj:`starfinder.segmentation.segment`
* :py:obj:`starfinder.segmentation.SEGMENTATION_METHODS`
* :py:obj:`starfinder.segmentation.SegmentationBackendUnavailableError`
* :py:obj:`starfinder.segmentation.SegmentationInput`
* :py:obj:`starfinder.segmentation.SegmentationPlan`
* :py:obj:`starfinder.segmentation.SegmentationResult`
* :py:obj:`starfinder.segmentation.SegmentationRun`
* :py:obj:`starfinder.segmentation.SegmentationSpec`
* :py:obj:`starfinder.segmentation.StarDistConfig`
* :py:obj:`starfinder.segmentation.to_label_dtype`
* :py:obj:`starfinder.segmentation.ZExtensionConfig`

starfinder.spot_finding
~~~~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.spot_finding.ChannelOverride`
* :py:obj:`starfinder.spot_finding.fetch_weights`
* :py:obj:`starfinder.spot_finding.find_spots`
* :py:obj:`starfinder.spot_finding.KNOWN_WEIGHTS`
* :py:obj:`starfinder.spot_finding.KnownWeights`
* :py:obj:`starfinder.spot_finding.LocalMaximaConfig`
* :py:obj:`starfinder.spot_finding.MissingWeightsError`
* :py:obj:`starfinder.spot_finding.NoiseLandmarkConfig`
* :py:obj:`starfinder.spot_finding.PercentileCentroidConfig`
* :py:obj:`starfinder.spot_finding.PiscisConfig`
* :py:obj:`starfinder.spot_finding.plot_detections`
* :py:obj:`starfinder.spot_finding.resolve_weights`
* :py:obj:`starfinder.spot_finding.SPOT_FINDING_METHODS`
* :py:obj:`starfinder.spot_finding.SpotFindingBackendUnavailableError`
* :py:obj:`starfinder.spot_finding.SpotFindingPlan`
* :py:obj:`starfinder.spot_finding.SpotFindingResult`
* :py:obj:`starfinder.spot_finding.SpotFindingSpec`
* :py:obj:`starfinder.spot_finding.SpotFindingWarning`
* :py:obj:`starfinder.spot_finding.SpotiflowConfig`
* :py:obj:`starfinder.spot_finding.StarfishLogConfig`
* :py:obj:`starfinder.spot_finding.WeightsFile`
* :py:obj:`starfinder.spot_finding.WeightsHashMismatchError`

starfinder.synthetic
~~~~~~~~~~~~~~~~~~~~

* :py:obj:`starfinder.synthetic.BackgroundConfig`
* :py:obj:`starfinder.synthetic.BENCHMARK_PRESETS`
* :py:obj:`starfinder.synthetic.benchmark_scene_preset`
* :py:obj:`starfinder.synthetic.CALIBRATED_CONDITIONS`
* :py:obj:`starfinder.synthetic.calibrated_scene_preset`
* :py:obj:`starfinder.synthetic.deformation_geometry`
* :py:obj:`starfinder.synthetic.DEFORMATION_PRESETS`
* :py:obj:`starfinder.synthetic.development_codebook`
* :py:obj:`starfinder.synthetic.DEVELOPMENT_FACTORS`
* :py:obj:`starfinder.synthetic.DEVELOPMENT_FIXTURES`
* :py:obj:`starfinder.synthetic.development_preset_factors`
* :py:obj:`starfinder.synthetic.development_scene_preset`
* :py:obj:`starfinder.synthetic.DEVELOPMENT_SIZES`
* :py:obj:`starfinder.synthetic.formed_scene_preset`
* :py:obj:`starfinder.synthetic.FormedScene`
* :py:obj:`starfinder.synthetic.FormedSceneConfig`
* :py:obj:`starfinder.synthetic.forward_displacement`
* :py:obj:`starfinder.synthetic.generate_codebook`
* :py:obj:`starfinder.synthetic.generate_dataset`
* :py:obj:`starfinder.synthetic.generate_formed_scene`
* :py:obj:`starfinder.synthetic.generate_registration_pair`
* :py:obj:`starfinder.synthetic.GeometryConfig`
* :py:obj:`starfinder.synthetic.NoiseConfig`
* :py:obj:`starfinder.synthetic.PRESET_VERSION`
* :py:obj:`starfinder.synthetic.ReadoutEffectsConfig`
* :py:obj:`starfinder.synthetic.registration_scene_preset`
* :py:obj:`starfinder.synthetic.save_formed_scene`
* :py:obj:`starfinder.synthetic.ScalarDistribution`
* :py:obj:`starfinder.synthetic.SCENE_PRESETS`
* :py:obj:`starfinder.synthetic.SyntheticDataset`
* :py:obj:`starfinder.synthetic.TextureConfig`

Update this inventory and the owning autosummary list when exports change.
Generated object stubs and HTML are disposable build products, never authored API.
