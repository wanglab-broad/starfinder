starfinder.readout
==================

Codebooks, intensity extraction, read decoding and read quality control. The
:doc:`readout guide <../readout-contract>` shows the workflow; the
:doc:`readout reference <../readout-algorithms>` lists methods, options and tables.

.. currentmodule:: starfinder.readout

Codebooks and encodings
-----------------------

.. autosummary::
   :toctree: generated

   BarcodeLayout
   bases_to_colors
   Codebook
   colors_to_bases
   DirectPanel
   EncodingConfig
   ENCODINGS
   EncodingSpec
   load_codebook
   load_direct_panel
   OneBaseEncodingConfig
   Segment

Intensity extraction
--------------------

.. autosummary::
   :toctree: generated

   extract_intensities
   IntensityExtractionResult
   InvalidIntensityError
   LocalBackgroundConfig
   NeighborhoodSumConfig

Decoding
--------

.. autosummary::
   :toctree: generated

   CodebookAwareDecoderConfig
   decode_direct
   decode_reads
   DECODING_METHODS
   DecodingSpec
   DirectDecoderConfig
   ReadDecodingResult
   WtaDecoderConfig

Scoring, deduplication and filtering
------------------------------------

.. autosummary::
   :toctree: generated

   deduplicate_reads
   DeduplicationConfig
   filter_reads
   ReadDeduplicationResult
   ReadFilterConfig
   ReadFilteringResult
   ReadScoreConfig
   ReadScoringResult
   score_reads

Inspection and export
---------------------

.. autosummary::
   :toctree: generated

   explain_read
   export_spots
   inspect_read
   plot_read
   summarize_reads
