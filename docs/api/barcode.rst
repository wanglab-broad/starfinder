starfinder.barcode
==================

Independent codebook validation, intensity extraction, decoding and filtering.
See :doc:`contracts` for labels, identities, score meanings and numerical policy.

:py:data:`~starfinder.barcode.ENCODINGS` maps each encoding config type
(``two_base``, ``one_base``) to its :py:class:`~starfinder.barcode.EncodingSpec`, and
:py:data:`~starfinder.barcode.DECODING_METHODS` maps each decoder config type (``wta``,
``codebook_aware`` for readout mode ``multiplexed``, ``direct`` for readout mode
``direct``) to its :py:class:`~starfinder.barcode.DecodingSpec`; lookups use
the exact config type. In readout mode ``direct``, a
:py:class:`~starfinder.barcode.DirectPanel` gives the gene of each (round, channel) and
:py:func:`~starfinder.barcode.assign_direct` assigns each candidate from its own round.
A codebook row is an entry (``entry_id``) of a gene, and
:py:class:`~starfinder.barcode.BarcodeLayout` describes its segments. Extraction
measures a local background and noise next to the sums
(:py:class:`~starfinder.barcode.LocalBackgroundConfig`, on by default), and
:py:func:`~starfinder.barcode.score_reads` adds the shared read-QC score, a ranking
that never changes a call. Optional deduplication
(:py:func:`~starfinder.barcode.deduplicate_reads`, off by default) marks cross-channel
reads of one amplicon as duplicates of one representative, which
:py:func:`~starfinder.barcode.filter_reads` then rejects, and
:py:func:`~starfinder.barcode.inspect_read`,
:py:func:`~starfinder.barcode.summarize_reads` and
:py:func:`~starfinder.barcode.explain_read` are the read inspection, population
summary and decision inspection diagnostics. See :doc:`../readout-contract`.

.. currentmodule:: starfinder.barcode

.. autosummary::
   :toctree: generated/

   assign_direct
   BarcodeDecodingResult
   BarcodeLayout
   Codebook
   CodebookAwareDecoderConfig
   decode_barcodes
   decode_color_sequence
   DECODING_METHODS
   DecodingSpec
   deduplicate_reads
   DeduplicationConfig
   DirectAssignmentConfig
   DirectPanel
   encode_bases
   EncodingConfig
   ENCODINGS
   EncodingSpec
   explain_read
   extract_intensities
   filter_reads
   inspect_read
   IntensityExtractionResult
   InvalidIntensityError
   load_codebook
   load_direct_panel
   LocalBackgroundConfig
   NeighborhoodSumConfig
   OneBaseEncodingConfig
   plot_read
   ReadDeduplicationResult
   ReadFilterConfig
   ReadFilteringResult
   ReadScoreConfig
   ReadScoringResult
   score_reads
   Segment
   summarize_reads
   WtaDecoderConfig
