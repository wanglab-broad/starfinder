starfinder.readout
==================

Independent codebook validation, intensity extraction, decoding and filtering.
See :doc:`contracts` for labels, identities, score meanings and numerical policy.

:py:data:`~starfinder.readout.ENCODINGS` maps each encoding config type
(``two_base``, ``one_base``) to its :py:class:`~starfinder.readout.EncodingSpec`, and
:py:data:`~starfinder.readout.DECODING_METHODS` maps each decoder config type (``wta``,
``codebook_aware`` for readout mode ``multiplexed``, ``direct`` for readout mode
``direct``) to its :py:class:`~starfinder.readout.DecodingSpec`; lookups use
the exact config type. In readout mode ``direct``, a
:py:class:`~starfinder.readout.DirectPanel` gives the gene of each (round, channel) and
:py:func:`~starfinder.readout.decode_direct` assigns each candidate from its own round.
A codebook row is an entry (``entry_id``) of a gene, and
:py:class:`~starfinder.readout.BarcodeLayout` describes its segments. Extraction
measures a local background and noise next to the sums
(:py:class:`~starfinder.readout.LocalBackgroundConfig`, on by default), and
:py:func:`~starfinder.readout.score_reads` adds the shared read-QC score, a ranking
that never changes a call. Optional deduplication
(:py:func:`~starfinder.readout.deduplicate_reads`, off by default) marks cross-channel
reads of one amplicon as duplicates of one representative, which
:py:func:`~starfinder.readout.filter_reads` then rejects, and
:py:func:`~starfinder.readout.inspect_read`,
:py:func:`~starfinder.readout.summarize_reads` and
:py:func:`~starfinder.readout.explain_read` are the read inspection, population
summary and decision inspection diagnostics. See :doc:`../readout-contract`.
:py:func:`~starfinder.readout.export_spots` writes the joined spots as a one-based
MATLAB-compatible CSV.

.. currentmodule:: starfinder.readout

.. autosummary::
   :toctree: generated/

   BarcodeLayout
   bases_to_colors
   Codebook
   CodebookAwareDecoderConfig
   colors_to_bases
   decode_direct
   decode_reads
   DECODING_METHODS
   DecodingSpec
   deduplicate_reads
   DeduplicationConfig
   DirectDecoderConfig
   DirectPanel
   EncodingConfig
   ENCODINGS
   EncodingSpec
   explain_read
   export_spots
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
   ReadDecodingResult
   ReadDeduplicationResult
   ReadFilterConfig
   ReadFilteringResult
   ReadScoreConfig
   ReadScoringResult
   score_reads
   Segment
   summarize_reads
   WtaDecoderConfig
