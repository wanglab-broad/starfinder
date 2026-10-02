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
:py:class:`~starfinder.barcode.BarcodeLayout` describes its segments. See
:doc:`../readout-contract`.

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
   DirectAssignmentConfig
   DirectPanel
   encode_bases
   EncodingConfig
   ENCODINGS
   EncodingSpec
   extract_intensities
   filter_reads
   IntensityExtractionResult
   InvalidIntensityError
   load_codebook
   load_direct_panel
   NeighborhoodSumConfig
   OneBaseEncodingConfig
   ReadFilterConfig
   ReadFilteringResult
   Segment
   WtaDecoderConfig
