starfinder.barcode
==================

Independent codebook validation, intensity extraction, decoding and filtering.
See :doc:`contracts` for labels, identities, score meanings and numerical policy.

.. currentmodule:: starfinder.barcode

.. autosummary::
   :toctree: generated/

   BarcodeDecodingResult
   Codebook
   CodebookAwareDecoderConfig
   EncodingConfig
   IntensityExtractionResult
   InvalidIntensityError
   NeighborhoodSumConfig
   ReadFilterConfig
   ReadFilteringResult
   WtaDecoderConfig
   decode_barcodes
   decode_color_sequence
   encode_bases
   extract_intensities
   filter_reads
   load_codebook
