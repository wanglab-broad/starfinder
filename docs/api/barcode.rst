starfinder.barcode
==================

Color labels are one-based channel numbers encoded as strings. Spatial coordinates remain zero-based. Raw tensors have shape ``(N, C, R)``. See :doc:`contracts`.

.. currentmodule:: starfinder.barcode

.. autosummary::
   :toctree: generated

   build_one_error_index
   candidate_sequences
   channel_probabilities
   decode_color_seq
   decode_codebook_aware
   encode_bases
   extract_from_location
   extract_intensity_tensor
   filter_reads
   load_codebook
   score_candidates
   wta_color_sequences

.. autodata:: starfinder.barcode.encoding.BASE_PAIR_TO_COLOR

.. autodata:: starfinder.barcode.encoding.COLOR_TO_BASE_PAIRS

.. autodata:: starfinder.barcode.encoding.COLOR_TO_CHANNEL
