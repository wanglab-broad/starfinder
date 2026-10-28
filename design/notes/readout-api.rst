starfinder.readout
==================

Independent codebook validation, intensity extraction, decoding and filtering.
See `Array and coordinate contracts <../../docs/api/contracts.md>`_ for labels and identities, and `Readout reference <../../docs/readout-algorithms.md>`_
for score fields. The `readout specification
<https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/readout.md>`_
gives numerical policies.

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
summary and decision inspection diagnostics. See `Readout reference <../../docs/readout-algorithms.md>`_.
:py:func:`~starfinder.readout.export_spots` writes the joined spots as a typed,
zero-based CSV with their identities: the accepted molecules (``molecules.csv``) or every
read (``reads.csv``).

