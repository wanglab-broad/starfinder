starfinder.dataset
==================

Dataset/FOV coordination with typed scientific stages and separate residency.
FOV remains the public acronym. See `Array and coordinate contracts <../../docs/api/contracts.md>`_ and `Dataset and FOV coordination <../../docs/coordination.md>`_.

Evaluating a ``Dataset`` or ``FOV`` shows a plain-text summary of names and
structure, never array values or table rows. ``FOV.results`` is a read-only
mapping of the stages that have run (``registration``, ``spot_finding``,
``extraction``, ``decoding``, ``filtering``) to the stored result objects;
the existing attributes remain. ``Codebook``, the spot finding, extraction,
decoding, filtering, registration and evaluation results have one-line
summaries. See `inspecting results <../../docs/coordination.md#inspecting-results>`_.

``PipelineConfig.registration`` is a :py:class:`~starfinder.registration.RegistrationRecipe` (or ``None``): global steps,
at most one local step, one signal and one final resampling per moving round; see
`Registration <../../docs/registration-contract.md>`_. ``FOV.register`` also registers other rounds to a
reference round or a :py:class:`~starfinder.registration.ExternalReference` through a shared stain.

``FOV.run`` and ``FOV.register`` handle the reference round and the sequencing rounds
only. ``FOV.prepare_morphology(MorphologyConfig(...), checkpoints=...)``, an entry
beside ``FOV.run``, loads and rotates the reference stain (the image
``reference_stain``) and loads, rotates and registers each other round, and can save
each prepared image; ``FOV.load_registered_round`` restores one, and
``FOV.save_registration_record()`` writes the registered rounds and their attempts to
``fovs/<fov_id>/nuclei_registration.json``. ``FOV.load_images``
takes ``rotation_degrees``. See `Dataset and FOV coordination <../../docs/coordination.md>`_, "Sequencing rounds and other
rounds". ``FOV.load_reference_image()`` restores the reference merged image that
``FOV.save_reference_image`` wrote, so that a process with no sequencing round resident
can segment from the saved images; see the `segmentation specification
<https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/segmentation.md#coordination-per-fov>`_, "Coordination per FOV".

Each channel is a ``ChannelInfo`` (file pattern, content name, wavelength in nm).
``Dataset.channel_info(round)`` returns a round's channels (``reference_stain``: the
reference stains), and ``Dataset.channel_index(round, key)`` is the one lookup by
index, pattern or name; see `Dataset and FOV coordination <../../docs/coordination.md>`_, "Channels".

``Dataset.report(sample, fov_ids, …)`` writes a sample's processing report from the saved
records, without running a stage, and ``Dataset.compare_samples(samples)`` writes the
dataset's comparison page (``report/index.html`` and ``report/samples.csv``) from the finished
sample reports only; see `starfinder.report <../../docs/api/report.rst>`_.

``dataset.load_checkpoint(path, stage)`` reads one stage of a per-FOV checkpoint folder (``fovs/<fov_id>/checkpoints``) into the
typed results; see `Checkpoint reference <../../docs/checkpoints-reference.md>`_ for the layout.

