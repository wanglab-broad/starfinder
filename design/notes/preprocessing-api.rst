starfinder.preprocessing
========================

Intensity normalization and background removal, and the step/recipe contract that
orders them. See `Array and coordinate contracts <../../docs/api/contracts.md>`_ for shared array conventions and
`Preprocessing <../../docs/preprocessing-contract.md>`_ for steps, recipes and the enforcement wrapper.
``FOV.run`` executes a :py:class:`~starfinder.preprocessing.PreprocessingRecipe` given as
``PipelineConfig.preprocessing``; each step is looked up in
:py:data:`~starfinder.preprocessing.PREPROCESSING_METHODS` by
its exact config type and run through :py:func:`~starfinder.preprocessing.run_step`.

Snapshots and sources
---------------------

A :py:class:`~starfinder.preprocessing.PreprocessingStep` with ``save_as`` keeps a named snapshot of
its output. The recipe's ``extraction_source`` and ``registration_source`` name the
snapshots that extraction and the registration signals read; both default to the
detection image, the last step's output. How the snapshots are registered with their
round and recorded, and the workflow key ``preprocessing``, are in the
`preprocessing guide <../../docs/preprocessing-contract.md>`_ ("What it does", "What you get",
"Workflow keys") and `Workflow configuration reference <../../docs/workflow-configuration.md>`_.

Background subtraction
----------------------

:py:class:`~starfinder.preprocessing.ScalarBackgroundConfig` (step ``scalar_background``)
subtracts one level per channel and round, the inverted-CDF ``percentile`` of the
channel's voxels, and clips at zero.
:py:class:`~starfinder.preprocessing.Background3DConfig` (step ``background_3d``) subtracts a
grey opening with an anisotropic ellipsoidal footprint, a volumetric white top-hat.
Both keep the input dtype and record the shared per-channel diagnostics (zero fraction,
median, MAD, noise threshold and ``mad_zero``). The XY methods
(:py:class:`~starfinder.preprocessing.ReconstructionConfig`,
:py:class:`~starfinder.preprocessing.TophatConfig`) are unchanged. Choosing the 3D radius
relative to puncta size is described in `Python recipes <../../docs/recipes.md>`_.

Sample-level statistics
-----------------------

Steps with ``fit="supplied"`` (percentile normalization, scalar background and the
histogram-matching reference) read their values from the recipe's ``supplied_statistics`` file, schema
``starfinder.preprocessing.supplied/1``, as described in the
`preprocessing reference <../../docs/preprocessing-algorithms.md>`_. A summary pass runs the steps before the fitted
step (:py:func:`~starfinder.preprocessing.summary_recipe`), records the integer
histograms of each FOV at that step's input
(:py:func:`~starfinder.preprocessing.summarize_histograms`), sums them
(:py:func:`~starfinder.preprocessing.merge_histograms`) and fits the step's section
(:py:func:`~starfinder.preprocessing.fit_supplied_section`). The application pass runs the
full recipe; ``FOV.run`` validates the file against the recipe, the dataset
``channel_order`` and the rounds before any step runs. The manual two-pass example
below summarizes three FOVs after a white top-hat, merges them, writes the file and
applies it; from ``src/python``, run
``uv run python ../../docs/examples/percentile_two_pass.py <new output directory>``.
When only scalar background steps lie between two supplied steps,
:py:func:`~starfinder.preprocessing.scalar_background_histograms` derives the later
summary exactly from the earlier one, per FOV before merging (``fit="fov"``) or from the
merged counts (``fit="supplied"``), without another pass over the images.

.. literalinclude:: ../../docs/examples/percentile_two_pass.py
   :language: python

API
---

