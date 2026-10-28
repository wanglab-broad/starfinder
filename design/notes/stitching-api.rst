starfinder.stitching
====================

Stitching places the FOVs of one sample in one physical frame, ``sample:<sample>``
in µm. It is a sample-level computation outside ``PipelineConfig`` and ``FOV.run``: it
reads only saved per-FOV files and runs no per-FOV stage. See
`Stitching <../../docs/stitching-contract.md>`_ and `Stitching reference <../../docs/stitching-algorithms.md>`_.

.. Stitching (§2.10)

A :py:class:`~starfinder.stitching.TilePositionTable` holds one row per acquired FOV in
acquisition order: ``fov_id``, ``position_id``, ``nominal_z``, ``nominal_y``,
``nominal_x`` (float64, µm), ``source`` (``stage``, ``grid`` or ``supplied``) and the grid
cell, with a record of the converter and its parameters. Only the three converters build
one, and none reads an image: :py:func:`~starfinder.stitching.positions_from_leica` reads a
LAS X stage file through an explicit :py:class:`~starfinder.stitching.StageAxisMapping`
(no default) and, with a :py:class:`~starfinder.stitching.GridSnap`, snaps the positions
to a regular grid; :py:func:`~starfinder.stitching.positions_from_grid` places FOV ``k``
by ``order`` (``column_major_down_right``, ``row_major_right_down`` and their
``_serpentine`` forms) with the step ``(1 − overlap) × tile × spacing``;
:py:func:`~starfinder.stitching.positions_from_table` reads a supplied CSV or frame in
µm, nm or mm. :py:func:`~starfinder.stitching.neighbor_pairs` returns every pair whose
nominal boxes overlap (diagonals included) as
:py:class:`~starfinder.stitching.NeighborPair` with its overlap box, by one box rule
for every converter.

Every placement is one :py:class:`~starfinder.stitching.PlacementResult`: ``fovs`` (per
FOV its status ``image_supported``, ``metadata_only``, ``placeholder``, ``missing`` or
``failed``, its origin, nominal position, correction, the method's own translation, its
component, kept links and flags), ``links`` (per neighbour pair the overlap box, shift,
quality, kept flag, reason and residual) and ``record`` (frame, gauge, method provenance,
table, inputs and review). The origin is the sample-frame position of index ``(0, 0, 0)``
of the FOV's reference grid, unrounded; :py:func:`~starfinder.stitching.placed_metadata`
gives the FOV's grid metadata in the sample frame, so that ``index_to_world(p)`` is
``origin + p ⊙ spacing``.

:py:func:`~starfinder.stitching.stitch` checks its
:py:class:`~starfinder.stitching.StitchingInput` list against the table and the grids'
geometry, derives the neighbour pairs, calls the method registered for the config in
``STITCHING_METHODS`` (a :py:class:`~starfinder.stitching.StitchingSpec` returning a
:py:class:`~starfinder.stitching.MethodPlacement`) once, assigns the statuses, applies
gauge G1 (zero mean correction over the ``image_supported`` FOVs) and the review
policy. A missing optional dependency raises
:py:class:`~starfinder.stitching.StitchingBackendUnavailableError`, and importing the
module imports no backend. A FOV that no kept link joins to the reference component is
``failed``, or with ``fallback="nominal"`` ``metadata_only`` at its nominal position and
flagged ``fallback``.

The first registered method is ``multiview_stitcher``
(:py:class:`~starfinder.stitching.MultiviewStitcherConfig`, the ``multiview-stitcher``
extra, imported only when it runs), a second backend beside BigStitcher, which stays the
default. Its defaults are the contract's starting configuration: each tile becomes a view
at its nominal origin (a Z=1 tile as a 2D view); the library's overlap graph, restricted
to the neighbour pairs, is pruned to the alternating pattern; phase correlation registers
each pair at binning 4; link rule L1 keeps a link whose quality is at least 0.7 and drops
NaN-quality links, which the library keeps; and the library's global optimization
resolves translations in µm, view to common frame. Pairs it does not register are
recorded ``pruned``, links the optimizer removes ``optimizer_removed``. With no link left
the library raises ``NotEnoughOverlapError``, and every FOV is ``failed``, the result
still returned.

:py:func:`~starfinder.stitching.placement_from_table` turns a known placement (for
example a synthetic scene's truth) into the same record, with the nominal positions from
the table and, optionally, the origins given separately: every FOV ``metadata_only`` with
the flag ``supplied``, no link, gauge ``none``.

BigStitcher, the initial default, runs in Fiji as an engine task, and the package owns its
two ends. :py:func:`~starfinder.stitching.export_bigstitcher` writes ``tile_{j}.tif`` in
the column-major order of BigStitcher's ``Down & Right`` grid (a blank for each empty
cell, a padding view, and with ``pad_missing=True`` for each acquired FOV without an
image, a placeholder view) and returns the macro text, whose parameters are those of
:py:class:`~starfinder.stitching.BigStitcherConfig` (today's macro by default) with the
reference grids' spacing as voxel size, and the view map (``view``, ``grid_column``,
``grid_row``, ``fov_id``, ``kind``). :py:func:`~starfinder.stitching.import_bigstitcher`
reads ``dataset.xml`` into the same record: each view maps to its FOV through the
``ViewSetup`` tile attribute and the view map, every transform but ``calibration`` is
composed (translation only), world units become µm through the calibration and each
FOV's reference spacing, each ``PairwiseResult`` becomes a kept link, and the statuses,
gauge, fallback and review are those of :py:func:`~starfinder.stitching.stitch`. Padding
views have no row; their raw translations stay in the method record. A view whose affine
holds a non-finite number refuses the file with ``ValueError`` (naming the file, the setup
and the transform) under either fallback.
:py:func:`~starfinder.stitching.legacy_tile_config_view` applies today's integer cast,
shift and rectangles to the raw translations of every view, which reproduces
``tile_config_{sample}.csv`` whatever the fallback.

:py:class:`~starfinder.stitching.ReviewPolicy` is criterion V1:
:py:func:`~starfinder.stitching.review_placement` requires review when a FOV is not
``image_supported``, a correction exceeds ``max_correction_fraction`` (0.5) of the FOV's
overlap width, a kept link's residual exceeds ``max_residual_um`` (1.0 µm) or the method
removed a link, and always, with the reason ``supplied_placement``, for a supplied
placement. :py:func:`~starfinder.stitching.approve_placement` records the reviewer, UTC
time, note and the SHA-256 of both tables, and changes no origin; sample assembly raises
:py:class:`~starfinder.stitching.PlacementReviewRequiredError` for a placement that needs
review and holds no approval of its current tables.

:py:func:`~starfinder.stitching.save_placement` writes the folder
``stitching/<sample>/`` (``placement.json``, ``fovs.csv`` and ``links.csv``, float64 bit
for bit; for a BigStitcher import also ``view_map.csv``, with the macro's and
``dataset.xml``'s SHA-256 in the record) as a whole or not at all, and
:py:func:`~starfinder.stitching.load_placement` reads it back, checking the tables'
SHA-256. ``Dataset.stitch(sample, fov_ids,
table=…, config=…)`` reads each FOV's reference grid from its reference image (``fovs/{fovID}/reference.tif`` by default)
and its stitching image (``reference_merged``, the saved ``reference_stain`` or a
``file:`` pattern) one FOV at a time, builds the table from the workflow's ``positions``
mapping when given one, calls :py:func:`~starfinder.stitching.stitch` and, with
``output_dir``, saves the result.

