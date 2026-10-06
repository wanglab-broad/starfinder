starfinder.assignment
=====================

Assignment (§2.9) places the molecules of one FOV in cells; it is the third of the three
calls per FOV (``FOV.run``, ``FOV.segment``, ``FOV.assign``). See
:doc:`../assignment-contract` and :doc:`../assignment-algorithms`.

A :py:class:`~starfinder.assignment.MoleculeTable` holds the molecules: ``spot_namespace``,
``spot_id``, ``z``, ``y``, ``x`` (float64, zero-based voxel coordinates of the molecule
run's reference grid) and ``gene_id``, with the ordered gene list of the count matrix, the
population, the source record and a SHA-256 that does not depend on row order. It refuses
repeated or null keys, rows of another FOV, non-finite coordinates and genes outside the
gene list (``ValueError`` naming them). :py:func:`~starfinder.assignment.molecule_table`
joins a :py:class:`~starfinder.spot_finding.SpotFindingResult` with a read result by
``(spot_namespace, spot_id)``, as :py:func:`~starfinder.io.export_spots` does (``final``:
the accepted reads; ``called``: the reads called ``assigned``);
:py:func:`~starfinder.assignment.molecule_table_from_csv` reads a legacy one-based
``x, y, z, gene`` CSV with integer or float coordinates.

:py:func:`~starfinder.assignment.assign_molecules` takes the molecules, a cell
:py:class:`~starfinder.segmentation.SegmentationResult` (a ``cell`` run, or a ``nucleus``
run used as territories), the molecule run's
:py:class:`~starfinder.segmentation.ReferenceGrid`, optional nuclei and an optional
supplied correspondence. Before any sampling it checks the config, the targets, the grid
(a ``plane`` run against the projection of the grid;
:py:class:`~starfinder.image.IncompatibleGeometryError` otherwise), the nuclei on the cell
grid, the FOV identity of the label namespaces, that the cell run was not expanded by
segmentation, and the expansion unit against the calibration. It never reads or writes
a file.

Sampling reads each label image at ``floor(c + 0.5)`` per axis
(:py:func:`~starfinder.assignment.sample_labels`); a ``plane`` is read at ``y, x``. Every
molecule gets one status of ``ASSIGNMENT_STATUSES``: ``assigned``, ``unassigned``
(outside every territory), ``excluded_cell`` (in an excluded cell) or ``outside_grid``;
nothing is dropped. :py:class:`~starfinder.assignment.AssignmentConfig` sets the one
optional expansion, applied here with :py:func:`~starfinder.segmentation.expand_labels`
(``um`` on a calibrated grid, ``pixel`` on an uncalibrated one, or with
``legacy_pixel_expansion``); the original and the expanded territories are both kept, and
each molecule records ``in_expansion`` and ``original_cell_id``.

With nuclei, :py:func:`~starfinder.assignment.match_nuclei` relates each nucleus to the
cell holding more than ``match_fraction`` of its voxels
(:py:class:`~starfinder.assignment.CorrespondenceConfig`; by default ``match_fraction`` 0.5
and ``outside_tolerance`` 0.1, a provisional value) and flags ``several_nuclei``, ``ambiguous_nucleus``, ``nucleus_outside_cell`` and
``foreign_nucleus``; nothing is repaired. A cell's correspondence is one of
``CELL_CORRESPONDENCE`` and its compartments one of ``COMPARTMENT_STATES``: nuclear and
cytoplasmic counts exist only for ``available`` cells and sum to the whole-cell counts;
withheld cells keep their whole-cell counts. Cells whose correspondence is ``no_nucleus``
are excluded by default (``exclude_cells_without_nucleus``; ``CELL_STATUSES``), keep their
row, masks and molecules, and have no count rows; ``ambiguous`` cells are kept. Nuclei have
a status of ``NUCLEUS_STATUSES``.

An :py:class:`~starfinder.assignment.AssignmentResult` holds the molecule, cell, count and
nucleus tables, the original territories, the expanded ones and the nucleus labels, and
the run record (input hashes, grid check, config, sampling rule, expansion, calibration and
totals). Construction checks the count identities of the contract.
``AssignmentResult.matrix(compartment, cells=…)`` gives a dense ``int64`` cells × genes
matrix; for ``nucleus`` and ``cytoplasm`` the cells whose compartments are not available
are absent, not zero. :py:func:`~starfinder.assignment.summarize_assignment` counts
statuses, flags and states, with quantiles and the totals before and after the exclusion
(the same five keys in both; the nuclear and cytoplasmic totals are equal, because an
excluded cell has no compartment counts).
:py:func:`~starfinder.assignment.plot_assignment` draws one row of four panels: the stain
in grey scale with the territory outlines in green and a red dot at each cell's centroid;
the same with the molecules, ``assigned`` blue, ``unassigned`` red and ``excluded_cell``
orange (``outside_grid`` not drawn); and the histograms of voxels and of molecules per
cell. ``view="z_max"`` draws the Z maximum, ``view="single_layer"`` one plane ``z``
(default ``Z // 2``) of a volumetric result.

``FOV.assign(config, cells="cell", nuclei=None, name="default", population="final")``
builds the molecule table from the FOV's spot and read results and the codebook's genes,
calls the entry on ``FOV.reference_grid()`` (also the grid of the saved reference image
in a process that segmented from saved images) and stores the result in
``FOV.assignment_results[name]``. With ``checkpoints=CheckpointConfig(…)`` it also
writes ``<checkpoint dir>/<fov_id>/assignment/<name>/``: the four tables, ``assignment.json``
and every label image it used that is not saved under its run, linking the saved ones;
``FOV.load_assignment(name)`` reads the folder and the linked files back, checking every
recorded SHA-256 (see :doc:`../checkpoints`).

.. currentmodule:: starfinder.assignment

.. autosummary::
   :toctree: generated

   assign_molecules
   ASSIGNMENT_STATUSES
   AssignmentConfig
   AssignmentResult
   CELL_CORRESPONDENCE
   CELL_STATUSES
   COMPARTMENT_STATES
   CorrespondenceConfig
   match_nuclei
   molecule_table
   molecule_table_from_csv
   MoleculeTable
   NUCLEUS_STATUSES
   plot_assignment
   sample_labels
   summarize_assignment
