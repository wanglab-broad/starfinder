starfinder.assembly
===================

Sample assembly (§2.10) puts the per-FOV §2.9 results of one sample into the sample frame
of its placement and decides which observations represent the sample. It is a
sample-level computation: it runs no per-FOV stage, matches no cell across FOVs and moves
no molecule to another cell. See :doc:`../assembly-contract` and
:doc:`../assembly-algorithms`.

:py:func:`~starfinder.assembly.assemble` takes a
:py:class:`~starfinder.stitching.PlacementResult` (from any backend, or from
:py:func:`~starfinder.stitching.placement_from_table`; one whose review is required must
hold an approval) and one :py:class:`~starfinder.assembly.AssemblyInput` per FOV with
observations: its reference grid and either its
:py:class:`~starfinder.assignment.AssignmentResult` (in memory, or the path of its saved
``assignment/<name>/`` folder) or a :py:class:`~starfinder.assignment.MoleculeTable`
alone for a sample without segmentation. It applies eight checks in order (config,
review, FOV identity, kind, observation identity, grid, genes, one layer of tiles), then
maps every molecule and cell centroid into sample coordinates, ``origin + p ⊙ spacing``
in µm, float64 and never rounded.

Ownership is computed from the placement by :py:func:`~starfinder.assembly.ownership_map`
(an :py:class:`~starfinder.assembly.OwnershipMap` with a point test ``owner``): a point
belongs to the owning FOV whose footprint contains it and whose footprint centre is
nearest in Y and X, an exact tie going to the FOV first in table order. Only placed FOVs
with an input whose status is ``image_supported`` or ``metadata_only`` own points; a
``placeholder`` keeps its footprint and centre in the record and owns nothing. The
optional ``candidates`` restricts the FOVs that may own a point.
:py:class:`~starfinder.assembly.AssemblyConfig` selects ``midpoint_box`` (boxes whose
edges lie in the middle of each grid neighbour's overlap) instead, the centroid ownership
tests (``territory`` or ``original``), the tie rule, and whether the ``conflict`` and
``unclaimed`` flags are computed.

A cell is retained when the owner of its ownership centroid is its own FOV; every
molecule assigned to it follows it, wherever it lies; an unassigned molecule is retained
by its own position; ``outside_grid`` molecules and every observation of a FOV without an
origin are excluded. Every row keeps its §2.9 columns and gets one status of
``ASSEMBLY_STATUSES`` with a reason of ``CELL_ASSEMBLY_REASONS`` or
``MOLECULE_ASSEMBLY_REASONS`` and the flags of ``CELL_ASSEMBLY_FLAGS`` and
``MOLECULE_ASSEMBLY_FLAGS``, which never change a status. The
:py:class:`~starfinder.assembly.AssemblyResult` holds the four sample tables (``fovs``,
``cells``, ``molecules`` and the long ``counts`` of the counted cells: retained and
§2.9 ``kept``), the genes and the record; its construction checks the count identities.

:py:func:`~starfinder.assembly.sample_anndata` writes ``sample.h5ad``, one row per
counted cell with ``obsm["spatial"]`` the ownership centroid's X and Y in µm, through
:py:func:`~starfinder.assignment.write_h5ad`; it needs the ``anndata`` extra.
:py:func:`~starfinder.assembly.save_assembly` writes the sample's export folder
(``tables/fovs``, ``tables/cells``, ``tables/molecules``, ``tables/counts`` as CSV or
Parquet, ``sample.h5ad``, and the ``assembly`` section of ``export.json`` with the
SHA-256 of every file and, per FOV read from a saved folder, the territory file's path
beside its SHA-256), replacing only that section; :py:func:`~starfinder.assembly.load_assembly`
reads it back, checking every SHA-256.

.. currentmodule:: starfinder.assembly

.. autosummary::
   :toctree: generated

   assemble
   ASSEMBLY_STATUSES
   AssemblyConfig
   AssemblyInput
   AssemblyResult
   CELL_ASSEMBLY_FLAGS
   CELL_ASSEMBLY_REASONS
   load_assembly
   MOLECULE_ASSEMBLY_FLAGS
   MOLECULE_ASSEMBLY_REASONS
   ownership_map
   OwnershipMap
   sample_anndata
   save_assembly
