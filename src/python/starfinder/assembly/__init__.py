"""Sample assembly: sample coordinates, ownership, retention, accounting and the sample tables.

:func:`assemble` takes a sample's :class:`~starfinder.stitching.PlacementResult` and one
:class:`AssemblyInput` per FOV (its :class:`~starfinder.assignment.AssignmentResult`
or saved assignment folder, or a :class:`~starfinder.assignment.MoleculeTable` alone) and
returns an :class:`AssemblyResult`: every cell and molecule in sample coordinates
(``origin + p ⊙ spacing``, µm, float64), with one assembly status of
``ASSEMBLY_STATUSES`` and a reason of ``CELL_ASSEMBLY_REASONS`` or
``MOLECULE_ASSEMBLY_REASONS``, the flags of ``CELL_ASSEMBLY_FLAGS`` and
``MOLECULE_ASSEMBLY_FLAGS``, the assignment counts of the counted cells and the per-FOV
``fovs`` table. Ownership is computed from the placement by :func:`ownership_map`
(:class:`OwnershipMap`): nearest footprint centre among the owning FOVs, ties by table
order, decided in Y and X; a placeholder owns no point. A cell is retained when its
ownership centroid lies in its own FOV's region, with every molecule assigned to it.
With ``AssemblyConfig.duplicate_threshold`` set (default 0.5), two retained cells of
different FOVs whose territories share at least that share of the smaller one are two
observations of one cell: the larger stays and the other is excluded
``duplicate_observation`` with its molecules. No observation is merged across FOVs.
With ``segmentation`` (the route after fusion), :func:`assemble` assigns the retained
molecules of the ``molecules_only`` assembly to the cells of the sample segmentation, one
row per cell (design/specs/sample-segmentation.md).
:func:`write_sample_h5ad`
writes ``sample.h5ad`` through :func:`starfinder.assignment.write_h5ad`, and
:func:`save_assembly` and :func:`load_assembly` write and read the sample's export folder
(``tables/`` and the ``assembly`` section of ``export.json``). :func:`fov_ownership`
gives one FOV's ownership region and excluded observations as the
:class:`~starfinder.assignment.FOVOwnership` that
:func:`~starfinder.assignment.plot_assignment` draws. :class:`AssemblyConfig`
selects the ownership rule, the centroid, the tie rule, the conflict flags and the
duplicate rule. See
design/specs/assembly.md and design/specs/assembly-algorithms.md.
"""
# §2.10, §2.9
from ._anndata import write_sample_h5ad
from ._assemble import AssemblyResult, assemble
from ._config import (ASSEMBLY_STATUSES, CELL_ASSEMBLY_FLAGS, CELL_ASSEMBLY_REASONS, MOLECULE_ASSEMBLY_FLAGS,
                      MOLECULE_ASSEMBLY_REASONS, AssemblyConfig, AssemblyInput)
from ._ownership import OwnershipMap, ownership_map
from ._persist import load_assembly, save_assembly
from ._region import fov_ownership

__all__ = ["assemble", "AssemblyResult", "AssemblyInput", "AssemblyConfig", "ownership_map", "OwnershipMap",
           "ASSEMBLY_STATUSES", "CELL_ASSEMBLY_REASONS", "MOLECULE_ASSEMBLY_REASONS", "CELL_ASSEMBLY_FLAGS",
           "MOLECULE_ASSEMBLY_FLAGS", "write_sample_h5ad", "save_assembly", "load_assembly", "fov_ownership"]
