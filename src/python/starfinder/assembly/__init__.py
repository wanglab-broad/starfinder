"""Sample assembly (§2.10): sample coordinates, ownership, retention, accounting and the sample tables.

:func:`assemble` takes a sample's :class:`~starfinder.stitching.PlacementResult` and one
:class:`AssemblyInput` per FOV (its §2.9 :class:`~starfinder.assignment.AssignmentResult`
or saved assignment folder, or a :class:`~starfinder.assignment.MoleculeTable` alone) and
returns an :class:`AssemblyResult`: every cell and molecule in sample coordinates
(``origin + p ⊙ spacing``, µm, float64), with one assembly status of
``ASSEMBLY_STATUSES`` and a reason of ``CELL_ASSEMBLY_REASONS`` or
``MOLECULE_ASSEMBLY_REASONS``, the flags of ``CELL_ASSEMBLY_FLAGS`` and
``MOLECULE_ASSEMBLY_FLAGS``, the §2.9 counts of the counted cells and the per-FOV
``fovs`` table. Ownership is computed from the placement by :func:`ownership_map`
(:class:`OwnershipMap`): nearest footprint centre among the owning FOVs, ties by table
order, decided in Y and X; a placeholder owns no point. A cell is retained when its
ownership centroid lies in its own FOV's region, with every molecule assigned to it.
There is no matching or merging of observations across FOVs. :func:`sample_anndata`
writes ``sample.h5ad`` through :func:`starfinder.assignment.write_h5ad`, and
:func:`save_assembly` and :func:`load_assembly` write and read the sample's export folder
(``tables/`` and the ``assembly`` section of ``export.json``). :func:`fov_ownership`
gives one FOV's ownership region and excluded observations as the
:class:`~starfinder.assignment.FOVOwnership` that
:func:`~starfinder.assignment.plot_assignment` draws. :class:`AssemblyConfig`
selects the ownership rule, the centroid, the tie rule and the conflict flags. See
docs/assembly-contract.md and docs/assembly-algorithms.md.
"""
from ._anndata import sample_anndata
from ._assemble import AssemblyResult, assemble
from ._config import (ASSEMBLY_STATUSES, CELL_ASSEMBLY_FLAGS, CELL_ASSEMBLY_REASONS, MOLECULE_ASSEMBLY_FLAGS,
                      MOLECULE_ASSEMBLY_REASONS, AssemblyConfig, AssemblyInput)
from ._ownership import OwnershipMap, ownership_map
from ._persist import load_assembly, save_assembly
from ._region import fov_ownership

__all__ = ["assemble", "AssemblyResult", "AssemblyInput", "AssemblyConfig", "ownership_map", "OwnershipMap",
           "ASSEMBLY_STATUSES", "CELL_ASSEMBLY_REASONS", "MOLECULE_ASSEMBLY_REASONS", "CELL_ASSEMBLY_FLAGS",
           "MOLECULE_ASSEMBLY_FLAGS", "sample_anndata", "save_assembly", "load_assembly", "fov_ownership"]
