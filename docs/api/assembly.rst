starfinder.assembly
===================

Sample coordinates, overlap ownership and sample tables. The
:doc:`assembly guide <../assembly-contract>` shows how to assemble a sample; the
:doc:`assembly reference <../assembly-algorithms>` lists options and outputs.

.. currentmodule:: starfinder.assembly

Assembly
--------

.. autosummary::
   :toctree: generated

   assemble
   AssemblyConfig
   AssemblyInput
   AssemblyResult

Ownership
---------

.. autosummary::
   :toctree: generated

   fov_ownership
   ownership_map
   OwnershipMap

Persistence
-----------

.. autosummary::
   :toctree: generated

   load_assembly
   save_assembly
   write_sample_h5ad

Status vocabularies
-------------------

.. autosummary::
   :toctree: generated

   ASSEMBLY_STATUSES
   CELL_ASSEMBLY_FLAGS
   CELL_ASSEMBLY_REASONS
   MOLECULE_ASSEMBLY_FLAGS
   MOLECULE_ASSEMBLY_REASONS
