starfinder.assignment
=====================

Molecules to cells, nucleus–cell correspondence and cell-by-gene counts.
The :doc:`assignment guide <../assignment-contract>` shows the workflow; the
:doc:`assignment reference <../assignment-algorithms>` lists options and outputs.

.. currentmodule:: starfinder.assignment

Molecules and input tables
--------------------------

.. autosummary::
   :toctree: generated

   import_legacy_molecules_csv
   load_molecules
   molecule_table
   MoleculeTable

Assignment and correspondence
-----------------------------

.. autosummary::
   :toctree: generated

   assign_molecules
   AssignmentConfig
   AssignmentResult
   CorrespondenceConfig
   labels_at_positions
   match_nuclei

Inspection and output
---------------------

.. autosummary::
   :toctree: generated

   FOVOwnership
   plot_assignment
   summarize_assignment
   write_h5ad

Status vocabularies
-------------------

.. autosummary::
   :toctree: generated

   ASSIGNMENT_STATUSES
   CELL_CORRESPONDENCE
   CELL_STATUSES
   COMPARTMENT_STATES
   NUCLEUS_STATUSES
