starfinder.stitching
====================

FOV positions and placement of a sample. The
:doc:`stitching guide <../stitching-contract>` shows the workflow; the
:doc:`stitching reference <../stitching-algorithms>` lists methods, options and files.

.. currentmodule:: starfinder.stitching

Positions and neighbours
------------------------

.. autosummary::
   :toctree: generated

   GridSnap
   neighbor_pairs
   NeighborPair
   positions_from_grid
   positions_from_leica
   positions_from_table
   StageAxisMapping
   TilePositionTable

Stitching and supplied placements
---------------------------------

.. autosummary::
   :toctree: generated

   MethodPlacement
   MultiviewStitcherConfig
   placed_metadata
   placement_from_table
   PlacementResult
   stitch
   STITCHING_METHODS
   StitchingBackendUnavailableError
   StitchingInput
   StitchingSpec

BigStitcher boundary
--------------------

.. autosummary::
   :toctree: generated

   BigStitcherConfig
   export_bigstitcher
   import_bigstitcher
   legacy_tile_config_view

Review and persistence
----------------------

.. autosummary::
   :toctree: generated

   approve_placement
   load_placement
   PlacementReviewRequiredError
   review_placement
   ReviewPolicy
   save_placement
