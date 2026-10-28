"""Stitching: tile positions, neighbour pairs, the placement record, the stitch entry and the review policy.

A sample's FOVs are placed into one physical frame, ``sample:<sample>`` in µm. Their
nominal positions are a :class:`TilePositionTable`, built by one of three converters
(:func:`positions_from_leica` with an explicit :class:`StageAxisMapping` and an optional
:class:`GridSnap`, :func:`positions_from_grid`, :func:`positions_from_table`);
:func:`neighbor_pairs` derives the pairs whose nominal boxes overlap. Every placement,
whichever backend made it, is one :class:`PlacementResult`: one row per FOV with its
status, origin (the sample-frame position of index ``(0, 0, 0)`` of its reference grid,
gauge G1), correction and flags, one row per neighbour pair with its shift, quality and
residual, and a record with the frame, gauge, provenance, inputs and review.
:func:`stitch` runs one method of ``STITCHING_METHODS`` on :class:`StitchingInput` images;
the first is ``multiview_stitcher`` (:class:`MultiviewStitcherConfig`, the
``multiview-stitcher`` extra, imported only when it runs). :func:`placement_from_table`
turns a known placement into the same record; :func:`placed_metadata` gives a FOV's sample-frame ``ImageMetadata``.
BigStitcher runs in Fiji as an engine task: :func:`export_bigstitcher` writes its tiles and
returns the macro (:class:`BigStitcherConfig`) and the view map, :func:`import_bigstitcher`
reads its ``dataset.xml`` into the same record, and :func:`legacy_tile_config_view` gives
today's ``tile_config_{sample}.csv`` from the import.
:func:`review_placement` applies :class:`ReviewPolicy` (criterion V1),
:func:`approve_placement` records an approval, and :func:`save_placement` and
:func:`load_placement` write and read ``stitching/<sample>/`` (option S1).
``Dataset.stitch`` reads the saved per-FOV images and calls :func:`stitch`. Importing this
module imports no stitching backend. See design/specs/stitching.md and
design/specs/stitching-algorithms.md.
"""
# §2.10
from ._bigstitcher import BigStitcherConfig, export_bigstitcher, import_bigstitcher, legacy_tile_config_view
from ._errors import PlacementReviewRequiredError, StitchingBackendUnavailableError
from ._methods import STITCHING_METHODS as _STITCHING_METHODS
from ._methods import MethodPlacement, StitchingSpec
from ._multiview import MultiviewStitcherConfig
from ._neighbors import NeighborPair, neighbor_pairs
from ._persist import load_placement, save_placement
from ._placement import PlacementResult, placed_metadata, placement_from_table
from ._positions import (GridSnap, StageAxisMapping, TilePositionTable, positions_from_grid, positions_from_leica,
                         positions_from_table)
from ._review import ReviewPolicy, approve_placement, review_placement
from ._stitch import StitchingInput, stitch

#: The stitching method registry, mapping each exact frozen config type to its StitchingSpec.
#: stitch derives its method set from it; ``multiview_stitcher`` is registered.
STITCHING_METHODS = _STITCHING_METHODS

__all__ = ["TilePositionTable", "GridSnap", "StageAxisMapping", "positions_from_leica", "positions_from_grid",
           "positions_from_table", "NeighborPair", "neighbor_pairs", "StitchingInput", "PlacementResult",
           "MethodPlacement", "StitchingSpec", "STITCHING_METHODS", "MultiviewStitcherConfig", "stitch",
           "placement_from_table", "placed_metadata", "ReviewPolicy", "review_placement", "approve_placement",
           "save_placement", "load_placement", "BigStitcherConfig", "export_bigstitcher", "import_bigstitcher",
           "legacy_tile_config_view", "StitchingBackendUnavailableError", "PlacementReviewRequiredError"]
