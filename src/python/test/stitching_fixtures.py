"""Hand-built stitching fixtures of docs/stitching-assembly-validation.md ("Fixtures").

``stub_grid``: 2 rows × 3 columns of FOV grids of 4×16×24 voxels at (0.25, 0.25, 0.25) µm
(dyadic, so the review boundaries are exact), identity direction, overlap 0.25, zero
images, positions from ``positions_from_grid`` (``column_major_down_right``), and a
test-only method (``StubConfig``) that returns supplied translations (µm), one link per
neighbour pair with the shift those translations imply, and the overrides a case needs.
``stub_variants`` are built from these pieces in the tests. ``hand_records``' stage files
are written as text from the constant values of W-324 ``scripts/parity_fixtures.py``,
stated here as literals: nothing is copied or read from a run directory.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from starfinder.image import ImageMetadata
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import (MethodPlacement, StageAxisMapping, StitchingInput, StitchingSpec,
                                  positions_from_grid, positions_from_leica)

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
STUB_IDS = tuple(f"FOV{k}" for k in range(6))
STUB_SHAPE = (4, 16, 24)
STUB_SPACING = (0.25, 0.25, 0.25)
LN_SPACING = (0.346255, 0.194398, 0.194398)
LINK_COLUMNS = ("fov_a", "fov_b", "shift_z", "shift_y", "shift_x", "quality", "kept", "reason")

# W-324 parity_fixtures.py, stage files: (PositionID, StageXPos, StageYPos) in metres, as written.
MAF_3X2_GAP = ((1, "0.076", "0.048"), (2, "0.076", "0.048261"), (3, "0.076261", "0.048"),
               (4, "0.076522", "0.048"), (5, "0.076522", "0.048261"))
MAF_2X3_JITTER = ((1, "0.0500004", "0.02"), (2, "0.0499989", "0.020261"), (3, "0.05", "0.020522"),
                  (4, "0.0502619", "0.02"), (5, "0.0502604", "0.020261"), (6, "0.0502612", "0.020522"))
# Grids for the stage-file fixtures: 288 µm tiles in Y and X over the 261 µm stage steps.
STAGE_SPACING = (1.0, 18.0, 12.0)


def maf_text(points):
    """A LAS X stage file with the element and attribute names the converters read (parity_fixtures.maf)."""
    lines = ['<?xml version="1.0" encoding="utf-8"?>', '<XYZStagePointDefinitionList StageOrderNumber="0">']
    for pid, x, y in points:
        lines.append(f' <XYZStagePointDefinition PositionID="{pid}" PositionIdentifier="Position{pid}" '
                     f'Sections="12" StageXPos="{x}" StageYPos="{y}" />')
    lines += ["</XYZStagePointDefinitionList>", ""]
    return "\n".join(lines)


def position_ids(points):
    return {pid: f"Position{pid:03d}" for pid, *_ in points}


def metadata(spacing=STUB_SPACING, *, unit="µm", direction=IDENTITY, frame="fov"):
    return ImageMetadata(frame, spacing, (0.0, 0.0, 0.0), direction, unit)


def grid(shape=STUB_SHAPE, spacing=STUB_SPACING, **kw):
    return ReferenceGrid(shape, metadata(spacing, **kw), "declared")


def stub_table(ids=STUB_IDS, *, tile=STUB_SHAPE[1:], spacing_yx=STUB_SPACING[1:], columns=3, rows=2):
    return positions_from_grid(columns=columns, rows=rows, order="column_major_down_right", tile_shape_yx=tile,
                               spacing_yx=spacing_yx, overlap=0.25, fov_ids=list(ids))


def grids_for(table, shape=STUB_SHAPE, spacing=STUB_SPACING, **changes):
    """One grid per FOV of the table; changes maps a fov_id to its own (shape, spacing)."""
    out = {}
    for fov in table.fov_ids:
        own_shape, own_spacing = changes.get(fov, (shape, spacing))
        out[fov] = grid(own_shape, own_spacing)
    return out


def inputs_for(table, grids, *, absent=(), yx=False):
    items = []
    for fov in table.fov_ids:
        shape = grids[fov].shape_zyx
        image = None if fov in absent else np.zeros(shape[1:] if yx else shape, np.uint8)
        items.append(StitchingInput(fov, image, grids[fov], "reference_merged"))
    return items


def stage_table(tmp_path, points, name, **kw):
    path = tmp_path / f"{name}.maf"
    path.write_text(maf_text(points))
    return positions_from_leica(path, stage_axes=StageAxisMapping(x="+x", y="+y"), fov_ids=position_ids(points),
                                **kw)


@dataclass(frozen=True)
class StubConfig:
    """The test-only method: supplied translations (µm, table order) and link overrides.

    Every neighbour pair is a kept link with quality 0.9 and the shift the
    translations imply (``t_b − t_a``), unless ``overrides`` holds a row
    ``(fov_a, fov_b, quality, kept, reason, extra_shift_zyx)``; ``kept`` None
    leaves the pair unregistered (``pruned``). A FOV without an image gets NaN
    and no link.
    """

    translations: tuple
    overrides: tuple = ()
    method: str = field(default="stub", init=False)


def stub_run(tiles, grids, nominal_zyx, pairs, *, config):
    translations = np.array(config.translations, dtype=np.float64).reshape(len(tiles), 3)
    absent = {i for i, tile in enumerate(tiles) if tile is None}
    translations[sorted(absent)] = np.nan
    overrides = {(o[0], o[1]): o[2:] for o in config.overrides}
    rows = []
    for pair in pairs:
        quality, kept, reason, extra = overrides.get((pair.fov_a, pair.fov_b), (0.9, True, "", (0.0, 0.0, 0.0)))
        if kept is None or pair.index_a in absent or pair.index_b in absent:
            continue
        shift = translations[pair.index_b] - translations[pair.index_a] + np.asarray(extra, dtype=np.float64)
        rows.append((pair.fov_a, pair.fov_b, *shift, quality, kept, reason))
    links = pd.DataFrame(rows, columns=list(LINK_COLUMNS))
    return MethodPlacement(translations, np.zeros(len(tiles), dtype=np.int64), links,
                           {"pairs": len(pairs), "array": np.zeros(2)})


STUB_SPEC = StitchingSpec("stub", stub_run)


def zero(n):
    return tuple((0.0, 0.0, 0.0) for _ in range(n))


# Fractional translations, multiples of 2**-6 µm, so the expected sums are exact.
FRACTIONAL = ((1 / 64, -3 / 64, 5 / 64), (-2 / 64, 7 / 64, 0.0), (0.0, 1 / 64, -9 / 64),
              (3 / 64, -1 / 64, 2 / 64), (-1 / 64, 0.0, 11 / 64), (2 / 64, 5 / 64, -6 / 64))


def stub_variant_results():
    """The stub-path placements of ``stub_variants`` used by PL16: name -> PlacementResult (stub registered)."""
    from starfinder.stitching import positions_from_table, stitch

    def run(table, grids, translations, overrides=(), absent=(), fallback=None):
        return stitch(inputs_for(table, grids, absent=absent), table, sample="s1", fallback=fallback,
                      config=StubConfig(tuple(map(tuple, translations)), tuple(overrides)))

    table = stub_table()
    grids = grids_for(table)
    moved = table.positions.drop(columns=["grid_column", "grid_row"])
    moved.loc[5, "nominal_x"] += 12.0
    moved = positions_from_table(moved)
    nan = tuple((a, b, np.nan, False, "nan_quality", (0.0, 0.0, 0.0)) for a, b in
                (("FOV0", "FOV3"), ("FOV1", "FOV3"), ("FOV2", "FOV3"), ("FOV3", "FOV4"), ("FOV3", "FOV5")))
    small = stub_table(STUB_IDS[:4], columns=2, rows=2)
    across = tuple((a, b, 0.5, False, "quality_below_threshold", (0.0, 0.0, 0.0)) for a, b in
                   (("FOV0", "FOV2"), ("FOV0", "FOV3"), ("FOV1", "FOV2"), ("FOV1", "FOV3")))
    lmax = np.zeros((6, 3))
    lmax[0, 2], lmax[5, 2] = 0.75 + 2.0 ** -20, -(0.75 + 2.0 ** -20)
    return {
        "fractional": run(table, grids, FRACTIONAL),
        "ln_calibration": run(stub_table(spacing_yx=LN_SPACING[1:]), grids_for(table, spacing=LN_SPACING),
                              FRACTIONAL),
        "nan_quality": run(table, grids, FRACTIONAL, overrides=nan, fallback="nominal"),
        "moved": run(moved, grids, FRACTIONAL),
        "two_components": run(small, grids_for(small), zero(4), overrides=across),
        "correction_bound": run(table, grids, lmax),
        "residual": run(table, grids, zero(6),
                        overrides=(("FOV0", "FOV1", 0.9, True, "", (0.0, 0.0, 1.0 + 2.0 ** -20)),)),
        "missing": run(table, grids, FRACTIONAL, absent=("FOV4",)),
    }


def determinism_hashes(directory):
    """Save each stub-path placement under directory; return name -> (SHA-256 of fovs.csv, of links.csv)."""
    import hashlib
    from pathlib import Path

    from starfinder.stitching import STITCHING_METHODS, save_placement
    STITCHING_METHODS[StubConfig] = STUB_SPEC
    try:
        results = stub_variant_results()
    finally:
        del STITCHING_METHODS[StubConfig]
    hashes = {}
    for name, result in results.items():
        folder = save_placement(result, Path(directory) / name)
        hashes[name] = [hashlib.sha256((folder / f).read_bytes()).hexdigest() for f in ("fovs.csv", "links.csv")]
    return hashes
