"""Fixtures of the real-appearance scene's generator checks (docs/synthetic-real-appearance-validation.md, "Fixtures").

The bank's part (W-516): ``bank_cases`` and ``fall_off_cases`` are hand-made label and stain arrays with a known outcome;
``cf_banks`` are the closed-form banks of the three routine context presets (CF1, ``n`` = 4: the seeds 42 to 45, the
seed 45 image held out); ``cf_bank_files`` are the same scenes written by ``save_context_scene`` and configured as file
sources of ``build_appearance_bank``. Nothing is read from a run directory or a real image: every array is built in
the session and every file is written under the test's temporary directory.
"""
from dataclasses import dataclass
from functools import cache
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import tifffile

from starfinder.synthetic import closed_form_bank, save_context_scene

ROUTINE = ("context-dense3d-v1", "context-tissue2d-v1", "context-culture-v1")
SEEDS = (42, 43, 44, 45)
HELD_OUT = 45


@dataclass
class Case:
    """A hand-made source image: stains by channel, labels, nucleus labels (culture), spacing, the build settings and
    the known outcome of each label."""

    name: str
    stains: dict
    labels: np.ndarray
    nucleus_labels: np.ndarray | None
    spacing: tuple
    settings: dict
    expected: dict
    plateau: float
    background: float


def _box(array, lo, hi, value):
    array[tuple(slice(a, b) for a, b in zip(lo, hi))] = value


@cache
def bank_cases():
    """``bank_cases``: a dense image of nucleus labels and a culture image of cells with nucleus labels.

    Nuclei (spacing 0.4, 0.2, 0.2 µm; background 10, every label at 100 except the two small and large ones):
    1 touches the Y border; 2 touches a Z end; 3 is the smallest and 4 the largest of the labels that pass rule 1
    (below the 5th and above the 95th percentile of 8, 48, 75, 144, 500 voxels); 5 (3 × 2 × 8 voxels, every voxel on
    its rim) shares its 3 × 8 face with 6, a rim share of 24 / 48 = 0.5; 6 (3 × 6 × 8) has 24 of its 120 rim voxels on
    that face, 0.2; 7 passes every rule. The plateau is the median of the label medians, 100.

    Cells (spacing 0.4 µm; DAPI background 5, Flamingo 3 on the background and 50 in the cells): 1 touches the X border;
    2 is the smallest and 3 the largest cell passing rule 1; 4 holds a nucleus label of 20 voxels only; 5 holds two
    nucleus labels, of 80 voxels (DAPI 150) and 60 voxels (DAPI 60); 6 passes every rule. Each other nucleus has DAPI
    100, so the plateau (on the nucleus labels) is 100.
    """
    shape = (14, 40, 120)
    labels = np.zeros(shape, dtype=np.uint16)
    levels = {1: 100.0, 2: 100.0, 3: 90.0, 4: 110.0, 5: 100.0, 6: 100.0, 7: 100.0}
    _box(labels, (5, 0, 4), (8, 4, 9), 1)            # Y border
    _box(labels, (0, 10, 4), (3, 14, 9), 2)          # Z end
    _box(labels, (5, 10, 16), (7, 12, 18), 3)        # 8 voxels, the smallest
    _box(labels, (4, 25, 16), (9, 35, 26), 4)        # 500 voxels, the largest
    _box(labels, (5, 10, 40), (8, 12, 48), 5)        # 48 voxels, rim share 0.5 against 6
    _box(labels, (5, 12, 40), (8, 18, 48), 6)        # 144 voxels, rim share 0.2
    _box(labels, (5, 26, 60), (8, 31, 65), 7)        # 75 voxels, passes every rule
    stain = np.full(shape, 10.0)
    for label, level in levels.items():
        stain[labels == label] = level
    nuclei = Case("nuclei", {"DAPI": stain}, labels, None, (0.4, 0.2, 0.2),
                  dict(entry="nucleus", keep_z=False, margin_um=(1.5, 1.5, 1.5),
                       patches=dict(shape_zyx=(4, 8, 8), overlap_zyx=(0, 4, 4), per_origin=5, max_tries=2000)),
                  {1: "border", 2: "border", 3: "size", 4: "size", 5: "squashed", 6: "kept", 7: "kept"}, 100.0, 10.0)

    shape = (6, 30, 120)
    cells = np.zeros(shape, dtype=np.uint16)
    nucleus = np.zeros(shape, dtype=np.uint16)
    _box(cells, (0, 5, 0), (6, 12, 6), 1)            # X border
    _box(cells, (0, 5, 12), (6, 11, 18), 2)          # 216 voxels, the smallest
    _box(cells, (0, 3, 24), (6, 27, 44), 3)          # 2,880 voxels, the largest
    _box(cells, (0, 5, 50), (6, 15, 60), 4)          # 600 voxels; a nucleus of 20
    _box(cells, (0, 5, 66), (6, 17, 80), 5)          # 1,008 voxels; nuclei of 80 and 60
    _box(cells, (0, 5, 86), (6, 15, 96), 6)          # 600 voxels; passes every rule
    _box(nucleus, (1, 7, 2), (4, 10, 4), 1)          # in cell 1
    _box(nucleus, (1, 6, 13), (4, 10, 17), 2)        # in cell 2, 48 voxels
    _box(nucleus, (1, 10, 30), (5, 15, 36), 3)       # in cell 3
    _box(nucleus, (2, 8, 53), (3, 12, 58), 4)        # in cell 4, 20 voxels
    _box(nucleus, (1, 7, 68), (5, 11, 73), 5)        # in cell 5, 80 voxels
    _box(nucleus, (1, 12, 68), (4, 16, 73), 6)       # in cell 5, 60 voxels
    _box(nucleus, (1, 7, 88), (5, 12, 92), 7)        # in cell 6, 80 voxels
    dapi = np.full(shape, 5.0)
    for label in (1, 2, 3, 4, 7):
        dapi[nucleus == label] = 100.0
    dapi[nucleus == 5] = 150.0
    dapi[nucleus == 6] = 60.0
    flamingo = np.where(cells > 0, 50.0, 3.0)
    culture = Case("cells", {"DAPI": dapi, "Flamingo": flamingo}, cells, nucleus, (0.4, 0.4, 0.4),
                   dict(entry="cell", keep_z=True, margin_um=(0.0, 1.5, 1.5),
                        patches=dict(shape_zyx=(0, 8, 8), overlap_zyx=(0, 4, 4), per_origin=5, max_tries=2000)),
                   {1: "border", 2: "size", 3: "size", 4: "nucleus", 5: "kept", 6: "kept"}, 100.0, 5.0)
    return {"nuclei": nuclei, "cells": culture}


def _ball(shape, centre, radius, spacing):
    grid = np.indices(shape, dtype=np.float64)
    return sum(((g - c) * s) ** 2 for g, c, s in zip(grid, centre, spacing)) <= radius ** 2


@cache
def fall_off_cases():
    """``fall_off_cases``: OUT1's three hand-made cases on a grid of 0.2 × 0.1 × 0.1 µm (excess ``r`` = 10 over a
    background of 20).

    (a) Cap: label 1, a ball of radius 1 µm alone in its box, excess ``r`` inside, ``2r`` on the free voxels within
    1 µm of it and 0 beyond, so that the smoothed surroundings stay at or above the cap and the weight v is 1.
    (b) Neighbour: (a) with label 2, a box sharing a face with label 1, at excess ``10r`` (``b10``) and ``20r``
    (``b20``). (c) Rising: (b10) with the free-side surround 0 below 0.2 µm, rising linearly to ``0.8 r`` at 0.5 µm
    and ``0.8 r`` to 1 µm, and the excess ``r`` on the inner rim of label 1 only (0 deeper inside), so that the
    smoothed surroundings near the mask stay below the cap and the per-distance median of the capped values rises
    from about 0.37 r to 0.47 r between 0.05 and 0.25 µm before the cap brings it down. Three more balls at excess
    ``r``, far from label 1, keep the plateau (the median of the label medians) at ``r`` plus the background in every
    case.
    """
    shape, spacing, r, background = (24, 80, 140), (0.2, 0.1, 0.1), 10.0, 20.0
    centre = (11.5, 39.5, 39.5)
    one = _ball(shape, centre, 1.0, spacing)
    labels = np.zeros(shape, dtype=np.uint16)
    labels[one] = 1
    for k, x in enumerate((95.0, 115.0, 130.0)):
        labels[_ball(shape, (11.5, 39.5, x), 0.6, spacing)] = 3 + k
    d = ndi.distance_transform_edt(~one, sampling=spacing)
    cases = {}
    for name, neighbour, rising in (("a", None, False), ("b10", 10.0, False), ("b20", 20.0, False),
                                    ("c", 10.0, True)):
        L = labels.copy()
        if neighbour is not None:
            box = np.zeros(shape, dtype=bool)
            box[8:15, 32:47, 50:60] = True
            box &= ~one
            # the box from the first voxel column outside ball 1 on +x, so that the two share a face
            L[box & (L == 0)] = 2
        X = np.zeros(shape)
        X[L == 1] = r
        free = L == 0
        if rising:
            X[ndi.binary_erosion(one, structure=ndi.generate_binary_structure(3, 1))] = 0.0
            ring = np.clip((d - 0.2) / 0.3, 0.0, 1.0) * 0.8 * r
            X[free & (d <= 1.0)] = ring[free & (d <= 1.0)]
        else:
            X[free & (d <= 1.0)] = 2 * r
        if neighbour is not None:
            X[L == 2] = neighbour * r
        X[L >= 3] = r
        cases[name] = Case(name, {"DAPI": X + background}, L, None, spacing,
                           dict(entry="nucleus", keep_z=False, guard=100.0,
                                selection=dict(size_percentiles=(0.0, 100.0), rim_share_max=1.0),
                                patches=dict(shape_zyx=(4, 8, 8), overlap_zyx=(0, 4, 4), per_origin=2, max_tries=200)),
                           {1: "kept"}, r + background, background)
    return cases


def write_case(case, directory):
    """Write a hand-made case as TIFF files under ``directory`` and return its build configuration."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stains = {}
    for channel, array in case.stains.items():
        tifffile.imwrite(directory / f"{channel}.tif", array.astype(np.float32))
        stains[channel] = str(directory / f"{channel}.tif")
    tifffile.imwrite(directory / "labels.tif", case.labels)
    source = dict(image=case.name, stains=stains, labels=str(directory / "labels.tif"))
    if case.nucleus_labels is not None:
        tifffile.imwrite(directory / "nuclei.tif", case.nucleus_labels)
        source["nucleus_labels"] = str(directory / "nuclei.tif")
    return dict(case.settings, bank=f"case-{case.name}", channels=list(case.stains),
                spacing_zyx=list(case.spacing), sources=[source], held_out=[])


@cache
def cf_bank(preset):
    """``cf_banks``: the closed-form bank of a routine context preset, ``n`` = 4 (CF1)."""
    return closed_form_bank(preset, 4)


def cf_sources(preset):
    """The source images of ``cf_bank(preset)`` as the build loads them (stains, labels, nucleus labels, spacing),
    rendered by the generator from the cached scenes; the oracle recomputes the bank from these arrays."""
    from starfinder.synthetic._appearance_bank import _closed_form_source
    return [_closed_form_source(preset, seed) for seed in SEEDS if seed != HELD_OUT]


def cf_bank_files(preset, directory):
    """``cf_bank_files``: the four scenes of ``cf_bank(preset)`` written by ``save_context_scene`` under
    ``directory``, each with its stains on the sample grid (``sample_stains/<name>.tif``, the rendering the
    closed-form bank reads, which ``save_context_scene`` writes only per FOV at the FOVs' offsets), and the build
    configuration that names each scene's stain files and truth label files (``labels/sample_nuclei.tif``,
    ``labels/sample_cells.tif``) as one source image, the seed 45 scene's files held out. The sources are the arrays
    of ``cf_sources(preset)`` under the same image IDs, so the file bank equals ``cf_bank(preset)``."""
    from starfinder.synthetic._appearance_bank import (_closed_form_channels, _closed_form_configuration,
                                                       _closed_form_scene, _closed_form_stain, _context_of)
    directory = Path(directory)
    context = _context_of(preset)
    sources, held_out = [], []
    for seed in SEEDS:
        root = directory / f"seed-{seed}"
        scene = _closed_form_scene(preset, seed)
        stains = {}
        for name in _closed_form_channels(preset):
            path = root / "sample_stains" / f"{name}.tif"
            if not root.exists() or not path.exists():
                if not root.exists():
                    save_context_scene(scene, root)
                path.parent.mkdir(exist_ok=True)
                tifffile.imwrite(path, _closed_form_stain(preset, seed, name).astype(np.uint8))
            stains[name] = str(path)
        labels = root / "labels" / ("sample_cells.tif" if context == "culture" else "sample_nuclei.tif")
        source = dict(image=f"{preset}/seed-{seed}", stains=stains, labels=str(labels), project_z=context == "tissue2d")
        if context == "culture":
            source["nucleus_labels"] = str(root / "labels" / "sample_nuclei.tif")
        if seed == HELD_OUT:
            held_out.append(dict(image=source["image"], files=sorted(
                [*stains.values(), source["labels"], *([source["nucleus_labels"]] if context == "culture" else [])])))
        else:
            sources.append(source)
    return dict(_closed_form_configuration(preset), bank=f"files-{preset}", sources=sources, held_out=held_out)


def write_config(config, path):
    """Write a build configuration as JSON and return its path."""
    Path(path).write_text(json.dumps(config, indent=1), encoding="utf-8")
    return Path(path)
