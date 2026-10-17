"""Fixtures of the real-appearance scene's generator checks (docs/synthetic-real-appearance-validation.md, "Fixtures").

The bank's part (W-516): ``bank_cases`` and ``fall_off_cases`` are hand-made label and stain arrays with a known outcome;
``cf_banks`` are the closed-form banks of the three routine context presets (CF1, ``n`` = 4: the seeds 42 to 45, the
seed 45 image held out); ``cf_bank_files`` are the same scenes written by ``save_context_scene`` and configured as file
sources of ``build_appearance_bank``. Nothing is read from a run directory or a real image: every array is built in
the session and every file is written under the test's temporary directory.
"""
from dataclasses import dataclass, replace
from functools import cache
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

from starfinder.synthetic import (AppearanceSceneConfig, BankReference, CellTypeConfig, CompositionConfig,
                                  ContextSceneConfig, MorphologyRoundConfig, NoiseConfig, OutputScaleConfig,
                                  ScalarDistribution, StainChannelConfig, TerritoryConfig, closed_form_bank,
                                  generate_appearance_scene, save_context_scene)
from starfinder.synthetic._appearance_layout import AugmentationConfig, LayoutConfig, _place_appearance

from .cellular_scenes import float64, plane, tiny

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
    shape = (14, 40, 64)
    labels = np.zeros(shape, dtype=np.uint16)
    levels = {1: 100.0, 2: 100.0, 3: 90.0, 4: 110.0, 5: 100.0, 6: 100.0, 7: 100.0}
    _box(labels, (5, 0, 4), (8, 4, 9), 1)            # Y border
    _box(labels, (0, 10, 4), (3, 14, 9), 2)          # Z end
    _box(labels, (5, 10, 16), (7, 12, 18), 3)        # 8 voxels, the smallest
    _box(labels, (4, 25, 4), (9, 35, 14), 4)         # 500 voxels, the largest
    _box(labels, (5, 10, 28), (8, 12, 36), 5)        # 48 voxels, rim share 0.5 against 6
    _box(labels, (5, 12, 28), (8, 18, 36), 6)        # 144 voxels, rim share 0.2
    _box(labels, (5, 26, 24), (8, 31, 29), 7)        # 75 voxels, passes every rule
    stain = np.full(shape, 10.0)
    for label, level in levels.items():
        stain[labels == label] = level
    nuclei = Case("nuclei", {"DAPI": stain}, labels, None, (0.4, 0.2, 0.2),
                  dict(entry="nucleus", keep_z=False, margin_um=(1.5, 1.5, 1.5),
                       patches=dict(shape_zyx=(4, 8, 8), overlap_zyx=(0, 4, 4), per_origin=5, max_tries=2000)),
                  {1: "border", 2: "border", 3: "size", 4: "size", 5: "squashed", 6: "kept", 7: "kept"}, 100.0, 10.0)

    shape = (6, 64, 64)
    cells = np.zeros(shape, dtype=np.uint16)
    nucleus = np.zeros(shape, dtype=np.uint16)
    _box(cells, (0, 5, 0), (6, 12, 6), 1)            # X border
    _box(cells, (0, 5, 12), (6, 11, 18), 2)          # 216 voxels, the smallest
    _box(cells, (0, 3, 24), (6, 27, 44), 3)          # 2,880 voxels, the largest
    _box(cells, (0, 35, 2), (6, 45, 12), 4)          # 600 voxels; a nucleus of 20
    _box(cells, (0, 35, 18), (6, 47, 32), 5)         # 1,008 voxels; nuclei of 80 and 60
    _box(cells, (0, 35, 40), (6, 45, 50), 6)         # 600 voxels; passes every rule
    _box(nucleus, (1, 7, 2), (4, 10, 4), 1)          # in cell 1
    _box(nucleus, (1, 6, 13), (4, 10, 17), 2)        # in cell 2, 48 voxels
    _box(nucleus, (1, 10, 30), (5, 15, 36), 3)       # in cell 3
    _box(nucleus, (2, 38, 5), (3, 42, 10), 4)        # in cell 4, 20 voxels
    _box(nucleus, (1, 37, 20), (5, 41, 25), 5)       # in cell 5, 80 voxels
    _box(nucleus, (1, 42, 20), (4, 46, 25), 6)       # in cell 5, 60 voxels
    _box(nucleus, (1, 37, 42), (5, 42, 46), 7)       # in cell 6, 80 voxels
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
    between 0.05 and 0.25 µm before the cap brings it down. The grid is 16 × 64 × 64 voxels, so label 1's box is cut
    by the Z ends and the Y = 0 edge. Three balls of radius 0.5 µm at excess ``r``, outside label 1's box, keep the
    plateau (the median of the label medians) at ``r`` plus the background in every case.
    """
    shape, spacing, r, background = (16, 64, 64), (0.2, 0.1, 0.1), 10.0, 20.0
    centre = (7.5, 24.5, 24.5)
    one = _ball(shape, centre, 1.0, spacing)
    labels = np.zeros(shape, dtype=np.uint16)
    labels[one] = 1
    for k, x in enumerate((8.0, 30.0, 52.0)):
        labels[_ball(shape, (7.5, 57.0, x), 0.5, spacing)] = 3 + k
    d = ndi.distance_transform_edt(~one, sampling=spacing)
    cases = {}
    for name, neighbour, rising in (("a", None, False), ("b10", 10.0, False), ("b20", 20.0, False),
                                    ("c", 10.0, True)):
        L = labels.copy()
        if neighbour is not None:
            box = np.zeros(shape, dtype=bool)
            box[4:11, 17:32, 35:45] = True
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


# ------------------------------------------------------------------------------------------------ the layout's part
# (W-517) docs/synthetic-real-appearance-validation.md, "Fixtures": ``app_tiny`` (sample-tiny-v1, contact 1.0, DN3) in
# three copies on the closed-form banks, ``app_plane`` (the same on ``scene_plane``), ``app_morph`` (the culture copy
# with a morphology round), ``app_crowded`` and ``app_streams`` (copies of the dense 3D one), ``app_cu_offset`` and
# ``app_cases`` (hand-made banks). Each DN3 count is the count reached by the same configuration when placement first
# stops at a run of 50 refused proposals ("the count that fills the domain at the first refusal run"; the run of 50 is
# app_crowded's ``max_refusals``); GA10 checks the dense 3D count against app_crowded.

APP_BANKS = dict(dense3d="context-dense3d-v1", tissue2d="context-tissue2d-v1", culture="context-culture-v1")
REFUSAL_RUN = 50
APP_COUNTS = {"dense3d": 45, "tissue2d": 19, "culture": 2, "dense3d-plane": 21, "tissue2d-plane": 19,
              "culture-plane": 2, "culture-morph-drawn": 2, "culture-morph-supplied": 2, "cu-offset": 16}
# CT1's growth: the accepted growth laws of the dense 3D and 2D tissue context presets (TerritoryConfig, T2).
GROWTH = dict(dense3d=ScalarDistribution("lognormal", (math.log(0.78), 0.15)),
              tissue2d=ScalarDistribution("lognormal", (math.log(0.5), 0.15)))
# app_streams: AU1's gain range changed, and type 2's size scale 1.5.
GAIN = (0.95, 1.05)
# app_morph: a whole-voxel supplied translation of one plane for every FOV, or MT1's drawn one. The bound grows Ω by
# T ⊙ s on each side; (1, 0.5, 0.5) voxels keeps the sample grid within 8 × 64 × 64 (the page's (0, 2, −3) would need
# 69 voxels in Y and X, above the routine bound of 32 × 64 × 64).
MORPH_WHOLE = (1.0, 0.0, 0.0)
MORPH_BOUND = (1.0, 0.5, 0.5)


def _stains(copy, morphology):
    """The stain channels of a copy (names, rounds and file patterns of the bank's channels)."""
    if copy == "tissue2d":
        return (StainChannelConfig("PI"),)
    dapi = StainChannelConfig("DAPI")
    if copy == "dense3d":
        return (dapi,)
    flamingo = StainChannelConfig("Flamingo", source="territory", channel="ch05")
    if morphology is None:
        return dapi, flamingo
    return dapi, replace(dapi, round="morphology", channel="ch00"), replace(flamingo, round="morphology",
                                                                            channel="ch02")


def app_context(copy, flat=False, morphology=None, planes=None, types=None):
    """The ContextSceneConfig of an app_tiny copy: sample-tiny-v1 with float64 output and no sample-scene stain, on
    ``scene_plane`` when ``flat``, with ``planes`` planes per FOV when given (app_cu_offset), CT1's growth law in the
    dense contexts and the copy's stains."""
    _, sample = tiny()
    sample = float64(sample)
    if flat:
        sample = plane(sample)
    if planes is not None:
        shape = (planes, *sample.acquisition.fov_shape_zyx[1:])
        sample = replace(sample, acquisition=replace(sample.acquisition, fov_shape_zyx=shape),
                         fov=replace(sample.fov, shape_zyx=shape))
    territories = None if copy == "culture" else TerritoryConfig("grown", growth=GROWTH[copy])
    return ContextSceneConfig(sample=sample, territories=territories, types=types,
                              stains=_stains(copy, morphology), morphology=morphology)


def morphology_round(mode, flat=False):
    """app_morph's round ``organelle``: MT1's drawn fractional translation, or the supplied whole-voxel one."""
    bound = (0.0, *MORPH_BOUND[1:]) if flat else MORPH_BOUND
    if mode == "drawn":
        return MorphologyRoundConfig(translation_max_zyx=bound)
    return MorphologyRoundConfig(translation="supplied", translation_max_zyx=bound,
                                 translation_zyx={fov: MORPH_WHOLE for fov in ("FOV_001", "FOV_002", "FOV_003",
                                                                               "FOV_004")})


def app_layout(count, **changes):
    """DN3 with ``count`` objects, contact 1.0."""
    return LayoutConfig(density="count", count=count, contact=1.0, **changes)


@cache
def app_tiny(name):
    """A placed copy by name: ``dense3d``, ``tissue2d``, ``culture`` (app_tiny), the same with ``-plane``
    (app_plane; the culture with MT1's drawn round), ``culture-morph-drawn`` and ``culture-morph-supplied``
    (app_morph), ``<copy>-gain`` and ``<copy>-size-scale`` (app_streams; the page's are the dense 3D ones). Returns
    (context, bank, augmentation, layout, placed)."""
    copy = name.split("-")[0]
    flat = name.endswith("-plane")
    morphology = None
    if name.startswith("culture-morph-"):
        morphology = morphology_round(name.rsplit("-", 1)[1])
    elif name == "culture-plane":
        morphology = morphology_round("drawn", flat=True)
    types = CellTypeConfig(size_scale=(1.0, 1.5, 1.0, 1.0)) if name.endswith("-size-scale") else None
    context = app_context(copy, flat=flat, morphology=morphology, types=types)
    augmentation = AugmentationConfig(gain=GAIN) if name.endswith("-gain") else AugmentationConfig()
    key = name if name in APP_COUNTS else copy
    layout = app_layout(APP_COUNTS[key])
    bank = cf_bank(APP_BANKS[copy])
    return context, bank, augmentation, layout, _place_appearance(context, bank, augmentation, layout)


@cache
def app_crowded(variant):
    """app_crowded: the dense 3D app_tiny with a DN3 count larger than its domain holds (100), stopping after a run of
    50 refusals at ``floor`` 0.1 (``record``) or 0.9 (``raise``, returns the error), or after 30 proposals
    (``proposals``)."""
    context, bank, augmentation = app_context("dense3d"), cf_bank(APP_BANKS["dense3d"]), AugmentationConfig()
    changes = dict(record=dict(floor=0.1, max_refusals=REFUSAL_RUN), proposals=dict(floor=0.1, max_proposals=30),
                   **{"raise": dict(floor=0.9, max_refusals=REFUSAL_RUN)})[variant]
    layout = app_layout(100, **changes)
    try:
        return context, bank, layout, _place_appearance(context, bank, augmentation, layout)
    except ValueError as error:
        return context, bank, layout, error


def cu_offset_case():
    """app_cu_offset's source image: 7 planes of 64 × 64 voxels (0.346255 × 0.4 × 0.4 µm), four cells of 16 × 16
    voxels through every plane (box centre on plane 3), each with a nucleus of 6 × 6 voxels filling planes 3 to 5
    (centroid on plane 4, 0.346 µm above the box centre); DAPI 100 in the nuclei and 5 elsewhere, Flamingo 50 in the
    cells and 3 elsewhere. Every rule is set so that the four cells are kept."""
    shape = (7, 64, 64)
    cells = np.zeros(shape, dtype=np.uint16)
    nucleus = np.zeros(shape, dtype=np.uint16)
    for label, (y, x) in enumerate(((8, 8), (8, 40), (40, 8), (40, 40)), start=1):
        _box(cells, (0, y, x), (7, y + 16, x + 16), label)
        _box(nucleus, (3, y + 5, x + 5), (6, y + 11, x + 11), label)
    dapi = np.where(nucleus > 0, 100.0, 5.0)
    flamingo = np.where(cells > 0, 50.0, 3.0)
    return Case("cu-offset", {"DAPI": dapi, "Flamingo": flamingo}, cells, nucleus, (0.346255, 0.4, 0.4),
                dict(entry="cell", keep_z=True, margin_um=(0.0, 1.5, 1.5),
                     selection=dict(size_percentiles=(0.0, 100.0), rim_share_max=1.0),
                     patches=dict(shape_zyx=(0, 8, 8), overlap_zyx=(0, 4, 4), per_origin=2, max_tries=200)),
                {1: "kept", 2: "kept", 3: "kept", 4: "kept"}, 100.0, 5.0)


def spheres_case():
    """app_cases' dense bank: two balls of radius 1 µm (the voxels within 4 voxels of their centre on a grid of
    0.25 µm, exact in binary), each centred on a voxel with 10 voxels to every side of it, so that its box (margin
    1.5 µm, 6 voxels) is centred on it; stain 100 inside, 10 outside. The source is 21 × 21 × 43 voxels."""
    shape = (21, 21, 43)
    labels = np.zeros(shape, dtype=np.uint16)
    z, y, x = np.indices(shape)
    for label, cx in ((1, 10), (2, 32)):
        labels[(z - 10) ** 2 + (y - 10) ** 2 + (x - cx) ** 2 <= 16] = label
    stain = np.where(labels > 0, 100.0, 10.0)
    return Case("spheres", {"DAPI": stain}, labels, None, (0.25, 0.25, 0.25),
                dict(entry="nucleus", keep_z=False, margin_um=(1.5, 1.5, 1.5),
                     selection=dict(size_percentiles=(0.0, 100.0), rim_share_max=1.0),
                     patches=dict(shape_zyx=(4, 8, 8), overlap_zyx=(0, 4, 4), per_origin=1, max_tries=50)),
                {1: "kept", 2: "kept"}, 100.0, 10.0)


def case_placements(rows):
    """A placements table written by hand: (object_id, entry_id, centre_zyx), the identity transform and gain 1."""
    table = pd.DataFrame([dict(object_id=k, entry_id=entry, centre_z=c[0], centre_y=c[1], centre_x=c[2], theta=0.0,
                               flip_z=1, flip_y=1, scale_z=1.0, scale_yx=1.0, gain=1.0) for k, entry, c in rows])
    return table.astype(dict(object_id=np.uint32))


# ---------------------------------------------------------------------------------------------- the composition's part
# (W-518) docs/synthetic-real-appearance-validation.md, "Fixtures": the generated scenes of ``app_tiny`` (float64, cast
# off) and ``app_tiny_uint8`` (the 8-bit cast), ``app_plane``, ``app_morph``, ``app_coarse`` (the dense 3D copy at
# twice the bank's source spacing in Y and X, 0.388796 µm, so that IN2's prefilter is active on every object),
# ``app_bg1`` (BG1 from the closed-form bank's patches instead of BG2), ``app_dn1`` (DN1 with its target and threshold
# from the closed-form bank's source images, a check after every placement; the short copies with ``max_proposals``
# too small to reach the target at ``floor`` 0.1 and 0.99), ``app_streams``' size-scale and dropped-stain copies, and
# GA17's copies with the cell types of ``ctx_types``. Each is generated by ``generate_appearance_scene`` on the
# closed-form bank of its context. Additions beyond the page's copies, so that each check also runs in the routine
# tier where the page's copy measures 5 s or more: the 2D tissue copies of app_bg1 and app_morph, and GA17's copies
# with 20 molecules per cell (``-light``) beside ctx_types' 200.

COARSE_YX = 2 * 0.194398
APP_SCENE_COUNTS = {"dense3d-coarse": 15}
DN1_SHORT = 10
# GA17's routine copy: ctx_types with 20 molecules per cell (median) instead of 200, M2 and NF1 kept; the page's
# copies with 200 render thousands of amplicons and measure 5 s or more.
LIGHT_MOLECULES = 20.0


def _uint8(stains):
    return tuple(replace(s, output=OutputScaleConfig(dtype="uint8")) for s in stains)


def appearance_config(name):
    """The AppearanceSceneConfig of a generated copy by name: ``<copy>`` (app_tiny), ``<copy>-uint8``
    (app_tiny_uint8), ``<copy>-plane`` (app_plane), ``culture-morph-drawn``, ``culture-morph-supplied`` and
    ``tissue2d-morph-supplied`` (app_morph), ``dense3d-coarse`` (app_coarse), ``<copy>-bg1`` (app_bg1),
    ``<copy>-dn1``, ``<copy>-dn1-short-0.1`` and ``<copy>-dn1-short-0.99`` (app_dn1), ``<copy>-types`` (GA17),
    ``<copy>-size-scale`` (app_streams, with ``ctx_types``), the same with ``-light`` (20 molecules per cell),
    ``<copy>-types-light-spread0`` (per_cell_spread 0, GN11's M1), a ``-quiet`` suffix (no FOV noise, GN11's rendering
    identity) and ``culture-no-flamingo`` (app_streams' dropped stain),
    with ``<copy>`` one of ``dense3d``, ``tissue2d``, ``culture``."""
    from .cellular_scenes import MOLECULES, TYPES
    copy = name.split("-")[0]
    rest = name[len(copy):]
    flat = rest == "-plane"
    morphology = None
    if rest.startswith("-morph-"):
        morphology = morphology_round(rest.rsplit("-", 1)[1])
    elif name == "culture-plane":
        morphology = morphology_round("drawn", flat=True)
    types = molecules = None
    quiet = rest.endswith("-quiet")
    if quiet:
        rest = rest[:-len("-quiet")]
    if rest in ("-types", "-size-scale", "-types-light", "-size-scale-light", "-types-light-spread0"):
        types, molecules = TYPES, MOLECULES
        if "-light" in rest:
            molecules = replace(MOLECULES, per_cell_median=LIGHT_MOLECULES)
        if rest.endswith("-spread0"):
            molecules = replace(molecules, per_cell_spread=0.0)
        if rest.startswith("-size-scale"):
            types = replace(TYPES, size_scale=(1.0, 1.5, 1.0, 1.0))
    if copy == "tissue2d" and morphology is not None:
        context = app_context(copy, flat=flat, types=types)
        pi = context.stains[0]
        context = replace(context, stains=(pi, replace(pi, round="morphology", channel="ch00")), morphology=morphology)
    else:
        context = app_context(copy, flat=flat, morphology=morphology, types=types)
    if molecules is not None:
        context = replace(context, molecules=molecules)
    if quiet:
        # GA17's rendering identity: the amplicon images without the FOV noise (GN11's quiet copy).
        context = replace(context, sample=replace(context.sample, fov=replace(context.sample.fov, noise=NoiseConfig())))
    if rest == "-uint8":
        context = replace(context, stains=_uint8(context.stains))
    if name == "culture-no-flamingo":
        context = replace(context, stains=tuple(s for s in context.stains if s.name != "Flamingo"))
    if rest == "-coarse":
        acquisition = context.sample.acquisition
        spacing = (acquisition.spacing_zyx[0], COARSE_YX, COARSE_YX)
        context = replace(context, sample=replace(context.sample, acquisition=replace(acquisition,
                                                                                      spacing_zyx=spacing)))
    key = name if name in APP_COUNTS else name if name in APP_SCENE_COUNTS else copy
    count = APP_SCENE_COUNTS.get(key, APP_COUNTS.get(key, APP_COUNTS.get(f"{copy}-plane" if flat else copy)))
    if rest.startswith("-morph-"):
        count = APP_COUNTS["culture-morph-" + rest.rsplit("-", 1)[1]] if copy == "culture" else APP_COUNTS[copy]
    layout = app_layout(count)
    if rest.startswith("-dn1"):
        floor = float(rest.rsplit("-", 1)[1]) if "-short-" in rest else 0.9
        layout = LayoutConfig(density="visible", contact=1.0, max_refusals=REFUSAL_RUN, floor=floor,
                              **(dict(max_proposals=DN1_SHORT) if "-short-" in rest else {}))
    composition = CompositionConfig(background="patches") if rest == "-bg1" else CompositionConfig()
    return AppearanceSceneConfig(context, BankReference(preset=APP_BANKS[copy]), layout=layout,
                                 composition=composition)


@cache
def app_scene(name):
    """A generated copy by name (``appearance_config``): (codebook, config, scene), or with a stop below ``floor``
    (codebook, config, the ValueError)."""
    book, _ = tiny()
    config = appearance_config(name)
    try:
        return book, config, generate_appearance_scene(book, config=config)
    except ValueError as error:
        return book, config, error


def bg2_settings(preset, name):
    """The stain background of a closed-form bank's preset as the oracle's input for BG2 (the fixture's own
    configuration): the preset's sample grid origin −(J + T) ⊙ s and spacing, its BackgroundConfig's baseline, tissue
    weight and regions (index units), and its output scale (the output plateau over the median of the brightness
    law)."""
    from starfinder.synthetic import context_scene_preset
    _, config = context_scene_preset(preset)
    stain = next(s for s in config.stains if s.name == name)
    acquisition = config.sample.acquisition
    spacing = np.asarray(acquisition.spacing_zyx, dtype=np.float64)
    grow = np.zeros(3) if config.morphology is None else np.asarray(config.morphology.translation_max_zyx)
    background = stain.background
    regions = (list(zip(background.region_centers, background.region_sigma_zyx, background.region_heights))
               if background.regions_enabled else [])
    brightness = stain.brightness
    median = math.exp(brightness.parameters[0]) if brightness.mode == "lognormal" else float(brightness.parameters[0])
    return dict(origin=(-(np.asarray(acquisition.jitter_max_zyx, dtype=np.float64) + grow) * spacing).tolist(),
                spacing=spacing.tolist(),
                baseline=float(np.asarray(background.baseline)[0, 0]) if background.baseline_enabled else 0.0,
                weight=float(np.asarray(background.tissue_weights)[0, 0]) if background.regions_enabled else 0.0,
                regions=[(tuple(c), tuple(w), float(h)) for c, w, h in regions],
                scale=float(stain.output.plateau) / median)
