"""Chain checks of the cellular scene with real appearance (W-487): the §2.9 methods through each context's route.

docs/synthetic-cellular-scene-validation.md ("Chain checks", CD0 to CD4, CT0 to CT4, CU0 to CU4) as amended by
docs/synthetic-real-appearance-validation.md ("Consequences for the chain checks", with CD5, CT5 and CU5): one chain
per context on the appearance presets (VA1), each step its own test so that a miss in one step does not hide the
others. It is an engineering check of the wiring from stain to segmentation, assignment, assembly and report against
the scene's truth; it claims no accuracy and compares no methods. The truth nuclei and culture cells of the `slow`
presets are a method's own labels of real images, so agreement there is read as "The claim boundary" of the
appearance validation page states (the same method on its own labels, or a related one).

The chains (fixtures of the two pages, built once per module and only when a selected test needs them):

* ``chain_dense3d``: ``appearance-dense3d-slow-v1`` (bank ``ln-sample1-v1``), route per FOV; per FOV StarDist
  ``3D_spleen`` (scale 1.0) on the reference stain ``DAPI`` and the cells by ``assign_molecules`` with
  ``ExpandLabelsConfig(0.7776, "um", "planar")`` (W-382's LN setting).
* ``chain_tissue2d``: ``appearance-tissue2d-slow-v1`` (bank ``tissue2d-v1``, 0.18927 µm), route after fusion;
  ``Dataset.stitch`` on ``PI``; the export's ``segmentation`` block with StarDist ``2D_versatile_fluo`` at the scale
  0.5 (the model sees 0.37854 µm pixels) on the fused ``PI`` and ``seeded_watershed`` (sigma 1.5 µm, Otsu) on the
  fused nuclei–amplicon composite, seeded by the nuclei, device CPU. The route has no composite of its own and one
  device per export: as W-393 did, the composite is made per FOV (``composite_nuclei_amplicon`` of ``PI`` and the
  reference round's channel maximum) and fused, and every run is on the CPU (stated, not fixed here).
* ``chain_culture``: ``appearance-culture-slow-v1`` (bank ``culture3d-v1``, 0.388796 µm) at the seeds 42 and 43,
  fixed in advance (VA9); route per FOV; per FOV ``FOV.prepare_morphology(MorphologyConfig())`` (MR1), the nuclei by
  Cellpose ``cpsam_v2`` on the Z maximum of the registered morphology ``DAPI``, extended through z with W-382's
  culture nucleus setting, and the cells by Cellpose ``cpsam_v2`` on the ``Flamingo`` and ``DAPI`` projections of the
  registered round, extended through z with W-382's culture cell setting (``extend_labels_through_z`` on the 3D
  Flamingo: median 1.94398 µm, Otsu, 7.5581 µm², dilation 1.94398 µm, holes twice: W-382's pixels × 0.194398 µm).
  Cellpose's diameters are W-382's 60 and 240 pixels at 0.194398 µm divided by the pixel ratio (LS2).
* ``chain_*_routine``: the same on ``appearance-<context>-v1`` with its closed-form bank (seed 42 only).
* ``chain_*_truth``: each chain with the method replaced by the truth labels and no model called: on the route per
  FOV each FOV's truth labels imported with ``import_labels(…, relabel=False)`` and assigned without expansion; on
  the route after fusion the truth nuclei and cells on the segmentation grid, evaluated by the appearance point
  evaluator at the grid's pixel centres (V3; the Z maximum of the IDs over the sample grid's planes, PT1),
  imported as ``truth_import`` and ``truth_runs`` build them.

Every chain: the molecules at their truth positions (each FOV's legacy CSV, V2 a); ``Dataset.stitch`` with
``MultiviewStitcherConfig()`` on the reference stain, its placement used when every FOV is ``image_supported`` (2D
tissue: when TH5 holds) and the truth placement otherwise (SC1); ``Dataset.export``; ``Dataset.report`` and
``load_report``. The truth-label chain uses the placement its method chain uses.

Each stage that generates, stitches, segments, assigns and exports, or reports runs in its own process under
``/usr/bin/time -v`` (one CPU thread, no GPU, ``HF_HUB_OFFLINE``), so that each step's wall time and peak RSS are
measured; the test process compares. ``STARFINDER_CHAIN_STEP_WRAPPER`` names a command prefixed to each step (the
worker's record of a run); ``STARFINDER_CHAIN_OUTPUT`` names the folder of the per-object tables and step logs
(default: the module's temporary folder). The models are read by path (``STARFINDER_STARDIST_3D_SPLEEN``,
``STARFINDER_WEIGHTS_DIR``) and checked against the SHA-256 of W-392's ``inventory/models.json``, the banks through
``STARFINDER_APPEARANCE_BANKS`` with the SHA-256 their preset names; a test whose extra, model or bank is absent
skips with its reason, and nothing is downloaded. The thresholds TH1 to TH5 are W-393's, approved for the appearance
presets at W-515, unchanged; a miss is a finding ("What a failure means").
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy import ndimage

from .cellular_measures import object_shapes, touching_share, volume_fraction
from .test_segmentation_validation import l12_agreement

pytestmark = [pytest.mark.integration, pytest.mark.validation]

# --- The thresholds (docs/synthetic-cellular-scene-validation.md, "Thresholds and where they come from"), as the
# gate W-515 approved them for the appearance presets, unchanged.
TH1_MATCHED = 0.9        # matched fraction at IoU >= 0.5 of the interior truth nuclei, at least
TH2_MERGED = 0.10        # merged truth nuclei over the truth count, at most
TH3_SPLITS = 0           # splits
TH4_AGREEMENT = 0.85     # agreement among assigned molecules, at least
TH5_ERROR_UM = 1.0       # every FOV image_supported and every per-axis origin error (mean removed) at most, µm
TOL = 1e-9               # the project's default float tolerance (identities a test recomputes)
# The amendment of 2026-10-18 (W-487; operator decision pending Jiahao's confirmation; docs/synthetic-real-appearance-
# validation.md, "Consequences for the chain checks"): on the slow presets, CD1's TH1 and CU1's TH3 are reported beside
# their threshold, not asserted (known limits: the dense 3D scene's label density, L1 and L4; the culture's multi-piece
# and two-lobed truth masks, L6). Every other threshold stays asserted; no threshold value changes.
AMENDED = {("CD1", "TH1"), ("CU1", "TH3")}
AMENDMENT = "reported beside the threshold, not asserted: amendment of 2026-10-18 (W-487)"
# W-393's ranges of the method's labels of real images, for step 5 (TR1, TR2, TR11; reported, never asserted).
STEP5_RANGES = {"dense3d": {"TR1": (0.44, 0.57), "TR2": (0.07, 0.16), "TR11": (90.0, 185.0)},
                "tissue2d": {"TR1": (0.76, 0.82), "TR2": (0.30, 0.44)},
                "culture": {"TR1": (0.33, 0.72), "TR2": (0.17, 0.45)}}

# --- The contexts, their presets, routes and settings.
CONTEXTS = {
    "dense3d": dict(slow="appearance-dense3d-slow-v1", routine="appearance-dense3d-v1", bank="ln-sample1-v1",
                    stain="DAPI", route="per_fov", seeds={"slow": (42,), "routine": (42,)},
                    models=("3D_spleen",), extras=("stardist",)),
    "tissue2d": dict(slow="appearance-tissue2d-slow-v1", routine="appearance-tissue2d-v1", bank="tissue2d-v1",
                     stain="PI", route="after_fusion", seeds={"slow": (42,), "routine": (42,)},
                     models=("2D_versatile_fluo",), extras=("stardist",)),
    "culture": dict(slow="appearance-culture-slow-v1", routine="appearance-culture-v1", bank="culture3d-v1",
                    stain="DAPI", route="per_fov", seeds={"slow": (42, 43), "routine": (42,)},
                    models=("cpsam_v2",), extras=("cellpose",)),
}
TIERS = ("slow", "routine")
LN_EXPANSION_UM = 0.7776                  # W-382's LN setting (W-393 notes.md §4)
STARDIST_2D_PIXEL_UM = 0.37854            # the pixel StarDist 2D sees on tissue-2D's route (scale 0.25 of 0.094635)
W382_PIXEL_UM = 0.194398                  # the culture's acquired pixel, W-382's settings in pixels of it
CELLPOSE_DIAMETERS_PX = {"nucleus": 60, "cell": 240}
# W-382's culture extension (scripts/w382.py, EXTEND_NUCLEI_PX and EXTEND_CELLS_PX) in µm of its pixel.
EXTENSION_PX = {"nucleus": dict(median=10, min_area=10, dilation=5, fill_holes="once"),
                "cell": dict(median=10, min_area=200, dilation=10, fill_holes="twice")}
MORPHOLOGY = "organelle"
REPORT_PAIRS = (("balanced-01", "balanced-03"), ("balanced-01", "balanced-02"))   # exclusive, co-expressed (C8)

# --- The models (docs/synthetic-cellular-scene-validation.md, "Execution conditions"): each file's SHA-256 as W-392's
# inventory/models.json records it (runs/W-392/20261009T0323Z-inputs/), checked before the first call.
MODELS = {
    "3D_spleen": dict(variable="STARFINDER_STARDIST_3D_SPLEEN", folder="", path="", files={
        "config.json": "130a82e96339cef2e4460e509e1a3599b5f76bfb9d72022fd79839bac0b477a4",
        "thresholds.json": "b350f3e66950cb4fcebb90315af058e0b2ec42440f07c61087c601b5ecd16d53",
        "weights_best.h5": "30a5bbc8987dd7e58b6db57705780d8adb317c64ed7d82fcc4369eb7f6da8091"}),
    "2D_versatile_fluo": dict(variable="STARFINDER_WEIGHTS_DIR", folder="stardist/2D_versatile_fluo", path="", files={
        "config.json": "836da16282c3e0db1ba2e58f377e977419887ace8c737ebde16a094d58d50f74",
        "thresholds.json": "5cd6aac6e923f8659b63a9e297485920d45a59dad35371ddb8b6ae4be398d805",
        "weights_best.h5": "42202bd269c8106782316f1a2c75afb3f5ffa65e525c2e155ee0ced3a95da349"}),
    "cpsam_v2": dict(variable="STARFINDER_WEIGHTS_DIR", folder="cellpose/cpsam_v2", path="cpsam_v2", files={
        "cpsam_v2": "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667"}),
}
STEP_WRAPPER = "STARFINDER_CHAIN_STEP_WRAPPER"
OUTPUT = "STARFINDER_CHAIN_OUTPUT"
TIME = Path("/usr/bin/time")
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
AXES = "zyx"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _ns(*parts):
    return json.dumps(list(parts), separators=(",", ":"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str) + "\n", encoding="utf-8")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ===================================================================================================== the steps
# Each runs in its own process: ``python -c BOOT <step> <arguments as JSON>``; it writes its summary to
# ``arguments["summary"]``.

def _codebook(preset):
    from starfinder.synthetic import appearance_scene_preset
    return appearance_scene_preset(preset)[0]


def _raw_dataset(raw, root):
    """A Dataset of the raw dataset ``save_cellular_dataset`` wrote, from the fields ``dataset.json`` records, with its
    output under ``root`` (as the appearance contract tests build it)."""
    from starfinder.readout import load_codebook
    from starfinder.dataset import ChannelInfo, Dataset, RoundState
    record = _read_json(Path(raw) / "dataset.json")
    rounds = record["rounds"]
    dataset = Dataset(raw, root, record["dataset_id"], record["sample_id"], "output",
                      RoundState(rounds["sequencing"], rounds["other"], rounds["reference"]),
                      tuple(record["channel_order"]),
                      other_channel_order={name: tuple(ChannelInfo(c["channel"], c["name"]) for c in channels)
                                           for name, channels in record["other_channel_order"].items()},
                      reference_stains=tuple(ChannelInfo(c["channel"], c["name"]) for c in record["reference_stains"]),
                      voxel_size_xy=record["voxel_size_xy"], voxel_size_z=record["voxel_size_z"])
    labels = record["codebook"]
    dataset.codebook = load_codebook(Path(raw) / labels["path"], round_labels=tuple(labels["round_labels"]),
                                     channel_labels=tuple(labels["channel_labels"]))
    return record, dataset


def _work_root(base, summary):
    """The output root of a scene's shared files (its images, molecule tables and placements)."""
    return Path(base) / "work"


def _variant_root(base, summary, variant):
    return Path(base) / variant


def _grid(root, fov):
    from starfinder.segmentation import reference_grid_from_file
    return reference_grid_from_file(Path(root) / "images" / "ref_merged" / f"{fov}.tif")


def _image(root, name, fov):
    return np.asarray(tifffile.imread(Path(root) / "images" / name / f"{fov}.tif"), dtype=np.float32)


def _write_zyx(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.ascontiguousarray(image), photometric="minisblack", metadata={"axes": "ZYX"})


def _step_generate(a):
    """The scene of a preset at a seed: generated, saved (``save_appearance_scene``, ``save_cellular_dataset``) and
    written as the chain's shared inputs: the reference image (the reference round's channel maximum, with the FOV's
    grid), each stain as a float32 ZYX file, the 2D tissue composite, and each FOV's truth-position molecules."""
    from starfinder.io import save_volume
    from starfinder.segmentation import composite_nuclei_amplicon
    from starfinder.synthetic import (appearance_scene_preset, generate_appearance_scene, save_appearance_scene,
                                      save_cellular_dataset)
    base = Path(a["base"])
    book, config = appearance_scene_preset(a["preset"])
    sample_config = config.context.sample
    if a["seed"] != sample_config.fov.seed:
        config = replace(config, context=replace(config.context, sample=replace(
            sample_config, fov=replace(sample_config.fov, seed=a["seed"]))))
    scene = generate_appearance_scene(book, config=config)
    save_appearance_scene(scene, base / "scene")
    save_cellular_dataset(scene.context, base / "raw")
    sample = scene.context.sample
    fov_ids = list(sample.positions.fov_id)
    summary = dict(preset=a["preset"], seed=a["seed"], context=a["context"], dataset_id=config.context.sample.fov.
                   dataset_version, sample=sample.sample, fov_ids=fov_ids, genes=list(book.genes),
                   placed=scene.record["placed"], proposals=scene.record["proposals"],
                   density={k: v for k, v in scene.record["density"].items() if k != "trace"},
                   bank={k: scene.record["bank"][k] for k in ("name", "sha256", "entries")},
                   spacing_zyx=list(sample.record["sample_grid"]["spacing_zyx"]),
                   fov_shape_zyx=list(sample_config.acquisition.fov_shape_zyx),
                   sample_grid=sample.record["sample_grid"])
    root = _work_root(base, summary)
    reference = sample.record["effective_config"]["reference_round"]
    for fov in fov_ids:
        metadata = sample.fov_metadata[fov]
        merged = np.asarray(sample.fovs[fov].rounds[reference], dtype=np.float32).max(axis=-1)
        save_volume(merged, root / "images" / "ref_merged" / f"{fov}.tif", metadata=metadata)
        for name, image in scene.context.stains[fov].items():
            _write_zyx(root / "images" / name / f"{fov}.tif", np.asarray(image, dtype=np.float32))
        if a["context"] == "tissue2d":
            composite, _ = composite_nuclei_amplicon(scene.context.stains[fov]["PI"], merged)
            _write_zyx(root / "images" / "composite" / f"{fov}.tif", np.asarray(composite, dtype=np.float32))
        target = root / "signal" / f"{fov}_goodSpots.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / "scene" / "fovs" / fov / "molecules.csv", target)
    _write_json(a["summary"], summary)


def _step_stitch(a):
    """``Dataset.stitch`` with ``MultiviewStitcherConfig()`` on the reference stain; the placement used (SC1): the
    stitch's, approved, when every FOV is ``image_supported`` (and, in 2D tissue, every per-axis error is at most
    1 µm: TH5), else the truth placement (``placement_from_table`` with the true origins, approved)."""
    from starfinder.dataset import Dataset
    from starfinder.stitching import (MultiviewStitcherConfig, approve_placement, placement_from_table,
                                      positions_from_table, save_placement)
    base = Path(a["base"])
    summary = _read_json(base / "generate.json")
    sample, fovs = summary["sample"], summary["fov_ids"]
    dataset = Dataset(base / "raw", base / "work", summary["dataset_id"], sample, "output")
    root = Path(dataset.output_root)
    table = positions_from_table(base / "scene" / "positions.csv")
    truth = pd.read_csv(base / "scene" / "placement_truth.csv", float_precision="round_trip").set_index("fov_id")
    grids = {f: _grid(root, f) for f in fovs}
    out = dict(stitch="error", status={}, error_um=None, placement_used="truth")
    try:
        result = dataset.stitch(sample, fovs, table=table, config=MultiviewStitcherConfig(),
                                input=f"file:images/{a['stain']}/{{fovID}}.tif",
                                output_dir=root / "stitching_stitch" / sample)
        frame = result.fovs.set_index("fov_id")
        out["status"] = {f: str(frame.loc[f, "status"]) for f in fovs}
        placed = [f for f in fovs if out["status"][f] in ("image_supported", "metadata_only")]
        if placed:
            got = np.array([[float(frame.loc[f, f"origin_{x}"]) for x in AXES] for f in placed])
            true = np.array([[float(truth.loc[f, f"true_origin_{x}"]) for x in AXES] for f in placed])
            error = (got - got.mean(axis=0)) - (true - true.mean(axis=0))
            out["error_um"] = {f: error[k].tolist() for k, f in enumerate(placed)}
            out["max_abs_error_um"] = np.abs(error).max(axis=0).tolist()
        supported = all(out["status"][f] == "image_supported" for f in fovs)
        out["stitch"] = "placed" if supported else "stopped"
        within = out["error_um"] is not None and max(out["max_abs_error_um"]) <= TH5_ERROR_UM
        use = supported and (within if a["context"] == "tissue2d" else True)
        if use:
            placement = approve_placement(result, reviewer="test", note="synthetic scene; stitched placement")
            out["placement_used"] = "stitch"
    except Exception as error:   # noqa: BLE001 - where the stitch stops is the recorded outcome (SC1)
        out["error"] = f"{type(error).__name__}: {error}"
    if out["placement_used"] != "stitch":
        origins = {f: tuple(float(truth.loc[f, f"true_origin_{x}"]) for x in AXES) for f in fovs}
        placement = approve_placement(placement_from_table(table, grids, sample=sample, origins=origins),
                                      reviewer="test", note="truth placement (SC1)")
    save_placement(placement, root / "stitching" / sample)
    placed_origins = placement.fovs.set_index("fov_id")
    out["origins_used"] = {f: [float(placed_origins.loc[f, f"origin_{x}"]) for x in AXES] for f in fovs}
    _write_json(a["summary"], out)


def _step_register(a):
    """CU0: per FOV ``FOV.prepare_morphology(MorphologyConfig())`` on the raw dataset (MR1: one translation on the
    shared ``DAPI``); the registered round's ``DAPI`` and ``Flamingo`` written as float32 ZYX files and each FOV's
    recovered displacement."""
    from starfinder.dataset import MorphologyConfig
    base = Path(a["base"])
    summary = _read_json(base / "generate.json")
    record, dataset = _raw_dataset(base / "raw", base / "work")
    root = Path(dataset.output_root)
    channels = [c["name"] for c in record["other_channel_order"][MORPHOLOGY]]
    out = {}
    for fov in summary["fov_ids"]:
        prepared = dataset.fov(fov).prepare_morphology(MorphologyConfig())
        image = prepared.images[MORPHOLOGY]
        for c, name in enumerate(channels):
            _write_zyx(root / "images" / f"registered_{name}" / f"{fov}.tif", np.asarray(image[..., c], np.float32))
        transforms = prepared.registration_chains[MORPHOLOGY].transforms
        out[fov] = dict(displacement_zyx=[float(v) for v in transforms[0].displacement_zyx],
                        transforms=len(transforms))
    _write_json(a["summary"], dict(channels=channels, fovs=out))


def _model_path(name):
    entry = MODELS[name]
    folder = Path(os.environ[entry["variable"]]) / entry["folder"]
    return folder / entry["path"] if entry["path"] else folder


def _step_spleen(a):
    """CD: per FOV StarDist ``3D_spleen`` (by path, scale 1.0) on the reference stain ``DAPI``, role ``nuclear``."""
    from starfinder.segmentation import SegmentationInput, StarDistConfig, segment
    base = Path(a["base"])
    summary = _read_json(base / "generate.json")
    root = _work_root(base, summary)
    config = StarDistConfig(model_path=str(_model_path("3D_spleen")), scale=1.0)
    out = {}
    for fov in summary["fov_ids"]:
        grid = _grid(root, fov)
        stain = _image(root, "DAPI", fov)
        result = segment(SegmentationInput(stain[..., np.newaxis], grid, ("nuclear",), ({},)), config=config,
                         target="nucleus", device="cpu",
                         label_namespace=_ns(summary["dataset_id"], summary["sample"], fov, None, "nucleus_spleen"))
        labels = np.asarray(result.labels, dtype=np.uint32)
        _write_zyx(base / "labels" / fov / "nuclei.tif", labels)
        out[fov] = dict(n_labels=int(len(np.unique(labels[labels > 0]))),
                        device=result.record["methods"][0]["execution"]["device"])
    _write_json(a["summary"], out)


def _extension(target):
    from starfinder.segmentation import ZExtensionConfig
    px = EXTENSION_PX[target]
    return ZExtensionConfig(median_um=px["median"] * W382_PIXEL_UM, threshold="otsu",
                            min_area_um2=px["min_area"] * W382_PIXEL_UM ** 2, dilation_um=px["dilation"] * W382_PIXEL_UM,
                            fill_holes=px["fill_holes"])


def _step_cellpose(a):
    """CU: per FOV Cellpose ``cpsam_v2`` (by path) on the registered round's projections: ``nucleus`` on the ``DAPI``
    Z maximum, ``cell`` on the ``Flamingo`` and ``DAPI`` Z maxima (cytoplasm, nuclear), each extended through z with
    W-382's culture setting on its 3D stain (``DAPI`` for the nuclei, ``Flamingo`` for the cells)."""
    from starfinder.segmentation import CellposeConfig, SegmentationInput, extend_labels_through_z, segment
    base, target = Path(a["base"]), a["target"]
    summary = _read_json(base / "generate.json")
    root = _work_root(base, summary)
    spacing = summary["spacing_zyx"]
    diameter = round(CELLPOSE_DIAMETERS_PX[target] * W382_PIXEL_UM / spacing[1], 1)
    config = CellposeConfig(diameter=diameter, model_path=str(_model_path("cpsam_v2")))
    out = dict(diameter_px=diameter, fovs={})
    for fov in summary["fov_ids"]:
        grid = _grid(root, fov)
        dapi = _image(root, "registered_DAPI", fov)
        if target == "cell":
            flamingo = _image(root, "registered_Flamingo", fov)
            image = np.stack([flamingo.max(axis=0), dapi.max(axis=0)], axis=-1)[np.newaxis]
            roles, stain = ("cytoplasm", "nuclear"), flamingo
        else:
            image = dapi.max(axis=0)[np.newaxis, ..., np.newaxis]
            roles, stain = ("nuclear",), dapi
        result = segment(SegmentationInput(np.ascontiguousarray(image), grid.projected(method="max"), roles,
                                           tuple({} for _ in roles)), config=config, target=target, device="cpu",
                         label_namespace=_ns(summary["dataset_id"], summary["sample"], fov, None,
                                             f"{target}_cellpose_2d"))
        labels, record = extend_labels_through_z(result.labels, stain, grid.metadata, config=_extension(target))
        labels = np.asarray(labels, dtype=np.uint32)
        _write_zyx(base / "labels" / fov / f"{'nuclei' if target == 'nucleus' else 'cells'}.tif", labels)
        plane = np.asarray(result.labels)
        out["fovs"][fov] = dict(n_labels_2d=int(len(np.unique(plane[plane > 0]))),
                                n_labels=int(len(np.unique(labels[labels > 0]))), pixels=record.get("pixels"))
    _write_json(a["summary"], out)


def _fusion_images(context, stain):
    names = (stain, "composite") if context == "tissue2d" else (stain,)
    return tuple(f"{name}=file:images/{name}/{{fov_id}}.tif" for name in names)


def _fused_projection(root, sample, fovs, context, stain):
    from starfinder.export import FusionConfig, fused_grid
    from starfinder.stitching import load_placement
    placement = load_placement(Path(root) / "stitching" / sample)
    fusion = FusionConfig(images=_fusion_images(context, stain), dimensionality="projection", projection="max",
                          scale_yx=1.0)
    grid = fused_grid(placement, {f: _grid(root, f) for f in fovs}, config=fusion)
    return placement, fusion, grid


def _step_truth_grid(a):
    """CT (V3): the truth nuclei and cells on the segmentation grid of the route after fusion, by the appearance point
    evaluator at the fused projection's pixel centres, moved into the truth frame by the mean origin difference of the
    placement used, at the Z of every plane of the sample grid, the Z maximum of the IDs (PT1); written as uint32
    OME-TIFFs with the grid's metadata, for ``truth_runs``."""
    from starfinder.io._ome import write_ome_tiff
    from starfinder.synthetic import appearance_scene_preset, load_appearance_bank
    from starfinder.synthetic._appearance import _point_evaluator
    from .cellular_appearance_measures import load_saved_scene
    base = Path(a["base"])
    summary = _read_json(base / "generate.json")
    root = _work_root(base, summary)
    placement, _, grid = _fused_projection(root, summary["sample"], summary["fov_ids"], "tissue2d", a["stain"])
    spacing, origin, shape = grid.level("projection")
    metadata = grid.level_metadata("projection")
    truth = pd.read_csv(base / "scene" / "placement_truth.csv", float_precision="round_trip").set_index("fov_id")
    used = placement.fovs.set_index("fov_id")
    offset = np.mean([[float(truth.loc[f, f"true_origin_{x}"]) - float(used.loc[f, f"origin_{x}"]) for x in AXES]
                      for f in summary["fov_ids"]], axis=0)
    scene = load_saved_scene(base / "scene", reference_round=False)
    evaluator = _point_evaluator(scene, load_appearance_bank(appearance_scene_preset(summary["preset"])[1].bank))
    sample_grid = summary["sample_grid"]
    planes = sample_grid["origin_zyx"][0] + np.arange(sample_grid["shape_zyx"][0]) * sample_grid["spacing_zyx"][0]
    y = origin[1] + np.arange(shape[1]) * spacing[1] + offset[1]
    x = origin[2] + np.arange(shape[2]) * spacing[2] + offset[2]
    yy, xx = np.meshgrid(y, x, indexing="ij")
    cells = np.zeros(shape[1:], dtype=np.uint32)
    nuclei = np.zeros(shape[1:], dtype=np.uint32)
    for z in planes:
        points = np.stack([np.full(yy.shape, z), yy, xx], axis=-1)
        C, N = evaluator.labels(points)
        cells = np.maximum(cells, np.asarray(C, dtype=np.uint32).reshape(yy.shape))
        nuclei = np.maximum(nuclei, np.asarray(N, dtype=np.uint32).reshape(yy.shape))
    out = base / "truth_grid"
    out.mkdir(parents=True, exist_ok=True)
    for name, labels in (("cell", cells), ("nucleus", nuclei)):
        write_ome_tiff(out / f"{name}.ome.tif", labels, metadata=metadata, annotation={"truth": name})
    _write_json(a["summary"], dict(shape_zyx=list(shape), origin_zyx=list(origin), spacing_zyx=list(spacing),
                                   offset_um=offset.tolist(), planes=len(planes),
                                   nuclei=int(len(np.unique(nuclei[nuclei > 0]))),
                                   cells=int(len(np.unique(cells[cells > 0])))))


def _copy_shared(base, summary, variant):
    """The variant's output root: the shared images, molecule tables and the placement used, copied."""
    work, root = _work_root(base, summary), _variant_root(base, summary, variant)
    for part in ("images", "signal", Path("stitching") / summary["sample"]):
        shutil.copytree(work / part, root / part, dirs_exist_ok=True)
    return root


def _step_assign_export(a):
    """The assignment and the export of one chain. Route per FOV: per FOV the molecules (``import_legacy_molecules_csv``
    of the truth-position CSV), the labels (the method's, or the FOV's truth labels, imported ``relabel=False``),
    ``assign_molecules`` (dense 3D method: the 0.7776-µm planar expansion; otherwise the default) saved as the §2.9
    folder ``default``, then ``Dataset.export`` (``route: per_fov``, ``dimensionality: stack``). Route after fusion:
    ``Dataset.export`` (``dimensionality: projection``, Z maximum, ``scale_yx`` 1) with the method's segmentation
    block or the truth runs."""
    from starfinder.assignment import AssignmentConfig, assign_molecules, import_legacy_molecules_csv
    from starfinder.assignment._persist import assignment_directory, write_assignment
    from starfinder.dataset import Dataset
    from starfinder.export import ExportConfig, FusionConfig, SampleSegmentationConfig, SampleSegmentationRun
    from starfinder.segmentation import ExpandLabelsConfig, SeededWatershedConfig, StarDistConfig, import_labels
    from .sample_segmentation_fixtures import truth_runs
    base, variant, context = Path(a["base"]), a["variant"], a["context"]
    summary = _read_json(base / "generate.json")
    ds, sample, fovs = summary["dataset_id"], summary["sample"], summary["fov_ids"]
    book = _codebook(summary["preset"])
    root = _copy_shared(base, summary, variant)
    dataset = Dataset(base / "raw", base / variant, ds, sample, "output", codebook=book)
    assert Path(dataset.output_root) == root
    out = dict(variant=variant)
    if CONTEXTS[context]["route"] == "per_fov":
        for fov in fovs:
            grid = _grid(root, fov)
            molecules = import_legacy_molecules_csv(root / "signal" / f"{fov}_goodSpots.csv",
                                                spot_namespace=_ns(ds, sample, fov, None), genes=tuple(book.genes))
            if variant == "truth":
                labels = {target: import_labels(base / "scene" / "fovs" / fov / "labels" / file, grid=grid,
                                                target=target, relabel=False,
                                                label_namespace=_ns(ds, sample, fov, None, f"truth_{target}"))
                          for target, file in (("cell", "cells.tif"), ("nucleus", "nuclei.tif"))}
                config = AssignmentConfig()
            elif context == "dense3d":
                nuclei = import_labels(base / "labels" / fov / "nuclei.tif", grid=grid, target="nucleus",
                                       relabel=False, label_namespace=_ns(ds, sample, fov, None, "nucleus_spleen"))
                labels = {"cell": nuclei, "nucleus": nuclei}
                config = AssignmentConfig(expansion=ExpandLabelsConfig(LN_EXPANSION_UM, "um", "planar"))
            else:
                labels = {target: import_labels(base / "labels" / fov / file, grid=grid, target=target, relabel=False,
                                                label_namespace=_ns(ds, sample, fov, None, f"{target}_cellpose"))
                          for target, file in (("cell", "cells.tif"), ("nucleus", "nuclei.tif"))}
                config = AssignmentConfig()
            result = assign_molecules(molecules, labels["cell"], grid=grid, nuclei=labels["nucleus"], config=config)
            result = replace(result, record=dict(result.record, name="default"))
            write_assignment(assignment_directory(root / "checkpoints" / fov, "default"), result,
                             metadata=grid.metadata)
        export = ExportConfig(route="per_fov", fusion=FusionConfig(images=_fusion_images(context, a["stain"]),
                                                                    dimensionality="stack", scale_yx=1.0))
    else:
        if variant == "truth":
            segmentation = truth_runs(base / "truth_grid" / "cell.ome.tif", base / "truth_grid" / "nucleus.ome.tif",
                                      device="cpu")
        else:
            scale = round(summary["spacing_zyx"][1] / STARDIST_2D_PIXEL_UM, 6)
            out["stardist_scale"] = scale
            segmentation = SampleSegmentationConfig(
                runs=(SampleSegmentationRun("nucleus", "nucleus", (("nuclear", a["stain"]),),
                                            StarDistConfig(scale=scale, model_path=str(_model_path(
                                                "2D_versatile_fluo")))),
                      SampleSegmentationRun("cell", "cell", (("composite", "composite"),), SeededWatershedConfig(),
                                            seeds="nucleus")),
                cells="cell", nuclei="nucleus", device="cpu")
        export = ExportConfig(route="after_fusion", segmentation=segmentation,
                              fusion=FusionConfig(images=_fusion_images(context, a["stain"]),
                                                  dimensionality="projection", projection="max", scale_yx=1.0))
    record = dataset.export(sample, placement=root / "stitching" / sample, output_dir=root / "export" / sample,
                            config=export)
    out["complete"] = bool(getattr(record, "complete", True))
    _write_json(a["summary"], out)


def _step_report(a):
    """``Dataset.report`` on the export (no run records: A2), with the marker groups of the four types and the
    exclusive and co-expressed pairs (C8)."""
    from starfinder.dataset import Dataset
    from starfinder.report import MarkerGroup, ReportConfig
    base, variant = Path(a["base"]), a["variant"]
    summary = _read_json(base / "generate.json")
    dataset = Dataset(base / "raw", base / variant, summary["dataset_id"], summary["sample"], "output")
    config = ReportConfig(markers=tuple(MarkerGroup(name, tuple(genes)) for name, genes in a["markers"]),
                          marker_pairs=tuple(tuple(p) for p in a["pairs"]))
    result = dataset.report(summary["sample"], summary["fov_ids"], config=config)
    _write_json(a["summary"], dict(path=str(result.path)))


STEPS = {"generate": _step_generate, "stitch": _step_stitch, "register": _step_register, "spleen": _step_spleen,
         "cellpose": _step_cellpose, "truth_grid": _step_truth_grid, "assign_export": _step_assign_export,
         "report": _step_report}
BOOT = f"import json, sys; from {__name__} import _main; _main(sys.argv[1], json.loads(sys.argv[2]))"


def _main(step, arguments):
    STEPS[step](arguments)


# ============================================================================================ running the steps

def _seconds(clock):
    seconds = 0.0
    for part in clock.split(":"):
        seconds = 60 * seconds + float(part)
    return seconds


def _output_dir(base):
    path = Path(os.environ[OUTPUT]) if os.environ.get(OUTPUT) else Path(base) / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class Runner:
    """Runs each step in its own process under /usr/bin/time -v and keeps one row per step."""

    base: Path
    rows: list = field(default_factory=list)

    @property
    def output(self):
        return _output_dir(self.base)

    def run(self, tag, step, **arguments):
        env = dict(os.environ, **{k: "1" for k in THREADS}, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                   MPLBACKEND="Agg")
        env["PYTHONPATH"] = os.pathsep.join([str(PACKAGE_ROOT), *([env["PYTHONPATH"]] if env.get("PYTHONPATH")
                                                                  else [])])
        logs = self.output / "steps"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{tag}.time.log"
        command = [sys.executable, "-c", BOOT, step, json.dumps(arguments)]
        if os.environ.get(STEP_WRAPPER):
            command = [*os.environ[STEP_WRAPPER].split(), tag, *command]
        done = subprocess.run([str(TIME), "-v", "-o", str(log), *command], env=env, cwd=PACKAGE_ROOT,
                              capture_output=True, text=True)
        (logs / f"{tag}.output.log").write_text(done.stdout + done.stderr, encoding="utf-8")
        text = log.read_text(encoding="utf-8")
        row = dict(tag=tag, step=step, exit=done.returncode,
                   wall_s=_seconds(re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", text)
                                   .group(1)),
                   cpu_s=float(re.search(r"User time \(seconds\): (\S+)", text).group(1))
                   + float(re.search(r"System time \(seconds\): (\S+)", text).group(1)),
                   peak_rss_kib=int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", text).group(1)))
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.output / "steps.csv", index=False)
        print(f"\nSTEP {tag}: exit {row['exit']}, wall {row['wall_s']:.2f} s, cpu {row['cpu_s']:.2f} s, "
              f"peak RSS {row['peak_rss_kib']} KiB", flush=True)
        assert done.returncode == 0, f"step {tag} exited {done.returncode}:\n{(done.stdout + done.stderr)[-6000:]}"
        return _read_json(arguments["summary"])


# ========================================================================================= skips and checks

def require(*modules):
    for module in modules:
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"the chain needs the {module} extra, which is not installed")


_CHECKED = {}


def checked_model(name):
    """The model's path; the test skips when its variable is unset or a file is missing, and fails when a file's
    SHA-256 differs from W-392's inventory."""
    entry = MODELS[name]
    if not os.environ.get(entry["variable"]):
        pytest.skip(f"model {name} is found through {entry['variable']}, which is not set")
    folder = Path(os.environ[entry["variable"]]) / entry["folder"]
    for file in entry["files"]:
        if not (folder / file).is_file():
            pytest.skip(f"model {name}: no file {folder / file} ({entry['variable']}); nothing is downloaded")
    if name not in _CHECKED:
        _CHECKED[name] = {file: _sha256(folder / file) for file in entry["files"]}
        print(f"\nMODEL {name}: {folder} " + ", ".join(f"{f} {d}" for f, d in _CHECKED[name].items()))
    assert _CHECKED[name] == entry["files"], f"model {name}: SHA-256 differs from W-392's inventory/models.json"
    return _model_path(name)


def checked_bank(context):
    """The real bank of a context's slow preset through STARFINDER_APPEARANCE_BANKS, its SHA-256 the one the preset
    names, or a skip naming the variable or the file."""
    from starfinder.synthetic import APPEARANCE_PRESETS
    from starfinder.synthetic._appearance_bank import BANK_DIRECTORY_VARIABLE
    from .cellular_appearance_scenes import real_bank_missing
    name = CONTEXTS[context]["bank"]
    reason = real_bank_missing(name)
    if reason is not None:
        pytest.skip(reason)
    path = Path(os.environ[BANK_DIRECTORY_VARIABLE]) / f"{name}.npz"
    key = f"bank:{name}"
    if key not in _CHECKED:
        _CHECKED[key] = _sha256(path)
        print(f"\nBANK {name}: {path} {_CHECKED[key]}")
    assert _CHECKED[key] == APPEARANCE_PRESETS[CONTEXTS[context]["slow"]]["bank_sha256"], name


# ============================================================================================= the chains

@dataclass
class Scene:
    """One generated scene, its shared steps' summaries and its truth tables (read lazily)."""

    context: str
    tier: str
    seed: int
    base: Path
    summary: dict
    stitch: dict = None
    register: dict = None
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def label(self):
        return f"{self.context}-{self.tier}-seed{self.seed}"

    @property
    def fovs(self):
        return self.summary["fov_ids"]

    @property
    def scene_dir(self):
        return self.base / "scene"

    def table(self, name):
        if name not in self._cache:
            self._cache[name] = pd.read_csv(self.scene_dir / f"{name}.csv", float_precision="round_trip")
        return self._cache[name]

    def fov_labels(self, fov, target):
        key = (fov, target)
        if key not in self._cache:
            file = "nuclei.tif" if target == "nucleus" else "cells.tif"
            labels = np.asarray(tifffile.imread(self.scene_dir / "fovs" / fov / "labels" / file), dtype=np.uint32)
            self._cache[key] = labels.reshape(self.summary["fov_shape_zyx"])
        return self._cache[key]

    def markers(self):
        """The marker groups of the four types: the genes with weight in one type only (G1)."""
        abundances = self.table("type_abundances")
        genes = self.summary["genes"]
        weights = {}
        for row in abundances.itertuples(index=False):
            weights.setdefault(row.type, {})[row.gene_id] = row.weight
        groups = []
        for name, row in weights.items():
            if name == "ambient":
                continue
            own = [g for g in genes if row.get(g, 0) > 0 and sum(w.get(g, 0) > 0 for t, w in weights.items()
                                                                  if t != "ambient") == 1]
            groups.append((name, own))
        return groups


@dataclass
class Chain:
    scene: Scene
    variant: str
    export: dict
    report: dict
    segmentation: dict = None

    @property
    def root(self):
        return _variant_root(self.scene.base, self.scene.summary, self.variant)

    @property
    def sample(self):
        return self.scene.summary["sample"]

    @property
    def export_dir(self):
        return self.root / "export" / self.sample

    @property
    def report_dir(self):
        return self.root / "report" / self.sample

    @property
    def fov_ids(self):
        return self.scene.fovs

    @property
    def dataset_id(self):
        return self.scene.summary["dataset_id"]

    def run_record(self, fov):   # the report's checks ask; the chains have no run records (A2)
        raise AssertionError(f"no run record for {fov}")


class Chains:
    """The module's chains, each stage built on first use and kept; a stage that failed fails again at once, so that
    no step runs twice."""

    def __init__(self, base):
        self.base = Path(base)
        self.runner = Runner(self.base)
        self.scenes, self.segmented, self.chains, self.labels, self.failed = {}, {}, {}, {}, {}

    def _once(self, kind, key, build):
        cache = getattr(self, kind)
        if key in cache:
            return cache[key]
        if (kind, key) in self.failed:
            pytest.fail(f"{kind} {key} failed earlier in this module: {self.failed[(kind, key)]}")
        try:
            cache[key] = build()
        except (AssertionError, Exception) as error:
            self.failed[(kind, key)] = f"{type(error).__name__}: {str(error)[:2000]}"
            raise
        return cache[key]

    @staticmethod
    def require_method(context):
        require("anndata", "multiview_stitcher", *CONTEXTS[context]["extras"])
        for model in CONTEXTS[context]["models"]:
            checked_model(model)

    def scene(self, context, tier, seed):
        """The generated scene, its stitch and the placement used (and in 2D tissue the truth on the grid)."""
        require("anndata", "multiview_stitcher")
        if not TIME.is_file():
            pytest.skip(f"no file {TIME}: every chain step is measured under /usr/bin/time -v")
        if tier == "slow":
            checked_bank(context)

        def build():
            spec = CONTEXTS[context]
            base = self.base / f"{context}-{tier}-seed{seed}"
            tag = f"{context}-{tier}-seed{seed}"
            summary = self.runner.run(f"{tag}.generate", "generate", base=str(base), preset=spec[tier], seed=seed,
                                      context=context, summary=str(base / "generate.json"))
            scene = Scene(context, tier, seed, base, summary)
            scene.stitch = self.runner.run(f"{tag}.stitch", "stitch", base=str(base), context=context,
                                           stain=spec["stain"], summary=str(base / "stitch.json"))
            if context == "tissue2d":
                self.runner.run(f"{tag}.truth_grid", "truth_grid", base=str(base), stain=spec["stain"],
                                summary=str(base / "truth_grid.json"))
            return scene
        return self._once("scenes", (context, tier, seed), build)

    def method_labels(self, context, tier, seed):
        """The scene with the method's labels of the route per FOV: dense 3D ``3D_spleen``; culture the registration
        and the two Cellpose calls (2D tissue segments inside its export)."""
        self.require_method(context)
        scene = self.scene(context, tier, seed)

        def build():
            base, tag = scene.base, f"{scene.label}.method"
            if context == "dense3d":
                return self.runner.run(f"{tag}.spleen", "spleen", base=str(base), summary=str(base / "spleen.json"))
            if context == "culture":
                scene.register = self.runner.run(f"{scene.label}.register", "register", base=str(base),
                                                 summary=str(base / "register.json"))
                return {target: self.runner.run(f"{tag}.cellpose_{target}", "cellpose", base=str(base),
                                                target=target, summary=str(base / f"cellpose_{target}.json"))
                        for target in ("nucleus", "cell")}
            return None
        self._once("segmented", (context, tier, seed), build)
        return scene

    def chain(self, context, tier, seed, variant):
        """The chain's assignment, export and report (the method's labels first for ``method``)."""
        if variant == "method":
            scene = self.method_labels(context, tier, seed)
        else:
            scene = self.scene(context, tier, seed)

        def build():
            spec, base, tag = CONTEXTS[context], scene.base, f"{scene.label}.{variant}"
            export = self.runner.run(f"{tag}.assign_export", "assign_export", base=str(base), variant=variant,
                                     context=context, stain=spec["stain"],
                                     summary=str(base / f"assign_export_{variant}.json"))
            report = self.runner.run(f"{tag}.report", "report", base=str(base), variant=variant,
                                     markers=scene.markers(), pairs=REPORT_PAIRS,
                                     summary=str(base / f"report_{variant}.json"))
            return Chain(scene, variant, export, report, self.segmented.get((context, tier, seed)))
        return self._once("chains", (context, tier, seed, variant), build)

    def all(self, context, tier, variant):
        return [self.chain(context, tier, seed, variant) for seed in CONTEXTS[context]["seeds"][tier]]

    def all_labelled(self, context, tier):
        return [self.method_labels(context, tier, seed) for seed in CONTEXTS[context]["seeds"][tier]]


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    return Chains(tmp_path_factory.mktemp("chains"))


def write_table(chains, name, rows, columns=None):
    path = chains.runner.output / f"{name}.csv"
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False)
    print(f"TABLE {path} ({len(frame)} rows)")
    return frame


# ================================================================== step 1: nuclei against truth (W-393 §2)

def match_nuclei(truth, pred):
    """W-393's definitions (notes.md §2) on L12's matching (``l12_agreement``: one to one at IoU ≥ 0.5, the larger
    IoU first): a predicted label is owned by the truth nucleus holding at least half of its voxels; a split is a
    truth nucleus owning at least two predicted labels; a merge is a predicted label holding at least half of each of
    at least two truth nuclei; missed is a truth nucleus not matched and in no split or merge; a false positive is a
    predicted label not matched and owned by no truth nucleus. Each truth nucleus is counted once by its ID with all
    its voxels (V4)."""
    t, p = np.asarray(truth).ravel(), np.asarray(pred).ravel()
    tid, tsz = np.unique(t[t > 0], return_counts=True)
    pid, psz = np.unique(p[p > 0], return_counts=True)
    ts, ps = dict(zip(tid.tolist(), tsz.tolist())), dict(zip(pid.tolist(), psz.tolist()))
    both = (t > 0) & (p > 0)
    pairs, overlaps = (np.unique(np.stack([t[both], p[both]]), axis=1, return_counts=True) if both.any()
                       else (np.zeros((2, 0), np.int64), np.zeros(0, np.int64)))
    candidates, owner, held = [], {}, {}
    for (i, j), o in zip(pairs.T.tolist(), overlaps.tolist()):
        iou = o / (ts[i] + ps[j] - o)
        if iou >= 0.5:
            candidates.append((-iou, i, j))
        if o >= 0.5 * ps[j]:
            owner[j] = i
        if o >= 0.5 * ts[i]:
            held.setdefault(j, []).append(i)
    used_t, used_p, matched = set(), set(), {}
    for negative, i, j in sorted(candidates):
        if i not in used_t and j not in used_p:
            used_t.add(i)
            used_p.add(j)
            matched[j] = (i, -negative)
    owned = {}
    for j, i in owner.items():
        owned.setdefault(i, []).append(j)
    split = {i for i, js in owned.items() if len(js) >= 2}
    merges = {j for j, ii in held.items() if len(ii) >= 2}
    merged = {i for j in merges for i in held[j]}
    missed = [i for i in ts if i not in used_t and i not in split and i not in merged]
    false_positive = [j for j in ps if j not in used_p and j not in owner]
    # The matching is L12's: its matched fraction of the truth labels is this one.
    n_truth, _, fraction, _ = l12_agreement(truth, pred)
    assert n_truth == len(ts) and abs(fraction * n_truth - len(matched)) <= TOL * max(1, n_truth)
    return dict(truth=ts, pred=ps, matched=matched, split=split, merges=merges, merged=merged, missed=missed,
                false_positive=false_positive, held=held, owned=owned)


def border_flags(labels):
    """Per label: (cut by the Y or X border, cut by a Z end) of the image compared."""
    out = {}
    for index, slices in enumerate(ndimage.find_objects(np.asarray(labels, dtype=np.int64)), start=1):
        if slices is None:
            continue
        xy = any(s.start == 0 or s.stop == n for s, n in zip(slices[-2:], labels.shape[-2:]))
        z = labels.shape[0] > 1 and (slices[0].start == 0 or slices[0].stop == labels.shape[0])
        out[index] = (bool(xy), bool(z))
    return out


def pieces(labels):
    """The labels of more than one connected piece (26-connected in 3D, 8-connected in a plane)."""
    labels = np.asarray(labels)
    structure = ndimage.generate_binary_structure(labels.ndim, labels.ndim)
    out = []
    for index, slices in enumerate(ndimage.find_objects(labels.astype(np.int64)), start=1):
        if slices is not None and ndimage.label(labels[slices] == index, structure=structure)[1] > 1:
            out.append(index)
    return out


COUNTS = ("n_truth", "n_interior", "matched", "matched_interior", "missed", "missed_interior", "missed_at_xy_border",
          "merged_truth", "merges", "split", "false_positive", "n_pred", "at_z_end", "multi_piece")


def nuclei_counts(label, fov, truth, pred, rows):
    """The counts of step 1 on one image and its per-object rows (appended to ``rows``)."""
    m = match_nuclei(truth, pred)
    flags = border_flags(truth)
    multi = pieces(truth)
    interior = {i for i in m["truth"] if not flags[i][0]}
    matched_truth = {i: (j, iou) for j, (i, iou) in m["matched"].items()}
    counts = dict(n_truth=len(m["truth"]), n_interior=len(interior), matched=len(m["matched"]),
                  matched_interior=sum(i in interior for i in matched_truth), missed=len(m["missed"]),
                  missed_interior=sum(i in interior for i in m["missed"]),
                  missed_at_xy_border=sum(i not in interior for i in m["missed"]), merged_truth=len(m["merged"]),
                  merges=len(m["merges"]), split=len(m["split"]), false_positive=len(m["false_positive"]),
                  n_pred=len(m["pred"]), at_z_end=sum(flags[i][1] for i in m["truth"]), multi_piece=len(multi))
    centres = dict(truth=ndimage.center_of_mass(truth > 0, truth, list(m["truth"])) if m["truth"] else [],
                   pred=ndimage.center_of_mass(pred > 0, pred, list(m["pred"])) if m["pred"] else [])
    centre = {kind: dict(zip(list(m[kind]), centres[kind])) for kind in ("truth", "pred")}
    for i in m["truth"]:
        kind = ("matched" if i in matched_truth else "split" if i in m["split"] else "merged" if i in m["merged"]
                else "missed")
        related = ([matched_truth[i][0]] if kind == "matched" else m["owned"].get(i, []) if kind == "split"
                   else [j for j in m["merges"] if i in m["held"][j]] if kind == "merged" else [])
        rows.append(dict(scene=label, fov=fov, object="truth", id=int(i), kind=kind, related=json.dumps(related),
                         iou=matched_truth[i][1] if kind == "matched" else None, n_voxels=m["truth"][i],
                         interior=i in interior, at_xy_border=flags[i][0], at_z_end=flags[i][1],
                         multi_piece=i in multi, **dict(zip(("z", "y", "x"), centre["truth"][i][-3:]))
                         if truth.ndim == 3 else dict(z=0.0, y=centre["truth"][i][0], x=centre["truth"][i][1])))
    for j in m["pred"]:
        kind = "merge" if j in m["merges"] else "false_positive" if j in m["false_positive"] else (
            "matched" if j in m["matched"] else "owned")
        related = m["held"].get(j, []) if kind == "merge" else [m["matched"][j][0]] if kind == "matched" else []
        point = centre["pred"][j]
        rows.append(dict(scene=label, fov=fov, object="pred", id=int(j), kind=kind, related=json.dumps(related),
                         iou=m["matched"][j][1] if j in m["matched"] else None, n_voxels=m["pred"][j],
                         interior=None, at_xy_border=None, at_z_end=None, multi_piece=None,
                         **(dict(z=point[0], y=point[1], x=point[2]) if pred.ndim == 3
                            else dict(z=0.0, y=point[0], x=point[1]))))
    return counts, {j: i for j, (i, _) in m["matched"].items()}


def pooled(rows):
    return {k: int(sum(r[k] for r in rows)) for k in COUNTS}


def step1_fractions(counts):
    n = counts["n_truth"]
    return dict(matched_fraction=counts["matched_interior"] / counts["n_interior"] if counts["n_interior"] else None,
                merged_share=counts["merged_truth"] / n if n else None,
                false_positive_per_truth=counts["false_positive"] / n if n else None)


def assert_step1(row, counts, label):
    """TH1, TH2 and TH3 on a slow preset's counts (TH-FP reported). A row of ``AMENDED`` is printed with its value,
    its threshold, its outcome and the amendment, as TH-FP is reported, and not asserted; every other row is asserted.
    Returns the printed lines of the amended rows."""
    f = step1_fractions(counts)
    rows = [("TH1", f["matched_fraction"] is not None and f["matched_fraction"] >= TH1_MATCHED,
             f"matched fraction {f['matched_fraction']} of {counts['n_interior']} interior truth nuclei "
             f"({counts['matched_interior']} matched), threshold at least {TH1_MATCHED}"),
            ("TH2", f["merged_share"] is not None and f["merged_share"] <= TH2_MERGED,
             f"merged truth nuclei {counts['merged_truth']} of {counts['n_truth']} = {f['merged_share']}, threshold at "
             f"most {TH2_MERGED}"),
            ("TH3", counts["split"] == TH3_SPLITS, f"{counts['split']} splits, threshold {TH3_SPLITS}")]
    reported, misses = [], []
    for name, held, text in rows:
        if (row, name) in AMENDED:
            reported.append(f"{row} {label}: {name}: {text}: {'within' if held else 'outside'} the threshold; "
                            f"{AMENDMENT}")
            print(reported[-1])
        elif not held:
            misses.append(f"{name}: {text}")
    assert not misses, f"{row} {label}: " + "; ".join(misses)
    return reported


def report_step1(row, tier, per_scene, total):
    print(f"\n{row} {tier}: per scene " + "; ".join(f"{k}: {v} {step1_fractions(v)}" for k, v in per_scene.items()))
    print(f"{row} {tier}: {'pooled ' if len(per_scene) > 1 else ''}{total} {step1_fractions(total)} "
          f"(TH-FP false positives per truth nucleus reported)")


def per_fov_nuclei(scene, rows):
    """Step 1 on the route per FOV: each FOV's method labels against its truth nucleus labels; counts summed over the
    FOVs (a nucleus in an overlap counts in each FOV that holds it). Returns the counts and per FOV the match map."""
    counts, maps = [], {}
    for fov in scene.fovs:
        pred = np.asarray(tifffile.imread(scene.base / "labels" / fov / "nuclei.tif"), np.uint32).reshape(
            scene.summary["fov_shape_zyx"])
        c, maps[fov] = nuclei_counts(scene.label, fov, scene.fov_labels(fov, "nucleus"), pred, rows)
        counts.append(c)
    return pooled(counts), maps


def export_labels(chain, run):
    record = _read_json(chain.export_dir / "export.json")
    entry = next(r for r in record["segmentation"]["runs"] if r["name"] == run)
    labels = np.asarray(tifffile.imread(chain.export_dir / entry["path"]), np.uint32)
    return labels[np.newaxis] if labels.ndim == 2 else labels


def truth_on_grid(chain, target):
    path = chain.scene.base / "truth_grid" / f"{target}.ome.tif"
    labels = np.asarray(tifffile.imread(path), np.uint32)
    return labels[np.newaxis] if labels.ndim == 2 else labels


def fused_nuclei(chain, rows):
    """Step 1 on the route after fusion: the export's nucleus labels against P on the segmentation grid."""
    truth, pred = truth_on_grid(chain, "nucleus"), export_labels(chain, "nucleus")
    assert truth.shape == pred.shape, (truth.shape, pred.shape)
    counts, matches = nuclei_counts(chain.scene.label, "sample", truth[0], pred[0], rows)
    return counts, matches


# ===================================================================== step 2: the cell of each molecule

def assignment(chain, fov):
    from starfinder.assignment._persist import assignment_directory, read_assignment
    return read_assignment(assignment_directory(chain.root / "checkpoints" / fov, "default"), "default")


def agreement(frame):
    """W-393's molecule agreement over the rows of ``frame`` (columns ``assigned``: the cell of a molecule whose §2.9
    status is ``assigned``, else 0; ``named``: the truth cell that cell names, 0 for none, −1 for no truth cell;
    ``true``: its true cell, 0 for none)."""
    assigned = frame["assigned"].to_numpy() > 0
    agree = frame["named"].to_numpy() == frame["true"].to_numpy()
    with_cell = frame["true"].to_numpy() > 0
    return dict(n_molecules=int(len(frame)), n_with_true_cell=int(with_cell.sum()), n_assigned=int(assigned.sum()),
                agree_among_assigned_n=int(agree[assigned].sum()), agree_with_true_cell_n=int(agree[with_cell].sum()),
                unassigned_with_true_cell_n=int((~assigned & with_cell).sum()))


def agreement_fractions(c):
    return dict(agree_among_assigned=c["agree_among_assigned_n"] / c["n_assigned"] if c["n_assigned"] else None,
                agree_with_true_cell=c["agree_with_true_cell_n"] / c["n_with_true_cell"] if c["n_with_true_cell"]
                else None,
                unassigned_with_true_cell=c["unassigned_with_true_cell_n"] / c["n_with_true_cell"]
                if c["n_with_true_cell"] else None)


def _true_cells(scene):
    molecules = scene.table("molecules")
    return dict(zip(molecules.amplicon_id, molecules.true_cell_id.fillna(0).astype(np.int64)))


def _assigned(molecules):
    """The cell of each molecule whose §2.9 status is ``assigned``, else 0 (``unassigned``, ``outside_grid``, and
    ``excluded_cell``: a molecule of a cell excluded for having no matched nucleus, which no output counts)."""
    cells = molecules.cell_id.astype("float64").fillna(0).to_numpy()
    status = molecules.assignment_status.astype(object).eq("assigned").fillna(False).to_numpy(bool)
    return np.where(status, cells, 0).astype(np.int64)


def per_fov_molecules(chain, maps):
    """Step 2 on the route per FOV: per FOV the rows of molecule_fovs.csv with in_footprint, each with its assigned
    cell named through its nucleus's match (dense 3D: the cell keeps the seed nucleus's label; culture: the nuclei
    the assignment matched to the cell, which must name one truth cell), against true_cell_id."""
    scene = chain.scene
    n2c = dict(zip(scene.table("nuclei").nucleus_id.astype(np.int64), scene.table("nuclei").cell_id.astype(np.int64)))
    true_cell = _true_cells(scene)
    views = scene.table("molecule_fovs")
    frames = []
    for fov in scene.fovs:
        result = assignment(chain, fov)
        molecules = result.molecules.copy()
        molecules["csv_row"] = molecules.spot_id.str.slice(4).astype(int)
        view = views[views.fov_id == fov].set_index("csv_row")
        molecules = molecules.join(view[["amplicon_id", "in_footprint"]], on="csv_row")
        molecules = molecules[molecules.in_footprint.astype(bool)].copy()
        pmap = maps[fov]
        if scene.context == "culture":
            nuclei = result.nuclei
            matched = nuclei[nuclei.cell_id.notna().to_numpy(bool)]
            named = {}
            for cell, group in matched.groupby(matched.cell_id.astype("int64")):
                cells = {n2c.get(pmap.get(int(n), -1), -1) for n in group.nucleus_id.astype(np.int64)} - {-1}
                named[int(cell)] = cells.pop() if len(cells) == 1 else -1
        else:
            named = {int(j): n2c.get(i, -1) for j, i in pmap.items()}
        molecules["assigned"] = _assigned(molecules)
        molecules["named"] = [named.get(c, -1) if c > 0 else 0 for c in molecules["assigned"]]
        molecules["true"] = molecules.amplicon_id.map(true_cell).astype(np.int64)
        molecules["fov"] = fov
        frames.append(molecules)
    return pd.concat(frames, ignore_index=True)


def fused_molecules(chain, matches):
    """Step 2 on the route after fusion: the export's retained molecules, each cell (the seeded watershed's, which
    keeps its seed nucleus's label) named through the nucleus match, against true_cell_id."""
    scene = chain.scene
    n2c = dict(zip(scene.table("nuclei").nucleus_id.astype(np.int64), scene.table("nuclei").cell_id.astype(np.int64)))
    molecules = export_table(chain, "molecules")
    molecules = molecules[molecules.assembly_status.eq("retained").to_numpy(bool)].copy()
    molecules["csv_row"] = molecules.spot_id.str.slice(4).astype(int)
    views = scene.table("molecule_fovs").set_index(["fov_id", "csv_row"]).amplicon_id
    molecules["amplicon_id"] = [views.get((f, r)) for f, r in zip(molecules.fov_id, molecules.csv_row)]
    molecules["assigned"] = _assigned(molecules)
    molecules["named"] = [n2c.get(matches.get(c, -1), -1) if c > 0 else 0 for c in molecules["assigned"]]
    molecules["true"] = molecules.amplicon_id.map(_true_cells(scene)).astype(np.int64)
    molecules["fov"] = molecules.fov_id
    return molecules


def wrong_cell_rows(label, frame):
    wrong = frame[frame["named"] != frame["true"]]
    return [dict(scene=label, fov=r.fov, amplicon_id=r.amplicon_id, spot_id=r.spot_id, gene_id=r.gene_id,
                 assigned_cell=int(r.assigned), named_truth_cell=int(r.named), true_cell_id=int(r.true),
                 assignment_status=r.assignment_status, z=float(r.z), y=float(r.y), x=float(r.x),
                 voxel_y=getattr(r, "voxel_y", None), voxel_x=getattr(r, "voxel_x", None))
            for r in wrong.itertuples(index=False)]


def assert_step2(row, counts, label):
    f = agreement_fractions(counts)
    assert f["agree_among_assigned"] is not None and f["agree_among_assigned"] >= TH4_AGREEMENT, (
        f"{row} {label}: TH4: agreement among assigned molecules {f['agree_among_assigned']} "
        f"({counts['agree_among_assigned_n']} of {counts['n_assigned']}) < {TH4_AGREEMENT}")


# ========================================================================== steps 3 and 4: assembly, report

def export_table(chain, name):
    from .test_report_measurements import export_table as read
    return read(chain.root, chain.sample, name)


def check_per_fov_identities(chain):
    """Identities 1 to 5 of "Count accounting" on the export (AS6, EE3): ``load_assembly`` (which checks them on
    construction), ``check_accounting`` recomputing them from the per-FOV §2.9 results, ``check_export``, and
    ``n_obs`` of sample.h5ad equal to the counted cells."""
    from starfinder.assembly import load_assembly
    from starfinder.export import check_export
    from starfinder.stitching import load_placement
    from .assembly_fixtures import check_accounting
    from .test_report_measurements import matrix
    loaded = load_assembly(chain.export_dir)
    placement = load_placement(chain.root / "stitching" / chain.sample)
    check_accounting(loaded, {fov: assignment(chain, fov) for fov in chain.fov_ids}, placement)
    check_export(chain.export_dir)
    cells = loaded.cells
    counted = cells["assembly_status"].eq("retained").to_numpy(bool) & cells["status"].eq("kept").to_numpy(bool)
    _, _, _, X = matrix(chain.root, chain.sample)
    assert X.shape[0] == int(counted.sum())
    return loaded


class _Folder:
    """The placement and grids of an export after fusion, as the T4 helper reads a folder."""

    def __init__(self, chain):
        from starfinder.stitching import load_placement
        self.placement = load_placement(chain.root / "stitching" / chain.sample)
        self.grids = {f: _grid(chain.root, f) for f in chain.fov_ids}


def check_export_after_fusion(folder):
    """``check_export``'s conditions on an export after fusion, by its own functions in its order (R1 to R3, F, T1,
    T5, T2, T4, L, S), with T3 evaluated by the route tests' ``t3_failures``, the same condition: the package's
    ``_t3`` compares ``owner_fov_id`` with ``Series.eq`` on an object column holding ``pd.NA`` and raises
    ``TypeError`` for a molecule that no FOV owns (excluded ``outside_grid``, a valid export), which these chains hold
    (W-487 worker notes, a finding for a later issue; the package is not changed here)."""
    from starfinder.assembly import load_assembly
    from starfinder.export import _check as checks
    from .test_sample_segmentation_route import t3_failures
    folder = Path(folder)
    record = checks._r1(folder)
    checks._r2(folder, record)
    checks._r3(folder, record)
    checks._check_fusion(folder, record)
    section = record.get("segmentation")
    assert checks.part_status(record, "segment") in ("written", "reused") and isinstance(section, dict)
    checks._t1(folder, record, section)
    assert checks.part_status(record, "assemble") in ("written", "reused")
    checks._t5(record, section)
    assert checks._after_fusion(record.get("assembly"))
    assembly = load_assembly(folder)
    checks._t2(folder, section, assembly.cells)
    assert t3_failures(folder, record, assembly.molecules) == []
    checks._t4(record["assembly"], assembly.cells)
    checks._check_labels(folder, record)
    checks._check_spatialdata(folder, record)


def check_fused_identities(chain):
    """SA7's identities (``load_assembly``), ``check_export``'s conditions (R1 to R3, F, L, S, T1 to T5;
    ``check_export_after_fusion``), with T1 to T5 also evaluated by the route tests' helpers; ``n_obs`` equal to the
    counted cells."""
    from starfinder.assembly import load_assembly
    from .test_report_measurements import matrix
    from .test_sample_segmentation_route import t1, t2, t3_failures, t4, t5
    loaded = load_assembly(chain.export_dir)
    check_export_after_fusion(chain.export_dir)
    record = _read_json(chain.export_dir / "export.json")
    t1(chain.export_dir, record)
    t2(chain.export_dir, record, loaded.cells)
    assert t3_failures(chain.export_dir, record, loaded.molecules) == []
    t4(_Folder(chain), loaded.cells)
    t5(record)
    cells = loaded.cells
    counted = cells["assembly_status"].eq("retained").to_numpy(bool) & cells["status"].eq("kept").to_numpy(bool)
    _, _, _, X = matrix(chain.root, chain.sample)
    assert X.shape[0] == int(counted.sum())
    return loaded


def ee_case(chain):
    """The case the EE2 oracle reads: the truth tables, the FOV truth labels and, as the FOV origins, those of the
    placement the chain used (the truth placement's are the true origins; SC1)."""
    from types import SimpleNamespace
    scene = chain.scene
    truth = scene.table("placement_truth").set_index("fov_id").copy()
    for fov, origin in scene.stitch["origins_used"].items():
        for axis, value in zip(AXES, origin):
            truth.loc[fov, f"true_origin_{axis}"] = value
    labels = SimpleNamespace(fov_labels={f: {"cell": scene.fov_labels(f, "cell")} for f in scene.fovs})
    return dict(scene=labels, truth=truth, ids=list(scene.fovs),
                tables={name: scene.table(name) for name in ("molecules", "cells", "molecule_fovs", "cell_fovs")})


def check_ee1(chain):
    """EE1 on the truth-label chain: per FOV, every molecule with in_footprint assigned to fov_cell_label exactly (0
    unassigned) and not outside_grid; every other one outside_grid. Returns label_agrees false per FOV."""
    views = chain.scene.table("molecule_fovs")
    disagree = {}
    for fov in chain.fov_ids:
        result = assignment(chain, fov).molecules
        rows = {int(s.split(":")[1]): r for s, r in zip(result.spot_id, result.itertuples(index=False))}
        for r in views[views.fov_id == fov].itertuples(index=False):
            got = rows.pop(int(r.csv_row))
            if r.in_footprint:
                assert (0 if pd.isna(got.cell_id) else int(got.cell_id)) == int(r.fov_cell_label), (fov, r.csv_row)
                assert got.assignment_status != "outside_grid", (fov, r.csv_row)
                disagree[fov] = disagree.get(fov, 0) + (not r.label_agrees)
            else:
                assert got.assignment_status == "outside_grid", (fov, r.csv_row)
        assert not rows, fov
    return disagree


def check_ee2(chain, loaded):
    """EE2: the assembly's cell and molecule statuses, reasons, owners and edge flags equal the oracle's."""
    from .test_stitching_assembly_scene import molecule_rows, none, oracle
    cells, molecules = oracle(ee_case(chain))
    got = {(r.fov_id, int(r.cell_id)): (r.assembly_status, none(r.assembly_reason), none(r.owner_fov_id),
                                        "fov_edge" in r.assembly_flags.split(";"),
                                        "z_edge" in r.assembly_flags.split(";")) for r in loaded.cells.itertuples()}
    assert got == cells
    rows = molecule_rows(loaded)
    assert {k: (r.assembly_status, none(r.assembly_reason), none(r.owner_fov_id)) for k, r in rows.items()} \
        == molecules


def check_sa3(chain, loaded):
    """SA3 on the truth-label chain: every truth cell with a pixel on the segmentation grid has exactly one row with
    its truth ID, and the cell run's territories equal the truth on the grid pixel for pixel."""
    from .test_sample_segmentation_route import territory_image
    record = _read_json(chain.export_dir / "export.json")
    truth = truth_on_grid(chain, "cell")
    ids = sorted(int(v) for v in np.unique(truth) if v)
    assert sorted(loaded.cells["cell_id"].astype("int64").tolist()) == ids
    np.testing.assert_array_equal(territory_image(chain.export_dir, record), truth)


def sa5_share(chain, loaded):
    """SA5 on the truth-label chain: T3 for every retained molecule (``check_fused_identities``); every retained
    molecule whose cell_id differs from its true cell is one whose sampled pixel's imported label differs from its
    true cell. Returns (in their true cell, retained)."""
    from .test_sample_segmentation_route import territory_image
    record = _read_json(chain.export_dir / "export.json")
    labels = territory_image(chain.export_dir, record)
    frame = fused_molecules(chain, {})
    same = 0
    for r in frame.itertuples(index=False):
        if r.assigned == r.true:
            same += 1
            continue
        assert int(labels[0, int(r.voxel_y), int(r.voxel_x)]) != r.true, (r.fov, r.spot_id)
    return same, len(frame)


def recompute_mecr(chain, pairs, threshold):
    """DF8 from X of sample.h5ad (read with anndata)."""
    from .test_report_measurements import matrix
    _, _, genes, X = matrix(chain.root, chain.sample)
    rows = []
    for a, b in pairs:
        pa, pb = X[:, genes.index(a)] >= threshold, X[:, genes.index(b)] >= threshold
        both, either = int((pa & pb).sum()), int((pa | pb).sum())
        rows.append(dict(gene_a=a, gene_b=b, cells_a=int(pa.sum()), cells_b=int(pb.sum()), both=both, either=either,
                         mecr=both / either if either else None))
    return rows


def mixing_cells(chain, pair):
    """The counted cells holding both genes of ``pair`` (at least one retained assigned molecule each) and, per cell,
    its retained marker molecules: (gene, the molecule's true cell, label_agrees on the route per FOV or None)."""
    scene = chain.scene
    cells, molecules = export_table(chain, "cells"), export_table(chain, "molecules")
    counted = cells[cells["assembly_status"].eq("retained").to_numpy(bool) & cells["status"].eq("kept").to_numpy(bool)]
    keys = {(ns, int(i)) for ns, i in zip(counted.cell_namespace, counted.cell_id)}
    kept = molecules[molecules.assembly_status.eq("retained").to_numpy(bool)
                     & molecules.assignment_status.eq("assigned").fillna(False).to_numpy(bool)
                     & molecules.gene_id.isin(pair).to_numpy(bool)].copy()
    views = scene.table("molecule_fovs").set_index(["fov_id", "csv_row"])
    true_cell = _true_cells(scene)
    out = {}
    for r in kept.itertuples(index=False):
        key = (r.cell_namespace, int(r.cell_id))
        if key not in keys:
            continue
        view = views.loc[(r.fov_id, int(r.spot_id.split(":")[1]))]
        out.setdefault(key, []).append((r.gene_id, int(true_cell[view.amplicon_id]), bool(view.label_agrees)
                                        if CONTEXTS[scene.context]["route"] == "per_fov" else None, int(r.cell_id)))
    return {k: v for k, v in out.items() if {g for g, *_ in v} == set(pair)}


def check_report(chain):
    """Step 4 on one chain: load_report equal to the files, RC12 and PG1; reads_in_cells equal to Σ X and to the
    molecules of the counted cells (identity 4); mecr.csv equal to DF8 recomputed from X; the cells per FOV by RT1's
    rule. Returns the page and the quantities reported."""
    from starfinder.report import load_report
    from . import test_report_page as page_checks
    from .test_report_chain import ChainCase, check_null_values
    from .test_report_measurements import expected_reads_in_cells, matrix
    record = load_report(chain.report_dir)
    page = page_checks.Page(ChainCase(chain, _report_config(chain)), chain.report_dir, record)
    for name, entry in ((e["path"], e) for e in record["files"]):
        assert hashlib.sha256((chain.report_dir / name).read_bytes()).hexdigest() == entry["sha256"], name
    page_checks.test_pg1_embedded_data_equal_the_tables(lambda maker, variant: page, "chain", None)
    check_null_values(page)
    # Reads assigned to cells (DF7, CH3) and identity 4.
    _, _, _, X = matrix(chain.root, chain.sample)
    cells, molecules = export_table(chain, "cells"), export_table(chain, "molecules")
    counted = cells["assembly_status"].eq("retained").to_numpy(bool) & cells["status"].eq("kept").to_numpy(bool)
    summary = record["summary"]
    assert summary["reads_in_cells"] == int(np.rint(X.sum())) == int(cells["n_molecules"][counted].sum())
    assert summary["reads_in_cells_equal"] is True
    assert summary["reads_in_cells"] == sum(expected_reads_in_cells(molecules, cells, chain.fov_ids).values())
    # mecr.csv against DF8 recomputed here.
    config = _report_config(chain)
    mecr = page.tables["mecr"]
    expected = recompute_mecr(chain, config.marker_pairs, config.positive_threshold)
    for row, want in zip(mecr.to_dict("records"), expected):
        for key in ("gene_a", "gene_b", "cells_a", "cells_b", "both", "either"):
            assert row[key] == want[key], (key, row, want)
        if want["mecr"] is None:
            assert pd.isna(row["mecr"])
        else:
            assert abs(row["mecr"] - want["mecr"]) <= TOL, (row, want)
    assert len(mecr) == len(expected)
    # The cells per FOV (RT1's rule): the retained cells of cells.csv whose ownership centroid the FOV owns, by
    # ownership_map of the placement, the export's grids and its assembly configuration.
    fovs = page.tables["fovs"].set_index("fov_id")
    by_owner = owned_counts(chain, cells[cells["assembly_status"].eq("retained").to_numpy(bool)])
    assert fovs["n_cells_retained"].astype("int64").to_dict() == by_owner, (fovs["n_cells_retained"].to_dict(),
                                                                            by_owner)
    counted_by_fov = cells[counted].groupby("fov_id").size().reindex(chain.fov_ids, fill_value=0).to_dict()
    return page, dict(mecr=mecr, cells_per_fov=fovs["n_cells_retained"].astype("int64").to_dict(),
                      counted_per_fov={k: int(v) for k, v in counted_by_fov.items()},
                      reads_in_cells=summary["reads_in_cells"])


def owned_counts(chain, cells):
    """Per FOV the rows of ``cells`` whose ownership centroid the FOV owns, by ``ownership_map`` of the placement, the
    export's FOV grids and its assembly configuration (RT1's rule, as ``test_report_chain.owned`` counts it)."""
    from starfinder.assembly import AssemblyConfig, ownership_map
    from starfinder.image import ImageMetadata
    from starfinder.segmentation import ReferenceGrid
    from starfinder.stitching import load_placement
    export = _read_json(chain.export_dir / "export.json")
    grids = {e["fov_id"]: ReferenceGrid(tuple(e["grid"]["shape_zyx"]), ImageMetadata(**e["grid"]["metadata"]),
                                        "declared") for e in export["assembly"]["fovs"]}
    owner = ownership_map(load_placement(chain.root / "stitching" / chain.sample), grids,
                          config=AssemblyConfig(**export["assembly"]["config"]))
    points = np.stack([cells[f"sample_ownership_centroid_{a}"].astype("float64").to_numpy(np.float64, na_value=np.nan)
                       for a in AXES], axis=1)
    owners = owner.owner(points) if len(points) else []
    return {f: int(sum(o == f for o in owners)) for f in chain.fov_ids}


def _report_config(chain):
    from starfinder.report import MarkerGroup, ReportConfig
    return ReportConfig(markers=tuple(MarkerGroup(name, tuple(genes)) for name, genes in chain.scene.markers()),
                        marker_pairs=REPORT_PAIRS)


# ============================================================================================= the tests

def _seeds(context, tier):
    return CONTEXTS[context]["seeds"][tier]


def _step0_scenes(chains, context, tier):
    chains.require_method(context)
    if context == "culture":
        return chains.all_labelled(context, tier)
    return [chains.scene(context, tier, seed) for seed in _seeds(context, tier)]


def _step0_rows(scenes, context):
    """CD0, CU0 (placement) and CT0: the stitch's statuses and per-axis errors (mean removed) and the placement used,
    per scene; CU0 also the registration's recovered displacement against morphology_truth.csv."""
    rows, registration = [], []
    for scene in scenes:
        stitch = scene.stitch
        for fov in scene.fovs:
            error = (stitch.get("error_um") or {}).get(fov)
            rows.append(dict(scene=scene.label, fov=fov, status=stitch["status"].get(fov), stitch=stitch["stitch"],
                             placement_used=stitch["placement_used"],
                             **{f"error_{x}_um": None if error is None else error[k] for k, x in enumerate(AXES)}))
        if context == "culture":
            truth = scene.table("morphology_truth").set_index("fov_id")
            for fov, entry in scene.register["fovs"].items():
                want = [float(truth.loc[fov, f"pull_displacement_{x}_vox"]) for x in AXES]
                registration.append(dict(scene=scene.label, fov=fov, transforms=entry["transforms"],
                                         **{f"recovered_{x}_vox": v for x, v in zip(AXES, entry["displacement_zyx"])},
                                         **{f"truth_{x}_vox": v for x, v in zip(AXES, want)},
                                         **{f"error_{x}_vox": got - w for x, got, w in
                                            zip(AXES, entry["displacement_zyx"], want)}))
    return rows, registration


def _print_step0(row, tier, scenes):
    for scene in scenes:
        stitch = scene.stitch
        print(f"\n{row} {tier} {scene.label}: stitch {stitch['stitch']}, statuses {stitch['status']}, largest per-axis "
              f"error (mean removed) {stitch.get('max_abs_error_um')} µm, placement used {stitch['placement_used']}"
              + (f", error {stitch['error']}" if stitch.get("error") else ""))


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier", TIERS)
def test_cd0_placement(chains, tier):
    scenes = _step0_scenes(chains, "dense3d", tier)
    rows, _ = _step0_rows(scenes, "dense3d")
    write_table(chains, f"cd0-{tier}-placement", rows)
    _print_step0("CD0", tier, scenes)
    for scene in scenes:
        stitch = scene.stitch
        assert stitch["placement_used"] == ("stitch" if stitch["stitch"] == "placed" else "truth")


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier", TIERS)
def test_ct0_stitch_on_the_projection_sample(chains, tier):
    scenes = _step0_scenes(chains, "tissue2d", tier)
    rows, _ = _step0_rows(scenes, "tissue2d")
    write_table(chains, f"ct0-{tier}-placement", rows)
    _print_step0("CT0", tier, scenes)
    (scene,) = scenes
    stitch = scene.stitch
    supported = stitch["stitch"] == "placed"
    largest = None if stitch.get("max_abs_error_um") is None else max(stitch["max_abs_error_um"])
    held = supported and largest is not None and largest <= TH5_ERROR_UM
    assert stitch["placement_used"] == ("stitch" if held else "truth")
    if tier == "slow":
        assert held, (f"CT0 slow: TH5: every FOV image_supported ({stitch['status']}) and every per-axis error at most "
                      f"{TH5_ERROR_UM} µm (largest {stitch.get('max_abs_error_um')}); CT1 to CT4 run on the truth "
                      "placement (SC1)")


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier", TIERS)
def test_cu0_registration_and_placement(chains, tier):
    scenes = _step0_scenes(chains, "culture", tier)
    rows, registration = _step0_rows(scenes, "culture")
    write_table(chains, f"cu0-{tier}-placement", rows)
    frame = write_table(chains, f"cu0-{tier}-registration", registration)
    _print_step0("CU0", tier, scenes)
    print(f"CU0 {tier}: TH-REG, the recovered displacement against morphology_truth.csv (voxels), largest per axis "
          + str({x: float(frame[f"error_{x}_vox"].abs().max()) for x in AXES}))
    assert (frame["transforms"] == 1).all()
    for scene in scenes:
        stitch = scene.stitch
        assert stitch["placement_used"] == ("stitch" if stitch["stitch"] == "placed" else "truth")


def _step1(chains, context, tier, row):
    per_scene, objects = {}, []
    if CONTEXTS[context]["route"] == "per_fov":
        for scene in chains.all_labelled(context, tier):
            counts, chains.labels[scene.label] = per_fov_nuclei(scene, objects)
            per_scene[scene.label] = counts
    else:
        for chain in chains.all(context, tier, "method"):
            counts, chains.labels[chain.scene.label] = fused_nuclei(chain, objects)
            per_scene[chain.scene.label] = counts
    write_table(chains, f"{row.lower()}-{tier}-nuclei", objects)
    total = pooled(list(per_scene.values()))
    write_table(chains, f"{row.lower()}-{tier}-counts",
                [dict(scene=k, **v, **step1_fractions(v)) for k, v in per_scene.items()]
                + ([dict(scene="pooled", **total, **step1_fractions(total))] if len(per_scene) > 1 else []))
    report_step1(row, tier, per_scene, total)
    if tier == "slow":
        assert_step1(row, total, "pooled over the seeds 42 and 43" if len(per_scene) > 1 else "")


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier", TIERS)
def test_cd1_nuclei_against_truth(chains, tier):
    _step1(chains, "dense3d", tier, "CD1")


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier", TIERS)
def test_ct1_nuclei_against_truth(chains, tier):
    _step1(chains, "tissue2d", tier, "CT1")


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier", TIERS)
def test_cu1_nuclei_against_truth(chains, tier):
    _step1(chains, "culture", tier, "CU1")


def _step2(chains, context, tier, row):
    per_scene, wrong = {}, []
    for chain in chains.all(context, tier, "method"):
        label = chain.scene.label
        if label not in chains.labels:
            rows = []
            chains.labels[label] = (per_fov_nuclei(chain.scene, rows)[1] if CONTEXTS[context]["route"] == "per_fov"
                                    else fused_nuclei(chain, rows)[1])
        maps = chains.labels[label]
        frame = per_fov_molecules(chain, maps) if CONTEXTS[context]["route"] == "per_fov" else \
            fused_molecules(chain, maps)
        per_scene[label] = agreement(frame)
        wrong += wrong_cell_rows(label, frame)
    write_table(chains, f"{row.lower()}-{tier}-molecules-not-in-their-true-cell", wrong)
    total = {k: sum(v[k] for v in per_scene.values()) for k in next(iter(per_scene.values()))}
    write_table(chains, f"{row.lower()}-{tier}-agreement",
                [dict(scene=k, **v, **agreement_fractions(v)) for k, v in per_scene.items()]
                + ([dict(scene="pooled", **total, **agreement_fractions(total))] if len(per_scene) > 1 else []))
    print(f"\n{row} {tier}: per scene " + "; ".join(f"{k}: {v} {agreement_fractions(v)}" for k, v in per_scene.items()))
    print(f"{row} {tier}: {'pooled ' if len(per_scene) > 1 else ''}{total} {agreement_fractions(total)} (TH-UN, the "
          "unassigned share of the molecules with a true cell, reported)")
    if tier == "slow":
        assert_step2(row, total, "pooled over the seeds 42 and 43" if len(per_scene) > 1 else "")


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier", TIERS)
def test_cd2_cell_of_each_molecule(chains, tier):
    _step2(chains, "dense3d", tier, "CD2")


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier", TIERS)
def test_ct2_cell_of_each_molecule(chains, tier):
    _step2(chains, "tissue2d", tier, "CT2")


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier", TIERS)
def test_cu2_cell_of_each_molecule(chains, tier):
    _step2(chains, "culture", tier, "CU2")


def _truth_labels_ee1(chains, context, tier, row):
    for chain in chains.all(context, tier, "truth"):
        disagree = check_ee1(chain)
        print(f"\n{row} {tier} truth labels {chain.scene.label}: EE1 holds; label_agrees false per FOV {disagree}")


@pytest.mark.slow
@pytest.mark.parametrize("tier", TIERS)
def test_cd2_truth_labels_ee1(chains, tier):
    """CD2 on the truth labels: EE1 holds exactly."""
    _truth_labels_ee1(chains, "dense3d", tier, "CD2")


@pytest.mark.slow
@pytest.mark.parametrize("tier", TIERS)
def test_cu2_truth_labels_ee1(chains, tier):
    """CU2 on the truth labels: EE1 holds exactly, on each scene."""
    _truth_labels_ee1(chains, "culture", tier, "CU2")


@pytest.mark.slow
@pytest.mark.parametrize("tier", TIERS)
def test_ct2_truth_labels_sa5(chains, tier):
    """CT2 on the truth labels: SA5 holds (T3 for every retained molecule; each molecule not in its true cell lies
    where the imported label is not its true cell)."""
    (chain,) = chains.all("tissue2d", tier, "truth")
    loaded = check_fused_identities(chain)
    same, total = sa5_share(chain, loaded)
    print(f"\nCT2 {tier} truth labels: SA5 holds; {same} of {total} retained molecules in their true cell "
          f"({same / total if total else float('nan'):.4f}), reported")


def _step3(chains, context, tier, variant, seed):
    chain = chains.chain(context, tier, seed, variant)
    if CONTEXTS[context]["route"] == "per_fov":
        loaded = check_per_fov_identities(chain)
        if variant == "truth":
            check_ee2(chain, loaded)
    else:
        loaded = check_fused_identities(chain)
        if variant == "truth":
            check_sa3(chain, loaded)
    counts = loaded.record["counts"]
    print(f"\n{context} {tier} {variant} {chain.scene.label}: the count identities hold; counts {counts}")


SCENES = [(tier, seed) for tier in TIERS for seed in (42, 43) if seed in CONTEXTS["culture"]["seeds"][tier]]


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier", TIERS)
def test_cd3_count_identities(chains, tier):
    _step3(chains, "dense3d", tier, "method", 42)


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier", TIERS)
def test_ct3_count_identities(chains, tier):
    _step3(chains, "tissue2d", tier, "method", 42)


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier, seed", SCENES)
def test_cu3_count_identities(chains, tier, seed):
    _step3(chains, "culture", tier, "method", seed)


@pytest.mark.slow
@pytest.mark.parametrize("tier", TIERS)
def test_cd3_count_identities_truth_labels(chains, tier):
    _step3(chains, "dense3d", tier, "truth", 42)


@pytest.mark.slow
@pytest.mark.parametrize("tier", TIERS)
def test_ct3_count_identities_truth_labels(chains, tier):
    _step3(chains, "tissue2d", tier, "truth", 42)


@pytest.mark.slow
@pytest.mark.parametrize("tier, seed", SCENES)
def test_cu3_count_identities_truth_labels(chains, tier, seed):
    _step3(chains, "culture", tier, "truth", seed)


def _marker_truth(chain):
    """The co-expressed pair's truth row of marker_pairs.csv."""
    pairs = chain.scene.table("marker_pairs")
    return pairs.to_dict("records")


def _step4(chains, context, tier, variant, seed, row):
    chain = chains.chain(context, tier, seed, variant)
    page, values = check_report(chain)
    exclusive = REPORT_PAIRS[0]
    mixing = mixing_cells(chain, exclusive)
    mecr = values["mecr"].set_index(["gene_a", "gene_b"])
    both = int(mecr.loc[exclusive, "both"])
    assert len(mixing) == both, (len(mixing), both)
    rows = [dict(scene=chain.scene.label, variant=variant, cell_namespace=k[0], cell_id=k[1], gene_id=g,
                 molecule_true_cell=t, label_agrees=a) for k, v in mixing.items() for g, t, a, _ in v]
    write_table(chains, f"{row.lower()}-{tier}-{variant}-seed{seed}-mixing-cells", rows)
    if variant == "truth":
        # Every mixing cell holds a retained marker molecule whose sampled label differs from its true cell
        # (label_agrees false on the route per FOV; after fusion, SA5's condition: the cell is not its true cell).
        for key, molecules in mixing.items():
            if CONTEXTS[context]["route"] == "per_fov":
                assert any(a is False for _, _, a, _ in molecules), key
            else:
                assert any(t != cell for _, t, _, cell in molecules), key
        if not mixing:
            assert mecr.loc[exclusive, "mecr"] == 0 or pd.isna(mecr.loc[exclusive, "mecr"])
    print(f"\n{row} {tier} {variant} {chain.scene.label}: reads_in_cells {values['reads_in_cells']} = Σ X; mecr.csv "
          f"equals DF8 recomputed; TH-MECR {mecr.reset_index().to_dict('records')}; mixing cells of the exclusive pair "
          f"{len(mixing)}; the truth marker pairs {_marker_truth(chain)}; TH-CPF cells per FOV (RT1) "
          f"{values['cells_per_fov']}, counted cells per FOV {values['counted_per_fov']}")
    return values


def _cells_per_fov(chains, context, tier, seed, row):
    method = _step4(chains, context, tier, "method", seed, row)
    truth = chains.chain(context, tier, seed, "truth")
    _, values = check_report(truth)
    difference = {f: method["cells_per_fov"][f] - values["cells_per_fov"][f] for f in truth.fov_ids}
    print(f"{row} {tier} seed {seed}: TH-CPF, the method's cells per FOV less the truth-label chain's {difference}")


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier", TIERS)
def test_cd4_report(chains, tier):
    _cells_per_fov(chains, "dense3d", tier, 42, "CD4")


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier", TIERS)
def test_ct4_report(chains, tier):
    _cells_per_fov(chains, "tissue2d", tier, 42, "CT4")


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier, seed", SCENES)
def test_cu4_report(chains, tier, seed):
    _cells_per_fov(chains, "culture", tier, seed, "CU4")


@pytest.mark.slow
@pytest.mark.parametrize("tier", TIERS)
def test_cd4_report_truth_labels(chains, tier):
    _step4(chains, "dense3d", tier, "truth", 42, "CD4")


@pytest.mark.slow
@pytest.mark.parametrize("tier", TIERS)
def test_ct4_report_truth_labels(chains, tier):
    _step4(chains, "tissue2d", tier, "truth", 42, "CT4")


@pytest.mark.slow
@pytest.mark.parametrize("tier, seed", SCENES)
def test_cu4_report_truth_labels(chains, tier, seed):
    _step4(chains, "culture", tier, "truth", seed, "CU4")


# ================================================================ step 5: the scene as the method sees it (VA5)

def _expanded_volumes(labels, metadata):
    """The volumes (µm³) of the nuclei expanded by 0.7776 µm (planar, the route's expansion): of the cells not cut by
    a border (W-393's shape rule, ``object_shapes``), and of every cell (border included, reported beside it)."""
    from starfinder.segmentation import ExpandLabelsConfig, expand_labels
    expanded, _ = expand_labels(labels, metadata, config=ExpandLabelsConfig(LN_EXPANSION_UM, "um", "planar"))
    expanded = np.asarray(expanded, np.int64)
    voxel = float(np.prod(metadata.spacing_zyx))
    counts = np.bincount(expanded.ravel())
    every = [float(c) * voxel for c in counts[1:] if c]
    return [r["size"] for r in object_shapes(expanded, metadata.spacing_zyx)], every


def _median(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.median(values)) if values else None


def _step5(chains, context, tier, row):
    """W-393's touching share (TR1) and volume (area) fraction (TR2) per image, and in dense 3D the volume of the
    nuclei expanded by 0.7776 µm (TR11), on the method's nucleus labels and on the truth beside them; each with its
    value (the median over the images, or over the cells), W-393's range and its object count. Reported only: on the
    slow presets as the page states, and on the routine presets too (an addition, reported the same way)."""
    rows, per_image = [], []
    if CONTEXTS[context]["route"] == "per_fov":
        sources = [(scene, None) for scene in chains.all_labelled(context, tier)]
    else:
        sources = [(chain.scene, chain) for chain in chains.all(context, tier, "method")]
    for scene, chain in sources:
        if chain is None:
            images = []
            for fov in scene.fovs:
                pred = np.asarray(tifffile.imread(scene.base / "labels" / fov / "nuclei.tif"), np.uint32).reshape(
                    scene.summary["fov_shape_zyx"])
                images.append((fov, pred, scene.fov_labels(fov, "nucleus"),
                               _grid(_work_root(scene.base, scene.summary), fov).metadata))
        else:
            images = [("sample", export_labels(chain, "nucleus")[0], truth_on_grid(chain, "nucleus")[0], None)]
        for labels_of in ("method", "truth"):
            values = {"TR1": [], "TR2": [], "TR11": [], "TR11 border included": []}
            objects = {"TR1": 0, "TR2": 0, "TR11": 0, "TR11 border included": 0}
            for fov, pred, truth, metadata in images:
                labels = pred if labels_of == "method" else truth
                share, present = touching_share(labels)
                fraction = volume_fraction(labels)
                values["TR1"].append(share)
                values["TR2"].append(fraction)
                objects["TR1"] += present
                objects["TR2"] += present
                if context == "dense3d":
                    volumes, every = _expanded_volumes(labels, metadata)
                    values["TR11"] += volumes
                    objects["TR11"] += len(volumes)
                    values["TR11 border included"] += every
                    objects["TR11 border included"] += len(every)
                per_image.append(dict(scene=scene.label, image=fov, labels=labels_of, touching_share=share,
                                      volume_fraction=fraction, objects=present))
            keys = list(STEP5_RANGES[context]) + (["TR11 border included"] if context == "dense3d" else [])
            for key in keys:
                low, high = STEP5_RANGES[context][key.split()[0]]
                rows.append(dict(scene=scene.label, row=key, labels=labels_of, value=_median(values[key]), low=low,
                                 high=high, objects=objects[key], outcome="reported (VA5)"))
    if context == "culture":
        for labels_of in ("method", "truth"):
            pool = [r for r in per_image if r["labels"] == labels_of]
            for key, column in (("TR1", "touching_share"), ("TR2", "volume_fraction")):
                low, high = STEP5_RANGES[context][key]
                rows.append(dict(scene="pooled", row=key, labels=labels_of, value=_median([r[column] for r in pool]),
                                 low=low, high=high, objects=sum(r["objects"] for r in pool),
                                 outcome="reported (VA5)"))
    write_table(chains, f"{row.lower()}-{tier}-per-image", per_image)
    frame = write_table(chains, f"{row.lower()}-{tier}", rows)
    print(f"\n{row} {tier} (reported, nothing asserted):\n" + frame.to_string(index=False))


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier", TIERS)
def test_cd5_scene_as_the_method_sees_it(chains, tier):
    _step5(chains, "dense3d", tier, "CD5")


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier", TIERS)
def test_ct5_scene_as_the_method_sees_it(chains, tier):
    _step5(chains, "tissue2d", tier, "CT5")


@pytest.mark.slow
@pytest.mark.learned
@pytest.mark.parametrize("tier", TIERS)
def test_cu5_scene_as_the_method_sees_it(chains, tier):
    _step5(chains, "culture", tier, "CU5")


# ================================================================ the step 1 measures on hand-built labels

def test_step1_measures_on_hand_built_labels():
    """The definitions of step 1 on labels whose outcome is known: a match; a truth nucleus split in three (each piece
    owned by it, none at IoU 0.5); a merge of two truth nuclei (matched to the larger, IoU 0.6); a miss at the Y
    border; a false positive; a two-piece nucleus whose detached piece a label covers alone, owned by it and so not
    a false positive (V4), which makes that nucleus own two labels, a split. And W-393's molecule agreement."""
    truth = np.zeros((1, 12, 24), np.uint32)
    truth[0, 1:4, 1:4] = 1
    truth[0, 1:4, 6:12] = 2
    truth[0, 6:9, 1:4] = 3
    truth[0, 6:9, 4:6] = 4
    truth[0, 9:12, 20:23] = 5
    truth[0, 1:3, 15:18] = 6
    truth[0, 5, 15] = 6
    pred = np.zeros_like(truth)
    pred[0, 1:4, 1:4] = 11
    pred[0, 1:4, 6:8], pred[0, 1:4, 8:10], pred[0, 1:4, 10:12] = 12, 13, 18
    pred[0, 6:9, 1:6] = 14
    pred[0, 10:12, 10:13] = 15
    pred[0, 1:3, 15:18] = 16
    pred[0, 5, 15] = 17
    rows = []
    counts, matches = nuclei_counts("hand", "fov", truth, pred, rows)
    assert counts == dict(n_truth=6, n_interior=5, matched=3, matched_interior=3, missed=1, missed_interior=0,
                          missed_at_xy_border=1, merged_truth=2, merges=1, split=2, false_positive=1, n_pred=8,
                          at_z_end=0, multi_piece=1)
    assert matches == {11: 1, 14: 3, 16: 6}
    assert step1_fractions(counts) == pytest.approx(dict(matched_fraction=3 / 5, merged_share=2 / 6,
                                                         false_positive_per_truth=1 / 6), abs=TOL)
    kinds = {(r["object"], r["id"]): r["kind"] for r in rows}
    assert kinds[("truth", 2)] == "split" and kinds[("truth", 4)] == "merged" and kinds[("truth", 5)] == "missed"
    assert kinds[("pred", 14)] == "merge" and kinds[("pred", 15)] == "false_positive" and kinds[("pred", 17)] == "owned"
    # The amendment of 2026-10-18: CD1's TH1 and CU1's TH3 are printed, not asserted; every other row is asserted.
    passing = dict(counts, merged_truth=0, split=0)
    assert assert_step1("CD1", passing, "hand") == [
        "CD1 hand: TH1: matched fraction 0.6 of 5 interior truth nuclei (3 matched), threshold at least 0.9: outside "
        f"the threshold; {AMENDMENT}"]
    for row in ("CT1", "CU1"):
        with pytest.raises(AssertionError, match="TH1"):
            assert_step1(row, passing, "hand")
    with pytest.raises(AssertionError, match="TH3") as missed:
        assert_step1("CD1", dict(passing, split=1), "hand")
    assert "TH1" not in str(missed.value)
    assert len(assert_step1("CU1", dict(passing, matched_interior=5, split=2), "hand")) == 1
    with pytest.raises(AssertionError, match="TH2"):
        assert_step1("CU1", dict(passing, matched_interior=5, merged_truth=2, split=2), "hand")
    frame = pd.DataFrame(dict(assigned=[0, 5, 5, 7, 0], named=[0, 2, 3, 2, 0], true=[0, 2, 2, 2, 4]))
    assert agreement_fractions(agreement(frame)) == pytest.approx(dict(
        agree_among_assigned=2 / 3, agree_with_true_cell=2 / 4, unassigned_with_true_cell=1 / 4), abs=TOL)
