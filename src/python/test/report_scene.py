"""The synthetic chains of the §2.11 report (design/specs/report-algorithms.md, "Fixtures").

``scene_chain``: ``sample_scene_preset("sample-tiny-v1")`` (seed 42; 2 × 2 FOVs of
5 × 36 × 36, four rounds of four channels). The FOV round images are written as one
OME-TIFF per channel in the dataset layout ``input/<round>/<fov>/<channel>.ome.tif`` (with
the FOV's metadata), so that ``FOV.run`` loads them and ``run.json`` records its inputs.
``FOV.run`` runs W-384's synthetic recipe (``scripts/synthetic_run.py``, read, not copied):
min–max normalization to uint8 (``rounding="truncate"``) and a white top-hat of radius 3,
one translation step on the channel maximum, local maxima, extraction, WTA decoding,
scoring, deduplication and a filter with ``end_bases``, with the checkpoint stages
``candidates`` and ``pre_qc``; then the rule outputs (reference image, molecules.csv, score
log, processing log), the truth labels imported with ``import_labels(…, relabel=False)``
and ``FOV.assign_molecules`` with checkpoints, the stain saved as ``reference_stain``; the truth
placement (``placement_from_table`` on ``positions.csv`` with the origins of
``placement_truth.csv``, approved) saved; ``Dataset.export`` on the route per FOV with
``dimensionality: both``. ``scene_chain_registered``: the same with the stage
``registered`` too. ``scene_chain_plane``: the same chain on ``scene_plane``
(``sample_segmentation_fixtures.scene_plane_config``: one-plane FOVs of 1 × 36 × 36),
exported twice, each export in its own output root (the report reads the export at its
fixed place): on the route per FOV, and on the route after fusion with the truth nuclei
projection imported and cells by ``seeded_watershed`` on the fused ``reference_merged``
image, as ``seeded_runs`` builds them.

Each chain is built once per test module (module-scoped fixtures) under pytest's temporary
folder; nothing is read from a run directory and no image is committed. The chains need the
``anndata`` extra (the export) and ``multiview-stitcher`` (the fused image) and skip with a
reason without them. :func:`build_chain` builds a variant (other stages, another recipe,
fewer FOVs, no export) for the checks that need one.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SEED = 42
STAGES = ("candidates", "pre_qc")
STAGES_REGISTERED = ("registered", "candidates", "pre_qc")
STAIN_FILE = Path("other_rounds") / "reference_stain" / "image.ome.tif"


def require_extras():
    pytest.importorskip("anndata", reason="the report's chains write an export, which needs the anndata extra")
    pytest.importorskip("multiview_stitcher", reason="the report's chains fuse images, which needs the "
                                                     "multiview-stitcher extra")


@dataclass
class Chain:
    """A chain's folders and what made it: the output root the report reads, the input root, the scene and the
    pipeline configuration."""

    base: Path
    root: Path
    input_root: Path
    scene_dir: Path
    sample: str
    dataset_id: str
    fov_ids: list
    rounds: list
    channels: list
    pipeline: object
    scene: object = field(repr=False)
    book: object = field(repr=False)
    route: str | None = None

    def run_record(self, fov) -> dict:
        return json.loads((self.root / "fovs" / fov / "run.json").read_text())

    def write_run_record(self, fov, record) -> None:
        (self.root / "fovs" / fov / "run.json").write_text(json.dumps(record, indent=2) + "\n")

    def collect(self, **changes):
        from starfinder.report._collect import collect_report
        arguments = dict(sample=self.sample, fov_ids=self.fov_ids, dataset_id=self.dataset_id,
                         sample_id=self.sample) | changes
        return collect_report(self.root, **arguments)

    def copy(self, destination) -> Chain:
        """A copy of the whole chain (inputs, scene and output root) whose run records name the copied inputs.

        A version-2 run record keeps its input paths, relative to the input root, and records the copy's input root
        (``input_root``); a version-1 record's absolute input paths are rewritten to the copied files.
        """
        destination = Path(destination)
        shutil.copytree(self.base, destination)
        copy = replace(self, base=destination, root=destination / self.root.relative_to(self.base),
                       input_root=destination / self.input_root.relative_to(self.base),
                       scene_dir=destination / self.scene_dir.relative_to(self.base))
        for fov in self.fov_ids:
            path = copy.root / "fovs" / fov / "run.json"
            if path.is_file():
                record = json.loads(path.read_text())
                if record.get("format_version", 1) == 1:
                    for entry in record["inputs"]:
                        entry["path"] = str(copy.input_root / Path(entry["path"]).relative_to(self.input_root))
                else:
                    record["input_root"] = str(copy.input_root)
                copy.write_run_record(fov, record)
        return copy


def scene_config(plane=False):
    if plane:
        from .sample_segmentation_fixtures import scene_plane_config
        return scene_plane_config()
    from starfinder.synthetic import sample_scene_preset
    return sample_scene_preset("sample-tiny-v1")


def pipeline_config(channels, *, registration=None, preprocessing=None):
    """W-384's synthetic recipe with deduplication; ``registration`` and ``preprocessing`` replace its recipes."""
    from starfinder.readout import (DeduplicationConfig, NeighborhoodSumConfig, ReadFilterConfig, ReadScoreConfig,
                                    WtaDecoderConfig)
    from starfinder.dataset import PipelineConfig
    from starfinder.registration import RegistrationRecipe, RegistrationStep
    from starfinder.io import ImageLoadConfig
    from starfinder.preprocessing import MinMaxNormalizationConfig, PreprocessingRecipe, PreprocessingStep, TophatConfig
    from starfinder.registration import RegistrationSignalConfig, TranslationConfig
    from starfinder.spot_finding import LocalMaximaConfig
    if preprocessing is None:
        preprocessing = PreprocessingRecipe((
            PreprocessingStep(MinMaxNormalizationConfig("uint8", (0, 255), rounding="truncate")),
            PreprocessingStep(TophatConfig(radius_yx=3))))
    if registration is None:
        registration = RegistrationRecipe(steps=(RegistrationStep(TranslationConfig()),),
                                          signal=RegistrationSignalConfig(mode="max"))
    return PipelineConfig(load=ImageLoadConfig(channel_labels=tuple(channels)), preprocessing=preprocessing,
                          registration=registration, spot_finding=LocalMaximaConfig(exclude_border=False),
                          extraction=NeighborhoodSumConfig(),
                          decoding=WtaDecoderConfig(negative_policy="clip_negative"),
                          scoring=ReadScoreConfig(), deduplication=DeduplicationConfig(),
                          filtering=ReadFilterConfig(end_bases="CC", start_base="C"))


def _write_inputs(scene, rounds, channels, root):
    from starfinder.io import save_volume
    for fov, formed in scene.fovs.items():
        for r in rounds:
            for c, channel in enumerate(channels):
                save_volume(formed.rounds[r][..., c:c + 1], root / r / fov / f"{channel}.ome.tif",
                            metadata=scene.fov_metadata[fov])


def build_chain(base, *, plane=False, stages=STAGES, registration=None, preprocessing=None, fov_ids=None,
                export=True) -> Chain:
    """One chain under ``base``: the scene, its inputs, the per-FOV runs and assignments, the truth placement and,
    with ``export``, the export on the route per FOV (``dimensionality: both``, or ``projection`` on a plane)."""
    from starfinder.dataset import CheckpointConfig, Dataset, RoundState
    from starfinder.io import save_volume
    from starfinder.segmentation import ReferenceGrid, import_labels
    from starfinder.stitching import approve_placement, placement_from_table, positions_from_table, save_placement
    from starfinder.synthetic import generate_sample_scene, save_sample_scene
    base = Path(base)
    book, config = scene_config(plane)
    scene = generate_sample_scene(book, config=config)
    scene_dir = base / "scene"
    save_sample_scene(scene, scene_dir)
    sample, dataset_id = scene.sample, config.fov.dataset_version
    rounds = list(next(iter(scene.fovs.values())).round_labels)
    channels = list(book.channel_labels)
    input_root, root = base / "input", base / "output"
    _write_inputs(scene, rounds, channels, input_root)
    dataset = Dataset(input_root, root, dataset_id, sample, "output",
                      rounds=RoundState(sequencing_rounds=rounds, reference_round=rounds[0]), channel_order=channels)
    dataset.codebook = book
    checkpoints = CheckpointConfig(stages=tuple(stages))
    pipeline = pipeline_config(channels, registration=registration, preprocessing=preprocessing)
    all_ids = list(scene.positions.fov_id)
    fov_ids = all_ids if fov_ids is None else list(fov_ids)
    grids = {}
    for fov_id in fov_ids:
        fov = dataset.fov(fov_id)
        fov.run(pipeline, checkpoints=checkpoints)
        fov.save_reference_image()
        fov.export_spots()
        grid = grids[fov_id] = fov.reference_grid()
        namespace = json.dumps([dataset_id, sample, fov_id, None, "truth"], separators=(",", ":"))
        labels = {target: import_labels(scene_dir / "fovs" / fov_id / "labels" / file, grid=grid, target=target,
                                        relabel=False, label_namespace=namespace)
                  for target, file in (("cell", "cells.tif"), ("nucleus", "nuclei.tif"))}
        fov.assign_molecules(cells=labels["cell"], nuclei=labels["nucleus"], checkpoints=checkpoints)
        stain = scene.stains[fov_id]
        save_volume(stain[..., None] if stain.ndim == 3 else stain, root / "fovs" / fov_id / STAIN_FILE,
                    metadata=scene.fov_metadata[fov_id])
    chain = Chain(base, root, input_root, scene_dir, sample, dataset_id, fov_ids, rounds, channels, pipeline, scene,
                  book)
    if export:
        table = positions_from_table(scene_dir / "positions.csv")
        truth = pd.read_csv(scene_dir / "placement_truth.csv", float_precision="round_trip").set_index("fov_id")
        origins = {f: tuple(float(v) for v in truth.loc[f, ["true_origin_z", "true_origin_y", "true_origin_x"]])
                   for f in all_ids}
        shapes = {f: ReferenceGrid(config.acquisition.fov_shape_zyx, scene.fov_metadata[f], "declared")
                  for f in all_ids}
        placement = approve_placement(placement_from_table(table, shapes, sample=sample, origins=origins),
                                      reviewer="test", note="truth placement")
        save_placement(placement, root / "stitching" / sample)
        _export_per_fov(chain, dataset, plane)
    return chain


def _export_per_fov(chain, dataset, plane):
    from starfinder.export import ExportConfig, FusionConfig
    fusion = FusionConfig(dimensionality="projection" if plane else "both")
    dataset.export(chain.sample, placement=chain.root / "stitching" / chain.sample,
                   output_dir=chain.root / "export" / chain.sample, config=ExportConfig(route="per_fov", fusion=fusion))
    chain.route = "per_fov"


def export_after_fusion(chain, destination) -> Chain:
    """A copy of ``chain`` whose export is on the route after fusion: the stains ``reference_stain`` and
    ``reference_merged`` fused, the truth nuclei projection imported on the segmentation grid and cells by
    ``seeded_watershed`` (sigma 1 µm) on the fused ``reference_merged`` image, as W-384's synthetic run did."""
    from starfinder.assignment import AssignmentConfig
    from starfinder.dataset import Dataset, RoundState
    from starfinder.io import save_volume
    from starfinder.export import (ExportConfig, FusionConfig, SampleSegmentationConfig, SampleSegmentationRun,
                                          fused_grid)
    from starfinder.segmentation import LabelImportConfig, ReferenceGrid, SeededWatershedConfig
    from starfinder.stitching import load_placement
    copy = chain.copy(destination)
    shutil.rmtree(copy.root / "export")
    dataset = Dataset(copy.input_root, copy.root, copy.dataset_id, copy.sample, "output",
                      rounds=RoundState(sequencing_rounds=copy.rounds, reference_round=copy.rounds[0]),
                      channel_order=copy.channels)
    dataset.codebook = copy.book
    scene = copy.scene
    placement = load_placement(copy.root / "stitching" / copy.sample)
    shape = next(iter(scene.fovs.values())).rounds[copy.rounds[0]].shape[:3]
    grids = {f: ReferenceGrid(tuple(shape), scene.fov_metadata[f], "declared") for f in copy.fov_ids}
    fusion = FusionConfig(images=("reference_stain", "reference_merged"))
    s_seg, o_seg, n_seg = fused_grid(placement, grids, config=fusion).level("projection")
    largest = scene.sample_labels["nucleus"].max(axis=0)
    sg = scene.sample_grid
    iy = np.floor((o_seg[1] + np.arange(n_seg[1]) * s_seg[1] - sg.origin_zyx[1]) / sg.spacing_zyx[1] + 0.5).astype(int)
    ix = np.floor((o_seg[2] + np.arange(n_seg[2]) * s_seg[2] - sg.origin_zyx[2]) / sg.spacing_zyx[2] + 0.5).astype(int)
    inside = ((iy >= 0) & (iy < largest.shape[0]))[:, None] & ((ix >= 0) & (ix < largest.shape[1]))[None, :]
    nuclei = np.where(inside, largest[iy.clip(0, largest.shape[0] - 1)[:, None],
                                      ix.clip(0, largest.shape[1] - 1)[None, :]], 0).astype(np.uint32)
    nuclei_file = copy.base / "truth" / "nuclei_projection.tif"
    save_volume(nuclei[np.newaxis], nuclei_file,
                metadata=fused_grid(placement, grids, config=fusion).level_metadata("projection"))
    segmentation = SampleSegmentationConfig(
        runs=(SampleSegmentationRun("nucleus", "nucleus", (), LabelImportConfig(str(nuclei_file), "nucleus")),
              SampleSegmentationRun("cell", "cell", (("amplicon", "reference_merged"),),
                                    SeededWatershedConfig(sigma_um=1.0), seeds="nucleus")),
        cells="cell", nuclei="nucleus", assignment=AssignmentConfig())
    dataset.export(copy.sample, placement=copy.root / "stitching" / copy.sample,
                   output_dir=copy.root / "export" / copy.sample,
                   config=ExportConfig(route="after_fusion", fusion=fusion, segmentation=segmentation))
    copy.route = "after_fusion"
    return copy


@pytest.fixture(scope="module")
def scene_chain(tmp_path_factory) -> Chain:
    require_extras()
    return build_chain(tmp_path_factory.mktemp("scene_chain"))


@pytest.fixture(scope="module")
def scene_chain_registered(tmp_path_factory) -> Chain:
    require_extras()
    return build_chain(tmp_path_factory.mktemp("scene_chain_registered"), stages=STAGES_REGISTERED)


@pytest.fixture(scope="module")
def scene_chain_plane(tmp_path_factory) -> dict:
    """``{"per_fov": chain, "after_fusion": chain}``: the plane chain and its copy exported after fusion."""
    require_extras()
    per_fov = build_chain(tmp_path_factory.mktemp("scene_chain_plane"), plane=True)
    after = export_after_fusion(per_fov, tmp_path_factory.mktemp("scene_chain_plane_after") / "chain")
    return {"per_fov": per_fov, "after_fusion": after}
