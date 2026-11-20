"""Computed segmentation runs on the saved stain, for the report's chain on the route per FOV (W-884).

``build_chain(computed_runs=True)`` replaces the imported truth labels by runs saved the way ``FOV.segment`` saves
them: a ``nucleus`` and a ``cell`` run on the FOV's stain, a threshold of the stain at Otsu's level (the cell run at 0.8
of it, its objects grown by 2 px; a test-only method, registered for the call only), written with ``write_run`` as
``fovs/<fov>/segmentation/<run>/`` (``labels.tif``, ``input.ome.tif`` and ``segmentation.json``). The report then shows
the run's input as the example background, not ``reference.tif``.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

#: The level of each run as a fraction of Otsu's threshold of the stain, and the growth of its objects in pixels.
RUNS = {"nucleus": (1.0, 0), "cell": (0.8, 2)}
METHOD = "w884_stain_threshold"


@dataclass(frozen=True)
class StainThresholdConfig:
    """A test-only method: the connected components of channel 0 above ``level``."""

    level: float = 0.0
    grow: int = 0
    method: str = field(default=METHOD, init=False)


def _run(image, config, context):
    from skimage.measure import label
    from skimage.segmentation import expand_labels
    labels = label(image[..., 0] > config.level, connectivity=1)
    if config.grow:
        labels = np.stack([expand_labels(plane, config.grow) for plane in labels]) if labels.ndim == 3 \
            else expand_labels(labels, config.grow)
    return labels.astype(np.int32), {"effective": {"level": config.level, "grow": config.grow}}


@contextmanager
def registered():
    """The test-only method in ``SEGMENTATION_METHODS`` for the block."""
    from starfinder.segmentation import SEGMENTATION_METHODS, SegmentationSpec
    spec = SegmentationSpec(METHOD, _run, targets=frozenset({"nucleus", "cell"}), roles=frozenset({"nuclear"}),
                            required_roles=(frozenset({"nuclear"}),), seeds="none", dimensions=frozenset({2, 3}),
                            models=False, devices=frozenset({"cpu"}))
    SEGMENTATION_METHODS[StainThresholdConfig] = spec
    try:
        yield
    finally:
        SEGMENTATION_METHODS.pop(StainThresholdConfig, None)


def save_stain_runs(root, fov_id, stain, grid, dataset_id, sample) -> dict:
    """Run and save the ``nucleus`` and ``cell`` runs on a FOV's stain (ZYX or ZYXC with one channel) under
    ``<root>/fovs/<fov_id>/segmentation/``; ``{target: saved SegmentationResult}``."""
    from skimage.filters import threshold_otsu

    from starfinder import _layout
    from starfinder.segmentation import SegmentationInput, segment
    from starfinder.segmentation._labels import array_sha256
    from starfinder.segmentation._persist import run_directory, write_run
    stain = np.asarray(stain)
    image = np.ascontiguousarray(stain if stain.ndim == 4 else stain[..., None])
    saved = {}
    with registered():
        otsu = float(threshold_otsu(image[..., 0]))
        for name, (fraction, grow) in RUNS.items():
            level = fraction * otsu
            source = {"round": "reference_stain", "channel": None, "name": None, "wavelength": None,
                      "reference_merged": False, "registration": None,
                      "sha256": array_sha256(image[..., 0])}
            segmentation_input = SegmentationInput(image, grid, ("nuclear",), (source,))
            namespace = json.dumps([dataset_id, sample, fov_id, None, name], separators=(",", ":"))
            result = segment(segmentation_input, config=StainThresholdConfig(level, grow), target=name,
                             label_namespace=namespace)
            directory = run_directory(_layout.fov_dir(root, fov_id), name)
            saved[name] = write_run(directory, result, segmentation_input)
    return saved


def export_after_fusion_on_stain(chain, destination, *, export_name="export"):
    """A copy of ``chain`` exported after fusion as ``report_scene.export_after_fusion`` does it, but with the cell run
    segmenting the fused reference stain: the truth nuclei projection imported as the seeds, and the cell run's input the
    image ``reference_stain`` (role ``cytoplasm``, the role ``seeded_watershed`` takes), so that the page's background
    after fusion is the stain, not ``reference_merged``. ``report_scene.export_after_fusion`` is not changed.

    The copied per-FOV export is removed and the export written in its place (``export_name`` ``"export"``, the folder the
    report reads by default); with another ``export_name`` the copied export stays and the new one is written to
    ``<root>/<export_name>/<sample>`` (the report reads it with ``export_dir``), so that no file is written twice."""
    import shutil

    import numpy as np

    from starfinder.assignment import AssignmentConfig
    from starfinder.dataset import Dataset, RoundState
    from starfinder.export import (ExportConfig, FusionConfig, SampleSegmentationConfig, SampleSegmentationRun,
                                   fused_grid)
    from starfinder.io import save_volume
    from starfinder.segmentation import LabelImportConfig, ReferenceGrid, SeededWatershedConfig
    from starfinder.stitching import load_placement

    copy = chain.copy(destination)
    if export_name == "export":
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
              SampleSegmentationRun("cell", "cell", (("cytoplasm", "reference_stain"),),
                                    SeededWatershedConfig(sigma_um=1.0), seeds="nucleus")),
        cells="cell", nuclei="nucleus", assignment=AssignmentConfig())
    dataset.export(copy.sample, placement=copy.root / "stitching" / copy.sample,
                   output_dir=copy.root / export_name / copy.sample,
                   config=ExportConfig(route="after_fusion", fusion=fusion, segmentation=segmentation))
    copy.route = "after_fusion"
    return copy
