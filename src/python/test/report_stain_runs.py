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
                      "reference_merged": False, "prepare": None, "prepare_channel": None, "registration": None,
                      "sha256": array_sha256(image[..., 0])}
            segmentation_input = SegmentationInput(image, grid, ("nuclear",), (source,))
            namespace = json.dumps([dataset_id, sample, fov_id, None, name], separators=(",", ":"))
            result = segment(segmentation_input, config=StainThresholdConfig(level, grow), target=name,
                             label_namespace=namespace)
            directory = run_directory(_layout.fov_dir(root, fov_id), name)
            saved[name] = write_run(directory, result, segmentation_input)
    return saved
