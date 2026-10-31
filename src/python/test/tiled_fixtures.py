"""The exactness fixtures of tiled execution (design/validation/orchestration.md, "Fixtures").

``exactness_fov`` is W-644's exactness fixture (``runs/W-644/20261029T200039Z-9454d4c8/scripts/fixture_synthetic.py``):
the benchmark preset ``small`` (its twelve-gene codebook, four rounds, four channels, ``uint16``, seed 42, the e2e
translation of every non-reference round) at 16 × 64 × 64 voxels with 60 amplicons, the appearance recomputed for that
shape. ``exactness_fov_deformed`` adds the deformation preset ``gaussian_small`` to the moving rounds. ``processing``
is W-644's processing of the fixture (``datasets.py``, ``synthetic``) in the Python shape: min-max to ``uint8`` with
SNR gate 5, one translation step, local maxima with noise threshold 5, extraction box (1, 2, 2), end bases CC.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

SHAPE = (16, 64, 64)
COUNT = 60
FOV = "FOV_001"
SAMPLE = "sample"

PROCESSING = {
    "load_codebook": {},
    "preprocessing": {"steps": [{"method": "min_max_normalization", "output_dtype": "uint8", "output_range": [0, 255],
                                 "rounding": "truncate", "snr_threshold": 5.0}]},
    "registration": {"steps": [{"method": "translation"}], "signal": {"mode": "max"}},
    "spot_finding": {"method": "local_maxima", "threshold_mode": "noise", "threshold_value": 5.0},
    "reads_extraction": {"voxel_size": [1, 2, 2]},
    "reads_filtration": {"end_base": "CC", "start_base": "C"},
    "streaming": True,
}


def write_exactness_fov(root, *, deformed=False):
    """Write the fixture under ``root/<dataset>/sample`` (``<round>/FOV_001/<channel>.tif`` and ``codebook.csv``);
    return the dataset id (``exactness`` or ``exactness-deformed``)."""
    from starfinder.synthetic import benchmark_scene_preset, generate_amplicon_dataset, save_amplicon_dataset
    from starfinder.synthetic._presets import _appearance, deformation_geometry
    book, config = benchmark_scene_preset("small")
    config = replace(config, shape_zyx=SHAPE, count=COUNT, max_count=1024, dataset_version="w644-exactness-v1",
                     scene_key="w644/exactness", **_appearance(SHAPE, len(book.round_labels), "uint16"))
    name = "exactness"
    if deformed:
        geometry = deformation_geometry("gaussian_small", SHAPE, reference_round=book.round_labels[0],
                                        translation_max_zyx=config.geometry.translation_max_zyx)
        config = replace(config, geometry=geometry)
        name = "exactness-deformed"
    dataset = generate_amplicon_dataset(book, config, fov_ids=(FOV,), preset="small")
    save_amplicon_dataset(dataset, Path(root) / name / SAMPLE)
    return name


def configuration(root, output_root, name, **processing):
    """The Python-shape configuration of the fixture ``name`` under ``root``, its processing block updated."""
    block = copy.deepcopy(PROCESSING)
    block.update(processing)
    return {"backend": "python", "root_input_path": str(root), "dataset_id": name, "sample_id": SAMPLE,
            "root_output_path": str(output_root), "output_id": "run", "n_rounds": 4, "ref_round": "round1",
            "seq_channel_order": ["ch00", "ch01", "ch02", "ch03"], "rotate_angle": 0, "voxel_size_xy": 0.5,
            "voxel_size_z": 1.0, "codebook": "codebook.csv", "processing": block}
