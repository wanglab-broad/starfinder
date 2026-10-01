"""Shared pieces of the extended-tier Spotiflow and Piscis tests (W-272): scenes, matching, digests, processes.

seam_scene ports W-266 ``w266_common.seam_layout`` and the seam branch of
``isolated_positions``, keeping only the spots 0.5, 1 and 2 px from a
keep-boundary of 32-pixel Piscis tiles and the four controls (check S8, 18
spots); positions come from W-266's random stream for the whole layout, so
the kept spots sit where W-266 put them. The scenes are rendered in session
with the isolated-spot appearance; nothing is written to disk.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from starfinder.barcode import Codebook
from starfinder.evaluation.spot_finding import evaluate_spots, localization_errors
from starfinder.image import ImageMetadata
from starfinder.spot_finding import find_spots
from starfinder.synthetic import (BackgroundConfig, FormedSceneConfig, NoiseConfig, ScalarDistribution,
    generate_formed_scene)

META = ImageMetadata("learned-detectors")
NAMESPACE = "learned-detectors/test"
# W-266's isolated-spot policy (docs/spot-finding-algorithms.md, "Engineering validation design").
W266_MATCH = dict(policy="greedy", threshold=3.0, boundary="inclusive", units="voxel")
SEAM_SHAPES = {"seam_z1": (1, 64, 64), "seam3d": (32, 64, 64)}
SEAM_OFFSETS = (0.5, 1.0, 2.0)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "NUMBA_NUM_THREADS")


def detect(image, config, **options):
    return find_spots(image, config=config, metadata=META, spot_namespace=NAMESPACE, **options)


def evaluate(spots, truth):
    """evaluate_spots with W-266's policy and the localization errors of its matched pairs."""
    detected = spots[["z", "y", "x"]].to_numpy()
    match = evaluate_spots(detected, truth, reference_metadata=META, observed_metadata=META, **W266_MATCH)
    return match, localization_errors(match, detected, truth)


def table_digest(spots):
    """SHA-256 over the column names and dtypes and the CSV text (floats in %.17g), as the golden test digests."""
    header = json.dumps([[str(c), str(t)] for c, t in spots.dtypes.items()])
    text = spots.to_csv(index=False, float_format="%.17g", na_rep="<NA>")
    return hashlib.sha256((header + "\n" + text).encode()).hexdigest()


def one_thread_environment(**extra):
    """The environment of a second process: one numerical thread, no GPU, this checkout first on the path."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", **{name: "1" for name in THREAD_VARIABLES})
    source = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([source] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env.update(extra)
    return env


def run_python(code, env=None):
    """Run code in a second Python process from src/python; return its stdout."""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env or
                            one_thread_environment(), cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr[-4000:]
    return result.stdout


def seam_layout():
    """(y, x, kind, offset) of W-266's seam scene; the keep-boundaries of 32-pixel tiles on 64 px are 30.5 and 59.5."""
    spots = []
    for x, d in zip((5, 11, 17, 23, 37, 51), (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0)):
        spots.append((30.5 + d, float(x), "y-seam 30.5", d))
    for y, d in zip((5, 11, 17, 23, 37, 51), (1.0, 0.5, 0.0, -0.5, -1.0, -2.0)):
        spots.append((float(y), 30.5 + d, "x-seam 30.5", d))
    for x, d in zip((8, 20, 37), (-1.0, 0.0, 1.0)):
        spots.append((59.5 + d, float(x), "y-seam 59.5", d))
    for y, d in zip((8, 20, 38), (1.0, 0.0, -1.0)):
        spots.append((float(y), 59.5 + d, "x-seam 59.5", d))
    spots += [(30.5, 30.5, "corner 30.5/30.5", 0.0), (59.5, 59.5, "corner 59.5/59.5", 0.0)]
    spots += [(y, x, "control", 0.0) for y, x in ((12.0, 12.0), (12.0, 50.0), (50.0, 12.0), (50.0, 50.0))]
    return spots


def seam_positions(case, seed):
    """W-266's truth centres (ZYX) of the whole seam layout: the seam axis exact, the other jittered."""
    rng = np.random.default_rng([266, seed, 3 if case == "seam_z1" else 4])
    out = []
    for y, x, kind, _ in seam_layout():
        jitter = rng.uniform(-0.5, 0.5, size=2)
        if kind.startswith("y-seam"):
            x += jitter[1]
        elif kind.startswith("x-seam"):
            y += jitter[0]
        elif kind == "control":
            y, x = y + jitter[0], x + jitter[1]
        z = 0.0 if case == "seam_z1" else 16.0 + rng.uniform(-0.5, 0.5)
        out.append((z, y, x))
    return np.array(out)


def seam_scene(case, seed):
    """(uint16 ZYX image, truth (18, 3) ZYX, (kind, offset) per spot) of the S8 seam scene.

    The spots 0.5, 1 and 2 px from a keep-boundary and the controls, rendered
    with the isolated-spot appearance (brightness 1500, sigma Z 1.5 and YX
    1.3, baseline 100, Poisson plus read noise 3).
    """
    keep = [i for i, (_, _, kind, offset) in enumerate(seam_layout())
            if kind == "control" or (kind.endswith("-seam 30.5") or kind.endswith("-seam 59.5"))
            and abs(offset) in SEAM_OFFSETS]
    coords = seam_positions(case, seed)[keep]
    book = Codebook(pd.DataFrame(dict(gene_id=["iso"], color_sequence=["1"])), ("round1",),
                    ("ch00", "ch01", "ch02", "ch03"))
    config = FormedSceneConfig(
        dataset_version="w266-isolated-v1", sample_id="w266", scene_key=f"w266/{case}", seed=seed,
        shape_zyx=SEAM_SHAPES[case], dtype="uint16", coordinates=tuple(map(tuple, coords)),
        brightness=ScalarDistribution(parameters=(1500.0,)),
        axial_width=ScalarDistribution(parameters=(1.5,)), lateral_width=ScalarDistribution(parameters=(1.3,)),
        background=BackgroundConfig(baseline_enabled=True, baseline=((100.0, 100.0, 100.0, 100.0),)),
        noise=NoiseConfig(dependent_enabled=True, alpha=1.0, model="poisson", independent_enabled=True, sigma=3.0))
    scene = generate_formed_scene(book, config=config)
    layout = [seam_layout()[i][2:] for i in keep]
    return scene.rounds["round1"][..., 0], scene.round_truth[["z", "y", "x"]].to_numpy(float), layout
