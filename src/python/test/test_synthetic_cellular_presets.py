"""Generator checks of the six context presets (design/validation/synthetic-cellular-scene.md).

GN7 on the presets (``context_routine``, ``context_slow``): K1 for every pair of sites recomputed by the test oracle,
the placement record, and F2's floor reached, with the reached density read back from the saved ``context.json``.
GN5 on ``context_slow`` (W-486): the lobes and LB1 by ``cellular_checks.check_lobes``, on the module's slow scenes.
GN24 on the routine presets: each generated in its own process under ``/usr/bin/time -v``, within the project's 4 GiB.
GN23 (W-393's target ranges on ``context_slow`` and ``context_slow_acquired``) and GN24's ``slow`` and acquired-pixel
cases (RG1: one CPU, 16 GiB and 3,600 s per step) were moved by W-536 to ``benchmarks/cellular_evidence.py``, run by
the Chapter II evidence project.

Each GN24 row is printed as one ``GN24 ROW`` JSON line, so a run with ``-s`` holds the table.
"""
from dataclasses import replace
import json
import math
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

from starfinder.synthetic import cellular_scene_preset, generate_cellular_scene, save_cellular_scene

from .cellular_checks import check_lobes
from .cellular_oracle import Oracle

pytestmark = [pytest.mark.synthetic, pytest.mark.validation]

ROUTINE = ["context-dense3d-v1", "context-tissue2d-v1", "context-culture-v1"]
SLOW = ["context-dense3d-slow-v1", "context-tissue2d-slow-v1", "context-culture-slow-v1"]
# RG1 (C26), granted at W-396: one CPU, 16 GiB and 3,600 s per step (benchmarks/cellular_evidence.py); the project's
# 4 GiB for a routine step.
BOUND_S, BOUND_KIB, ROUTINE_KIB = 3600.0, 16 * 1024 * 1024, 4 * 1024 * 1024


def generate(name):
    book, config = cellular_scene_preset(name)
    return config, generate_cellular_scene(book, config=config)


# ---------------------------------------------------------------- GN7: packing on the presets

def nominal_size(scene, planar):
    lobes = scene.lobes
    if planar:
        return float((math.pi * lobes.lateral_radius ** 2 * lobes.elongation).sum())
    return float((4 / 3 * math.pi * lobes.axial_radius * lobes.lateral_radius ** 2 * lobes.elongation).sum())


def check_packing(name, config, scene, tmp_path):
    """K1 and both power tests for every pair of sites, the record, and F2's floor read back from context.json."""
    oracle = Oracle(scene)
    packing = config.packing
    planar = scene.record["effective_config"]["planar"]
    centres, radii = oracle.centres, oracle.radii
    pairs = touching = 0
    for i in range(len(centres)):
        d_all = np.linalg.norm(centres - centres[i], axis=1)
        for k in range(i + 1, len(centres)):
            d = float(d_all[k])
            assert oracle.label_power(centres[k], [k])[0] < oracle.label_power(centres[k], [i])[0], (i, k)
            assert oracle.label_power(centres[i], [i])[0] < oracle.label_power(centres[i], [k])[0], (i, k)
            assert d > math.sqrt(abs(radii[i] ** 2 - radii[k] ** 2)) - 1e-9, (i, k)
            if d >= radii[i] + radii[k] + packing.gap:
                continue
            pairs += 1
            t = oracle.contact_distance(i, k, (centres[k] - centres[i]) / d)
            # K1 within 1e-9 µm: the test's float64 arithmetic is not the generator's.
            assert d >= (1 - packing.overlap) * t + packing.gap - 1e-9, (i, k, d, t)
            touching += d <= t
    record = scene.record["effective_config"]["packing"]
    sample = scene.sample
    omega = sample.record["domain"]["omega"]
    extent = np.array(omega["hi_zyx"]) - np.array(omega["lo_zyx"])
    domain = float(extent[1] * extent[2]) if planar else float(np.prod(extent))
    assert record["mode"] == "density" and record["density"] == packing.density
    assert record["placed_seeds"] == len(sample.cells) and record["placed_nuclei"] == len(sample.nuclei)
    # Non-integers the row compares exactly: within the default absolute 1e-9 (µm³, or µm² in a monolayer).
    assert abs(record["domain_size"] - domain) <= 1e-9
    assert abs(record["nominal_size"] - nominal_size(scene, planar)) <= 1e-9
    assert abs(record["reached_density"] - record["nominal_size"] / record["domain_size"]) <= 1e-12
    target = packing.density * record["domain_size"]
    if not record["stopped"]:
        # Seeds are added until the summed nominal size reaches φ · ‖Ω‖; the last one reaches it.
        last = sample.cells.cell_id.max()
        own = scene.lobes[scene.lobes.nucleus_id.isin(sample.nuclei.nucleus_id[sample.nuclei.cell_id == last])]
        before = record["nominal_size"] - nominal_size(replace(scene, lobes=own), planar)
        assert before < target <= record["nominal_size"]
    # F2: every preset reaches its floor (with on_crowded="stop" a stop below it would have raised).
    assert record["reached_density"] >= packing.floor * packing.density
    # The reached density and the placed seeds as context.json records them.
    root = save_cellular_scene(scene, tmp_path / name)
    try:
        saved = json.loads((root / "context.json").read_text(encoding="utf-8"))["effective_config"]["packing"]
    finally:
        shutil.rmtree(root)
    assert saved == json.loads(json.dumps(record))
    print(f"GN7 {name}: {len(centres)} sites, {pairs} pairs within reach, all K1 within 1e-9 µm, {touching} at or "
          f"inside their contact distance; reached density {saved['reached_density']:.4f} of {packing.density} "
          f"(floor {packing.floor * packing.density:.4f}), {saved['placed_seeds']} seeds, {saved['placed_nuclei']} "
          f"nuclei, stopped {saved['stopped']}")


@pytest.mark.parametrize("name", ROUTINE)
def test_gn7_routine_preset_reaches_its_floor(name, tmp_path):
    config, scene = generate(name)
    check_packing(name, config, scene, tmp_path)


@pytest.fixture(scope="module", params=SLOW)
def slow_scene(request):
    config, scene = generate(request.param)
    yield request.param, config, scene
    del scene


@pytest.mark.slow
def test_gn7_slow_preset_reaches_its_floor(slow_scene, tmp_path):
    name, config, scene = slow_scene
    check_packing(name, config, scene, tmp_path)


# ---------------------------------------------------------------- GN5 on the slow presets (W-486)

@pytest.mark.slow
def test_gn5_lobes_on_the_slow_presets(slow_scene):
    name, config, scene = slow_scene
    check_lobes(name, config, scene)


# ---------------------------------------------------------------- GN24: generation cost, one process per scene

GENERATE = """
import json, sys, time
from dataclasses import replace
from starfinder.synthetic import cellular_scene_preset, generate_cellular_scene
name, spacing = sys.argv[1], json.loads(sys.argv[2])
book, config = cellular_scene_preset(name)
if spacing is not None:
    acquisition = replace(config.sample.acquisition, spacing_zyx=tuple(spacing))
    config = replace(config, sample=replace(config.sample, acquisition=acquisition))
start = time.perf_counter()
scene = generate_cellular_scene(book, config=config)
seconds = time.perf_counter() - start
print(json.dumps(dict(generation_s=seconds, nuclei=len(scene.sample.nuclei), cells=len(scene.sample.cells),
                      molecules=len(scene.sample.molecules), bright_regions=len(scene.bright_regions))))
"""
TIME_FIELDS = {"wall_s": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)",
               "user_s": r"User time \(seconds\): (\S+)", "system_s": r"System time \(seconds\): (\S+)",
               "peak_rss_kib": r"Maximum resident set size \(kbytes\): (\d+)", "exit_status": r"Exit status: (\d+)"}


def clock(text):
    parts = [float(p) for p in text.split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def generate_in_own_process(name, spacing):
    result = subprocess.run(["/usr/bin/time", "-v", sys.executable, "-c", GENERATE, name, json.dumps(spacing)],
                            capture_output=True, text=True, timeout=BOUND_S + 60)
    found = {key: re.search(pattern, result.stderr) for key, pattern in TIME_FIELDS.items()}
    assert all(found.values()), result.stderr[-2000:]
    row = {key: clock(m.group(1)) if key == "wall_s" else float(m.group(1)) for key, m in found.items()}
    assert result.returncode == 0 and row["exit_status"] == 0, result.stderr[-2000:]
    row.update(json.loads(result.stdout.strip().splitlines()[-1]))
    return row


# The routine presets; the slow presets and the acquired-pixel variants are benchmarks/cellular_evidence.py's.
CASES = [(name, None, "routine") for name in ROUTINE]


@pytest.mark.slow
@pytest.mark.parametrize("name,spacing,tier", CASES, ids=[f"{n}{'@acquired' if s else ''}" for n, s, _ in CASES])
def test_gn24_generation_cost_in_its_own_process(name, spacing, tier):
    row = dict(generate_in_own_process(name, spacing), scene=name + ("@acquired" if spacing else ""), tier=tier)
    print("GN24 ROW " + json.dumps(row, sort_keys=True))
    limit = ROUTINE_KIB if tier == "routine" else BOUND_KIB
    assert row["wall_s"] <= BOUND_S and row["peak_rss_kib"] <= limit, row
