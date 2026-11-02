"""X5 and X6 of design/validation/orchestration.md: tiled execution against direct on the exactness fixtures.

X5 (without local registration): on ``exactness_fov`` at tile sizes 32 (a 2 × 2 grid) and 22 (3 × 3) with halos 6 and
12, the global task, every tile task and the merge (``starfinder.dataset.tiled``) give the candidate table (the
``candidates`` stage) and the molecule table of direct execution (``FOV.run`` and ``FOV.export_spots``): their integer
and string columns equal, their floats (the peak values, the signals and backgrounds, the scores) within float64
``rtol=1e-9``, ``atol=1e-12`` (docs/contributing.md), and ``run.json``'s counts equal (I-discrete).

X6 (with local registration, one demons step, halo 12): on ``exactness_fov_deformed`` direct registers each moving round
with the translation and the demons step over the FOV, tiled with the translation in the global task and the demons
step in each tile's window. The tiled record's proposed tolerance: at least 95 % of direct's molecules have a tiled
molecule of the same gene within one voxel (Chebyshev) and at least 95 % of the tiled molecules a direct one, and the
counts differ by at most 5 % (not exact: three proportions).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starfinder import _layout

from . import tiled_fixtures as F

pytestmark = [pytest.mark.workflow, pytest.mark.dataset, pytest.mark.extended]

RTOL, ATOL = 1e-9, 1e-12


def _adapted(root, out, name):
    """The fixture's processing block translated (the plan's resolution and the kept translator), codebook loaded."""
    from starfinder.dataset.workflow import _load_reference
    from starfinder.workflow import _translate
    from starfinder.workflow._plan import resolve
    resolved = resolve(F.configuration(root, out, name), {"path": "test", "sha256": None})["configuration"]
    adapted = _translate.processing(resolved).workflow
    adapted.dataset.output_root = Path(out) / name / "run"
    _load_reference(adapted, Path(root) / name / F.SAMPLE / "codebook.csv")
    return adapted


def _demons(pipeline):
    from starfinder.registration import DemonsConfig, RegistrationRecipe, RegistrationSignalConfig, RegistrationStep
    return RegistrationRecipe((RegistrationStep(DemonsConfig()),), signal=RegistrationSignalConfig("max"))


def run_direct(root, out, name, *, local=False):
    """Direct execution: FOV.run with the candidates and pre_qc stages, FOV.export_spots; returns the FOV folder."""
    from dataclasses import replace

    from starfinder.dataset.config import CheckpointConfig
    from starfinder.registration import RegistrationRecipe
    adapted = _adapted(root, out, name)
    pipeline = adapted.pipeline
    if local:
        recipe = pipeline.registration
        pipeline = replace(pipeline, registration=RegistrationRecipe(recipe.steps + _demons(pipeline).steps,
                                                                     signal=recipe.signal, warp=recipe.warp))
    fov = adapted.dataset.fov(F.FOV)
    fov.run(pipeline, execution=adapted.execution, checkpoints=CheckpointConfig(stages=("candidates", "pre_qc"),
                                                                                overwrite=True))
    fov.export_spots()
    return _layout.fov_dir(adapted.dataset.output_root, F.FOV)


def run_tiled(root, out, name, size, halo, *, local=False):
    """Tiled execution through the API entries: run_tile_global, run_tile per tile, merge_tiles; the FOV folder."""
    from starfinder.dataset import tiled as T
    from starfinder.dataset.config import CheckpointConfig
    adapted = _adapted(root, out, name)
    folder = _layout.fov_dir(adapted.dataset.output_root, F.FOV)
    record = T.run_tile_global(adapted.dataset.fov(F.FOV), adapted.pipeline, execution=adapted.execution,
                               directory=_layout.tiled_dir(folder))
    tiles = T.tile_grid(record["shape_zyxc"][1:3], size, halo)
    directories = {tile.id: _layout.tile_dir(folder, tile.id) for tile in tiles}
    for tile in tiles:
        again = _adapted(root, out, name)
        T.run_tile(again.dataset.fov(F.FOV), again.pipeline, tile,
                   store=_layout.tiled_dir(folder) / _layout.CHUNKED_CHECKPOINT, statistics=record["statistics"],
                   directory=directories[tile.id], halo=halo,
                   local_registration=_demons(again.pipeline) if local else None)
    merged = _adapted(root, out, name)
    fov = merged.dataset.fov(F.FOV)
    run_record = json.loads((_layout.tiled_dir(folder) / _layout.RUN_RECORD).read_text())
    T.merge_tiles(fov, merged.pipeline, tiles, tile_directories=directories, global_record=record,
                  global_run_record=run_record, execution=merged.execution,
                  checkpoints=CheckpointConfig(stages=("candidates", "pre_qc"), overwrite=True))
    fov.export_spots()
    return folder, len(tiles)


def _table(path, dtypes=None):
    return pd.read_csv(path, dtype=dtypes, keep_default_na=False, na_values={"qc_score": [""]})


def assert_tables_equal(direct, tiled, what):
    """Integer, Boolean and string columns exactly; floats within float64 rtol=1e-9, atol=1e-12 (NaN where NaN)."""
    assert list(direct.columns) == list(tiled.columns), what
    assert len(direct) == len(tiled), what
    for column in direct.columns:
        a, b = direct[column], tiled[column]
        if a.dtype.kind == "f" or b.dtype.kind == "f":
            a, b = a.to_numpy(dtype=np.float64), b.to_numpy(dtype=np.float64)
            assert np.array_equal(np.isnan(a), np.isnan(b)), (what, column)
            np.testing.assert_allclose(b, a, rtol=RTOL, atol=ATOL, err_msg=f"{what}: {column}")
        else:
            assert a.astype(str).tolist() == b.astype(str).tolist(), (what, column)


@pytest.fixture(scope="module")
def exactness(tmp_path_factory):
    root = tmp_path_factory.mktemp("exactness")
    name = F.write_exactness_fov(root / "in")
    return root, name, run_direct(root / "in", root / "direct", name)


@pytest.mark.parametrize("size,halo,grid", [(32, 6, 4), (32, 12, 4), (22, 6, 9), (22, 12, 9)])
def test_x5_tiled_equals_direct_without_local_registration(exactness, size, halo, grid):
    root, name, direct = exactness
    tiled, tiles = run_tiled(root / "in", root / f"tiled-{size}-{halo}", name, size, halo)
    assert tiles == grid
    stage = _layout.CHECKPOINTS
    candidates = [_table(folder / stage / "candidates.csv") for folder in (direct, tiled)]
    assert len(candidates[0]) > 0
    assert_tables_equal(*candidates, "candidates")
    molecules = [_table(folder / _layout.SPOT_TABLES["molecules"]) for folder in (direct, tiled)]
    assert len(molecules[0]) > 0
    assert_tables_equal(*molecules, "molecules")
    reads = [_table(folder / stage / "pre_qc.csv") for folder in (direct, tiled)]
    assert_tables_equal(*reads, "pre_qc")
    records = [json.loads((folder / _layout.RUN_RECORD).read_text()) for folder in (direct, tiled)]
    assert records[1]["status"] == "succeeded" and records[1]["format_version"] == 2
    assert_counts_equal(records[0]["counts"], records[1]["counts"])
    # The merge removed nothing it does not own; the chunked checkpoint stays for the caller to remove.
    assert (_layout.tiled_dir(tiled) / _layout.CHUNKED_CHECKPOINT).is_file()


def assert_counts_equal(direct, tiled, path="counts"):
    """run.json's counts: integers, strings and structure exactly, floats (quantiles, medians) within float64
    rtol=1e-9, atol=1e-12."""
    if isinstance(direct, dict):
        assert isinstance(tiled, dict) and list(direct) == list(tiled), path
        for key in direct:
            assert_counts_equal(direct[key], tiled[key], f"{path}.{key}")
    elif isinstance(direct, list):
        assert isinstance(tiled, list) and len(direct) == len(tiled), path
        for i, (a, b) in enumerate(zip(direct, tiled)):
            assert_counts_equal(a, b, f"{path}[{i}]")
    elif isinstance(direct, float) and not isinstance(tiled, bool):
        assert abs(tiled - direct) <= ATOL + RTOL * abs(direct), path
    else:
        assert type(direct) is type(tiled) and direct == tiled, path


@pytest.mark.registration
@pytest.mark.validation
@pytest.mark.slow
@pytest.mark.parametrize("size,grid", [(32, 4), (22, 9)])
def test_x6_tiled_with_local_registration_meets_the_proposed_proportions(tmp_path_factory, size, grid):
    root = tmp_path_factory.mktemp(f"deformed-{size}")
    name = F.write_exactness_fov(root / "in", deformed=True)
    direct = _table(run_direct(root / "in", root / "direct", name, local=True) / _layout.SPOT_TABLES["molecules"])
    folder, tiles = run_tiled(root / "in", root / "tiled", name, size, 12, local=True)
    assert tiles == grid
    tiled = _table(folder / _layout.SPOT_TABLES["molecules"])
    assert len(direct) > 0
    from starfinder.evaluation.matching import evaluate_molecules
    from starfinder.image import ImageMetadata
    metadata = ImageMetadata("x6/reference")
    agreement = evaluate_molecules(direct, tiled, policy="any_neighbor", threshold=1 + 1e-9,
                                   units="voxel", reference_metadata=metadata, observed_metadata=metadata)
    found, kept = agreement.values["reference_fraction"], agreement.values["observed_fraction"]
    print(json.dumps({"size": size, "direct": len(direct), "tiled": len(tiled), "direct_matched": found,
                      "tiled_matched": kept, "count_difference": agreement.values["count_difference"]}))
    assert found >= 0.95
    assert kept >= 0.95
    assert abs(len(tiled) - len(direct)) <= 0.05 * len(direct)
