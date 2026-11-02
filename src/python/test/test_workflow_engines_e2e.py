"""X1–X3: real engines on tiny_sample, completion hashes and strict output comparison."""
import pytest

from . import workflow_engine_validation as E

pytestmark = [pytest.mark.workflow, pytest.mark.integration, pytest.mark.extended, pytest.mark.slow]


def _uncalibrated_run(tmp_path, engine, *, tiled=False):
    """Copy tiny_sample in the session and remove descriptions, keeping pixel arrays unchanged."""
    import tifffile
    from . import workflow_examples as X
    E.require(engine)
    work = tmp_path / ("uncalibrated-tiled" if tiled else "uncalibrated-direct")
    X.golden().write_fixture(work)
    for path in X.input_root(work).rglob("*.tif"):
        image = tifffile.imread(path)
        # The .ome.tif filename must not make tifffile infer a new OME C axis.
        tifffile.imwrite(path, image, photometric="minisblack", metadata=None, ome=False)
    run = E.EngineRun(work, engine, fixture=False)
    # Exercise the spacing-dependent segmentation method as well as stitching.
    run.config["segmentation"]["runs"][1] = {
        "name": "cell", "target": "cell", "method": "seeded_watershed", "seeds": "nucleus",
        "inputs": [{"role": "composite", "round": "reference_stain", "channel": "ch04"}]}
    if tiled:
        run.config["processing"].update(mode="tiled", tiled={"tile_size_yx": 18})
        run.config["processing"]["checkpoints"] = ["registered", "candidates", "pre_qc"]
    run.replan(run.config)
    return run


def _assert_declared_grids(run):
    """All persisted task grids and prepared images carry the declared micrometre spacing."""
    import json
    import subprocess
    import sys
    from starfinder.io import load_volume, load_volume_zyxc
    from . import workflow_examples as X
    def check(value):
        if isinstance(value, dict):
            if "spacing_zyx" in value:
                assert value["spacing_zyx"] == pytest.approx((1., .5, .5), rel=0, abs=1e-9)
                assert value["spatial_unit"] == "micrometer"
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    checked = []
    for path in (run.out / "fovs").rglob("*.json"):
        check(json.loads(path.read_text()))
        checked.append(str(path.relative_to(run.out)))
    for fov in X.FOVS:
        folder = run.out / "fovs" / fov
        for path, loader in [(folder / "reference.tif", load_volume),
                             (folder / "other_rounds/reference_stain/image.ome.tif", load_volume_zyxc),
                             (folder / "other_rounds/flamingo/image.ome.tif", load_volume_zyxc)]:
            metadata = loader(path).metadata
            assert metadata.spacing_zyx == pytest.approx((1., .5, .5), rel=0, abs=1e-9)
            assert metadata.spatial_unit == "micrometer"
            checked.append(str(path.relative_to(run.out)))
        if run.config["processing"].get("mode") == "tiled":
            from starfinder.io.chunked import read_chunked_record, read_window
            store = folder / "tiled/registered.h5"
            rounds = read_chunked_record(store)["rounds"]
            # Read the actual window source of every tile task, in every round.
            tiles = sorted((folder / "tiles").glob("*/tile.json"))
            assert len(tiles) == 4
            for path in tiles:
                tile = json.loads(path.read_text())
                for name in rounds:
                    metadata = read_window(store, name, tile["window_yx"]).metadata
                    assert metadata.spacing_zyx == pytest.approx((1., .5, .5), rel=0, abs=1e-9)
                    assert metadata.spatial_unit == "micrometer"
                    checked.append(f"{fov}/{tile['identity']['tile']}/{name}")
    E.evidence(run.work, "calibrated-grids.json", {"checked": checked, "spacing_zyx": [1., .5, .5]})
    command = [sys.executable, "-m", "starfinder", "workflow", "verify", "--manifest", str(run.manifest),
               "--hash", "--final"]
    done = subprocess.run(command, capture_output=True, text=True, timeout=30)
    E.evidence(run.work, "verify-final-cli.json", {"command": command, "returncode": done.returncode,
                                                "stdout": done.stdout, "stderr": done.stderr})
    assert done.returncode == 0, done.stdout + done.stderr


def test_uncalibrated_tiny_sample_snakemake(tmp_path):
    run = _uncalibrated_run(tmp_path, "snakemake")
    assert len(run.data["tasks"]) == 29
    run.invoke({t["id"] for t in run.data["tasks"]})
    _assert_declared_grids(run)


@pytest.mark.nextflow
def test_uncalibrated_tiny_sample_nextflow(tmp_path):
    run = _uncalibrated_run(tmp_path, "nextflow")
    assert len(run.data["tasks"]) == 29
    run.invoke({t["id"] for t in run.data["tasks"]})
    _assert_declared_grids(run)


def test_uncalibrated_tiny_sample_tiled_snakemake(tmp_path):
    run = _uncalibrated_run(tmp_path, "snakemake", tiled=True)
    assert len(run.data["tasks"]) == 49
    run.invoke({t["id"] for t in run.data["tasks"]})
    _assert_declared_grids(run)


@pytest.mark.parametrize("variant", ["per-fov", "molecules-only"])
def test_x1_snakemake_end_to_end(tmp_path, variant):
    run = E.EngineRun(tmp_path / variant, "snakemake", variant)
    run.invoke({t["id"] for t in run.data["tasks"]})


@pytest.mark.nextflow
@pytest.mark.parametrize("variant", ["per-fov", "molecules-only", "supplied-xml"])
def test_x2_nextflow_end_to_end(tmp_path, variant):
    run = E.EngineRun(tmp_path / variant, "nextflow", variant)
    run.invoke({t["id"] for t in run.data["tasks"]})


@pytest.mark.nextflow
@pytest.mark.parametrize("variant", ["per-fov", "molecules-only"])
def test_x3_engines_have_byte_equal_declared_outputs(tmp_path, variant):
    # Same input bytes, mtimes, resolved configuration and output path in one environment.
    # Each engine starts with no output tree and its own clean work/cache folder.
    E.require("nextflow")
    run = E.EngineRun(tmp_path / variant, "snakemake", variant)
    run.invoke({t["id"] for t in run.data["tasks"]})
    first = E.snapshot(run)
    import shutil
    shutil.rmtree(run.out)
    run.engine = "nextflow"
    E.require(run.engine)
    run.invoke({t["id"] for t in run.data["tasks"]})
    E.compare(run, first, E.snapshot(run), name=f"x3-{variant}.json", left_engine="snakemake", right_engine="nextflow")
