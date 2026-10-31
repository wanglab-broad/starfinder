"""C5 of design/validation/orchestration.md: the plan's refusals.

The two refused variants of the worked example, a missing round folder, a channel label that matches two files, a
run name with ``/``, each key that left the Python dialect and ``backend: matlab``: ``starfinder workflow plan``
exits 2 with a message that names the key, FOV, round or label, and writes nothing. And the refusals that moved here
from the tests of the old rules' parse-time checks (W-750): a channel label that matches no file, a reference image
pattern without ``{fov_id}``, the legacy sample rules of a Python configuration, a stage file of a fovs.csv without
PositionIDs and the route RD1 refuses; these plan configurations of the test's own on an input root whose raw
images are empty files (``workflow_tasks``).
"""
import copy
import shutil
from pathlib import Path

import pytest

from starfinder.__main__ import main

from starfinder.workflow._plan import PlanRefused

from . import workflow_examples as X
from . import workflow_tasks as W

pytestmark = [pytest.mark.workflow, pytest.mark.contract]


@pytest.fixture(scope="module")
def fixture_work(tmp_path_factory):
    work = tmp_path_factory.mktemp("tiny") / "work"
    X.golden().write_fixture(work)
    X.write_supplied(work)
    return work


@pytest.fixture
def work(fixture_work, tmp_path):
    target = tmp_path / "work"
    shutil.copytree(fixture_work, target, symlinks=True)
    return target


def _refused(work, config, capsys, *parts):
    """Plan ``config``: exit 2, a message holding every one of ``parts``, and no file under the output root."""
    path = X.write_config(work, "refused", config)
    assert main(["workflow", "plan", "--config", str(path)]) == 2
    message = capsys.readouterr().err
    assert message.startswith("plan refused: ")
    for part in parts:
        assert part in message, (part, message)
    assert not (work / "output").exists()
    return message


def test_the_refused_assignment_variant(work, capsys):
    message = _refused(work, X.variants(work)["refused-assignment"], capsys)
    assert message.strip() == (
        "plan refused: the assignment block assigns each FOV's molecules, which only the processing block produces "
        "(with spot_finding, reads_extraction and reads_filtration): add a processing block")


def test_the_refused_matlab_variant(work, capsys):
    message = _refused(work, X.variants(work)["refused-matlab"], capsys)
    assert message.strip() == (
        "plan refused: this configuration has no backend and holds keys of the MATLAB dialect (workflow_mode, rules); "
        "the default backend is python: add backend: matlab to run the MATLAB workflow, or migrate it to the Python "
        "configuration (docs/migration.md, \"The Python configuration shape\")")


def test_backend_matlab(work, capsys):
    _refused(work, dict(X.variants(work)["molecules-only"], backend="matlab"), capsys, "backend: matlab",
             "frozen MATLAB workflow")


def test_a_missing_round_folder(work, capsys):
    shutil.rmtree(X.input_root(work) / "round3" / "FOV_002")
    _refused(work, X.variants(work)["molecules-only"], capsys, "FOV FOV_002", "round round3", "does not exist")


def test_a_channel_label_that_matches_two_files(work, capsys):
    folder = X.input_root(work) / "round2" / "FOV_004"
    shutil.copyfile(folder / "ch01.ome.tif", folder / "copy-ch01.tif")
    _refused(work, X.variants(work)["molecules-only"], capsys, "FOV FOV_004", "round round2", "'ch01'",
             "found 2")


def test_a_run_name_with_a_slash(work, capsys):
    config = X.variants(work)["morphology-only"]
    config["segmentation"]["runs"][0]["name"] = "nucleus/2d"
    _refused(work, config, capsys, "segmentation run name 'nucleus/2d'", "'/'")


#: Each key that left the Python dialect (the contract's migration table) and the place the message names.
LEFT = {
    "config_path": ("config_path", "the plan records the file it read"),
    "starfinder_path": ("starfinder_path", "installed starfinder command"),
    "fiji_path": ("fiji_path", "execution.fiji"),
    "envs_path": ("envs_path", "no rule read it"),
    "img_row": ("img_row", "retired"),
    "img_col": ("img_col", "retired"),
    "img_z": ("img_z", "retired"),
    "maximum_projection": ("maximum_projection", "processing.reference_image_view.projection"),
    "workflow_mode": ("workflow_mode", "a present block enables its step"),
    "rules": ("rules", "the step blocks"),
    "matlab_launcher": ("matlab_launcher", "backend: matlab"),
    "matlab_single_thread": ("matlab_single_thread", "backend: matlab"),
    "dapi_round": ("dapi_round", "reference_stains"),
    "channel_order": ("channel_order", "seq_channel_order"),
    "fov_pattern": ("fov_pattern", "fovs.csv"),
}
LEFT_VALUES = {"rules": {"rsf_single_fov": {"run": True}}, "workflow_mode": "free", "img_row": 36, "img_col": 36,
               "img_z": 5, "maximum_projection": False, "matlab_single_thread": True, "dapi_round": "round1",
               "channel_order": ["ch00"], "fov_pattern": "FOV_{i:03d}"}


@pytest.mark.parametrize("key", sorted(LEFT))
def test_each_key_that_left_the_python_dialect(work, capsys, key):
    config = dict(X.variants(work)["molecules-only"], **{key: LEFT_VALUES.get(key, "/somewhere")})
    _refused(work, config, capsys, *LEFT[key])


NESTED = {
    "processing.spot_finding.run": (lambda c: c["processing"]["spot_finding"].update(run=True),
                                    "processing.spot_finding.run"),
    "processing.create_subtiles": (lambda c: c["processing"].update(create_subtiles={"sqrt_pieces": 2}),
                                   "processing.create_subtiles"),
    "assembly.previews": (lambda c: c["assembly"].update(previews=True), "assembly.previews"),
    "assignment.checkpoints": (lambda c: c.update(assignment={"checkpoints": {}}), "assignment.checkpoints"),
}


@pytest.mark.parametrize("key", sorted(NESTED))
def test_each_nested_key_that_left_the_python_dialect(work, capsys, key):
    config = copy.deepcopy(X.variants(work)["molecules-only"])
    change, named = NESTED[key]
    change(config)
    _refused(work, config, capsys, named)


def test_an_unknown_key_is_named(work, capsys):
    _refused(work, dict(X.variants(work)["molecules-only"], spot_size=3), capsys, "'spot_size'")


def _tiled(work, **tiled):
    """The molecules-only variant with its processing tiled (a tile size of 18: 2 x 2 tiles of each 36 x 36 FOV)."""
    config = copy.deepcopy(X.variants(work)["molecules-only"])
    config["processing"].update(mode="tiled", tiled=dict({"tile_size_yx": 18}, **tiled))
    return config


def test_a_tile_size_whose_predicted_peak_exceeds_the_task_memory_is_refused(work, capsys):
    """The resource model (0.21 GB + 1.93 x the window's bytes; 4 rounds of 5 x 36 x 36 x 4 uint8 voxels): a tile
    size of 4096 with a halo of 6 is predicted at 0.21 GB + 1.93 x 4 x 5 x 4108 ** 2 x 4 = 2816 MB, above 1000 MB, and
    the largest multiple of 128 that fits is 2176 (2188 ** 2 x 154.4 + 0.21 GB = 0.949 GB; 2304 does not)."""
    config = _tiled(work, tile_size_yx=4096)
    config["execution"] = {"tasks": {"tile": {"memory_mb": 1000}}}
    _refused(work, config, capsys, "processing.tiled.tile_size_yx 4096", "execution.tasks.tile.memory_mb 1000",
             "predicted to peak at 2816 MB", "tile_size_yx 2176 fits")


def test_a_task_memory_below_the_model_s_intercept_fits_no_tile_size(work, capsys):
    config = _tiled(work)
    config["execution"] = {"tasks": {"tile": {"memory_mb": 200}}}
    _refused(work, config, capsys, "processing.tiled.tile_size_yx 18", "no tile size fits that memory")


@pytest.mark.parametrize("change,named", [
    ({"min_distance_voxels": 2}, "spot_finding.min_distance_voxels 2"),
    ({"merge_radius_zyx": [1, 2, 2]}, "spot_finding.merge_radius_zyx")])
def test_a_detector_without_a_finite_footprint_is_refused_with_exact(work, capsys, change, named):
    config = _tiled(work)
    config["processing"]["spot_finding"].update(change)
    _refused(work, config, capsys, "processing.tiled.exact true", named, "processing.tiled.exact: false")


def test_exact_false_accepts_a_detector_without_a_finite_footprint(work):
    from starfinder.workflow import plan
    config = _tiled(work, exact=False)
    config["processing"]["spot_finding"].update(min_distance_voxels=2)
    manifest = plan(X.write_config(work, "inexact", config))
    assert sum(t["kind"] == "tile" for t in manifest["tasks"]) == 16


def test_a_detector_other_than_local_maxima_is_refused_in_tiled_mode(work, capsys):
    config = _tiled(work, exact=False)
    config["processing"]["spot_finding"] = {"method": "starfish_log", "min_sigma": 1, "max_sigma": 2, "num_sigma": 2,
                                            "threshold": 0.01}
    _refused(work, config, capsys, "spot_finding method 'starfish_log'", "local_maxima only")


def test_a_halo_below_the_footprints_names_the_operation(work, capsys):
    _refused(work, _tiled(work, halo=2), capsys, "processing.tiled.halo 2", "minimum 6", "reads_extraction (6)")


def test_the_tiled_settings_are_refused_in_direct_mode(work, capsys):
    config = copy.deepcopy(X.variants(work)["molecules-only"])
    config["processing"]["tiled"] = {"tile_size_yx": 18}
    _refused(work, config, capsys, "processing.tiled", "processing.mode tiled only")


# --- the refusals of the old rules' parse-time checks (W-750) -------------------------------------------------------

def _one_fov(tmp_path, fovs="sample,fov_id\ns1,P1\n", **keys):
    """The processing block on fovs.csv's FOVs, raw images as empty files."""
    config = W.dataset_keys(tmp_path, processing=copy.deepcopy(X.PROCESSING), **keys)
    W.write_inputs(config, fovs)
    return config


def _refused_plan(tmp_path, config, *parts):
    with pytest.raises(PlanRefused) as refusal:
        W.plan(tmp_path, "refused", config)
    for part in parts:
        assert part in str(refusal.value), (part, str(refusal.value))
    assert not (W.output_root(config) / "workflow").exists()     # the plan writes nothing
    return str(refusal.value)


def test_a_channel_label_that_matches_no_file(tmp_path):
    """From test_segmentation_workflow.py, test_get_dapi_input_needs_exactly_one_file[registration-py.smk] (the
    reference round's stain file of the Python rotate_nuclei rule): a label of a FOV's round that matches no file is
    refused, naming the FOV, the round, the label and the count (two files: test_a_channel_label_that_matches_two_files
    above)."""
    config = _one_fov(tmp_path, reference_stains=[{"channel": "ch04", "name": "DAPI"}], morphology={},
                      additional_round=[{"round_name": "morph", "channel_order": [{"channel": "ch00",
                                                                                     "name": "DAPI"}]}])
    (W.input_root(config) / "round1" / "P1" / "P1_ch04.tif").unlink()
    _refused_plan(tmp_path, config, "FOV P1", "round round1", "'ch04'", "found 0")


@pytest.mark.parametrize("value", ["fovs/{fovID}/reference.tif", "images/x.tif", "", 3])
def test_a_reference_image_without_fov_id_is_refused(tmp_path, value):
    """From test_stitching_reference_image.py, test_a_pattern_without_fov_id_raises_naming_the_setting (the rule
    inputs of the reference image): a reference_image pattern without {fov_id} is refused, naming the setting."""
    _refused_plan(tmp_path, _one_fov(tmp_path, reference_image=value), "reference_image")


LEGACY_SAMPLE_RULES = [
    (["stitching_preparation", "create_tile_config"], {}),
    (["create_sample_h5ad"], {"stitching": {"method": "multiview_stitcher", "positions": X.grid()}}),
    (["create_sample_reads_assignment"], {}),
    (["reads_assignment"], {}),
]


@pytest.mark.parametrize("rules, blocks", LEGACY_SAMPLE_RULES)
def test_the_legacy_sample_rules_are_refused_naming_the_blocks(tmp_path, rules, blocks):
    """From test_fov_membership.py, test_the_legacy_sample_rules_are_refused_for_python_naming_the_blocks: a Python
    configuration names no rule (D4, D7): the plan refuses the rules and names the step blocks that replace them,
    where the parse-time check named the blocks each legacy rule needs."""
    config = _one_fov(tmp_path, rules={rule: {"run": True} for rule in rules}, **blocks)
    _refused_plan(tmp_path, config, "rules left the Python configuration",
                  "the step blocks (processing, morphology, segmentation, assignment, stitching, assembly, export, "
                  "report)")


def test_a_stage_file_without_the_position_ids_of_fovs_csv_is_refused(tmp_path):
    """From test_fov_membership.py, test_the_python_stitching_parse_needs_only_fovs_csv_and_a_stage_file (its
    refusal): a stage file names the FOVs by fovs.csv's position_id; without that column the plan refuses it (V6),
    as the rules' parse did."""
    from .stitching_fixtures import maf_text
    stage = tmp_path / "positions.maf"
    stage.write_text(maf_text(((1, "0.076", "0.048"), (2, "0.076", "0.048261"))))
    config = _one_fov(tmp_path, fovs="sample,fov_id\ns1,F1\ns1,F2\n", reference_image_import=str(
        tmp_path / "refs" / "{fov_id}.tif"), stitching={"method": "multiview_stitcher", "positions": {
            "source": "stage_file", "stage_axes": {"x": "+x", "y": "+y"}, "stage_file": str(stage)}})
    config.pop("processing")
    _refused_plan(tmp_path, config, "position_id column of fovs.csv")


def _route(tmp_path, combination):
    """The route_configs block of the combination (test_sample_segmentation_workflow.py) on the plan's dataset: its
    source images under the test's folder (the relative pattern under the output root, where the export reads it)."""
    from .test_sample_segmentation_workflow import python_route_config
    config = python_route_config(tmp_path, *combination)
    flamingo = tmp_path / "flamingo" / "{fov_id}.tif"
    config["export"]["images"] = [i.replace("/abs/flamingo/{fov_id}.tif", str(flamingo))
                                  for i in config["export"]["images"]]
    fovs = "sample,fov_id\nsample,F1\nsample,F2\n"
    W.write_inputs(config, fovs, codebook="gene,barcode\n")
    out = W.output_root(config)
    for fov in ("F1", "F2"):
        for path in (out / "images" / "dapi" / f"{fov}.tif", Path(str(flamingo).replace("{fov_id}", fov)),
                     tmp_path / f"{fov}.tif"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
    return config


@pytest.mark.parametrize("segment", [True, False])
@pytest.mark.parametrize("fuse", [True, False])
def test_a_configuration_rd1_refuses_names_export_segmentation_and_route_per_fov(tmp_path, segment, fuse):
    """From test_wf1_a_configuration_rd1_refuses_plans_its_earlier_jobs_with_route_per_fov (the route_configs that
    RD1 refuses by its default: no route, dimensionality projection, an assignment, no export.segmentation): the plan
    refuses each, naming export.segmentation and route: per_fov; with route: per_fov added it plans the route per FOV,
    whose export task reads each FOV's saved assignment and no molecule table (it also declares the folder's tables
    and the fusion's source images, D2)."""
    config = _route(tmp_path, (None, "projection", segment, fuse, False, "default"))
    _refused_plan(tmp_path, config, "export.segmentation", "route: per_fov")
    config["export"]["route"] = "per_fov"
    _, manifest = W.plan(tmp_path, "per_fov", config)
    (export,) = [t for t in manifest["tasks"] if t["kind"] == "export"]
    assert "stitching/sample/placement.json" in export["inputs"]
    for fov in ("F1", "F2"):
        assert f"fovs/{fov}/assignment/default/assignment.json" in export["inputs"]
        assert f"fovs/{fov}/molecules.csv" not in export["inputs"]
