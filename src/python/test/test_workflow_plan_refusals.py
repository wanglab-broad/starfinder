"""C5 of design/validation/orchestration.md: the plan's refusals.

The two refused variants of the worked example, a missing round folder, a channel label that matches two files, a
run name with ``/``, each key that left the Python dialect and ``backend: matlab``: ``starfinder workflow plan``
exits 2 with a message that names the key, FOV, round or label, and writes nothing.
"""
import copy
import shutil

import pytest

from starfinder.__main__ import main

from . import workflow_examples as X

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


def test_tiled_processing_waits_for_the_tiled_kinds(work, capsys):
    config = copy.deepcopy(X.variants(work)["molecules-only"])
    config["processing"].update(mode="tiled", tiled={"tiles": 4})
    _refused(work, config, capsys, "processing.mode tiled", "W-735")
