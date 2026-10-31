"""The inputs of the Python workflow (W-567): one gene authority, the reference-file keys and image_dir_template.

OM-13: the reference file of the sequencing rules is the file the Python-only key ``codebook``
(``direct_panel`` in readout_mode direct) names, absolute or relative to the input root, else
``<input_root>/genes.csv``; the translators ``_reference_file``, ``_reference_genes`` and ``_export_dataset`` read
it, and the count matrix's genes are the loaded codebook's (or panel's) in its order. NM-11: the acquisition
images are laid out round-first (``"{round}/{fov}"``, the default) or FOV-first (``"{fov}/{round}"``);
``FOV.input_dir`` follows ``image_dir_template``, and the MATLAB rules stay round-first. The package validates the
keys of a Python configuration (the Python schema, as the plan does; W-750), and the plan's inputs follow the template
and the codebook (test_workflow_plan.py, W-750); the MATLAB schema refuses the three keys without ``backend: python``
(OC5). Fixtures, all written in the test: a two-FOV scene of two rounds stored in both layouts, and the repository's
minimal example configuration with a three-gene codebook. The MATLAB rule files are evaluated in part, as in
``test_fov_membership.py`` and ``test_dataset_fov_layout.py``; Snakemake itself never runs.
"""
import copy
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import tifffile
import yaml

from starfinder import _layout
from starfinder.dataset import Dataset, RoundState, from_workflow_config
from starfinder.dataset.types import REFERENCE_STAIN
from starfinder.dataset.workflow import _assembly_block, _export_dataset, _reference_file, _reference_genes

from . import workflow_tasks as T
from .test_dataset_fov_layout import rule_files
from .test_fov_membership import MULTIVIEW, common_names, matlab_config, python_example

pytestmark = [pytest.mark.workflow, pytest.mark.dataset, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
RULES = ROOT / "workflow" / "rules"
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
ROUNDS = ("round1", "round2")
CHANNELS = ("ch00", "ch01", "ch02", "ch03")
FOVS = ("Position001", "Position002")
# A codebook of the minimal example (four rounds, end bases CC) whose gene order is not alphabetical.
CODEBOOK = "Vip,CCAAC\nGad1,CAACC\nSst,CACAC\n"


def scene():
    """The scene's channel arrays by (round, FOV): the four colours, and in the reference round the stain ch04."""
    rng = np.random.default_rng(567)
    return {(r, f): {c: rng.integers(0, 255, (3, 12, 10), dtype=np.uint8)
                     for c in CHANNELS + (("ch04",) if r == "round1" else ())}
            for r in ROUNDS for f in FOVS}


def store(root, arrays, template):
    """Write the scene under root in the layout of template; returns the folders written."""
    for (round_name, fov_id), channels in arrays.items():
        folder = _layout.image_dir(root, round_name, fov_id, template)
        folder.mkdir(parents=True, exist_ok=True)
        for label, array in channels.items():
            tifffile.imwrite(folder / f"{fov_id}_{label}.tif", array)
    return sorted(p.relative_to(root).as_posix() for p in root.glob("*/*"))


def dataset(root, **fields):
    return Dataset(root, root.parent / "output", "data", "sample", "out",
                   RoundState(list(ROUNDS), reference_round="round1"), CHANNELS, reference_stains=("ch04",), **fields)


# --- image_dir_template ------------------------------------------------------------------------------------------

def test_both_layouts_load_the_same_images(tmp_path):
    """NM-11: one scene stored round-first and FOV-first loads as the same arrays, sequencing rounds and the
    reference stain, each equal to the arrays written."""
    arrays = scene()
    loaded = {}
    for template, name in zip(_layout.IMAGE_DIR_TEMPLATES, ("round_first", "fov_first")):
        root = tmp_path / name
        folders = store(root, arrays, template)
        first, second = (ROUNDS, FOVS) if template == "{round}/{fov}" else (FOVS, ROUNDS)
        assert folders == [f"{a}/{b}" for a in first for b in second]
        ds = dataset(root, image_dir_template=template)
        for fov_id in FOVS:
            fov = ds.fov(fov_id)
            assert fov.input_dir("round2") == root / template.format(round="round2", fov=fov_id)
            fov.load_images().load_images(rounds=[REFERENCE_STAIN])
            loaded[name, fov_id] = {r: fov.images[r] for r in (*ROUNDS, REFERENCE_STAIN)}
    for fov_id in FOVS:
        for round_name in ROUNDS:
            expected = np.stack([arrays[round_name, fov_id][c] for c in CHANNELS], axis=-1)
            assert np.array_equal(loaded["round_first", fov_id][round_name], expected)
            assert np.array_equal(loaded["fov_first", fov_id][round_name], expected)
        stain = arrays["round1", fov_id]["ch04"][..., None]
        assert np.array_equal(loaded["round_first", fov_id][REFERENCE_STAIN], stain)
        assert np.array_equal(loaded["fov_first", fov_id][REFERENCE_STAIN], stain)


def test_the_default_layout_is_round_first_and_a_layout_reads_only_its_own_folders(tmp_path):
    arrays = scene()
    store(tmp_path / "in", arrays, "{fov}/{round}")
    assert dataset(tmp_path / "in").image_dir_template == "{round}/{fov}"
    with pytest.raises(FileNotFoundError):
        dataset(tmp_path / "in").fov(FOVS[0]).load_images()


@pytest.mark.parametrize("template", ["{round}_{fov}", "{fov}/{round}/raw", "{round}", "", None, "{FOV}/{ROUND}"])
def test_an_unknown_image_dir_template_raises(tmp_path, template):
    with pytest.raises(ValueError, match=r"image_dir_template must be one of '\{round\}/\{fov\}', '\{fov\}/\{round\}'"):
        dataset(tmp_path, image_dir_template=template)
    with pytest.raises(ValueError, match="image_dir_template must be one of"):
        _layout.image_dir(tmp_path, "round1", FOVS[0], template)
    config = python_example(tmp_path, image_dir_template=template)
    if template is None:  # null is the default
        assert from_workflow_config(config).dataset.image_dir_template == "{round}/{fov}"
    else:
        with pytest.raises(ValueError, match="image_dir_template must be one of"):
            from_workflow_config(config)
        assert T.validation_error(T.dataset_keys(tmp_path, image_dir_template=template))


def test_the_workflow_key_sets_the_dataset_field_and_needs_the_python_backend(tmp_path):
    config = python_example(tmp_path, image_dir_template="{fov}/{round}")
    assert T.validation_error(T.dataset_keys(tmp_path, image_dir_template="{fov}/{round}")) is None
    assert from_workflow_config(config).dataset.image_dir_template == "{fov}/{round}"
    matlab = dict(config, backend="matlab")
    with pytest.raises(ValueError, match="image_dir_template is Python-only and need backend: python"):
        from_workflow_config(matlab)


# --- The reference file: codebook, direct_panel and <input_root>/genes.csv ---------------------------------------

def input_root(config):
    return Path(config["root_input_path"]) / config["dataset_id"] / config["sample_id"]


@pytest.mark.parametrize("given", [None, "relative", "absolute"])
def test_the_codebook_key_names_the_reference_file_else_genes_csv(tmp_path, given):
    supplied = tmp_path / "panels" / "codebook-v2.csv"
    value = {None: None, "relative": "../../../panels/codebook-v2.csv", "absolute": str(supplied)}[given]
    config = python_example(tmp_path, **({} if value is None else {"codebook": value}))
    expected = input_root(config) / "genes.csv" if value is None else input_root(config) / value
    assert _reference_file(config) == expected
    if value is not None:
        assert expected.resolve() == supplied.resolve()
        assert T.validation_error(T.dataset_keys(tmp_path, codebook=value)) is None


def test_the_direct_panel_key_names_the_panel_of_readout_mode_direct(tmp_path):
    config = python_example(tmp_path, readout_mode="direct", direct_panel="panel.csv")
    assert _reference_file(config) == input_root(config) / "panel.csv"
    assert _reference_file(dict(config, direct_panel=None)) == input_root(config) / "genes.csv"
    with pytest.raises(ValueError, match="codebook names the file of readout_mode multiplexed; readout_mode direct "
                                         "reads direct_panel"):
        _reference_file(dict(config, codebook="genes.csv"))
    with pytest.raises(ValueError, match="direct_panel names the file of readout_mode direct; readout_mode "
                                         "multiplexed reads codebook"):
        _reference_file(python_example(tmp_path, direct_panel="panel.csv"))


@pytest.mark.parametrize("key", ["codebook", "direct_panel", "image_dir_template"])
def test_the_matlab_backend_refuses_the_python_input_keys(tmp_path, monkeypatch, key):
    """MATLAB reads <INPUT>/genes.csv round-first: the MATLAB schema refuses the keys for backend matlab and without
    backend (OC5: the refusal moved from common.smk to the schema's Python-only clause, which the MATLAB rule set
    validates first; null stays accepted), and so does the package; without them CODEBOOK is <INPUT>/genes.csv."""
    config = matlab_config(tmp_path)
    assert common_names(config, monkeypatch)["CODEBOOK"] == f"{input_root(config)}/genes.csv"
    value = "{fov}/{round}" if key == "image_dir_template" else "other.csv"
    assert jsonschema.Draft202012Validator(SCHEMA).is_valid(config)
    assert jsonschema.Draft202012Validator(SCHEMA).is_valid(dict(config, **{key: None}))
    for backend in ({"backend": "matlab"}, {}):
        errors = list(jsonschema.Draft202012Validator(SCHEMA).iter_errors(dict(config, **{key: value}, **backend)))
        assert errors and all(e.validator == "not" for e in errors), key
    if key != "image_dir_template":
        with pytest.raises(ValueError, match=f"{key} is Python-only"):
            _reference_file(dict(config, **{key: value}))
    # The MATLAB rule set itself refuses them (the base's refusal in common.smk, W-750): legacy.smk validates the
    # configuration against that schema before it includes common.smk or any rule file.
    legacy = (RULES / "legacy.smk").read_text()
    validation = legacy.index('\nvalidate(config, schema="../schemas/config.schema.yaml")\n')
    assert validation < legacy.index('\ninclude: "common.smk"\n') == legacy.index("\ninclude: ")


def test_the_matlab_rules_read_the_round_first_folders():
    """From test_the_python_rule_inputs_follow_the_template, both cases: MATLAB's rules stay round-first, whatever
    image_dir_template the Python workflow reads (registration.smk's two image folder inputs and the stain file of
    rotate_nuclei)."""
    matlab = (RULES / "registration.smk").read_text()
    assert matlab.count('expand("{input_dir}/{rounds}/{{fovID}}", input_dir=INPUT_DIR, rounds=ROUND)') == 2
    assert 'pattern = f"{INPUT_DIR}/{config[\'dapi_round\']}/{wildcards.fovID}/*ch04.tif"' in matlab


def test_the_rules_read_the_configured_codebook(tmp_path, monkeypatch):
    """CODEBOOK is <INPUT>/genes.csv for the MATLAB rule set: the codebook input of the shared reads_assignment rule,
    right after its molecule table. (The Python workflow's tasks declare the configured codebook:
    test_workflow_plan.py, test_the_plan_declares_the_configured_codebook, W-750.)"""
    config = matlab_config(tmp_path)
    names = common_names(config, monkeypatch, checks=True)
    expected = f"{input_root(config)}/genes.csv"
    assert names["CODEBOOK"] == expected
    names.update(READS_ASSIGNMENT_GEOMETRY=[])
    inputs = rule_files(RULES / "reads-assignment.smk", "reads_assignment", "input", names)[0]
    assert inputs[inputs.index(names["MOLECULE_TABLE"]) + 1] == expected
    assert not [f for f in inputs if "documents/genes" in f]


def write_codebook(config, relative):
    path = input_root(config) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CODEBOOK)
    return path


def test_the_genes_and_the_export_codebook_come_from_the_configured_file(tmp_path):
    """_reference_genes and _export_dataset load the configured codebook; the genes keep its order."""
    config = python_example(tmp_path, codebook="panels/codebook.csv", stitching=MULTIVIEW,
                            assembly={"assignment": None})
    path = write_codebook(config, "panels/codebook.csv")
    assert _reference_genes(config) == ("Vip", "Gad1", "Sst") == _reference_genes(config, path)
    dataset = _export_dataset(config, _assembly_block(config))
    assert dataset.codebook.genes == ["Vip", "Gad1", "Sst"]
    assert Path(dataset._reference_files["codebook"][1]) == path.absolute()
    # Without the key, <input_root>/genes.csv; a missing file raises rather than falling back.
    default = copy.deepcopy(config)
    del default["codebook"]
    with pytest.raises(FileNotFoundError):
        _reference_genes(default)
    write_codebook(default, "genes.csv")
    files = _export_dataset(default, _assembly_block(default))._reference_files
    assert Path(files["codebook"][1]) == (input_root(default) / "genes.csv").absolute()


def test_the_genes_of_a_direct_panel_are_in_its_table_order(tmp_path):
    config = python_example(tmp_path, readout_mode="direct", direct_panel="panel.csv")
    config["rules"]["rsf_single_fov"]["parameters"]["spot_finding"]["rounds"] = ["round1", "round2"]
    config["rules"]["rsf_single_fov"]["parameters"]["reads_filtration"] = {"run": True}
    path = input_root(config) / "panel.csv"
    path.parent.mkdir(parents=True)
    path.write_text("round,channel,gene_id\nround2,ch01,Sst\nround1,ch00,Vip\nround1,ch03,Gad1\n")
    assert _reference_genes(config) == ("Sst", "Vip", "Gad1")


def test_a_matlab_configuration_without_channel_order_loads_its_codebook_for_the_genes(tmp_path):
    """MATLAB's default channel order names no labels; the genes do not depend on them."""
    config = matlab_config(tmp_path, root_input_path=str(tmp_path / "input"))
    assert not config.get("seq_channel_order")
    path = input_root(config) / "genes.csv"
    path.parent.mkdir(parents=True)
    rounds = config["n_rounds"]
    path.write_text("".join(f"{g},{b}\n" for g, b in zip(("Sst", "Gad1", "Vip"),
                                                         ("CA" + "A" * (rounds - 1), "GA" + "A" * (rounds - 1),
                                                          "TA" + "A" * (rounds - 1)))))
    assert _reference_genes(config, path) == ("Sst", "Gad1", "Vip")
