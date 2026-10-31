"""W-564: the FOV folder, the one reference-image resolver and the channel output paths by label.

design/specs/checkpoints.md ("Layout") and docs/migration.md. A FOV's durable results go to
``<output_root>/fovs/<fov_id>/`` (``CheckpointConfig.directory`` names the parent of the FOV
folders), its restart material to the folder's ``checkpoints/``; the reference image is
written and read through ``_layout.reference_image_path`` (default
``fovs/<fov_id>/reference.tif``, or the file ``reference_image_import`` names, which nothing
writes). The cases are the hand-built 4-channel, 2-round FOV of test_checkpoints, the
other-round textures of test_registration_other_rounds and the small views of
test_projection_views; the MATLAB rule files are parsed with a stand-in for Snakemake's
``expand``. The Python rules' declarations of these files moved to the plan's
(test_workflow_declared_outputs.py), the processing rule with an imported reference image to the fov
command and the channel folders by label to the morphology command (test_workflow_fov_tasks.py,
test_workflow_cell_tasks.py), W-750; the Python readers of the retired stitch_subtile (D5) and
create_nuclei_amplicon_overlay (D6) rules are retired.
"""
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pytest

from starfinder import _layout
from starfinder.dataset import CheckpointConfig, Dataset, RoundState, from_workflow_config
from starfinder.dataset._declared import reference_grid, reference_settings
from starfinder.dataset._export import source_path
from starfinder.io import save_volume
from starfinder.segmentation import InputChannel, LabelImportConfig, SegmentationPlan, SegmentationRun

from . import hash_oracles
from .test_checkpoints import dataset, full, resident
from .test_morphology_rounds import ThresholdConfig, threshold  # noqa: F401 (the fixture registers the method)
from .test_projection_views import CHANNELS as VIEW_CHANNELS, METADATA as VIEW_METADATA, reference_round, workflow_config
from .test_registration_other_rounds import SEQUENCING

pytestmark = [pytest.mark.dataset, pytest.mark.workflow]

ROOT = Path(__file__).resolve().parents[3]
RULES = ROOT / "workflow" / "rules"


def tree(root):
    return sorted(p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file())


def hashes(root):
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(root).rglob("*") if p.is_file()}


def segment_and_assign(fov, tmp_path, checkpoints):
    """An imported two-cell label run ``cells`` on the FOV's grid, saved, and the assignment ``default`` on it."""
    grid = fov.reference_grid()
    labels = np.zeros(grid.shape_zyx, np.uint32)
    labels[:, 2:7, 3:8] = 1
    labels[:, 6:11, 8:13] = 2
    path = tmp_path / "masks" / "cells.tif"
    save_volume(labels, path, metadata=grid.metadata)
    plan = SegmentationPlan((SegmentationRun("cells", "cell", (), LabelImportConfig(str(path), "cell")),))
    fov.segment(plan, checkpoints=checkpoints)
    fov.assign_molecules(cells="cells", checkpoints=checkpoints)


# --- The FOV folder ------------------------------------------------------------------------------------------

FOV_TREE = [
    "fovs/FOV/assignment/default/assignment.json", "fovs/FOV/assignment/default/cells.csv",
    "fovs/FOV/assignment/default/counts.csv", "fovs/FOV/assignment/default/molecules.csv",
    "fovs/FOV/checkpoints/candidates.csv", "fovs/FOV/checkpoints/candidates.json",
    "fovs/FOV/checkpoints/pre_qc.csv", "fovs/FOV/checkpoints/pre_qc.json",
    "fovs/FOV/checkpoints/registered/round1.ome.tif", "fovs/FOV/checkpoints/registered/round2.ome.tif",
    "fovs/FOV/checkpoints/registered/transforms.json",
    "fovs/FOV/molecules.csv", "fovs/FOV/reference.tif", "fovs/FOV/run.json",
    "fovs/FOV/segmentation/cells/labels.tif", "fovs/FOV/segmentation/cells/segmentation.json",
]


@pytest.mark.contract
def test_a_checkpointed_fov_writes_the_exact_tree_of_its_folder(tmp_path):
    """FOV.run with checkpoints, export_spots, save_reference_image, a segmentation and an assignment write the
    FOV folder and nothing else; run.json lists its stage files relative to the folder, and the assignment links
    the saved labels of its sibling segmentation folder."""
    ds = dataset(tmp_path)
    checkpoints = CheckpointConfig()
    fov = resident(ds).run(full(), checkpoints=checkpoints)
    molecules = fov.export_spots()
    reference = fov.save_reference_image()
    segment_and_assign(fov, tmp_path, checkpoints)

    out = tmp_path / "out"
    assert tree(out) == FOV_TREE
    folder = out / "fovs" / "FOV"
    assert (fov.paths.fov_dir, fov.paths.checkpoint_dir, fov.paths.run_record) == (
        folder, folder / "checkpoints", folder / "run.json")
    assert (molecules, reference) == (folder / "molecules.csv", folder / "reference.tif")
    record = json.loads((folder / "run.json").read_text())
    assert record["checkpoint_directory"] == "checkpoints"
    # run.json names each stage record; each record names its payload files relative to its folder.
    listed = []
    for entry in record["checkpoints"].values():
        listed.append(entry["path"])
        parent = Path(entry["path"]).parent
        listed += [(parent / f["path"]).as_posix() for f in json.loads((folder / entry["path"]).read_text())["files"]]
    assert sorted(listed) == sorted(name[len("fovs/FOV/"):] for name in FOV_TREE if "/checkpoints/" in name)
    assert record["outputs"]["molecules"]["path"] == "molecules.csv"
    saved = json.loads((folder / "assignment" / "default" / "assignment.json").read_text())
    assert saved["inputs"]["cells"]["file"]["path"] == "../../segmentation/cells/labels.tif"
    # Every reader finds the results where they were written.
    again = ds.fov("FOV").load_checkpoint("candidates").load_checkpoint("pre_qc")
    assert again.load_segmentation("cells").labels.max() == 2
    assert len(again.load_assignment("default").molecules) == len(fov.assignment_results["default"].molecules)


def test_the_overwrite_guard_covers_run_json_and_checkpoints_only(tmp_path):
    """Without overwrite an existing run.json or checkpoints/ folder stops a run before processing; the other
    folders of the FOV folder do not. With overwrite, only checkpoints/ is cleared."""
    ds = dataset(tmp_path)
    fov = resident(ds).run(full(), checkpoints=CheckpointConfig())
    fov.save_reference_image()
    segment_and_assign(fov, tmp_path, CheckpointConfig())
    folder = tmp_path / "out" / "fovs" / "FOV"
    with pytest.raises(FileExistsError, match="overwrite"):
        resident(ds).run(full(), checkpoints=CheckpointConfig())
    kept = {p: h for p, h in hashes(folder).items() if "checkpoints" not in p.parts and p.name != "run.json"}
    resident(ds).run(full(), checkpoints=CheckpointConfig(stages=("candidates",), overwrite=True))
    assert tree(folder / "checkpoints") == ["candidates.csv", "candidates.json"]
    assert {p: h for p, h in hashes(folder).items() if p in kept} == kept
    # A FOV folder with results of other calls but no run.json nor checkpoints/ takes a run without overwrite.
    (folder / "run.json").unlink()
    for path in sorted((folder / "checkpoints").rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    (folder / "checkpoints").rmdir()
    resident(ds).run(full(), checkpoints=CheckpointConfig(stages=()))
    assert (folder / "run.json").is_file() and {p: h for p, h in hashes(folder).items() if p in kept} == kept


def test_checkpoint_directory_names_the_parent_of_the_fov_folders(tmp_path):
    """CheckpointConfig.directory replaces <output_root>/fovs (the subtile folders subtile_<n> are retired, W-735)."""
    ds = dataset(tmp_path)
    checkpoints = CheckpointConfig(directory=tmp_path / "elsewhere", stages=("candidates",))
    resident(ds).run(full(), checkpoints=checkpoints)
    assert tree(tmp_path / "elsewhere") == [
        "FOV/checkpoints/candidates.csv", "FOV/checkpoints/candidates.json", "FOV/run.json"]
    assert not (tmp_path / "out").exists()
    assert _layout.run_record("/o", "F") == Path("/o/fovs/F/run.json")


# --- The reference image ----------------------------------------------------------------------------------

def reference_fov(root, **fields):
    ds = Dataset(root, root / "out", "data", "sample", "run", RoundState(["round1"], reference_round="round1"),
                 VIEW_CHANNELS, **fields)
    fov = ds.fov("FOV")
    fov.images = {"round1": reference_round(np.uint16)}
    fov.metadata = {"round1": VIEW_METADATA}
    return fov


def merged_segmentation_record(ds):
    """The input record of a threshold run on the reference merged image read from the saved reference image."""
    fresh = ds.fov("FOV").load_reference_image()
    run = SegmentationRun("merged", "nucleus", (InputChannel("nuclear", reference_merged=True),), ThresholdConfig(1.0))
    fresh.segment(SegmentationPlan((run,)))
    (entry,) = fresh.segmentation_results["merged"].record["input"]["channels"]
    return fresh, entry["reference_image"]


@pytest.mark.contract
@pytest.mark.parametrize("setting", ["default", "relative", "absolute"])
def test_the_reference_image_is_written_and_read_where_reference_image_names(tmp_path, threshold, setting):
    """The writer, FOV.load_reference_image, the grid and export readers, the workflow declarations and the
    segmentation record resolve one file: the default, a relative pattern under the output root, or an absolute
    one outside it, which the segmentation record keeps by its absolute path and SHA-256."""
    pattern = {"default": None, "relative": "refs/{fov_id}_ref.tif",
               "absolute": str(tmp_path / "elsewhere" / "{fov_id}.tif")}[setting]
    expected = {"default": tmp_path / "out" / "fovs" / "FOV" / "reference.tif",
                "relative": tmp_path / "out" / "refs" / "FOV_ref.tif",
                "absolute": tmp_path / "elsewhere" / "FOV.tif"}[setting]
    fov = reference_fov(tmp_path, **({} if pattern is None else {"reference_image": pattern}))
    ds = fov.dataset
    path = fov.save_reference_image()
    assert path == expected == _layout.reference_image_path(ds.output_root, "FOV", ds.reference_image)
    assert tree(tmp_path / "out") == ([] if setting == "absolute" else [expected.relative_to(tmp_path / "out")
                                                                       .as_posix()])
    fresh, linked = merged_segmentation_record(ds)
    assert fresh.reference_grid().source == f"file:{path.resolve()}"
    assert reference_grid(ds.output_root, "FOV", **reference_settings(ds)) == fresh.reference_grid()
    assert source_path(ds, "FOV", "reference_merged") == path
    recorded = str(path) if setting == "absolute" else path.relative_to(ds.output_root).as_posix()
    assert linked == {"path": recorded, "sha256": hash_oracles.file_sha256(path)}
    # The plan declares the same file (the fov task's output, its readers' input): test_workflow_declared_outputs.py,
    # test_the_reference_image_is_declared_where_reference_image_names_it (W-750).


@pytest.mark.contract
@pytest.mark.parametrize("where", ["relative", "absolute"])
def test_an_imported_reference_image_is_read_by_every_reader_and_never_written(tmp_path, threshold, where):
    """reference_image_import names a file another tool wrote: FOV.save_reference_image refuses to write it,
    every reader reads it, and the workflow declares it as an input and as no output."""
    pattern = {"relative": "imports/{fov_id}.tif", "absolute": str(tmp_path / "tool" / "{fov_id}.tif")}[where]
    imported = {"relative": tmp_path / "out" / "imports" / "FOV.tif", "absolute": tmp_path / "tool" / "FOV.tif"}[where]
    save_volume(reference_round(np.uint16, seed=3).max(axis=-1), imported, metadata=VIEW_METADATA)
    before = imported.read_bytes()
    fov = reference_fov(tmp_path, reference_image_import=pattern)
    ds = fov.dataset
    with pytest.raises(ValueError, match="imported"):
        fov.save_reference_image()
    assert imported.read_bytes() == before
    assert tree(tmp_path / "out") == ([] if where == "absolute" else ["imports/FOV.tif"])
    assert _layout.reference_image_path(ds.output_root, "FOV", ds.reference_image, ds.reference_image_import) == \
        imported
    fresh, linked = merged_segmentation_record(ds)
    assert fresh.reference_grid().source == f"file:{imported.resolve()}"
    assert reference_grid(ds.output_root, "FOV", **reference_settings(ds)) == fresh.reference_grid()
    assert source_path(ds, "FOV", "reference_merged") == imported
    recorded = str(imported) if where == "absolute" else "imports/FOV.tif"
    assert linked == {"path": recorded, "sha256": hash_oracles.file_sha256(imported)}
    # The plan declares it as an input of its readers and as no output: test_workflow_declared_outputs.py,
    # test_the_reference_image_is_declared_where_reference_image_names_it (W-750).


@pytest.mark.contract
def test_the_two_reference_settings_exclude_each_other(tmp_path):
    with pytest.raises(ValueError, match="reference_image_import .* excludes reference_image"):
        Dataset(tmp_path, tmp_path / "out", "d", "s", "o", reference_image="a/{fov_id}.tif",
                reference_image_import="b/{fov_id}.tif")
    with pytest.raises(ValueError, match="reference_image_import must be a file pattern with"):
        Dataset(tmp_path, tmp_path / "out", "d", "s", "o", reference_image_import="b/fixed.tif")
    with pytest.raises(ValueError, match="excludes"):
        reference_settings({"dataset_id": "d", "output_id": "o", "reference_image": "a/{fov_id}.tif",
                            "reference_image_import": "b/{fov_id}.tif"})
    config = workflow_config(tmp_path)
    assert from_workflow_config(dict(config, reference_image_import="b/{fov_id}.tif")).dataset.reference_image_import \
        == "b/{fov_id}.tif"
    with pytest.raises(ValueError, match="excludes"):
        from_workflow_config(dict(config, reference_image="a/{fov_id}.tif", reference_image_import="b/{fov_id}.tif"))


# --- Channel output paths by label ------------------------------------------------------------------------

def nuclei_rule_config(tmp_path, morph):
    """The translator's view of a Python configuration with the other round ``morph`` (its channel_order)."""
    return dict(starfinder_path=str(ROOT), root_input_path=str(tmp_path / "input"), dataset_id="dataset",
                sample_id="sample", root_output_path=str(tmp_path / "output"), output_id="output",
                n_rounds=2, ref_round="round1", seq_channel_order=list(SEQUENCING[:3]),
                reference_stains=[dict(channel=SEQUENCING[3], name="DAPI")],
                ref_channel="DAPI", rotate_angle=0, maximum_projection=False, backend="python",
                additional_round=[dict(round_name="morph", channel_order=morph)],
                rules=dict(nuclei_registration=dict(run=True)))


@pytest.mark.contract
@pytest.mark.parametrize("label", ["a/b", "../ch00", ".hidden", "ch 00", "*ch00", ""])
def test_an_unsafe_channel_label_is_rejected_when_the_dataset_is_built(tmp_path, label):
    error = ValueError
    match = "nonempty" if label == "" else "not safe single path components"
    with pytest.raises(error, match=match):
        Dataset(tmp_path, tmp_path / "out", "d", "s", "o", RoundState(["round1"], ["morph"], "round1"), ("ch01",),
                other_channel_order={"morph": ("ch00", label)})
    with pytest.raises(error, match=match):
        Dataset(tmp_path, tmp_path / "out", "d", "s", "o", RoundState(["round1"], reference_round="round1"), (label,))
    if label:
        config = nuclei_rule_config(tmp_path, [dict(channel="ch00", name="DAPI"), dict(channel=label, name="Nissl")])
        with pytest.raises(error, match=match):
            from_workflow_config(config)


# --- The rule files of both backends ----------------------------------------------------------------------

def expand(pattern, **values):
    """Snakemake's expand on one pattern: every combination of the values, ``{{x}}`` kept as ``{x}``."""
    keys = list(values)
    lists = [value if isinstance(value, (list, tuple)) else [value] for value in values.values()]
    return [pattern.format(**dict(zip(keys, combination))) for combination in itertools.product(*lists)]


def rule_section(path, rule, section, namespace):
    """One section (input or output) of a rule, evaluated as Snakemake evaluates it at parse time: (positional
    entries, named entries), each as the rule declares it (a string, a list or an input function)."""
    text = path.read_text()
    block = text[text.index(f"rule {rule}:"):]
    end = re.search(r"\n(?=\S)", block[1:])
    block = block if end is None else block[:end.start() + 1]
    body = re.search(rf"^    {section}:\n((?:        .*\n)+)", block, re.M).group(1)
    return eval(f"_collect({body.strip()})", {**namespace, "_collect": lambda *a, **k: (a, k), "expand": expand,
                                             "temp": lambda value: value})


def rule_files(path, rule, section, namespace):
    """The files of one section of a rule: (positional files flattened, named entries as declared)."""
    args, kwargs = rule_section(path, rule, section, namespace)
    flat = [str(f) for item in args for f in (item if isinstance(item, list) else [item])]
    return flat, kwargs


def common_value(name):
    """The MATLAB value common.smk gives a variable before the Python block: its literal assignment, evaluated."""
    text = (RULES / "common.smk").read_text()
    match = re.search(rf"^{name} = (.*?)\n(?=\S|\Z)", text, re.M | re.S)
    return match.group(1)


def backend_namespace(tmp_path, backend="matlab", **keys):
    """The names the MATLAB rule files read from common.smk, for a configuration under tmp_path."""
    output = tmp_path / "data" / "run"
    config = dict(dataset_id="data", output_id="run", backend=backend, n_rounds=1, ref_round="round1",
                  root_input_path=str(tmp_path), root_output_path=str(tmp_path), sample_id="sample",
                  seq_channel_order=["ch01"], additional_round=[dict(round_name="flamingo", channel_order=[
                      dict(channel="ch00", name="DAPI"), dict(channel="ch03", name="Flamingo")])], **keys)
    namespace = dict(OUTPUT_DIR=output, INPUT_DIR=tmp_path / "in", ROUND=["round1"], N_SUBTILE=[1], config=config)
    namespace["SHARED_IMAGE_INPUTS"] = eval(common_value("SHARED_IMAGE_INPUTS"), dict(OUTPUT_DIR=output))
    return namespace, output


@pytest.mark.contract
def test_the_matlab_backend_keeps_its_rules_and_the_python_backend_follows_the_layout(tmp_path):
    """registration.smk and the shared segmentation.smk parsed for MATLAB give MATLAB's files (images/ref_merged/,
    the name folders of the flamingo round, the logs). (The Python workflow's tasks declare the FOV folder's run.json
    and reference image, none with reference_image_import: test_workflow_declared_outputs.py,
    test_the_reference_image_is_declared_where_reference_image_names_it, W-750; the label folders of the flamingo
    round that enhance_dapi_with_flamingo read are retired with the Python branch of that rule, D6.)"""
    matlab, out = backend_namespace(tmp_path, "matlab")
    assert rule_files(RULES / "registration.smk", "rsf_single_fov", "output", matlab)[0] == [
        f"{out}/log/{{fovID}}_rsf.txt", f"{out}/log/sf_scores/{{fovID}}.txt", f"{out}/images/ref_merged/{{fovID}}.tif",
        f"{out}/signal/{{fovID}}_goodSpots.csv"]
    assert rule_files(RULES / "registration.smk", "gr_single_fov_subtile", "output", matlab)[0][:2] == [
        f"{out}/log/{{fovID}}_gr.txt", f"{out}/images/ref_merged/{{fovID}}.tif"]
    segmentation = RULES / "segmentation.smk"
    assert rule_files(segmentation, "enhance_dapi_with_flamingo", "input", matlab)[1] == {
        "dapi_img": f"{out}/images/flamingo/DAPI/{{fovID}}.tif",
        "flamingo_img": f"{out}/images/flamingo/Flamingo/{{fovID}}.tif"}
    assert rule_files(segmentation, "create_nuclei_amplicon_overlay", "input", matlab)[1] == {
        "dapi_img": [f"{out}/images/DAPI/{{fovID}}.tif"], "amplicon_img": f"{out}/images/ref_merged/{{fovID}}.tif"}
