"""W-564: the FOV folder, the one reference-image resolver and the channel output paths by label.

docs/checkpoints.md ("Layout") and docs/migration.md. A FOV's durable results go to
``<output_root>/fovs/<fov_id>/`` (``CheckpointConfig.directory`` names the parent of the FOV
folders), its restart material to the folder's ``checkpoints/``; the reference image is
written and read through ``_layout.reference_image_path`` (default
``fovs/<fov_id>/reference.tif``, or the file ``reference_image_import`` names, which nothing
writes); the Python ``nuclei_registration`` rule names each channel's folder by its unique
channel label. The cases are the hand-built 4-channel, 2-round FOV of test_checkpoints, the
other-round textures of test_registration_other_rounds and the small views of
test_projection_views; the rule files are parsed for both backends with a stand-in for
Snakemake's ``expand`` (Snakemake is not installed in the test environment).
"""
import hashlib
import itertools
import json
import re
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from starfinder import _layout
from starfinder._records import file_sha256
from starfinder._snakemake import (_reference_image_inputs, _reference_image_outputs, _run_record_outputs,
                                   _shared_image_inputs)
from starfinder.dataset import CheckpointConfig, Dataset, RoundState, from_workflow_config
from starfinder.dataset._declared import reference_grid, reference_settings
from starfinder.dataset._export import source_path
from starfinder.image import ImageMetadata
from starfinder.io import save_volume
from starfinder.segmentation import InputChannel, LabelImportConfig, SegmentationPlan, SegmentationRun

from .test_checkpoints import dataset, full, resident
from .test_morphology_rounds import ThresholdConfig, threshold  # noqa: F401 (the fixture registers the method)
from .test_projection_views import CHANNELS as VIEW_CHANNELS, METADATA as VIEW_METADATA, SINGLE, reference_round, \
    workflow_config
from .test_registration_other_rounds import OTHER, SEQUENCING, rounds

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
    """CheckpointConfig.directory replaces <output_root>/fovs; a subtile's folder is subtile_<n> of its FOV's."""
    ds = dataset(tmp_path)
    checkpoints = CheckpointConfig(directory=tmp_path / "elsewhere", stages=("candidates",))
    resident(ds).run(full(), checkpoints=checkpoints)
    subtile = resident(ds)
    subtile.subtile_id = 2
    subtile.run(full(), checkpoints=checkpoints)
    assert tree(tmp_path / "elsewhere") == [
        "FOV/checkpoints/candidates.csv", "FOV/checkpoints/candidates.json", "FOV/run.json",
        "FOV/subtile_2/checkpoints/candidates.csv", "FOV/subtile_2/checkpoints/candidates.json",
        "FOV/subtile_2/run.json"]
    assert not (tmp_path / "out").exists()
    assert _layout.run_record("/o", "F", 3) == Path("/o/fovs/F/subtile_3/run.json")


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
    assert linked == {"path": recorded, "sha256": file_sha256(path)}
    config = {"dataset_id": "data", "output_id": "run", **({} if pattern is None else {"reference_image": pattern})}
    declared = str(_layout.reference_image_path(ds.output_root, "{fovID}", ds.reference_image))
    assert _reference_image_outputs(config, ds.output_root, ["{fovID}"]) == [declared]
    assert _reference_image_inputs(config, ds.output_root, ["{fovID}"]) == [declared]


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
    assert linked == {"path": recorded, "sha256": file_sha256(imported)}
    config = {"dataset_id": "data", "output_id": "run", "reference_image_import": pattern}
    assert _reference_image_outputs(config, ds.output_root, ["{fovID}"]) == []
    assert _reference_image_inputs(config, ds.output_root, ["{fovID}"]) == [
        str(_layout.reference_image_path(ds.output_root, "{fovID}", imported=pattern))]


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


def test_with_reference_image_import_the_processing_rule_writes_no_reference_image(tmp_path):
    """The Python rsf_single_fov writes run.json and molecules.csv in the FOV folder and no reference image; the
    imported file is left as it is."""
    raw = reference_round(np.uint16, seed=1)
    for name in ("round1", "round2"):
        for c, channel in enumerate(VIEW_CHANNELS):
            save_volume(raw[..., c], tmp_path / "data" / "sample" / name / "FOV" / f"{channel}.tif",
                        metadata=VIEW_METADATA)
    codebook = tmp_path / "genes.csv"
    codebook.write_text("gene,barcode\ngene,AAA\n")
    imported = tmp_path / "tool" / "FOV.tif"
    save_volume(raw.max(axis=-1), imported, metadata=VIEW_METADATA)
    before = imported.read_bytes()
    config = workflow_config(tmp_path, enhance_contrast={"run": True}, global_registration=SINGLE,
                             spot_finding={"run": True, "intensity_estimation": "adaptive", "intensity_threshold": .1},
                             reads_extraction={"run": True, "voxel_size": [0, 0, 0]}, reads_filtration={"run": True})
    config.update(starfinder_path=str(ROOT), fov_id_pattern="%s", reference_image_import=str(tmp_path / "tool" /
                                                                                              "{fov_id}.tif"))
    snakemake = SimpleNamespace(config=config, input=["unused", str(codebook)], wildcards=SimpleNamespace(fovID="FOV"))
    runpy.run_path(str(ROOT / "workflow" / "scripts" / "rsf_single_fov.py"), init_globals={"snakemake": snakemake})
    assert tree(tmp_path / "out" / "data" / "run") == ["fovs/FOV/molecules.csv", "fovs/FOV/run.json"]
    assert imported.read_bytes() == before


# --- Channel output paths by label ------------------------------------------------------------------------

def nuclei_rule_config(tmp_path, morph):
    return dict(starfinder_path=str(ROOT), root_input_path=str(tmp_path / "input"), dataset_id="dataset",
                sample_id="sample", root_output_path=str(tmp_path / "output"), output_id="output",
                fov_id_pattern="FOV%d", n_rounds=2, ref_round="round1", seq_channel_order=list(SEQUENCING),
                ref_channel="DAPI", rotate_angle=0, maximum_projection=False, backend="python",
                additional_round=[dict(round_name="morph", channel_order=morph)],
                rules=dict(nuclei_registration=dict(run=True)))


@pytest.mark.contract
def test_channels_that_share_a_descriptive_name_write_one_file_each_under_their_labels(tmp_path):
    """Two channels named Nissl in one round write images/morph/ch01/ and images/morph/ch02/, each its own
    channel; MATLAB's name folders would have written one file for both."""
    reference, sequencing, morph, _, _ = rounds()
    morph_channels = [dict(wavelength=405, channel="ch00", name="DAPI"), dict(wavelength=488, channel="ch01",
                      name="Nissl"), dict(wavelength=647, channel="ch02", name="Nissl")]
    config = nuclei_rule_config(tmp_path, morph_channels)
    source = tmp_path / "input" / "dataset" / "sample"
    for name, image, labels in (("round1", reference, SEQUENCING), ("round2", sequencing, SEQUENCING),
                                ("morph", morph, OTHER)):
        for c, label in enumerate(labels):
            save_volume(image[..., c], source / name / "FOV" / f"FOV_{label}.tif")
    snakemake = SimpleNamespace(config=config, wildcards=SimpleNamespace(fovID="FOV"), input=[], output=[])
    runpy.run_path(str(ROOT / "workflow" / "scripts" / "nuclei_registration.py"), init_globals={"snakemake": snakemake})
    output = tmp_path / "output" / "dataset" / "output"
    assert tree(output) == ["fovs/FOV/nuclei_registration.json", "images/morph/ch00/FOV.tif",
                            "images/morph/ch01/FOV.tif", "images/morph/ch02/FOV.tif"]
    first, second = (tifffile.imread(output / "images" / "morph" / label / "FOV.tif") for label in ("ch01", "ch02"))
    assert not np.array_equal(first, second)


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


class Namedlist(list):
    """The shape of ``snakemake.io.Namedlist``, the ``snakemake.input`` of a job (Snakemake is not installed in the
    test environment): every file in order, and each named entry also an attribute and an item by name, a string for
    an entry declared as one path and a ``Namedlist`` of its files for an entry declared as a list."""

    def __getitem__(self, key):
        return getattr(self, key) if isinstance(key, str) else super().__getitem__(key)


def job_input(path, rule, namespace, **wildcards):
    """The ``snakemake.input`` of one job of a rule: its parsed input section with the wildcards filled in, input
    functions called with them, in the shape Snakemake gives each entry."""
    def fill(value):
        if callable(value):
            value = value(SimpleNamespace(**wildcards))
        if isinstance(value, list):
            return Namedlist(fill(v) for v in value)
        return str(value).format(**wildcards)
    args, kwargs = rule_section(path, rule, "input", namespace)
    job = Namedlist()
    for value in [*args, *kwargs.values()]:
        filled = fill(value)
        job.extend(filled if isinstance(filled, list) else [filled])
    for name, value in kwargs.items():
        setattr(job, name, fill(value))
    return job


def common_value(name):
    """The MATLAB value common.smk gives a variable before the Python block: its literal assignment, evaluated."""
    text = (RULES / "common.smk").read_text()
    match = re.search(rf"^{name} = (.*?)\n(?=\S)", text, re.M | re.S)
    return match.group(1)


def common_python_names(config, output):
    """The names the Python block of common.smk gives the Python rule files, by executing its own assignments."""
    text = (RULES / "common.smk").read_text()
    names = ("RUN_RECORD", "SUBTILE_RUN_RECORD", "REFERENCE_IMAGE_OUTPUT", "REFERENCE_IMAGE_INPUT",
             "SHARED_IMAGE_INPUTS")
    lines = re.findall(rf"^    ((?:{'|'.join(names)}) = .*)$", text, re.M)
    assert [line.split(" = ")[0] for line in lines] == list(names)
    namespace = dict(config=config, OUTPUT_DIR=output, _reference_image_inputs=_reference_image_inputs,
                     _reference_image_outputs=_reference_image_outputs, _run_record_outputs=_run_record_outputs,
                     _shared_image_inputs=_shared_image_inputs)
    exec("\n".join(lines), namespace)
    return {name: namespace[name] for name in names}


def backend_namespace(tmp_path, backend, **keys):
    output = tmp_path / "data" / "run"
    config = dict(dataset_id="data", output_id="run", backend=backend, n_rounds=1, ref_round="round1",
                  root_input_path=str(tmp_path), root_output_path=str(tmp_path), sample_id="sample",
                  seq_channel_order=["ch01"], additional_round=[dict(round_name="flamingo", channel_order=[
                      dict(channel="ch00", name="DAPI"), dict(channel="ch03", name="Flamingo")])], **keys)
    namespace = dict(OUTPUT_DIR=output, INPUT_DIR=tmp_path / "in", ROUND=["round1"], N_SUBTILE=[1], config=config)
    if backend == "python":
        namespace.update(common_python_names(config, output))
    else:
        namespace["SHARED_IMAGE_INPUTS"] = eval(common_value("SHARED_IMAGE_INPUTS"), dict(OUTPUT_DIR=output))
    return namespace, output


@pytest.mark.contract
def test_the_matlab_backend_keeps_its_rules_and_the_python_backend_follows_the_layout(tmp_path):
    """registration.smk and the shared segmentation.smk parsed for MATLAB give MATLAB's files (images/ref_merged/,
    the name folders of the flamingo round, the logs); registration-py.smk and segmentation.smk for Python give the
    FOV folder's run.json and reference image (none with reference_image_import) and the label folders."""
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

    python, out = backend_namespace(tmp_path, "python")
    registration = RULES / "registration-py.smk"
    assert rule_files(registration, "rsf_single_fov", "output", python)[0] == [
        f"{out}/fovs/{{fovID}}/run.json", f"{out}/fovs/{{fovID}}/reference.tif", f"{out}/fovs/{{fovID}}/molecules.csv"]
    for rule in ("gr_single_fov_subtile", "deep_create_subtile"):
        assert rule_files(registration, rule, "output", python)[0][:2] == [
            f"{out}/fovs/{{fovID}}/run.json", f"{out}/fovs/{{fovID}}/reference.tif"], rule
    assert rule_files(RULES / "spot-finding-py.smk", "lrsf_single_fov_subtile", "output", python)[0][0] == \
        f"{out}/fovs/{{fovID}}/subtile_{{n_subtile}}/run.json"
    assert rule_files(RULES / "spot-finding-py.smk", "stitch_subtile", "input", python)[1] == {
        "reference": f"{out}/fovs/{{fovID}}/reference.tif"}
    named = rule_files(segmentation, "enhance_dapi_with_flamingo", "input", python)[1]
    wildcards = SimpleNamespace(fovID="FOV")
    assert {key: function(wildcards) for key, function in named.items()} == {
        "dapi_img": f"{out}/images/flamingo/ch00/FOV.tif", "flamingo_img": f"{out}/images/flamingo/ch03/FOV.tif"}
    assert rule_files(segmentation, "create_nuclei_amplicon_overlay", "input", python)[1]["amplicon_img"] == \
        f"{out}/fovs/{{fovID}}/reference.tif"

    imported, out = backend_namespace(tmp_path, "python", reference_image_import="tool/{fov_id}.tif")
    assert rule_files(registration, "rsf_single_fov", "output", imported)[0] == [
        f"{out}/fovs/{{fovID}}/run.json", f"{out}/fovs/{{fovID}}/molecules.csv"]
    assert rule_files(segmentation, "create_nuclei_amplicon_overlay", "input", imported)[1]["amplicon_img"] == \
        f"{out}/tool/{{fovID}}.tif"


@pytest.mark.contract
@pytest.mark.parametrize("setting", ["default", "configured", "imported"])
def test_the_readers_of_the_reference_image_get_one_path_in_the_shape_snakemake_gives(tmp_path, setting):
    """The Python stitch_subtile and create_nuclei_amplicon_overlay jobs, their inputs built from the parsed rules in
    the shape of snakemake.input (one path for an entry declared as one path), read the reference image the setting
    names: the default, a configured pattern or an imported file. stitch_subtile passes its input reference to
    TiffFile, which refuses a list."""
    keys = {"default": {}, "configured": {"reference_image": "refs/{fov_id}_ref.tif"},
            "imported": {"reference_image_import": str(tmp_path / "tool" / "{fov_id}.tif")}}[setting]
    namespace, out = backend_namespace(tmp_path, "python", **keys)
    reference = {"default": out / "fovs" / "FOV" / "reference.tif", "configured": out / "refs" / "FOV_ref.tif",
                 "imported": tmp_path / "tool" / "FOV.tif"}[setting]
    image = reference_round(np.uint16, seed=4).max(axis=-1)
    save_volume(image, reference, metadata=VIEW_METADATA)
    subtiles = out / "output" / "subtile" / "FOV"
    subtiles.mkdir(parents=True)
    (subtiles / "subtile_coords.csv").write_text("t,scoords_x,scoords_y,ecoords_x,ecoords_y\n1,1,1,7,6\n")
    (subtiles / "subtile_goodSpots_1.csv").write_text("x,y,z,gene\n4,5,1,Gene1\n")

    job = job_input(RULES / "spot-finding-py.smk", "stitch_subtile", namespace, fovID="FOV")
    assert job.reference == str(reference) and job[-1] == str(reference)
    signal = out / "signal"
    signal.mkdir()
    snakemake = SimpleNamespace(wildcards=SimpleNamespace(fovID="FOV"), config=namespace["config"], input=job,
                                output=[str(signal / "FOV_goodSpots.csv"), str(signal / "FOV_goodSpots.png")])
    stitched = runpy.run_path(str(ROOT / "workflow" / "scripts" / "stitch_subtile.py"),
                              init_globals={"snakemake": snakemake})
    np.testing.assert_array_equal(stitched["reference_img"], image.max(axis=0))
    assert (signal / "FOV_goodSpots.png").stat().st_size > 0

    dapi = out / "images" / "DAPI" / "FOV.tif"
    dapi.parent.mkdir(parents=True)
    tifffile.imwrite(dapi, np.full(image.shape, 7, np.uint16), photometric="minisblack")
    job = job_input(RULES / "segmentation.smk", "create_nuclei_amplicon_overlay", namespace, fovID="FOV")
    assert job["amplicon_img"] == str(reference) and list(job["dapi_img"]) == [str(dapi)]
    overlay = out / "images" / "overlay" / "FOV.tif"
    overlay.parent.mkdir(parents=True)
    runpy.run_path(str(ROOT / "workflow" / "scripts" / "create_nuclei_amplicon_overlay.py"),
                   init_globals={"snakemake": SimpleNamespace(input=job, output=[str(overlay)],
                                                              config=namespace["config"])})
    assert tifffile.imread(overlay).shape == image.shape
