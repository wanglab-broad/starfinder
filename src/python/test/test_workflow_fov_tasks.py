"""The fov command (moved by W-750 from the tests that ran ``workflow/scripts/rsf_single_fov.py``, the Python
processing rule's adapter: test_readout_direct.py, test_projection_views.py, test_report_run_record_additions.py and
test_dataset_fov_layout.py).

Each test writes the raw images the old test wrote, plans the rule's parameters as the ``processing`` block of a
configuration in the Python shape (the run flags left with D4) and runs the FOV's ``fov`` task by its command. What
changed with the contract is named where a test asserts it: the task writes its record and log under ``workflow/``
(D8), the run record names the resolved configuration it ran with (D2, D8; design/specs/checkpoints.md), and the
default checkpoint stages are ``candidates`` and ``pre_qc`` (D8).
"""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starfinder import _layout
from starfinder.dataset import from_workflow_config
from starfinder.readout import DirectDecoderConfig
from starfinder.io import save_volume

from . import workflow_tasks as T

pytestmark = [pytest.mark.workflow, pytest.mark.dataset, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(root, *, leave=("workflow/",)):
    """The files under ``root``, POSIX and sorted, without those under the folders ``leave`` (the plan's and the
    tasks' records, D8)."""
    return sorted(name for name in (p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file())
                  if not name.startswith(leave))


def run_fov(work, config, fov, fovs_csv):
    """Plan ``config`` on its input root's raw images and fovs.csv and run the FOV's fov task; (manifest, output
    root)."""
    root = T.input_root(config)
    root.mkdir(parents=True, exist_ok=True)
    _layout.fovs_file(root).write_text(fovs_csv)
    manifest, _ = T.plan(work, f"fov-{fov}", config)
    task = T.task(manifest, "fov", fov=fov)
    assert T.run(manifest, task["id"]) == 0
    return manifest, T.output_root(config)


# --- the reference image of reference_image_import (from test_dataset_fov_layout.py) -------------------------------

def test_with_reference_image_import_the_fov_task_writes_no_reference_image(tmp_path):
    """From test_with_reference_image_import_the_processing_rule_writes_no_reference_image: the fov command writes
    run.json and molecules.csv in the FOV folder and no reference image, which it does not declare; the imported
    file is left as it is."""
    from .test_projection_views import CHANNELS, METADATA, NATIVE_DETECTION, reference_round
    raw = reference_round(np.uint16, seed=1)
    for name in ("round1", "round2"):
        for c, channel in enumerate(CHANNELS):
            save_volume(raw[..., c], tmp_path / "data" / "sample" / name / "FOV" / f"{channel}.tif",
                        metadata=METADATA)
    (tmp_path / "data" / "sample" / "genes.csv").write_text("gene,barcode\ngene,AAA\n")
    imported = tmp_path / "tool" / "FOV.tif"
    save_volume(raw.max(axis=-1), imported, metadata=METADATA)
    before = imported.read_bytes()
    processing = {k: copy.deepcopy(v) for k, v in NATIVE_DETECTION.items() if k != "reference_image_view"}
    for block in ("spot_finding", "reads_extraction", "reads_filtration"):
        processing[block].pop("run")
    config = dict(T.dataset_keys(tmp_path, n_rounds=2, output_id="run", seq_channel_order=list(CHANNELS)),
                  root_input_path=str(tmp_path), root_output_path=str(tmp_path / "out"),
                  reference_image_import=str(tmp_path / "tool" / "{fov_id}.tif"),
                  processing=dict(processing, checkpoints=[]))
    manifest, out = run_fov(tmp_path, config, "FOV", "sample,fov_id\nsample,FOV\n")
    assert out == tmp_path / "out" / "data" / "run"
    assert tree(out) == ["fovs/FOV/molecules.csv", "fovs/FOV/run.json"]
    assert not [o for o in T.task(manifest, "fov", fov="FOV")["outputs"] if "tool" in o or o.endswith(".tif")]
    assert imported.read_bytes() == before


# --- the direct readout (from test_readout_direct.py) ---------------------------------------------------------------

def test_the_fov_command_direct_route_writes_the_four_genes(tmp_path):
    """From test_yaml_direct_route_writes_the_four_genes: readout_mode direct with the direct panel (direct_panel)
    writes the FOV's molecule table of the four genes, at the seeded positions."""
    from .test_readout_direct import CHANNELS, DIRECT_MAPPING, DIRECT_SPOTS, GENES, METADATA, ROUNDS, direct_images
    for name, image in direct_images().items():
        for c, channel in enumerate(CHANNELS):
            save_volume(image[..., c], tmp_path / "data" / "sample" / name / "FOV" / f"{channel}.tif",
                        metadata=METADATA)
    panel = tmp_path / "panel.csv"
    DIRECT_MAPPING.to_csv(panel, index=False)
    processing = {"spot_finding": {"threshold_mode": "noise", "threshold_value": 10.0, "rounds": list(ROUNDS)},
                  "reads_extraction": {}, "reads_filtration": {}}
    config = dict(T.dataset_keys(tmp_path, n_rounds=2, output_id="run", seq_channel_order=list(CHANNELS)),
                  root_input_path=str(tmp_path), root_output_path=str(tmp_path / "out"), readout_mode="direct",
                  direct_panel=str(panel), processing=processing)
    manifest, out = run_fov(tmp_path, config, "FOV", "sample,fov_id\nsample,FOV\n")
    resolved = json.loads((out / "workflow" / "resolved.json").read_text())["configuration"]
    assert resolved["readout_mode"] == "direct"
    assert resolved["api"]["processing"]["pipeline"]["decoding"]["type"] == DirectDecoderConfig.__name__
    good = pd.read_csv(_layout.spot_table(out, "FOV"))
    assert list(good.columns) == ["spot_namespace", "spot_id", "z", "y", "x", "gene_id", "entry_id", "qc_score"]
    assert good.gene_id.tolist() == GENES
    assert list(zip(good.x, good.y, good.z)) == [(x + 0, y + 0, z + 0) for _, _, z, y, x in DIRECT_SPOTS]


# --- the reference image's view (from test_projection_views.py) ----------------------------------------------------

@pytest.mark.parametrize("maximum_projection", [False, True])
def test_the_fov_command_writes_the_detection_view(tmp_path, maximum_projection):
    """From test_rsf_single_fov_script_writes_the_detection_view: the reference image the fov command writes is the
    single-channel view of the registered reference round (channel 2), its Z maximum with
    reference_image_view.projection max."""
    from .test_projection_views import (CHANNEL_VIEW, CHANNELS, METADATA, NATIVE_DETECTION, read_back,
                                        reference_round, workflow_config)
    raw = reference_round(np.uint16, seed=1)
    for name in ("round1", "round2"):
        for c, channel in enumerate(CHANNELS):
            save_volume(raw[..., c], tmp_path / "data" / "sample" / name / "FOV" / f"{channel}.tif",
                        metadata=METADATA)
    (tmp_path / "data" / "sample" / "genes.csv").write_text("gene,barcode\ngene,AAA\n")
    processing = copy.deepcopy(NATIVE_DETECTION)
    for block in ("spot_finding", "reads_extraction", "reads_filtration"):
        processing[block].pop("run")
    processing["reference_image_view"] = dict(CHANNEL_VIEW, **({"projection": "max"} if maximum_projection else {}))
    config = dict(T.dataset_keys(tmp_path, n_rounds=2, output_id="run", seq_channel_order=list(CHANNELS)),
                  root_input_path=str(tmp_path), root_output_path=str(tmp_path / "out"),
                  processing=dict(processing, checkpoints=[]))
    _, out = run_fov(tmp_path, config, "FOV", "sample,fov_id\nsample,FOV\n")
    # The expected view, by the API on the same raw images (as the old test computed it).
    old = workflow_config(tmp_path, maximum_projection=maximum_projection, backend="python", **NATIVE_DETECTION)
    adapted = from_workflow_config(old)
    fov = adapted.dataset.fov("FOV")
    fov.run(replace(adapted.pipeline, registration=None, spot_finding=None, extraction=None, decoding=None,
                    filtering=None))
    expected = fov.images["round1"][..., 2]
    expected = expected.max(axis=0) if maximum_projection else expected
    saved, axes = read_back(_layout.reference_image_path(out, "FOV"))
    np.testing.assert_array_equal(saved, expected, strict=True)
    assert saved.dtype == np.uint8 and axes == ("YX" if maximum_projection else "ZYX")


# --- the run record of the fov task (from test_report_run_record_additions.py, R4 and R5) -------------------------

@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """The golden scene of test_report_records_golden.py (FOV_001, input root <root>/data/in) and its codebook."""
    from .test_report_records_golden import write_scene
    from .test_report_run_record_additions import write_codebook
    root = tmp_path_factory.mktemp("workflow")
    write_scene(root / "data")
    return root, write_codebook(root / "genes.csv")


def scene_config(root, codebook, out, **processing):
    """The rule's configuration of the golden scene (native keys, translation registration) in the Python shape."""
    from .test_report_records_golden import CHANNELS
    block = {"registration": {"steps": [{"method": "translation"}]},
             "spot_finding": {"threshold_mode": "adaptive", "threshold_value": 0.2},
             "reads_extraction": {"voxel_size": [1, 1, 1]}, "reads_filtration": {}, **processing}
    return dict(T.dataset_keys(root, n_rounds=2, output_id="run", seq_channel_order=list(CHANNELS)),
                root_input_path=str(root), root_output_path=str(root / out), sample_id="in",
                codebook=str(codebook), processing=block)


def run_scene(root, config, name):
    """The fov task of FOV_001: (run.json, the FOV folder's run.json and checkpoint files, the files under log/)."""
    manifest, out = run_fov(root / name, config, "FOV_001", "sample,fov_id\nin,FOV_001\n")
    folder = _layout.fov_dir(out, "FOV_001")
    files = sorted(n for n in (p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
                   if n == "run.json" or n.startswith("checkpoints/"))
    logs = sorted(p.relative_to(out).as_posix() for p in (out / "log").rglob("*"))
    return json.loads((folder / "run.json").read_text()), files, logs, out


def test_r4_r5_the_fov_task_writes_run_json_and_no_log_in_the_fov_folder(scene):
    """From test_r4_r5_the_rule_writes_run_json_alone_and_no_log: with processing.checkpoints [] the fov task writes
    run.json alone in the FOV folder and no log beside it (its log is workflow/logs/, D8); the record names the
    resolved configuration it ran with (workflow/resolved.json, D2 and D8), the codebook, the transforms, the inputs'
    hashes, the filter's fractions, the summary and the round roles."""
    from .test_report_records_golden import ROUNDS
    root, codebook = scene
    record, files, logs, out = run_scene(root, scene_config(root, codebook, "out", checkpoints=[]), "r4")
    assert files == ["run.json"]
    assert record["status"] == "succeeded"
    assert record["config"]["checkpoints"] == {"stages": [], "directory": None, "table_format": "csv",
                                               "hash_inputs": True, "overwrite": True}
    resolved = out / "workflow" / "resolved.json"
    assert record["workflow_config"] == {"path": "workflow/resolved.json", "sha256": f"sha256:{sha256(resolved)}"}
    assert record["codebook"] == {"kind": "codebook", "path": str(codebook), "sha256": sha256(codebook),
                                  "entries": 4, "genes": 4}
    assert list(record["transforms"]) == ["round2"]
    assert all(entry["sha256"] is not None for entry in record["inputs"])
    assert logs == []
    assert (out / "workflow" / "logs" / "fov.in.FOV_001.log").is_file()
    assert set(record["counts"]["filtering"]["fractions"]) == {"accepted"}
    assert record["counts"]["summary"]["filtering"]["accepted"] == record["counts"]["filtering"]["accepted"]
    assert record["rounds"]["reference_round"] == "round1" and record["rounds"]["sequencing_rounds"] == list(ROUNDS)


def test_r5_the_checkpoints_key_adds_stages_and_a_rerun_replaces_the_record(scene):
    """From the test of the same name: processing.checkpoints adds the stages, a rerun into the same output root
    replaces the record, which names the new resolved configuration; without the key the stages are the default,
    candidates and pre_qc (D8, where the rule's default was none)."""
    root, codebook = scene
    first, files, _, _ = run_scene(root, scene_config(root, codebook, "rerun", checkpoints=[]), "rerun")
    assert files == ["run.json"]
    staged = scene_config(root, codebook, "rerun", checkpoints=["candidates", "pre_qc"])
    second, files, _, _ = run_scene(root, staged, "rerun")
    assert files == ["checkpoints/candidates.csv", "checkpoints/candidates.json", "checkpoints/pre_qc.csv",
                     "checkpoints/pre_qc.json", "run.json"]
    assert second["config"]["checkpoints"]["stages"] == ["candidates", "pre_qc"]
    assert second["started_at"] >= first["ended_at"]
    # The resolved configuration changed with the key, and the record names its new bytes.
    assert second["workflow_config"]["sha256"] != first["workflow_config"]["sha256"]
    third, files, _, _ = run_scene(root, scene_config(root, codebook, "rerun"), "rerun")
    assert third["config"]["checkpoints"]["stages"] == ["candidates", "pre_qc"]
    assert files == ["checkpoints/candidates.csv", "checkpoints/candidates.json", "checkpoints/pre_qc.csv",
                     "checkpoints/pre_qc.json", "run.json"]


@pytest.mark.workflow
def test_the_fov_task_records_every_declared_output_after_the_processing(scene):
    """From test_run_record_version_2.py, test_the_processing_rules_record_every_declared_output_after_the_processing:
    the fov task's run.json (format 2) records molecules.csv and the reference image after the processing, the inputs
    relative to the input root it names, the codebook and the resolved configuration it ran with (D2, D8). (The
    subtile rules, retired for Python by D5, left the configuration: the plan refuses create_subtiles,
    test_workflow_plan_refusals.py, test_each_nested_key_that_left_the_python_dialect.)"""
    root, codebook = scene
    record, _, _, out = run_scene(root, scene_config(root, codebook, "outputs", checkpoints=[]), "outputs")
    folder = _layout.fov_dir(out, "FOV_001")
    assert record["status"] == "succeeded" and record["format_version"] == 2
    assert record["inputs"][0]["path"] == "round1/FOV_001/ch00.tif"
    assert record["input_root"] == str(root / "data" / "in")
    assert record["codebook"]["path"] == str(codebook)
    assert record["workflow_config"]["path"] == "workflow/resolved.json"
    outputs = record["outputs"]
    assert set(outputs) == {"molecules", "reference_image"}
    assert (outputs["molecules"]["path"], outputs["reference_image"]["path"]) == ("molecules.csv", "reference.tif")
    assert all(e["sha256"] == sha256(folder / e["path"]) for e in outputs.values())


@pytest.mark.parametrize("streaming", [False, True])
def test_the_fov_command_and_the_shared_filenames(tmp_path, streaming):
    """From test_coordination_contract.py, test_five_workflow_adapters_and_shared_filenames: the fov command (with
    processing.streaming as given) writes the FOV folder's molecules.csv with the molecule table's columns, its
    reference image and run.json, where the package's paths name them. (The four subtile adapters, retired for
    Python by D5, refused; the plan refuses their create_subtiles key, test_workflow_plan_refusals.py.)"""
    from starfinder.image import ImageMetadata
    from .test_coordination_contract import images
    channels = ["a", "b", "c", "d"]
    for name, image in images().items():
        for c, channel in enumerate(channels):
            save_volume(image[..., c], tmp_path / "data" / "sample" / name / "FOV" / f"{channel}.tif",
                        metadata=ImageMetadata("common"))
    (tmp_path / "data" / "sample" / "genes.csv").write_text("gene,barcode\ngene,AAA\n")
    processing = dict(streaming=streaming, spot_finding={"threshold_mode": "adaptive", "threshold_value": .1},
                      reads_extraction={"voxel_size": [0, 0, 0]}, reads_filtration={})
    config = dict(T.dataset_keys(tmp_path, n_rounds=2, output_id="run", seq_channel_order=channels),
                  root_input_path=str(tmp_path), root_output_path=str(tmp_path / "out"), processing=processing)
    _, out = run_fov(tmp_path, config, "FOV", "sample,fov_id\nsample,FOV\n")
    folder = tmp_path / "out" / "data" / "run" / "fovs" / "FOV"
    assert _layout.spot_table(out, "FOV") == folder / "molecules.csv"
    assert list(pd.read_csv(folder / "molecules.csv")) == ["spot_namespace", "spot_id", "z", "y", "x", "gene_id",
                                                           "entry_id", "qc_score"]
    assert (folder / "reference.tif").is_file() and (folder / "run.json").is_file()
