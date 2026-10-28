"""run.json version 2 and the stage records that own their payload hashes (W-565, design/specs/checkpoints.md, "run.json").

A checkpointed run of the synthetic FOV of ``test_report_records_golden.py`` (two rounds of
four channels, 4 × 24 × 24) writes run.json version 2: the output and stage paths relative to
the FOV folder, the input paths relative to the input root, the roots as provenance, the
sources outside the input root (codebook, workflow configuration, supplied statistics) by
absolute path and SHA-256, each stage by its record and that record's SHA-256, and the files
written after the processing under ``outputs``. The stage records (header version 3) hash their
payload files and ``FOV.load_checkpoint`` checks them. A copy of the output and input trees
reports at its new place with the new input root, and the replay verifies the inputs there.
Version-1 run records (the report fixtures) and headers of versions 1 and 2 stay readable.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
from types import SimpleNamespace

import pytest

from starfinder.dataset import CheckpointConfig, Dataset, PipelineConfig, RoundState, from_workflow_config
from starfinder.dataset._run_record import READ_VERSIONS, resolve_source
from starfinder.preprocessing import (HistogramMatchingConfig, MinMaxNormalizationConfig, PreprocessingRecipe,
                                      PreprocessingStep, TophatConfig, fit_supplied_section, merge_histograms,
                                      save_supplied_statistics, summarize_histograms, summary_recipe,
                                      supplied_statistics)
from starfinder.registration import DemonsConfig, RegistrationRecipe, RegistrationStep, TranslationConfig
from starfinder.dataset import SubtileConfig

from .report_fixtures import report_records
from .test_report_records_golden import CHANNELS, ROUNDS, pipeline, write_scene
from .test_report_run_record_additions import ROOT, run_rule, workflow_config, write_codebook

pytestmark = [pytest.mark.dataset, pytest.mark.contract]

SITK = pytest.mark.skipif(importlib.util.find_spec("SimpleITK") is None,
                          reason="the demons step (a dense field file) requires the local-registration extra")
MINMAX = PreprocessingStep(MinMaxNormalizationConfig("uint8", (0, 255)))


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_dataset(root: Path, *, input_root=None, output_root=None) -> Dataset:
    dataset = Dataset(input_root or root / "in", output_root or root / "out", "golden", "sample", "run",
                      RoundState(list(ROUNDS), reference_round="round1"), CHANNELS)
    # The codebook lies outside the input root: an external source, recorded by absolute path and SHA-256.
    dataset.load_codebook(root / "external" / "genes.csv")
    return dataset


def supplied_recipe(root: Path, dataset: Dataset) -> PreprocessingRecipe:
    """Min-max normalization, then histogram matching to a reference supplied in ``external/supplied.json``."""
    path = root / "external" / "supplied.json"
    recipe = PreprocessingRecipe((MINMAX, PreprocessingStep(HistogramMatchingConfig(fit="supplied"))),
                                 supplied_statistics=path)
    prefix, after = summary_recipe(recipe, "histogram_matching")
    fov = dataset.fov("FOV_001").run(PipelineConfig(load=pipeline().load, preprocessing=prefix))
    merged = merge_histograms([summarize_histograms({r: fov.images[r] for r in ROUNDS}, channel_labels=CHANNELS,
                                                    fov_id="FOV_001", summarized_after=after)])
    section = fit_supplied_section(recipe.steps[1].config, merged, reference_round="round1")
    save_supplied_statistics(supplied_statistics(merged, {"histogram_matching": section}), path)
    return recipe


@pytest.fixture
def scene(tmp_path):
    root = tmp_path / "first"
    write_scene(root)
    (root / "external").mkdir()
    write_codebook(root / "external" / "genes.csv")
    return root


# ---------------------------------------------------------------------------------------
# What a checkpointed run writes.

def test_a_checkpointed_run_writes_run_json_version_2(scene):
    dataset = new_dataset(scene)
    dataset.subtile = SubtileConfig(sqrt_pieces=2, overlap_ratio=0)
    dataset.subtile.compute_windows(24, 24)
    # The workflow configuration file, as from_workflow_config records it for a workflow rule.
    workflow = scene / "external" / "config.yaml"
    workflow.write_text("dataset_id: golden\n")
    dataset._workflow_config = workflow
    config = replace(pipeline(), preprocessing=supplied_recipe(scene, dataset))
    parent = scene / "out" / "elsewhere"
    fov = dataset.fov("FOV_001").run(config, checkpoints=CheckpointConfig(directory=parent))
    folder = parent / "FOV_001"
    record = json.loads((folder / "run.json").read_text())
    assert record["format_version"] == 2 and record["status"] == "succeeded"
    # The roots, absolute, as provenance.
    assert record["input_root"] == str(scene / "in") and record["output_root"] == str(scene / "out")
    # Image inputs relative to the input root, each with its SHA-256.
    assert [e["path"] for e in record["inputs"]] == [f"{r}/FOV_001/{c}.tif" for r in ROUNDS for c in CHANNELS]
    assert all(e["sha256"] == sha256(scene / "in" / e["path"]) for e in record["inputs"])
    # Output and stage paths relative to the FOV folder.
    assert record["checkpoint_directory"] == "checkpoints"
    assert record["config"]["checkpoints"]["directory"] == ".."
    # External sources by absolute path and SHA-256.
    codebook, supplied = scene / "external" / "genes.csv", scene / "external" / "supplied.json"
    assert (record["codebook"]["path"], record["codebook"]["sha256"]) == (str(codebook), sha256(codebook))
    assert record["workflow_config"] == {"path": str(workflow), "sha256": sha256(workflow)}
    assert record["preprocessing"]["supplied_statistics"] == {"path": str(supplied), "sha256": sha256(supplied)}
    # Each stage by its record and that record's SHA-256; the record names and hashes its payload files.
    headers = {"registered": "checkpoints/registered/transforms.json", "candidates": "checkpoints/candidates.json",
               "pre_qc": "checkpoints/pre_qc.json"}
    assert record["checkpoints"] == {stage: {"path": path, "sha256": sha256(folder / path)}
                                     for stage, path in headers.items()}
    payload = {"registered": ["round1.ome.tif", "round2.ome.tif"], "candidates": ["candidates.csv"],
               "pre_qc": ["pre_qc.csv"]}
    for stage, path in headers.items():
        header = json.loads((folder / path).read_text())
        assert header["format_version"] == 3
        assert [e["path"] for e in header["files"]] == payload[stage]
        assert all(e["sha256"] == sha256((folder / path).parent / e["path"]) for e in header["files"])
    # outputs: empty when the processing ends; each output added, path relative to the FOV folder, as written.
    assert record["outputs"] == {}
    fov.export_spots()
    fov.export_spots("reads")
    fov.save_reference_image()
    fov.save_subtiles()
    record = json.loads((folder / "run.json").read_text())
    assert record["status"] == "succeeded"
    outputs = record["outputs"]
    assert set(outputs) == {"molecules", "reads", "reference_image", "subtile_coords", "subtile_data"}
    subtiles = os.path.relpath(fov.paths.subtile_dir, folder)
    assert outputs["molecules"]["path"] == os.path.relpath(fov.paths.spot_table("molecules"), folder)
    assert outputs["reference_image"]["path"] == os.path.relpath(fov._reference_image_path(), folder)
    assert outputs["subtile_coords"]["path"] == f"{subtiles}/subtile_coords.csv"
    assert outputs["subtile_coords"]["rows"] == 4 and sorted(outputs["subtile_data"]) == ["1", "2", "3", "4"]
    entries = [outputs[k] for k in ("molecules", "reads", "reference_image", "subtile_coords")]
    entries += list(outputs["subtile_data"].values())
    assert all(not Path(e["path"]).is_absolute() and e["sha256"] == sha256(folder / e["path"]) for e in entries)


@pytest.mark.workflow
def test_the_processing_rules_record_every_declared_output_after_the_processing(tmp_path):
    """rsf_single_fov records molecules.csv and the reference image; gr_single_fov_subtile the reference image and
    the subtile files; lrsf_single_fov_subtile its subtile_goodSpots_<n>.csv in the subtile's run.json."""
    write_scene(tmp_path / "data")
    codebook = write_codebook(tmp_path / "genes.csv")
    record, _, _ = run_rule(workflow_config(tmp_path, "out"), codebook)
    folder = from_workflow_config(workflow_config(tmp_path, "out")).dataset.fov("FOV_001").paths.fov_dir
    assert record["status"] == "succeeded" and record["format_version"] == 2
    assert record["inputs"][0]["path"] == "round1/FOV_001/ch00.tif"
    assert record["input_root"] == str(tmp_path / "data" / "in")
    assert record["codebook"]["path"] == str(codebook)
    assert record["workflow_config"]["path"] == str(tmp_path / "config.yaml")
    outputs = record["outputs"]
    assert set(outputs) == {"molecules", "reference_image"}
    assert (outputs["molecules"]["path"], outputs["reference_image"]["path"]) == ("molecules.csv", "reference.tif")
    assert all(e["sha256"] == sha256(folder / e["path"]) for e in outputs.values())
    # The subtile rules.
    rules = {"gr_single_fov_subtile": {"run": True, "parameters": {
                 "registration": {"steps": [{"method": "translation"}]},
                 "create_subtiles": {"run": True, "sqrt_pieces": 2}}},
             "lrsf_single_fov_subtile": {"run": True, "parameters": {
                 "spot_finding": {"run": True, "threshold_mode": "adaptive", "threshold_value": 0.2},
                 "reads_extraction": {"run": True, "voxel_size": [1, 1, 1]}, "reads_filtration": {"run": True}}}}
    config = dict(workflow_config(tmp_path, "sub"), workflow_mode="subtile", rules=rules)
    Path(config["config_path"]).write_text(json.dumps(config))
    paths = from_workflow_config(config, "gr_single_fov_subtile").dataset.fov("FOV_001").paths
    subtiles = paths.subtile_dir
    for rule, inputs, wildcards in (
            ("gr_single_fov_subtile", SimpleNamespace(codebook=str(codebook)), {}),
            ("lrsf_single_fov_subtile", SimpleNamespace(codebook=str(codebook),
                                                        subtile=str(subtiles / "subtile_data_1.npz")),
             {"n_subtile": "1"})):
        snakemake = SimpleNamespace(config=config, input=inputs, wildcards=SimpleNamespace(fovID="FOV_001", **wildcards))
        runpy.run_path(str(ROOT / "workflow" / "scripts" / f"{rule}.py"), init_globals={"snakemake": snakemake})
    folder = paths.fov_dir
    outputs = json.loads((folder / "run.json").read_text())["outputs"]
    assert set(outputs) == {"reference_image", "subtile_coords", "subtile_data"}
    assert outputs["subtile_coords"]["path"] == "../../output/subtile/FOV_001/subtile_coords.csv"
    assert sorted(outputs["subtile_data"]) == ["1", "2", "3", "4"]
    entries = [outputs["reference_image"], outputs["subtile_coords"], *outputs["subtile_data"].values()]
    assert all(e["sha256"] == sha256(folder / e["path"]) for e in entries)
    subtile = json.loads((folder / "subtile_1" / "run.json").read_text())
    assert subtile["status"] == "succeeded" and set(subtile["outputs"]) == {"subtile_spots"}
    entry = subtile["outputs"]["subtile_spots"]
    assert entry["path"] == "../../../output/subtile/FOV_001/subtile_goodSpots_1.csv"
    assert entry["sha256"] == sha256(subtiles / "subtile_goodSpots_1.csv")


def test_an_input_outside_the_input_root_is_absolute_and_a_record_without_roots_never_reads_the_cwd(
        tmp_path, monkeypatch):
    record = {"format_version": 2, "input_root": None}
    assert resolve_source(record, "round1/a.tif") is None
    monkeypatch.chdir(tmp_path)
    assert resolve_source(record, "round1/a.tif", tmp_path / "in") == tmp_path / "in" / "round1/a.tif"
    assert resolve_source(dict(record, input_root="/recorded"), "round1/a.tif") == Path("/recorded/round1/a.tif")
    assert resolve_source(record, "/abs/genes.csv", tmp_path) == Path("/abs/genes.csv")
    # Version 1 keeps the path as it was recorded.
    assert resolve_source({"format_version": 1}, "relative/a.tif", tmp_path) == Path("relative/a.tif")
    assert READ_VERSIONS == (1, 2)


# ---------------------------------------------------------------------------------------
# Portability.

@pytest.mark.report
def test_a_copied_output_and_input_tree_reports_at_its_new_place(scene, tmp_path, monkeypatch):
    dataset = new_dataset(scene)
    config = replace(pipeline(), preprocessing=PreprocessingRecipe((MINMAX, PreprocessingStep(TophatConfig(2)))))
    dataset.fov("FOV_001").run(config, checkpoints=CheckpointConfig())
    moved = tmp_path / "moved"
    shutil.copytree(scene / "out", moved / "results")
    shutil.copytree(scene / "in", moved / "raw")
    # The original trees are gone; only the external codebook stays where it was recorded.
    shutil.rmtree(scene / "out")
    shutil.rmtree(scene / "in")
    copy = new_dataset(scene, input_root=moved / "raw", output_root=moved / "results")
    report = copy.report("sample", ["FOV_001"])
    replay = report.record["replay"]
    assert report.record["stages"]["run_records"] == ["FOV_001"]
    assert report.record["stages"]["version_1_run_records"] == []
    assert replay["fov_id"] == "FOV_001" and replay["verified"] is True
    assert len(replay["inputs"]) == 8
    for entry in replay["inputs"]:
        assert Path(entry["path"]).is_relative_to((moved / "raw").resolve())
        assert entry["equal"] is True and entry["sha256"] == entry["recorded_sha256"]
    # Without the new input root the record's input root is read, and the relative paths are never opened
    # against the current folder, even when it holds them.
    monkeypatch.chdir(moved / "raw")
    from starfinder.report._replay import replay_fov
    missing = replay_fov(moved / "results", "FOV_001").missing
    assert missing.reason == "input_missing"
    assert replay_fov(moved / "results", "FOV_001", input_root=moved / "raw").record["verified"] is True


# ---------------------------------------------------------------------------------------
# Tampering.

@pytest.mark.parametrize("stage, name", [
    ("candidates", "candidates.csv"), ("pre_qc", "pre_qc.csv"), ("registered", "registered/round1.ome.tif"),
    pytest.param("registered", "registered/round2_field.npz", marks=SITK)])
def test_one_changed_byte_of_a_stage_payload_makes_load_checkpoint_raise(scene, stage, name):
    dataset = new_dataset(scene)
    registration = RegistrationRecipe((RegistrationStep(TranslationConfig()),
                                       RegistrationStep(DemonsConfig(iterations=(1,)))))
    config = replace(pipeline(), registration=registration) if name.endswith("_field.npz") else pipeline()
    fov = dataset.fov("FOV_001").run(config, checkpoints=CheckpointConfig())
    path = fov.paths.checkpoint_dir / name
    # The record loads unchanged.
    dataset.fov("FOV_001").load_checkpoint(stage)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match=f"{stage} checkpoint file {path} has SHA-256 [0-9a-f]{{64}}, the record"):
        dataset.fov("FOV_001").load_checkpoint(stage)
    path.unlink()
    with pytest.raises(ValueError, match=f"{stage} checkpoint file {path} is missing"):
        dataset.fov("FOV_001").load_checkpoint(stage)


# ---------------------------------------------------------------------------------------
# Version 1 and the earlier headers.

@pytest.mark.parametrize("version", [1, 2])
def test_headers_of_versions_1_and_2_load_without_payload_hashes(scene, version):
    dataset = new_dataset(scene)
    fov = dataset.fov("FOV_001").run(pipeline(), checkpoints=CheckpointConfig(stages=("candidates", "pre_qc")))
    directory = fov.paths.checkpoint_dir
    current = {stage: dataset.fov("FOV_001").load_checkpoint(stage) for stage in ("candidates", "pre_qc")}
    for stage in ("candidates", "pre_qc"):
        header = json.loads((directory / f"{stage}.json").read_text())
        earlier = {k: v for k, v in header.items() if k != "files"}
        (directory / f"{stage}.json").write_text(json.dumps(dict(earlier, format_version=version)))
        # An earlier header holds no hashes, so a changed table is not detected by a hash.
        loaded = dataset.fov("FOV_001").load_checkpoint(stage)
        for name in ("spot_result", "decoding_result"):
            if getattr(current[stage], name) is not None:
                assert (getattr(loaded, name).table if name == "decoding_result" else getattr(loaded, name).spots
                        ).equals(getattr(current[stage], name).table if name == "decoding_result"
                                 else getattr(current[stage], name).spots)


@pytest.mark.report
def test_a_version_1_record_stays_report_readable_with_the_note(tmp_path):
    case = report_records(tmp_path / "root", inputs=True)
    assert {case.run_record(f)["format_version"] for f in case.fov_ids} == {1}
    _, record = case.collect()
    assert record["stages"]["run_records"] == list(case.fov_ids)
    # Read as recorded, and listed as not eligible for verified resume.
    assert record["stages"]["version_1_run_records"] == list(case.fov_ids)
    assert record["replay"]["verified"] is True
    # A record of an unknown version is refused by check 3.
    data = case.run_record(case.fov_ids[0])
    case.write_run_record(case.fov_ids[0], dict(data, format_version=3))
    with pytest.raises(ValueError, match="format_version is 3, expected one of \\[1, 2\\]"):
        case.collect()


def test_the_record_keeps_a_relative_checkpoint_directory_for_a_subtile(scene):
    dataset = new_dataset(scene)
    fov = dataset.fov("FOV_001")
    fov.subtile_id = 2
    fov.run(PipelineConfig(load=pipeline().load), checkpoints=CheckpointConfig(stages=(), directory=scene / "ck"))
    record = json.loads((scene / "ck" / "FOV_001" / "subtile_2" / "run.json").read_text())
    assert record["config"]["checkpoints"] == dict(json.loads(json.dumps(asdict(CheckpointConfig(stages=())))),
                                                   directory="../..")
