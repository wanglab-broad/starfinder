"""The records §2.11 adds to run.json and the workflow's run record (W-451, check RR1 of design/specs/report-algorithms.md).

design/specs/report.md, "Records added to run.json": R1 ``transforms`` (the entries of
``registered/transforms.json``, ``_checkpoint.py``: ``_transform_json``, with the field
file named only when the stage ``registered`` wrote it), R2 ``codebook`` (the file
``Dataset.load_codebook`` or ``load_direct_panel`` read), R3
``counts.filtering.endpoint_valid`` (the filter's correct-form count), R4
``workflow_config`` (the configuration file of a workflow rule) and R5 (the processing
rules write ``run.json``, alone by default). Every run is made here on the synthetic FOV of
``test_report_records_golden.py`` (two rounds of four channels, 4 × 24 × 24, round2 shifted
by one voxel in X); nothing outside the test is read. ``format_version`` stayed 1 (V1) until
run.json version 2 (W-565, design/notes/migration.md), which these records are.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import jsonschema
import pandas as pd
import pytest
import yaml

from starfinder.readout import BarcodeLayout, Codebook, ReadFilterConfig, Segment
from starfinder.dataset import (CheckpointConfig, Dataset, ExecutionConfig, PipelineConfig, RoundState,
                                from_workflow_config)
from starfinder.registration import RegistrationRecipe, RegistrationStep
from starfinder._records import jsonable as _jsonable
from starfinder._checkpoint import _transform_json
from starfinder.registration import (AffineConfig, DemonsConfig, RegistrationQcConfig, RegistrationRejectedError,
                                     TranslationConfig)

from .test_report_records_golden import (ALL_STAGES, CHANNELS, GENES, NO_REGISTERED, ROUNDS, differences, pinned_record,
                                         pipeline, write_scene)

pytestmark = [pytest.mark.dataset, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
ADDED = ("transforms", "codebook", "workflow_config")
# 3-base barcodes whose two_base colors are 22, 44, 33 and 11; first and last base C in read orientation.
BARCODES = {"g1": "CAC", "g2": "CGC", "g3": "CTC", "g4": "CCC"}
TRANSLATION = RegistrationStep(TranslationConfig())


def jsonable(value):
    """``value`` as run.json holds it after a write and a read."""
    return json.loads(json.dumps(_jsonable(value)))


def golden_dataset(root: Path) -> Dataset:
    write_scene(root)
    dataset = Dataset(root / "in", root / "out", "golden", "sample", "run",
                      RoundState(list(ROUNDS), reference_round="round1"), CHANNELS)
    dataset.codebook = Codebook(pd.DataFrame({"gene_id": list(GENES), "color_sequence": list(GENES.values())}),
                                ROUNDS, CHANNELS)
    return dataset


def run(dataset: Dataset, stages, config: PipelineConfig | None = None, overwrite=False):
    """FOV_001 run with checkpoints; (the FOV, its run.json, the files of its FOV folder)."""
    fov = dataset.fov("FOV_001")
    fov.run(config or pipeline(), execution=ExecutionConfig(),
            checkpoints=CheckpointConfig(stages=stages, overwrite=overwrite))
    folder = fov.paths.fov_dir
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    return fov, json.loads((folder / "run.json").read_text()), files


def expected_transforms(fov, registered):
    """R1 from the FOV's own results: _transform_json of each result, with the field file of the stage registered."""
    expected = {}
    for name, results in fov.registration_results.items():
        field_name = f"{name}_field.npz" if registered and any(
            r.transform.__class__.__name__ in ("DenseDisplacementTransform", "BSplineTransform") for r in results) else None
        expected[name] = {"reference": "round1",
                          "results": [_transform_json(r, i, field_name) for i, r in enumerate(results)],
                          "application_config": fov.registration_record["application"][name]}
    return jsonable(expected)


def numbers(value):
    """The count of numbers in a list, nested lists included."""
    return sum(numbers(v) if isinstance(v, list) else 1 for v in value)


def no_long_list(value, where="transforms"):
    """Paths of lists longer than 16 numbers (an array written into run.json); [] when there are none."""
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in no_long_list(v, f"{where}.{k}")]
    if isinstance(value, list):
        if numbers(value) > 16:
            return [where]
        return [p for i, v in enumerate(value) for p in no_long_list(v, f"{where}[{i}]")]
    return []


# ---------------------------------------------------------------------------------------
# R1. The transforms.

RECIPES = {
    "translation": RegistrationRecipe((TRANSLATION,)),
    "translation_affine": RegistrationRecipe((TRANSLATION, RegistrationStep(AffineConfig(iterations=20)))),
    "translation_demons": RegistrationRecipe((TRANSLATION, RegistrationStep(DemonsConfig(iterations=(1,))))),
}
ELASTIX = pytest.mark.skipif(importlib.util.find_spec("itk") is None,
                             reason="the affine step requires the registration-elastix extra")
SITK = pytest.mark.skipif(importlib.util.find_spec("SimpleITK") is None,
                          reason="the demons step requires the local-registration extra")


@pytest.fixture(scope="module", params=[
    ("translation", NO_REGISTERED), ("translation", ALL_STAGES),
    pytest.param(("translation_affine", NO_REGISTERED), marks=ELASTIX),
    pytest.param(("translation_affine", ALL_STAGES), marks=ELASTIX),
    pytest.param(("translation_demons", NO_REGISTERED), marks=SITK),
    pytest.param(("translation_demons", ALL_STAGES), marks=SITK),
], ids=lambda p: f"{p[0]}-{'registered' if 'registered' in p[1] else 'no_registered'}")
def registered_run(request, tmp_path_factory):
    recipe, stages = request.param
    dataset = golden_dataset(tmp_path_factory.mktemp("rr1"))
    fov, record, files = run(dataset, stages, replace(pipeline(), registration=RECIPES[recipe]))
    return recipe, stages, fov, record, files


def test_r1_transforms_are_the_entries_of_transforms_json(registered_run):
    recipe, stages, fov, record, files = registered_run
    registered = "registered" in stages
    assert record["status"] == "succeeded" and record["format_version"] == 2
    # The reference round has no entry; round2's results are whole and in order.
    assert list(record["transforms"]) == ["round2"]
    kinds = [r["transform"]["kind"] for r in record["transforms"]["round2"]["results"]]
    assert kinds == {"translation": ["translation"], "translation_affine": ["translation", "affine"],
                     "translation_demons": ["translation", "dense"]}[recipe]
    assert [r["step"] for r in record["transforms"]["round2"]["results"]] == list(range(len(kinds)))
    assert record["transforms"] == expected_transforms(fov, registered)
    assert record["transforms"]["round2"]["application_config"] == jsonable(
        fov.registration_record["application"]["round2"])
    if registered:
        # Exactly the entries and the application the stage registered wrote.
        saved = json.loads((fov.paths.checkpoint_dir / "registered" / "transforms.json").read_text())
        assert record["transforms"]["round2"]["results"] == saved["transforms"]["round2"]
        assert record["transforms"]["round2"]["application_config"] == \
            saved["applications"]["round2"]["application_config"]
    dense = [r["transform"] for r in record["transforms"]["round2"]["results"] if r["transform"]["kind"] == "dense"]
    for transform in dense:
        assert transform["field"] == ("round2_field.npz" if registered else None)
        assert ("checkpoints/registered/round2_field.npz" in files) == registered
    if recipe == "translation_affine":
        affine = record["transforms"]["round2"]["results"][1]["transform"]
        assert len(affine["matrix_zyx"]) == 4 and all(len(row) == 4 for row in affine["matrix_zyx"])


def test_r1_no_field_image_or_array_is_written_into_run_json(registered_run):
    _, _, _, record, _ = registered_run
    assert no_long_list(record["transforms"]) == []


def test_r1_without_registration_and_with_a_failed_round(tmp_path):
    _, record, _ = run(golden_dataset(tmp_path / "none"), NO_REGISTERED, replace(pipeline(), registration=None))
    assert record["transforms"] == {} and record["registration"] == {}
    # A round rejected by the QC has no entry; its attempts stay in registration.
    rejected = RegistrationRecipe((TRANSLATION,), qc=RegistrationQcConfig(max_translation_voxels=0.5))
    dataset = golden_dataset(tmp_path / "failed")
    with pytest.raises(RegistrationRejectedError):
        run(dataset, NO_REGISTERED, replace(pipeline(), registration=rejected))
    record = json.loads((dataset.fov("FOV_001").paths.fov_dir / "run.json").read_text())
    assert record["status"] == "failed" and record["error"]["round"] == "round2"
    assert record["transforms"] == {}
    assert [a["outcome"] for a in record["registration"]["round2"]] == ["rejected"]


# ---------------------------------------------------------------------------------------
# R2. The codebook file; R3. The correct-form count.

def write_codebook(path: Path) -> Path:
    path.write_text("gene,barcode\n" + "".join(f"{g},{b}\n" for g, b in BARCODES.items()))
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r2_and_r3_the_golden_fov_with_a_codebook_set_without_a_file(tmp_path):
    fov, record, _ = run(golden_dataset(tmp_path), NO_REGISTERED)
    assert record["codebook"] == {"kind": "codebook", "path": None, "sha256": None, "entries": 4, "genes": 4}
    # end_bases: the count of the filter's diagnostics, with no per-segment counts, and the fractions (W-563).
    diagnostics = fov.filtering_result.diagnostics
    assert record["counts"]["filtering"] == {**fov.filtering_result.counts,
                                             "endpoint_valid": diagnostics["endpoint_valid_count"],
                                             "fractions": fov.filtering_result.fractions}
    assert "endpoint_valid_segments" not in record["counts"]["filtering"]
    assert record["workflow_config"] is None


def test_r2_and_r3_a_codebook_file_with_segment_ends_and_a_filter_without_endpoints(tmp_path):
    dataset = golden_dataset(tmp_path)
    path = write_codebook(tmp_path / "genes.csv")
    layout = BarcodeLayout((Segment("A", 3, (("C", "C"),)),))
    dataset.load_codebook(path, layout=layout)
    # The layout's segment ends: the filter checks them without end_bases.
    fov, record, _ = run(dataset, NO_REGISTERED, replace(pipeline(), filtering=ReadFilterConfig()))
    assert record["codebook"] == {"kind": "codebook", "path": str(path.absolute()), "sha256": sha256(path),
                                  "entries": 4, "genes": 4}
    diagnostics = fov.filtering_result.diagnostics
    assert record["counts"]["filtering"]["endpoint_valid"] == diagnostics["endpoint_valid_count"]
    assert record["counts"]["filtering"]["endpoint_valid_segments"] == diagnostics["endpoint_valid_segment_counts"]
    assert list(record["counts"]["filtering"]["endpoint_valid_segments"]) == ["A"]
    # No endpoint check (no end_bases, no segment ends): neither key.
    dataset.load_codebook(path)
    fov, record, _ = run(dataset, NO_REGISTERED, replace(pipeline(), filtering=ReadFilterConfig()), overwrite=True)
    assert "endpoint_valid_count" not in fov.filtering_result.diagnostics
    assert record["counts"]["filtering"] == {**fov.filtering_result.counts, "fractions": fov.filtering_result.fractions}
    # A codebook set on the Dataset after the file: no file is named.
    dataset.codebook = Codebook(dataset.codebook.table, ROUNDS, CHANNELS)
    _, record, _ = run(dataset, (), PipelineConfig(), overwrite=True)
    assert record["codebook"] == {"kind": "codebook", "path": None, "sha256": None, "entries": 4, "genes": 4}


def test_r2_a_direct_panel_file_and_no_codebook(tmp_path):
    panel = tmp_path / "panel.csv"
    panel.write_text("round,channel,gene_id\nround1,ch00,g1\nround1,ch01,g2\nround2,ch02,g3\n")
    dataset = Dataset(tmp_path, tmp_path / "out", "d", "s", "o", RoundState(list(ROUNDS), reference_round="round1"),
                      CHANNELS, readout_mode="direct")
    _, record, _ = run(dataset, (), PipelineConfig())
    assert record["codebook"] is None
    dataset.load_direct_panel(panel)
    _, record, _ = run(dataset, (), PipelineConfig(), overwrite=True)
    # The panel lies in the input root, so version 2 records it relative to the root.
    assert record["codebook"] == {"kind": "direct_panel", "path": "panel.csv", "sha256": sha256(panel),
                                  "entries": 3, "genes": 3}


# ---------------------------------------------------------------------------------------
# Older records and the size of the record.

def test_an_older_record_and_the_format_version(tmp_path):
    _, record, _ = run(golden_dataset(tmp_path), NO_REGISTERED)
    assert record["format_version"] == 2
    # A record written before the additions is the same record without them (each is an additive key) ...
    older = {k: v for k, v in record.items() if k not in ADDED}
    older["counts"] = dict(older["counts"], filtering={k: v for k, v in record["counts"]["filtering"].items()
                                                       if not k.startswith("endpoint_valid")})
    (tmp_path / "older.json").write_text(json.dumps(older))
    older = json.loads((tmp_path / "older.json").read_text())
    assert older["format_version"] == 2 and not set(ADDED) & set(older)
    # ... and every key and value pinned at the base revision is in the older form.
    older = json.loads(json.dumps(older).replace(str(tmp_path), "<root>"))
    older["__root__"] = str(tmp_path)
    assert differences(pinned_record(NO_REGISTERED), older) == []


# ---------------------------------------------------------------------------------------
# R4 and R5. The workflow writes run.json and records its configuration file.


# The keys that name and select the FOVs of a MATLAB configuration (the Python backend reads fovs.csv and refuses
# them).
MATLAB_FOVS = {"fov_id_pattern": "%s", "n_fovs": 1, "subset_range": False, "subset_start": 1, "subset_end": 1,
               "subset_random": False, "n_random_tests": 1}


def workflow_config(root: Path, out: str, **parameters) -> dict:
    """The rsf_single_fov configuration of the golden scene (native keys, translation registration)."""
    rule = {"registration": {"steps": [{"method": "translation"}]},
            "spot_finding": {"run": True, "threshold_mode": "adaptive", "threshold_value": 0.2},
            "reads_extraction": {"run": True, "voxel_size": [1, 1, 1]}, "reads_filtration": {"run": True},
            **parameters}
    return dict(config_path=str(root / "config.yaml"), starfinder_path=str(ROOT), root_input_path=str(root),
                root_output_path=str(root / out), dataset_id="data", sample_id="in", output_id="run",
                n_rounds=2, ref_round="round1", rotate_angle=0, img_col=24,
                img_row=24, seq_channel_order=list(CHANNELS), backend="python",
                rules={"rsf_single_fov": {"run": True, "parameters": rule}})


def run_rule(config: dict, codebook: Path):
    """Write the configuration file and run workflow/scripts/rsf_single_fov.py with a stub snakemake object."""
    jsonschema.validate(config, SCHEMA)
    Path(config["config_path"]).write_text(yaml.safe_dump(config))
    snakemake = SimpleNamespace(config=config, input=SimpleNamespace(codebook=str(codebook)),
                                wildcards=SimpleNamespace(fovID="FOV_001"))
    runpy.run_path(str(ROOT / "workflow" / "scripts" / "rsf_single_fov.py"), init_globals={"snakemake": snakemake})
    paths = from_workflow_config(config).dataset.fov("FOV_001").paths
    folder = paths.fov_dir
    # The run's own files: run.json and its checkpoint folder (the rule also writes the reference image and molecules).
    files = sorted(name for name in (p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
                   if name == "run.json" or name.startswith("checkpoints/")) if folder.exists() else []
    logs = sorted(p.relative_to(paths.output_root).as_posix() for p in (paths.output_root / "log").rglob("*"))
    record = json.loads((folder / "run.json").read_text()) if "run.json" in files else None
    return record, files, logs


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    root = tmp_path_factory.mktemp("workflow")
    write_scene(root / "data")
    return root, write_codebook(root / "genes.csv")


def test_r4_r5_the_rule_writes_run_json_alone_and_no_log(scene):
    root, codebook = scene
    record, files, logs = run_rule(workflow_config(root, "out"), codebook)
    assert files == ["run.json"]
    assert record["status"] == "succeeded"
    assert record["config"]["checkpoints"] == {"stages": [], "directory": None, "table_format": "csv",
                                               "hash_inputs": True, "overwrite": True}
    config_file = root / "config.yaml"
    assert record["workflow_config"] == {"path": str(config_file), "sha256": sha256(config_file)}
    assert record["codebook"] == {"kind": "codebook", "path": str(codebook), "sha256": sha256(codebook),
                                  "entries": 4, "genes": 4}
    assert list(record["transforms"]) == ["round2"]
    assert all(entry["sha256"] is not None for entry in record["inputs"])
    # The rule writes no log file (W-563): run.json holds what its logs held.
    assert logs == []
    assert set(record["counts"]["filtering"]["fractions"]) == {"accepted"}
    assert record["counts"]["summary"]["filtering"]["accepted"] == record["counts"]["filtering"]["accepted"]
    assert record["rounds"]["reference_round"] == "round1" and record["rounds"]["sequencing_rounds"] == list(ROUNDS)


def test_r5_the_checkpoints_key_adds_stages_and_a_rerun_replaces_the_record(scene):
    root, codebook = scene
    first, files, _ = run_rule(workflow_config(root, "rerun"), codebook)
    assert files == ["run.json"]
    staged = workflow_config(root, "rerun", checkpoints={"stages": ["candidates", "pre_qc"]})
    second, files, _ = run_rule(staged, codebook)
    assert files == ["checkpoints/candidates.csv", "checkpoints/candidates.json", "checkpoints/pre_qc.csv",
                     "checkpoints/pre_qc.json", "run.json"]
    assert second["config"]["checkpoints"]["stages"] == ["candidates", "pre_qc"]
    assert second["started_at"] >= first["ended_at"]
    # The configuration file changed with the key, and the record names the new bytes.
    assert second["workflow_config"]["sha256"] == sha256(root / "config.yaml") != first["workflow_config"]["sha256"]
    third, files, _ = run_rule(workflow_config(root, "rerun", checkpoints=None), codebook)
    assert files == ["run.json"] and third["config"]["checkpoints"]["stages"] == []


def test_r5_the_checkpoints_key_is_checked(tmp_path):
    config = workflow_config(tmp_path, "out", checkpoints={"stages": ["registered"]})
    jsonschema.validate(config, SCHEMA)
    for value in ({"stages": ["registred"]}, {"directory": "elsewhere"}, ["pre_qc"]):
        invalid = workflow_config(tmp_path, "out", checkpoints=value)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, SCHEMA)
        with pytest.raises((ValueError, TypeError)):
            from_workflow_config(invalid)
    with pytest.raises(ValueError, match="Python-only"):
        from_workflow_config(dict(config, backend="matlab"))
    # Outside the workflow, and for a configuration without config_path, workflow_config is null.
    assert from_workflow_config(config).dataset._workflow_config == tmp_path / "config.yaml"
    assert from_workflow_config({k: v for k, v in config.items() if k != "config_path"}).dataset._workflow_config \
        is None


PROCESSING_RULES = ("rsf_single_fov", "gr_single_fov_subtile", "lrsf_single_fov_subtile", "deep_create_subtile",
                    "deep_rsf_subtile")


@pytest.mark.parametrize("rule", PROCESSING_RULES)
def test_r5_the_schema_allows_the_checkpoints_key_with_the_python_backend_only(tmp_path, rule):
    # The workflow validates its configuration with this schema (workflow/Snakefile) before any rule runs; a
    # MATLAB run never reaches the adapter's own check, so the schema must refuse the key there.
    base = workflow_config(tmp_path, "out")
    matlab = dict(base, **MATLAB_FOVS)
    without = {k: v for k, v in matlab.items() if k != "backend"}
    keyed = {"rules": {rule: {"run": True, "parameters": {"checkpoints": {"stages": ["pre_qc"]}}}}}
    plain = {"rules": {rule: {"run": True, "parameters": {}}}}
    jsonschema.validate(dict(base, **keyed), SCHEMA)
    # backend: matlab refuses it; without the key the same configuration is valid.
    jsonschema.validate(dict(matlab, backend="matlab", **plain), SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(matlab, backend="matlab", **keyed), SCHEMA)
    # The backend omitted is the schema's default, matlab: refused as well.
    assert SCHEMA["properties"]["backend"]["default"] == "matlab"
    jsonschema.validate(dict(without, **plain), SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(without, **keyed), SCHEMA)
