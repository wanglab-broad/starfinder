"""RS5, WF1 and WF2 of docs/sample-segmentation-validation.md: the route in the workflow (W-430).

RS5: the static schema's ``export`` keys equal the fields of ``ExportConfig`` with ``route``,
``segment`` and ``segmentation``, and a run's keys besides ``name``, ``target``, ``inputs``,
``seeds`` and ``method`` equal the init fields of its method's config, kept equal to
``SEGMENTATION_METHODS`` as the §2.9 schema test does; the example configurations of each route on
the workflow configuration page validate. WF1: the inputs of ``export_sample`` by route
(``_export_sample_files``, which ``export.smk`` calls), and the parse-time check of the route.
WF2: the assignment figures of W-330 are not drawn again on the route after fusion, and are on
the route per FOV. The rule-level evidence of WF1, a read-only Snakemake dry run with the
starfinder-v9 Snakemake, is in the W-430 worker notes; the locked environment has no Snakemake.

Fixtures: ``route_configs`` (workflow ``export`` blocks for every combination of ``route``,
``dimensionality``, ``segment``, ``fuse``, with and without ``segmentation``, and ``assignment``
set or null; the repository's example configurations and copies with the blocks), and
``fused_pair`` written as a dataset with ``truth_import`` and ``per_fov_routes``
(``sample_segmentation_fixtures.py``). The adapters run through stub ``snakemake`` objects, as in
``test_stitching_workflow.py``. No model is loaded or run; dask runs with the synchronous scheduler.
"""
import copy
import importlib.util
import itertools
import json
import re
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
import yaml

import starfinder.dataset.workflow as workflow
from starfinder import _snakemake
from starfinder._registry import names
from starfinder.export import (
    ROUTES,
    ExportConfig,
    FusionConfig,
    LabelConfig,
    SampleSegmentationConfig,
    SampleSegmentationRun,
    SpatialDataExportConfig,
    check_export,
)
from starfinder.segmentation import SEGMENTATION_METHODS, LabelImportConfig
from starfinder.segmentation._methods import ROLES

from .fusion_fixtures import listing
from .sample_segmentation_fixtures import (
    DS,
    IMAGES,
    SAMPLE,
    SCENES,
    codebook,
    length_configs,
    per_fov_routes,
    write_scene,
)
from .test_sample_segmentation_route import export_grid, truth_config

pytestmark = [pytest.mark.workflow, pytest.mark.validation]

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
EXPORT_BLOCK = SCHEMA["$defs"]["export_block"]
VALIDATOR = jsonschema.validators.validator_for(SCHEMA)(SCHEMA)  # checked once, reused for every configuration
needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None,
                                   reason="the multiview-stitcher extra (starfinder[multiview-stitcher]) is not "
                                          "installed")
needs_anndata = pytest.mark.skipif(importlib.util.find_spec("anndata") is None,
                                   reason="the anndata extra (starfinder[anndata]) is not installed")
MULTIVIEW = {"method": "multiview_stitcher", "input": "reference_stain",
             "positions": {"source": "regular_grid",
                           "regular_grid": {"columns": 2, "rows": 1, "order": "column_major_down_right",
                                            "overlap": 0.25}}}
FOVS = ["Position001", "Position002"]
OUT = Path("/out")
# The images of the route_configs blocks: the four kinds of source entry, one of them an absolute pattern.
ROUTE_IMAGES = ["reference_stain", "reference_merged", "dapi=file:images/dapi/{fov_id}.tif",
                "flamingo=file:/abs/flamingo/{fov_id}.tif"]
SEGMENTATION = {"device": "cpu", "cells": "cell", "nuclei": "nucleus", "runs": [
    {"name": "nucleus", "target": "nucleus", "method": "import", "path": "/in/nuclei.ome.tif"},
    {"name": "cell", "target": "cell", "inputs": [{"role": "cytoplasm", "image": "flamingo"}], "seeds": "nucleus",
     "method": "seeded_watershed", "sigma_um": 1.0}]}


@pytest.fixture(autouse=True)
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


def base_config(**blocks):
    """The repository's full example configuration on the Python backend with a stitching block and the given
    blocks."""
    config = yaml.safe_load((ROOT / "docs/examples/workflow-full.yaml").read_text())
    return {**config, "backend": "python", "stitching": copy.deepcopy(MULTIVIEW), **copy.deepcopy(blocks)}


# --- route_configs -------------------------------------------------------------------------------------------

ROUTE_VALUES = (None, "per_fov", "after_fusion")
DIMENSIONALITIES = ("projection", "stack", "both")
COMBINATIONS = list(itertools.product(ROUTE_VALUES, DIMENSIONALITIES, (True, False), (True, False), (True, False),
                                      ("default", None)))


def route_block(route, dimensionality, segment, fuse, segmentation):
    block = {"images": list(ROUTE_IMAGES), "dimensionality": dimensionality, "segment": segment, "fuse": fuse}
    if route is not None:
        block["route"] = route
    if segmentation:
        block["segmentation"] = copy.deepcopy(SEGMENTATION)
    return block


def route_workflow_config(route, dimensionality, segment, fuse, segmentation, assignment):
    return base_config(assembly={"assignment": assignment}, export=route_block(route, dimensionality, segment, fuse,
                                                                              segmentation))


def resolved(route, dimensionality):
    """The route of the contract's rule, by hand: the set value, else after_fusion for projection only."""
    return route or ("after_fusion" if dimensionality == "projection" else "per_fov")


def refused(route, dimensionality, segment, fuse, segmentation, assignment):
    """Whether check 7 refuses the configuration alone (the reuse conditions and the models aside)."""
    if resolved(route, dimensionality) != "after_fusion":
        return None
    if dimensionality == "stack":
        return "dimensionality 'stack'"
    if assignment is None:
        return None
    if not segmentation:
        return r"export\.segmentation"
    if segment and not fuse:
        return "fuse: false"
    return None


def per_fov_inputs(fovs=FOVS, root=OUT, name="default"):
    """The inputs of export_sample on the route per FOV: the placement and the one output of reads_assignment, the
    saved folder's record (W-568: no raw.h5ad or reads_assignment.csv)."""
    return ([f"{root}/stitching/{SAMPLE}/placement.json"]
            + [f"{root}/fovs/{f}/assignment/{name}/assignment.json" for f in fovs])


def molecule_inputs(fovs=FOVS, root=OUT):
    return [f"{root}/stitching/{SAMPLE}/placement.json"] + [f"{root}/fovs/{f}/molecules.csv" for f in fovs]


def after_fusion_inputs(fuse, fovs=FOVS, root=OUT, reference="fovs/{fov_id}/reference.tif"):
    """The molecule tables, the reference images and, with fuse on, the source images that are workflow files:
    reference_merged (the reference image) and each file pattern; not the saved form of reference_stain."""
    inputs = molecule_inputs(fovs, root) + [f"{root}/{reference.format(fov_id=f)}" for f in fovs]
    if fuse:
        inputs += [f"{root}/images/dapi/{f}.tif" for f in fovs] + [f"/abs/flamingo/{f}.tif" for f in fovs]
    return inputs


# --- RS5: configuration schema -------------------------------------------------------------------------------

def test_rs5_configuration_schema():
    """The export keys, the route values, the segmentation keys, a run's keys and the method list."""
    declared = EXPORT_BLOCK["properties"]
    nested = {"assembly", "table_format", "fusion", "labels", "spatialdata"}
    expected = ({f.name for f in fields(ExportConfig) if f.name not in nested}
                | {f.name for f in fields(FusionConfig) if f.init}
                | {f"labels_{f.name}" for f in fields(LabelConfig) if f.init}
                | {f"spatialdata_{f.name}" for f in fields(SpatialDataExportConfig) if f.init})
    assert set(declared) == expected == set(workflow._export_keys())
    assert {"route", "segment", "segmentation"} <= set(declared)
    assert declared["route"]["enum"] == [*ROUTES, None]
    assert declared["segment"]["type"] == "boolean"
    # The segmentation sub-block: the SampleSegmentationConfig fields; its assignment is the assignment block's.
    block = declared["segmentation"]
    assert block["type"] == ["object", "null"] and block["additionalProperties"] is False
    assert set(block["properties"]) == {f.name for f in fields(SampleSegmentationConfig)} - {"assignment"}
    assert block["properties"]["device"]["enum"] == ["cpu", "cuda"]
    run = block["properties"]["runs"]["items"]
    assert set(run["properties"]) == set(workflow._SAMPLE_SEGMENTATION_RUN_KEYS) \
        == {f.name for f in fields(SampleSegmentationRun)}
    assert run["additionalProperties"] is True  # the method's config fields, validated by the adapter
    methods = run["properties"]["method"]["anyOf"]
    assert set(methods[0]["enum"]) == set(names(SEGMENTATION_METHODS)) and len(methods[0]["enum"]) == len(
        set(methods[0]["enum"]))
    assert methods[1] == {"const": "import"}
    item = run["properties"]["inputs"]["items"]
    assert set(item["properties"]) == {"role", "image"} and item["properties"]["role"]["enum"] == list(ROLES)


@pytest.mark.parametrize("method", [*names(SEGMENTATION_METHODS), "import"])
def test_rs5_a_run_s_method_keys_equal_the_config_fields(method, monkeypatch):
    """The keys the adapter accepts beside the run keys are the init fields of the method's config (of
    LabelImportConfig without its target, which is the run's, for import)."""
    seen = []
    real = workflow._known
    monkeypatch.setattr(workflow, "_known", lambda values, allowed, context: seen.append(set(allowed)) or
                        real(values, allowed, context))
    with pytest.raises((ValueError, TypeError)):
        workflow._run_method({"not_a_field": 1}, "run", method, "cell", "export.segmentation run")
    if method == "import":
        expected = {f.name for f in fields(LabelImportConfig) if f.init} - {"target"}
    else:
        config_type = next(t for t, spec in SEGMENTATION_METHODS.items() if spec.name == method)
        expected = {f.name for f in fields(config_type) if f.init}
    assert seen == [expected]


def test_rs5_every_init_field_of_a_method_config_translates(tmp_path):
    """length_configs and an import, each written as a run with every init field, give back the same config."""
    configs = dict(length_configs(tmp_path), imported=LabelImportConfig(str(tmp_path / "n.tif"), "nucleus"))
    runs = []
    for key, config in configs.items():
        values = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(config).items()}
        if key == "imported":
            runs.append({"name": "nucleus", "target": "nucleus", "method": "import",
                         **{k: v for k, v in values.items() if k != "target"}})
            continue
        spec = SEGMENTATION_METHODS[type(config)]
        target = "nucleus" if "nucleus" in spec.targets else "cell"
        role = sorted(spec.roles)[0]
        runs.append({"name": key, "target": target, "inputs": [{"role": role, "image": "dapi"}], "method": spec.name,
                     **({"seeds": "nucleus"} if spec.seeds == "required" else {}), **values})
    runs.insert(0, runs.pop())  # the import first: the seeds of the watershed
    block = {"images": ["dapi=file:images/dapi/{fov_id}.tif"], "route": "after_fusion",
             "segmentation": {"cells": "seeded_watershed", "nuclei": "nucleus", "runs": runs}}
    config = base_config(assembly={"assignment": "default"}, export=block)
    VALIDATOR.validate(config)
    export, _ = workflow._export_config(config)
    expected = {"nucleus": configs.pop("imported"), **configs}
    assert {run.name: run.method for run in export.segmentation.runs} == expected


@pytest.mark.parametrize("combination", COMBINATIONS, ids=lambda c: "-".join(map(str, c)))
def test_rs5_route_configs_validate_and_translate(combination):
    """Every route_configs block validates against the schema and translates to the ExportConfig of the same
    values; the segmentation's assignment settings are the assignment block's."""
    route, dimensionality, segment, fuse, segmentation, assignment = combination
    config = route_workflow_config(*combination)
    VALIDATOR.validate(config)
    export, block = workflow._export_config(config)
    assert block.assignment == assignment
    assert (export.route, export.segment, export.fuse, export.fusion.dimensionality) == (route, segment, fuse,
                                                                                         dimensionality)
    assert export.fusion.images == tuple(ROUTE_IMAGES)
    if not segmentation:
        assert export.segmentation is None
        return
    nucleus, cell = export.segmentation.runs
    assert (nucleus.name, nucleus.target, nucleus.inputs, nucleus.method) == (
        "nucleus", "nucleus", (), LabelImportConfig("/in/nuclei.ome.tif", "nucleus"))
    assert (cell.inputs, cell.seeds, cell.method.sigma_um) == ((("cytoplasm", "flamingo"),), "nucleus", 1.0)
    assert (export.segmentation.cells, export.segmentation.nuclei, export.segmentation.device) == (
        "cell", "nucleus", "cpu")


@pytest.mark.parametrize("change", [
    {"route": "per_sample"},
    {"route": "after-fusion"},
    {"segment": "no"},
    {"segmentation": dict(SEGMENTATION, device="gpu")},
    {"segmentation": dict(SEGMENTATION, expansion=1.0)},              # an assignment setting: the assignment block's
    {"segmentation": dict(SEGMENTATION, runs=[])},
    {"segmentation": dict(SEGMENTATION, runs=[{"name": "cell", "target": "cell", "method": "watershed"}])},
    {"segmentation": dict(SEGMENTATION, runs=[{"name": "cell", "target": "membrane", "method": "cellpose"}])},
    {"segmentation": dict(SEGMENTATION, runs=[{"name": "cell", "target": "cell", "method": "cellpose",
                                               "inputs": [{"role": "dapi", "image": "dapi"}]}])},
    {"segmentation": dict(SEGMENTATION, runs=[{"name": "cell", "target": "cell", "method": "cellpose",
                                               "inputs": [{"role": "nuclear"}]}])},
])
def test_rs5_invalid_route_keys_fail_the_schema(change):
    config = base_config(assembly={"assignment": "default"}, export=dict(route_block(None, "projection", True, True,
                                                                                     True), **change))
    with pytest.raises(jsonschema.ValidationError):
        VALIDATOR.validate(config)


def page_examples():
    """The YAML examples of the workflow configuration page that set export.route."""
    text = (ROOT / "docs/workflow-configuration.md").read_text()
    blocks = [yaml.safe_load(b) for b in re.findall(r"```yaml\n(.*?)```", text, re.DOTALL)]
    return [b for b in blocks if isinstance(b, dict) and isinstance(b.get("export"), dict)
            and "route" in b["export"]]


@pytest.mark.contract
def test_rs5_the_example_configuration_of_each_route():
    """docs/workflow-configuration.md gives one example per route; each validates, translates and passes the
    parse-time check of the route. It reads the page, so it is ``contract`` and runs in Q: this module holds no
    ``slow``, ``learned`` or ``extended`` test (docs/contributing.md, "Test markers"; W-602)."""
    examples = page_examples()
    assert sorted(e["export"]["route"] for e in examples) == ["after_fusion", "per_fov"]
    for example in examples:
        config = base_config(**example)
        VALIDATOR.validate(config)
        workflow._check_export_route(config)
        export, block = workflow._export_config(config)
        assert workflow._after_fusion(export, block) is (example["export"]["route"] == "after_fusion")


# --- WF1: the rule's inputs by route ---------------------------------------------------------------------------

@pytest.mark.parametrize("combination", COMBINATIONS, ids=lambda c: "-".join(map(str, c)))
def test_wf1_the_rule_s_inputs_by_route(combination):
    """route_configs: on the route per FOV the outputs of reads_assignment of every FOV, as before; on the route
    after fusion the molecule tables and the source images and no reads_assignment output; molecules only on
    either route the molecule tables (fovs/<fov>/molecules.csv). A configuration that check 7 refuses by itself is refused when
    the rule files are parsed, naming the key."""
    route, dimensionality, segment, fuse, segmentation, assignment = combination
    config = route_workflow_config(*combination)
    error = refused(*combination)
    if error is not None:
        with pytest.raises(ValueError, match=error):
            workflow._check_export_route(config)
        return
    workflow._check_export_route(config)
    inputs, output = _snakemake._export_sample_files(config, OUT, FOVS, SAMPLE)
    assert output == f"{OUT}/export/{SAMPLE}/export.json"
    if assignment is None:
        assert inputs == molecule_inputs()
    elif resolved(route, dimensionality) == "per_fov":
        assert inputs == per_fov_inputs()
    else:
        assert inputs == after_fusion_inputs(fuse)
        assert not [p for p in inputs if "/expr/" in p or "/assignment/" in p]


def test_wf1_the_inputs_follow_reference_image_and_the_named_images():
    """The reference images are the files reference_image names; reference_merged is that file, listed once; with
    reference_stain alone no source image is declared (its saved form is no declared output of any rule)."""
    config = route_workflow_config("after_fusion", "projection", True, True, True, "default")
    config["reference_image"] = "images/flamingo/DAPI/{fov_id}.tif"
    reference = "images/flamingo/DAPI/{fov_id}.tif"
    assert _snakemake._export_sample_files(config, OUT, FOVS, SAMPLE)[0] == after_fusion_inputs(True,
                                                                                              reference=reference)
    config["export"]["images"] = ["reference_stain"]
    config["export"]["segmentation"]["runs"][1]["inputs"] = [{"role": "cytoplasm", "image": "reference_stain"}]
    workflow._check_export_route(config)
    assert _snakemake._export_sample_files(config, OUT, FOVS, SAMPLE)[0] == after_fusion_inputs(False,
                                                                                              reference=reference)


def wf_config(name):
    return yaml.safe_load((ROOT / "docs/examples" / name).read_text())


@pytest.mark.parametrize("name", ["workflow-full.yaml", "workflow-minimal.yaml", "workflow-recipe-2.yaml"])
def test_wf1_existing_configurations_without_the_new_keys(name):
    """The amended row WF1. The examples have no assembly block, so export_sample is not included. Their copies with
    the blocks and without the new keys that RD1 accepts (a stack, both, a projection without an assignment) keep
    the inputs they had at the base revision; the copy that RD1 refuses (a projection with an assignment and no
    export.segmentation) is refused when the rules are parsed, naming export.segmentation and route: per_fov, and
    with route: per_fov added it has the inputs it had at the base revision, as has any copy with route: per_fov."""
    config = wf_config(name)
    assert "assembly" not in config and "export" not in config
    common = (ROOT / "workflow/rules/common.smk").read_text()
    assert re.search(r"if ASSEMBLY_BLOCK:\n        _export_config\(config\)\n        _check_export_route\(config\)",
                     common)
    blocks = dict(config, backend="python", stitching=copy.deepcopy(MULTIVIEW))
    for export, assignment, expected in ((None, "default", None), ({"dimensionality": "stack"}, "default", "per_fov"),
                                         ({"dimensionality": "both"}, "default", "per_fov"),
                                         (None, None, "molecules")):
        copied = dict(copy.deepcopy(blocks), assembly={"assignment": assignment})
        if export is not None:
            copied["export"] = export
        VALIDATOR.validate(copied)
        if expected is None:
            with pytest.raises(ValueError, match=r"export\.segmentation.*route: per_fov"):
                workflow._check_export_route(copied)
            copied["export"] = {"route": "per_fov"}  # the one line that restores the earlier behaviour
            expected = "per_fov"
            VALIDATOR.validate(copied)
        workflow._check_export_route(copied)
        inputs = _snakemake._export_sample_files(copied, OUT, FOVS, SAMPLE)[0]
        assert inputs == (per_fov_inputs() if expected == "per_fov" else molecule_inputs())


# The route_configs without the new key route that RD1 refuses by its default.
RD1_REFUSED = [c for c in COMBINATIONS if c[0] is None and refused(*c) == r"export\.segmentation"]


@pytest.mark.parametrize("combination", RD1_REFUSED, ids=lambda c: "-".join(map(str, c)))
def test_wf1_a_configuration_rd1_refuses_plans_its_earlier_jobs_with_route_per_fov(combination):
    """The amended row WF1, second part, on route_configs: every configuration RD1 refuses (no route, a projection,
    an assignment, no export.segmentation) is refused when the rules are parsed, naming export.segmentation and
    route: per_fov; with route: per_fov added it passes and export_sample has the inputs of the base revision (the
    planned jobs are compared in the dry run, case default_projection_per_fov, in the W-430 worker notes)."""
    route, dimensionality, segment, fuse, segmentation, assignment = combination
    assert (route, dimensionality, segmentation, assignment) == (None, "projection", False, "default")
    config = route_workflow_config(*combination)
    with pytest.raises(ValueError) as refusal:
        workflow._check_export_route(config)
    assert "export.segmentation" in str(refusal.value) and "route: per_fov" in str(refusal.value)
    config["export"]["route"] = "per_fov"
    VALIDATOR.validate(config)
    workflow._check_export_route(config)
    assert _snakemake._export_sample_files(config, OUT, FOVS, SAMPLE)[0] == per_fov_inputs()


# --- WF2: the assignment figures --------------------------------------------------------------------------------

def scene_workflow(tmp_path):
    """fused_pair as a workflow dataset (output root tmp/data/output), its codebook, the per-FOV assignment folders
    of per_fov_routes and the truth labels on the segmentation grid; returns (folder, the export block of the
    route after fusion)."""
    folder = write_scene(tmp_path / "data", SCENES["fused_pair"]())
    per_fov_routes(folder)
    genes = tmp_path / "in" / DS / SAMPLE / "genes.csv"
    genes.parent.mkdir(parents=True)
    codebook().table[["entry_id", "gene_id", "color_sequence"]].to_csv(genes, index=False)
    segmentation = truth_config(folder, export_grid(folder))
    runs = [{"name": run.name, "target": run.target, "method": "import", "path": run.method.path,
             "relabel": False} for run in segmentation.runs]
    block = {"route": "after_fusion", "images": list(IMAGES), "scale_yx": 1.0,
             "segmentation": {"cells": "cell", "nuclei": "nucleus", "runs": runs}}
    return folder, block


def scene_config(tmp_path, export, previews=None):
    """The scene's configuration; ``previews`` None leaves assembly.previews out (its default, false)."""
    assembly = {"assignment": "default", "table_format": "csv", **({} if previews is None else {"previews": previews})}
    return {"backend": "python", "dataset_id": DS, "sample_id": SAMPLE, "output_id": "output",
            "root_input_path": str(tmp_path / "in"), "root_output_path": str(tmp_path),
            "fov_id_pattern": "FOV{i}", "n_rounds": 3, "ref_round": "round1", "rotate_angle": 0, "img_col": 24,
            "img_row": 24, "voxel_size_xy": 0.5, "voxel_size_z": 1.0, "rules": {},
            "seq_channel_order": ["ch00", "ch01", "ch02", "ch03"], "stitching": copy.deepcopy(MULTIVIEW),
            "assembly": assembly, "export": export}


def run_rule(folder, config, monkeypatch, extra_inputs=()):
    """export_sample through a stub with the inputs export.smk gives it; returns (record, figure calls)."""
    calls = []
    real = _snakemake._write_assignment_figures
    monkeypatch.setattr(_snakemake, "_write_assignment_figures",
                        lambda result, image, out, **kwargs: calls.append((Path(out).name, kwargs)) or
                        real(result, image, out, **kwargs))
    inputs, output = _snakemake._export_sample_files(config, folder.root, list(folder.scene.ids), SAMPLE)
    _snakemake._run_export(SimpleNamespace(config=config, input=[*inputs, *map(str, extra_inputs)], output=[output],
                                         wildcards=SimpleNamespace(sample=SAMPLE)))
    monkeypatch.undo()
    return json.loads(Path(output).read_text()), calls, inputs


def figure_paths(folder):
    return {p.relative_to(folder.root).as_posix(): p.read_bytes() for p in sorted(folder.root.rglob("*.png"))}


@needs_anndata
@needs_library
def test_wf2_the_assignment_figures(tmp_path, monkeypatch):
    """With assembly.previews on (W-568): on the route after fusion export_sample draws no assignment figure, also
    with the FOVs' saved assignments present and passed as inputs; on the route per FOV it draws them for every FOV
    with its ownership, as W-330 specifies, into export/<sample>/previews/<fov>/, and leaves the local figures of
    reads_assignment in fovs/<fov>/previews/ as they are."""
    folder, block = scene_workflow(tmp_path)
    for fov in folder.scene.ids:  # the figures reads_assignment wrote first
        (folder.root / "fovs" / fov / "previews").mkdir(parents=True)
        (folder.root / "fovs" / fov / "previews" / "assignment.png").write_bytes(b"written by reads_assignment")
    before = figure_paths(folder)
    config = scene_config(tmp_path, block, previews=True)
    workflow._check_export_route(config)
    saved = [folder.root / "fovs" / f / "assignment" / "default" / "assignment.json" for f in folder.scene.ids]
    assert all(p.is_file() for p in saved)
    record, calls, inputs = run_rule(folder, config, monkeypatch, extra_inputs=saved)
    assert record["parts"] == {"assemble": "written", "fuse": "written", "segment": "written", "fuse_labels": "off",
                               "export_spatialdata": "off"}
    assert record["assembly"]["route"] == "after_fusion" and record["complete"] is True
    assert set(record["assembly"]["fovs"][i]["kind"] for i in range(2)) == {"molecules"}
    assert calls == [] and figure_paths(folder) == before
    assert not [p for p in inputs if "/expr/" in p or "/assignment/" in p]
    # The route per FOV of the same sample, into the same folder: the figures are drawn again with ownership.
    config = scene_config(tmp_path, {"route": "per_fov", "fuse": False}, previews=True)
    exported = listing(folder.export_dir)
    record, calls, _ = run_rule(folder, config, monkeypatch)
    assert record["assembly"]["route"] == "per_fov" and record["parts"]["segment"] == "off"
    assert sorted(fov for fov, _ in calls) == sorted(folder.scene.ids)
    assert all(kwargs["ownership"] is not None for _, kwargs in calls)
    after = figure_paths(folder)
    assert {k: v for k, v in after.items() if k.startswith("fovs/")} == before
    previews = f"export/{SAMPLE}/previews"
    assert {k.rsplit("/", 1)[0] for k in after if not k.startswith("fovs/")} == {f"{previews}/{fov}"
                                                                                 for fov in folder.scene.ids}
    assert all(after[f"{previews}/{fov}/assignment.png"] != before[f"fovs/{fov}/previews/assignment.png"]
               for fov in folder.scene.ids)
    assert listing(folder.export_dir) != exported
    check_export(folder.export_dir)


@needs_anndata
@needs_library
@pytest.mark.parametrize("assemble", [True, False], ids=["assemble", "no_assemble"])
@pytest.mark.parametrize("route", ["per_fov", "after_fusion"])
def test_wf2_every_declared_output_of_export_sample_exists(tmp_path, route, assemble):
    """W-568 repair: with assembly.previews on, export_sample declares export/<sample>/previews/ only on the route per
    FOV with assemble on (_export_preview_outputs), and after the adapter runs every declared output exists, on each
    route with the assemble part on and off; the folder holds each FOV's figures exactly when declared."""
    folder, block = scene_workflow(tmp_path)
    # After fusion, assemble false is accepted with segment false only (check 7), which reuses a saved segmentation.
    export = dict(block, assemble=assemble, **({} if assemble else {"segment": False})) if route == "after_fusion" \
        else {"route": "per_fov", "fuse": False, "assemble": assemble}
    config = scene_config(tmp_path, export, previews=True)
    workflow._check_export_route(config)
    inputs, output = _snakemake._export_sample_files(config, folder.root, list(folder.scene.ids), SAMPLE)
    if route == "after_fusion" and not assemble:   # the saved sample segmentation it reuses, from a first export
        first = scene_config(tmp_path, block, previews=True)
        _snakemake._run_export(SimpleNamespace(config=first, input=inputs, output=[output],
                                             wildcards=SimpleNamespace(sample=SAMPLE)))
    previews = _snakemake._export_preview_outputs(config, folder.root, SAMPLE)
    assert previews == ([str(folder.export_dir / "previews")] if (route, assemble) == ("per_fov", True) else [])
    declared = [output, *previews]       # export.smk: export.json, then the preview folder when declared
    _snakemake._run_export(SimpleNamespace(config=config, input=inputs, output=declared,
                                         wildcards=SimpleNamespace(sample=SAMPLE)))
    assert [p for p in declared if not Path(p).exists()] == []
    drawn = {p.relative_to(folder.export_dir).as_posix() for p in folder.export_dir.rglob("*.png")}
    expected = {f"previews/{fov}/{name}" for fov in folder.scene.ids
                for name in ("assignment.png", "assignment_single_layer.png")}
    assert drawn == (expected if previews else set())
    check_export(folder.export_dir)


@needs_anndata
@needs_library
def test_wf2_the_declared_preview_folder_exists_when_the_assemble_part_cannot_run(tmp_path, monkeypatch):
    """W-568 repair: when the folder is declared but the assemble part is neither written nor reused (here not_run,
    as with an extra missing), export_sample still creates it, empty, and draws no figure."""
    folder, _ = scene_workflow(tmp_path)
    config = scene_config(tmp_path, {"route": "per_fov", "fuse": False}, previews=True)
    inputs, output = _snakemake._export_sample_files(config, folder.root, list(folder.scene.ids), SAMPLE)
    declared = [output, *_snakemake._export_preview_outputs(config, folder.root, SAMPLE)]
    assert declared[1:] == [str(folder.export_dir / "previews")]

    class NotRun:
        def export(self, sample, *, output_dir, **kwargs):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "export.json").write_text("{}")
            return SimpleNamespace(parts={"assemble": "not_run"}, folder=Path(output_dir))

    monkeypatch.setattr(_snakemake, "_export_dataset", lambda *args: NotRun())
    monkeypatch.setattr(_snakemake, "_write_ownership_figures", lambda *args: pytest.fail("no assembly to draw"))
    _snakemake._run_export(SimpleNamespace(config=config, input=inputs, output=declared,
                                         wildcards=SimpleNamespace(sample=SAMPLE)))
    assert all(Path(p).exists() for p in declared)
    assert list(Path(declared[1]).iterdir()) == []
