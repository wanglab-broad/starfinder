"""The workflow's checks of design/validation/compressed-image-outputs.md (W-922): CW1 and CR1 through the commands,
CW5 at plan time, WF1 to WF4 and MT1.

``execution.image_compression`` (``zlib``, the default, or ``none``) is outside every hash; the ``fov``,
``tile_global``, ``tile_merge`` and ``morphology`` commands pass it to the writers of the stage ``registered``, the
reference image and the prepared images, and the records name it. On one FOV of ``tiny_sample`` (the worked example's
fixture, ``workflow_fixture``), each task run in this process with a clean code identity. The comparisons with v0.3.0
(``70edc72``) run its package in a subprocess (``image_compression_cases``) and skip, naming the reason, without Git or
that revision.
"""
import copy
import hashlib
import json
import shutil

import h5py
import pytest
import tifffile

from starfinder import _layout
from starfinder.__main__ import main
from starfinder.io.tiff import _tiff_compression
from starfinder.workflow import plan
from starfinder.workflow._plan import PlanRefused

from . import image_compression_cases as C
from . import workflow_fixture as F
from . import workflow_variants as V

pytestmark = [pytest.mark.workflow, pytest.mark.dataset, pytest.mark.validation]

FOV = "FOV_001"
CODES = {"zlib": 8, "none": 1}
#: The fov task's tables, compared byte for byte between runs in this environment.
TABLES = ["molecules.csv", "checkpoints/candidates.csv", "checkpoints/pre_qc.csv"]
STAGES = ["registered", "candidates", "pre_qc"]


@pytest.fixture(scope="module")
def fixture_work(tmp_path_factory):
    work = tmp_path_factory.mktemp("tiny") / "work"
    F.golden().write_fixture(work)
    return work


@pytest.fixture
def work(fixture_work, tmp_path, monkeypatch):
    target = tmp_path / "work"
    shutil.copytree(fixture_work, target, symlinks=True)
    F.fix_code(monkeypatch)
    return target


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """``src/python`` of v0.3.0 (``base_reader``), or a skip naming why it is unavailable."""
    return C.base_source(tmp_path_factory.mktemp("base"))


def codes(path):
    """The TIFF Compression tags of every page of a file."""
    with tifffile.TiffFile(path) as tif:
        return {int(page.compression) for page in tif.pages}


def config(work, output_id, setting=None, *, checkpoints=STAGES, morphology=None, **processing):
    """One FOV with the molecules-only processing block, ``checkpoints``, ``morphology`` when given and
    ``execution.image_compression: setting`` when given (None leaves the key out, as v0.3.0's shape has none)."""
    out = V.single_fov(work, output_id, **({} if setting is None else {"image_compression": setting}))
    out["processing"]["checkpoints"] = list(checkpoints)
    out["processing"].update(processing)
    if morphology is not None:
        out["morphology"] = morphology
    return out


def run(work, name, configuration, *kinds):
    """Plan ``configuration`` and run its tasks of ``kinds`` (all when none is given) in order; (manifest path, FOV
    folder)."""
    manifest = plan(F.write_config(work, name, configuration))
    path = F.output_root(work, configuration["output_id"]) / "workflow" / "manifest.json"
    for task in manifest["tasks"]:
        if not kinds or task["kind"] in kinds:
            assert F.run_task(path, task["id"]) == 0, task["id"]
    return path, _layout.fov_dir(path.parents[1], FOV)


def digests(folder, names):
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in names}


# --- CW1, CR1: the commands pass the setting, and the records name it ---------------------------------------------

SEGMENTATION = {"runs": [
    {"name": "nucleus", "target": "nucleus", "method": "import", "path": "{work}/labels/{fov_id}.tif"},
    {"name": "cell", "target": "cell", "method": "seeded_watershed", "seeds": "nucleus",
     "inputs": [{"role": "amplicon", "reference_merged": True}]}]}


@pytest.mark.parametrize("setting", ["zlib", "none", None])
def test_cw1_cr1_the_tasks_write_and_record_the_setting(work, setting):
    """The fov, morphology and segment tasks: I1, I3 and I4 take the setting (zlib when the key is absent), every
    page; I6 and I7 stay zlib and I10 and the raw inputs (I14) uncompressed, in either setting. run.json, the stage
    record and registration.json name the codec, which the file's tags give."""
    expected = setting or "zlib"
    configuration = config(work, "codec", setting, morphology={"channel_images": {"projection": None}})
    configuration["segmentation"] = json.loads(json.dumps(SEGMENTATION).replace("{work}", str(work)))
    path, folder = run(work, "codec", configuration)
    resolved = json.loads((path.parent / "resolved.json").read_text())
    assert resolved["configuration"]["execution"]["image_compression"] == expected
    record = json.loads((folder / "run.json").read_text())
    header = json.loads((folder / "checkpoints" / "registered" / "transforms.json").read_text())
    assert record["config"]["checkpoints"]["image_compression"] == expected
    assert header["image_compression"] == expected and header["format_version"] == 3
    assert all(set(entry) == {"path", "sha256"} for entry in header["files"])
    assert record["outputs"]["reference_image"]["compression"] == expected
    rounds = [folder / "checkpoints" / "registered" / f"round{r}.ome.tif" for r in range(1, 5)]
    prepared = sorted((folder / "other_rounds").glob("*/registration.json"))
    assert [p.parent.name for p in prepared] == ["flamingo", "reference_stain"]
    images = [*rounds, folder / "reference.tif"]
    for item in prepared:
        entry = json.loads(item.read_text())["image"]
        assert entry["compression"] == expected
        images.append(item.parent / entry["path"])
    for image in images:
        assert codes(image) == {CODES[expected]} and _tiff_compression(image) == expected, image
    saved = folder / "segmentation"
    assert codes(saved / "nucleus" / "labels.tif") == codes(saved / "cell" / "labels.tif") == {8}
    assert codes(saved / "cell" / "input.ome.tif") == {8}
    channel_images = sorted((path.parents[1] / "images").rglob(f"{FOV}.tif"))
    assert len(channel_images) == 2 and all(codes(image) == {1} for image in channel_images)
    raw = sorted(F.input_root(work).rglob("*.tif"))
    assert raw and all(codes(image) == {1} for image in raw if image.parent.name != "labels")


# --- CW5: the plan refuses another value ----------------------------------------------------------------------------

@pytest.mark.contract
def test_cw5_the_plan_refuses_another_codec(work, capsys):
    configuration = config(work, "refused", "gzip")
    path = F.write_config(work, "refused", configuration)
    with pytest.raises(PlanRefused, match=r"^execution\.image_compression: 'gzip' is not one of \['zlib', 'none'\]$"):
        plan(path)
    assert main(["workflow", "plan", "--config", str(path)]) == 2
    assert "execution.image_compression: 'gzip' is not one of ['zlib', 'none']" in capsys.readouterr().err


# --- WF1: outside every hash -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["per-fov", "after-fusion", "reference-only"])
def test_wf1_the_codec_is_outside_every_hash(work, variant):
    """Planned with zlib and with none, every task's blocks_sha256 and fingerprint and the configuration_sha256 are
    equal; the resolved configuration holds the setting, and the default written out when the key is absent."""
    F.write_supplied(work)
    plans = {}
    for setting in ("zlib", "none", None):
        configuration = copy.deepcopy(V.variants(work)[variant])
        execution = dict(configuration.get("execution") or {})
        if setting is not None:
            execution["image_compression"] = setting
        configuration["execution"] = execution
        plans[setting] = plan(F.write_config(work, f"{variant}-{setting}", configuration))
        resolved = F.output_root(work, configuration["output_id"]) / "workflow" / "resolved.json"
        assert json.loads(resolved.read_text())["configuration"]["execution"]["image_compression"] == (setting or "zlib")
    keyed = {s: {t["id"]: (t["blocks_sha256"], t["fingerprint"]) for t in m["tasks"]} for s, m in plans.items()}
    assert keyed["zlib"] == keyed["none"] == keyed[None]
    assert len({m["configuration"]["configuration_sha256"] for m in plans.values()}) == 1
    assert all("image_compression" not in t["execution"] for m in plans.values() for t in m["tasks"])


# --- WF2: a resume after a switch ----------------------------------------------------------------------------------

class Kill(BaseException):
    """A stop of the task in the middle of FOV.run, as a scheduler's kill."""


def test_wf2_a_resume_after_a_switch_keeps_the_stage_and_its_codec(work, monkeypatch):
    """A fov task stopped after its stage registered under none resumes under zlib: not refused, resumed from
    registered; the kept round images stay Compression 1 and their record says none; reference.tif is zlib; the
    tables equal a run without the stop; run.json names this attempt's setting."""
    from starfinder.dataset.fov import FOV as FOVClass
    _, full = run(work, "full", config(work, "full", "zlib"), "fov")
    configuration = config(work, "switch", "none", checkpoints=["registered", "candidates", "pre_qc"])
    configuration["execution"]["resume_from_checkpoint"] = True
    manifest = plan(F.write_config(work, "switch", configuration))
    path = F.output_root(work, "switch") / "workflow" / "manifest.json"
    task = manifest["tasks"][0]["id"]
    with monkeypatch.context() as patch:
        def stop(self, *args, **kwargs):
            raise Kill()
        patch.setattr(FOVClass, "_assemble_intensities", stop)
        with pytest.raises(Kill):
            F.run_task(path, task)
    folder = _layout.fov_dir(path.parents[1], FOV)
    stopped = json.loads((folder / "run.json").read_text())
    assert stopped["status"] == "interrupted" and list(stopped["checkpoints"]) == ["registered"]
    stage = digests(folder, [f"checkpoints/registered/round{r}.ome.tif" for r in range(1, 5)])
    configuration["execution"]["image_compression"] = "zlib"
    switched = plan(F.write_config(work, "switch", configuration))
    assert switched["tasks"][0]["blocks_sha256"] == manifest["tasks"][0]["blocks_sha256"]
    assert F.run_task(path, task) == 0
    record = json.loads((folder / "run.json").read_text())
    assert record["workflow"]["resumed_from"] == "registered"
    assert digests(folder, stage) == stage
    assert all(codes(folder / name) == {1} for name in stage)
    header = json.loads((folder / "checkpoints" / "registered" / "transforms.json").read_text())
    assert header["image_compression"] == "none"
    assert record["config"]["checkpoints"]["image_compression"] == "zlib"
    assert record["outputs"]["reference_image"]["compression"] == "zlib" and codes(folder / "reference.tif") == {8}
    assert digests(folder, TABLES) == digests(full, TABLES)


# --- WF3: a saved set of either codec -------------------------------------------------------------------------------

@pytest.fixture
def raw_run(work):
    """The fov task from the raw images with the default zlib: its FOV folder."""
    return run(work, "raw", config(work, "raw"), "fov")[1]


@pytest.mark.parametrize("written_by", ["none", "v0.3.0"])
def test_wf3_a_saved_set_of_either_codec_is_read(work, raw_run, written_by, request):
    """A set written with none, or by v0.3.0 (uncompressed, no codec key), is read by a task with
    spot_finding_input and the default zlib: no difference from check_input; the tables equal those of a run from the
    raw images; reference.tif is zlib and decodes equal."""
    stages = ["registered", "candidates", "pre_qc"]
    if written_by == "none":
        _, set_folder = run(work, "set", config(work, "set", "none", checkpoints=stages), "fov")
    else:
        source = request.getfixturevalue("base")
        set_config = F.write_config(work, "set", config(work, "set", checkpoints=stages))
        assert C.run_base(source, "run_workflow", config=str(set_config), tasks=[f"fov/sample/{FOV}"]) == {
            f"fov/sample/{FOV}": 0}
        set_folder = _layout.fov_dir(F.output_root(work, "set"), FOV)
    header = json.loads((set_folder / "checkpoints" / "registered" / "transforms.json").read_text())
    assert header.get("image_compression", "none") == "none"
    assert ("image_compression" in header) == (written_by == "none")
    assert all(codes(path) == {1} for path in (set_folder / "checkpoints" / "registered").glob("*.ome.tif"))
    configuration = config(work, "from-set", checkpoints=["candidates", "pre_qc"],
                           spot_finding_input=str(F.output_root(work, "set")))
    _, folder = run(work, "from-set", configuration, "fov")
    record = json.loads((folder / "run.json").read_text())
    assert record["spot_finding_input"]["path"] == str(F.output_root(work, "set"))
    assert digests(folder, TABLES) == digests(raw_run, TABLES)
    assert codes(folder / "reference.tif") == {8}
    assert C.identity(tifffile.imread(folder / "reference.tif")) == C.identity(
        tifffile.imread(raw_run / "reference.tif"))


# --- WF4: direct and tiled --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("setting", ["zlib", "none"])
def test_wf4_direct_and_tiled_write_the_same_reference_image_and_an_unfiltered_chunked_checkpoint(work, setting):
    """The tiled route with the stage registered kept: reference.tif byte-equal to direct's in the setting's codec,
    recorded by the merge from the file; tiled/registered.h5 has no filter in either setting."""
    _, direct = run(work, "direct", config(work, "direct", setting), "fov")
    tiled = config(work, "tiled", setting, mode="tiled", tiled={"tile_size_yx": 18})
    _, folder = run(work, "tiled", tiled)
    assert (folder / "reference.tif").read_bytes() == (direct / "reference.tif").read_bytes()
    assert codes(folder / "reference.tif") == {CODES[setting]}
    record = json.loads((folder / "run.json").read_text())
    assert record["tiled"]["checkpoint"]["kept"] is True
    assert record["outputs"]["reference_image"]["compression"] == setting
    assert record["config"]["checkpoints"]["image_compression"] == setting
    with h5py.File(_layout.tiled_dir(folder) / _layout.CHUNKED_CHECKPOINT, "r") as store:
        filters = []
        store.visititems(lambda name, item: filters.append(
            (item.compression, item.shuffle, item.fletcher32, item.scaleoffset)) if isinstance(item, h5py.Dataset)
            else None)
    assert filters and set(filters) == {(None, False, False, None)}


# --- MT1: the MATLAB-era channel images --------------------------------------------------------------------------

@pytest.mark.parametrize("projection", [None, "max"])
def test_mt1_the_channel_images_are_unchanged(work, base, projection):
    """With morphology.channel_images on, images/<round>/<label>/<fov>.tif are Compression 1 under both settings and
    byte-equal to the files v0.3.0 writes from the same input."""
    morphology = {"channel_images": {"projection": projection}}
    written = {}
    for setting in ("zlib", "none"):
        _, folder = run(work, f"mt1-{setting}", config(work, f"mt1-{setting}", setting, checkpoints=[],
                                                       morphology=morphology), "morphology")
        images = folder.parents[1] / "images"
        written[setting] = {p.relative_to(images).as_posix(): p.read_bytes() for p in sorted(images.rglob("*.tif"))}
        assert len(written[setting]) == 2 and all(codes(images / name) == {1} for name in written[setting])
    path = F.write_config(work, "mt1-base", config(work, "mt1-base", checkpoints=[], morphology=morphology))
    task = f"morphology/sample/{FOV}"
    assert C.run_base(base, "run_workflow", config=str(path), tasks=[task]) == {task: 0}
    images = F.output_root(work, "mt1-base") / "images"
    old = {p.relative_to(images).as_posix(): p.read_bytes() for p in sorted(images.rglob("*.tif"))}
    assert written["zlib"] == written["none"] == old


# --- ID3: the golden comparison's W-909 exception -------------------------------------------------------------------

@pytest.mark.contract
def test_id3_the_w909_exception_explains_the_two_codec_keys_and_nothing_else(tmp_path):
    """G3's comparison (workflow_golden.compare_migrated, run by benchmarks/orchestration_golden_evidence.py) on a
    copy of the expected values whose run.json files gain the two codec keys: every difference is the W-909
    exception's (migration M3); another new key of config.checkpoints is not; OC3's reasons on the reference image's
    digests name the codec (M4)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("workflow_golden", C.REPOSITORY / "src" / "python" / "scripts" /
                                                  "workflow_golden.py")
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    expected = C.REPOSITORY / "src" / "python" / "test" / "data" / "expected" / "test_workflow_orchestration_golden"
    actual = G.expected_with_exceptions(expected, tmp_path / "actual")
    records = sorted((actual / "records" / "fovs").glob("*/run.json"))
    assert len(records) == 4
    for path in records:
        record = json.loads(path.read_text())
        record["config"]["checkpoints"]["image_compression"] = "zlib"
        record["outputs"]["reference_image"]["compression"] = "zlib"
        path.write_text(json.dumps(record, indent=1) + "\n")
    unexplained, explained = G.compare_migrated(expected, actual, tmp_path / "expected")
    assert unexplained == [] and list(explained) == ["W-909"] and len(explained["W-909"]) == 8
    record = json.loads(records[0].read_text())
    record["config"]["checkpoints"]["codec"] = "zlib"
    records[0].write_text(json.dumps(record, indent=1) + "\n")
    unexplained, _ = G.compare_migrated(expected, actual, tmp_path / "again")
    assert unexplained == ["records/fovs/FOV_001/run.json.config.checkpoints.codec: only in actual"]
    reasons = [reason for decision, _, reason in G.EXCEPTIONS if decision == "OC3"]
    assert [("W-909" in reason) for reason in reasons] == [True, True, True, False]
