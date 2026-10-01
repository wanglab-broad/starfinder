"""Detection in several rounds and the candidates checkpoint (§2.7 checks S10 and S11; W-273).

The ``multiround`` fixture of docs/spot-finding-algorithms.md is hand-built here (no §2.12
generator): 3 rounds × 16×64×64 × 2 uint16 channels on one grid, 15 spots per round and
channel, 5 of them at the same positions in every round. Spots have integer centres on a
lattice of sites 7 voxels apart (Z layers 4 and 11), the iso3d appearance (brightness 1500,
sigma Z 1.5 and YX 1.3, baseline 100, Poisson plus read noise 3) and come from one seed
(100, 101 or 102). Every bound is provisional: option A of the contract is a new interface
and a new checkpoint content without a W-266 reference.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starfinder.barcode import NeighborhoodSumConfig, WtaDecoderConfig
from starfinder.dataset import CheckpointConfig, Dataset, ExecutionConfig, PipelineConfig, RoundState
from starfinder.dataset.workflow import from_workflow_config
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io import read_checkpoint
from starfinder.io._checkpoint import candidates_frame, write_candidates
from starfinder.spot_finding import (ChannelOverride, LocalMaximaConfig, PiscisConfig, SpotFindingPlan,
    SpotFindingResult, SpotiflowConfig, StarfishLogConfig, find_spots)

from .spot_finding_scenes import SEEDS, isolated_scene
from .test_spot_finding_golden import PINNED_FIND_SPOTS, golden_dataset, table_digest

SHAPE_ZYX = (16, 64, 64)
ROUNDS = ("round1", "round2", "round3")
CHANNELS = ("ch00", "ch01")
GRID = ImageMetadata("multiround/grid")
SITES = np.array([(z, y, x) for z in (4, 11) for y in range(7, 57, 7) for x in range(7, 57, 7)], dtype=float)
N_SHARED, N_PER_ROUND = 5, 15
# The starfish ISS tutorial values W-266 used for LoG; an example, not a default.
METHODS = {"local_maxima": LocalMaximaConfig(),
           "starfish_log": StarfishLogConfig(min_sigma=1, max_sigma=10, num_sigma=30, threshold=0.01)}
DATA = Path(__file__).parent / "data" / "spot_finding_candidates_7a178da"
PLAN_KEYS = ("detection_rounds", "detection_plan", "execution", "weights")


def _render(rng, centres):
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in SHAPE_ZYX), indexing="ij")
    signal = np.full(SHAPE_ZYX, 100.0)
    for cz, cy, cx in centres:
        signal += 1500.0 * np.exp(-((z - cz) / 1.5) ** 2 / 2 - ((y - cy) / 1.3) ** 2 / 2 - ((x - cx) / 1.3) ** 2 / 2)
    noisy = rng.poisson(signal) + rng.normal(0.0, 3.0, SHAPE_ZYX)
    return np.clip(np.rint(noisy), 0, 65535).astype(np.uint16)


def multiround(seed):
    """({round: uint16 ZYXC image}, shared centres (2, 5, 3) per channel) of the multiround fixture."""
    rng = np.random.default_rng(seed)
    shared, unique = [], []
    for _ in CHANNELS:
        sites = rng.permutation(len(SITES))
        shared.append(SITES[sites[:N_SHARED]])
        per_round = N_PER_ROUND - N_SHARED
        unique.append([SITES[sites[N_SHARED + i * per_round:N_SHARED + (i + 1) * per_round]]
                       for i in range(len(ROUNDS))])
    images = {name: np.stack([_render(rng, np.vstack([shared[c], unique[c][r]])) for c in range(len(CHANNELS))],
                             axis=-1) for r, name in enumerate(ROUNDS)}
    return images, np.stack(shared)


def multiround_fov(root, images, metadata=None):
    dataset = Dataset(root, root / "out", "multiround", "sample", "out",
                      rounds=RoundState(sequencing_rounds=list(ROUNDS), reference_round="round1"),
                      channel_order=list(CHANNELS))
    fov = dataset.fov("FOV_001")
    for name, image in images.items():
        fov.images[name] = image.copy()
        fov.metadata[name] = (metadata or {}).get(name, GRID)
    return fov


@pytest.fixture(scope="module", params=SEEDS)
def scene(request):
    return multiround(request.param)


def replace_labels(config):
    return replace(config, channel_labels=CHANNELS)


def detected(tmp_path, images, config, rounds=ROUNDS):
    fov = multiround_fov(tmp_path, images)
    fov.find_spots(config=SpotFindingPlan(config, rounds=rounds) if rounds is not None else config)
    return fov.spot_result


# --- S10: identities ------------------------------------------------------------------------------------

def test_the_fixture_stays_within_the_resource_bounds(scene):
    images, shared = scene
    assert len(images) == 3 and all(image.shape == (*SHAPE_ZYX, 2) for image in images.values())
    assert shared.shape == (2, N_SHARED, 3)


@pytest.mark.parametrize("method", METHODS)
def test_every_row_has_its_round_and_the_identities_are_unique(tmp_path, scene, method):
    images, _ = scene
    result = detected(tmp_path, images, METHODS[method])
    spots = result.spots
    assert isinstance(spots["round"].dtype, pd.StringDtype)
    assert spots.spot_id.tolist() == [str(i) for i in range(len(spots))]
    assert list(dict.fromkeys(spots["round"])) == list(ROUNDS)
    labelled = replace_labels(METHODS[method])
    for name in ROUNDS:
        own = find_spots(images[name], config=labelled, metadata=GRID, spot_namespace=result.spot_namespace).spots
        rows = spots[spots["round"] == name].drop(columns=["spot_id", "round"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(rows, own.drop(columns="spot_id"), check_exact=True)
    frame = candidates_frame(result)
    merged = spots.assign(spot_namespace=result.spot_namespace).merge(
        frame, on=["spot_namespace", "spot_id"], validate="one_to_one", suffixes=("", "_frame"))
    assert len(merged) == len(spots) == len(frame)


@pytest.mark.parametrize("method", METHODS)
def test_each_shared_position_gives_one_row_per_round(tmp_path, scene, method):
    images, shared = scene
    spots = detected(tmp_path, images, METHODS[method]).spots
    for c in range(len(CHANNELS)):
        rows = spots[spots.channel == c]
        zyx = rows[["z", "y", "x"]].to_numpy()
        for centre in shared[c]:
            near = rows[np.linalg.norm(zyx - centre, axis=1) <= 1.5]
            assert sorted(near["round"]) == sorted(ROUNDS), (method, c, centre)


@pytest.mark.parametrize("method", METHODS)
def test_the_reference_round_rows_equal_the_default_table(tmp_path, scene, method):
    images, _ = scene
    combined = detected(tmp_path / "rounds", images, METHODS[method]).spots
    default = detected(tmp_path / "default", images, METHODS[method], rounds=None)
    reference = combined[combined["round"] == "round1"].drop(columns="round").reset_index(drop=True)
    pd.testing.assert_frame_equal(reference, default.spots, check_exact=True)
    assert "round" not in default.spots and "rounds" not in default.diagnostics
    assert default.plan == SpotFindingPlan(replace_labels(METHODS[method]))


def test_rounds_follow_the_run_order_and_fov_run_equals_find_spots(tmp_path):
    images, _ = multiround(100)
    plan = SpotFindingPlan(LocalMaximaConfig(), rounds=("round3", "round1"))
    fov = multiround_fov(tmp_path / "find", images)
    fov.find_spots(config=plan)
    assert list(dict.fromkeys(fov.spot_result.spots["round"])) == ["round1", "round3"]
    run = multiround_fov(tmp_path / "run", images)
    run.run(PipelineConfig(detection=plan), checkpoints=CheckpointConfig(stages=(), directory=tmp_path / "ck"))
    pd.testing.assert_frame_equal(run.spot_result.spots, fov.spot_result.spots, check_exact=True)
    assert run.spot_result.plan == fov.spot_result.plan == replace_plan(plan)
    record = json.loads((tmp_path / "ck" / "FOV_001" / "run.json").read_text())
    detections = [s for s in record["steps"] if s["name"] == "find_round_spots"]
    assert [s["round"] for s in detections] == ["round1", "round3"]
    assert all(s["methods"][0]["method"] == "local_maxima" for s in detections)


def replace_plan(plan):
    return SpotFindingPlan(replace_labels(plan.config), plan.channel_overrides, plan.rounds)


def test_multi_round_diagnostics_move_under_rounds(tmp_path):
    images, _ = multiround(101)
    result = detected(tmp_path, images, LocalMaximaConfig())
    diagnostics = result.diagnostics
    assert set(diagnostics["rounds"]) == set(ROUNDS)
    assert not {"thresholds", "counts", "outcomes", "noise"} & set(diagnostics)
    for name in ROUNDS:
        rows = result.spots[result.spots["round"] == name]
        entry = diagnostics["rounds"][name]
        assert entry["counts"] == {label: int((rows.channel == c).sum()) for c, label in enumerate(CHANNELS)}
        assert entry["outcomes"] == {label: "ok" for label in CHANNELS}
        assert len(entry["thresholds"]) == 2 and set(entry["noise"]) == set(CHANNELS)


def test_a_plan_with_overrides_and_rounds(tmp_path):
    images, _ = multiround(102)
    base = replace_labels(LocalMaximaConfig())
    override = LocalMaximaConfig(threshold_value=8.0)
    plan = SpotFindingPlan(base, (ChannelOverride("ch01", override),), ROUNDS)
    fov = multiround_fov(tmp_path, images)
    fov.find_spots(config=plan)
    result = fov.spot_result
    for name in ROUNDS:
        single = find_spots(images[name], config=SpotFindingPlan(base, plan.channel_overrides), metadata=GRID,
                            spot_namespace=result.spot_namespace).spots
        rows = result.spots[result.spots["round"] == name].drop(columns=["spot_id", "round"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(rows, single.drop(columns="spot_id"), check_exact=True)
    assert result.diagnostics["effective_settings"]["ch01"]["threshold_value"] == 8.0
    assert result.plan == plan


# --- Guards -------------------------------------------------------------------------------------------

def test_decoding_a_multi_round_set_raises_naming_the_readout_mode(tmp_path):
    images, _ = multiround(100)
    fov = multiround_fov(tmp_path, images)
    pipeline = PipelineConfig(detection=SpotFindingPlan(LocalMaximaConfig(), rounds=ROUNDS),
                              extraction=NeighborhoodSumConfig(), decoding=WtaDecoderConfig())
    with pytest.raises(ValueError, match="§2.8"):
        fov.run(pipeline)
    assert fov.spot_result is None
    fov.find_spots(config=SpotFindingPlan(LocalMaximaConfig(), rounds=ROUNDS))
    with pytest.raises(ValueError, match=r"readout mode \(§2\.8\)"):
        fov.decode_barcodes()
    with pytest.raises(ValueError, match="§2.8"):
        fov.run(PipelineConfig(extraction=NeighborhoodSumConfig(), decoding=WtaDecoderConfig()))


@pytest.mark.parametrize("change", ["metadata", "shape"])
def test_a_round_off_the_reference_grid_raises(tmp_path, change):
    images, _ = multiround(100)
    if change == "metadata":
        fov = multiround_fov(tmp_path, images, {"round2": ImageMetadata("multiround/other")})
    else:
        fov = multiround_fov(tmp_path, {**images, "round2": images["round2"][:, :48]})
    plan = SpotFindingPlan(LocalMaximaConfig(), rounds=ROUNDS)
    with pytest.raises(IncompatibleGeometryError, match="round2"):
        fov.find_spots(config=plan)
    with pytest.raises(IncompatibleGeometryError, match="round2"):
        fov.run(PipelineConfig(detection=plan))
    # Only the listed rounds are checked.
    fov.find_spots(config=SpotFindingPlan(LocalMaximaConfig(), rounds=("round1", "round3")))


def test_extraction_of_a_multi_round_set_equals_extraction_without_the_round_column(tmp_path):
    images, _ = multiround(101)
    fov = multiround_fov(tmp_path / "rounds", images)
    fov.find_spots(config=SpotFindingPlan(LocalMaximaConfig(), rounds=ROUNDS))
    fov.extract_intensities(config=NeighborhoodSumConfig())
    result = fov.spot_result
    plain = multiround_fov(tmp_path / "plain", images)
    plain.spot_result = SpotFindingResult(result.spots.drop(columns="round"), result.metadata,
                                          result.spot_namespace, result.config, {})
    plain.extract_intensities(config=NeighborhoodSumConfig())
    np.testing.assert_array_equal(fov.intensity_result.values, plain.intensity_result.values)
    np.testing.assert_array_equal(fov.intensity_result.valid, plain.intensity_result.valid)
    assert fov.intensity_result.spot_ids == tuple(result.spots.spot_id)
    # FOV.run extracts every sequencing round at every candidate after the last detected round.
    run = multiround_fov(tmp_path / "run", images)
    run.run(PipelineConfig(detection=SpotFindingPlan(LocalMaximaConfig(), rounds=ROUNDS),
                           extraction=NeighborhoodSumConfig()))
    np.testing.assert_array_equal(run.intensity_result.values, fov.intensity_result.values)
    with pytest.raises(ValueError, match="retain_images"):
        multiround_fov(tmp_path / "stream", images).run(
            PipelineConfig(detection=SpotFindingPlan(LocalMaximaConfig(), rounds=ROUNDS),
                           extraction=NeighborhoodSumConfig()), execution=ExecutionConfig(mode="streaming"))


def test_plan_rounds_are_validated(tmp_path):
    config = LocalMaximaConfig()
    assert SpotFindingPlan(config, rounds=["round1", "round2"]).rounds == ("round1", "round2")
    for rounds in [(), ("round1", "round1"), ("",)]:
        with pytest.raises(ValueError, match="rounds"):
            SpotFindingPlan(config, rounds=rounds)
    with pytest.raises(TypeError, match="rounds"):
        SpotFindingPlan(config, rounds="round1")
    images, _ = multiround(100)
    with pytest.raises(ValueError, match="detects one image"):
        find_spots(images["round1"], config=SpotFindingPlan(config, rounds=("round1",)), metadata=GRID,
                   spot_namespace="multiround")
    with pytest.raises(ValueError, match="round9"):
        multiround_fov(tmp_path, images).find_spots(config=SpotFindingPlan(config, rounds=("round1", "round9")))


def test_the_workflow_rounds_key_builds_a_plan():
    def workflow(block):
        return {"n_rounds": 3, "ref_round": "round1", "dataset_id": "d", "sample_id": "s", "output_id": "o",
                "root_input_path": "in", "root_output_path": "out", "seq_channel_order": list(CHANNELS),
                "rules": {"rsf_single_fov": {"parameters": {"load_raw_images": {"run": False},
                                                            "spot_finding": {"run": True, **block}}}}}
    detection = from_workflow_config(workflow({"rounds": ["round1", "round3"]})).pipeline.detection
    assert detection == SpotFindingPlan(LocalMaximaConfig(), rounds=("round1", "round3"))
    detection = from_workflow_config(workflow({"rounds": ["round2"], "channel_overrides": {
        "ch01": {"threshold_value": 6.0}}})).pipeline.detection
    assert detection.rounds == ("round2",) and detection.channel_overrides[0].channel == "ch01"


# --- S11: candidates checkpoint -------------------------------------------------------------------------

def _round_trip(result, directory, table_format):
    write_candidates(directory, {}, result, None, table_format)
    header = json.loads((directory / "candidates.json").read_text())
    return header, read_checkpoint(directory, "candidates")["spot_result"]


def assert_reloaded(result, reloaded):
    pd.testing.assert_frame_equal(reloaded.spots, result.spots, check_exact=True)
    assert reloaded.config == result.config
    assert reloaded.plan == result.plan


@pytest.mark.parametrize("table_format", ["csv", "parquet"])
@pytest.mark.parametrize("method", METHODS)
def test_multi_round_sets_round_trip_through_the_candidates_checkpoint(tmp_path, scene, method, table_format):
    images, _ = scene
    config = METHODS[method]
    override = {"local_maxima": LocalMaximaConfig(threshold_value=6.0),
                "starfish_log": StarfishLogConfig(min_sigma=1, max_sigma=10, num_sigma=30, threshold=0.02)}[method]
    plan = SpotFindingPlan(config, (ChannelOverride("ch01", override),), ROUNDS)
    fov = multiround_fov(tmp_path, images)
    fov.run(PipelineConfig(detection=plan),
            checkpoints=CheckpointConfig(stages=("candidates",), directory=tmp_path / "ck", table_format=table_format))
    header = json.loads((tmp_path / "ck" / "FOV_001" / "candidates.json").read_text())
    assert header["format_version"] == 2
    assert header["detection_rounds"] == list(ROUNDS)
    assert header["detection_plan"] == [{"channel": "ch01", "config": {**json.loads(json.dumps(
        _plain(override))), "method": method}}]
    assert header["execution"]["device"] == "cpu" and header["weights"] == []
    assert header["dtypes"]["round"] == "string"
    reloaded = multiround_fov(tmp_path, {}).load_checkpoint(
        "candidates", checkpoints=CheckpointConfig(directory=tmp_path / "ck"))
    assert_reloaded(fov.spot_result, reloaded.spot_result)
    assert reloaded.spot_result.diagnostics["rounds"] == fov.spot_result.diagnostics["rounds"]


def _plain(config):
    from starfinder.io._checkpoint import _jsonable
    return {k: v for k, v in _jsonable(config).items() if k != "method"}


@pytest.mark.parametrize("table_format", ["csv", "parquet"])
def test_iso3d_log_tables_round_trip(tmp_path, table_format):
    image, _ = isolated_scene("iso3d", 100)
    result = find_spots(image, config=METHODS["starfish_log"], metadata=ImageMetadata("iso3d"),
                        spot_namespace="iso3d/100")
    assert len(result.spots) and "radius" in result.spots
    header, reloaded = _round_trip(result, tmp_path, table_format)
    assert header["format_version"] == 2 and all(key in header for key in PLAN_KEYS)
    assert header["detection_rounds"] is None and header["detection_plan"] == []
    assert_reloaded(result, reloaded)


def test_a_checkpoint_from_the_start_revision_loads_unchanged(tmp_path):
    header = json.loads((DATA / "FOV_001" / "candidates.json").read_text())
    assert header["format_version"] == 2 and not any(key in header for key in PLAN_KEYS)
    fov = golden_dataset(tmp_path).fov("FOV_001").load_checkpoint(
        "candidates", checkpoints=CheckpointConfig(directory=DATA.parent / DATA.name))
    result = fov.spot_result
    assert table_digest(result.spots) == PINNED_FIND_SPOTS[("global", True, "z1")][1]
    assert result.plan == SpotFindingPlan(result.config) and result.plan.rounds is None
    assert result.config == LocalMaximaConfig(threshold_mode="global", threshold_value=0.01,
                                              channel_labels=("ch00", "ch01", "ch02", "ch03"))
    assert "round" not in result.spots and fov.intensity_result is None


@pytest.fixture
def one_thread(monkeypatch):
    """One numerical thread and no GPU for the learned detectors (the checks also set these before start)."""
    from .learned_detectors import THREAD_VARIABLES
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    torch.set_num_threads(1)


@pytest.mark.extended
@pytest.mark.parametrize("config", [SpotiflowConfig("synth_3d"), SpotiflowConfig("smfish_3d"), PiscisConfig("20230905"),
                                    PiscisConfig("20251212")], ids=lambda c: f"{c.method}-{c.model}")
def test_iso3d_learned_tables_round_trip(tmp_path, one_thread, config):
    pytest.importorskip(config.method)
    from starfinder.spot_finding import _model_artifacts
    image, _ = isolated_scene("iso3d", 100)
    result = find_spots(image, config=config, metadata=ImageMetadata("iso3d"), spot_namespace="iso3d/100")
    assert len(result.spots)
    for table_format in ("csv", "parquet"):
        header, reloaded = _round_trip(result, tmp_path / table_format, table_format)
        assert header["format_version"] == 2 and all(key in header for key in PLAN_KEYS)
        assert header["weights"] == json.loads(json.dumps(_model_artifacts(result.diagnostics["model"])))
        assert header["weights"] and header["execution"]["framework"]["name"] == "torch"
        assert_reloaded(result, reloaded)
