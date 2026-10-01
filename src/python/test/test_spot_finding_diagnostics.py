"""Compact spot-finding diagnostics and the detection overlay (§2.7 check S6, part; W-273).

On the golden fixture (12×48×48, four uint16 channels) and on constructed images of at
most 16×64×64 and two channels: ``counts``, ``outcomes`` (``ok``, ``empty``, ``constant``; a constant
channel never reaches the private method function, which a spy on the registry entry
checks), ``native`` (minimum, median and maximum of LoG's ``radius`` and Spotiflow's
``probability``), ``software``, the typed empty tables and ``plot_detections`` (Agg
backend). Every bound is provisional: new diagnostics without a W-266 reference (W-266's
empty parity case supports only LoG's typed empty table).
"""
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from starfinder.dataset import CheckpointConfig, PipelineConfig  # noqa: E402
from starfinder.image import ImageMetadata  # noqa: E402
from starfinder.spot_finding import (SPOT_FINDING_METHODS, LocalMaximaConfig, PiscisConfig, SpotFindingPlan,  # noqa: E402
    SpotiflowConfig, StarfishLogConfig, find_spots, plot_detections)

from .test_spot_finding_golden import CHANNELS, fixture_image, fov_with_fixture, golden_dataset  # noqa: E402
from .test_spot_finding_rounds import ROUNDS, multiround, multiround_fov, one_thread  # noqa: E402,F401

META = ImageMetadata("diagnostics")
NAMESPACE = "diagnostics/test"
LOG = StarfishLogConfig(min_sigma=1, max_sigma=10, num_sigma=30, threshold=0.01)
DEFAULT_METHODS = {"local_maxima": LocalMaximaConfig(), "starfish_log": LOG}
PAIRS = [("ok", "constant"), ("empty", "constant"), ("ok", "empty")]


def detect(image, config, **options):
    return find_spots(image, config=config, metadata=META, spot_namespace=NAMESPACE, **options)


def constructed(kinds):
    """16×64×64×2 uint16 with the two named channels: ``ok`` (four iso3d-like spots with Poisson
    noise), ``constant`` (100 everywhere) or ``empty`` (an X ramp 0…63).

    The ramp has no interior maximum (local maxima excludes its border face) and a LoG
    response far below 0.01 on the dtype-scaled image.
    """
    rng = np.random.default_rng(100)
    shape = (16, 64, 64)
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in shape), indexing="ij")
    spots = np.full(shape, 100.0)
    for cz, cy, cx in [(5, 10, 12), (8, 30, 40), (11, 50, 20), (6, 40, 52)]:
        spots += 1500.0 * np.exp(-((z - cz) / 1.5) ** 2 / 2 - ((y - cy) / 1.3) ** 2 / 2 - ((x - cx) / 1.3) ** 2 / 2)
    channels = {"ok": rng.poisson(spots), "constant": np.full(shape, 100.0),
                "empty": np.broadcast_to(np.arange(64, dtype=np.float64), shape)}
    return np.clip(np.stack([channels[k] for k in kinds], axis=-1), 0, 65535).astype(np.uint16)


@pytest.fixture
def spy(monkeypatch):
    """Record the channels each call of a method's private function receives."""
    calls = []

    def install(config_type):
        spec = SPOT_FINDING_METHODS[config_type]

        def run(image, config, context):
            calls.append(context.channels)
            return spec.run(image, config, context)

        monkeypatch.setitem(SPOT_FINDING_METHODS, config_type, replace(spec, run=run))
        return calls

    return install


# --- counts and outcomes ---------------------------------------------------------------------------------

@pytest.mark.parametrize("dims", ["3d", "z1"])
@pytest.mark.parametrize("config", [LocalMaximaConfig(channel_labels=CHANNELS), replace(LOG, channel_labels=CHANNELS)],
                         ids=["local_maxima", "starfish_log"])
def test_counts_equal_the_per_channel_rows_of_the_golden_fixture(config, dims):
    result = detect(fixture_image(dims), config)
    rows = result.spots.channel.value_counts()
    assert result.diagnostics["counts"] == {label: int(rows.get(c, 0)) for c, label in enumerate(CHANNELS)}
    assert result.diagnostics["outcomes"] == {label: "ok" if rows.get(c, 0) else "empty"
                                              for c, label in enumerate(CHANNELS)}
    assert sum(result.diagnostics["counts"].values()) == len(result.spots)


@pytest.mark.parametrize("kinds", PAIRS, ids="-".join)
@pytest.mark.parametrize("method", DEFAULT_METHODS)
def test_outcomes_are_ok_empty_or_constant_as_constructed(method, kinds, spy):
    config = replace(DEFAULT_METHODS[method], channel_labels=kinds)
    calls = spy(type(config))
    result = detect(constructed(kinds), config)
    assert result.diagnostics["outcomes"] == {kind: kind for kind in kinds}
    rows = result.spots.channel.value_counts()
    assert result.diagnostics["counts"] == {kind: int(rows.get(c, 0)) for c, kind in enumerate(kinds)}
    assert all((result.diagnostics["counts"][kind] > 0) == (kind == "ok") for kind in kinds)
    # A constant channel never reaches the method function.
    assert calls == [tuple(c for c, kind in enumerate(kinds) if kind != "constant")]
    assert len(result.diagnostics["thresholds"]) == 2


@pytest.mark.parametrize("method", DEFAULT_METHODS)
def test_a_constant_image_does_not_call_the_method_and_gives_the_typed_empty_table(method, spy):
    config = DEFAULT_METHODS[method]
    calls = spy(type(config))
    for image in (np.zeros((8, 32, 32, 2), dtype=np.uint16), np.full((8, 32, 32), 100, dtype=np.uint16)):
        result = detect(image, config)
        assert calls == []
        assert set(result.diagnostics["outcomes"].values()) == {"constant"}
        assert_typed_empty(result, config)


def assert_typed_empty(result, config):
    spec = SPOT_FINDING_METHODS[type(config)]
    declared = [c.rstrip("?") for c in spec.output_columns
                if not c.endswith("?") or getattr(config, spec.column_fields[c.rstrip("?")])]
    spots = result.spots
    assert len(spots) == 0
    assert list(spots.columns) == ["spot_id", *declared]
    assert isinstance(spots.spot_id.dtype, pd.StringDtype)
    assert {c: str(spots[c].dtype) for c in declared} == {c: "int64" if c == "channel" else "float64"
                                                          for c in declared}


def test_the_typed_empty_table_follows_the_optional_columns():
    for measure in (True, False):
        config = LocalMaximaConfig(measure_peak_intensity=measure)
        assert_typed_empty(detect(np.full((4, 16, 16), 7, dtype=np.uint16), config), config)


def test_a_constant_channel_keeps_its_noise_record_and_threshold():
    kinds = ("ok", "constant")
    image = constructed(kinds)
    result = detect(image, replace(DEFAULT_METHODS["local_maxima"], channel_labels=kinds))
    record = result.diagnostics["noise"]["constant"]
    assert record == {"zero_fraction": 0.0, "median": 100.0, "mad": 0.0, "threshold": 100.0}
    assert result.diagnostics["thresholds"][1] == 100.0
    log = detect(image, replace(DEFAULT_METHODS["starfish_log"], channel_labels=kinds))
    assert log.diagnostics["thresholds"] == (0.01, 0.01)
    assert log.diagnostics["geometry"]["scale_space_bytes_estimate"] == 10.4 * 30 * 16 * 64 * 64


# --- native summaries ------------------------------------------------------------------------------------

@pytest.mark.parametrize("dims", ["3d", "z1"])
def test_native_radius_summary_equals_a_direct_computation(dims):
    result = detect(fixture_image(dims), replace(LOG, channel_labels=CHANNELS))
    spots = result.spots
    for c, label in enumerate(CHANNELS):
        radius = spots.loc[spots.channel == c, "radius"].to_numpy()
        expected = ({"min": float(radius.min()), "median": float(np.median(radius)), "max": float(radius.max())}
                    if len(radius) else {"min": None, "median": None, "max": None})
        assert result.diagnostics["native"][label] == {"radius": expected}


def test_local_maxima_has_no_native_columns_and_records_software():
    result = detect(fixture_image("3d"), LocalMaximaConfig())
    assert "native" not in result.diagnostics
    assert set(result.diagnostics["software"]) == {"starfinder", "numpy", "scipy", "scikit-image"}
    assert result.diagnostics["software"]["numpy"] == np.__version__


def test_native_of_a_constant_channel_is_empty():
    kinds = ("ok", "constant")
    result = detect(constructed(kinds), replace(DEFAULT_METHODS["starfish_log"], channel_labels=kinds))
    assert result.diagnostics["native"]["constant"] == {"radius": {"min": None, "median": None, "max": None}}


def test_multi_round_native_and_counts(tmp_path):
    images, _ = multiround(100)
    fov = multiround_fov(tmp_path, images)
    fov.find_spots(config=SpotFindingPlan(LOG, rounds=ROUNDS))
    result = fov.spot_result
    for c, label in enumerate(result.diagnostics["channel_labels"]):
        radius = result.spots.loc[result.spots.channel == c, "radius"]
        assert result.diagnostics["native"][label]["radius"] == {
            "min": float(radius.min()), "median": float(np.median(radius)), "max": float(radius.max())}
    assert sum(sum(r["counts"].values()) for r in result.diagnostics["rounds"].values()) == len(result.spots)


# --- plot_detections ---------------------------------------------------------------------------------------

def plotted(ax):
    (collection,) = ax.collections
    return collection.get_offsets()


@pytest.mark.parametrize("window", [None, ((10, 40), (5, 45))])
def test_plot_detections_draws_one_marker_per_detection_in_the_window(window):
    image = fixture_image("3d")
    result = detect(image, LocalMaximaConfig(channel_labels=CHANNELS))
    z = 6
    ax = plot_detections(image, result, channel="ch00", z=z, yx_window=window)
    spots = result.spots[result.spots.channel == 0]
    (y0, y1), (x0, x1) = window or ((0, 48), (0, 48))
    index = np.rint(spots[["z", "y", "x"]].to_numpy())
    inside = ((index[:, 0] == z) & (index[:, 1] >= y0) & (index[:, 1] < y1) & (index[:, 2] >= x0)
              & (index[:, 2] < x1))
    assert inside.sum() > 0
    np.testing.assert_array_equal(np.asarray(plotted(ax)), spots[["x", "y"]].to_numpy()[inside])
    assert ax.get_images()[0].get_array().shape == (y1 - y0, x1 - x0)
    plt.close(ax.figure)


def test_plot_detections_selects_a_round_and_a_channel_index(tmp_path):
    images, _ = multiround(101)
    fov = multiround_fov(tmp_path, images)
    fov.find_spots(config=SpotFindingPlan(LocalMaximaConfig(), rounds=ROUNDS))
    spots = fov.spot_result.spots
    figure, ax = plt.subplots()
    returned = plot_detections(images["round2"], fov.spot_result, channel=1, z=4, round="round2", ax=ax)
    assert returned is ax
    rows = spots[(spots["round"] == "round2") & (spots.channel == 1) & (np.rint(spots.z) == 4)]
    assert len(plotted(ax)) == len(rows) > 0
    plt.close(figure)
    with pytest.raises(ValueError, match="channel"):
        plot_detections(images["round2"], fov.spot_result, channel="ch09", z=4)


def test_fov_run_writes_no_figure(tmp_path):
    before = plt.get_fignums()
    fov = fov_with_fixture(golden_dataset(tmp_path), "3d")
    fov.run(PipelineConfig(spot_finding=LocalMaximaConfig()), checkpoints=CheckpointConfig(directory=tmp_path / "ck"))
    assert plt.get_fignums() == before
    written = {p.suffix for p in (tmp_path / "ck").rglob("*") if p.is_file()}
    assert not written & {".png", ".pdf", ".svg", ".jpg"}


# --- learned methods (extended tier) --------------------------------------------------------------------

@pytest.mark.extended
@pytest.mark.parametrize("config", [SpotiflowConfig("smfish_3d"), PiscisConfig("20251212")],
                         ids=["spotiflow", "piscis"])
def test_learned_methods_skip_constant_channels_with_typed_empty_tables(config, spy, one_thread):
    pytest.importorskip(config.method)
    calls = spy(type(config))
    result = detect(np.zeros((8, 32, 32, 2), dtype=np.uint16), config)
    assert calls == []
    assert result.diagnostics["outcomes"] == {"0": "constant", "1": "constant"}
    assert_typed_empty(result, config)
    # The weights are still verified, so the model record is there without a model call.
    assert result.diagnostics["model"]["model"] == config.model


@pytest.mark.extended
def test_spotiflow_native_probability_summary(one_thread):
    pytest.importorskip("spotiflow")
    from .spot_finding_scenes import isolated_scene
    image = isolated_scene("iso3d", 100)[0][4:20]   # 16 planes holding the Z layers 8 and 16
    result = detect(np.stack([image, np.full_like(image, 100)], axis=-1), SpotiflowConfig("smfish_3d"))
    probability = result.spots.loc[result.spots.channel == 0, "probability"].to_numpy()
    assert result.diagnostics["native"]["0"] == {"probability": {
        "min": float(probability.min()), "median": float(np.median(probability)), "max": float(probability.max())}}
    assert result.diagnostics["outcomes"] == {"0": "ok", "1": "constant"}
    assert {"spotiflow", "torch"} <= set(result.diagnostics["software"])
