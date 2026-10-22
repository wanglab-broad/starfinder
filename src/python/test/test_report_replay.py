"""The replay of the examples and the image examples of the report (W-454).

Checks of docs/report-algorithms.md ("Validation design"): RP1 to RP3 (the replay against
the saved registered image, its failures, the two sources of the transforms), DF11 (the
bins of ``intensity.csv``), the figures of the Preprocessing and Registration examples and
of the intensity distributions, and RT1's tile part (the segmentation previews on both
routes). DF11 runs on hand-written step outputs; the other checks run the synthetic chains
of ``report_scene.py`` (seed 42) and are ``slow``. Expected values are computed here from the
fixtures' own arrays and files, or come from ``read_checkpoint`` (the reference of RP1),
never from the report's own output.
"""
from __future__ import annotations

import io
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
import pytest
import tifffile

from starfinder.dataset import read_checkpoint
from starfinder._records import _read_table
from starfinder.report import ReportConfig
from starfinder.report._collect import report_examples
from starfinder.report._figures import (EXAMPLE_TILE, intensity_figure, preprocessing_figure,
                                        registration_figure, segmentation_figure)
from starfinder.report._replay import intensity_bins, replay_fov

from .report_scene import (STAGES, Chain, build_chain, scene_chain_plane,  # noqa: F401 (module-scoped fixtures)
                           scene_chain_registered)
from .report_sizes import INTENSITY_SIZE, preprocessing_tile, tiles_width

pytestmark = [pytest.mark.report, pytest.mark.validation]

MAIN = "FOV_001"
NAMED = (MAIN, "FOV_002", "FOV_003")
REPLAY_PARTS = ("preprocessing.example", "preprocessing.intensity", "registration.example")


def tables(folder, record):
    return {e["path"].split("/")[-1][:-4]: _read_table(folder / e["path"], e["dtypes"]) for e in record["files"]
            if "dtypes" in e}


def rows_of(record, parts=REPLAY_PARTS):
    return [(u["part"], u["row_key"], u["reason"]) for u in record["unavailable"] if u["part"] in parts]


def decoded(figure):
    """The figure decoded through matplotlib's image reader (Pillow): (height, width)."""
    from matplotlib.image import imread
    image = imread(io.BytesIO(figure.data), format=figure.format)
    return image.shape[:2]


def percentiles(values):
    lo, hi = np.percentile(np.asarray(values, dtype=np.float64), [0.1, 99.9])
    return float(lo), float(hi)


# --- DF11: the bins of intensity.csv -----------------------------------------------------------------------------

def expected_rows(image, channels, edges):
    """Per channel and bin, the voxels with ``low <= v < high`` (the last float bin closed), as the test counts
    them."""
    image = image if image.ndim == 4 else image[..., None]
    rows = []
    for c, channel in enumerate(channels):
        values = image[..., c].astype(np.float64).ravel()
        for k in range(len(edges) - 1):
            low, high = edges[k], edges[k + 1]
            last = k == len(edges) - 2 and image.dtype.kind == "f"
            count = int(((values >= low) & ((values <= high) if last else (values < high))).sum())
            rows.append((channel, k, low, high, count))
    return rows


def check_bins(image, channels, edges):
    rows = intensity_bins(image, channels)
    expected = expected_rows(image, channels, edges)
    assert [(r[0], r[1], r[4]) for r in rows] == [(r[0], r[1], r[4]) for r in expected]
    np.testing.assert_allclose([r[2:4] for r in rows], [r[2:4] for r in expected], rtol=0, atol=1e-9)
    voxels = image[..., 0].size if image.ndim == 4 else image.size
    for channel in channels:
        assert sum(r[4] for r in rows if r[0] == channel) == voxels
    # The channels of one entry share their edges.
    per_channel = [[r[2:4] for r in rows if r[0] == channel] for channel in channels]
    assert all(edges_of == per_channel[0] for edges_of in per_channel)
    return rows


def test_df11_intensity_bins():
    # uint8 with values 3 to 250: 248 bins of width 1 from 3.
    values = np.arange(3, 251, dtype=np.uint8)
    image = np.stack([values.reshape(1, 8, 31), values[::-1].reshape(1, 8, 31)], axis=-1)
    rows = check_bins(image, ("a", "b"), 3 + np.arange(249, dtype=np.float64) - 0.5)
    assert len(rows) == 2 * 248 and rows[0][2:4] == (2.5, 3.5)
    # uint16 with values 0 to 1000: width ceil(1001 / 256) = 4, ceil(1001 / 4) = 251 bins from 0.
    values = np.arange(0, 1001, dtype=np.uint16)
    image = np.stack([values[:1000].reshape(2, 20, 25), np.full((2, 20, 25), 1000, np.uint16)], axis=-1)
    width = math.ceil(1001 / 256)
    assert width == 4
    rows = check_bins(image, ("a", "b"), np.arange(252, dtype=np.float64) * width - 0.5)
    assert len(rows) == 2 * 251
    # float32 over [0.25, 0.75]: 256 equal bins, the last closed. The values are the bin centres (dyadic, so exact)
    # and both ends.
    step = 0.5 / 256
    centres = (0.25 + (np.arange(256) + 0.5) * step).astype(np.float32)
    values = np.concatenate([centres, np.float32([0.25, 0.75])]).reshape(1, 2, 129)
    image = np.stack([values, values], axis=-1)
    rows = check_bins(image, ("a", "b"), 0.25 + np.arange(257) * step)
    assert len(rows) == 2 * 256
    counts = [r[4] for r in rows if r[0] == "a"]
    assert counts[0] == 2 and counts[-1] == 2 and set(counts[1:-1]) == {1}
    # A constant float output: one bin [lo, hi].
    image = np.full((2, 3, 4, 2), 2.5, dtype=np.float32)
    rows = check_bins(image, ("a", "b"), np.array([2.5, 2.5]))
    assert rows == [("a", 0, 2.5, 2.5, 24), ("b", 0, 2.5, 2.5, 24)]
    # bool: the two bins 0 and 1.
    image = np.zeros((1, 2, 3, 2), dtype=bool)
    image[0, 0, :, 0] = True
    rows = check_bins(image, ("a", "b"), np.array([-0.5, 0.5, 1.5]))
    assert [(r[0], r[1], r[4]) for r in rows] == [("a", 0, 3), ("a", 1, 3), ("b", 0, 6), ("b", 1, 0)]


# --- RP1: the replay against the saved registered image ------------------------------------------------------

@pytest.mark.slow
def test_rp1_replay_equals_the_saved_registered_image(scene_chain_registered):  # noqa: F811
    chain = scene_chain_registered
    start = time.perf_counter()
    replay = replay_fov(chain.root, MAIN, keep_final=True)
    seconds = time.perf_counter() - start
    print(f"\nRP1: the replay of {MAIN}, four rounds, took {seconds:.3f} s")
    saved = read_checkpoint(chain.root / "checkpoints" / MAIN, "registered")
    assert replay.missing is None and replay.round_missing == {}
    assert list(replay.rounds) == chain.rounds
    for name in chain.rounds:
        replayed, stored = replay.rounds[name].final, saved["images"][name]
        assert replayed.dtype == stored.dtype and replayed.shape == stored.shape
        difference = np.abs(replayed.astype(np.float64) - stored.astype(np.float64)).max()
        print(f"RP1: {name} largest absolute difference {difference}")
        assert difference == 0
        assert np.array_equal(replayed, stored)
    # The replayed fitted values equal the recorded ones, entry by entry.
    record = chain.run_record(MAIN)
    steps = replay.record["steps"]
    assert [(s["round"], s["step"]) for s in steps] == [
        (name, e["step"]) for name in chain.rounds for e in record["preprocessing"]["rounds"][name]]
    assert all(s["fitted_equal"] for s in steps) and replay.record["fitted_equal"] is True
    assert replay.record["verified"] is True and replay.record["transform_source"] == "run.json"
    # The two rounds the report replays, through the report.
    folder, report = chain.collect(config=ReportConfig(example_fovs=NAMED))
    assert report["replay"]["rounds"] == {"reference": "round1", "moving": ["round2"]}
    assert report["replay"]["fitted_equal"] is True and report["replay"]["verified"] is True
    assert rows_of(report) == []
    intensity = tables(folder, report)["intensity"]
    assert list(dict.fromkeys(intensity["round"])) == ["round1", "round2"]


# --- RP2: replay failures --------------------------------------------------------------------------------------

def _raise_fitted(chain):
    record = chain.run_record(MAIN)
    record["preprocessing"]["rounds"]["round1"][0]["fitted"]["groups"][0]["max"] += 1
    chain.write_run_record(MAIN, record)


def _change_input(chain):
    path = chain.input_root / "round2" / MAIN / "ch01.ome.tif"
    data = bytearray(path.read_bytes())
    data[-1] = (data[-1] + 1) % 256
    path.write_bytes(bytes(data))


def _remove_input(chain):
    (chain.input_root / "round2" / MAIN / "ch01.ome.tif").unlink()


@pytest.mark.slow
@pytest.mark.parametrize("reason,edit", [("replay_differs", _raise_fitted), ("input_changed", _change_input),
                                         ("input_missing", _remove_input)],
                         ids=["fitted_raised", "input_changed", "input_removed"])
def test_rp2_replay_failures(scene_chain_registered, tmp_path, reason, edit):  # noqa: F811
    chain = scene_chain_registered.copy(tmp_path / "copy")
    edit(chain)
    folder, record = chain.collect(config=ReportConfig(example_fovs=NAMED))
    key = f"fov_id={MAIN};role=example_1"
    assert rows_of(record) == [(part, key, reason) for part in REPLAY_PARTS]
    assert tables(folder, record)["intensity"].empty
    replay = record["replay"]
    assert replay["fov_id"] == MAIN and replay["rounds"] == {"reference": "round1", "moving": ["round2"]}
    if reason == "replay_differs":
        assert replay["fitted_equal"] is False
        assert replay["steps"][0] == {"round": "round1", "step_index": 1, "phase": "steps",
                                      "step": "min_max_normalization", "fitted_equal": False}
    else:
        [listed] = [e for e in replay["inputs"] if e["path"].endswith(f"/round2/{MAIN}/ch01.ome.tif")]
        assert listed["verified"] is True and listed["equal"] is False
        assert (listed["sha256"] is None) == (reason == "input_missing")
        assert replay["steps"] == [] and replay["fitted_equal"] is None


@pytest.mark.slow
def test_rp2_field_not_saved(tmp_path):
    """A recipe with a ``demons`` step and a ``post_registration`` entry, run without the stage ``registered``:
    the Registration example and the outputs of the moving round's ``post_registration`` entry are unavailable
    with ``field_not_saved`` (RE1, PR1); the round's ``input`` and ``steps`` rows of ``intensity.csv`` are
    written, and the reference round's rows of every phase."""
    pytest.importorskip("SimpleITK", reason="the demons step needs the local-registration extra")
    from starfinder.registration import RegistrationRecipe, RegistrationStep
    from starfinder.preprocessing import (MinMaxNormalizationConfig, PreprocessingRecipe, PreprocessingStep,
                                          ReconstructionConfig, TophatConfig)
    from starfinder.registration import DemonsConfig, RegistrationSignalConfig, TranslationConfig
    registration = RegistrationRecipe(steps=(RegistrationStep(TranslationConfig()),
                                             RegistrationStep(DemonsConfig(iterations=(10,)))),
                                      signal=RegistrationSignalConfig(mode="max"))
    preprocessing = PreprocessingRecipe(
        (PreprocessingStep(MinMaxNormalizationConfig("uint8", (0, 255), rounding="truncate")),
         PreprocessingStep(TophatConfig(radius_yx=3))),
        post_registration=(PreprocessingStep(ReconstructionConfig()),))
    chain = build_chain(tmp_path / "demons", stages=STAGES, registration=registration, preprocessing=preprocessing,
                        fov_ids=[MAIN], export=False)
    run = chain.run_record(MAIN)
    assert [r["transform"]["kind"] for r in run["transforms"]["round2"]["results"]] == ["translation", "dense"]
    assert run["transforms"]["round2"]["results"][1]["transform"]["field"] is None
    folder, record = chain.collect()
    key = f"fov_id={MAIN};role=example_1"
    assert rows_of(record) == [
        ("preprocessing.example", f"{key};round=round2;phase=post_registration", "field_not_saved"),
        ("preprocessing.intensity", f"{key};round=round2;phase=post_registration", "field_not_saved"),
        ("registration.example", key, "field_not_saved")]
    intensity = tables(folder, record)["intensity"]
    phases = intensity.groupby("round", sort=False)["phase"].unique().apply(list).to_dict()
    assert phases == {"round1": ["input", "steps", "post_registration"], "round2": ["input", "steps"]}
    steps = intensity.drop_duplicates(["round", "step_index"])[["round", "step_index", "phase", "step"]]
    assert steps.values.tolist() == [["round1", 0, "input", "input"], ["round1", 1, "steps", "min_max_normalization"],
                                     ["round1", 2, "steps", "white_tophat"],
                                     ["round1", 3, "post_registration", "reconstruction"],
                                     ["round2", 0, "input", "input"], ["round2", 1, "steps", "min_max_normalization"],
                                     ["round2", 2, "steps", "white_tophat"]]
    assert record["replay"]["fitted_equal"] is True and record["replay"]["transform_source"] is None


# --- RP3: the two sources of the transforms ---------------------------------------------------------------------

@pytest.mark.slow
def test_rp3_two_sources_of_the_transforms(scene_chain_registered, tmp_path):  # noqa: F811
    chain = scene_chain_registered
    older = chain.copy(tmp_path / "older")
    for fov in older.fov_ids:
        record = older.run_record(fov)
        del record["transforms"]
        older.write_run_record(fov, record)
    first = replay_fov(chain.root, MAIN, keep_final=True)
    second = replay_fov(older.root, MAIN, keep_final=True)
    assert list(first.rounds) == list(second.rounds) == chain.rounds
    for name in chain.rounds:
        a, b = first.rounds[name], second.rounds[name]
        assert np.array_equal(a.final, b.final) and a.final.dtype == b.final.dtype
        for x, y in ((a.before, b.before), (a.after, b.after)):
            assert (x is None) == (y is None) and (x is None or np.array_equal(x, y))
        assert [(s.step_index, s.phase, s.step) for s in a.states] == [(s.step_index, s.phase, s.step)
                                                                       for s in b.states]
        for s, t in zip(a.states, b.states):
            assert all(np.array_equal(u, v) and u.dtype == v.dtype for u, v in zip(s.signal, t.signal))
        assert a.rows == b.rows
    assert first.record["transform_source"] == "run.json"
    assert second.record["transform_source"] == "transforms.json"
    # Through the report: the same intensity.csv, the transform source of each.
    one = chain.collect(config=ReportConfig(example_fovs=NAMED), output_dir=tmp_path / "report_run_json")
    two = older.collect(config=ReportConfig(example_fovs=NAMED))
    assert (one[0] / "tables" / "intensity.csv").read_bytes() == (two[0] / "tables" / "intensity.csv").read_bytes()
    assert (one[1]["replay"]["transform_source"], two[1]["replay"]["transform_source"]) == ("run.json",
                                                                                            "transforms.json")


# --- The figures ---------------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_figures_of_the_examples(scene_chain_registered):  # noqa: F811
    chain = scene_chain_registered
    replay, previews = report_examples(chain.root, chain.sample, chain.fov_ids, dataset_id=chain.dataset_id,
                                       sample_id=chain.sample, config=ReportConfig(example_fovs=NAMED))
    record = chain.run_record(MAIN)
    channels = [c["channel"] for c in record["channels"]["round1"]]
    entries = record["preprocessing"]["rounds"]["round1"]
    # The Preprocessing grid: the reference round, one row per channel, one column per entry plus the input.
    figure = preprocessing_figure(replay)
    columns, rows = len(entries) + 1, len(channels)
    size = (figure.height, tiles_width(columns, preprocessing_tile(columns)))
    assert figure.width == size[1] and figure.height > rows * preprocessing_tile(columns) and decoded(figure) == size
    assert figure.format == "jpeg"
    assert figure.round == "round1"
    assert [(t["step_index"], t["channel"]) for t in figure.tiles] == [(j, c) for j in range(columns)
                                                                       for c in channels]
    # Each tile's range: the 0.1th and 99.9th percentile of its displayed values (36 px < 256 px: one pixel per
    # block). The input from the raw files, the last state from the saved registered image (the round has no
    # post_registration entry), the others from the replayed states.
    raw = {c: tifffile.imread(chain.input_root / "round1" / MAIN / f"{c}.ome.tif") for c in channels}
    saved = read_checkpoint(chain.root / "checkpoints" / MAIN, "registered")["images"]
    states = replay.rounds["round1"].states
    for tile in figure.tiles:
        c = channels.index(tile["channel"])
        if tile["step_index"] == 0:
            displayed = np.asarray(raw[tile["channel"]]).reshape(saved["round1"].shape[:3]).max(axis=0)
        elif tile["step_index"] == columns - 1:
            displayed = saved["round1"][..., c].max(axis=0)
        else:
            displayed = states[tile["step_index"]].signal[c]
        assert (tile["lo"], tile["hi"]) == pytest.approx(percentiles(displayed), abs=1e-9)
    # The intensity panels: one per column of the grid.
    figure = intensity_figure(replay)
    assert (figure.width, figure.height) == INTENSITY_SIZE and decoded(figure) == INTENSITY_SIZE[::-1]
    assert figure.format == "png" and figure.round == "round1" and len(figure.tiles) == columns
    # The Registration example: the first moving round, three tiles of 512 px.
    figure = registration_figure(replay)
    size = (figure.height, tiles_width(3, EXAMPLE_TILE))
    assert figure.width == size[1] and figure.height > EXAMPLE_TILE and decoded(figure) == size
    assert figure.round == "round2" and [t["round"] for t in figure.tiles] == ["round2"] * 3
    reference = saved["round1"].max(axis=(0, 3))
    after = saved["round2"].max(axis=(0, 3))
    before = np.max([s for s in replay.rounds["round2"].states[-1].signal], axis=0)
    assert figure.tiles[0]["moving"] == pytest.approx(percentiles(before), abs=1e-9)
    for tile in figure.tiles:
        assert tile["reference"] == pytest.approx(percentiles(reference), abs=1e-9)
    for tile in figure.tiles[1:]:
        assert tile["moving"] == pytest.approx(percentiles(after), abs=1e-9)
    # The segmentation examples on the route per FOV: three tiles of 512 px, titled with their FOVs.
    figure = segmentation_figure(previews)
    size = (figure.height, tiles_width(3, EXAMPLE_TILE))
    assert figure.width == size[1] and figure.height > EXAMPLE_TILE and decoded(figure) == size
    assert [t["fov_id"] for t in figure.tiles] == list(NAMED)


# --- RT1: the segmentation previews on both routes ----------------------------------------------------------------

def outline_of(labels):
    """The pixels whose label differs from one of their four neighbours inside the image, counted pixel by pixel."""
    ny, nx = labels.shape
    out = np.zeros(labels.shape, dtype=bool)
    for y in range(ny):
        for x in range(nx):
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if 0 <= y + dy < ny and 0 <= x + dx < nx and labels[y + dy, x + dx] != labels[y, x]:
                    out[y, x] = True
    return out


@pytest.mark.slow
@pytest.mark.parametrize("route", ["per_fov", "after_fusion"])
def test_rt1_segmentation_previews(scene_chain_plane, route):  # noqa: F811
    chain: Chain = scene_chain_plane[route]
    folder, record = chain.collect()
    assert record["route"] == route
    assert [u for u in record["unavailable"] if u["part"] == "assembly.examples"] == []
    replay, previews = report_examples(chain.root, chain.sample, chain.fov_ids, dataset_id=chain.dataset_id,
                                       sample_id=chain.sample)
    roles = {role: entry["fov_id"] for role, entry in record["examples"].items()}
    assert {role: p.fov_id for role, p in previews.items()} == roles
    export_dir = chain.root / "export" / chain.sample
    export = json.loads((export_dir / "export.json").read_text())
    fovs = pd.read_csv(export_dir / "tables" / "fovs.csv", float_precision="round_trip").set_index("fov_id")
    for role, preview in previews.items():
        fov = preview.fov_id
        if route == "after_fusion":
            # The fused label image over the footprint's pixel range, floor((q − O) / S + ½), end excluded.
            section = export["segmentation"]
            grid = section["inputs"]["grid"]
            (_, sy, sx), (_, oy, ox) = grid["spacing_zyx"], grid["origin_zyx"]
            row = fovs.loc[fov]
            y0, y1 = (int(np.floor((row[k] - oy) / sy + 0.5)) for k in ("footprint_y0", "footprint_y1"))
            x0, x1 = (int(np.floor((row[k] - ox) / sx + 0.5)) for k in ("footprint_x0", "footprint_x1"))
            runs = {r["name"]: r for r in section["runs"]}
            cells = tifffile.imread(export_dir / runs[section["cells"]]["path"])
            nuclei = tifffile.imread(export_dir / runs[section["nuclei"]]["path"])
            assert cells.shape == tuple(grid["shape_zyx"][1:])
            y0, y1, x0, x1 = max(0, y0), min(cells.shape[0], y1), max(0, x0), min(cells.shape[1], x1)
            expected_cells, expected_nuclei = cells[y0:y1, x0:x1], nuclei[y0:y1, x0:x1]
            assert preview.window == (y0, y1, x0, x1)
        else:
            # The FOV's own label images (one plane).
            folder_of = chain.root / "checkpoints" / fov / "assignment" / "default"
            expected_cells = tifffile.imread(folder_of / "cell_labels.tif").reshape(-1, *preview.cells.shape)[0]
            expected_nuclei = tifffile.imread(folder_of / "nucleus_labels.tif").reshape(-1, *preview.cells.shape)[0]
        assert preview.route == route
        assert np.array_equal(preview.cells, expected_cells) and np.array_equal(preview.nuclei, expected_nuclei)
        assert np.array_equal(preview.cell_outline, outline_of(expected_cells))
        assert np.array_equal(preview.nucleus_outline, outline_of(expected_nuclei))
        assert preview.cells.any()
    figure = segmentation_figure(previews)
    size = (figure.height, tiles_width(3, EXAMPLE_TILE))
    assert decoded(figure) == size and [t["fov_id"] for t in figure.tiles] == [roles[r] for r in
                                                                                ("example_1", "example_2",
                                                                                 "example_3")]
    shutil.rmtree(folder)
