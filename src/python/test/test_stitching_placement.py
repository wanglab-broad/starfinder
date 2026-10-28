"""PL3 to PL11 of docs/stitching-assembly-validation.md, their stub parts, and PL4's scene part.

Fixtures: ``stub_grid`` and ``stub_variants`` (fractional translations, spacing (1, 2, 2) µm
and the LN calibration, Z = 1 grids from YX inputs, unequal shapes, unsnapped stage
positions, the L shape, a NaN-quality FOV, a moved FOV, two components, the review
boundaries) with the test-only method of ``stitching_fixtures``; ``scene_tiny``
(``sample-tiny-v1``, seed 42) with that method returning the true corrections.
"""
import dataclasses
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import tifffile

from starfinder.image import IncompatibleGeometryError
from starfinder.segmentation import ReferenceGrid, reference_grid_from_file
from starfinder.stitching import (STITCHING_METHODS, PlacementResult, PlacementReviewRequiredError, ReviewPolicy,
                                  StitchingInput, StitchingSpec, approve_placement, load_placement, placed_metadata,
                                  placement_from_table, positions_from_table, review_placement, save_placement,
                                  stitch)
from starfinder.stitching._review import require_approval

from . import hash_oracles
from .stitching_fixtures import (FRACTIONAL, LN_SPACING, MAF_2X3_JITTER, MAF_3X2_GAP, STAGE_SPACING, STUB_SHAPE,
                                 STUB_SPACING, STUB_SPEC, StubConfig, check_components, grid, grids_for, inputs_for,
                                 metadata, stage_table, stub_run, stub_table, stub_variant_results, zero)

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

STATUSES = {"image_supported", "metadata_only", "placeholder", "missing", "failed"}
PLACED = {"image_supported", "metadata_only", "placeholder"}
ORIGIN = ["origin_z", "origin_y", "origin_x"]
NOMINAL = ["nominal_z", "nominal_y", "nominal_x"]
CORRECTION = ["correction_z", "correction_y", "correction_x"]
TRANSLATION = ["method_translation_z", "method_translation_y", "method_translation_x"]
TINY = 2.0 ** -20


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setitem(STITCHING_METHODS, StubConfig, STUB_SPEC)


def run(table, grids, translations, *, overrides=(), absent=(), yx=False, **kw):
    inputs = inputs_for(table, grids, absent=absent, yx=yx)
    result = stitch(inputs, table, config=StubConfig(tuple(map(tuple, translations)), tuple(overrides)),
                    sample="s1", **kw)
    check_components(result)
    return result


def bits(frame, columns):
    return frame[columns].to_numpy(dtype=np.float64).tobytes()


def check_record(result, table):
    """The record assertions of design/specs/stitching.md ("Method record")."""
    fovs = result.fovs
    assert list(fovs.fov_id) == table.fov_ids
    assert set(fovs.status) <= STATUSES
    assert all(fovs[c].dtype == np.float64 for c in ORIGIN + NOMINAL + CORRECTION + TRANSLATION)
    finite = np.isfinite(fovs[ORIGIN].to_numpy()).all(axis=1)
    assert list(finite) == [s in PLACED for s in fovs.status]
    assert bits(fovs, NOMINAL) == table.nominal_zyx().tobytes()
    difference = fovs[ORIGIN].to_numpy() - fovs[NOMINAL].to_numpy()
    assert difference.tobytes() == fovs[CORRECTION].to_numpy().tobytes()
    supported = (fovs.status == "image_supported").to_numpy()
    if supported.any():
        assert np.abs(fovs.loc[supported, CORRECTION].to_numpy().mean(axis=0)).max() <= 1e-9
    kept = result.links[result.links.kept]
    counts = [int(((kept.fov_a == f) | (kept.fov_b == f)).sum()) for f in fovs.fov_id]
    assert list(fovs.n_links_kept) == counts
    assert list(fovs.component == -1) == list(fovs.status == "missing")
    for fov in table.fov_ids:
        entry = result.record["inputs"][fov]
        assert set(entry) == {"source", "file_sha256", "grid"}
        assert set(entry["grid"]) >= {"shape_zyx", "metadata", "sha256"}
    assert result.record["frame"] == {"frame_id": "sample:s1", "unit": "µm", "direction": "identity"}
    assert result.record["format_version"] == 1 and result.record["stage"] == "stitching"


def test_fractional_translations_give_the_g1_gauge_bitwise(stub):
    table = stub_table()
    grids = grids_for(table)
    before = {fov: dataclasses.replace(g) for fov, g in grids.items()}
    result = run(table, grids, FRACTIONAL)
    c = np.array(FRACTIONAL)
    nominal = table.nominal_zyx()
    expected = nominal + c - c.mean(axis=0)
    assert set(result.fovs.status) == {"image_supported"}
    assert bits(result.fovs, ORIGIN) == expected.tobytes()
    assert bits(result.fovs, TRANSLATION) == c.tobytes()
    assert (np.round(expected * 64) != expected * 64).any()   # the gauge mean is not a multiple of 2⁻⁶
    assert (result.fovs[ORIGIN].to_numpy() * 64 % 1 != 0).any()
    check_record(result, table)
    assert result.record["gauge"] == "G1"
    assert result.record["gauge_mean_correction_zyx"] == list(c.mean(axis=0))
    method = result.record["method"]
    assert (method["stage"], method["method"]) == ("stitching", "stub")
    assert method["config"]["translations"] == [list(t) for t in FRACTIONAL]
    assert result.record["method_diagnostics"] == {"components": [0] * 6, "pairs": 11}
    assert result.record["review"]["state"] == "not_needed" and result.record["review"]["reasons"] == []
    assert grids == before


def test_scene_origins_equal_the_g1_truth(monkeypatch, tmp_path):
    from starfinder.synthetic import generate_sample_scene, sample_scene_preset, save_sample_scene
    book, config = sample_scene_preset("sample-tiny-v1")
    scene = generate_sample_scene(book, config=config)
    save_sample_scene(scene, tmp_path)
    table = positions_from_table(tmp_path / "positions.csv")
    truth = pd.read_csv(tmp_path / "placement_truth.csv", float_precision="round_trip").set_index("fov_id")
    shape = config.acquisition.fov_shape_zyx
    grids = {fov: ReferenceGrid(shape, scene.fov_metadata[fov], "declared") for fov in table.fov_ids}
    metadata_before = dict(scene.fov_metadata)
    corrections = truth.loc[table.fov_ids, ["true_correction_z", "true_correction_y", "true_correction_x"]]
    monkeypatch.setitem(STITCHING_METHODS, StubConfig, STUB_SPEC)
    inputs = [StitchingInput(fov, scene.stains[fov], grids[fov], "reference_stain") for fov in table.fov_ids]
    result = stitch(inputs, table, config=StubConfig(tuple(map(tuple, corrections.to_numpy()))), sample="s1")
    assert set(result.fovs.status) == {"image_supported"}
    g1 = truth.loc[table.fov_ids, ["true_origin_g1_z", "true_origin_g1_y", "true_origin_g1_x"]].to_numpy()
    assert np.abs(result.fovs[ORIGIN].to_numpy() - g1).max() <= 1e-9
    check_record(result, table)
    assert scene.fov_metadata == metadata_before


def test_index_to_sample_rule_and_its_inverse(stub):
    table = stub_table()
    grids = grids_for(table)
    result = run(table, grids, FRACTIONAL)
    for fov in table.fov_ids:
        placed = placed_metadata(result, fov)
        origin = result.fovs.set_index("fov_id").loc[fov, ORIGIN].to_numpy(dtype=np.float64)
        spacing = np.array(STUB_SPACING)
        assert placed.frame_id == "sample:s1" and placed.spacing_zyx == STUB_SPACING
        for p in ([0, 0, 0], [3, 15, 23], [1.25, 7.5, 0.375], [-0.5, 2.0, 23.5]):
            push = origin + np.array(p) * spacing
            assert np.abs(placed.index_to_world(p) - push).max() <= 1e-9
            assert np.abs((push - origin) / spacing - np.array(p)).max() <= 1e-9
            assert np.abs(placed.world_to_index(push) - np.array(p)).max() <= 1e-9
        start = (1.0, 4.0, 6.0)
        cropped = placed.cropped(start, frame_id="crop")
        q = np.array([0.5, 2.0, 3.0])
        assert np.abs(cropped.index_to_world(q) - placed.index_to_world(q + np.array(start))).max() <= 1e-9


def test_contract_example_of_the_index_rule():
    frame = pd.DataFrame({"fov_id": ["a"], "nominal_z": [0.0], "nominal_y": [250.0], "nominal_x": [100.0]})
    result = placement_from_table(positions_from_table(frame), {"a": grid((50, 1496, 1496), LN_SPACING)},
                                  sample="s1", origins={"a": (0.0, 250.5, 100.25)})
    point = placed_metadata(result, "a").index_to_world((2, 10, 0))
    assert np.abs(point - np.array([0.69251, 252.44398, 100.25])).max() <= 1e-9


@pytest.mark.parametrize("spacing", [(1.0, 2.0, 2.0), LN_SPACING], ids=["spacing_1_2_2", "ln_calibration"])
def test_method_translations_stay_in_micrometres(stub, spacing):
    table = stub_table(spacing_yx=spacing[1:])
    grids = grids_for(table, spacing=spacing)
    result = run(table, grids, FRACTIONAL)
    c = np.array(FRACTIONAL)
    expected = table.nominal_zyx() + c - c.mean(axis=0)
    assert set(result.fovs.status) == {"image_supported"}
    assert np.abs(result.fovs[ORIGIN].to_numpy() - expected).max() <= 1e-9
    assert bits(result.fovs, TRANSLATION) == c.tobytes()
    check_record(result, table)


def test_yx_inputs_give_z1_grids_and_keep_the_method_z(stub, tmp_path):
    path = tmp_path / "plane.tif"
    tifffile.imwrite(path, np.zeros((16, 24), np.uint8))
    plane = reference_grid_from_file(path, metadata=metadata())
    assert plane.shape_zyx == (1, 16, 24)
    table = stub_table()
    grids = {fov: grid((1, 16, 24)) for fov in table.fov_ids}
    result = run(table, grids, zero(6), yx=True)
    assert set(result.fovs.status) == {"image_supported"}
    assert (result.fovs.origin_z == 0.0).all() and (result.fovs.nominal_z == 0.0).all()
    assert all(entry["grid"]["shape_zyx"] == [1, 16, 24] for entry in result.record["inputs"].values())


@pytest.mark.parametrize("change,match", [
    (((1, 16, 24), STUB_SPACING), r"one Z size; FOV0, FOV1, FOV2, FOV3, FOV4: 4; FOV5: 1"),
    ((STUB_SHAPE, (0.25, 0.25, 0.5)), r"one spacing; .*FOV5: \(0.25, 0.25, 0.5\)"),
])
def test_mixed_geometry_is_refused_naming_the_fovs(stub, change, match):
    table = stub_table()
    grids = grids_for(table, FOV5=change)
    with pytest.raises(IncompatibleGeometryError, match=match):
        run(table, grids, zero(6))


@pytest.mark.parametrize("kw,match", [
    (dict(spacing=None), "unknown spacing: FOV2"),
    (dict(unit="nm"), "a unit other than µm: FOV2 \\('nm'\\)"),
    (dict(direction=((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))), "a direction other than the identity: FOV2"),
])
def test_grids_outside_the_sample_frame_are_refused(stub, kw, match):
    table = stub_table()
    grids = grids_for(table)
    spacing = kw.pop("spacing", STUB_SPACING)
    grids["FOV2"] = ReferenceGrid(STUB_SHAPE, metadata(spacing, **kw) if spacing else
                                  dataclasses.replace(metadata(), spacing_zyx=None), "declared")
    with pytest.raises(IncompatibleGeometryError, match=match):
        run(table, grids, zero(6))


@dataclasses.dataclass(frozen=True)
class VolumeOnlyConfig(StubConfig):
    pass


def test_a_spec_without_dimension_2_refuses_z1_inputs(monkeypatch):
    monkeypatch.setitem(STITCHING_METHODS, VolumeOnlyConfig,
                        StitchingSpec("volume_only", stub_run, dimensions=frozenset({3}), min_shape_zyx=(2, 16, 16)))
    table = stub_table()
    planes = {fov: grid((1, 16, 24)) for fov in table.fov_ids}
    with pytest.raises(IncompatibleGeometryError, match="'volume_only' does not accept Z=1 tiles"):
        stitch(inputs_for(table, planes), table, config=VolumeOnlyConfig(zero(6)))
    small = grids_for(table, FOV1=((4, 8, 24), STUB_SPACING))
    with pytest.raises(IncompatibleGeometryError, match=r"at least \(2, 16, 16\); FOVs FOV1 \(4, 8, 24\)"):
        stitch(inputs_for(table, small), table, config=VolumeOnlyConfig(zero(6)))


def test_non_square_tiles_place_with_their_own_sizes(stub):
    table = stub_table()
    grids = grids_for(table, FOV2=((4, 16, 20), STUB_SPACING))
    result = run(table, grids, zero(6))
    assert set(result.fovs.status) == {"image_supported"}
    origins = result.fovs.set_index("fov_id")
    for fov, (n_y, n_x) in (("FOV0", (16, 24)), ("FOV2", (16, 20))):
        corner = placed_metadata(result, fov).index_to_world((0, n_y - 1, n_x - 1))
        expected = origins.loc[fov, ORIGIN].to_numpy(dtype=np.float64) + np.array([0.0, (n_y - 1) * 0.25,
                                                                                    (n_x - 1) * 0.25])
        assert np.abs(corner - expected).max() <= 1e-9
    assert result.record["inputs"]["FOV2"]["grid"]["shape_zyx"] == [4, 16, 20]


@pytest.mark.parametrize("name,rows", [("maf_2x3_jitter", 6), ("maf_3x2_gap", 5)])
def test_irregular_footprints_are_placed_and_joined(stub, tmp_path, name, rows):
    points = {"maf_2x3_jitter": MAF_2X3_JITTER, "maf_3x2_gap": MAF_3X2_GAP}[name]
    table = stage_table(tmp_path, points, name)
    grids = {fov: grid(STUB_SHAPE, STAGE_SPACING) for fov in table.fov_ids}
    result = run(table, grids, zero(rows))
    assert len(result.fovs) == rows and list(result.fovs.fov_id) == table.fov_ids
    assert set(result.fovs.status) == {"image_supported"} and set(result.fovs.component) == {0}
    if name == "maf_3x2_gap":
        # The missing position (column 1, row 1) has no row, and its centre lies in no footprint.
        assert "Position006" not in set(result.fovs.fov_id)
        centre = np.array([261.0 + 144.0, 261.0 + 144.0])
        covered = [np.all((o <= centre) & (centre <= o + 288.0))
                   for o in result.fovs[["origin_y", "origin_x"]].to_numpy()]
        low, high = result.fovs[["origin_y", "origin_x"]].min(), result.fovs[["origin_y", "origin_x"]].max() + 288
        assert not any(covered) and np.all((low.to_numpy() <= centre) & (centre <= high.to_numpy()))


def nan_links(fov, pairs_of):
    return tuple((a, b, np.nan, False, "nan_quality", (0.0, 0.0, 0.0)) for a, b in pairs_of if fov in (a, b))


STUB_PAIRS = [("FOV0", "FOV1"), ("FOV0", "FOV2"), ("FOV0", "FOV3"), ("FOV1", "FOV2"), ("FOV1", "FOV3"),
              ("FOV2", "FOV3"), ("FOV2", "FOV4"), ("FOV2", "FOV5"), ("FOV3", "FOV4"), ("FOV3", "FOV5"),
              ("FOV4", "FOV5")]


def test_a_tile_with_only_nan_quality_links(stub):
    table = stub_table()
    grids = grids_for(table)
    overrides = nan_links("FOV3", STUB_PAIRS)
    for fallback, status in ((None, "failed"), ("nominal", "metadata_only")):
        result = run(table, grids, FRACTIONAL, overrides=overrides, fallback=fallback)
        fovs = result.fovs.set_index("fov_id")
        nan = result.links[result.links.quality.isna()]
        assert len(nan) == 5 and not nan.kept.any() and set(nan.reason) == {"nan_quality"}
        assert "nan_quality_link" in fovs.loc["FOV3", "flags"].split(";")
        assert fovs.loc["FOV3", "status"] == status and fovs.loc["FOV3", "n_links_kept"] == 0
        assert result.record["review"]["state"] == "required"
        assert result.record["review"]["reasons"][0] == "unplaced_fovs"
        if fallback is None:
            assert fovs.loc["FOV3", ORIGIN].isna().all()
            assert fovs.loc["FOV3", "flags"] == "nan_quality_link"
        else:
            assert bits(fovs.loc[["FOV3"]], ORIGIN) == bits(fovs.loc[["FOV3"]], NOMINAL)
            assert fovs.loc["FOV3", "flags"] == "fallback;nan_quality_link"
        others = fovs.drop(index="FOV3")
        assert set(others.status) == {"image_supported"}
        check_record(result, table)


def moved_table():
    frame = stub_table().positions.drop(columns=["grid_column", "grid_row"])
    frame.loc[5, "nominal_x"] += 2 * 6.0
    return positions_from_table(frame)


def test_stage_position_fallback_for_a_disconnected_tile(stub):
    table = moved_table()
    grids = grids_for(table)
    nominal = table.nominal_zyx()
    failed = run(table, grids, FRACTIONAL)
    fallback = run(table, grids, FRACTIONAL, fallback="nominal")
    assert not ((failed.links.fov_a == "FOV5") | (failed.links.fov_b == "FOV5")).any()
    for result in (failed, fallback):
        fovs = result.fovs.set_index("fov_id")
        assert set(fovs.drop(index="FOV5").status) == {"image_supported"}
        assert fovs.loc["FOV5", "component"] == 1
        assert result.record["review"]["state"] == "required"
        assert result.record["review"]["flagged"]["unplaced_fovs"] == ["FOV5"]
        assert bits(fovs.loc[["FOV5"]], TRANSLATION) == np.array(FRACTIONAL[5]).tobytes()
        check_record(result, table)
    assert failed.fovs.status[5] == "failed" and np.isnan(failed.fovs.loc[5, ORIGIN].to_numpy()).all()
    assert failed.fovs["flags"][5] == "" and failed.record["fallback"] is None
    assert fallback.fovs.status[5] == "metadata_only" and fallback.fovs["flags"][5] == "fallback"
    assert fallback.fovs.loc[5, ORIGIN].to_numpy(dtype=np.float64).tobytes() == nominal[5].tobytes()
    assert (fallback.fovs.loc[5, CORRECTION] == 0.0).all() and fallback.record["fallback"] == "nominal"
    # The gauge uses only image_supported FOVs, so the fallback changes no other origin.
    assert bits(failed.fovs.iloc[:5], ORIGIN) == bits(fallback.fovs.iloc[:5], ORIGIN)


def two_by_two():
    return stub_table([f"FOV{k}" for k in range(4)], columns=2, rows=2)


def dropped(pairs):
    return tuple((a, b, 0.5, False, "quality_below_threshold", (0.0, 0.0, 0.0)) for a, b in pairs)


def test_components_and_the_reference_component(stub):
    table = two_by_two()
    grids = grids_for(table)
    across = [("FOV0", "FOV2"), ("FOV0", "FOV3"), ("FOV1", "FOV2"), ("FOV1", "FOV3")]
    tie = run(table, grids, zero(4), overrides=dropped(across))
    assert list(tie.fovs.component) == [0, 0, 1, 1]
    assert list(tie.fovs.status) == ["image_supported", "image_supported", "failed", "failed"]
    first_alone = run(table, grids, zero(4), overrides=dropped([("FOV0", "FOV1"), ("FOV0", "FOV2"),
                                                                 ("FOV0", "FOV3")]))
    assert list(first_alone.fovs.component) == [0, 1, 1, 1]
    assert list(first_alone.fovs.status) == ["failed", "image_supported", "image_supported", "image_supported"]
    nominal = run(table, grids, zero(4), overrides=dropped(across), fallback="nominal")
    assert list(nominal.fovs.status) == ["image_supported", "image_supported", "metadata_only", "metadata_only"]
    assert list(nominal.fovs["flags"]) == ["", "", "fallback", "fallback"]


def test_an_absent_image_is_missing(stub):
    table = stub_table()
    result = run(table, grids_for(table), FRACTIONAL, absent=("FOV4",))
    fovs = result.fovs.set_index("fov_id")
    assert fovs.loc["FOV4", "status"] == "missing" and fovs.loc["FOV4", "component"] == -1
    assert fovs.loc["FOV4", ORIGIN].isna().all() and fovs.loc["FOV4", CORRECTION].isna().all()
    assert result.record["review"]["state"] == "required"
    assert result.record["review"]["flagged"]["unplaced_fovs"] == ["FOV4"]
    check_record(result, table)


@pytest.mark.parametrize("axis,fovs,bound", [("x", ("FOV0", "FOV5"), 0.75), ("y", ("FOV1", "FOV4"), 0.5)])
def test_correction_bound_boundary(stub, axis, fovs, bound):
    table = stub_table()
    grids = grids_for(table)
    k = {"y": 1, "x": 2}[axis]
    for value, flagged in ((bound, False), (bound + TINY, True)):
        translations = np.zeros((6, 3))
        translations[int(fovs[0][-1]), k], translations[int(fovs[1][-1]), k] = value, -value
        result = run(table, grids, translations)
        frame = result.fovs.set_index("fov_id")
        assert frame.loc[fovs[0], f"correction_{axis}"] == value
        assert frame.loc[fovs[1], f"correction_{axis}"] == -value
        marked = [f for f, text in zip(frame.index, frame["flags"]) if "correction_above_bound" in text]
        assert marked == (list(fovs) if flagged else [])
        review = result.record["review"]
        assert review["state"] == ("required" if flagged else "not_needed")
        assert review["reasons"] == (["correction_above_bound"] if flagged else [])
        assert set(frame.status) == {"image_supported"}


def test_residual_boundary_and_optimizer_removed_links(stub):
    table = stub_table()
    grids = grids_for(table)
    for value, flagged in ((1.0, False), (1.0 + TINY, True)):
        result = run(table, grids, zero(6), overrides=(("FOV0", "FOV1", 0.9, True, "", (0.0, 0.0, value)),))
        link = result.links[(result.links.fov_a == "FOV0") & (result.links.fov_b == "FOV1")].iloc[0]
        assert link.residual == value
        marked = [f for f, text in zip(result.fovs.fov_id, result.fovs["flags"]) if "residual_above_tolerance" in text]
        assert marked == (["FOV0", "FOV1"] if flagged else [])
        assert result.record["review"]["reasons"] == (["residual_above_tolerance"] if flagged else [])
    removed = run(table, grids, zero(6), overrides=(("FOV0", "FOV1", 0.9, False, "optimizer_removed",
                                                     (0.0, 0.0, 0.0)),))
    marked = [f for f, text in zip(removed.fovs.fov_id, removed.fovs["flags"]) if text == "optimizer_removed_link"]
    assert marked == ["FOV0", "FOV1"] and set(removed.fovs.status) == {"image_supported"}
    assert removed.record["review"]["state"] == "required"
    assert removed.record["review"]["reasons"] == ["optimizer_removed_link"]
    assert np.isnan(removed.links.residual[0])
    pruned = run(table, grids, zero(6), overrides=(("FOV0", "FOV1", 0.9, None, "", (0.0, 0.0, 0.0)),))
    first = pruned.links.iloc[0]
    assert (first.fov_a, first.fov_b, first.kept, first.reason) == ("FOV0", "FOV1", False, "pruned")
    assert np.isnan(first.quality) and np.isnan(first.shift_x)
    assert pruned.record["review"]["state"] == "not_needed"


def test_approval_records_the_tables_and_changes_no_origin(stub, tmp_path):
    table = moved_table()
    result = run(table, grids_for(table), FRACTIONAL, fallback="nominal")
    start = datetime.now(timezone.utc)
    approved = approve_placement(result, reviewer="jiahao", note="FOV5 has no overlap")
    (approval,) = approved.record["review"]["approvals"]
    assert approval["reviewer"] == "jiahao" and approval["note"] == "FOV5 has no overlap"
    when = datetime.strptime(approval["time_utc"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    assert start <= when <= datetime.now(timezone.utc)
    assert approval["fovs_sha256"] == hash_oracles.placement_table_sha256(result.fovs)
    assert approval["links_sha256"] == hash_oracles.placement_table_sha256(result.links)
    assert re.fullmatch(r"[0-9a-f]{64}", approval["fovs_sha256"])
    pd.testing.assert_frame_equal(approved.fovs, result.fovs, check_exact=True)
    pd.testing.assert_frame_equal(approved.links, result.links, check_exact=True)
    assert approved.record["review"]["state"] == "required"
    # An approval of other tables does not count.
    other = approve_placement(run(table, grids_for(table), zero(6), fallback="nominal"), reviewer="x", note="")
    stale = dataclasses.replace(result, record=dict(result.record, review=other.record["review"]))
    with pytest.raises(PlacementReviewRequiredError):
        require_approval(stale)
    with pytest.raises(ValueError, match="reviewer"):
        approve_placement(result, reviewer=" ", note="")


def test_review_placement_is_idempotent_and_keeps_matching_approvals(stub):
    table = moved_table()
    result = approve_placement(run(table, grids_for(table), FRACTIONAL, fallback="nominal"), reviewer="a", note="")
    again = review_placement(result)
    pd.testing.assert_frame_equal(again.fovs, result.fovs, check_exact=True)
    assert again.record == result.record
    strict = review_placement(result, policy=ReviewPolicy(max_correction_fraction=0.0))
    assert "correction_above_bound" in strict.record["review"]["reasons"]
    assert strict.record["review"]["approvals"] == []


def test_placement_result_rules():
    table = stub_table()
    result = placement_from_table(table, grids_for(table), sample="s1")
    with pytest.raises(ValueError, match="origins must be finite exactly"):
        PlacementResult("s1", result.fovs.assign(status="failed"), result.links, result.record)
    with pytest.raises(ValueError, match="status must be one of"):
        PlacementResult("s1", result.fovs.assign(status="placed"), result.links, result.record)
    with pytest.raises(ValueError, match="flags must be among"):
        PlacementResult("s1", result.fovs.assign(flags="guess"), result.links, result.record)
    with pytest.raises(ValueError, match="columns"):
        PlacementResult("s1", result.fovs.drop(columns="flags"), result.links, result.record)
    with pytest.raises(ValueError, match="no origin"):
        placed_metadata(dataclasses.replace(result, fovs=result.fovs.assign(
            status="failed", origin_z=np.nan, origin_y=np.nan, origin_x=np.nan)), "FOV0")


@pytest.mark.parametrize("fallback", [None, "nominal"], ids=["fallback_none", "fallback_nominal"])
def test_a_present_fov_with_a_non_finite_translation_is_failed_and_keeps_a_component(stub, fallback):
    """Every image present, FOV3's translation NaN and its links dropped as nan_quality or reported kept.

    design/specs/stitching-algorithms.md ("Failure behavior"): "A non-finite translation: `failed`", whatever
    the fallback; the component rule of W-365 holds (its own component, −1 only for ``missing``).
    """
    table = stub_table()
    grids = grids_for(table)
    translations = np.array(FRACTIONAL)
    translations[3] = np.nan
    nominal = table.nominal_zyx()
    supported = [k for k in range(6) if k != 3]
    c = translations[supported]
    # The other FOVs as they are without FOV3's translation: FOV3 disconnected with a finite one.
    without = run(table, grids, FRACTIONAL, overrides=nan_links("FOV3", STUB_PAIRS)).fovs.set_index("fov_id")
    for overrides in (nan_links("FOV3", STUB_PAIRS), ()):     # links dropped, or reported kept by the method
        result = run(table, grids, translations, overrides=overrides, fallback=fallback)
        fovs = result.fovs.set_index("fov_id")
        reference = check_components(result)
        assert fovs.loc["FOV3", "status"] == "failed"
        assert "fallback" not in fovs.loc["FOV3", "flags"].split(";")
        if overrides:
            assert fovs.loc["FOV3", "flags"] == "nan_quality_link" and fovs.loc["FOV3", "n_links_kept"] == 0
        else:
            assert fovs.loc["FOV3", "flags"] == "" and fovs.loc["FOV3", "n_links_kept"] == 5
        assert fovs.loc["FOV3", ORIGIN].isna().all() and fovs.loc["FOV3", CORRECTION].isna().all()
        assert fovs.loc["FOV3", TRANSLATION].isna().all()
        assert reference == 0 and fovs.loc["FOV3", "component"] == 1
        others = fovs.drop(index="FOV3")
        assert set(others.status) == {"image_supported"} and set(others.component) == {0}
        assert bits(others, ORIGIN) == (nominal[supported] + c - c.mean(axis=0)).tobytes()
        assert bits(others, ORIGIN) == bits(without.drop(index="FOV3"), ORIGIN)
        assert list(others.status) == list(without.drop(index="FOV3").status)
        assert result.record["fallback"] == fallback
        assert result.record["review"]["state"] == "required"
        assert result.record["review"]["reasons"] == ["unplaced_fovs"]
        assert result.record["review"]["flagged"] == {"unplaced_fovs": ["FOV3"]}
        check_record(result, table)


@pytest.mark.parametrize("fallback", [None, "nominal"], ids=["fallback_none", "fallback_nominal"])
def test_a_method_without_a_finite_translation_fails_every_fov(stub, tmp_path, fallback):
    """Every translation NaN: every present FOV is ``failed`` under both fallbacks; saved and reloaded."""
    table = stub_table()
    grids = grids_for(table)
    translations = np.full((6, 3), np.nan)
    every_link = tuple((a, b, np.nan, False, "nan_quality", (0.0, 0.0, 0.0)) for a, b in STUB_PAIRS)
    for name, overrides in (("kept", ()), ("nan_quality", every_link)):
        result = run(table, grids, translations, overrides=overrides, fallback=fallback)
        fovs = result.fovs
        assert set(fovs.status) == {"failed"} and not set(fovs.status) & {"image_supported", "metadata_only"}
        assert not any("fallback" in text.split(";") for text in fovs["flags"])
        assert fovs[ORIGIN].isna().all().all() and fovs[TRANSLATION].isna().all().all()
        assert check_components(result) is None and list(fovs.component) == list(range(6))
        assert result.record["gauge_mean_correction_zyx"] is None and result.record["fallback"] == fallback
        assert result.record["review"]["state"] == "required"
        assert result.record["review"]["reasons"] == ["unplaced_fovs"]
        assert result.record["review"]["flagged"] == {"unplaced_fovs": table.fov_ids}
        check_record(result, table)
        loaded = load_placement(save_placement(result, tmp_path / name))
        pd.testing.assert_frame_equal(loaded.fovs, result.fovs, check_exact=True)
        pd.testing.assert_frame_equal(loaded.links, result.links, check_exact=True)
        assert loaded.record == result.record
        check_components(loaded)


@pytest.mark.parametrize("fallback", [None, "nominal"], ids=["fallback_none", "fallback_nominal"])
def test_a_disconnected_fov_with_a_finite_translation_keeps_the_fallback_rule(stub, fallback):
    """PL9 to PL11: the moved FOV and the second component, finite translations, under each fallback."""
    moved = moved_table()
    small = two_by_two()
    across = [("FOV0", "FOV2"), ("FOV0", "FOV3"), ("FOV1", "FOV2"), ("FOV1", "FOV3")]
    cases = ((moved, run(moved, grids_for(moved), FRACTIONAL, fallback=fallback), ["FOV5"]),
             (small, run(small, grids_for(small), zero(4), overrides=dropped(across), fallback=fallback),
              ["FOV2", "FOV3"]))
    for table, result, outside in cases:
        fovs = result.fovs.set_index("fov_id")
        nominal = pd.DataFrame(table.nominal_zyx(), index=table.fov_ids, columns=ORIGIN)
        assert set(fovs.drop(index=outside).status) == {"image_supported"}
        assert np.isfinite(fovs.loc[outside, TRANSLATION].to_numpy()).all()
        if fallback is None:
            assert set(fovs.loc[outside, "status"]) == {"failed"} and set(fovs.loc[outside, "flags"]) == {""}
            assert fovs.loc[outside, ORIGIN].isna().all().all()
        else:
            assert set(fovs.loc[outside, "status"]) == {"metadata_only"}
            assert set(fovs.loc[outside, "flags"]) == {"fallback"}
            assert bits(fovs.loc[outside], ORIGIN) == bits(nominal.loc[outside], ORIGIN)
            assert (fovs.loc[outside, CORRECTION] == 0.0).all().all()
        assert result.record["review"]["state"] == "required"
        assert result.record["review"]["flagged"]["unplaced_fovs"] == outside
        check_record(result, table)


def test_the_reference_component_holds_placed_fovs_only(stub):
    """FOV0 first in table order with a NaN translation; the others form two pairs of equal size."""
    table = two_by_two()
    grids = grids_for(table)
    translations = np.zeros((4, 3))
    translations[0] = np.nan
    across = dropped([("FOV1", "FOV2"), ("FOV1", "FOV3"), ("FOV2", "FOV3")])
    result = run(table, grids, translations, overrides=across)
    assert list(result.fovs.component) == [0, 1, 2, 3]
    assert check_components(result) is None and set(result.fovs.status) == {"failed"}
    joined = run(table, grids, translations, overrides=dropped([("FOV1", "FOV3"), ("FOV2", "FOV3")]))
    assert list(joined.fovs.component) == [0, 1, 1, 2]
    assert check_components(joined) == 1
    assert list(joined.fovs.status) == ["failed", "image_supported", "image_supported", "failed"]


def test_present_fovs_joined_by_no_kept_link_have_their_own_component(stub):
    moved = moved_table()
    for fallback in (None, "nominal"):
        result = run(moved, grids_for(moved), FRACTIONAL, fallback=fallback)
        reference = check_components(result)
        assert reference == 0 and result.fovs.component[5] == 1 and result.fovs.status[5] != "missing"
    table = two_by_two()
    across = [("FOV0", "FOV2"), ("FOV0", "FOV3"), ("FOV1", "FOV2"), ("FOV1", "FOV3")]
    result = run(table, grids_for(table), zero(4), overrides=dropped(across))
    reference = check_components(result)
    assert reference == 0 and list(result.fovs.component[2:]) == [1, 1]


def test_every_stub_variant_follows_the_component_rule(stub):
    results = stub_variant_results()
    assert len(results) == 8
    for name, result in results.items():
        check_components(result)
    assert results["missing"].fovs.component[4] == -1
