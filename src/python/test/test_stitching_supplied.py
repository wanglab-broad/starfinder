"""PL18 of docs/stitching-assembly-validation.md: ``placement_from_table``, without and with ``origins``.

Fixtures: ``stub_grid``'s table and grids; ``own_cases``' table (two FOVs of 4×16×16 at
1 µm, nominal x 0 and 12 µm, one layer) with the origins (0.3, 0.25, 1.5) and (−0.7,
−0.125, 15.0) µm; a one-FOV table; Z = 1 grids; one grid of 4×16×20 beside 4×16×24;
``stub_grid``'s table with origins equal to its nominal positions; ``scene_tiny``'s
table from ``positions.csv`` with the ``true_origin_*`` of ``placement_truth.csv``
(seed 42); a table whose ``nominal_z`` differ; and the failure cases. The clauses on
``assemble`` belong to the assembly issue.
"""
import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest

from starfinder.image import IncompatibleGeometryError
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import (PlacementResult, approve_placement, load_placement,
                                  placement_from_table, positions_from_table, review_placement, save_placement)
from starfinder.stitching._review import require_approval

from .stitching_fixtures import STUB_SPACING, grid, grids_for, metadata, stub_table

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

ORIGIN = ["origin_z", "origin_y", "origin_x"]
NOMINAL = ["nominal_z", "nominal_y", "nominal_x"]
CORRECTION = ["correction_z", "correction_y", "correction_x"]
TRANSLATION = ["method_translation_z", "method_translation_y", "method_translation_x"]
LINK_COLUMNS = ["fov_a", "fov_b", "overlap_z0", "overlap_z1", "overlap_y0", "overlap_y1", "overlap_x0",
                "overlap_x1", "shift_z", "shift_y", "shift_x", "quality", "kept", "reason", "residual"]
OWN_ORIGINS = {"FOV1": (0.3, 0.25, 1.5), "FOV2": (-0.7, -0.125, 15.0)}


def bits(frame, columns):
    return frame[columns].to_numpy(dtype=np.float64).tobytes()


def own_cases():
    frame = pd.DataFrame({"fov_id": ["FOV1", "FOV2"], "nominal_z": [0.0, 0.0], "nominal_y": [0.0, 0.0],
                          "nominal_x": [0.0, 12.0]})
    table = positions_from_table(frame)
    return table, {fov: grid((4, 16, 16), (1.0, 1.0, 1.0)) for fov in table.fov_ids}


def one_fov():
    table = positions_from_table(pd.DataFrame({"fov_id": ["solo"], "nominal_z": [0.5], "nominal_y": [1.25],
                                               "nominal_x": [-2.0]}))
    return table, {"solo": grid()}


def z1():
    table = stub_table()
    return table, {fov: grid((1, 16, 24)) for fov in table.fov_ids}


def unequal():
    table = stub_table()
    return table, grids_for(table, FOV2=((4, 16, 20), STUB_SPACING))


def stub():
    table = stub_table()
    return table, grids_for(table)


CASES = {"stub_grid": stub, "own_cases": own_cases, "one_fov": one_fov, "z1": z1, "unequal_yx": unequal}


def check_common(result, table, grids, sample="s1"):
    assert isinstance(result, PlacementResult) and result.sample == sample
    fovs = result.fovs
    assert list(fovs.fov_id) == table.fov_ids
    assert set(fovs.status) == {"metadata_only"} and set(fovs["flags"]) == {"supplied"}
    assert bits(fovs, NOMINAL) == table.nominal_zyx().tobytes()
    assert fovs[TRANSLATION].isna().all().all()
    assert list(fovs.n_links_kept) == [0] * len(fovs)
    assert list(fovs.component) == list(range(len(fovs)))
    assert list(result.links.columns) == LINK_COLUMNS and len(result.links) == 0
    assert result.record["method"] == {"stage": "stitching", "method": "supplied", "config_type": "",
                                       "implementation": "starfinder.stitching.placement_from_table",
                                       "config": {}, "requires": {}, "artifacts": []}
    assert result.record["gauge"] == "none"
    assert result.record["frame"]["frame_id"] == f"sample:{sample}"
    assert result.record["table"] == table.record
    for fov in table.fov_ids:
        entry = result.record["inputs"][fov]
        assert entry["source"] == "supplied" and entry["file_sha256"] is None
        assert entry["grid"]["shape_zyx"] == list(grids[fov].shape_zyx)
        assert entry["grid"]["metadata"] == json.loads(json.dumps(dataclasses.asdict(grids[fov].metadata)))
        assert entry["grid"]["sha256"] == grids[fov].sha256
    review = result.record["review"]
    assert review["state"] == "required" and review["reasons"] == ["supplied_placement"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_without_origins_each_fov_sits_at_its_nominal_position(case):
    table, grids = CASES[case]()
    result = placement_from_table(table, grids, sample="s1")
    check_common(result, table, grids)
    assert bits(result.fovs, ORIGIN) == table.nominal_zyx().tobytes()
    assert (result.fovs[CORRECTION] == 0.0).all().all()
    assert result.record["origins"] == "nominal"
    if case == "one_fov":
        assert len(result.fovs) == 1 and result.fovs.component[0] == 0


def test_with_origins_the_supplied_values_are_kept_bitwise(tmp_path):
    table, grids = own_cases()
    result = placement_from_table(table, grids, sample="s1", origins=OWN_ORIGINS)
    check_common(result, table, grids)
    supplied = np.array([OWN_ORIGINS[f] for f in table.fov_ids])
    assert bits(result.fovs, ORIGIN) == supplied.tobytes()
    assert bits(result.fovs, CORRECTION) == (supplied - table.nominal_zyx()).tobytes()
    assert list(result.fovs.correction_x) == [1.5, 3.0]        # above 1.0 µm and above 0.5 × the 4 µm overlap
    assert result.record["origins"] == "supplied"
    reviewed = review_placement(result)
    assert reviewed.record["review"]["state"] == "required"
    assert reviewed.record["review"]["reasons"] == ["supplied_placement"]
    assert set(reviewed.fovs["flags"]) == {"supplied"} and set(reviewed.fovs.status) == {"metadata_only"}
    pd.testing.assert_frame_equal(reviewed.fovs, result.fovs, check_exact=True)
    approved = approve_placement(reviewed, reviewer="jiahao", note="truth")
    assert bits(approved.fovs, ORIGIN + CORRECTION) == bits(result.fovs, ORIGIN + CORRECTION)
    folder = save_placement(approved, tmp_path / "stitching" / "s1")
    loaded = load_placement(folder)
    pd.testing.assert_frame_equal(loaded.fovs, approved.fovs, check_exact=True)
    pd.testing.assert_frame_equal(loaded.links, approved.links, check_exact=True)
    assert loaded.record == approved.record and loaded.sample == "s1"
    header = (folder / "fovs.csv").read_text().splitlines()[0].split(",")
    assert set(ORIGIN + NOMINAL + CORRECTION) <= set(header)
    assert (folder / "links.csv").read_text() == ",".join(LINK_COLUMNS) + "\n"
    assert json.loads((folder / "placement.json").read_text())["origins"] == "supplied"


def test_origins_equal_to_the_nominal_positions_give_the_same_table(tmp_path):
    table, grids = stub()
    plain = placement_from_table(table, grids, sample="s1")
    origins = {fov: tuple(row) for fov, row in zip(table.fov_ids, table.nominal_zyx())}
    given = placement_from_table(table, grids, sample="s1", origins=origins)
    pd.testing.assert_frame_equal(given.fovs, plain.fovs, check_exact=True)
    a = save_placement(plain, tmp_path / "plain")
    b = save_placement(given, tmp_path / "given")
    assert (a / "fovs.csv").read_bytes() == (b / "fovs.csv").read_bytes()
    record_a = json.loads((a / "placement.json").read_text())
    record_b = json.loads((b / "placement.json").read_text())
    assert {k for k in record_a if record_a[k] != record_b[k]} == {"origins"}


def test_scene_truth_placement(tmp_path):
    from starfinder.synthetic import generate_sample_scene, sample_scene_preset, save_sample_scene
    book, config = sample_scene_preset("sample-tiny-v1")
    scene = generate_sample_scene(book, config=config)
    save_sample_scene(scene, tmp_path)
    table = positions_from_table(tmp_path / "positions.csv")
    truth = pd.read_csv(tmp_path / "placement_truth.csv", float_precision="round_trip").set_index("fov_id")
    origins = {fov: tuple(truth.loc[fov, ["true_origin_z", "true_origin_y", "true_origin_x"]]) for fov in truth.index}
    grids = {fov: ReferenceGrid(config.acquisition.fov_shape_zyx, scene.fov_metadata[fov], "declared")
             for fov in table.fov_ids}
    result = approve_placement(placement_from_table(table, grids, sample="sample-tiny-v1", origins=origins),
                               reviewer="test", note="truth placement")
    require_approval(result)
    check_common(result, table, grids, sample="sample-tiny-v1")
    true_origin = truth.loc[table.fov_ids, ["true_origin_z", "true_origin_y", "true_origin_x"]].to_numpy()
    true_correction = truth.loc[table.fov_ids, ["true_correction_z", "true_correction_y",
                                                "true_correction_x"]].to_numpy()
    assert bits(result.fovs, ORIGIN) == true_origin.tobytes()
    assert np.abs(result.fovs[CORRECTION].to_numpy() - true_correction).max() <= 1e-9
    assert (result.fovs.correction_z != 0.0).any()        # Z jitter included
    assert set(result.fovs.nominal_z) == {0.0}


def test_a_table_whose_nominal_z_differ_is_still_a_placement():
    frame = pd.DataFrame({"fov_id": ["a", "b"], "nominal_z": [0.0, 1.0], "nominal_y": [0.0, 0.0],
                          "nominal_x": [0.0, 4.0]})
    table = positions_from_table(frame)
    grids = {"a": grid(), "b": grid()}
    plain = placement_from_table(table, grids, sample="s1")
    given = placement_from_table(table, grids, sample="s1", origins={"a": (0.5, 0.0, 0.0), "b": (0.5, 0.0, 4.0)})
    assert list(plain.fovs.nominal_z) == list(given.fovs.nominal_z) == [0.0, 1.0]
    assert list(given.fovs.origin_z) == [0.5, 0.5] and list(given.fovs.correction_z) == [0.5, -0.5]


def test_failures_name_the_fovs():
    table, grids = own_cases()
    with pytest.raises(ValueError, match="no row"):
        placement_from_table(dataclasses.replace(table, positions=table.positions.iloc[:0]), {}, sample="s1")
    with pytest.raises(ValueError, match="FOVs of the table without a grid: FOV2"):
        placement_from_table(table, {"FOV1": grids["FOV1"]}, sample="s1")
    with pytest.raises(ValueError, match="a grid given for FOVs the table lacks: FOV9"):
        placement_from_table(table, dict(grids, FOV9=grids["FOV1"]), sample="s1")
    bad_grids = {
        "unknown spacing": ReferenceGrid((4, 16, 16), dataclasses.replace(metadata(), spacing_zyx=None), "declared"),
        "a unit other than µm": grid((4, 16, 16), (1.0, 1.0, 1.0), unit="nm"),
        "a direction other than the identity": grid((4, 16, 16), (1.0, 1.0, 1.0),
                                                   direction=((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))),
        "one Z size": grid((5, 16, 16), (1.0, 1.0, 1.0)),
    }
    for what, bad in bad_grids.items():
        with pytest.raises(IncompatibleGeometryError, match=what):
            placement_from_table(table, dict(grids, FOV2=bad), sample="s1")
    faulty = {
        "without an origin: FOV2": {"FOV1": (0.0, 0.0, 0.0)},
        "given for FOVs the table lacks: FOV3": dict(OWN_ORIGINS, FOV3=(0.0, 0.0, 0.0)),
        "three finite numbers .*; FOVs FOV1 are not": dict(OWN_ORIGINS, FOV1=(math.nan, 0.0, 0.0)),
        "three finite numbers .*; FOVs FOV2 are not": dict(OWN_ORIGINS, FOV2=(0.0, math.inf, 12.0)),
        "three finite numbers .*; FOVs FOV1, FOV2 are not": {"FOV1": (0.0, 1.0), "FOV2": (0.0, 0.0, 12.0, 1.0)},
        "without an origin: FOV1, FOV2": {},
    }
    for match, origins in faulty.items():
        with pytest.raises(ValueError, match=match):
            placement_from_table(table, grids, sample="s1", origins=origins)
    accepted = placement_from_table(*unequal(), sample="s1")
    assert accepted.record["inputs"]["FOV2"]["grid"]["shape_zyx"] == [4, 16, 20]
