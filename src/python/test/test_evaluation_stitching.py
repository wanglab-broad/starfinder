"""M62/M68 common-frame placement and overlap populations."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from starfinder.image import ImageMetadata, IncompatibleGeometryError

pytestmark = [pytest.mark.evaluation, pytest.mark.stitching, pytest.mark.contract, pytest.mark.validation]
META = ImageMetadata("common", spacing_zyx=(1., 1., 1.), origin_zyx=(0., 0., 0.),
                     direction_zyx=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), spatial_unit="micrometer")
GEOMETRY = dict(reference_metadata=META, observed_metadata=META, units="micrometer")


def origins():
    return pd.DataFrame(dict(fov_id=["A", "B"], origin_z=[0., 0.], origin_y=[0., 0.], origin_x=[0., 10.]))


def molecules(x, gene=None):
    return pd.DataFrame(dict(z=np.zeros(len(x)), y=np.zeros(len(x)), x=x,
                             gene_id=gene if gene is not None else ["g"] * len(x)))


def test_gauge_aligned_origin_error():
    from starfinder.evaluation.stitching import gauge_aligned_origin_error

    ref = origins()
    obs = ref.assign(origin_z=5., origin_y=2., origin_x=[1., 13.])
    result = gauge_aligned_origin_error(ref, obs, anchor="A", **GEOMETRY)
    assert result.values == pytest.approx(dict(median_error=1., max_error=2.), rel=0, abs=1e-9)
    assert [r["error"] for r in result.details["origins"]] == pytest.approx([0., 2.], rel=0, abs=1e-9)
    assert result.config["anchor"] == "A"
    assert result.config["gauge_offset_zyx"] == pytest.approx([5., 2., 1.], rel=0, abs=1e-9)
    missing = gauge_aligned_origin_error(ref, obs.iloc[:1], anchor="A", **GEOMETRY)
    assert missing.status == "missing"
    assert missing.details["missing_placements"] == ["B"]
    assert missing.counts["evaluated"] == 1 and missing.counts["total_reference"] == 2
    nan = gauge_aligned_origin_error(ref, obs.assign(origin_x=[1., np.nan]), anchor="A", **GEOMETRY)
    assert nan.details["missing_placements"] == ["B"]
    with pytest.raises(ValueError, match="anchor"):
        gauge_aligned_origin_error(ref, obs.iloc[1:], anchor="A", **GEOMETRY)
    with pytest.raises(IncompatibleGeometryError):
        gauge_aligned_origin_error(ref, obs, anchor="A", **(GEOMETRY | dict(observed_metadata=ImageMetadata("common"))))
    with pytest.raises(ValueError, match="unique"):
        gauge_aligned_origin_error(ref, pd.concat([obs, obs]), anchor="A", **GEOMETRY)


def test_overlap_molecules_common_frame():
    from starfinder.evaluation.stitching import overlap_molecules

    a, b = molecules([0., 10.]), molecules([.2, 10.4])
    config = GEOMETRY | dict(bounds_zyx=((-1., -1., 0.), (1., 1., 11.)), threshold=.5)
    result = overlap_molecules(a, b, **config)
    assert result.values == pytest.approx(dict(reference_match_rate=1., observed_match_rate=1.,
                                              median_distance=.3, max_distance=.4), rel=0, abs=1e-9)
    assert [p[2] for p in result.details["matched_pairs"]] == pytest.approx([.2, .4], rel=0, abs=1e-9)
    duplicate = overlap_molecules(a, molecules([.2, 10.4, .3]), **config)
    assert duplicate.counts["matched"] == 2
    assert duplicate.values["observed_match_rate"] == pytest.approx(2 / 3, rel=1e-9, abs=1e-12)
    assert duplicate.values["reference_match_rate"] == pytest.approx(1., rel=1e-9, abs=1e-12)
    mismatch = overlap_molecules(a, molecules([.2, 10.4], ["wrong", None]), **config)
    assert mismatch.counts["matched"] == 0
    assert mismatch.values["median_distance"] is None and mismatch.reasons["median_distance"]
    assert mismatch.values["observed_match_rate"] == pytest.approx(0., rel=1e-9, abs=1e-12)
    # Lower bounds enter, upper bounds do not; threshold equality enters.
    boundary = overlap_molecules(molecules([0., 11.]), molecules([.5, 11.]), **config)
    assert boundary.counts["eligible_reference"] == boundary.counts["eligible_observed"] == 1
    assert boundary.counts["matched"] == 1
    assert boundary.details["matched_pairs"][0][:2] == (0, 0)
    assert boundary.values["max_distance"] == pytest.approx(.5, rel=0, abs=1e-9)
    with pytest.raises(IncompatibleGeometryError):
        overlap_molecules(a, b, **(config | dict(observed_metadata=replace(META, frame_id="other"))))
    with pytest.raises(IncompatibleGeometryError):
        overlap_molecules(a, b, **(config | dict(units="voxel")))
    none = overlap_molecules(a, b, **(config | dict(bounds_zyx=((0., 0., 0.), (0., 1., 11.)))))
    assert all(v is None for v in none.values.values())
    assert none.counts["eligible_reference"] == none.counts["eligible_observed"] == 0
    assert none.reasons["reference_match_rate"]
    excluded = overlap_molecules(a, b, **(config | dict(bounds_zyx=((20., 20., 20.), (21., 21., 21.)))))
    assert all(v is None for v in excluded.values.values())


@pytest.mark.parametrize("change", [dict(threshold=-1), dict(bounds_zyx=((1., 0., 0.), (0., 1., 1.))),
                                    dict(bounds_zyx=((0., 0.), (1., 1.)))])
def test_overlap_rejects_invalid_population(change):
    from starfinder.evaluation.stitching import overlap_molecules

    with pytest.raises(ValueError):
        overlap_molecules(molecules([0.]), molecules([0.]),
                          **(GEOMETRY | dict(bounds_zyx=((-1., -1., -1.), (1., 1., 1.)), threshold=.5) | change))
