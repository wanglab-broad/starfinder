"""The memory of registration_qc (W-461): the values it had before it read the signals plane by plane, and its peak.

The pinned values are registration_qc's results at the starting revision of W-461 (454b3de), saved by that issue's
scripts/qc_values.py on the same fixtures; floats compare to an absolute 1e-9, everything else exactly.
"""
import math
import tracemalloc

import numpy as np
import pytest
from scipy import ndimage

from starfinder.evaluation.registration import registration_qc
from starfinder.image import ImageMetadata
from starfinder.registration import (
    AffineTransform,
    BSplineTransform,
    DenseDisplacementTransform,
    RegistrationQcConfig,
    TransformChain,
    TranslationTransform,
    WarpConfig,
)
from starfinder.registration._chain import resample

pytestmark = [pytest.mark.registration, pytest.mark.evaluation]

SHAPE = (8, 48, 64)
SPACING = (0.346255, 0.194398, 0.194398)
UNITS = {"coverage": "fraction", "ncc_before": "dimensionless", "ncc_after": "dimensionless",
         "ncc_gain": "dimensionless", "ssim_before": "dimensionless", "ssim_after": "dimensionless"}


def geometry(shape=SHAPE):
    return dict(reference_shape_zyx=shape, moving_shape_zyx=shape,
                reference_metadata=ImageMetadata("reference", spacing_zyx=SPACING),
                moving_metadata=ImageMetadata("moving", spacing_zyx=SPACING))


def center():
    return (np.asarray(SHAPE, dtype=np.float64) - 1) / 2


def about_center(a, shift=(0.0, 0.0, 0.0)):
    """The pull map p -> A (p - c) + c + shift as an AffineTransform."""
    a = np.asarray(a, dtype=np.float64)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = a, center() - a @ center() + np.asarray(shift, dtype=np.float64)
    return AffineTransform(m, **geometry())


def rotation_yx(degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def bump():
    """A smooth ZYX3 field that is zero on the grid's border planes (polynomials only: no SIMD sin or cos)."""
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float64) / (n - 1) for n in SHAPE), indexing="ij")
    s = 64 * z * (1 - z) * y * (1 - y) * x * (1 - x)
    return np.stack([0.3 * s, 0.8 * s * (1 - 2 * x), -0.6 * s * (1 - 2 * y)], axis=-1)


def inward():
    """A field that pulls every voxel towards the centre (0.05 of its offset in Z, 0.08 in Y and X)."""
    p = np.moveaxis(np.indices(SHAPE, dtype=np.float64), 0, -1)
    return (center() - p) * np.array([0.05, 0.08, 0.08])


def folded(field):
    """field with two voxels whose X displacement folds the map (det(I + grad u) <= 0 next to them)."""
    field = np.array(field, dtype=np.float64)
    field[4, 20, 30, 2] -= 3.0
    field[2, 30, 40, 1] += 3.0
    return field


def dense(field):
    return DenseDisplacementTransform(field, **geometry())


def translation(d):
    return TranslationTransform(tuple(float(v) for v in d), **geometry())


def bspline(uniform):
    """A cubic B-spline with one control point at the centre (and, with uniform, every coefficient raised)."""
    spacing_xyz = (2.0, 2.0, 0.6)
    center_xyz = (center() * np.asarray(SPACING))[::-1]
    origin = tuple(float(c - 4 * s) for c, s in zip(center_xyz, spacing_xyz))
    coefficients = np.zeros((3, 9, 9, 9))
    coefficients[0, 4, 4, 4], coefficients[1, 4, 4, 4], coefficients[2, 4, 4, 4] = 0.25, -0.2, 0.05
    if uniform:
        coefficients[0] += 0.4
        coefficients[1] -= 0.3
    return BSplineTransform((9, 9, 9), origin, spacing_xyz, tuple(np.eye(3).ravel()), coefficients, SPACING,
                            **geometry())


def cases():
    """{name: transform}: seven transforms, each with every pull point inside the moving grid and with some outside."""
    contraction = np.diag([0.9, 0.9, 0.9])
    return {
        "translation_inside": translation((0, 0, 0)),
        "translation_outside": translation((0.5, 2.25, -3.5)),
        "affine_inside": about_center(np.diag([0.9, 0.92, 0.94]) + np.array([[0, 0, 0], [0, 0, 0.01], [0, 0.02, 0]])),
        "affine_outside": about_center(rotation_yx(6.0), (0.0, 1.5, -2.0)),
        "bspline_inside": bspline(False),
        "bspline_outside": bspline(True),
        "dense_inside": dense(inward() + bump()),
        "dense_outside": dense(folded(np.array([0.0, 1.7, -2.2]) + bump())),
        "chain_translation_inside": TransformChain((translation((0, 0, 0)),)),
        "chain_translation_outside": TransformChain((translation((0.0, 2.25, -3.5)),)),
        "chain_translation_affine_inside": TransformChain((translation((0.2, 1.0, -1.0)), about_center(contraction))),
        "chain_translation_affine_outside": TransformChain((translation((0.0, 3.0, -4.0)),
                                                            about_center(rotation_yx(5.0)))),
        "chain_translation_dense_inside": TransformChain((translation((0.1, 0.5, -0.5)), dense(inward() + bump()))),
        "chain_translation_dense_outside": TransformChain((translation((0.0, 2.0, -3.0)),
                                                           dense(folded(np.array([0.0, 0.5, 0.5]) + bump())))),
    }


def images():
    """(reference uint8, moving float32): a smoothed noise volume and its copy shifted by (0.4, 1.6, -2.3)."""
    rng = np.random.default_rng(461)
    smooth = ndimage.gaussian_filter(rng.random(SHAPE), (1.0, 2.0, 2.0))
    reference = np.round(255 * (smooth - smooth.min()) / (smooth.max() - smooth.min())).astype(np.uint8)
    moving = ndimage.shift(reference.astype(np.float64), (0.4, 1.6, -2.3), order=1) + 3 * rng.random(SHAPE)
    return reference, moving.astype(np.float32)


PINNED = {"translation_inside": {"status": "ok",
                                 "values": {"coverage": 1.0,
                                            "ncc_before": 0.18028392792725287,
                                            "ncc_after": 0.18028392792725287,
                                            "ncc_gain": 0.0,
                                            "ssim_before": 0.2826204561090706,
                                            "ssim_after": 0.2826204561090706},
                                 "counts": {"total": 24576, "valid": 24576, "valid_columns": 3072,
                                            "ssim_columns": 2436},
                                 "reasons": {},
                                 "data_range": 132.0,
                                 "transform": {"kind": "translation", "displacement_zyx": [0.0, 0.0, 0.0]}},
          "translation_outside": {"status": "undefined",
                                  "values": {"coverage": 0.76904296875,
                                             "ncc_before": 0.15842152161764586,
                                             "ncc_after": 0.5714469764827137,
                                             "ncc_gain": 0.4130254548650678,
                                             "ssim_before": None,
                                             "ssim_after": None},
                                  "counts": {"total": 24576, "valid": 18900, "valid_columns": 0, "ssim_columns": 0},
                                  "reasons": {"ssim_before": "the eroded valid columns are empty",
                                              "ssim_after": "the eroded valid columns are empty"},
                                  "data_range": None,
                                  "transform": {"kind": "translation", "displacement_zyx": [0.5, 2.25, -3.5]}},
          "affine_inside": {"status": "ok",
                            "values": {"coverage": 1.0,
                                       "ncc_before": 0.18028392792725287,
                                       "ncc_after": 0.2931215733283053,
                                       "ncc_gain": 0.11283764540105243,
                                       "ssim_before": 0.2826204561090706,
                                       "ssim_after": 0.31111473831467074},
                            "counts": {"total": 24576, "valid": 24576, "valid_columns": 3072, "ssim_columns": 2436},
                            "reasons": {},
                            "data_range": 132.0,
                            "transform": {"kind": "affine",
                                          "matrix": [[0.9, 0.0, 0.0], [0.0, 0.92, 0.01], [0.0, 0.02, 0.94]],
                                          "offset": [0.3500000000000001, 1.5649999999999977, 1.4200000000000017],
                                          "det": 0.77814,
                                          "reflection_or_collapse": False,
                                          "max_singular_value_a_minus_i": 0.09999999999999998,
                                          "rotation_angle": None}},
          "affine_outside": {"status": "ok",
                             "values": {"coverage": 0.9127604166666666,
                                        "ncc_before": 0.1856116582072656,
                                        "ncc_after": 0.23076965782044903,
                                        "ncc_gain": 0.045157999613183425,
                                        "ssim_before": 0.2765530097915778,
                                        "ssim_after": 0.5101928408236053},
                             "counts": {"total": 24576, "valid": 22432, "valid_columns": 2804, "ssim_columns": 2172},
                             "reasons": {},
                             "data_range": 132.0,
                             "transform": {"kind": "affine",
                                           "matrix": [[1.0, 0.0, 0.0],
                                                      [0.0, 0.9945218953682733, -0.10452846326765347],
                                                      [0.0, 0.10452846326765347, 0.9945218953682733]],
                                           "offset": [0.0, 4.921382051776664, -4.283858590890468],
                                           "det": 1.0,
                                           "reflection_or_collapse": False,
                                           "max_singular_value_a_minus_i": 0.10467191248588768,
                                           "rotation_angle": None}},
          "bspline_inside": {"status": "ok",
                             "values": {"coverage": 1.0,
                                        "ncc_before": 0.18028392792725287,
                                        "ncc_after": 0.17909621416566912,
                                        "ncc_gain": -0.0011877137615837474,
                                        "ssim_before": 0.2826204561090706,
                                        "ssim_after": 0.2804242686588933},
                             "counts": {"total": 24576, "valid": 24576, "valid_columns": 3072, "ssim_columns": 2436},
                             "reasons": {},
                             "data_range": 132.0,
                             "transform": {"kind": "bspline",
                                           "displacement_voxels": {"median": 0.0,
                                                                   "p95": 0.0804648789264615,
                                                                   "max": 0.4344765104943875},
                                           "displacement_physical": {"median": 0.0,
                                                                     "p95": 0.015771312968750655,
                                                                     "max": 0.08515845814967404},
                                           "fold_fraction": 0.0}},
          "bspline_outside": {"status": "ok",
                              "values": {"coverage": 0.9296875,
                                         "ncc_before": 0.222282718741711,
                                         "ncc_after": 0.030115772282364787,
                                         "ncc_gain": -0.1921669464593462,
                                         "ssim_before": 0.3153698187476243,
                                         "ssim_after": 0.008318356313327913},
                              "counts": {"total": 24576, "valid": 22848, "valid_columns": 2856, "ssim_columns": 2200},
                              "reasons": {},
                              "data_range": 132.0,
                              "transform": {"kind": "bspline",
                                            "displacement_voxels": {"median": 2.572042922252289,
                                                                    "p95": 2.6521716538708517,
                                                                    "max": 3.0049177724568428},
                                            "displacement_physical": {"median": 0.5000000000000004,
                                                                      "p95": 0.5155807981575563,
                                                                      "max": 0.584251203989414},
                                            "fold_fraction": 0.03125}},
          "dense_inside": {"status": "ok",
                           "values": {"coverage": 1.0,
                                      "ncc_before": 0.18028392792725287,
                                      "ncc_after": 0.25937535762146197,
                                      "ncc_gain": 0.0790914296942091,
                                      "ssim_before": 0.2826204561090706,
                                      "ssim_after": 0.299909813729555},
                           "counts": {"total": 24576, "valid": 24576, "valid_columns": 3072, "ssim_columns": 2436},
                           "reasons": {},
                           "data_range": 132.0,
                           "transform": {"kind": "dense",
                                         "displacement_voxels": {"median": 1.780400235902029,
                                                                 "p95": 2.7397003489035576,
                                                                 "max": 3.1488767838707186},
                                         "displacement_physical": {"median": 0.3486961858288169,
                                                                   "p95": 0.5334977477971051,
                                                                   "max": 0.6141856865199236},
                                         "fold_fraction": 0.0}},
          "dense_outside": {"status": "ok",
                            "values": {"coverage": 0.9134114583333334,
                                       "ncc_before": 0.1776178464123189,
                                       "ncc_after": 0.3709379271709184,
                                       "ncc_gain": 0.1933200807585995,
                                       "ssim_before": 0.277797880762436,
                                       "ssim_after": 0.9053658319020773},
                            "counts": {"total": 24576, "valid": 22448, "valid_columns": 2806, "ssim_columns": 2200},
                            "reasons": {},
                            "data_range": 132.0,
                            "transform": {"kind": "dense",
                                          "displacement_voxels": {"median": 2.7802877548915688,
                                                                  "p95": 2.9598115475250775,
                                                                  "max": 5.570707931267544},
                                          "displacement_physical": {"median": 0.5404823789754112,
                                                                    "p95": 0.5775380771396189,
                                                                    "max": 1.0860460166003922},
                                          "fold_fraction": 8.138020833333333e-05}},
          "chain_translation_inside": {"status": "ok",
                                       "values": {"coverage": 1.0,
                                                  "ncc_before": 0.18028392792725287,
                                                  "ncc_after": 0.18028392792725287,
                                                  "ncc_gain": 0.0,
                                                  "ssim_before": 0.2826204561090706,
                                                  "ssim_after": 0.2826204561090706},
                                       "counts": {"total": 24576,
                                                  "valid": 24576,
                                                  "valid_columns": 3072,
                                                  "ssim_columns": 2436},
                                       "reasons": {},
                                       "data_range": 132.0,
                                       "transform": {"kind": "dense",
                                                     "displacement_voxels": {"median": 0.0, "p95": 0.0, "max": 0.0},
                                                     "displacement_physical": {"median": 0.0, "p95": 0.0, "max": 0.0},
                                                     "fold_fraction": 0.0}},
          "chain_translation_outside": {"status": "ok",
                                        "values": {"coverage": 0.87890625,
                                                   "ncc_before": 0.17335650940419242,
                                                   "ncc_after": 0.33198861224892684,
                                                   "ncc_gain": 0.15863210284473442,
                                                   "ssim_before": 0.27446859229345555,
                                                   "ssim_after": 0.7073109102603472},
                                        "counts": {"total": 24576,
                                                   "valid": 21600,
                                                   "valid_columns": 2700,
                                                   "ssim_columns": 2106},
                                        "reasons": {},
                                        "data_range": 132.0,
                                        "transform": {"kind": "dense",
                                                      "displacement_voxels": {"median": 4.16082924427331,
                                                                              "p95": 4.16082924427331,
                                                                              "max": 4.16082924427331},
                                                      "displacement_physical": {"median": 0.8088568834282428,
                                                                                "p95": 0.8088568834282428,
                                                                                "max": 0.8088568834282428},
                                                      "fold_fraction": 0.0}},
          "chain_translation_affine_inside": {"status": "ok",
                                              "values": {"coverage": 1.0,
                                                         "ncc_before": 0.18028392792725287,
                                                         "ncc_after": 0.4894681236779645,
                                                         "ncc_gain": 0.30918419575071165,
                                                         "ssim_before": 0.2826204561090706,
                                                         "ssim_after": 0.42656927760545466},
                                              "counts": {"total": 24576,
                                                         "valid": 24576,
                                                         "valid_columns": 3072,
                                                         "ssim_columns": 2436},
                                              "reasons": {},
                                              "data_range": 132.0,
                                              "transform": {"kind": "dense",
                                                            "displacement_voxels": {"median": 2.4510201957552304,
                                                                                    "p95": 4.288064831599446,
                                                                                    "max": 5.361669516111561},
                                                            "displacement_physical": {"median": 0.48390587828374454,
                                                                                      "p95": 0.8391360608310565,
                                                                                      "max": 1.0541444897649663},
                                                            "fold_fraction": 0.0}},
          "chain_translation_affine_outside": {"status": "ok",
                                               "values": {"coverage": 0.865234375,
                                                          "ncc_before": 0.16206056396593194,
                                                          "ncc_after": 0.19203108454634304,
                                                          "ncc_gain": 0.02997052058041111,
                                                          "ssim_before": 0.270816977497998,
                                                          "ssim_after": 0.32583676566410147},
                                               "counts": {"total": 24576,
                                                          "valid": 21264,
                                                          "valid_columns": 2658,
                                                          "ssim_columns": 2046},
                                               "reasons": {},
                                               "data_range": 132.0,
                                               "transform": {"kind": "dense",
                                                             "displacement_voxels": {"median": 5.336516851678976,
                                                                                     "p95": 7.326611688859786,
                                                                                     "max": 8.318046892068413},
                                                             "displacement_physical": {"median": 1.0374082029326894,
                                                                                       "p95": 1.4242786590909644,
                                                                                       "max": 1.6170116797243153},
                                                             "fold_fraction": 0.0}},
          "chain_translation_dense_inside": {"status": "ok",
                                             "values": {"coverage": 1.0,
                                                        "ncc_before": 0.18028392792725287,
                                                        "ncc_after": 0.3604101045689095,
                                                        "ncc_gain": 0.18012617664165662,
                                                        "ssim_before": 0.2826204561090706,
                                                        "ssim_after": 0.40506206988495885},
                                             "counts": {"total": 24576,
                                                        "valid": 24576,
                                                        "valid_columns": 3072,
                                                        "ssim_columns": 2436},
                                             "reasons": {},
                                             "data_range": 132.0,
                                             "transform": {"kind": "dense",
                                                           "displacement_voxels": {"median": 1.850666937940424,
                                                                                   "p95": 3.0762436455062536,
                                                                                   "max": 3.8549221782028256},
                                                           "displacement_physical": {"median": 0.3652252110841906,
                                                                                     "p95": 0.5999731683691355,
                                                                                     "max": 0.7535204542224951},
                                                           "fold_fraction": 0.0}},
          "chain_translation_dense_outside": {"status": "ok",
                                              "values": {"coverage": 0.8935546875,
                                                         "ncc_before": 0.17351623137054228,
                                                         "ncc_after": 0.35442587822296057,
                                                         "ncc_gain": 0.18090964685241828,
                                                         "ssim_before": 0.2736203133305344,
                                                         "ssim_after": 0.8040429326205701},
                                              "counts": {"total": 24576,
                                                         "valid": 21960,
                                                         "valid_columns": 2745,
                                                         "ssim_columns": 2145},
                                              "reasons": {},
                                              "data_range": 132.0,
                                              "transform": {"kind": "dense",
                                                            "displacement_voxels": {"median": 3.5355339059327378,
                                                                                    "p95": 3.7199723678578294,
                                                                                    "max": 6.1410324659201825},
                                                            "displacement_physical": {"median": 0.6873007202455123,
                                                                                      "p95": 0.7247897839473822,
                                                                                      "max": 1.1966277101611782},
                                                            "fold_fraction": 8.138020833333333e-05}}}


def differences(pinned, actual, where):
    """Where actual differs from pinned: the same keys, floats within 1e-9, everything else equal."""
    if isinstance(pinned, dict):
        if not isinstance(actual, dict) or list(actual) != list(pinned):
            return [f"{where}: keys {list(actual) if isinstance(actual, dict) else actual!r} != {list(pinned)}"]
        return [d for key in pinned for d in differences(pinned[key], actual[key], f"{where}.{key}")]
    if isinstance(pinned, list):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(pinned):
            return [f"{where}: {actual!r} != {pinned!r}"]
        return [d for i, (p, a) in enumerate(zip(pinned, actual)) for d in differences(p, a, f"{where}[{i}]")]
    if isinstance(pinned, float):
        ok = isinstance(actual, float) and abs(actual - pinned) <= 1e-9
        return [] if ok else [f"{where}: {actual!r} != {pinned!r}"]
    return [] if type(actual) is type(pinned) and actual == pinned else [f"{where}: {actual!r} != {pinned!r}"]


@pytest.mark.golden
@pytest.mark.parametrize("name", list(PINNED))
def test_qc_values_equal_those_of_the_starting_revision(name):
    if name.startswith("bspline"):
        pytest.importorskip("SimpleITK")
    transform = cases()[name]
    reference, moving = images()
    chain = transform if isinstance(transform, TransformChain) else TransformChain((transform,))
    after = resample([moving], chain, WarpConfig(backend="scipy"))[0]
    qc = registration_qc(reference, moving, after, transform, config=RegistrationQcConfig(projections=True))
    pinned = PINNED[name]
    actual = {"status": qc.status, "values": qc.values, "counts": qc.counts, "reasons": qc.reasons,
              "data_range": qc.config["data_range"], "transform": qc.details["transform"]}
    assert differences(pinned, actual, name) == []
    assert qc.units == UNITS
    # The projections are the float64 Z maxima of the signals, as when the signals were copied to float64 first.
    for label, image in (("reference", reference), ("before", moving), ("after", after)):
        projection = qc.details["projections"][label]
        assert projection.dtype == np.float64
        np.testing.assert_array_equal(projection, image.astype(np.float64).max(axis=0))


def test_peak_memory_of_the_chain_qc_is_at_most_eight_volumes():
    """A chain of one translation on float64 signals, as FOV._register_round passes the round's application QC.

    At the starting revision the peak was 21.3 such volumes (float64 copies of
    the three signals, the pull field, a nine-component Jacobian, gradients).
    """
    shape = (16, 256, 256)
    rng = np.random.default_rng(4610)
    reference = rng.random(shape)
    moving = np.roll(reference, (1, 3, -2), axis=(0, 1, 2)) + 0.1 * rng.random(shape)
    chain = TransformChain((TranslationTransform((1.0, 3.0, -2.0), **geometry(shape)),))
    after = resample([moving], chain, WarpConfig())[0]
    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    try:
        start = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        qc = registration_qc(reference, moving, after, chain, config=RegistrationQcConfig())
        peak = tracemalloc.get_traced_memory()[1] - start
    finally:
        if not tracing:
            tracemalloc.stop()
    volume = reference.nbytes
    assert peak <= 8 * volume, f"peak {peak / volume:.2f} float64 volumes"
    # The chain keeps the summary of a dense field: |u| is 14**0.5 voxels everywhere, and nothing folds.
    summary = qc.details["transform"]
    assert summary["kind"] == "dense" and summary["fold_fraction"] == 0.0
    assert summary["displacement_voxels"] == pytest.approx({"median": math.sqrt(14), "p95": math.sqrt(14),
                                                            "max": math.sqrt(14)}, abs=1e-9)
