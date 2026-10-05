"""Golden baseline for the current segmentation input preparation and label steps (W-307, §2.9).

Pins, with exact SHA-256 digests, on one small seeded fixture (16×64×64 uint8 DAPI,
merged amplicon and Flamingo images):

(a) the DAPI–amplicon composite of ``workflow/scripts/create_nuclei_amplicon_overlay.py``
    with and without its maximum projection, and the Flamingo enhancement of
    ``workflow/scripts/enhance_dapi_with_flamingo.py``, both run unchanged through a stub
    ``snakemake`` object, and both on constant and all-zero inputs (a constant image passes
    through their quantile stretch unchanged);
(b) the foreground gate decision of ``workflow/scripts/stardist_segmentation.py`` (Otsu
    threshold, connected components, largest area > 100) in 3D and 2D;
(c) the script's nearest-neighbour rescale of labels back to the input grid after its
    fixed 0.5 shrink in Y and X;
(d) the script's per-slice label expansion and its final ``uint16`` cast.

The StarDist script cannot be imported in the locked environment (it imports StarDist
and ``tifffile.imsave``, which tifffile 2026.1.28 no longer has), so (b) to (d) run
``legacy_stardist_steps``, whose lines are cited against the script; a test checks that
those lines are still in the script. The model call is replaced by a stand-in
(``stand_in_model``: Otsu threshold and connected components), so nothing here pins
model inference; the W-306 parity outputs pin it (docs/segmentation-baseline.md).
Four tests document legacy behavior that §2.9 changes: the gate raises on an image with
no foreground, the composite needs 3D inputs, the rescale round trip changes an odd
grid, and the ``uint16`` cast wraps labels above 65,535.

The digests were produced with the locked project environment (NumPy 2.2.6,
scikit-image 0.26.0, SciPy 1.17.0, tifffile 2026.1.28) and are bit-identical over
repeated single-thread runs; see docs/segmentation-baseline.md. The §2.9 work may
replace the body of ``legacy_stardist_steps``, ``composite`` and
``flamingo_enhancement`` with calls to the package functions; every pinned digest
stays, except where docs/segmentation-contract.md names an edit.
"""
import hashlib
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.segmentation import expand_labels
from skimage.transform import rescale

from starfinder.preprocessing import ProjectionConfig, project_image
from starfinder.segmentation import composite_nuclei_amplicon, enhance_with_flamingo

pytestmark = [pytest.mark.workflow, pytest.mark.segmentation, pytest.mark.golden]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "workflow" / "scripts"
SHAPE_ZYX = (16, 64, 64)
SEED = 20261005
DISTANCE = 4  # the expansion distance of the W-306 parity runs
# (z, y, x) centre and (z, y, x) semi-axes in voxels. Nuclei 1 and 2 lie 3 voxels apart, so
# their expansions meet; the two small blobs hold fewer than 100 voxels each.
NUCLEI = (((7, 14, 14), (3, 6, 6)), ((8, 14, 29), (3, 5, 5)), ((6, 40, 18), (2, 6, 7)),
          ((9, 44, 46), (3, 7, 6)), ((10, 20, 50), (2, 5, 5)))
SMALL = (((4, 54, 8), (1, 2, 2)), ((12, 6, 40), (1, 2, 3)))
ODD_YX = (61, 63)  # the odd-sized 2D crop of the rescale round trip


def _ellipsoids(shape, blobs):
    """Boolean mask of the union of the given ellipsoids."""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    mask = np.zeros(shape, bool)
    for (cz, cy, cx), (rz, ry, rx) in blobs:
        mask |= ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1
    return mask


def _uint8(values):
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def fixture():
    """The seeded uint8 images: DAPI, merged amplicon, Flamingo, and DAPI with small blobs only."""
    rng = np.random.default_rng(SEED)
    nuclei, small = _ellipsoids(SHAPE_ZYX, NUCLEI), _ellipsoids(SHAPE_ZYX, SMALL)
    stained = nuclei | small
    background = 12 + rng.normal(0, 2, SHAPE_ZYX)
    dapi = _uint8(background + 110 * stained + rng.normal(0, 6, SHAPE_ZYX) * stained)
    small_only = _uint8(12 + rng.normal(0, 2, SHAPE_ZYX) + 110 * small)
    # Cytoplasm: larger ellipsoids around each nucleus, dimmer inside the nucleus.
    cells = _ellipsoids(SHAPE_ZYX, [(c, (r[0] + 1, r[1] + 4, r[2] + 4)) for c, r in NUCLEI])
    flamingo = _uint8(15 + rng.normal(0, 3, SHAPE_ZYX) + 90 * cells - 50 * nuclei)
    # Amplicons: 80 single bright voxels, a few saturated, on a dim background.
    amplicon = 8 + rng.normal(0, 2, SHAPE_ZYX)
    points = tuple(rng.integers(0, n, 80) for n in SHAPE_ZYX)
    amplicon[points] = rng.uniform(120, 255, 80)
    amplicon[points[0][:4], points[1][:4], points[2][:4]] = 255
    return {"dapi": dapi, "amplicon": _uint8(amplicon), "flamingo": flamingo,
            "small_only": small_only}


def digest(array):
    """SHA-256 over dtype, shape and C-order bytes."""
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode())
    h.update(array.tobytes())
    return h.hexdigest()


def run_script(name, tmp_path, inputs, config):
    """Run workflow/scripts/<name> unchanged with a stub snakemake object; return its output."""
    paths = {}
    for key, image in inputs.items():
        paths[key] = str(tmp_path / f"{key}.tif")
        tifffile.imwrite(paths[key], image)
    output = tmp_path / "output.tif"
    snakemake = SimpleNamespace(input=paths, output=[str(output)], config=config)
    runpy.run_path(str(SCRIPTS / name), init_globals={"snakemake": snakemake})
    return tifffile.imread(output)


def composite(tmp_path, dapi, amplicon, maximum_projection):
    """create_nuclei_amplicon_overlay.py (rule create_nuclei_amplicon_overlay)."""
    parameters = {"maximum_projection": maximum_projection}
    config = {"rules": {"create_nuclei_amplicon_overlay": {"parameters": parameters}}}
    return run_script("create_nuclei_amplicon_overlay.py", tmp_path,
                      {"dapi_img": dapi, "amplicon_img": amplicon}, config)


def flamingo_enhancement(tmp_path, dapi, flamingo):
    """enhance_dapi_with_flamingo.py (rule enhance_dapi_with_flamingo)."""
    return run_script("enhance_dapi_with_flamingo.py", tmp_path,
                      {"dapi_img": dapi, "flamingo_img": flamingo}, {})


def stand_in_model(image):
    """Stands in for normalize + predict_instances: Otsu foreground, connected components, int32."""
    return label(image > threshold_otsu(image)).astype(np.int32)


def foreground_gate(image):
    """stardist_segmentation.py lines 17-21: the Otsu threshold and the component areas."""
    threshold = threshold_otsu(image)                           # :17
    bw_img = image > threshold                                  # :18
    props = regionprops(label(bw_img))                          # :19-20
    return threshold, np.array([prop.area for prop in props])  # :21


def legacy_stardist_steps(image, *, rescale_labels, expand, distance=DISTANCE, min_area=100,
                          factor=0.5, model=stand_in_model):
    """workflow/scripts/stardist_segmentation.py at 6b384cd without the model, line by line.

    model(image) returns the label image of the image it is given; it replaces the
    normalization and predict_instances calls (lines 33-34 and 37-38 in 3D, 53-54 and
    57-58 in 2D). min_area and factor are the script's literals 100 and 0.5, exposed only
    for the digest-change test; the script reads rescale, expand_labels and distance from
    rules.stardist_segmentation.parameters.
    """
    back = 1 / factor
    _, areas = foreground_gate(image)
    if areas.max() > min_area:                                                  # :23
        if len(image.shape) == 3:                                               # :24
            if rescale_labels:                                                  # :31
                labels = model(rescale(image, [1, factor, factor]))             # :32-34
                labels = rescale(labels, [1, back, back], order=0, preserve_range=True)  # :35
            else:
                labels = model(image)                                           # :37-38
            if expand:                                                          # :40
                for z in range(labels.shape[0]):                                # :41
                    labels[z, :, :] = expand_labels(labels[z, :, :], distance=distance)  # :42
        else:
            if rescale_labels:                                                  # :51
                labels = model(rescale(image, [factor, factor]))                # :52-54
                labels = rescale(labels, [back, back], order=0, preserve_range=True)  # :55
            else:
                labels = model(image)                                           # :57-58
            if expand:                                                          # :60
                labels = expand_labels(labels, distance=distance)               # :61
        return labels.astype("uint16")                                          # :63
    return np.zeros(image.shape, dtype="uint16")                                # :65


# The lines of stardist_segmentation.py that legacy_stardist_steps follows.
_PARAMETERS = "snakemake.config['rules']['stardist_segmentation']['parameters']"
SCRIPT_LINES = {
    17: "threshold = threshold_otsu(current_img)",
    18: "bw_img = current_img > threshold",
    19: "test_img = label(bw_img)",
    20: "props = regionprops(test_img)",
    21: "areas = np.array([prop.area for prop in props])",
    23: "if areas.max() > 100:",
    24: "if len(current_img.shape) == 3:",
    32: "current_img = rescale(current_img, [1, .5, .5])",
    35: "current_label = rescale(labels, [1, 2, 2], order=0, preserve_range=True)",
    41: "for z in range(current_label.shape[0]):",
    42: ("current_label[z,:,:] = expand_labels(current_label[z,:,:], "
         f"distance={_PARAMETERS}['distance'])"),
    52: "current_img = rescale(current_img, [.5, .5])",
    55: "current_label = rescale(labels, [2, 2], order=0, preserve_range=True)",
    61: f"current_label = expand_labels(current_label, distance={_PARAMETERS}['distance'])",
    63: "imsave(snakemake.output[0], current_label.astype('uint16'), compression='zlib')",
    65: "current_label = np.zeros(current_img.shape, dtype='uint16')",
}

INPUT_DIGESTS = {
    "dapi": "cc0b54f3e857c174cac174787503951164f5fbabdfa8723c4f512d4265018dfd",
    "amplicon": "b8ffdf9b46627a06967e7775431a7e49f9ff6fd1d8e040bffc675885b119ad39",
    "flamingo": "d8d604e4d4dace727c82601623818c464e56497f023b9f275d472f8f15a1cbfe",
    "small_only": "f01d59dbc825f11ea902687df7b672fcc16c7781e5516e5d9ebc53e64984cde2",
}
# maximum_projection -> composite
COMPOSITE_DIGESTS = {False: "3a7320da2ab158dfb5f780835fdd3cc26e4291d753203895d7b2a20e8d58dac7",
                     True: "3352e74a9d068b8fe8298ba0981eeae098030b5cc9180a1421489cf4fe75d6c9"}
FLAMINGO_DIGEST = "a66bd5989fb80be850802f3ceed8924ecb7e49305adb9e0c6d60a129a651875f"
# image -> (Otsu threshold, number of components, largest area, gate passes)
GATE = {
    "dapi": (22.0, 7, 521, True),
    "dapi_2d": (22.0, 7, 131, True),
    "small_only": (21.0, 2, 21, False),
}
GATE_AREAS_DIGESTS = {
    "dapi": "b9720200737505bfb34c3ab5b3f27f3ef639088b5e205e02d94e7e64d2bc41ae",
    "dapi_2d": "228af7d8bc0715ff0603ea35cad26d16359a2e80173f10a6b37e671fe00b5997",
    "small_only": "2b74379c7e4f296245b6b1b0aaee4c31b8f6d49804bb973c8c1b49ff483e82eb",
}
# (rescale, expand) -> digest of the uint16 label image, in 3D and in 2D (the DAPI Z maximum)
LABELS_3D = {
    (False, False): "ec97107d690a88a4844e9fb257bbb119562e3d78310630d2f43faa221409bbb4",
    (False, True): "343c4dc33550ccaa2dff16dfbaa9466a0651a45954898449030f05f77f323c48",
    (True, False): "669c1cccf0f383c0e25c58a1c754a9f1d919dd032661ec7dee3ffe8cb300f8c2",
    (True, True): "df47e26cb841fa90b8b3ed4888d8b9b0fd153e6b2e2da012391fdd512e7544d0",
}
LABELS_2D = {
    (False, False): "213dcecea9784ee9d2c36b70717839a729e522d87e424b44f74f8c2a0011c61e",
    (False, True): "00f2964c40e8f8130b4105795f55c5f14ba61bc2fee93f53272b665b36e36f6b",
    (True, False): "15db8fd1a58211f8c9c285eb25ca46bf708bcf04783f7744d5f1d0133c073bf2",
    (True, True): "94898d39122fc589c3719170811a86bbb5844f7f766d854d1021ea30a0f7ed6b",
}
# rescale(dapi, [1, .5, .5]): float64, 16×32×32
SHRUNK_IMAGE_DIGEST = "11e543854295fe0c085cb7d40d87fc87649d07f8a17ea948ba704803eb308cb3"
# the stand-in labels on the shrunk grid
SHRUNK_LABELS_DIGEST = "9a1bd53e5353a3cbbc68f6e28e32cb5309a00a3889882c861ee0baf9bf40f8d7"
# (dtype, digest) of the labels rescaled back to 16×64×64
RESTORED_LABELS = ("<i4", "e4f1917decc37e7624442672437bd40f75e6589ee9e3f3f89cd6f84f17db512e")
# Constant and all-zero inputs (W-307 repair): the composite with a constant (50) or all-zero
# nuclear image, the Flamingo enhancement with a constant (80) or all-zero Flamingo image
CONSTANT_DIGESTS = {
    "composite_constant": "e3243c99e77c1157d05452adc0c9cc827900931e2eb37761ba7007d2113acdf2",
    "composite_zero": "22b2aded212f0c6aa03dfb392c9fbd6cc533ed4987da7e5343489cbb3597f53e",
    "flamingo_constant": "1fd6fc43ecbe16c439425d52710940ea96b32e90f7493a206beb9db9f0e9e816",
    "flamingo_zero": "a81c7feae65ae58f9df9e14ea1489c4beb7452054fca8a585d71c4a812c9ec4c",
}


@pytest.fixture(scope="module")
def images():
    return fixture()


def test_the_helper_follows_the_script():
    lines = (SCRIPTS / "stardist_segmentation.py").read_text().splitlines()
    assert {n: lines[n - 1].strip() for n in SCRIPT_LINES} == SCRIPT_LINES


def test_fixture_digests(images):
    assert all(image.shape == SHAPE_ZYX and image.dtype == np.uint8 for image in images.values())
    assert {name: digest(image) for name, image in images.items()} == INPUT_DIGESTS


@pytest.mark.parametrize("maximum_projection", [False, True])
def test_composite(tmp_path, images, maximum_projection):
    result = composite(tmp_path, images["dapi"], images["amplicon"], maximum_projection)
    assert result.dtype == np.uint8
    assert result.shape == (SHAPE_ZYX[1:] if maximum_projection else SHAPE_ZYX)
    assert digest(result) == COMPOSITE_DIGESTS[maximum_projection]


def test_composite_needs_3d_inputs(tmp_path, images):
    """Legacy: the channel maximum is over axis 3, so projected 2D inputs raise (script line 33)."""
    with pytest.raises(np.exceptions.AxisError):
        composite(tmp_path, images["dapi"].max(axis=0), images["amplicon"].max(axis=0), False)


def test_flamingo_enhancement(tmp_path, images):
    result = flamingo_enhancement(tmp_path, images["dapi"], images["flamingo"])
    assert result.dtype == np.uint8 and result.shape == SHAPE_ZYX
    assert digest(result) == FLAMINGO_DIGEST


@pytest.mark.parametrize("maximum_projection", [False, True])
def test_composite_function(images, maximum_projection):
    """composite_nuclei_amplicon gives the script's composite; the projection is a run's z maximum."""
    result, _ = composite_nuclei_amplicon(images["dapi"], images["amplicon"])
    if maximum_projection:
        result = project_image(result, config=ProjectionConfig())[0]
    assert result.dtype == np.uint8
    assert result.shape == (SHAPE_ZYX[1:] if maximum_projection else SHAPE_ZYX)
    assert digest(result) == COMPOSITE_DIGESTS[maximum_projection]


def test_flamingo_enhancement_function(images):
    result, _ = enhance_with_flamingo(images["dapi"], images["flamingo"])
    assert result.dtype == np.uint8 and result.shape == SHAPE_ZYX
    assert digest(result) == FLAMINGO_DIGEST


def test_constant_and_zero_inputs(tmp_path, images):
    """A constant image passes through the quantile stretch unchanged; it is not set to zero.

    Its two quantiles are equal, so skimage's rescale_intensity clips it to the output
    range (the dtype range for uint8), which leaves its grey level as it is.
    """
    c50, c80, zero = (np.full(SHAPE_ZYX, value, np.uint8) for value in (50, 80, 0))
    assert np.unique(composite(tmp_path, c50, c80, False)).tolist() == [80]  # the maximum
    assert np.unique(composite(tmp_path, zero, zero, False)).tolist() == [0]
    # 50 × (1 − 80/255) = 34.3
    assert np.unique(flamingo_enhancement(tmp_path, c50, c80)).tolist() == [34]
    zero_nuclear = flamingo_enhancement(tmp_path, zero, images["flamingo"])
    assert np.unique(zero_nuclear).tolist() == [0]
    constant = composite(tmp_path, c50, images["amplicon"], False)
    alone = composite(tmp_path, zero, images["amplicon"], False)  # the stretched amplicon alone
    assert np.array_equal(constant, np.maximum(alone, 50))
    results = {"composite_constant": constant, "composite_zero": alone,
               "flamingo_constant": flamingo_enhancement(tmp_path, images["dapi"], c80),
               "flamingo_zero": flamingo_enhancement(tmp_path, images["dapi"], zero)}
    assert {name: digest(result) for name, result in results.items()} == CONSTANT_DIGESTS


@pytest.mark.parametrize("name", ["dapi", "dapi_2d", "small_only"])
def test_foreground_gate(images, name):
    image = images["dapi"].max(axis=0) if name == "dapi_2d" else images[name]
    threshold, areas = foreground_gate(image)
    decision = (float(threshold), len(areas), int(areas.max()), bool(areas.max() > 100))
    assert decision == GATE[name]
    assert digest(areas) == GATE_AREAS_DIGESTS[name]


def test_closed_gate_writes_an_empty_uint16_image(images):
    labels = legacy_stardist_steps(images["small_only"], rescale_labels=True, expand=True)
    assert labels.dtype == np.uint16 and labels.shape == SHAPE_ZYX and not labels.any()


def test_gate_raises_on_an_image_without_foreground():
    """Legacy: areas.max() of no components raises before the zero-label fallback (line 23)."""
    with pytest.raises(ValueError, match="zero-size array"):
        legacy_stardist_steps(np.zeros(SHAPE_ZYX, np.uint8), rescale_labels=False, expand=False)


def test_rescale_round_trip(images):
    shrunk = rescale(images["dapi"], [1, .5, .5])
    assert shrunk.dtype == np.float64 and shrunk.shape == (16, 32, 32)
    assert digest(shrunk) == SHRUNK_IMAGE_DIGEST
    labels = stand_in_model(shrunk)
    assert digest(labels) == SHRUNK_LABELS_DIGEST
    restored = rescale(labels, [1, 2, 2], order=0, preserve_range=True)
    assert restored.shape == SHAPE_ZYX
    assert (restored.dtype.str, digest(restored)) == RESTORED_LABELS
    # Nearest neighbour: every restored voxel holds the label of its 2×2 parent.
    assert np.array_equal(restored, labels.repeat(2, axis=1).repeat(2, axis=2))


def test_rescale_round_trip_changes_an_odd_grid(images):
    """Legacy: 61×63 comes back as round(round(n × 0.5) × 2) = 60×64 (NumPy rounds half to even)."""
    image = images["dapi"].max(axis=0)[:ODD_YX[0], :ODD_YX[1]]
    assert rescale(image, [.5, .5]).shape == (30, 32)
    assert legacy_stardist_steps(image, rescale_labels=True, expand=False).shape == (60, 64)


@pytest.mark.parametrize("rescale_labels", [False, True])
@pytest.mark.parametrize("expand", [False, True])
def test_label_steps_3d(images, rescale_labels, expand):
    labels = legacy_stardist_steps(images["dapi"], rescale_labels=rescale_labels, expand=expand)
    assert labels.dtype == np.uint16 and labels.shape == SHAPE_ZYX
    assert digest(labels) == LABELS_3D[(rescale_labels, expand)]


@pytest.mark.parametrize("rescale_labels", [False, True])
@pytest.mark.parametrize("expand", [False, True])
def test_label_steps_2d(images, rescale_labels, expand):
    image = images["dapi"].max(axis=0)
    labels = legacy_stardist_steps(image, rescale_labels=rescale_labels, expand=expand)
    assert labels.dtype == np.uint16 and labels.shape == SHAPE_ZYX[1:]
    assert digest(labels) == LABELS_2D[(rescale_labels, expand)]


def test_expansion_is_per_slice(images):
    """3D expansion grows labels in Y and X only: empty planes stay empty, no label moves plane."""
    labels = stand_in_model(images["dapi"])
    expanded = legacy_stardist_steps(images["dapi"], rescale_labels=False, expand=True)
    expanded = expanded.astype(np.int32)
    empty = ~labels.any(axis=(1, 2))
    assert empty.any() and not expanded[empty].any()
    for z in range(SHAPE_ZYX[0]):
        assert set(np.unique(expanded[z])) == set(np.unique(labels[z]))
    assert np.array_equal(expanded[labels > 0], labels[labels > 0])
    assert (expanded > 0).sum() > (labels > 0).sum()


def test_uint16_cast_wraps():
    """Legacy: the final astype('uint16') (line 63) maps label 65,536 to 0 and 65,537 to 1."""
    labels = np.array([[1, 65535, 65536, 65537]], dtype=np.int32)
    assert labels.astype("uint16").tolist() == [[1, 65535, 0, 1]]


@pytest.mark.parametrize("change", ["distance", "min_area", "factor", "maximum_projection"])
def test_changing_a_pinned_parameter_changes_a_digest(tmp_path, images, change):
    if change == "maximum_projection":
        # The projected composite differs from the stack; it equals the stack's Z maximum.
        stack = composite(tmp_path, images["dapi"], images["amplicon"], False)
        assert digest(stack.max(axis=0)) == COMPOSITE_DIGESTS[True] != COMPOSITE_DIGESTS[False]
        return
    options = {"distance": dict(distance=DISTANCE - 1), "min_area": dict(min_area=GATE["dapi"][2]),
               "factor": dict(factor=0.25)}[change]
    labels = legacy_stardist_steps(images["dapi"], rescale_labels=True, expand=True, **options)
    assert digest(labels) != LABELS_3D[(True, True)]
