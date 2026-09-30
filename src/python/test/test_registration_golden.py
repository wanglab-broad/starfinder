"""Golden baseline for the current registration path through ``FOV.run`` (W-245).

Pins, with exact SHA-256 digests, the registered images and the transforms of a
translation-only run and of a translation -> demons run on one small seeded
synthetic fixture, using the current signal behavior (``merged``: the float64
channel sum). Any behavior change, including a changed demons iteration count
or a changed signal mode, changes a digest. The digests were produced with the
locked project environment (SimpleITK 2.5.3, NumPy 2.2.6, SciPy 1.17.0) and are
bit-identical over repeated single-thread runs; see docs/registration-baseline.md.

Every registration configuration is built by ``registration_config``, which
also holds the only imports of registration configuration types. The §2.6
refactor (``RegistrationRecipe``) replaces only that helper. The fixture, the
runs, and the input, translation-only and demons-field digests stay; the
translation -> demons image digest changes only through the reviewed change to
one final resampling described in docs/registration-contract.md.
"""
import hashlib

import numpy as np
import pytest

from starfinder.dataset import Dataset, RoundState
from starfinder.image import ImageMetadata

pytest.importorskip("SimpleITK")

SHAPE_ZYX = (8, 32, 32)
CHANNELS = 4
SEED = 20260929
SHIFT_ZYX = (1.0, 3.0, -2.0)  # content displacement of the moving round, in voxels
BUMP_ZYX = (0.0, 1.5, -1.0)  # peak of the smooth local displacement added on top
DEMONS_ITERATIONS = (100, 50, 25)  # DemonsConfig default


def registration_config(methods, *, signal="merged", demons_iterations=DEMONS_ITERATIONS):
    """The only place that builds registration configuration.

    methods is an ordered tuple of "translation" and "demons". signal is the
    current per-step image representation for both rounds: "merged" (the
    float64 channel sum) or "single-channel" (channel 0). Other settings are
    the defaults of each config.
    """
    from starfinder.dataset import PipelineConfig, RegistrationStep
    from starfinder.registration import DemonsConfig, TranslationConfig

    configs = {"translation": lambda: TranslationConfig(),
               "demons": lambda: DemonsConfig(iterations=tuple(demons_iterations))}
    return PipelineConfig(registration=tuple(
        RegistrationStep(configs[method](), signal, signal, 0) for method in methods))


def _render(rng, centers, channels, amplitudes, gains):
    """uint16 ZYXC: channel offsets, Gaussian noise and anisotropic Gaussian puncta."""
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in SHAPE_ZYX), indexing="ij")
    volume = np.empty(SHAPE_ZYX + (CHANNELS,), dtype=np.float64)
    for c, offset in enumerate((100.0, 150.0, 200.0, 250.0)):
        volume[..., c] = offset + rng.normal(0.0, 10.0, SHAPE_ZYX)
    for (cz, cy, cx), c, amplitude in zip(centers, channels, amplitudes):
        volume[..., c] += gains[c] * amplitude * np.exp(
            -((z - cz) ** 2) / (2 * 1.0**2) - ((y - cy) ** 2 + (x - cx) ** 2) / (2 * 1.2**2))
    return np.clip(np.rint(volume), 0, 65535).astype(np.uint16)


def fixture_rounds():
    """round1 is the reference; round2 images the same puncta displaced and with other gains.

    A punctum at q in round1 appears at q + SHIFT_ZYX + b(q) in round2, where b
    is a Gaussian bump of peak BUMP_ZYX centred in YX (sigma 6 voxels).
    """
    rng = np.random.RandomState(SEED)
    n = 60
    centers = np.column_stack([rng.uniform(1.5, SHAPE_ZYX[0] - 2.5, n),
                               rng.uniform(5.0, SHAPE_ZYX[1] - 5.0, n),
                               rng.uniform(5.0, SHAPE_ZYX[2] - 5.0, n)])
    channels = rng.randint(CHANNELS, size=n)
    amplitudes = rng.uniform(800.0, 3000.0, n)
    bump = np.exp(-((centers[:, 1] - 16.0) ** 2 + (centers[:, 2] - 16.0) ** 2) / (2 * 6.0**2))
    moved = centers + np.asarray(SHIFT_ZYX) + bump[:, None] * np.asarray(BUMP_ZYX)
    return {"round1": _render(rng, centers, channels, amplitudes, (1.0, 1.0, 1.0, 1.0)),
            "round2": _render(rng, moved, channels, amplitudes, (0.8, 1.2, 1.0, 1.5))}


def digest(array):
    """SHA-256 over dtype, shape and C-order bytes."""
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode())
    h.update(array.tobytes())
    return h.hexdigest()


def run(tmp_path, methods, **options):
    """FOV.run on the resident fixture; returns image digests and transform summaries."""
    rounds = RoundState(sequencing_rounds=["round1", "round2"], reference_round="round1")
    dataset = Dataset(tmp_path, tmp_path / "out", "golden", "sample", "out", rounds=rounds,
                      channel_order=["ch00", "ch01", "ch02", "ch03"])
    fov = dataset.fov("FOV_001")
    for name, volume in fixture_rounds().items():
        fov.images[name] = volume
        fov.metadata[name] = ImageMetadata(f"FOV_001/{name}")
    fov.run(registration_config(methods, **options))
    transforms = []
    for result in fov.registration_results["round2"]:
        transform = result.transform
        if hasattr(transform, "correction_zyx"):
            transforms.append(("translation", transform.correction_zyx))
        else:
            transforms.append((result.diagnostics.method, digest(transform.displacement_zyx)))
    return {"images": {name: digest(image) for name, image in fov.images.items()},
            "transforms": transforms}


PINNED_INPUTS = {
    "round1": "f6772a500ecc9fcbe57e983c813f58a5ce2b7264b26bff97f189f33fa6edac32",
    "round2": "ea6e8746ad03f4e958ac89ebff53b42749a4c4b8ac2357a75ab0fdb5e2d4f7a1",
}
# Pull source index = reference index - correction. The Y correction is -4, not
# -3: near the centre the bump adds up to 1.5 voxels to the 3-voxel shift. The
# reference round is never transformed.
PINNED_RUNS = {
    ("translation",): {
        "images": {"round1": PINNED_INPUTS["round1"],
                   "round2": "74c50550ff358c3f51f60a678ce14e7b0c8f9647966a7e7a7506f633afa47ba9"},
        "transforms": [("translation", (-1.0, -4.0, 2.0))],
    },
    ("translation", "demons"): {
        "images": {"round1": PINNED_INPUTS["round1"],
                   "round2": "6abb8c28e591a5e8260bb50058b5265e45d76da8dd1833de2dd4f2c4b1e8bd06"},
        "transforms": [
            ("translation", (-1.0, -4.0, 2.0)),
            ("demons", "54ff31f9a458cc255f7b4a2d19b9703a6eb6ed9d22b1f3209734f4cb3dc04968"),
        ],
    },
}


def test_fixture_inputs_are_pinned():
    assert {name: digest(volume) for name, volume in fixture_rounds().items()} == PINNED_INPUTS


@pytest.mark.parametrize("methods", list(PINNED_RUNS), ids="-".join)
def test_registered_images_and_transforms_are_pinned(tmp_path, methods):
    assert run(tmp_path, methods) == PINNED_RUNS[methods]


def test_a_changed_demons_iteration_count_changes_the_digests(tmp_path):
    methods = ("translation", "demons")
    changed = run(tmp_path, methods, demons_iterations=(100, 50, 24))
    assert changed["images"]["round2"] != PINNED_RUNS[methods]["images"]["round2"]
    assert changed["transforms"][1] != PINNED_RUNS[methods]["transforms"][1]


def test_a_changed_signal_mode_changes_the_digests(tmp_path):
    methods = ("translation", "demons")
    changed = run(tmp_path, methods, signal="single-channel")
    assert changed["images"]["round2"] != PINNED_RUNS[methods]["images"]["round2"]
    assert changed["transforms"][1] != PINNED_RUNS[methods]["transforms"][1]
