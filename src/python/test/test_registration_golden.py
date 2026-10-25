"""Golden baseline for the current registration path through ``FOV.run`` (W-245).

Pins the registered images and the transforms of a translation-only run and of
a translation -> demons run on one small seeded synthetic fixture, using the
current signal behavior (``merged``: the float64 channel sum). The uint16 images
keep their exact SHA-256 digests and the translation is exact (whole voxels);
the demons displacement field is an explicit expected array
(``data/expected/test_registration_golden/demons.npz``, W-574) compared with the
float64 tolerance of docs/contributing.md. Any behavior change, including a
changed demons iteration count or a changed signal mode, changes the round2
digest and moves the field outside that tolerance. The values were produced
with the locked project environment (SimpleITK 2.5.3, NumPy 2.2.6, SciPy 1.17.0);
see docs/registration-baseline.md.

Every registration configuration is built by ``registration_config``, which
also holds the only imports of registration configuration types; the pinned
runs use each config's defaults. The §2.6 refactor (``RegistrationRecipe``)
replaces that helper. Its one other named edit is the translation -> demons
image digest, which changes with the reviewed one final resampling described in
docs/registration-contract.md. The fixture, the runs, and the input,
translation-only and demons-field digests stay (the field is the expected array
since W-574).
"""
from pathlib import Path

import numpy as np
import pytest

from starfinder.dataset import Dataset, RoundState
from starfinder.image import ImageMetadata

from .coordination_helpers import digest
from .expected_values import FLOAT64, read_arrays

pytestmark = [pytest.mark.registration, pytest.mark.golden]

pytest.importorskip("SimpleITK")

SHAPE_ZYX = (8, 32, 32)
CHANNELS = 4
SEED = 20260929
EXPECTED = Path(__file__).resolve().parent / "data" / "expected" / "test_registration_golden"
SHIFT_ZYX = (1.0, 3.0, -2.0)  # content displacement of the moving round, in voxels
BUMP_ZYX = (0.0, 1.5, -1.0)  # peak of the smooth local displacement added on top


def registration_config(methods, *, signal="merged", demons_iterations=None):
    """The only place that builds registration configuration.

    methods is an ordered tuple of "translation" and "demons". signal is the
    current per-step image representation for both rounds: "merged" (the
    float64 channel sum) or "single-channel" (channel 0). Other settings are
    the defaults of each config; demons_iterations=None keeps the DemonsConfig
    default, so a changed default changes the pinned digests.
    """
    from starfinder.dataset import PipelineConfig
    from starfinder.registration import (DemonsConfig, RegistrationRecipe, RegistrationSignalConfig, RegistrationStep,
                                         TranslationConfig)

    configs = {"translation": lambda: TranslationConfig(),
               "demons": lambda: (DemonsConfig() if demons_iterations is None
                                  else DemonsConfig(iterations=tuple(demons_iterations)))}
    return PipelineConfig(registration=RegistrationRecipe(
        steps=tuple(RegistrationStep(configs[method]()) for method in methods),
        signal=RegistrationSignalConfig(mode="sum" if signal == "merged" else "channel",
                                        reference_channel=None if signal == "merged" else 0)))


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


def run(tmp_path, methods, **options):
    """FOV.run on the resident fixture; returns image digests and the transforms (translation, demons field)."""
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
        transforms.append((result.diagnostics.method, transform.displacement_zyx))
    return {"images": {name: digest(image) for name, image in fov.images.items()},
            "transforms": transforms}


PINNED_INPUTS = {
    "round1": "f6772a500ecc9fcbe57e983c813f58a5ce2b7264b26bff97f189f33fa6edac32",
    "round2": "ea6e8746ad03f4e958ac89ebff53b42749a4c4b8ac2357a75ab0fdb5e2d4f7a1",
}
# Pull source index = reference index + displacement. The Y displacement is 4, not
# 3: near the centre the bump adds up to 1.5 voxels to the 3-voxel shift. The
# reference round is never transformed.
PINNED_RUNS = {
    ("translation",): {
        "images": {"round1": PINNED_INPUTS["round1"],
                   "round2": "74c50550ff358c3f51f60a678ce14e7b0c8f9647966a7e7a7506f633afa47ba9"},
        "transforms": [("translation", (1.0, 4.0, -2.0))],
    },
    ("translation", "demons"): {
        "images": {"round1": PINNED_INPUTS["round1"],
                   # One final resampling (W-256): re-pin approved by Jiahao in W-246 (2026-09-30), on the operator's
                   # reading that both W-245 pull points lie one voxel inside their grids; the reading was
                   # confirmed by Jiahao on 2026-09-30.
                   "round2": "aee3f8c33c613371790fdf1e83e9f7431538c6c64b820a75eba735eb5c224697"},
        # The demons field: the array of demons.npz (expected_demons_field), at the float64 tolerance.
        "transforms": [
            ("translation", (1.0, 4.0, -2.0)),
            ("demons", "demons.npz"),
        ],
    },
}


def expected_demons_field():
    """The demons displacement field (Z, Y, X, ZYX components) of the translation -> demons run, float64."""
    return read_arrays(EXPECTED / "demons.npz")["displacement_zyx"]


def assert_transforms_match(transforms, pinned):
    """The methods in order; a translation exactly; a demons field within FLOAT64 of the expected array."""
    assert [method for method, _ in transforms] == [method for method, _ in pinned]
    for (method, value), (_, expected) in zip(transforms, pinned):
        if method == "translation":
            assert value == expected
        else:
            expected = expected_demons_field()
            assert value.dtype == expected.dtype and value.shape == expected.shape
            np.testing.assert_allclose(value, expected, **FLOAT64)


def test_fixture_inputs_are_pinned():
    assert {name: digest(volume) for name, volume in fixture_rounds().items()} == PINNED_INPUTS


@pytest.mark.parametrize("methods", list(PINNED_RUNS), ids="-".join)
def test_registered_images_and_transforms_are_pinned(tmp_path, methods):
    result = run(tmp_path, methods)
    assert result["images"] == PINNED_RUNS[methods]["images"]
    assert_transforms_match(result["transforms"], PINNED_RUNS[methods]["transforms"])


def test_a_changed_demons_iteration_count_changes_the_digests(tmp_path):
    methods = ("translation", "demons")
    changed = run(tmp_path, methods, demons_iterations=(100, 50, 24))
    assert changed["images"]["round2"] != PINNED_RUNS[methods]["images"]["round2"]
    assert changed["transforms"][1][0] == "demons"
    assert not np.allclose(changed["transforms"][1][1], expected_demons_field(), **FLOAT64)


def test_a_changed_signal_mode_changes_the_digests(tmp_path):
    methods = ("translation", "demons")
    changed = run(tmp_path, methods, signal="single-channel")
    assert changed["images"]["round2"] != PINNED_RUNS[methods]["images"]["round2"]
    assert changed["transforms"][1][0] == "demons"
    assert not np.allclose(changed["transforms"][1][1], expected_demons_field(), **FLOAT64)
