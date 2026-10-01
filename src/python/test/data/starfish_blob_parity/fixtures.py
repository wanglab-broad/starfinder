"""W-266 starfish BlobDetector parity fixtures (pure NumPy, deterministic).

Each fixture is a float32 array of shape (round, channel, Z, Y, X) with values
in [0, 1], the form starfish ``ImageStack.from_numpy`` accepts without
conversion. Spots are sampled Gaussians at fractional positions drawn from the
fixture seed. The arrays are regenerated from this file; none is saved.

Parity cases pair a fixture with BlobDetector settings. ``LOG_SETTINGS`` are the
starfish ISS tutorial values (min_sigma=1, max_sigma=10, num_sigma=30,
threshold=0.01). ``LOG_SETTINGS_ANISOTROPIC`` uses per-axis ZYX sigmas that stay
at or above one voxel, so the expected tables stay small: the spacing-converted
tutorial values (1, 0.5, 0.5)-(10, 5, 5) gave about 2,450 noise detections per
channel on volume-2ch (also exact parity), recorded in the worker notes.
"""
import numpy as np

LOG_SETTINGS = dict(min_sigma=1, max_sigma=10, num_sigma=30, threshold=0.01)
LOG_SETTINGS_ANISOTROPIC = dict(min_sigma=(2.0, 1.0, 1.0), max_sigma=(6.0, 3.0, 3.0),
                                num_sigma=10, threshold=0.01)
# Converted ISS values for ZYX spacing (1, 2, 2); raises for a squeezed plane.
LOG_SETTINGS_SPACING_122 = dict(min_sigma=(1.0, 0.5, 0.5), max_sigma=(10.0, 5.0, 5.0),
                                num_sigma=30, threshold=0.01)
# overlap, exclude_border and is_volume stay at the BlobDetector defaults
# (0.5, False, True); detector_method is blob_log; no reference image.


def _spots(shape, centers, sigmas, amplitudes):
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    out = np.zeros(shape, dtype=np.float64)
    for center, amplitude in zip(centers, amplitudes):
        r2 = sum(((g - c) / s) ** 2 for g, c, s in zip(grids, center, sigmas))
        out += amplitude * np.exp(-0.5 * r2)
    return out


def _volume(rng, shape, n, sigmas, *, close_pair=False, background=0.05, noise=0.002):
    margin = np.array([2.0] * (len(shape) - 2) + [4.0, 4.0])[-len(shape):]
    upper = np.array(shape, dtype=float) - 1 - margin
    centers = rng.uniform(margin, upper, size=(n, len(shape)))
    if close_pair:  # two spots 2.5 voxels apart in X exercise the overlap pruning
        mid = (np.array(shape, dtype=float) - 1) / 2
        offset = np.zeros(len(shape))
        offset[-1] = 1.25
        centers = np.vstack([centers, mid - offset, mid + offset])
    amplitudes = rng.uniform(0.3, 0.8, size=len(centers))
    image = background + _spots(shape, centers, sigmas, amplitudes)
    image += rng.normal(0.0, noise, size=shape)
    return np.clip(image, 0.0, 1.0).astype(np.float32), centers, amplitudes


def fixture(name):
    """Return (array RCZYX float32, truth dict) for a fixture name."""
    if name == "volume-2ch":  # 16x64x64, two channels, anisotropic spot shape
        rng = np.random.default_rng(266001)
        ch0, c0, a0 = _volume(rng, (16, 64, 64), 10, (1.5, 1.3, 1.3), close_pair=True)
        ch1, c1, a1 = _volume(rng, (16, 64, 64), 8, (2.0, 1.0, 1.0))
        return np.stack([ch0, ch1])[None], {"centers": [c0, c1], "amplitudes": [a0, a1]}
    if name == "plane":  # 1x64x64 single plane: exercises the singleton-Z squeeze
        rng = np.random.default_rng(266002)
        plane, c, a = _volume(rng, (64, 64), 12, (1.3, 1.3), close_pair=True)
        return plane[None, None, None], {"centers": [c], "amplitudes": [a]}
    if name == "empty":  # all zeros: typed empty result
        return np.zeros((1, 1, 8, 32, 32), dtype=np.float32), {"centers": [], "amplitudes": []}
    raise ValueError(name)


FIXTURES = ("volume-2ch", "plane", "empty")
#: case name -> (fixture, settings); every case compares one table per (round, channel).
CASES = {
    "volume-2ch-isotropic": ("volume-2ch", LOG_SETTINGS),
    "volume-2ch-anisotropic": ("volume-2ch", LOG_SETTINGS_ANISOTROPIC),
    "plane-isotropic": ("plane", LOG_SETTINGS),
    "empty-isotropic": ("empty", LOG_SETTINGS),
}
#: Cases where both implementations must raise the same exception type.
ERROR_CASES = {
    "plane-anisotropic-3tuple": ("plane", LOG_SETTINGS_SPACING_122),
}
COLUMNS = ("intensity", "z", "y", "x", "radius", "spot_id")
