"""Independent W-155 arithmetic/seed oracle; not a scene/effects implementation."""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata

import numpy as np

CONTRACT = "starfinder.synthetic/1"
COMPONENTS = (
    "count", "placement", "identity", "brightness", "width.axial",
    "width.lateral", "elongation", "angle", "round.dropout", "round.weakening",
    "round.loss", "geometry.translation", "geometry.local", "background.count",
    "background.placement", "background.width", "background.brightness",
    "noise.dependent", "noise.independent",
)


def stream_descriptor(component, *, split="development", seed=42,
                      scene="formed-v1", entity=None, round_label=None,
                      channel_label=None):
    """Reference byte encoding; reserved split descriptors do not generate scenes."""
    if split not in ("development", "calibration", "evaluation"):
        raise ValueError("unknown split")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if component not in COMPONENTS:
        raise ValueError("unknown component")
    for label in (scene, entity, round_label, channel_label):
        if label is not None and (not isinstance(label, str) or not label
                                  or unicodedata.normalize("NFC", label) != label):
            raise ValueError("labels must be nonempty NFC strings or optional null")
    if scene is None:
        raise ValueError("scene is required")
    return json.dumps(
        [CONTRACT, split, seed, scene, component, entity, round_label, channel_label],
        ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def stream_digest(component, **keys):
    return hashlib.sha256(stream_descriptor(component, **keys)).hexdigest()


def reference_draws(component, **keys):
    seed = int(stream_digest(component, **keys), 16)
    return np.random.Generator(np.random.PCG64(seed)).standard_normal(4)


def stream_probe():
    """Small development-only draw payload for separate-process comparison."""
    return {
        component: {
            "digest": stream_digest(component, entity="amplicon-0"),
            "draws": reference_draws(component, entity="amplicon-0").tolist(),
        }
        for component in COMPONENTS
    }


def check_arithmetic():
    """Evaluate definitions against hand-calculated constants, never a renderer."""
    for calculated, literal in (
        (8 * math.exp(-0.5), 4.852245277701067),
        (8 * math.exp(-8), 0.002683701023220095),
        (8 * math.exp(-0.125), 7.059975220676764),
        (8 * math.exp(-2), 1.0826822658929016),
    ):
        assert math.isclose(calculated, literal, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(float(np.float32(calculated)), literal,
                            rel_tol=0, abs_tol=1e-6)
    intended = np.array([8., 8., 8.])
    trend = intended * 0.5 ** np.arange(3)
    np.testing.assert_array_equal(trend, [8, 4, 2])
    np.testing.assert_array_equal(trend * [1, 0, 1], [8, 0, 2])
    np.testing.assert_array_equal(trend * [1, 0, 0], [8, 0, 0])
    np.testing.assert_array_equal(trend * [1, 0.25, 1], [8, 1, 2])
    np.testing.assert_array_equal(intended, [8, 8, 8])
    mixing = np.eye(4)
    mixing[1, 0] = 0.25
    np.testing.assert_array_equal(mixing @ [8, 0, 0, 0], [8, 2, 0, 0])
    np.testing.assert_array_equal(mixing @ [4, 0, 0, 0] + [1, 2, 3, 4], [5, 3, 3, 4])
    assert 9 + math.sqrt(4 * 9) * -0.5 + 2 * 0.25 == 6.5
    np.testing.assert_array_equal(
        np.clip(np.rint([-1, 0.5, 1.5, 255.5]), 0, 255).astype(np.uint8),
        [0, 0, 2, 255],
    )
    np.testing.assert_array_equal(np.array([1, 2, 3]) + [0.5, -1, 2], [1.5, 1, 5])
    assert 3 + 0.25 + 0.5 == 3.75
    # Invert an independent scalar nonconstant analytic example at its center.
    observed = 3.25
    q = observed
    for _ in range(100):
        updated = observed - 0.25 * math.exp(-(q - 3)**2 / 8)
        if abs(updated - q) <= 1e-10:
            q = updated
            break
        q = updated
    else:
        raise AssertionError("analytic inverse did not converge")
    assert abs(q - 3) <= 2e-10
    assert abs(q + 0.25 * math.exp(-(q - 3)**2 / 8) - observed) <= 2e-10


if __name__ == "__main__":
    check_arithmetic()
    print(json.dumps(stream_probe(), sort_keys=True))
    print("Synthetic specification examples passed.")
