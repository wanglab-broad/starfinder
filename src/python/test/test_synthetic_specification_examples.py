"""Independent specification checks; renderer conformance belongs to W-157+."""
from pathlib import Path
import os
import runpy
import subprocess
import sys

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[3] / "docs/examples/synthetic_specification.py"
ORACLE = runpy.run_path(str(EXAMPLE))


def test_independent_arithmetic():
    ORACLE["check_arithmetic"]()


def test_descriptor_bytes():
    assert ORACLE["stream_descriptor"]("count") == (
        b'["starfinder.synthetic/1","development",42,"formed-v1","count",null,null,null]'
    )
    # SHA-256 standard test vector distinguishes hash algorithm/encoding mistakes.
    assert ORACLE["hashlib"].sha256(b"abc").hexdigest() == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_cross_process_hash_seed_independence():
    outputs = []
    for seed in ("1", "7654321"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outputs.append(subprocess.run(
            [sys.executable, str(EXAMPLE)], env=env, check=True,
            capture_output=True, text=True, timeout=30,
        ).stdout)
    assert outputs[0] == outputs[1]
    assert outputs[0].endswith("Synthetic specification examples passed.\n")


def test_component_isolation_and_scheduling():
    draw = ORACLE["reference_draws"]
    before = {name: draw(name, entity="0") for name in ORACLE["COMPONENTS"]}
    # An unrelated factor may consume more draws/change amplitude; no shared state.
    for index in range(3):
        _ = (index + 1) * draw("noise.independent", round_label="round2")
    after = {name: draw(name, entity="0") for name in reversed(ORACLE["COMPONENTS"])}
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    _ = draw("placement", entity="new-object")
    np.testing.assert_array_equal(before["placement"], draw("placement", entity="0"))


def test_namespace_separation():
    digest = ORACLE["stream_digest"]
    base = digest("placement")
    variants = [digest("identity"), digest("placement", seed=43)]
    for key, value in (("split", "calibration"), ("split", "evaluation"),
                       ("scene", "other"), ("entity", "0"),
                       ("round_label", "round2"), ("channel_label", "ch00")):
        variants.append(digest("placement", **{key: value}))
    assert len(set([base, *variants])) == 9
    # JSON field boundaries avoid collisions from concatenating labels.
    assert digest("placement", scene="ab", entity="c") != digest(
        "placement", scene="a", entity="bc")


@pytest.mark.parametrize("keys", [
    {"seed": True}, {"seed": -1}, {"seed": 2**64}, {"seed": 1.5},
    {"split": "test"}, {"scene": ""}, {"scene": None}, {"entity": "e\u0301"},
    {"round_label": 1},
])
def test_invalid_stream_keys(keys):
    with pytest.raises(ValueError):
        ORACLE["stream_descriptor"]("placement", **keys)


def test_unknown_component():
    with pytest.raises(ValueError, match="component"):
        ORACLE["stream_descriptor"]("typo")
