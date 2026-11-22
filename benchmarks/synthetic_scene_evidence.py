"""The synthetic scene's checks on the three ``slow`` context presets that W-899 moved out of the test suite.

What it protects: the cellular scene's generator on the presets ``context-dense3d-slow-v1``,
``context-tissue2d-slow-v1`` and ``context-culture-slow-v1`` (``context_slow``), as the suite checks the three routine
presets in its fast tier (Jiahao's decisions of 2026-11-19 on W-893, design/proposals/test-suite-slimming.md, "The
other categories"):

* K5, design/validation/synthetic-cellular-scene.md: GN7, packing (K1 for every pair of sites recomputed by the test
  oracle, the placement record and F2's floor, read back from the saved ``context.json``), and GN5, the lobes and LB1
  (``cellular_checks.check_lobes``), on each ``slow`` preset, generated once per module as the suite did;
* O1, design/validation/synthetic-real-appearance-scene.md: GA1 (b), the group digests of everything
  ``save_cellular_scene`` writes for each ``slow`` preset against its pins, an identity contract (Jiahao's decision of
  2026-11-19 to move the three ``slow`` presets' digests here). The routine presets' digests stay in the suite.

GN7 and GN5 are the tests of ``src/python/test/test_synthetic_cellular_presets.py`` at ``ORIGIN_REVISION``, moved here
unchanged with their module fixture ``slow_scene`` (module scope) and their markers; they import ``generate`` and
``check_packing`` from that module and ``check_lobes`` from ``cellular_checks.py``. GA1 is the test function of
``test_synthetic_appearance_golden.py`` itself, called with the three ``slow`` presets: the same digests (``PINNED``,
which stays in that module), canonical forms and assertions. Each case keeps the node ID's parameter it had in the
suite.

Software: the package's base dependencies (no extra); it reads no real data and needs no appearance bank and no model:
every case generates its preset in its temporary folder. About 13 CPU-minutes for the nine cases on one CPU (W-893's
inventory: 6.2 for K5 and 6.9 for O1; most of it the generation of each preset, twice per preset).

When it runs: at the end of a batch whose changes touch the synthetic generator (``src/python/starfinder/synthetic/``)
or these tests' helpers, and with every evidence script before a tag or a release (Jiahao, 2026-11-19); the
coordinating session runs it on UGER, in the image with the commit's code laid over it.

    cd src/python
    uv run python ../../benchmarks/synthetic_scene_evidence.py --output <run dir>/synthetic-scene
    uv run python ../../benchmarks/synthetic_scene_evidence.py --list

``-k`` selects cases as pytest's ``-k`` does (for example ``-k tissue2d``), and ``--case <test>[<parameter>]`` names
one (repeatable); ``evidence_runner.py`` describes the output and the exit status. Do not run with ``python -O``, which
drops the assertions.
"""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "python"
if str(PACKAGE) not in sys.path[:1]:
    sys.path.insert(0, str(PACKAGE))

from test import test_synthetic_appearance_golden as GA1  # noqa: E402
from test.cellular_checks import check_lobes  # noqa: E402
from test.test_synthetic_cellular_presets import check_packing, generate  # noqa: E402

# The last revision whose test suite held these tests (W-899's starting revision).
ORIGIN_REVISION = "a7f4b8f51d76ae742d051e7a2ad8189bba3f9f03"
HERE = "benchmarks/synthetic_scene_evidence.py"
PRESETS = "src/python/test/test_synthetic_cellular_presets.py"
GOLDEN = "src/python/test/test_synthetic_appearance_golden.py"
GA1_TEST = "test_saved_context_preset_outputs_equal_the_base_revision_pins"
SLOW = ["context-dense3d-slow-v1", "context-tissue2d-slow-v1", "context-culture-slow-v1"]
#: Each case's node ID here: (its node ID in the suite at ORIGIN_REVISION, its group of W-893's proposal).
MOVED = {
    **{f"{HERE}::{test}[{name}]": (f"{PRESETS}::{test}[{name}]", "K5") for name in SLOW
       for test in ("test_gn7_slow_preset_reaches_its_floor", "test_gn5_lobes_on_the_slow_presets")},
    **{f"{HERE}::{GA1_TEST}[{name}]": (f"{GOLDEN}::{GA1_TEST}[{name}]", "O1") for name in SLOW},
}


# --- K5: from test_synthetic_cellular_presets.py (its module markers: synthetic, validation) ------------------------

@pytest.fixture(scope="module", params=SLOW)
def slow_scene(request):
    config, scene = generate(request.param)
    yield request.param, config, scene
    del scene


@pytest.mark.synthetic
@pytest.mark.validation
@pytest.mark.slow
def test_gn7_slow_preset_reaches_its_floor(slow_scene, tmp_path):
    name, config, scene = slow_scene
    check_packing(name, config, scene, tmp_path)


# ---------------------------------------------------------------- GN5 on the slow presets (W-486)

@pytest.mark.synthetic
@pytest.mark.validation
@pytest.mark.slow
def test_gn5_lobes_on_the_slow_presets(slow_scene):
    name, config, scene = slow_scene
    check_lobes(name, config, scene)


# --- O1: GA1's slow presets, the test of test_synthetic_appearance_golden.py (its markers: synthetic, golden) -------

@pytest.mark.synthetic
@pytest.mark.golden
@pytest.mark.parametrize("name", [pytest.param(name, marks=pytest.mark.slow) for name in SLOW])
def test_saved_context_preset_outputs_equal_the_base_revision_pins(tmp_path, name):
    GA1.test_saved_context_preset_outputs_equal_the_base_revision_pins(tmp_path, name)


def main(argv=None):
    spec = importlib.util.spec_from_file_location("evidence_runner", Path(__file__).resolve().parent / "evidence_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner.run(__file__, MOVED, ORIGIN_REVISION, argv,
                      description="Run the synthetic scene's checks on the three slow context presets that W-899 moved "
                                  "out of the test suite: GN7 and GN5 (K5) and GA1's digests (O1). Needs no extra, no "
                                  "bank and no real data.",
                      epilog="Run from src/python: uv run python ../../benchmarks/synthetic_scene_evidence.py "
                             "--output <run dir>/synthetic-scene")


if __name__ == "__main__":
    sys.exit(main())
