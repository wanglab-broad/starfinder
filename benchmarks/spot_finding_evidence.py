"""The spot-finding validation rows that W-899 moved out of the test suite, as an evidence script.

What it protects: the engineering validation S1 to S16 of design/specs/spot-finding-algorithms.md ("Engineering
validation design") on the seeds and the Piscis model the suite no longer runs. The design parametrizes S1 to S8, S10
and S16 by the seeds 100 to 102 with the same assertion, and S1 to S15 by both Piscis models; the suite keeps seed 100
and Piscis ``20251212`` (Jiahao's decisions of 2026-11-19 on W-893, design/proposals/test-suite-slimming.md, "The other
categories"):

* K4: seeds 101 and 102 of S1 to S8, S10 and S16, every method (local maxima, the Starfish LoG, the four Spotiflow
  models and both Piscis models), except S10's Piscis seed 102, which stays in the suite as the only seed that passes
  there (seeds 100 and 101 are strict expected failures). The strict expected failures recorded per seed (S4 Piscis,
  S5 ``smfish_3d`` seed 102, S7 ``synth_3d``, S10 Piscis seed 101) move with their seeds and stay strict here: a case
  that starts to pass fails the run;
* O3: every row of Piscis ``20230905`` in S1 to S15 (seed 100, S10's seed 102, S9, S11, S13 and S15). Both models'
  weight identity and recorded settings stay in the suite (``test_piscis.py``).

The cases are the test functions of ``src/python/test/test_spot_finding_validation.py`` themselves, called with the rows
that module leaves to this script (its ``EVIDENCE``, built by the same ``params`` as the suite's rows): the same
assertions, tolerances, strict expected failures, markers and cached detections as the suite's rows, and each case's
parameter ID as it was in the suite at ``ORIGIN_REVISION``. The fixtures are generated in session (at most 32x64x64
voxels, four channels, three rounds); nothing is read but the package, the test helpers and the model weights.

Software: the ``spotiflow`` and ``piscis`` extras with PyTorch on the CPU, and the weights of the four Spotiflow models
and both Piscis models through ``STARFINDER_WEIGHTS_DIR`` (``starfinder weights fetch``; the controller's learned checks
set the same variable). A learned case whose extra is missing skips with the reason (exit status 2); a missing weight
fails its case, naming the fetch command. It needs no appearance bank and no network. Every case runs on the CPU with
one numerical thread (the cases set ``CUDA_VISIBLE_DEVICES=""`` and the thread variables). About 35 CPU-minutes for the
192 cases on one CPU (W-893's inventory: 24.8 for K4, 10.0 for O3; a Piscis ``iso3d`` detection takes about 25 s), and
about 1.7 GB of peak RSS (the validation run of W-274).

When it runs: at the end of a batch whose changes touch spot finding (``src/python/starfinder/spot_finding/``, the
evaluation of spot finding or these tests' helpers), and with every evidence script before a tag or a release (Jiahao,
2026-11-19); the coordinating session runs it on UGER, in the image with the commit's code laid over it.

    cd src/python
    STARFINDER_WEIGHTS_DIR=<weights> uv run python ../../benchmarks/spot_finding_evidence.py \\
        --output <run dir>/spot-finding
    uv run python ../../benchmarks/spot_finding_evidence.py --list

``-k`` selects cases as pytest's ``-k`` does (for example ``-k 20230905``, or ``-k "not learned"`` for the local maxima
and LoG cases, which need no extra), and ``--case <test>[<parameter>]`` names one (repeatable); ``evidence_runner.py``
describes the output and the exit status (an expected failure that stays expected counts as passed). Do not run with
``python -O``, which drops the assertions.
"""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "python"
if str(PACKAGE) not in sys.path[:1]:
    sys.path.insert(0, str(PACKAGE))

from test import test_spot_finding_validation as V  # noqa: E402
# The module's fixtures: one numerical thread and no GPU for every case (autouse), and S6's registry spy.
from test.test_spot_finding_validation import one_thread, spy  # noqa: E402, F401

# The last revision whose test suite held these rows (W-899's starting revision).
ORIGIN_REVISION = "a7f4b8f51d76ae742d051e7a2ad8189bba3f9f03"
HERE = "benchmarks/spot_finding_evidence.py"
ORIGIN = "src/python/test/test_spot_finding_validation.py"
#: The test function of each check that has rows here.
TESTS = {
    "S1": "test_s1_isolated_spot_recall_and_precision",
    "S2": "test_s2_localization_of_the_s1_matches",
    "S3": "test_s3_every_member_of_a_resolvable_pair_is_matched",
    "S4": "test_s4_offset_channels_and_a_channel_override",
    "S5": "test_s5_coincident_spots_keep_one_row_per_channel",
    "S6": "test_s6_empty_input_zero_channels_and_the_mad_diagnostics",
    "S7": "test_s7_spots_near_the_faces",
    "S8": "test_s8_tiling_seams",
    "S9": "test_s9_explicit_scaling",
    "S10": "test_s10_multi_round_identities",
    "S11": "test_s11_candidates_checkpoint_round_trip",
    "S13": "test_s13_tables_are_identical_twice_in_one_process_and_in_a_second",
    "S15": "test_s15_dimensionality_rules",
    "S16": "test_s16_the_w218_option_and_exclude_border_on_formed16",
}
#: Each case's node ID here: (its node ID in the suite at ORIGIN_REVISION, its group of W-893's proposal).
MOVED = {f"{HERE}::{TESTS[check]}[{param.id}]": (f"{ORIGIN}::{TESTS[check]}[{param.id}]", group)
         for check, rows in V.EVIDENCE.items() for param, group in rows}

pytestmark = [pytest.mark.spot_finding, pytest.mark.validation]


def rows(check):
    """The rows of one check that the suite module leaves to this script, with their marks."""
    return [param for param, _ in V.EVIDENCE[check]]


@pytest.mark.parametrize("method, name, seed", rows("S1"))
def test_s1_isolated_spot_recall_and_precision(method, name, seed, record_property):
    V.test_s1_isolated_spot_recall_and_precision(method, name, seed, record_property)


@pytest.mark.parametrize("method, name, seed", rows("S2"))
def test_s2_localization_of_the_s1_matches(method, name, seed, record_property):
    V.test_s2_localization_of_the_s1_matches(method, name, seed, record_property)


@pytest.mark.parametrize("method, case, seed", rows("S3"))
def test_s3_every_member_of_a_resolvable_pair_is_matched(method, case, seed, record_property):
    V.test_s3_every_member_of_a_resolvable_pair_is_matched(method, case, seed, record_property)


@pytest.mark.parametrize("method, seed", rows("S4"))
def test_s4_offset_channels_and_a_channel_override(method, seed, record_property):
    V.test_s4_offset_channels_and_a_channel_override(method, seed, record_property)


@pytest.mark.parametrize("method, seed", rows("S5"))
def test_s5_coincident_spots_keep_one_row_per_channel(method, seed, record_property):
    V.test_s5_coincident_spots_keep_one_row_per_channel(method, seed, record_property)


@pytest.mark.parametrize("method, seed", rows("S6"))
def test_s6_empty_input_zero_channels_and_the_mad_diagnostics(method, seed, spy, record_property):
    V.test_s6_empty_input_zero_channels_and_the_mad_diagnostics(method, seed, spy, record_property)


@pytest.mark.parametrize("method, exclude_border, dims, seed", rows("S7"))
def test_s7_spots_near_the_faces(method, exclude_border, dims, seed, record_property):
    V.test_s7_spots_near_the_faces(method, exclude_border, dims, seed, record_property)


@pytest.mark.parametrize("method, case, seed", rows("S8"))
def test_s8_tiling_seams(method, case, seed, record_property):
    V.test_s8_tiling_seams(method, case, seed, record_property)


@pytest.mark.parametrize("method, case", rows("S9"))
def test_s9_explicit_scaling(method, case):
    V.test_s9_explicit_scaling(method, case)


@pytest.mark.parametrize("method, seed", rows("S10"))
def test_s10_multi_round_identities(method, seed, record_property):
    V.test_s10_multi_round_identities(method, seed, record_property)


@pytest.mark.parametrize("method, source, seed, table_format", rows("S11"))
def test_s11_candidates_checkpoint_round_trip(method, source, seed, table_format, tmp_path):
    V.test_s11_candidates_checkpoint_round_trip(method, source, seed, table_format, tmp_path)


@pytest.mark.slow  # every row shares the second process of its group, 5 s or more
@pytest.mark.parametrize("method, name", rows("S13"))
def test_s13_tables_are_identical_twice_in_one_process_and_in_a_second(method, name, record_property):
    V.test_s13_tables_are_identical_twice_in_one_process_and_in_a_second(method, name, record_property)


@pytest.mark.parametrize("method, shape", rows("S15"))
def test_s15_dimensionality_rules(method, shape, record_property):
    V.test_s15_dimensionality_rules(method, shape, record_property)


@pytest.mark.parametrize("part, variant, seed", rows("S16"))
def test_s16_the_w218_option_and_exclude_border_on_formed16(part, variant, seed, record_property):
    V.test_s16_the_w218_option_and_exclude_border_on_formed16(part, variant, seed, record_property)


def main(argv=None):
    spec = importlib.util.spec_from_file_location("evidence_runner", Path(__file__).resolve().parent / "evidence_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner.run(__file__, MOVED, ORIGIN_REVISION, argv,
                      description="Run the spot-finding validation rows that W-899 moved out of the test suite: seeds "
                                  "101 and 102 of S1 to S8, S10 and S16 (K4) and every row of Piscis 20230905 (O3). "
                                  "Needs the spotiflow and piscis extras and their weights through "
                                  "STARFINDER_WEIGHTS_DIR.",
                      epilog="Run from src/python: STARFINDER_WEIGHTS_DIR=<weights> uv run python "
                             "../../benchmarks/spot_finding_evidence.py --output <run dir>/spot-finding")


if __name__ == "__main__":
    sys.exit(main())
