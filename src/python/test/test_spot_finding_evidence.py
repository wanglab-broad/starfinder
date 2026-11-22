"""The split of the spot-finding validation rows between the suite and benchmarks/spot_finding_evidence.py (W-899).

W-893's K4 and O3 (Jiahao, 2026-11-19) keep seed 100 and Piscis 20251212 of test_spot_finding_validation.py in the
suite and move seeds 101 and 102 (except S10's Piscis seed 102) and every Piscis 20230905 row to the evidence script,
as design/specs/spot-finding-algorithms.md states ("Implemented checks", "Where the rows run").
``test_the_suite_keeps_seed_100_and_piscis_20251212`` reads the suite's parametrization of each check;
``test_the_evidence_script_collects_exactly_the_moved_rows`` collects the script (``--list``, in a process of its own)
and checks it holds exactly the rows the module leaves to it, each with the node ID it had in the suite.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from . import test_spot_finding_validation as V
from .spot_finding_scenes import SEEDS

pytestmark = [pytest.mark.spot_finding, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "benchmarks" / "spot_finding_evidence.py"
CHECKS = {"S1": V.test_s1_isolated_spot_recall_and_precision, "S2": V.test_s2_localization_of_the_s1_matches,
          "S3": V.test_s3_every_member_of_a_resolvable_pair_is_matched,
          "S4": V.test_s4_offset_channels_and_a_channel_override,
          "S5": V.test_s5_coincident_spots_keep_one_row_per_channel,
          "S6": V.test_s6_empty_input_zero_channels_and_the_mad_diagnostics, "S7": V.test_s7_spots_near_the_faces,
          "S8": V.test_s8_tiling_seams, "S9": V.test_s9_explicit_scaling, "S10": V.test_s10_multi_round_identities,
          "S11": V.test_s11_candidates_checkpoint_round_trip,
          "S13": V.test_s13_tables_are_identical_twice_in_one_process_and_in_a_second,
          "S14": V.test_s14_dependency_and_weights_errors, "S15": V.test_s15_dimensionality_rules,
          "S16": V.test_s16_the_w218_option_and_exclude_border_on_formed16}


def suite_rows(test):
    """The parameter values of a test's rows in the suite."""
    (mark,) = [mark for mark in test.pytestmark if mark.name == "parametrize"]
    return [param.values for param in mark.args[1]]


def test_the_suite_keeps_seed_100_and_piscis_20251212():
    for check, test in CHECKS.items():
        rows = suite_rows(test)
        assert rows and not any(row[0] == V.EVIDENCE_MODEL for row in rows), check
        if check in V.SEEDED:
            seeds = {row[-1] for row in rows}
            kept = {row for row in rows if row[-1] != SEEDS[0]}
            assert SEEDS[0] in seeds, check
            assert all(check == "S10" and V.piscis(row) and row[-1] == SEEDS[2] for row in kept), (check, kept)
        moved = [param.values for param, _ in V.EVIDENCE.get(check, [])]
        assert not set(moved) & set(rows), check
        assert {V.evidence_group(check, row) for row in moved} <= {"K4", "O3"}, check
    assert set(V.EVIDENCE) == set(CHECKS) - {"S14"}
    # S10's Piscis 20251212 keeps seed 100 (a strict expected failure) and 102, its only passing seed.
    assert [row for row in suite_rows(V.test_s10_multi_round_identities) if V.piscis(row)] == [
        ("piscis-20251212", SEEDS[0]), ("piscis-20251212", SEEDS[2])]


def test_the_evidence_script_collects_exactly_the_moved_rows():
    """``--list`` exits 1 unless the script collects exactly its MOVED cases, each with its former node ID."""
    done = subprocess.run([sys.executable, str(SCRIPT), "--list"], cwd=ROOT / "src/python",
                          env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    listing = json.loads(done.stdout)
    assert listing["complete"], listing
    expected = {f"src/python/test/test_spot_finding_validation.py::{CHECKS[check].__name__}[{param.id}]": group
                for check, rows in V.EVIDENCE.items() for param, group in rows}
    assert {case["origin_node_id"]: case["group"] for case in listing["cases"]} == expected
    assert len(listing["cases"]) == len(expected)
